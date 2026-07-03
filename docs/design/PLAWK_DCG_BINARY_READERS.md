<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# PLAWK DCG Binary Readers: Grammar-Driven Record Parsing

Phase 3's remaining goal is reading binary formats whose layout is not
one fixed-size struct: length-prefixed fields, tagged unions,
repetition. In Prolog terms these are DCGs over byte lists; in PLAWK
terms they must become **native readers in the AOT-compiled record
loop** — no interpreter, no allocation per record, constant memory.
This document fixes the design: which grammar shapes lower to native
code, what the in-memory contract is, and where the WAM bytecode
fallback begins.

## The core idea: parse on the wire, fixed layout in memory

The fixed-record pipeline works because every downstream consumer
(guards, prints, scalar updates, assoc keys, writebin sources) reads
typed values at compile-time offsets from one record buffer `%rec`.
The DCG reader design keeps that invariant:

> A grammar-driven reader may consume a *variable* number of bytes from
> the stream, but it always **materializes the record into the same
> fixed access layout** that `BINFMT` describes. Wire shape and access
> shape are decoupled; only the reader knows the wire.

This means the entire existing emitter stack works unchanged over
grammar-read records. A field's **stored type** (what the reader
handles) maps to an **access type** (what consumers see):

| stored (wire) type | wire bytes | access type | in-memory slot |
|---|---|---|---|
| `i64` | 8 | `i64` | 8 bytes |
| `f64` | 8 | `f64` | 8 bytes |
| `sN` | N | `sN` | N bytes |
| `lpsN` (landed) | 8 + len, len ≤ N | `sN` | N bytes, NUL-padded |
| tagged union (planned) | 8-byte tag + arm | tag `i64` + widest arm | tag slot + max-arm slot |
| bounded repetition (planned) | 8-byte count + count×elem, count ≤ K | count `i64` + K elems | count slot + K elem slots |

## The lowering spectrum

A general DCG can do things a streaming native reader cannot
(unbounded lookahead, backtracking across records, building terms).
The design splits grammar shapes into three tiers:

**Tier 1 — native, streaming, constant memory (the PLAWK reader).**
Grammars that are *LL(1) over field boundaries with compile-time size
bounds*: each field's byte count is known either statically (`i64`,
`sN`) or from an already-read header (`lpsN` length prefix, a union
tag, a repetition count), and every variable quantity has a
compile-time cap so the access layout is fixed. These lower to a
field-by-field read sequence in the driver loop — straight-line native
code with one branch per header check. No backtracking exists because
the format is self-describing left to right. This tier is what PLAWK
implements, slice by slice.

**Tier 2 — native readers for *record framing*, WAM for interpretation.**
Formats where the record boundary is Tier-1 but the payload needs real
parsing (say, a length-prefixed blob containing a nested grammar). The
loop stays native and hands the payload slice to a compiled Prolog
predicate through the existing foreign-call bridge (~0.2µs, constant
memory via heap-top save/restore). The DCG runs as WAM bytecode over
the byte slice. This tier needs no new machinery — it composes the
binary reader with `prolog_call`.

**Tier 3 — full DCG fallback (future, Phase 5 adjacent).** Grammars
with genuine nondeterminism or unbounded structure run entirely as WAM
bytecode DCGs; the stream feeds them via buffered byte lists. This is
correct but not streaming-fast, and it is also where the Phase 5
runtime-DCG JIT would eventually apply: a grammar loaded at runtime
compiles through the same WAM→LLVM path, promoting Tier-3 rules to
Tier-1/2 readers on the fly.

The design rule of thumb: **a format specification (`BINFMT`) is a
degenerate grammar, and each new stored type is one production**
added to the Tier-1 reader generator. There is no separate "DCG
engine" in Tier 1 — the grammar is compiled away entirely.

## Slice 1 (landed): length-prefixed strings, `lpsN`

`BEGIN { BINFMT = "i64 lps16" }` declares a record that is 8 fixed
bytes followed by one length-prefixed string: an 8-byte native-endian
length `L` (validated `0 ≤ L ≤ 16` unsigned), then `L` payload bytes.

- **Reader:** records with any `lps` field switch the driver from the
  one-shot fixed-size read to a *varlen read sequence*
  (`llvm_emit_varlen_stream_driver_ir/5` + `plawk_varlen_read_ir/3`):
  each numeric field reads its 8 bytes straight into `%rec` at its
  access offset; each `lpsN` field reads its length into a shared
  8-byte scratch alloca, bounds-checks it, zeroes the N-byte slot, and
  reads the payload into it. Zero-length payloads are free
  (`@wam_stream_read_record` returns 1 immediately for size 0).
- **EOF discipline:** clean EOF is legal only before a record's first
  read (→ END actions run, exit 0). A short read anywhere else in the
  record — truncated payload, missing fields, an oversized length —
  exits through `fail_read` (exit 11), exactly like a trailing partial
  fixed record.
- **Access:** `lpsN` maps to access type `sN`
  (`plawk_binfmt_access_type/2`), so prints (`%.*s` via strnlen),
  equality guards (memcmp + NUL check), and `sN` OUTFMT passthrough
  work unchanged, and the record buffer size is the sum of *access*
  widths. i64/f64 fields in the same record keep their arithmetic,
  guards, assoc-key, and scalar-update roles.
- **Not in this slice:** `lps` in OUTFMT (writers stay fixed-layout;
  a varlen writer is the natural next writer slice), and `lps` caps
  are mandatory — an unbounded string would break the fixed access
  layout and constant-memory guarantee.

## Planned slices

1. **Tagged unions:** `BINFMT = "i64 union(0: i64 i64 | 1: lps16)"`
   style — an 8-byte discriminator selects one of several arm layouts;
   the access layout reserves the widest arm plus the tag as `$K`;
   guards on the tag route rules per arm. Requires surface syntax
   design (the awk-flavored form is the open question, e.g. rule
   patterns like `$tag == 1 { ... }` with per-arm field numbering).
2. **Bounded repetition:** `count` header + up to K fixed elements;
   access layout is a count slot plus K element slots; `for` over
   elements in rule bodies is the surface question.
3. **Varlen writers:** `lpsN` in OUTFMT emitting length + payload from
   `sN`/`lpsN` sources — closing the varlen pipeline loop the same way
   fixed writers did.
4. **Tier-2 composition sugar:** a declared payload type whose slice is
   passed to a compiled Prolog DCG via the foreign bridge without
   hand-written glue.

## Why not a real DCG engine in the loop?

Because the loop's performance contract is the whole point of PLAWK:
the fixed-record loop measured 5.6× faster than mawk precisely because
there is no dispatch, no allocation, and no interpretation per record.
Tier-1 grammars compile to the same shape of code a C programmer would
write by hand for that format. Anything that genuinely needs
unification or backtracking already has a home — the WAM interpreter
in the same binary — and the bridge between the tiers is a function
call, not an architecture change.
