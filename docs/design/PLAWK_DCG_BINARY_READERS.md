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
| tagged union (landed) | 8-byte tag + arm | per-arm `$1..$N` via `case` blocks | widest arm (tag in SSA only) |
| bounded repetition (landed) | 8-byte count + count×elem, count ≤ K | count `i64` + K flat elem fields; `foreach` | count slot + K elem slots, zeroed past count |

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

1. **Tagged unions (landed):** `BINFMT = "case(i64 f64 | lps16 i64)"`
   — an 8-byte discriminator selects an arm layout. The chosen surface
   is per-arm blocks (`case K { rules }`): the arm is lexically scoped,
   so every rule's field types are decided by the parser, not by
   guard-shape analysis. Case blocks flatten into one scalar rule chain
   whose guards prepend a tag check (`arm_pat/3`); the reader is a
   native `switch` on the tag dispatching per-arm field-read sequences
   that all materialize at offset 0 of a buffer sized to the widest arm
   (the tag lives only in the `%vr_tag` SSA value — no case block needs
   it at runtime because the arm is static inside the block). Unknown
   tags and truncated arms are malformed input (`fail_read`); arms
   without a case block are still read and skipped so the stream stays
   framed. The tag-guard spelling (landed) is accepted as sugar:
   `TAG == K && P { actions }` desugars into `case K { P { actions } }`
   (one single-rule block per rule, so source order is preserved and
   the two spellings compile to identical IR). Every rule must lead
   with a tag guard, and a tag test under `||`/`!` or in a non-leftmost
   conjunct is rejected. Arms may carry a `repK(...)` (landed):
   `foreach` inside a case block resolves against that arm's own
   layout, per-arm staging rides the same access-type expansion, and
   the union buffer (max record size across arms) covers it
   automatically. writebin inside case blocks (landed): OUTFMT is
   program-wide (one output layout regardless of arm) while each
   rule's source fields type against its own arm, so a union stream
   normalizes into one fixed layout; a pure normalizer (every rule
   just writebins) needs no END and no scalar state. Assoc arrays
   inside case blocks (landed): rules are assoc increments whose keys
   are raw i64 field values typed per arm, all arms updating one
   shared table per array name; both END report shapes (for-in print,
   integer lookups) work, and the assoc rule chain resolves guards and
   key loads through the same per-rule arm descriptor as the scalar
   chain. for-in writebin over a union input (landed): the END table
   walk writebins one fixed-layout (key, count, ...) record per group,
   mirroring the plain binary group-by-to-binary-output clause. Union
   (tagged) output (landed): `OUTFMT = "case(arm0 | arm1)"` (same
   spelling as BINFMT), and each writebin site statically targets one
   arm -- `writebin case K, args` emits the 8-byte tag K then arm K's
   slots through the per-slot varlen writer (the shared buffer sizes
   to the widest arm; its element pointer is resolved once in the
   entry block). Output is byte-compatible with the union reader, so
   tagged plawk-to-plawk pipelines round-trip; works from flat or
   union inputs and in the single-rule endless shape. Arm slots are
   i64/f64/sN/lpsN (a tagged rep write is a later slice); plain
   writebin with a union OUTFMT, arm writes against a flat OUTFMT,
   out-of-range arms, and arity mismatches all reject the program.
2. **Bounded repetition (landed):** `repK(elem types)` — an 8-byte
   count (≤ K) then that many elements. Fixed-width elements read as
   one bulk count×elemsize read after a memset of the element region
   (element slots past the count are deterministic zeros); elements
   containing `lpsN` strings have per-element wire sizes, so the reader
   emits a runtime loop that parses one element at a time into its
   fixed in-memory slot group (the lps materializes as an NUL-padded
   `sN` slot, same as at top level). Access layout flattens: the count
   is an i64 field and each element's fields are plain record fields.
   The surface answer to "for over elements" is `foreach { actions }`:
   inside the block `$1..$M` are the current element's fields, and the
   compiler emits a genuine runtime loop — the current element is
   memcpy'd into a hidden staging slot group appended to the record
   buffer and every scalar slot rides a loop-carried typed phi, so code
   size is O(body) at any cap. One rep per layout (per arm in a
   union), no rep/blob nesting; those are later extensions.
3. **Varlen writers (landed):** `lpsN` in OUTFMT emits the 8-byte
   length plus exactly the payload bytes, sourced from literals,
   `sM`/`lpsM` input fields (`M ≤ cap`), or text-mode slices clamped to
   the cap. Records with an lps slot switch from the single-buffer
   fwrite to per-slot fwrites emitted left to right (buffered in libc,
   so still memcpy cost per record). Writer output is byte-compatible
   with the `lpsN` reader. `repK(elems)` in OUTFMT (landed) is a
   passthrough slot: the argument names the input rep's count field,
   and the writer emits the live count plus the live elements — one
   bulk fwrite of count×elemsize bytes when the elements are all
   fixed-width (in-memory layout == wire layout), or a writer-side
   loop when they contain `lpsN` strings (each iteration recovers the
   live length from the NUL-padded slot with strnlen and emits the
   prefix plus exactly those bytes, mirroring the reader loop). Caps
   and element layouts must match the input rep exactly. Guarded rules
   therefore make byte-exact stream filters, in plain and union input
   modes.
4. **Tier-2 composition sugar (landed):** `blobN` — a length-prefixed
   binary payload whose only consumer is a compiled-Prolog foreign
   call. The record loop frames natively (length read, cap check, bulk
   payload read into the record buffer); passing `$K` to a
   `prolog_call`/`prolog_guard` copies the payload into the shared
   transient buffer (`@wam_transient_atom_from_bytes`, constant memory,
   no interning) and marshals it as the transient atom, which the
   compiled predicate reads with `atom_codes/2` and parses with an
   ordinary WAM DCG — real choice points and all. i64 fields marshal
   as WAM integers and f64 fields as WAM floats (typed loads, no text;
   the f64 packs its bits under the Float tag via `@value_float`).
   Double-returning calls (landed) are spelled `float(name(args))`:
   the site calls a `{double, ok}` wrapper that accepts an Integer or
   Float output and promotes via `@value_to_double`; a failed call
   contributes 0.0, and the written scalar types as a double through
   the ordinary slot-type fixpoint. Constraints: one blob argument per
   call (one shared buffer), NUL-free payloads (the transient atom is
   a C string), blob output is a later slice.

## Embedded Prolog blocks (landed)

`@prolog ... @end` blocks in the plawk source hold ordinary Prolog
clauses (including DCG rules, which expand on the way in). Markers sit
alone on their line; a heredoc-style tag (`@prolog-TAG ... @end-TAG`,
exact tag match) makes the fence unambiguous when the Prolog text
itself contains an `@end`-shaped line. `plawk_parse_source/3` lifts
the blocks before the awk grammar runs and term-reads them;
`plawk_prolog_block_preds/2` installs the clauses (reset-then-assert,
so recompiles replace) and returns the `user:Name/Arity` list that
`write_wam_llvm_project/3` takes — from there the existing foreign
bridge does everything (guards, i64/f64/blob marshaling,
`float(name(args))`). One source file now carries the whole Tier-2
story: native framing above, the payload grammar below.

awk-style expression functions are sugar over the same bridge:
`function scale(a, b) { return a * b + 1 }` desugars at parse time to
`scale(A, B, R) :- R is A * B + 1` (awk precedence, `%` → mod, float
literals allowed; every identifier in the body must be a parameter)
and installs through the identical clause path. Full imperative awk
functions (locals, loops, early returns) are deliberately out of
scope: between `foreach`, `if/else`, and Prolog blocks they don't earn
their native-codegen complexity, and a hot expression function can be
inlined natively later without changing the surface.

## Known design debt

- **(FIXED) The "if-then-else grammar returned 0" incident — root
  cause was neither if-then-else nor cut.** Bisection showed ITE
  (including binding conditions), cut, and recursive cut all lower
  correctly; the failing grammar differed in calling `code_type/2`,
  which was not a WAM builtin. A call to an unknown predicate lowered
  to **label index 0** with only a stderr warning, so the digit check
  failed silently and the whole parse summed to 0. Two fixes:
  `code_type/2` is now a builtin (id 172, entering `char_type/2`'s
  classifier through phis at `ctp.dispatch` — same type names, ASCII
  semantics, check mode); and unknown labels are a **compile-time
  existence error** by default (`wam_strict_labels(false)` restores
  the legacy index-0 fallback). The originally-failing grammar is the
  standing regression test, alongside a test that an uncompiled
  callee fails the compile.
- **(FIXED) WAM bug: constant-in-list clause heads never matched.**
  `unify_constant` built its read-mode comparison value with a
  hardcoded Atom tag (op2 was unused), so an integer constant in a
  list/structure head — `p(..., [44|T], ...)` — compared `Atom(44)`
  against the heap's `Integer(44)` and always failed. The fix packs
  the tag into `op2 >> 16` (both encoders, mirroring `get_constant`)
  and routes read mode through the general `wam_unify_value`, which
  also closes a second hole: an unbound sub-arg used to "match"
  without being bound; it now binds with trailing. The Tier-2 payload
  DCG's comma clause is the standing regression test.


- **(FIXED) `foreach` unrolling did not scale in code size.** The
  original surface unrolled its block Cap times — O(Cap × body) IR.
  It is now a real runtime loop, the one loop in the emitter stack:
  an index phi, one loop-carried phi per scalar slot (typed i64 or
  double), and a per-iteration `memcpy` of the current element into a
  hidden staging group appended to the record buffer — so the body's
  field accesses remain compile-time offsets and every existing
  emitter (updates, doubles, prints, inner if/else, next/break) works
  unchanged inside the loop. Code size is O(body) at any cap
  (regression test: `rep64` emits one increment site), user-visible
  field numbering and `NF` are untouched, and exit values are the
  head phis themselves. The staging copy costs one small memcpy per
  element — noise next to the per-element work itself.

## Why not a real DCG engine in the loop?

Because the loop's performance contract is the whole point of PLAWK:
the fixed-record loop measured 5.6× faster than mawk precisely because
there is no dispatch, no allocation, and no interpretation per record.
Tier-1 grammars compile to the same shape of code a C programmer would
write by hand for that format. Anything that genuinely needs
unification or backtracking already has a home — the WAM interpreter
in the same binary — and the bridge between the tiers is a function
call, not an architecture change.
