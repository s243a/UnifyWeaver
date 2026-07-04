<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# PLAWK JIT Roadmap

Where the runtime-loadable-grammar (JIT) arc stands, and what comes next.

## What has landed

The dynamic-grammar surface is feature-complete for numeric work:

| Capability | Where |
|---|---|
| `.wamo` object format + writer (`write_wam_object/3`) | #3460 |
| Native loader + object-call primitive (`@wam_object_load`, `@wam_object_call_i64`) | #3460 |
| plawk `DYNLOAD` + `dyncall(args...)` (one fixed object) | #3461 |
| In-memory loader (`@wam_object_load_bytes`) | #3463 |
| `dyncall_at(Source, args...)` (runtime-chosen source) + path cache (`on`/`mtime`/`off`) | #3465 |
| `float(dyncall(...))` / `float(dyncall_at(...))` (double returns) | #3467 |

A grammar is compiled ahead of time to a `.wamo`, loaded at runtime (from a
fixed path, an in-memory buffer, or a per-call runtime source), cached with
optional mtime invalidation, and called returning `i64` or `double`.

**Loadable subset today:** the tier-2 grammar shape — `try_me_else` chains,
`get`/`put`/`set`/`unify` variable+value+constant, `get`/`put_list`,
`get`/`put_structure`, `allocate`/`deallocate`, `call`/`execute`,
`proceed`, `builtin_call`, `cut`/`get_level`. Excluded: **float constants**,
`switch_on_constant` tables, and `call/N` meta-calls.

## Next steps, in recommended order

Items 1–3 are largely independent; the ordering reflects value-per-effort
and the fact that each earlier item feeds the later ones. Items 4–5 have
real dependencies on the earlier subset work.

### 1. Lift float constants into the loadable subset — *small, quick win*

**What:** allow a grammar clause like `scale(X, R) :- R is X * 1.5` (a
`float` constant, tag 2) to compile into a `.wamo`. Today `write_wam_object`
rejects float literals; the object encoding for `get`/`put`/`set_constant`
already has a tag lane and `set_constant_literal_parts` knows how to emit a
`bitcast (double c to i64)`, so this is mostly relaxing the writer's
`wamo_const` guard and carrying the float payload through the relocation
(no relocation needed — a float constant is self-contained).

**Why:** `float(dyncall(...))` (just landed) makes float-returning grammars
first-class, but a grammar can currently only *reach* a Float by
computation (`R is X / 2`), not by writing one (`R is X * 1.5`). This closes
that gap. It is also the first, smallest step of the subset expansion that
item 5 (source-eval) ultimately needs.

**Effort:** small. **Depends on:** nothing.

### 2. Binary-data returns — opaque bytes — *moderate, high value*

**What:** let a grammar return a **byte string**, read by a new
`blob(dyncall(...))` / `blob(dyncall_at(...))` form. The grammar binds its
output to an Atom whose interned string is the payload (atoms in this
runtime are byte strings); a `@wam_object_call_bytes` primitive checks the
output tag is Atom, reads `@wam_atom_to_string` + length, and returns
`{ i8* ptr, i64 len, i1 ok }`. The atom lives in the (persistent) atom
table, so the pointer survives the arena rewind. In plawk the result is a
byte **slice** — exactly the `%WamSlice` (ptr,len) shape that
`llvm_emit_atom_field_slice` already produces — so it plugs into the
existing consumers: `print` (`%.*s`), `writebin` into an `sN`/`lpsN` slot,
equality guards, assoc keys.

**Why:** this is the "binary data return" you raised, in its
**no-deserialization** form — the bytes are opaque to plawk, consumed as a
string/blob. It opens grammars that *emit* encoded or textual output (a
formatter, an encoder, a template filler) rather than a single number. It
reuses the slice machinery, so it's mostly a new call primitive + surface,
not new consumer code. It is also the foundation for item 4.

**Effort:** moderate. **Depends on:** nothing (independent of item 1).

### 3. Multi-entry objects — *moderate, ergonomics*

**What:** let one `.wamo` expose several entry predicates, selected by name
at the call site — e.g. `dyncall@parse($1)` / `dyncall@classify($1)` over a
single `DYNLOAD = "lib.wamo"`. Needs the writer to emit a name→label-index
table, the loader to expose entry-by-name lookup, and a naming surface
(binding entries at declaration rather than overloading `dyncall`'s arg
list — a leading entry-name arg can't be told apart from a value).

**Why:** today one `.wamo` = one entry, so a "grammar library" means one
file per predicate. Multi-entry lets a related family ship as one object.
Purely additive; no dependency on the other items.

**Effort:** moderate (there's a surface fork to settle). **Depends on:** nothing.

### 4. Binary-data returns — structured records (deserialization) — *large, capstone*

**What:** let a grammar return a **compound term** (e.g. `rec(42, 3.14,
"name")`) that plawk **deserializes** into typed fields against a declared
return shape — `dyncall(...) as (i64 f64 s16)` → walk the compound's args,
type each (arg0 Integer→i64, arg1 Float→f64, arg2 Atom→sN), and materialize
into a record buffer so `$1`,`$2`,`$3` address it like any binary record.

**Why:** this is the *other half* of your binary-return idea — the case
that **does** need deserialization. It is also the endgame that closes the
loop with the DCG-binary-readers design (`PLAWK_DCG_BINARY_READERS.md`,
Tier 2/3): a grammar parses a payload (bytes in, via the existing blob
bridge) and returns a **structured record** (term out), which plawk lays
out into the same fixed access layout `BINFMT` describes. Bytes-in +
record-out = a grammar-driven reader for formats too irregular for the
native Tier-1 reader, without leaving the compiled loop.

**Effort:** large — a return-shape surface, a term-walking marshaller in
the call primitive, and typed materialization. **Depends on:** item 2 (the
byte-return primitive and the "output is a term, not a scalar" plumbing),
and benefits from item 1 (float constants) for grammars that build float
fields.

### 5. Source-`eval` via the compiler-as-`.wamo` bootstrap — *largest*

**What:** compile a grammar from **source text at runtime** — `g =
compile($1); total += dyncall_at(g, $2)` — by shipping the WAM compiler
itself as a `.wamo` and loading it on demand: `eval(source)` = load
`compiler.wamo` → run it on the source string → load the object bytes it
produces (via the in-memory loader from #3463).

**Why:** true query/userspace mode — define a grammar and run it live. The
`mtime` cache-invalidation path (#3465) is its natural home. Pay-for-what-
you-use holds: the compiler-object only loads when an `eval` surface is used.

**Effort:** largest. **Depends on:** a substantially expanded loadable
subset — the compiler leans on `findall`/`assert`/`read_term`/meta-call,
which are outside the subset today. Items 1 (float constants) and the
broader builtin/subset work are stepping stones; this is genuinely last.

## The binary-return question, specifically

Your instinct — "in some cases this might require deserialization" — is the
exact fault line:

- **Opaque bytes (item 2):** the grammar returns a byte string, plawk treats
  it as a slice/blob. **No deserialization** — the bytes are handed through
  unchanged (print, writebin, compare). This is the smaller, high-value
  first step.
- **Structured records (item 4):** the grammar returns a *term* whose shape
  plawk must interpret into typed fields. **This is the deserialization
  case** — a return-shape declaration drives a term-walker that materializes
  fields. Larger, and it's where the JIT arc and the binary-readers arc meet.

Both are new for compiled foreign calls too (the existing bridge passes
bytes *in* but returns only `i64`/`double`), so whatever we build for
`dyncall` naturally extends the compiled `name(args)` bridge as well.

## Cross-cutting notes

- **Performance invariant** (kept throughout): a program with no dynamic
  calls emits zero extra IR — every capability rides `emit_wamo_loader(true)`
  and per-site arity collection.
- **Subset expansion** (items 1, 5) is the through-line: each builtin/opcode
  added to the loadable subset makes richer grammars loadable and inches
  toward the compiler being self-hostable as a `.wamo`.
