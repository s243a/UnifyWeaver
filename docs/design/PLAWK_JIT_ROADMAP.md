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
| `blob(dyncall(...))` / `blob(dyncall_at(...))` (opaque byte returns) | #3470 |
| Multi-entry objects — writer `wamo_entries([...])` + loader name resolution (`@wam_object_entry_index`) | #3471 |
| plawk surface A — `dyncall@name(...)` (named entry, compile-time-fixed, cached PC) | #3473 |
| `float(dyncall@name(...))` / `blob(dyncall@name(...))` (named double / byte returns, shared resolver) | #3474 |
| Structured returns — `@wam_object_call_record` (deserialize a returned Compound's args into typed i64/f64 slots) | #3475 |
| Destructure surface — `(a, b) = dyncall[@name](args) as (i64 f64)` binds fields to typed scalars | #3476 |
| Record-view surface — `dyncall[@name](args) as (i64 f64) { … $1 $2 … }` reads the return like the current record | #3477 |
| String/atom record fields (mechanism) — `@wam_object_call_record` typecode 2 → `(ptr,len)` via `out_slots`+`out_lens` | #3478 |
| String fields in the record view — `... as (i64 string) { print $2 }` (slice from hidden ptr/len temps) | #3479 |
| Assoc-return (mechanism) — `@wam_object_call_assoc` walks `[K-V,...]` into an i64 assoc table | #3481 |
| Assoc-return surface — `arr = dyncall@name(...) as assoc` populates a plawk assoc array; END `arr[k]` reads it | this PR |

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

### 1. Lift float constants into the loadable subset — *LANDED*

**What:** a grammar clause like `scale(X, R) :- R is X * 1.5` (a `float`
constant, tag 2) now compiles into a `.wamo`. `put_constant`/`set_constant`
accept floats (matching AOT `set_constant_literal_parts`); the object stores
the float's decimal text in the C-string table (reloc class `float`, id 3)
and the loader `strtod`s it at load, writing the i64 bit pattern into op1 —
the same double the AOT `bitcast (double c to i64)` yields.
`get_constant`/`unify_constant` stay integer/atom only (AOT never emits a
float there). Verified: `scale(3)` via `float(dyncall($1))` yields `4.5`.

**Why:** `float(dyncall(...))` made float-returning grammars first-class, but
a grammar could only *reach* a Float by computation (`R is X / 2`), not by
writing one. This closes that gap and is the first step of the subset
expansion that item 5 (source-eval) needs.

### 2. Binary-data returns — opaque bytes — *LANDED (print position)*

**What:** a grammar returns a **byte string** (its entry binds the output
to an Atom, a byte string here), read by `blob(dyncall(...))` /
`blob(dyncall_at(...))`. The `@wam_object_call_bytes` primitive checks the
output tag is Atom, reads `@wam_atom_to_string` + `strlen`, and returns
`{ i8* ptr, i64 len, i1 ok }` — the pointer is into the persistent atom
table, so it survives the arena rewind. In plawk the result is a byte
**slice** (`%Base_ptr`/`%Base_len`), printed via `%.*s` (empty on
failure). `blob(dyncall($1))` echoing a text field, and
`blob(dyncall_at($1))` over a dynamic source returning `hello`, both
verified. NUL-free by the blob convention. Also lifted `jump` (tag 32, a
self-relative label) into the loadable subset so if-then-else grammars
compile.

**Why:** the "binary data return" in its **no-deserialization** form — the
bytes are opaque, consumed as a string/blob. Opens grammars that *emit*
encoded/textual output rather than a single number, and is the foundation
for item 4.

**Still to do within item 2:** the slice currently plugs into `print`;
`writebin` into an `sN`/`lpsN` slot, equality guards, and assoc keys reuse
the same `(ptr,len)` shape and are the natural follow-on (each needs the
blob node wired into that consumer's path).

### 3. Multi-entry objects — *LANDED (mechanism + surface A)*

**What (mechanism):** one `.wamo` can expose several named entry
predicates. The writer takes `wamo_entries([P/A, ...])` and emits a
name→label-index table early in the stream (right after the default-entry
index), so `@wam_object_load` steps past it with a tiny skip loop and pays
nothing at call time. Two loader primitives resolve a name to its label
index: `@wam_object_entry_index_bytes(buf, total, name, namelen)` scans the
table in an already-read buffer (reads only the early table, stops at the
first match — never touches the code section), and
`@wam_object_entry_index(path, name, namelen)` is the path convenience
(reads the file, scans, frees). `@wam_label_pc` turns the returned label
index into a PC to call against the loaded VM.

**What (surface A — `dyncall@name`):** `dyncall@square($1)` /
`dyncall@cube($1)` over a single `DYNLOAD` selects a named entry at the
call site. The `@name` is a compile-time token, so the shim
(`@plawk_dyncall_named_<name>_<N>`) resolves the entry's label index once
at startup via `@wam_object_entry_index`, caches the PC in a per-entry
global (sentinel `-1` = unresolved), and every later call skips straight to
the object call — no per-call dispatch. A name no entry exposes yields `0`
(the shim's resolve-fail path). Rides `emit_wamo_loader(true)` + per-site
entry collection, so a program with no named calls emits none of this IR.
Verified end to end: one object exposing `square/2` and `cube/2`, summed by
name in one binary → `c - s = 22` (reachable only if both resolved).

**float / blob named — LANDED:** `float(dyncall@name(...))` reads a named
entry's numeric output as a double, `blob(dyncall@name(...))` reads its Atom
output as a byte slice — mirroring the bare `float`/`blob` forms. All three
return kinds (i64 / double / bytes) for one entry share a **single** per-entry
PC resolver `@plawk_dyncall_resolve_<name>_<N>` (the shims differ only in
which `@wam_object_call_*` they call), so an entry used in more than one
return position resolves once and shares the cached PC — no duplicate globals.
Verified: `float(dyncall@halve($1))` summing `N/2` over 3,5 → `4`;
`blob(dyncall@greet($1))` echoing a field.

**Deferred follow-on:** `dyncall_at@name(Src, ...)` (named entry on a
*runtime* source), which needs the PC cached per (object, name) pair rather
than per entry. And surface **B** below.

**Surface B — declaration-bound library names (planned):** for a fixed
`DYNLOAD` shipping a known family, bind entries once at declaration and call
them like ordinary functions: `DYNENTRY parse` → `parse($1)`. Cleaner call
sites than A, and resolution stays static (baked PC) because the object is
compile-time-fixed. **A is for the userspace/dynamic case (call site says
`@name`, cost is visible); B is for the pre-compiled library case (bare
call, cost read from the declaration).** The rule that makes B coherent:
*entry resolution inherits the object's sourcing* — fixed `DYNLOAD` ⇒ static
PC bake; a runtime source ⇒ per-load resolution. Build B when a real
multi-entry library exists and A's `@name` call sites feel noisy.

**Namespace rule for B (settled):** `DYNENTRY name` **reserves** `name` for
the compiled object — it removes that identifier from userspace, so the two
name sets are disjoint *by construction*. A bare `name(...)` call resolves
by set membership: in the `DYNENTRY` set → compiled entry; otherwise →
userspace (foreign call / builtin / user function). A rule freely mixes both
(`parse($1)` compiled, `score($2)` userspace) with no per-call ambiguity.
Declaring `DYNENTRY` over a name that is already a builtin/foreign name is a
**compile error**, never silent shadowing — so B never calls a userspace
name. **Optional guaranteed-no-shadow prefix — two directions:** a prefix can
carve compiled and userspace names into disjoint lexical zones, and which
side gets the prefix depends on the program's center of gravity:

- *Declaration-side (reserve):* `DYNENTRY parse` pulls `parse` out of
  userspace; bare `parse($1)` is compiled. Default-compiled, opt-out to
  userspace — for a grammar-library program where most calls are compiled.
- *Call-site-side (reach in):* userspace stays the default; a prefix like
  `static.parse($1)` (or a configurable `DYNPREFIX`) *reaches into* compiled
  space per call and statically guarantees no collision. Default-userspace,
  opt-in to compiled — for a mostly-userspace program that occasionally calls
  a compiled entry.

The two are not exclusive: a program can reserve the names it uses heavily
and prefix the occasional one. Same static-safety-vs-brevity trade B itself
makes, offered per program in whichever direction fits.

**Why:** today one `.wamo` = one entry, so a "grammar library" means one
file per predicate. Multi-entry lets a related family ship as one object.
Purely additive; no dependency on the other items.

**Effort:** A landed. B is moderate (declaration table + a resolution pass +
the reserved-name check). **Depends on:** nothing.

### 4. Binary-data returns — structured records (deserialization) — *mechanism + destructure surface LANDED (numeric fields)*

**What:** let a grammar return a **compound term** (e.g. `rec(42, 3.14,
"name")`) that plawk **deserializes** into typed fields against a declared
return shape. Because deserialization means *choosing the target type*, the
same walked compound can materialize into more than one plawk container
(see "Target containers" below) — the first shipped target is a set of
typed scalar variables.

**Mechanism landed:** the native primitive
`@wam_object_call_record(vm, pc, nargs, args, out_reg, nfields, typecodes,
out_slots)`. It runs the entry, requires the output cell to deref to a
**Compound** of arity `nfields`, and deserializes each arg into `out_slots[i]`
per `typecodes[i]` (`0` → i64, arg must be Integer; `1` → f64, arg must be a
number, stored as double bits) — **before** the arena rewind, since the
compound and its arg cells live in the arena (atoms survive; a plain
`@wam_object_call_*` would rewind them away). Same heap-save/rewind discipline
as the scalar variants, so a per-record call stays constant-memory. This is
the "output is a term, not a scalar" plumbing item 2 anticipated.

**Destructure surface landed:** `(v1, ..., vn) = dyncall[@name](args) as
(T1 ... Tn)` binds each returned field to a typed scalar variable — field i
lands in `vi`, an i64 scalar for `T=i64` or an f64 scalar for `T=f64`. The
record shims (`@plawk_dyncall_rec_N` / `@plawk_dyncall_named_rec_<Sym>`)
forward the call-site typecodes + a stack slot array to the primitive, then
each field is loaded and threaded into the variable's scalar slot like any
assignment (so `total += n` after a bind works). A failed call zeroes the
slots. Verified end to end: `rec(X) -> pair(X, X+0.5)`, `(n, half) =
dyncall@rec($1) as (i64 f64)` over 10,20 → `total=30`, `sum=31.0`. Rides
`emit_wamo_loader(true)` + per-site collection (no IR when unused).

**Target containers (the "choose the type" generalization):** a return shape
is really a *marshalling target*, and the walked compound can land in
different plawk containers — this is the through-line for the rest of item 4:

- **Typed scalars** (`(a,b) = ... as (i64 f64)`) — *landed*.
- **Record view / field reindex** (`... as (i64 f64) { $1 $2 ... }`) —
  *landed*. The returned compound reads like the current record inside the
  block: `$k` accesses field k. Implemented by **desugaring** to a
  destructure into hidden per-site temporaries plus the block body with
  every `$k` (1≤k≤nfields) rewritten to the k-th temporary — so it rides the
  destructure machinery with no field-pointer repoint. A body field outside
  1..nfields (including `$0`) leaves the view uncompilable (the record has no
  such field). Recurses into if-branches; a view nested in a for-in body is a
  follow-on. Verified: `dyncall@rec($1) as (i64 f64) { total += $1 }` sums
  the i64 field to 30; `{ sum += $2 }` sums the f64 field to 31.
- **Associative array** (`arr = dyncall@name(...) as assoc`) — *mechanism +
  surface LANDED (named entry, integer keys).* A grammar returning a list of
  pairs (`[K1-V1, K2-V2, ...]`) materializes into an i64 assoc table via
  `@wam_object_call_assoc(vm, pc, nargs, args, out_reg, %WamAssocI64Table*)`:
  it walks the cons list (functor `[|]`, identified by `strcmp` against the
  module's cons-functor string, since the loaded object's functor pointers
  are its own malloc'd copies), takes each arity-2 element's (Integer key,
  Integer value), and `@wam_assoc_i64_inc`s it into the caller's table —
  before the arena rewind, like the record primitive. `@wam_assoc_i64_*` is
  always in the module (WAM helpers), so no new runtime coupling. **Surface:**
  the desugar `dynassoc_bind(var(arr), Call)` becomes a per-record assoc
  action in the BINFMT assoc driver — a `@plawk_dyncall_assoc_<Sym>` shim
  fills `arr`'s table each record, keyed into the same assoc plan as
  `arr[k]++`, so END `arr[k]` lookups see the accumulated result. Verified:
  `tally($1) -> [1-$1, 2-100]` over inputs 5,7 gives `arr[1]=12`,
  `arr[2]=200`. **Deferred:** the default-entry `dyncall(...) as assoc` (named
  only for now), for-in iteration over a grammar-populated table, and
  atom/string keys (interned — an extension of the integer-key mechanism).
- **Positional array** — fields by numeric index into one array value.

Each target reuses one marshaller over the same walked compound; they differ
only in where fields are written (scalar slots, the line record, an assoc
table).

**String/atom fields — mechanism + record-view surface LANDED:** the
cross-cutting field type. `@wam_object_call_record` has typecode `2` (string)
and an `out_lens` array: a string field's atom is read as
`@wam_atom_to_string` + `strlen`, `out_slots[i]` gets the pointer (into the
persistent atom table, so it survives the arena rewind, like `blob`) and
`out_lens[i]` gets the length — a `(ptr,len)` byte slice per field. Numeric
fields set `out_lens[i]=0`.

The **record-view surface** exposes string fields: `dyncall@info($1) as
(i64 string) { total += $1 ; print $2 }` sums the i64 field and prints the
string field per record. Because plawk scalars are numeric-only, a string
field can't bind to a single scalar variable; instead the view desugar binds
a string field to a **(ptr,len) pair of hidden i64 temps** and rewrites its
`$k` to a `blob`-style byte slice built from those two scalars (printed
`%.*s`, empty on a failed call). Numeric fields stay single scalar temps.
This also required the rule-body `print` path to substitute scalar-slot
reads (it didn't before), which now lets any rule-body print reference a
scalar. Verified: `info(X) -> tag(X, big/small)` over 5,200,7 prints
`small/big/small` and sums `$1` to 212.

The **destructure** target still rejects string fields (no scalar to bind
them to); the **assoc** target (string values / interned keys) is the
remaining container.

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

### 5. Source-`eval` via the compiler-as-`.wamo` bootstrap — *largest, STARTED*

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

**Status:** design + first subset increment landed. The full plan and its
six milestones live in [PLAWK_EVAL_BOOTSTRAP.md](./PLAWK_EVAL_BOOTSTRAP.md).
Item 5 is a **subset-expansion campaign** with the bootstrap as its payoff;
each milestone is its own PR(s) and richer hand-written grammars become
loadable along the way.

- **Milestone 1 — clause indexing — LANDED.** All clause-indexing dispatch
  (`switch_on_term`, `switch_on_structure`, `switch_on_constant`,
  `switch_on_constant_a2`) is now in the loadable `.wamo` subset as
  nop-fallthroughs. Safe because the tier-2 compiler emits every indexing
  instruction *inline at the head of the predicate*, immediately before the
  `try_me_else` chain it dispatches into; dropping the switch and falling
  through runs every clause in order — correct, just unindexed. This lets
  atom-keyed multi-clause predicates (pervasive in the compiler) load.
- **Milestones 2–6** (meta-call in objects, the compiler's builtin closure,
  byte-buffer output, the `eval`/`compile` surface, self-host) — see the
  bootstrap doc.

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
