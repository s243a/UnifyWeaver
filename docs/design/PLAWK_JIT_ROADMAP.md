<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# PLAWK JIT Roadmap

Where the runtime-loadable-grammar (JIT) arc stands, and what comes next.

> **Top-level map:** for a one-stop architecture summary of the whole
> eval arc — the layers, how they compose, and where each design doc
> fits — see [PLAWK_EVAL_ARCHITECTURE.md](./PLAWK_EVAL_ARCHITECTURE.md).

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

### 2. Binary-data returns — opaque bytes — **LANDED (all consumer positions)**

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

**Follow-ons LANDED — the slice now feeds every listed consumer:**
`writebin` into `sN`/`lpsN` slots (zero-filled clamped copy for fixed
slots, clamped pointer+length for lps payloads; a failed call writes an
empty payload), **equality guards** (`blob(dyncall...) == "literal"` as
a rule pattern — length check + memcmp, null-safe, usable wherever
`field_eq` patterns go), and **assoc keys**
(`counts[blob(dyncall...)]++` interns the returned bytes exactly like a
text-field slice; a failed call skips the increment like a missing
field). A generic blob-node walk feeds the shim-arity collectors from
every position (print fields, writebin slots, assoc keys, patterns), so
the eval surface composes too: `compile(...)` sources inside any blob
position ship the compiler object. The assoc apply path gained a
global-constant channel (marker lines partitioned into the rule's
existing guard-globals stream) so blob-key argument marshaling can emit
its per-arg constants. Tests: `tests/test_plawk_blob_consumers.pl`.

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

**`dyncall_at@name(Src, ...)` — LANDED (i64):** a named entry on a
*runtime* source. The loader now MATERIALIZES the `.wamo` entry-name
table into the loaded VM (`%WamEntryRow` rows on two new `%WamState`
fields, mirroring the milestone-2 meta-call table), and
`@wam_object_vm_entry_pc(vm, name, len)` resolves a name against an
already-loaded VM — no file re-scan, so it works uniformly for path
sources AND `compile(...)` handles (verified:
`dyncall_at@sq(compile("[(sq(...)...)]"), $1)` resolves the entry of a
runtime-compiled grammar). Resolution is per call (a short in-memory
scan; caching a PC by VM pointer would go stale across an mtime-mode
reload at the same address). One `@plawk_dyncall_at_named_<Sym>` shim
per Name-NArgs, in cached and off modes (the off variant frees the
fresh VM on the resolve-miss path too). **Multi-entry compiled handles
LANDED via `cgfullm/2`:** rather than changing `cgfull`'s header (whose
bytes every self-host golden compares against), the bootstrap module
gained a second entry sharing the whole compilation core
(`cgfull_core`) with a multi-entry serializer (`wzam_serialize`) — one
"name/arity" → group-label row per predicate. The plawk CLI ships
`cgfullm` as `<bin>.evalc.wamo`, so a `compile(...)` source holding a
grammar FAMILY exposes every predicate to `dyncall_at@name` (verified:
two grammars in one source, two named call sites, one content-deduped
handle → 194). Single-predicate sources serialize byte-identically to
`cgfull` (NE=1, first predicate named), so handles, dedup, and every
existing compile are unchanged — and `cgfull` stays the untouched
self-host fixpoint subject. The `float`/`blob` named-at variants have
LANDED too (`float(dyncall_at@name(...))` / `blob(dyncall_at@name(...))`
— the i64 shim shape with the f64/bytes call primitives, cached and
off modes), so the named-at surface covers all three return kinds.
Surface **B** below has landed as well (see next section).

**Surface B — declaration-bound library names — LANDED:** for a fixed
`DYNLOAD` shipping a known family, bind entries once at declaration and call
them like ordinary functions: `DYNENTRY parse, score` → `parse($1)`,
`float(score($2))`. Implemented as PARSE-TIME sugar: a declared name's
bare-call nodes rewrite to the named-entry nodes (`dyncall_named` /
`float_dyncall_named`), so the whole surface-A machinery — the shared
startup-resolved cached-PC resolver — carries B with zero new codegen.
**A is for the userspace/dynamic case (call site says `@name`, cost is
visible); B is for the pre-compiled library case (bare call, cost read
from the declaration).** The rule that makes B coherent: *entry
resolution inherits the object's sourcing* — fixed `DYNLOAD` ⇒ static
PC bake; a runtime source ⇒ per-load resolution (`dyncall_at@name`).

**Namespace rule for B (implemented as settled):** `DYNENTRY name` **reserves** `name` for
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

### 4. Binary-data returns — structured records (deserialization) — **LANDED (all target containers; posarray/assoc string values deferred)**

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
  such field). Verified: `dyncall@rec($1) as (i64 f64) { total += $1 }` sums
  the i64 field to 30; `{ sum += $2 }` sums the f64 field to 31.
  **Binds/views inside if-branches LANDED:** a record destructure or view
  can sit inside `if { ... } else { ... }`. The sequence walker already
  lowered `dynrec_bind` in any position; the gap was branch-body
  VALIDATION (`plawk_scalar_rule_body_plain_action` did not list the bind
  among a branch's allowed actions), now closed. Test:
  `tests/test_plawk_dyncall_rec_if.pl`.
  **Binds/views inside a `foreach` loop body LANDED:** a `foreach { ... }`
  over a record's repetition elements can call a grammar per element and
  destructure or view the returned record. The foreach body already lowers
  through the scalar action-sequence walker, so the branch-body validation
  fix carried it, and the record-view desugar now recurses into `foreach`
  (and `for_in`) bodies so a view there desugars in place — its block `$k`
  become the view's hidden temps while the view's Call args (e.g. `$1`
  passing the current element) ride the later foreach-element rebind. Test:
  `tests/test_plawk_dyncall_rec_loop.pl` (destructure, view, and
  view-in-if-in-foreach).
  **Recommended "iterate a collection, object per item" pattern:** a
  `foreach` over a repetition field whose elements have MORE THAN ONE
  field iterates TUPLES — element `$1, $2, …` are the tuple's fields (a
  (key, value) pair for `rep(i64 i64)`) — and the body can decode any
  field into a structured record via a grammar. So "loop and process each
  item as an object" is served today for list-shaped (repeating-field)
  data. Test: `tests/test_plawk_foreach_tuples.pl` (tuple fields;
  tuple→struct destructure; tuple→struct view).
  **The `for (k in arr)` assoc for-in stays print-only** by construction
  (its body is a per-key print plan, not a scalar action sequence), so
  record binds there remain a separate, larger surface. Iterating a HASH
  table with real per-entry work needs two read forms plumbed through the
  expression model — the loop key `k` as a readable value and the `arr[k]`
  value-lookup — plus loop-carried state for accumulation, and a surface
  answer for referencing the current value as a grammar argument. Deferred
  as a distinct feature; `foreach`-over-tuples covers the list-shaped case
  without it.
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
  `arr[2]=200`. **Atom/string keys LANDED:** a grammar returning
  `[Atom-V, ...]` pairs populates the table under the atom's
  global-registry id — the same keyspace text-mode field slices and blob
  keys intern into — so END `for (k in arr)` reports resolve the key
  names and `arr["literal"]` lookups intern to the same ids (verified:
  a bucketing grammar over text records reports `big 4 / small 2 /
  seen 4` and answers literal lookups). One key kind per table (integer
  keys collide with atom ids — the documented text-mode ambiguity).
  Landing this also fixed a latent text-mode bug: the dynassoc action
  emitter DISCARDED its arg-marshal globals (text-mode field args emit
  a fallback constant each), leaving the IR referencing undefined
  values; they now ride the apply path's global channel (added in the
  blob-consumers round). **Default-entry LANDED:**
  `arr = dyncall(args) as assoc` runs the DYNLOAD object's `wamo_entry`
  through `@plawk_dyncall_assoc_default_<N>` (entry PC recorded at
  object-load time, no resolver). **String VALUES LANDED — the second
  table kind:** `arr = dyncall[@name](args) as assoc(str)` declares a
  str-valued table; the grammar returns `[K-Atom, ...]` pairs,
  `@wam_object_call_assoc_str` stores each value's registry id via
  `@wam_assoc_i64_set` (insert-or-REPLACE — accumulating ids is
  meaningless, a repeated key keeps the latest label), and reads
  (for-in value prints, END `arr["literal"]` lookups) resolve the id
  back to text through `@wam_atom_to_string`, exactly as key prints
  already did. Same table layout; only the declared value kind differs.
  **Rule-body for-in LANDED:** `for (k in arr)` iterates a
  grammar-populated table inside the rule''s action chain (a per-record
  running snapshot), not only in END.
- **Positional array — LANDED (named + default entry, i64 AND str values):**
  `arr = dyncall[@name](args) as array` binds a FLAT returned list
  `[V1, ..., Vn]` into one array value by POSITION — element i at key i
  (1-indexed, the awk `split` convention). `@wam_object_call_posarray`
  walks the list into the shared i64 table via `@wam_assoc_i64_set`
  (REPLACE semantics, so the array reflects the most recent record), and
  the `posarray(...)` spec wrapper rides the whole assoc pipeline
  (planning, table, for-in, lookups) — only the shim name and the
  runtime walk differ. A positional table''s keys are integer positions,
  never interned atom ids, so int-key reads (`arr[1]`, and a for-in loop
  key) are unambiguous and permitted in TEXT mode too, unlike a regular
  integer-keyed assoc (which stays binary-only to avoid the atom-id
  collision). **String values (`as array(str)`):** a grammar returning a
  flat `[Atom, ...]` list gives a str-valued positional table — the
  element atoms' registry ids are stored by position
  (`@wam_object_call_posarray_str`), and reads (END `arr[1]`, for-in
  values) resolve them back to text through `@wam_atom_to_string`. Such a
  table is BOTH positional-keyed (int positions, int reads work in text
  mode) and str-valued (its name joins both the posarray set and the
  str-array set), completing the container × value-kind matrix. Tests:
  `tests/test_plawk_dyncall_posarray.pl`.

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

The **destructure** target still rejects string fields by design (no
scalar to bind them to — the record view is the string surface); the
**assoc** target takes interned atom keys AND, with the `(str)` value
kind, atom values (see above) — every string shape now has a container.

**Why:** this is the *other half* of your binary-return idea — the case
that **does** need deserialization. It is also the endgame that closes the
loop with the DCG-binary-readers design (`PLAWK_DCG_BINARY_READERS.md`,
Tier 2/3): a grammar parses a payload (bytes in, via the existing blob
bridge) and returns a **structured record** (term out), which plawk lays
out into the same fixed access layout `BINFMT` describes. Bytes-in +
record-out = a grammar-driven reader for formats too irregular for the
native Tier-1 reader, without leaving the compiled loop.

**Grammar-driven reader capstone — LANDED (proven, no new code):** the
endgame above composes today, and now has a standing end-to-end test
(`tests/test_plawk_grammar_reader.pl`). A `BINFMT` `blobN` field frames
a binary payload natively; the payload flows — as a transient atom,
constant-memory, no interning — into a grammar shipped as a `.wamo`
(`(s, c) = dyncall@parse($2) as (i64 i64)`), which parses the bytes with
a real WAM DCG (choice points and all) and returns a `pair(Sum, Count)`
compound; `@wam_object_call_record` deserializes that into the typed
plawk scalars. The novelty is routing bytes-in + record-out **through a
loaded object** rather than a compiled-in foreign predicate, so the
reader grammar is a shippable, swappable artifact. It fell out of the
existing pieces — item 2's blob→transient-atom arg marshalling
(`plawk_foreign_args_ir`, `≤1` blob per call) meeting this item's record
destructure (`dynrec_bind`, legal as a binary-mode action) — with no
connecting code needed; a scalar id field can still guard the record
while the grammar reads the payload.

**Fully-JIT reader — LANDED (runtime-compiled grammar):** the record
destructure now also works over a RUNTIME source
(`(s, c) = dyncall_at@parse(compile("[...]"), $2) as (i64 i64)`), so the
reader grammar itself is compiled from source text inside the running
binary (via the shipped bootstrap-compiler object) and loaded into a
fresh VM; its named entry resolves per record against that VM's
materialized entry table (`@wam_object_vm_entry_pc`). A new at-record
shim family (`@plawk_dyncall_at[_named]_rec_*`, cached + off cache modes)
threads the source `(path, len)` ahead of the boxed args into
`@wam_object_call_record`, mirroring the i64 `dyncall_at` call site; a
`compile(...)` handle travels as the `(null, handle-id)` pair the
registry getter already speaks. Kept separate from the i64/float/blob
at-collectors so a record-only entry emits no spurious scalar shim. The
reader source stays inside the bootstrap compiler's subset (a constant
char literal in a list head — e.g. a comma separator — is a documented
bootstrap-subset gap, so the test reader parses one number per record).
Test: `tests/test_plawk_jit_reader.pl`.

**Effort:** large — a return-shape surface, a term-walking marshaller in
the call primitive, and typed materialization. **Depends on:** item 2 (the
byte-return primitive and the "output is a term, not a scalar" plumbing),
and benefits from item 1 (float constants) for grammars that build float
fields.

### 5. Source-`eval` via the compiler-as-`.wamo` bootstrap — **LANDED (the payoff runs)**

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

**Status:** ALL SIX MILESTONES LANDED, and the payoff surface runs:
`dyncall_at(compile("<prolog source>"), $1)` compiles a grammar from
source text INSIDE the running binary — the CLI ships the self-hosted
bootstrap compiler (promoted to
`src/unifyweaver/targets/wam_bootstrap_compiler.pl`) as
`<bin>.evalc.wamo` when a program has compile sites, `@plawk_compile`
dedups by source text and hands out registry handles, and the existing
dyncall_at shims consume them as `(null, handle)`. See
[PLAWK_EVAL_BOOTSTRAP.md](./PLAWK_EVAL_BOOTSTRAP.md) ("The landed
surface") and `tests/test_plawk_eval_compile.pl`. The full plan and its
six milestones live in the same doc; item 5 was a **subset-expansion
campaign** with the bootstrap as its payoff, and each milestone was its
own PR(s).

- **Milestone 1 — clause indexing — LANDED, now REAL dispatch.** Every
  `switch_on_*` dispatch variant first entered the loadable `.wamo` subset
  as a nop-fallthrough — safe because the tier-2 compiler emits every
  indexing instruction *inline at the head of the predicate*, immediately
  before the `try_me_else` chain it dispatches into, so falling through
  runs every clause in order (correct, just unindexed). That let
  atom-keyed multi-clause predicates (pervasive in the compiler) load
  early. Post-fixpoint, `switch_on_constant`/`_a2` are REAL indexed
  dispatch in loaded objects: the writer carries their key→label tables
  in a trailing section (emitted only when non-empty — switch-free and
  bootstrap-emitted objects stay byte-identical), and the loader builds
  the same `%SwitchEntry` arrays the AOT dispatcher feeds
  `@wam_switch_on_constant`, so the shared step cases run them with no
  new dispatch code (7.6× on a 200-clause fact-table probe; the lift
  also fixed a latent quoted-key parsing bug shared with the AOT switch
  tables — see PLAWK_EVAL_BOOTSTRAP.md). `switch_on_term`/`_structure`
  and `*_fallthrough` remain nops, matching the AOT dispatcher.
- **Milestone 2 — `call/N` meta-call in objects — LANDED.** A meta-call
  encodes `op1 = -1`; dispatch resolves the runtime goal through a new
  **per-object meta-call table** (format version 2) hung off two new
  `%WamState` fields, so a loaded object can `call/N` its own predicates —
  atom goals and compound (partial-application) goals alike. Falls back to the
  host-global table for host VMs. The spine of any runtime-built goal.
- **Milestone 3 — the compiler's builtin closure — audit done, aggregates
  landed.** The audit sorted the compiler's constructs by how the tier-2
  compiler lowers them (`builtin_call`, aggregate opcodes, unimplemented
  builtins, library-predicate calls). Landed this PR: **aggregate control**
  `begin_aggregate`/`end_aggregate` into the loadable subset, so `findall`,
  `setof` and `bagof` over user-predicate goals load and run from a `.wamo`
  (setof/bagof via `inline_bagof_setof`, now the `.wamo` default). Milestone 3b
  adds `term_to_atom/2` (write direction) — a recursive term→text writer that
  works in loaded objects (byte-based cons detection) — and the **reader**
  (`read_term_from_atom/2`): a recursive-descent parser with operator
  precedence, variables, control operators, floats and quoted atoms. A loaded
  object can now parse whole clauses from source text. Milestone 3b-db (PR 1 +
  PR 2) adds a **dynamic clause store**: a process-global, malloc-backed clause
  database (survives the arena rewind) with `assertz`/`asserta`/`retractall`;
  calling dynamic facts via `call/1` AND via direct calls (`counter(N)`,
  rewritten to `call/1` at compile time); and nondet `retract/1` (a
  remove+unify+backtrack iterator). **Milestone 3c** adds `catch`/`throw`: a
  process-global side stack of catch frames (op1 sentinels -5/-6), no
  cross-object linkage needed. **PR 3** completes the store with **rule bodies**
  (`assertz((H :- B))`): a var-preserving clause copy (head↔body sharing, fresh
  vars per call) + a deterministic body interpreter over `,`/2, builtins, and
  predicate calls (incl. nested rules). See PLAWK_DYNAMIC_DB.md. The dynamic
  store is now feature-complete for the eval bootstrap. See the bootstrap doc
  for the full loadability matrix.
- **Milestone 4 — byte-buffer output from a grammar — LANDED.** A loaded
  grammar assembles a byte string at runtime (via the milestone-3 string/codes
  builtins) and returns it as an Atom; the host reads it back through
  `@wam_object_call_bytes` (`{ptr, len, ok}`). No new target IR — it composes
  existing primitives. This is the path the `eval`/`compile` surface hands
  assembled `.wamo` text back across.
- **Milestone 5 — eval/compile pipeline — LANDED (runtime).**
  `@wam_object_eval` chains the compiler run (`@wam_object_call_bytes` on a
  source arg) into `@wam_object_load_bytes`, so a grammar's emitted `.wamo`
  bytes load and run in the same process; `@wam_object_load_cached` lazy-loads
  and memoizes the compiler object (the `DYNCACHE` role). Verified end to end
  with a stand-in (echo) compiler: source text → emitted bytes → load → run →
  `42`. A real source-to-bytecode compiler is milestone 6.
- **Milestone 6** (self-host) — **COMPLETE — the fixpoint runs.** A **minimal** Prolog→`.wamo`
  compiler written in the loadable subset (not the full ~22 000-line host
  compiler), run through the existing `@wam_object_eval` pipeline. The key
  enabler: `.wamo` is a **text** format, so emitting it is string assembly —
  already proven loadable in milestone 4. Staged: **(A) a `.wamo` serializer in
  the subset — LANDED**; **(B) minimal codegen for one clause shape — LANDED**
  (`cgcompile/2` parses source text with the reader, walks the clause to
  instructions, and serializes; `p(R) :- R = 42` and `p(R) :- R is 6*7` compile
  from source and run to `42` end to end via `@wam_object_eval` — the first
  source→bytecode compile); **(C) predicate calls — LANDED** (`cgcprog/2`
  compiles a multi-clause program — a list of clauses — into a multi-predicate
  `.wamo` with per-clause labels and `execute(CalleeLabel)` tail calls, so one
  clause calls another; `[(main0(R):-helper(R)), helper(42)]` → `42`), plus
  **conjunction + register allocation — LANDED** (`cgconj/2`: `numbervars`→
  Y-registers, first/subsequent occurrence → `put_variable`/`put_value`,
  `builtin_call =/2`; `pconj(R):-Y=42,R=Y` → `42`, byte-identical to the host),
  plus **runtime arithmetic — LANDED** (`cgarith/2`: `Var is op(A,B)` →
  `put_structure` + `set_value`/`set_constant` + `builtin_call is/2`, with a
  functor table `NF>0`; `ca(R):-X is 6*7, R=X` → `42`, byte-identical to the
  host), plus **non-tail calls — LANDED** (the unified `cgfull/2`: multi-clause +
  labels + register allocation + conjunction + arithmetic + `call(Label,arity)`
  goals; a call whose callee computes, `main0` calls `add1(41,V)` with
  `add1(X,Y):-Y is X+1` → `42`; also enlarged the transient arena 1→16 MiB, which
  an allocation-heavy compiler grammar needs); **(D) STARTED — multi-clause
  predicates LANDED**: clause grouping + `try_me_else`/`retry_me_else`/`trust_me`
  chains with per-alternative labels, so backtracking dispatch and **recursion**
  compile from source (two-clause dispatch needing the second clause → `42`;
  recursive factorial `fact(3)` → `6`); **lists + atom table LANDED**: head
  patterns `[]`/`[H|T]` (`get_list` + `unify_*`), repeated head vars
  (`get_value` — fixing a latent silent-overwrite bug), list literals in call
  args (`put_list`/`set_*`/write-mode `put_structure`), and the atom-table
  section with reloc-1 atom constants; the classic sum-over-a-list compiles
  from source → `42`; **comparison guards + if-then-else LANDED**: comparison
  builtins as goals, and `( C -> T ; E )` via `try_me_else`/`cut_ite`/`jump`/
  `trust_me` with mid-clause else/join labels (codegen is now PC- and
  label-aware; max-of-two exercises both branches → `42`); **general structure
  patterns LANDED**: arbitrary compounds in heads and call args — flat, nested
  (X-temp deferral), pairs, and the compiler's own `enc/4` shape — all → `42`;
  **builtin goals LANDED**: ~37 whitelisted builtins (term inspection, text,
  lists, type checks) as staged-args + `builtin_call`, `=/2` upgraded to
  full-term operands, data atoms collected into the atom table; and **the
  FIXPOINT first slice LANDED** — the loaded bootstrap compiler compiled the
  source of its own Stage A serializer, and the doubly-compiled serializer
  reproduced the golden `.wamo` byte stream exactly (checksum 2263 = byte sum
  + length, matched against the Stage A implementation). The compiler has
  compiled its own back end. **GEN 3 LANDED** — the loaded compiler compiled
  a mini-COMPILER (reader as a compiled goal, `=..` clause decomposition, a
  dispatching ITE codegen decision, atom-table emission), and the
  doubly-compiled compiler compiled two golden programs byte-exactly
  (combined checksum 4676): source text in, correct object bytes out, two
  compile generations deep. **Nested arithmetic** now compiles (the
  `is`-expression is staged as an ordinary term through the structure
  builder), and the compiler **fails fast** on unsupported constructs
  (catch-all `throw/1` diagnostics in the goal/operand/head-arg walkers
  instead of a silent catastrophic-backtracking hang). **THE MIDDLE first
  slice LANDED** — the loaded compiler compiled a cut-free restatement of
  its own single-clause codegen (numbervars register allocation,
  first-occurrence init tracking, head/goal/expression compilation,
  functor-table collection), and the doubly-compiled codegen reproduced
  the production compiler's bytes exactly on an arithmetic clause
  (checksum 8755). The reader gained quoted functor applications, and the
  `'$VAR'` marker clauses now guard on integer arguments so source-level
  `'$VAR'` patterns compile as ordinary structures. **THE FRONT LANDED** —
  clause grouping, labels, and try/retry/trust chain building
  self-compiled; the doubly-compiled compiler compiled a multi-predicate
  program (facts, a two-clause chain, a predicate call, constants,
  arithmetic) byte-identically to the production cgfull (checksum
  13679). **THE WALKERS LANDED** — ITE codegen with labels and init-set
  intersection, comparison guards, and the builtin whitelist
  self-compiled and byte-exact loaded (22412); the X-temp deferral
  paths are byte-exact in SWI (33858) but blocked loaded on **finding
  no. 12 (root cause established)**: an instrumented trace proved that
  a post-success failure backtracks into the stale chain CP of a
  completed call, re-runs an OVERLAPPING dispatch clause on the same
  term, and the divergent re-execution observes a register Ref above
  the rewound heap top. The runtime half is FIXED: backtrack no
  longer rewinds heap_top (the heap is monotonic per top-level call,
  like the arena), eliminating the dangling-Ref class — arena compound
  slots are raw pointers the trail cannot cover, so heap rewind
  deallocated cells that surviving slots still referenced. The semantic
  divergence is also CLOSED: cgfull's FACTS emitted no environment, so
  their Y-register head variables clobbered the caller's Y window —
  invisible for tail-position fact calls (all earlier slices), fatal
  for the walkers' first non-tail fact call. Facts now allocate. The
  FULL walkers golden — deferral included — is byte-exact loaded
  (35309). **THE CAPSTONE LANDED — the fixpoint runs**: the AOT-compiled
  cgfull (gen1) compiles the compiler's own source (entry
  `main2(Src, W)` — the source is a runtime argument, no quine needed)
  to gen2 (36151 bytes); gen2, loaded, compiles the same source to
  gen3; **gen2 == gen3 byte-identical** (F(F) = F), and gen3 compiles a
  fresh golden byte-identically to the production `cgfull_term/2` —
  the self-compiled compiler is behaviorally the compiler
  (`selfhost_capstone_fixpoint`).
  The compile budget for the full self-compile is closed: the **chained
  arena** removed the memory cliff (blocks link on exhaustion and never
  move; marks are virtual offsets so mark/rewind work across growth), and
  the serializer's **difference-list linearisation** removed the quadratic
  time/allocation (an 11.9 KB source compiles loaded in 40 ms / 35 MB where
  the quadratic style took 20 s / 3.7 GB at half that size). The
  table-collection walk's dedup scan now uses the NATIVE memberchk
  builtin instead of an interpreted hand-rolled scan — on an atom-rich
  20 KB synthetic fact-table grammar (the pathological shape for table
  building) the loaded compile drops ~23% with byte-identical output;
  typical grammars are hundreds of bytes and compile in single-digit
  milliseconds, once per distinct source (the compile() surface dedups),
  so the remaining mildly-superlinear tail is not worth further
  restatement.
  The campaign keeps surfacing and fixing latent runtime bugs — **fifteen found
  so far**: a 64-register-file ceiling corrupting memory for large clauses;
  `get_structure` not comparing the functor; the choice-point saved-register
  block not widened with the register file (failed clause bodies leaked Y17+
  across backtrack); and `copy_term/2` aliasing instead of copying (Refs
  returned unchanged, no sharing preservation); and `get_list` read mode
  accepting any compound (no cons/arity check — `[H|T]` wrongly matched
  `foo(A,B)`); the loaded reader missing `=..`/`=\=`/`\==` operators; and
  `=../2` compose mode broken three ways (no deref of Ref-linked list
  spines, id-based atom payload used as a functor pointer, result bound to
  the register instead of through the Ref); and quadratic accumulator-append
  allocation exhausting the 16 MiB arena (the chained arena above); and a
  variable-identity collapse in two paths — `get_value` var-var left the
  two variables unlinked (a silent no-op unification), and `builtin_append`
  seeded its result tail with the collapsed Unbound sentinel when the second
  argument was a bare unbound variable — both fixed via
  `@wam_deref_keep_var`; and an uncaught throw behaving as a plain failure
  (`@backtrack` resumed into live choice points over the half-unwound
  state, spinning inside append on corrupted terms) — fixed with an
  explicit abort flag checked at backtrack entry; and the reader
  var-dict silently falling back to fresh-per-occurrence variables past
  128 distinct names (the self-hosted compiler miscompiled its own
  serializer) — fixed with a growable dict; plus the re-entry pair
  above (no. 12: monotonic heap + fact environments), and the capstone
  finding (no. 13): loaded arithmetic had **no error channel** — unknown
  functors evaluated to a benign 0, and the first-byte dispatch let
  unknown names *alias* real ops (`f(2)` ran as `floor(2)`) — fixed
  with an arith-error flag failing `is/2` and the comparisons, plus a
  full-name whitelist gate before dispatch. A follow-up closed the last
  aliasing residue *inside* the whitelist: `//` (integer division)
  shares its first byte with `/` and ran as float division — the `/`
  branch now checks the second byte and routes `//` to a truncating
  `sdiv` (SWI's default), with float operands and division by zero
  failing through the same error flag. Finding **no. 14** came from the
  subset-growth campaign's empty-findall case: the AGGREGATE frame
  never saved/restored `stack_size` — an inner goal that allocated an
  environment and then failed left its frame orphaned, so the caller's
  later `deallocate` popped the orphan (masked whenever every inner
  clause was allocate-free) — and `wam_finalize_aggregate` restored
  only 512 of the 2048 register bytes `begin_aggregate` saves (the
  finding-no.-3 narrow-block class again). Finding **no. 15** came from
  the catch/throw emission round: `wam_catch_setup` stored the
  fully-deref'd catcher in the catch frame, so an UNBOUND catcher — the
  common `catch(G, E, Rec)` shape — collapsed to the addressless Unbound
  sentinel, which the throw-side `wam_unify_value` cannot bind through
  (its bind arm requires a Ref); every ball sailed past a variable
  catcher to the uncaught halt. Masked because every prior catch test
  used a compound catcher (`myerr(V)` keeps its struct identity). The
  frame now stores the chain-end Ref (`@wam_deref_keep_var`, the same
  helper that closed the earlier variable-identity collapses). See
  [PLAWK_SELFHOST.md](./PLAWK_SELFHOST.md).

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
