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
5. **Grammar-driven reader through a LOADED object (landed):** the same
   `blobN` payload can flow into a grammar shipped as a `.wamo` rather
   than a compiled-in predicate, and that grammar can return a
   **structured record**: `(s, c) = dyncall@parse($2) as (i64 i64)`
   frames the payload natively, marshals it as the transient atom (same
   `@wam_transient_atom_from_bytes` path, one blob per call), and
   `@wam_object_call_record` deserializes the returned compound into the
   typed scalars. This is the JIT-roadmap item-4 endgame — bytes-in +
   record-out — with the reader grammar as a swappable artifact (the
   object can be a shipped `DYNLOAD` library or a runtime `compile(...)`
   handle). It composes from the item-2 blob bridge and item-4 record
   destructure with no new codegen; standing test
   `tests/test_plawk_grammar_reader.pl`. Tier-3 (a full DCG engine in
   the loop) remains the only unbuilt rung, and stays deliberately out
   of scope — the loaded-grammar reader covers the irregular-format
   cases without one.

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

## Phase 5 (JIT) slice 1 (landed): runtime-loadable WAM objects (`.wamo`)

The Tier-3 note above imagined a grammar *loaded at runtime* that
compiles through the same WAM→LLVM path. Slice 1 delivers the loading
half: a `.wamo` object is a self-contained, position-independent WAM
program that any host binary (a module built with
`emit_wamo_loader(true)`) reads at runtime and executes on the shared
WAM runtime. The grammar is compiled ahead of time; the host swaps it
in without recompiling. `examples/plawk/bin/plawk` already carries the
full runtime, so a plawk binary is the natural host.

**Format** (`write_wam_object/3`, `wam_object_encode/3`) — a
whitespace-separated token stream (`WAMO\n`, version, entry-label index,
then length-prefixed atom/functor tables, label PCs, and `tag op1 op2
reloc` code quads). Length-prefixed strings mean the native loader needs
no escaping, which keeps it small and robust.

**The three relocation classes** are the whole trick. A compiled WAM
instruction can't carry absolute addresses across a process boundary, so
each operand that would be one is tagged:

- **atom** — a constant's atom id is meaningless in another process. The
  object stores the atom *string* and its table index; at load time the
  loader re-interns via `@wam_intern_atom` (which copies the string) and
  patches the runtime id into the operand. Atom equality is by id, and
  interning is deduplicating, so identity is preserved.
- **functor** — `get/put_structure` operands are functor *pointers*, and
  the runtime compares functors by pointer. The loader makes exactly one
  `malloc`'d copy per functor name and points every referencing
  instruction at that single copy, so structure unification inside the
  object stays correct. (Arithmetic — `is`, comparisons — inspects
  functor *bytes*, so the copy is fine there too.)
- **none** — everything else is already self-relative.
  `call`/`execute`/`try_me_else`/`retry_me_else` operands are indexes
  into the object's *own* labels array, and `@wam_label_pc` reads the
  VM's installed labels — so they need no relocation. `builtin_call` ids
  are host-stable per UnifyWeaver version.

**Loadable subset.** Tier-2-style grammars (try_me_else chains, no
switch tables) use only: get/put/set/unify variable+value+constant,
get/put_list, get/put_structure, allocate/deallocate, call/execute,
proceed, try_me_else/retry_me_else/trust_me, builtin_call, cut_ite,
get_level/cut. First-argument *type* switches (`switch_on_term` and
friends) lower to nop-fallthrough exactly as the interpreter does — the
`try_me_else` chain still runs, just unindexed. Outside slice 1 and
rejected loudly at write time: float constants (tag 2),
`switch_on_constant` tables (need a global table), and `call/N`
meta-calls (need the apply machinery).

**Runtime.** `@wam_object_load(path)` reads the file, re-interns atoms,
copies functor strings, patches operands, `malloc`s the code + label
arrays, and returns `{ %WamState*, entry_pc }`. `@wam_object_call_i64`
mirrors the plawk foreign bridge: push an unbound output cell, set the
argument registers, `@run_loop`, read back the Integer result, and
rewind the arena (heap-top save/restore) so repeated calls run in
constant memory. See `tests/test_wam_object.pl` — the round-trip test
builds a host, writes a sum grammar, and the host computes `119` from an
object it never saw at compile time; the swap test runs a *second*
grammar (`1020`) through the *same* host binary with no rebuild.

**plawk surface (`dyncall`).** A plawk program opts into a runtime object
with `BEGIN { DYNLOAD = "file.wamo" }` and calls into it with
`dyncall(args...)`, which yields the entry's integer result (or 0 on
load/call failure). `dyncall` is a reserved call form that parses to its
own AST node — it never touches the compiled-foreign-call machinery, so
the spelling itself marks the runtime-JIT boundary (the object file can be
absent or swapped, unlike a compiled call). Codegen emits one
`@plawk_dyncall_N` shim per arity that boxes N `%Value` args and calls
`@wam_object_call_i64`; the object loads lazily on the first `dyncall`
(mirroring `@plawk_foreign_vm_get`) and is reused, so no driver-startup
plumbing is needed. The CLI turns on `emit_wamo_loader(true)` whenever a
program uses `dyncall`. The entry is read as `entry(A0..A_{N-1},
out=A_N)`, so a grammar predicate has arity N+1 (N inputs + one output).
`tests/test_plawk_dyncall.pl` builds a binary that sums `dyncall($1)` over
i64 records (sum of squares = 150), then swaps `square.wamo` for a
doubling grammar and reruns the *same binary* → 44, no rebuild.

One thing kept open for later: a `.wamo` currently exposes a single
entry, so `dyncall(args)` is unambiguous. Multiple entry points would bind
at declaration (e.g. a named-entry directive) rather than overloading
`dyncall`'s argument list, since a leading entry-name arg can't be told
apart from a value.

**Dynamic source (`dyncall_at`).** Where `dyncall` binds one fixed object
at compile time, `dyncall_at(Source, args...)` chooses the object *per
call* from a runtime value: `Source` is a field or string literal naming a
`.wamo` path, `args...` are the entry inputs. The source is marshalled to
a NUL-terminated path (a field slice is copied into a stack buffer, capped
at 4095 bytes; a literal becomes a constant global) and everything else
mirrors `dyncall`. Object management is set by `BEGIN { DYNCACHE = "..." }`
— a compile-time constant, so the mode is baked into the emitted shim, not
dispatched at runtime:

- **`on`** (default) — a fixed-capacity (64) cache keyed by the interned
  path id. Each distinct grammar loads once via `@wam_object_load` and is
  reused; per-call cost is one intern + a linear scan, and memory grows
  only with the number of *distinct* grammars (beyond 64, load without
  caching). Ideal for "pick one of N grammars by a tag."
- **`mtime`** — like `on`, but the cache also keys on the file's
  modification time (`st_mtim` at offset 88/96, combined to nanoseconds,
  matching `time_file/2`). Recompiling the `.wamo` bumps its mtime, which
  busts the cached entry (freeing the stale VM via `@wam_state_free`) and
  reloads — so a query/userspace **redefinition** takes effect with no
  reload-per-call tax and no rebuild of the host binary. The nanosecond
  key catches sub-second edits.
- **`off`** — no cache: load fresh and `@wam_state_free` after every call.
  Always current, correct for genuinely-changing sources, but pays a full
  parse+relocate+build per call (fine when calls are rare, wrong in a hot
  loop).

`tests/test_plawk_dyncall_at.pl` covers all three: `on` picks the grammar
by a filename column and reuses a repeated file (sum 30 = 7+9+7+7), `off`
reloads each call, and `mtime` returns 7, then 11 after `g.wamo` is
redefined — same binary, no rebuild.

**Float returns (`float(dyncall(...))`).** `dyncall`/`dyncall_at` read the
entry output as an Integer; `float(dyncall(...))` and
`float(dyncall_at(...))` read it as a double via a `@wam_object_call_f64`
primitive (`@value_is_number` + `@value_to_double`), exactly like the
compiled bridge's `float(name(args))`. This matters two ways: it puts an
integer result into the f64 lane so plawk-side float math applies, and —
crucially — a grammar whose output is a *Float* (e.g. `R is X / 2`, since
runtime `/` yields float) is **unreadable by the integer form**, which
demands tag=1 and returns 0. `tests/test_plawk_dyncall_float.pl` pins the
contrast: for a `half(X,R):-R is X/2` grammar and input 7, `dyncall($1)`
yields `0` while `float(dyncall($1))` yields `3.5`; `float(dyncall_at($1))`
does the same over a dynamic source.

**Float constants in grammars (landed).** `put_constant`/`set_constant`
now accept float literals, so a grammar may write one directly (`R is X *
1.5`), not only reach a Float by computation. The object stores the float's
decimal text (C-string table, reloc class `float`) and the loader `strtod`s
it, reproducing the exact double the AOT `bitcast` would. `scale(3)` via
`float(dyncall($1))` yields `4.5`.

## Why not a real DCG engine in the loop?

Because the loop's performance contract is the whole point of PLAWK:
the fixed-record loop measured 5.6× faster than mawk precisely because
there is no dispatch, no allocation, and no interpretation per record.
Tier-1 grammars compile to the same shape of code a C programmer would
write by hand for that format. Anything that genuinely needs
unification or backtracking already has a home — the WAM interpreter
in the same binary — and the bridge between the tiers is a function
call, not an architecture change.
