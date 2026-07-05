<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk eval bootstrap (JIT roadmap item 5)

Compiling a grammar from **source text at runtime** — the endgame of the
dynamic-grammar arc. This is the plan and the milestone sequence; it is a
multi-PR effort, not a single change.

## The goal

```awk
BEGIN { DYNLOAD = "..." }
{ g = compile($1) ; total += dyncall_at(g, $2) }
```

`eval(Source)` / `compile(Source)` = compile a grammar from a source string
**inside the running binary**, then load and call it. Concretely:

1. Ship the WAM compiler itself as a `.wamo` object (`compiler.wamo`).
2. `eval(src)` loads `compiler.wamo` (once, lazily — the pay-for-what-you-use
   invariant: the compiler object only loads when an `eval` surface is used).
3. Run the compiler object on the source string → it returns the compiled
   grammar's **`.wamo` bytes** (a blob — the byte-return bridge from item 2).
4. Load those bytes with the **in-memory loader** `@wam_object_load_bytes`
   (#3463) → a `%WamState*` + entry PC.
5. Call the freshly compiled grammar like any `dyncall`.

Steps 3–5 already have their primitives: `@wam_object_call_bytes` returns a
byte slice, and `@wam_object_load_bytes` + `@wam_object_call_i64` load and run
a buffer. The `mtime` cache-invalidation path (#3465) is the natural home for
"recompile when the source changes." **The hard part is step 1**: getting the
compiler to run *as a loaded object*, which needs the loadable WAM subset
expanded to cover what the compiler leans on.

## Why it is genuinely last

The compiler is ordinary Prolog compiled to WAM, but it uses constructs the
`.wamo` loadable subset does not yet cover. Until the subset reaches the
compiler's needs, the compiler can be compiled *ahead of time* (as the host
does today) but not *loaded and run* from a `.wamo`. So item 5 is really a
**subset-expansion campaign** with the bootstrap as its payoff.

The loadable subset today: `try_me_else`/`retry_me_else` chains,
`get`/`put`/`set`/`unify` variable+value+constant (integer, atom, float),
`get`/`put_list`, `get`/`put_structure`, `allocate`/`deallocate`,
`call`/`execute` (**including `call/N` meta-calls** — milestone 2, landed),
`proceed`, `builtin_call`, `cut`/`get_level`/`cut_ite`, `jump`, **all
clause-indexing dispatch** — every `switch_on_*` variant, matched by prefix
so the `_fallthrough` and `_a2` forms are covered too — as nop-fallthroughs,
and **aggregate control** `begin_aggregate`/`end_aggregate` (milestone 3 —
`findall`/`setof`/`bagof`/`aggregate_all`).

### Why indexing dispatch is a safe nop

The tier-2 compiler emits every indexing instruction **inline at the head of
the predicate**, immediately before the `try_me_else` chain, e.g. for an
atom-keyed `col/2`:

```
switch_on_constant red:default green:L_col_2_2_body blue:L_col_2_3_body
try_me_else L_col_2_2
get_constant red, A1
...
L_col_2_2: retry_me_else L_col_2_3
...
L_col_2_3: trust_me
...
```

The switch is an *optimization*: its entries only point deeper into the same
`try_me_else`/`retry`/`trust` chain that follows it inline. Dropping the
switch and falling through runs every clause in order — correct, just
unindexed. The loader already did this for `switch_on_term`; extending it to
`switch_on_constant`/`_a2` (this PR) means grammars with atom-keyed clauses —
pervasive in the compiler — now load. Cost: unindexed dispatch in loaded
objects (an optimization gap, not a correctness one).

## Milestones (subset expansion → bootstrap)

Rough dependency order. Each is its own PR(s); the earlier ones are
independently useful (richer hand-written grammars load sooner).

1. **Clause indexing** — *LANDED (this PR).* `switch_on_constant`/`_a2` as
   nop-fallthroughs. Atom-keyed multi-clause predicates load.
2. **`call/N` meta-call in objects** — *LANDED.* A meta-call encodes
   `op1 = -1` (`op2` = total arity), exactly as the host AOT path does, and
   dispatches through `@wam_dispatch_meta_call`. The subtlety is that
   dispatch resolves a *goal* (atom/functor + arity) to a *label index*, and
   a loaded object's predicates are not in the host's compile-time dispatch
   table. So the format grew a **per-object meta-call table** (version 2): a
   trailing section listing each predicate as `(atomIdx, funIdx, arity,
   labelIdx)` into the object's own atom/functor tables. The loader
   materializes it into a `%WamMetaRow` array hung off two new `%WamState`
   fields (25/26); `@wam_dispatch_meta_call` consults the VM's table when
   present (via `@wam_meta_find_atom` / `@wam_meta_find_compound`) and falls
   back to the host-global arrays otherwise. Compound-goal matching works
   because a loaded object's runtime functor pointers are its own malloc'd
   copies — the very pointers the table stores — so identity holds within one
   object. The table is only emitted for objects that actually contain a
   meta-call (pay-for-what-you-use). This is the spine of any grammar that
   calls a goal built at runtime — an interpreter or compiler.
3. **The builtin closure the compiler needs** — *audit done; aggregates
   landed.* `builtin_call` is already in the subset, so a builtin works in a
   loaded object *if the host runtime provides it AND the construct is in the
   loadable subset*. The audit sorted the compiler's constructs into four
   buckets by **how the tier-2 compiler lowers them**, which decides what
   "make it loadable" means:

   | Construct | Lowers to | Loadable now? |
   |---|---|---|
   | `is`/`=`/`==`/comparison, `functor`/`arg`/`=..`, `copy_term`, `atom_codes`/`number_codes`/`char_code`, `sub_atom`/`atom_concat`, `sort`/`msort`/`keysort`, `length`/`append`/`nth`/`reverse`, `sum_list` … | `builtin_call <id>` (id in `builtin_op_to_id`, host-implemented) | **yes** — already in the subset and loader-safe (operate on VM regs/heap + libc) |
   | `findall`/`aggregate_all`, `setof`/`bagof` | `begin_aggregate`/`end_aggregate` (tags 28/29) | **yes, this PR** — opcodes lifted into the subset; setof/bagof additionally need `inline_bagof_setof(true)`, now the `.wamo` default |
   | `assert`/`asserta`/`assertz`/`retract`, `term_to_atom`/`read_term`/`read_term_from_atom` | `builtin_call <name>` with **no `builtin_op_to_id` entry and no host runtime impl** (writer would emit the unknown-id 200) | **no** — genuinely unimplemented; needs new runtime builtins (milestone 3b). `assert`/`retract` against a loaded object's own clauses is the subtle one — mutable clause state in an arena-rewound VM |
   | `catch`/`throw` | `execute catch/3` — a call to a runtime *library predicate*, not a builtin | **no** — the loaded object references `catch/3`, which lives in the host, not the object; needs cross-object/host predicate linkage (milestone 3c) |

   **Landed here:** the aggregate opcodes, verified end-to-end — `findall`,
   `setof`, `bagof` over a *user predicate* goal load and run from a `.wamo`
   (the case a compiler actually hits, iterating over clauses). Also note a
   **pre-existing host bug** surfaced by the audit: a *backtracking list
   builtin as a findall goal* (e.g. `findall(X, member(X, L), _)`) segfaults
   even in AOT-compiled host code — orthogonal to the loader, filed
   separately, and not a blocker since compiler-style `findall` iterates over
   predicates, not `member`.

   **Remaining (3b/3c):** `assert`/`retract`/`term_to_atom`/`read_term` runtime
   builtins, and `catch`/`throw` predicate linkage. These are the true long
   pole for self-hosting the compiler.
4. **Byte-buffer output from a grammar.** The compiler object must *emit*
   `.wamo` bytes. It returns them as an Atom/byte string (the item-2 blob
   bridge already carries bytes out); building that byte string inside the
   grammar needs the string/codes builtins from milestone 3.
5. **The `eval` / `compile` surface + pipeline.** Wire the plawk surface:
   `compile(src)` → run `compiler.wamo` on `src` (blob out) →
   `@wam_object_load_bytes` → a handle usable by `dyncall_at`. Lazy-load the
   compiler object; cache per `DYNCACHE`.
6. **Self-host.** Compile the actual WAM compiler to a `.wamo` and run the
   whole pipeline from source text end to end. The capstone.

## Cross-cutting notes

- **Pay-for-what-you-use holds:** the compiler object (large) loads only when
  an `eval`/`compile` surface is compiled into the program, riding
  `emit_wamo_loader(true)` like every other dynamic capability.
- **Correctness-first, speed-later:** loaded objects run unindexed (milestone
  1). If the eval loop becomes hot, a real switch-table in the loader is a
  later optimization — the format already carries the switch operands we
  currently drop.
- **Milestones 1–4 are the long pole**; 5–6 are plumbing over primitives that
  already exist (`@wam_object_load_bytes`, `@wam_object_call_bytes`, the
  `mtime` cache).
