# Dynamic clause store (assert / retract) for the WAM/LLVM target

Status: **PR 1 landed — ground dynamic facts, called via `call/1`.** This is
milestone **3b-db** of the eval bootstrap (see PLAWK_EVAL_BOOTSTRAP.md). It is
the last genuinely-new runtime machinery before the eval/compile surface: a
mutable clause database in a VM whose term arena is rewound between queries.

## Why it is the subtle one

Every other term the VM builds lives in the **arena** — a bump allocator that
`@wam_arena_reset`/`@wam_arena_rewind` roll back between queries and on
backtrack. A clause you `assertz` must outlive all of that: it has to survive
the arena rewind, later backtracks, and any number of subsequent queries.
`@wam_copy_term_value` (the existing deep-copier) allocates from the arena, so
it is *not* durable — an asserted clause copied that way would be freed under
the next query. The store therefore needs its own **malloc-backed, durable**
storage, following the same growable-global-table idiom the runtime-interned
atom table uses (`@wam_atom_dyn_ptr` / `_count` / `_cap`, realloc-doubling,
per-entry malloc that the table owns).

## Storage model

A single process-global table (in `state.ll.mustache`, alongside the atom
table):

```
@wam_dyn_ptr   = global %WamDynClause* null   ; the growable array
@wam_dyn_count = global i32 0
@wam_dyn_cap   = global i32 0
```

with one row per asserted clause:

```
%WamDynClause = type { i8*, i32, i32, %Value* }  ; functor, arity, live, args
```

- **functor**: a malloc'd copy of the predicate's functor name (so the row does
  not depend on the lifetime or provenance of the source functor pointer —
  AOT `@.fn_*` global, atom-table entry, or reader-built are all fine). Matching
  uses `@wam_functor_eq` (pointer fast-path + `strcmp` fallback), so a
  goal built from any source unifies against the stored copy.
- **arity**: base arity of the head.
- **live**: 1 = present, 0 = retracted. Retraction is a **lazy tombstone** —
  the row stays in the array (so a `retract` mid-iteration does not shift the
  indices the consult iterator is walking) and is skipped on scan.
- **args**: a malloc'd array of `arity` durably-copied argument `%Value`s.
  `@wam_dyn_copy_durable` recurses: atomic values are self-contained and copied
  by value; compounds get a fresh malloc'd `%Compound` + args with the functor
  string duplicated and each arg copied recursively.

### Scope of PR 1 — ground facts

Asserted terms are treated as **facts** (the head is the whole term) and are
expected to be **ground** — which is exactly what a self-hosting compiler's
dynamic tables are (gensym counters, symbol tables, collected clauses become
ground once fully built). Concretely, in PR 1:

- `assertz(T)` / `asserta(T)` store `T` as a fact. A `T` of the form `(H :- B)`
  is stored verbatim as a `:-`/2 fact (not interpreted as a rule) — calling a
  rule body is milestone 3b-db-rules, a separate follow-up that needs a body
  interpreter.
- Unbound variables in an asserted term are copied as fresh `Unbound`
  sentinels with **no variable-map sharing** — the same documented limitation
  the existing `@wam_copy_term_value` carries. Ground facts are unaffected;
  non-ground asserts are a documented follow-up (a var-map durable copy).

## Calling a dynamic predicate

The probe that motivated the split: `getc(N) :- counter(N)` lowers to
`execute counter/1` with an **unresolved label** (no compiled clauses for
`counter/1`), which today silently defaults to label index 0. Routing that
*direct* call to the store needs compile-time `:- dynamic` tracking plus a new
runtime dispatch — that is **PR 2**.

PR 1 reaches the store through the path that already resolves runtime-built
goals: the `call/1` meta-call. `G = counter(X), call(G)` lowers to
`call call/1` (label index −1), handled by `@wam_dispatch_meta_call`, which
resolves the goal's functor/arity against the compiled meta-call table and
`fail`s when nothing matches. That `fail` is the hook: on a miss we consult the
dynamic store instead of failing.

```
@wam_dispatch_meta_call:
    atom goal, meta table miss  ─▶ @wam_dyn_consult(vm, atom_string,  0,        after_pc)
    compound goal, table miss   ─▶ @wam_dyn_consult(vm, functor,      base_arity, after_pc)
```

(PR 1 consults only when the goal in reg 0 is the *complete* goal — i.e. no
`call/N` closure extra-args — since the iterator reads the goal's arguments
straight out of reg 0. `call(foo, A, B)`-style partial application is a
follow-up.)

### The consult iterator (a new choice-point kind)

Backtracking over the matching clauses reuses the choice-point machinery, with
a new `agg_type = -3` ("dynamic clause iterator"), modelled on the existing
foreign-result iterator (`agg_type = -2`). `@backtrack` dispatches `-3` to
`@wam_dyn_iter_next` just as it dispatches `-2` to `@wam_foreign_iter_next`.

`@wam_dyn_consult` pushes the CP (saving regs/trail/heap, and stashing the
functor pointer, arity, a cursor = 0, and the return PC in the repurposed
aggregate fields) and immediately calls `@wam_dyn_iter_next` for the first
solution. Each `@wam_dyn_iter_next`:

1. scans the store from `cursor` for the next **live** row whose functor
   (`@wam_functor_eq`) and arity match;
2. restores the saved registers and unwinds the trail to the CP's mark (so the
   previous attempt's bindings are gone);
3. unifies each goal argument (from reg 0) against the stored row's args with
   `@wam_unify_value` (which trails the goal-side bindings);
4. on full success: advances `cursor` past this row, sets PC to the return PC,
   returns true — the fact "ran", its (empty) body is `true`, execution
   continues at the caller's continuation;
5. on unify failure: continues the scan (re-unwinding for the next row);
6. on exhaustion: unwinds to the mark, pops the CP, returns false — genuine
   goal failure / no (more) matching clauses.

The scan is O(n) per step (re-scanned each backtrack) — fine for the small
tables a compiler keeps; a functor+arity index is a later optimization if a
hot eval loop needs it.

## Builtins

`assertz/1`, `asserta/1` and `retractall/1` are in `is_builtin_pred`
(`wam_target.pl`), so the tier-2 compiler lowers them to `builtin_call`. PR 1
adds their `@execute_builtin` cases, `builtin_op_to_id` entries, switch arms,
and `wamo_enc` loadable-subset entries (so a loaded `.wamo` can build its own
database — the eval-bootstrap need):

| builtin | id | runtime |
|---|---|---|
| `assertz/1` | 175 | `@wam_dyn_assert(vm, A1, /*prepend*/ false)` |
| `asserta/1` | 176 | `@wam_dyn_assert(vm, A1, /*prepend*/ true)` |
| `retractall/1` | 177 | `@wam_dyn_retractall(vm, A1)` — tombstone every live row whose head unifies with the pattern (pattern bindings unwound after each test); always succeeds |

`retract/1` lowers to `call retract/1` (nondet, dispatched as a choice-point
iterator per `wam_target.pl`), so it lands with the dispatch work in **PR 2**.

## Roadmap

- **PR 1 (this):** durable store; `assertz`/`asserta`/`retractall`; calling
  ground dynamic facts via `call/1`. AOT + loaded-object tests.
- **PR 2:** compile-time `:- dynamic P/N` tracking so direct calls
  (`counter(N)`) resolve to the store; nondet `retract/1` as a CP iterator;
  `call/N` partial-application consult.
- **PR 3 (later):** rule bodies (`assertz((H :- B))`) — a body interpreter for
  asserted clauses. The true long pole; likely unneeded for the eval bootstrap
  if the compiler's dynamic predicates are all fact tables.
- Optimizations: functor+arity index over the store; variable-map durable copy
  for non-ground asserts.
