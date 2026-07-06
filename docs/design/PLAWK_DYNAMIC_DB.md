# Dynamic clause store (assert / retract) for the WAM/LLVM target

Status: **PR 1 + PR 2 landed.** PR 1: the store + `assertz`/`asserta`/
`retractall` + ground facts called via `call/1`. PR 2: **direct** calls to
`:- dynamic` predicates (no explicit `call/1`) and **nondet `retract/1`**.
This is milestone **3b-db** of the eval bootstrap (see PLAWK_EVAL_BOOTSTRAP.md):
a mutable clause database in a VM whose term arena is rewound between queries —
the last genuinely-new runtime machinery before the eval/compile surface.

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

`getc(N) :- counter(N)` lowers to `execute counter/1` with an **unresolved
label** (no compiled clauses for `counter/1`), which used to silently default
to label index 0. **PR 2** routes such *direct* calls to the store without any
new runtime dispatch: the tier-2 compiler (`dynamic_store_goal/1` in
`wam_target.pl`) detects a body goal calling a `:- dynamic` predicate that has
no compiled clauses and **rewrites it to a `call/1` meta-call** —
`counter(N)` → `call(counter(N))`, which lowers to `execute call/1` and flows
through the exact path below. Detection is `predicate_property(Mod:Head,
dynamic)` + no `clause/2` (the compile module is recorded in `b_setval`); a
dynamic predicate that also has compiled clauses is left as a static call
(mixing compiled + asserted clauses for one predicate is out of scope).

Both PR 1's explicit `call/1` and PR 2's rewritten direct calls reach the store
through the path that resolves runtime-built goals: the `call/1` meta-call.
`G = counter(X), call(G)` lowers to
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

### retract/1 — nondet (PR 2)

`retract/1` is NOT in `is_builtin_pred`, so it lowers to `call retract/1` /
`execute retract/1` (a nondet operation, not a plain builtin). PR 2 recognises
`"retract/1"` in the LLVM call/execute lowering and emits the op1 sentinel
**−3** (alongside −1 for meta-call), which the `do_call` / `do_execute` opcode
handlers route to `@wam_dyn_retract_consult`. That reads the pattern from
reg 0, extracts its functor/arity, and pushes a **new `agg_type = −4`
choice point** — a retract iterator (`@wam_dyn_retract_iter_next`, dispatched
from `@backtrack` next to −2/−3). It is the consult iterator plus one step: on
a matching unify it **tombstones the row** (`live = 0`) before yielding, so
each solution removes one clause and backtracking removes the next. Works in
call position (`retract(X), More`) and last-call position, and drives cleanly
under `findall(X, retract(p(X)), L)` to remove every match.

## catch/3 and throw/1 (milestone 3c)

The last runtime primitive before the eval surface, and — like the store — it
turned out not to need cross-object linkage. `catch/3` and `throw/1` aren't
`is_builtin_pred`, so they lower to `call`/`execute` with op1 sentinels **−5**
(catch) and **−6** (throw), which `do_call` / `do_execute` route to dedicated
handlers. State lives in a **process-global side stack of `%WamCatchFrame`**
(separate from the choice-point stack), reset per top-level query in
`@wam_prepare_call`.

- `@wam_catch_setup` reads A2/A3 (Catcher, Recovery), pushes a frame
  snapshotting the VM marks (trail / heap / stack / cp / cp_count) and the
  post-catch continuation PC, then meta-calls Goal (in reg 0) — reusing
  `@wam_dispatch_meta_call`, so Goal resolves against the object's meta table /
  the clause store exactly like `call/1`. (This is why `wamo_has_meta_call`
  treats −5/−6 as needing the meta table built.)
- `@wam_throw` deep-copies the ball durably (`@wam_dyn_copy_durable`, so it
  survives the heap unwind), then walks the side stack top-down: for each frame
  it restores the VM to that frame's marks and unifies the catcher with the
  ball; on a match it runs Recovery via the meta-call and continues at the
  frame's return PC, otherwise it keeps unwinding. No matching frame → the throw
  is uncaught and the computation fails (halts).

Goal, Catcher and Recovery are built before `catch/3` runs, so they sit below
the frame's heap mark and survive the unwind; the catcher's bindings are made
*after* the unwind and are the ones Recovery sees.

**Scoping limitation (documented):** true ISO `catch/3` protects only Goal. A
frame here is popped on backtracking past it and on the per-query reset, but on
a *forward* path a frame can linger after Goal has exited — so a `throw`
sequenced after a `catch` in the same clause body, before any backtrack, may be
caught by it. This is correct for the dominant error-boundary usage
(`catch(work(X), E, handle(E))`); exact ISO forward-scoping is a follow-up.
Recovery is run through the meta-call, so (like `call/1`) it should be a
predicate goal, not a bare builtin.

## Rule bodies (PR 3)

`assertz((H :- B))` stores a **rule**: the row is keyed by `H`'s functor/arity
with `B` kept as a durable body term (facts store an Unbound-sentinel body).
When the head is consulted, the body runs.

**Variable sharing — the load-bearing piece.** A rule like `p(X) :- q(X)`
shares `X` between head and body, so the durable copy must preserve variable
identity (the naive `@wam_dyn_copy_durable` does not). `@wam_dyn_copy_var`
copies the clause assigning each distinct source variable an **index**, stored
as a Ref whose payload is that index; sharing one var-map across the head and
body copy preserves head↔body sharing, and `nvars` is recorded on the row. At
call time `@wam_dyn_instantiate` copies the durable head/body into the arena,
mapping each var-index to a **fresh heap cell** — so every clause use gets
fresh, correctly-shared variables. The consult iterator splits: a ground
bodyless fact (`nvars == 0`, sentinel body) takes the existing fast path
(unify goal args vs stored args directly); a rule or var-clause takes a slow
path (allocate `nvars` fresh cells, instantiate + unify the head, then run the
body). `retract`/`retractall` match only ground facts (`nvars == 0`); retract
of rules is a follow-up.

**Body execution (`@wam_dyn_run_body`)** is deterministic (first solution):
`,`/2 recurses; a builtin goal (`is/2`, `<`,`>`,`>=`,`=<`,`=:=`,`=\=`, `==`,
`=`) marshals its args into registers and runs through `@execute_builtin`; any
other goal is a predicate call solved via `@wam_dyn_solve_pred` — a nested
`@run_loop` capped by a **barrier choice point** (`agg_type = −8`, which
`@backtrack` turns into a `false`) and a `cp = 0` halt continuation, then the
goal's choice points are cut (deterministic). A dynamic body goal is run inline
by the consult during dispatch (it halts), so `solve_pred` must not clear
`halted` before `run_loop`; a compiled body goal leaves `halted` false and
`run_loop` drives it. Nested rules work (a rule body may call another rule).
Not yet: `;`/`->`/`!` inside bodies, and cross-goal backtracking (first
solution only).

(This PR also fixed a latent tail-position bug: a consulted goal whose
continuation is the top-level `cp = 0` must **halt** rather than jump to PC 0 —
the consult/retract `success` blocks now mirror `proceed`.)

## Roadmap

- **PR 1:** durable store; `assertz`/`asserta`/`retractall`; calling ground
  dynamic facts via `call/1`.
- **PR 2:** compiler rewrite of direct `:- dynamic` calls to `call/1`
  (`dynamic_store_goal/1`); nondet `retract/1` as an `agg_type = −4` CP
  iterator (op1 = −3 sentinel).
- **Milestone 3c:** `catch/3` + `throw/1` (side stack of catch frames, op1
  sentinels −5/−6).
- **PR 3:** rule bodies (`assertz((H :- B))`) — var-preserving clause copy +
  a deterministic body interpreter. Landed.
- Follow-ups / optimizations: `;`/`->`/`!` and cross-goal backtracking in
  bodies; retract of rules; `call/N` partial-application consult; a
  functor+arity index over the store (the scan is O(n) per backtrack); mixing
  compiled + asserted clauses for one predicate.
