# WAM Mode Analysis: Specification

> **Companion docs:** `WAM_HASKELL_MODE_ANALYSIS_PHILOSOPHY.md`
> (the *why*) and `WAM_HASKELL_MODE_ANALYSIS_PLAN.md` (the work
> breakdown). This doc fixes the contracts: data structures,
> propagation rules, public API, and integration sites.

## 1. Data structures

### 1.1 `binding_state/1`

Three-valued domain. Strict bottom-of-lattice = `unknown`; `bound`
and `unbound` are incomparable peaks. There is no merge below
`unknown` in this analysis (we never join paths).

```prolog
%% binding_state(?State)
%  State is one of:
%    unbound  — variable definitely unbound at this program point
%    bound    — variable definitely bound to a non-variable term
%    unknown  — analysis cannot prove either
binding_state(unbound).
binding_state(bound).
binding_state(unknown).
```

### 1.2 `binding_env/1`

Per-variable map keyed by Prolog variable identity. Stored as an
association list (`assoc/1` from SWI-Prolog's `library(assoc)`)
keyed by the variable's stable address — same approach
`clause_body_analysis.pl` uses for its `VarMap`.

```prolog
%% binding_env(BindingEnv)
%  Opaque association: VarKey -> binding_state.
%  VarKey is a stable identifier obtained via variable_key/2 (see
%  clause_body_analysis for the existing helper).
```

Variables not present in the map are treated as `unknown`. The map
only holds non-default entries to keep printing useful in tests.

### 1.3 `goal_binding/3`

The output record per goal in a clause body.

```prolog
%% goal_binding(GoalIdx, BeforeEnv, AfterEnv)
%  GoalIdx    — 1-based index of the goal in the clause body
%               (after normalisation, see §2.1).
%  BeforeEnv  — binding_env at the program point immediately before
%               this goal executes.
%  AfterEnv   — binding_env at the program point immediately after
%               this goal completes successfully.
```

A clause body of N goals produces N records. A consumer asking
"what is the binding state of T at the `=..` goal" reads
`BeforeEnv` of the record whose `GoalIdx` matches the `=..` goal's
position.

## 2. Algorithm

### 2.1 Body normalisation

Reuse `clause_body_analysis:normalise_body/2`. After normalisation
the body is a flat list of goals; control constructs (if-then-else,
disjunction, conjunction nesting) have been flattened or wrapped.
The analysis treats every flattened goal as one program point.

For if-then-else `(Cond -> Then ; Else)` the analysis is conservative:
it walks the *Then* branch with `Cond` having executed (to capture
guard-induced bindings), but the post-state `AfterEnv` of the entire
construct is the **meet** of the Then-branch and Else-branch
post-states — variables only marked `bound`/`unbound` in *both*
branches retain that state. Anything else collapses to `unknown`.

For disjunction `(A ; B)`, same meet rule: post-state is the meet
of the per-branch post-states.

For findall/bag/set aggregates, the inner goal is analysed in
isolation; the outer call sees only the result variable transitioning
to `bound` after the aggregate completes.

### 2.2 Initial environment

`initial_binding_env(+Head, +ModeDecl, -Env)` constructs the
binding state at the start of the body:

- If `ModeDecl` is `mode(Pred(M1, ..., Mn))` from
  `demand_analysis:read_mode_declaration/3`, then for each head
  argument `H_i` with mode `M_i`:
  - `M_i = +` → if `H_i` is a variable, set state to `bound`;
    otherwise (atom/structure/list literal) leave at default
    `unknown` (the head match itself binds the variable, see §2.3
    for unification handling).
  - `M_i = -` → if `H_i` is a variable, set state to `unbound`;
    otherwise inconsistent declaration, emit a warning and treat
    as `unknown`.
  - `M_i = ?` → leave at `unknown`.
- If no mode declaration is present, every head argument variable
  starts at `unknown`. (Head match still propagates structural
  bindings; see §2.3.)

After mode-driven initialisation, walk the head pattern itself and
mark any variable that appears as a sub-term inside a literal head
argument as `bound` — those are bound by the head unification at
clause entry. Atomic head args don't introduce variables.

### 2.3 Per-goal propagation

`propagate_goal(+Goal, +BeforeEnv, -AfterEnv)` is the workhorse.
The rule table is hardcoded, indexed on goal head + arity. Order:
guards first (don't bind anything, only consume), then explicit
table, then fallback.

#### 2.3.1 Guard goals

For any goal classified as a guard by
`clause_body_analysis:is_guard_goal/2` — comparisons (`>/2`, `<`,
`==/2`, `=\\=`, `@</2`, etc.), type tests (`var/1`, `nonvar/1`,
`is_list/1`, `atom/1`, `integer/1`, `compound/1`), and arithmetic
predicates that don't bind:

- `AfterEnv = BeforeEnv`.
- **Exception**: `nonvar(X)` ⇒ `X` becomes `bound`.
  `var(X)` ⇒ `X` becomes `unbound`.
  These are guards but they prove a binding state.
- **Exception**: `is_list(X)` ⇒ `X` becomes `bound` (a proper list
  is bound). `atom(X)` ⇒ `bound`. `integer(X)` ⇒ `bound`. Etc.

#### 2.3.2 Unification

`X = Y`:
- If `X` is a variable and `Y` is a variable:
  - If `state(X) == bound` or `state(Y) == bound`, both become
    `bound` (one ground side propagates).
  - If `state(X) == unbound` and `state(Y) == unbound`, both
    become `unknown` (aliased fresh vars — analysis does not
    track aliasing).
  - Otherwise both become `unknown`.
- If `X` is a variable and `Y` is a non-variable term:
  - `X` becomes `bound`. Sub-variables of `Y` retain their state
    (or become `bound` if `Y` is fully ground at compile time).
  - Symmetric for `Y` variable, `X` non-variable.
- If neither side is a variable: `AfterEnv = BeforeEnv` (the
  unification either succeeds without binding new variables, or
  fails — analysis is success-conditional).

#### 2.3.3 Arithmetic: `X is Expr`

`X` becomes `bound` (regardless of prior state). Variables in
`Expr` must be `bound` for the goal to succeed; if any are not
proven `bound`, the analysis still proceeds (success-conditional)
but optionally emits a soft diagnostic in verbose mode.

#### 2.3.4 Term-inspection builtins

| Goal | Pre-condition for success | Post-state effect |
|------|---------------------------|-------------------|
| `functor(T, Name, Arity)` | At least one of T, Name+Arity is bound | If T pre-bound: Name and Arity → `bound`. If T pre-unbound and Name+Arity bound: T → `bound`. Else all → `unknown`. |
| `arg(N, T, A)` | N bound, T bound | A → `unknown` (could be unbound or bound depending on T's contents) |
| `T =.. L` | At least one of T, L is bound | If T pre-bound: L → `bound`. If T pre-unbound and L bound to list of length ≥1 with first element bound to atom: T → `bound`. Else all → `unknown`. |
| `copy_term(T, C)` | T bound (for meaningful semantics) | C → `bound` |

#### 2.3.5 Negation: `\\+ Goal`

`\\+ Goal` does not bind anything — `AfterEnv = BeforeEnv`.

#### 2.3.6 Cut: `!`

Cut does not bind anything — `AfterEnv = BeforeEnv`.

#### 2.3.7 User predicate calls

Two paths:

- **Foreign-classified or external_source predicates**: opaque,
  all argument variables become `unknown` after the call.
- **User predicates with `:- mode/1` declaration**: the input
  positions (modes `+`) require their argument to be `bound`
  before the call (analysis emits a `mode_violation` warning if
  not, but does not fail compilation); output positions (modes
  `-`) make their argument `bound` after the call. `?` modes
  leave the argument at its pre-call state.
- **User predicates without mode declaration**: conservative —
  every argument variable becomes `unknown` after the call.

#### 2.3.8 Control constructs

- `(A, B)`: sequential, `propagate_goal(A, …) ∘ propagate_goal(B, …)`.
- `(A ; B)`: meet of post-states, see §2.1.
- `(Cond -> Then ; Else)`: meet of post-states, see §2.1.
- `aggregate_all(Tmpl, Goal, Result)` /
  `findall/3` / `bagof/3` / `setof/3`: inner goal analysed in
  isolation with no effect on outer env except `Result` →
  `bound`.

#### 2.3.9 Fallback

Any goal not matched by the above table is treated like an
opaque user predicate call: all argument variables become
`unknown`.

### 2.4 Analyser entry point

```prolog
%% analyse_clause_bindings(+Head, +Body, -GoalBindings)
%
%% Computes per-goal binding-state records for a clause body.
%%
%% Head — the clause head, e.g. `build_term(F, A, T)`.
%% Body — the clause body, may contain control constructs.
%% GoalBindings — list of `goal_binding/3` records, one per
%%   normalised goal in the body, in source order.
%%
%% Side-effect-free. Reads `:- mode/1` declarations via
%% demand_analysis:read_mode_declaration/3 if available.
analyse_clause_bindings(+Head, +Body, -GoalBindings).
```

Public API. Lives in `src/unifyweaver/core/binding_state_analysis.pl`.

A second entry point provides the lookup the WAM compiler will use:

```prolog
%% binding_state_at(+GoalIdx, +Var, +GoalBindings, -State)
%% Reads the BeforeEnv of the goal at index GoalIdx and returns
%% the binding_state of Var in that environment. Defaults to
%% `unknown` if the variable is not in the env.
binding_state_at(+GoalIdx, +Var, +GoalBindings, -State).
```

## 3. Integration with `wam_target.pl`

### 3.1 Plumbing

`compile_clause/3` (or whichever predicate currently sequences
head-pattern compilation, body normalisation, and goal-by-goal
emission) gains a one-line call:

```prolog
analyse_clause_bindings(Head, Body, GoalBindings),
```

The resulting `GoalBindings` list is threaded into the body walk
alongside the existing `V0` register-allocator state. The walk
becomes:

```prolog
compile_goals_with_bindings([], _Idx, _GB, V, _, V, "").
compile_goals_with_bindings([Goal|Rest], Idx, GoalBindings, V0, HasEnv, Vf, Code) :-
    compile_goal_call_with_bindings(Goal, Idx, GoalBindings, V0, V1, GoalCode),
    NIdx is Idx + 1,
    compile_goals_with_bindings(Rest, NIdx, GoalBindings, V1, HasEnv, Vf, RestCode),
    format(string(Code), "~w~n~w", [GoalCode, RestCode]).
```

`compile_goal_call_with_bindings/6` is a thin wrapper that pulls
`BeforeEnv` from `GoalBindings` for the current `Idx` and passes
it to a new `compile_goal_call/5` arity. Existing `compile_goal_call/4`
becomes a default-`unknown`-env wrapper for backwards compatibility
with non-WAM-Haskell callers.

### 3.2 The `=../2` lowering decision

In `compile_goal_call/5` (the new arity), add an early clause:

```prolog
compile_goal_call(Goal, BeforeEnv, V0, Vf, Code) :-
    Goal = (T =.. L),
    is_list_with_var_head_and_fixed_tail(L, Name, FixedArgs),
    binding_state_at_var(BeforeEnv, T, unbound),
    binding_state_at_var(BeforeEnv, Name, bound),
    !,
    emit_put_structure_dyn_lowering(T, Name, FixedArgs, V0, Vf, Code).
```

`is_list_with_var_head_and_fixed_tail/3` recognises the pattern
`[NameVar, Arg1, Arg2, ..., ArgN]` where:
- `NameVar` is a variable.
- The list literal has a fixed length N ≥ 0.
- Each `Arg_i` is a term (var, atom, structure, list literal — the
  existing `compile_put_argument` machinery handles all of these).

### 3.3 The lowering itself

`emit_put_structure_dyn_lowering(T, Name, FixedArgs, V0, Vf, Code)`
emits:

```
    put_value Reg(Name), A1                  # nameReg
    put_constant N, A2                       # arityReg (literal int)
    put_structure_dyn A1, A2, A3             # construct Str at A3
    set_value/set_variable for FixedArgs[1..N]
    get_value Reg(T), A3                     # unify constructed term with T
```

A few details:
- If `Name` does not yet have a register, allocate one via
  `next_x_reg/3` and emit `put_variable` in its place. The
  binding-state precondition (`bound`) is structural — Name
  *must* be reachable from a register at runtime — so the
  fallback should be unreachable in practice but we handle it
  for safety.
- The arity register slot does need a real `put_constant`
  emission; `PutStructureDyn` reads an `Integer` value from
  `arityReg`, not a literal in the instruction.
- `T` may or may not have a register. If it does, emit
  `get_value`; if not (T is a fresh variable here, which is
  precisely the precondition), allocate a register, bind the
  Prolog variable to it via `bind_var/4`, and emit
  `get_variable`.

### 3.4 Falling through to the existing path

If any of the precondition predicates fail (pattern mismatch,
binding state not provable, list shape too dynamic), the
compile_goal_call clause matches no head and falls through to
the existing `Goal =.. [Pred|Args]` general builtin path. No
behavioural change for any unmatched call.

## 4. Test surface

The implementation plan covers the test-by-test list, but the
spec fixes what we test *for*:

- **Unit-level, analyser**:
  - Fresh variable in head with mode `-` is `unbound` at goal 1.
  - Fresh variable bound by `is/2` becomes `bound` after the goal.
  - Variable bound by head pattern unification (e.g. head
    `foo(f(X))` ⇒ X is `bound` at start of body).
  - Meet at if-then-else collapses to `unknown` when branches
    disagree.
  - Opaque user call without mode declaration yields `unknown`
    for every argument.
- **Unit-level, lowering**:
  - `T =.. [Name | Args]` with T proven `unbound` and Name
    proven `bound` emits `PutStructureDyn` in generated WAM
    text.
  - `T =.. [Name | Args]` with T `unknown` falls through to
    `builtin_call =../2`.
  - `T =.. [Name | Args]` with T `bound` falls through.
- **Integration**:
  - `build_pair/4` from the existing smoke fixture, given
    `:- mode build_pair(?, ?, ?, -).`, generates a
    `Predicates.hs` containing `PutStructureDyn` and the
    project builds and runs.
  - `split_pair/4` from the same fixture, given
    `:- mode split_pair(+, ?, ?, ?).`, keeps the existing
    `BuiltinCall "=../2"` path and the project builds and runs.
  - Without any mode declaration, both predicates compile via
    the existing builtin path (no regressions on un-annotated
    code).

## 5. Failure modes and diagnostics

The analysis must never crash. Three specific failure modes:

- **Mode declaration parse error**: warn and proceed as if no
  declaration was present.
- **Unhandled goal shape** (e.g. a meta-call `call(F, X)`):
  treat all arg variables as `unknown` after the goal, emit
  no warning (silent fallback).
- **Mode violation** (a goal requires a `+` argument bound, but
  analysis cannot prove it): emit a warning at compile time
  but proceed. The runtime `=../2` builtin handles the actual
  failure semantics.

In verbose mode (controlled by an existing
`unifyweaver_debug` flag if there is one — to be confirmed in
the implementation plan), print the per-goal binding env to
stderr alongside the existing WAM text dump for the predicate.

## 6. Out-of-scope for this spec

Explicitly deferred to follow-up arcs:

- `functor/3` lowering to `PutStructureDyn` + `SetVariable` ×N.
- `arg/3` lowering to indexed `GetValue`.
- `\\+ member(X, L)` lowering to native set lookup.
- Sharing analysis (alias tracking).
- Multi-mode predicates compiled as separate WAM bodies.
- Backwards/demand-driven mode propagation.

The analysis API (§2.4) is shaped to accommodate those follow-ups
without further changes — they would add new propagation rule
table entries and new lowering clauses, not alter the contract.
