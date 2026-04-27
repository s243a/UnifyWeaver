# WAM Mode Analysis: Implementation Plan

> **Companion docs:** `WAM_HASKELL_MODE_ANALYSIS_PHILOSOPHY.md`
> (the *why*) and `WAM_HASKELL_MODE_ANALYSIS_SPEC.md` (the *what*).
> This doc is the *how*: ordered phases, file-by-file edits, test
> gates between phases, and rollback advice.

## Phases at a glance

| Phase | Goal | Lines (rough) | Test gate |
|------:|------|---------------|-----------|
| **M1** | Skeleton analyser module + initial-env construction from `:- mode/1` | ~150 LOC | unit tests for initial env |
| **M2** | Per-goal propagation table for guards, `=/2`, `is/2`, term-inspection builtins | ~250 LOC | unit tests for propagation on synthetic goal lists |
| **M3** | Control-construct handling (if-then-else, disjunction, aggregates) | ~100 LOC | unit tests for meet semantics |
| **M4** | User-call handling (mode-declared and opaque) | ~80 LOC | unit tests for input/output mode propagation |
| **M5** | Wire `analyse_clause_bindings` into `wam_target.pl` body walk | ~50 LOC threading change | existing test suite stays green |
| **M6** | `=../2` lowering decision in `compile_goal_call/5` | ~80 LOC for new clause + emit helper | new codegen unit tests |
| **M7** | End-to-end: regenerate `dyn-struct-smoke` with mode declarations, build, run, verify `PutStructureDyn` is in the output | ~30 LOC test fixture | smoke fixture builds and produces correct results |
| **M8** | Documentation: update `MEMORY.md`, `WAM_PERF_OPTIMIZATION_LOG.md`, the project status memory file | docs only | no test impact |

Total estimated LOC: ~700 production, ~400 test, ~150 docs.

## Phase M1 — Skeleton analyser

### Files
- **New**: `src/unifyweaver/core/binding_state_analysis.pl`
- **New**: `tests/core/test_binding_state_analysis.pl`
- **Read-only reference**: `src/unifyweaver/core/clause_body_analysis.pl`
  (for `variable_key/2` and `normalise_body/2`).
- **Read-only reference**: `src/unifyweaver/core/demand_analysis.pl`
  (for `read_mode_declaration/3`, lines 133–149).

### Tasks

1. Create the module skeleton:
   ```prolog
   :- module(binding_state_analysis, [
       analyse_clause_bindings/3,
       binding_state_at/4,
       binding_state_at_var/3
   ]).
   ```
2. Implement `binding_state/1` validator and three constructors.
3. Implement `binding_env/1` as `assoc/1`-wrapped, with helpers:
   - `empty_binding_env(-Env)`
   - `set_binding_state(+Env0, +Var, +State, -Env1)`
   - `get_binding_state(+Env, +Var, -State)` — defaults to `unknown`.
4. Implement `initial_binding_env/3`:
   ```prolog
   initial_binding_env(+Head, +ModeDecl, -Env)
   ```
   Reads `Head =.. [_|HeadArgs]`, walks them in lockstep with the
   mode list (or uses `?` for every position when no mode), and
   builds an env with appropriate states.
5. Add a stub `analyse_clause_bindings/3` that returns
   `goal_binding(1, InitEnv, InitEnv)` for every body goal — no
   propagation yet. This lets us land the integration test gate in
   M5 without blocking on M2–M4.

### Test gate (M1)

Add `tests/core/test_binding_state_analysis.pl` with:
- `test_initial_env_no_mode` — every head arg → `unknown`.
- `test_initial_env_input_mode` — `+` head var → `bound`.
- `test_initial_env_output_mode` — `-` head var → `unbound`.
- `test_initial_env_any_mode` — `?` head var → `unknown`.
- `test_initial_env_structured_head` — head `foo(f(X))` → X is
  `bound` regardless of mode.
- `test_get_default_unknown` — un-set var reads as `unknown`.

Wire the test file into the project test runner (likely the
`run_all_core_tests` predicate, mirror what
`tests/core/test_demand_analysis.pl` does).

## Phase M2 — Propagation table for guards, =, is, term inspection

### Files
- **Edit**: `src/unifyweaver/core/binding_state_analysis.pl`
- **Edit**: `tests/core/test_binding_state_analysis.pl`

### Tasks

1. Implement `propagate_goal/3` with the rule table from spec §2.3:
   - Guards (delegate to `clause_body_analysis:is_guard_goal/2`)
     including the `var/1`, `nonvar/1`, type-test exceptions.
   - `X = Y` unification with the four sub-cases.
   - `X is Expr` → X bound.
   - `functor/3`, `arg/3`, `=../2`, `copy_term/2` per the table.
   - `\\+/1`, `!/0` → no-op.
2. Make `analyse_clause_bindings/3` actually walk the body:
   ```prolog
   analyse_clause_bindings(Head, Body, GoalBindings) :-
       lookup_mode_decl(Head, ModeDecl),
       initial_binding_env(Head, ModeDecl, Env0),
       clause_body_analysis:normalise_body(Body, NormBody),
       walk_body(NormBody, 1, Env0, GoalBindings).
   ```
3. Implement `walk_body/4` as a fold that emits a `goal_binding/3`
   per goal and threads the env forward.

### Test gate (M2)

- `test_propagate_unify_var_term` — `X = foo(a)` ⇒ X bound.
- `test_propagate_unify_two_vars_one_bound` — bound flag propagates.
- `test_propagate_is` — `Y is X + 1` ⇒ Y bound.
- `test_propagate_nonvar_guard` — `nonvar(X)` ⇒ X bound.
- `test_propagate_var_guard` — `var(X)` ⇒ X unbound.
- `test_propagate_functor_compose` — `functor(T, foo, 2)` with T
  unbound ⇒ T bound.
- `test_propagate_functor_decompose` — `functor(T, N, A)` with T
  bound ⇒ N and A bound.
- `test_propagate_univ_compose` — `T =.. [foo, X, Y]` ⇒ T bound
  (since list is statically bound).
- `test_propagate_negation_noop` — env unchanged through `\\+ G`.

## Phase M3 — Control constructs

### Files
- **Edit**: `src/unifyweaver/core/binding_state_analysis.pl`
- **Edit**: `tests/core/test_binding_state_analysis.pl`

### Tasks

1. Add an env-meet operator:
   ```prolog
   meet_env(+EnvA, +EnvB, -EnvAB)
   %% Variable-by-variable meet:
   %%   bound  ⊓ bound   = bound
   %%   unbound ⊓ unbound = unbound
   %%   anything else      = unknown
   ```
2. Handle if-then-else and disjunction in `walk_body/4` by
   recursively analysing each branch from the same `Env0` and
   meeting the post-states.
3. Handle aggregates (`findall/3`, `bagof/3`, `setof/3`,
   `aggregate_all/3`): inner goal analysed in isolation; outer
   env gains `bound` for the result variable; template variables
   get `unknown`.

### Test gate (M3)

- `test_ite_meet_disagree` — Then sets X bound, Else doesn't ⇒
  post-state X is unknown.
- `test_ite_meet_agree` — both branches set X bound ⇒ X bound.
- `test_disjunction_meet` — same shape via `(A ; B)`.
- `test_findall_result_bound` — `findall(_, _, R)` ⇒ R bound,
  inner template var unaffected.

## Phase M4 — User-call handling

### Files
- **Edit**: `src/unifyweaver/core/binding_state_analysis.pl`
- **Edit**: `tests/core/test_binding_state_analysis.pl`

### Tasks

1. In `propagate_goal/3` fallback, classify the goal:
   - Foreign / external_source / unknown predicate ⇒ all arg
     vars become `unknown`.
   - User predicate with `:- mode/1` declaration ⇒ apply mode
     pattern: `+` args required `bound` (warning if not), `-`
     args become `bound` after, `?` args unchanged.
2. Reuse `demand_analysis:read_mode_declaration/3` verbatim;
   do not re-implement the directive parser.
3. Add an optional warning emission helper
   `emit_mode_violation/2` that prints to stderr only when a
   debug flag is set.

### Test gate (M4)

- `test_call_unknown_pred_opacity` — args become unknown.
- `test_call_mode_decl_input_propagation` — `+` arg required
  bound (analyser does not block, just notes).
- `test_call_mode_decl_output_propagation` — `-` arg becomes
  bound after the call.
- `test_call_mode_decl_any` — `?` args unchanged.

## Phase M5 — Wire into wam_target.pl

### Files
- **Edit**: `src/unifyweaver/targets/wam_target.pl`
  (`compile_clause/3` and the body walk).
- **Edit**: `src/unifyweaver/targets/wam_target.pl` to add an
  arity-5 `compile_goal_call/5` and `compile_goal_execute/5`
  that take a `BeforeEnv`. Existing arity-4 wrappers default to
  empty env.

### Tasks

1. Add `:- use_module('../core/binding_state_analysis')` at the
   top of `wam_target.pl`.
2. In `compile_clause/3` (or wherever the body sequence is built):
   ```prolog
   binding_state_analysis:analyse_clause_bindings(
       Head, Body, GoalBindings),
   ```
3. Replace the existing `compile_goals/5` with
   `compile_goals_with_bindings/7` that threads
   `(Idx, GoalBindings)` alongside the existing `(V0, HasEnv, Vf)`.
4. New arity:
   ```prolog
   compile_goal_call(Goal, BeforeEnv, V0, Vf, Code).
   compile_goal_execute(Goal, BeforeEnv, V0, Vf, Code).
   ```
   Default `BeforeEnv = empty_binding_env` clause kept for any
   non-WAM-Haskell target that may eventually call these.
5. **Critical**: keep the existing arity-4 versions intact and
   delegate to the arity-5 with empty env. The Rust, Go, ILAsm,
   LLVM, Elixir targets all share `wam_target.pl` — we cannot
   force them to thread a new state.

### Test gate (M5)

- Run the **full existing test suite** (`run_tests` in
  `tests/test_wam_haskell_target.pl` plus the core test runner).
  No regressions.
- Generate `/tmp/dyn-struct-smoke` from the existing fixture
  (no mode declarations) — the generated `Predicates.hs` is
  byte-for-byte identical to current main. Confirms the wiring
  is a no-op without mode declarations.

## Phase M6 — `=../2` lowering decision

### Files
- **Edit**: `src/unifyweaver/targets/wam_target.pl`
- **Edit**: `tests/test_wam_haskell_target.pl`

### Tasks

1. In `compile_goal_call/5`, add an early clause matching
   `Goal = (T =.. L)`:
   ```prolog
   compile_goal_call(T =.. L, BeforeEnv, V0, Vf, Code) :-
       parse_univ_list_pattern(L, Name, FixedArgs),
       binding_state_analysis:binding_state_at_var(BeforeEnv, T, unbound),
       binding_state_analysis:binding_state_at_var(BeforeEnv, Name, bound),
       !,
       emit_put_structure_dyn_lowering(T, Name, FixedArgs, V0, Vf, Code).
   ```
2. Implement `parse_univ_list_pattern(+List, -NameVar, -FixedArgs)`:
   - Accepts `[NameVar | RestArgs]` where `NameVar` is unbound
     (a Prolog variable) and `RestArgs` is a fixed-length proper
     list.
   - Fails (no lowering) if list is partial, contains a tail
     variable, or NameVar is not a variable.
3. Implement `emit_put_structure_dyn_lowering/6` per spec §3.3:
   - Allocate or look up reg for Name, emit `put_value` (or
     `put_variable` if not yet bound to a reg — defensive).
   - Allocate a fresh X-reg for arity, emit `put_constant N`.
   - Allocate a fresh X-reg for the constructed term, emit
     `put_structure_dyn`.
   - For each `Arg_i` in `FixedArgs`, emit the appropriate
     `set_value` / `set_variable` / `set_constant` /
     nested `put_structure` (reuse `compile_set_arguments/4`).
   - For T: if already bound to a reg, emit `get_value`; else
     bind T to the constructed-term reg via `bind_var/4` and
     emit nothing (T is now aliased to the term reg).
4. Mirror the change in `compile_goal_execute/5` for the
   tail-call form, ending with `proceed` instead of falling
   through to the next goal.

### Test gate (M6)

- `test_haskell_univ_compose_lowered_to_put_structure_dyn` —
  given a clause body that the analyser proves is compose-mode,
  the generated WAM text contains `put_structure_dyn` instead
  of `builtin_call =../2`.
- `test_haskell_univ_unknown_keeps_builtin` — without mode
  declaration, the generated WAM text contains
  `builtin_call =../2`.
- `test_haskell_univ_decompose_keeps_builtin` — with `:- mode`
  marking T as `+`, the generated WAM text contains
  `builtin_call =../2`.

## Phase M7 — End-to-end smoke

### Files
- **Edit (or new)**: `tests/test_wam_haskell_target.pl` —
  add an integration test that:
  1. Asserts the existing `dyn-struct-smoke` fixture predicates
     into a clean db.
  2. Asserts a `:- mode build_pair(?, ?, ?, -).` directive.
  3. Calls `write_wam_haskell_project/3` to a temp dir.
  4. `cabal v2-build`s the project, asserts exit code 0.
  5. Greps the generated `Predicates.hs` for
     `PutStructureDyn`, asserts present in `build_pair`'s
     instruction list.
  6. Greps the same file for `BuiltinCall "=../2"`, asserts
     present in `split_pair`'s instruction list (since it has
     no mode declaration).

### Test gate (M7)

The integration test passes. The smoke binary (built via
`cabal v2-run dyn-struct-smoke`) produces the same output as
before, confirming end-to-end runtime correctness.

## Phase M8 — Docs

### Files
- **Edit**: `MEMORY.md` (the user's auto-memory). Add an entry
  pointing at a new project-memory file
  `project_wam_haskell_mode_analysis.md` summarising what was
  built and what is now possible.
- **Edit**: `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` — append
  an entry for the lowering, with the analysis pass as the
  prerequisite.
- **Edit**: `docs/design/HASKELL_TARGET_ROADMAP.md` (if
  present) — mark `=../2` lowering as done.

## Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Mode propagation rule table grows unboundedly as we add builtins | medium | Cap at the spec's table; new builtins fall through to "all unknown". Adding rules later is additive. |
| `clause_body_analysis:variable_key/2` doesn't exist or has a different shape | low | Confirm in M1; if missing, write a local helper using `term_to_atom/2` over a copy_term. |
| `wam_target.pl` body walk is shared with non-WAM-Haskell targets and a thread-through breaks them | medium | M5 test gate explicitly verifies the no-op default-env path. |
| `:- mode` declarations are scoped per-module and the analyser misses the user's module | medium | Reuse `demand_analysis`'s lookup verbatim; if it works for demand, it works here. |
| Analyser silently produces wrong `bound`/`unbound` answer | low (but high impact) | Test gate at M2 covers the canonical builtins; M6 tests verify the lowering only fires when both pre-conditions are present. |
| Existing tests that exercise `=../2` regress because the analyser now sees `unknown` everywhere | low | M5 test gate confirms byte-identical codegen for un-annotated code. |

## Rollback plan

The analysis pass is additive: it produces records but the
existing compiler ignores them unless a new
`compile_goal_call/5` clause matches. To roll back:

1. Comment out the M6 `compile_goal_call(T =.. L, ...)`
   clause — falls through to existing builtin path.
2. Optionally comment out the M5 thread-through — the
   analyser still runs but its output is unused.
3. Optionally comment out the M5 `analyse_clause_bindings/3`
   call — the analyser becomes dead code.

No data structures change. No public APIs change. No existing
tests rely on the new module. Rollback is a single-clause
deletion away.

## Acceptance for the arc

Arc complete when:

1. `tests/core/test_binding_state_analysis.pl` runs and is green.
2. `tests/test_wam_haskell_target.pl` runs and is green
   (existing tests + new M6 codegen tests + M7 integration).
3. The `dyn-struct-smoke` smoke fixture, with a mode
   declaration on `build_pair`, generates and builds a Haskell
   project containing `PutStructureDyn` in the compiled instruction
   list, and the project runs to completion.
4. The `MEMORY.md` index points at a new
   `project_wam_haskell_mode_analysis.md` summary with status,
   benchmarks (if any), and follow-up arcs.
5. No regression on `tests/core/test_purity_certificate.pl`,
   `tests/core/test_demand_analysis.pl`, or any other existing
   core test.

## Suggested branch / PR strategy

Single feature branch:
`feat/wam-haskell-mode-analysis`.

One PR is acceptable for the whole arc — phases M1–M4 are
self-contained (analyser + tests, no compiler integration), M5
is a no-op wiring change, and M6–M7 are the actual
optimisation. Reviewers can step through phase-by-phase via
the commits if they want.

If preferred, split into two PRs:
- PR 1: M1–M5 (analyser module + wiring + green test suite).
- PR 2: M6–M7 (lowering + integration smoke).

PR 1 is risk-free (no behavioural change). PR 2 is where the
actual lowering lands and where any regression would manifest.
