# WAM Elixir Parity — Specification

Concrete gap inventory, dependency graph, and per-PR phasing for
closing `wam_elixir_target.pl`'s feature gap with the C++ WAM
target. For *why* each decision, see
`WAM_ELIXIR_PARITY_PHILOSOPHY.md`.

Scope boundary vs other Elixir docs:
- This doc: **feature parity** (catch/throw, ISO errors, meta-call).
- `WAM_ELIXIR_CORRECTNESS_GAPS.md`: bug fixes in existing features.
- `WAM_ELIXIR_PERF_PHASE_A_PLAN.md`: container/data-structure perf
  to match Haskell/Rust baselines.

## 1. Gap inventory

Survey columns: ✅ shipped · ⚠️ partial · ❌ missing.

### 1.1 Builtins and control flow

| Feature | Elixir | C++ | Rust | Haskell | Notes |
|---|---|---|---|---|---|
| `is/2` | ✅ | ✅ | ✅ | ✅ | All four. |
| Arith compares (`>/2`, `</2`, `>=/2`, `=</2`, `=:=/2`) | ✅ | ✅ | ✅ | ⚠️ | Haskell partial — needs audit. |
| `=\=/2` | ❌ | ✅ | ❌ | ❌ | Trivial gap; bundle with sweep. |
| `\+/1` (negation as failure) | ✅ | ✅ | ⚠️ | ⚠️ | |
| `not/1` | ✅ | ✅ | ⚠️ | ⚠️ | Alias for `\+/1`. |
| `findall/3` | ⚠️ | ✅ | ⚠️ | ⚠️ | Elixir has partial — needs gap-closing audit. |
| `call/N` (meta-call) | ⚠️ | ✅ | ⚠️ | ⚠️ | Required by `catch/3`. |
| `catch/3` | ❌ | ✅ | ❌ | ❌ | **Foundation for ISO error stack.** |
| `throw/1` | ❌ | ✅ | ❌ | ❌ | |
| `bagof/3`, `setof/3` | ❌ | ✅ | ❌ | ❌ | Out of initial scope. |
| `succ/2` | ❌ | ✅ | ❌ | ❌ | |
| `between/3` | ❌ | ❌ | ❌ | ❌ | All targets missing. |
| `format/1`, `format/2` | ✅ | ✅ | ✅ | ✅ | |

### 1.2 ISO error infrastructure

| Component | Elixir | C++ | Notes |
|---|---|---|---|
| `iso_errors_default_to_iso/2` table | ❌ | ✅ | 8 entries in C++. |
| `iso_errors_default_to_lax/2` table | ❌ | ✅ | |
| `iso_errors_resolve_options/2` config loader | ❌ | ✅ | |
| `iso_errors_rewrite/4` per-pred dispatch | ❌ | ✅ | |
| Audit predicate (`wam_*_iso_audit/3`) | ❌ | ✅ | |
| Runtime: `make_type_error` / `make_*_error` | ❌ | ✅ | |
| Runtime: `throw_iso_error` helper | ❌ | ✅ | Depends on `throw/1` runtime. |
| `is_iso/2` + `is_lax/2` | ❌ | ✅ | First ISO-aware builtin. |
| `succ_iso/2` + `succ_lax/2` | ❌ | ✅ | |
| Iso/lax variants of arith compares | ❌ | ✅ | |
| Lax IEEE-754 float divide (inf/nan) | ❌ | ✅ | Small breaking change vs current Elixir. |

### 1.3 Meta-call and synthetic ops

| Feature | Elixir | C++ | Notes |
|---|---|---|---|
| `invoke_goal_as_call` (goal-term dispatch) | ⚠️ | ✅ | C++ has explicit fn; Elixir has partial via `WamDispatcher.call`. |
| Synthetic `catch_return` op | ❌ | ✅ | Auto-injected on dispatch table tail. |
| Synthetic `negation_return` op | ❌ | ✅ | For `\+/1`. Elixir's `\+/1` works without it; audit. |
| Synthetic `findall_collect` op | ⚠️ | ✅ | |
| Synthetic `conj_return` (for `,/2` as goal-term) | ❌ | ⚠️ | |
| Synthetic `disj_alt` (for `;/2`) | ❌ | ⚠️ | |
| Synthetic `if_then_commit` / `if_then_else` | ❌ | ✅ | |
| Synthetic `aggregate_next_group` | ❌ | ✅ | For `bagof/setof`; out of scope. |

### 1.4 Items API

| Status | Elixir | C++ | Rust | Haskell |
|---|---|---|---|---|
| Consumes `compile_predicate_to_wam_items` | ❌ | ❌ | ❌ | ❌ |
| Calls `parse_wam_text` (legacy text path) | ✅ | ✅ | ✅ | ✅ |
| `format(string(...))` site count | 56 | 22 | 74 | 88 |

Phase 1 of the Items API hasn't landed in `wam_target.pl` yet.
Tracked separately in `WAM_ITEMS_API_SPECIFICATION.md`.

## 2. Dependency graph

Solid arrows = hard prerequisites. Dotted = beneficial but not
required.

```
                          [catch/3 + throw/1]
                                  │
                                  │  needs runtime unwinding
                                  ▼
                       [ISO errors plumbing]
                       (config loader, rewrite,
                        audit, key tables empty)
                                  │
                                  ├──────────────────────┐
                                  ▼                      ▼
                           [is_iso/2 +              [succ_iso/2 +
                            is_lax/2]                arith iso/lax sweep]

   [call/N meta-call] ────► [catch/3]    (catch dispatches A1 as a goal)

   [Items API Phase 1] ┄┄┄► [any of the above]
   (cross-cutting; reduces                   (rewrites become
    each PR ~30%)                             swap_key_in_item walks
                                              instead of multi-shape
                                              text matching)
```

`call/N` is the only non-obvious prerequisite. C++ ships
`call/N` as a precursor commit to `catch/3` (commit `b5fc6041`)
because `catch(Goal, Catcher, Recovery)` needs to dispatch `Goal`
as a tail call regardless of Goal's shape — including
`Module:Pred(...)` and partial-application forms. Elixir's
current meta-call is partial; the audit in PR #1 below
determines whether it's enough.

## 3. Per-PR phasing

Five PRs in the initial roadmap. Sized small for review.

### PR #1 — `call/N` meta-call audit + completion

**Scope:** survey what Elixir's existing meta-call does, identify
the gap, and close enough of it to support `catch/3`'s goal-term
dispatch in PR #2.

Acceptance:
- `?- call(append, [a, b], [c, d], X).` succeeds with
  `X = [a, b, c, d]`.
- `?- G = (X is 1 + 1), call(G), write(X).` writes `2`.
- `?- call(true).` succeeds.
- `?- call(fail).` fails (not throws).

Files: `src/unifyweaver/targets/wam_elixir_target.pl`,
`tests/test_wam_elixir_target.pl`.
Reference: C++ commit `b5fc6041`.

### PR #2 — `catch/3` + `throw/1`

**Scope:** mirror C++ commit `151c0178`. Side-stack
`catcher_frames`, synthetic `:catch_return` instruction,
`execute_catch` and `execute_throw` helpers, integration with
existing `:fail` propagation. Option A from PHILOSOPHY §5.

Acceptance test set mirrors C++ (`tests/test_wam_elixir_target.pl`):
- `elixir_e2e_catch_match` — `catch(throw(foo), foo, write(caught))`
  binds catcher and runs recovery.
- `elixir_e2e_catch_match_compound` — compound pattern unifies,
  binding pattern variables in the recovery goal.
- `elixir_e2e_catch_no_throw` — protected goal proceeds normally;
  recovery NOT invoked.
- `elixir_e2e_catch_uncaught` — `throw` walks past all frames,
  prints diagnostic to stderr, returns `:fail`.
- `elixir_e2e_catch_nested` — inner catcher doesn't match, frame
  popped, outer catcher matches.
- `elixir_e2e_catch_fail_propagates` — protected goal fails (not
  throws); catch propagates failure; recovery NOT invoked.

Files: `src/unifyweaver/targets/wam_elixir_target.pl`,
`tests/test_wam_elixir_target.pl`.
Reference: C++ commit `151c0178`.

### PR #3 — ISO errors plumbing (no behaviour change)

**Scope:** mirror C++ commit `f7d2c932`. Config loader
(`iso_errors_resolve_options/2`), rewrite hook
(`iso_errors_rewrite/4`), audit predicate
(`wam_elixir_iso_audit/3`), runtime helpers (`throw_iso_error`,
`make_type_error`, `make_instantiation_error`, etc.). Key tables
shipped empty.

Behaviour unchanged because the rewrite tables are empty — no
default key gets rewritten. Smoke-tested via:
- `test_iso_errors_config_loader` — parse a sample config,
  verify `iso_errors_mode_for/3` returns expected modes.
- `test_iso_errors_inline_wins` — inline option overrides file
  config.
- `test_iso_errors_audit` — audit returns expected per-site
  records (all `source=default`, `resolved` == original key).
- `test_iso_errors_multi_module_warning` — bare PI matches
  multiple modules; loader emits warning.

Files: `src/unifyweaver/targets/wam_elixir_target.pl`,
`tests/test_wam_elixir_target.pl`.
Reference: C++ commit `f7d2c932`.

### PR #4 — `is_iso/2` + `is_lax/2`

**Scope:** mirror C++ commit `32567157`. First ISO-aware builtin.
Add `is_iso/2` and `is_lax/2` runtime branches alongside `is/2`.
Populate `iso_errors_default_to_iso("is/2", "is_iso/2")` and
`iso_errors_default_to_lax("is/2", "is_lax/2")`.

Acceptance:
- `elixir_e2e_iso_is_instantiation` — `is_iso(X, _)` throws
  `error(instantiation_error, _)`.
- `elixir_e2e_iso_is_type_error` — `is_iso(X, foo)` throws
  `error(type_error(evaluable, foo/0), _)`.
- `elixir_e2e_iso_is_zero_divisor` — `is_iso(X, 1//0)` throws
  `error(evaluation_error(zero_divisor), _)`.
- `elixir_e2e_lax_is_unchanged` — existing `is/2` tests pass
  unchanged.
- `elixir_e2e_iso_explicit_lax_in_iso_mode` — predicate annotated
  ISO with explicit `is_lax(X, _)` call site fails silently
  (three-form guarantee).

Files: `src/unifyweaver/targets/wam_elixir_target.pl`,
`tests/test_wam_elixir_target.pl`.
Reference: C++ commit `32567157`.

### PR #5 — ISO sweep (compares + `succ_iso/2` + IEEE-754 floats)

**Scope:** mirror C++ commit `0dda9d1b`. Add iso/lax variants of
all arithmetic compares (`>_iso/2`, `<_iso/2`, `>=_iso/2`,
`=<_iso/2`, `=:=_iso/2`, `=\=_iso/2`), `succ_iso/2` /
`succ_lax/2`, and the lax IEEE-754 float-divide behaviour
(`1.0/0.0` → `inf`, `0.0/0.0` → `nan`, current uniform-fail
preserved for integer divide).

This is the largest PR. Test surface: one e2e test trio (iso /
lax / explicit-bypass) per builtin × 7 builtins = 21 tests, plus
the IEEE-754 float-divide tests.

Files: `src/unifyweaver/targets/wam_elixir_target.pl`,
`tests/test_wam_elixir_target.pl`.
Reference: C++ commit `0dda9d1b`.

## 4. Architectural decisions

### 4.1 Side-stack `catcher_frames` (Option A)

Per PHILOSOPHY §5 we ship Option A in PR #2. State shape:

```elixir
defmodule WamState do
  defstruct [
    # ... existing fields ...

    # Side stack of catch frames. Walked manually on throw;
    # popped when protected goal proceeds via :catch_return.
    catcher_frames: [],

    # PC of the auto-injected :catch_return instruction. Set once
    # at compile time by the generator. catch/3 sets cp = this
    # before dispatching A1.
    catch_return_pc: 0
  ]
end

defmodule CatcherFrame do
  defstruct [
    :catcher_term,        # A2 from catch/3 (pattern)
    :recovery_term,       # A3 from catch/3 (goal on match)
    :saved_cp,            # proceed target after recovery
    :trail_mark,
    :base_cp_count,
    :saved_cut_barrier,
    :saved_regs,          # snapshot of A-regs and X-regs
    :saved_mode_stack,
    :saved_env_stack
  ]
end
```

`execute_catch/1`:
1. Snapshot full VM state into a `%CatcherFrame{}`.
2. Push onto `state.catcher_frames`.
3. Dispatch A1 as a goal-term tail call with `cp =
   state.catch_return_pc`.

`execute_throw/1`:
1. Walk `state.catcher_frames` from the top.
2. For each frame, attempt unification of the thrown term against
   `frame.catcher_term`. On success: restore state from frame,
   bind catcher pattern's vars, dispatch `frame.recovery_term`
   with `cp = frame.saved_cp`, drop popped frames.
3. If walk exhausts the stack: print diagnostic to stderr,
   return `:fail`.

`:catch_return` op (single auto-injected instruction): pop the
top `catcher_frames` frame, set `cp = frame.saved_cp`, proceed.

### 4.2 Reuse vs duplicate the existing `:fail` mechanism

Elixir's existing `try / catch {:fail, state}` pattern is
unrelated to user-level Prolog `throw/1` — they share BEAM
mechanics but tag values differently:

- Internal: `throw({:fail, state})` — caught by `backtrack/1`,
  drives WAM choice-point retry.
- User-level (new): `throw({:wam_throw, prolog_term, _})` — caught
  by the dispatch loop, drives Prolog `catch/3`.

The two never collide because the catch arms tag-discriminate.
Unit-tested before any code lands.

(Note: this is *internal* BEAM throw mechanics. We're not yet
switching to Option B from PHILOSOPHY §5 — that's a follow-up
optimisation. PR #2 still ships side-stack frames; the BEAM
`throw` here is just how the dispatch loop receives the
exceptional return.)

### 4.3 Distinguishing `catch/3` from existing dispatch

The C++ generator dispatches `catch/3` as `execute "catch/3"` /
`call "catch/3", 3`. The runtime's instruction handler matches on
the operator string:

```cpp
if (instr.a == "catch/3") { cp = pc + 1; return execute_catch(); }
```

Elixir mirror:

```elixir
{:call, "catch/3", 3} -> %{state | cp: state.pc + 1} |> execute_catch()
{:execute, "catch/3"} -> execute_catch(state)
{:call, "throw/1", 1} -> execute_throw(state)
{:execute, "throw/1"} -> execute_throw(state)
```

Same dispatch shape; clauses go in the existing `step/2`
dispatch.

## 5. Test strategy

Three layers per PR.

- **Unit tests** for generator-side helpers (config loader,
  rewrite, audit). Pure Prolog, no Elixir compile/run.
- **e2e tests** that compile a fixture predicate and run the
  generated Elixir, asserting on the program's output.
  `tests/test_wam_elixir_target.pl` is the existing harness.
- **Regression tests** that mirror the C++ test set so any
  divergence in observable behaviour fails loudly.

The e2e tests are slow (Elixir compile + run per case). PR #2
adds ~6 e2e cases; PR #4 adds ~5; PR #5 adds ~21. Roughly 30s
of incremental CI time across the series; acceptable.

## 6. Performance constraints

Per PHILOSOPHY §3, the right comparison is
Elixir-before-PR vs Elixir-after-PR. Three call sites per
feature:

| Site | Required behaviour | How verified |
|---|---|---|
| Happy path (no catch active, default-mode pred) | Byte-identical to baseline. No new branches in step dispatch unless guarded by zero-cost compile-time pattern match. | Bench fixtures from `tests/bench_wam_*.pl` show <1% delta. |
| Feature-active path (catch frame present, no throw) | Quantified per-PR. | Bench fixture with N nested catchers; report µs/instr delta. |
| Exceptional path (throw triggered) | No constraint beyond "completes". | Bench omitted; correctness-only. |

Per-PR bench protocol:
1. Save `bench_wam_elixir_baseline_<sha>.txt` from `main` before
   the PR's first commit.
2. Re-run after the PR's last commit; diff into PR description.
3. If happy-path delta exceeds 1%, pause and investigate.

We do NOT bench Elixir against Rust or Haskell as part of PR
acceptance. Cross-target results live in
`WAM_CROSS_TARGET_BENCHMARK_RESULTS.md` and update on their own
cadence.

### 6.1 Specific bench fixtures

- `bench_wam_elixir_no_catch.pl` — pure arithmetic loop, no
  catch/throw on the call path. Gates the happy-path constraint
  for every PR.
- `bench_wam_elixir_catch_unused.pl` — predicate body sits inside
  `catch(Goal, _, _)` but never throws. Gates the
  feature-active path constraint for PR #2 onwards.
- `bench_wam_elixir_iso_lax_compare.pl` — ISO-mode predicate
  with arith compares; lax-mode predicate with same compares.
  Gates the rewrite-cost constraint for PR #4 / PR #5.

Fixtures land with PR #2; PR #1's `call/N` audit may also
pre-land them if it adds bench infrastructure incidentally.

## 7. Open architectural questions

These don't block PR #2 but should be settled before PR #5.

- **Q1: Lift to BEAM `throw` for the unwinding mechanics
  (Option B from PHILOSOPHY §5)?** Defer until PR #2's bench
  fixtures show whether the side-stack walk is a measurable
  cost.
- **Q2: Share `is_lax/2` body with default `is/2` via clause
  ordering, or use distinct function clauses?** C++ uses an
  `if (op == "is/2" || op == "is_lax/2")` guard. BEAM's
  function-clause pattern matching may make a distinct clause
  cheaper than the OR; verify with PR #4's bench.
- **Q3: Audit predicate naming — `wam_elixir_iso_audit/3` or
  generic `wam_iso_audit/4`?** If a third target adopts ISO
  errors before this work is done, extract. Otherwise keep
  Elixir-specific to mirror C++ exactly.
- **Q4: Should generator-side ISO config loader be shared with
  C++?** Both targets read the same Prolog config-file shape.
  A shared `iso_errors_config_loader.pl` would dedupe ~80 LOC.
  Worth doing in a separate refactor PR after both targets ship.

## 8. Files touched (estimated)

Per-PR LOC estimates, generator + tests:

| PR | Generator | Tests | Total |
|---|---:|---:|---:|
| #1 — `call/N` audit + completion | 50–150 | 60 | 110–210 |
| #2 — `catch/3` + `throw/1` | 250 | 100 | 350 |
| #3 — ISO plumbing | 200 | 80 | 280 |
| #4 — `is_iso/2` + `is_lax/2` | 80 | 60 | 140 |
| #5 — ISO sweep | 250 | 200 | 450 |
| **Total** | **830–930** | **500** | **~1.4 KLOC** |

Sized to land over ~5 PRs in normal review cadence, with PR #2
the largest single change (mirrors C++'s 275 LOC for the same
feature).

## 9. Out-of-scope follow-ups

After this roadmap completes, natural follow-ups:

- **`bagof/3` + `setof/3`** — mirror C++ PR series #2086 → #2112.
  Larger scope; needs witness-grouping infrastructure and the
  `aggregate_next_group` synthetic op.
- **Items API migration** — Elixir is #6 in the
  `WAM_ITEMS_API_SPECIFICATION.md` Phase 2 plan. Drops the 56
  `format(string(...))` calls.
- **Option B for catch/throw** — if PR #2 bench shows side-stack
  walk is a hot-path cost, lift unwinding mechanics to BEAM
  `throw` while keeping state management explicit.
- **Shared ISO config loader** — extract once a third target
  adopts ISO errors.
- **Runtime parser** (`prolog_term_parser.pl` transpiled to
  Elixir) — only when Elixir grows `read/2` /
  `read_term_from_atom/2,3`. Tracked under
  `RUNTIME_PARSER_TRANSPILATION_SPECIFICATION.md`.
