# Phase 4d Handoff: Nested-Fork Suppression Test

Branch: `feat/wam-elixir-tier2-phase4d-nested-fork-suppression-test`

## Goal

Add an explicit runtime test for the Tier-2 nested-fork suppression gate
documented in the Phase 4 proposal §4.6 / risk #4
(`docs/proposals/WAM_ELIXIR_TIER2_FINDALL_PHASE4.md`, line 205).

The gate is in `wam_elixir_lowered_emitter.pl:1194`:

```elixir
Map.get(state, :parallel_depth, 0) > 0 ->
  <EntryImplFunc>(state)   # fall back to sequential
```

When a Tier-2 parallel fan-out is in progress (`parallel_depth = 1`),
any nested Tier-2-eligible predicate's super-wrapper sees depth > 0 and
runs its clauses sequentially via `_impl` instead of spawning another
`Task.async_stream`. This prevents scheduler exhaustion from unbounded
fork nesting.

The Phase 4c test (merged, PR #1710) did NOT exercise this path — all
Phase 4c scenarios have flat (non-nested) Tier-2 predicates.

---

## What Is Already Done (committed)

### 1. `branch_sentinel` fix — Phase 4b bug discovered by Phase 4d probe

**Bug (Phase 4b, `wam_elixir_target.pl:383`):**
`backtrack/1` returned `{:branch_exhausted, accum}` for ANY agg-typed
CP when `branch_mode: true`. When a nested inner findall's agg CP
(pushed inside a branch body) reached `backtrack`, it returned the
inner accum as the branch result instead of finalising the inner findall
normally. Symptom: 9 unbound vars (3 outer branches × 3 anonymous inner
template vars from `findall(_, inner(_), _)`) instead of `['a','b','c']`.

**Fix (committed in this branch, `wam_elixir_lowered_emitter.pl:1214`):**
The super-wrapper now stamps the topmost CP at branch entry with
`branch_sentinel: true`:

```elixir
[parent_agg_cp | rest_cps] = state.choice_points
stamped_parent = Map.put(parent_agg_cp, :branch_sentinel, true)
branch_state = %{state |
  branch_mode: true,
  ...
  choice_points: [stamped_parent | rest_cps]
}
```

`backtrack/1` now guards on BOTH `branch_mode` AND `branch_sentinel`
(`wam_elixir_target.pl:341`):

```elixir
Map.get(state, :branch_mode, false) and
    Map.get(cp, :branch_sentinel, false) ->
  {:branch_exhausted, Enum.reverse(Map.get(cp, :agg_accum, []))}
```

Nested agg CPs (no sentinel) `finalise_aggregate` normally.

**Tests updated:** `test_findall_phase4b_backtrack_dispatches_on_branch_mode`
and the super-wrapper emit test in `tests/test_wam_elixir_target.pl` —
all passing.

### 2. Phase 4d test file written — `tests/test_wam_elixir_lowered_phase4d.pl`

Runtime smoke test. Mirrors Phase 4c structure. One scenario:
- `phase4d_inner/1`: 3-clause pure predicate (Tier-2 eligible).
- `phase4d_outer_p/1`: 3-clause pure predicate. Each clause body calls
  `findall(_, phase4d_inner(_), _)`.
- `phase4d_nested/1`: single-clause top-level whose `findall(X, phase4d_outer_p(X), L)`
  fans the outer into 3 branches. Each branch calls inner's
  super-wrapper at `parallel_depth=1` → gate fires → sequential.
- Expected result: `L = ['a','b','c']` (set-equality assertion).

**Current status: test is FAILING.** Exit code 0 but assertion fails — the
result is `[]` (empty list). The `branch_sentinel` fix was confirmed
correct (no longer leaking inner accum), but now the outer collects
nothing.

---

## The Remaining Problem: Register Clobbering in the Fixture

### Root cause analysis

In `phase4d_outer_p('a') :- findall(_, phase4d_inner(_), _)`:

The WAM lowers this body to roughly:

```elixir
# Head match: bind reg A1 = "a"
state = bind(state, 1, "a")               # regs[original_id] = "a", regs[1] = original_ref

# Inner findall template setup (begin_aggregate for findall(_, ...)):
fresh = {:unbound, make_ref()}
state = state
  |> put_reg(102, fresh)                  # reg 102 = fresh (template var)
  |> put_reg(1, fresh)                    # reg 1  = fresh (OVERWRITES 'a' link!)

state = push_aggregate_frame(state, :findall, 102, 101)
# ^ Saves regs snapshot: regs[1]=fresh, regs[102]=fresh.
#   The original_id→"a" binding still exists in the map but reg 1
#   now points to `fresh`, not `original_id`. Deref of reg 1 = unbound.

# Inner goal arg
fresh2 = {:unbound, make_ref()}
state = state |> put_reg(103, fresh2) |> put_reg(1, fresh2)
state = %{state | cp: &k1/1}
WamDispatcher.call("phase4d_inner/1", state)
```

After inner finalise, `restore` reinstates `inner_agg_cp.regs` (the
snapshot), which has `regs[1] = fresh` (unbound). The outer findall's
`aggregate_collect(state, X_reg)` where `X_reg = 1` deref's an unbound
var — collecting nothing meaningful.

The issue is `put_reg(1, fresh)` overwriting `reg 1` with the inner
template/goal unbound, breaking the chain from `reg 1` to the
head-bound constant `"a"`.

### Is this a pre-existing compiler bug or fixture-specific?

**Unknown — not yet confirmed.** We generated a sequential project at
`$PREFIX/tmp/phase4d_seq/` (using `intra_query_parallel(false)`) to
investigate, but the session ended before running it.

To confirm: run
```
elixir $PREFIX/tmp/phase4d_seq/smoke_driver.exs
```
(you'll need to write a `smoke_driver.exs` first — see the parallel
version as template).

If the sequential run also returns `[]`, it is a pre-existing WAM
compiler register-allocation bug affecting this clause shape. The
fix then belongs in the compiler (not Phase 4d scope).

If the sequential run returns `['a','b','c']`, the parallel path has
an additional issue that needs investigating.

---

## How to Fix the Test Fixture (Recommended)

Regardless of whether there's a compiler bug, the simplest fix is to
redesign the fixture so the outer clause's head-bound value ('a'/'b'/'c')
is NOT in register A1 at the time the inner findall's `begin_aggregate`
runs. Two approaches:

### Option A — Indirect call (cleanest)

Add a single-clause helper that does the inner findall. The outer's
clauses just call the helper; their head-bound values are collected
before the inner findall ever touches A1:

```prolog
user:phase4d_outer_p('a') :- phase4d_do_inner.
user:phase4d_outer_p('b') :- phase4d_do_inner.
user:phase4d_outer_p('c') :- phase4d_do_inner.

% Not Tier-2 eligible (1 clause), but calls phase4d_inner's
% super-wrapper at parallel_depth=1 → gate fires.
user:phase4d_do_inner :- findall(_, phase4d_inner(_), _).
```

This avoids the clobbering entirely: the outer's `aggregate_collect`
runs (via `state.cp` set before calling `phase4d_outer_p`) before
`phase4d_do_inner` is ever entered.

### Option B — Named template variable

Use a named (non-anonymous) variable for the inner template. A named
variable gets a separate register (possibly a Y-reg via `allocate`),
not shared with A1:

```prolog
user:phase4d_outer_p('a') :- findall(Y, phase4d_inner(Y), _Ls).
user:phase4d_outer_p('b') :- findall(Y, phase4d_inner(Y), _Ls).
user:phase4d_outer_p('c') :- findall(Y, phase4d_inner(Y), _Ls).
```

This may or may not fix the issue depending on how the register
allocator assigns Y — worth trying but Option A is more reliable.

---

## Key Files

| File | Role | Status |
|------|------|--------|
| `src/unifyweaver/targets/wam_elixir_lowered_emitter.pl` | Super-wrapper emission; `branch_sentinel` stamp at lines 1205–1220 | **Modified + committed** |
| `src/unifyweaver/targets/wam_elixir_target.pl` | `backtrack/1` dispatch; `branch_sentinel` guard at lines 341–343 | **Modified + committed** |
| `tests/test_wam_elixir_target.pl` | Emit-and-grep tests; Phase 4b/4d dispatch test updated | **Modified + committed** |
| `tests/test_wam_elixir_lowered_phase4d.pl` | Runtime smoke test; currently failing | **Added + committed** |
| `docs/proposals/WAM_ELIXIR_TIER2_FINDALL_PHASE4.md` | Proposal — §4.6 describes the gate; §6 risk #4 is the probe this PR closes | **Not modified** |

---

## Passing Tests

- `swipl -g run_tests -t halt tests/test_wam_elixir_target.pl` — all pass
- `swipl -g run_tests -t halt tests/test_wam_elixir_lowered_phase4c.pl` — all 3 pass (no regression)
- `swipl -g run_tests -t halt tests/test_wam_elixir_lowered_phase3.pl` — all pass (no regression)
- `swipl -g run_tests -t halt tests/test_wam_elixir_lowered_phase4d.pl` — **FAILS** (fixture issue)

---

## Recommended Next Steps

1. **Try Option A fixture** (indirect call via `phase4d_do_inner`). Update
   `tests/test_wam_elixir_lowered_phase4d.pl` — replace the three
   `phase4d_outer_p` clauses and add `phase4d_do_inner`. Add
   `user:phase4d_do_inner/0` to `phase4d_predicates` and
   compile/run.

2. **Verify Option A passes.** If it does, also verify
   `test_wam_elixir_target.pl` and Phase 4c still pass (regression check).

3. **Commit the fixture fix**, then push and open the PR.

4. **PR description:** This PR closes Phase 4 risk #4 (nested-fork
   suppression probe). Two deliverables: (a) `branch_sentinel` fix for
   Phase 4b's `backtrack/1` dispatch, (b) runtime fixture confirming
   nested Tier-2 doesn't fork-bomb and produces correct results.

---

## Background Context

Phase 4 sequence: 4a substrate (#1694) → 4b super-wrapper (#1706) →
4b.5 `_branch` variants (#1708) → 4c parallel runtime tests (#1710) →
4d this branch.

The `parallel_depth` gate at
`wam_elixir_lowered_emitter.pl:1194` was shipped in the Tier-2
activation PR (#1624 / #1586 area). What was missing was an explicit
runtime test that exercised the gate end-to-end under parallel mode.
Phase 4d provides that test, and surfaced the `branch_sentinel` bug in
the process.
