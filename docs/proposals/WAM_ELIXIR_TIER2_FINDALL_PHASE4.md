# Elixir WAM findall — Phase 4: parallel under Tier-2

**Status:** Draft for design review (pre-implementation).
**Target audience:** Perplexity review + Phase 4 implementer.
**Depends on:** All of `WAM_ELIXIR_TIER2_FINDALL.md` shipped via PRs #1627, #1643, #1647, #1649, #1650, #1658, #1659, #1661, #1663, #1667, #1669. Tier-2 wiring (`par_wrap_segment/4`, super-wrapper template) shipped via #1608, #1624, #1646.
**Companion proposal:** `WAM_ELIXIR_TIER2_FINDALL.md` (sequential findall — completed Phase 3c).

## 0. Implementation status

| Phase | Scope | Status |
|---|---|---|
| Phase 1–3c | Sequential findall + aggregate_all complete across all proposal §6 risks | ✅ Shipped |
| Phase 4a | Substrate — branch-local accum, parent-merge protocol, agg-frame-aware branch backtrack | ⏳ Planning |
| Phase 4b | Super-wrapper rework — replace #1608's stub with a findall-aware fan-out | ⏳ Planning |
| Phase 4c | Runtime integration tests — activate `intra_query_parallel(true)` for the existing Phase 3 scenarios where the inner goal is Tier-2-eligible | ⏳ Planning |

## 1. Context

The Tier-2 super-wrapper from PR #1608 (emitter) and PR #1624 (wiring activation) emits a `cond do` block that gates on `in_forkable_aggregate_frame?/1` and `parallel_depth > 0`. The intent: when the inner goal of a findall is a multi-clause pure predicate, fan its clauses out via `Task.async_stream` and merge results.

The super-wrapper's parallel arm is currently **dead code at runtime**. Phase 3 scenarios kill-switch it via `intra_query_parallel(false)`. Finding 2 from PR #1647 is the proximate reason: when the parallel arm fires, branches return state tuples instead of solution values, and `merge_into_aggregate/2` writes the wrong shape into the shared `:agg_accum`. The fix isn't local — it requires a different protocol for branch execution and result merging, which the Haskell target solved via the branch-local-accum pattern at `wam_haskell_target.pl:1502-1516`.

This proposal describes the protocol needed to make parallel findall work correctly on the WAM-Elixir target.

## 2. Goal

When `findall(Template, Goal, L)` calls a Tier-2-eligible predicate (≥3 clauses, declared/inferred pure, kill-switch off), the predicate's clauses fan out via `Task.async_stream`. Each branch runs in isolation, accumulates its own local solutions, and returns them to the parent. The parent merges branch contributions into the agg frame's `:agg_accum`, then finalises normally.

**In scope:**
- Parallel `findall/3` and `aggregate_all/3` over Tier-2-eligible inner predicates.
- All aggregator types from #1663 (`:findall`/`:bag`/`:set`/`:sum`/`:count`/`:max`/`:min`).
- Cut barrier preservation across the parallel boundary (cut inside one branch must not affect siblings).

**Out of scope (Phase 4 limitations):**
- Nested parallel — Tier-2-eligible inner goal whose clauses themselves contain Tier-2-eligible findalls. Nested-fork suppression (parallel_depth > 0 gate) shifts these to sequential.
- Side-effecting inner goals — the purity gate already excludes these.
- `bagof/3`, `setof/3` — witness-variable semantics, separate proposal (out of scope of the Phase 1–4 findall proposal series).

## 3. Current state

### 3.1 What exists

- **Tier-2 super-wrapper template** (#1608, `par_wrap_segment/4` in `wam_elixir_lowered_emitter.pl`): emits the `cond do` block with `Task.async_stream` fan-out and `merge_into_aggregate` call.
- **Tier-2 wiring activation** (#1624): `render_compiled_module/8` calls `par_wrap_segment/4`; the super-wrapper takes the `clause_main` slot when the gate passes; per-clause bodies move to `*_impl`.
- **`merge_into_aggregate/2`** (#1586): walks `state.choice_points` to find the topmost agg frame and appends a list to `:agg_accum`. Used by the current super-wrapper.
- **`in_forkable_aggregate_frame?/1`** (#1586): the gate. Returns true when an `:agg_type` of `:findall` or `:aggregate_all` is on the CP stack.
- **`parallel_depth` field** (#1586): nested-fork suppression — cleared at fork entry, gate rejects when `> 0`.

### 3.2 What's broken

The existing super-wrapper's Task.async_stream block:

```elixir
branches
|> Task.async_stream(fn branch ->
     try do
       branch.(branch_state)   # branch is &clause_main_impl/1 etc.
     catch
       {:fail, _state} -> []
       {:return, result} when is_list(result) -> result
       {:return, result} -> [result]
     end
   end, ...)
|> Enum.flat_map(fn
     {:ok, solutions} when is_list(solutions) -> solutions
     _ -> []
   end)
```

Three problems for findall:

1. **`branch.(branch_state)` returns `{:ok, state}`** (via `terminal_cp` after the branch's clause completes its post-end_aggregate sub-segment, post-#1669). That's a tuple, not a list. The `:return` catch pattern doesn't match it; the `Enum.flat_map` discards it.

2. **Each branch's post-end_aggregate path runs `deallocate + proceed`** assuming top-level invocation context. When the branch returns to `terminal_cp`, the agg frame's update_topmost_agg_cp pointer (#1669) is consumed in the branch's task — but the **parent's** state never sees that update.

3. **Branches share the parent's agg frame indirectly via the snapshot**, but `merge_into_aggregate` operates on the *parent's* `state.choice_points`, not the branches'. Each branch's aggregate_collect mutates its own (immutable) snapshot copy. The collected values vanish when the branch returns.

### 3.3 Haskell precedent

The Haskell target (`wam_haskell_target.pl:1480-1539`) solved this via the branch-local-accum protocol:

- **`runBranchLoop`** drives branch execution with a *custom backtrack*. When a branch's normal control flow encounters its own aggregate-frame CP (via backtrack after enumeration exhaustion), instead of calling `finalizeAggregate` (which would clear `wsAggAccum` and resume at `agg_return_pc`), it returns the branch's local `wsAggAccum` to the parent.
- **`forkParBranches`** sparks branches via `parMap rdeepseq`, each returning its local accum. The parent concatenates: `allValues = concat branchResults`. Then merges into the parent's `wsAggAccum` and calls `finalizeAggregate` on the combined result.
- **Nested-fork suppression** rewrites `Par*` instructions to sequential equivalents inside branches.

## 4. Proposed design

### 4.1 Branch-local accumulator pattern (parallel to Haskell)

Each branch runs in its own Task with a snapshot of the parent's state. The branch's local agg frame (the same agg frame the parent has, but in the branch's snapshot) is treated as a **branch-completion marker** rather than a finalisation trigger. When the branch's backtrack encounters this frame, the branch returns its local `:agg_accum` to the parent without finalising.

The parent's super-wrapper:
1. Captures the parent's agg frame's snapshot before forking.
2. Dispatches each branch with `Task.async_stream`.
3. Each branch returns a list of solution values (its local accum).
4. The parent flattens all branch results.
5. Calls `merge_into_aggregate(parent_state, all_branch_results)` to feed the parent's agg frame.
6. Throws `{:fail, parent_state}` to drive the standard backtrack→finalise flow on the parent.

### 4.2 New runtime helper: `branch_backtrack/1`

`backtrack/1` currently dispatches aggregate-frame CPs to `finalise_aggregate/4` (added in #1627). For the parallel-branch context, we need a variant that **returns the local accum** instead of finalising:

```elixir
def branch_backtrack(state) do
  case state.choice_points do
    [] -> {:branch_exhausted, []}   # no agg frame found; branch produced nothing
    [cp | _rest] ->
      case Map.get(cp, :agg_type) do
        nil -> backtrack_ordinary(state, cp, tl(state.choice_points))
                # ↑ existing path; resume the next clause CP
        _agg_type -> {:branch_exhausted, Enum.reverse(cp.agg_accum)}
                # ↑ branch's own agg frame reached; return local accum
      end
  end
end
```

Branches use `branch_backtrack` instead of `backtrack` for their wrap_segment catch arms (or via a different super-wrapper-installed catch).

### 4.3 Lowering — branch entry wrapper

Each branch in the super-wrapper's `Task.async_stream` block runs through a wrapper that:

1. Sets `state.cp` to a special `:branch_return_cp` that returns the branch's local accum on invocation.
2. Calls the clause's `*_impl` function.
3. Catches `{:branch_exhausted, accum}` (thrown via `branch_backtrack`'s alternative throw shape, or returned via the stub cp).
4. Returns the accum (a list) to the Task wrapper.

Sketch:

```elixir
fn branch ->
  branch_state = %{state |
    cp: fn s -> {:branch_exhausted, []} end,   # stub for normal completion
    cut_point: state.choice_points,
    parallel_depth: Map.get(state, :parallel_depth, 0) + 1
  }
  try do
    case branch.(branch_state) do
      {:branch_exhausted, accum} -> accum
      _ -> []
    end
  catch
    {:fail, s} ->
      case WamRuntime.branch_backtrack(s) do
        {:branch_exhausted, accum} -> accum
        _ -> []
      end
  end
end
```

### 4.4 MergeStrategy — separate from `:agg_type`

Haskell's `inferMergeStrategy` separates "what is the aggregate" from "how to combine parallel branch contributions." Our Elixir runtime conflates these in `finalise_aggregate/4` (folds `:agg_accum` per `:agg_type`). For Phase 4 the merge strategy is implicit:

- `:findall` / `:bag` / `:collect` / `:aggregate_all` → concat branch accums (preserves multiset; order is non-deterministic but the aggregator is order-independent by Tier-2's purity gate)
- `:set` → concat then `Enum.uniq` at finalise time (existing behaviour handles this)
- `:sum` → concat (each accum element is a value to sum); `Enum.sum` at finalise sums them all
- `:count` → concat; `length` at finalise counts them
- `:max` / `:min` → concat; `Enum.max`/`Enum.min` at finalise picks the global extremum

So the existing `finalise_aggregate/4` works unchanged — concatenation is the universal merge for the agg types we support. No new `MergeStrategy` abstraction needed at the runtime layer; the differentiation lives entirely in `finalise_aggregate`'s per-type case.

### 4.5 Cut barrier across the parallel boundary

Cut inside a branch's inner-goal evaluation should:
- Prune the branch's own enumeration (existing semantics from #1667 — cut preserves agg frames as cut barriers, just truncates ordinary CPs above)
- **Not** affect sibling branches (each branch has its own snapshot of `choice_points`; mutations are local)
- **Not** affect the parent's agg frame after merging (the parent's state is independent of the branches' snapshots)

The existing cut fix from #1667 already handles this correctly because Elixir's immutable maps give branches their own copies. No additional cut work needed.

### 4.6 Nested-fork suppression — already in place

The `parallel_depth > 0` gate in the super-wrapper rejects nested forks (sets `parallel_depth: parent_depth + 1` on entry; nested findall's super-wrapper sees `> 0` and short-circuits to sequential). This is the same insight as Haskell's `Par* → seq` rewrite at lines 1488-1494. Already shipped via #1624.

## 5. Approaches considered and rejected

### 5.1 Mutate parent's `:agg_accum` from branches via a GenServer or ETS table

Have an external state holder (GenServer or ETS) that branches write to as they collect solutions. Parent reads at the end. **Rejected**: introduces side-effecting state outside the WAM model, complicates testing (can't reproduce a state by replaying instructions), and BEAM's existing parMap idiom (snapshot + return) is well-understood. Branch-local accum is the standard pattern.

### 5.2 Throw the branch's accum from inside the branch via a tagged exception

Instead of returning, branches throw `{:branch_done, accum}`. **Considered seriously**: this is what the stub-cp approach essentially does, just with a tagged tuple instead of a thrown exception. The catch shape is similar. Slight preference for return-via-stub-cp because it keeps the throw channel for actual failure (`{:fail, state}`).

### 5.3 Run branches sequentially inside the super-wrapper

Just iterate the clauses sequentially under a `parallel_depth > 0` guard. **Rejected**: defeats the entire point of Tier-2. The whole purpose of the super-wrapper is parallel speedup; sequential fallback is what the gate-reject path does.

### 5.4 Branch-local accum with parent merge (chosen)

Pattern A from §4. Mirrors the Haskell precedent; lowest friction with the existing substrate (no new global state, no protocol changes to existing helpers, the merge step uses the existing `merge_into_aggregate/2` from #1586).

## 6. Risks and open questions

1. **Branch return shape consistency.** Each branch must return a list (its local accum). The wrapper in §4.3 enforces this via the `case` pattern match plus the `:branch_exhausted` tuple. Probe needed: confirm clause bodies that succeed normally (proceed → state.cp.(state)) return cleanly via the stub cp without leaking other return shapes.

2. **Snapshot cost at fork.** Each branch gets a copy of the parent's state. For large heaps or deep CP stacks, this is O(N). Acceptable in BEAM (immutable structures share memory) but worth measuring. Haskell's `parMap rdeepseq` does deep-copying; BEAM's send-to-process for `Task.async_stream` does similar.

3. **Heap address conflicts across branches.** Each branch's `put_structure` writes to `state.heap[heap_len..]`. Different branches share the parent's heap_len at fork time, so all branches start writing at the same address. Different branches' heaps are independent (each is an immutable copy). When branches return their accums (via `deep_copy_value` per #1663), the values are self-contained — no heap refs survive across the parallel boundary. ✓

4. **Cut barrier with parallel branches and nested findall.** A Tier-2-eligible predicate whose body contains its own findall would have its outer cut interact with the inner agg frame. The parallel_depth > 0 gate prevents the inner findall from forking, so the inner runs sequentially with the existing cut-barrier protection from #1667. Probe needed: explicit test fixture.

5. **`Task.async_stream` ordering.** Branches complete in non-deterministic order. For order-independent aggregators (the only ones Tier-2 allows) this is fine. But the **test assertions** need to be order-tolerant. Phase 3 substring matching is already order-tolerant; explicit list-equality assertions in Phase 4 tests should sort or use set comparison.

6. **Branch failure modes.** A branch's clause may legitimately fail (no head match for the inner goal's args). The branch should return `[]` (no contribution). The catch arm in §4.3 handles this via `:branch_exhausted` from `branch_backtrack`. Probe needed: confirm a branch whose head fails immediately produces `[]`, not a dropped exception.

7. **Re-entry into parallel from inside a branch.** Branch i is running, it makes a recursive call to itself (or another Tier-2-eligible predicate). The recursive call's super-wrapper sees `parallel_depth > 0` and goes sequential. ✓ Already handled by the existing gate.

8. **Determinism for measurement.** Phase 4 perf testing wants reproducibility. `Task.async_stream` with `ordered: false` is non-deterministic; `ordered: true` is deterministic but pays sync cost. The proposal's `ordered: false` choice matches Haskell precedent and yields better throughput; perf tests should run multiple iterations and take medians.

## 7. Test matrix

**Phase 4a (substrate):**

- Unit test: `branch_backtrack/1` returns `{:branch_exhausted, []}` from empty CP stack.
- Unit test: `branch_backtrack/1` returns `{:branch_exhausted, [v1, v2]}` from a state with an agg frame holding `agg_accum: [v2, v1]` (reversed).
- Unit test: `branch_backtrack/1` falls through to `backtrack_ordinary` when the topmost CP is not an agg frame (i.e., a try_me_else CP from clause enumeration).

**Phase 4b (lowering):**

- Emit-and-grep: super-wrapper's Task.async_stream block contains the new branch wrapper shape (stub cp, catch on `:branch_exhausted`).
- Emit-and-grep: super-wrapper invokes `merge_into_aggregate(parent_state, all_branch_results)` then throws fail to drive parent finalise.

**Phase 4c (runtime integration):**

Activate `intra_query_parallel(true)` for these scenarios from Phase 3:

- Scenario 1: `findall(X, phase3_smoke_p(X), L) → [a, b, c]`. With Tier-2 active, phase3_smoke_p (3 clauses) fans out. Result must contain all three values; order is non-deterministic so use set comparison.
- Scenario 11: nested findall. Outer parallel, inner sequential (per nested-fork suppression). Result `[[a, b, c]]`.
- Scenario 16: two inline findalls. Each independently parallelisable (or sequential if the inner predicate is < 3 clauses). Both `L` and `Rest` contain `[1, 2, 3]`.
- New scenario: `findall(p(X, Y), phase3_q(X, Y), L)` — compound Template, parallel inner predicate. Tests interaction of compound construction with the parallel boundary.
- New scenario: `findall(X, (phase3_smoke_p(X), !), L)` — cut inside parallel inner. Cut prunes its branch's enumeration (one solution from each clause); siblings unaffected.

**Cross-validation:**

Compare results to the Haskell target's parallel findall output for each scenario. Haskell has had the protocol shipping for some time; matching results = strong correctness signal.

## 8. Rollback plan

Phase 4 introduces a new `branch_backtrack/1` runtime helper and reworks the super-wrapper's Task.async_stream block. If a bug surfaces post-merge:

- Reverting the super-wrapper to the #1608 stub restores the kill-switched-by-default state. `intra_query_parallel(true)` would re-trigger the same Finding 2 broken behaviour, so callers should re-add `intra_query_parallel(false)` to their project options.
- The substrate helper `branch_backtrack/1` is additive and harmless if unused.
- The `merge_into_aggregate/2` semantics are unchanged.

## 9. Questions for design review

1. **Is `branch_backtrack/1` the right shape?** Specifically, returning `{:branch_exhausted, accum}` vs. throwing a tagged tuple. The proposal favours return-via-stub-cp because it keeps the throw channel for actual `{:fail, state}` propagation.

2. **Should `MergeStrategy` be a first-class abstraction?** Haskell separates it. Our analysis (§4.4) shows concat is universal for the agg types we support, so the runtime layer doesn't need it. Worth adopting Haskell's separation anyway for future extensibility (custom aggregators, non-commutative merges)?

3. **Branch-local cp stub vs. branch-local backtrack hook.** §4.3 uses a stub `state.cp = fn s -> {:branch_exhausted, []} end` to handle "branch's clause completed normally, no inner enumeration to drive." Alternative: have the branch wrapper run a loop that calls `branch_backtrack` until exhaustion. The stub-cp approach handles single-solution clauses cleanly; the loop handles multi-solution gracefully but requires more setup. Both work. Proposal favours stub-cp for simpler control flow.

4. **Determinism in tests.** The proposal recommends order-tolerant assertions for Phase 4c (set comparison). Should Phase 4c also include an `ordered: true` variant for tests that care about determinism? Probably not — order independence is Tier-2's defining property. Tests that need ordering should use sequential.

5. **Cross-validation against Haskell — formalise as CI?** The Haskell target's parallel findall is a known-good reference. Could be a recurring CI step that compiles the same Prolog through both pipelines and asserts result equality (modulo order). Useful long-term but probably out of scope for the initial Phase 4 PR — would require a Haskell build environment in CI.

## 10. What this proposal does NOT cover

- **`bagof/3` / `setof/3`** — separate witness-variable design.
- **Side-effecting inner goals.** Tier-2's purity gate excludes these.
- **Custom merge strategies** for user-defined aggregators.
- **Performance tuning** of `Task.async_stream`'s `max_concurrency`. Default is `System.schedulers_online()`, which is reasonable.
- **`forkMinBranches` threshold tuning.** Stays at the hardcoded 3 from #1608 / Haskell's convention.
- **Cross-target parity automation.** Manual cross-validation (compile-and-compare) recommended; CI integration deferred.
- **Profiling and optimisation.** The Phase 4 protocol is correctness-first. Perf tuning (snapshot cost, ordered vs. unordered, max_concurrency) waits on measurements.
