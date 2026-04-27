# Elixir WAM findall/aggregate_all proposal

**Status:** Phases 1–3 shipped (substrate, instruction lowering, runtime smoke + module-qualifier unwrap). Phase 3c edge cases and Phase 4 (parallel branch-local accum) remain.
**Target audience:** Perplexity review + Phase 3c / Phase 4 implementer.
**Depends on:** PR #1586 (Tier-2 substrate), PR #1608 (`par_wrap_segment/4`), PR #1624 (Tier-2 wiring activation), PR #1626 (this proposal).
**Shipped:** PR #1627 (substrate — `push_aggregate_frame/4`, `aggregate_collect/2`, `finalise_aggregate/4`, `backtrack/1` dispatch). PR #1643 (instruction lowering). PR #1647 / #1649 (Phase 3a/3b runtime smoke). PR #1650 (`compile_findall` Y-reg fix). The current PR (static module-qualifier unwrap in `compile_goal_call/4` + `compile_goal_execute/4`) closes Finding 1 from #1647 end-to-end.

## 0. Implementation status

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Runtime substrate (4 helpers + backtrack dispatch) | ✅ Shipped #1627 |
| Phase 2 | `begin_aggregate` / `end_aggregate` lowering + `:collect → :findall` translation | ✅ Shipped #1643 |
| Phase 3a | Runtime harness + first scenario (3-clause sequential findall) | ✅ Shipped #1647 |
| Phase 3b | Runtime aggregator coverage (8 scenarios spanning `:findall` / `:count` / `:bag` / `:set` / `:max` / `:min`) | ✅ Shipped #1649 |
| WAM-compiler fix | `compile_findall` Y-reg allocation (Finding 1, byte-shape) | ✅ Shipped #1650 |
| Static module-qualifier unwrap | `compile_goal_call/4` + `compile_goal_execute/4` recognise `M:Goal` with atom/string M (Finding 1, end-to-end runtime closure) | ✅ This PR |
| Phase 3c | Edge cases — single-clause, Y-register Template deep-copy probe (§6 risk #1), cut in inner goal (§6 risk #3), nested findall (§6 risk #7) | ⏳ Next |
| Phase 4 | Tier-2 × findall — branch-local-accum protocol from Haskell | ⏳ Future |

Deviations from the proposal as written, documented in commits and code:

- **§4.1 CP shape:** the proposal listed a separate `:agg_return_cp` field, but it always equalled `:cp` at push time, so it was dropped during implementation. `finalise_aggregate/4` tail-calls via `restored.cp.(restored)`. Documented in `push_aggregate_frame/4`'s `@doc`.
- **§4.5 / §9 Q5 (`:collect` vs `:findall`):** translation lands at the emission site (`agg_type_atom/2`), keeping the substrate's existing alphabet (`:findall` / `:aggregate_all`) authoritative rather than broadening it. `in_forkable_aggregate_frame?/1`'s `@doc` documents the alphabet contract.
- **§6 risk #6 (empty-accumulator semantics) refined during implementation:** list aggregators (`:collect` / `:findall` / `:aggregate_all` / `:bag` / `:set`) return `[]`; `:sum` / `:count` use natural identities (`0`, `0`); `:max` / `:min` throw `{:fail, state}` because no identity exists and silently returning `nil` would propagate a non-WAM value into downstream `get_constant` unification. Documented in `finalise_aggregate/4`'s `@doc`.

## 1. Context

The WAM compiler at `src/unifyweaver/targets/wam_target.pl:755` already recognises `findall(Template, Goal, Result)` and `aggregate_all(Template, Goal, Result)` body goals. It compiles them into ordinary instructions sandwiched between two opcodes:

```
    begin_aggregate AggType, ValueReg, ResultReg
    ... ordinary WAM instructions for the inner goal ...
    end_aggregate ValueReg
```

The Goal is flattened (via `flatten_conjunction/2`) and lowered as ordinary `call` instructions at compile time — there is **no metacall**, no Goal-as-data at runtime.

Three other targets already lower these instructions:

- `wam_rust_target.pl:2659-2669` — emits `Instruction::BeginAggregate / EndAggregate`.
- `wam_go_target.pl:903-911` — emits the Go runtime variants.
- `wam_llvm_target.pl:2417-2496` — full LLVM IR implementation, including the fail-driven enumeration mechanism (most detailed precedent — see §3.3).

The Elixir lowered emitter has **zero** handler for either instruction. The Tier-2 substrate from PR #1586 added `in_forkable_aggregate_frame?/1` and `merge_into_aggregate/2` runtime helpers but no producer pushes the frame they read. As a result, the Tier-2 super-wrapper wired up in PR #1624 always short-circuits to sequential at runtime.

## 2. Goal

Add `begin_aggregate` / `end_aggregate` lowering to the Elixir target and the runtime mechanism that supports them, so `findall/3` and `aggregate_all/3` work end-to-end. As a side effect, the Tier-2 super-wrapper becomes operationally reachable on declared-pure inner goals.

**In scope:** `findall/3`, `aggregate_all/3` with collect/sum/count/max/min/bag/set aggregators (the set already supported by `compile_aggregate_all/5`).

**Out of scope:** `bagof/3`, `setof/3` (witness-variable semantics, separate proposal); user-defined aggregators; `findall/4` (variant taking a tail list).

## 3. Current state

### 3.1 What exists

- **WAM compiler:** emits `begin_aggregate`/`end_aggregate`. Handles all aggregator types via `compile_aggregate_all/5` (`wam_target.pl:710`). `findall/3` is a thin wrapper that calls `compile_aggregate_all` with `collect-Template` (`wam_target.pl:756`).
- **Elixir runtime substrate (PR #1586):** `WamRuntime.in_forkable_aggregate_frame?/1` reads `:agg_type`, returns true only for `:findall` / `:aggregate_all` (forkable types). `WamRuntime.merge_into_aggregate/2` appends a list of branch results to the nearest aggregate CP's `:agg_accum`. Both consumer-side; no producer.
- **Tier-2 super-wrapper:** when emitted, gates on `in_forkable_aggregate_frame?/1`. With no producer, the gate always returns false; the super-wrapper short-circuits to its `*_impl` sequential path.

### 3.2 What's missing (now: shipped status)

Three things, all in the Elixir target — **all three shipped across PRs #1627 and #1643**:

1. ✅ **`begin_aggregate` instruction lowering** (#1643). Push an aggregate-typed CP onto `state.choice_points` carrying `:agg_type`, `:agg_value_reg`, `:agg_result_reg`, `:agg_accum: []`, and a snapshot of `state.heap_len` / `state.trail_len` / `state.regs` so backtracking-to-finalize can restore. (Note: `:agg_return_cp` was dropped — see §0 deviations.)
2. ✅ **`end_aggregate` instruction lowering** (#1643). Read the value register, deref it, copy into the nearest aggregate CP's `:agg_accum`, then `throw({:fail, state})` to drive backtracking — fail-driven enumeration. The next inner-goal alternative gets re-tried; when none remain, backtrack pops the aggregate CP and finalizes.
3. ✅ **Backtrack-to-finalize semantics in `WamRuntime.backtrack/1`** (#1627). When the topmost CP popped is an aggregate frame, do not re-execute its `pc` (aggregate frames have no failure target). Instead: aggregate the `:agg_accum` per the `:agg_type`, bind the result to `:agg_result_reg`, restore the saved environment minus the popped frame, and tail-call the restored `state.cp`.

### 3.3 LLVM precedent (closest analogue)

The LLVM target (`wam_llvm_target.pl:2417-2496`) uses fail-driven enumeration:

- `begin_aggregate` pushes a CP with `agg_type`, `agg_value_reg`, `agg_result_reg`, `agg_return_pc`, plus a `saved_heap_top`. Resets the accumulator counter.
- `end_aggregate` reads the value register, pushes it onto the accumulator via `wam_agg_push`, updates the agg frame's `agg_return_pc` to PC+1, then **returns failure** — explicit fail to drive backtrack.
- Backtrack hits the agg frame, recognises it (`agg_type != 0`), and either resumes the inner goal's prior CP (more solutions to enumerate) or finalises (no more solutions: bind accumulator → result reg, jump to `agg_return_pc`).

The Elixir design will mirror this control flow but in BEAM idioms (struct-shaped CPs, `throw({:fail, ...})` rather than ret-false).

## 4. Proposed design

### 4.1 Aggregate CP shape

Extend the existing CP map with aggregate-frame fields. A normal try_me_else CP carries `pc, regs, heap, heap_len, cp, trail, trail_len, stack`. An aggregate CP additionally carries:

```elixir
%{
  # Sentinel: presence of :agg_type marks this as an aggregate frame.
  agg_type: :findall | :aggregate_all | :sum | :count | :max | :min | :bag | :set | :collect,
  agg_value_reg: integer,        # which register holds the per-solution Template value
  agg_result_reg: integer,        # which register receives the final aggregate
  agg_accum: list,                # accumulator (list of values, or running sum/count/etc.)
  agg_return_cp: (state -> any),  # continuation called after finalisation
  # Plus the usual CP snapshot fields (regs, heap, heap_len, trail, trail_len, stack)
  # so finalize can restore the pre-begin_aggregate state cleanly before resuming.
}
```

The presence of `:agg_type` is the discriminant — `in_forkable_aggregate_frame?/1` already reads it.

### 4.2 `begin_aggregate` lowering

```prolog
wam_elixir_lower_instr(begin_aggregate(AggTypeStr, ValueReg, ResultReg),
                       _PC, _Labels, _FuncName, _Suffix, Code) :-
    agg_type_atom(AggTypeStr, AggType),
    reg_id(ValueReg, ValReg),
    reg_id(ResultReg, ResReg),
    format(string(Code),
'    state = WamRuntime.push_aggregate_frame(state, ~w, ~w, ~w, state.cp)',
        [AggType, ValReg, ResReg]).
```

The `push_aggregate_frame/5` runtime helper builds the CP and pushes it onto `state.choice_points`. `state.cp` is captured as `:agg_return_cp` so finalize knows where to jump.

### 4.3 `end_aggregate` lowering

```prolog
wam_elixir_lower_instr(end_aggregate(ValueReg), _PC, _Labels, _FuncName, _Suffix, Code) :-
    reg_id(ValueReg, ValReg),
    format(string(Code),
'    state = WamRuntime.aggregate_collect(state, ~w)
    throw({:fail, state})', [ValReg]).
```

`aggregate_collect/2` derefs the value register, deep-copies the resolved value (so trail unwinding during backtrack doesn't mutate it), and prepends it to the nearest aggregate frame's `:agg_accum`. The throw drives backtracking.

### 4.4 `backtrack/1` extension

Today's `backtrack/1` (in `wam_elixir_target.pl:362`) pops the topmost CP and resumes at its `pc`. The extension: before resuming, check `:agg_type`:

```elixir
def backtrack(state) do
  case state.choice_points do
    [] -> :fail
    [cp | rest] ->
      case Map.get(cp, :agg_type) do
        nil ->
          # Ordinary CP — existing path.
          state = restore_from_cp(state, cp, rest)
          cp.pc.(state)
        agg_type ->
          # Aggregate frame — finalise.
          finalise_aggregate(state, cp, rest, agg_type)
      end
  end
end
```

`finalise_aggregate/4` aggregates `:agg_accum` per `:agg_type` (list reverse for `:collect`/`:findall`, sum for `:sum`, etc.), binds the result to `:agg_result_reg`, restores from the CP snapshot, and tail-calls `cp.agg_return_cp.(state)`.

### 4.5 Interaction with Tier-2

When the inner goal is a Tier-2-eligible predicate (declared pure, ≥ 3 clauses), the super-wrapper's `cond do` checks `in_forkable_aggregate_frame?(state)`. After `begin_aggregate` lands, this returns `true`. The super-wrapper takes the `Task.async_stream` arm, runs branches in parallel, and calls `merge_into_aggregate/2` with the batch results — directly populating `:agg_accum` without per-solution `end_aggregate` calls.

**Crucial property:** `merge_into_aggregate/2` already exists and writes to `:agg_accum`. Its results are visible to `finalise_aggregate/4` as ordinary accumulator entries. The two collection paths (sequential per-solution `end_aggregate` and parallel bulk `merge_into_aggregate`) share the same accumulator and the same finalise step.

When Tier-2 fans out, control does not flow through `end_aggregate` — the super-wrapper consumes the inner-goal's clauses directly and returns to the caller. So `finalise_aggregate` needs to be triggered explicitly after the super-wrapper completes. **Design choice (see §6 risk #2):** the super-wrapper, after `merge_into_aggregate`, throws `{:fail, state}` to drop into `backtrack/1`, which finds the aggregate frame and finalises. Same control flow as the sequential path.

## 5. Approaches considered and rejected

### 5.1 Eager collection inside the inner goal's tail

Have `compile_aggregate_all` emit code that directly accumulates into a Y-register list as the goal proceeds, without using a separate aggregate CP. **Rejected**: this requires the inner goal to know it's inside an aggregate, which breaks the local-reasoning property — the same predicate body would compile differently in/out of aggregate context. The aggregate-CP approach localises all aggregate-specific code at the surrounding `begin_aggregate`/`end_aggregate` boundaries.

### 5.2 Synchronous "find next solution" dispatcher

Instead of fail-driven enumeration, expose a `WamRuntime.next_solution/1` that the aggregate-finalise loop calls repeatedly until exhaustion. **Rejected**: Prolog's call-and-fail enumeration is exactly what `try_me_else` / `retry_me_else` / `trust_me` already implement. Re-implementing that dispatch loop on top of next_solution would duplicate the CP machinery. Fail-driven enumeration uses the existing CP chain unchanged.

### 5.3 Materialising the inner goal as a stream

Lazy stream evaluation — drive the goal via `Stream.unfold/2`. **Rejected**: BEAM's process-per-task model + the WAM trail's mutable bindings make lazy enumeration awkward. Snapshotting state for a yield-and-resume is more invasive than fail-driven enumeration. Worth revisiting only if perf measurements show enumeration overhead.

### 5.4 Fail-driven enumeration with finalise on backtrack (chosen)

Mirror the LLVM precedent. Lowest-friction interaction with existing CP machinery.

## 6. Risks and open questions

1. **Trail-unwind safety of `:agg_accum`.** ⏳ Open for Phase 3. When `end_aggregate` collects a value and triggers backtrack, the trail unwind restores variable bindings to the pre-CP state. If a Template value contains heap references to bound variables that the trail unwinds, the captured value mutates retroactively. The Phase 1 implementation uses `deref_var` to resolve the value before capture — atomic values (strings/numbers/atoms) are BEAM value types and survive unwind; compound values that reference heap cells need a recursive deep-copy. **Haskell precedent:** `wam_haskell_target.pl:2254` uses `derefVar (wsBindings s)` — same pattern, suggesting our atomic-value approach is sound for the simple cases. A Phase 3 probe with `findall(X, p(X), L)` where `p/1` produces compound terms will tell us if the deep-copy follow-up is needed.

2. **Tier-2 finalise trigger — sequential confirmed, parallel needs divergent path.** When the super-wrapper takes the parallel arm, control returns from `merge_into_aggregate/2` to the caller of the super-wrapper. The aggregate frame is still on `state.choice_points`.
    - (A) Super-wrapper throws `{:fail, state}` after merge — falls through to backtrack which finalises.
    - (B) Super-wrapper explicitly calls `finalise_aggregate` itself.

   **Updated guidance from Haskell precedent (`wam_haskell_target.pl:1502-1516`):** option (A) works for the **sequential** case but breaks for **nested-fork** cases. When a parallel branch backtracks and hits its OWN aggregate-frame CP (e.g. an outer findall fanning out a Tier-2-eligible predicate that itself contains a findall in some clause), naïvely calling `finalizeAggregate` from the branch wipes the *outer* `wsAggAccum`. Haskell's branch-loop instead checks `case wsCPs s of (cp : _) | Just _ <- cpAggFrame cp -> wsAggAccum s` — i.e. the branch returns its **local** accum to the parent without finalising. The parent's `forkParBranches` then merges via the strategy.

   **Implication for Elixir Phase 3+:** the current single-collector design (sequential `aggregate_collect/2` and parallel `merge_into_aggregate/2` both writing to the same `:agg_accum`) is correct for the sequential and shallow-parallel cases, but a Tier-2 fan-out whose branches contain their own findall needs a branch-local accumulator pattern. This is Phase 4 territory; Phase 3 stays sequential and validates option (A) on shallow cases only.

3. **Cut barrier interaction.** ⏳ Open for Phase 3. `cut_point` (PR #1535) snapshots `state.choice_points` so `!` prunes back to that snapshot. An aggregate CP under a cut should not be pruned by an inner-goal cut — the aggregate semantics depend on the frame staying alive until enumeration completes. Audit whether `cut_to/1` walks past aggregate frames or treats them as cut barriers. Phase 3 test fixture: `findall(X, (p(X), q(X), !), L)` where the cut commits inside the aggregate's inner goal.

4. ✅ **`agg_type` alphabet mismatch — resolved.** Translation lands at the emission site via `agg_type_atom/2` (#1643): `:collect → :findall`; all other atoms (`sum/count/max/min/bag/set/aggregate_all`) pass through unchanged. `in_forkable_aggregate_frame?/1`'s `@doc` documents the alphabet contract: only `:findall` and `:aggregate_all` ever reach the substrate.

5. **Heap snapshot cost.** Open as a measurement question for Phase 3 / 4. Aggregate frames now capture the same snapshot fields as ordinary CPs — additive, no new copying. If profiling shows finalise-time heap rewind dominates, it's a separate optimisation.

6. ✅ **Empty result — resolved with refined semantics** (see §0 deviations). `findall(X, fail, L)` binds `L = []` (list aggregators have a natural identity). `:sum` / `:count` use 0. `:max` / `:min` throw `{:fail, state}` because no identity exists and `nil` would propagate a non-WAM value. Phase 3 test fixture asserts the list-empty case end-to-end.

7. **Nested findalls.** ⏳ Open for Phase 3. `findall(X, findall(Y, p(X,Y), Ys), Pairs)` pushes two aggregate frames. `aggregate_collect/2` walks `state.choice_points` to find the **nearest** frame (consistent with `merge_into_aggregate/2`'s semantics). Inner finalise should pop the inner agg frame before outer enumeration continues. **Haskell precedent:** the "only the OUTERMOST `ParTryMeElse` actually forks; nested ones use sequential equivalents" pattern (`wam_haskell_target.pl:1486-1492`) matches our `parallel_depth > 0` gate in the Tier-2 super-wrapper — the nested-fork explosion problem has the same shape and the same fix in both implementations. For sequential nested findalls (no Tier-2), the nearest-frame walk is the standard Prolog semantics; Phase 3 confirms via a fixture.

8. **Aggregate frames inside parallel branches.** Cross-references risk #2's branch-local-accum guidance. Each Task gets a snapshotted state, so branch-local aggregate frames stay branch-local — but `merge_into_aggregate/2` writes to the **nearest** frame, which is the branch's local one. The branch finalises locally and the resulting Template value is what propagates up. **Confirmed correct by construction** in the Haskell implementation; Elixir Phase 3 carries a test fixture for this even though it's a Phase 4 (nested parallel) concern.

## 7. Test matrix

**Phases 1–2 covered (emit-and-grep):** parser entries, lowering shape per aggregator type, end-to-end pipeline test that compiles a `findall/3`-using predicate through the WAM compiler and asserts the lowered Elixir contains `push_aggregate_frame(state, :findall, ...)`, `aggregate_collect(state, ...)`, and `throw({:fail, state})`. 79/79 passing post-Phase-2.

**Phase 3 plan — runtime integration tests** (this is the next PR):

Sequential findall:

- `findall(X, member(X, [a,b,c]), L)` → `L = [a, b, c]`.
- `findall(X, fail, L)` → `L = []` (empty result).
- `findall(X, p(X), L)` over a 3-clause fact predicate `p/1` → `L = [v1, v2, v3]` (order matches clause order under sequential enumeration).

aggregate_all:

- `aggregate_all(count, member(_, [a,b,c]), N)` → `N = 3`.
- `aggregate_all(sum(X), member(X, [1,2,3]), S)` → `S = 6`.
- `aggregate_all(max(X), member(X, [3,1,2]), M)` → `M = 3`.
- `aggregate_all(max(X), fail, M)` → fail (no identity for max over empty bag — see §0 deviations).

Tier-2 interaction (sequential only at Phase 3; nested parallel deferred to Phase 4):

- 3-clause declared-pure predicate `p/1` invoked via `findall(X, p(X), L)`. Assert: `@tier2_eligible true` was emitted (already covered), super-wrapper's `cond do` parallel arm fires, `merge_into_aggregate` populates `:agg_accum`, finalise binds `L = [...]` with all three solutions present (set equality, since parallel order is non-deterministic).
- Same 3-clause predicate invoked OUTSIDE a findall (e.g. as a directly-called body goal). Super-wrapper's `cond do` falls through to `*_impl` sequential — proves the gate works when no aggregate frame is present.

Edge cases (probes for §6 risks #1, #3, #7):

- Nested findall (§6 risk #7).
- findall with cut in the inner goal (§6 risk #3).
- findall whose Template references a Y-register variable bound during enumeration (§6 risk #1, deep-copy probe).
- Empty-result and single-result paths.

Cross-target validation (recommended Phase 3 addition):

- For each scenario above, compile the same Prolog source through the **Haskell** target's pipeline and compare result lists. Haskell has end-to-end findall + Tier-2 working (see §11) and serves as a known-good correctness reference. Cross-validation turns "tests we wrote that pass" into "behaviour matches a proven implementation" — strongest signal for the §6 risks the proposal flagged.

Compilability:

- `mix compile` on a generated module containing both the new instruction lowerings and the existing wiring. No warnings on unused functions.

## 8. Rollback plan

The new code paths are additive — `begin_aggregate` and `end_aggregate` instructions don't appear in any existing emitted module (the WAM compiler only produces them when a clause body contains `findall`/`aggregate_all`, which existing test predicates don't). If a bug surfaces post-merge, removing the two new `wam_elixir_lower_instr` clauses and the `backtrack/1` aggregate-frame extension restores byte-for-byte pre-PR behaviour.

The Tier-2 substrate from PR #1586 stays inert (as it has been) — `in_forkable_aggregate_frame?/1` returns false for empty CP stacks.

## 9. Questions for design review (resolved)

1. ✅ **Fail-driven enumeration confirmed.** Implemented per the LLVM precedent. Both Phase 1 and Phase 2 reviews confirmed the fit with existing CP machinery. Stream/iterator semantics not pursued.

2. ✅ **Two-collector design kept separate.** `merge_into_aggregate/2` (parallel batch, from #1586) and `aggregate_collect/2` (sequential per-solution, from #1627) both write to the same `:agg_accum`. `finalise_aggregate/4` consumes the result uniformly. Haskell precedent (§11) shows this design needs refinement under nested parallel — flagged as a Phase 4 concern.

3. ✅ **`finalise_aggregate` lives in `WamRuntime`** alongside `merge_into_aggregate`, as proposed. All aggregate-frame logic co-located in `compile_aggregate_helpers_to_elixir/1`.

4. ⏳ **Tier-2 finalise trigger — sequential validated, parallel deferred.** Option (A) shipped for sequential and shallow-parallel; nested parallel needs the branch-local-accum pattern from Haskell (§6 risk #2 update). Phase 3 tests the sequential case; Phase 4 will revisit.

5. ✅ **`:collect → :findall` translation at emission time.** Implemented in `agg_type_atom/2` (#1643). The substrate's `:findall` / `:aggregate_all` alphabet stays authoritative.

## 10. What this proposal does NOT cover

- **`bagof/3`, `setof/3`** — witness-variable semantics need separate design (the substrate `in_forkable_aggregate_frame?/1` already explicitly excludes these as non-forkable).
- **User-defined aggregators.**
- **`findall/4`** (the variant taking a tail list).
- **Performance tuning of the parallel-vs-sequential gate.** `forkMinBranches = 3` stays hardcoded.
- **Compile-time inlining of small enumerable sources** (e.g. `findall(X, member(X, [a,b,c]), L)` → directly emit `L = ["a","b","c"]`). Worthwhile peephole opt, separate proposal.
- **Nested-parallel branch-local accumulators** (§6 risk #2 update). Phase 4 territory — needs the per-branch local-accum + parent-merge pattern from Haskell when a Tier-2 fan-out branch contains its own findall.
- **`MergeStrategy` separation from `agg_type`.** Haskell's `inferMergeStrategy` (§11) splits "what is the aggregate" from "how do parallel branches combine results" — necessary once Phase 4 introduces parallel-under-aggregate.
- **Dynamic `Module:Goal` meta-call on the Elixir runtime.** The static-module-qualifier unwrap in `compile_goal_call/4` / `compile_goal_execute/4` handles the common case (`findall(X, user:p(X), L)`) at compile time — `M:Goal` with atom or string `M` recursively compiles to a regular `call p/Arity, N`, identical to the unqualified case. The dynamic case (`Module = m, Module:Goal` where `M` is a Prolog variable at compile time) still routes through the `:/2` builtin, which `WamRuntime.execute_builtin/3` does not implement (catch-all returns `:fail`). When the dynamic case actually surfaces in a real predicate, the correct runtime fix is the CP-frame extension to `backtrack/1` — install a meta-call frame at `:/2` entry that backtracking knows to handle by re-entering the call site for each enumerated solution. This is workaround #2 from the design analysis Perplexity reviewed prior to this PR; it's an architectural change (meta-call as a first-class CP type, like aggregate frames are today), not a bug fix. Other targets (Rust / Go / LLVM / Haskell) have native `:/2` runtime support and handle the dynamic case directly. No existing test predicate uses dynamic meta-call.

## 11. Cross-target precedent (Haskell)

The Haskell target (`src/unifyweaver/targets/wam_haskell_target.pl`) has an end-to-end working implementation of the same fail-driven-enumeration findall + Tier-2 (parallel-forkable) machinery this proposal designs. It has been a useful design oracle for our Phase 3 planning and is the recommended cross-validation reference per §7.

**Direct correspondences:**

| Concept | Elixir (this proposal) | Haskell (`wam_haskell_target.pl`) |
|---|---|---|
| Aggregate-frame CP | `cp` map with `:agg_type`, `:agg_value_reg`, `:agg_result_reg`, `:agg_accum` | `ChoicePoint` record with `cpAggFrame :: Maybe AggFrame` |
| Per-solution collect | `aggregate_collect/2` (#1627) | line 2253 — `EndAggregate` opcode body |
| Backtrack dispatch | `Map.get(cp, :agg_type)` arm in `backtrack/1` (#1627) | `cpAggFrame cp` pattern match in backtrack |
| Empty-accumulator | List → `[]`, sum/count → 0, max/min → throw fail (#1627) | `finalizeAggregate` |
| Begin/end opcodes | `wam_elixir_lower_instr(begin_aggregate/end_aggregate)` (#1643) | `BeginAggregate` / `EndAggregate` (lines 2231 / 2252) |
| Fail-driven enumeration | `throw({:fail, state})` after `aggregate_collect/2` | `backtrackInner` + `finalizeAggregate` (line 1280) |
| `:collect → :findall` translation | `agg_type_atom/2` (#1643) | `inferMergeStrategy "findall" = MergeFindall` (line 3242) — different shape but similar role |
| Nested-fork suppression | `parallel_depth > 0` gate in super-wrapper (#1624) | `seqInstr` rewrite of `Par* → sequential` inside branches (line 1493) |
| Branch-local accumulator | ⏳ Phase 4 follow-up (§6 risk #2 update) | line 1502 — branch returns local `wsAggAccum` to parent without finalising |
| `MergeStrategy` (parallel result combining) | ⏳ Phase 4 follow-up (§10 non-goal) | `inferMergeStrategy` separated from `agg_type` |

**What we adopted from Haskell:**

- The fail-driven enumeration model itself (originally cited from LLVM in §3.3 — Haskell is the direct functional analogue).
- The single-dispatch backtrack (no separate `backtrackInner`) — Haskell needed it because their inner-goal CPs and aggregate frames live in the same `wsCPs` list with PC-based resumption; our Elixir CPS-with-continuations design naturally routes to the right place.
- The "nested forks suppression" pattern: only the outermost parallel layer forks. Confirms our `parallel_depth > 0` gate design is correct.
- Branch-local accum vs global accum — informs §6 risk #2 update; flagged for Phase 4 in §10.

**What we deliberately don't adopt:**

- PC-based aggregate-return-PC storage. Haskell stores `returnPC = wsPC s + 1` inside the agg frame at end_aggregate time and `updateNearestAggFrame` walks the CP list to write it. Our Elixir version captures `state.cp` at *push* time — equivalent semantics in our CPS world, simpler implementation. (See §0 deviations: `:agg_return_cp` was redundant with `:cp`.)
- `MergeStrategy` for now. Phase 3 is sequential-only, so a dedicated merge-strategy abstraction is premature. Phase 4 will introduce it when nested-parallel needs the divergent path.
