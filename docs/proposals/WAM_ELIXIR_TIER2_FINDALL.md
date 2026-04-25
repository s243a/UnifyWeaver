# Elixir WAM findall/aggregate_all proposal

**Status:** Draft for design review (pre-implementation).
**Target audience:** Perplexity review + future implementer.
**Depends on:** PR #1586 (Tier-2 substrate), PR #1608 (`par_wrap_segment/4`), PR #1624 (Tier-2 wiring activation).

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

### 3.2 What's missing

Three things, all in the Elixir target:

1. **`begin_aggregate` instruction lowering.** Push an aggregate-typed CP onto `state.choice_points` carrying `:agg_type`, `:agg_value_reg`, `:agg_result_reg`, `:agg_accum: []`, `:agg_return_cp` (continuation), and a snapshot of `state.heap_len` / `state.trail_len` / `state.regs` so backtracking-to-finalize can restore.
2. **`end_aggregate` instruction lowering.** Read the value register, deref it, copy into the nearest aggregate CP's `:agg_accum`, then `throw({:fail, state})` to drive backtracking — fail-driven enumeration. The next inner-goal alternative gets re-tried; when none remain, backtrack pops the aggregate CP and finalizes.
3. **Backtrack-to-finalize semantics in `WamRuntime.backtrack/1`.** When the topmost CP popped is an aggregate frame, do not re-execute its `pc` (aggregate frames have no failure target). Instead: aggregate the `:agg_accum` per the `:agg_type` (`:collect`/`:findall` → list, `:sum` → sum, `:count` → length, etc.), bind the result to `:agg_result_reg`, restore the saved environment minus the popped frame, and resume at `:agg_return_cp`.

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

1. **Trail-unwind safety of `:agg_accum`.** When `end_aggregate` collects a value and triggers backtrack, the trail unwind restores variable bindings to the pre-CP state. If a Template value contains heap references to bound variables that the trail unwinds, the captured value mutates retroactively. The deep-copy in `aggregate_collect/2` must walk the value through `deref_var` and copy concrete heap cells — turning `{:struct, addr}` references into self-contained tuples. Worth a probe with a multi-clause inner goal whose Template references unify-time-bound Y-registers.

2. **Tier-2 finalise trigger.** When the super-wrapper takes the parallel arm, control returns from `merge_into_aggregate/2` to the caller of the super-wrapper — which is the predicate's `clause_main`, which returns to its caller's `state.cp`. But the aggregate frame is still on `state.choice_points`. Two options:
    - (A) Super-wrapper throws `{:fail, state}` after merge — cleanest, falls through to backtrack which finalises.
    - (B) Super-wrapper explicitly calls `finalise_aggregate` itself.
   Option (A) shares the finalise path with the sequential case and is the proposal's recommended approach. Verify with a Tier-2-eligible inner goal.

3. **Cut barrier interaction.** `cut_point` (PR #1535) snapshots `state.choice_points` so `!` prunes back to that snapshot. An aggregate CP under a cut should not be pruned by an inner-goal cut — the aggregate semantics depend on the frame staying alive until enumeration completes. Audit whether `cut_to/1` walks past aggregate frames or treats them as cut barriers.

4. **`agg_type` alphabet mismatch.** `compile_aggregate_all/5` uses atoms `sum/count/max/min/bag/set/collect`; the runtime substrate's `in_forkable_aggregate_frame?/1` only forks on `:findall` and `:aggregate_all`. The `:collect` atom (used by findall via the `collect-Template` wrapper) needs to be recognised as forkable, or the wrapper needs to translate `collect → findall` at the begin_aggregate emission site. Proposed: emit `:findall` for findall/3, `:aggregate_all` for aggregate_all/3 with non-list aggregators (sum/count/etc.), keep `:collect` as a synonym only inside the WAM compiler's intermediate representation.

5. **Heap snapshot cost.** Aggregate frames need to capture enough state to restore after finalise. The Elixir CP map already captures regs/heap/heap_len/trail/trail_len/stack — adding aggregate fields is additive, no new copying. If a future profiling pass shows finalise-time heap rewind dominates, that's a separate optimisation.

6. **Empty result.** `findall(X, fail, L)` should bind L = []. Verify `begin_aggregate` followed immediately by inner-goal failure backtracks straight to the aggregate frame (no inner enumeration), and `finalise_aggregate` correctly emits the empty-list result.

7. **Nested findalls.** `findall(X, findall(Y, p(X,Y), Ys), Pairs)` pushes two aggregate frames. `aggregate_collect` walks `state.choice_points` to find the **nearest** frame (consistent with `merge_into_aggregate/2`'s existing semantics). Verify the nested case binds correctly — inner finalise should pop the inner agg frame before outer enumeration continues.

8. **Aggregate frames inside parallel branches.** A Tier-2 fan-out branch could itself contain a findall. Each branch runs in its own Task with a snapshotted state — its inner aggregate frame stays branch-local. The branch's `merge_into_aggregate` writes to its own frame, which is then consumed by the branch's local finalise; the resulting Template value is what propagates up to the outer fan-out's collection. This is correct by construction but worth a test.

## 7. Test matrix

Sequential findall:

- `findall(X, member(X, [a,b,c]), L)` → `L = [a, b, c]`.
- `findall(X, fail, L)` → `L = []` (empty result).
- `findall(X, p(X), L)` over a 3-clause fact predicate `p/1` → `L = [v1, v2, v3]` (order matches clause order under sequential enumeration).

aggregate_all:

- `aggregate_all(count, member(_, [a,b,c]), N)` → `N = 3`.
- `aggregate_all(sum(X), member(X, [1,2,3]), S)` → `S = 6`.
- `aggregate_all(max(X), member(X, [3,1,2]), M)` → `M = 3`.

Tier-2 interaction:

- 3-clause declared-pure predicate `p/1` invoked via `findall(X, p(X), L)`. Assert: `@tier2_eligible true` was emitted (already covered), super-wrapper's `cond do` parallel arm fires, `merge_into_aggregate` populates `:agg_accum`, finalise binds `L = [...]` with all three solutions present (set equality, since parallel order is non-deterministic).
- Same 3-clause predicate invoked OUTSIDE a findall (e.g. as a directly-called body goal). Super-wrapper's `cond do` falls through to `*_impl` sequential — proves the gate works when no aggregate frame is present.

Edge cases:

- Nested findall.
- findall with cut in the inner goal.
- findall whose Template references a Y-register variable bound during enumeration.
- Empty-result and single-result paths.

Compilability:

- `mix compile` on a generated module containing both the new instruction lowerings and the existing wiring. No warnings on unused functions.

## 8. Rollback plan

The new code paths are additive — `begin_aggregate` and `end_aggregate` instructions don't appear in any existing emitted module (the WAM compiler only produces them when a clause body contains `findall`/`aggregate_all`, which existing test predicates don't). If a bug surfaces post-merge, removing the two new `wam_elixir_lower_instr` clauses and the `backtrack/1` aggregate-frame extension restores byte-for-byte pre-PR behaviour.

The Tier-2 substrate from PR #1586 stays inert (as it has been) — `in_forkable_aggregate_frame?/1` returns false for empty CP stacks.

## 9. Questions for design review

1. **Is fail-driven enumeration the right control-flow model**, or should the Elixir target diverge from the LLVM precedent and use explicit stream/iterator semantics? The proposal favours fail-driven for consistency with existing CP machinery.

2. **Should `merge_into_aggregate/2` and per-solution `end_aggregate` collection be unified into one helper**, or is the two-collector design (one batch, one incremental) the right shape? The proposal keeps them separate because they serve different tier paths and the substrate already lives in shipped code.

3. **`finalise_aggregate` placement.** Does it live in `WamRuntime` (alongside `merge_into_aggregate`) or in the lowered emitter as inline Elixir? The proposal puts it in `WamRuntime` so all aggregate-frame logic is co-located.

4. **Tier-2 finalise trigger** (§6 risk #2): super-wrapper throws `{:fail, state}` after `merge_into_aggregate`, vs. explicit `finalise_aggregate` call. The proposal recommends throw-and-let-backtrack-finalise.

5. **`:collect` vs `:findall` atom for findall/3.** Translate at emission time, or keep `:collect` as a synonym throughout? The proposal translates at emission time so the runtime substrate's existing `:findall`/`:aggregate_all` discrimination remains the source of truth.

## 10. What this proposal does NOT cover

- **`bagof/3`, `setof/3`** — witness-variable semantics need separate design (the substrate `in_forkable_aggregate_frame?/1` already explicitly excludes these as non-forkable).
- **User-defined aggregators.**
- **`findall/4`** (the variant taking a tail list).
- **Performance tuning of the parallel-vs-sequential gate.** `forkMinBranches = 3` stays hardcoded.
- **Compile-time inlining of small enumerable sources** (e.g. `findall(X, member(X, [a,b,c]), L)` → directly emit `L = ["a","b","c"]`). Worthwhile peephole opt, separate proposal.
