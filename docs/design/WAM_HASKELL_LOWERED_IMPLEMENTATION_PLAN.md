# WAM-Lowered Haskell: Implementation Plan

This document is the step-by-step plan for adding the **WAM-lowered
Haskell** code-generation path (path 3) to UnifyWeaver.

Companion docs:
- [`WAM_HASKELL_LOWERED_BACKGROUND.md`](WAM_HASKELL_LOWERED_BACKGROUND.md)
  — descriptive tour of the existing Haskell code-generation paths, with
  source line references. Read this first if you are new to the area.
- [`WAM_HASKELL_LOWERED_PHILOSOPHY.md`](WAM_HASKELL_LOWERED_PHILOSOPHY.md)
  — *why* we are adding this third path.
- [`WAM_HASKELL_LOWERED_SPECIFICATION.md`](WAM_HASKELL_LOWERED_SPECIFICATION.md)
  — *what* the emitter must produce.

This document is strictly about ordering and dependencies.

## Branch strategy

- New branch: `feat/wam-haskell-lowered`, forked from whatever main-
  line state the FFI-optimization branch merged into.
- Do **not** touch `wam_rust_target.pl`, `wam_c_target.pl`, or the
  other WAM targets. This is Haskell-only.
- Do **not** touch `haskell_target.pl` (path 1). It stays as-is.
- The interpreter path (path 2) stays the default throughout this
  branch. The branch ships with `emit_mode(interpreter)` still the
  default.

## Phase 0: prerequisites (no code changes)

Before writing a single line of the new emitter, verify these hold:

- [ ] Current `feat/wam-haskell-ffi-optimization` work is merged or
      its performance/correctness results are frozen, so we have a
      stable baseline to measure lowered code against.
- [ ] The `effective_distance` 10k benchmark runs end-to-end through
      `wam_haskell_target.pl` and produces output byte-identical to
      the Prolog reference. This is the regression oracle.
- [ ] The WAM IR instruction set as consumed by
      `wam_haskell_target.pl` is documented or inspectable (read:
      `wam_instr_to_haskell/2` in `wam_haskell_target.pl` already
      lists every instruction the Haskell target supports — use that
      list as the initial lowerability whitelist).

None of the phases below start until Phase 0 is green.

## Phase 1: selector plumbing (no lowering yet)

Goal: add the `emit_mode/1` option and the `wam_haskell_emit_mode/1`
dynamic fact, wire them through `write_wam_haskell_project/3`, and
verify that `emit_mode(interpreter)` (the default) produces exactly the
current output.

- [ ] Add the `emit_mode/1` option parsing to
      `write_wam_haskell_project/3`. Accepted values:
      `interpreter`, `functions`, `mixed(_)`. Unknown values raise an
      existence error.
- [ ] Add `user:wam_haskell_emit_mode/1` lookup as the second level
      of the selector hierarchy. Missing is fine; the generator falls
      through to the default.
- [ ] Add a stub `wam_haskell_lowerable(+PredIndicator, +WamCode, -Reason)`
      that **always** fails with reason
      `"lowering emitter not yet implemented"`. This makes
      `emit_mode(functions)` and `mixed(_)` route everything to the
      interpreter for now — no behavior change, but the plumbing is
      in place.
- [ ] Regression check: run the `effective_distance` 10k benchmark
      with no `emit_mode` option. Output must be byte-identical to
      the pre-Phase-1 state.
- [ ] Regression check: run with `emit_mode(interpreter)`
      explicitly. Same.
- [ ] Regression check: run with `emit_mode(functions)`. Because
      the stub lowerability check always fails, everything still
      routes to the interpreter, and output is still identical.
- [ ] Land as one commit.

## Phase 2: `Lowered.hs` skeleton (empty lowered partition)

Goal: emit a `Lowered.hs` file even when no predicates are lowered,
wire it into `WamContext` and `WamRuntime.step`, and verify the
interpreter still works.

- [ ] Add `wcLoweredPredicates :: Map.Map String (WamContext -> WamState -> Maybe WamState)`
      to `WamContext` (cold field). Default initialization is
      `Map.empty`.
- [ ] Modify `step` in the `Call` dispatch chain to check
      `wcLoweredPredicates` first, before `executeForeign`. On miss,
      fall through to the existing chain unchanged.
- [ ] Emit `Lowered.hs` in `write_wam_haskell_project/3` even when
      the lowered partition is empty. Its body is the module header,
      the imports, and `loweredPredicates = Map.empty`.
- [ ] Modify `Main.hs` generation to populate `wcLoweredPredicates`
      from `Lowered.loweredPredicates` when building `ctx`.
- [ ] Regression check: `effective_distance` 10k benchmark byte-
      identical to Phase 1.
- [ ] Land.

## Phase 3: first lowered predicate (smoke test)

Goal: hand-lower one trivial predicate end-to-end, prove the interop
contract, and use it as the correctness oracle for the rest of the
work.

Target predicate: `max_depth/1` (the dynamic fact with one clause,
`max_depth(10).`). Its WAM is essentially one `GetConstant` and a
`Proceed`. There is no backtracking, no cut, no aggregation. It is
the smallest thing that actually exercises the function-call protocol.

- [ ] Create `src/unifyweaver/targets/wam_haskell_lowered_emitter.pl`.
      Export `lower_predicate_to_haskell/4` and
      `wam_haskell_lowerable/3`.
- [ ] Implement the whitelist-based lowerability check for the
      instructions in `max_depth/1`: `GetConstant`, `Proceed`.
- [ ] Implement the emitter for those two instructions only. Output
      a function `lowered_max_depth_1 :: WamContext -> WamState -> Maybe WamState`
      that inlines the same logic as the interpreter's `step` cases.
- [ ] Wire `lower_predicate_to_haskell/4` into
      `write_wam_haskell_project/3` so that
      `emit_mode(mixed([max_depth/1]))` actually lowers it and the
      rest stay interpreted.
- [ ] Add `tests/test_wam_haskell_lowered_smoke.pl`: generate a tiny
      project with `mixed([max_depth/1])`, build it, run a query that
      hits `max_depth/1`, and verify the answer matches the
      interpreter-only run.
- [ ] Land when the smoke test passes.

After Phase 3, the generator has one lowered predicate that works
end-to-end. This is where we stop if we want to bail out; everything
from here is incremental expansion.

## Phase 4: simple register and control instructions

Goal: expand the lowering whitelist to cover predicates made of
straight-line code with no backtracking.

- [ ] Add to the whitelist: `GetVariable`, `GetValue`, `GetStructure`,
      `PutConstant`, `PutVariable`, `PutValue`, `PutStructure`,
      `CallResolved`, `Proceed`, `Allocate`, `Deallocate`.
- [ ] Implement the Haskell emission for each. Use the interpreter's
      `step` implementation as the semantic reference — the lowered
      version produces the same `WamState` delta inline.
- [ ] Pick a test predicate from the existing benchmark suite whose
      body uses these instructions but no choice points (e.g.,
      `dimension_n/1`, or the first clause of a simple helper).
- [ ] Add a test comparing lowered vs interpreted output.
- [ ] Land.

## Phase 5: choice points and backtracking

Goal: support `TryMeElse`, `RetryMeElse`, `TrustMe`, and prove
backtracking crosses the path boundary correctly.

- [ ] Add to the whitelist: `TryMeElse`, `RetryMeElse`, `TrustMe`,
      and the `Call` (string-dispatched) variant for predicates whose
      target is not known at lowering time.
- [ ] Implement the `wsPC`-dispatched entry described in
      `WAM_HASKELL_LOWERED_SPECIFICATION.md` §2.4. Each lowered
      predicate that has multiple clause entry PCs starts with a
      `case wsPC s of ...` that routes to the right clause body.
- [ ] Verify the `cpNextPC`-as-interpreter-PC invariant in practice:
      a lowered predicate whose `TryMeElse` alternate is also lowered
      still records the interpreter PC in the CP, and a backtrack
      into it still works because the `step` dispatch finds it in
      `wcLoweredPredicates`.
- [ ] Add a cross-path backtracking test: an interpreted predicate
      that calls a lowered predicate whose first clause fails, and
      the backtrack restores the correct state and tries clause 2.
- [ ] Add the reverse: a lowered predicate that calls an interpreted
      predicate which creates its own CPs, and verify backtracking
      from inside the interpreted call correctly unwinds out.
- [ ] Land.

## Phase 6: builtins

Goal: support the BuiltinCall variants the interpreter handles.

- [ ] Add to the whitelist: `BuiltinCall "is/2"`, `BuiltinCall "!/0"`,
      `BuiltinCall "\\+/1"`, `BuiltinCall "length/2"`,
      `BuiltinCall "member/2"`, `BuiltinCall "</2"`,
      `BuiltinCall ">/2"`.
- [ ] For each builtin, emit a direct Haskell call into a shared
      helper that the interpreter also uses. Refactor
      `WamRuntime.hs` if needed so the helpers are top-level
      functions, not inlined into the `step` case bodies. This is a
      code-shape refactor; it must not change interpreter behavior.
- [ ] Add a test predicate using each builtin and verify lowered vs
      interpreted parity.
- [ ] Land.

## Phase 7: `SwitchOnConstant` and indexed dispatch

Goal: support the one non-trivial dispatch instruction and verify
parity with the interpreter on predicates that use it.

- [ ] Add `SwitchOnConstant` to the whitelist.
- [ ] Emit either an inline Haskell `case` (for small dispatch tables,
      threshold: ≤ 32 entries) or a `Map.lookup` (for larger tables),
      matching the interpreter's runtime semantics.
- [ ] Test on a fact-indexed predicate: `category_parent/2` probably
      works. Verify parity.
- [ ] Land.

## Phase 8: benchmark with `emit_mode(functions)`

Goal: measure path 3 against path 2 honestly, document the number.

- [ ] Run the `effective_distance` 10k benchmark in
      `emit_mode(interpreter)` (baseline) and
      `emit_mode(functions)` (lowered), 5 runs each, report median
      query_ms and total_ms.
- [ ] Verify output is byte-identical between the two.
- [ ] If `emit_mode(functions)` is slower, document that honestly.
      Do *not* ship performance claims ahead of measurements.
- [ ] Save the numbers in `docs/benchmarks/WAM_HASKELL_LOWERED_RESULTS.md`
      (new file). Include the commit hash and GHC version.

## Phase 9: `mixed(HotPreds)` ergonomics

Goal: make mixed mode first-class, not just a debugging artifact.

- [ ] Add `emit_mode(mixed([category_ancestor/4, power_sum_bound/4]))`
      as the default shape tested in the `wam_haskell` benchmark
      target. (Still not the generator's overall default — that's
      Phase 10.)
- [ ] Benchmark the mixed case vs both all-interpreted and all-
      lowered. Document.
- [ ] If the mixed mode beats both, update
      `examples/benchmark/benchmark_effective_distance.py` to use it
      for the `wam_haskell` target by default.

## Phase 10 (deferred): flip the default

Only do this if all of the following hold:

- Path 3 passes every correctness test path 2 passes on every
  benchmark currently checked in.
- Path 3 is faster than path 2 on the `effective_distance` 10k
  benchmark by at least 1.5× median total_ms.
- At least one externally-reported workload has used path 3 in
  `mixed(HotPreds)` mode without filing a correctness bug.
- The lowerability check has been expanded to cover every
  instruction used by every checked-in benchmark predicate, OR the
  generator's automatic fallback has been exercised and verified to
  produce correct output.

When all of the above hold:

- [ ] Change the default in `write_wam_haskell_project/3` from
      `interpreter` to `functions`.
- [ ] Update `WAM_HASKELL_LOWERED_PHILOSOPHY.md` to reflect the new
      default.
- [ ] Leave `interpreter` mode fully supported — it is still the
      fallback and the reference oracle.

## Deferred / out of scope for this branch

- **Aggregation lowering** (`AggPush`/`AggPop`/`AggEmit`). Stays
  interpreted. A future follow-up handles aggregation in lowered
  code after the lowering aggregation shape is designed.
- **`PutStructureDyn`**. Tracked separately (task 16 in the current
  todo list).
- **Cross-target lowering parity** (lowered Rust, lowered C, etc.).
  Different branch.
- **Profile-guided `HotPreds` selection.** The user picks the list.
- **Per-call-site inlining.** Whole predicate is the unit.
- **Automated benchmark runs on CI.** Nice-to-have.

## Rollback plan

Each phase lands as its own commit(s). If a phase breaks something
and can't be fixed within the phase, revert the phase's commits
(git-revert, not force-push) and the generator is back to its
previous shape. Because the interpreter is the default and the
lowered emitter is additive, reverting any phase of path 3 cannot
break path-2 users.

## Coordination with other branches

- **`feat/wam-haskell-ffi-optimization`**: merge first, freeze its
  benchmark numbers, use them as the path-2 baseline for all path-3
  comparisons.
- **Task 16 (`PutStructureDyn`)**: independent. If it lands first,
  the lowering whitelist can include it; if it lands second, the
  lowering emitter gains it at that point.
- **Task 13 (Rust target parity)**: separate branch, separate
  design, no interaction.

## Review checklist (before landing each phase)

- [ ] Does the phase change `emit_mode(interpreter)` output in any
      way? If yes, that is a bug and must be fixed before landing.
- [ ] Does the phase's new lowered code produce byte-identical
      output to the interpreter on every test case the phase adds?
- [ ] Do previously-passing tests still pass?
- [ ] Is `wam_haskell_lowerable/3` correctly rejecting predicates
      that use instructions not yet supported? (A missed rejection
      manifests as a generator crash or a GHC compile error, not
      a runtime failure — both are worse than falling back to the
      interpreter.)
- [ ] Is the commit message specific enough that `git log
      --oneline` tells a future reader which phase this is?
