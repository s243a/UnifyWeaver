# Haskell WAM Target: Roadmap

## Current state (2026-04-13)

The Haskell WAM target beats SWI-Prolog by 1.16-1.56x on the effective-
distance benchmark (300 scale) with FFI. The pure interpreter is ~8x
slower than FFI but 47% faster than the pre-optimization baseline.

| Metric | Value |
|---|---|
| Haskell + FFI (300 scale) | 285ms |
| SWI-Prolog optimized | 311ms |
| Rust + FFI | 126ms |
| Pure interpreter (no FFI) | 2518ms |
| Lowered predicates | 5 of 7 |
| Detected kernels | category_ancestor (auto-generated FFI) |

## Phase 1: Seed-Level Parallelism

**Goal:** Multi-core speedup with minimal code change.

**Tasks:**
1. Add `-threaded` to cabal GHC options
2. Add `NFData` instances for `Value`, `WamState`
3. Replace `mapM querySeed seedCats` with `parMap rdeepseq querySeed`
   in Main.hs template
4. Benchmark with `+RTS -N4` on the 300 and 5k scale datasets
5. Verify output correctness (identical to sequential)

**Expected outcome:** ~3-4x speedup for seed-dominated workloads.
At 300 scale (386 seeds, 4 cores): ~70-90ms.

**Risk:** Low. Immutable WamContext is shared read-only. Each seed
gets independent WamState. No architectural change needed.

## Phase 2: Atom Interning

**Goal:** Eliminate string comparison and hashing from the hot loop.

**Tasks:**
1. Add `AtomTable` type (bidirectional String ↔ Int mapping)
2. Change `Value` to `Atom !Int` (intern ID, not string)
3. Update Main.hs fact loading to intern atom strings at load time
4. Update `executeForeign` to intern/de-intern at the FFI boundary
5. Update `SwitchOnConstantPc` to `Map.Map Int Int` (intern ID → PC)
6. Benchmark pure interpreter (expected ~10% gain)

**Expected outcome:** hashWithSalt drops to ~0%, `==` for atoms becomes
O(1) integer comparison.

**Risk:** Medium. Touches the `Value` type which appears everywhere.
The AtomTable must be threaded through or stored in WamContext.

## Phase 3: Expanded Lowering

**Goal:** Bypass the interpreter for more predicates.

**Tasks:**
1. Lower `begin_aggregate`/`end_aggregate` via a `run`-based wrapper
   (delegate the aggregate body to the interpreter's run loop but
   keep the setup/teardown in the lowered function)
2. Support nested if-then-else in lowered functions
3. Lower predicates with `get_structure`/`unify_variable` (requires
   read/write mode in the lowered emitter)
4. Auto-detect lowerability: report which predicates can't be lowered
   and why (helps prioritize whitelist expansion)

**Expected outcome:** lowered=6-7 of 7 predicates in the optimized
benchmark. Each lowered predicate avoids the step dispatch + record
update overhead.

## Phase 4: Intra-Query Parallelism

**Goal:** Parallel exploration of choice point branches.

**Prerequisites:**
- Phase 1 (seed parallelism) working and benchmarked
- C# demand analysis can estimate branch work
- Purity annotations or proofs for target predicates

**Tasks:**
1. Add `ParTryMeElse` / `ParTrustMe` instructions to WAM compiler
2. Add purity annotation syntax (`:- parallel(pred/arity)`)
3. Implement fork/merge in the Haskell runtime using `par`/`pseq`
4. Add merge strategy selection based on enclosing aggregate context
5. Add work-estimation threshold to avoid forking cheap branches
6. Benchmark on the 5k and 10k scale datasets

**Expected outcome:** Additional 2-4x speedup for deep recursive
predicates with multiple independent branches.

**Risk:** High. Requires careful merge semantics, work estimation
heuristics, and interaction with cut/aggregate.

## Phase 5: Demand-Driven Mutable Sections

**Goal:** ST monad for non-forking hot sections.

**Prerequisites:**
- Phase 4 (intra-query parallelism) working
- C# demand analysis can identify non-forking sections
- State monad adopted as code style (for easy ST transition)

**Tasks:**
1. Adopt State monad in the step function (pure refactor, no perf gain)
2. Identify non-forking sections via demand analysis annotations
3. Replace State with ST for annotated sections
4. Add freeze/thaw at fork point boundaries
5. Profile and benchmark to verify the allocation reduction

**Expected outcome:** ~30-40% interpreter speedup in non-forking
sections, with no loss of parallelism at fork points.

**Risk:** High. Large refactor (~184 field references). Must not
regress parallelism or correctness.

## Non-goals

- **Beating Rust on single-threaded speed.** Rust's mutable vectors
  will always be faster for sequential execution. Haskell's advantage
  is parallelism and correctness, not raw throughput.
- **Full Prolog semantics.** The WAM target compiles a subset of Prolog
  (the predicates UnifyWeaver generates). It doesn't need assert/retract,
  meta-predicates, or module system support.
- **Production deployment.** The Haskell target is a benchmark and
  research platform. Production workloads use the C# query engine.

## References

- `docs/vision/HASKELL_TARGET_PHILOSOPHY.md` — strategic rationale
- `docs/vision/HASKELL_TARGET_PARALLELIZATION_SPEC.md` — parallelism
  semantics and merge strategies
- `docs/design/WAM_HASKELL_PERF_IMPLEMENTATION_PLAN.md` — profiling
  data and optimization history
- `docs/design/WAM_HASKELL_LOWERED_IMPLEMENTATION_PLAN.md` — lowering
  architecture and phased plan
