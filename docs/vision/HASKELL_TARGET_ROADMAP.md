# Haskell WAM Target: Roadmap

## Current state (2026-04-20, post-atom-interning)

The Haskell WAM target **outperforms the Rust WAM target** on the effective-
distance benchmark (300 scale) thanks to seed-level parallelism that Rust's
mutable-vector design can't trivially match. Median numbers over 10 runs:

| Metric | Value | vs Rust |
|---|---|---|
| Haskell + FFI + parallel (300 scale, 4 cores) | **75ms total, 32ms query** | 1.75x faster total |
| Haskell + FFI single-core (300 scale) | 193ms total, 107ms query | — |
| Rust + FFI (300 scale) | 126ms | baseline |
| SWI-Prolog optimized | 311ms | — |
| Pure interpreter (no FFI, pre-interning) | 2518ms | — |
| Haskell at 5k scale (4 cores) | 213ms total, 86ms query | 3.5x parallel speedup |
| Pure interpreter (10k WAM-only, interned) | **15,546ms** | 19% faster than baseline |
| Lowered predicates | 5 of 7 | — |
| Detected kernels | category_ancestor (auto-generated FFI) | — |

**Phases 1, 2, and 2c are now complete** (seed parallelism, FFI-boundary
interning, system-wide atom interning).
Additional optimizations beyond the original roadmap: `skip_fact_wam`
(eliminates redundant WAM-compilation of FFI-owned facts, -70% total at
300 scale), O(n) intern-table construction. See
`docs/design/WAM_PERF_OPTIMIZATION_LOG.md` for the complete history.

## Phase 1: Seed-Level Parallelism — DONE

**Goal:** Multi-core speedup with minimal code change.

**Delivered (PR #1377, commit 74d9e9b4):**
1. `-threaded -rtsopts` added to cabal GHC options
2. `deepseq` + `parallel` build-deps added
3. Query body generator emits pure `let { ... } in (cat, result)` so
   it can be used in a `parMap rdeepseq` lambda
4. Main.hs template replaces `mapM` with `parMap rdeepseq`, forces
   sparks via `seedResultsForced \`deepseq\` ()`
5. No NFData instances needed — `(String, Double)` is already NFData
   through base. WamState is ephemeral inside each seed's pure
   computation and never crosses a spark boundary.

**Actual outcome:** 3.3x speedup at 300 scale (query 107ms → 32ms with
4 cores), 3.5x at 5k scale. 8-core regressed on this WSL2 host due to
scheduling/GC contention; 4 cores is the sweet spot.

## Phase 2: Atom Interning — DONE (scoped)

**Goal:** Eliminate string comparison and hashing from the hot loop.

**Delivered (PRs #1376, #1377):**
- Rather than changing the `Value` type (landmine: mixed atom/integer
  dispatch in SwitchOnConstant, pervasive `Atom ""` defaults), interning
  happens **at the FFI boundary** only. The WAM path still uses
  `Atom String`.
- `WamContext` gains `wcAtomIntern`/`wcAtomDeintern`/`wcFfiFacts` fields.
- Kernel templates (`kernel_category_ancestor`, `kernel_transitive_closure`)
  now take `IntMap [Int]` and `Int` args.
- executeForeign emitter interns atom/vlist_atoms register reads,
  de-interns atom results.
- Main.hs builds the intern table once at startup via a single O(n)
  foldl' (the initial O(n²) `Map.size`-per-insert version was fixed
  in the parallel-seeds PR).

**Actual outcome:** query_ms dropped from ~200ms to ~107ms (median)
on a single core — beat Rust's 126ms on query time without needing
parallelism.

## Phase 2c: System-Wide Atom Interning — DONE

**Goal:** Promote atom interning from the FFI boundary to the entire system.
Change `Atom String` to `Atom !Int` everywhere — WAM interpreter, lowered
emitter, FFI kernels, fact dispatch.

**Delivered (branch `feat/wam-haskell-atom-interning`):**
1. `data Value`: `Atom !Int`, `Str !Int [Value]`
2. `InternTable` type with forward (String→Int) and reverse (Int→String) maps
3. Well-known atom constants: atomTrue=0, atomFail=1, atomNil=2, atomDot=3
4. `SwitchOnConstantPc` changed from `Map.Map String Int` to `IM.IntMap Int`
5. `compileTimeAtomTable` emitted in Predicates.hs, extended at load time
6. FFI boundary simplified — atoms already Int, no intern/deintern step
7. Lowered emitter updated to emit `Atom <id>` via `intern_atom/2`
8. `evalArith` threaded through `InternTable` for reverse lookup

**Actual outcome:** 19% faster on the WAM-only interpreter path at 10k
scale (19,173ms → 15,546ms). FFI path unchanged (already used Int
comparison). Correctness verified: identical output at 1k and 10k.

**Also fixed:** FFI query dispatch bug — effective-distance benchmark was
producing `tuple_count=0` when kernels were detected (fact code skipped
but query entered WAM code directly). Added `collectForeignSolutions`
using `executeForeign` dispatch. This fix was also applied to main.

## Phase 2b (unplanned): Skip-Facts

**Goal:** Don't WAM-compile fact predicates that FFI already handles.

**Delivered (PR #1375, commit fcaa885c):**
- When any FFI kernel is detected, the Main.hs template skips the
  `buildFact2Code`/`buildFact1Code` calls for category_parent,
  article_category, and root_category — they'd be allocated but never
  executed.

**Actual outcome:** 300-scale total_ms dropped from 740ms to 225ms
(-70%), eliminating the largest single startup cost.

## Phase 3: Expanded Lowering

**Goal:** Bypass the interpreter for more predicates.

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
