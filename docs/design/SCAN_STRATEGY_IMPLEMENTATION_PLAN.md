# Scan Strategy — Implementation Plan

Phased rollout for the design specified in
`SCAN_STRATEGY_SPECIFICATION.md`. Each phase produces a landable PR
with tests; later phases depend on earlier ones either for code or
for measurements. Cancel/replan after any phase if measurements
contradict assumptions — the plan is a default, not a contract.

## Sequencing summary

| phase | size | depends on | gates |
|---|---|---|---|
| P0: Algorithm-manifest abstraction | small | — | pure infrastructure; doesn't ship any optimization |
| P1: Pluggable cost-function slot | small | P0 (so the slot can be declared in a manifest) | none — pure refactor |
| P2: At-scale validation re-ingest | small | (data only, no code) | needs P0+P1 in for the matrix bench to exercise the slot, but P2 itself doesn't change code |
| P3: Warm-build + snapshot core | medium | P1 (cost-function abstraction); P2 results inform tuning | empirical signal that warming is worth it |
| P4: `scan_strategy(auto)` resolver + matrix-bench mode | small-medium | P3 | P3 lands |
| P5: Spark routing via retained ranking | medium | P3 (snapshot produces ranked view); validation of P3 | tree retention has a real consumer |
| P6: Adaptive fixed-point iteration | small | P3, P4 | initial measurements show diminishing returns |
| P7+: Live-mode tree updates | medium | P3, P5; research-y use cases | when an algorithm actually needs post-warm tree updates |

P0–P4 are the core deliverable. P5–P7 are extensions; treat as
optional based on P3/P4 data.

---

## Phase 0 — Algorithm-manifest abstraction

### Goal

Introduce the algorithm + optimization-manifest split specified in
`ALGORITHM_MANIFEST_SPECIFICATION.md`. Without this, every new
resolver requires every bench harness to re-declare the same
options, and the scan-strategy options end up duplicated all over
the place.

### Deliverables

- `src/unifyweaver/core/algorithm_manifest.pl` — new module:
  - `load_algorithm_manifest/2` — merges
    `user:algorithm/2` + `user:algorithm_optimization/2` into the
    codegen's option list. Caller options win on conflict.
  - `merge_caller_and_manifest_options/3` — explicit
    caller-wins merge semantics (not the SWI library
    `merge_options/3`, whose convention is different).
  - Validation: warn on orphan optimization, error on duplicate
    `algorithm/2` declarations.
- `src/unifyweaver/targets/wam_haskell_target.pl` — call
  `load_algorithm_manifest/2` at the top of
  `write_wam_haskell_project/3` and
  `compile_wam_runtime_to_haskell/3`, before any resolver.
- `tests/core/test_algorithm_manifest.pl` — unit tests:
  - manifest absent → Options0 unchanged
  - manifest present → options merged
  - caller key wins over manifest key
  - multiple `algorithm_optimization/2` facts concat
  - orphan optimization → warning, ignored
  - duplicate `algorithm/2` → error

### Scope notes

- This phase ships no new optimization. It's pure infrastructure;
  the existing resolvers continue to work as today, just reading
  from a slightly-enriched option list.
- Backwards-compat: workloads without `algorithm/2` declarations
  are completely unaffected. `user:demand_filter_spec/2`,
  `user:max_depth/1` etc. still work alongside manifests.

### Success criteria

- All 167+ existing WAM-Haskell tests pass (no behaviour change).
- New manifest unit tests pass.
- A demo workload using `algorithm/2` + `algorithm_optimization/2`
  emits the same Haskell code as the equivalent caller-options
  invocation.

### Estimated effort

~1 session.

---

## Phase 1 — Pluggable cost-function slot

### Goal

Refactor the existing `Flux` panic stub in
`runDemandBFS` / `runDemandBFSCursor` into a pluggable strategy
slot. The stub becomes one implementation of a generic
`CostFn`-parameterised path. No new functionality lands; this is
preparation.

### Deliverables

- `src/unifyweaver/core/cost_function.pl` — new module:
  - `tree_cost_function(+Name, +Params)` term constructor
  - `validate_cost_function/1` predicate
  - Registry of supported names: `flux`, `hop_distance`,
    `semantic_similarity`
- `templates/targets/haskell_wam/cost_function.hs.mustache` — new
  Haskell-side `CostFn` record + concrete implementations.
- `src/unifyweaver/targets/wam_haskell_target.pl` — codegen for
  the cost-function strategy slot; ensures the existing Flux path
  routes through the new abstraction.
- Tests for the registry, term validation, and codegen emission.

### Scope notes

- The Flux implementation matches today's stub *exactly* —
  `error "Flux strategy not implemented (Phase 2.5)"`. Same panic
  at runtime; just routed through the new abstraction. P3 lands
  the real Flux scoring.
- `hop_distance` gets a real but simple implementation here
  (it's cheap; ships as a working alternative even before Flux).

### Success criteria

- All existing tests pass.
- A new test verifies that
  `tree_cost_function(hop_distance, [max_hops(5)])` emits
  code that produces a non-zero score for nodes within 5 hops.
- The existing Flux-panic test still passes (regression check).

### Estimated effort

~1 session. Small refactor.

---

## Phase 2 — At-scale validation re-ingest

### Goal

Re-create the simplewiki and enwiki Phase 1 LMDB fixtures that
were swept in an earlier workspace reset. Validate the
already-landed cost-model resolvers against measured wall-clock
numbers, not just unit tests.

### Deliverables

No code changes. Tooling and measurement only.

- Re-run streaming pipeline to produce
  `data/benchmark/simplewiki_cats/lmdb_proj/lmdb` and
  `data/benchmark/enwiki_cats/lmdb_proj/lmdb`.
- Run `convert_lmdb_to_phase1_layout.py` to produce
  `lmdb_proj_resident` variants.
- Generate matrix-bench projects at 1k, simplewiki, enwiki
  scales with `resident_auto` mode.
- Sweep: 3 trials × {-N1, -N2, -N4} × 2 roots × resident_auto.
- Append Phase L appendix #14 to `WAM_PERF_OPTIMIZATION_LOG.md`
  with:
  - Measured wall-clock vs `resident_cursor` baseline.
  - Resolver decisions (verbose traces).
  - Miss-rate stats if instrumentation exists.
  - Branching-factor distribution of the category graph (input
    to P3's flux decay tuning).

### Success criteria

- `resident_auto` runs within 5% of `resident_cursor` baseline at
  every scale (resolver agrees with empirical best).
- Branching-factor distribution captured: median, P90, P99 for
  parent and child legs separately.

### Estimated effort

~1 session for ingest + sweep + write-up. Wall-clock dominated by
the streaming pipeline.

---

## Phase 3 — Warm-build + snapshot core

### Goal

Implement the three-phase warm-build / snapshot / steady-state
architecture from the spec. This is the core deliverable.

### Deliverables

Prolog side (`src/unifyweaver/`):

- `core/cost_function.pl` extended with concrete `flux/3`
  implementation (real, not stub).
- `core/scan_strategy.pl` — new module:
  - `warm_budget_resolve/3` — compute defaults from options
  - `stage2_threshold_resolve/3` — derive scan-vs-seek crossover
  - `warm_phase_options/2` — bundle for codegen

Haskell side (`templates/targets/haskell_wam/`):

- `scan_strategy.hs.mustache` — new template with:
  - `WarmHeap`, `WarmIndex`, `Frontier`, `Visited` types
  - `runWarmBuild :: CostFn -> WarmConfig -> IO (WarmHeap, WarmIndex)`
  - `snapshotCache :: WarmHeap -> IntMap NodeId [EdgeTarget]`
  - `snapshotRanked :: WarmHeap -> [(NodeId, Cost)]`
  - Stage-1 cursor loop + stage-2 scan trigger
- `main.hs.mustache` — wire scan-strategy path conditionally on
  `scan_strategy(auto)`.

Tests:

- `tests/core/test_scan_strategy.pl` — option resolution, threshold
  computation, default values.
- `tests/test_wam_haskell_target.pl` — codegen emission, integration.

### Scope notes

- Flux implementation uses the parent/child decay split from the
  spec, with tuning constants from P2's branching-distribution data.
- Stage-2 fixed-point is bounded to ≤ 2 scans for the first cut
  (the spec's recommendation).
- Tree retention defaults to `discard`. Retention is implemented
  but unused by any consumer until P5.

### Success criteria

- Unit tests pass for warm-budget computation, threshold derivation,
  cost-function dispatch.
- End-to-end smoke build of a `scan_strategy(auto)`-enabled project
  on the 1k fixture.
- Functional run on simplewiki produces correct results (tuple
  count matches `resident_cursor` baseline within tolerance).
- Measured wall-clock at simplewiki: within 10% of
  `resident_cursor` at -N4 (no regression at fixture sizes where
  warming should be approximately neutral).

### Estimated effort

~3 sessions. The biggest deliverable in the plan.

---

## Phase 4 — `scan_strategy(auto)` resolver + matrix-bench mode

### Goal

Add the codegen-time auto-resolver and a new matrix-bench mode so
the bench can exercise scan strategies end-to-end.

### Deliverables

- `resolve_auto_scan_strategy/2` in `wam_haskell_target.pl`:
  - Reads workload metadata + cost-model outputs.
  - Picks default `tree_cost_function`, `warm_budget_nodes`,
    `stage2_scan_threshold`, `iterations`.
- `parse_lmdb_mode(resident_warm, ...)` in
  `generate_wam_haskell_matrix_benchmark.pl`:
  - Bundles `scan_strategy(auto)` with sensible defaults.
- Tests pinning resolver behaviour at known input combinations
  (mirror of `cache_strategy(auto)` test set).
- Phase L appendix #15: empirical comparison of
  `resident_cursor` vs `resident_warm` at 1k, simplewiki, enwiki.

### Scope notes

- The resolver is a thin wrapper; most decisions defer to spec
  defaults. Picks cost-function based on availability of
  embeddings (semantic if available, hop_distance otherwise) and
  graph properties (flux if branching distribution exists from
  P2; hop_distance fallback).

### Success criteria

- All resolver tests pass deterministically.
- `resident_warm` at simplewiki shows measurable improvement over
  `resident_cursor` at -N4, *or* — equally informative — measured
  worsening that tells us the warm-build path isn't worth it at
  this scale. Either outcome is publishable.
- enwiki regression check: `resident_warm` doesn't break what
  `resident_cursor` already runs successfully.

### Estimated effort

~1 session for the resolver + 1 for the measurement sweep.

---

## Phase 5 — Spark routing via retained tree (snapshot_only mode)

### Goal

Use the snapshot's ranked view (`tree_retention(snapshot_only)`)
to partition parMap-driven sparks into capability-local chunks.
Closes the MoE-spark-routing gap that has been blocking the
per-HEC L1 cache from being useful.

### Deliverables

- `partitionByRank :: Int -> RankedView -> [[NodeId]]` in
  `scan_strategy.hs.mustache`.
- `main.hs.mustache` — when `tree_retention(retain)` is set,
  replace the seed `parMap rdeepseq` with a per-capability
  `mapConcurrently` driven by `partitionByRank`.
- New option: `spark_routing(capability_local)` (default
  `unsharded`).
- Resolver hook: `scan_strategy(auto)` + `expected_query_count > 1`
  → automatically opts into retention + capability_local routing
  if L1 is the chosen cache mode.
- Tests pinning the partition algorithm + routing emission.
- Phase L appendix #16: empirical comparison of
  `parMap` baseline vs capability-routed L1 at simplewiki -N4.

### Success criteria

- Capability-routed L1 captures non-overlapping coverage in
  threadCapability traces.
- Measured -N4 speedup matches or exceeds sharded L2 baseline
  (the existing default) on simplewiki.

### Estimated effort

~2 sessions. Depends on P3 producing a working ranked view.

---

## Phase 6 — Adaptive fixed-point iteration

### Goal

Lift the `stage2_max_scans(2)` bound to a measurement-driven
fixed-point. Implement only if P3/P4 measurements show diminishing
returns from the first scan don't actually plateau at 2 — i.e. if
a third or fourth scan would still meaningfully grow the tree.

### Deliverables

- `convergence_metric/2` predicate in `scan_strategy.pl`.
- Loop in `runWarmBuild` that scans until convergence or limit.
- New option `convergence_threshold(F)`.

### Scope notes

May be unnecessary. Filed as optional. Likely de-prioritised in
favour of cache-warming follow-ups if P3/P4 data shows 2 scans
captures >95% of the tree.

### Estimated effort

~1 session if pursued.

---

## Phase 7+ — Live-mode tree updates

### Goal

Support `tree_retention(live)` — keep `WarmHeap` + `WarmIndex`
alive past snapshot and let downstream algorithms update them.
Unlocks the class of algorithms that need post-warm re-ranking:
query-history-driven re-rank, adaptive cost-function switching,
hot-region expansion, workload prioritisation.

### Deliverables (per-algorithm; not a single PR)

Per actual `live`-mode algorithm we want to ship:

- The algorithm-specific update logic (what triggers a re-rank,
  what cost is computed, how the cache reflects re-ranks).
- Thread-safety for `WarmHeap` access (`MVar`/`IORef` guards;
  concurrent priority queue if measurement shows contention).
- Optional `treeRebuild` (re-snapshot the cache from the
  updated tree) — useful when the workload pattern shifts enough
  that the original snapshot is stale.

### Scope notes

Each `live`-mode algorithm is a research-y effort. Don't ship
the infrastructure for a hypothetical algorithm — only land
guards / concurrency support when a specific algorithm needs
them. The spec leaves the door open structurally; this phase
walks through it.

### Estimated effort

Per-algorithm; multiple PRs. ~2–3 sessions per algorithm
including measurement and tuning.

---

## Cross-cutting concerns

### Determinism

All scan-strategy code paths must be deterministic given the same
inputs (same fixture, same options, same seed). Avoid
non-deterministic `parMap` ordering inside the warm phase; it's
fine in steady-state but warm needs reproducibility for tests.

### Memory budgets

P3's warm phase peaks at:

```
peak_warm_mem ≈ |WarmHeap| × (cost + node_id + edges)
              + |WarmIndex| × (node_id + cost)
              + |Frontier| × (node_id + cost)
              + |Visited| × node_id
```

For warm_budget_nodes = 10% of fact_count = 30k at simplewiki,
average 5 children/node, ~30 bytes per entry: ≈ 5 MB. Fits
easily. enwiki at 10% = 990k nodes × ~30 bytes = ~30 MB. Still
fine. Article-level scales would push this; defer until that
data lands.

### Telemetry

P3+ must support optional verbose tracing:

```
[WAM-Haskell] scan_strategy: warm_nodes=29950 warm_ms=2341 stage2_scans=1
[WAM-Haskell] scan_strategy: snapshot bytes=4.8MB ranked_retained=True
[WAM-Haskell] scan_strategy: steady-state hits=12839 misses=144 (1.1% miss rate)
```

Reuses the `cache_strategy_verbose(true)` pattern. Add a new
option `scan_strategy_verbose(true)` for full traces.

### What we won't do

Out of scope, file and forget:

- Online cache growth (cache misses don't memoise).
- Mid-workload re-warming.
- Cost-function autoselect via ML — the registry has 3 options
  for now; pick one explicitly.
- Distributed warm phases (single-process only).
- Persistence: warm/snapshot artefacts are per-process, not
  written to disk.

## Open questions to resolve via measurement

P2 and P3 data will inform answers:

1. **Default `warm_budget_nodes(N)` fraction.** Spec says 10%; is
   that right at simplewiki? At enwiki?
2. **Default `parent_decay` / `child_decay` for flux.** Spec says
   0.5 / 0.3 as starting points; what does the branching
   distribution actually argue for?
3. **`iterations(1)` vs `iterations(3)`.** Where does the
   accuracy/speed curve hit diminishing returns?
4. **`stage2_max_scans(N)`.** Spec says 2. Validate via P3.
5. **Whether semantic_similarity is worth supporting** at all,
   or whether flux + hop_distance covers the workloads we care
   about. P5 might inform via spark-routing accuracy.

## See also

- `SCAN_STRATEGY_PHILOSOPHY.md` — the why.
- `SCAN_STRATEGY_SPECIFICATION.md` — the what.
- `CACHE_COST_MODEL_PHILOSOPHY.md` — the cost model this layer
  builds on.
- `WAM_PERF_OPTIMIZATION_LOG.md` — appendices #11–#13 (cost-model
  resolver work); appendices #14+ for the validation and warm-
  build measurements.
