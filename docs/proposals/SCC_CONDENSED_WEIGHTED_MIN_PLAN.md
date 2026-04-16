# SCC-Condensed Weighted Min: Implementation Plan

## Goal

Make weighted `min` on `PathAwareAccumulationNode` meaningfully faster
than `All` while preserving exact per-path semantics.

Current status:

- correctness is in place
- positive-additive weighted `min` now has a fast runtime path and
  clearly beats `All`
- SCC instrumentation is in place in the weighted benchmark harness
- an internal SCC-condensed weighted-min candidate now exists for
  positive-additive `PathAwareAccumulationNode` workloads, with bounded
  measured selection against the existing layered dynamic-programming path
- the additive fast-path boundary now includes non-negative steps, so
  zero-cost source weights no longer force the exact visited-state frontier
  fallback when the additive form is otherwise safe and depth-bounded
- negative additive steps and non-additive recurrence expressions are now
  covered by explicit fallback-shape survey tests and runtime strategy
  labeling
- the exact weighted `Min` frontier fallback now records lightweight
  frontier metrics for candidate counts, dominance checks, subset checks,
  and retained frontier bucket sizes
- on the current positive-additive benchmark shape, the measured selector
  rejects SCC condensation because the layered path is still cheaper after
  SCC build/probe overhead

So the next step remains algorithmic, but should focus on cases where the
frontier fallback still appears instead of replacing the already-fast
positive-additive layered path unconditionally.

## Phase 0: Baseline Preservation

Before adding a new strategy:

1. keep the current exact frontier implementation as the fallback
2. preserve the existing benchmark script and output-comparison checks
3. do not change user-facing syntax

This gives us a known-correct baseline to compare against.

## Phase 1: Graph Structure Measurement

Status: completed for the current benchmark harness.

Add instrumentation to the weighted benchmark path to measure:

- SCC count
- SCC size distribution
- largest SCC
- condensation DAG size

Current measured shape is favorable:

- `10k`: `8247` nodes, `25227` edges
- `8204` SCCs total
- `17` cyclic SCCs
- largest cyclic SCC size `35`

Deliverable:

- a benchmark note or doc update summarizing whether the benchmark graph
  is structurally favorable for SCC condensation

Reason:

- if SCCs are huge, condensation may not help much
- if SCCs are mostly small with a sparse DAG between them, the approach
  is promising

## Phase 2: Internal Runtime Prototype

Status: implemented as a measured candidate for the current
positive-additive runtime shape.

Prototype a new internal runtime strategy for the remaining weighted
`min` cases not already handled by the positive-additive fast path:

- `PathAwareAccumulationNode`
- `TableMode.Min`

Algorithm sketch:

1. build SCC ids for the edge relation
2. build condensation DAG
3. define component entry/exit boundary states
4. compute exact local transfer costs inside SCCs
5. compose those costs on the condensation DAG with DAG-style `min`
   propagation

Deliverable:

- an internal C# runtime path behind the existing node
- no planner change required yet
- current implementation records SCC graph metrics, SCC probe phases, and
  probe local/outer state counts through `QueryExecutionTrace`

## Phase 3: Strategy Selection

Status: implemented for additive non-negative candidate paths with bounded
measured probes.

Add a runtime applicability check:

- use SCC-condensed strategy when safe and the bounded probe beats the
  layered additive-min path by a margin
- otherwise fall back to current exact frontier

Current selection boundary:

- strictly positive additive `Min` keeps the existing positive layered/SCC
  measured path
- non-negative additive `Min` now uses the same depth-bounded layered/SCC
  measured candidate and reports separate non-negative trace labels
- use the exact frontier fallback otherwise

Next selection work:

- use the explicit frontier-fallback labels to measure the remaining
  unsupported shapes before adding more shortcuts
- determine whether a restricted non-additive class, such as positive
  multiplicative weights, can be transformed safely without losing exact
  simple-path semantics
- keep negative-step additive recurrences on the exact frontier path until
  there is a proof that pruning and condensation remain sound

Deliverable:

- strategy split with explicit trace/debug labeling

Examples:

- `PathAwareAccumulation-Min-Frontier`
- `PathAwareAccumulation-Min-SccCondensed`
- `PathAwareAccumulationSeededMinFrontierFallback`

## Phase 4: Benchmark Loop

Rerun:

```bash
python examples/benchmark/benchmark_weighted_shortest_path.py --scales 300,1k,5k,10k --repetitions 1
```

Track:

- output match
- total runtime
- query time
- SCC metrics
- SCC probe/solve phases
- local state counts and outer condensation-DAG state counts

Current positive-additive benchmark results already satisfy the intended
performance goal:

- `300`: `2.93x`
- `1k`: `2.62x`
- `5k`: `4.56x`
- `10k`: `6.84x`

So for the SCC work, success criteria should be read as applying to the
broader remaining weighted `min` cases that still fall back to the exact
frontier algorithm.

## Phase 5: Documentation and Rollout

Once the runtime strategy is validated:

1. update benchmark docs with the new results
2. update mode-directed tabling docs to reflect the weighted `min` fast
   path
3. explicitly document the fallback boundary

## Phase 6: Rust Follow-Up

Only after the C# runtime strategy is stable:

1. extend Rust native lowering for counted `min` if needed
2. extend Rust native lowering for weighted `min`
3. prefer the same condensed-graph abstraction rather than a
   frontier-heavy path-state clone

Reason:

- Rust should inherit the better abstraction, not the current C#
  exploratory intermediate form

## Risks

### Risk 1: SCCs are too large

Mitigation:

- keep the exact frontier fallback
- use the SCC strategy only when condensation materially simplifies the
  graph

### Risk 2: Internal transfer summarization becomes lossy

Mitigation:

- compare against `All` benchmark outputs continuously
- keep the prototype behind runtime strategy labeling until validated

### Risk 3: SCC preprocessing overhead dominates

Mitigation:

- measure SCC construction separately
- cache condensation structures per relation when feasible

## Immediate Next Coding Step

The original first code step after this document set was:

1. add SCC measurement instrumentation to the weighted benchmark/runtime
2. record actual SCC structure for `300/1k/5k/10k`
3. decide whether the component-DAG strategy is justified by the data

That step is now complete.

The fallback survey step now covers two representative cases:

1. negative additive weighted `Min`, which matches the additive expression
   shape but is rejected by the non-negative proof
2. positive multiplicative weighted `Min`, which is monotone for the test
   data but is not additive and therefore still requires exact path-state
   evaluation

The frontier metric step now records:

1. `min_frontier_candidate_count`
2. `min_frontier_dominance_check_count`
3. `min_frontier_dominance_candidate_check_count`
4. `min_frontier_same_fingerprint_candidate_check_count`
5. `min_frontier_lower_count_candidate_check_count`
6. `min_frontier_lower_count_index_probe_count`
7. `min_frontier_subset_check_count`
8. `min_frontier_dominated_state_count`
9. `min_frontier_recorded_state_count`
10. `min_frontier_removed_state_count`
11. `min_frontier_target_bucket_count`
12. `min_frontier_bucket_count`
13. `min_frontier_bucket_state_count`
14. `min_frontier_bucket_max_size`
15. `min_frontier_bucket_avg_size`

The path-state partitioning step is now complete:

1. weighted `Min` fallback states carry deterministic path fingerprints
2. target-local frontiers partition retained states by path length and
   fingerprint before exact subset/dominance verification
3. `min_frontier_target_bucket_count` preserves the old target-bucket view,
   while `min_frontier_bucket_*` now describes exact path-state partitions

The lower-cardinality prefilter step is also now implemented:

1. same-cardinality checks use direct fingerprint indexes
2. lower-cardinality checks use lazy representative-node indexes only after a
   path-count bucket is large enough for the index to pay for itself
3. small buckets direct-scan to avoid dictionary overhead
4. eager dominated-state removal is no longer scanned during insert, because
   retained dominated states are correctness-safe and the measured removals
   stayed at `0`

The positive multiplicative step is now implemented as a direct-product layered
strategy rather than as a geometric-mean or log-output strategy:

1. recurrence must be `Acc is Acc1 * Factor` with the base expression matching
   `Factor`
2. every reachable factor must be finite and at least `1`
3. subunit factors stay on the exact frontier fallback because they would
   become negative steps under a log transform
4. the runtime minimizes products directly, avoiding log/exp output drift

The cross-workload comparison step is now complete, and counted simple-path
closure now uses compact visited paths for cycle checks.
`PathAwareTransitiveClosureNode` emits `path_state_*` metrics, giving a
non-weighted non-DAG comparison point before adding more generic frontier
indexes.

## Frontier Fallback Metric Survey

Measured with:

```bash
python examples/benchmark/benchmark_weighted_shortest_path.py \
  --scales 300,1k --repetitions 1 --weight-mode negative \
  --recurrence-mode additive

python examples/benchmark/benchmark_weighted_shortest_path.py \
  --scales 300,1k --repetitions 1 --weight-mode positive \
  --recurrence-mode multiplicative
```

The negative-additive fallback preserves output agreement between `All` and
`Min`. After lazy lower-count representative prefiltering, the exact `Min`
fallback is still slower than `All` on these benchmark-scale cyclic cases, but
it is faster than the previous path-state partitioning-only fallback.

| Shape | Scale | All | Min | Speedup | Dominance Candidates | Lower Candidates | Index Probes | Subset Checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| negative additive | 300 | 0.871s | 1.478s | 0.59x | 20,404,270 | 19,826,544 | 912,133 | 12,621 |
| negative additive | 1k | 0.563s | 1.217s | 0.46x | 16,522,183 | 16,183,799 | 1,138,494 | 11,157 |

Positive multiplicative recurrence now uses the product layered strategy on the
benchmark shape:

| Shape | Scale | All | Min | Speedup | Strategy |
| --- | ---: | ---: | ---: | ---: | --- |
| multiplicative | 300 | 0.892s | 0.264s | 3.38x | `PathAwareAccumulationSeededMinNonNegativeMultiplicativeLayered` |
| multiplicative | 1k | 0.587s | 0.212s | 2.76x | `PathAwareAccumulationSeededMinNonNegativeMultiplicativeLayered` |

Counted simple-path closure provides the non-weighted comparison point. It does
not use the weighted `min_frontier_*` dominance machinery; its measured cost is
raw path-state traversal:

| Shape | Scale | All | Min | Speedup | All Output Rows | Min Output Rows | All Successor Candidates | Min Successor Candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| counted shortest path | 300 | 0.634s | 0.214s | 2.97x | 602,808 | 30,968 | 982,581 | 101,371 |
| counted shortest path | 1k | 0.450s | 0.180s | 2.50x | 352,522 | 10,328 | 592,698 | 38,196 |

Counted-closure phase split after typed row buffering, pre-sized
materialization, edge-state node-id preindexing, and removing per-row buffer
timing from the traversal hot path:

| Scale | Mode | Traversal | Row Creation | Result Materialization | Best-Known Flush/Sort |
| --- | --- | ---: | ---: | ---: | ---: |
| 300 | All | 217.951ms | 0.000ms | 99.800ms | n/a |
| 300 | Min | 53.414ms | 0.000ms | 7.487ms | 13.907ms |
| 1k | All | 122.026ms | 0.000ms | 63.798ms | n/a |
| 1k | Min | 21.147ms | 0.000ms | 1.794ms | 5.640ms |

Interpretation:

- dominance-candidate scans are still the dominant counter, but lazy
  representative-node prefiltering cuts the path-state-partitioning-only
  counts by roughly `1.35x` to `2.1x` on these runs
- same-cardinality checks are now visible separately from lower-count scans,
  which confirms that the remaining work is almost entirely lower-count
  dominance probing
- `min_frontier_removed_state_count` stayed at `0` for these runs, so the
  runtime avoids eager removal scans and keeps dominated retained states as a
  correctness-safe tradeoff
- positive multiplicative recurrence no longer contributes frontier counters on
  the benchmark shape because it does not enter the fallback
- counted closure confirms that not every non-DAG path-state workload has the
  same bottleneck: its measured cost is successor expansion and depth-limit
  pruning, not subset-dominance lookup
- compact visited paths reduce counted-closure allocation overhead while
  leaving the measured traversal counters unchanged
- typed row buffering and pre-sized final materialization reduce avoidable
  row-output overhead, but traversal is still the largest counted-closure
  phase
- edge-state node-id preindexing removes the per-successor candidate node-id
  dictionary lookup from traversal while preserving output hashes and
  `path_state_*` counters
- row-buffer recording no longer starts a stopwatch for every emitted path
  row; the explicit `path_state_row_creation` phase is now `0`, and row-buffer
  work is included in traversal timing
- the next broad optimization should avoid adding more generic frontier indexes
  until another dominance-heavy fallback shape appears; for counted closure,
  remaining work should target expansion/materialization overhead
