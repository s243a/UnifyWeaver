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
4. `min_frontier_subset_check_count`
5. `min_frontier_dominated_state_count`
6. `min_frontier_recorded_state_count`
7. `min_frontier_removed_state_count`
8. `min_frontier_target_bucket_count`
9. `min_frontier_bucket_count`
10. `min_frontier_bucket_state_count`
11. `min_frontier_bucket_max_size`
12. `min_frontier_bucket_avg_size`

That coding step is now complete:

1. weighted `Min` fallback states carry deterministic path fingerprints
2. target-local frontiers partition retained states by path length and
   fingerprint before exact subset/dominance verification
3. `min_frontier_target_bucket_count` preserves the old target-bucket view,
   while `min_frontier_bucket_*` now describes exact path-state partitions

The next coding step should keep the multiplicative-to-additive transform as a
narrower later optimization and focus first on whether the remaining lower-cardinality
candidate scans need additional exact prefilters.

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

Both fallback cases preserve output agreement between `All` and `Min`. After
path-state partitioning, the exact `Min` fallback is still slower than `All` on
these benchmark-scale cyclic cases, but dominance-candidate scans are much lower
than the previous broad target-bucket frontier.

| Shape | Scale | All | Min | Speedup | Dominance Candidates | Dominance Checks | Subset Checks | Avg Bucket | Max Bucket |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| negative additive | 300 | 0.938s | 1.590s | 0.59x | 34,704,185 | 1,163,485 | 15,431 | 1.00 | 1 |
| negative additive | 1k | 0.636s | 1.332s | 0.48x | 35,271,278 | 682,397 | 13,775 | 1.00 | 1 |
| multiplicative | 300 | 0.911s | 1.101s | 0.83x | 10,906,078 | 863,266 | 104,355 | 1.00 | 1 |
| multiplicative | 1k | 0.692s | 0.993s | 0.70x | 11,043,000 | 505,982 | 69,064 | 1.00 | 1 |

Interpretation:

- dominance-candidate scans are still the dominant counter, but exact
  path-state partitioning cuts them by roughly `3x` to `3.6x` on these runs
- subset checks drop substantially for negative additive fallback because
  same-cardinality non-matching path states no longer reach exact subset
  verification
- `min_frontier_removed_state_count` stayed at `0` for these runs, so the
  current retained buckets grow but rarely compact themselves
- the remaining candidate scans come from lower-cardinality states that still
  must be considered for exact subset dominance, so any next frontier
  optimization should add exact prefilters there rather than rely only on
  same-cardinality fingerprints
