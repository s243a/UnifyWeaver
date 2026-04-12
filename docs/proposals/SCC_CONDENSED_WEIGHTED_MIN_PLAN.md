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

Status: implemented for the positive-additive candidate path with bounded
measured probes.

Add a runtime applicability check:

- use SCC-condensed strategy when safe and the bounded probe beats the
  layered positive-min path by a margin
- otherwise fall back to current exact frontier

Current selection boundary:

- use the layered dynamic-programming fast path for strictly positive
  additive `min` unless the SCC-condensed measured probe wins
- use the exact frontier fallback otherwise

Next selection work:

- determine where SCC-condensed evaluation can replace the non-positive
  frontier fallback safely

Deliverable:

- strategy split with explicit trace/debug labeling

Examples:

- `PathAwareAccumulation-Min-Frontier`
- `PathAwareAccumulation-Min-SccCondensed`

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

The next coding step should be:

1. identify weighted `Min` recurrence shapes not covered by the current
   positive-additive fast path
2. prototype SCC-condensed evaluation for one of those broader cases
3. benchmark it against the exact frontier fallback
