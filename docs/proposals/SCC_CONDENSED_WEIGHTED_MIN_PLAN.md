# SCC-Condensed Weighted Min: Implementation Plan

## Goal

Make weighted `min` on `PathAwareAccumulationNode` meaningfully faster
than `All` while preserving exact per-path semantics.

Current status:

- correctness is in place
- local runtime optimizations reduced the cost of the exact frontier
- weighted `min` is still slower than `All`

So the next step must be algorithmic.

## Phase 0: Baseline Preservation

Before adding a new strategy:

1. keep the current exact frontier implementation as the fallback
2. preserve the existing benchmark script and output-comparison checks
3. do not change user-facing syntax

This gives us a known-correct baseline to compare against.

## Phase 1: Graph Structure Measurement

Add instrumentation to the weighted benchmark path to measure:

- SCC count
- SCC size distribution
- largest SCC
- condensation DAG size

Deliverable:

- a benchmark note or doc update summarizing whether the benchmark graph
  is structurally favorable for SCC condensation

Reason:

- if SCCs are huge, condensation may not help much
- if SCCs are mostly small with a sparse DAG between them, the approach
  is promising

## Phase 2: Internal Runtime Prototype

Prototype a new internal runtime strategy for:

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

## Phase 3: Strategy Selection

Add a runtime applicability check:

- use SCC-condensed strategy when safe
- otherwise fall back to current exact frontier

Possible early rule:

- enable only for monotone additive `min`
- disable for unsupported expression shapes

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
- local state counts

Success criteria:

- exact output match at all scales
- weighted `min` faster than `all` at larger scales
- visible crossover toward a meaningful win, ideally around `2x`

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

The first code step after this document set should be:

1. add SCC measurement instrumentation to the weighted benchmark/runtime
2. record actual SCC structure for `300/1k/5k/10k`
3. decide whether the component-DAG strategy is justified by the data

That keeps the next implementation grounded in the actual graph rather
than intuition alone.
