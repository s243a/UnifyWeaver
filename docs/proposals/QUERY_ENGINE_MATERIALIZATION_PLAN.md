# Query Engine Materialization Plan

## Goal

Move the parameterized query engine toward engine-owned materialization by
making streamed ingestion the preferred route and external pre-materialization
a supported fallback.

## Stage 1: Narrow Benchmark-Driven Streaming Paths

Status:

- implemented for the current DAG benchmark paths

Work:

- allow providers to expose streamed delimited sources
- let DAG-specialized runtime nodes ingest those sources directly
- stop preloading benchmark TSV rows into `InMemoryRelationProvider` when the
  runtime can ingest them itself

Success criteria:

- correctness unchanged
- benchmark harness becomes thinner
- engine owns more of the retained-state decision

## Stage 2: Clarify Retention Modes In The Runtime

Status:

- implemented for streamed and replayable relation bindings in the current C# query runtime

Work:

- make the distinction clearer between:
  - streaming source
  - replayable source
  - operator-owned retained state
  - externally materialized source
- document which operators can stay single-pass and which need replay/indexed
  access

Success criteria:

- fewer ambiguous materialization decisions outside the engine
- clearer runtime contracts for new operators
- explicit provider/runtime hooks for choosing streaming, replayable, or
  external-materialized access

## Stage 3: Expand Streamed Ingestion Beyond Current DAG Cases

Status:

- started for scan-oriented and path-aware benchmark paths using the shared retention boundary
- path-aware shortest-path operators now build compact operator-owned edge state
  instead of consuming generic replayed edge tuples
- shortest-path and weighted-shortest-path operators now emit compact
  operator-owned `(group, root, min_value)` summaries instead of consuming
  broad seeded path rows plus benchmark-side regrouping
- effective-distance and category-influence operators now build compact
  operator-owned `(group, root, weight_sum)` state instead of consuming
  generic path rows plus benchmark-side regrouping

Work:

- identify additional operators that can ingest directly from streamed sources
- move retained-state construction into those operators where it pays off
- keep replacing generic replay/path rows with narrower operator-owned summaries
- avoid generic `object[]` fact preloading when a narrower retained form is
  sufficient

Success criteria:

- more benchmark/program paths use engine-owned retention
- reduced parser-side or harness-side eager structure building

## Stage 4: Cost-Based Strategy Selection

Status:

- started for shortest-path and weighted-shortest-path grouped minima
- started for effective-distance and category-influence grouped weight sums
- started for path-aware edge-relation retention, where the runtime can now
  choose between direct streamed edge-state build, replayable buffering, and
  external materialized fallback before building operator-owned edge state
- started for DAG grouped reach-count and DAG longest-depth relation retention,
  where the runtime can now choose between direct streamed DAG build,
  replayable buffering, and external materialized fallback before building
  operator-owned DAG state, and where the fallback edge/seed materialization
  path now also follows that planner-selected retention strategy explicitly
- started for generic scan relation retention, where `RelationScanNode`,
  `PatternScanNode`, and scan-heavy join/negation/aggregate consumers can now
  choose between direct streamed scan access, replayable buffering, and
  external materialized fallback before building list/set views
- focused validation for that generic scan planner surface now lives in
  `examples/benchmark/benchmark_scan_materialization.py`, mirroring the
  earlier focused closure harness
- started for generic closure relation retention, where `TransitiveClosureNode`,
  grouped closure variants, and `PathAwareAccumulationNode` auxiliary-relation
  loading can now choose between direct streamed access, replayable
  buffering, and external materialized fallback before building closure
  indices
- started for generic closure pair strategy planning, where seeded and grouped
  closure-pair probes can now choose among forward, backward, mixed,
  mixed-with-pair-probe-cache, and memoized source/target strategies through
  an explicit planner surface, and `Auto` can now run bounded measured probes
  for ambiguous mixed request sets
- focused validation for that closure-pair planner surface now lives in
  `examples/benchmark/benchmark_closure_pair_planning.py`; the grouped mode now
  produces non-empty rows, the summary reports the best effective pair plan
  separately from the requested override label, and measured runs emit
  `closure_pair_probe_*` phases
- started for path-aware support relations, where grouped minima and weight-sum
  operators can now choose streaming, replayable buffering, or external
  materialized access for `RootRelation` and `SeedRelation` before grouped
  summary construction
- the runtime can now choose between compact grouped summaries and legacy
  seeded-row regrouping for both operator families through a shared
  grouped-summary policy layer
- `QueryExecutorOptions.PathAwareGroupedMinStrategy` can force `Auto`,
  `CompactGrouped`, or `LegacySeededRows` for grouped minima benchmarking
- `QueryExecutorOptions.PathAwareWeightSumStrategy` can force `Auto`,
  `CompactGrouped`, or `LegacySeededRows` for grouped weight-sum benchmarking
- the current implementation shares one grouped-summary heuristic/resolution
  path internally, now records measured cost buckets at the retention
  decision points, and keeps per-family override knobs for benchmarking
- `QueryExecutorOptions.PathAwareEdgeRetentionStrategy` can force `Auto`,
  `StreamingDirect`, `ReplayableBuffer`, or `ExternalMaterialized` for
  path-aware edge-retention benchmarking
- `QueryExecutorOptions.DagRelationRetentionStrategy` can force `Auto`,
  `StreamingDirect`, `ReplayableBuffer`, or `ExternalMaterialized` for DAG
  relation-retention benchmarking
- the current selector only runs bounded measured probes in ambiguous cases;
  obvious grouped-summary shapes still short-circuit structurally, while
  path-aware edge retention and DAG relation retention now measure competing
  source paths directly when the structural signal is unclear
- the current implementation now shares one internal relation-retention policy
  layer for DAG and path-aware edge selection, while keeping separate public
  override knobs for benchmarking and diagnosis
- the runtime now shares one internal materialization-planner layer above
  the shared relation-retention and grouped-summary policy components
- the path-aware grouped-summary family uses both branches of that planner,
  and now also routes root/seed support-relation access through the shared
  relation-retention side of the same planner surface, while the DAG family,
  the generic scan family, and the generic closure family currently use the
  relation-retention branch of that planner
- `QueryExecutorOptions.ClosureRelationRetentionStrategy` can force `Auto`,
  `StreamingDirect`, `ReplayableBuffer`, or `ExternalMaterialized` for
  generic closure benchmarking and diagnosis
- `QueryExecutorOptions.ClosurePairStrategy` can force `Auto`, `Forward`,
  `Backward`, `MemoizedBySource`, `MemoizedByTarget`, `MixedDirection`, or
  `MixedDirectionWithPairProbeCache` for closure-pairs benchmarking and
  diagnosis
- the current per-family override knobs and trace surface are still preserved

Work:

- use measured cost buckets to guide whether the engine prefers:
  - direct streaming
  - replayable buffering
  - indexed retained state
  - fallback external materialization
- keep broadening the shared planner so relation-retention and
  grouped-summary choices can be coordinated through one runtime layer when
  both policy components exist for an operator family
- keep this heuristic-driven at first, then refine as profiling improves

Success criteria:

- the materialization strategy is chosen where operator knowledge exists
- measured comparisons can confirm when `Auto` picks the right retained form
- eager fallback remains available but is no longer the default instinct

## Guardrails

- do not remove external materialization compatibility prematurely
- do not force purely streaming execution onto operators that clearly need
  retained state
- do not push retention policy back into the parser just to simplify the
  engine's current implementation

## Current Recommendation

Near-term implementation work should continue to prefer:

- streamed sources into the runtime
- explicit runtime retention requests (`Streaming`, `Replayable`,
  `ExternalMaterialized`)
- operator-owned retained state
- external materialization only when the streamed/operator-owned path is not yet
  available or is measurably worse
- heuristic or measured strategy selection when multiple operator-owned retained
  forms are already available
