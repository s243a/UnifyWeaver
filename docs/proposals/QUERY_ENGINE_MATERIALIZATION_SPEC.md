# Query Engine Materialization Specification

## Scope

This document specifies the intended materialization boundary for the
parameterized query engine.

It does not attempt to fully redesign every provider or plan node. It defines a
narrow architectural contract that current and future implementations should
move toward.

## Terminology

- streamed source: tuples are decoded on demand and not eagerly retained by the
  parser layer
- replayable source: the engine can ask for a second pass without requiring the
  original caller to rebuild the source manually
- retained state: operator-owned in-memory state built by the engine
- external materialization: facts are fully built outside the engine and then
  handed in as in-memory relations

## Current Runtime Hooks

The current runtime now exposes a narrow explicit retention contract:

- `RelationRetentionMode`
  - `Streaming`
  - `Replayable`
  - `ExternalMaterialized`
- `RelationBinding`
- `IRetentionAwareRelationProvider`
- `IReplayableRelationSource`

These hooks do not solve every ingestion case yet, but they make the
retention choice explicit at the runtime/provider boundary instead of hiding it
inside benchmark-specific wiring. The current runtime now also shares one
internal measured relation-retention policy layer between DAG relation
selection, path-aware edge selection, generic scan selection, and generic
closure relation selection, while preserving separate public override
surfaces for those families. The runtime now also shares one internal
materialization-planner layer above those policy components: the path-aware
grouped-summary family uses both relation-retention and grouped-summary
planning through that shared layer, while the DAG, generic scan, and
generic closure families currently use the relation-retention branch of the
same planner.

## Required Capabilities

### 1. Streaming-capable providers

A relation provider may expose a streamed source for a predicate.

Current narrow example:

- delimited two-column TSV relation sources for benchmark DAG workloads

Future streamed providers may expose:

- other delimited layouts
- structured binary sources
- database-backed iterators
- socket or pipe-backed tuple streams

### 2. Engine-owned retained state

A plan node may consume a streamed source and build only the retained state it
needs.

Examples:

- DAG grouped reachability builds adjacency and grouped bitset/count state
- DAG longest depth builds adjacency and scalar suffix-depth state
- where direct streamed DAG build, replayable buffering, and external
  materialized rows are all viable inputs for those DAG operators, the runtime
  can choose among them through a measured retention selector and still expose
  an explicit override via `QueryExecutorOptions.DagRelationRetentionStrategy`
- when the streamed DAG fast path is not used, the fallback edge/seed
  materialization path now also honors that same planner-selected retention
  strategy instead of dropping back to generic raw fact-list loading
- path-aware shortest-path operators can build compact source->targets edge state
  instead of retaining generic edge tuples
- where direct streaming, replayable buffering, and external materialized rows
  are all viable sources for that edge state, the runtime can choose among
  them through a measured retention selector and still expose an explicit
  override via `QueryExecutorOptions.PathAwareEdgeRetentionStrategy`
- path-aware grouped-summary operators can now also route their root and seed
  support relations through that same measured relation-retention boundary,
  with an explicit override via
  `QueryExecutorOptions.PathAwareSupportRelationRetentionStrategy`
- generic closure operators can route edge and support relations through the
  same measured relation-retention boundary before building successor,
  predecessor, or auxiliary lookup indices, with an explicit override via
  `QueryExecutorOptions.ClosureRelationRetentionStrategy`
- delimited relation ingestion now honors `DelimitedRelationSource.ExpectedWidth`
  beyond 2-column rows, so grouped benchmark and planner harness sources can
  flow through the runtime without silently dropping extra fields
- generic closure pair probes can now choose among forward, backward, mixed,
  mixed-with-pair-probe-cache, and memoized source/target strategies through
  an explicit planner surface, with an override via
  `QueryExecutorOptions.ClosurePairStrategy`; `Auto` can now use bounded
  measured probes for ambiguous mixed request sets, and focused validation in
  `examples/benchmark/benchmark_closure_pair_planning.py` now produces
  non-empty grouped rows and reports the best effective pair plan separately
  from the raw override label
- generic scan planning now also has focused validation coverage in
  `examples/benchmark/benchmark_scan_materialization.py`, so scan-heavy join,
  negation, aggregate, relation-scan, and pattern-scan paths can be checked
  directly rather than only through indirect workload coverage
- shortest-path and weighted-shortest-path operators can emit compact
  `(group, root, min_value)` summaries instead of retaining full seeded path rows
- where both grouped minima and legacy seeded-row regrouping are available, the
  runtime can choose between them through a shared grouped-summary policy
  layer, record measured cost buckets for that choice, and still expose an
  explicit override via `QueryExecutorOptions.PathAwareGroupedMinStrategy`
- effective-distance and category-influence operators can build compact
  `(group, root, weight_sum)` summaries instead of retaining full path rows
- where grouped weight sums and legacy seeded-row regrouping both exist, the
  runtime can choose between them through that same grouped-summary policy
  layer, record measured cost buckets for that choice, and still expose an
  explicit override via `QueryExecutorOptions.PathAwareWeightSumStrategy`
- when an operator family has both relation-retention and grouped-summary
  policy layers available, the runtime can coordinate them through a shared
  materialization planner layer instead of treating them as unrelated
  decisions
- other operators may request replay buffers or indexes when needed

### 3. External materialization fallback

The engine must still work when facts are already externally materialized.

That means:

- existing `IRelationProvider` behavior remains valid
- `InMemoryRelationProvider` remains supported
- streamed ingestion augments the engine rather than replacing compatibility
  paths

## Preferred Execution Rule

When both of these are true:

- the provider can stream tuples for a predicate
- the operator has a specialized streamed-ingestion path

then the engine should prefer the streamed path over eager external
materialization.

## Non-Goals

This specification does not require:

- purely streaming execution for every recursive operator
- zero retained state
- removal of all external materialization support
- one universal streaming strategy for all workloads

Some operators will still need retained state. The design point is that the
engine, not the parser, should decide when that is warranted.

## Current Narrow Implementation Pattern

For the current benchmark/runtime surface, the streamed path is:

1. provider exposes delimited relation sources
2. the runtime requests either `Streaming` or `Replayable` access
3. replayable bindings can cache a reusable relation source instead of ad hoc `ToList()` calls
4. DAG, scan, and path-aware operators read rows through that retention boundary
5. DAG grouped reach-count and longest-depth operators can use measured
   relation-retention selection to choose between direct streamed DAG build,
   replayable buffering, and external materialized fallback before building
   their operator-owned graph state
6. path-aware shortest-path operators can build a compact edge-state cache
   directly from streamed facts instead of generic replayed edge rows
7. where streaming, replayable buffering, and external materialized rows are
   all viable sources for that cache, the runtime can use measured edge-retention
   buckets and bounded probes to choose among them through that same shared
   relation-retention policy layer before honoring an explicit executor override
8. shortest-path and weighted-shortest-path operators can emit compact
   grouped root minima directly from streamed edge/seed inputs
9. where grouped minima and legacy seeded-row regrouping both exist, the runtime
   can select between them through a shared grouped-summary policy layer,
   record measured cost buckets for that decision, and use bounded probes in
   ambiguous cases before falling back to an explicit executor option
10. effective-distance and category-influence operators can emit compact
   grouped root-weight summaries directly from streamed edge/seed inputs
11. where grouped weight sums and legacy seeded-row regrouping both exist, the
   runtime can select between them through that same grouped-summary policy
   layer, record measured cost buckets for that decision, and use bounded
   probes in ambiguous cases before falling back to an explicit executor option
12. generic `RelationScanNode` and `PatternScanNode` access now also route
   through measured relation-retention selection, so scan-heavy join,
   negation, and aggregate paths can choose between direct streaming,
   replayable buffering, and external materialized fallback before building
   list/set views
13. generic closure operators now also route edge and support relations
   through measured relation-retention selection before building successor,
   predecessor, or auxiliary lookup indices
14. the current runtime now routes path-aware, DAG, generic scan, and
   generic closure planning through a shared internal
   materialization-planner layer
15. the current path-aware grouped-summary family now also routes its
   root and seed support relations through measured planner-driven relation
   retention before grouped-summary construction
16. for the current path-aware grouped-summary family, that planner
   coordinates the earlier edge-retention choice with the later
   grouped-summary choice and records the combined plan in trace output,
   while support-relation loads now use the same planner trace surface
17. for the current DAG family, that same planner currently records the
   coordinated relation-retention plan before the operator-owned DAG state is
   built
18. for the current generic scan family, that same planner currently records
   the coordinated relation-retention plan before scan-heavy consumers build
   list/set views
19. for the current generic closure family, that same planner currently
   records the coordinated relation-retention plan before closure consumers
   build edge or auxiliary indices
20. the operator builds only the retained state it actually needs
21. benchmark code avoids preloading raw facts into in-memory relations first

This is still a first step, not the full endpoint, but it is now broader than
just the original DAG-only fast paths.
