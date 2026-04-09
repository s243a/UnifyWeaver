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
inside benchmark-specific wiring.

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
- path-aware shortest-path operators can build compact source->targets edge state
  instead of retaining generic edge tuples
- where direct streaming, replayable buffering, and external materialized rows
  are all viable sources for that edge state, the runtime can choose among
  them through a measured retention selector and still expose an explicit
  override via `QueryExecutorOptions.PathAwareEdgeRetentionStrategy`
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
5. path-aware shortest-path operators can build a compact edge-state cache
   directly from streamed facts instead of generic replayed edge rows
6. where streaming, replayable buffering, and external materialized rows are
   all viable sources for that cache, the runtime can use measured edge-retention
   buckets and bounded probes to choose among them before honoring an explicit
   executor override
7. shortest-path and weighted-shortest-path operators can emit compact
   grouped root minima directly from streamed edge/seed inputs
8. where grouped minima and legacy seeded-row regrouping both exist, the runtime
   can select between them through a shared grouped-summary policy layer,
   record measured cost buckets for that decision, and use bounded probes in
   ambiguous cases before falling back to an explicit executor option
9. effective-distance and category-influence operators can emit compact
   grouped root-weight summaries directly from streamed edge/seed inputs
10. where grouped weight sums and legacy seeded-row regrouping both exist, the
   runtime can select between them through that same grouped-summary policy
   layer, record measured cost buckets for that decision, and use bounded
   probes in ambiguous cases before falling back to an explicit executor option
11. the operator builds only the retained state it actually needs
12. benchmark code avoids preloading raw facts into in-memory relations first

This is still a first step, not the full endpoint, but it is now broader than
just the original DAG-only fast paths.
