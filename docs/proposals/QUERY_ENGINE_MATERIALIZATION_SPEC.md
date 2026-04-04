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

For the current DAG benchmark operators, the streamed path is:

1. provider exposes delimited relation sources
2. engine reads rows directly from those sources
3. engine builds only the graph/group state needed by the DAG operator
4. benchmark code avoids preloading raw facts into in-memory relations first

This is a first step, not the full endpoint.
