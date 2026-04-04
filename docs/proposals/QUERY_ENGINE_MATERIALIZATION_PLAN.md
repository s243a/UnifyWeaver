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

Work:

- identify additional operators that can ingest directly from streamed sources
- move retained-state construction into those operators where it pays off
- avoid generic `object[]` fact preloading when a narrower retained form is
  sufficient

Success criteria:

- more benchmark/program paths use engine-owned retention
- reduced parser-side or harness-side eager structure building

## Stage 4: Cost-Based Strategy Selection

Work:

- use measured cost buckets to guide whether the engine prefers:
  - direct streaming
  - replayable buffering
  - indexed retained state
  - fallback external materialization
- keep this heuristic-driven at first, then refine as profiling improves

Success criteria:

- the materialization strategy is chosen where operator knowledge exists
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
