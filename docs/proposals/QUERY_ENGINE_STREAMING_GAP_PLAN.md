# Query Engine Streaming Gap Plan

## Purpose

This document turns the streaming vision and recent benchmark interpretation
into a concrete implementation agenda.

The central problem is:

- the query engine is intended to own retained state
- but some current benchmark paths still build too much retained structure
  before the engine gets to make that decision

## Problem Statement

The current longest-depth/load profiling suggests that the main missing piece
is not another DAG recurrence idea.

Instead, the gap is that tuple ingestion and retained-state ownership are not
yet aligned with the engine's intended model.

In other words:

- too much state is still created by benchmark-side or program-side ingestion
- too little of that state selection is owned by the query runtime itself

## Near-Term Goals

### 1. Preserve What Already Works

Do not regress the query-engine paths that already demonstrate strong wins on
computationally heavy workloads.

In particular:

- keep the grouped reach-count runtime gains
- keep the DAG longest-depth runtime specialization
- keep using benchmark profiling to distinguish compute cost from ingestion cost

### 2. Move Toward Streamed Tuple Ingestion

The parser layer should aim to provide streamed decoded tuples with minimal
retained state.

The query engine should increasingly decide whether to:

- retain nothing
- retain summaries
- retain indexes
- retain a full relation representation when truly needed

### 3. Let Operators Own Their State

Operators like these should explicitly own their retained state strategy:

- grouped DAG reachability
- DAG longest depth
- recursive grouped aggregation

That means the engine should become the component that says:

- "this operator needs adjacency"
- "this operator needs grouped bitsets"
- "this operator only needs counts"

rather than inheriting a pre-built structure from a generic ingestion layer.

## Concrete Stages

### Stage 1: Keep Profiling Boundaries Clear

Continue measuring these separately whenever possible:

- load/ingestion
- graph/state build
- solve
- emit

This is necessary to distinguish:

- baseline runtime issues
- query-engine issues
- benchmark-wrapper issues

### Stage 2: Shrink Parser Responsibility

Move the parser toward:

- row decoding
- tuple streaming
- minimal canonicalization only when clearly beneficial

Avoid turning the parser into a second planner or index builder.

### Stage 3: Add Engine-Owned Retention Paths

Add or strengthen runtime paths where the engine consumes tuple streams and
constructs only the state required by the operator.

Priority examples:

- streamed edge ingestion for DAG operators
- streamed grouped seed ingestion
- count/summarize directly rather than materializing relations first

### Stage 4: Re-evaluate Streaming Benchmarks

Once the engine owns more of the retained-state choice, revisit benchmarks
like `dependency_longest_depth` and ask:

- does the lazy/streaming advantage now appear more clearly?
- how much of the remaining gap is still just baseline C# cost?

## What Not To Do

### Do not overfit the parser

A highly specialized benchmark parser may improve one benchmark while moving
architectural responsibility in the wrong direction.

### Do not treat every loss as a query-engine failure

Some benchmark losses are still baseline C# ingestion/runtime losses.
Those should be recognized separately.

### Do not assume graph-theory improvements are the next step by default

On several recent longest-depth iterations, the missing win was not a better
recurrence but lower-level ingestion/runtime ownership.

## Success Criteria

This gap is meaningfully closed when:

- benchmark ingestion paths are thinner
- the engine owns more retained-state decisions directly
- load-dominated workloads stop masking the engine's intended lazy/streaming
  advantages as much as they do today
- pruning and streaming are both visible as distinct strengths of the query
  engine in the benchmark suite
