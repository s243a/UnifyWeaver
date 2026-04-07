# Query Engine Streaming Vision

## Purpose

This document clarifies an intended property of the query engine that has
been easy to lose sight of while chasing benchmark performance:

- the query engine is meant to own retained state
- the parser/ingestion layer is meant to stream tuples into the engine
- eager pre-materialization outside the engine is a compromise, not the ideal

This is not a new design direction. It is a restatement of the engine's
existing architectural intent, sharpened by recent profiling work.

## Core Principle

The parser should do as little as possible beyond decoding rows into tuples.

The query engine should decide:

- what state to retain
- when to index or group
- when full materialization is actually required
- when a plan can stay streaming or partially streaming

So the architectural split should be:

- parser responsibility: stream decoded tuples
- query-engine responsibility: build only the retained state required by the operator plan

## Why This Matters

A parser that eagerly constructs heavyweight data structures is making a
query-planning decision too early.

That has several costs:

- peak memory goes up before the engine can prune or aggregate
- the runtime pays allocation and ingestion overhead even for queries that do
  not need all rows retained
- benchmark results become dominated by pre-engine setup rather than by query
  execution properties

This is especially harmful when the query engine is supposed to be compared
against hand-written pipelines partly on the basis of lazy execution.

## What Profiling Has Already Shown

Recent benchmark work has already shown one major strength clearly:

- the query engine wins strongly on computationally heavy, prunable workloads

This is already visible in the grouped recursive benchmarks where pruning and
avoiding wasted derivations matter more than raw ingestion speed.

But the profiling has also shown an important gap:

- the query engine's intended streaming/lazy advantage is not yet fully
  expressed in every benchmark path

The clearest example is `dependency_longest_depth`:

- the actual DAG dynamic programming is cheap
- load/build overhead dominates much more of the total time
- too much eager structure-building still happens before the engine fully owns
  the retained state story

## Architectural Position

The right goal is not "no data structures".

The right goal is:

- no retained state earlier than necessary
- no retained state outside the engine unless the engine explicitly requires it
- retained state chosen by the operator plan, not by the parser

That means some operators will still build substantial state.
For example:

- grouped reachability on DAGs
- longest-depth DAG summaries
- recursive aggregation indexes

But the engine should build those intentionally, from streamed tuples, rather
than inheriting them from an eager pre-processing layer by default.

## Practical Implication

When a benchmark currently performs badly because of load/setup cost, that
should not always be interpreted as an algorithmic weakness of the query
engine.

Sometimes it is revealing something narrower:

- the benchmark path is not yet allowing the engine to exercise its intended
  streaming ownership of state

That is a real implementation gap, not a refutation of the streaming vision.
