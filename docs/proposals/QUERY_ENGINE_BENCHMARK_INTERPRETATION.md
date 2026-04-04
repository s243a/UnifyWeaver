# Query Engine Benchmark Interpretation

## Purpose

This document explains how the current benchmark results should be read.
The key point is that the query engine already demonstrates one of its major
advantages clearly, while another intended advantage is only partially
realized in the current implementation paths.

## Two Positive Stories

The current benchmark work supports two positive claims about the query
engine:

1. pruning and grouped recursive execution can produce strong wins on
   computationally heavy workloads
2. the engine is intended to gain additional advantages from lazy/streamed
   ownership of retained state

Only the first is fully demonstrated right now.

## What Is Already Demonstrated Well

The strongest demonstrated result so far is this:

- when the workload is computationally heavy and the engine can avoid wasted
  derivations or materialization, the query engine can beat the DFS-style
  baselines substantially

This is visible in workloads like:

- grouped reach-count after direct count aggregation was moved into the runtime
- earlier grouped/prunable recursive benchmarks

So the pruning story is real and already supported by measured results.

## What Is Not Fully Demonstrated Yet

The second intended story is this:

- a lazy/stream-oriented query engine should avoid eager materialization costs
  that scale poorly in more traditional pipelines

Current profiling suggests this story is not yet fully expressed in all
benchmark paths.

The clearest example is `dependency_longest_depth`:

- the solve step is comparatively cheap
- load/build overhead dominates much more of total time
- handwritten and query-engine C# paths both still pay substantial eager
  ingestion/setup costs before the main computation wins can show up

So when the query engine loses on such a benchmark, the interpretation should
be careful.

It does not necessarily mean:

- the engine's computational model is weak

It may instead mean:

- the current benchmark path is still too eager before the engine gets to own
  retention and streaming behavior

## Why The Handwritten C# Baseline Matters

Recent profiling also showed that handwritten `csharp-dfs` is itself slower
than Rust and Go on `dependency_longest_depth`.

That matters because it splits the gap into two layers:

1. C# baseline gap relative to Rust/Go
2. query-engine gap relative to handwritten C#

So not every deficit should be blamed on the query engine specifically.
Some of it belongs to a broader managed-runtime ingestion/setup story.

## Recommended Reading Of Current Results

The clean reading of the benchmark portfolio today is:

- pruning/computational efficiency is already a demonstrated strength of the
  query engine
- lazy/streaming efficiency remains an intended strength whose full benchmark
  expression is still incomplete
- therefore current losses on load-dominated workloads should be read as a
  combination of:
  - C# baseline ingestion/runtime overhead
  - incomplete engine ownership of streaming/retained-state decisions

This is still a positive result, because it tells us where the missing work is
and what kind of work it is.
