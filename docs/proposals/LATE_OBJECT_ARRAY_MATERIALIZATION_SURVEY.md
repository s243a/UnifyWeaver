# Late `object[]` Materialization Survey

This note surveys where the C# query runtime currently commits to `object[]`
row materialization and what would need to change to keep rows in a narrower
internal representation longer.

It follows from:

- [`COUNTED_PATH_ALL_ROW_ALLOCATION_CONTRACT.md`](./COUNTED_PATH_ALL_ROW_ALLOCATION_CONTRACT.md)
- [`COUNTED_PATH_ALL_ORDERING_CONTRACT.md`](./COUNTED_PATH_ALL_ORDERING_CONTRACT.md)

## Current Boundary Map

The runtime currently commits to `object[]` at multiple levels, not just in
counted-path replay:

### Execution Interfaces

- `IRelationProvider.GetFacts(PredicateId)` returns `IEnumerable<object[]>`
- `IReplayableRelationSource.Stream()` returns `IEnumerable<object[]>`
- `IReplayableRelationSource.Materialize()` returns `List<object[]>`

### Runtime Storage And Caches

- `ReplayableRelationSource` buffers `List<object[]>`
- `EvaluationContext` stores facts, totals, deltas, and materialized nodes as
  `List<object[]>`
- closure and grouped-closure memoized results use `IReadOnlyList<object[]>`

### Equality And Indexing

- `RowWrapper` wraps `object[]`
- `StructuralArrayComparer` compares `object[]`
- fact/join indexes store `List<object[]>`

### Tests And Harnesses

- the Prolog/C# runtime harnesses expect row outputs as `object[]`
- exact-row-order tests operate on fully materialized row sequences

## What This Means

There is no single counted-path-local switch that moves the runtime off
`object[]`.

A later materialization strategy is only plausible if it introduces a narrower
internal representation at one of these broader boundary layers:

1. replay buffers before `IEnumerable<object[]>` exposure
2. replayable/materialized relation sources
3. cached closure result containers
4. indexing/wrapper utilities that currently assume `object[]`

## Plausible Escape Hatches

The most plausible directions are:

### Option 1: Narrow Internal Replay Buffers With Late Conversion

Keep counted-path `All` rows in a specialized internal shape longer, but only
if the conversion to `object[]` can happen after:

- replay batching
- cache admission decisions
- any internal consumers that do not require `object[]`

This helps only if the runtime can avoid immediate conversion back to
`object[]` at the very next boundary.

### Option 2: Typed Replayable Source Layer

Generalize `IReplayableRelationSource` or add a sibling abstraction that can
hold typed rows internally and only expose `object[]` at the outer execution
boundary.

This is broader, but it aligns with where the current boundary actually lives.

### Option 3: Cache/Index Layer Generalization

Introduce a row-shape abstraction under:

- `RowWrapper`
- structural equality/hashing
- fact/join index storage

This is the most invasive option, but it is also the first one that would make
late `object[]` conversion a real runtime capability rather than a local trick.

## Non-Goals

This survey does **not** propose changing:

- counted-path `All` row order
- counted-path `Min` row order
- user-visible output schema

It only identifies where a broader late-materialization design would have to
land to matter.

## Practical Implication

The next serious optimization step is not another counted-path-local allocator
experiment.

The next serious step is a broader design/prototype around one of these
boundary layers, most likely:

1. replayable source abstraction
2. cached result container abstraction
3. row wrapper/index abstraction

Without that broader move, counted-path `All` remains boxed in by the current
`object[]` runtime boundary.
