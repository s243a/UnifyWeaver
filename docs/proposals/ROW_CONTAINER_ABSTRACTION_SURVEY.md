# Row Container Abstraction Survey

This note surveys the smallest plausible abstraction seam for delaying
`object[]` row materialization in the C# query runtime.

It follows from:

- [`LATE_OBJECT_ARRAY_MATERIALIZATION_SURVEY.md`](./LATE_OBJECT_ARRAY_MATERIALIZATION_SURVEY.md)
- [`COUNTED_PATH_ALL_ROW_ALLOCATION_CONTRACT.md`](./COUNTED_PATH_ALL_ROW_ALLOCATION_CONTRACT.md)

## Problem Statement

Counted-path `All` replay surveys show that row allocation is the largest
write-side cost. However, the runtime currently exposes and stores rows as
`object[]` across execution, replayable sources, caches, wrappers, and indexes.

That means a counted-path-local row shape is not enough. The runtime needs an
abstraction seam where a narrower row container can survive long enough to
avoid immediate conversion back to `object[]`.

## Candidate Seams

### Seam 1: Replayable Source Layer

Add a sibling abstraction to `IReplayableRelationSource` that can store typed
rows internally and expose `object[]` only when a consumer requires the public
row shape.

Pros:

- narrower than changing every execution node
- aligns with existing replay/materialization behavior
- a good fit for counted-path replay output

Cons:

- consumers still need an `object[]` view
- fact and join indexes still force conversion unless they are taught the new
  row shape too

### Seam 2: Cached Result Container Layer

Introduce a cache container that can hold either `object[]` rows or a typed
row representation, with explicit conversion at `IEnumerable<object[]>`
boundaries.

Pros:

- directly targets closure and grouped-closure result caches
- avoids changing the top-level public execution API first
- provides a migration path for hot internal producers

Cons:

- every cache consumer must be audited for whether it needs arrays or only
  structural access
- indexing still needs an abstraction if cached rows are probed by key

### Seam 3: Row Wrapper And Index Layer

Generalize `RowWrapper`, structural equality, fact indexes, and join indexes
to accept a row-view interface rather than only `object[]`.

Pros:

- strongest foundation for late materialization
- lets typed rows participate in indexing without immediate conversion

Cons:

- largest surface area
- highest risk of behavioral regressions
- likely requires changes across many plan nodes, not just counted paths

## Recommended Prototype Path

The smallest useful prototype is the cached result container layer.

Rationale:

- replayable sources are too close to the public `IEnumerable<object[]>`
  boundary by themselves
- row wrapper/index generalization is too broad for the first prototype
- cached closure result containers are broad enough to matter but narrow
  enough to audit

The first prototype should introduce a small internal container abstraction
that can:

1. expose an `IReadOnlyList<object[]>` view for existing consumers
2. optionally retain a specialized internal row representation
3. make conversion points explicit and measurable
4. preserve exact row order and row equality behavior

## Prototype Status

The first prototype seam is `CachedResultRows`.

It is intentionally narrow:

- `TransitiveClosureResults` stores `CachedResultRows`
- seeded transitive-closure source/target caches, including grouped variants,
  store `CachedResultRows`
- counted-path replay can construct `CachedResultRows` from compact
  path-aware target/depth buffers
- ungrouped seeded source/target closure caches can store two-column results
  as parallel value buffers and materialize `object[]` rows on cache hits
- existing consumers still call `AsObjectRows()` and receive
  `IReadOnlyList<object[]>`
- public execution APIs and row order are unchanged

This proves the cached-result-container boundary can be inserted without a
full row-wrapper/index rewrite. The compact counted-path replay prototype
keeps final `object[]` row materialization at the boundary, but avoids the
previous intermediate target-value and boxed-depth arrays. The seeded closure
cache prototype extends the same seam into cache storage for the simple
two-column shape while leaving grouped and wider row shapes on object rows.
This trades retained cache size and row-array isolation against hit-path
latency, because compact cache hits must allocate fresh public `object[]` rows
instead of reusing cached row arrays. The next step is to add retained-memory
measurement before moving more cache families onto specialized buffers.

## Non-Goals

The first abstraction should not change:

- the public `QueryExecutor.Execute(...)` row type
- external relation provider APIs
- counted-path `All` ordering
- counted-path `Min` ordering

Those can be revisited only after the internal conversion boundary is proven.

## Open Questions

- Which existing closure caches are hot enough to justify container migration
  first?
- Can key probes operate over a typed row view without constructing full
  arrays?
- Should the first container be generic, or should it start as a counted-path
  target/depth container and generalize later?
- How should trace metrics report delayed conversion cost versus true row
  write cost?

## Practical Implication

The next implementation step should not attempt a full row-interface rewrite.

It should prototype a cached result container boundary for one measured hot
path, then use metrics to decide whether row wrapper/index generalization is
worth the larger surface-area cost.
