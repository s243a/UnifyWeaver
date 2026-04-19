# Counted Path `All` Row Allocation Contract

This note defines the current row-allocation contract for counted-path `All`
materialization in the C# query runtime.

It complements
[`COUNTED_PATH_ALL_ORDERING_CONTRACT.md`](./COUNTED_PATH_ALL_ORDERING_CONTRACT.md),
which explains the current observable row ordering.

## Current Runtime Contract

Today, counted-path `All` rows are materialized as fresh `object[]` instances
at the replay boundary in `PathAwareTransitiveClosureNode`.

That is not an isolated implementation detail. It fits into a broader runtime
shape where query rows are represented pervasively as `object[]`:

- plan execution surfaces return `IEnumerable<object[]>`
- replay/materialization surfaces use `List<object[]>`
- caches and retained result maps store `IReadOnlyList<object[]>`
- set/index helpers such as `RowWrapper` and fact/join indexes are defined in
  terms of `object[]`
- the C# runtime test harnesses print and compare rows through `object[]`

In code, the relevant boundaries include:

- `IReplayableRelationSource`
- `ReplayableRelationSource`
- `EvaluationContext` result caches and materialized stores
- `RowWrapper`
- `MaterializePathAwareDepthRows`

all in
[`QueryRuntime.cs`](../../src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs),
plus the test harness entry points in
[`test_csharp_query_target.pl`](../../tests/core/test_csharp_query_target.pl).

## Why This Matters

The recent counted-path `All` replay survey showed that row allocation is the
largest write-side cost on the current benchmark shape.

That naturally raises the question:

"Can counted-path `All` avoid allocating a fresh `object[]` per row?"

Under the current runtime structure, the answer is effectively "not locally."

The materialization site is only the last step in a runtime that already
expects rows to be `object[]` across:

- execution interfaces
- replayable sources
- memoized result caches
- structural row wrappers
- fact and join indexing
- test harness consumption

That means a narrower counted-path-only row-shape substitution would still
have to cross back into `object[]` immediately at the surrounding runtime
boundary, unless the broader row representation contract changes too.

## What This Does And Does Not Rule Out

This contract does **not** mean counted-path `All` can never get faster.

It does mean:

- allocator swaps at the final `object[]` construction site are likely to be
  low-signal unless they materially change allocation behavior
- the runtime is near the end of easy counted-path-only row-shape wins that do
  not affect broader row representation
- larger gains probably require either fewer rows, later conversion to
  `object[]`, or a broader runtime/container contract change

## Practical Implication

If we want a larger counted-path `All` replay win from row shaping, the next
question is not:

"Which tiny `object[]` allocation variant should we try next?"

The next question is:

"Is there a broader runtime boundary where rows can remain in a narrower
internal representation longer before conversion to `object[]`?"

Until that broader contract changes, fresh `object[]` row materialization is a
real current runtime boundary, not just a local counted-path implementation
choice.
