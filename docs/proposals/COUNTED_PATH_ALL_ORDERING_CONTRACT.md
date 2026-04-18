# Counted Path `All` Ordering Contract

This note defines the current ordering contract for counted-path `All`
materialization in the C# query runtime.

It complements
[`COUNTED_PATH_MIN_ORDERING_CONTRACT.md`](./COUNTED_PATH_MIN_ORDERING_CONTRACT.md),
which covers the separate retained-min flush path.

## Current Runtime Contract

For counted-path `All` in `PathAwareTransitiveClosureNode`, the runtime does
not sort rows by target before materialization.

Instead, it currently emits rows in two layers of order:

1. seeds are processed in deterministic seed-sorted order via
   `CompareCacheSeedValues`
2. rows for each seed are replayed in the same order they were buffered during
   traversal

That means counted-path `All` is currently:

- deterministic across runs for the same edge insertion order
- grouped by sorted seed
- ordered within each seed by traversal/discovery order, not target sort
- multiplicity-preserving when multiple distinct paths reach the same target

In code, that contract currently lives in:

- `ExecutePathAwareTransitiveClosure`
- `ExecuteSeededPathAwareTransitiveClosure`
- `AppendPathAwareRowsForSeed`
- `MaterializePathAwareDepthRows`

all in
[`QueryRuntime.cs`](../../src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs).

## Why This Matters

For counted-path `All`, replay/materialization optimizations are only valid if
they preserve the current row sequence, or if the consumer contract is
explicitly relaxed.

Unlike counted-path `Min`, the `All` path does not currently pay for a final
target sort. Its observable order is tied to traversal details such as:

- reverse bucket iteration
- stack push/pop behavior
- when rows are buffered relative to child-state pushes

Those details are implementation choices, but they are still part of the
current public runtime behavior because exact-row-order tests can observe them.

## Which Current Checks Depend On It

The benchmark harness commonly normalizes output order before hashing, so the
benchmarks alone do not define the public order contract.

The stronger constraint comes from the core runtime tests, which compare exact
row lists. In particular:

- `verify_path_aware_transitive_closure_plan`
- `verify_path_aware_transitive_closure_all_mode_order_plan`
- `verify_path_aware_transitive_closure_min_mode_plan`

in
[`test_csharp_query_target.pl`](../../tests/core/test_csharp_query_target.pl)

make row order observable.

## Practical Implication

If we want to optimize counted-path `All` replay/materialization further, the
first question is not "can we reorder rows more cheaply?"

The first question is:

"Which execution paths are actually allowed to change counted-path `All`
output order?"

Until that contract changes, safe optimization means:

1. keep per-seed traversal/discovery order intact
2. keep sorted-seed grouping intact
3. make buffering/replay cheaper without introducing a different row sequence
