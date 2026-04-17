# Counted Path `Min` Ordering Contract

This note defines the current ordering contract for counted-path `Min`
materialization in the C# query runtime and explains when sort-elision would
or would not be valid.

It complements
[`COUNTED_PATH_MIN_FLUSH_THEORY.md`](./COUNTED_PATH_MIN_FLUSH_THEORY.md),
which explains why `path_state_best_known_flush_sort` can still be prominent
inside a globally faster `Min` run.

## Current Runtime Contract

For counted-path `Min` in `PathAwareTransitiveClosureNode`, the runtime does
not expose retained minima in discovery order.

Instead, after traversal/pruning finishes, it:

1. keeps one best depth per target for the current seed
2. sorts retained targets with `CompareCacheSeedValues`
3. materializes final rows in that deterministic target order

In code, that contract currently lives in the counted-path `Min` flush path:

- `AppendPathAwareRowsForSeed` in
  [`QueryRuntime.cs`](../../src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs)
- `CompareCacheSeedValues` in
  [`QueryRuntime.cs`](../../src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs)

That means the observable order for counted-path `Min` is:

- per-seed
- deterministic
- target-sorted under `CompareCacheSeedValues`

It is **not** just "whatever order traversal happened to discover first".

## Why This Matters

Without the final sort, output order would depend on traversal details such as:

- edge insertion order
- reverse bucket iteration
- stack push/pop shape
- internal dictionary insertion order

Those are implementation details. The current runtime contract hides them and
replaces them with a stable deterministic order.

That stability matters because the runtime test suite frequently checks exact
row sequences, not merely set equality.

## Which Current Checks Depend On It

The benchmark harnesses often normalize row order before hashing, so they do
**not** by themselves require the runtime to preserve deterministic row order.

For example:

- [`benchmark_common.py:221`](../../examples/benchmark/benchmark_common.py#L221)

That is only a benchmark comparison convenience.

The stronger constraint comes from the core C# query runtime tests, which
compare exact row lists. Relevant counted/weighted `Min` examples include:

- `verify_path_aware_transitive_closure_min_mode_plan` in
  [`test_csharp_query_target.pl`](../../tests/core/test_csharp_query_target.pl)
- `verify_path_aware_accumulation_min_mode_plan` in
  [`test_csharp_query_target.pl`](../../tests/core/test_csharp_query_target.pl)

Those tests encode deterministic target order directly in their expected row
lists. In other words, sort-elision would currently be a user-visible behavior
change, not a private optimization.

## What The Benchmark Comparison Does And Does Not Prove

The counted shortest-path benchmark shows that at `1k`, end-to-end `Min`
beats `All` while `path_state_best_known_flush_sort` remains visible.

That measurement proves:

- the finishing step is now a meaningful share of `Min`
- pruning already removed much of the larger traversal/output burden

It does **not** prove:

- that the final sort is unnecessary
- that runtime consumers do not care about order
- that removing sort would be semantics-preserving for the public runtime path

The benchmark hash comparison is order-insensitive because the harness makes it
order-insensitive on purpose.

## When Sort-Elision Would Be Valid

Sort-elision or delayed ordering is valid only when the consumer contract is
explicitly unordered.

Examples of potentially valid cases:

- an internal benchmark-only path that normalizes rows before comparison
- an internal aggregation stage that consumes minima as a set/map
- a future explicit `unordered` execution mode with tests and docs to match

In those cases, the optimization target is a different contract, not the
current one.

## When Sort-Elision Is Not Valid

Sort-elision is not valid for the current public counted-path `Min`
materialization path when:

- output is returned as ordinary runtime rows
- exact-row-order tests are in scope
- deterministic replay across runs is expected

Under the current contract, removing the sort would change observable
behavior.

## Practical Implication

If we want to optimize `path_state_best_known_flush_sort`, the first question
is not "can we skip ordering?"

The first question is:

"Which execution paths are actually allowed to be unordered?"

Until that contract changes, the right optimization direction is:

1. keep deterministic ordering
2. make the retained-minima flush/sort/materialization path cheaper
3. only introduce sort-elision behind an explicitly narrower contract
