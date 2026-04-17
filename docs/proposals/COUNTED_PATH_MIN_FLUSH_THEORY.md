# Counted Path `Min` vs Best-Known Flush/Sort

This note explains an easy-to-misread result in the counted simple-path
measurements for `PathAwareTransitiveClosureNode`:

- end-to-end `Min` can be much faster than `All`
- while `path_state_best_known_flush_sort` can still be a visible cost inside
  the `Min` run

Those statements are not in tension. They describe different levels of the
same execution.

For the separate question of whether that finishing sort is semantically
required on the public runtime path, see
[`COUNTED_PATH_MIN_ORDERING_CONTRACT.md`](./COUNTED_PATH_MIN_ORDERING_CONTRACT.md).

## The Comparison

The measured phases for counted shortest path include:

- `path_state_traversal`
- `path_state_result_materialization`
- `path_state_best_known_flush_sort`

For `All`, the runtime:

1. explores all exact simple-path states up to the depth limit
2. buffers every emitted `(target, depth)` result
3. materializes all of those rows

For `Min`, the runtime:

1. explores exact simple-path states
2. prunes states that cannot improve the best known depth for a target
3. keeps one best depth per target
4. flushes the retained minima by sorting targets and materializing the final
   rows

So `best_known_flush_sort` is not a competitor to `Min`.
It is one terminal phase inside `Min`.

## Why `Min` Still Wins

On the benchmarked counted shortest-path workload, `Min` wins because pruning
collapses the size of the retained result set and reduces the amount of
traversal that survives to output.

That means:

- `All` pays for huge output cardinality
- `Min` pays for much less retained output
- but `Min` still has to finish by sorting and materializing its retained
  minima

The important point is that the flush/sort step is applied to a much smaller
set than `All` would materialize.

## Why Flush/Sort Can Look Large

After the traversal and buffering improvements, the counted-path `Min` path is
no longer dominated by raw search alone. Once pruning has made traversal
cheap enough, the fixed finishing work becomes more visible.

At that point:

- traversal shrinks
- retained minima are few
- but deterministic final ordering still costs something

This is the same pattern you see in many optimized pipelines:
when the main body gets cheaper, the tail becomes a larger share of the total.

That does **not** mean the tail is larger than the old body in absolute terms.
It means it is now prominent inside the optimized run.

## What The 1k Result Means

At `1k`, the counted shortest-path measurements show that:

- `Min` remains globally faster than `All`
- `best_known_flush_sort` is still visible inside `Min`

The implication is that the `Min` implementation has crossed a threshold where
future wins are less about generic path-state machinery and more about the
post-prune finishing path.

In practical terms:

- improving `All` still means reducing traversal and large-output replay costs
- improving `Min` increasingly means reducing final retained-row ordering and
  materialization cost

## Optimization Implications

This comparison argues against adding more generic frontier indexing by
default for counted shortest path.

Why:

- the counted shortest-path bottleneck is not subset-dominance search
- the runtime already gets most of the `Min` win from pruning
- the remaining visible `Min` cost is largely end-stage flush/materialization

So the next `Min`-specific optimization ideas should look like:

1. reduce retained-row finishing overhead
2. keep deterministic ordering only where it is semantically required
3. avoid extra object churn during final projection
4. consider whether sort-elision is valid for a specific consumer contract

The next `All`-specific optimization ideas should look like:

1. reduce traversal-state overhead
2. reduce replay/materialization overhead for large buffered output
3. keep exact simple-path semantics unchanged

## What Not To Infer

Do not read a large `best_known_flush_sort` number as evidence that:

- `Min` is losing to `All`
- the pruning strategy is wrong
- generic frontier indexes are the obvious next step

The correct reading is narrower:

- pruning already worked
- the retained minima are cheap enough that the finishing step is now visible
- the finishing step is therefore the next place to look for `Min`-path wins

## Rule Of Thumb

For counted shortest path:

- if `All` is slow, think traversal and replay volume
- if `Min` is already fast but `best_known_flush_sort` is prominent, think
  finishing-path cost, not search-theory cost

That distinction is the practical takeaway from the current measurements.
