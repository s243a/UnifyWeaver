# SCC-Condensed Weighted Min: Specification

## Scope

This specification defines a candidate fast path for weighted `min`
evaluation of `PathAwareAccumulationNode` workloads in the C# query
engine.

It is intended for recursive accumulation shapes that already match the
current path-aware accumulation lowering:

```prolog
pred(X, Y, Acc) :-
    edge(X, Y),
    aux(X, V),
    Acc is BaseExpr(V).

pred(X, Z, Acc) :-
    edge(X, Y),
    aux(X, V),
    pred(Y, Z, Acc1),
    Acc is StepExpr(Acc1, V).
```

with:

- `TableMode.Min`
- per-path uniqueness
- monotone non-negative additive accumulation for the current fast path

## Supported Semantic Class

The fast path is only sound when all of the following hold:

1. The recursive accumulator is monotone increasing along a path.
   Examples:
   - `Acc is Acc1 + Cost`
   - `Acc is Acc1 + log(Deg) / log(N)` with positive step
   - zero-cost additive steps when the query is depth-bounded

2. The result contract is minimum accumulated cost per `(source,target)`.

3. Paths are simple:
   - no repeated nodes in a path

4. The optimizer may transform the graph via SCC condensation, but may
   not change the final result set.

Still on the exact frontier fallback path:

- negative increments
- non-additive recurrence expressions such as `Acc is Acc1 * Factor`
- `max`, `first`, `sum`, `count`
- multi-auxiliary or non-linear accumulation beyond the current path-aware
  accumulation shape

## High-Level Execution Model

### Input

- edge relation `edge(U, V)`
- auxiliary relation `aux(U, Value)`
- source seeds
- weighted `min` accumulation expressions

### Output

For each seeded source `S` and reachable target `T`:

- exactly one row `(S, T, MinAcc)`
- where `MinAcc` equals the minimum accumulated cost over all valid
  simple paths from `S` to `T`

### Intermediate Graph Model

The runtime may construct:

1. SCC partition of the edge graph
2. condensation DAG over SCC ids
3. local entry/exit transfer structure for each SCC

The runtime may then evaluate:

- local exact weighted path costs inside SCCs
- global `min` composition across the condensation DAG

## Required Correctness Properties

1. Exact agreement with current `TableMode.All` weighted results after
   downstream `min` aggregation.

2. No path may be accepted if it repeats a node.

3. SCC condensation must not merge semantically distinct boundary costs.

4. The fast path may use summaries, but only if they are lossless for the
   supported workload class.

## Runtime Shape

Preferred runtime structure:

- keep `PathAwareAccumulationNode`
- add an internal execution strategy, not a new user-facing Prolog syntax

Possible C# runtime strategies:

- `PathAwareAccumulation-Min-Frontier` (current exact fallback)
- `PathAwareAccumulation-Min-SccCondensed` (new fast path)

The planner does not need a new source-level feature to select this.
Selection may be runtime-internal as long as semantics are preserved.

## Planner Contract

No new Prolog syntax is required for the initial version.

The planner already provides:

- `path_aware_accumulation`
- `table_modes:[..., ..., min]`
- base and recursive arithmetic expressions

The runtime may inspect the existing node and decide whether the SCC
fast path is applicable.

## Applicability Test

The runtime fast path should only activate when:

- `AccumulatorMode == TableMode.Min`
- the accumulation expression is monotone additive in practice
- graph condensation is available
- the bounded SCC-condensed probe beats the existing layered positive-min
  path by a margin

The runtime should fall back to the current exact frontier algorithm
when:

- applicability cannot be proven
- the additive step is negative for any reachable edge/auxiliary row
- the recursive expression is non-additive
- SCC preprocessing fails
- internal transfer summarization would be lossy
- measured SCC overhead dominates the existing positive-min path

## Instrumentation Requirements

The implementation must report enough data to evaluate whether the new
strategy is worth keeping.

At minimum:

- number of SCCs
- largest SCC size
- condensation DAG edge count
- number of exact local states explored
- number of outer DAG states explored
- SCC condensation, SCC probe, and SCC solve phase timings
- exact frontier fallback candidate count
- exact frontier fallback dominance and subset-check counts
- exact frontier fallback target-bucket count plus path-state partition count,
  total retained states, maximum partition size, and average partition size
- final output row count
- total runtime

These should be available to benchmark harnesses, at least through
stderr metrics.

## Benchmark Contract

Primary benchmark:

- `examples/benchmark/benchmark_weighted_shortest_path.py`

Success criteria:

1. `all_vs_min = match`
2. `min` becomes faster than `all`
3. the gain remains visible at `5k` and `10k`

Target threshold:

- weighted `min` should ideally approach at least `~2x` faster than
  `all` on large-scale workloads

That threshold is aspirational rather than a hard semantic requirement,
but it is the practical bar for considering the fast path successful.
