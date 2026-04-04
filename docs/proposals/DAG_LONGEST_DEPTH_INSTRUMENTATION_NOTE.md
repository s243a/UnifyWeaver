# DAG Longest-Depth Instrumentation Note

This note records the first phase-timing pass for the C# query-engine
`dependency_longest_depth` benchmark.

The purpose of this pass was not to change semantics or claim a new
optimization. It was to answer a narrower question:

- where is the current longest-depth query time actually going?

## Setup

Instrumentation was added to the `SeedGroupedDagLongestDepthNode` runtime
path and the generated `csharp_query` benchmark program was updated to
print per-phase timings to `stderr`.

The measured phases were:

1. `build_graph`
2. `seed_grouping`
3. `reachable_cone`
4. `topological_order`
5. `suffix_depth_dp`
6. `group_reduction`

## 10k Measurement

For the synthetic `10k` dependency benchmark, a direct run of the
generated `csharp_query` longest-depth program produced:

- `load_ms=38`
- `query_ms=27`
- `aggregation_ms=0`
- `total_ms=65`
- `seed_count=500`
- `tuple_count=500`
- `project_count=500`

And the query-phase breakdown was:

| Phase | Time |
|---|---:|
| `build_graph` | `12.591 ms` |
| `seed_grouping` | `0.514 ms` |
| `reachable_cone` | `0.282 ms` |
| `topological_order` | `0.230 ms` |
| `suffix_depth_dp` | `0.113 ms` |
| `group_reduction` | `0.046 ms` |

## Main Finding

The dominant bucket is:

- **graph building**

Not:

- reachable-cone restriction
- topological ordering
- the scalar suffix-depth DP itself
- the final grouped reduction

That matters because the recent optimization guesses had focused more on
execution structure than on graph construction cost.

The measurement says the opposite:

- the suffix-depth algorithm is already cheap
- the runtime is spending much more time turning tuples into an in-memory
  graph than it is solving the DAG longest-depth problem

## Interpretation

For this benchmark family, the current longest-depth performance gap is
now best explained as:

- **graph construction overhead**
  plus
- residual generic runtime cost

rather than a weak depth-DP algorithm.

This also explains why some earlier ideas did not pay off:

- global/full-graph variants did not help because they did not attack the
  graph-build cost enough
- transitive reduction did not help because the DP is already cheap
- prefix/suffix decomposition is conceptually correct, but does not by
  itself solve the dominant cost center

## Recommended Next Step

The next longest-depth optimization pass should focus on:

1. reducing `build_graph` cost
2. not on further DAG-math changes first

Promising directions include:

- pre-sizing and allocation reduction in graph construction
- reducing object/tuple decoding overhead from fact rows
- reusing a more direct graph representation when the benchmark shape
  permits it

Less promising immediate directions are:

- more cone-restriction tuning
- more topological-order tuning
- more suffix-depth recurrence changes

## Relationship To Other Notes

This note complements:

- `DAG_QUERY_EXECUTION_THEORY.md`
- `DAG_RUNTIME_COST_EXPERIMENTS_NOTE.md`

Those notes explain:

- the DAG/non-DAG conceptual split
- prior runtime-cost experiments

This note adds the first phase-level evidence for the remaining
longest-depth bottleneck.
