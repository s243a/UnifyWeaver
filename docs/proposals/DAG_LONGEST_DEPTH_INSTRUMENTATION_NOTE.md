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

## Follow-Up Experiment: String-Specialized Row Decoding

After the instrumentation pass, a targeted experiment tried to reduce
`build_graph` cost by adding a string-specialized fast path for
`SeedGroupedDagLongestDepthNode`.

The idea was:

- the synthetic dependency benchmark uses string-valued edge rows
- the current runtime path builds the graph from generic `object[]` rows
- perhaps avoiding generic `object` lookups/casts would materially reduce
  graph-build overhead

The experiment was intentionally narrow:

- keep the existing DAG recurrence unchanged
- keep the same phase structure
- only specialize graph construction and seed grouping when all relevant
  values are strings

### Result

The specialized path was:

- correct
- benchmark-compatible
- but slower end to end

Full benchmark timings regressed to approximately:

| Scale | C# Query |
|---|---:|
| `300` | `0.071s` |
| `1k` | `0.073s` |
| `5k` | `0.099s` |
| `10k` | `0.114s` |

Those numbers were worse than the current `main` baseline, so the change
was reverted.

### Interpretation

This is useful because it narrows the bottleneck further.

It suggests that the remaining graph-build cost is **not** solved by a
simple “string instead of object” specialization layered on top of the
current implementation.

In practice, that fast path added enough extra checking and duplicated
logic that it lost any benefit from typed dictionary access.

So the next promising target is more likely:

- lower-level fact-row access overhead
- relation-provider / executor overhead around graph construction
- or a more direct graph ingestion path

and less likely:

- another shallow specialization of the same graph-build loop


## Follow-Up Experiment: Allocation And GC Comparison

A direct `10k` comparison was also run between the generated:

- `csharp-dfs` benchmark program
- `csharp-query` benchmark program

with both programs instrumented to report:

- total wall-clock time
- `GC.CollectionCount(0/1/2)` deltas
- total allocated bytes

### 10k Comparison

#### C# DFS

- `total_ms=35`
- `gc0_collections=0`
- `gc1_collections=0`
- `gc2_collections=0`
- `allocated_bytes=7,880,720`
- `project_count=500`

#### C# Query

- `load_ms=39`
- `query_ms=22`
- `aggregation_ms=0`
- `total_ms=62`
- `gc0_collections=0`
- `gc1_collections=0`
- `gc2_collections=0`
- `allocated_bytes=8,952,104`
- `seed_count=500`
- `tuple_count=500`
- `project_count=500`

### Main Finding

This is strong evidence that the remaining longest-depth gap is **not**
currently being driven by garbage collection pauses.

At `10k`:

- both C# implementations completed with `0` GC collections
- the query path does allocate somewhat more than the DFS path
- but the bigger difference is still CPU/runtime overhead rather than GC
  interruption

So the current diagnosis becomes more precise:

- **not**: "the query engine is slow because GC is firing repeatedly"
- more likely: "the query engine still spends extra CPU time in managed
  runtime/framework overhead, including graph ingestion and generic query
  machinery"

### Interpretation

This matters because it narrows the next optimization track again.

If GC were the dominant issue, the next work would focus on:

- reducing collection frequency
- reducing temporary object lifetimes
- or reusing large buffers to avoid triggering collections

But with `0` collections in both programs at `10k`, the more promising
next targets are:

- CPU cost of graph ingestion
- generic tuple/object handling overhead
- query-framework overhead that remains even after the dedicated DAG node
  work

That does **not** mean allocations are irrelevant. The query path still
allocates more bytes than the DFS path, so lowering allocation pressure may
still help CPU/cache behavior. But the evidence so far does not support GC
itself as the main bottleneck.

## Questions Worth Handing To External Research

If we want to ask Perplexity or another research assistant for ideas, the
useful questions are now fairly concrete:

1. For a DAG longest-path query over string-labeled edges, what exact
   engineering techniques reduce graph-build overhead more reliably than
   typed dictionary specialization?
2. In query engines that ingest generic tuple rows, where do successful
   DAG implementations usually win:
   - row decoding
   - symbol interning
   - adjacency construction
   - or result materialization?
3. What are good exact designs for building a temporary DAG view from a
   generic relation provider with minimal allocation?
4. When the DP itself is cheap, what benchmark patterns typically expose
   tuple/object overhead as the dominant cost in graph workloads?

## Relationship To Other Notes

This note complements:

- `DAG_QUERY_EXECUTION_THEORY.md`
- `DAG_RUNTIME_COST_EXPERIMENTS_NOTE.md`

Those notes explain:

- the DAG/non-DAG conceptual split
- prior runtime-cost experiments

This note adds the first phase-level evidence for the remaining
longest-depth bottleneck.
