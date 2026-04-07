# DAG Runtime-Cost Experiments Note

This note records the recent **runtime-cost** experiments on the C#
query-engine DAG paths after the successful introduction of:

1. project-grouped DAG reachability
2. DAG longest-depth execution

The goal of these experiments was narrower than the earlier theory work:

- keep the DAG execution model the same
- reduce runtime overhead
- avoid mixing in non-DAG path-state concerns

## Current Baseline

The current committed baseline on `main` has two important C# query DAG
paths:

1. `SeedGroupedTransitiveClosureNode`
   - used by the synthetic dependency **reach-count** benchmark
2. `SeedGroupedDagLongestDepthNode`
   - used by the synthetic dependency **longest-depth** benchmark

At the time of these experiments, the relevant baseline results were:

### Dependency Reach

| Scale | C# Query | C# DFS | Rust DFS | Go DFS |
|---|---:|---:|---:|---:|
| 300 | 0.060s | 0.045s | 0.002s | 0.003s |
| 1k | 0.097s | 0.051s | 0.007s | 0.009s |
| 5k | 0.199s | 0.167s | 0.130s | 0.106s |
| 10k | 0.485s | 0.511s | 0.572s | 0.457s |

### Dependency Longest Depth

| Scale | C# Query | C# DFS | Rust DFS | Go DFS |
|---|---:|---:|---:|---:|
| 300 | 0.063s | 0.041s | 0.002s | 0.002s |
| 1k | 0.060s | 0.044s | 0.003s | 0.003s |
| 5k | 0.098s | 0.051s | 0.006s | 0.006s |
| 10k | 0.104s | 0.057s | 0.011s | 0.011s |

These are the numbers to compare later experiments against.

## Experiment 1: Shared DAG Preprocessing

### Idea

Both DAG-specialized runtime nodes were independently rebuilding similar
structures:

- node ids
- local reachable cone
- adjacency
- topological order

The hypothesis was that introducing a shared cached DAG analysis would:

- reduce duplicate preprocessing
- improve both reach-count and longest-depth

### Outcome

The refactor was structurally clean but did **not** produce a clear
performance win:

- dependency reach was roughly neutral, maybe slightly better at large
  scale
- dependency longest depth was neutral to slightly worse

### Conclusion

This was not worth landing as a performance change.

The likely reason is that the two benchmark paths do not reuse the DAG
analysis enough within a single benchmark invocation to amortize the
extra caching/plumbing overhead.

This may still become valuable later if the DAG runtime grows into a
more general shared compiler-emitted substrate, but it was not the right
next optimization for the current benchmark paths.

## Experiment 2: Local Transitive Reduction

### Idea

After building the seed-reachable local DAG, remove redundant edges
before the grouped reachability or longest-depth DP:

- if `u -> v` is already implied by `u -> w -> ... -> v`
- then drop the direct `u -> v`

The hypothesis was that fewer edges would mean:

- fewer bitset merges for grouped reachability
- fewer successor scans for longest depth

### Implementation Shape

The prototype:

1. built descendant bitsets over the local DAG
2. removed redundant direct edges
3. ran the existing DP over the reduced graph

### Outcome

This was **not** a win overall.

#### Dependency Reach

Results were only marginally changed and not clearly better than the
baseline:

| Scale | Trial C# Query | Baseline C# Query |
|---|---:|---:|
| 300 | 0.065s | 0.060s |
| 1k | 0.100s | 0.097s |
| 5k | 0.232s | 0.199s |
| 10k | 0.501s | 0.485s |

Even when the results were close, the extra preprocessing cost was hard
to justify.

#### Dependency Longest Depth

This regressed more clearly:

| Scale | Trial C# Query | Baseline C# Query |
|---|---:|---:|
| 300 | 0.059s | 0.063s |
| 1k | 0.061s | 0.060s |
| 5k | 0.105s | 0.098s |
| 10k | 0.114s | 0.104s |

The effect was small at low scale and clearly negative at `10k`.

### Conclusion

Local transitive reduction is **not** the next high-value DAG
optimization for this runtime.

The preprocessing cost of proving edge redundancy outweighed the savings
from scanning fewer edges in these workloads.

This does not mean transitive reduction is never useful. It means it is
not the right low-hanging fruit for the current synthetic dependency
benchmarks.

## Main Takeaway

The next DAG optimization should focus on **runtime overhead**, not more
graph rewriting.

The current likely cost centers are:

1. seed-reachable cone construction
2. node-id and adjacency building
3. result materialization and aggregation overhead

The key observation is:

- the core DAG dynamic programming itself is already cheap
- a growing share of the remaining cost is in surrounding runtime work

## Recommended Next Steps

1. Profile and tighten the seed-reachable cone build.
2. Reduce allocation overhead in node/id/adjacency construction.
3. Reduce result materialization overhead, especially for grouped reach.
4. Keep the DAG theory notes separate from these runtime-cost notes.

## Relationship To Existing Notes

This note complements, rather than replaces:

- `DAG_QUERY_EXECUTION_THEORY.md`
- `DAG_GROUPED_REACH_STRATEGY_NOTE.md`
- `DAG_SEED_RESTRICTED_GROUPED_REACH_NOTE.md`

Those notes are about:

- the correct DAG execution model
- the right grouping abstraction
- the separation from non-DAG work

This note is specifically about:

- what runtime-cost experiments were tried next
- what failed to pay off
- where the next engineering time should go
