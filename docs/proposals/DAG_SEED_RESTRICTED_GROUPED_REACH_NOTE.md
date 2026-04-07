# DAG Seed-Restricted Grouped Reach Note

## Summary

For the synthetic dependency reach-count benchmark, the next C# query
optimization should not be framed as "seed-restricted reachability over
raw dependency seeds." The more useful grouped state is the **project
label**, not the dependency seed itself.

The benchmark asks:

- for each project
- count the unique packages reachable from that project's direct
  dependency set

The current query-engine benchmark path computes:

1. reachability from each unique direct dependency seed
2. then unions those per-seed results back into per-project counts

That is semantically fine, but it means the runtime is optimizing the
wrong grouped dimension. The useful shared state is not "which direct
dependency seed reaches this node," but "which project reaches this
node."

## What We Learned

On the synthetic dataset family, the grouping cardinalities are:

| Scale | Projects | Unique Dependency Seeds | Project Edges |
|---|---:|---:|---:|
| 300 | 15 | 45 | 45 |
| 1k | 50 | 150 | 150 |
| 5k | 250 | 750 | 750 |
| 10k | 500 | 1500 | 1500 |

So the raw dependency-seed space is consistently about `3x` larger than
the project space.

A seed-restricted DAG propagation over raw dependency seeds therefore
still carries more grouped state than the benchmark result actually
needs.

## Why The Raw-Seed Grouped Experiment Regressed

The attempted seed-restricted grouped DAG pass:

- restricted the graph to the union of reachable nodes from the active
  dependency seeds
- propagated exact seed bitsets over that restricted DAG
- emitted exact `(seed, reachable)` closure rows

This preserved correctness, but it regressed relative to the existing
descendant-bitset DAG fast path.

The reason is structural:

- the benchmark result is grouped by project, not by dependency seed
- the runtime still had to carry a larger grouped state than necessary
- and it still had to materialize closure rows at the raw-seed level
  before the benchmark could union them back per project

So the experiment answered the wrong grouped question efficiently, but
not the right grouped question.

## Better Next Step

The next meaningful runtime direction is:

1. model the grouped state as **project labels**
2. propagate project bitsets across the DAG
3. derive per-project reach counts directly

Conceptually:

- initial project-to-dependency membership seeds the project bitsets
- DAG propagation carries project membership forward through the graph
- each reachable package contributes to the projects whose bit is set

That should reduce grouped state from:

- `unique dependency seeds`

to:

- `projects`

which is the natural output grain of the benchmark.

## Relationship To Existing DAG Work

This note does not replace the existing DAG descendant-bitset fast path.
That fast path materially improved large-scale dependency reach for the
C# query engine and should remain the current implementation baseline.

This note explains why the next DAG optimization should likely be a
**project-grouped reachability path**, not another refinement of
raw-seed closure.

## Theory Direction

If we later write a fuller theory note, it should separate three DAG
execution shapes:

1. **pair reachability**
   - source/target closure
   - best served by descendant summaries or memoized reachability

2. **seed-restricted grouped reachability**
   - grouped by a higher-level label such as project
   - best served by grouped label propagation

3. **longest depth on DAGs**
   - scalar dynamic-programming summary
   - max over predecessors or successors

Those are related but not identical problems, and conflating them makes
it harder to choose the right runtime state.
