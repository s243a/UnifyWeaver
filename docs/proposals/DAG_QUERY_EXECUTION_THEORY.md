# DAG Query Execution Theory

## Summary

This note captures the current theory direction behind the recent DAG
work in the C# query runtime.

The central split is:

- **DAG workloads**
  - state can usually be summarized per node or per node-plus-group
  - dynamic programming is the right execution model
- **non-DAG simple-path workloads**
  - state must include path or visited-set information
  - exact execution is fundamentally more expensive

This note is only about the DAG side.

## Core Principle

For DAGs, the useful state is not a path. It is a **summary over a node
in topological context**.

That summary can take different forms depending on the query:

1. **pair reachability**
   - question: can `u` reach `v`?
   - natural summary: descendant or ancestor reachability sets

2. **grouped reachability**
   - question: which labels/groups reach this node?
   - natural summary: propagated group bitsets on nodes

3. **longest depth**
   - question: what is the longest path from this node to a sink?
   - natural summary: a single integer depth per node

The mistake to avoid is treating all DAG problems as path enumeration.

## Why DAGs Are Different

In a DAG:

- topological order exists
- recursion is well-founded
- no visited-set/path-state is needed for exactness

So fixed-point evaluation can often be collapsed to:

- one preprocessing pass
- one topological dynamic-programming pass

rather than repeated frontier growth over richer path states.

## Reachability Shapes

### 1. Plain Reachability

For ungrouped transitive closure:

- state: reachable descendants per node
- useful execution: topological propagation with exact bitsets or other
  memoized descendant summaries

This was the basis of the earlier DAG reachability fast path.

### 2. Grouped Reachability

For dependency reach count, the important lesson was:

- the output is grouped by **project**
- not by raw dependency seed

So the right grouped state is:

- project membership propagated over reachable nodes

not:

- raw dependency-seed membership propagated over reachable nodes

That is why the recent `SeedGroupedTransitiveClosureNode` uses:

- a plain edge relation
- a separate `(group, seed)` relation
- grouped propagation inside the runtime

instead of materializing raw `(seed, reachable)` closure and regrouping
later.

## Longest Depth Shape

For longest depth on a DAG:

- each node needs only one value:
  - `depth(node) = 1 + max(depth(successor))`
  - or `1` for sinks

This is simpler than grouped reachability:

- no bitsets are needed
- only a reverse topological pass and a scalar memo

At the grouped level, project longest depth becomes:

- `max(depth(seed))` over the direct dependency seeds of the project

So the grouped state for longest depth lives mostly at the **seed
relation**, not in the propagation itself.

That is why the runtime node for the benchmark can be:

- `SeedGroupedDagLongestDepthNode(edgeRelation, seedRelation, predicate)`

which:

1. computes scalar node depths on the DAG
2. reduces them by group over the seed relation

## Design Rule

When choosing a DAG execution strategy, first ask:

- what is the smallest exact summary state that answers this workload?

Examples:

- reachability pair query:
  - descendant summary
- grouped reach count:
  - group-bitset summary
- longest depth:
  - scalar max-depth summary

The runtime should not carry richer state than the query requires.

## Relationship To Non-DAG Work

This theory note is intentionally separate from the non-DAG path-state
design docs.

For non-DAG simple-path workloads:

- state includes visited/path information
- hashing can help with indexing
- exact verification is still required

That is a different execution family.

The point of keeping these notes separate is to prevent DAG workloads
from inheriting unnecessary path-state complexity.

## Current Practical Takeaway

The C# query runtime now has two useful DAG-specialized directions:

1. **project-grouped reachability**
2. **grouped DAG longest depth**

Both are exact.
Both avoid path-state tracking.
Both are better understood as topological dynamic programming than as
general recursive search.

## Likely Next Theory Questions

- when should DAG preprocessing be shared across multiple query-runtime
  nodes on the same edge relation?
- when is transitive reduction worth the preprocessing cost?
- how much of the current benchmark-driven DAG runtime should become
  planner-emitted general-purpose lowering?
