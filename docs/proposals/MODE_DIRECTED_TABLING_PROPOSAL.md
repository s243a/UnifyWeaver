# Proposal: Mode-Directed Tabling for Recursive Optimization

## Problem

UnifyWeaver can compile recursive predicates to efficient DFS with
per-path visited semantics. But some queries only need an aggregate
over all paths (e.g., shortest path = min, widest path = max), and
the engine wastes time exploring paths that can't improve on the
current best.

Example: if the shortest known path from A to Physics is 3 hops,
exploring a branch that's already 4 hops deep is provably useless.
Today the engine explores it anyway, producing 7-16x more tuples
than needed.

The user has no declarative way to express "I want the minimum" in
a way that enables pruning. They write:

```prolog
shortest_path(X, Y, MinH) :-
    aggregate_all(min(H), path(X, Y, H), MinH).
```

This collects ALL paths then takes the min — no early termination.

## Proposed Solution: Mode-Directed Tabling

Mode-directed tabling (from XSB Prolog, also in SWI-Prolog) lets the
user declare how answers should be aggregated at the table level:

```prolog
:- table path(_, _, min).
path(X, Y, 1) :- edge(X, Y).
path(X, Z, H) :- edge(X, Y), path(Y, Z, H1), H is H1 + 1.
```

The `min` mode on the third argument means:
- For each unique `(X, Y)` pair, keep only the minimum `H`
- New derivations worse than the tabled answer are discarded
- The table acts as a dynamic upper bound for pruning

This is fully declarative — the user states the aggregation intent,
not the pruning strategy.

## How It Maps to Each Compilation Strategy

### C# Query Engine: Branch-and-Bound in PathAwareTransitiveClosureNode

The runtime maintains a "best known" table alongside the DFS:

```csharp
var bestKnown = new Dictionary<(object?, object?), int>();  // (seed, target) -> min depth

// In AppendPathAwareRowsForSeed:
if (bestKnown.TryGetValue((seed, next), out var best) && nextDepth >= best)
    continue;  // prune: can't improve
bestKnown[(seed, next)] = nextDepth;
output.Add(new object[] { seed!, next!, nextDepth });
```

This is a local change to the existing DFS loop. The table mode
(`min`, `max`, `sum`, `first`) determines the comparison operator.

Expected impact: at 10K scale, shortest path would drop from ~3.6M
tuples to ~264K (matching d_eff's dedup count), with corresponding
speedup from ~3s to ~1.5s.

### Native Targets (Go, Rust, Python, AWK): Same Pattern

The DFS template gains a `bestKnown` map:

```go
// Go example
bestKnown := make(map[string]int)  // target -> min depth

for _, neighbor := range adj[current] {
    if visited[neighbor] { continue }
    newDepth := depth + 1
    if best, ok := bestKnown[neighbor]; ok && newDepth >= best {
        continue  // prune
    }
    bestKnown[neighbor] = newDepth
    results = append(results, Result{seed, neighbor, newDepth})
    // ... recurse
}
```

### Prolog Target: Use Native Tabling

SWI-Prolog already supports `:- table pred(_, _, min).` natively.
For the Prolog target, UnifyWeaver simply passes through the tabling
directive.

## Supported Table Modes

| Mode | Semantics | Pruning strategy | Use case |
|------|-----------|-----------------|----------|
| `min` | Keep minimum value | Prune branches >= current best | Shortest path |
| `max` | Keep maximum value | Prune branches <= current best | Widest/best path |
| `first` | Keep first derivation | Stop after first answer per key | Reachability |
| `sum` | Accumulate sum | No pruning (all paths needed) | d_eff, influence |
| `count` | Count derivations | No pruning | Path counting |
| `all` | Keep all (default) | No pruning | Enumeration |

The `sum`, `count`, and `all` modes don't enable pruning but still
inform the query engine about deduplication strategy:
- `sum` needs all derivations but can discard individual rows after
  accumulating
- `count` only needs the count, not the rows themselves
- `all` is the default (current behavior)

## Interaction with Existing Features

### Per-Path Visited

Mode-directed tabling composes with per-path visited semantics:
- Visited prevents cycles within a single path (correctness)
- Tabling prunes across paths by tracking global best (performance)

These are orthogonal — a branch is pruned if EITHER it revisits a
node (visited) OR it can't improve the tabled answer (bound).

### MaxDepth

MaxDepth is a static bound. The tabled best is a dynamic bound that
can be tighter. With `min` tabling:
- MaxDepth provides the initial bound
- As shorter paths are found, the effective bound tightens
- The static MaxDepth becomes a fallback for unreachable targets

### Dedup (emitted HashSet)

The current `emitted` HashSet deduplicates output by `(target, depth)`.
With `min` tabling, this becomes redundant — the table itself ensures
only the minimum is emitted. The `emitted` set could be replaced by
the tabling table.

## Syntax

### User-Facing (Prolog Source)

```prolog
% Standard XSB/SWI syntax
:- table category_ancestor(_, _, min).

category_ancestor(Cat, Parent, 1) :-
    category_parent(Cat, Parent).
category_ancestor(Cat, Ancestor, Hops) :-
    category_parent(Cat, Mid),
    category_ancestor(Mid, Ancestor, H1),
    Hops is H1 + 1.
```

### Compiler Detection

The compiler sees `:- table pred(Modes...)` and:
1. Parses the mode list (`_` = lattice/all, `min`/`max`/`first` = directed)
2. Checks that the predicate is recursive
3. Checks that the directed argument is the accumulator
4. Emits a specialized plan node or DFS template with pruning

### Generated C# Plan

```csharp
new PathAwareTransitiveClosureNode(
    edgeRelation, predicate, baseDepth, depthIncrement, maxDepth,
    tableMode: TableMode.Min  // new parameter
)
```

Or a new node type if the modes need richer representation:

```csharp
new TabledTransitiveClosureNode(
    edgeRelation, predicate, baseDepth, depthIncrement, maxDepth,
    tableModes: new[] { TableMode.Lattice, TableMode.Lattice, TableMode.Min }
)
```

## Implementation Phases

### Phase 1: Directive Parsing

Parse `:- table pred(Modes...)` in the shared core. Store the table
modes in the predicate metadata.

**Location**: `src/unifyweaver/core/` (new or extend existing directive handling)

### Phase 2: C# Query Engine — Min/Max Pruning

Add branch-and-bound to `AppendPathAwareRowsForSeed` when the table
mode is `min` or `max`. This is a small runtime change.

**Location**: `QueryRuntime.cs`, `csharp_target.pl`

### Phase 3: Native Target Templates

Add `bestKnown` map to DFS templates for Go, Rust, Python when table
mode is `min` or `max`.

**Location**: `go_target.pl`, `rust_target.pl`, `python_target.pl`

### Phase 4: Prolog Target Passthrough

Pass `:- table` directives through to the generated Prolog output.

### Phase 5: Sum/Count Modes

Implement in-place aggregation for `sum` and `count` modes — emit
accumulated values instead of individual rows. This is a different
optimization (memory, not pruning).

## Correctness Argument

For `min` tabling on a monotonically increasing accumulator:

1. The accumulator strictly increases along any path (each hop adds
   >= baseDepth which is >= 1)
2. Therefore, if the current branch depth >= the tabled minimum for
   (seed, target), no extension of this branch can produce a shorter
   path
3. Pruning is sound: no minimum-valued answer is lost
4. The tabled answer is updated monotonically (only decreases), so
   the pruning bound only tightens over time

For `max` on a bounded accumulator (e.g., with MaxDepth), the
argument is symmetric.

## Performance Expectations

Based on the shortest path benchmark results:

| Scale | Current (no pruning) | With min-tabling (estimated) |
|-------|---------------------|------------------------------|
| 300 | 0.57s (603K tuples) | ~0.35s (~84K tuples) |
| 1K | 0.35s (353K tuples) | ~0.20s (~26K tuples) |
| 5K | 1.19s (1.3M tuples) | ~0.60s (~85K tuples) |
| 10K | 2.97s (3.6M tuples) | ~1.50s (~264K tuples) |

The pruned tuple counts should approach the d_eff deduped counts,
since both effectively keep one answer per `(source, target)` pair.

## Prior Art

| System | Feature | Notes |
|--------|---------|-------|
| XSB Prolog | Mode-directed tabling | Original implementation, `min`, `max`, `first`, `last` |
| SWI-Prolog | `table/1` directive | Compatible with XSB syntax, lattice modes |
| Soufflé (Datalog) | Lattice operations | Functors with lub/glb for aggregation |
| Differential Dataflow | `distinct`, `reduce` | Incremental aggregation over recursive relations |
| LogicBlox | Lattice types | Built-in min/max/sum lattice types on relations |

## Related UnifyWeaver Work

| Document | Relevance |
|----------|-----------|
| `docs/proposals/COMPUTED_RECURSIVE_INCREMENT_*.md` | Computed increments (tabling optimizes their evaluation) |
| `docs/proposals/CSHARP_QUERY_ENGINE_PER_PATH_VISITED.md` | Per-path visited (orthogonal to tabling) |
| `docs/proposals/CROSS_TARGET_EFFECTIVE_DISTANCE_THEORY.md` | d_eff benchmark (sum mode, no pruning benefit) |
| `examples/benchmark/shortest_path_to_root.pl` | Primary evaluation workload for min tabling |

## Open Questions

1. **Should tabling subsume dedup?** The current `emitted` HashSet in
   `AppendPathAwareRowsForSeed` deduplicates by `(target, depth)`.
   With `min` tabling, this is redundant. Should we unify them?

2. **Dynamic MaxDepth interaction**: If min-tabling finds a path of
   length 3, should that override MaxDepth=10 for that seed? Or
   should MaxDepth remain a static upper bound independent of tabling?

3. **Multi-argument tabling**: `:- table pred(_, min, max).` — min on
   one argument, max on another. Is this needed? XSB supports it.

4. **Lattice generalization**: Instead of fixed modes, should we
   support arbitrary lattice operations? This is what Soufflé and
   LogicBlox do. More general but harder to optimize.
