# Codex Handoff: Mode-Directed Tabling Runtime Support

## What's Done

### Phase 1: Compiler-side (complete)

The compiler now parses `:- table pred(_, _, min).` directives and
passes them through to the generated C# code.

**Files changed:**

1. `src/unifyweaver/core/advanced/pattern_matchers.pl`
   - `declared_table_modes/3` — looks up `user:table/1` declarations
   - `parse_table_spec/2` — parses table args into mode atoms
     (`lattice`, `min`, `max`, `first`, `sum`, `count`)
   - `has_directed_table_mode/4`, `table_enables_pruning/2` — helpers

2. `src/unifyweaver/targets/csharp_target.pl`
   - `maybe_path_aware_transitive_closure_plan` now adds
     `table_modes:[lattice, lattice, min]` to the plan dict
   - `emit_plan_expression` for `path_aware_transitive_closure` emits
     a `TableMode.Min` (or `.Max`/`.All`/etc.) parameter
   - `table_mode_csharp_enum/2` maps Prolog modes to C# enum strings

3. `tests/core/test_csharp_query_target.pl`
   - Updated path-aware TC test dicts to include `table_modes`

The compiler currently emits:
```csharp
new PathAwareTransitiveClosureNode(
    new PredicateId("category_parent", 2),
    new PredicateId("category_ancestor", 3),
    1, 1, 10, TableMode.Min)
```

But the C# runtime doesn't define `TableMode` yet — that's Phase 2.

## What Codex Needs To Do

### Phase 2: C# Query Engine Runtime

**Goal**: Add branch-and-bound pruning to `AppendPathAwareRowsForSeed`
when the table mode is `min` or `max`.

#### Step 2.1: Add TableMode enum

```csharp
public enum TableMode
{
    All,      // default — keep all derivations (no pruning)
    Min,      // keep minimum accumulator value, prune worse branches
    Max,      // keep maximum accumulator value, prune worse branches
    First,    // keep first derivation per key, stop after first
    Sum,      // accumulate sum (no pruning, but in-place aggregation)
    Count     // count derivations (no pruning, memory optimization)
}
```

#### Step 2.2: Add TableMode to PathAwareTransitiveClosureNode

```csharp
public sealed record PathAwareTransitiveClosureNode(
    PredicateId EdgeRelation,
    PredicateId Predicate,
    int BaseDepth,
    int DepthIncrement,
    int MaxDepth = 0,
    TableMode AccumulatorMode = TableMode.All  // new
) : PlanNode;
```

#### Step 2.3: Implement branch-and-bound in AppendPathAwareRowsForSeed

When `AccumulatorMode` is `Min`:

```csharp
// Track best known depth per (seed, target)
var bestKnown = new Dictionary<object?, int>();

// In the DFS loop, after computing nextDepth:
if (accumulatorMode == TableMode.Min)
{
    if (bestKnown.TryGetValue(next, out var best) && nextDepth >= best)
        continue;  // prune — can't improve
    bestKnown[next] = nextDepth;
}
```

This replaces the `emitted` HashSet for `Min` mode — the bestKnown
dict serves both as dedup and pruning bound.

For `Max` mode, the comparison is reversed (`nextDepth <= best`).

For `All` mode (default), keep current behavior with `emitted` HashSet.

#### Step 2.4: Update display string

In the plan node formatting:
```csharp
PathAwareTransitiveClosureNode closure =>
    $"PathAwareTransitiveClosure edge={closure.EdgeRelation} ... mode={closure.AccumulatorMode}",
```

### Expected Performance Impact

Shortest path benchmark without pruning:

| Scale | Tuples | Time |
|-------|--------|------|
| 300 | 603K | 0.57s |
| 1K | 353K | 0.35s |
| 5K | 1.3M | 1.19s |
| 10K | 3.6M | 2.97s |

With `Min` pruning, tuples should drop to ~d_eff levels (~84K at 300,
~26K at 1K, etc.) with corresponding ~2x speedup.

### Testing

The existing test predicate `test_pathaware_reach/3` doesn't have a
table directive, so it uses `TableMode.All` (backward compatible).

To test `Min` mode, add a new fixture:

```prolog
:- dynamic user:table/1.
assertz(user:table(test_pathaware_min_reach(_, _, min))),
% ... same edges as test_pathaware_reach ...
```

Expected behavior: only the shortest path per (source, target) pair
is returned.

## Prior Work and Context

| Document | Contents |
|----------|----------|
| `docs/proposals/MODE_DIRECTED_TABLING_PROPOSAL.md` | Full proposal with correctness argument, all modes, prior art |
| `docs/proposals/COMPUTED_RECURSIVE_INCREMENT_PHILOSOPHY.md` | Theory: spectral dimensionality, semantic distance, collapsed trees |
| `docs/proposals/COMPUTED_RECURSIVE_INCREMENT_SPEC.md` | Pattern detection spec for computed increments (related) |
| `docs/proposals/COMPUTED_RECURSIVE_INCREMENT_PLAN.md` | Implementation plan for computed increments |
| `examples/benchmark/shortest_path_to_root.pl` | Prolog benchmark that would benefit from `Min` tabling |
| `examples/benchmark/effective_distance.pl` | d_eff benchmark (uses `All` mode — no pruning benefit) |
| `examples/benchmark/category_influence.pl` | Influence benchmark (uses `All` mode) |

## Benchmark Results for Context

Query engine vs DFS (depth-first search) pipelines:

| Target | 300 art | 1K art | 5K art | 10K art |
|--------|---------|--------|--------|---------|
| C# Query Engine | 0.40s | 0.22s | 0.66s | 1.51s |
| C# DFS pipeline | 0.96s | 1.57s | 5.81s | 10.29s |
| Rust DFS pipeline | 0.33s | 1.33s | 6.86s | 12.44s |

The query engine is 2.4-10x faster than DFS pipelines due to seed
deduplication. Min-tabling would further reduce tuple counts for
shortest-path queries.

## Commits to Review

- `baeb1208` — Phase 1: table directive parsing + C# plan builder wiring
- `dad56e09` — Mode-directed tabling proposal
- `5a5eee21` — Shortest path benchmark (the eval workload)
- `0c1943cc` / `dbb3ac0d` — Original dedup + max depth fix for PathAwareTransitiveClosureNode
