# Proposal: Per-Path Visited Tracking in C# Query Engine

## Status

As of 2026-03-30, the C# query engine now handles the **canonical counted
transitive-closure shape** with a dedicated
`PathAwareTransitiveClosureNode`:

```prolog
path(X, Y, H) :- edge(X, Y), H is Base.
path(X, Z, H) :- edge(X, Y), path(Y, Z, H1), H is H1 + Increment.
```

The compiler recognizes that pattern and the runtime evaluates it with
DFS plus a copied visited set per branch, which gives the intended
per-path cycle semantics for counted reachability on cyclic graphs.

What is still not implemented in the C# query engine is **generic
visited-list lowering** from Prolog source patterns such as
`\+ member(X, Visited)` and `[X|Visited]`. This document therefore
serves partly as historical context and partly as a note on the broader
remaining work.

## Problem

The C# query engine's `FixpointNode` produces **semantically incorrect
results** for recursive predicates with hop counters on cyclic graphs.

### Demonstrated Failure

Query: `category_ancestor("Relativity", Ancestor, Hops)`

| Engine | Hops to "Physics" | Total results |
|--------|-------------------|---------------|
| **Prolog** (correct) | [2] | 71 unique pairs |
| **C# FixpointNode** (incorrect) | [2, 17, 18, 19, ... 51] | 1748 tuples |

The C# engine finds "Physics" at 36 different hop counts because it
traverses cycles repeatedly. Hops 17-51 are from cycle paths (going
around loops and reaching Physics again). Prolog's Visited list correctly
prevents this.

### Root Cause

The `FixpointNode` uses **set-at-a-time** semi-naive evaluation:
- Each iteration processes ALL delta tuples
- Deduplication is on full `(source, ancestor, hops)` triples
- `(Relativity, Physics, 2)` ≠ `(Relativity, Physics, 17)` → both kept
- No per-path cycle detection → cycles produce unbounded new tuples

The `MaxFixpointIterations` bound (added in this PR) prevents OOM but
doesn't fix the semantics — it just caps the number of cycle traversals.

### What Other Targets Do

All other targets solve this with **per-path visited tracking**:

| Target | Mechanism |
|--------|-----------|
| Prolog | `Visited` list parameter, `\+ member(X, Visited)` |
| Go | `map[string]bool` copied per branch |
| Rust | `HashSet<String>` cloned per branch |
| Python | `frozenset` union per branch |
| AWK | DFS stack with path string |

These are all **path-at-a-time** (DFS — depth-first search) approaches. The C# query engine
needs an equivalent that works within its plan-based architecture.

## Implemented Solution

### PathAwareTransitiveClosureNode (implemented)

The implemented node is `PathAwareTransitiveClosureNode`. It specializes
the common counted-reachability case instead of extending the generic
`FixpointNode`.

The runtime:

1. Builds an adjacency index from the edge relation
2. Enumerates seeds either from the full edge set or from bound source
   parameters
3. Runs DFS from each seed
4. Copies the visited set on each branch so sibling paths do not
   interfere
5. Deduplicates output by `(target, depth)` per seed — multiple simple
   paths reaching the same node at the same hop count produce one row
6. Enforces `MaxDepth` to match Prolog's `max_depth/1` semantics
7. Emits `(source, target, hops)` rows with the configured base depth
   and increment

#### Bugs Fixed (2026-03-30)

Codex's initial implementation (commit `08df76b`) had two issues:

1. **No output deduplication**: Multiple distinct simple paths to the
   same `(target, hops)` produced duplicate rows. E.g., 56 paths from
   Relativity to Container_categories at hops=20 → 56 identical output
   rows. Fix: `emitted` HashSet keyed on `(target, depth)` per seed.

2. **No max depth**: Without a depth limit, the DFS explores all simple
   paths of arbitrary length. With per-path visited, simple paths in a
   cyclic graph can be very long (hops 19-24+). Fix: `MaxDepth` field
   on the record, enforced in `AppendPathAwareRowsForSeed`.

Results on dev dataset (198 edges), querying "Relativity":

| Version | Total rows | Unique pairs | Physics hops |
|---------|-----------|-------------|-------------|
| Before fix | 3930 | 522 | [2, 19, 20, 21, 22, 23, 24] |
| Dedup only | 522 | 522 | [2, 19, 20, 21, 22, 23, 24] |
| Dedup + MaxDepth=10 | 71 | 71 | [2] |
| Prolog reference | 115* | 71 | [2] |

*Prolog reports 115 derivations but 71 unique `(ancestor, hops)` pairs.

#### Implementation Sketch

```csharp
private static void AppendPathAwareRowsForSeed(
    object? seed,
    IReadOnlyDictionary<object, List<object[]>> succIndex,
    int baseDepth, int depthIncrement,
    ICollection<object[]> output, int maxDepth = 0)
{
    var emitted = new HashSet<(object?, int)>();
    var stack = new Stack<(object? Node, int Depth, HashSet<object?> Visited)>();
    stack.Push((seed, 0, new HashSet<object?> { seed }));

    while (stack.Count > 0)
    {
        var (current, depth, visited) = stack.Pop();
        foreach (var neighbor in GetNeighbors(current, succIndex))
        {
            if (visited.Contains(neighbor)) continue;
            var nextDepth = depth == 0 ? baseDepth : depth + depthIncrement;
            if (maxDepth > 0 && nextDepth > maxDepth) continue;
            if (emitted.Add((neighbor, nextDepth)))
                output.Add(new object[] { seed!, neighbor!, nextDepth });
            var nextVisited = new HashSet<object?>(visited) { neighbor };
            stack.Push((neighbor, nextDepth, nextVisited));
        }
    }
}
```

#### When to Use

The compiler now emits `PathAwareTransitiveClosureNode` when it detects
the counted transitive-closure shape above. The `MaxDepth` defaults to
10 (matching Prolog's `max_depth/1`).

For standard Datalog predicates without counters, the existing
`TransitiveClosureNode`, `GroupedTransitiveClosureNode`, or generic
fixpoint lowering still apply.

### Remaining Generalization Work

The broader visited-list compilation path is still open. In particular,
predicates whose Prolog source explicitly threads a `Visited` list are
not yet lowered into an equivalent query-plan node.

## Prior Work

### In this project

| PR/Commit | Description | Relevance |
|-----------|-------------|-----------|
| PR #1054 | Cross-target effective distance benchmark | Original benchmark revealing this issue |
| PR #1056 | Go/AWK arity-3 deepening | Per-path visited in Go/AWK targets |
| PR #1057 | C# parameterized query engine fix | `is/2` support, mode declarations |
| PR #1063 | Per-path visited pattern detection | `is_per_path_visited_pattern/4` in shared core |
| PR #1065 | Per-path visited implementation plan | Cross-target implementation guide |
| PR #1068 | Self-contained benchmark pipelines | DFS pipeline approach (what we're comparing against) |
| PR #1093 | 1K/5K benchmark — C# takes lead | Performance data motivating this work |
| PR #1094 | 10K benchmark — C# widens lead | C# 1.3x faster than Rust at 10K |
| PR #1092 | Codon + AWK optimization + demand analysis docs | Optimization analysis |
| PR #1100 | Transpilation audit + fixpoint non-convergence | Identified the bug |
| `dce3945b` | Demand closure for parameterized mutual SCCs | Top-down evaluation in engine |
| `88346fe9` | Parameterized mutual recursion | SCC-aware top-down evaluation |
| `e738fee3` | Seed transitive closure for parameters | ParamSeedNode for TC |

### Design documents

| Document | Contents |
|----------|----------|
| `docs/design/PER_PATH_VISITED_RECURSION.md` | Theory, taxonomy, three compilation strategies |
| `docs/design/PER_PATH_VISITED_IMPLEMENTATION_PLAN.md` | Cross-target implementation steps |
| `docs/design/FULL_TRANSPILATION_AUDIT.md` | Which predicates compile on which targets |
| `docs/proposals/PROLOG_TARGET_DEMAND_ANALYSIS.md` | Demand analysis extraction from C# engine |
| `examples/benchmark/README.md` | Full scaling results across 7 targets |

### Education repo

Book 3 (C# target) was restructured (education repo PR #18) with:
- **Ch 3**: Parameterized query engine (modes, `is/2`, demand analysis)
- **Ch 4**: Performance at scale (benchmark results, C# beating Rust/Go)
- **Appendix A**: Old non-parameterized engine (pedagogical stepping stone)

The per-path visited fix would be referenced in Ch 3 as an advanced topic.

### Profiling results

**Query engine vs DFS pipelines (2026-03-30):**

| Target | 300 art | 1K art | 5K art | 10K art |
|--------|---------|--------|--------|---------|
| **C# Query Engine** | **0.40s** | **0.22s** | **0.66s** | **1.51s** |
| C# DFS pipeline | 0.96s | 1.57s | 5.81s | 10.29s |
| Rust DFS pipeline | 0.33s | 1.33s | 6.86s | 12.44s |
| Go DFS pipeline | 0.43s | 1.96s | 11.36s | 18.71s |
| Codon DFS pipeline | 0.67s | 2.55s | 10.98s | 22.14s |

The query engine is **2.4-10x faster** than DFS pipelines at all scales.
The speedup comes from seed deduplication: at 1K scale, 1000 articles
map to only 89 unique category seeds, so the engine does ~11x less DFS
work. The precomputed ancestor index then makes per-article aggregation
nearly free.

### Benchmark data

The `examples/benchmark/` directory contains:
- Dev dataset (19 articles, 198 edges) — for correctness testing
- 300-article dataset (6008 edges) — for performance comparison
- 1K/5K/10K datasets — for scaling analysis
- `generate_pipeline.py` — DFS pipeline generator (the correct reference)
- `compute_effective_distance.py` — d_eff aggregation (validation tool)

## Key Files

| File | Role |
|------|------|
| `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs:11221` | `ExecuteFixpoint` — current fixpoint loop |
| `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs:326` | `QueryExecutorOptions` — configuration |
| `src/unifyweaver/targets/csharp_target.pl:1472` | `build_pipeline_seeded` — plan construction |
| `src/unifyweaver/core/advanced/pattern_matchers.pl` | `is_per_path_visited_pattern/4` — pattern detection |
| `docs/design/PER_PATH_VISITED_RECURSION.md` | Design for per-path visited across all targets |

## Test Case

```csharp
// This should produce results matching Prolog's Visited-list semantics
var (provider, plan) = CategoryAncestorQueryModule.Build();
var options = new QueryExecutorOptions(MaxFixpointIterations: 50);
var executor = new QueryExecutor(provider, options);

var results = executor.Execute(plan,
    new List<object[]> { new object[] { "Relativity" } }).ToList();

// Expected: ~71 unique (ancestor, hops) pairs (matching Prolog)
// Current:  1748 tuples (includes cycle paths — WRONG)

// Specifically, "Physics" should appear only at hops=2
// Current: appears at hops 2, 17, 18, 19, ... 51
```

## Benchmark Reference

For performance validation, compare against the DFS pipeline benchmark
at 10K scale:

| Target | Execute (10K articles) |
|--------|----------------------|
| C# DFS pipeline | 9.48s |
| C# Query Engine | TBD (should be similar or better) |

The query engine should be competitive with or faster than the DFS
pipeline because it has the same HashSet-based evaluation, plus
potential for JIT optimization of the plan evaluation hot path.

## Priority

The counted transitive-closure case is now implemented. The remaining
high-priority work is:
1. Lower explicit visited-list predicates into query plans
2. Benchmark the specialized query-engine path against the DFS pipeline
3. Extend the specialization beyond the current canonical counted shape
