# Proposal: Per-Path Visited Tracking in C# Query Engine

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

These are all **path-at-a-time** (DFS) approaches. The C# query engine
needs an equivalent that works within its plan-based architecture.

## Proposed Solution

### Option 1: PathAwareFixpointNode (recommended)

Add a new plan node `PathAwareFixpointNode` that tracks per-path visited
state during evaluation. This node would:

1. Evaluate the base plan (same as `FixpointNode`)
2. For each delta tuple, track which "source" nodes led to it
3. During recursive evaluation, exclude tuples that would revisit
   a node already in the current derivation chain
4. Use a `HashSet<string>` per derivation path

#### Implementation Sketch

```csharp
private IEnumerable<object[]> ExecutePathAwareFixpoint(
    PathAwareFixpointNode fixpoint, EvaluationContext? context)
{
    // Instead of global delta/total sets, maintain per-source DFS
    var adj = BuildAdjacencyFromRelation(fixpoint.StepRelation, context);
    var results = new List<object[]>();

    foreach (var seed in GetSeeds(fixpoint, context))
    {
        // DFS with per-path visited (same as Go/Rust/Python)
        var stack = new Stack<(string node, int hops, HashSet<string> visited)>();
        stack.Push((seed, 0, new HashSet<string> { seed }));

        while (stack.Count > 0)
        {
            var (cur, hops, visited) = stack.Pop();
            foreach (var neighbor in adj.GetValueOrDefault(cur, Array.Empty<string>()))
            {
                if (visited.Contains(neighbor)) continue;
                results.Add(new object[] { seed, neighbor, hops + 1 });

                var newVisited = new HashSet<string>(visited) { neighbor };
                stack.Push((neighbor, hops + 1, newVisited));
            }
        }
    }

    return results;
}
```

#### When to Use

The compiler should emit `PathAwareFixpointNode` instead of `FixpointNode`
when it detects the per-path visited pattern in the Prolog source
(`is_per_path_visited_pattern/4` from `pattern_matchers.pl`).

For standard Datalog predicates (no visited list, no counters), the
existing `FixpointNode` with semi-naive evaluation is correct and more
efficient.

### Option 2: Shortest-Path Dedup (approximation)

Deduplicate on `(source, ancestor)` pairs, keeping only the shortest
hops. This is semantically different (loses multiple valid simple paths)
but gives a close approximation for the d_eff formula where short paths
dominate.

Useful for applications that only need shortest-path distances, not
all-simple-paths enumeration.

### Option 3: Cycle Guard in Plan

Add a `CycleGuardNode` that wraps the recursive step and filters out
tuples that would create a cycle. This requires tracking the derivation
chain in the evaluation context.

More complex than Option 1 but preserves the plan-based architecture
better.

### Note on Engine Capabilities

The C# query engine already has both **bottom-up** (semi-naive fixpoint)
and **top-down** (demand-driven, parameterized) evaluation capabilities.
The per-path visited pattern is a natural fit for the top-down path —
the `ParamSeedNode` already seeds evaluation from known inputs, and
demand closure computes backward reachability. Extending this to track
per-path visited state during top-down evaluation should align with the
existing architecture rather than fighting it.

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

Execute time scaling (DFS pipeline, NOT query engine):

| Scale | C# | Rust | Go | Codon |
|-------|-----|------|-----|-------|
| 300 art | 0.43s | 0.33s | 0.43s | 0.67s |
| 1K art | 1.13s | 1.33s | 1.96s | 2.55s |
| 5K art | 4.74s | 6.86s | 11.36s | 10.98s |
| 10K art | 9.48s | 12.44s | 18.71s | 22.14s |

Once the query engine supports per-path visited, it should be
benchmarked against these DFS pipeline numbers to verify it's
competitive or better.

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

When fixed, compare against the DFS pipeline benchmark at 10K scale:

| Target | Execute (10K articles) |
|--------|----------------------|
| C# DFS pipeline | 9.48s |
| C# Query Engine | TBD (should be similar or better) |

The query engine should be competitive with or faster than the DFS
pipeline because it has the same HashSet-based evaluation, plus
potential for JIT optimization of the plan evaluation hot path.

## Priority

This is **high priority** because:
1. The query engine produces incorrect results without it
2. The benchmark's headline result ("C# beats Rust at scale") is
   from the DFS pipeline, not the query engine — we need the engine
   to match
3. The per-path visited pattern is the most common graph traversal
   pattern in real applications
