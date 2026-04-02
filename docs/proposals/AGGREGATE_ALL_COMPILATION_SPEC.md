# aggregate_all/3 Compilation: Specification

## Supported Forms

### Phase 1: Scalar aggregation over a single goal

```prolog
aggregate_all(AggOp(Expr), Goal, Result)
```

Where:
- `AggOp` is one of: `sum`, `count`, `min`, `max`
- `Expr` is an arithmetic expression over variables bound by `Goal`
- `Goal` is a conjunction of relation lookups and/or recursive calls
- `Result` is bound to the aggregate value

Examples:

```prolog
% Sum of weighted path distances
aggregate_all(sum(W),
    (category_ancestor(Cat, Root, Hops), W is Hops ^ (-5)),
    TotalWeight).

% Count of ancestors
aggregate_all(count,
    category_ancestor(Cat, _, _),
    NumAncestors).

% Minimum hop count (equivalent to min tabling, but post-hoc)
aggregate_all(min(H),
    category_ancestor(Cat, Root, H),
    MinHops).
```

### Phase 2: Grouped aggregation

```prolog
% For each Root, sum the weighted paths from Cat
aggregate_all(sum(W),
    (category_ancestor(Cat, Root, Hops), W is Hops ^ (-5)),
    TotalWeight),
% ... grouped by Root
```

In Prolog, grouping is achieved by having free variables in the Goal
that are bound outside the aggregate:

```prolog
category_influence(Root, Score) :-
    root_category(Root),
    aggregate_all(sum(W),
        (article_category(Art, Cat),
         category_ancestor(Cat, Root, Hops),
         W is (Hops + 1) ^ (-5)),
        Score).
```

Here `Root` is bound before `aggregate_all` is called, so the
aggregation is implicitly grouped by `Root`.

### Phase 3: Collection aggregation

```prolog
aggregate_all(bag(H),
    category_ancestor(Cat, Root, H),
    AllHops).

aggregate_all(set(Root),
    (article_category(Art, Cat), category_ancestor(Cat, Root, _)),
    UniqueRoots).
```

## Pattern Detection

### Compiler recognition

The compiler must detect `aggregate_all/3` in clause bodies and
classify the goal into compilation-friendly categories:

```prolog
classify_aggregate(aggregate_all(Template, Goal, Result), AggInfo) :-
    parse_aggregate_template(Template, AggOp, AggExpr),
    classify_aggregate_goal(Goal, GoalInfo),
    AggInfo = agg_info{
        op: AggOp,
        expr: AggExpr,
        goal: GoalInfo,
        result_var: Result
    }.
```

### Goal classification

The goal inside `aggregate_all` falls into categories:

| Category | Example | Compilation strategy |
|----------|---------|---------------------|
| Single fact scan | `fact(X, Y)` | Scan + aggregate |
| Recursive predicate | `category_ancestor(C, A, H)` | TC output + aggregate |
| Conjunction with recursion | `fact(A, C), rec(C, R, H)` | Join + TC output + aggregate |
| Conjunction with arithmetic | `rec(C, R, H), W is f(H)` | TC output + compute + aggregate |
| Multi-recursive | `rec1(...), rec2(...)` | Out of scope for Phase 1 |

Phase 1 targets the first four categories. The conjunction case
(fact + recursion + arithmetic) is the most common in real queries.

### Template parsing

```prolog
parse_aggregate_template(sum(Expr), sum, Expr).
parse_aggregate_template(count, count, 1).
parse_aggregate_template(min(Expr), min, Expr).
parse_aggregate_template(max(Expr), max, Expr).
parse_aggregate_template(bag(Expr), bag, Expr).
parse_aggregate_template(set(Expr), set, Expr).
```

## Target-Specific Compilation

### C# Query Engine: AggregateNode

Add a new plan node that wraps a sub-plan and applies reduction:

```csharp
public sealed record AggregateNode(
    PlanNode Input,
    AggregateOp Op,       // Sum, Count, Min, Max, Bag, Set
    int ValueIndex,        // which column of input to aggregate
    int? GroupIndex = null  // optional: which column to group by
) : PlanNode;

public enum AggregateOp
{
    Sum, Count, Min, Max, Bag, Set
}
```

Execution:

```csharp
private IEnumerable<object[]> ExecuteAggregate(
    AggregateNode node, EvaluationContext context)
{
    var inputRows = ExecuteNode(node.Input, context);

    if (node.GroupIndex is null)
    {
        // Scalar aggregation
        double result = node.Op switch
        {
            AggregateOp.Sum => inputRows.Sum(r => Convert.ToDouble(r[node.ValueIndex])),
            AggregateOp.Count => inputRows.Count(),
            AggregateOp.Min => inputRows.Min(r => Convert.ToDouble(r[node.ValueIndex])),
            AggregateOp.Max => inputRows.Max(r => Convert.ToDouble(r[node.ValueIndex])),
            _ => throw new NotSupportedException()
        };
        return new[] { new object[] { result } };
    }
    else
    {
        // Grouped aggregation
        var groups = inputRows.GroupBy(r => r[node.GroupIndex.Value]);
        return groups.Select(g => new object[]
        {
            g.Key!,
            g.Aggregate(node.Op, node.ValueIndex)
        });
    }
}
```

This composes with existing plan nodes:

```
AggregateNode(Sum, valueIndex=2,
    ArithmeticNode(Hops ^ (-5),
        PathAwareTransitiveClosureNode(...)))
```

### Go: Native aggregation wrapper

Generate Go code that calls the recursive worker and aggregates:

```go
// Generated from: aggregate_all(sum(W), (cat_anc(C,R,H), W is H^(-5)), Sum)
func categoryInfluence(root string, adj map[string][]string, artCats map[string][]string) float64 {
    sum := 0.0
    for art := range artCats {
        for _, cat := range artCats[art] {
            results := categoryAncestorWorker(cat, make(map[string]bool), adj, 0)
            for _, r := range results {
                if r.ancestor == root {
                    dist := float64(r.hops + 1)
                    sum += math.Pow(dist, -5.0)
                }
            }
        }
    }
    return sum
}
```

The compiler generates this wrapper function from the `aggregate_all`
clause body. The recursive worker is already compiled. The wrapper
adds the aggregation loop.

### Rust: Same pattern as Go

```rust
fn category_influence(root: &str, adj: &HashMap<String, Vec<String>>,
                      art_cats: &HashMap<String, Vec<String>>) -> f64 {
    let mut sum = 0.0_f64;
    for (_, cats) in art_cats {
        for cat in cats {
            let results = category_ancestor_worker(cat, &HashSet::new(), adj, 0);
            for (ancestor, hops) in &results {
                if ancestor == root {
                    let dist = (hops + 1) as f64;
                    sum += dist.powf(-5.0);
                }
            }
        }
    }
    sum
}
```

### Key difference: node vs wrapper

| Target | Strategy | Why |
|--------|----------|-----|
| C# query engine | AggregateNode in plan IR | Composable, optimizable by planner |
| Go | Generated wrapper function | No plan framework, code is the plan |
| Rust | Generated wrapper function | Same as Go |
| Python/Codon | Generated wrapper function | Same pattern |

The C# approach is more general (the planner can optimize, e.g.,
pushing aggregation into the TC node). The native approach is simpler
and sufficient for Phase 1.

## Composition with Existing Machinery

### With PathAwareTransitiveClosureNode (All mode)

Category influence needs all paths:
```
aggregate_all(sum(W), ..., Score)
  └── PathAwareTransitiveClosureNode(mode=All)
```

All paths are enumerated, each contributes to the sum.

### With min tabling

Shortest path could use post-hoc min:
```
aggregate_all(min(H), category_ancestor(Cat, Root, H), MinH)
```

Or equivalently, min tabling inside the TC:
```
:- table category_ancestor(_, _, min).
```

Both produce the same result. Tabling is faster (pruning during DFS),
but `aggregate_all(min(...))` is more general (works over any goal,
not just the recursive predicate itself).

### With computed increments

Degree-corrected semantic distance:
```
aggregate_all(sum(W),
    (semantic_distance(Cat, Root, D), W is D ^ (-5)),
    Score)
```

The `semantic_distance` predicate uses computed increments (Phase 0
of the computed recursive increment plan). `aggregate_all` wraps
the result with standard reduction.

## Correctness Criteria

### Phase 1 validation

1. `aggregate_all(sum(W), (category_ancestor(C, R, H), W is H^(-5)), S)`
   matches the d_eff weight computation from the benchmark
2. `aggregate_all(count, category_ancestor(C, _, _), N)` matches
   the tuple count from query engine execution
3. `aggregate_all(min(H), category_ancestor(C, R, H), M)` matches
   shortest path results
4. Cross-target: C# query engine, Go, Rust produce identical results

### Phase 2 validation

5. `category_influence(Root, Score)` using grouped aggregation
   matches the pipeline benchmark results
6. Grouping by multiple keys works correctly

## Test Queries

### Non-recursive sanity (all targets should pass immediately)

```prolog
:- table test_score/2.
test_score(alice, 90).
test_score(alice, 85).
test_score(bob, 70).

test_avg(Name, Avg) :-
    aggregate_all(sum(S), test_score(Name, S), Total),
    aggregate_all(count, test_score(Name, _), N),
    Avg is Total / N.
```

### Recursive + aggregation (Phase 1)

```prolog
total_weight(Cat, Root, TotalW) :-
    aggregate_all(sum(W),
        (category_ancestor(Cat, Root, H), W is (H+1) ^ (-5)),
        TotalW).
```

### Grouped recursive aggregation (Phase 2)

```prolog
category_influence(Root, Score) :-
    root_category(Root),
    aggregate_all(sum(W),
        (article_category(Art, Cat),
         category_ancestor(Cat, Root, Hops),
         W is (Hops + 1) ^ (-5)),
        Score),
    Score > 0.
```
