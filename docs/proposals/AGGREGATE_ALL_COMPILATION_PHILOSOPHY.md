# aggregate_all/3 Compilation: Philosophy

## The Abstraction Boundary

Prolog's `aggregate_all/3` is the natural boundary between **relational
enumeration** (finding solutions) and **scalar reduction** (combining
them into a single value). Every analytics query eventually crosses
this boundary:

```prolog
aggregate_all(sum(Weight),
    (article_category(Art, Cat), category_ancestor(Cat, Root, Hops),
     Weight is Hops ^ (-5)),
    TotalWeight).
```

This says: "enumerate all `(Art, Cat, Root, Hops)` bindings from the
relational goal, compute `Weight` for each, sum them."

UnifyWeaver can already compile the relational part — transitive
closure, per-path visited, computed increments, min tabling. But it
cannot compile the reduction. Every benchmark so far works around this
by hand-writing the aggregation loop in native code (the pipeline
approach).

Compiling `aggregate_all/3` closes this gap. It is the minimum
abstraction that lets UnifyWeaver compile **complete analytical
queries** from Prolog to target languages, not just the recursive
core.

## Why This Matters

Without `aggregate_all`, UnifyWeaver is a **recursive relation
compiler**. With it, UnifyWeaver becomes an **analytical query
compiler**.

The difference:

| Capability | Without aggregate_all | With aggregate_all |
|-----------|----------------------|-------------------|
| Transitive closure | Yes | Yes |
| Path enumeration | Yes | Yes |
| Shortest path (min) | Yes (via tabling) | Yes |
| Effective distance | Partial (TC only) | Complete |
| Category influence | Partial (TC only) | Complete |
| Any GROUP BY query | No | Yes |

Every real analytical workload has an aggregation step. The category
hierarchy benchmarks demonstrate this clearly: the recursive TC is
fast, but the user still needs to write aggregation code by hand.

## The Composition Model

`aggregate_all/3` composes with the existing recursive machinery
rather than replacing it:

```
Prolog source
  ├── recursive predicate (category_ancestor/3)
  │     └── compiled to: PathAwareTransitiveClosureNode / native DFS
  │
  └── aggregate_all(sum(W), Goal, Sum)
        └── compiled to: aggregation wrapper around the recursive result
```

The recursive predicate produces a **relation** (set of tuples). The
aggregation reduces that relation to a **scalar or grouped result**.
These are separate compilation concerns:

1. **Relation production** — already solved (TC nodes, fixpoint, DFS)
2. **Relation reduction** — the new work (aggregate_all)

This separation means we don't need to change the recursive machinery.
We need to add a layer that consumes its output.

## Relationship to Existing Features

### Mode-directed tabling (min/max)

Tabling applies aggregation **inside** the recursion — pruning branches
during DFS. It's an optimization, not a general aggregation mechanism.

`aggregate_all` applies aggregation **after** the recursion — reducing
the complete result set. It's the general mechanism.

They compose naturally:
- Use `min` tabling inside the TC for shortest-path pruning
- Use `aggregate_all(sum(...))` outside the TC for influence scores

### Pipeline approach (generate_pipeline.py)

The pipeline approach generates complete native programs that include
both recursion AND aggregation. It works, but it's not compilation —
it's code generation from a template, bypassing UnifyWeaver's compiler.

`aggregate_all` compilation lets UnifyWeaver emit the same code that
`generate_pipeline.py` produces, but from Prolog source.

### C# query engine GroupedTransitiveClosureNode

The grouped TC node handles a specific case: transitive closure with
invariant columns (grouping by label/category). This is related but
different — it groups the recursive computation, not the aggregation.

`aggregate_all` with grouping would sit on TOP of the grouped TC,
reducing each group's results to a scalar.

## Design Principle: Wrapper vs Node

There are two strategies for compiling `aggregate_all`:

**Strategy A: Aggregation as a plan node** (C# query engine)

Add an `AggregateNode` to the query plan IR that wraps a sub-plan
and applies reduction:

```
AggregateNode(sum, position=2,
    input=PathAwareTransitiveClosureNode(...))
```

This is the right approach for the query engine because it composes
within the plan evaluation framework and can be optimized by the
planner.

**Strategy B: Aggregation as wrapper code** (Go, Rust, Python)

Generate native code that:
1. Calls the compiled recursive predicate
2. Collects results into a data structure
3. Applies the aggregation in a loop

```go
results := categoryAncestorWorker(cat, visited, adj, 0)
sum := 0.0
for _, r := range results {
    sum += math.Pow(float64(r.hops), -5.0)
}
```

This is the right approach for native targets because they don't have
a plan evaluation framework — the aggregation is just code.

Both strategies produce the same output. The distinction matters for
the compiler architecture, not the user.

## Scoping: What aggregate_all/3 Means

In standard Prolog:

```prolog
aggregate_all(Template, Goal, Result)
```

- `Goal` is called; all solutions are collected
- `Template` is evaluated for each solution
- `Result` is the aggregation of all template values

Templates we should support:

| Template | Result | SQL equivalent |
|----------|--------|---------------|
| `sum(X)` | Sum of X values | `SUM(x)` |
| `count` | Number of solutions | `COUNT(*)` |
| `min(X)` | Minimum X | `MIN(x)` |
| `max(X)` | Maximum X | `MAX(x)` |
| `bag(X)` | List of all X values | `ARRAY_AGG(x)` |
| `set(X)` | Unique X values | `ARRAY_AGG(DISTINCT x)` |

Phase 1 should support `sum`, `count`, `min`, `max`. Phase 2 adds
`bag` and `set`. Phase 3 adds grouped aggregation.

## Non-Goals

- Replacing `findall/3` or `setof/3` — those are list-collection
  predicates, not aggregation. `aggregate_all` subsumes them for
  the analytical use case.
- Supporting arbitrary templates — focus on the standard aggregate
  functions that map to SQL/LINQ equivalents.
- Nested aggregation — `aggregate_all` inside `aggregate_all`. This
  is out of scope for now.
