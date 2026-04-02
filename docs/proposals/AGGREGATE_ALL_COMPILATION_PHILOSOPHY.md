# aggregate_all/3 Compilation: Philosophy

This document revises the earlier proposal at commit `a25e0a7`.
Reference version:

```text
git show a25e0a7:docs/proposals/AGGREGATE_ALL_COMPILATION_PHILOSOPHY.md
```

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

Compiling `aggregate_all/3` closes this gap. More precisely, it closes
the gap for targets that still require hand-written aggregation wrappers
around already-compiled recursive predicates. The C# query engine is not
starting from zero here: it already has `AggregateNode` and
`AggregateSubplanNode` support in the runtime and partial compiler-side
handling of aggregate forms. The real work is to make `aggregate_all/3`
a coherent cross-target compilation feature rather than a partial C#
special case.

## Why This Matters

Without general `aggregate_all/3` compilation, UnifyWeaver is still
primarily a **recursive relation compiler** with some target-specific
aggregation exceptions. With it, UnifyWeaver becomes a more general
**analytical query compiler**.

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
fast, but outside the partial C# path the user still needs to write
aggregation code by hand.

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

This separation means we do not need to redesign the recursive
machinery first. We need a more general layer that consumes its output
consistently across targets.

## Relationship to Existing Features

### Mode-directed tabling (min/max)

Tabling applies aggregation **inside** the recursion — pruning branches
during DFS. It's an optimization, not a general aggregation mechanism.

`aggregate_all` applies aggregation **after** the recursion — reducing
the complete result set. It's the general mechanism.

They compose naturally:
- Use `min` tabling inside the TC for shortest-path pruning
- Use `aggregate_all(sum(...))` outside the TC for influence scores

### Pipeline approach (`generate_pipeline.py`)

The pipeline approach generates complete native programs that include
both recursion AND aggregation. It works, but it's not compilation —
it's code generation from a template, bypassing UnifyWeaver's compiler.

`aggregate_all` compilation should eventually let UnifyWeaver emit the
same code that `generate_pipeline.py` produces, but from Prolog source
and through the ordinary compiler path.

### Existing C# aggregation support

The current repository already has:

- `AggregateNode`
- `AggregateSubplanNode`
- compiler-side parsing of several aggregate forms in
  [csharp_target.pl](/home/s243a/Projects/UnifyWeaver/context/gemini/UnifyWeaver/src/unifyweaver/targets/csharp_target.pl)
- runtime execution in
  [QueryRuntime.cs](/home/s243a/Projects/UnifyWeaver/context/gemini/UnifyWeaver/src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs)

So the philosophical question is not “should C# invent an aggregation
node?” but rather:

- how do we normalize aggregate detection and semantics across targets?
- how much of the existing C# aggregation IR should become the shared
  conceptual model?
- which forms should be wrapper code on native targets versus plan
  nodes in the query engine?

### C# query engine `GroupedTransitiveClosureNode`

The grouped TC node handles a specific case: transitive closure with
invariant columns (grouping by label/category). This is related but
different — it groups the recursive computation, not the aggregation.

`aggregate_all` with grouping would sit on TOP of the grouped TC,
reducing each group's results to a scalar.

## Design Principle: Wrapper vs Node

There are two strategies for compiling `aggregate_all`:

**Strategy A: Aggregation as a plan node** (C# query engine)

Use the existing `AggregateNode`/`AggregateSubplanNode` plan concepts
as the query-engine realization of `aggregate_all/3`:

```
AggregateSubplanNode(sum, valueIndex=2,
    subplan=PathAwareTransitiveClosureNode(...))
```

This is the right approach for the query engine because it composes
within the plan evaluation framework and already exists in partial form.
The task is to broaden and regularize it.

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
`bag` and `set` if they prove necessary for real workloads. Phase 3
adds grouped recursive aggregation.

## Non-Goals

- Replacing `findall/3` or `setof/3` wholesale. Those are adjacent but
  different collection semantics. `aggregate_all/3` is the right first
  analytical target, but not a full replacement for every collection
  predicate.
- Supporting arbitrary templates — focus on the standard aggregate
  functions that map to SQL/LINQ equivalents.
- Nested aggregation — `aggregate_all` inside `aggregate_all`. This
  is out of scope for now.

## Change Summary

Edited by `gpt-5.4 (medium)`.

Main changes relative to `a25e0a7`:

- clarified that C# already has partial aggregate runtime/compiler
  support
- reframed the problem from “add AggregateNode” to “generalize and
  unify aggregate_all/3 compilation across targets”
- made the wrapper-vs-node discussion concrete for the existing C#
  architecture
- narrowed some overbroad claims about `findall/3` replacement and
  “no aggregation support”
