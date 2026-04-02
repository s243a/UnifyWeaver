# aggregate_all/3 Compilation: Implementation Plan

This document revises the earlier proposal at commit `a25e0a7`.
Reference version:

```text
git show a25e0a7:docs/proposals/AGGREGATE_ALL_COMPILATION_PLAN.md
```

## Overview

This plan adds `aggregate_all/3` compilation to UnifyWeaver, starting
with scalar aggregation over recursive goals and building toward
grouped recursive aggregation. The primary evaluation workload is
category influence propagation.

## Phase 0: Pattern Detection and Normalization (shared core)

**Goal**: Detect `aggregate_all/3` in clause bodies and classify the
template, goal, and grouping structure in a shared way that C#, Go,
and Rust can all consume.

**Location**:
- `src/unifyweaver/core/advanced/pattern_matchers.pl`
- with possible extraction/reuse from existing aggregate parsing in
  `src/unifyweaver/targets/csharp_target.pl`

### Step 0.1: Template parser

```prolog
:- module(pattern_matchers, [
    ...,
    classify_aggregate/2,       % +AggTerm, -AggInfo
    parse_aggregate_template/3  % +Template, -Op, -Expr
]).

parse_aggregate_template(sum(Expr), sum, Expr).
parse_aggregate_template(count, count, 1).
parse_aggregate_template(min(Expr), min, Expr).
parse_aggregate_template(max(Expr), max, Expr).
parse_aggregate_template(bag(Expr), bag, Expr).
parse_aggregate_template(set(Expr), set, Expr).
```

### Step 0.2: Goal classifier

Analyze the goal inside `aggregate_all` to determine compilation
strategy:

```prolog
classify_aggregate_goal(Goal, GoalInfo) :-
    conjunction_to_list(Goal, Goals),
    partition(is_recursive_goal, Goals, RecGoals, NonRecGoals),
    partition(is_arithmetic_goal, NonRecGoals, ArithGoals, RelGoals),
    GoalInfo = agg_goal{
        recursive: RecGoals,
        arithmetic: ArithGoals,
        relational: RelGoals
    }.
```

### Step 0.3: Grouping detection

When `aggregate_all` appears inside a clause, variables bound before
the aggregate call define implicit grouping:

```prolog
detect_aggregate_grouping(ClauseHead, PreGoals, AggGoal, GroupVars) :-
    term_variables(ClauseHead, HeadVars),
    term_variables(PreGoals, PreBound),
    term_variables(AggGoal, AggVars),
    intersection(PreBound, AggVars, GroupVars).
```

**Depends on**: nothing (can start immediately)

## Phase 1: C# Query Engine — Reuse and Generalize Existing Aggregate IR

**Goal**: Reuse the existing `AggregateNode` / `AggregateSubplanNode`
runtime support and generalize compiler-side `aggregate_all/3`
lowering around it.

**Location**:
- `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs`
- `src/unifyweaver/targets/csharp_target.pl`

### Step 1.1: Audit current support

Before adding runtime surface, audit what already exists:

- `AggregateNode`
- `AggregateSubplanNode`
- current compiler-side parsing of `aggregate_all`
- current tests for aggregate support in C#

### Step 1.2: Fill the semantic gaps

```csharp
case AggregateNode agg:
    var inputRows = ExecuteNode(agg.Input, context).ToList();
    if (agg.GroupIndex is null)
    {
        // Scalar
        var value = agg.Op switch
        {
            AggregateOp.Sum => inputRows.Sum(r => ToDouble(r[agg.ValueIndex])),
            AggregateOp.Count => (double)inputRows.Count,
            AggregateOp.Min => inputRows.Min(r => ToDouble(r[agg.ValueIndex])),
            AggregateOp.Max => inputRows.Max(r => ToDouble(r[agg.ValueIndex])),
            _ => throw new NotSupportedException()
        };
        result = new[] { new object[] { value } };
    }
    else
    {
        // Grouped
        result = inputRows
            .GroupBy(r => r[agg.GroupIndex.Value])
            .Select(g => new object[] { g.Key!, Reduce(g, agg.Op, agg.ValueIndex) });
    }
    break;
```

### Step 1.3: Shared plan builder path

In `csharp_target.pl`, stop treating aggregate parsing as a partially
special path and align it with the shared aggregate classification from
Phase 0.

```prolog
build_aggregate_plan(AggInfo, SubPlan, AggregatePlan) :-
    get_dict(op, AggInfo, Op),
    get_dict(expr, AggInfo, Expr),
    % ... determine value index from Expr position in sub-plan schema
    AggregatePlan = aggregate{
        type: aggregate,
        input: SubPlan,
        op: Op,
        value_index: ValueIndex
    }.
```

### Step 1.4: Emitter and tests

```prolog
emit_plan_expression(Node, Expr) :-
    is_dict(Node, aggregate), !,
    get_dict(input, Node, Input),
    get_dict(op, Node, Op),
    get_dict(value_index, Node, ValueIndex),
    emit_plan_expression(Input, InputExpr),
    aggregate_op_csharp(Op, OpStr),
    format(atom(Expr),
        'new AggregateNode(~w, AggregateOp.~w, ~w)',
        [InputExpr, OpStr, ValueIndex]).
```

Start with:

- non-recursive `sum/count/min/max`
- recursive scalar aggregate over a single recursive predicate
- arithmetic post-processing over recursive output

**Depends on**: Phase 0

## Phase 2: Go Target — Aggregation Wrapper

**Goal**: Generate Go wrapper functions that aggregate over compiled
recursive predicate output.

**Location**: `src/unifyweaver/targets/go_target.pl`

### Step 2.1: Detect `aggregate_all/3` in clause body

When compiling a predicate whose body contains `aggregate_all`, the
Go target should:
1. Compile the recursive sub-goal as a worker function (existing)
2. Generate an aggregation wrapper that calls the worker and reduces

### Step 2.2: Generate aggregation loop

```prolog
emit_go_aggregate_wrapper(AggInfo, WorkerName, Code) :-
    get_dict(op, AggInfo, sum),
    format(atom(Code),
        'sum := 0.0\nfor _, r := range ~w(...) {\n  sum += ~w\n}\n',
        [WorkerName, ValueExpr]).
```

### Step 2.3: Test

Use the same semantic cases as C#:

- non-recursive sanity
- recursive scalar aggregate
- arithmetic post-processing over recursive output

**Depends on**: Phase 0

## Phase 3: Rust Target — Aggregation Wrapper

**Goal**: Same as Go, for Rust.

**Location**: `src/unifyweaver/targets/rust_target.pl`

Mirrors Phase 2:
- Detect `aggregate_all` in clause body
- Generate `iter().map().sum()` or equivalent Rust idioms
- Rust's iterator combinators make this particularly clean:
  ```rust
  let sum: f64 = results.iter()
      .filter(|(anc, _)| anc == root)
      .map(|(_, hops)| ((hops + 1) as f64).powf(-5.0))
      .sum();
  ```

**Depends on**: Phase 0

## Phase 4: Grouped Aggregation

**Goal**: Support `aggregate_all` with implicit grouping (variables
bound before the aggregate call).

### Step 4.1: Extend Phase 0 grouping detection

Emit grouping columns into the AggInfo dict.

### Step 4.2: C# query engine — existing grouping hooks

Use the existing grouping fields/indices in the aggregate plan/runtime
path rather than adding a fresh grouping concept.

### Step 4.3: Native targets — grouped loops

Generate nested loops or map-based grouping:

```go
groups := make(map[string]float64)
for _, r := range results {
    groups[r.root] += math.Pow(float64(r.hops+1), -5.0)
}
```

### Step 4.4: Test with category_influence

The full category influence query:
```prolog
category_influence(Root, Score) :-
    root_category(Root),
    aggregate_all(sum(W),
        (article_category(_, Cat),
         category_ancestor(Cat, Root, Hops),
         W is (Hops + 1) ^ (-5)),
        Score),
    Score > 0.
```

This should compile to all targets and produce results matching the
pipeline benchmark.

**Depends on**: Phases 1-3

## Phase 5: Non-Recursive Aggregation Sanity

**Goal**: Verify that `aggregate_all` works over non-recursive goals
(simple fact scans). This is simpler than the recursive case and
should be tested first as a sanity check.

```prolog
test_sum(Total) :-
    aggregate_all(sum(V), test_scores(_, V), Total).
```

This should be implemented before broad recursive aggregation work, as a
simpler semantic baseline.

**Depends on**: Phase 0

## Priority and Dependencies

```
Phase 0 (pattern detection)
  ├── Phase 5 (non-recursive sanity — can start first)
  ├── Phase 1 (C# query engine AggregateNode)
  ├── Phase 2 (Go aggregation wrapper)
  └── Phase 3 (Rust aggregation wrapper)
Phase 4 (grouped aggregation) — after any target completes
```

Phase 0 is the critical path. Phase 5 (non-recursive sanity) should run
first. Phase 1 should focus on normalizing and broadening existing C#
support, not rebuilding it. Phases 2-3 are then the real cross-target
expansion.

## Effort Estimates

| Phase | Effort | Notes |
|-------|--------|-------|
| 0 | Low-Medium | Template parsing, goal classification |
| 1 | Medium | Reuse existing C# aggregate runtime, broaden lowering/tests |
| 2 | Low-Medium | Wrapper generation, pattern exists in pipeline |
| 3 | Low-Medium | Mirrors Go work |
| 4 | Medium | Grouping detection + grouped execution |
| 5 | Low | Simple test case |

## Benchmark Validation

### Phase 1 target

`total_weight("Relativity", "Physics", W)` should produce a value
consistent with the d_eff benchmark's weight sum for that pair.

### Phase 4 target

`category_influence(Root, Score)` for all roots should match the
output from the category influence pipeline benchmark at each scale.

### Performance baseline

The aggregation overhead should be negligible compared to the TC
computation. At 10K scale:
- TC computation: ~1.5s (query engine)
- Aggregation: <100ms expected (single pass over results)
- Total should be competitive with the pipeline benchmark

## Related Work

| Document | Relevance |
|----------|-----------|
| `docs/proposals/AGGREGATE_ALL_COMPILATION_PHILOSOPHY.md` | Why aggregate_all, composition model |
| `docs/proposals/AGGREGATE_ALL_COMPILATION_SPEC.md` | Supported forms, target strategies, test queries |
| `docs/proposals/MODE_DIRECTED_TABLING_PROPOSAL.md` | Min/max inside recursion (complementary) |
| `docs/proposals/COMPUTED_RECURSIVE_INCREMENT_*.md` | Computed increments (composes with aggregation) |
| `docs/design/FULL_TRANSPILATION_AUDIT.md` | Lists aggregate_all as a gap for all targets |
| `examples/benchmark/category_influence.pl` | Primary evaluation workload for Phase 4 |
| `examples/benchmark/effective_distance.pl` | Uses aggregate_all in Prolog source |

## Change Summary

Edited by `gpt-5.4 (medium)`.

Main changes relative to `a25e0a7`:

- corrected Phase 1 from “add AggregateNode” to “audit, reuse, and
  broaden existing C# aggregate support”
- strengthened Phase 0 as shared normalization rather than target-local
  parsing
- promoted non-recursive aggregate sanity checks as a true early gate
- tightened the initial recursive scope to aggregation-friendly,
  single-branch forms
