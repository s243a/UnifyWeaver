# Specification: Demand-Driven Prolog Target Optimization

## Overview

Extract the demand analysis from the C# parameterized query engine into
a shared optimization pass, then apply it to generate optimized Prolog
code with pre-computed demand guards.

## Inputs

1. **User's Prolog source**: Declarative predicates with recursive
   transitive closure and per-path visited tracking
2. **Query binding**: Which arguments are known at query time
   (e.g., `root_category(Physics)`)
3. **Fact base**: The materialized `category_parent/2` and
   `article_category/2` relations

## Outputs

1. **Demand set**: The set of categories that can reach the root
   (backward reachability from the query binding)
2. **Optimized Prolog**: User's predicates with demand guards inserted
3. **Initialization code**: Prolog directives to pre-compute the
   demand set at load time

## Demand Analysis Algorithm

```
Input: root_category(Root), category_parent(Child, Parent) facts

1. Initialize demand set D = {Root}
2. Repeat until fixpoint:
   For each category_parent(Child, Parent) where Parent ∈ D:
     Add Child to D
3. Output D as can_reach_root/1 facts
```

This is backward reachability — the same computation as the C#
query engine's demand closure but producing Prolog facts instead
of a query plan node.

## Guard Insertion Rules

For each clause in the user's recursive predicate that calls
`category_parent(Cat, Mid)` followed by a recursive call:

```prolog
%% Original:
pred(Cat, ...) :-
    category_parent(Cat, Mid),
    pred(Mid, ...).

%% Optimized:
pred(Cat, ...) :-
    category_parent(Cat, Mid),
    can_reach_root(Mid),    % ← inserted
    pred(Mid, ...).
```

The guard is inserted **after** the step relation and **before** the
recursive call, ensuring we only recurse into categories that can
contribute to the answer.

## Detection Criteria

The optimizer should apply demand guards when:

1. The query has a **fixed target** — a root/goal category known at
   compile time (from `root_category/1` or mode declarations)
2. The predicate has **recursive transitive closure** — detected by
   `is_per_path_visited_pattern/4` or similar
3. The step relation is a **fact table** — `category_parent/2` etc.

## Interface with C# Query Engine

### What to Extract

From `src/unifyweaver/targets/csharp_target.pl`:

- **Demand closure computation** (around `build_pipeline_seeded`):
  The backward reachability from seeded parameters
- **Predicate dependency analysis**: Which predicates feed into the
  recursive call
- **Parameter binding propagation**: How input modes (`+`) propagate
  through the predicate call graph

### What to Keep Target-Specific

- **C# target**: Continues to generate `ParamSeedNode` in the query plan
- **Prolog target**: Generates `can_reach_root/1` facts + guard clauses
- **Go/Rust/Python**: Could generate a precomputed `reachable` set
  initialized at startup

### Shared Module

Create `src/unifyweaver/core/demand_analysis.pl`:

```prolog
:- module(demand_analysis, [
    compute_demand_set/4,        % +StepRelation, +Target, +Direction, -DemandSet
    insert_demand_guards/4,      % +Clauses, +DemandPred, +StepRel, -OptClauses
    generate_demand_init/3       % +DemandSet, +DemandPred, -InitCode
]).
```

## Correctness Criteria

1. **Sound**: The optimized code produces a **subset** of the original
   code's results — specifically, only results that include the target
   root. No valid results are lost.
2. **Complete**: Every result the original code produces that involves
   the target root is also produced by the optimized code.
3. **Equivalent d_eff**: The effective distance values must match the
   unoptimized version within floating-point tolerance (1e-6).

## Testing

1. Compare optimized Prolog output against unoptimized on dev dataset
   (19 articles) — must be identical
2. Compare against Python/Go/Rust reference at 300 scale — must match
3. Measure speedup: expect 4-5x at 300 articles, more at larger scales
