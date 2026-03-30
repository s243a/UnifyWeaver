# Proposal: Demand-Driven Optimization for the Prolog Target

## Problem Statement

In the cross-target effective distance benchmark (PR #1054–#1087), the
Prolog target is the slowest of the interpreted targets:

| Target | Execute (300 articles) | Visited Set |
|--------|----------------------|-------------|
| CPython | 0.73s | `frozenset` O(1) |
| Prolog | 1.26s | `member/2` O(n) |
| AWK | 2.46s | `in` array O(1) — still slow |

Two optimizations were tested:
1. **Dict-based visited** (O(1) lookup): 1.26s vs 1.28s — negligible
   improvement because visited sets are small (≤10 elements at max_depth=10)
2. **Tabling**: 2.58s — 2x worse because the Visited list makes every
   call unique, preventing cache hits

Neither addresses the real bottleneck: **Prolog explores all paths from
every source category**, including branches that can never reach the root.
With 6008 category edges, most DFS branches lead to dead ends that waste
computation.

## Proposed Solution: Demand Analysis

Apply the same demand-driven optimization that the C# parameterized query
engine uses (`FixpointNode` + `ParamSeedNode` + demand closure) to the
Prolog target. Instead of generating a C# query plan, generate **optimized
Prolog code** with pre-computed demand guards.

### How the C# Query Engine Does It

The C# target already implements demand analysis:

- **Demand closure** (commit `dce3945b`): For parameterized recursive
  predicates, computes which tuples are "needed" by working backwards
  from the query
- **ParamSeedNode** (commit `e738fee3`): Seeds the evaluation with known
  input parameter bindings
- **Semi-naive fixpoint**: Only processes newly discovered (delta) tuples

Key file: `src/unifyweaver/targets/csharp_target.pl` — the
`build_pipeline_seeded` and demand closure predicates.

### What the Prolog Target Would Generate

Given the user's declarative Prolog:

```prolog
%% User writes:
effective_distance(Article, Root, Deff) :-
    root_category(Root),
    path_to_root(Article, Root, Hops), ...

category_ancestor(Cat, Ancestor, Hops, Visited) :-
    category_parent(Cat, Mid),
    \+ member(Mid, Visited),
    category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.
```

The optimizer would generate:

```prolog
%% Compiled Prolog (auto-generated):

%% Pre-computed demand set: categories that can reach root
:- dynamic can_reach_root/1.

init_demand :-
    root_category(Root),
    assert(can_reach_root(Root)),
    repeat,
    (   category_parent(Child, Parent),
        can_reach_root(Parent),
        \+ can_reach_root(Child)
    ->  assert(can_reach_root(Child)), fail
    ;   !
    ).

%% Optimized: only explore categories that can reach root
category_ancestor(Cat, Ancestor, Hops, Visited) :-
    category_parent(Cat, Mid),
    can_reach_root(Mid),          % ← demand guard (auto-inserted)
    \+ member(Mid, Visited),
    category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.

:- initialization(init_demand).
```

The `can_reach_root/1` guard prunes branches immediately, avoiding DFS
into categories that can never contribute to the final answer.

## O(1) Optimization Opportunities

Beyond demand analysis, consider these O(1) improvements:

### 1. Dict-Based Visited (O(1) per check)

Replace `member/2` list with SWI-Prolog dicts:
```prolog
\+ get_dict(Mid, Visited, _)      % O(1) instead of O(n)
put_dict(Mid, Visited, true, New)  % O(1) copy-on-write
```
Negligible at depth ≤ 10, significant if max_depth increases.
Requires SWI-Prolog 7+ (built-in, no packages).

### 2. Assert-Based Demand Set (O(1) per check)

`can_reach_root/1` uses Prolog's first-argument indexing:
```prolog
can_reach_root('Physics').  % O(1) hash lookup via arg indexing
can_reach_root('Quantum_mechanics').
...
```
SWI-Prolog indexes `dynamic` predicates on the first argument,
giving O(1) amortized lookup.

### 3. Compiled Demand Guard (no runtime overhead)

If the demand set is known at compile time (root is fixed), the
optimizer can inline the guard as a fact table — no runtime
computation needed.

## Relationship to C# Parameterized Query Engine

| Concept | C# Query Engine | Prolog Target (proposed) |
|---------|----------------|--------------------------|
| Demand analysis | `build_pipeline_seeded` | `init_demand` pre-computation |
| Input seeding | `ParamSeedNode` | `root_category(Root)` binding |
| Demand guard | Join condition in plan | `can_reach_root(Mid)` guard |
| Fixpoint evaluation | `FixpointNode` + delta sets | Prolog backtracking (native) |
| Cycle detection | `HashSet<T>` dedup | `Visited` list / dict |

The key insight: **the demand analysis is target-independent**. The same
backward reachability computation drives both the C# plan construction
and the Prolog guard insertion. The difference is only the output format.

## Architecture

```
User Prolog source
    ↓
[Shared demand analysis]  ← extracted from C# query engine pipeline
    ↓
┌─────────────────────────┐
│ C# target: query plan   │ (existing)
│ Prolog target: opt code │ (proposed)
│ Go target: pruned DFS   │ (future)
│ etc.                    │
└─────────────────────────┘
```

## Expected Impact

At 300 articles (6008 edges), roughly 1500 categories are under Physics
but only ~300 can actually reach the root via `category_parent`. The
demand guard would prune ~80% of DFS branches, potentially reducing
Prolog execution from 1.26s to ~0.25s (estimated, needs verification).

At larger scales the impact grows — more categories means more dead-end
branches to prune.

## Prior Work

- C# demand closure: `dce3945b` — `feat(csharp_query): demand-closure for
  parameterized mutual SCCs`
- C# ParamSeedNode: `e738fee3` — `perf(csharp-query): seed transitive
  closure for parameters`
- Benchmark results: PRs #1054–#1087 (cross-target effective distance)
- Prolog optimization doc: `docs/design/PROLOG_TARGET_OPTIMIZATION.md`
- Per-path visited design: `docs/design/PER_PATH_VISITED_RECURSION.md`
