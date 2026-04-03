# TypR Per-Path Unique-Path Design

## Goal

Define the next TypR design step for path-aware recursion where correctness
depends on path-local visited state rather than endpoint memoization.

This document is narrower than the cross-target plan in
`PER_PATH_VISITED_IMPLEMENTATION_PLAN.md`. It focuses on:

- what TypR already supports
- what remains unsupported
- what recent C#/Go/Rust path-aware aggregation work implies
- what should be implemented next without adding another brittle matcher family

## Core Semantic Distinction

There are two materially different recursion models:

1. Endpoint memoization
   - caches results by input or endpoint tuple
   - valid for ordinary transitive closure and DP-style recursion
   - not sufficient for all-simple-path or path-sensitive semantics

2. Per-path visited / unique-path semantics
   - recursion state includes the current path's visited set
   - sibling branches must not share visited state
   - required for cycle-safe path enumeration and path-aware accumulation

The second model is not "transitive closure with a different cache key". It is
a different execution state model.

## Current TypR Support

TypR already supports a conservative native per-path slice with:

- `VisitedPos`-driven detection
- mode-driven input/output positions when `user:mode/1` is available
- one recursion-driving input plus conservative invariant non-visited inputs
- weighted outputs with additive numeric accumulation
- seeded and `*_from_vectors` runtime helpers
- `input(stdin|file|vfs|function)` wrappers
- typed runtime parsing for:
  - scalar nodes such as `integer` and `number`
  - conservative pair-of-scalar nodes

Primary implementation points:

- [typr_target.pl](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/src/unifyweaver/targets/typr_target.pl)
- [per_path_visited.mustache](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/templates/targets/typr/per_path_visited.mustache)
- [ppv_definitions.mustache](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/templates/targets/typr/ppv_definitions.mustache)

## Current Boundary

The first helper/guard slice between the step relation and the recursive call
now stays native in TypR.

The next real missing cases are:

- broader invariant-input threading beyond the current conservative shape
- multiple helper/guard stages or richer branch-local path-state control
- runtime-backed step relations that still preserve path-local uniqueness

Examples:

```prolog
category_ancestor_limited(Cat, Ancestor, Hops, Limit, Visited) :-
    category_parent(Cat, Mid),
    Mid =< Limit,
    \+ member(Mid, Visited),
    category_ancestor_limited(Mid, Ancestor, H1, Limit, [Mid|Visited]),
    Hops is H1 + 1.
```

and:

```prolog
category_ancestor_guarded(Cat, Ancestor, Hops, Limit, Visited) :-
    category_parent(Cat, Mid),
    Allowed is Limit - 1,
    Mid =< Allowed,
    \+ member(Mid, Visited),
    category_ancestor_guarded(Mid, Ancestor, H1, Limit, [Mid|Visited]),
    Hops is H1 + 1.
```

These are still structurally per-path, but they already want a more explicit
internal model than the current narrow worker template.

## Why Recent C#/Go/Rust Work Matters

Recent target work in other backends points to the right abstraction.

Relevant recent commits:

- `bdaf9c1e` `feat(csharp-query): add min tabling for path-aware closure`
- `bc89fb28` `feat(csharp-query): add path-aware accumulation node`
- `c01d9c45` `feat(rust): add path-aware accumulation lowering`
- `090bc667` `feat(rust): support recursive aggregate_all lowering`
- `8c11f593` `feat(go): add aggregate_all subplan filters`
- `ad355213` `feat(go): add native min recursion lowering`

What they imply for TypR:

1. Path-aware closure and path-aware accumulation are adjacent problems.
2. Once path identity matters, aggregation is not just post-processing.
3. Endpoint memoization is the wrong abstraction for unique-path work.

So the next TypR step should be designed with path-state and future
path-aware aggregation in mind, even if the immediate implementation remains
conservative.

## Proposed Internal Model

Do not jump straight to a broad new IR. But do make the per-path worker model
explicit enough that it can evolve into one.

For TypR, the next implementation-friendly model should carry:

- recursion-driving input
- invariant inputs
- current visited/path state
- step-produced next node
- local helper/guard values
- yielded outputs

Conceptually:

```text
worker(State, Invariants, Visited) ->
    step(State, Next),
    helper/guard checks,
    not member(Next, Visited),
    yield(BaseOrProjectedResult),
    recurse(Next, Invariants, [Next|Visited])
```

That is enough to cover the next slice without committing yet to a full generic
path-state IR.

## Aggregation-Oriented Follow-Up

With invariant guards in place, the next high-signal per-path extension should
not be more loader work. It should be the first path-aware aggregation shape.

The first conservative aggregation slice now lands for counted unique-path
reachability wrappers of the form:

- `aggregate_all(count, per_path_goal, N)`

where the inner per-path goal already stays native in the current TypR worker
path.

The next conservative candidates are:

- grouped counted unique-path reachability
- min-cost unique-path reachability
- grouped path-aware accumulation over a seeded relation

If TypR can carry path-local state cleanly into those next aggregation slices,
that will tell us whether the current worker model is enough or whether a
broader IR is actually required.

## Non-Goals

Do not mix this design/implementation line with:

- SCC generalization work
- wrapped-R fallback reduction
- unrelated generic producer-family audits
- broader transitive-closure template cleanup

## Recommended Probes

Initial guarded probe:

```prolog
category_ancestor_limited(Cat, Ancestor, Hops, Limit, Visited) :-
    category_parent(Cat, Ancestor),
    Ancestor =< Limit,
    \+ member(Ancestor, Visited),
    Hops = 1.
category_ancestor_limited(Cat, Ancestor, Hops, Limit, Visited) :-
    category_parent(Cat, Mid),
    Mid =< Limit,
    \+ member(Mid, Visited),
    category_ancestor_limited(Mid, Ancestor, H1, Limit, [Mid|Visited]),
    Hops is H1 + 1.
```

Aggregation-oriented follow-up:

```prolog
category_ancestor_min_cost(Cat, Ancestor, Cost, Limit, Visited) :-
    ...
```

## References

- [typr-per-path-next-steps.md](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/docs/handoff/typr-per-path-next-steps.md)
- [PER_PATH_VISITED_IMPLEMENTATION_PLAN.md](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/docs/design/PER_PATH_VISITED_IMPLEMENTATION_PLAN.md)
- [TYPR_TARGET_DESIGN.md](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/docs/design/TYPR_TARGET_DESIGN.md)
