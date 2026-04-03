# Prolog Target Optimization

## Context

Prolog is a compilation target like any other in UnifyWeaver. The source
Prolog (what the user writes) should remain declarative — using standard
idioms like `\+ member(X, Visited)` and `[X|Visited]`. The **compiled**
Prolog (what UnifyWeaver generates for the Prolog target) can differ,
using SWI-specific optimizations while preserving semantics.

## Current Status

The Prolog target now has an initial SWI-Prolog-specific branch-pruning
lowering for parameterized per-path visited recursion.

When the compiler sees all of the following:
- a recursive predicate matching `is_per_path_visited_pattern/4`
- a concrete `user:mode/1` declaration with at least one non-visited input
- SWI-Prolog as the target dialect

it can generate:
- a normal recursive worker
- a pruned worker used when the mode inputs are bound
- a tabled prune relation over the driver input and invariant inputs
- a small cache/guard layer for grounded prune checks

This is conceptually similar to the C# parameterized query engine's
demand seeding: known inputs restrict the search space before full
recursive exploration. The generated code is still ordinary Prolog,
just with helper predicates such as `'$worker'`, `'$pruned'`, and
`'$prune'`.

## Implemented Strategy

### 1. Mode-Directed Branch Pruning

For a predicate shaped roughly like:

```prolog
category_ancestor(Cat, Ancestor, Hops, Visited) :-
    category_parent(Cat, Ancestor),
    \+ member(Ancestor, Visited),
    Hops = 1.

category_ancestor(Cat, Ancestor, Hops, Visited) :-
    category_parent(Cat, Mid),
    Mid \= blocked,
    \+ member(Mid, Visited),
    category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.

:- mode(category_ancestor(-, +, -, +)).
```

the compiler can derive a prune relation keyed by the bound query inputs.
If those inputs are known at call time, the generated wrapper dispatches
to a pruned worker that checks whether a branch can still reach the
requested answer before recursing.

- **Benefit**: Avoids exploring branches that cannot satisfy the bound query
- **Impact**: Best when one or more answer-shaping inputs are fixed
- **SWI-specific**: Uses `:- table` and a dynamic cache helper
- **Requires**: The visited argument must be identified from the actual
  per-path pattern, not guessed from the first `+` mode position

## Current Performance

At 10x scale (195 articles, 3932 edges, max-depth=10):
- SWI-Prolog with `member/2` list: 0.68s
- SWI-Prolog with dict visited: 0.66s (negligible difference at this scale)
- Tabling: 2.58s (worse — Visited list makes every call unique)

## Optimization Strategies

### 2. Dict-Based Visited Set (O(1) lookup)

Replace `\+ member(X, Visited)` with `\+ get_dict(X, Visited, _)` and
`[X|Visited]` with `put_dict(X, Visited, true, NewVisited)`.

- **Benefit**: O(1) lookup vs O(n) for lists
- **Impact**: Negligible at depth ≤ 10 (lists are short). Significant
  at deeper traversals or if max_depth is increased.
- **SWI-specific**: Dicts are SWI-Prolog 7+. Not portable.
- **Requires**: No extra packages (built-in).

### 3. Root-Reachability Pruning

Precompute which categories can reach the root, then prune DFS branches
that lead to unreachable categories:

```prolog
%% Precompute: which categories can reach Physics?
:- dynamic can_reach_root/1.
precompute_reachability :-
    root_category(Root),
    assert(can_reach_root(Root)),
    repeat,
    (   category_parent(Child, Parent),
        can_reach_root(Parent),
        \+ can_reach_root(Child)
    ->  assert(can_reach_root(Child)), fail
    ;   !
    ).

%% In DFS, only explore neighbors that can reach root
category_ancestor(Cat, Mid, ...) :-
    category_parent(Cat, Mid),
    can_reach_root(Mid),  % prune unreachable branches
    ...
```

- **Benefit**: Massive pruning — only explore categories on paths to root
- **Impact**: Potentially orders of magnitude at large scale
- **Requires**: Pre-processing step (one-time BFS from root backwards)

This is now partially implemented in a more general, mode-directed form:
rather than precomputing a single fixed root relation, the generated
Prolog can derive a prune relation from the bound inputs of the query.

### 4. Cuts for Early Termination

Add cuts (`!`) to prevent exploring branches once sufficient paths are
found. For d_eff with n=5, paths beyond ~3x the shortest contribute
negligibly:

```prolog
category_ancestor(Cat, Ancestor, Hops, Visited) :-
    max_depth(MaxD),
    length(Visited, Depth),
    Depth < MaxD, !,  % already have this
    ...
```

Could also add: once shortest path to root is found, only explore paths
up to `K * shortest` depth.

### 5. Tabling (NOT recommended for the main recursive predicate)

Tested: tabling `category_ancestor/4` is slower (2.58s vs 0.68s) because
the Visited list makes every call unique, preventing cache hits. Tabling
is effective for standard transitive closure but not per-path visited.

The current implementation only tables the derived prune helper, where
the arguments are the demand-signature inputs rather than the full
visited path.

## Prolog-Specific Packages

| Package | Ships with SWI | Purpose |
|---------|---------------|---------|
| `library(assoc)` | Yes | AVL trees, O(log n) — not needed if dicts available |
| `library(rbtrees)` | Yes | Red-black trees, O(log n) — same |
| `library(tabling)` | Yes | Auto-memoization — not suitable for this pattern |
| SWI dicts | Built-in (7+) | O(1) hash — best option for visited sets |
| `library(ordsets)` | Yes | Sorted lists — O(n) but with fast merge |

No external packages needed. All optimizations use built-in SWI features.

## Relationship to Other Targets

The Prolog target optimization mirrors what other targets do:
- Go/Rust/C#: `HashSet` / `map` → O(1) visited (already implemented)
- Python: `frozenset` → O(1) visited (already implemented)
- AWK: string scan → O(n) visited (needs optimization, see
  `docs/design/PER_PATH_VISITED_RECURSION.md`)

The UnifyWeaver compiler now does this for one useful subset:
parameterized per-path recursion with explicit mode information in the
SWI target. Other optimizations in this note remain future work.
