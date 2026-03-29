# Design: Per-Path Visited Recursion

## Problem

Standard transitive closure deduplicates on `(source, target)` pairs — each
pair is discovered once. But the effective distance formula
`d_eff = (Σ dᵢ^(-n))^(-1/n)` requires **all simple paths** between nodes,
where each path has a different length that contributes to the weighted sum.

A "simple path" is one that visits no node more than once. In a cyclic graph,
without this constraint, there are infinitely many paths (loops can be
traversed arbitrarily many times).

Prolog handles this naturally via a **Visited list** threaded through each
recursive branch:

```prolog
category_ancestor(Cat, Parent, 1, Visited) :-
    category_parent(Cat, Parent),
    \+ member(Parent, Visited).

category_ancestor(Cat, Ancestor, Hops, Visited) :-
    category_parent(Cat, Mid),
    \+ member(Mid, Visited),
    category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.
```

Key semantics:
- Each branch of backtracking gets its own independent copy of `Visited`
- A node can appear in different branches (different paths) but not twice
  on the same path
- `\+ member(Mid, Visited)` is negation-as-failure membership check
- `[Mid|Visited]` extends the path before recursing

## Taxonomy of Recursion Patterns

| Pattern | State | Dedup | Use Case |
|---------|-------|-------|----------|
| Standard TC | None | `(source, target)` | Reachability |
| Counted TC | Counter | `(source, target, hops)` | Shortest path |
| Tail recursion | Accumulator | N/A (loop) | Aggregation |
| Global memoization | Cache | Input → Output | Pure functions |
| **Per-path visited** | **Visited set per branch** | **Node ∉ current path** | **All simple paths** |

Per-path visited is distinct because:
1. State is **per-branch**, not global
2. The same computation can yield **multiple results** for the same input
   (different paths to the same target with different lengths)
3. Requires **negation** (`\+ member`) which interacts with the visited set

## Target Compilation Strategies

### Strategy A: Recursive Function with Set Parameter (Go, Python, Rust)

Languages with recursion and set/hashset types can directly translate the
Prolog pattern:

```python
def category_ancestor(cat, visited=None):
    if visited is None:
        visited = frozenset()
    for parent in category_parent[cat]:
        if parent not in visited:
            yield (parent, 1)
            # Recurse with extended visited set (immutable copy per branch)
            for (ancestor, h) in category_ancestor(parent, visited | {parent}):
                yield (ancestor, h + 1)
```

Key requirements:
- **Immutable or copied visited set per branch** — `visited | {parent}`
  creates a new set for each recursive call, so sibling branches don't
  interfere
- **Generator/yield pattern** — multiple results per input
- **Set membership test** — `parent not in visited`

This is what Go and Python already generate (with `visited` parameter),
but it needs to be recognized as a **pattern** that the compiler extracts
from the Prolog `\+ member(X, Visited)` + `[X|Visited]` idiom.

### Strategy B: Explicit DFS with Stack (AWK, C)

Languages without recursion (AWK) or where explicit stack control is
preferred (C for performance) simulate DFS:

```awk
# Stack-based DFS with per-path visited encoded as string
sp = 1
stack_node[sp] = source
stack_hops[sp] = 0
stack_path[sp] = source  # comma-separated path

while (sp > 0) {
    cur = stack_node[sp]; hops = stack_hops[sp]; path = stack_path[sp]
    sp--
    for each neighbor of cur:
        if neighbor not in path:
            record (source, neighbor, hops + 1)
            sp++
            stack_node[sp] = neighbor
            stack_hops[sp] = hops + 1
            stack_path[sp] = path "," neighbor
}
```

Key requirements:
- **Explicit stack arrays** — AWK associative arrays indexed by stack pointer
- **Path encoding** — visited set encoded as delimited string for containment
  check via `index()`
- **No recursion limit** — stack depth bounded by graph diameter

### Strategy C: Semi-Naive with Path Tracking (C# Query Engine)

The C# parameterized query engine uses `FixpointNode` with delta sets.
To support per-path visited, the tuples would need to carry path
information:

- Tuple: `(source, ancestor, hops, path_hash)` where `path_hash`
  encodes the visited set
- Deduplication on the full tuple including path_hash
- Join condition: new intermediate node not in path

This is more complex and may not be worth implementing in the IR —
the C# query engine could instead generate a custom C# function that
does DFS directly, bypassing the plan-based evaluation for this pattern.

## Pattern Recognition

The compiler should recognize this pattern from Prolog source:

```
% Indicators of per-path visited recursion:
1. A list parameter that grows by consing: [X|Visited]
2. Negated membership check: \+ member(X, Visited)
3. The parameter is threaded through the recursive call
4. Base case checks the same negation pattern
```

## Existing Infrastructure

Key findings from codebase analysis:

1. **`classify_goal_sequence/3`** (`clause_body_analysis.pl`) does NOT handle
   `\+` negation — negated goals fall through to `passthrough`. Negation-as-
   guard classification needs to be added.

2. **`split_body_at_recursive_call/5`** (`pattern_matchers.pl`) can identify
   where the recursive call sits in the body — useful for finding the Visited
   parameter position.

3. **`is_tail_recursive_accumulator/2`** (`pattern_matchers.pl`) detects
   accumulator patterns but uses a simple heuristic (arity-3, position 2).
   The Visited list pattern is structurally different (list cons + negated
   membership).

4. **Go target** is the only target with explicit visited-set code generation.
   The pattern needs to be generalized across targets.

5. **TypR target** has "threaded context arguments" (positions 3+) for
   invariant state in mutual recursion — conceptually similar but not
   per-path mutable state.

## Pattern Recognition

The compiler should recognize the Visited-list pattern from these indicators
in the Prolog source:

```prolog
%% Indicators:
%% 1. A list parameter that grows via cons: [X|Visited]
%% 2. Negated membership check: \+ member(X, Visited)
%% 3. The list parameter is threaded through the recursive call
%% 4. Base case also has a negated membership check

is_per_path_visited_pattern(Name, Arity, Clauses, VisitedPos) :-
    % Separate base and recursive clauses
    partition(is_recursive_clause_for(Name), Clauses, RecClauses, BaseClauses),
    RecClauses \= [], BaseClauses \= [],

    % Find the Visited parameter position:
    % In the recursive clause, find which head arg appears in [X|Arg]
    member((RecHead, RecBody), RecClauses),
    RecHead =.. [Name|RecArgs],

    % Find the arg position that has list-cons in body
    nth1(VisitedPos, RecArgs, VisitedVar),
    body_contains_cons(RecBody, _, VisitedVar),  % [_|VisitedVar]

    % Verify negated membership check exists
    body_contains_negated_member(RecBody, _, VisitedVar),

    % Verify the extended list is passed to recursive call
    body_recursive_call_arg(RecBody, Name, VisitedPos, ExtendedList),
    ExtendedList = [_|VisitedVar].

%% Helper: check body contains \+ member(X, List)
body_contains_negated_member((\+ member(X, List), _), X, List) :- !.
body_contains_negated_member((_, Rest), X, List) :-
    body_contains_negated_member(Rest, X, List).
body_contains_negated_member(\+ member(X, List), X, List).

%% Helper: check body contains [H|T] pattern
body_contains_cons(Body, H, T) :-
    sub_term([H|T], Body).
```

## Compilation Pipeline

1. **Detect** the `\+ member(X, Visited)` + `[X|Visited]` pattern via
   `is_per_path_visited_pattern/4`
2. **Extract** which argument position holds the visited list (`VisitedPos`)
3. **Generate** target-specific code using Strategy A, B, or C:
   - Strategy A targets receive the Visited position and generate a set
     parameter with copy-on-branch semantics
   - Strategy B targets receive the adjacency structure and generate DFS
     with stack-encoded path strings
4. **External API omits** the visited parameter — it's internal state
   managed by the generated code

The external signature is `category_ancestor(Cat, Ancestor, Hops)` — the
Visited parameter is an implementation detail that the compiler manages.

## Interaction with Effective Distance

For the benchmark, the full pipeline is:

```
category_ancestor/4 (with Visited)
    ↓ per-path visited recursion
    yields all (Cat, Ancestor, Hops) triples for simple paths
    ↓
effective_distance/3
    aggregates: sum(Hops^(-5)) per (Article, Root) pair
    ↓
d_eff = WeightSum^(-1/5)
```

The per-path visited ensures finite enumeration (no cycles), and the
aggregation combines all path lengths into the dimensional distance metric.

## Implementation Priority

| Target | Strategy | Effort | Notes |
|--------|----------|--------|-------|
| Python | A (recursive + frozenset) | Low | Already has visited param, just needs pattern recognition |
| Go | A (recursive + map copy) | Low | Already has visited param |
| Rust | A (recursive + HashSet clone) | Low | Needs arity-3 deepening first |
| AWK | B (explicit DFS stack) | Medium | No recursion, needs stack simulation |
| C# Query | C (custom function) | High | May bypass FixpointNode for this pattern |
| C/C++ | B (explicit DFS stack) | Medium | Stack-based for performance |

## Prior Work

- `src/unifyweaver/core/clause_body_analysis.pl` — classify_goal_sequence
  framework with multifile hooks (native deepening)
- `docs/design/RECURSIVE_TEMPLATE_LOWERING.md` — recursive template → native
  lowering architecture
- `docs/design/CLAUSE_BODY_ANALYSIS_EXTRACTION.md` — extraction of shared
  clause body analysis from typR
- Go/Python already generate `visited` parameter — this design formalizes
  the pattern and adds compiler-level recognition
