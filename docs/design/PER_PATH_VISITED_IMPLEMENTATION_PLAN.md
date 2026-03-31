# Per-Path Visited Recursion: Cross-Target Implementation Plan

## Context

PR #1063 introduced:
- **Design doc**: `docs/design/PER_PATH_VISITED_RECURSION.md` — theory,
  taxonomy, compilation strategies
- **Core pattern detection**: `is_per_path_visited_pattern/4` in
  `src/unifyweaver/core/advanced/pattern_matchers.pl` — detects the
  `\+ member(X, Visited)` + `[X|Visited]` idiom, returns Visited
  parameter position
- **AWK DFS implementation**: All-simple-paths via explicit stack with
  per-path visited encoded as comma-delimited string

This document specifies how to wire the pattern detection into each
target's compilation pipeline, so that when a user writes
`category_ancestor/4` with a Visited list, the compiler automatically
generates cycle-safe code in the target language.

## Current State per Target

### Already Implemented

| Target | Status | Commit | Approach | Limitation |
|--------|--------|--------|----------|------------|
| **AWK** | ✅ Complete | `4adf62e3` | DFS with stack + path string | Hardcoded in `compile_general_recursive_to_awk`; not wired to pattern detection |
| **Go** | ⚠️ Partial | `d731adf5` | Recursive function + `map[string]bool` copy | Generates visited code for ALL general recursion, not specifically when Visited pattern is detected |
| **Python** | ⚠️ Partial | `6e0cdc32` | Recursive generator + `visited set` | Same — generates visited code for all arity-3 general recursion |
| **C# Query** | ✅ For counted closure | local | `PathAwareTransitiveClosureNode` + DFS + copied `HashSet` per branch | Covers the canonical counted transitive-closure shape, but not generic visited-list lowering |

### Not Yet Implemented

| Target | Effort | Notes |
|--------|--------|-------|
| **Rust** | Low | Has `classify_goal_sequence` framework; needs arity-3 deepening first (task #16), then visited-set pattern |
| **C/C++** | Medium | Has native deepening; needs DFS stack approach (like AWK) or recursive with `malloc`'d visited sets |
| **TypeScript** | Low | Has native deepening; `Set<string>` + recursive generator |
| **Haskell** | Low | Has native deepening; `Set` from `Data.Set` + recursive list |
| **Lua** | Medium | Has native deepening; table-based visited set |
| **Others** | Variable | Perl, Ruby, F#, Elixir, Clojure, Scala, Kotlin all have native deepening |

## Key Files

| File | Role |
|------|------|
| `src/unifyweaver/core/advanced/pattern_matchers.pl` | `is_per_path_visited_pattern/4` — shared pattern detection |
| `src/unifyweaver/core/clause_body_analysis.pl` | `classify_goal_sequence/3` — needs `\+` negation-as-guard |
| `src/unifyweaver/targets/go_target.pl:6520-6700` | Go visited-set code generation |
| `src/unifyweaver/targets/python_target.pl:6023-6120` | Python visited-set code generation |
| `src/unifyweaver/targets/awk_target.pl:110-245` | AWK DFS stack implementation |
| `docs/design/PER_PATH_VISITED_RECURSION.md` | Theory + strategy reference |

## Specification

### What the Pattern Detection Returns

```prolog
is_per_path_visited_pattern(Name, Arity, Clauses, VisitedPos)
% Example: is_per_path_visited_pattern(category_ancestor, 4, Clauses, 4)
%   → VisitedPos = 4 (the 4th argument is the Visited list)
```

### What Each Target Must Generate

Given a predicate `pred/N` where argument `VisitedPos` is the Visited
list, the target should generate code that:

1. **Exposes an API without the Visited parameter**: The external
   function has arity `N-1`. The Visited parameter is initialized
   internally (empty set/list) and threaded through recursive calls.

2. **Per-branch copy semantics**: Each recursive branch gets its own
   copy of the visited set. Sibling branches must not interfere.
   - Go: `newVisited := make(map[string]bool, ...)` copy
   - Python: `visited | {node}` creates new frozenset
   - Rust: `visited.clone()` or `HashSet::from_iter`
   - AWK: path string concatenation (implicit copy)

3. **Membership check before recursing**: Check that the next node is
   not in the current path's visited set before recursing.

4. **Multiple results**: The function yields/returns ALL results found
   across all branches (all simple paths), not just the first or shortest.

### External vs Internal Signatures

```
Prolog source:       category_ancestor(Cat, Ancestor, Hops, Visited)
External API:        category_ancestor(Cat) → [(Ancestor, Hops), ...]
Internal recursive:  _category_ancestor_worker(Cat, Visited) → [(Ancestor, Hops), ...]
```

The compiler strips the Visited parameter from the public API and
initializes it in the wrapper.

## Generalization Beyond Arity-3

**Important note for the implementing agent**: The current Go and Python
visited-set code is hardcoded for arity-3 predicates (e.g., in
`generate_ternary_worker` and `compile_ternary_recursive_go`). When
wiring the pattern detection into each target, look for opportunities
to **generalize beyond fixed arity**:

1. `is_per_path_visited_pattern/4` already works for any arity ≥ 3 —
   it returns `VisitedPos` regardless of total arity.

2. The generated code should use `VisitedPos` to determine which argument
   is the visited set, and treat all other arguments positionally — not
   assume arity is exactly 3 or 4.

3. Consider predicates like:
   ```prolog
   reach(Source, Target, Hops, Cost, Visited) :- ...  % arity 5
   ```
   The pattern detection will return `VisitedPos = 5`. The generated code
   should handle 3 output args (Target, Hops, Cost), not just 2.

4. The external API should have arity `N-1` (stripping Visited), and the
   result type should be a tuple of all non-input, non-visited args.

5. Use the **mode declaration** (if available via `user:mode/1`) to
   distinguish input (+), output (-), and visited (list) parameters.
   When no mode is declared, infer from the pattern: first arg is input,
   last arg (VisitedPos) is visited, rest are outputs.

The arity-specific handlers (`generate_ternary_worker`,
`compile_ternary_recursive_go`, etc.) should ideally be refactored into
a single arity-generic handler parameterized by `VisitedPos` and the
list of output positions. This refactoring can be done incrementally —
first wire the pattern detection, then generalize the code generation.

## Implementation Steps

### Step 1: Wire Pattern Detection into Target Dispatch

For each target, add a check in the recursive compilation dispatch
chain:

```prolog
% In target's compile_predicate_to_X_normal:
;   is_per_path_visited_pattern(Pred, Arity, Clauses, VisitedPos) ->
    format('Type: per_path_visited_recursion (visited at position ~w)~n', [VisitedPos]),
    compile_per_path_visited_to_X(Pred, Arity, Clauses, VisitedPos, Options, Code)
```

This should be checked BEFORE the general recursion handler so it takes
priority when the Visited pattern is detected.

### Step 2: Extract Compilation Parameters

From the clauses and VisitedPos, extract:
- **Step relation**: The non-recursive predicate in the body
  (e.g., `category_parent/2`)
- **Result positions**: Which head args are outputs (excluding VisitedPos)
- **Counter arithmetic**: The `is/2` expression for hop counting
- **Base case constant**: The constant in the base case (e.g., `1`)

Use existing `extract_goals_list` and `split_body_at_recursive_call/5`
helpers.

### Step 3: Generate Target Code

#### Strategy A: Recursive + Set (Go, Python, Rust, TypeScript, etc.)

```
function pred_worker(input, visited):
    if input in visited: return []
    new_visited = visited ∪ {input}
    results = []
    for each (mid) in step_relation(input):
        results.add((mid, base_constant))          # base case
        for each (ancestor, h) in pred_worker(mid, new_visited):
            results.add((ancestor, h + increment)) # recursive case
    return results

function pred(input):
    return pred_worker(input, empty_set())
```

#### Strategy B: DFS + Stack (AWK, C, C++)

```
function pred(source):
    build adjacency list from step relation
    stack = [(source, 0, "source")]
    results = []
    while stack not empty:
        (cur, hops, path) = stack.pop()
        for each neighbor of cur:
            if neighbor not in path:
                results.add((source, neighbor, hops+1))
                stack.push((neighbor, hops+1, path+","+neighbor))
    return results
```

#### Strategy C: Parameterized Query (C# Query Engine)

Current state: the C# query engine now recognizes the canonical counted
transitive-closure shape and lowers it to
`PathAwareTransitiveClosureNode`, evaluated with DFS plus a copied
visited set per branch.

That means the main semantic bug for counted reachability on cyclic
graphs is solved without extending the generic `FixpointNode`.

Remaining work for C# Query is broader than this specialized case:

- Lower explicit visited-list Prolog patterns into query plans
- Generalize beyond the current `edge + recursive call + arithmetic`
  shape
- Re-benchmark against the DFS pipeline and aggregation workloads

### Step 4: Add `\+` Negation-as-Guard to classify_goal_sequence

In `clause_body_analysis.pl`, add a classification clause for negated
goals:

```prolog
classify_single_goal_(\+ Goal, VarMap, guard(\+ Goal, NegExpr)) :-
    % Negated goal with no output variables → guard condition
    term_variables(Goal, GoalVars),
    forall(member(V, GoalVars), lookup_var(VarMap, V, _)),
    render_negation(Goal, NegExpr).
```

This allows targets using `classify_goal_sequence` to handle
`\+ member(X, Visited)` as a guard condition rather than a passthrough.

### Step 5: Test

For each target:
1. Compile `category_ancestor/4` (with Visited) → target code
2. Execute on dev dataset (198 edges, 121 categories)
3. Compare path counts against SWI-Prolog reference
4. Verify no cycles (paths should have no repeated nodes)

## Implementation Priority

Recommended order (easiest first, most impactful first):

| Priority | Target | Why |
|----------|--------|-----|
| 1 | **Go** | Already has visited code; just needs wiring to pattern detection + arity-4 support |
| 2 | **Python** | Same — already has visited code |
| 3 | **Rust** | Needs arity-3 deepening first (task #16), then visited pattern is straightforward |
| 4 | **TypeScript** | `Set<string>` is native; recursive generators via `yield*` |
| 5 | **C/C++** | DFS stack approach; needs memory management for visited sets |
| 6 | **AWK** | Already done (DFS stack with path strings) |
| 7 | **C# Query** | Specialized counted closure is done; remaining work is generic visited-list lowering |
| 8 | **Others** | Perl, Ruby, Haskell, etc. — follow the same Strategy A pattern |

## Testing Strategy

### Correctness Test

```bash
# Reference: SWI-Prolog path count for each (source, ancestor) pair
swipl -g "aggregate_all(count, category_ancestor(X, Y, _, [X]), N), ..."

# Target: should produce same count
./target_binary | awk '{print $1, $2}' | sort | uniq -c
```

### Cycle Safety Test

```bash
# No path should contain repeated nodes
./target_binary | while read src anc hops; do
    # verify no node appears twice in the path that produced this tuple
done
```

### Performance Test (at 50K scale)

| Metric | Target |
|--------|--------|
| Wall-clock time | Total for all paths from all sources |
| Peak memory | Max RSS during execution |
| Path count | Total simple paths found |
| Max path length | Longest simple path discovered |

## References

- `docs/design/PER_PATH_VISITED_RECURSION.md` — Theory and strategy overview
- `docs/design/RECURSIVE_TEMPLATE_LOWERING.md` — Native deepening architecture
- `docs/design/CLAUSE_BODY_ANALYSIS_EXTRACTION.md` — Shared clause body analysis
- PR #1063 — Pattern detection + AWK DFS implementation
- PR #1056 — Go/AWK arity-3 deepening (`d731adf5`, `e791d043`)
- PR #1054 — Python arity-3 deepening (`6e0cdc32`)
- PR #1057 — C# parameterized query engine fix (`1106388c`)
