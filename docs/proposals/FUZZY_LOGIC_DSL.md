# Fuzzy Logic DSL for UnifyWeaver

## Overview

This proposal defines a fuzzy logic domain-specific language (DSL) for UnifyWeaver,
enabling probabilistic scoring and matching in semantic search, bookmark filing,
and other applications where boolean logic is too rigid.

## Quick Start

```prolog
% Load the fuzzy logic DSL
:- use_module('src/unifyweaver/fuzzy/fuzzy').

% Example: Evaluate fuzzy AND with term scores
?- eval_fuzzy_expr(
       f_and([w(bash, 0.9), w(shell, 0.5)]),
       [bash-0.8, shell-0.6],
       Result
   ).
% Result = 0.216  (0.9*0.8 * 0.5*0.6)

% Example: Fuzzy OR
?- eval_fuzzy_expr(
       f_or([w(bash, 0.9), w(shell, 0.5)]),
       [bash-0.8, shell-0.6],
       Result
   ).
% Result = 0.804  (1 - (1-0.72)(1-0.3))

% Example: With operator syntax (after loading operators)
:- use_module('src/unifyweaver/fuzzy/operators').
?- fuzzy_and(bash:0.9 & shell:0.5, Result).
```

## Motivation

Embedding models don't always recognize domain-specific terms (e.g., "bash-reduce").
Users need to boost or filter results using fuzzy logic operations:

- **Fuzzy AND**: Require multiple terms with varying importance
- **Fuzzy OR**: Match any of several alternatives
- **Distributed OR**: Combine base score with alternatives
- **Boolean matching**: Filter by metadata (type, account, tags)

A Prolog-based DSL allows:
1. Declarative query composition
2. Compilation to multiple targets (Python, SQL, etc.)
3. Mixing fuzzy and boolean operations
4. Integration with UnifyWeaver's existing transpiler

## Core Functors

### Weighted Terms (Core Form)

The canonical representation uses explicit `w/2` functors:

```prolog
% w(Term, Weight) - a term with its weight
w(bash, 0.9)
w(shell, 0.5)

% Unweighted terms use weight 1.0
w(bash, 1.0)   % or just: bash (expanded by preprocessor)
```

### Fuzzy Logic Operations

Each functor has two forms:
- **Symbolic** (no Result): for building expressions, used by operator sugar
- **Evaluation** (with Result): for computing scores

This mirrors Prolog arithmetic: `2 + 3` is symbolic, `X is 2 + 3` evaluates.

```prolog
% f_and: Fuzzy AND - multiply weighted scores
% Formula: w1*t1 * w2*t2 * ...
f_and([w(Term1, Weight1), w(Term2, Weight2), ...])          % Symbolic
f_and([w(Term1, Weight1), w(Term2, Weight2), ...], Result)  % Evaluation

% f_or: Fuzzy OR - probabilistic sum
% Formula: 1 - (1 - w1*t1) * (1 - w2*t2) * ...
f_or([w(Term1, Weight1), w(Term2, Weight2), ...])           % Symbolic
f_or([w(Term1, Weight1), w(Term2, Weight2), ...], Result)   % Evaluation

% f_dist_or: Distributed OR - base score distributed into each term, then OR
% Formula: 1 - (1 - Base*w1*t1) * (1 - Base*w2*t2) * ...
% Note: f_dist_or(1.0, Terms, R) is equivalent to f_or(Terms, R)
f_dist_or(BaseScore, [w(Term1, Weight1), w(Term2, Weight2), ...])          % Symbolic
f_dist_or(BaseScore, [w(Term1, Weight1), w(Term2, Weight2), ...], Result)  % Evaluation

% f_union: Non-distributed OR - base multiplies the OR result
% Formula: Base * (1 - (1-w1*t1)(1-w2*t2)...)
f_union(BaseScore, [w(Term1, Weight1), w(Term2, Weight2), ...])            % Symbolic
f_union(BaseScore, [w(Term1, Weight1), w(Term2, Weight2), ...], Result)    % Evaluation

% f_not: Fuzzy NOT - complement
% Formula: 1 - Score
f_not(Expr)                % Symbolic
f_not(Score, Result)       % Evaluation
```

### Evaluating Symbolic Expressions

```prolog
% eval_fuzzy/3 - evaluate a symbolic fuzzy expression
eval_fuzzy(Expr, Context, Result) :-
    ...

% Example usage
query(Query, Result) :-
    Expr = f_and([w(bash, 0.9), w(shell, 0.5)]),  % Build symbolic
    semantic_search(Query, Context),
    eval_fuzzy(Expr, Context, Result).            % Evaluate

% Operators build symbolic, then evaluate
query2(Query, Result) :-
    semantic_search(Query, Ctx),
    eval_fuzzy(bash:0.9 & shell:0.5, Ctx, Result).  % Operator -> symbolic -> eval
```

### Boolean Metadata Operations

```prolog
% and: Boolean AND - all conditions must match
and([Condition1, Condition2, ...])

% or: Boolean OR - any condition matches
or([Condition1, Condition2, ...])

% not: Boolean NOT - negation
not(Condition)
```

### Metadata Predicates

```prolog
is_type(tree)              % Match item type
is_type(pearl)
has_account(s243a)         % Match account
has_parent(bash)           % Match parent folder name
in_subtree("BASH (Unix/Linux)")  % Match path contains
has_tag(linux)             % Match tag (future)
```

### Hierarchical Filters

Filters for navigating tree structure:

```prolog
% Relationship filters (return 1.0 or 0.0)
child_of(Node)             % Direct children only
descendant_of(Node)        % Any depth below Node (alias: in_subtree)
parent_of(Node)            % The immediate parent
ancestor_of(Node)          % Any depth above Node
sibling_of(Node)           % Same parent as Node
root_of(Tree)              % Root nodes of a tree/account

% Depth filters
has_depth(N)               % Exactly at depth N
depth_between(Min, Max)    % Depth in range [Min, Max]
depth_at_least(N)          % Depth >= N
depth_at_most(N)           % Depth <= N

% Path pattern matching
path_matches(Pattern)      % Glob-style: "*/STEM/*/Python"
path_regex(Regex)          % Regex on full path

% Distance-based (fuzzy, returns score in [0,1])
near(Node, Decay)          % Score decays with tree distance
                           % near(X, 0.5) = 0.5^distance
```

### Custom Filters

Users can define custom filters using standard Prolog:

```prolog
% Define a custom filter predicate
% Must return Score in [0.0, 1.0] for fuzzy, or boolean for crisp
:- module(my_filters, [
    is_programming_related/2,
    recent_item/2
]).

% Custom fuzzy filter: programming-related items
is_programming_related(Item, Score) :-
    item_path(Item, Path),
    (  path_contains(Path, "Programming") -> Score = 1.0
    ;  path_contains(Path, "STEM") -> Score = 0.7
    ;  path_contains(Path, "Tech") -> Score = 0.5
    ;  Score = 0.0
    ).

% Custom boolean filter: items modified recently
recent_item(Item, Score) :-
    item_modified(Item, Timestamp),
    days_ago(Timestamp, Days),
    (  Days < 7 -> Score = 1.0
    ;  Days < 30 -> Score = 0.5
    ;  Score = 0.0
    ).
```

### Registering Custom Filters

```prolog
% Register for use in fuzzy expressions
:- register_filter(is_programming_related/2).
:- register_filter(recent_item/2).

% Now usable in queries
f_and([
    w(python, 0.9),
    w(is_programming_related, 1.0),
    w(recent_item, 0.5)
], Result).
```

### Filter Combinators

Build complex filters from simpler ones:

```prolog
% Define compound filter
my_scope(Item, Score) :-
    child_of("Programming", Item, S1),
    is_type(tree, Item, S2),
    Score is S1 * S2.  % Fuzzy AND

% Or using DSL combinators
:- define_filter(my_scope,
    f_and([
        child_of("Programming"),
        is_type(tree)
    ])
).

% Parameterized filters
in_account_subtree(Account, Subtree, Item, Score) :-
    has_account(Account, Item, S1),
    in_subtree(Subtree, Item, S2),
    Score is S1 * S2.

% Partial application for reuse
:- define_filter(my_bash_folders,
    in_account_subtree(s243a, "BASH (Unix/Linux)")
).
```

### Filter Interface

All filters implement a common interface:

```prolog
% filter(+Item, -Score)
% Item: the item being evaluated
% Score: 0.0-1.0 for fuzzy, 0.0 or 1.0 for boolean

% Built-in filters follow this pattern
child_of(Parent, Item, Score) :-
    item_parent(Item, ItemParent),
    (  ItemParent == Parent -> Score = 1.0
    ;  Score = 0.0
    ).

% Curried form for use in f_and/f_or
child_of(Parent) :-
    current_item(Item),
    child_of(Parent, Item, Score),
    assert_score(Score).
```

## Syntax

### Core Form (Verbose)

The core form uses explicit `w/2` functors with Result argument:

```prolog
f_and([w(bash, 0.9), w(shell, 0.5)], Result)
f_or([w(bash, 0.9), w(shell, 0.5), w(scripting, 0.3)], Result)
f_dist_or(BaseScore, [w(bash, 0.9), w(shell, 0.5)], Result)
```

Unweighted terms (weight 1.0) can omit the wrapper:

```prolog
f_and([bash, shell], Result)   % Expanded to: f_and([w(bash, 1.0), w(shell, 1.0)], Result)
```

### Composition

Fuzzy and boolean operations can be chained:

```prolog
% Fuzzy boost with boolean filter
query(Query, Result) :-
    semantic_search(Query, BaseScores),
    f_and([w(bash, 0.9), w(is_type(tree), 1.0)], TermScores),
    multiply_scores(BaseScores, TermScores, Result).

% Fuzzy OR with subtree constraint
query(Query, Result) :-
    semantic_search(Query, BaseScores),
    f_dist_or(BaseScores, [w(bash, 0.9), w(shell, 0.5)], Boosted),
    apply_filter(Boosted, in_subtree("Unix & Linux"), Result).

% Complex query with pipeline
boost_query(Query, Result) :-
    semantic_search(Query, S0),
    f_dist_or(S0, [w(bash, 0.9), w(shell, 0.5)], S1),
    apply_filter(S1, not(in_subtree("Puppylinux")), S2),
    top_k(S2, 10, Result).
```

## Convenience Operators Module

For concise syntax, an **optional** operator module provides syntactic sugar:

```prolog
:- module(fuzzy_operators, [...]).

% Operator definitions (separate directives, not in export list)
:- op(400, xfy, &).      % Fuzzy AND
:- op(400, xfy, v).      % Fuzzy OR (using 'v' to avoid conflict with Prolog's \/)
:- op(200, fy, ~).       % Fuzzy NOT

% Note: : (colon) is not redefined as an operator to avoid conflict
% with Prolog's module qualification. Use : inside lists only.

% Colon expansion: Term:Weight -> w(Term, Weight)
expand_weighted(Term:Weight, w(Term, Weight)) :- !.
expand_weighted(Term, w(Term, 1.0)).   % Bare term -> weight 1.0

% Expand list of weighted terms
expand_weighted_list([], []).
expand_weighted_list([H|T], [HExp|TExp]) :-
    expand_weighted(H, HExp),
    expand_weighted_list(T, TExp).

% Term expansion for fuzzy functors
user:term_expansion(f_and(List), f_and(Expanded)) :-
    is_list(List),
    expand_weighted_list(List, Expanded).
user:term_expansion(f_or(List), f_or(Expanded)) :-
    is_list(List),
    expand_weighted_list(List, Expanded).
user:term_expansion(f_dist_or(List), f_dist_or(Expanded)) :-
    is_list(List),
    expand_weighted_list(List, Expanded).

% Operator expansion for &, \/, |/
user:term_expansion(A & B, Expanded) :-
    expand_fuzzy_and(A & B, Expanded).
user:term_expansion(A \/ B, Expanded) :-
    expand_fuzzy_or(A \/ B, Expanded).
```

### Usage with Operators

With `fuzzy_operators` imported:

```prolog
:- use_module(fuzzy/operators).

% Colon notation (expands to w/2) - works in lists
f_and([bash:0.9, shell:0.5])           % -> f_and([w(bash,0.9), w(shell,0.5)])
f_or([bash:0.9, shell:0.5])            % -> f_or([w(bash,0.9), w(shell,0.5)])

% Infix operators (alternative to functor form)
fuzzy_and(bash:0.9 & shell:0.5, R).    % -> f_and([w(bash,0.9), w(shell,0.5)], R)
fuzzy_or(bash:0.9 v shell:0.5, R).     % -> f_or([w(bash,0.9), w(shell,0.5)], R)

% Direct expansion
expand_fuzzy(bash:0.9 & shell:0.5, Expanded).
% Expanded = f_and([w(bash, 0.9), w(shell, 0.5)])
```

### Without Operators (Pure Core)

Without importing `fuzzy_ops`, use the verbose form:

```prolog
f_and([w(bash, 0.9), w(shell, 0.5)])   % No sugar, explicit w/2
```

This avoids any potential conflicts with `:` (module qualification) or other operators.

## Mathematical Foundations

### Fuzzy AND (Product T-norm)

```
f_and([w(a,w1), w(b,w2)]) = (w1 * a) * (w2 * b)
```

For score blending:
```
result = base_score * (w1 * sim(query, a)) * (w2 * sim(query, b))
```

### Fuzzy OR (Probabilistic Sum)

```
f_or([w(a,w1), w(b,w2)]) = 1 - (1 - w1*a) * (1 - w2*b)
```

Equivalent to:
```
f_or([a, b]) = a + b - a*b  (when weights = 1)
```

### Distributed OR

The base score is distributed into each term before OR:

```
f_dist_or([w(a,w1), w(b,w2)]) with base_score S:
  = 1 - (1 - S*w1*a) * (1 - S*w2*b)
```

This is NOT equivalent to `S * f_or([w(a,w1), w(b,w2)])` because multiplication
does not distribute over fuzzy OR:

```
S * (a OR b) = S * (a + b - ab) = Sa + Sb - Sab
(S*a) OR (S*b) = Sa + Sb - S²ab
```

These differ by a factor of S in the last term (Sab vs S²ab).

Note: Fuzzy AND and OR are both associative; it's distributivity that fails.

### Non-Distributed OR (Union) - Future

For set-union semantics, base score multiplies the whole OR result:

```prolog
f_union(BaseScore, [w(a,w1), w(b,w2)], Result)
% Result = BaseScore * f_or([w(a,w1), w(b,w2)])
%        = BaseScore * (1 - (1-w1*a)(1-w2*b))
```

**Comparison of f_or, f_union, and f_dist_or:**

```
f_or([a, b], R)        →  R = a + b - ab
f_union(S, [a,b], R)   →  R = S*(a + b - ab) = Sa + Sb - Sab
f_dist_or(S, [a,b], R) →  R = 1-(1-Sa)(1-Sb) = Sa + Sb - S²ab
```

The difference is the interaction term: `Sab` (union) vs `S²ab` (dist_or).

- Use `f_or` when combining term scores without a base
- Use `f_union` when scaling combined OR by base (multiplicative)
- Use `f_dist_or` when blending base into each term before OR

## Compilation Targets

### Python (NumPy)

```python
# f_and([w(bash, 0.9), w(shell, 0.5)])
def fuzzy_and(scores, weights):
    result = np.ones(len(scores[0]))
    for s, w in zip(scores, weights):
        result *= w * s
    return result

# f_dist_or([w(bash, 0.9), w(shell, 0.5)])
def fuzzy_dist_or(base_score, scores, weights):
    result = np.ones(len(base_score))
    for s, w in zip(scores, weights):
        result *= (1 - base_score * w * s)
    return 1 - result
```

### SQL (Future)

```sql
-- f_and([w(col1, 0.9), w(col2, 0.5)])
SELECT *, (0.9 * col1_score) * (0.5 * col2_score) AS fuzzy_score
FROM results;

-- Boolean and()
SELECT * FROM results
WHERE type = 'tree' AND account = 's243a';
```

### Rust (Future)

```rust
fn fuzzy_and(scores: &[&[f64]], weights: &[f64]) -> Vec<f64> {
    scores[0].iter().enumerate().map(|(i, _)| {
        scores.iter().zip(weights).fold(1.0, |acc, (s, w)| acc * w * s[i])
    }).collect()
}
```

## Integration with Bookmark Filing

### Current Python Implementation

The `infer_phone.py` script implements fuzzy logic directly:

```bash
python infer_phone.py --query "bash-reduce" \
    --boost-and "bash:0.9" \
    --boost-or "shell:0.5,scripting:0.3"
```

(The CLI uses `term:weight` string syntax, parsed into the internal representation.)

### Future Prolog Query

```prolog
file_bookmark("bash-reduce", Result) :-
    semantic_search("bash-reduce", BaseScores),
    f_and([w(bash, 0.9)], BoostScores),
    multiply_scores(BaseScores, BoostScores, Boosted),
    apply_filter(Boosted, in_subtree("Unix"), Filtered),
    top_k(Filtered, 10, Result).

% Or using f_dist_or for distributed boost
file_bookmark_v2("bash-reduce", Result) :-
    semantic_search("bash-reduce", BaseScores),
    f_dist_or(BaseScores, [w(bash, 0.9), w(shell, 0.5)], Boosted),
    apply_filter(Boosted, in_subtree("Unix"), Filtered),
    top_k(Filtered, 10, Result).
```

Or with the optional `fuzzy_ops` module (colon syntax):

```prolog
:- use_module(fuzzy_ops).

file_bookmark("bash-reduce", Result) :-
    semantic_search("bash-reduce", BaseScores),
    f_dist_or(BaseScores, [bash:0.9, shell:0.5], Boosted),  % : sugar
    apply_filter(Boosted, in_subtree("Unix"), Filtered),
    top_k(Filtered, 10, Result).
```

Compiled to Python via UnifyWeaver.

## Implementation Status

### Completed: Core Prolog DSL
All core fuzzy logic operations are implemented and tested:

```
src/unifyweaver/fuzzy/
  fuzzy.pl         # Main module (re-exports all)
  core.pl          # f_and, f_or, f_dist_or, f_union, f_not
  boolean.pl       # b_and, b_or, b_not
  predicates.pl    # is_type, has_account, in_subtree, hierarchical filters
  operators.pl     # Optional operator sugar (& for AND, v for OR)
  eval.pl          # Evaluation engine with context management
  test_fuzzy.pl    # Test suite
```

### Test Results

All tests pass:

```
=== Fuzzy Logic DSL Tests ===

--- Core Operations ---
f_and: PASS (0.216)        # 0.9*0.8 * 0.5*0.6
f_or: PASS (0.804)         # 1 - (1-0.72)(1-0.3)
f_dist_or(0.7): PASS (0.60816)
f_union(0.7): PASS (0.5628)
f_not: PASS (0.7)

--- Eval Module ---
eval_fuzzy_expr: PASS (0.216)
fallback score: PASS (0.5)  # Default when no term score set

--- Operators Module ---
expand_fuzzy AND: PASS
fuzzy_and: PASS (0.216)
fuzzy_or: PASS (0.804)

--- Boolean Operations ---
b_and: PASS
b_or: PASS
```

### Remaining Implementation

#### Phase 2: Python Target ✅ COMPLETE
1. ✅ Add fuzzy logic to UnifyWeaver's Python code generator (`python_fuzzy_target.pl`)
2. ✅ Generate NumPy-based implementations (`fuzzy_logic.py`)
3. ✅ Test with bookmark filing use case (17 tests pass)

Files added:
- `src/unifyweaver/targets/python_fuzzy_target.pl` - Prolog code generator
- `src/unifyweaver/targets/python_runtime/fuzzy_logic.py` - NumPy runtime
- `src/unifyweaver/targets/python_runtime/test_fuzzy_logic.py` - Test suite

#### Phase 3: SQL Target (Future)
1. SQL compilation target for database queries
2. Streaming evaluation for large datasets

## Open Questions

1. ~~Should weights be normalized (sum to 1) or raw?~~
   **Decision**: Use raw weights. Normalization is an optional post-processing step.
2. How to handle missing/null scores in fuzzy operations?
   Current: fallback to 0.5 (neutral score) when term not found.
3. Should `f_dist_or` be renamed to clarify semantics (e.g., `f_blend_or`)?
4. ~~Precedence of `&` vs `\/` operators?~~
   **Resolved**: Both have precedence 400. Using `v` instead of `\/` to avoid Prolog builtin conflict.

## References

- Zadeh, L.A. (1965). "Fuzzy Sets". Information and Control.
- Klir, G.J. & Yuan, B. (1995). "Fuzzy Sets and Fuzzy Logic: Theory and Applications".
- UnifyWeaver transpiler documentation: `docs/ARCHITECTURE.md`
