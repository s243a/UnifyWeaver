# Fuzzy Logic DSL for UnifyWeaver

## Overview

This proposal defines a fuzzy logic domain-specific language (DSL) for UnifyWeaver,
enabling probabilistic scoring and matching in semantic search, bookmark filing,
and other applications where boolean logic is too rigid.

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

```prolog
% f_and: Fuzzy AND - multiply weighted scores
% Formula: score * w1*t1 * w2*t2 * ...
f_and([w(Term1, Weight1), w(Term2, Weight2), ...])

% f_or: Fuzzy OR - probabilistic sum
% Formula: 1 - (1 - w1*t1) * (1 - w2*t2) * ...
f_or([w(Term1, Weight1), w(Term2, Weight2), ...])

% f_dist_or: Distributed OR - score AND (t1 OR t2 ...)
% Formula: 1 - (1 - score*w1*t1) * (1 - score*w2*t2) * ...
f_dist_or([w(Term1, Weight1), w(Term2, Weight2), ...])

% f_not: Fuzzy NOT - complement
% Formula: 1 - score
f_not(Term)
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

## Syntax

### Core Form (Verbose)

The core form uses explicit `w/2` functors:

```prolog
f_and([w(bash, 0.9), w(shell, 0.5)])
f_or([w(bash, 0.9), w(shell, 0.5), w(scripting, 0.3)])
```

Unweighted terms (weight 1.0) can omit the wrapper:

```prolog
f_and([bash, shell])           % Expanded to: f_and([w(bash, 1.0), w(shell, 1.0)])
```

### Composition

Fuzzy and boolean operations can be nested:

```prolog
% Fuzzy boost with boolean filter
f_and([
    w(bash, 0.9),
    and([is_type(tree), has_account(s243a)])
])

% Fuzzy OR with subtree constraint
query(Q) :-
    f_dist_or([w(bash, 0.9), w(shell, 0.5)]),
    in_subtree("Unix & Linux").

% Complex query
boost_query(
    f_and([
        f_dist_or([w(bash, 0.9), w(shell, 0.5)]),
        not(in_subtree("Puppylinux"))
    ])
).
```

## Convenience Operators Module

For concise syntax, an **optional** operator module provides syntactic sugar:

```prolog
:- module(fuzzy_ops, [
    op(600, xfy, :),    % Weight notation (Term:Weight -> w(Term, Weight))
    op(400, xfy, &),    % Fuzzy AND
    op(400, xfy, \/),   % Fuzzy OR
    op(400, xfy, |/),   % Distributed OR
    op(200, fy, ~)      % Fuzzy NOT
]).

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

With `fuzzy_ops` imported:

```prolog
:- use_module(fuzzy_ops).

% Colon notation (expands to w/2)
f_and([bash:0.9, shell:0.5])           % -> f_and([w(bash,0.9), w(shell,0.5)])
f_or([bash:0.9, shell:0.5])            % -> f_or([w(bash,0.9), w(shell,0.5)])

% Infix operators (alternative to functor form)
boost(bash:0.9 & shell:0.5).           % -> f_and([w(bash,0.9), w(shell,0.5)])
boost(bash:0.9 \/ shell:0.5).          % -> f_or([w(bash,0.9), w(shell,0.5)])
boost(bash:0.9 |/ shell:0.5).          % -> f_dist_or([w(bash,0.9), w(shell,0.5)])
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

For set-union semantics:
```
f_union([w(a,w1), w(b,w2)]) = base_score * f_or([w(a,w1), w(b,w2)])
                            = base_score * (1 - (1-w1*a)(1-w2*b))
```

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
    semantic_search("bash-reduce", Candidates),
    apply_boost(Candidates, f_and([w(bash, 0.9)]), Boosted),
    apply_filter(Boosted, in_subtree("Unix"), Filtered),
    top_k(Filtered, 10, Result).
```

Or with the optional `fuzzy_ops` module:

```prolog
:- use_module(fuzzy_ops).

file_bookmark("bash-reduce", Result) :-
    semantic_search("bash-reduce", Candidates),
    apply_boost(Candidates, f_and([bash:0.9]), Boosted),  % sugar
    apply_filter(Boosted, in_subtree("Unix"), Filtered),
    top_k(Filtered, 10, Result).
```

Compiled to Python via UnifyWeaver.

## Implementation Plan

### Phase 1: Core Functors
1. Define `f_and/1`, `f_or/1`, `f_dist_or/1`, `f_not/1` in Prolog
2. Define `and/1`, `or/1`, `not/1` for boolean operations
3. Implement evaluation predicates

### Phase 2: Python Target
1. Add fuzzy logic to UnifyWeaver's Python code generator
2. Generate NumPy-based implementations
3. Test with bookmark filing use case

### Phase 3: Operator Module
1. Create `fuzzy_ops` convenience module
2. Implement term expansion for operators
3. Document operator precedence

### Phase 4: Advanced Features
1. Add `f_union/1` for non-distributed OR
2. SQL compilation target
3. Streaming evaluation for large datasets

## File Structure

```
src/prolog/
  fuzzy/
    core.pl          # f_and, f_or, f_dist_or, f_not
    boolean.pl       # and, or, not
    predicates.pl    # is_type, has_account, in_subtree
    operators.pl     # Optional operator sugar
    eval.pl          # Evaluation/scoring

targets/
  python/
    fuzzy_codegen.py # Generate NumPy code
  sql/
    fuzzy_codegen.sql # Generate SQL (future)
```

## Open Questions

1. Should weights be normalized (sum to 1) or raw?
2. How to handle missing/null scores in fuzzy operations?
3. Should `f_dist_or` be renamed to clarify semantics (e.g., `f_blend_or`)?
4. Precedence of `&` vs `\/` operators?

## References

- Zadeh, L.A. (1965). "Fuzzy Sets". Information and Control.
- Klir, G.J. & Yuan, B. (1995). "Fuzzy Sets and Fuzzy Logic: Theory and Applications".
- UnifyWeaver transpiler documentation: `docs/ARCHITECTURE.md`
