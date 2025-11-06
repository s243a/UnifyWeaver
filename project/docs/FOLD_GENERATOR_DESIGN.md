# Fold Helper Generator - Design Documentation

## Overview

The fold helper generator automatically transforms tree-recursive predicates into a two-phase fold pattern:
1. **Graph builder** (`pred_graph/2`): Builds dependency tree structure
2. **Fold computer** (`fold_pred/2`): Folds over structure to compute result
3. **Wrapper** (`pred_fold/2`): Combines both phases

## Current Implementation Status

**Version**: v0.1 (Basic Framework)
**File**: `src/unifyweaver/core/advanced/fold_helper_generator.pl`
**Status**: Framework complete, transformation logic needs refinement

### What Works
- Module structure and API complete
- Detection of base vs recursive clauses
- Basic clause transformation skeleton
- Test framework in place

### What Needs Work
1. **Graph Builder Transformation**
   - Issue: Variable replacement in body creates malformed expressions
   - Example: `N1 is N - 1` becomes `N1 is N + -1` (incorrect parsing)
   - Root cause: `replace_calls_in_body/7` doesn't preserve non-recursive goals correctly

2. **Fold Computer Generation**
   - Issue: `replace_vars_in_op/4` doesn't properly substitute variables in arithmetic expressions
   - Example: Should replace `F1 + F2` with `VL + VR` but variable unification fails
   - Root cause: `replace_vars_list/3` uses term identity (`==`) which fails for expressions

3. **General vs Specific Trade-off**
   - Current approach: General term transformation (works for any predicate structure)
   - Alternative: Template-based approach (simpler but less general)

## Design Approach: General Term Transformation

### Philosophy
The current implementation attempts to be **fully general** - it should work for any tree-recursive predicate regardless of:
- Number of recursive calls (2, 3, 4+)
- Complexity of body (guards, arithmetic, pattern matching)
- Type of combination operation (addition, multiplication, max, etc.)

### Algorithm

#### Phase 1: Graph Builder Generation

**Input**: Original clause
```prolog
fib(N, F) :- N > 1, N1 is N-1, N2 is N-2, fib(N1,F1), fib(N2,F2), F is F1+F2.
```

**Transformation Steps**:
1. **Identify recursive calls**: `[fib(N1,F1), fib(N2,F2)]`
2. **Create graph variables**: `[G1, G2]`
3. **Replace each recursive call**:
   - `fib(N1, F1)` → `fib_graph(N1, G1)`
   - `fib(N2, F2)` → `fib_graph(N2, G2)`
4. **Remove result computation**: Drop `F is F1+F2` (moves to fold phase)
5. **Build node head**: `fib_graph(N, node(N, [G1, G2]))`

**Expected Output**:
```prolog
fib_graph(N, node(N, [G1, G2])) :- N > 1, N1 is N-1, N2 is N-2, fib_graph(N1,G1), fib_graph(N2,G2).
```

**Current Issue**: Step 3 replaces calls but also corrupts other goals in the body.

**Implementation**: `replace_calls_in_body/7`
- Recursively walks body structure (conjunction, disjunction, etc.)
- Maintains index to track which recursive call we're on
- Problem: Doesn't correctly preserve non-matching goals

#### Phase 2: Fold Computer Generation

**Input**: Original recursive clause body
```prolog
N > 1, N1 is N-1, N2 is N-2, fib(N1,F1), fib(N2,F2), F is F1+F2
```

**Transformation Steps**:
1. **Extract result variables**: `[F1, F2]` from recursive calls
2. **Extract combination operation**: `F is F1+F2`
3. **Create fold variables**: `[VL, VR]`
4. **Generate fold calls**: `fold_fib(L, VL), fold_fib(R, VR)`
5. **Replace variables in expression**: `F is F1+F2` → `F is VL+VR`

**Expected Output**:
```prolog
fold_fib(node(_, [L, R]), F) :- fold_fib(L, VL), fold_fib(R, VR), F is VL+VR.
```

**Current Issue**: Step 5 fails because variable replacement doesn't work on compound expressions.

**Implementation**: `replace_vars_in_op/4` and `replace_vars_list/3`
- Uses `copy_term/2` to create fresh copy
- Attempts to unify variables
- Problem: `==` identity check doesn't work for variables in copied terms

### Why This Approach Is General

1. **Handles arbitrary arity**: Works with 2, 3, 4+ recursive calls
2. **Preserves guards**: Conditional logic (like `N > 1`) stays in graph builder
3. **Arbitrary combination**: Not limited to `+`, works with `*`, `max`, custom operations
4. **Structure-preserving**: Maintains original clause structure

### Implementation Challenges

#### Challenge 1: Term Transformation
Prolog terms are immutable. Replacing sub-terms requires reconstructing the entire term.

**Current Approach**: Recursive descent through term structure
```prolog
replace_calls_in_body(Goal, Pred, GraphPred, RecCalls, GraphVars, Index, NewGoal) :-
    ( functor(Goal, Pred, _) ->
        % This is a recursive call - replace it
        ...
    ; Goal = (A, B) ->
        % Conjunction - recursively process both sides
        ...
    ; Goal = ... ->
        % Handle other structures
    ;
        % Leaf - keep as-is
        NewGoal = Goal
    ).
```

**Problem**: Loses variable bindings and corrupts expressions

**Possible Solutions**:
- Use `term_variables/2` to track all variables
- Build variable substitution map
- Apply substitution using `=..` (univ) operator
- Consider using `copy_term_nat/2` for attributed variables

#### Challenge 2: Variable Substitution

Goal: Replace `[F1, F2]` with `[VL, VR]` in expression `F1 + F2`

**Current Approach**: Walk term and unify
```prolog
replace_vars_list([Old|Olds], [New|News], Term) :-
    ( var(Term), Term == Old ->
        Term = New
    ; true ),
    replace_vars_list(Olds, News, Term).
```

**Problem**: After `copy_term/2`, variables are fresh - `==` always fails

**Better Approach**: Create substitution map first
```prolog
% Build substitution: [F1/VL, F2/VR]
% Apply using maplist or similar
% OR use named templates (mustache pattern)
```

#### Challenge 3: Expression Preservation

Arithmetic expressions like `N - 1` are compound terms: `-(N, 1)`

When transformed incorrectly, they become malformed.

**Root Cause**: The replacement logic treats `-` as binary operator but doesn't preserve its structure.

## Alternative Approach: Template-Based Generation

### Philosophy
Instead of general term transformation, use **templates** for common patterns.

### Trade-offs

| Aspect | General Transformation | Template-Based |
|:-------|:----------------------|:---------------|
| Generality | Works for any predicate | Limited to template patterns |
| Complexity | High (complex term manipulation) | Low (simple pattern matching) |
| Maintainability | Harder to debug | Easier to understand |
| Extensibility | Harder to add patterns | Easy to add templates |
| Correctness | More bugs (term manipulation) | Fewer bugs (simpler logic) |

### Template Approach Design

#### Step 1: Pattern Classification

Classify predicates into templates:
- **Binary operator template**: `F is F1 + F2` (fibonacci, factorial with +)
- **List accumulation template**: `append(L1, L2, L)` (list operations)
- **Maximum template**: `F is max(F1, F2)` (tree max/min)
- **Custom template**: User-defined combination

#### Step 2: Template Application

For fibonacci (binary operator template):

**Graph Builder Template**:
```prolog
fib_graph(0, leaf(0)).
fib_graph(1, leaf(1)).
fib_graph(N, node(N, [L, R])) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib_graph(N1, L),
    fib_graph(N2, R).
```

**Fold Computer Template**:
```prolog
fold_fib(leaf(V), V).
fold_fib(node(_, [L, R]), V) :-
    fold_fib(L, VL),
    fold_fib(R, VR),
    V is VL + VR.  % <- Operator extracted from original
```

#### Step 3: Template Instantiation

Given: `fib(N, F) :- ... F is F1 + F2`

1. Extract operator: `+`
2. Extract guards: `[N > 1]`
3. Extract computations: `[N1 is N-1, N2 is N-2]`
4. Instantiate template with extracted values
5. Generate clauses

### Implementation Strategy

```prolog
% 1. Detect pattern type
detect_pattern_type(Pred/Arity, PatternType) :-
    find_recursive_clause(Pred/Arity, RecClause),
    extract_combination_op(RecClause, CombineOp),
    classify_operation(CombineOp, PatternType).

% 2. Extract template parameters
extract_template_params(Pred/Arity, PatternType, Params) :-
    Params = params(
        guards: GuardList,
        computations: ComputeList,
        operator: Op,
        num_recursive_calls: N
    ).

% 3. Instantiate template
instantiate_template(PatternType, Params, Clauses) :-
    template(PatternType, Template),
    apply_params(Template, Params, Clauses).
```

### Mustache/Named Template Pattern

Instead of positional replacement, use **named placeholders**:

**Template Definition**:
```prolog
template(binary_tree_fold, [
    graph_base_cases(BaseCases),
    graph_recursive(
        '{{pred}}_graph({{input}}, node({{input}}, [L, R])) :-',
        '    {{guards}},',
        '    {{computations}},',
        '    {{pred}}_graph({{arg1}}, L),',
        '    {{pred}}_graph({{arg2}}, R).'
    ),
    fold_cases([
        'fold_{{pred}}(leaf(V), V).',
        'fold_{{pred}}(node(_, [L, R]), V) :-',
        '    fold_{{pred}}(L, VL),',
        '    fold_{{pred}}(R, VR),',
        '    V is VL {{operator}} VR.'
    ])
]).
```

**Parameter Binding**:
```prolog
Bindings = [
    pred: 'fib',
    input: 'N',
    guards: 'N > 1',
    computations: 'N1 is N - 1, N2 is N - 2',
    arg1: 'N1',
    arg2: 'N2',
    operator: '+'
].
```

**Instantiation**:
```prolog
substitute_template(Template, Bindings, Code).
```

This approach is:
- ✅ Simple and maintainable
- ✅ Easy to debug
- ✅ Easy to extend with new templates
- ❌ Less general (doesn't handle arbitrary predicates)
- ❌ Requires template for each pattern

## Recommendation

### Short-term: Use Template-Based Approach
1. Implement 2-3 common templates (binary tree, ternary tree, max/min)
2. Generate clean, correct code for 80% of use cases
3. Get feedback from real usage

### Long-term: Return to General Approach
1. Document lessons learned from template approach
2. Design better variable substitution algorithm
3. Use insights from template parameters to guide general transformation
4. Implement hybrid: templates for common cases, general fallback for others

## Next Steps

1. **Document current general approach** (this file) ✓
2. **Implement simple template-based generator**
   - Binary tree template (fibonacci, binomial)
   - Extract operator from original clause
   - Generate clean code
3. **Test template generator**
   - Fibonacci
   - Binomial coefficients
   - Tree sum (if pattern matches)
4. **Commit template-based implementation**
5. **Revisit general approach later** with better algorithm

## References

- Current implementation: `fold_helper_generator.pl`
- Example manual implementations: `examples/fibonacci_fold.pl`, `examples/binomial_fold.pl`
- Pattern detection: `pattern_matchers.pl::is_tree_fold_pattern/1`
- Documentation: `docs/FOLD_HELPER_PATTERN.md`

## Related Work

### Variable Substitution in Prolog

**Standard Approaches**:
1. **`=..` (univ)**: Convert between term and list representation
2. **`copy_term/2`**: Create fresh copy with new variables
3. **`term_variables/2`**: Extract all variables from term
4. **`subsumes_term/2`**: Check term subsumption

**Advanced Approaches**:
1. **DCGs**: Use grammar rules for term transformation
2. **Partial Evaluation**: Specialize code at compile-time
3. **Term Rewriting**: Apply rewrite rules systematically

**Library Support**:
- `library(apply)`: maplist, include, exclude for term operations
- `library(aggregate)`: Complex aggregations
- `library(lambda)`: Lambda expressions for anonymous predicates

### Mustache/Template Systems in Prolog

**Available Libraries**:
- `library(quasi_quotations)`: Embedded DSLs in Prolog
- Mustache implementations exist for SWI-Prolog
- Can build custom template system with DCGs

**Example**:
```prolog
template_substitute(Template, Bindings, Result) :-
    phrase(template_dcg(Bindings), Template, Result).

template_dcg(Bindings) -->
    "{{", identifier(Name), "}}",
    { member(Name=Value, Bindings) },
    [Value],
    template_dcg(Bindings).
template_dcg(_) --> [C], template_dcg.
template_dcg(_) --> [].
```

## Open Questions

1. **How many templates do we need?**
   - Binary tree (2 recursive calls)
   - Ternary tree (3 recursive calls)
   - N-ary tree (variable number)
   - List-based patterns?

2. **Should we support user-defined templates?**
   - Allow users to register new templates
   - Template DSL or Prolog predicates?

3. **How to handle edge cases?**
   - Predicates with guards in multiple clauses
   - Multiple recursive clauses with different structures
   - Non-uniform recursion patterns

4. **Integration with bash generation?**
   - How to compile fold patterns to bash?
   - Two-phase execution in bash (array building + folding)
   - Performance implications

## Version History

- **v0.1** (2025-01-13): Initial framework, general transformation approach
  - Basic structure complete
  - Transformation bugs identified
  - This design document created

- **v0.2** (planned): Template-based implementation
  - Binary tree template
  - Operator extraction
  - Clean code generation

- **v1.0** (future): Hybrid approach
  - Templates for common patterns
  - General transformation for edge cases
  - Full bash compilation support
