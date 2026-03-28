# Recursive Template-Then-Lower Architecture

## The Insight

Templates and native lowering should not be an either/or choice
at the predicate level. They should interleave recursively at
every level of the AST:

```
For each expression/goal:
  1. Try template (recognized pattern → idiomatic output)
  2. Fall back to native lowering (goal-by-goal translation)
     Within each lowered goal:
       Repeat: try template first, then lower sub-expressions
```

This produces idiomatic code wherever a recognized pattern exists,
and mechanically correct code everywhere else — at every nesting
level.

## Current Architecture (One-Shot)

```
compile_recursive(Pred/Arity, Options, Code) →
  classify_predicate → Pattern
  ├─ transitive_closure → compile_tc_from_template (TEMPLATE)
  ├─ tail_recursion → compile_tail_pattern (TEMPLATE)
  ├─ linear_recursion → compile_linear_pattern (TEMPLATE)
  ├─ tree_recursion → compile_tree_pattern (TEMPLATE)
  ├─ non_recursive → compile_non_recursive
  │   └─ native_python_clause_body (NATIVE LOWERING)
  └─ fail
```

The problem: if a predicate is classified as non-recursive but
its body contains a sub-expression that IS a recognized pattern
(e.g., a nested if-then-else, a call to a TC predicate, or a
fold-like accumulation), the template for that sub-expression is
never tried. The entire body gets native-lowered mechanically.

## Proposed Architecture (Recursive)

```
compile_expression(Target, Expr, VarMap, Code) →
  1. try_template(Target, Expr, Code)
     ├─ if-then-else output template
     ├─ disjunction if/elif/else template
     ├─ accumulator loop template
     ├─ binding call template
     └─ ...
  2. native_lower(Target, Expr, VarMap, Code)
     ├─ decompose Expr into sub-expressions
     ├─ for each sub-expression:
     │     compile_expression(Target, SubExpr, VarMap, SubCode)
     │     (recursively tries template first!)
     └─ assemble sub-codes into target syntax
```

The key: `compile_expression` is the recursive entry point.
Both templates and native lowering call it for sub-expressions,
creating a mutual recursion between template matching and
mechanical lowering.

## Examples

### Example 1: Guard + Nested Template

```prolog
safe_max(X, Y, R) :- X >= 0, Y >= 0, (X > Y -> R = X ; R = Y).
```

Current output (native lowering only):
```python
def safe_max(arg1, arg2):
    if arg1 >= 0 and arg2 >= 0:
        return arg1 if arg1 > arg2 else arg2
```

This already works because `classify_goal_sequence` detects the
if-then-else output. But it's a flat dispatch — the ternary
template is hardcoded in the classified goal renderer.

With recursive template-then-lower, the flow would be:
1. Classify body goals: [guard, guard, output_ite]
2. For each guard: `compile_expression` → native_lower → comparison
3. For the output_ite: `compile_expression` →
   `try_template(ite_output)` → ternary template
4. Within the ternary template, each branch:
   `compile_expression` → native_lower → simple assignment

### Example 2: Unrecognized Structure With Recognized Parts

```prolog
process(X, Y, R) :-
    transform(X, T),         % binding call
    (T > 0 -> R is T * Y    % if-then-else output
    ; R is abs(T) * Y).
```

Current: if `transform/2` is a binding, `classify_goal_sequence`
returns `[passthrough(transform(X,T)), output_ite(...)]`. The
passthrough gets handled by `python_output_goal` (binding lookup),
and the ite gets the ternary template. This already works for
two levels.

But what if the branches themselves contain complex expressions?

```prolog
process(X, Y, R) :-
    transform(X, T),
    (T > 0 ->
        filter(data, T > threshold, Filtered),  % pandas op
        R = Filtered
    ;   R is abs(T) * Y).
```

Currently the pandas `filter` inside the then-branch would NOT
be handled by the pandas template — it's inside an ite branch
which is rendered as a block, not as individual classified goals.

With recursive template-then-lower, the then-branch would be
decomposed into goals, each getting `compile_expression`:
- `filter(data, T > threshold, Filtered)` →
  `try_template(pandas_filter)` → `arg3 = arg1[arg2 > threshold]`
- `R = Filtered` → native_lower → simple assignment

### Example 3: Recursive Call Inside Native Lowering

```prolog
double_ancestor(X, Y) :- ancestor(X, Z), ancestor(Z, Y).
```

If `ancestor/2` has a TC template, the recursive approach could
detect the `ancestor` calls and inline or reference the TC-compiled
version. Currently this would be `passthrough(ancestor(X,Z))` →
fail (no binding for `ancestor`).

## Implementation Strategy

### Phase 1: Formalize compile_expression

```prolog
%% compile_expression(+Target, +Goal, +VarMap, -Code, -VarMapOut)
%  The recursive entry point. Tries templates, falls back to lowering.
compile_expression(Target, Goal, VarMap, Code, VarMapOut) :-
    (   try_goal_template(Target, Goal, VarMap, Code, VarMapOut)
    ->  true
    ;   native_lower_goal(Target, Goal, VarMap, Code, VarMapOut)
    ).
```

### Phase 2: Register goal-level templates

```prolog
%% try_goal_template(+Target, +Goal, +VarMap, -Code, -VarMapOut)
try_goal_template(Target, Goal, VarMap, Code, VarMapOut) :-
    if_then_else_goal(Goal, If, Then, Else),
    if_then_else_shared_output_vars(Then, Else, VarMap, SharedVars),
    SharedVars \= [],
    !,
    compile_ite_output_template(Target, If, Then, Else, SharedVars,
                                VarMap, Code, VarMapOut).

try_goal_template(Target, Goal, VarMap, Code, VarMapOut) :-
    compound(Goal), functor(Goal, Pred, Arity),
    binding(Target, Pred/Arity, TargetName, Inputs, [_], _),
    !,
    compile_binding_template(Target, Goal, TargetName, Inputs,
                             VarMap, Code, VarMapOut).
```

### Phase 3: Native lowering calls compile_expression

```prolog
%% native_lower_goal(+Target, +Goal, +VarMap, -Code, -VarMapOut)
native_lower_goal(Target, (A, B), VarMap, Code, VarMapOut) :-
    compile_expression(Target, A, VarMap, CodeA, VarMap1),
    compile_expression(Target, B, VarMap1, CodeB, VarMapOut),
    atomic_list_concat([CodeA, CodeB], '\n', Code).
```

### Phase 4: Templates call compile_expression for branches

```prolog
%% compile_ite_output_template(+Target, +If, +Then, +Else, ...)
compile_ite_output_template(Target, If, Then, Else, SharedVars,
                            VarMap, Code, VarMapOut) :-
    compile_guard_expression(Target, If, VarMap, IfCode),
    %% Recursively compile each branch!
    compile_expression(Target, Then, VarMap, ThenCode, _),
    compile_expression(Target, Else, VarMap, ElseCode, _),
    format_ite(Target, IfCode, ThenCode, ElseCode, SharedVars,
               VarMap, Code, VarMapOut).
```

## Relationship to Existing Code

The current `classify_goal_sequence` + `python_render_classified_goals`
is already a partial implementation of this architecture:

- `classify_goal_sequence` is the "try templates" step
- `passthrough` is the "fall back to native lowering" step
- The classified renderers call `python_guard_condition` and
  `python_output_goal` which are the target-specific lowerers

The missing piece: the branch bodies in ite/disjunction templates
are rendered as opaque blocks (via `python_branch_value`), not
recursively through `compile_expression`. Making `python_branch_value`
call `compile_expression` instead of extracting a single value
would complete the recursive architecture.

## Non-Goals

This is NOT about:
- Adding new recursion patterns (that's the multifile dispatch)
- Replacing the advanced recursion compiler (it stays)
- TypR-specific features (R fallback, @{ }@ wrapping)

It IS about:
- Making templates and lowering compose at every nesting level
- Ensuring recognized sub-patterns get idiomatic output even when
  the containing structure is unrecognized
- A clean recursive architecture that's easy to extend

## Priority

Medium — the current flat dispatch already handles most cases well.
The recursive approach matters most for complex predicates where
the body has multiple nesting levels of control flow and bindings.
The design should be validated with concrete failing test cases
before implementation.
