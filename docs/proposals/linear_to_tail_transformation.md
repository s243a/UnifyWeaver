<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->
# Proposal: Automatic Linear-to-Tail Recursion Transformation with Purity Analysis

## Motivation

Currently, a predicate like `factorial/2` is classified as **linear recursive** because the recursive call is not in tail position:

```prolog
factorial(0, 1).
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),   % recursive call
    F is N * F1.         % post-recursion computation
```

The compiled output uses memoization or a bottom-up for loop. However, this can be mechanically transformed into a **tail-recursive accumulator** form:

```prolog
factorial(N, F) :- factorial_acc(N, 1, F).
factorial_acc(0, Acc, Acc).
factorial_acc(N, Acc, F) :-
    N > 0,
    Acc1 is Acc * N,
    N1 is N - 1,
    factorial_acc(N1, Acc1, F).
```

The tail-recursive form compiles to a simple while loop — no array, no memoization, O(1) space instead of O(n).

This transformation is only safe when the post-recursion operations are **pure** (no side effects) and the accumulation operation is **associative** (or at least admits an accumulator formulation). Today the compiler has no purity analysis, so it cannot make this determination automatically. The user must either write the tail-recursive form manually or declare `unordered` constraints.

## Proposal

### 1. Purity Analysis for Goals

Add a predicate `is_pure_goal/1` that determines whether a goal is free of side effects:

```prolog
:- module(purity_analysis, [
    is_pure_goal/1,       % +Goal
    is_pure_body/1,       % +Body (conjunction)
    pure_builtin/1        % +Functor/Arity — whitelisted builtins
]).

%% Whitelisted pure builtins
pure_builtin(is/2).
pure_builtin((>)/2).
pure_builtin((<)/2).
pure_builtin((>=)/2).
pure_builtin((=<)/2).
pure_builtin((=:=)/2).
pure_builtin((=\=)/2).
pure_builtin((=)/2).        % unification
pure_builtin((\=)/2).
pure_builtin(succ/2).
pure_builtin(plus/3).
pure_builtin(length/2).
pure_builtin(append/3).
pure_builtin(msort/2).
pure_builtin(sort/2).
pure_builtin(nth0/3).
pure_builtin(nth1/3).
pure_builtin(last/2).
pure_builtin(member/2).     % pure but nondeterministic
pure_builtin(between/3).
pure_builtin(number/1).
pure_builtin(integer/1).
pure_builtin(float/1).
pure_builtin(atom/1).
pure_builtin(compound/1).
pure_builtin(is_list/1).
pure_builtin(functor/3).
pure_builtin(arg/3).
pure_builtin((=..)/2).
pure_builtin(copy_term/2).

%% Known impure — do NOT whitelist:
%% assert/1, retract/1, write/1, read/1, format/2,
%% nb_setval/2, nb_getval/2, open/3, close/1, etc.

%% is_pure_goal(+Goal)
is_pure_goal(true).
is_pure_goal(Goal) :-
    functor(Goal, F, A),
    pure_builtin(F/A).
is_pure_goal(Goal) :-
    % User-declared pure predicate
    declared_pure(Goal).

%% is_pure_body(+Body)
is_pure_body(true).
is_pure_body((A, B)) :- is_pure_body(A), is_pure_body(B).
is_pure_body(Goal) :- is_pure_goal(Goal).
```

**Key insight:** For the compiler's purposes, we only need to verify that the *post-recursion* goals are pure — the goals that appear after the recursive call. If those are all `is/2` and comparison operators, the transformation is safe.

### 2. Accumulator Transformation Detection

Add a predicate that detects when a linear-recursive predicate can be transformed to tail-recursive form:

```prolog
%% can_transform_to_tail(+Pred/Arity, -TransformInfo)
%
%  Succeeds when:
%  1. Predicate is linear recursive (single recursive call per clause)
%  2. The recursive call is NOT already in tail position
%  3. All goals after the recursive call are pure
%  4. The post-recursion goals compute the result via an associative
%     or accumulator-compatible operation
%
%  TransformInfo = transform(
%      AccInit,        % initial accumulator value (from base case)
%      AccOp,          % accumulation operation (e.g., *, +, -)
%      Direction       % left_fold or right_fold
%  )

can_transform_to_tail(Pred/Arity, TransformInfo) :-
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Partition into base/recursive
    partition(is_recursive_for(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],

    % Must be linear (single recursive call per clause)
    forall(
        member(clause(_, RecBody), RecClauses),
        has_exactly_one_recursive_call(RecBody, Pred)
    ),

    % Recursive call must NOT already be in tail position
    forall(
        member(clause(_, RecBody), RecClauses),
        \+ is_tail_call(RecBody, Pred)
    ),

    % Extract post-recursion goals and verify purity
    member(clause(_, RecBody), RecClauses),
    split_at_recursive_call(RecBody, Pred, _PreGoals, _RecCall, PostGoals),
    is_pure_body(PostGoals),

    % Extract the accumulation pattern
    extract_accumulation_pattern(PostGoals, BaseClauses, TransformInfo).
```

### 3. Accumulation Pattern Extraction

The key transformation patterns:

| Post-recursion form | Accumulator Op | Init (from base case) | Associative? |
|---------------------|---------------|----------------------|-------------|
| `F is N * F1` | `*` | 1 | Yes |
| `F is N + F1` | `+` | 0 | Yes |
| `F is F1 + X` | `+` | 0 | Yes |
| `F is F1 - X` | Not transformable | — | No (non-assoc) |
| `append(F1, X, F)` | `append` | `[]` | Yes |

```prolog
%% extract_accumulation_pattern(+PostGoals, +BaseClauses, -TransformInfo)
%
%  Match the post-recursion `is` expression against known patterns.

extract_accumulation_pattern(PostGoals, BaseClauses, transform(Init, Op, left_fold)) :-
    % Find the final `Result is Expr` in post-goals
    find_result_computation(PostGoals, ResultVar, Expr),
    % Decompose: Expr should combine recursive result with current value
    Expr =.. [Op, A, B],
    is_associative_op(Op),
    % Extract initial value from base case
    extract_base_value(BaseClauses, ResultVar, Init).

is_associative_op(*).
is_associative_op(+).
% Note: - and / are NOT associative, so not included
```

For subtraction: `F is N - F1` would need special handling (negate the accumulator). This can be a future extension. The initial implementation should handle `*` and `+` which cover the most common patterns (factorial, sum, product).

### 4. Integration with Existing Compiler

The transformation integrates at the classification stage in `compile_dispatch`:

```
classify_predicate(Pred/Arity, Options, Pattern) :-
    ...
    % After checking for tail recursion and before linear recursion:
    (   can_compile_tail_recursion(Pred/Arity) ->
        Pattern = tail_recursion
    ;   can_transform_to_tail(Pred/Arity, TransformInfo) ->
        Pattern = linear_as_tail(TransformInfo)
    ;   can_compile_linear_recursion(Pred/Arity) ->
        Pattern = linear_recursion
    ;   ...
    ).
```

When `linear_as_tail(TransformInfo)` is selected:
1. The compiler generates the accumulator-based code directly (without actually asserting a transformed predicate)
2. The generated code is a while loop with an accumulator variable — same as genuine tail recursion
3. This is purely a code-generation strategy; the original Prolog predicate is unchanged

### 5. Interaction with Existing Constraints

The `unordered` constraint currently controls whether the linear recursion compiler uses hash-based or sort-based memoization. With this proposal:

- **`unordered(true)` (default):** Compiler is free to attempt linear-to-tail transformation, since reordering is permitted.
- **`unordered(false)` (ordered):** Compiler skips the transformation and uses hash-based memoization, since output order must match evaluation order.
- **Purity analysis makes `unordered` unnecessary for pure arithmetic:** If the post-recursion body is provably pure (all goals are whitelisted builtins), the compiler knows reordering is safe regardless of the `unordered` constraint. This means users don't need to declare `unordered` for purely arithmetic predicates — the compiler deduces safety automatically.

The hierarchy:
1. Explicit `unordered(false)` → forbid transformation (user override)
2. Purity analysis proves safe → allow transformation (no declaration needed)
3. Impure post-goals, no `unordered` declared → skip transformation (conservative)

### 6. Skip/Pattern Integration

The recently added `pattern(P)` and `skip(P)` options interact naturally:

- `pattern(tail_recursion)` — forces tail compilation; if the predicate isn't natively tail-recursive, the compiler attempts the linear-to-tail transform
- `pattern(linear_recursion)` — forces linear compilation, skipping the transform even if eligible
- `skip(linear_as_tail)` or `skip(tail_recursion)` — prevents the transform

### 7. Code Generation

For the transformed pattern, code generation reuses the existing tail recursion templates with minor additions. Example for `factorial/2` → R target:

```r
factorial <- function(n) {
  acc <- 1
  while (n > 0) {
    acc <- acc * n
    n <- n - 1
  }
  return(acc)
}
```

This is identical to what `compile_tail_pattern(r, ...)` already generates for genuine tail-recursive predicates. The only difference is that `AccInit`, `AccOp`, and `Step` are extracted from the linear form rather than from explicit accumulator arguments.

## Implementation Plan

### Phase 1: Purity Analysis Module (Small)
- New file: `src/unifyweaver/core/advanced/purity_analysis.pl`
- ~60 lines: whitelist of pure builtins + `is_pure_goal/1` + `is_pure_body/1`
- No changes to existing code

### Phase 2: Transform Detection (Medium)
- New predicates in `pattern_matchers.pl`: `can_transform_to_tail/2`, `split_at_recursive_call/5`, `extract_accumulation_pattern/3`
- ~80 lines
- Uses purity_analysis module

### Phase 3: Compiler Integration (Small)
- Add `linear_as_tail` case in `compile_dispatch` (recursive_compiler.pl and advanced_recursive_compiler.pl)
- Reuse existing `compile_tail_pattern` multifile with adjusted parameters
- ~30 lines per file

### Phase 4: Tests
- Test purity analysis: pure builtins pass, impure builtins fail
- Test transform detection: `factorial/2` detected as transformable, `ancestor/2` not (relational, not arithmetic)
- Test end-to-end: `factorial/2` compiled via `linear_as_tail` produces correct while-loop code for all targets
- Test constraint interaction: `unordered(false)` blocks transform; pure arithmetic needs no declaration

## Estimated Complexity

**Low-medium.** The core purity analysis is a simple whitelist lookup. The transform detection is pattern matching on the post-recursion body. Code generation reuses existing tail recursion templates. Total: ~200 lines of new Prolog across 3 files, plus tests.

## Risks

1. **False positives in purity analysis:** A user-defined predicate called from post-goals could have side effects. Mitigation: only whitelist known SWI-Prolog builtins; user-defined goals require explicit `unordered` declaration or `:- pure(my_pred/2)` annotation.

2. **Non-associative operations:** `F is F1 - N` looks like a candidate but subtraction isn't associative. The compiler must only transform for provably associative operations (`+`, `*`). Future work can handle anti-commutative patterns.

3. **Nondeterministic post-goals:** `member/2` is pure but nondeterministic — the transformation may change solution order. Since the default is `unordered(true)`, this is acceptable, but should be documented.
