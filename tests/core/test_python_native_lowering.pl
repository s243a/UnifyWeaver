:- module(test_python_native_lowering, [test_python_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/python_target').

test_python_native_lowering :-
    run_tests([python_native_lowering]).

:- begin_tests(python_native_lowering).

% Helper: compile using user module for predicate lookup
compile_py(Pred/Arity, Code) :-
    python_target:compile_predicate_to_python(user:Pred/Arity, [], Code).

% Helper: check substring exists (deterministic)
has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% ============================================================================
% Tier 1: Multi-clause predicates → if/elif/else chains
% ============================================================================

test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_py(classify/2, Code),
    has(Code,"def classify(arg1)"),
    has(Code,"if arg1 > 0 and arg1 < 10"),
    has(Code,"return \"small\""),
    has(Code,"elif arg1 >= 10"),
    has(Code,"return \"large\""),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    compile_py(positive/2, Code),
    has(Code,"def positive(arg1)"),
    has(Code,"arg1 > 0"),
    has(Code,"return \"yes\""),
    retractall(user:positive(_, _)).

test(pure_fact) :-
    assert(user:greeting(hello, world)),
    compile_py(greeting/2, Code),
    has(Code,"def greeting(arg1)"),
    has(Code,"arg1 == \"hello\""),
    has(Code,"return \"world\""),
    retractall(user:greeting(_, _)).

test(multi_fact) :-
    assert(user:color(red, warm)),
    assert(user:color(blue, cool)),
    assert(user:color(green, cool)),
    compile_py(color/2, Code),
    has(Code,"def color(arg1)"),
    has(Code,"if arg1 == \"red\""),
    has(Code,"elif arg1 == \"blue\""),
    has(Code,"elif arg1 == \"green\""),
    retractall(user:color(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_py(double/2, Code),
    has(Code,"def double(arg1)"),
    has(Code,"return (arg1 * 2)"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_py(identity/2, Code),
    has(Code,"def identity(arg1)"),
    has(Code,"return arg1"),
    retractall(user:identity(_, _)).

% ============================================================================
% Tier 2: If-then-else and nested conditionals
% ============================================================================

test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_py(abs_val/2, Code),
    has(Code,"def abs_val(arg1)"),
    has(Code,"if arg1 >= 0"),
    has(Code,"return arg1"),
    has(Code,"return (-arg1)"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_py(range_classify/2, Code),
    has(Code,"def range_classify(arg1)"),
    has(Code,"if arg1 < 0"),
    has(Code,"return \"negative\""),
    has(Code,"elif arg1 == 0"),
    has(Code,"return \"zero\""),
    has(Code,"return \"positive\""),
    retractall(user:range_classify(_, _)).

test(three_way_nested) :-
    assert(user:(sign(X, R) :-
        (X > 0 -> R = positive
        ; (X < 0 -> R = negative
        ; R = zero)))),
    compile_py(sign/2, Code),
    has(Code,"def sign(arg1)"),
    has(Code,"if arg1 > 0"),
    has(Code,"elif arg1 < 0"),
    retractall(user:sign(_, _)).

% ============================================================================
% Tier 1: Guard separation with arity > 2
% ============================================================================

test(guard_with_computation) :-
    assert(user:(safe_div(X, Y, R) :- Y > 0, R is X / Y)),
    compile_py(safe_div/3, Code),
    has(Code,"def safe_div(arg1, arg2)"),
    has(Code,"arg2 > 0"),
    retractall(user:safe_div(_, _, _)).

% ============================================================================
% Verify fallback works for unsupported patterns
% ============================================================================

test(fallback_for_recursive) :-
    assert(user:factorial(0, 1)),
    assert(user:(factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1)),
    compile_py(factorial/2, Code),
    % Recursive predicates go through compile_recursive_predicate
    % so native lowering doesn't apply — should not produce def factorial(arg1)
    \+ once(sub_string(Code, _, _, _, "def factorial(arg1)")),
    retractall(user:factorial(_, _)).

% ============================================================================
% Uses shared clause_body_analysis module
% ============================================================================

test(uses_shared_analysis_module) :-
    current_predicate(clause_body_analysis:normalize_goals/2),
    current_predicate(clause_body_analysis:if_then_else_goal/4),
    current_predicate(clause_body_analysis:build_head_varmap/3),
    current_predicate(clause_body_analysis:classify_goal/3).

:- end_tests(python_native_lowering).
