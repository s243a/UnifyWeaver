:- module(test_go_native_lowering, [test_go_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/go_target').

test_go_native_lowering :-
    run_tests([go_native_lowering]).

:- begin_tests(go_native_lowering).

% Helper: compile using user module
compile_go(Pred/Arity, Code) :-
    go_target:compile_predicate_to_go(user:Pred/Arity, [], Code).

% Helper: check substring exists (deterministic)
has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% ============================================================================
% Tier 1: Multi-clause predicates → if/else if/else chains
% ============================================================================

test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_go(classify/2, Code),
    has(Code, "func classify(arg1 interface{})"),
    has(Code, "if arg1 > 0 && arg1 < 10"),
    has(Code, "return \"small\""),
    has(Code, "else if arg1 >= 10"),
    has(Code, "return \"large\""),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    compile_go(positive/2, Code),
    has(Code, "func positive(arg1 interface{})"),
    has(Code, "arg1 > 0"),
    has(Code, "return \"yes\""),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_go(double/2, Code),
    has(Code, "func double(arg1 interface{})"),
    has(Code, "return (arg1 * 2)"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_go(identity/2, Code),
    has(Code, "func identity(arg1 interface{})"),
    has(Code, "return arg1"),
    retractall(user:identity(_, _)).

test(multi_fact_native) :-
    assert(user:color2(red, warm)),
    assert(user:color2(blue, cool)),
    assert(user:color2(green, cool)),
    compile_go(color2/2, Code),
    has(Code, "func color2(arg1 interface{})"),
    has(Code, "if arg1 == \"red\""),
    has(Code, "else if arg1 == \"blue\""),
    has(Code, "else if arg1 == \"green\""),
    retractall(user:color2(_, _)).

% ============================================================================
% Tier 2: If-then-else and nested conditionals
% ============================================================================

test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_go(abs_val/2, Code),
    has(Code, "func abs_val(arg1 interface{})"),
    has(Code, "if arg1 >= 0"),
    has(Code, "return arg1"),
    has(Code, "return (-arg1)"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_go(range_classify/2, Code),
    has(Code, "func range_classify(arg1 interface{})"),
    has(Code, "if arg1 < 0"),
    has(Code, "return \"negative\""),
    has(Code, "else if arg1 == 0"),
    has(Code, "return \"zero\""),
    has(Code, "return \"positive\""),
    retractall(user:range_classify(_, _)).

test(three_way_nested) :-
    assert(user:(sign(X, R) :-
        (X > 0 -> R = positive
        ; (X < 0 -> R = negative
        ; R = zero)))),
    compile_go(sign/2, Code),
    has(Code, "func sign(arg1 interface{})"),
    has(Code, "if arg1 > 0"),
    has(Code, "else if arg1 < 0"),
    retractall(user:sign(_, _)).

% ============================================================================
% Tier 1: Guard separation with arity > 2
% ============================================================================

test(guard_with_computation) :-
    assert(user:(safe_div(X, Y, R) :- Y > 0, R is X / Y)),
    compile_go(safe_div/3, Code),
    has(Code, "func safe_div(arg1 interface{}, arg2 interface{})"),
    has(Code, "arg2 > 0"),
    retractall(user:safe_div(_, _, _)).

% ============================================================================
% Go-specific syntax
% ============================================================================

test(go_package_and_import) :-
    assert(user:(double2(X, R) :- R is X * 2)),
    compile_go(double2/2, Code),
    has(Code, "package main"),
    has(Code, "import \"fmt\""),
    retractall(user:double2(_, _)).

test(go_uses_interface_type) :-
    assert(user:(id2(X, R) :- R = X)),
    compile_go(id2/2, Code),
    has(Code, "interface{}"),
    retractall(user:id2(_, _)).

% ============================================================================
% Verify shared module is loaded
% ============================================================================

test(uses_shared_analysis_module) :-
    current_predicate(clause_body_analysis:normalize_goals/2),
    current_predicate(clause_body_analysis:if_then_else_goal/4),
    current_predicate(clause_body_analysis:build_head_varmap/3).

:- end_tests(go_native_lowering).
