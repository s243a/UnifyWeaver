:- module(test_c_native_lowering, [test_c_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/c_target').

test_c_native_lowering :-
    run_tests([c_native_lowering]).

:- begin_tests(c_native_lowering).

% Helper: compile using the public API
compile_c(Pred/Arity, Code) :-
    c_target:compile_predicate_to_c(Pred/Arity, [], Code).

% Helper: check substring exists (deterministic)
has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% ============================================================================
% Tier 1: Multi-clause predicates → if/else if/else chains
% ============================================================================

test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_c(classify/2, Code),
    has(Code, "const char* classify(int arg1)"),
    has(Code, "arg1 > 0 && arg1 < 10"),
    has(Code, "\"small\""),
    has(Code, "else if (arg1 >= 10)"),
    has(Code, "\"large\""),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    compile_c(positive/2, Code),
    has(Code, "const char* positive(int arg1)"),
    has(Code, "arg1 > 0"),
    has(Code, "\"yes\""),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_c(double/2, Code),
    has(Code, "const char* double(int arg1)"),
    has(Code, "(arg1 * 2)"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_c(identity/2, Code),
    has(Code, "const char* identity(int arg1)"),
    has(Code, "arg1"),
    retractall(user:identity(_, _)).

test(multi_fact_native) :-
    assert(user:color2(red, warm)),
    assert(user:color2(blue, cool)),
    assert(user:color2(green, cool)),
    compile_c(color2/2, Code),
    has(Code, "const char* color2(int arg1)"),
    has(Code, "strcmp(arg1, \"red\") == 0"),
    has(Code, "strcmp(arg1, \"blue\") == 0"),
    has(Code, "strcmp(arg1, \"green\") == 0"),
    retractall(user:color2(_, _)).

% ============================================================================
% Tier 2: If-then-else and nested conditionals
% ============================================================================

test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_c(abs_val/2, Code),
    has(Code, "const char* abs_val(int arg1)"),
    has(Code, "arg1 >= 0"),
    has(Code, "arg1"),
    has(Code, "(-arg1)"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_c(range_classify/2, Code),
    has(Code, "const char* range_classify(int arg1)"),
    has(Code, "arg1 < 0"),
    has(Code, "\"negative\""),
    has(Code, "arg1 == 0"),
    has(Code, "\"zero\""),
    has(Code, "\"positive\""),
    retractall(user:range_classify(_, _)).

test(three_way_nested) :-
    assert(user:(sign(X, R) :-
        (X > 0 -> R = positive
        ; (X < 0 -> R = negative
        ; R = zero)))),
    compile_c(sign/2, Code),
    has(Code, "const char* sign(int arg1)"),
    has(Code, "arg1 > 0"),
    has(Code, "arg1 < 0"),
    retractall(user:sign(_, _)).

% ============================================================================
% Tier 1: Guard separation with arity > 2
% ============================================================================

test(guard_with_computation) :-
    assert(user:(safe_div(X, Y, R) :- Y > 0, R is X / Y)),
    compile_c(safe_div/3, Code),
    has(Code, "const char* safe_div(int arg1, int arg2)"),
    has(Code, "arg2 > 0"),
    retractall(user:safe_div(_, _, _)).

% ============================================================================
% C-specific syntax
% ============================================================================

test(c_uses_strcmp) :-
    assert(user:(greet(hello, hi) :- true)),
    compile_c(greet/2, Code),
    has(Code, "strcmp"),
    retractall(user:greet(_, _)).

test(c_uses_exit) :-
    assert(user:(only_pos(X, yes) :- X > 0)),
    compile_c(only_pos/2, Code),
    has(Code, "exit(1)"),
    retractall(user:only_pos(_, _)).

% ============================================================================
% Expanded: three-clause classify
% ============================================================================

test(three_clause_classify) :-
    assert(user:(grade(X, low) :- X < 50)),
    assert(user:(grade(X, mid) :- X >= 50, X < 80)),
    assert(user:(grade(X, high) :- X >= 80)),
    compile_c(grade/2, Code),
    has(Code, "arg1 < 50"),
    has(Code, "arg1 >= 50"),
    has(Code, "arg1 >= 80"),
    has(Code, "\"low\""),
    has(Code, "\"high\""),
    retractall(user:grade(_, _)).

test(mod_guard) :-
    assert(user:(parity(X, even) :- 0 =:= X mod 2)),
    assert(user:(parity(X, odd) :- 0 =\= X mod 2)),
    compile_c(parity/2, Code),
    has(Code, "%"),
    has(Code, "\"even\""),
    has(Code, "\"odd\""),
    retractall(user:parity(_, _)).

test(complex_arithmetic) :-
    assert(user:(formula(X, Y) :- Y is (X * X) + (X * 2) + 1)),
    compile_c(formula/2, Code),
    has(Code, "arg1 * arg1"),
    has(Code, "arg1 * 2"),
    retractall(user:formula(_, _)).

test(negation_output) :-
    assert(user:(negate(X, Y) :- Y is 0 - X)),
    compile_c(negate/2, Code),
    has(Code, "0 - arg1"),
    retractall(user:negate(_, _)).

% ============================================================================
% Verify shared module is loaded
% ============================================================================

test(uses_shared_analysis_module) :-
    current_predicate(clause_body_analysis:normalize_goals/2),
    current_predicate(clause_body_analysis:if_then_else_goal/4),
    current_predicate(clause_body_analysis:build_head_varmap/3).

:- end_tests(c_native_lowering).
