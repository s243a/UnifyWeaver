:- module(test_cpp_native_lowering, [test_cpp_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/cpp_target').

test_cpp_native_lowering :-
    run_tests([cpp_native_lowering]).

:- begin_tests(cpp_native_lowering).

% Helper: compile using the public API
compile_cpp(Pred/Arity, Code) :-
    cpp_target:compile_predicate_to_cpp(Pred/Arity, [], Code).

% Helper: check substring exists (deterministic)
has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% ============================================================================
% Tier 1: Multi-clause predicates → if/else if/else chains
% ============================================================================

test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_cpp(classify/2, Code),
    has(Code, "std::string classify(int arg1)"),
    has(Code, "arg1 > 0 && arg1 < 10"),
    has(Code, "\"small\""),
    has(Code, "else if (arg1 >= 10)"),
    has(Code, "\"large\""),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    compile_cpp(positive/2, Code),
    has(Code, "std::string positive(int arg1)"),
    has(Code, "arg1 > 0"),
    has(Code, "\"yes\""),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_cpp(double/2, Code),
    has(Code, "std::string double(int arg1)"),
    has(Code, "(arg1 * 2)"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_cpp(identity/2, Code),
    has(Code, "std::string identity(int arg1)"),
    has(Code, "arg1"),
    retractall(user:identity(_, _)).

test(multi_fact_native) :-
    assert(user:color2(red, warm)),
    assert(user:color2(blue, cool)),
    assert(user:color2(green, cool)),
    compile_cpp(color2/2, Code),
    has(Code, "std::string color2(int arg1)"),
    has(Code, "arg1 == std::string(\"red\")"),
    has(Code, "arg1 == std::string(\"blue\")"),
    has(Code, "arg1 == std::string(\"green\")"),
    retractall(user:color2(_, _)).

% ============================================================================
% Tier 2: If-then-else and nested conditionals
% ============================================================================

test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_cpp(abs_val/2, Code),
    has(Code, "std::string abs_val(int arg1)"),
    has(Code, "arg1 >= 0"),
    has(Code, "arg1"),
    has(Code, "(-arg1)"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_cpp(range_classify/2, Code),
    has(Code, "std::string range_classify(int arg1)"),
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
    compile_cpp(sign/2, Code),
    has(Code, "std::string sign(int arg1)"),
    has(Code, "arg1 > 0"),
    has(Code, "arg1 < 0"),
    retractall(user:sign(_, _)).

% ============================================================================
% Tier 1: Guard separation with arity > 2
% ============================================================================

test(guard_with_computation) :-
    assert(user:(safe_div(X, Y, R) :- Y > 0, R is X / Y)),
    compile_cpp(safe_div/3, Code),
    has(Code, "std::string safe_div(int arg1, int arg2)"),
    has(Code, "arg2 > 0"),
    retractall(user:safe_div(_, _, _)).

% ============================================================================
% C++-specific syntax
% ============================================================================

test(cpp_uses_else_if) :-
    assert(user:(grade(X, pass) :- X >= 50)),
    assert(user:(grade(X, fail) :- X < 50)),
    compile_cpp(grade/2, Code),
    has(Code, "else if"),
    retractall(user:grade(_, _)).

test(cpp_uses_runtime_error) :-
    assert(user:(only_pos(X, yes) :- X > 0)),
    compile_cpp(only_pos/2, Code),
    has(Code, "std::runtime_error"),
    retractall(user:only_pos(_, _)).

% ============================================================================
% Verify shared module is loaded
% ============================================================================

test(uses_shared_analysis_module) :-
    current_predicate(clause_body_analysis:normalize_goals/2),
    current_predicate(clause_body_analysis:if_then_else_goal/4),
    current_predicate(clause_body_analysis:build_head_varmap/3).

:- end_tests(cpp_native_lowering).
