:- module(test_r_native_lowering, [test_r_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/r_target').

test_r_native_lowering :-
    run_tests([r_native_lowering]).

:- begin_tests(r_native_lowering).

compile_r_test(Pred/Arity, Code) :-
    r_target:compile_predicate_to_r(Pred/Arity, [], Code).

has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% Tier 1: Multi-clause → if/else chain
test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_r_test(classify/2, Code),
    has(Code, "classify <- function"),
    has(Code, "arg1 > 0 && arg1 < 10"),
    has(Code, "\"small\""),
    has(Code, "arg1 >= 10"),
    has(Code, "\"large\""),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    assert(user:(positive(X, no) :- X =< 0)),
    compile_r_test(positive/2, Code),
    has(Code, "positive <- function"),
    has(Code, "arg1 > 0"),
    has(Code, "\"yes\""),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_r_test(double/2, Code),
    has(Code, "double <- function"),
    has(Code, "(arg1 * 2)"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_r_test(identity/2, Code),
    has(Code, "<- function"),
    has(Code, "arg1"),
    retractall(user:identity(_, _)).

% Tier 2: If-then-else
test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_r_test(abs_val/2, Code),
    has(Code, "arg1 >= 0"),
    has(Code, "arg1"),
    has(Code, "(-arg1)"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_r_test(range_classify/2, Code),
    has(Code, "arg1 < 0"),
    has(Code, "\"negative\""),
    has(Code, "arg1 == 0"),
    has(Code, "\"zero\""),
    has(Code, "\"positive\""),
    retractall(user:range_classify(_, _)).

% R-specific syntax
test(r_uses_arrow_assignment) :-
    assert(user:(inc(X, Y) :- Y is X + 1)),
    compile_r_test(inc/2, Code),
    has(Code, "<- function"),
    retractall(user:inc(_, _)).

test(r_uses_double_percent_mod) :-
    assert(user:(parity(X, even) :- 0 =:= X mod 2)),
    assert(user:(parity(X, odd) :- 0 =\= X mod 2)),
    compile_r_test(parity/2, Code),
    has(Code, "%%"),
    has(Code, "\"even\""),
    retractall(user:parity(_, _)).

test(three_clause) :-
    assert(user:(grade(X, low) :- X < 50)),
    assert(user:(grade(X, mid) :- X >= 50, X < 80)),
    assert(user:(grade(X, high) :- X >= 80)),
    compile_r_test(grade/2, Code),
    has(Code, "arg1 < 50"),
    has(Code, "arg1 >= 80"),
    has(Code, "\"low\""),
    has(Code, "\"high\""),
    retractall(user:grade(_, _)).

test(complex_arithmetic) :-
    assert(user:(formula(X, Y) :- Y is (X * X) + (X * 2) + 1)),
    compile_r_test(formula/2, Code),
    has(Code, "(arg1 * arg1)"),
    has(Code, "(arg1 * 2)"),
    retractall(user:formula(_, _)).

test(negation_output) :-
    assert(user:(negate(X, Y) :- Y is 0 - X)),
    compile_r_test(negate/2, Code),
    has(Code, "(0 - arg1)"),
    retractall(user:negate(_, _)).

:- end_tests(r_native_lowering).
