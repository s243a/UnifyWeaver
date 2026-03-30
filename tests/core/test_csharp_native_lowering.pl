:- module(test_csharp_native_lowering, [test_csharp_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/csharp_native_target').

test_csharp_native_lowering :-
    run_tests([csharp_native_lowering]).

:- begin_tests(csharp_native_lowering).

compile_cs(Pred/Arity, Code) :-
    csharp_native_target:compile_predicate_to_csharp(Pred/Arity, [], Code).

has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% Tier 1: Multi-clause → if/else chain
test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_cs(classify/2, Code),
    has(Code, "Classify"),
    has(Code, "arg1)"),
    has(Code, "> Convert.ToInt32(0)"),
    has(Code, "< Convert.ToInt32(10)"),
    has(Code, "\"small\""),
    has(Code, ">= Convert.ToInt32(10)"),
    has(Code, "\"large\""),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    assert(user:(positive(X, no) :- X =< 0)),
    compile_cs(positive/2, Code),
    has(Code, "Positive"),
    has(Code, "> Convert.ToInt32(0)"),
    has(Code, "\"yes\""),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_cs(double/2, Code),
    has(Code, "Double"),
    has(Code, "Convert.ToInt32(arg1)"),
    has(Code, "* Convert.ToInt32(2)"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_cs(identity/2, Code),
    has(Code, "arg1"),
    retractall(user:identity(_, _)).

% Tier 2: If-then-else
test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_cs(abs_val/2, Code),
    has(Code, ">= Convert.ToInt32(0)"),
    has(Code, "arg1"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_cs(range_classify/2, Code),
    has(Code, "< Convert.ToInt32(0)"),
    has(Code, "\"negative\""),
    has(Code, "== Convert.ToInt32(0)"),
    has(Code, "\"zero\""),
    has(Code, "\"positive\""),
    retractall(user:range_classify(_, _)).

% C#-specific syntax
test(cs_uses_pascal_case) :-
    assert(user:(my_func(X, Y) :- Y is X + 1)),
    compile_cs(my_func/2, Code),
    has(Code, "MyFunc"),
    retractall(user:my_func(_, _)).

test(cs_uses_math_abs) :-
    assert(user:(abs_test(X, Y) :- Y is abs(X))),
    compile_cs(abs_test/2, Code),
    has(Code, "Math.Abs"),
    retractall(user:abs_test(_, _)).

test(three_clause) :-
    assert(user:(grade(X, low) :- X < 50)),
    assert(user:(grade(X, mid) :- X >= 50, X < 80)),
    assert(user:(grade(X, high) :- X >= 80)),
    compile_cs(grade/2, Code),
    has(Code, "< Convert.ToInt32(50)"),
    has(Code, ">= Convert.ToInt32(80)"),
    has(Code, "\"low\""),
    has(Code, "\"high\""),
    retractall(user:grade(_, _)).

test(complex_arithmetic) :-
    assert(user:(formula(X, Y) :- Y is (X * X) + (X * 2) + 1)),
    compile_cs(formula/2, Code),
    has(Code, "Convert.ToInt32(arg1)"),
    has(Code, "* Convert.ToInt32(2)"),
    retractall(user:formula(_, _)).

test(mod_guard) :-
    assert(user:(parity(X, even) :- 0 =:= X mod 2)),
    assert(user:(parity(X, odd) :- 0 =\= X mod 2)),
    compile_cs(parity/2, Code),
    has(Code, "% Convert.ToInt32(2)"),
    has(Code, "\"even\""),
    retractall(user:parity(_, _)).

:- end_tests(csharp_native_lowering).
