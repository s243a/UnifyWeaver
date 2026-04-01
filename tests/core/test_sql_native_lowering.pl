:- module(test_sql_native_lowering, [test_sql_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/sql_target').

test_sql_native_lowering :-
    run_tests([sql_native_lowering]).

:- begin_tests(sql_native_lowering).

compile_sql(Pred/Arity, Code) :-
    sql_target:compile_predicate_to_sql(Pred/Arity, [], Code).

compile_sql(Pred/Arity, Options, Code) :-
    sql_target:compile_predicate_to_sql(Pred/Arity, Options, Code).

has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% Tier 1: Multi-clause → CASE WHEN
test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_sql(classify/2, Code),
    has(Code, "CASE"),
    has(Code, "WHEN arg1 > 0 AND arg1 < 10 THEN"),
    has(Code, "'small'"),
    has(Code, "WHEN arg1 >= 10 THEN"),
    has(Code, "'large'"),
    has(Code, "END"),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    assert(user:(positive(X, no) :- X =< 0)),
    compile_sql(positive/2, Code),
    has(Code, "WHEN arg1 > 0 THEN"),
    has(Code, "'yes'"),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_sql(double/2, Code),
    has(Code, "arg1 * 2"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_sql(identity/2, Code),
    has(Code, "arg1"),
    retractall(user:identity(_, _)).

% Tier 2: If-then-else → nested CASE WHEN
test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_sql(abs_val/2, Code),
    has(Code, "CASE WHEN"),
    has(Code, "arg1 >= 0"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_sql(range_classify/2, Code),
    has(Code, "arg1 < 0"),
    has(Code, "'negative'"),
    has(Code, "arg1 = 0"),
    has(Code, "'zero'"),
    has(Code, "'positive'"),
    retractall(user:range_classify(_, _)).

% SQL-specific syntax
test(sql_uses_single_quotes) :-
    assert(user:(id(X, Y) :- Y = X)),
    compile_sql(id/2, Code),
    has(Code, "AS result"),
    retractall(user:id(_, _)).

test(sql_uses_abs_function) :-
    assert(user:(abs_test(X, Y) :- Y is abs(X))),
    compile_sql(abs_test/2, Code),
    has(Code, "ABS(arg1)"),
    retractall(user:abs_test(_, _)).

test(three_clause) :-
    assert(user:(grade(X, low) :- X < 50)),
    assert(user:(grade(X, mid) :- X >= 50, X < 80)),
    assert(user:(grade(X, high) :- X >= 80)),
    compile_sql(grade/2, Code),
    has(Code, "WHEN arg1 < 50 THEN 'low'"),
    has(Code, "WHEN arg1 >= 80 THEN 'high'"),
    retractall(user:grade(_, _)).

test(complex_arithmetic) :-
    assert(user:(formula(X, Y) :- Y is (X * X) + (X * 2) + 1)),
    compile_sql(formula/2, Code),
    has(Code, "arg1 * arg1"),
    has(Code, "arg1 * 2"),
    retractall(user:formula(_, _)).

test(mod_guard) :-
    assert(user:(parity(X, even) :- 0 =:= X mod 2)),
    assert(user:(parity(X, odd) :- 0 =\= X mod 2)),
    compile_sql(parity/2, Code),
    has(Code, "% 2"),
    has(Code, "'even'"),
    retractall(user:parity(_, _)).

% Output mode: create_function (PL/pgSQL)
test(create_function_mode) :-
    assert(user:(sign(X, pos) :- X > 0)),
    assert(user:(sign(X, neg) :- X =< 0)),
    compile_sql(sign/2, [sql_output_mode(create_function)], Code),
    has(Code, "CREATE OR REPLACE FUNCTION sign"),
    has(Code, "RETURNS TEXT"),
    has(Code, "IF arg1 > 0 THEN"),
    has(Code, "RETURN 'pos'"),
    has(Code, "ELSIF arg1 <= 0 THEN"),
    has(Code, "RETURN 'neg'"),
    has(Code, "LANGUAGE plpgsql"),
    retractall(user:sign(_, _)).

% Output mode: case_expression (bare CASE only)
test(case_expression_mode) :-
    assert(user:(bool(X, t) :- X > 0)),
    assert(user:(bool(X, f) :- X =< 0)),
    compile_sql(bool/2, [sql_output_mode(case_expression)], Code),
    has(Code, "CASE WHEN"),
    \+ has(Code, "SELECT"),
    \+ has(Code, "CREATE"),
    retractall(user:bool(_, _)).

:- end_tests(sql_native_lowering).
