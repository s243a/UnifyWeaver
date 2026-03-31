:- module(test_bash_native_lowering, [test_bash_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/bash_target').

test_bash_native_lowering :-
    run_tests([bash_native_lowering]).

:- begin_tests(bash_native_lowering).

compile_bash(Pred/Arity, Code) :-
    bash_target:compile_predicate_to_bash(Pred/Arity, [], Code).

compile_bash(Pred/Arity, Options, Code) :-
    bash_target:compile_predicate_to_bash(Pred/Arity, Options, Code).

has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% Tier 1: Multi-clause → if/elif/else chain
test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_bash(classify/2, Code),
    has(Code, "classify()"),
    has(Code, "arg1"),
    has(Code, "if (( arg1 > 0"),
    has(Code, "arg1 < 10"),
    has(Code, '"small"'),
    has(Code, "arg1 >= 10"),
    has(Code, '"large"'),
    has(Code, "elif"),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    assert(user:(positive(X, no) :- X =< 0)),
    compile_bash(positive/2, Code),
    has(Code, "positive()"),
    has(Code, "arg1 > 0"),
    has(Code, '"yes"'),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_bash(double/2, Code),
    has(Code, "double()"),
    has(Code, "$(( "),
    has(Code, "arg1 * 2"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_bash(identity/2, Code),
    has(Code, "$arg1"),
    retractall(user:identity(_, _)).

% Tier 2: If-then-else
test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_bash(abs_val/2, Code),
    has(Code, "arg1 >= 0"),
    has(Code, "arg1"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_bash(range_classify/2, Code),
    has(Code, "arg1 < 0"),
    has(Code, '"negative"'),
    has(Code, "arg1 == 0"),
    has(Code, '"zero"'),
    has(Code, '"positive"'),
    retractall(user:range_classify(_, _)).

% Bash-specific syntax
test(bash_uses_double_paren_arithmetic) :-
    assert(user:(inc(X, Y) :- Y is X + 1)),
    compile_bash(inc/2, Code),
    has(Code, "$(( "),
    has(Code, "arg1 + 1"),
    retractall(user:inc(_, _)).

test(bash_uses_echo_return) :-
    assert(user:(id(X, Y) :- Y = X)),
    compile_bash(id/2, Code),
    has(Code, "echo"),
    has(Code, "Return method: echo"),
    retractall(user:id(_, _)).

test(three_clause) :-
    assert(user:(grade(X, low) :- X < 50)),
    assert(user:(grade(X, mid) :- X >= 50, X < 80)),
    assert(user:(grade(X, high) :- X >= 80)),
    compile_bash(grade/2, Code),
    has(Code, "arg1 < 50"),
    has(Code, "arg1 >= 80"),
    has(Code, '"low"'),
    has(Code, '"high"'),
    retractall(user:grade(_, _)).

test(complex_arithmetic) :-
    assert(user:(formula(X, Y) :- Y is (X * X) + (X * 2) + 1)),
    compile_bash(formula/2, Code),
    has(Code, "arg1 * arg1"),
    has(Code, "arg1 * 2"),
    retractall(user:formula(_, _)).

test(mod_guard) :-
    assert(user:(parity(X, even) :- 0 =:= X mod 2)),
    assert(user:(parity(X, odd) :- 0 =\= X mod 2)),
    compile_bash(parity/2, Code),
    has(Code, "% 2"),
    has(Code, '"even"'),
    retractall(user:parity(_, _)).

% Return method tests
test(global_return_method) :-
    assert(user:(sign(X, pos) :- X > 0)),
    assert(user:(sign(X, neg) :- X =< 0)),
    compile_bash(sign/2, [return_method(global('~w_return'))], Code),
    has(Code, "sign_return="),
    has(Code, "Return method: global"),
    retractall(user:sign(_, _)).

test(nameref_return_method) :-
    assert(user:(sign2(X, pos) :- X > 0)),
    assert(user:(sign2(X, neg) :- X =< 0)),
    compile_bash(sign2/2, [return_method(nameref)], Code),
    has(Code, "__result="),
    has(Code, 'local -n __result'),
    has(Code, "Return method: nameref"),
    retractall(user:sign2(_, _)).

:- end_tests(bash_native_lowering).
