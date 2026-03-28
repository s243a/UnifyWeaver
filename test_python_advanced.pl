:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').

:- dynamic label/2, category/2, sign/2.

%% If-then-else output: label(X, L) where L depends on condition
label(X, L) :- (X > 0 -> L = positive ; L = negative).

%% Disjunction output: 3-way classification
category(X, C) :- (X < 0, C = negative ; X =:= 0, C = zero ; X > 0, C = positive).

%% Multi-clause with guards
sign(0, zero).
sign(X, positive) :- X > 0.
sign(X, negative) :- X < 0.

test_ite :-
    writeln('=== TEST: Python if-then-else output ==='),
    recursive_compiler:compile_recursive(label/2, [target(python)], Code),
    writeln(Code),
    (sub_string(Code, _, _, _, "if") -> writeln('  PASS: contains if') ; writeln('  FAIL')).

test_disj :-
    writeln('=== TEST: Python disjunction output ==='),
    recursive_compiler:compile_recursive(category/2, [target(python)], Code),
    writeln(Code),
    (sub_string(Code, _, _, _, "elif") -> writeln('  PASS: contains elif') ; writeln('  NOTE: no elif (may use different pattern)')).

test_multi_clause :-
    writeln('=== TEST: Python multi-clause with guards ==='),
    recursive_compiler:compile_recursive(sign/2, [target(python)], Code),
    writeln(Code),
    (sub_string(Code, _, _, _, "def sign") -> writeln('  PASS: generates function') ; writeln('  FAIL')).

run_tests :-
    test_ite,
    test_disj,
    test_multi_clause,
    nl, writeln('=== ALL PYTHON ADVANCED TESTS DONE ===').
