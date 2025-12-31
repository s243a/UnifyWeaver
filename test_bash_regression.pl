:- encoding(utf8).
% Regression tests for Bash stream compiler
% Tests basic functionality to ensure match predicate addition didn't break existing features

:- use_module('src/unifyweaver/core/stream_compiler').

% Test 1: Facts (basic compilation)
parent(alice, bob).
parent(bob, charlie).
parent(alice, dave).

% Test 2: Single rule without constraints
child(C, P) :- parent(P, C).

% Test 3: Single rule with inequality
sibling(X, Y) :-
    parent(P, X),
    parent(P, Y),
    X \= Y.

% Test 4: Single rule with arithmetic
double(X, Y) :-
    Y is X * 2.

% Test 5: Multiple rules (OR pattern)
related(X, Y) :- parent(X, Y).
related(X, Y) :- parent(Y, X).
related(X, Y) :- sibling(X, Y).

% Test 6: Single rule with match (new feature)
log_line('ERROR: timeout').
log_line('INFO: success').
log_line('ERROR: failed').

error_log(Line) :-
    log_line(Line),
    match(Line, 'ERROR').

% Run all tests
test_facts :-
    write('=== Test 1: Facts ==='), nl,
    compile_predicate(parent/2, [], Code),
    write(Code), nl, nl.

test_single_rule :-
    write('=== Test 2: Single Rule (no constraints) ==='), nl,
    compile_predicate(child/2, [], Code),
    write(Code), nl, nl.

test_inequality :-
    write('=== Test 3: Inequality Constraint ==='), nl,
    compile_predicate(sibling/2, [], Code),
    write(Code), nl, nl.

test_arithmetic :-
    write('=== Test 4: Arithmetic ==='), nl,
    compile_predicate(double/2, [], Code),
    write(Code), nl, nl.

test_multiple_rules :-
    write('=== Test 5: Multiple Rules (OR) ==='), nl,
    compile_predicate(related/2, [], Code),
    write(Code), nl, nl.

test_match_new :-
    write('=== Test 6: Match Predicate (NEW) ==='), nl,
    compile_predicate(error_log/1, [], Code),
    write(Code), nl, nl.

run_all :-
    write('======================================'), nl,
    write('BASH STREAM COMPILER REGRESSION TESTS'), nl,
    write('======================================'), nl, nl,
    test_facts,
    test_single_rule,
    test_inequality,
    test_arithmetic,
    test_multiple_rules,
    test_match_new,
    write('======================================'), nl,
    write('All regression tests completed!'), nl,
    write('======================================'), nl.

% Usage:
% ?- consult('test_bash_regression.pl').
% ?- run_all.
