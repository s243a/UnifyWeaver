:- encoding(utf8).
% Comprehensive tests for AWK target constraint mapping

:- use_module('src/unifyweaver/targets/awk_target').

% Test data
person(alice, 25).
person(bob, 17).
person(charlie, 45).
person(dave, 30).

% Test 1: Greater than constraint
adult(X, Age) :- person(X, Age), Age > 18.

% Test 2: Less than constraint
child(X, Age) :- person(X, Age), Age < 18.

% Test 3: Range constraint (combined)
working_age(X, Age) :- person(X, Age), Age >= 18, Age =< 65.

% Test 4: Inequality constraint
different(X, Y) :- X \= Y.

% Test 5: Arithmetic with is
double_age(X, Double) :- person(X, Age), Double is Age * 2.

% Run all tests
test_gt_constraint :-
    write('=== Test 1: Greater than (>)==='), nl, nl,
    awk_target:compile_predicate_to_awk(adult/2,
        [unique(true)], AwkCode),
    write(AwkCode), nl, nl.

test_lt_constraint :-
    write('=== Test 2: Less than (<) ==='), nl, nl,
    awk_target:compile_predicate_to_awk(child/2,
        [unique(true)], AwkCode),
    write(AwkCode), nl, nl.

test_range_constraint :-
    write('=== Test 3: Range (>= and =<) ==='), nl, nl,
    awk_target:compile_predicate_to_awk(working_age/2,
        [unique(true)], AwkCode),
    write(AwkCode), nl, nl.

test_inequality_constraint :-
    write('=== Test 4: Inequality (\\=) ==='), nl, nl,
    awk_target:compile_predicate_to_awk(different/2,
        [unique(true)], AwkCode),
    write(AwkCode), nl, nl.

run_all_tests :-
    test_gt_constraint,
    test_lt_constraint,
    test_range_constraint,
    test_inequality_constraint,
    write('All constraint tests completed!'), nl.

% Usage:
% ?- consult('test_awk_constraints.pl').
% ?- run_all_tests.
