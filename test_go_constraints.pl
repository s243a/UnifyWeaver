:- encoding(utf8).
% Test Go constraint support

:- use_module('src/unifyweaver/targets/go_target').

% Test data
person(alice, 25).
person(bob, 17).
person(charlie, 45).

% Test 1: Greater than constraint
adult(X, Age) :- person(X, Age), Age > 18.

% Test 2: Less than constraint
child(X, Age) :- person(X, Age), Age < 18.

% Test 3: Range constraint (combined)
working_age(X, Age) :- person(X, Age), Age >= 18, Age =< 65.

test_gt :-
    write('=== Test: Greater than (>) ==='), nl, nl,
    go_target:compile_predicate_to_go(adult/2, [], Code),
    write(Code), nl, nl.

test_lt :-
    write('=== Test: Less than (<) ==='), nl, nl,
    go_target:compile_predicate_to_go(child/2, [], Code),
    write(Code), nl, nl.

test_range :-
    write('=== Test: Range (>= and =<) ==='), nl, nl,
    go_target:compile_predicate_to_go(working_age/2, [], Code),
    write(Code), nl, nl.

run_all :-
    test_gt,
    test_lt,
    test_range,
    write('All constraint tests completed!'), nl.

% Usage:
% ?- consult('test_go_constraints.pl').
% ?- run_all.
