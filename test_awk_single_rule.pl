:- encoding(utf8).
% Test AWK target with single rules

:- use_module('src/unifyweaver/core/recursive_compiler').
:- use_module('src/unifyweaver/targets/awk_target').

% Facts
parent(alice, bob).
parent(bob, charlie).
parent(alice, charlie).

% Single rule: same as parent
parent_copy(X, Y) :- parent(X, Y).

% Test constraint-only rule
adult(X) :- X > 18.

% Run tests
test_single_predicate_rule :-
    write('=== Test 1: Single predicate rule (parent_copy) ==='), nl, nl,
    awk_target:compile_predicate_to_awk(parent_copy/2,
        [record_format(tsv), unique(true)], AwkCode),
    write(AwkCode), nl, nl.

test_constraint_only :-
    write('=== Test 2: Constraint-only rule (adult) ==='), nl, nl,
    awk_target:compile_predicate_to_awk(adult/1,
        [record_format(tsv), unique(true)], AwkCode),
    write(AwkCode), nl, nl.

run_tests :-
    test_single_predicate_rule,
    test_constraint_only,
    write('Tests completed!'), nl.

% Usage:
% ?- consult('test_awk_single_rule.pl').
% ?- run_tests.
