:- encoding(utf8).
% Test file for AWK target

% Load the necessary modules
:- use_module('src/unifyweaver/core/recursive_compiler').
:- use_module('src/unifyweaver/targets/awk_target').

% Simple facts for testing
person(alice).
person(bob).
person(charlie).

city(newyork).
city(london).
city(tokyo).

% Fact with arity 2
parent(alice, bob).
parent(bob, charlie).
parent(alice, charlie).

% Test compilation
test_awk_facts :-
    write('=== Testing AWK compilation of person/1 facts ==='), nl,
    awk_target:compile_predicate_to_awk(person/1,
        [record_format(tsv), unique(true)], AwkCode),
    write('Generated AWK code:'), nl,
    write(AwkCode), nl, nl.

test_awk_facts_arity2 :-
    write('=== Testing AWK compilation of parent/2 facts ==='), nl,
    awk_target:compile_predicate_to_awk(parent/2,
        [record_format(tsv), unique(true)], AwkCode),
    write('Generated AWK code:'), nl,
    write(AwkCode), nl, nl.

test_awk_via_compiler :-
    write('=== Testing AWK via recursive_compiler ==='), nl,
    recursive_compiler:compile_recursive(city/1,
        [target(awk), record_format(tsv)], AwkCode),
    write('Generated AWK code:'), nl,
    write(AwkCode), nl, nl.

% Run all tests
run_tests :-
    test_awk_facts,
    test_awk_facts_arity2,
    test_awk_via_compiler,
    write('All tests completed!'), nl.
