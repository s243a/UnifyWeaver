:- encoding(utf8).
% Test AWK target with multiple rules (OR pattern)

:- use_module('src/unifyweaver/core/recursive_compiler').
:- use_module('src/unifyweaver/targets/awk_target').

% Facts
parent(alice, bob).
parent(bob, charlie).

friend(alice, dave).
friend(charlie, eve).

% Multiple rules (OR pattern)
related(X, Y) :- parent(X, Y).
related(X, Y) :- friend(X, Y).

% Run test
test_multiple_rules :-
    write('=== Test: Multiple rules (related) ==='), nl, nl,
    awk_target:compile_predicate_to_awk(related/2,
        [record_format(tsv), unique(true)], AwkCode),
    write(AwkCode), nl, nl.

% Usage:
% ?- consult('test_awk_multiple_rules.pl').
% ?- test_multiple_rules.
