:- encoding(utf8).
% Test Go target - basic facts compilation

:- use_module('src/unifyweaver/targets/go_target').

% Test facts
parent(alice, bob).
parent(bob, charlie).
parent(alice, dave).

% Test single rule - reverse relationship
child(C, P) :- parent(P, C).

% Test compilation
test_facts :-
    write('=== Test: Compile Facts to Go ==='), nl, nl,
    go_target:compile_predicate_to_go(parent/2, [], GoCode),
    write(GoCode), nl.

test_single_rule :-
    write('=== Test: Compile Single Rule to Go ==='), nl, nl,
    go_target:compile_predicate_to_go(child/2, [], GoCode),
    write(GoCode), nl.

test_write_file :-
    write('=== Test: Write Go Program to File ==='), nl,
    go_target:compile_predicate_to_go(parent/2, [], GoCode),
    go_target:write_go_program(GoCode, 'test_parent.go').

test_write_child :-
    write('=== Test: Write Child Program to File ==='), nl,
    go_target:compile_predicate_to_go(child/2, [], GoCode),
    go_target:write_go_program(GoCode, 'test_child.go').

run_all :-
    test_facts,
    write('Go target test completed!'), nl.

% Usage:
% ?- consult('test_go_target.pl').
% ?- run_all.
