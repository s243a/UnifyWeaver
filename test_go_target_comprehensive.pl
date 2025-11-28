:- encoding(utf8).
% Comprehensive Go target tests
% Tests all major features: facts, rules, match, multiple rules

:- use_module('src/unifyweaver/targets/go_target').

%% Test 1: Facts compilation
user(alice, 25).
user(bob, 30).
user(charlie, 28).

test_facts :-
    write('=== Test 1: Facts Compilation ==='), nl,
    go_target:compile_predicate_to_go(user/2, [], Code),
    go_target:write_go_program(Code, 'output_test/test_facts.go'),
    write('Generated: output_test/test_facts.go'), nl, nl.

%% Test 2: Single rule - field reordering
parent(alice, bob).
parent(bob, charlie).

child(C, P) :- parent(P, C).

test_single_rule :-
    write('=== Test 2: Single Rule - Field Reordering ==='), nl,
    go_target:compile_predicate_to_go(child/2, [], Code),
    go_target:write_go_program(Code, 'output_test/test_child.go'),
    write('Generated: output_test/test_child.go'), nl, nl.

%% Test 3: Match predicate - boolean filtering
log('ERROR: connection timeout').
log('WARNING: slow response').
log('INFO: operation completed').
log('ERROR: authentication failed').

error_log(Line) :-
    log(Line),
    match(Line, 'ERROR').

test_match :-
    write('=== Test 3: Match Predicate ==='), nl,
    go_target:compile_predicate_to_go(error_log/1, [], Code),
    go_target:write_go_program(Code, 'output_test/test_error_log.go'),
    write('Generated: output_test/test_error_log.go'), nl, nl.

%% Test 4: Multiple rules - OR pattern
event('ERROR: disk full').
event('WARNING: low memory').
event('CRITICAL: service down').
event('INFO: startup complete').

alert(Line) :-
    event(Line),
    match(Line, 'ERROR').

alert(Line) :-
    event(Line),
    match(Line, 'WARNING').

alert(Line) :-
    event(Line),
    match(Line, 'CRITICAL').

test_multiple_rules :-
    write('=== Test 4: Multiple Rules - OR Pattern ==='), nl,
    go_target:compile_predicate_to_go(alert/1, [], Code),
    go_target:write_go_program(Code, 'output_test/test_alert.go'),
    write('Generated: output_test/test_alert.go'), nl, nl.

%% Test 5: Projection - extract single field
user_name(Name) :- user(Name, _).

test_projection :-
    write('=== Test 5: Projection - Single Field ==='), nl,
    go_target:compile_predicate_to_go(user_name/1, [], Code),
    go_target:write_go_program(Code, 'output_test/test_user_name.go'),
    write('Generated: output_test/test_user_name.go'), nl, nl.

%% Test 6: Tab delimiter
test_tab_delimiter :-
    write('=== Test 6: Tab Delimiter ==='), nl,
    go_target:compile_predicate_to_go(child/2, [field_delimiter(tab)], Code),
    go_target:write_go_program(Code, 'output_test/test_child_tab.go'),
    write('Generated: output_test/test_child_tab.go'), nl, nl.

%% Run all tests
run_all_tests :-
    test_facts,
    test_single_rule,
    test_match,
    test_multiple_rules,
    test_projection,
    test_tab_delimiter,
    write('========================================'), nl,
    write('All tests completed!'), nl,
    write('========================================'), nl.

% Usage:
% ?- consult('test_go_target_comprehensive.pl').
% ?- run_all_tests.
