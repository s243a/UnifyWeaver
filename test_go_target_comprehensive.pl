:- encoding(utf8).
% Comprehensive tests for Go target

:- use_module('src/unifyweaver/targets/go_target').

%% ============================================
%% TEST DATA - Facts
%% ============================================

% Simple facts
parent(alice, bob).
parent(bob, charlie).
parent(alice, dave).

user(john, 25).
user(jane, 30).
user(bob, 28).

%% ============================================
%% TEST DATA - Single Rules
%% ============================================

% Reverse relationship
child(C, P) :- parent(P, C).

% Different arity
user_name(Name) :- user(Name, _).

% Field reordering
age_user(Age, Name) :- user(Name, Age).

%% ============================================
%% TEST CASES
%% ============================================

test_facts_parent :-
    write('=== Test: Facts - parent/2 ==='), nl,
    go_target:compile_predicate_to_go(parent/2, [], Code),
    write(Code), nl.

test_facts_user :-
    write('=== Test: Facts - user/2 ==='), nl,
    go_target:compile_predicate_to_go(user/2, [], Code),
    write(Code), nl.

test_single_rule_child :-
    write('=== Test: Single Rule - child/2 ==='), nl,
    go_target:compile_predicate_to_go(child/2, [], Code),
    write(Code), nl.

test_single_rule_user_name :-
    write('=== Test: Single Rule - user_name/1 ==='), nl,
    go_target:compile_predicate_to_go(user_name/1, [], Code),
    write(Code), nl.

test_single_rule_age_user :-
    write('=== Test: Single Rule - age_user/2 ==='), nl,
    go_target:compile_predicate_to_go(age_user/2, [], Code),
    write(Code), nl.

%% Test with different delimiters
test_delimiter_tab :-
    write('=== Test: Tab delimiter ==='), nl,
    go_target:compile_predicate_to_go(child/2, [field_delimiter(tab)], Code),
    write(Code), nl.

test_delimiter_comma :-
    write('=== Test: Comma delimiter ==='), nl,
    go_target:compile_predicate_to_go(child/2, [field_delimiter(comma)], Code),
    write(Code), nl.

%% Write programs to files
write_all_programs :-
    write('=== Writing all test programs ==='), nl,

    go_target:compile_predicate_to_go(parent/2, [], ParentCode),
    go_target:write_go_program(ParentCode, 'output_test/parent.go'),

    go_target:compile_predicate_to_go(user/2, [], UserCode),
    go_target:write_go_program(UserCode, 'output_test/user.go'),

    go_target:compile_predicate_to_go(child/2, [], ChildCode),
    go_target:write_go_program(ChildCode, 'output_test/child.go'),

    go_target:compile_predicate_to_go(user_name/1, [], UserNameCode),
    go_target:write_go_program(UserNameCode, 'output_test/user_name.go'),

    go_target:compile_predicate_to_go(age_user/2, [], AgeUserCode),
    go_target:write_go_program(AgeUserCode, 'output_test/age_user.go'),

    write('All programs written to output_test/'), nl.

run_all_tests :-
    test_facts_parent,
    test_facts_user,
    test_single_rule_child,
    test_single_rule_user_name,
    test_single_rule_age_user,
    test_delimiter_tab,
    test_delimiter_comma,
    write('All tests completed!'), nl.

% Usage:
% ?- consult('test_go_target_comprehensive.pl').
% ?- run_all_tests.
% ?- write_all_programs.
