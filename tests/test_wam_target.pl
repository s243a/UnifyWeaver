:- encoding(utf8).
% Test suite for WAM target
% Usage: swipl -g run_tests -t halt tests/test_wam_target.pl

:- use_module('../src/unifyweaver/targets/wam_target').

%% Test data (facts) - MUST BE DYNAMIC for clause/2 to work across modules
:- dynamic test_parent/2, test_grandparent/2, test_ancestor/2.

test_parent(alice, bob).
test_parent(bob, charlie).

%% Grandparent rule
test_grandparent(X, Z) :- test_parent(X, Y), test_parent(Y, Z).

%% Recursive ancestor rule
test_ancestor(X, Y) :- test_parent(X, Y).
test_ancestor(X, Y) :- test_parent(X, Z), test_ancestor(Z, Y).

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]).

%% Tests
test_wam_facts :-
    Test = 'WAM: compile_facts',
    (   wam_target:compile_facts_to_wam(user:test_parent, 2, Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'test_parent/2:'),
        sub_string(S, _, _, _, 'get_constant alice, A1'),
        sub_string(S, _, _, _, 'try_me_else'),
        sub_string(S, _, _, _, 'trust_me')
    ->  pass(Test)
    ;   fail_test(Test, 'Incorrect WAM output for facts')
    ).

test_wam_single_clause :-
    Test = 'WAM: single clause rule',
    (   wam_target:compile_predicate_to_wam(user:test_grandparent/2, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'test_grandparent/2:'),
        sub_string(S, _, _, _, 'allocate'),
        sub_string(S, _, _, _, 'get_variable'),
        sub_string(S, _, _, _, 'call test_parent/2'),
        sub_string(S, _, _, _, 'deallocate'),
        sub_string(S, _, _, _, 'execute test_parent/2')
    ->  pass(Test)
    ;   wam_target:compile_predicate_to_wam(user:test_grandparent/2, [], Code2),
        format(user_error, 'DEBUG: single clause output:~n~w~n', [Code2]),
        fail_test(Test, 'Incorrect WAM output for single clause')
    ).

test_wam_recursion :-
    Test = 'WAM: recursive rule (ancestor)',
    (   wam_target:compile_predicate_to_wam(user:test_ancestor/2, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'test_ancestor/2:'),
        sub_string(S, _, _, _, 'try_me_else'),
        sub_string(S, _, _, _, 'trust_me'),
        sub_string(S, _, _, _, 'execute test_parent/2'),
        sub_string(S, _, _, _, 'allocate'),
        sub_string(S, _, _, _, 'call test_parent/2'),
        sub_string(S, _, _, _, 'deallocate'),
        sub_string(S, _, _, _, 'execute test_ancestor/2')
    ->  pass(Test)
    ;   wam_target:compile_predicate_to_wam(user:test_ancestor/2, [], Code2),
        format(user_error, 'DEBUG: recursion output:~n~w~n', [Code2]),
        fail_test(Test, 'Incorrect WAM output for recursion')
    ).

test_wam_module :-
    Test = 'WAM: compile_wam_module (template)',
    (   wam_target:compile_wam_module([user:test_parent/2, user:test_grandparent/2], [module_name('FamilyModule')], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'WAM Module: FamilyModule'),
        sub_string(S, _, _, _, 'test_parent/2:'),
        sub_string(S, _, _, _, 'test_grandparent/2:')
    ->  pass(Test)
    ;   wam_target:compile_wam_module([user:test_parent/2, user:test_grandparent/2], [module_name('FamilyModule')], Code2),
        format(user_error, 'DEBUG: Failed Code:~n~w~n', [Code2]),
        fail_test(Test, 'Incorrect WAM module output from template')
    ).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('WAM Target Test Suite~n'),
    format('========================================~n~n'),
    
    test_wam_facts,
    test_wam_single_clause,
    test_wam_recursion,
    test_wam_module,
    
    format('~n========================================~n'),
    format('All tests completed~n'),
    format('========================================~n').

:- initialization(run_tests, main).
