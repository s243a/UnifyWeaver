:- encoding(utf8).
% Test suite for WAM target
% Usage: swipl -g run_tests -t halt tests/test_wam_target.pl

:- use_module('../src/unifyweaver/targets/wam_target').

%% Test data (facts) - MUST BE DYNAMIC for clause/2 to work across modules
:- dynamic test_parent/2, test_grandparent/2, test_ancestor/2, test_wrap/2.

test_parent(alice, bob).
test_parent(bob, charlie).

%% Grandparent rule
test_grandparent(X, Z) :- test_parent(X, Y), test_parent(Y, Z).

%% Recursive ancestor rule
test_ancestor(X, Y) :- test_parent(X, Y).
test_ancestor(X, Y) :- test_parent(X, Z), test_ancestor(Z, Y).

%% Rule with compound body argument — exercises put_structure
%  The body goal `test_check(pair(X, done))` has a compound argument.
:- dynamic test_check/1.
test_check(pair(_, _)).
test_wrap(X) :- test_check(pair(X, done)).

%% Compound head — exercises get_structure + unify_constant
:- dynamic test_color/2.
test_color(rgb(255, 0, 0), red).

%% Multi-clause body containing a findall — regression for the bug
%  where compile_clauses_fragments only emitted allocate/deallocate
%  when goal-count > 1, missing the case where a single surface goal
%  is itself a findall/aggregate that internally produces a Call.
%  Without an env frame, the post-end_aggregate continuation has no
%  saved caller cp to return to; finalise_aggregate's cp chain loops
%  back into k2 forever. See compile_single_clause_wam/3 for the
%  same logic that was previously only applied to single-clause.
:- dynamic test_multi_findall/1, test_multi_inner/1.
test_multi_inner(1).
test_multi_inner(2).
test_multi_findall('a') :- findall(_, test_multi_inner(_), _).
test_multi_findall('b') :- findall(_, test_multi_inner(_), _).
test_multi_findall('c') :- findall(_, test_multi_inner(_), _).

%% Nested compound body arg — exercises recursive put_structure
:- dynamic test_nested_check/1.
test_nested_check(box(inner(_, _))).
:- dynamic test_nested_wrap/1.
test_nested_wrap(X) :- test_nested_check(box(inner(X, done))).

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

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
        sub_string(S, _, _, _, 'put_value'),
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
        sub_string(S, _, _, _, 'put_value'),
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

test_wam_put_structure :-
    Test = 'WAM: put_structure for compound body args',
    (   wam_target:compile_predicate_to_wam(user:test_wrap/1, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'put_structure pair/2')
    ->  pass(Test)
    ;   (   wam_target:compile_predicate_to_wam(user:test_wrap/1, [], Code2)
        ->  format(user_error, 'DEBUG: put_structure output:~n~w~n', [Code2])
        ;   format(user_error, 'DEBUG: compile_predicate_to_wam failed~n', [])
        ),
        fail_test(Test, 'Missing put_structure in compound body arg output')
    ).

test_wam_compound_head :-
    Test = 'WAM: compound head (get_structure)',
    (   wam_target:compile_predicate_to_wam(user:test_color/2, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'get_structure rgb/3'),
        sub_string(S, _, _, _, 'unify_constant 255'),
        sub_string(S, _, _, _, 'get_constant red')
    ->  pass(Test)
    ;   (   wam_target:compile_predicate_to_wam(user:test_color/2, [], Code2)
        ->  format(user_error, 'DEBUG: compound head output:~n~w~n', [Code2])
        ;   format(user_error, 'DEBUG: compile failed~n', [])
        ),
        fail_test(Test, 'Missing get_structure in compound head output')
    ).

test_wam_nested_put_structure :-
    Test = 'WAM: nested put_structure for compound body args',
    (   wam_target:compile_predicate_to_wam(user:test_nested_wrap/1, [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'put_structure box/1'),
        sub_string(S, _, _, _, 'put_structure inner/2')
    ->  pass(Test)
    ;   (   wam_target:compile_predicate_to_wam(user:test_nested_wrap/1, [], Code2)
        ->  format(user_error, 'DEBUG: nested put_structure output:~n~w~n', [Code2])
        ;   format(user_error, 'DEBUG: compile failed~n', [])
        ),
        fail_test(Test, 'Missing nested put_structure in output')
    ).

%% Multi-clause findall regression: every clause must have an
%  allocate/deallocate pair so the inner findall's post-end_aggregate
%  continuation can retrieve the caller cp from the env frame.
test_wam_multi_clause_findall_emits_allocate :-
    Test = 'WAM: multi-clause body with findall emits allocate/deallocate per clause',
    (   wam_target:compile_predicate_to_wam(user:test_multi_findall/1, [], Code),
        atom_string(Code, S),
        % Each of the three clause bodies should contain begin_aggregate
        % surrounded by allocate ... deallocate. Three pairs total.
        aggsubs_count(S, 'allocate', AllocCount),
        aggsubs_count(S, 'deallocate', DeallocCount),
        aggsubs_count(S, 'begin_aggregate', AggCount),
        AllocCount >= 3,
        DeallocCount >= 3,
        AggCount =:= 3
    ->  pass(Test)
    ;   wam_target:compile_predicate_to_wam(user:test_multi_findall/1, [], Code2),
        format(user_error, 'DEBUG: multi-clause findall output:~n~w~n', [Code2]),
        fail_test(Test, 'Multi-clause findall body missing allocate/deallocate per clause')
    ).

%% Count non-overlapping occurrences of Sub in S.
aggsubs_count(S, Sub, N) :-
    findall(_, sub_string(S, _, _, _, Sub), Occurrences),
    length(Occurrences, N).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('WAM Target Test Suite~n'),
    format('========================================~n~n'),
    
    test_wam_facts,
    test_wam_single_clause,
    test_wam_recursion,
    test_wam_put_structure,
    test_wam_nested_put_structure,
    test_wam_compound_head,
    test_wam_module,
    test_wam_multi_clause_findall_emits_allocate,
    
    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'),
        format('========================================~n'),
        halt(1)
    ;   format('All tests passed~n'),
        format('========================================~n')
    ).

:- initialization(run_tests, main).
