:- encoding(utf8).
% End-to-end test for WAM target and runtime
% Usage: swipl -g run_tests -t halt tests/test_wam_e2e.pl

:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/runtime/wam_runtime').

%% Test data
e2e_parent(alice, bob).
e2e_parent(bob, charlie).

%% Rule — exercises allocate/deallocate, call/execute, Yi registers
:- dynamic e2e_grandparent/2.
e2e_grandparent(X, Z) :- e2e_parent(X, Y), e2e_parent(Y, Z).

%% Recursive rule — multi-clause, backtracking, recursive calls
:- dynamic e2e_ancestor/2.
e2e_ancestor(X, Y) :- e2e_parent(X, Y).
e2e_ancestor(X, Y) :- e2e_parent(X, Z), e2e_ancestor(Z, Y).

%% Compound head facts — exercises get_structure + unify_* in the runtime
:- dynamic e2e_color/2.
e2e_color(rgb(255, 0, 0), red).
e2e_color(rgb(0, 255, 0), green).

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Tests
test_wam_compilation_and_execution :-
    Test = 'WAM E2E: Fact compilation and execution',
    (   % 1. Compile facts to WAM
        wam_target:compile_facts_to_wam(user:e2e_parent, 2, Code),
        % 2. Execute WAM code with query: e2e_parent(alice, bob)?
        wam_runtime:execute_wam(Code, e2e_parent(alice, bob), _FinalRegs)
    ->  pass(Test)
    ;   fail_test(Test, 'E2E execution failed')
    ).

test_wam_backtracking_simple :-
    Test = 'WAM E2E: Simple backtracking (second fact)',
    (   % 1. Compile facts
        wam_target:compile_facts_to_wam(user:e2e_parent, 2, Code),
        % 2. Execute with query: e2e_parent(bob, charlie)?
        % This exercises the backtracking path since the first fact (alice, bob) will fail.
        wam_runtime:execute_wam(Code, e2e_parent(bob, charlie), _)
    ->  pass(Test)
    ;   fail_test(Test, 'E2E execution failed for second fact')
    ).

test_wam_rule_execution :-
    Test = 'WAM E2E: Rule compilation and execution (grandparent)',
    (   % 1. Compile both parent facts and grandparent rule
        wam_target:compile_wam_module(
            [user:e2e_parent/2, user:e2e_grandparent/2], [], Code),
        % 2. Execute: grandparent(alice, charlie)? should succeed
        %    (alice->bob via parent, bob->charlie via parent)
        wam_runtime:execute_wam(Code, e2e_grandparent(alice, charlie), _)
    ->  pass(Test)
    ;   fail_test(Test, 'E2E rule execution failed for grandparent')
    ).

test_wam_recursive_execution :-
    Test = 'WAM E2E: Recursive rule (ancestor)',
    (   % Compile parent + ancestor, execute ancestor(alice, charlie)
        % Path: alice->bob (parent), bob->charlie (parent) via recursive clause
        wam_target:compile_wam_module(
            [user:e2e_parent/2, user:e2e_ancestor/2], [], Code),
        wam_runtime:execute_wam(Code, e2e_ancestor(alice, charlie), _)
    ->  pass(Test)
    ;   fail_test(Test, 'Recursive ancestor execution failed')
    ).

test_wam_compound_head :-
    Test = 'WAM E2E: Compound head unification (get_structure)',
    (   % Compile color/2 with compound first arg, query rgb(255,0,0) -> red
        wam_target:compile_predicate_to_wam(user:e2e_color/2, [], Code),
        wam_runtime:execute_wam(Code, e2e_color(rgb(255, 0, 0), red), _)
    ->  pass(Test)
    ;   fail_test(Test, 'Compound head unification failed')
    ).

run_tests :-
    format('~n========================================~n'),
    format('WAM Target E2E Test Suite~n'),
    format('========================================~n~n'),
    
    test_wam_compilation_and_execution,
    test_wam_backtracking_simple,
    test_wam_rule_execution,
    test_wam_recursive_execution,
    test_wam_compound_head,
    
    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'),
        format('========================================~n'),
        halt(1)
    ;   format('All E2E tests passed~n'),
        format('========================================~n')
    ).

:- initialization(run_tests, main).
