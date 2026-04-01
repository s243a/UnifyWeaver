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

%% Nested compound head — exercises get_structure on non-Ai register
:- dynamic e2e_nested/2.
e2e_nested(pair(a, b), yes).
e2e_nested(pair(c, d), no).

%% Rule that calls a predicate with compound head using a variable arg
%% This exercises write mode: grandparent computes Y via parent(alice,Y),
%% then calls parent(Y,Z) where Z is an unbound _V variable.
%% parent's get_constant on A2 must unify _V with the constant.
%% (Already tested by grandparent, but let's add an explicit write-mode
%% test by calling a compound-head predicate with a variable first arg.)
:- dynamic e2e_lookup/1.
e2e_lookup(X) :- e2e_color(rgb(X, 0, 0), red).

%% Built-in arithmetic — exercises builtin_call is/2 and >/2
:- dynamic e2e_double/2.
e2e_double(X, Y) :- Y is X * 2.

:- dynamic e2e_positive/1.
e2e_positive(X) :- X > 0.

%% Type check built-in — exercises builtin_call atom/1
:- dynamic e2e_is_atom/1.
e2e_is_atom(X) :- atom(X).

%% Predicate for solve_wam test — multi-clause facts
:- dynamic e2e_capital/2.
e2e_capital(france, paris).
e2e_capital(germany, berlin).

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

test_wam_nested_compound_head :-
    Test = 'WAM E2E: Nested compound head (get_structure on Xn)',
    (   % pair(a,b) has nested structure — get_structure on A1, then
        % unify_* for sub-args. Tests get_structure on non-Ai register
        % when compiler emits nested get_structure sequences.
        wam_target:compile_predicate_to_wam(user:e2e_nested/2, [], Code),
        wam_runtime:execute_wam(Code, e2e_nested(pair(a, b), yes), _)
    ->  pass(Test)
    ;   fail_test(Test, 'Nested compound head failed')
    ).

test_wam_write_mode :-
    Test = 'WAM E2E: Write mode (variable in compound query)',
    (   % e2e_lookup(X) calls e2e_color(rgb(X,0,0), red).
        % X is unbound, so put_structure builds rgb(_V,0,0) in A1.
        % color's get_structure enters read mode on the constructed term.
        % The unify_constant 255 must unify with the _V variable at
        % position 1, exercising variable unification within structures.
        wam_target:compile_wam_module(
            [user:e2e_color/2, user:e2e_lookup/1], [], Code),
        wam_runtime:execute_wam(Code, e2e_lookup(255), _)
    ->  pass(Test)
    ;   fail_test(Test, 'Write mode / variable compound query failed')
    ).

test_wam_builtin_arithmetic :-
    Test = 'WAM E2E: Built-in arithmetic (is/2, >/2)',
    (   wam_target:compile_predicate_to_wam(user:e2e_positive/1, [], Code),
        wam_runtime:execute_wam(Code, e2e_positive(5), _)
    ->  pass(Test)
    ;   fail_test(Test, 'Built-in arithmetic failed')
    ).

test_wam_builtin_is :-
    Test = 'WAM E2E: Built-in is/2 evaluation',
    (   wam_target:compile_predicate_to_wam(user:e2e_double/2, [], Code),
        wam_runtime:execute_wam(Code, e2e_double(3, 6), _)
    ->  pass(Test)
    ;   fail_test(Test, 'Built-in is/2 failed')
    ).

test_wam_builtin_type_check :-
    Test = 'WAM E2E: Built-in type check (atom/1)',
    (   wam_target:compile_predicate_to_wam(user:e2e_is_atom/1, [], Code),
        wam_runtime:execute_wam(Code, e2e_is_atom(hello), _)
    ->  pass(Test)
    ;   fail_test(Test, 'Built-in type check failed')
    ).

test_wam_solve :-
    Test = 'WAM E2E: solve_wam variable bindings',
    (   wam_target:compile_facts_to_wam(user:e2e_capital, 2, Code),
        wam_runtime:solve_wam(Code, e2e_capital(france, City),
            ['City'=City], Bindings),
        Bindings = ['City'=paris]
    ->  pass(Test)
    ;   fail_test(Test, 'solve_wam binding extraction failed')
    ).

test_wam_indexing :-
    Test = 'WAM E2E: First-argument indexing (switch_on_constant)',
    (   % Compile parent/2 — should have switch_on_constant index
        wam_target:compile_facts_to_wam(user:e2e_parent, 2, Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, 'switch_on_constant'),
        % Should still execute correctly with indexing
        wam_runtime:execute_wam(Code, e2e_parent(bob, charlie), _)
    ->  pass(Test)
    ;   fail_test(Test, 'First-argument indexing failed')
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
    test_wam_nested_compound_head,
    test_wam_write_mode,
    test_wam_builtin_arithmetic,
    test_wam_builtin_is,
    test_wam_builtin_type_check,
    test_wam_solve,
    test_wam_indexing,
    
    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'),
        format('========================================~n'),
        halt(1)
    ;   format('All E2E tests passed~n'),
        format('========================================~n')
    ).

:- initialization(run_tests, main).
