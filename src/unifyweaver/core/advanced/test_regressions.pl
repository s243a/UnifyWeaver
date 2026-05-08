:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_regressions.pl - Regression tests for fixed issues
% Ensures previously fixed bugs don't resurface

:- module(test_regressions, [
    test_regressions/0
]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(pattern_matchers).
:- use_module(linear_recursion).
:- use_module(tail_recursion).
:- use_module(tree_recursion).
:- use_module('../constraint_analyzer').

% Import is_transitive_closure from parent module
:- use_module('../recursive_compiler', [is_transitive_closure/5]).

%% test_regressions/0
%  Run all regression tests for POST_RELEASE_TODO items 1-3
test_regressions :-
    run_tests([advanced_regressions]).

:- begin_tests(advanced_regressions).

%% list_length_detection
%  Regression test for POST_RELEASE_TODO Item 1
%  Bug: list_length/2 was not detected as linear recursion
%  Fix: Updated has_structural_head_pattern to distinguish [H|T] from [V,L,R]
test(list_length_detection, [setup(clear_user_predicate(list_length/2)),
                             cleanup(clear_user_predicate(list_length/2))]) :-
    assertz(user:(list_length([], 0))),
    assertz(user:(list_length([_|T], N) :- list_length(T, N1), N is N1 + 1)),
    assertion(is_linear_recursive_streamable(list_length/2)),
    assertion(can_compile_linear_recursion(list_length/2)),
    compile_linear_recursion(list_length/2, [], Code),
    assertion(Code \= ""),
    assertion(sub_string(Code, _, _, _, "_memo")).

%% descendant_classification
%  Regression test for POST_RELEASE_TODO Item 2
%  Bug: descendant/2 was misclassified as tail_recursion
%  Fix: Added reverse transitive closure pattern support
test(descendant_classification, [setup((clear_user_predicate(parent/2),
                                        clear_user_predicate(descendant/2))),
                                 cleanup((clear_user_predicate(parent/2),
                                          clear_user_predicate(descendant/2)))]) :-
    assertz(user:(parent(alice, bob))),
    assertz(user:(parent(bob, charlie))),
    assertz(user:(descendant(X, Y) :- parent(X, Y))),
    assertz(user:(descendant(X, Z) :- parent(X, Y), descendant(Y, Z))),
    functor(Head, descendant, 2),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    assertion(member(clause(_, (parent(_, _), descendant(_, _))), Clauses)).

%% linear_recursion_codegen
%  Regression test for POST_RELEASE_TODO Item 3
%  Bug: Linear recursion bash generation had TODO placeholders
%  Fix: Implemented fold-based code generation with variable translation
test(linear_recursion_codegen, [setup(clear_user_predicate(factorial/2)),
                                cleanup(clear_user_predicate(factorial/2))]) :-
    assertz(user:(factorial(0, 1))),
    assertz(user:(factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1)),
    assertion(can_compile_linear_recursion(factorial/2)),
    compile_linear_recursion(factorial/2, [], Code),
    assertion(Code \= ""),
    assertion(sub_string(Code, _, _, _, "factorial")),
    assertion(sub_string(Code, _, _, _, "_memo")).

%% fibonacci_exclusion
%  Regression test for bug found during Item 3 implementation
%  Bug: Fibonacci (2 recursive calls) was incorrectly detected as linear
%  Fix: Added count check to require exactly 1 recursive call
test(fibonacci_exclusion, [setup(clear_user_predicate(fibonacci/2)),
                           cleanup(clear_user_predicate(fibonacci/2))]) :-
    assertz(user:(fibonacci(0, 0))),
    assertz(user:(fibonacci(1, 1))),
    assertz(user:(fibonacci(N, F) :-
        N > 1,
        N1 is N - 1,
        N2 is N - 2,
        fibonacci(N1, F1),
        fibonacci(N2, F2),
        F is F1 + F2
    )),
    assertion(is_tree_recursive(fibonacci/2)),
    assertion(\+ is_linear_recursive_streamable(fibonacci/2)).

%% runtime_option_precedence
%  Regression test for explicit caller options overriding declared/default
%  constraints in advanced recursion compilers.
test(runtime_option_precedence, [setup((clear_user_predicate(tail_override/3),
                                        clear_user_predicate(linear_override/2),
                                        clear_constraints(linear_override/2))),
                                 cleanup((clear_user_predicate(tail_override/3),
                                          clear_user_predicate(linear_override/2),
                                          clear_constraints(linear_override/2)))]) :-
    assertz(user:(tail_override([], Acc, Acc))),
    assertz(user:(tail_override([_|T], Acc, N) :- Acc1 is Acc + 1, tail_override(T, Acc1, N))),
    compile_tail_recursion(tail_override/3, [unique(false), target(bash)], TailCode),
    assertion(\+ sub_string(TailCode, _, _, _, "Unique constraint")),
    assertz(user:(linear_override(0, 0))),
    assertz(user:(linear_override(N, S) :- N > 0, N1 is N - 1, linear_override(N1, S1), S is N + S1)),
    declare_constraint(linear_override/2, [unique(false)]),
    compile_linear_recursion(linear_override/2, [unique(true), target(bash)], LinearCode),
    clear_constraints(linear_override/2),
    assertion(sub_string(LinearCode, _, _, _, "standard strategy")),
    assertion(\+ sub_string(LinearCode, _, _, _, "Memoization disabled")).

:- end_tests(advanced_regressions).

clear_user_predicate(Name/Arity) :-
    functor(Head, Name, Arity),
    retractall(user:Head),
    catch(abolish(user:Name/Arity), _, true).
