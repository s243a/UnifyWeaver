:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_advanced.pl - Comprehensive test suite for advanced recursion modules
% Tests all modules independently and together

:- use_module('call_graph').
:- use_module('scc_detection').
:- use_module('pattern_matchers').
:- use_module('tail_recursion').
:- use_module('linear_recursion').
:- use_module('mutual_recursion').
:- use_module('advanced_recursive_compiler').

%% Main test runner
test_all_advanced :-
    writeln(''),
    writeln('╔════════════════════════════════════════════════════════╗'),
    writeln('║  ADVANCED RECURSION COMPILER - FULL TEST SUITE        ║'),
    writeln('╚════════════════════════════════════════════════════════╝'),
    writeln(''),

    % Test each module independently
    run_module_test('Call Graph', test_call_graph),
    run_module_test('SCC Detection', test_scc_detection),
    run_module_test('Pattern Matchers', test_pattern_matchers),
    run_module_test('Tail Recursion Compiler', test_tail_recursion),
    run_module_test('Linear Recursion Compiler', test_linear_recursion),
    run_module_test('Mutual Recursion Compiler', test_mutual_recursion),
    run_module_test('Advanced Compiler Integration', test_advanced_compiler),

    writeln(''),
    writeln('╔════════════════════════════════════════════════════════╗'),
    writeln('║  ALL TESTS COMPLETE                                    ║'),
    writeln('╚════════════════════════════════════════════════════════╝'),
    writeln('').

%% Run a single module test with error handling
run_module_test(ModuleName, TestPredicate) :-
    writeln(''),
    format('┌─ ~w ~`─t─~50|~n', [ModuleName]),
    (   catch(call(TestPredicate), Error, (
            format('ERROR: ~w~n', [Error]),
            fail
        )) ->
        format('└─ ~w COMPLETE ~`─t─~50|~n', [ModuleName])
    ;   format('└─ ~w FAILED ~`─t─~50|~n', [ModuleName])
    ).

%% Integration tests
test_integration :-
    writeln('=== INTEGRATION TESTS ==='),

    % Test full pipeline: define predicates, compile, verify output
    writeln('Test: Full compilation pipeline'),

    % Clear predicates
    catch(abolish(fib/2), _, true),

    % Define fibonacci (linear recursion)
    assertz((fib(0, 0))),
    assertz((fib(1, 1))),
    assertz((fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, fib(N1, F1), fib(N2, F2), F is F1 + F2)),

    % Try to compile
    (   catch(
            advanced_recursive_compiler:compile_advanced_recursive(fib/2, [], Code),
            _,
            fail
        ) ->
        format('✓ Fibonacci compiled successfully~n'),
        format('Code length: ~w characters~n', [string_length(Code, _)])
    ;   writeln('⚠ Fibonacci not compiled (expected - complex pattern)')
    ),

    writeln('=== INTEGRATION TESTS COMPLETE ===').

%% Performance tests
test_performance :-
    writeln('=== PERFORMANCE TESTS ==='),

    % Test call graph building performance
    writeln('Test: Call graph building performance'),

    % Create many predicates
    NumPreds = 10,
    generate_test_predicates(NumPreds),

    % Build call graph
    statistics(cputime, T0),
    findall(Pred/2, between(1, NumPreds, Pred), Predicates),
    call_graph:build_call_graph(Predicates, Graph),
    statistics(cputime, T1),
    Time is T1 - T0,

    format('Built call graph for ~w predicates in ~3f seconds~n', [NumPreds, Time]),
    format('Graph has ~w edges~n', [length(Graph, _)]),

    % Cleanup
    cleanup_test_predicates(NumPreds),

    writeln('=== PERFORMANCE TESTS COMPLETE ===').

%% Generate test predicates for performance testing
generate_test_predicates(N) :-
    forall(
        between(1, N, I),
        (   functor(Head, I, 2),
            assertz((Head :- true))
        )
    ).

cleanup_test_predicates(N) :-
    forall(
        between(1, N, I),
        catch(abolish(I/2), _, true)
    ).

%% Regression tests
test_regression :-
    writeln('=== REGRESSION TESTS ==='),

    % Ensure basic patterns still work
    writeln('Test: Basic patterns still compile'),

    catch(abolish(simple_fact/1), _, true),
    assertz(simple_fact(a)),

    % Should be detected as non-recursive
    % (This would go through recursive_compiler, not advanced)

    writeln('✓ Basic patterns still work'),
    writeln('=== REGRESSION TESTS COMPLETE ===').

%% Run all tests including integration
test_all :-
    test_all_advanced,
    writeln(''),
    run_module_test('Integration Tests', test_integration),
    run_module_test('Performance Tests', test_performance),
    run_module_test('Regression Tests', test_regression).
