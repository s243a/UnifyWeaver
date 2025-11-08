/**
 * C# Target Tests using Janus Bridge
 *
 * This module provides tests for C# code generation using the Python helper
 * via the Janus bridge. This approach provides better subprocess handling
 * than Prolog's process_create/3.
 *
 * Prerequisites:
 *   - SWI-Prolog with Janus support
 *   - Python 3.8+
 *   - .NET SDK 9.0+
 *
 * Usage:
 *   ?- run_tests(csharp_janus).
 */

:- module(test_csharp_janus, [
    test_csharp_compilation/0,
    test_stream_target_simple_facts/0,
    test_stream_target_join/0
]).

:- use_module(library(plunit)).
:- use_module(library(janus)).

% Add helpers directory to Python path
:- initialization(setup_python_path, now).

setup_python_path :-
    working_directory(CWD, CWD),
    absolute_file_name('tests/helpers', HelpersDir,
                       [relative_to(CWD), file_type(directory)]),
    py_add_lib_dir(HelpersDir),
    format('[CSharpJanusTest] Added Python path: ~w~n', [HelpersDir]).

% Import the Python test helper module
:- py_call(importlib:import_module('csharp_test_helper'), _Module).

/**
 * compile_and_test_csharp(+Code, +ExpectedSubstring, -Result)
 *
 * Compile and run C# code, checking for expected output
 */
compile_and_test_csharp(Code, ExpectedSubstring, Result) :-
    py_call(csharp_test_helper:assert_output_contains(Code, ExpectedSubstring, 'janus_test'),
            Result).

/**
 * compile_and_run_csharp(+Code, -Result)
 *
 * Compile and run C# code, returning full result
 */
compile_and_run_csharp(Code, Result) :-
    py_call(csharp_test_helper:compile_and_run(Code, 'janus_test'),
            Result).

/**
 * check_dotnet_available/0
 *
 * Check if dotnet CLI is available
 */
check_dotnet_available :-
    py_call(csharp_test_helper:get_dotnet_version(), Version),
    (   Version = @(none)
    ->  format('[CSharpJanusTest] WARNING: dotnet CLI not found~n'),
        fail
    ;   format('[CSharpJanusTest] dotnet version: ~w~n', [Version])
    ).

% =====================================================================
% PLUnit Test Suite
% =====================================================================

:- begin_tests(csharp_janus, [
    condition(check_dotnet_available)
]).

test(stream_target_simple_facts, [
    condition(check_dotnet_available)
]) :-
    % Create test facts
    retractall(test_link(_, _)),
    assertz(test_link(a, b)),
    assertz(test_link(b, c)),
    assertz(test_link(c, d)),

    % Compile to C#
    use_module('../../src/unifyweaver/targets/csharp_stream_target'),
    compile_predicate_to_csharp(test_link/2, [], Code),

    % Test using Python helper
    compile_and_test_csharp(Code, "a:b", Result),

    % Verify result
    assertion(Result.success == @(true)),
    assertion(Result.assertion_passed == @(true)).

test(stream_target_join_query, [
    condition(check_dotnet_available)
]) :-
    % Create test facts
    retractall(test_parent(_, _)),
    assertz(test_parent(alice, bob)),
    assertz(test_parent(bob, charlie)),
    assertz(test_parent(alice, diane)),

    % Define grandparent rule
    retractall((test_grandparent(_, _) :- _)),
    assertz((test_grandparent(X, Z) :- test_parent(X, Y), test_parent(Y, Z))),

    % Compile to C#
    use_module('../../src/unifyweaver/targets/csharp_stream_target'),
    compile_predicate_to_csharp(test_grandparent/2, [], Code),

    % Test using Python helper
    compile_and_test_csharp(Code, "alice:charlie", Result),

    % Verify result
    assertion(Result.success == @(true)),
    assertion(Result.assertion_passed == @(true)).

test(stream_target_error_on_recursion, [
    condition(check_dotnet_available)
]) :-
    % Create recursive predicate
    retractall(test_path(_, _)),
    retractall(test_edge(_, _)),
    assertz(test_edge(a, b)),
    assertz(test_edge(b, c)),
    assertz((test_path(X, Y) :- test_edge(X, Y))),
    assertz((test_path(X, Z) :- test_edge(X, Y), test_path(Y, Z))),

    % Attempt to compile with Stream Target (should fail)
    use_module('../../src/unifyweaver/targets/csharp_stream_target'),
    \+ compile_predicate_to_csharp(test_path/2, [target(csharp_stream)], _Code).

test(query_runtime_basic_recursion, [
    condition(check_dotnet_available),
    blocked('Known issue: unsupported constraint operand')
]) :-
    % Create recursive predicate
    retractall(test_path2(_, _)),
    retractall(test_link2(_, _)),
    assertz(test_link2(a, b)),
    assertz(test_link2(b, c)),
    assertz(test_link2(c, d)),
    assertz((test_path2(X, Y) :- test_link2(X, Y))),
    assertz((test_path2(X, Z) :- test_link2(X, Y), test_path2(Y, Z))),

    % Compile with Query Runtime
    use_module('../../src/unifyweaver/targets/csharp_query_target'),
    compile_predicate_to_csharp(test_path2/2, [target(csharp_query)], Code),

    % Test using Python helper
    compile_and_test_csharp(Code, "a:d", Result),

    % Verify result
    assertion(Result.success == @(true)),
    assertion(Result.assertion_passed == @(true)).

test(dotnet_cli_available, []) :-
    check_dotnet_available.

:- end_tests(csharp_janus).

% =====================================================================
% Standalone Test Predicates
% =====================================================================

/**
 * test_csharp_compilation/0
 *
 * Run all Janus-based C# compilation tests
 */
test_csharp_compilation :-
    format('~n=== Running C# Janus Bridge Tests ===~n~n'),
    run_tests(csharp_janus).

/**
 * test_stream_target_simple_facts/0
 *
 * Test simple fact compilation with Stream Target
 */
test_stream_target_simple_facts :-
    format('~n=== Testing Stream Target (Simple Facts) ===~n~n'),
    run_tests([csharp_janus:stream_target_simple_facts]).

/**
 * test_stream_target_join/0
 *
 * Test join query compilation with Stream Target
 */
test_stream_target_join :-
    format('~n=== Testing Stream Target (Join Query) ===~n~n'),
    run_tests([csharp_janus:stream_target_join_query]).

% =====================================================================
% Example Usage
% =====================================================================

/**
 * Example: Manual test using Python helper
 *
 * ?- MinimalCode = "using System; class Program { static void Main() { Console.WriteLine(\"Hello from C#\"); } }",
 *    compile_and_run_csharp(MinimalCode, Result),
 *    format('Success: ~w~nOutput: ~w~n', [Result.success, Result.stdout]).
 */
