/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Integration tests for dotnet_glue module
 */

:- use_module('../../../src/unifyweaver/glue/dotnet_glue').

%% ============================================
%% Test Helpers
%% ============================================

run_tests :-
    format('~n=== .NET Glue Integration Tests ===~n~n'),
    run_all_tests,
    format('~nAll tests passed!~n').

run_all_tests :-
    test_ironpython_compatibility,
    test_powershell_bridge_generation,
    test_ironpython_bridge_generation,
    test_cpython_bridge_generation,
    test_csharp_host_generation,
    test_dotnet_pipeline_generation.

assert_contains(String, Substring, TestName) :-
    (   sub_atom(String, _, _, _, Substring)
    ->  format('  ✓ ~w~n', [TestName])
    ;   format('  ✗ ~w FAILED: "~w" not found~n', [TestName, Substring]),
        fail
    ).

assert_true(Goal, TestName) :-
    (   call(Goal)
    ->  format('  ✓ ~w~n', [TestName])
    ;   format('  ✗ ~w FAILED~n', [TestName]),
        fail
    ).

assert_false(Goal, TestName) :-
    (   \+ call(Goal)
    ->  format('  ✓ ~w~n', [TestName])
    ;   format('  ✗ ~w FAILED (should be false)~n', [TestName]),
        fail
    ).

%% ============================================
%% Test: IronPython Compatibility
%% ============================================

test_ironpython_compatibility :-
    format('Test: IronPython compatibility checking~n'),

    % Core modules should be compatible
    assert_true(ironpython_compatible(sys), 'sys is compatible'),
    assert_true(ironpython_compatible(os), 'os is compatible'),
    assert_true(ironpython_compatible(json), 'json is compatible'),
    assert_true(ironpython_compatible(re), 're is compatible'),
    assert_true(ironpython_compatible(collections), 'collections is compatible'),

    % .NET interop module
    assert_true(ironpython_compatible(clr), 'clr (IronPython special) is compatible'),

    % Data format modules
    assert_true(ironpython_compatible(csv), 'csv is compatible'),
    assert_true(ironpython_compatible(xml), 'xml is compatible'),

    % C extension modules should NOT be compatible
    assert_false(ironpython_compatible(numpy), 'numpy is NOT compatible'),
    assert_false(ironpython_compatible(pandas), 'pandas is NOT compatible'),
    assert_false(ironpython_compatible(tensorflow), 'tensorflow is NOT compatible'),

    % can_use_ironpython with compatible imports
    assert_true(can_use_ironpython([sys, json, re]), 'can use IronPython for [sys, json, re]'),
    assert_true(can_use_ironpython([]), 'can use IronPython for empty imports'),

    % can_use_ironpython with incompatible imports
    assert_false(can_use_ironpython([sys, numpy]), 'cannot use IronPython with numpy'),
    assert_false(can_use_ironpython([pandas, json]), 'cannot use IronPython with pandas'),

    % python_runtime_choice
    python_runtime_choice([sys, json], R1),
    assert_true(R1 == ironpython, 'chooses ironpython for [sys, json]'),

    python_runtime_choice([numpy, sys], R2),
    assert_true(R2 == cpython_pipe, 'chooses cpython_pipe for [numpy, sys]'),

    format('~n').

%% ============================================
%% Test: PowerShell Bridge Generation
%% ============================================

test_powershell_bridge_generation :-
    format('Test: PowerShell bridge generation~n'),

    % Basic generation
    generate_powershell_bridge([], Bridge1),
    assert_contains(Bridge1, 'namespace UnifyWeaver.Glue', 'default namespace'),
    assert_contains(Bridge1, 'public static class PowerShellBridge', 'default class name'),
    assert_contains(Bridge1, 'System.Management.Automation', 'uses PowerShell SDK'),
    assert_contains(Bridge1, 'Runspace', 'uses runspace'),
    assert_contains(Bridge1, 'Invoke<TInput, TOutput>', 'has generic Invoke method'),
    assert_contains(Bridge1, 'InvokeStream<TInput, TOutput>', 'has streaming method'),
    assert_contains(Bridge1, 'InvokeCommand<TOutput>', 'has command invocation'),

    % Custom options
    generate_powershell_bridge([namespace('MyApp.Glue'), class('PSBridge')], Bridge2),
    assert_contains(Bridge2, 'namespace MyApp.Glue', 'custom namespace'),
    assert_contains(Bridge2, 'public static class PSBridge', 'custom class name'),

    % Variable access
    assert_contains(Bridge1, 'SetVariable', 'has SetVariable method'),
    assert_contains(Bridge1, 'GetVariable<T>', 'has GetVariable method'),

    format('~n').

%% ============================================
%% Test: IronPython Bridge Generation
%% ============================================

test_ironpython_bridge_generation :-
    format('Test: IronPython bridge generation~n'),

    % Basic generation
    generate_ironpython_bridge([], Bridge1),
    assert_contains(Bridge1, 'namespace UnifyWeaver.Glue', 'default namespace'),
    assert_contains(Bridge1, 'public static class IronPythonBridge', 'default class name'),
    assert_contains(Bridge1, 'IronPython.Hosting', 'uses IronPython hosting'),
    assert_contains(Bridge1, 'ScriptEngine', 'uses script engine'),
    assert_contains(Bridge1, 'Python.CreateEngine()', 'creates Python engine'),

    % Methods
    assert_contains(Bridge1, 'Execute(string pythonCode)', 'has Execute method'),
    assert_contains(Bridge1, 'ExecuteWithInput<TInput>', 'has ExecuteWithInput'),
    assert_contains(Bridge1, 'ExecuteStream<TInput>', 'has streaming method'),
    assert_contains(Bridge1, 'CallFunction', 'has CallFunction'),
    assert_contains(Bridge1, 'DefineFunction', 'has DefineFunction'),
    assert_contains(Bridge1, 'Import(string moduleName)', 'has Import method'),

    % Dictionary conversion
    assert_contains(Bridge1, 'ToPythonDict', 'has ToPythonDict'),
    assert_contains(Bridge1, 'FromPythonDict', 'has FromPythonDict'),

    % Custom options
    generate_ironpython_bridge([namespace('ML.Bridges'), class('PyBridge')], Bridge2),
    assert_contains(Bridge2, 'namespace ML.Bridges', 'custom namespace'),
    assert_contains(Bridge2, 'public static class PyBridge', 'custom class name'),

    format('~n').

%% ============================================
%% Test: CPython Bridge Generation
%% ============================================

test_cpython_bridge_generation :-
    format('Test: CPython bridge generation~n'),

    % Basic generation
    generate_cpython_bridge([], Bridge1),
    assert_contains(Bridge1, 'namespace UnifyWeaver.Glue', 'default namespace'),
    assert_contains(Bridge1, 'public static class CPythonBridge', 'default class name'),
    assert_contains(Bridge1, 'Process', 'uses System.Diagnostics.Process'),
    assert_contains(Bridge1, 'ProcessStartInfo', 'uses ProcessStartInfo'),
    assert_contains(Bridge1, 'python3', 'default python path'),

    % Pipe communication
    assert_contains(Bridge1, 'RedirectStandardInput', 'redirects stdin'),
    assert_contains(Bridge1, 'RedirectStandardOutput', 'redirects stdout'),
    assert_contains(Bridge1, 'RedirectStandardError', 'redirects stderr'),

    % JSON serialization (default format)
    assert_contains(Bridge1, 'JsonSerializer', 'uses JSON serialization'),
    assert_contains(Bridge1, 'json.loads', 'Python reads JSON'),
    assert_contains(Bridge1, 'json.dumps', 'Python writes JSON'),

    % Streaming
    assert_contains(Bridge1, 'ExecuteStream<TInput, TOutput>', 'has streaming method'),

    % Custom Python path
    generate_cpython_bridge([python_path('/usr/bin/python3.11')], Bridge2),
    assert_contains(Bridge2, '/usr/bin/python3.11', 'custom python path'),

    format('~n').

%% ============================================
%% Test: C# Host Generation
%% ============================================

test_csharp_host_generation :-
    format('Test: C# host generation~n'),

    % Generate host with all bridges
    generate_csharp_host(
        [
            bridge(powershell, []),
            bridge(ironpython, []),
            bridge(cpython, [])
        ],
        [namespace('MyApp.Generated'), class('MultiTargetHost')],
        Host
    ),

    assert_contains(Host, 'namespace MyApp.Generated', 'custom namespace'),
    assert_contains(Host, 'public class MultiTargetHost', 'custom class name'),
    assert_contains(Host, 'Execute<TInput, TOutput>', 'has generic Execute'),

    % Target dispatch
    assert_contains(Host, '"powershell"', 'handles powershell target'),
    assert_contains(Host, '"ironpython"', 'handles ironpython target'),
    assert_contains(Host, '"cpython"', 'handles cpython target'),

    % Includes all bridges
    assert_contains(Host, 'PowerShellBridge', 'includes PowerShell bridge'),
    assert_contains(Host, 'IronPythonBridge', 'includes IronPython bridge'),
    assert_contains(Host, 'CPythonBridge', 'includes CPython bridge'),

    format('~n').

%% ============================================
%% Test: .NET Pipeline Generation
%% ============================================

test_dotnet_pipeline_generation :-
    format('Test: .NET pipeline generation~n'),

    % Generate pipeline with multiple steps
    generate_dotnet_pipeline(
        [
            step(filter, powershell, '$record | Where-Object { $_.status -eq "active" }', []),
            step(transform, ironpython, 'result = {"id": record["id"], "score": record["score"] * 100}', []),
            step(analyze, cpython, 'import numpy as np; result = {"mean": np.mean(record["values"])}', [])
        ],
        [namespace('DataPipeline'), class('ThreeStepPipeline')],
        Pipeline
    ),

    assert_contains(Pipeline, 'namespace DataPipeline', 'custom namespace'),
    assert_contains(Pipeline, 'public class ThreeStepPipeline', 'custom class name'),

    % Step methods
    assert_contains(Pipeline, 'Step 1: filter (PowerShell)', 'step 1 comment'),
    assert_contains(Pipeline, 'Step 2: transform (IronPython)', 'step 2 comment'),
    assert_contains(Pipeline, 'Step 3: analyze (CPython via pipes)', 'step 3 comment'),

    % Step method bodies
    assert_contains(Pipeline, 'PowerShellBridge.InvokeStream', 'step 1 uses PowerShell'),
    assert_contains(Pipeline, 'IronPythonBridge.ExecuteStream', 'step 2 uses IronPython'),
    assert_contains(Pipeline, 'CPythonBridge.ExecuteStream', 'step 3 uses CPython'),

    % Execute method chains steps
    assert_contains(Pipeline, 'current = Step1(current)', 'executes step 1'),
    assert_contains(Pipeline, 'current = Step2(current)', 'executes step 2'),
    assert_contains(Pipeline, 'current = Step3(current)', 'executes step 3'),

    format('~n').

%% ============================================
%% Main
%% ============================================

:- initialization(run_tests, main).
