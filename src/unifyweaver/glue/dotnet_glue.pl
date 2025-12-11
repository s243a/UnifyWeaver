/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * .NET Glue - Cross-target communication for .NET ecosystem
 *
 * This module generates glue code for:
 * - C# ↔ PowerShell in-process communication
 * - C# ↔ IronPython in-process communication
 * - CPython fallback via pipes when IronPython is incompatible
 */

:- module(dotnet_glue, [
    % Runtime detection
    detect_dotnet_runtime/1,        % detect_dotnet_runtime(-Runtime)
    detect_ironpython/1,            % detect_ironpython(-Available)
    detect_powershell/1,            % detect_powershell(-Version)

    % Compatibility checking
    ironpython_compatible/1,        % ironpython_compatible(+Module)
    can_use_ironpython/1,           % can_use_ironpython(+Imports)
    python_runtime_choice/2,        % python_runtime_choice(+Imports, -Runtime)

    % Bridge generation
    generate_powershell_bridge/2,   % generate_powershell_bridge(+Options, -Code)
    generate_ironpython_bridge/2,   % generate_ironpython_bridge(+Options, -Code)
    generate_cpython_bridge/2,      % generate_cpython_bridge(+Options, -Code)

    % Script wrappers
    generate_csharp_host/3,         % generate_csharp_host(+Bridges, +Options, -Code)
    generate_dotnet_pipeline/3,     % generate_dotnet_pipeline(+Steps, +Options, -Code)

    % Python Pipeline Hosting (Phase 3+ Integration)
    generate_python_predicate_wrapper/4,  % generate_python_predicate_wrapper(+Name, +Arity, +Options, -CSharpCode)
    generate_pipeline_host/3,             % generate_pipeline_host(+Predicates, +Options, -Code)
    compile_python_for_csharp/4,          % compile_python_for_csharp(+Predicate, +Options, -PythonCode, -CSharpCode)
    test_python_csharp_glue/0             % Run tests
]).

:- use_module(library(lists)).

%% ============================================
%% Runtime Detection
%% ============================================

%% detect_dotnet_runtime(-Runtime)
%  Detect available .NET runtime.
%  Runtime = dotnet_core | dotnet_framework | mono | none
%
detect_dotnet_runtime(Runtime) :-
    % Try .NET Core/5+/6+ first (modern)
    (   catch(
            (process_create(path(dotnet), ['--version'], [stdout(pipe(S))]),
             read_line_to_string(S, Version),
             close(S)),
            _, fail)
    ->  (   sub_string(Version, 0, 1, _, Major),
            atom_number(Major, N),
            N >= 5
        ->  Runtime = dotnet_modern  % .NET 5+
        ;   Runtime = dotnet_core    % .NET Core 3.x
        )
    % Try Mono
    ;   catch(
            (process_create(path(mono), ['--version'], [stdout(pipe(S2))]),
             read_line_to_string(S2, _),
             close(S2)),
            _, fail)
    ->  Runtime = mono
    % No .NET runtime found
    ;   Runtime = none
    ).

%% detect_ironpython(-Available)
%  Check if IronPython is available.
%
detect_ironpython(Available) :-
    (   catch(
            (process_create(path(ipy), ['--version'], [stdout(pipe(S))]),
             read_line_to_string(S, _),
             close(S)),
            _, fail)
    ->  Available = true
    ;   catch(
            (process_create(path(ipy64), ['--version'], [stdout(pipe(S2))]),
             read_line_to_string(S2, _),
             close(S2)),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

%% detect_powershell(-Version)
%  Detect PowerShell version (Core or Windows).
%
detect_powershell(Version) :-
    % Try PowerShell Core first (pwsh)
    (   catch(
            (process_create(path(pwsh), ['--version'], [stdout(pipe(S))]),
             read_line_to_string(S, VersionStr),
             close(S)),
            _, fail)
    ->  Version = core(VersionStr)
    % Try Windows PowerShell
    ;   catch(
            (process_create(path(powershell), ['-Command', '$PSVersionTable.PSVersion.ToString()'],
                          [stdout(pipe(S2))]),
             read_line_to_string(S2, VersionStr2),
             close(S2)),
            _, fail)
    ->  Version = windows(VersionStr2)
    ;   Version = none
    ).

%% ============================================
%% IronPython Compatibility
%% ============================================

%% ironpython_compatible(+Module)
%  True if the Python module is compatible with IronPython.
%
%  IronPython supports:
%  - Standard library (most of it)
%  - .NET interop (clr module)
%
%  IronPython does NOT support:
%  - C extension modules (numpy, pandas, etc.)
%  - Some newer Python 3 features
%

% Core modules - fully compatible
ironpython_compatible(sys).
ironpython_compatible(os).
ironpython_compatible(io).
ironpython_compatible(json).
ironpython_compatible(re).
ironpython_compatible(math).
ironpython_compatible(random).
ironpython_compatible(collections).
ironpython_compatible(itertools).
ironpython_compatible(functools).
ironpython_compatible(string).
ironpython_compatible(datetime).
ironpython_compatible(time).
ironpython_compatible(copy).
ironpython_compatible(types).
ironpython_compatible(operator).
ironpython_compatible(abc).

% File/path modules
ironpython_compatible(pathlib).
ironpython_compatible(glob).
ironpython_compatible(fnmatch).
ironpython_compatible(shutil).
ironpython_compatible(tempfile).

% Data format modules
ironpython_compatible(csv).
ironpython_compatible(xml).
ironpython_compatible('xml.etree').
ironpython_compatible('xml.etree.ElementTree').
ironpython_compatible(configparser).

% Text processing
ironpython_compatible(textwrap).
ironpython_compatible(difflib).

% .NET interop - IronPython special
ironpython_compatible(clr).

% Network (basic support)
ironpython_compatible(socket).
ironpython_compatible(http).
ironpython_compatible('http.client').
ironpython_compatible(urllib).
ironpython_compatible('urllib.parse').

% Compression
ironpython_compatible(gzip).
ironpython_compatible(zipfile).

% Hashing
ironpython_compatible(hashlib).

%% can_use_ironpython(+Imports)
%  True if all imports in the list are IronPython compatible.
%
can_use_ironpython([]).
can_use_ironpython([Import|Rest]) :-
    ironpython_compatible(Import),
    can_use_ironpython(Rest).

%% python_runtime_choice(+Imports, -Runtime)
%  Choose the appropriate Python runtime based on imports.
%  Runtime = ironpython | cpython_pipe
%
python_runtime_choice(Imports, ironpython) :-
    can_use_ironpython(Imports),
    !.
python_runtime_choice(_, cpython_pipe).

%% ============================================
%% PowerShell Bridge Generation
%% ============================================

%% generate_powershell_bridge(+Options, -Code)
%  Generate C# code to host and invoke PowerShell scripts.
%
%  Options:
%    - namespace(Name) : C# namespace (default: UnifyWeaver.Glue)
%    - class(Name) : Class name (default: PowerShellBridge)
%    - async(Bool) : Generate async version (default: false)
%
generate_powershell_bridge(Options, Code) :-
    option_or_default(namespace(Namespace), Options, 'UnifyWeaver.Glue'),
    option_or_default(class(ClassName), Options, 'PowerShellBridge'),
    option_or_default(async(Async), Options, false),

    (Async == true -> AsyncCode = async_powershell_methods ; AsyncCode = ''),

    format(atom(Code), '
// Generated PowerShell Bridge for C# integration
// Uses System.Management.Automation for in-process PowerShell hosting

using System;
using System.Collections.Generic;
using System.Management.Automation;
using System.Management.Automation.Runspaces;

namespace ~w
{
    /// <summary>
    /// Bridge for invoking PowerShell scripts from C# in-process.
    /// </summary>
    public static class ~w
    {
        private static readonly Runspace SharedRunspace;

        static ~w()
        {
            SharedRunspace = RunspaceFactory.CreateRunspace();
            SharedRunspace.Open();
        }

        /// <summary>
        /// Invoke a PowerShell script with input data.
        /// </summary>
        /// <typeparam name="TInput">Input type</typeparam>
        /// <typeparam name="TOutput">Expected output type</typeparam>
        /// <param name="script">PowerShell script text</param>
        /// <param name="input">Input object (accessible as $Input or via pipeline)</param>
        /// <returns>Collection of output objects</returns>
        public static IEnumerable<TOutput> Invoke<TInput, TOutput>(string script, TInput input)
        {
            using var ps = PowerShell.Create();
            ps.Runspace = SharedRunspace;
            ps.AddScript(script);
            ps.AddParameter("InputObject", input);

            var results = ps.Invoke<TOutput>();

            if (ps.HadErrors)
            {
                var errors = string.Join("; ", ps.Streams.Error);
                throw new InvalidOperationException($"PowerShell error: {errors}");
            }

            return results;
        }

        /// <summary>
        /// Invoke a PowerShell script that processes a stream of records.
        /// </summary>
        public static IEnumerable<TOutput> InvokeStream<TInput, TOutput>(
            string script,
            IEnumerable<TInput> inputStream)
        {
            using var ps = PowerShell.Create();
            ps.Runspace = SharedRunspace;
            ps.AddScript(script);

            foreach (var item in inputStream)
            {
                ps.Commands.Clear();
                ps.AddScript(script);
                ps.AddParameter("InputObject", item);

                foreach (var result in ps.Invoke<TOutput>())
                {
                    yield return result;
                }
            }
        }

        /// <summary>
        /// Invoke a PowerShell command by name with parameters.
        /// </summary>
        public static IEnumerable<TOutput> InvokeCommand<TOutput>(
            string commandName,
            IDictionary<string, object> parameters)
        {
            using var ps = PowerShell.Create();
            ps.Runspace = SharedRunspace;
            ps.AddCommand(commandName);

            foreach (var param in parameters)
            {
                ps.AddParameter(param.Key, param.Value);
            }

            return ps.Invoke<TOutput>();
        }

        /// <summary>
        /// Set a variable in the PowerShell runspace.
        /// </summary>
        public static void SetVariable(string name, object value)
        {
            SharedRunspace.SessionStateProxy.SetVariable(name, value);
        }

        /// <summary>
        /// Get a variable from the PowerShell runspace.
        /// </summary>
        public static T GetVariable<T>(string name)
        {
            return (T)SharedRunspace.SessionStateProxy.GetVariable(name);
        }
~w
    }
}
', [Namespace, ClassName, ClassName, AsyncCode]).

async_powershell_methods('
        /// <summary>
        /// Async version of Invoke.
        /// </summary>
        public static async Task<IEnumerable<TOutput>> InvokeAsync<TInput, TOutput>(
            string script,
            TInput input,
            CancellationToken cancellationToken = default)
        {
            return await Task.Run(() => Invoke<TInput, TOutput>(script, input), cancellationToken);
        }
').

%% ============================================
%% IronPython Bridge Generation
%% ============================================

%% generate_ironpython_bridge(+Options, -Code)
%  Generate C# code to host and invoke IronPython scripts.
%
generate_ironpython_bridge(Options, Code) :-
    option_or_default(namespace(Namespace), Options, 'UnifyWeaver.Glue'),
    option_or_default(class(ClassName), Options, 'IronPythonBridge'),

    format(atom(Code), '
// Generated IronPython Bridge for C# integration
// Uses IronPython.Hosting for in-process Python execution

using System;
using System.Collections.Generic;
using System.Dynamic;
using IronPython.Hosting;
using Microsoft.Scripting.Hosting;

namespace ~w
{
    /// <summary>
    /// Bridge for invoking Python scripts from C# using IronPython (in-process).
    /// </summary>
    public static class ~w
    {
        private static readonly ScriptEngine Engine;
        private static readonly ScriptScope SharedScope;

        static ~w()
        {
            Engine = Python.CreateEngine();
            SharedScope = Engine.CreateScope();

            // Add common imports
            Engine.Execute("import sys, os, json, re, collections", SharedScope);
        }

        /// <summary>
        /// Execute a Python expression and return the result.
        /// </summary>
        public static dynamic Execute(string pythonCode)
        {
            return Engine.Execute(pythonCode, SharedScope);
        }

        /// <summary>
        /// Execute a Python script with input data.
        /// </summary>
        /// <typeparam name="TInput">Input type</typeparam>
        /// <param name="script">Python script text</param>
        /// <param name="input">Input data (accessible as `input` variable)</param>
        /// <returns>Dynamic result from Python</returns>
        public static dynamic ExecuteWithInput<TInput>(string script, TInput input)
        {
            var scope = Engine.CreateScope();
            scope.SetVariable("input", input);
            return Engine.Execute(script, scope);
        }

        /// <summary>
        /// Execute a Python script that processes a stream of records.
        /// </summary>
        public static IEnumerable<dynamic> ExecuteStream<TInput>(
            string script,
            IEnumerable<TInput> inputStream)
        {
            // Compile the script once
            var source = Engine.CreateScriptSourceFromString(script);
            var compiled = source.Compile();

            foreach (var item in inputStream)
            {
                var scope = Engine.CreateScope();
                scope.SetVariable("record", item);
                compiled.Execute(scope);

                // Get result variable
                if (scope.TryGetVariable("result", out dynamic result))
                {
                    yield return result;
                }
            }
        }

        /// <summary>
        /// Call a Python function by name.
        /// </summary>
        public static dynamic CallFunction(string functionName, params object[] args)
        {
            var func = SharedScope.GetVariable<Func<object[], object>>(functionName);
            return func(args);
        }

        /// <summary>
        /// Define a Python function for later use.
        /// </summary>
        public static void DefineFunction(string functionCode)
        {
            Engine.Execute(functionCode, SharedScope);
        }

        /// <summary>
        /// Import a Python module into the shared scope.
        /// </summary>
        public static void Import(string moduleName)
        {
            Engine.Execute($"import {moduleName}", SharedScope);
        }

        /// <summary>
        /// Set a variable in the shared Python scope.
        /// </summary>
        public static void SetVariable(string name, object value)
        {
            SharedScope.SetVariable(name, value);
        }

        /// <summary>
        /// Get a variable from the shared Python scope.
        /// </summary>
        public static T GetVariable<T>(string name)
        {
            return SharedScope.GetVariable<T>(name);
        }

        /// <summary>
        /// Convert a C# dictionary to a Python dict.
        /// </summary>
        public static dynamic ToPythonDict<TKey, TValue>(IDictionary<TKey, TValue> dict)
        {
            var pyDict = Engine.Execute("dict()", SharedScope);
            foreach (var kvp in dict)
            {
                pyDict[kvp.Key] = kvp.Value;
            }
            return pyDict;
        }

        /// <summary>
        /// Convert a Python dict to a C# dictionary.
        /// </summary>
        public static Dictionary<string, object> FromPythonDict(dynamic pyDict)
        {
            var result = new Dictionary<string, object>();
            foreach (var key in pyDict.keys())
            {
                result[key.ToString()] = pyDict[key];
            }
            return result;
        }
    }
}
', [Namespace, ClassName, ClassName]).

%% ============================================
%% CPython Bridge Generation (Pipe-based fallback)
%% ============================================

%% generate_cpython_bridge(+Options, -Code)
%  Generate C# code to communicate with CPython via pipes.
%  Used when IronPython is not compatible with required modules.
%
generate_cpython_bridge(Options, Code) :-
    option_or_default(namespace(Namespace), Options, 'UnifyWeaver.Glue'),
    option_or_default(class(ClassName), Options, 'CPythonBridge'),
    option_or_default(python_path(PythonPath), Options, 'python3'),
    option_or_default(format(Format), Options, json),

    format_serialization(Format, SerializeCode, DeserializeCode),

    format(atom(Code), '
// Generated CPython Bridge for C# integration
// Uses process pipes for communication (fallback when IronPython incompatible)

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;

namespace ~w
{
    /// <summary>
    /// Bridge for invoking Python scripts via subprocess pipes.
    /// Used when IronPython is not compatible with required modules (numpy, pandas, etc.)
    /// </summary>
    public static class ~w
    {
        private static readonly string PythonPath = "~w";

        /// <summary>
        /// Execute a Python script with input, communicating via JSON.
        /// </summary>
        public static TOutput Execute<TInput, TOutput>(string script, TInput input)
        {
            // Wrap script to read from stdin and write to stdout
            var wrappedScript = $@"
import sys
import json

input_data = json.loads(sys.stdin.read())

{script}

print(json.dumps(result))
";

            var psi = new ProcessStartInfo
            {
                FileName = PythonPath,
                Arguments = "-c \\"" + wrappedScript.Replace("\\"", "\\\\\\"") + "\\"",
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi);

            // Send input
            ~w
            process.StandardInput.Close();

            // Read output
            var output = process.StandardOutput.ReadToEnd();
            var errors = process.StandardError.ReadToEnd();

            process.WaitForExit();

            if (process.ExitCode != 0)
            {
                throw new InvalidOperationException($"Python error (exit {process.ExitCode}): {errors}");
            }

            ~w
        }

        /// <summary>
        /// Execute a Python script that processes a stream of records.
        /// </summary>
        public static IEnumerable<TOutput> ExecuteStream<TInput, TOutput>(
            string script,
            IEnumerable<TInput> inputStream)
        {
            // Script wrapper for streaming
            var wrappedScript = $@"
import sys
import json

for line in sys.stdin:
    record = json.loads(line)
    {script}
    print(json.dumps(result))
    sys.stdout.flush()
";

            var psi = new ProcessStartInfo
            {
                FileName = PythonPath,
                Arguments = $"-c \\"{wrappedScript.Replace("\\"", "\\\\\\"")}\\"",
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi);

            // Write input in background
            var inputTask = System.Threading.Tasks.Task.Run(() =>
            {
                foreach (var item in inputStream)
                {
                    var json = JsonSerializer.Serialize(item);
                    process.StandardInput.WriteLine(json);
                }
                process.StandardInput.Close();
            });

            // Read output
            string line;
            while ((line = process.StandardOutput.ReadLine()) != null)
            {
                yield return JsonSerializer.Deserialize<TOutput>(line);
            }

            inputTask.Wait();
            process.WaitForExit();
        }

        /// <summary>
        /// Run a Python script file with arguments.
        /// </summary>
        public static string RunScript(string scriptPath, params string[] args)
        {
            var psi = new ProcessStartInfo
            {
                FileName = PythonPath,
                Arguments = $"\\"{scriptPath}\\" " + string.Join(" ", args),
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi);
            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();

            return output;
        }
    }
}
', [Namespace, ClassName, PythonPath, SerializeCode, DeserializeCode]).

format_serialization(json, SerializeCode, DeserializeCode) :-
    SerializeCode = 'var inputJson = JsonSerializer.Serialize(input);
            process.StandardInput.Write(inputJson);',
    DeserializeCode = 'return JsonSerializer.Deserialize<TOutput>(output);'.

format_serialization(tsv, SerializeCode, DeserializeCode) :-
    SerializeCode = '// TSV: Convert input to tab-separated
            process.StandardInput.Write(input.ToString());',
    DeserializeCode = '// TSV: Parse output
            return (TOutput)(object)output.Trim();'.

%% ============================================
%% C# Host Generation
%% ============================================

%% generate_csharp_host(+Bridges, +Options, -Code)
%  Generate a complete C# program that hosts multiple target bridges.
%
%  Bridges: List of bridge(Type, Options) where Type = powershell | ironpython | cpython
%
generate_csharp_host(Bridges, Options, Code) :-
    option_or_default(namespace(Namespace), Options, 'UnifyWeaver.Generated'),
    option_or_default(class(ClassName), Options, 'TargetHost'),

    maplist(generate_bridge_include, Bridges, BridgeIncludes),
    atomic_list_concat(BridgeIncludes, '\n', BridgeCode),

    format(atom(Code), '
// Generated UnifyWeaver Target Host
// Combines multiple language bridges for cross-target communication

using System;
using System.Collections.Generic;

namespace ~w
{
~w

    /// <summary>
    /// Main host class coordinating cross-target calls.
    /// </summary>
    public class ~w
    {
        /// <summary>
        /// Execute a predicate in the appropriate target runtime.
        /// </summary>
        public TOutput Execute<TInput, TOutput>(string targetName, string script, TInput input)
        {
            return targetName switch
            {
                "powershell" => ExecutePowerShell<TInput, TOutput>(script, input),
                "ironpython" => ExecuteIronPython<TInput, TOutput>(script, input),
                "cpython" => ExecuteCPython<TInput, TOutput>(script, input),
                _ => throw new ArgumentException($"Unknown target: {targetName}")
            };
        }

        private TOutput ExecutePowerShell<TInput, TOutput>(string script, TInput input)
        {
            var results = PowerShellBridge.Invoke<TInput, TOutput>(script, input);
            using var enumerator = results.GetEnumerator();
            return enumerator.MoveNext() ? enumerator.Current : default;
        }

        private TOutput ExecuteIronPython<TInput, TOutput>(string script, TInput input)
        {
            return (TOutput)IronPythonBridge.ExecuteWithInput(script, input);
        }

        private TOutput ExecuteCPython<TInput, TOutput>(string script, TInput input)
        {
            return CPythonBridge.Execute<TInput, TOutput>(script, input);
        }
    }
}
', [Namespace, BridgeCode, ClassName]).

generate_bridge_include(bridge(powershell, Opts), Code) :-
    generate_powershell_bridge(Opts, Code).
generate_bridge_include(bridge(ironpython, Opts), Code) :-
    generate_ironpython_bridge(Opts, Code).
generate_bridge_include(bridge(cpython, Opts), Code) :-
    generate_cpython_bridge(Opts, Code).

%% ============================================
%% .NET Pipeline Generation
%% ============================================

%% generate_dotnet_pipeline(+Steps, +Options, -Code)
%  Generate C# code for a multi-step pipeline within .NET ecosystem.
%
%  Steps: List of step(Name, Target, Script, StepOptions)
%         Target = csharp | powershell | ironpython | cpython
%
generate_dotnet_pipeline(Steps, Options, Code) :-
    option_or_default(namespace(Namespace), Options, 'UnifyWeaver.Generated'),
    option_or_default(class(ClassName), Options, 'Pipeline'),

    generate_step_methods(Steps, 1, StepMethods),
    generate_pipeline_execute(Steps, ExecuteCode),

    format(atom(Code), '
// Generated UnifyWeaver .NET Pipeline
// Multi-step data processing across C#, PowerShell, and Python

using System;
using System.Collections.Generic;
using System.Linq;

namespace ~w
{
    public class ~w
    {
~w

        /// <summary>
        /// Execute the full pipeline.
        /// </summary>
        public IEnumerable<dynamic> Execute(IEnumerable<dynamic> input)
        {
            var current = input;
~w
            return current;
        }
    }
}
', [Namespace, ClassName, StepMethods, ExecuteCode]).

generate_step_methods([], _, '').
generate_step_methods([step(Name, Target, Script, _Opts)|Rest], N, Code) :-
    generate_step_method(Name, Target, Script, N, MethodCode),
    N1 is N + 1,
    generate_step_methods(Rest, N1, RestCode),
    atom_concat(MethodCode, RestCode, Code).

generate_step_method(Name, powershell, Script, N, Code) :-
    format(atom(Code), '
        // Step ~w: ~w (PowerShell)
        private IEnumerable<dynamic> Step~w(IEnumerable<dynamic> input)
        {
            var script = @"~w";
            return PowerShellBridge.InvokeStream<dynamic, dynamic>(script, input);
        }
', [N, Name, N, Script]).

generate_step_method(Name, ironpython, Script, N, Code) :-
    format(atom(Code), '
        // Step ~w: ~w (IronPython)
        private IEnumerable<dynamic> Step~w(IEnumerable<dynamic> input)
        {
            var script = @"~w";
            return IronPythonBridge.ExecuteStream<dynamic>(script, input);
        }
', [N, Name, N, Script]).

generate_step_method(Name, cpython, Script, N, Code) :-
    format(atom(Code), '
        // Step ~w: ~w (CPython via pipes)
        private IEnumerable<dynamic> Step~w(IEnumerable<dynamic> input)
        {
            var script = @"~w";
            return CPythonBridge.ExecuteStream<dynamic, dynamic>(script, input);
        }
', [N, Name, N, Script]).

generate_step_method(Name, csharp, _Script, N, Code) :-
    format(atom(Code), '
        // Step ~w: ~w (C# - implement directly)
        private IEnumerable<dynamic> Step~w(IEnumerable<dynamic> input)
        {
            // TODO: Implement C# logic directly
            return input;
        }
', [N, Name, N]).

generate_pipeline_execute(Steps, Code) :-
    length(Steps, N),
    numlist(1, N, StepNums),
    maplist(step_call, StepNums, Calls),
    atomic_list_concat(Calls, '\n', Code).

step_call(N, Call) :-
    format(atom(Call), '            current = Step~w(current);', [N]).

%% ============================================
%% Utility Predicates
%% ============================================

option_or_default(Option, Options, _Default) :-
    member(Option, Options),
    !.
option_or_default(Option, _Options, Default) :-
    Option =.. [_, Default].

%% ============================================
%% Python Pipeline Hosting (Phase 3+ Integration)
%% ============================================
%%
%% These predicates integrate the Python pipeline mode (Phases 1-3) with
%% C# hosting, enabling compiled Python predicates to be called from C#
%% via IronPython (in-process) or CPython (subprocess).
%%

%% compile_python_for_csharp(+Predicate, +Options, -PythonCode, -CSharpCode)
%  Compile a Prolog predicate to Python and generate C# wrapper.
%
%  This is the main entry point for cross-target Python/C# glue.
%
%  Options:
%    - arg_names([...])     : Property names for pipeline output
%    - namespace(Name)      : C# namespace (default: UnifyWeaver.Generated)
%    - class_name(Name)     : C# class name (default: derived from predicate)
%    - runtime(Runtime)     : Force runtime (ironpython/cpython/auto)
%    - embed_python(Bool)   : Embed Python code in C# as string (default: true)
%    - python_file(Path)    : Write Python to file instead of embedding
%
compile_python_for_csharp(Module:Name/Arity, Options, PythonCode, CSharpCode) :-
    !,
    compile_python_for_csharp_impl(Module, Name, Arity, Options, PythonCode, CSharpCode).
compile_python_for_csharp(Name/Arity, Options, PythonCode, CSharpCode) :-
    compile_python_for_csharp_impl(user, Name, Arity, Options, PythonCode, CSharpCode).

compile_python_for_csharp_impl(Module, Name, Arity, Options, PythonCode, CSharpCode) :-
    % Determine runtime based on imports and context
    option_or_default(runtime(RequestedRuntime), Options, auto),
    determine_runtime(Module, Name, Arity, RequestedRuntime, Runtime),

    % Compile to Python with pipeline mode
    PythonOptions = [
        pipeline_input(true),
        output_format(object),
        runtime(Runtime)
        | Options
    ],

    % Import python_target if available
    (   catch(use_module('../targets/python_target'), _, fail)
    ->  python_target:compile_predicate_to_python(Module:Name/Arity, PythonOptions, PythonCode)
    ;   throw(error(module_not_found(python_target), compile_python_for_csharp/4))
    ),

    % Generate C# wrapper
    generate_python_predicate_wrapper(Name, Arity, [runtime(Runtime)|Options], CSharpCode).

%% determine_runtime(+Module, +Name, +Arity, +Requested, -Runtime)
%  Determine the actual runtime to use.
%
determine_runtime(_Module, _Name, _Arity, Runtime, Runtime) :-
    Runtime \= auto,
    !.
determine_runtime(Module, Name, Arity, auto, Runtime) :-
    % Try to detect imports from predicate
    (   catch(detect_predicate_imports(Module, Name, Arity, Imports), _, Imports = [])
    ->  true
    ;   Imports = []
    ),
    % Check IronPython compatibility
    (   can_use_ironpython(Imports)
    ->  Runtime = ironpython
    ;   Runtime = cpython
    ).

%% detect_predicate_imports(+Module, +Name, +Arity, -Imports)
%  Detect Python imports used by predicate (placeholder for future enhancement).
%
detect_predicate_imports(_Module, _Name, _Arity, []).

%% generate_python_predicate_wrapper(+Name, +Arity, +Options, -CSharpCode)
%  Generate C# class that wraps a Python predicate for in-process calling.
%
%  The generated class:
%  - Hosts IronPython (or manages CPython subprocess)
%  - Loads the compiled Python code
%  - Exposes typed methods matching the predicate signature
%  - Handles data marshaling between .NET and Python
%
generate_python_predicate_wrapper(Name, Arity, Options, CSharpCode) :-
    option_or_default(namespace(Namespace), Options, 'UnifyWeaver.Generated'),
    option_or_default(class_name(ClassName), Options, ''),
    option_or_default(runtime(Runtime), Options, ironpython),
    option_or_default(arg_names(ArgNames), Options, []),
    option_or_default(embed_python(EmbedPython), Options, true),
    option_or_default(python_file(PythonFile), Options, ''),

    % Generate class name from predicate if not specified
    (   ClassName == ''
    ->  atom_string(Name, NameStr),
        string_to_pascal_case(NameStr, PascalName),
        format(string(ActualClassName), "~sPredicate", [PascalName])
    ;   ActualClassName = ClassName
    ),

    % Generate arg names if not provided
    (   ArgNames == []
    ->  generate_arg_names(Arity, GeneratedArgNames)
    ;   GeneratedArgNames = ArgNames
    ),

    % Generate based on runtime
    (   Runtime == ironpython
    ->  generate_ironpython_wrapper(Name, Arity, GeneratedArgNames, Namespace, ActualClassName, EmbedPython, PythonFile, CSharpCode)
    ;   generate_cpython_wrapper(Name, Arity, GeneratedArgNames, Namespace, ActualClassName, EmbedPython, PythonFile, CSharpCode)
    ).

%% generate_ironpython_wrapper/8
%  Generate C# wrapper using IronPython for in-process execution.
%
generate_ironpython_wrapper(Name, Arity, ArgNames, Namespace, ClassName, EmbedPython, PythonFile, Code) :-
    atom_string(Name, NameStr),

    % Generate method parameters
    generate_csharp_parameters(ArgNames, Arity, MethodParams),

    % Generate Python load code
    (   EmbedPython == true
    ->  PythonLoadCode = '_engine.Execute(PythonCode, _scope);'
    ;   format(string(PythonLoadCode), '_engine.ExecuteFile("~w", _scope);', [PythonFile])
    ),

    % Generate property class for output
    generate_result_class(ArgNames, ResultClassCode),

    % Generate invoke arguments
    generate_invoke_args(ArgNames, Arity, InvokeArgs),

    format(string(Code),
'// Generated C# wrapper for Python predicate: ~w/~w
// Runtime: IronPython (in-process .NET integration)

using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using IronPython.Hosting;
using Microsoft.Scripting.Hosting;

namespace ~w
{
~w

    /// <summary>
    /// C# wrapper for the ~w Python predicate.
    /// Hosts IronPython engine for in-process execution.
    /// </summary>
    public class ~w : IDisposable
    {
        private readonly ScriptEngine _engine;
        private readonly ScriptScope _scope;
        private readonly dynamic _predicate;
        private bool _disposed;

        // Embedded Python code (set via SetPythonCode or constructor)
        private static string PythonCode = "";

        /// <summary>
        /// Initialize the IronPython engine and load the predicate.
        /// </summary>
        public ~w()
        {
            _engine = Python.CreateEngine();
            _scope = _engine.CreateScope();

            // Add common imports
            _engine.Execute("import sys, json, clr", _scope);
            _engine.Execute("clr.AddReference(\'System\')", _scope);
            _engine.Execute("from System.Collections.Generic import Dictionary, List", _scope);

            // Load the predicate code
            if (!string.IsNullOrEmpty(PythonCode))
            {
                ~w
                _predicate = _scope.GetVariable("~w");
            }
        }

        /// <summary>
        /// Set the Python code before instantiation.
        /// </summary>
        public static void SetPythonCode(string code)
        {
            PythonCode = code;
        }

        /// <summary>
        /// Invoke the predicate with a stream of input records.
        /// </summary>
        /// <param name="inputStream">Stream of input dictionaries</param>
        /// <returns>Stream of result objects</returns>
        public IEnumerable<~wResult> Invoke(IEnumerable<IDictionary<string, object>> inputStream)
        {
            var pyInputList = inputStream.Select(ToPythonDict).ToList();
            var results = _predicate(pyInputList);

            foreach (dynamic result in results)
            {
                yield return FromPythonDict(result);
            }
        }

        /// <summary>
        /// Invoke the predicate with a single input record.
        /// </summary>
        public IEnumerable<~wResult> InvokeSingle(~w)
        {
            var input = new Dictionary<string, object>
            {
~w
            };
            return Invoke(new[] { input });
        }

        /// <summary>
        /// Convert C# dictionary to Python dict.
        /// </summary>
        private dynamic ToPythonDict(IDictionary<string, object> dict)
        {
            var pyDict = _engine.Execute("dict()", _scope);
            foreach (var kvp in dict)
            {
                pyDict[kvp.Key] = kvp.Value;
            }
            return pyDict;
        }

        /// <summary>
        /// Convert Python dict to typed result.
        /// </summary>
        private ~wResult FromPythonDict(dynamic pyDict)
        {
            return new ~wResult
            {
~w
            };
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _engine?.Runtime?.Shutdown();
                _disposed = true;
            }
        }
    }
}
', [NameStr, Arity, Namespace, ResultClassCode, NameStr, ClassName, ClassName,
    PythonLoadCode, NameStr, ClassName, ClassName, MethodParams, InvokeArgs,
    ClassName, ClassName, generate_result_assignments(ArgNames)]).

%% generate_cpython_wrapper/8
%  Generate C# wrapper using CPython subprocess for execution.
%
generate_cpython_wrapper(Name, Arity, ArgNames, Namespace, ClassName, EmbedPython, PythonFile, Code) :-
    atom_string(Name, NameStr),

    % Generate method parameters
    generate_csharp_parameters(ArgNames, Arity, MethodParams),

    % Generate property class for output
    generate_result_class(ArgNames, ResultClassCode),

    % Generate invoke arguments
    generate_invoke_args(ArgNames, Arity, InvokeArgs),

    % Determine Python path handling
    (   EmbedPython == true
    ->  PythonPathCode = 'var tempFile = Path.GetTempFileName() + ".py";\n            File.WriteAllText(tempFile, PythonCode);',
        CleanupCode = 'File.Delete(tempFile);',
        ScriptPath = 'tempFile'
    ;   format(string(PythonPathCode), 'var tempFile = "~w";', [PythonFile]),
        CleanupCode = '',
        ScriptPath = 'tempFile'
    ),

    format(string(Code),
'// Generated C# wrapper for Python predicate: ~w/~w
// Runtime: CPython (subprocess with JSONL communication)

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace ~w
{
~w

    /// <summary>
    /// C# wrapper for the ~w Python predicate.
    /// Uses CPython subprocess with JSONL for communication.
    /// </summary>
    public class ~w : IDisposable
    {
        private Process _process;
        private bool _disposed;

        // Embedded Python code (set via SetPythonCode or constructor)
        private static string PythonCode = "";
        private static string PythonPath = "python3";

        /// <summary>
        /// Set the Python code before instantiation.
        /// </summary>
        public static void SetPythonCode(string code)
        {
            PythonCode = code;
        }

        /// <summary>
        /// Set the Python interpreter path.
        /// </summary>
        public static void SetPythonPath(string path)
        {
            PythonPath = path;
        }

        /// <summary>
        /// Invoke the predicate with a stream of input records.
        /// </summary>
        public IEnumerable<~wResult> Invoke(IEnumerable<IDictionary<string, object>> inputStream)
        {
            ~w

            var psi = new ProcessStartInfo
            {
                FileName = PythonPath,
                Arguments = ~w,
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            _process = Process.Start(psi);

            // Write input in background
            var writeTask = System.Threading.Tasks.Task.Run(() =>
            {
                foreach (var record in inputStream)
                {
                    var json = JsonSerializer.Serialize(record);
                    _process.StandardInput.WriteLine(json);
                }
                _process.StandardInput.Close();
            });

            // Read output
            string line;
            while ((line = _process.StandardOutput.ReadLine()) != null)
            {
                if (!string.IsNullOrWhiteSpace(line))
                {
                    var dict = JsonSerializer.Deserialize<Dictionary<string, object>>(line);
                    yield return FromDict(dict);
                }
            }

            writeTask.Wait();
            _process.WaitForExit();
            ~w
        }

        /// <summary>
        /// Invoke the predicate with a single input record.
        /// </summary>
        public IEnumerable<~wResult> InvokeSingle(~w)
        {
            var input = new Dictionary<string, object>
            {
~w
            };
            return Invoke(new[] { input });
        }

        /// <summary>
        /// Convert dictionary to typed result.
        /// </summary>
        private ~wResult FromDict(Dictionary<string, object> dict)
        {
            return new ~wResult
            {
~w
            };
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _process?.Kill();
                _process?.Dispose();
                _disposed = true;
            }
        }
    }
}
', [NameStr, Arity, Namespace, ResultClassCode, NameStr, ClassName,
    ClassName, PythonPathCode, ScriptPath, CleanupCode,
    ClassName, MethodParams, InvokeArgs,
    ClassName, ClassName, generate_result_assignments(ArgNames)]).

%% generate_result_class(+ArgNames, -Code)
%  Generate C# result class with properties for each argument.
%
generate_result_class(ArgNames, Code) :-
    maplist(generate_property, ArgNames, Properties),
    atomic_list_concat(Properties, '\n', PropertiesCode),
    format(string(Code),
'    /// <summary>
    /// Result type for predicate output.
    /// </summary>
    public class Result
    {
~w
    }', [PropertiesCode]).

generate_property(Name, Code) :-
    format(string(Code), '        public object ~w { get; set; }', [Name]).

%% generate_csharp_parameters(+ArgNames, +Arity, -Code)
%  Generate C# method parameter list.
%
generate_csharp_parameters(ArgNames, Arity, Code) :-
    length(ArgNames, Len),
    (   Len >= Arity
    ->  take_n(ArgNames, Arity, InputArgs)
    ;   generate_arg_names(Arity, InputArgs)
    ),
    maplist(format_param, InputArgs, Params),
    atomic_list_concat(Params, ', ', Code).

format_param(Name, Param) :-
    format(string(Param), 'object ~w', [Name]).

%% generate_invoke_args(+ArgNames, +Arity, -Code)
%  Generate dictionary initialization for invoke.
%
generate_invoke_args(ArgNames, Arity, Code) :-
    length(ArgNames, Len),
    (   Len >= Arity
    ->  take_n(ArgNames, Arity, InputArgs)
    ;   generate_arg_names(Arity, InputArgs)
    ),
    maplist(format_dict_entry, InputArgs, Entries),
    atomic_list_concat(Entries, ',\n', Code).

format_dict_entry(Name, Entry) :-
    format(string(Entry), '                ["~w"] = ~w', [Name, Name]).

%% generate_result_assignments(+ArgNames)
%  Generate property assignments from Python dict (for string interpolation).
%
generate_result_assignments(ArgNames) :-
    maplist(format_result_assignment, ArgNames, Assignments),
    atomic_list_concat(Assignments, ',\n', _Code).

format_result_assignment(Name, Assignment) :-
    format(string(Assignment), '                ~w = pyDict["~w"]', [Name, Name]).

%% generate_arg_names(+Arity, -Names)
%  Generate default argument names.
%
generate_arg_names(Arity, Names) :-
    findall(Name, (
        between(0, Arity, I),
        I < Arity,
        format(atom(Name), 'Arg~d', [I])
    ), Names).

%% take_n(+List, +N, -FirstN)
%  Take first N elements from list.
%
take_n(_, 0, []) :- !.
take_n([], _, []) :- !.
take_n([H|T], N, [H|Rest]) :-
    N > 0,
    N1 is N - 1,
    take_n(T, N1, Rest).

%% string_to_pascal_case(+String, -PascalCase)
%  Convert snake_case or lowercase to PascalCase.
%
string_to_pascal_case(String, PascalCase) :-
    split_string(String, "_", "", Parts),
    maplist(capitalize_first, Parts, CapParts),
    atomic_list_concat(CapParts, '', PascalCase).

capitalize_first("", "") :- !.
capitalize_first(Str, Cap) :-
    string_chars(Str, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HU, [HUC]),
    string_chars(Cap, [HUC|T]).

%% generate_pipeline_host(+Predicates, +Options, -Code)
%  Generate C# host that manages multiple Python predicates.
%
%  Predicates: List of Name/Arity or Module:Name/Arity
%
generate_pipeline_host(Predicates, Options, Code) :-
    option_or_default(namespace(Namespace), Options, 'UnifyWeaver.Generated'),
    option_or_default(class_name(ClassName), Options, 'PipelineHost'),

    % Generate wrapper for each predicate
    maplist(generate_predicate_field, Predicates, Fields),
    atomic_list_concat(Fields, '\n', FieldsCode),

    maplist(generate_predicate_init, Predicates, Inits),
    atomic_list_concat(Inits, '\n', InitsCode),

    format(string(Code),
'// Generated Pipeline Host for multiple Python predicates
// Manages IronPython/CPython bridges for cross-target calls

using System;
using System.Collections.Generic;

namespace ~w
{
    /// <summary>
    /// Host for managing multiple Python predicate wrappers.
    /// </summary>
    public class ~w : IDisposable
    {
~w

        public ~w()
        {
~w
        }

        /// <summary>
        /// Execute a predicate by name.
        /// </summary>
        public IEnumerable<IDictionary<string, object>> Execute(
            string predicateName,
            IEnumerable<IDictionary<string, object>> input)
        {
            return predicateName switch
            {
                // Add cases for each predicate
                _ => throw new ArgumentException($"Unknown predicate: {predicateName}")
            };
        }

        public void Dispose()
        {
            // Dispose all predicate wrappers
        }
    }
}
', [Namespace, ClassName, FieldsCode, ClassName, InitsCode]).

generate_predicate_field(Name/_Arity, Code) :-
    atom_string(Name, NameStr),
    string_to_pascal_case(NameStr, PascalName),
    format(string(Code), '        private ~wPredicate _~w;', [PascalName, NameStr]).
generate_predicate_field(_:Name/Arity, Code) :-
    generate_predicate_field(Name/Arity, Code).

generate_predicate_init(Name/_Arity, Code) :-
    atom_string(Name, NameStr),
    string_to_pascal_case(NameStr, PascalName),
    format(string(Code), '            _~w = new ~wPredicate();', [NameStr, PascalName]).
generate_predicate_init(_:Name/Arity, Code) :-
    generate_predicate_init(Name/Arity, Code).

%% ============================================
%% Tests for Python/C# Glue
%% ============================================

test_python_csharp_glue :-
    format('~n=== Python/C# Glue Tests ===~n~n', []),

    % Test 1: Generate IronPython wrapper
    format('[Test 1] Generate IronPython wrapper~n', []),
    generate_python_predicate_wrapper(user_info, 2, [
        runtime(ironpython),
        arg_names(['UserId', 'Email']),
        namespace('TestNamespace')
    ], Code1),
    (   sub_string(Code1, _, _, _, "IronPython.Hosting"),
        sub_string(Code1, _, _, _, "UserInfoPredicate"),
        sub_string(Code1, _, _, _, "ScriptEngine")
    ->  format('  [PASS] IronPython wrapper generated~n', [])
    ;   format('  [FAIL] IronPython wrapper missing components~n', [])
    ),

    % Test 2: Generate CPython wrapper
    format('[Test 2] Generate CPython wrapper~n', []),
    generate_python_predicate_wrapper(filter_users, 3, [
        runtime(cpython),
        arg_names(['Name', 'Age', 'Active']),
        namespace('TestNamespace')
    ], Code2),
    (   sub_string(Code2, _, _, _, "Process.Start"),
        sub_string(Code2, _, _, _, "FilterUsersPredicate"),
        sub_string(Code2, _, _, _, "JSONL")
    ->  format('  [PASS] CPython wrapper generated~n', [])
    ;   format('  [FAIL] CPython wrapper missing components~n', [])
    ),

    % Test 3: Generate pipeline host
    format('[Test 3] Generate pipeline host~n', []),
    generate_pipeline_host([user_info/2, filter_users/3], [
        namespace('TestNamespace'),
        class_name('MyPipelineHost')
    ], Code3),
    (   sub_string(Code3, _, _, _, "MyPipelineHost"),
        sub_string(Code3, _, _, _, "_user_info"),
        sub_string(Code3, _, _, _, "_filter_users")
    ->  format('  [PASS] Pipeline host generated~n', [])
    ;   format('  [FAIL] Pipeline host missing components~n', [])
    ),

    % Test 4: String to PascalCase conversion
    format('[Test 4] String to PascalCase~n', []),
    string_to_pascal_case("user_info", Pascal1),
    string_to_pascal_case("filteractive", Pascal2),
    (   Pascal1 == 'UserInfo',
        Pascal2 == 'Filteractive'
    ->  format('  [PASS] PascalCase conversion works~n', [])
    ;   format('  [FAIL] PascalCase: got ~w, ~w~n', [Pascal1, Pascal2])
    ),

    % Test 5: Generate arg names
    format('[Test 5] Generate default arg names~n', []),
    generate_arg_names(3, Names5),
    (   Names5 == ['Arg0', 'Arg1', 'Arg2']
    ->  format('  [PASS] Default arg names: ~w~n', [Names5])
    ;   format('  [FAIL] Expected [Arg0,Arg1,Arg2], got ~w~n', [Names5])
    ),

    format('~n=== Python/C# Glue Tests Complete ===~n', []).
