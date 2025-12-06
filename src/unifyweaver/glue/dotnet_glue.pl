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
    generate_dotnet_pipeline/3      % generate_dotnet_pipeline(+Steps, +Options, -Code)
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
                Arguments = $"-c \\"{wrappedScript.Replace("\\"", "\\\\\\"")}\\""
",
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
