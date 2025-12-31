% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% python_pyodide_target.pl - Pyodide Target for Browser-based Python
%
% Generates Python code and JavaScript wrappers for Pyodide execution.
% Pyodide compiles CPython to WebAssembly, enabling full Python in browsers.
%
% License: MPL 2.0 (Pyodide itself)
%
% Features:
% - Full NumPy, SciPy, pandas support in browser
% - No server required - runs entirely client-side
% - Browser sandbox security
% - Async/await JavaScript integration
%
% Use cases:
% - Interactive data visualization
% - Client-side computation (no server security concerns)
% - Educational/demo applications
%
% Example:
%   ?- compile_pyodide_module(matrix_ops, [packages([numpy])], Code).
%   ?- generate_pyodide_html(matrix_demo, [chart(true)], HTML).

:- module(python_pyodide_target, [
    compile_pyodide_function/3,
    compile_pyodide_module/3,
    generate_pyodide_loader/2,
    generate_pyodide_html/3,
    generate_pyodide_worker/2,
    pyodide_packages/1,
    init_pyodide_target/0
]).

:- use_module(python_target).
:- use_module('../core/binding_registry').
:- use_module('../core/component_registry').
:- use_module('python_runtime/custom_pyodide', []).

%% init_pyodide_target
%  Initialize Pyodide target
init_pyodide_target :-
    init_python_target.

%% pyodide_packages(-Packages)
%  List of commonly available Pyodide packages
pyodide_packages([
    numpy, scipy, pandas, matplotlib, scikit_learn,
    sympy, networkx, pillow, opencv_python, statsmodels
]).

%% compile_pyodide_function(+Pred/Arity, +Options, -Code)
%  Compile a predicate to Pyodide-compatible Python
%  Options:
%    - packages(List): Required packages to load
%    - async(true/false): Generate async function
compile_pyodide_function(Pred/Arity, Options, Code) :-
    % Get base Python code
    compile_predicate_to_python(Pred/Arity, Options, BaseCode),

    functor(Pred, Name, _),

    % Collect package imports
    (   member(packages(Packages), Options)
    ->  maplist(format_import, Packages, ImportLines),
        atomic_list_concat(ImportLines, '\n', ImportsCode)
    ;   ImportsCode = "import numpy as np"
    ),

    % Async wrapper if needed
    (   member(async(true), Options)
    ->  AsyncPrefix = "async ",
        AwaitNote = "# Use: result = await function_name(args)"
    ;   AsyncPrefix = "",
        AwaitNote = ""
    ),

    format(string(Code),
"# Pyodide-compatible Python
# Runs in browser via WebAssembly
~w
~w
~wdef ~w(~w):
    \"\"\"Generated from Prolog predicate ~w/~w.

    Runs in Pyodide (browser-based Python).
    ~w
    \"\"\"
~w
", [ImportsCode, AwaitNote, AsyncPrefix, Name, "args", Name, Arity, AwaitNote, BaseCode]).

format_import(Package, Line) :-
    format(string(Line), "import ~w", [Package]).

%% compile_pyodide_module(+ModuleName, +Options, -Code)
%  Compile a complete Pyodide module with multiple functions
%  Options:
%    - predicates(List): List of Pred/Arity to compile
%    - packages(List): Required packages
%    - exports(List): Functions to expose to JavaScript
compile_pyodide_module(ModuleName, Options, Code) :-
    % Package imports
    (   member(packages(Packages), Options)
    ->  maplist(format_import, Packages, ImportLines),
        atomic_list_concat(ImportLines, '\n', ImportsCode)
    ;   ImportsCode = "import numpy as np\nimport json"
    ),

    % Compile predicates
    (   member(predicates(Preds), Options)
    ->  maplist(compile_pred_for_pyodide(Options), Preds, PredCodes),
        atomic_list_concat(PredCodes, '\n\n', PredsCode)
    ;   PredsCode = "# Add functions here\npass"
    ),

    % Export registry for JavaScript access
    (   member(exports(Exports), Options)
    ->  maplist(format_export, Exports, ExportEntries),
        atomic_list_concat(ExportEntries, ',\n    ', ExportsStr),
        format(string(ExportRegistry),
"
# Registry for JavaScript access
EXPORTS = {
    ~w
}

def call_function(name, *args):
    \"\"\"Call exported function by name (for JS interop).\"\"\"
    if name in EXPORTS:
        return EXPORTS[name](*args)
    raise ValueError(f'Unknown function: {name}')
", [ExportsStr])
    ;   ExportRegistry = ""
    ),

    format(string(Code),
"# Pyodide Module: ~w
# Runs entirely in browser via WebAssembly
# No server required - secure client-side execution

~w

~w
~w
", [ModuleName, ImportsCode, PredsCode, ExportRegistry]).

compile_pred_for_pyodide(Options, Pred/Arity, Code) :-
    compile_pyodide_function(Pred/Arity, Options, Code).

format_export(FuncName, Entry) :-
    format(string(Entry), "'~w': ~w", [FuncName, FuncName]).

%% generate_pyodide_loader(+Options, -JSCode)
%  Generate JavaScript code to load and initialize Pyodide
%  Options:
%    - packages(List): Packages to pre-load
%    - python_code(Code): Python code to run after init
generate_pyodide_loader(Options, JSCode) :-
    % Packages to load
    (   member(packages(Packages), Options)
    ->  maplist(atom_string, Packages, PkgStrings),
        format(string(PackagesArray), "~w", [PkgStrings])
    ;   PackagesArray = "['numpy']"
    ),

    format(string(JSCode),
"// Pyodide Loader
// Generated by UnifyWeaver

class PyodideRunner {
    constructor() {
        this.pyodide = null;
        this.ready = false;
    }

    async init() {
        if (this.ready) return;

        console.log('Loading Pyodide...');
        this.pyodide = await loadPyodide();

        console.log('Loading packages...');
        await this.pyodide.loadPackage(~w);

        this.ready = true;
        console.log('Pyodide ready!');
    }

    async runPython(code) {
        if (!this.ready) await this.init();
        return await this.pyodide.runPythonAsync(code);
    }

    async callFunction(pythonCode, funcName, ...args) {
        if (!this.ready) await this.init();

        // Load the Python module
        await this.pyodide.runPythonAsync(pythonCode);

        // Call the function with JSON-serialized args
        const argsJson = JSON.stringify(args);
        const result = await this.pyodide.runPythonAsync(`
            import json
            args = json.loads('${argsJson}')
            result = ${funcName}(*args)
            json.dumps(result.tolist() if hasattr(result, 'tolist') else result)
        `);

        return JSON.parse(result);
    }

    // Convenience method for NumPy operations
    async numpy(operation) {
        return await this.runPython(`
import numpy as np
result = ${operation}
result.tolist() if hasattr(result, 'tolist') else result
        `);
    }
}

// Global instance
const pyodide = new PyodideRunner();
", [PackagesArray]).

%% generate_pyodide_html(+Title, +Options, -HTML)
%  Generate complete HTML page with Pyodide integration
%  Options:
%    - packages(List): Packages to load
%    - python_module(Code): Python code to include
%    - chart(true/false): Include Chart.js
generate_pyodide_html(Title, Options, HTML) :-
    % Chart.js inclusion
    (   member(chart(true), Options)
    ->  ChartScript = "<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>"
    ;   ChartScript = ""
    ),

    % Python module code
    (   member(python_module(PythonCode), Options)
    ->  format(string(PythonEmbed), "const PYTHON_CODE = `~w`;", [PythonCode])
    ;   PythonEmbed = "const PYTHON_CODE = '';"
    ),

    % Generate loader
    generate_pyodide_loader(Options, LoaderJS),

    format(string(HTML),
"<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>~w</title>
    ~w
    <script src=\"https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js\"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        .container {
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
        }
        canvas {
            background: #0f0f23;
            border-radius: 4px;
        }
        button {
            background: #00d4ff;
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            background: #00a8cc;
        }
        button:disabled {
            background: #666;
            cursor: wait;
        }
        input[type=\"number\"], input[type=\"range\"] {
            background: #0f0f23;
            border: 1px solid #00d4ff;
            color: #fff;
            padding: 8px;
            border-radius: 4px;
        }
        .status {
            color: #00d4ff;
            font-style: italic;
        }
        #output {
            font-family: monospace;
            white-space: pre-wrap;
            background: #0f0f23;
            padding: 10px;
            border-radius: 4px;
            max-height: 300px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <h1>~w</h1>
    <p class=\"status\" id=\"status\">Loading Pyodide...</p>

    <div class=\"container\">
        <div id=\"controls\"></div>
        <canvas id=\"chart\" width=\"800\" height=\"400\"></canvas>
    </div>

    <div class=\"container\">
        <h3>Output</h3>
        <pre id=\"output\"></pre>
    </div>

    <script>
~w

~w

// Initialize on load
document.addEventListener('DOMContentLoaded', async () => {
    const status = document.getElementById('status');
    const output = document.getElementById('output');

    try {
        await pyodide.init();
        status.textContent = 'Pyodide ready! NumPy available in browser.';

        if (PYTHON_CODE) {
            await pyodide.runPython(PYTHON_CODE);
            status.textContent = 'Python module loaded!';
        }
    } catch (e) {
        status.textContent = 'Error: ' + e.message;
        console.error(e);
    }
});
    </script>
</body>
</html>
", [Title, ChartScript, PythonEmbed, Title, LoaderJS]).

%% generate_pyodide_worker(+Options, -WorkerCode)
%  Generate Web Worker for running Pyodide in background thread
generate_pyodide_worker(Options, WorkerCode) :-
    (   member(packages(Packages), Options)
    ->  maplist(atom_string, Packages, PkgStrings),
        format(string(PackagesArray), "~w", [PkgStrings])
    ;   PackagesArray = "['numpy']"
    ),

    format(string(WorkerCode),
"// Pyodide Web Worker
// Runs Python in background thread, won't block UI

importScripts('https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js');

let pyodide = null;

async function init() {
    pyodide = await loadPyodide();
    await pyodide.loadPackage(~w);
    postMessage({ type: 'ready' });
}

onmessage = async (event) => {
    const { id, code } = event.data;

    try {
        if (!pyodide) await init();

        const result = await pyodide.runPythonAsync(code);
        postMessage({ id, type: 'result', data: result });
    } catch (error) {
        postMessage({ id, type: 'error', error: error.message });
    }
};

init();
", [PackagesArray]).

%% test_pyodide_target/0
%  Test Pyodide target code generation
test_pyodide_target :-
    format('Testing Pyodide target...~n'),

    % Test module generation
    compile_pyodide_module(test_module, [
        packages([numpy, scipy]),
        exports([compute, transform])
    ], ModCode),
    format('Module code:~n~w~n', [ModCode]),

    % Test loader generation
    generate_pyodide_loader([packages([numpy])], LoaderCode),
    format('Loader code length: ~w chars~n', [LoaderCode]),

    format('Pyodide target tests passed.~n').
