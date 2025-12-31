% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_pyodide.pl - Custom Pyodide Component Type
%
% Allows injecting Python code that runs in browser via Pyodide.
% Generates both Python code and JavaScript wrapper.
%
% Example:
%   declare_component(source, matrix_inverse, custom_pyodide, [
%       code("return np.linalg.inv(input)"),
%       packages([numpy]),
%       js_wrapper(true)
%   ]).

:- module(custom_pyodide, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom Pyodide Component'),
    version('1.0.0'),
    description('Injects Python code for browser execution via Pyodide/WASM')
)).

%% validate_config(+Config)
validate_config(Config) :-
    (   member(code(Code), Config), string(Code)
    ->  true
    ;   throw(error(missing_or_invalid_code_option))
    ).

%% init_component(+Name, +Config)
init_component(_Name, _Config).

%% invoke_component(+Name, +Config, +Input, -Output)
invoke_component(_Name, _Config, _Input, _Output) :-
    throw(error(runtime_invocation_not_supported(custom_pyodide))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Collect package imports
    (   member(packages(Packages), Config)
    ->  maplist(format_import, Packages, ImportLines),
        atomic_list_concat(ImportLines, '\n', ImportsCode)
    ;   ImportsCode = "import numpy as np"
    ),

    % JavaScript wrapper if requested
    (   member(js_wrapper(true), Config)
    ->  atom_string(Name, NameStr),
        format(string(JSWrapper),
"
// JavaScript wrapper for ~w
async function js_~w(input) {
    const code = `
import json
input = json.loads('${JSON.stringify(input)}')
result = comp_~w(input)
json.dumps(result.tolist() if hasattr(result, 'tolist') else result)
    `;
    const result = await pyodide.runPython(code);
    return JSON.parse(result);
}
", [NameStr, NameStr, NameStr])
    ;   JSWrapper = ""
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"# Pyodide Component: ~w
# Runs in browser via WebAssembly - no server required
# Security: Sandboxed in browser, no filesystem/network access

~w

def comp_~w(input):
    \"\"\"
    Pyodide component: ~w

    Runs entirely client-side in the browser.
    Full NumPy/SciPy available.
    \"\"\"
~w
~w
", [NameStr, ImportsCode, NameStr, NameStr, Body, JSWrapper]).

%% format_import(+Package, -Line)
format_import(Package, Line) :-
    format(string(Line), "import ~w", [Package]).

%% Register this component type
:- initialization((
    register_component_type(source, custom_pyodide, custom_pyodide, [
        description("Custom Pyodide/Browser Python Code")
    ])
), now).
