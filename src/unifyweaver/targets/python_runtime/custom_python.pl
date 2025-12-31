% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_python.pl - Custom Python Component Type
%
% Allows injecting raw Python code as a component.
% The code is wrapped in a class with an invoke() method.
%
% Example:
%   declare_component(source, my_transform, custom_python, [
%       code("return input.upper()"),
%       imports(["re", "json"])
%   ]).

:- module(custom_python, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
%
%  Metadata about the custom_python component type.
%
type_info(info(
    name('Custom Python Component'),
    version('1.0.0'),
    description('Injects custom Python code and exposes it as a component')
)).

%% validate_config(+Config)
%
%  Validate that the configuration contains required options.
%
validate_config(Config) :-
    (   member(code(Code), Config), string(Code)
    ->  true
    ;   throw(error(missing_or_invalid_code_option))
    ).

%% init_component(+Name, +Config)
%
%  No initialization needed for compile-time component.
%
init_component(_Name, _Config).

%% invoke_component(+Name, +Config, +Input, -Output)
%
%  Runtime invocation not supported in Prolog (compilation only).
%
invoke_component(_Name, _Config, _Input, _Output) :-
    throw(error(runtime_invocation_not_supported(custom_python))).

%% compile_component(+Name, +Config, +Options, -Code)
%
%  Compile the custom Python component to Python code.
%  Generates a class with an invoke() method.
%
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Collect imports if specified
    (   member(imports(Imports), Config)
    ->  maplist(format_python_import, Imports, ImportLines),
        atomic_list_concat(ImportLines, '\n', ImportsCode)
    ;   ImportsCode = ''
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"# Custom Component: ~w
~w

class Comp_~w:
    \"\"\"Custom component: ~w\"\"\"

    def invoke(self, input):
        \"\"\"Process input and return output.\"\"\"
~w
", [NameStr, ImportsCode, NameStr, NameStr, Body]).

%% format_python_import(+Module, -ImportLine)
%
%  Format a Python import statement.
%
format_python_import(Module, ImportLine) :-
    format(string(ImportLine), "import ~w", [Module]).

% ============================================================================
% REGISTRATION
% ============================================================================

%% Register this component type with the component registry
%
:- initialization((
    register_component_type(source, custom_python, custom_python, [
        description("Custom Python Code")
    ])
), now).
