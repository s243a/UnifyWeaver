% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_jython.pl - Custom Jython Component Type
%
% Allows injecting raw Jython/Python code as a component.
% Can use both Python and Java libraries.
%
% Example:
%   declare_component(source, my_transform, custom_jython, [
%       code("return input.upper()"),
%       imports(["java.util.ArrayList", "os"])
%   ]).

:- module(custom_jython, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom Jython Component'),
    version('1.0.0'),
    description('Injects custom Jython code and exposes it as a component')
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
    throw(error(runtime_invocation_not_supported(custom_jython))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    (   member(imports(Imports), Config)
    ->  maplist(format_jython_import, Imports, ImportLines),
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

%% format_jython_import(+Module, -ImportLine)
format_jython_import(Module, ImportLine) :-
    format(string(ImportLine), "import ~w", [Module]).

% ============================================================================
% REGISTRATION
% ============================================================================

:- initialization((
    register_component_type(source, custom_jython, custom_jython, [
        description("Custom Jython Code")
    ])
), now).
