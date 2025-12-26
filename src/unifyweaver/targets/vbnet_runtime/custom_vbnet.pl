% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_vbnet.pl - Custom VB.NET Component Type
%
% Allows injecting raw VB.NET code as a component.
% The code is wrapped in a Class with an Invoke function.
%
% Example:
%   declare_component(source, my_transform, custom_vbnet, [
%       code("Return input.ToUpper()"),
%       imports(["System.Text", "System.IO"])
%   ]).

:- module(custom_vbnet, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom VB.NET Component'),
    version('1.0.0'),
    description('Injects custom VB.NET code and exposes it as a component')
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
    throw(error(runtime_invocation_not_supported(custom_vbnet))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    (   member(imports(Imports), Config)
    ->  maplist(format_vbnet_imports, Imports, ImportLines),
        atomic_list_concat(ImportLines, '\n', ImportsCode)
    ;   ImportsCode = ''
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"' Custom Component: ~w
~w

''' <summary>
''' Custom component: ~w
''' </summary>
Public Class Comp~w(Of T)
    ''' <summary>
    ''' Process input and return output.
    ''' </summary>
    Public Function Invoke(input As T) As T
        ~w
    End Function
End Class
", [NameStr, ImportsCode, NameStr, NameStr, Body]).

%% format_vbnet_imports(+Namespace, -ImportLine)
format_vbnet_imports(Namespace, ImportLine) :-
    format(string(ImportLine), "Imports ~w", [Namespace]).

% ============================================================================
% REGISTRATION
% ============================================================================

:- initialization((
    register_component_type(source, custom_vbnet, custom_vbnet, [
        description("Custom VB.NET Code")
    ])
), now).
