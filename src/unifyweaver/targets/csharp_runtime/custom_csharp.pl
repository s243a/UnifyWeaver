% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_csharp.pl - Custom C# Component Type
%
% Allows injecting raw C# code as a component.
% The code is wrapped in a class with an Invoke() method.
%
% Example:
%   declare_component(source, my_transform, custom_csharp, [
%       code("return input.ToUpper();"),
%       usings(["System.Text", "System.Linq"])
%   ]).

:- module(custom_csharp, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
%
%  Metadata about the custom_csharp component type.
%
type_info(info(
    name('Custom C# Component'),
    version('1.0.0'),
    description('Injects custom C# code and exposes it as a component')
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
    throw(error(runtime_invocation_not_supported(custom_csharp))).

%% compile_component(+Name, +Config, +Options, -Code)
%
%  Compile the custom C# component to C# code.
%  Generates a class with an Invoke() method.
%
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Collect using directives if specified
    (   member(usings(Usings), Config)
    ->  maplist(format_csharp_using, Usings, UsingLines),
        atomic_list_concat(UsingLines, '\n', UsingsCode)
    ;   UsingsCode = ''
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"// Custom Component: ~w
~w

/// <summary>
/// Custom component: ~w
/// </summary>
public class Comp~w
{
    /// <summary>
    /// Process input and return output.
    /// </summary>
    public T Invoke<T>(T input)
    {
        ~w
    }
}
", [NameStr, UsingsCode, NameStr, NameStr, Body]).

%% format_csharp_using(+Namespace, -UsingLine)
%
%  Format a C# using directive.
%
format_csharp_using(Namespace, UsingLine) :-
    format(string(UsingLine), "using ~w;", [Namespace]).

% ============================================================================
% REGISTRATION
% ============================================================================

%% Register this component type with the component registry
%
:- initialization((
    register_component_type(source, custom_csharp, custom_csharp, [
        description("Custom C# Code")
    ])
), now).
