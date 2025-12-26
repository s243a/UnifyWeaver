% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_fsharp.pl - Custom F# Component Type
%
% Allows injecting raw F# code as a component.
% The code is wrapped in a module with an invoke function.
%
% Example:
%   declare_component(source, my_transform, custom_fsharp, [
%       code("input.ToUpper()"),
%       opens(["System.Text", "System.IO"])
%   ]).

:- module(custom_fsharp, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom F# Component'),
    version('1.0.0'),
    description('Injects custom F# code and exposes it as a component')
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
    throw(error(runtime_invocation_not_supported(custom_fsharp))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    (   member(opens(Opens), Config)
    ->  maplist(format_fsharp_open, Opens, OpenLines),
        atomic_list_concat(OpenLines, '\n', OpensCode)
    ;   OpensCode = ''
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"// Custom Component: ~w
~w

/// <summary>
/// Custom component: ~w
/// </summary>
module Comp~w =
    /// Process input and return output.
    let invoke (input: 'T) : 'T =
        ~w
", [NameStr, OpensCode, NameStr, NameStr, Body]).

%% format_fsharp_open(+Namespace, -OpenLine)
format_fsharp_open(Namespace, OpenLine) :-
    format(string(OpenLine), "open ~w", [Namespace]).

% ============================================================================
% REGISTRATION
% ============================================================================

:- initialization((
    register_component_type(source, custom_fsharp, custom_fsharp, [
        description("Custom F# Code")
    ])
), now).
