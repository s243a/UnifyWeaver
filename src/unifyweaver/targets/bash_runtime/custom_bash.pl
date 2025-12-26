% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_bash.pl - Custom Bash Component Type
%
% Allows injecting raw Bash code as a component.
% The code is wrapped in a function.
%
% Example:
%   declare_component(source, my_transform, custom_bash, [
%       code("echo \"${1^^}\""),
%       sources(["lib/utils.sh", "lib/helpers.sh"])
%   ]).

:- module(custom_bash, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom Bash Component'),
    version('1.0.0'),
    description('Injects custom Bash code and exposes it as a component')
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
    throw(error(runtime_invocation_not_supported(custom_bash))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    (   member(sources(Sources), Config)
    ->  maplist(format_bash_source, Sources, SourceLines),
        atomic_list_concat(SourceLines, '\n', SourcesCode)
    ;   SourcesCode = ''
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"#!/bin/bash
# Custom Component: ~w
~w

# Custom component: ~w
# Process input and return output.
comp_~w_invoke() {
    local input=\"$1\"
    ~w
}
", [NameStr, SourcesCode, NameStr, NameStr, Body]).

%% format_bash_source(+Script, -SourceLine)
format_bash_source(Script, SourceLine) :-
    format(string(SourceLine), "source \"~w\"", [Script]).

% ============================================================================
% REGISTRATION
% ============================================================================

:- initialization((
    register_component_type(source, custom_bash, custom_bash, [
        description("Custom Bash Code")
    ])
), now).
