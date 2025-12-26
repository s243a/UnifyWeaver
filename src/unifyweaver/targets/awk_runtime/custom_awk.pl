% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_awk.pl - Custom AWK Component Type
%
% Allows injecting raw AWK code as a component.
% The code is wrapped in a function.
%
% Example:
%   declare_component(source, my_transform, custom_awk, [
%       code("return toupper(input)"),
%       includes(["lib/utils.awk"])
%   ]).

:- module(custom_awk, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom AWK Component'),
    version('1.0.0'),
    description('Injects custom AWK code and exposes it as a component')
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
    throw(error(runtime_invocation_not_supported(custom_awk))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    (   member(includes(Includes), Config)
    ->  maplist(format_awk_include, Includes, IncludeLines),
        atomic_list_concat(IncludeLines, '\n', IncludesCode)
    ;   IncludesCode = ''
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"# Custom Component: ~w
~w

# Custom component: ~w
# Process input and return output.
function comp_~w_invoke(input) {
    ~w
}
", [NameStr, IncludesCode, NameStr, NameStr, Body]).

%% format_awk_include(+File, -IncludeLine)
format_awk_include(File, IncludeLine) :-
    format(string(IncludeLine), "@include \"~w\"", [File]).

% ============================================================================
% REGISTRATION
% ============================================================================

:- initialization((
    register_component_type(source, custom_awk, custom_awk, [
        description("Custom AWK Code")
    ])
), now).
