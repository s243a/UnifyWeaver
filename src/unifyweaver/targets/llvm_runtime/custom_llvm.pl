% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_llvm.pl - Custom LLVM IR Component Type
%
% Allows injecting raw LLVM IR code as a component.
% The code is wrapped in a function definition.
%
% Example:
%   declare_component(source, my_transform, custom_llvm, [
%       code("ret i64 %input"),
%       declares(["declare i64 @llvm.ctpop.i64(i64)"])
%   ]).

:- module(custom_llvm, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom LLVM IR Component'),
    version('1.0.0'),
    description('Injects custom LLVM IR code and exposes it as a component')
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
    throw(error(runtime_invocation_not_supported(custom_llvm))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    (   member(declares(Declares), Config)
    ->  atomic_list_concat(Declares, '\n', DeclaresCode)
    ;   DeclaresCode = ''
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"; Custom Component: ~w
~w

; Custom component: ~w
; Process input and return output.
define i64 @comp_~w_invoke(i64 %input) {
entry:
    ~w
}
", [NameStr, DeclaresCode, NameStr, NameStr, Body]).

% ============================================================================
% REGISTRATION
% ============================================================================

:- initialization((
    register_component_type(source, custom_llvm, custom_llvm, [
        description("Custom LLVM IR Code")
    ])
), now).
