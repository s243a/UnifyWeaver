% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_cython.pl - Custom Cython Component Type
%
% Allows injecting Cython code (.pyx) as a component.
% Generates code with cdef/cpdef and type declarations.
%
% Example:
%   declare_component(source, fast_multiply, custom_cython, [
%       code("return a * b"),
%       mode(cpdef),
%       types([double, double]),
%       return_type(double)
%   ]).

:- module(custom_cython, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom Cython Component'),
    version('1.0.0'),
    description('Injects Cython code with static typing as a component')
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
    throw(error(runtime_invocation_not_supported(custom_cython))).

%% cython_type_map(+PrologType, -CythonType)
cython_type_map(integer, 'long long').
cython_type_map(int, 'int').
cython_type_map(double, 'double').
cython_type_map(float, 'float').
cython_type_map(boolean, 'bint').
cython_type_map(string, 'str').
cython_type_map(object, 'object').

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Determine function mode
    (   member(mode(Mode), Config)
    ->  true
    ;   Mode = cpdef
    ),

    % Get return type
    (   member(return_type(RetType), Config),
        cython_type_map(RetType, CythonRetType)
    ->  true
    ;   CythonRetType = 'object'
    ),

    % Build typed arguments
    (   member(types(Types), Config)
    ->  build_typed_args(Types, TypedArgs),
        atomic_list_concat(TypedArgs, ', ', ArgsStr)
    ;   ArgsStr = "input"
    ),

    % Inline decorator
    (   member(inline(true), Config)
    ->  Inline = "@cython.inline\n"
    ;   Inline = ""
    ),

    % Nogil
    (   member(nogil(true), Config)
    ->  NogilSuffix = " nogil"
    ;   NogilSuffix = ""
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

# Cython Component: ~w
cimport cython
import numpy as np
cimport numpy as np

~w~w ~w comp_~w(~w)~w:
    \"\"\"Cython component: ~w\"\"\"
~w
", [NameStr, Inline, Mode, CythonRetType, NameStr, ArgsStr, NogilSuffix, NameStr, Body]).

%% build_typed_args(+Types, -TypedArgs)
build_typed_args(Types, TypedArgs) :-
    build_typed_args(Types, 0, TypedArgs).

build_typed_args([], _, []).
build_typed_args([Type|Types], N, [TypedArg|Rest]) :-
    cython_type_map(Type, CythonType),
    format(atom(TypedArg), "~w arg~w", [CythonType, N]),
    N1 is N + 1,
    build_typed_args(Types, N1, Rest).

%% Register this component type
:- initialization((
    register_component_type(source, custom_cython, custom_cython, [
        description("Custom Cython Code")
    ])
), now).
