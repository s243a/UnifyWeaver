% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_mypyc.pl - Custom mypyc Component Type
%
% Allows injecting Python code with full type annotations for mypyc.
% mypyc compiles type-annotated Python to C extensions.
%
% Example:
%   declare_component(source, typed_processor, custom_mypyc, [
%       code("return [x * 2 for x in input]"),
%       types([list(integer)]),
%       return_type(list(integer))
%   ]).

:- module(custom_mypyc, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom mypyc Component'),
    version('1.0.0'),
    description('Injects fully-typed Python code for mypyc compilation')
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
    throw(error(runtime_invocation_not_supported(custom_mypyc))).

%% mypyc_type_map(+PrologType, -MypycType)
mypyc_type_map(integer, 'int').
mypyc_type_map(float, 'float').
mypyc_type_map(boolean, 'bool').
mypyc_type_map(string, 'str').
mypyc_type_map(any, 'Any').
mypyc_type_map(none, 'None').
mypyc_type_map(list(Type), ListType) :-
    mypyc_type_map(Type, InnerType),
    format(atom(ListType), 'List[~w]', [InnerType]).
mypyc_type_map(dict(K, V), DictType) :-
    mypyc_type_map(K, KeyType),
    mypyc_type_map(V, ValType),
    format(atom(DictType), 'Dict[~w, ~w]', [KeyType, ValType]).
mypyc_type_map(optional(Type), OptType) :-
    mypyc_type_map(Type, InnerType),
    format(atom(OptType), 'Optional[~w]', [InnerType]).
mypyc_type_map(tuple(Types), TupleType) :-
    maplist(mypyc_type_map, Types, InnerTypes),
    atomic_list_concat(InnerTypes, ', ', TypesStr),
    format(atom(TupleType), 'Tuple[~w]', [TypesStr]).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Build typed arguments
    (   member(types(Types), Config)
    ->  build_mypyc_args(Types, TypedArgs),
        atomic_list_concat(TypedArgs, ', ', ArgsStr)
    ;   ArgsStr = "input: Any"
    ),

    % Return type
    (   member(return_type(RetType), Config),
        mypyc_type_map(RetType, MypycRetType)
    ->  format(string(RetHint), " -> ~w", [MypycRetType])
    ;   RetHint = " -> Any"
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"# mypyc Component: ~w
# Compile with: mypyc this_file.py

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Final
import math

def comp_~w(~w)~w:
    \"\"\"
    mypyc-compiled component: ~w

    Build commands:
        mypy --strict this_file.py  # Check types first
        mypyc this_file.py          # Compile to C extension
    \"\"\"
~w
", [NameStr, NameStr, ArgsStr, RetHint, NameStr, Body]).

%% build_mypyc_args(+Types, -TypedArgs)
build_mypyc_args(Types, TypedArgs) :-
    build_mypyc_args(Types, 0, TypedArgs).

build_mypyc_args([], _, []).
build_mypyc_args([Type|Types], N, [TypedArg|Rest]) :-
    mypyc_type_map(Type, MypycType),
    format(atom(TypedArg), "arg~w: ~w", [N, MypycType]),
    N1 is N + 1,
    build_mypyc_args(Types, N1, Rest).

%% Register this component type
:- initialization((
    register_component_type(source, custom_mypyc, custom_mypyc, [
        description("Custom mypyc Code")
    ])
), now).
