% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_codon.pl - Custom Codon Component Type
%
% Allows injecting Codon-compatible Python code as a component.
% Codon compiles a subset of Python directly to native code via LLVM.
%
% Example:
%   declare_component(source, fast_compute, custom_codon, [
%       code("return sum(data)"),
%       types([list(integer)]),
%       return_type(integer),
%       parallel(true)
%   ]).

:- module(custom_codon, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom Codon Component'),
    version('1.0.0'),
    description('Injects Codon-compatible Python code for native compilation')
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
    throw(error(runtime_invocation_not_supported(custom_codon))).

%% codon_type_map(+PrologType, -CodonType)
codon_type_map(integer, 'int').
codon_type_map(float, 'float').
codon_type_map(boolean, 'bool').
codon_type_map(string, 'str').
codon_type_map(list(integer), 'List[int]').
codon_type_map(list(float), 'List[float]').
codon_type_map(list(string), 'List[str]').

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Build typed arguments
    (   member(types(Types), Config)
    ->  build_codon_args(Types, TypedArgs),
        atomic_list_concat(TypedArgs, ', ', ArgsStr)
    ;   ArgsStr = "input"
    ),

    % Return type
    (   member(return_type(RetType), Config),
        codon_type_map(RetType, CodonRetType)
    ->  format(string(RetHint), " -> ~w", [CodonRetType])
    ;   RetHint = ""
    ),

    % JIT decorator for hybrid mode
    (   member(jit(true), Config)
    ->  JitDecorator = "@codon.jit\n"
    ;   JitDecorator = ""
    ),

    % Parallel decorator
    (   member(parallel(true), Config)
    ->  ParallelComment = "\n    # Use @par for parallel loops in this function"
    ;   ParallelComment = ""
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"# Codon Component: ~w
# Compile with: codon build -release this_file.py

from typing import List, Tuple, Optional
import math
~w
def comp_~w(~w)~w:
    \"\"\"
    Codon-compiled component: ~w

    Build commands:
        codon build this_file.py          # Debug
        codon build -release this_file.py # Optimized
        codon run this_file.py            # JIT
    \"\"\"~w
~w
", [NameStr, JitDecorator, NameStr, ArgsStr, RetHint, NameStr, ParallelComment, Body]).

%% build_codon_args(+Types, -TypedArgs)
build_codon_args(Types, TypedArgs) :-
    build_codon_args(Types, 0, TypedArgs).

build_codon_args([], _, []).
build_codon_args([Type|Types], N, [TypedArg|Rest]) :-
    codon_type_map(Type, CodonType),
    format(atom(TypedArg), "arg~w: ~w", [N, CodonType]),
    N1 is N + 1,
    build_codon_args(Types, N1, Rest).

%% Register this component type
:- initialization((
    register_component_type(source, custom_codon, custom_codon, [
        description("Custom Codon Code")
    ])
), now).
