% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_numba.pl - Custom Numba Component Type
%
% Allows injecting Numba JIT-compiled code as a component.
% Supports @jit, @njit, @vectorize decorators.
%
% Example:
%   declare_component(source, fast_sum, custom_numba, [
%       code("return np.sum(input)"),
%       decorator(njit),
%       parallel(true)
%   ]).

:- module(custom_numba, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom Numba JIT Component'),
    version('1.0.0'),
    description('Injects Numba JIT-compiled code as a component')
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
    throw(error(runtime_invocation_not_supported(custom_numba))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Determine decorator
    (   member(decorator(Decorator), Config)
    ->  true
    ;   Decorator = njit
    ),

    % Build decorator options
    findall(Opt, (
        (member(cache(true), Config) -> Opt = "cache=True" ; fail)
    ;   (member(parallel(true), Config) -> Opt = "parallel=True" ; fail)
    ;   (member(fastmath(true), Config) -> Opt = "fastmath=True" ; fail)
    ), OptList),

    (   OptList = []
    ->  format(string(DecoratorStr), "@~w", [Decorator])
    ;   atomic_list_concat(OptList, ', ', OptStr),
        format(string(DecoratorStr), "@~w(~w)", [Decorator, OptStr])
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"# Numba JIT Component: ~w
from numba import jit, njit, vectorize, prange
import numpy as np

~w
def comp_~w(input):
    \"\"\"Numba JIT-compiled component: ~w\"\"\"
~w
", [NameStr, DecoratorStr, NameStr, NameStr, Body]).

%% Register this component type
:- initialization((
    register_component_type(source, custom_numba, custom_numba, [
        description("Custom Numba JIT Code")
    ])
), now).
