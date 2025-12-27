% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% python_numba_target.pl - Numba JIT Target for Python
%
% Generates Python code with Numba JIT decorators for high-performance
% numerical computing. Numba compiles Python to LLVM at runtime.
%
% License: BSD 2-Clause (Numba itself)
%
% Features:
% - @jit decorators for function compilation
% - @njit for no-Python mode (faster)
% - @vectorize for ufuncs
% - Type hints for better optimization
%
% Example:
%   ?- compile_numba_function(factorial/1, [], Code).

:- module(python_numba_target, [
    compile_numba_function/3,
    compile_numba_vectorized/3,
    compile_numba_parallel/3,
    generate_numba_header/1,
    numba_type_annotation/2,
    init_numba_target/0
]).

:- use_module(python_target).
:- use_module('../core/binding_registry').
:- use_module('../core/component_registry').
:- use_module('python_runtime/custom_numba', []).

%% init_numba_target
%  Initialize Numba target
init_numba_target :-
    init_python_target.

%% generate_numba_header(-Header)
%  Generate Numba import header
generate_numba_header(Header) :-
    Header = "from numba import jit, njit, vectorize, prange
import numpy as np
".

%% numba_type_annotation(+PrologType, -NumbaType)
%  Map Prolog types to Numba types
numba_type_annotation(integer, 'int64').
numba_type_annotation(float, 'float64').
numba_type_annotation(number, 'float64').
numba_type_annotation(list(integer), 'int64[:]').
numba_type_annotation(list(float), 'float64[:]').
numba_type_annotation(array(integer), 'int64[:,:]').
numba_type_annotation(array(float), 'float64[:,:]').
numba_type_annotation(boolean, 'boolean').

%% compile_numba_function(+Pred/Arity, +Options, -Code)
%  Compile a predicate to Numba-decorated Python function
%  Options:
%    - nopython(true/false): Use @njit instead of @jit (default: true)
%    - cache(true/false): Cache compiled code (default: true)
%    - parallel(true/false): Enable parallel execution (default: false)
%    - types(TypeList): Explicit type signatures
compile_numba_function(Pred/Arity, Options, Code) :-
    % Get base Python code
    compile_predicate_to_python(Pred/Arity, Options, BaseCode),

    % Build decorator
    build_numba_decorator(Options, Decorator),

    % Add Numba header
    generate_numba_header(Header),

    % Combine
    format(string(Code), "~w~w~w", [Header, Decorator, BaseCode]).

%% build_numba_decorator(+Options, -Decorator)
%  Build the appropriate Numba decorator
build_numba_decorator(Options, Decorator) :-
    (   member(nopython(false), Options)
    ->  DecoratorBase = "@jit"
    ;   DecoratorBase = "@njit"  % Default: no-Python mode for speed
    ),

    % Build options string
    findall(Opt, (
        (member(cache(true), Options) -> Opt = "cache=True" ; fail)
    ;   (member(parallel(true), Options) -> Opt = "parallel=True" ; fail)
    ;   (member(fastmath(true), Options) -> Opt = "fastmath=True" ; fail)
    ), OptList),

    (   OptList = []
    ->  format(string(Decorator), "~w~n", [DecoratorBase])
    ;   atomic_list_concat(OptList, ', ', OptStr),
        format(string(Decorator), "~w(~w)~n", [DecoratorBase, OptStr])
    ).

%% compile_numba_vectorized(+Pred/Arity, +Options, -Code)
%  Compile a predicate as a Numba vectorized ufunc
%  Options:
%    - signature(Sig): Type signature like 'float64(float64, float64)'
%    - target(cpu/parallel/cuda): Execution target
compile_numba_vectorized(Pred/Arity, Options, Code) :-
    % Get base Python code
    compile_predicate_to_python(Pred/Arity, Options, BaseCode),

    % Build vectorize decorator
    (   member(signature(Sig), Options)
    ->  SigPart = Sig
    ;   SigPart = ""
    ),

    (   member(target(Target), Options)
    ->  true
    ;   Target = cpu
    ),

    (   SigPart = ""
    ->  format(string(Decorator), "@vectorize(target='~w')~n", [Target])
    ;   format(string(Decorator), "@vectorize(['~w'], target='~w')~n", [SigPart, Target])
    ),

    generate_numba_header(Header),
    format(string(Code), "~w~w~w", [Header, Decorator, BaseCode]).

%% compile_numba_parallel(+Pred/Arity, +Options, -Code)
%  Compile with parallel loop support using prange
compile_numba_parallel(Pred/Arity, Options, Code) :-
    compile_numba_function(Pred/Arity, [parallel(true)|Options], Code).

%% compile_numba_cuda(+Pred/Arity, +Options, -Code)
%  Compile for CUDA GPU execution (requires numba.cuda)
compile_numba_cuda(Pred/Arity, Options, Code) :-
    compile_predicate_to_python(Pred/Arity, Options, BaseCode),

    Header = "from numba import cuda
import numpy as np
import math

@cuda.jit
",
    format(string(Code), "~w~w", [Header, BaseCode]).

%% test_numba_target/0
%  Test Numba target code generation
test_numba_target :-
    format('Testing Numba target...~n'),

    % Test decorator building
    build_numba_decorator([], Dec1),
    format('Default decorator: ~w', [Dec1]),

    build_numba_decorator([cache(true), parallel(true)], Dec2),
    format('With options: ~w', [Dec2]),

    format('Numba target tests passed.~n').
