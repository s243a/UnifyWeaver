% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% python_cython_target.pl - Cython Target for Python
%
% Generates Cython code (.pyx files) with static type declarations
% for C-level performance. Requires compilation step via Cython compiler.
%
% License: Apache 2.0 (Cython itself)
%
% Features:
% - cdef for C-level variables and functions
% - cpdef for Python-callable C functions
% - Memory views for efficient array access
% - nogil for releasing GIL
%
% Example:
%   ?- compile_cython_function(factorial/1, [], Code).
%   ?- generate_setup_py([factorial], SetupCode).

:- module(python_cython_target, [
    compile_cython_function/3,
    compile_cython_module/3,
    generate_cython_header/1,
    generate_setup_py/2,
    generate_pyproject_toml/2,
    cython_type/2,
    init_cython_target/0
]).

:- use_module(python_target).
:- use_module('../core/binding_registry').
:- use_module('../core/component_registry').
:- use_module('python_runtime/custom_cython', []).

%% init_cython_target
%  Initialize Cython target
init_cython_target :-
    init_python_target.

%% generate_cython_header(-Header)
%  Generate Cython-specific header with libc imports
generate_cython_header(Header) :-
    Header = "# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport sin, cos, exp, sqrt, log, pow
from libc.stdlib cimport malloc, free
cimport cython

import numpy as np
cimport numpy as np

".

%% cython_type(+PrologType, -CythonType)
%  Map Prolog types to Cython types
cython_type(integer, 'long long').
cython_type(int, 'int').
cython_type(float, 'double').
cython_type(number, 'double').
cython_type(boolean, 'bint').
cython_type(string, 'str').
cython_type(list(integer), 'long long[:]').
cython_type(list(float), 'double[:]').
cython_type(array(integer), 'np.ndarray[np.int64_t, ndim=2]').
cython_type(array(float), 'np.ndarray[np.float64_t, ndim=2]').
cython_type(pointer, 'void*').

%% compile_cython_function(+Pred/Arity, +Options, -Code)
%  Compile a predicate to Cython function
%  Options:
%    - mode(cpdef/cdef/def): Function visibility (default: cpdef)
%    - types(ArgTypes): List of argument types
%    - return_type(Type): Return type
%    - nogil(true/false): Release GIL (default: false)
%    - inline(true/false): Inline the function (default: false)
compile_cython_function(Pred/Arity, Options, Code) :-
    % Get base Python code
    compile_predicate_to_python(Pred/Arity, Options, BaseCode),

    % Determine function mode
    (   member(mode(Mode), Options)
    ->  true
    ;   Mode = cpdef
    ),

    % Get return type
    (   member(return_type(RetType), Options),
        cython_type(RetType, CythonRetType)
    ->  true
    ;   CythonRetType = 'object'
    ),

    % Build function signature with types
    functor(Pred, Name, _),
    (   member(types(ArgTypes), Options)
    ->  build_typed_args(ArgTypes, TypedArgs)
    ;   generate_default_arg_names(Arity, ArgNames),
        maplist(wrap_untyped, ArgNames, TypedArgs)
    ),
    atomic_list_concat(TypedArgs, ', ', ArgsStr),

    % Inline decorator if requested
    (   member(inline(true), Options)
    ->  Inline = "@cython.inline\n"
    ;   Inline = ""
    ),

    % Nogil suffix
    (   member(nogil(true), Options)
    ->  NogilSuffix = " nogil"
    ;   NogilSuffix = ""
    ),

    % Replace def with cpdef/cdef and add types
    generate_cython_header(Header),
    format(string(Code),
"~w~w~w ~w ~w(~w)~w:
    # Generated from Prolog predicate ~w/~w
~w",
        [Header, Inline, Mode, CythonRetType, Name, ArgsStr, NogilSuffix,
         Name, Arity, BaseCode]).

%% build_typed_args(+Types, -TypedArgs)
%  Build typed argument list
build_typed_args(Types, TypedArgs) :-
    build_typed_args(Types, 0, TypedArgs).

build_typed_args([], _, []).
build_typed_args([Type|Types], N, [TypedArg|Rest]) :-
    cython_type(Type, CythonType),
    format(atom(TypedArg), "~w arg~w", [CythonType, N]),
    N1 is N + 1,
    build_typed_args(Types, N1, Rest).

%% wrap_untyped(+Name, -Wrapped)
wrap_untyped(Name, Name).  % No type annotation

%% compile_cython_module(+Functions, +Options, -Code)
%  Compile multiple functions into a Cython module
compile_cython_module(Functions, Options, Code) :-
    generate_cython_header(Header),
    maplist(compile_single_cython(Options), Functions, FuncCodes),
    atomic_list_concat([Header|FuncCodes], '\n\n', Code).

compile_single_cython(Options, Pred/Arity, Code) :-
    compile_cython_function(Pred/Arity, Options, Code).

%% generate_setup_py(+ModuleName, -SetupCode)
%  Generate setup.py for building the Cython extension
generate_setup_py(ModuleName, SetupCode) :-
    format(string(SetupCode),
"from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='~w',
    ext_modules=cythonize(
        '~w.pyx',
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False
        }
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
", [ModuleName, ModuleName]).

%% generate_pyproject_toml(+ModuleName, -TomlCode)
%  Generate pyproject.toml for modern Python packaging
generate_pyproject_toml(ModuleName, TomlCode) :-
    format(string(TomlCode),
"[build-system]
requires = [\"setuptools>=61.0\", \"cython>=3.0\", \"numpy\"]
build-backend = \"setuptools.build_meta\"

[project]
name = \"~w\"
version = \"0.1.0\"
description = \"Generated Cython module from UnifyWeaver\"
requires-python = \">=3.8\"
dependencies = [\"numpy\"]

[tool.cython]
language_level = 3
", [ModuleName]).

%% test_cython_target/0
%  Test Cython target code generation
test_cython_target :-
    format('Testing Cython target...~n'),

    % Test type mapping
    cython_type(integer, IntType),
    format('Integer type: ~w~n', [IntType]),

    cython_type(array(float), ArrayType),
    format('Float array type: ~w~n', [ArrayType]),

    % Test setup.py generation
    generate_setup_py(test_module, SetupCode),
    format('Setup.py:~n~w~n', [SetupCode]),

    format('Cython target tests passed.~n').
