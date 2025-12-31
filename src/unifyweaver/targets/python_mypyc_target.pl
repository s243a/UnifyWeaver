% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% python_mypyc_target.pl - mypyc Target for Python
%
% Generates Python code with type annotations for mypyc compilation.
% mypyc compiles type-annotated Python to C extensions.
%
% License: MIT (mypyc/mypy)
%
% Features:
% - Uses standard Python type annotations
% - Compiles to C extension modules
% - Full Python compatibility for non-compiled code
% - Part of the mypy ecosystem
%
% Example:
%   ?- compile_mypyc_function(factorial/1, [types([integer]), return_type(integer)], Code).
%   ?- generate_mypyc_build_command(my_module, Cmd).

:- module(python_mypyc_target, [
    compile_mypyc_function/3,
    compile_mypyc_module/3,
    compile_mypyc_class/3,
    generate_mypyc_header/1,
    generate_mypyc_build_command/2,
    generate_mypyc_setup_py/2,
    mypyc_type/2,
    init_mypyc_target/0
]).

:- use_module(python_target).
:- use_module('../core/binding_registry').
:- use_module('../core/component_registry').
:- use_module('python_runtime/custom_mypyc', []).

%% init_mypyc_target
%  Initialize mypyc target
init_mypyc_target :-
    init_python_target.

%% generate_mypyc_header(-Header)
%  Generate mypyc-optimized header with type imports
generate_mypyc_header(Header) :-
    Header = "#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
mypyc-compatible module with full type annotations.
Compile with: mypyc module.py
\"\"\"

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Final, Callable
from typing import TypeVar, Generic, Protocol
import math

# Type aliases for common patterns
IntList = List[int]
FloatList = List[float]
StringList = List[str]

".

%% mypyc_type(+PrologType, -MypycType)
%  Map Prolog types to Python/mypyc type annotations
mypyc_type(integer, 'int').
mypyc_type(float, 'float').
mypyc_type(number, 'float').
mypyc_type(boolean, 'bool').
mypyc_type(string, 'str').
mypyc_type(atom, 'str').
mypyc_type(any, 'Any').
mypyc_type(none, 'None').
mypyc_type(list(Type), ListType) :-
    mypyc_type(Type, InnerType),
    format(atom(ListType), 'List[~w]', [InnerType]).
mypyc_type(dict(KeyType, ValueType), DictType) :-
    mypyc_type(KeyType, KeyT),
    mypyc_type(ValueType, ValueT),
    format(atom(DictType), 'Dict[~w, ~w]', [KeyT, ValueT]).
mypyc_type(tuple(Types), TupleType) :-
    maplist(mypyc_type, Types, MypycTypes),
    atomic_list_concat(MypycTypes, ', ', TypesStr),
    format(atom(TupleType), 'Tuple[~w]', [TypesStr]).
mypyc_type(optional(Type), OptType) :-
    mypyc_type(Type, MypycType),
    format(atom(OptType), 'Optional[~w]', [MypycType]).
mypyc_type(callable(ArgTypes, RetType), CallableType) :-
    maplist(mypyc_type, ArgTypes, MypycArgTypes),
    mypyc_type(RetType, MypycRetType),
    atomic_list_concat(MypycArgTypes, ', ', ArgsStr),
    format(atom(CallableType), 'Callable[[~w], ~w]', [ArgsStr, MypycRetType]).
mypyc_type(final(Type), FinalType) :-
    mypyc_type(Type, MypycType),
    format(atom(FinalType), 'Final[~w]', [MypycType]).

%% compile_mypyc_function(+Pred/Arity, +Options, -Code)
%  Compile a predicate to fully-typed Python for mypyc
%  Options:
%    - types(ArgTypes): List of argument types
%    - return_type(Type): Return type
%    - pure(true/false): Mark as pure function (no side effects)
compile_mypyc_function(Pred/Arity, Options, Code) :-
    % Get base Python code
    compile_predicate_to_python(Pred/Arity, Options, BaseCode),

    functor(Pred, Name, _),

    % Build typed arguments
    (   member(types(ArgTypes), Options)
    ->  build_mypyc_args(ArgTypes, TypedArgs)
    ;   generate_default_arg_names(Arity, ArgNames),
        maplist(wrap_any_type, ArgNames, TypedArgs)
    ),
    atomic_list_concat(TypedArgs, ', ', ArgsStr),

    % Return type
    (   member(return_type(RetType), Options),
        mypyc_type(RetType, MypycRetType)
    ->  format(string(RetAnnotation), " -> ~w", [MypycRetType])
    ;   RetAnnotation = " -> Any"
    ),

    generate_mypyc_header(Header),
    format(string(Code),
"~wdef ~w(~w)~w:
    \"\"\"Generated from Prolog predicate ~w/~w.\"\"\"
~w",
        [Header, Name, ArgsStr, RetAnnotation, Name, Arity, BaseCode]).

%% build_mypyc_args(+Types, -TypedArgs)
build_mypyc_args(Types, TypedArgs) :-
    build_mypyc_args(Types, 0, TypedArgs).

build_mypyc_args([], _, []).
build_mypyc_args([Type|Types], N, [TypedArg|Rest]) :-
    mypyc_type(Type, MypycType),
    format(atom(TypedArg), "arg~w: ~w", [N, MypycType]),
    N1 is N + 1,
    build_mypyc_args(Types, N1, Rest).

wrap_any_type(Name, TypedArg) :-
    format(atom(TypedArg), "~w: Any", [Name]).

%% compile_mypyc_module(+ModuleName, +Options, -Code)
%  Compile multiple predicates into a mypyc module
compile_mypyc_module(ModuleName, Options, Code) :-
    generate_mypyc_header(Header),

    (   member(predicates(Preds), Options)
    ->  maplist(compile_pred_for_mypyc(Options), Preds, PredCodes),
        atomic_list_concat(PredCodes, '\n\n', PredsCode)
    ;   PredsCode = ""
    ),

    format(string(Code),
"# Module: ~w
~w~w", [ModuleName, Header, PredsCode]).

compile_pred_for_mypyc(Options, Pred/Arity, Code) :-
    compile_mypyc_function(Pred/Arity, Options, Code).

%% compile_mypyc_class(+ClassName, +Options, -Code)
%  Compile a typed class for mypyc
%  Options:
%    - fields(List): field(Name, Type) specifications
%    - methods(List): method(Name, Args, RetType) specifications
compile_mypyc_class(ClassName, Options, Code) :-
    % Collect fields
    (   member(fields(Fields), Options)
    ->  maplist(format_field, Fields, FieldLines),
        atomic_list_concat(FieldLines, '\n    ', FieldsCode)
    ;   FieldsCode = "pass"
    ),

    % Generate __init__
    (   member(fields(Fields), Options)
    ->  generate_init(Fields, InitCode)
    ;   InitCode = "    def __init__(self) -> None:\n        pass"
    ),

    % Collect methods
    (   member(methods(Methods), Options)
    ->  maplist(format_method, Methods, MethodCodes),
        atomic_list_concat(MethodCodes, '\n\n', MethodsCode)
    ;   MethodsCode = ""
    ),

    generate_mypyc_header(Header),
    format(string(Code),
"~wclass ~w:
    \"\"\"Typed class for mypyc compilation.\"\"\"

    # Typed attributes (mypyc requires these)
    ~w

~w

~w
", [Header, ClassName, FieldsCode, InitCode, MethodsCode]).

format_field(field(Name, Type), FieldLine) :-
    mypyc_type(Type, MypycType),
    format(string(FieldLine), "~w: ~w", [Name, MypycType]).

generate_init(Fields, InitCode) :-
    maplist(field_to_param, Fields, Params),
    atomic_list_concat(Params, ', ', ParamsStr),
    maplist(field_to_assign, Fields, Assigns),
    atomic_list_concat(Assigns, '\n        ', AssignsStr),
    format(string(InitCode),
"    def __init__(self, ~w) -> None:
        ~w", [ParamsStr, AssignsStr]).

field_to_param(field(Name, Type), Param) :-
    mypyc_type(Type, MypycType),
    format(atom(Param), "~w: ~w", [Name, MypycType]).

field_to_assign(field(Name, _), Assign) :-
    format(string(Assign), "self.~w = ~w", [Name, Name]).

format_method(method(Name, Args, RetType), MethodCode) :-
    maplist(format_method_arg, Args, ArgStrs),
    atomic_list_concat(ArgStrs, ', ', ArgsStr),
    mypyc_type(RetType, MypycRetType),
    format(string(MethodCode),
"    def ~w(self, ~w) -> ~w:
        raise NotImplementedError()", [Name, ArgsStr, MypycRetType]).

format_method_arg(arg(Name, Type), ArgStr) :-
    mypyc_type(Type, MypycType),
    format(atom(ArgStr), "~w: ~w", [Name, MypycType]).

%% generate_mypyc_build_command(+ModuleName, -Command)
%  Generate the mypyc build command
generate_mypyc_build_command(ModuleName, Command) :-
    format(string(Command),
"# mypyc build commands

# Simple compilation
mypyc ~w.py

# With optimization level
mypyc --opt-level 3 ~w.py

# Verbose output
mypyc -v ~w.py

# Check types first (recommended)
mypy --strict ~w.py && mypyc ~w.py

# Using setup.py (for distribution)
python setup.py build_ext --inplace
", [ModuleName, ModuleName, ModuleName, ModuleName, ModuleName]).

%% generate_mypyc_setup_py(+ModuleName, -SetupCode)
%  Generate setup.py for mypyc compilation
generate_mypyc_setup_py(ModuleName, SetupCode) :-
    format(string(SetupCode),
"from setuptools import setup
from mypyc.build import mypycify

setup(
    name='~w',
    version='0.1.0',
    packages=['~w'],
    ext_modules=mypycify([
        '~w.py',
    ]),
)
", [ModuleName, ModuleName, ModuleName]).

%% test_mypyc_target/0
%  Test mypyc target code generation
test_mypyc_target :-
    format('Testing mypyc target...~n'),

    % Test type mapping
    mypyc_type(integer, IntType),
    format('Integer type: ~w~n', [IntType]),

    mypyc_type(list(float), ListType),
    format('Float list type: ~w~n', [ListType]),

    mypyc_type(dict(string, integer), DictType),
    format('String->Int dict type: ~w~n', [DictType]),

    % Test class generation
    compile_mypyc_class(person, [
        fields([field(name, string), field(age, integer)]),
        methods([method(greet, [], string)])
    ], ClassCode),
    format('Class code:~n~w~n', [ClassCode]),

    format('mypyc target tests passed.~n').
