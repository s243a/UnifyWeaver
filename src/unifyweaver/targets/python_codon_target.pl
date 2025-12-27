% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% python_codon_target.pl - Codon Target for Python
%
% Generates Codon-compatible Python code for high-performance native compilation.
% Codon compiles a subset of Python directly to native machine code via LLVM.
%
% License: BSL 1.1 (Commercial use may require license, Apache 2.0 after 3 years)
%
% Features:
% - Near-C++ performance for numerical code
% - Native LLVM compilation (no interpreter overhead)
% - @codon.jit for JIT compilation within regular Python
% - GPU support via @par decorators
%
% Limitations:
% - Subset of Python (no dynamic features)
% - Limited library support
% - Static typing required for best performance
%
% Example:
%   ?- compile_codon_function(factorial/1, [], Code).
%   ?- generate_codon_build_command(my_program, Cmd).

:- module(python_codon_target, [
    compile_codon_function/3,
    compile_codon_module/3,
    compile_codon_parallel/3,
    generate_codon_header/1,
    generate_codon_build_command/2,
    codon_type/2,
    init_codon_target/0
]).

:- use_module(python_target).
:- use_module('../core/binding_registry').
:- use_module('../core/component_registry').
:- use_module('python_runtime/custom_codon', []).

%% init_codon_target
%  Initialize Codon target
init_codon_target :-
    init_python_target.

%% generate_codon_header(-Header)
%  Generate Codon-specific header
generate_codon_header(Header) :-
    Header = "# Codon-compatible Python code
# Compile with: codon build -release program.py
# Or run directly: codon run program.py

from typing import List, Tuple, Optional
import math

".

%% codon_type(+PrologType, -CodonType)
%  Map Prolog types to Codon types
codon_type(integer, 'int').
codon_type(float, 'float').
codon_type(number, 'float').
codon_type(boolean, 'bool').
codon_type(string, 'str').
codon_type(list(integer), 'List[int]').
codon_type(list(float), 'List[float]').
codon_type(list(string), 'List[str]').
codon_type(tuple(Types), TupleType) :-
    maplist(codon_type, Types, CodonTypes),
    atomic_list_concat(CodonTypes, ', ', TypesStr),
    format(atom(TupleType), 'Tuple[~w]', [TypesStr]).
codon_type(optional(Type), OptType) :-
    codon_type(Type, CodonType),
    format(atom(OptType), 'Optional[~w]', [CodonType]).

%% compile_codon_function(+Pred/Arity, +Options, -Code)
%  Compile a predicate to Codon-compatible Python
%  Options:
%    - types(ArgTypes): Argument type annotations
%    - return_type(Type): Return type annotation
%    - jit(true/false): Use @codon.jit decorator for hybrid Python
compile_codon_function(Pred/Arity, Options, Code) :-
    % Get base Python code
    compile_predicate_to_python(Pred/Arity, Options, BaseCode),

    % Build type annotations
    functor(Pred, Name, _),
    (   member(types(ArgTypes), Options)
    ->  build_codon_args(ArgTypes, TypedArgs)
    ;   generate_default_arg_names(Arity, ArgNames),
        maplist(wrap_untyped_codon, ArgNames, TypedArgs)
    ),
    atomic_list_concat(TypedArgs, ', ', ArgsStr),

    % Return type
    (   member(return_type(RetType), Options),
        codon_type(RetType, CodonRetType)
    ->  format(string(RetAnnotation), " -> ~w", [CodonRetType])
    ;   RetAnnotation = ""
    ),

    % JIT decorator for hybrid mode
    (   member(jit(true), Options)
    ->  JitDecorator = "@codon.jit\n"
    ;   JitDecorator = ""
    ),

    generate_codon_header(Header),
    format(string(Code),
"~w~wdef ~w(~w)~w:
    # Generated from Prolog predicate ~w/~w
~w",
        [Header, JitDecorator, Name, ArgsStr, RetAnnotation,
         Name, Arity, BaseCode]).

%% build_codon_args(+Types, -TypedArgs)
build_codon_args(Types, TypedArgs) :-
    build_codon_args(Types, 0, TypedArgs).

build_codon_args([], _, []).
build_codon_args([Type|Types], N, [TypedArg|Rest]) :-
    codon_type(Type, CodonType),
    format(atom(TypedArg), "arg~w: ~w", [N, CodonType]),
    N1 is N + 1,
    build_codon_args(Types, N1, Rest).

wrap_untyped_codon(Name, Name).

%% compile_codon_module(+ModuleName, +Options, -Code)
%  Compile multiple predicates into a Codon module
compile_codon_module(ModuleName, Options, Code) :-
    generate_codon_header(Header),

    (   member(predicates(Preds), Options)
    ->  maplist(compile_pred_for_codon(Options), Preds, PredCodes),
        atomic_list_concat(PredCodes, '\n\n', PredsCode)
    ;   PredsCode = ""
    ),

    % Main block
    (   member(main(MainPred), Options)
    ->  format(string(MainBlock),
"
if __name__ == '__main__':
    ~w()
", [MainPred])
    ;   MainBlock = ""
    ),

    format(string(Code),
"# Module: ~w
~w~w~w", [ModuleName, Header, PredsCode, MainBlock]).

compile_pred_for_codon(Options, Pred/Arity, Code) :-
    compile_codon_function(Pred/Arity, Options, Code).

%% compile_codon_parallel(+Pred/Arity, +Options, -Code)
%  Compile with Codon's parallel execution support
%  Codon supports @par decorator for GPU/parallel execution
compile_codon_parallel(Pred/Arity, Options, Code) :-
    compile_predicate_to_python(Pred/Arity, Options, BaseCode),

    functor(Pred, Name, _),

    generate_codon_header(Header),
    format(string(Code),
"~w
# Parallel execution with Codon's @par
# Uses OpenMP-style parallelism

def ~w_parallel(data: List[int]) -> List[int]:
    \"\"\"Parallel version of ~w\"\"\"
    result = [0] * len(data)

    @par(schedule='dynamic', chunk_size=64, num_threads=8)
    for i in range(len(data)):
        result[i] = process_item(data[i])

    return result

~w
", [Header, Name, Name, BaseCode]).

%% generate_codon_build_command(+FileName, -Command)
%  Generate the Codon build command
generate_codon_build_command(FileName, Command) :-
    format(string(Command),
"# Build commands for Codon

# Debug build
codon build ~w.py

# Release build (optimized)
codon build -release ~w.py

# Run directly (JIT)
codon run ~w.py

# Build shared library
codon build -release -lib ~w.py

# With specific optimizations
codon build -release -O3 ~w.py
", [FileName, FileName, FileName, FileName, FileName]).

%% test_codon_target/0
%  Test Codon target code generation
test_codon_target :-
    format('Testing Codon target...~n'),

    % Test type mapping
    codon_type(integer, IntType),
    format('Integer type: ~w~n', [IntType]),

    codon_type(list(float), ListType),
    format('Float list type: ~w~n', [ListType]),

    % Test build command
    generate_codon_build_command(test_program, Cmd),
    format('Build commands:~n~w~n', [Cmd]),

    format('Codon target tests passed.~n').
