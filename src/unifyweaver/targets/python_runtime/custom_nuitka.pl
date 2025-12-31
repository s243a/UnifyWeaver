% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% custom_nuitka.pl - Custom Nuitka Component Type
%
% Allows injecting Python code optimized for Nuitka compilation.
% Nuitka compiles full Python to C, so this focuses on structure and hints.
%
% Example:
%   declare_component(source, my_handler, custom_nuitka, [
%       code("return process(input)"),
%       framework(flask),
%       standalone(true)
%   ]).

:- module(custom_nuitka, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    compile_component/4
]).

:- use_module('../../core/component_registry').

%% type_info(-Info)
type_info(info(
    name('Custom Nuitka Component'),
    version('1.0.0'),
    description('Injects Python code optimized for Nuitka compilation')
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
    throw(error(runtime_invocation_not_supported(custom_nuitka))).

%% compile_component(+Name, +Config, +Options, -Code)
compile_component(Name, Config, _Options, Code) :-
    member(code(Body), Config),

    % Collect type hints if provided
    (   member(arg_types(ArgTypes), Config)
    ->  build_type_hints(ArgTypes, TypeHints),
        atomic_list_concat(TypeHints, ', ', ArgsStr)
    ;   ArgsStr = "input: Any"
    ),

    % Return type hint
    (   member(return_type(RetType), Config)
    ->  format(string(RetHint), " -> ~w", [RetType])
    ;   RetHint = " -> Any"
    ),

    % Framework imports
    (   member(framework(flask), Config)
    ->  FrameworkImport = "from flask import Flask, request, jsonify\n"
    ;   member(framework(fastapi), Config)
    ->  FrameworkImport = "from fastapi import FastAPI\nfrom pydantic import BaseModel\n"
    ;   FrameworkImport = ""
    ),

    atom_string(Name, NameStr),
    format(string(Code),
"# Nuitka-optimized Component: ~w
# Compile with: nuitka --module this_file.py

from typing import List, Dict, Optional, Any, Tuple
~w

def comp_~w(~w)~w:
    \"\"\"
    Nuitka-compiled component: ~w

    Build commands:
        nuitka --module this_file.py
        nuitka --standalone this_file.py
        nuitka --onefile this_file.py
    \"\"\"
~w
", [NameStr, FrameworkImport, NameStr, ArgsStr, RetHint, NameStr, Body]).

%% build_type_hints(+Types, -TypeHints)
build_type_hints(Types, TypeHints) :-
    build_type_hints(Types, 0, TypeHints).

build_type_hints([], _, []).
build_type_hints([Type|Types], N, [TypeHint|Rest]) :-
    format(atom(TypeHint), "arg~w: ~w", [N, Type]),
    N1 is N + 1,
    build_type_hints(Types, N1, Rest).

%% Register this component type
:- initialization((
    register_component_type(source, custom_nuitka, custom_nuitka, [
        description("Custom Nuitka-optimized Code")
    ])
), now).
