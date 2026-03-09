% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% r_bindings.pl - R-specific bindings
%
% This module defines bindings for R target language features.
%
% Categories:
%   - Core Built-ins
%   - Math Operations
%   - String Operations
%   - Type Conversion Operations
%   - Vector/List Operations
%   - File I/O Operations
%   - Data Frame Operations (Vectorized operations)
%
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md

:- module(r_bindings, [
    init_r_bindings/0,
    r_binding/5,               % Convenience: r_binding(Pred, TargetName, Inputs, Outputs, Options)
    test_r_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_r_bindings
%
%  Initialize all R bindings. Call this before using the compiler.
%
init_r_bindings :-
    register_builtin_bindings,
    register_math_bindings,
    register_string_bindings,
    register_type_conversion_bindings,
    register_vector_list_bindings,
    register_file_io_bindings,
    register_dataframe_bindings.

% ============================================================================
% CONVENIENCE PREDICATES
% ============================================================================

%% r_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query R bindings with reduced arity (Target=r implied).
%
r_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(r, Pred, TargetName, Inputs, Outputs, Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

:- multifile user:term_expansion/2.

user:term_expansion(
    (:- r_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(r, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% CORE BUILT-IN BINDINGS
% ============================================================================

register_builtin_bindings :-
    % -------------------------------------------
    % Length and Vectors
    % -------------------------------------------
    declare_binding(r, length/2, 'length',
        [any], [int],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Conditionals / Tests
    % -------------------------------------------
    declare_binding(r, true/0, 'TRUE',
        [], [],
        [pure, deterministic, total, pattern(command)]),

    declare_binding(r, fail/0, 'FALSE',
        [], [],
        [pure, deterministic, total, pattern(command)]),

    % -------------------------------------------
    % I/O
    % -------------------------------------------
    declare_binding(r, print/1, 'print',
        [any], [],
        [effect(io), deterministic, total]),

    declare_binding(r, cat/1, 'cat',
        [string], [],
        [effect(io), deterministic, total]).

% ============================================================================
% MATH BINDINGS
% ============================================================================

register_math_bindings :-
    declare_binding(r, sum/2, 'sum',
        [list(number)], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, mean/2, 'mean',
        [list(number)], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, min/2, 'min',
        [list(number)], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, max/2, 'max',
        [list(number)], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, abs/2, 'abs',
        [number], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, sqrt/2, 'sqrt',
        [number], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, floor/2, 'floor',
        [number], [int],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, ceiling/2, 'ceiling',
        [number], [int],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, round/2, 'round',
        [number], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, log/2, 'log',
        [number], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, log10/2, 'log10',
        [number], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, exp/2, 'exp',
        [number], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, sin/2, 'sin',
        [number], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, cos/2, 'cos',
        [number], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, tan/2, 'tan',
        [number], [number],
        [pure, deterministic, total, vectorized]).

% ============================================================================
% STRING BINDINGS
% ============================================================================

register_string_bindings :-
    declare_binding(r, string_concat/3, 'paste0',
        [string, string], [string],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, string_length/2, 'nchar',
        [string], [int],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, string_replace/4, 'gsub',
        [string, string, string], [string],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, string_sub/4, 'sub',
        [string, string, string], [string],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, string_upper/2, 'toupper',
        [string], [string],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, string_lower/2, 'tolower',
        [string], [string],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, string_trim/2, 'trimws',
        [string], [string],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, string_substr/4, 'substr',
        [string, int, int], [string],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, string_split/3, 'strsplit',
        [string, string], [list(string)],
        [pure, deterministic, total]),

    declare_binding(r, string_format/3, 'sprintf',
        [string, any], [string],
        [pure, deterministic, total]),

    declare_binding(r, string_grep/3, 'grep',
        [string, list(string)], [list(int)],
        [pure, deterministic, total]),

    declare_binding(r, string_grepl/3, 'grepl',
        [string, string], [logical],
        [pure, deterministic, total, vectorized]).

% ============================================================================
% TYPE CONVERSION BINDINGS
% ============================================================================

register_type_conversion_bindings :-
    declare_binding(r, to_numeric/2, 'as.numeric',
        [any], [number],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, to_integer/2, 'as.integer',
        [any], [int],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, to_string/2, 'as.character',
        [any], [string],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, to_logical/2, 'as.logical',
        [any], [logical],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, is_numeric/1, 'is.numeric',
        [any], [],
        [pure, deterministic, total, pattern(command)]),

    declare_binding(r, is_character/1, 'is.character',
        [any], [],
        [pure, deterministic, total, pattern(command)]),

    declare_binding(r, is_logical/1, 'is.logical',
        [any], [],
        [pure, deterministic, total, pattern(command)]).

% ============================================================================
% VECTOR/LIST BINDINGS
% ============================================================================

register_vector_list_bindings :-
    declare_binding(r, append/3, 'c',
        [any, any], [any],
        [pure, deterministic, total]),

    declare_binding(r, reverse/2, 'rev',
        [list(any)], [list(any)],
        [pure, deterministic, total]),

    declare_binding(r, sort/2, 'sort',
        [list(any)], [list(any)],
        [pure, deterministic, total]),

    declare_binding(r, unique/2, 'unique',
        [list(any)], [list(any)],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, which/2, 'which',
        [list(logical)], [list(int)],
        [pure, deterministic, total]),

    declare_binding(r, seq/3, 'seq',
        [number, number], [list(number)],
        [pure, deterministic, total]),

    declare_binding(r, rep/3, 'rep',
        [any, int], [list(any)],
        [pure, deterministic, total]),

    declare_binding(r, head_n/3, 'head',
        [list(any), int], [list(any)],
        [pure, deterministic, total]),

    declare_binding(r, tail_n/3, 'tail',
        [list(any), int], [list(any)],
        [pure, deterministic, total]).

% ============================================================================
% FILE I/O BINDINGS
% ============================================================================

register_file_io_bindings :-
    declare_binding(r, file_exists/1, 'file.exists',
        [string], [],
        [effect(io), deterministic, total, pattern(command)]),

    declare_binding(r, file_path/3, 'file.path',
        [string, string], [string],
        [pure, deterministic, total]),

    declare_binding(r, dirname/2, 'dirname',
        [string], [string],
        [pure, deterministic, total]),

    declare_binding(r, basename/2, 'basename',
        [string], [string],
        [pure, deterministic, total]),

    declare_binding(r, read_lines/2, 'readLines',
        [string], [list(string)],
        [effect(io), deterministic, total]),

    declare_binding(r, write_lines/2, 'writeLines',
        [list(string), string], [],
        [effect(io), deterministic, total]),

    declare_binding(r, normalize_path/2, 'normalizePath',
        [string], [string],
        [pure, deterministic, total]).

% ============================================================================
% DATAFRAME BINDINGS (PIPELINES)
% ============================================================================

register_dataframe_bindings :-
    % Used for pipeline streaming in R via dplyr or native pipes
    declare_binding(r, filter/3, 'subset',  % Base R alternative to dplyr::filter
        [dataframe, expr], [dataframe],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, sort_by/3, 'order',
        [dataframe, string], [dataframe],
        [pure, deterministic, total, vectorized]),

    declare_binding(r, group_by/3, 'aggregate', % Base R or dplyr::group_by
        [dataframe, string], [dataframe],
        [pure, deterministic, total, vectorized]).

% ============================================================================
% TESTING
% ============================================================================

test_r_bindings :-
    format('~n=== R Bindings Tests ===~n~n'),
    test_r_binding('Core', sum/2, 'sum'),
    test_r_binding('Math', abs/2, 'abs'),
    test_r_binding('String', string_upper/2, 'toupper'),
    test_r_binding('TypeConv', to_numeric/2, 'as.numeric'),
    test_r_binding('Vector', reverse/2, 'rev'),
    test_r_binding('FileIO', dirname/2, 'dirname'),
    test_r_binding('DataFrame', filter/3, 'subset'),
    % Count total bindings
    aggregate_all(count, r_binding(_, _, _, _, _), Count),
    format('~n  Total R bindings: ~w~n', [Count]),
    format('~n=== Tests Complete ===~n').

test_r_binding(Category, Pred, ExpectedTarget) :-
    (   r_binding(Pred, TargetName, _, _, _),
        TargetName == ExpectedTarget
    ->  format('  PASS: [~w] ~w -> ~w~n', [Category, Pred, TargetName])
    ;   format('  FAIL: [~w] ~w not found~n', [Category, Pred])
    ).
