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
        [pure, deterministic, total, vectorized]).

% ============================================================================
% DATAFRAME BINDINGS (PIPELINES)
% ============================================================================

register_dataframe_bindings :-
    % Used for pipeline streaming in R via dplyr or native pipes
    declare_binding(r, filter/2, 'subset',  % Base R alternative to dplyr::filter
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
    (   r_binding(sum/2, TargetName, _, _, _),
        TargetName == 'sum'
    ->  format('  PASS: sum/2 binding found~n')
    ;   format('  FAIL: sum/2 binding not found~n')
    ),
    format('~n=== Tests Complete ===~n').
