% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% awk_bindings.pl - AWK-specific bindings
%
% This module defines bindings for AWK target language features.
%
% Categories:
%   - Built-ins (length, print)
%   - String Operations (tolower, toupper, substr, index, split)
%   - Math Operations (sqrt, sin, cos, exp, log, int)
%
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md

:- module(awk_bindings, [
    init_awk_bindings/0,
    awk_binding/5,              % Convenience: awk_binding(Pred, TargetName, Inputs, Outputs, Options)
    test_awk_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_awk_bindings
%
%  Initialize all AWK bindings. Call this before using the compiler.
%
init_awk_bindings :-
    register_builtin_bindings,
    register_string_bindings,
    register_math_bindings.

% ============================================================================
% CONVENIENCE PREDICATE
% ============================================================================

%% awk_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query AWK bindings with reduced arity (Target=awk implied).
%
awk_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(awk, Pred, TargetName, Inputs, Outputs, Options).

% ============================================================================
% CORE BUILT-IN BINDINGS
% ============================================================================

register_builtin_bindings :-
    % -------------------------------------------
    % Built-ins
    % -------------------------------------------

    % length - length()
    declare_binding(awk, length/2, 'length(~w)',
        [string], [int],
        [pure, deterministic, total, pattern(function)]),

    % print - print
    declare_binding(awk, print/1, 'print ~w',
        [string], [],
        [effect(io), deterministic, total, pattern(statement)]).

% ============================================================================
% STRING OPERATION BINDINGS
% ============================================================================

register_string_bindings :-
    % -------------------------------------------
    % String Functions
    % -------------------------------------------

    % string_lower - tolower()
    declare_binding(awk, string_lower/2, 'tolower(~w)',
        [string], [string],
        [pure, deterministic, total, pattern(function)]),

    % string_upper - toupper()
    declare_binding(awk, string_upper/2, 'toupper(~w)',
        [string], [string],
        [pure, deterministic, total, pattern(function)]),

    % string_index - index()
    declare_binding(awk, string_index/3, 'index(~w, ~w)',
        [string, string], [int],
        [pure, deterministic, total, pattern(function)]),

    % string_contains - index() > 0
    declare_binding(awk, string_contains/2, 'index(~w, ~w) > 0',
        [string, string], [bool],
        [pure, deterministic, total, pattern(expression)]),

    % string_substr - substr()
    declare_binding(awk, string_substr/4, 'substr(~w, ~w, ~w)',
        [string, int, int], [string],
        [pure, deterministic, total, pattern(function)]).

% ============================================================================
% MATH OPERATION BINDINGS
% ============================================================================

register_math_bindings :-
    % -------------------------------------------
    % Math Functions
    % -------------------------------------------

    % sqrt - sqrt()
    declare_binding(awk, sqrt/2, 'sqrt(~w)',
        [number], [float],
        [pure, deterministic, total, pattern(function)]),

    % sin - sin()
    declare_binding(awk, sin/2, 'sin(~w)',
        [number], [float],
        [pure, deterministic, total, pattern(function)]),

    % cos - cos()
    declare_binding(awk, cos/2, 'cos(~w)',
        [number], [float],
        [pure, deterministic, total, pattern(function)]),

    % exp - exp()
    declare_binding(awk, exp/2, 'exp(~w)',
        [number], [float],
        [pure, deterministic, total, pattern(function)]),

    % log - log()
    declare_binding(awk, log/2, 'log(~w)',
        [number], [float],
        [pure, deterministic, total, pattern(function)]),

    % int - int()
    declare_binding(awk, to_int/2, 'int(~w)',
        [number], [int],
        [pure, deterministic, total, pattern(function)]),

    % rand - rand()
    declare_binding(awk, random/1, 'rand()',
        [], [float],
        [pure, deterministic, total, pattern(function)]).

% ============================================================================
% TESTS
% ============================================================================

test_awk_bindings :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  AWK Bindings Tests                   ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Initialize bindings
    format('[Test 1] Initializing AWK bindings~n', []),
    init_awk_bindings,
    format('[✓] AWK bindings initialized~n~n', []),

    % Test string bindings
    format('[Test 2] Checking string bindings~n', []),
    (   awk_binding(string_lower/2, 'tolower(~w)', _, _, _)
    ->  format('[✓] string_lower/2 binding exists~n', [])
    ;   format('[✗] string_lower/2 binding missing~n', []), fail
    ),

    % Test math bindings
    format('~n[Test 3] Checking math bindings~n', []),
    (   awk_binding(sqrt/2, 'sqrt(~w)', _, _, _)
    ->  format('[✓] sqrt/2 binding exists~n', [])
    ;   format('[✗] sqrt/2 binding missing~n', []), fail
    ),

    % Count total bindings
    format('~n[Test 4] Counting total bindings~n', []),
    findall(P, awk_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('[✓] Total AWK bindings: ~w~n', [Count]),

    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  All AWK Bindings Tests Passed        ║~n', []),
    format('╚════════════════════════════════════════╝~n', []).
