% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% bash_bindings.pl - Bash-specific bindings
%
% This module defines bindings for Bash target language features.
%
% Categories:
%   - Built-ins (test, echo, printf)
%   - String Operations (Parameter expansion)
%   - Math Operations (Arithmetic expansion)
%   - File Operations (cat, ls, mkdir, etc.)
%
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md

:- module(bash_bindings, [
    init_bash_bindings/0,
    bash_binding/5,             % Convenience: bash_binding(Pred, TargetName, Inputs, Outputs, Options)
    test_bash_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_bash_bindings
%
%  Initialize all Bash bindings. Call this before using the compiler.
%
init_bash_bindings :-
    register_builtin_bindings,
    register_string_bindings,
    register_math_bindings,
    register_file_bindings.

% ============================================================================
% CONVENIENCE PREDICATE
% ============================================================================

%% bash_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query Bash bindings with reduced arity (Target=bash implied).
%
bash_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(bash, Pred, TargetName, Inputs, Outputs, Options).

% ============================================================================
% CORE BUILT-IN BINDINGS
% ============================================================================

register_builtin_bindings :-
    % -------------------------------------------
    % Output
    % -------------------------------------------

    % print - echo
    declare_binding(bash, print/1, 'echo "$~w"',
        [string], [],
        [effect(io), deterministic, total, pattern(command)]),

    % -------------------------------------------
    % Conditionals / Tests
    % -------------------------------------------

    % true - true
    declare_binding(bash, true/0, 'true',
        [], [],
        [pure, deterministic, total, pattern(command)]),

    % false - false
    declare_binding(bash, fail/0, 'false',
        [], [],
        [pure, deterministic, total, pattern(command)]).

% ============================================================================
% STRING OPERATION BINDINGS
% ============================================================================

register_string_bindings :-
    % -------------------------------------------
    % Parameter Expansion (Requires Var Name)
    % -------------------------------------------

    % string_length - ${#var}
    declare_binding(bash, string_length/2, '${#~w}',
        [string], [int],
        [pure, deterministic, total, pattern(expansion)]),

    % string_upper - ${var^^}
    declare_binding(bash, string_upper/2, '${~w^^}',
        [string], [string],
        [pure, deterministic, total, pattern(expansion)]),

    % string_lower - ${var,,}
    declare_binding(bash, string_lower/2, '${~w,,}',
        [string], [string],
        [pure, deterministic, total, pattern(expansion)]),

    % string_replace_all - ${var//pattern/repl}
    declare_binding(bash, string_replace_all/4, '${~w//~w/~w}',
        [string, string, string], [string],
        [pure, deterministic, total, pattern(expansion)]).

% ============================================================================
% MATH OPERATION BINDINGS
% ============================================================================

register_math_bindings :-
    % -------------------------------------------
    % Arithmetic Expansion $((...)) (Requires Values $)
    % -------------------------------------------

    % add
    declare_binding(bash, add/3, '$(( $~w + $~w ))',
        [int, int], [int],
        [pure, deterministic, total, pattern(arithmetic)]),

    % subtract
    declare_binding(bash, subtract/3, '$(( $~w - $~w ))',
        [int, int], [int],
        [pure, deterministic, total, pattern(arithmetic)]),

    % multiply
    declare_binding(bash, multiply/3, '$(( $~w * $~w ))',
        [int, int], [int],
        [pure, deterministic, total, pattern(arithmetic)]),

    % divide
    declare_binding(bash, divide/3, '$(( $~w / $~w ))',
        [int, int], [int],
        [pure, deterministic, total, pattern(arithmetic)]),

    % modulo
    declare_binding(bash, mod/3, '$(( $~w % $~w ))',
        [int, int], [int],
        [pure, deterministic, total, pattern(arithmetic)]).

% ============================================================================
% FILE OPERATION BINDINGS
% ============================================================================

register_file_bindings :-
    % -------------------------------------------
    % Test Commands (Requires Values $)
    % -------------------------------------------

    % file_exists - [[ -f file ]]
    declare_binding(bash, file_exists/1, '[[ -f "$~w" ]]',
        [string], [],
        [effect(io), deterministic, total, pattern(test)]),

    % dir_exists - [[ -d dir ]]
    declare_binding(bash, dir_exists/1, '[[ -d "$~w" ]]',
        [string], [],
        [effect(io), deterministic, total, pattern(test)]),

    % -------------------------------------------
    % File Commands (Requires Values $)
    % -------------------------------------------

    % read_file - cat
    declare_binding(bash, read_file/2, 'cat "$~w"',
        [string], [string],
        [effect(io), deterministic, total, pattern(command_substitution)]),

    % write_file - echo >
    declare_binding(bash, write_file/2, 'echo "$~w" > "$~w"',
        [string, string], [],
        [effect(io), deterministic, total, pattern(command)]),

    % delete_file - rm
    declare_binding(bash, delete_file/1, 'rm "$~w"',
        [string], [],
        [effect(io), deterministic, total, pattern(command)]),

    % make_directory - mkdir -p
    declare_binding(bash, make_directory/1, 'mkdir -p "$~w"',
        [string], [],
        [effect(io), deterministic, total, pattern(command)]).

% ============================================================================
% TESTS
% ============================================================================

test_bash_bindings :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  Bash Bindings Tests                  ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Initialize bindings
    format('[Test 1] Initializing Bash bindings~n', []),
    init_bash_bindings,
    format('[✓] Bash bindings initialized~n~n', []),

    % Test string bindings
    format('[Test 2] Checking string bindings~n', []),
    (   bash_binding(string_length/2, '${#~w}', _, _, _)
    ->  format('[✓] string_length/2 binding exists~n', [])
    ;   format('[✗] string_length/2 binding missing~n', []), fail
    ),

    % Test file bindings
    format('~n[Test 3] Checking file bindings~n', []),
    (   bash_binding(file_exists/1, '[[ -f "$~w" ]]', _, _, _)
    ->  format('[✓] file_exists/1 binding exists~n', [])
    ;   format('[✗] file_exists/1 binding missing~n', []), fail
    ),

    % Count total bindings
    format('~n[Test 4] Counting total bindings~n', []),
    findall(P, bash_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('[✓] Total Bash bindings: ~w~n', [Count]),

    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  All Bash Bindings Tests Passed       ║~n', []),
    format('╚════════════════════════════════════════╝~n', []).