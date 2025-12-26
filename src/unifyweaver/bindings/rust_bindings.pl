% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% rust_bindings.pl - Rust-specific bindings
%
% This module defines bindings for Rust target language features.
%
% Categories:
%   - Core Built-ins (len, to_string, parse)
%   - String Operations (String methods)
%   - Math Operations (f64 methods, std::f64::consts)
%   - I/O Operations (std::fs, std::io)
%   - Regex Operations (regex crate)
%   - JSON Operations (serde_json crate)
%
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md

:- module(rust_bindings, [
    init_rust_bindings/0,
    rs_binding/5,               % Convenience: rs_binding(Pred, TargetName, Inputs, Outputs, Options)
    rs_binding_import/2,        % rs_binding_import(Pred, Import) - get required import/use
    test_rust_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_rust_bindings
%
%  Initialize all Rust bindings. Call this before using the compiler.
%
init_rust_bindings :-
    register_builtin_bindings,
    register_string_bindings,
    register_math_bindings,
    register_io_bindings,
    register_regex_bindings,
    register_json_bindings.

% ============================================================================
% CONVENIENCE PREDICATE
% ============================================================================

%% rs_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query Rust bindings with reduced arity (Target=rust implied).
%
rs_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(rust, Pred, TargetName, Inputs, Outputs, Options).

%% rs_binding_import(?Pred, ?Import)
%
%  Get the use/crate required for a Rust binding.
%
rs_binding_import(Pred, Import) :-
    rs_binding(Pred, _, _, _, Options),
    member(import(Import), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

%% :- rs_binding(Pred, TargetName, Inputs, Outputs, Options)
%
%  Directive for user-defined Rust bindings.
%  Allows users to declare bindings in their Prolog code.
%
%  Example:
%    :- rs_binding(my_hash/2, 'my_crate::hash', [string], [string], [import('my_crate')]).
%
:- multifile user:term_expansion/2.

user:term_expansion(
    (:- rs_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(rust, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% CORE BUILT-IN BINDINGS
% ============================================================================

register_builtin_bindings :-
    % -------------------------------------------
    % Length and Size
    % -------------------------------------------

    % length - len()
    declare_binding(rust, length/2, '.len()',
        [collection], [int],
        [pure, deterministic, total, pattern(method_call)]),

    % -------------------------------------------
    % Conversions
    % -------------------------------------------

    % to_string - .to_string()
    declare_binding(rust, to_string/2, '.to_string()',
        [any], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % parse_int - .parse::<i64>().unwrap()
    declare_binding(rust, parse_int/2, '.parse::<i64>()',
        [string], [int],
        [pure, deterministic, partial, effect(throws), pattern(method_call)]),

    % parse_float - .parse::<f64>().unwrap()
    declare_binding(rust, parse_float/2, '.parse::<f64>()',
        [string], [float],
        [pure, deterministic, partial, effect(throws), pattern(method_call)]).

% ============================================================================
% STRING OPERATION BINDINGS
% ============================================================================

register_string_bindings :-
    % -------------------------------------------
    % String Methods
    % -------------------------------------------

    % string_trim - .trim()
    declare_binding(rust, string_trim/2, '.trim()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % string_lower - .to_lowercase()
    declare_binding(rust, string_lower/2, '.to_lowercase()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % string_upper - .to_uppercase()
    declare_binding(rust, string_upper/2, '.to_uppercase()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % string_contains - .contains()
    declare_binding(rust, string_contains/2, '.contains',
        [string, string], [bool],
        [pure, deterministic, total, pattern(method_call)]),

    % string_replace - .replace()
    declare_binding(rust, string_replace/4, '.replace',
        [string, string, string], [string],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% MATH OPERATION BINDINGS
% ============================================================================

register_math_bindings :-
    % -------------------------------------------
    % f64 Methods and Consts
    % -------------------------------------------

    % sqrt - .sqrt()
    declare_binding(rust, sqrt/2, '.sqrt()',
        [float], [float],
        [pure, deterministic, total, pattern(method_call)]),

    % abs - .abs()
    declare_binding(rust, abs/2, '.abs()',
        [number], [number],
        [pure, deterministic, total, pattern(method_call)]),

    % floor - .floor()
    declare_binding(rust, floor/2, '.floor()',
        [float], [float],
        [pure, deterministic, total, pattern(method_call)]),

    % ceil - .ceil()
    declare_binding(rust, ceil/2, '.ceil()',
        [float], [float],
        [pure, deterministic, total, pattern(method_call)]),

    % round - .round()
    declare_binding(rust, round/2, '.round()',
        [float], [float],
        [pure, deterministic, total, pattern(method_call)]),

    % pow - .powf()
    declare_binding(rust, pow/3, '.powf',
        [float, float], [float],
        [pure, deterministic, total, pattern(method_call)]),

    % min - .min()
    declare_binding(rust, min/3, '.min',
        [number, number], [number],
        [pure, deterministic, total, pattern(method_call)]),

    % max - .max()
    declare_binding(rust, max/3, '.max',
        [number, number], [number],
        [pure, deterministic, total, pattern(method_call)]),

    % sin - .sin()
    declare_binding(rust, sin/2, '.sin()',
        [float], [float],
        [pure, deterministic, total, pattern(method_call)]),

    % cos - .cos()
    declare_binding(rust, cos/2, '.cos()',
        [float], [float],
        [pure, deterministic, total, pattern(method_call)]),

    % tan - .tan()
    declare_binding(rust, tan/2, '.tan()',
        [float], [float],
        [pure, deterministic, total, pattern(method_call)]),

    % pi - std::f64::consts::PI
    declare_binding(rust, pi/1, 'std::f64::consts::PI',
        [], [float],
        [pure, deterministic, total, pattern(static_value)]),

    % e - std::f64::consts::E
    declare_binding(rust, e/1, 'std::f64::consts::E',
        [], [float],
        [pure, deterministic, total, pattern(static_value)]).

% ============================================================================
% I/O OPERATION BINDINGS
% ============================================================================

register_io_bindings :-
    % -------------------------------------------
    % std::fs
    % -------------------------------------------

    % file_exists - std::path::Path::new(...).exists()
    declare_binding(rust, file_exists/1, 'std::path::Path::new',
        [string], [bool],
        [effect(io), deterministic, total, pattern(constructor_chain), suffix('.exists()')]),

    % read_file - std::fs::read_to_string
    declare_binding(rust, read_file/2, 'std::fs::read_to_string',
        [string], [string],
        [effect(io), deterministic, partial, effect(throws), pattern(static_call)]),

    % write_file - std::fs::write
    declare_binding(rust, write_file/2, 'std::fs::write',
        [string, string], [],
        [effect(io), deterministic, partial, effect(throws), pattern(static_call)]).

% ============================================================================
% REGEX OPERATION BINDINGS
% ============================================================================

register_regex_bindings :-
    % -------------------------------------------
    % regex crate
    % -------------------------------------------

    % regex_new - Regex::new
    declare_binding(rust, regex_compile/2, 'regex::Regex::new',
        [string], [regex],
        [pure, deterministic, partial, effect(throws), pattern(static_call), import('regex')]),

    % regex_is_match - .is_match()
    declare_binding(rust, regex_match/2, '.is_match',
        [regex, string], [bool],
        [pure, deterministic, total, pattern(method_call), import('regex')]).

% ============================================================================
% JSON OPERATION BINDINGS
% ============================================================================

register_json_bindings :-
    % -------------------------------------------
    % serde_json crate
    % -------------------------------------------

    % json_parse - serde_json::from_str
    declare_binding(rust, json_parse/2, 'serde_json::from_str',
        [string], [json],
        [pure, deterministic, partial, effect(throws), pattern(static_call), import('serde_json')]),

    % json_stringify - serde_json::to_string
    declare_binding(rust, json_stringify/2, 'serde_json::to_string',
        [json], [string],
        [pure, deterministic, partial, effect(throws), pattern(static_call), import('serde_json')]).

% ============================================================================
% TESTS
% ============================================================================

test_rust_bindings :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  Rust Bindings Tests                  ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Initialize bindings
    format('[Test 1] Initializing Rust bindings~n', []),
    init_rust_bindings,
    format('[✓] Rust bindings initialized~n~n', []),

    % Test builtin bindings
    format('[Test 2] Checking built-in bindings~n', []),
    (   rs_binding(length/2, '.len()', _, _, _)
    ->  format('[✓] length/2 -> .len() binding exists~n', [])
    ;   format('[✗] length/2 binding missing~n', []), fail
    ),

    % Test math bindings
    format('~n[Test 3] Checking math bindings~n', []),
    (   rs_binding(sqrt/2, '.sqrt()', _, _, _)
    ->  format('[✓] sqrt/2 binding exists~n', [])
    ;   format('[✗] sqrt/2 binding missing~n', []), fail
    ),

    % Test regex bindings with import
    format('~n[Test 4] Checking regex bindings with imports~n', []),
    (   rs_binding(regex_compile/2, _, _, _, Options),
        member(import('regex'), Options)
    ->  format('[✓] regex_compile/2 has import(regex)~n', [])
    ;   format('[✗] regex_compile/2 missing import~n', []), fail
    ),

    % Count total bindings
    format('~n[Test 5] Counting total bindings~n', []),
    findall(P, rs_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('[✓] Total Rust bindings: ~w~n', [Count]),

    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  All Rust Bindings Tests Passed       ║~n', []),
    format('╚════════════════════════════════════════╝~n', []).
