% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% csharp_bindings.pl - C#-specific bindings
%
% This module defines bindings for C# target language features.
%
% Categories:
%   - Core Built-ins (Length, ToString, etc.)
%   - String Operations (Split, Join, Replace, etc.)
%   - Math Operations (Math.Sqrt, Math.Abs, etc.)
%   - I/O Operations (File.ReadAllText, etc.)
%   - LINQ Operations (Select, Where, etc.)
%
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md

:- module(csharp_bindings, [
    init_csharp_bindings/0,
    cs_binding/5,               % Convenience: cs_binding(Pred, TargetName, Inputs, Outputs, Options)
    test_csharp_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_csharp_bindings
%
%  Initialize all C# bindings. Call this before using the compiler.
%
init_csharp_bindings :-
    register_builtin_bindings,
    register_string_bindings,
    register_math_bindings,
    register_io_bindings,
    register_linq_bindings.

% ============================================================================
% CONVENIENCE PREDICATE
% ============================================================================

%% cs_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query C# bindings with reduced arity (Target=csharp implied).
%
cs_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(csharp, Pred, TargetName, Inputs, Outputs, Options).

% ============================================================================
% CORE BUILT-IN BINDINGS
% ============================================================================

register_builtin_bindings :-
    % -------------------------------------------
    % Properties
    % -------------------------------------------

    % length - Array/String length
    declare_binding(csharp, length/2, '.Length',
        [sequence], [int],
        [pure, deterministic, total, pattern(property)]),

    % count - List/Collection count
    declare_binding(csharp, count/2, '.Count',
        [collection], [int],
        [pure, deterministic, total, pattern(property)]),

    % -------------------------------------------
    % Conversions
    % -------------------------------------------

    % to_string - ToString()
    declare_binding(csharp, to_string/2, '.ToString()',
        [any], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % parse_int - int.Parse
    declare_binding(csharp, parse_int/2, 'int.Parse',
        [string], [int],
        [pure, deterministic, partial, effect(throws), pattern(static_call)]),

    % parse_float - double.Parse
    declare_binding(csharp, parse_float/2, 'double.Parse',
        [string], [double],
        [pure, deterministic, partial, effect(throws), pattern(static_call)]),

    % to_bool - Convert.ToBoolean
    declare_binding(csharp, to_bool/2, 'Convert.ToBoolean',
        [any], [bool],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % to_list - ToList()
    declare_binding(csharp, to_list/2, '.ToList()',
        [enumerable], [list],
        [pure, deterministic, total, pattern(method_call), import('System.Linq')]).

% ============================================================================
% STRING OPERATION BINDINGS
% ============================================================================

register_string_bindings :-
    % -------------------------------------------
    % String Methods
    % -------------------------------------------

    % string_split - Split(char[])
    declare_binding(csharp, string_split/3, '.Split',
        [string, string], [array],
        [pure, deterministic, total, pattern(method_call)]),

    % string_join - String.Join
    declare_binding(csharp, string_join/3, 'String.Join',
        [string, enumerable], [string],
        [pure, deterministic, total, pattern(static_call)]),

    % string_replace - Replace
    declare_binding(csharp, string_replace/4, '.Replace',
        [string, string, string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % string_trim - Trim()
    declare_binding(csharp, string_trim/2, '.Trim()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % string_lower - ToLower()
    declare_binding(csharp, string_lower/2, '.ToLower()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % string_upper - ToUpper()
    declare_binding(csharp, string_upper/2, '.ToUpper()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % string_contains - Contains
    declare_binding(csharp, string_contains/2, '.Contains',
        [string, string], [bool],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% MATH OPERATION BINDINGS
% ============================================================================

register_math_bindings :-
    % -------------------------------------------
    % System.Math
    % -------------------------------------------

    % sqrt - Math.Sqrt
    declare_binding(csharp, sqrt/2, 'Math.Sqrt',
        [number], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % abs - Math.Abs
    declare_binding(csharp, abs/2, 'Math.Abs',
        [number], [number],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % pow - Math.Pow
    declare_binding(csharp, pow/3, 'Math.Pow',
        [number, number], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % round - Math.Round
    declare_binding(csharp, round/2, 'Math.Round',
        [number], [number],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % floor - Math.Floor
    declare_binding(csharp, floor/2, 'Math.Floor',
        [number], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % ceiling - Math.Ceiling
    declare_binding(csharp, ceil/2, 'Math.Ceiling',
        [number], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % min - Math.Min
    declare_binding(csharp, min/3, 'Math.Min',
        [number, number], [number],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % max - Math.Max
    declare_binding(csharp, max/3, 'Math.Max',
        [number, number], [number],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % log - Math.Log
    declare_binding(csharp, log/2, 'Math.Log',
        [number], [double],
        [pure, deterministic, partial, effect(throws), pattern(static_call), import('System')]),

    % log10 - Math.Log10
    declare_binding(csharp, log10/2, 'Math.Log10',
        [number], [double],
        [pure, deterministic, partial, effect(throws), pattern(static_call), import('System')]),

    % sin - Math.Sin
    declare_binding(csharp, sin/2, 'Math.Sin',
        [number], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % cos - Math.Cos
    declare_binding(csharp, cos/2, 'Math.Cos',
        [number], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % tan - Math.Tan
    declare_binding(csharp, tan/2, 'Math.Tan',
        [number], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % pi - Math.PI
    declare_binding(csharp, pi/1, 'Math.PI',
        [], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % e - Math.E
    declare_binding(csharp, e/1, 'Math.E',
        [], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]).

% ============================================================================
% I/O OPERATION BINDINGS
% ============================================================================

register_io_bindings :-
    % -------------------------------------------
    % System.IO
    % -------------------------------------------

    % file_exists - File.Exists
    declare_binding(csharp, file_exists/1, 'File.Exists',
        [string], [bool],
        [effect(io), deterministic, total, pattern(static_call), import('System.IO')]),

    % read_file - File.ReadAllText
    declare_binding(csharp, read_file/2, 'File.ReadAllText',
        [string], [string],
        [effect(io), deterministic, partial, effect(throws), pattern(static_call), import('System.IO')]),

    % write_file - File.WriteAllText
    declare_binding(csharp, write_file/2, 'File.WriteAllText',
        [string, string], [],
        [effect(io), deterministic, partial, effect(throws), pattern(static_call), import('System.IO')]),

    % path_combine - Path.Combine
    declare_binding(csharp, path_combine/3, 'Path.Combine',
        [string, string], [string],
        [pure, deterministic, total, pattern(static_call), import('System.IO')]).

% ============================================================================
% LINQ BINDINGS
% ============================================================================

register_linq_bindings :-
    % -------------------------------------------
    % System.Linq
    % -------------------------------------------

    % list_first - First()
    declare_binding(csharp, list_first/2, '.First()',
        [enumerable], [any],
        [pure, deterministic, partial, effect(throws), pattern(method_call), import('System.Linq')]),

    % list_last - Last()
    declare_binding(csharp, list_last/2, '.Last()',
        [enumerable], [any],
        [pure, deterministic, partial, effect(throws), pattern(method_call), import('System.Linq')]),

    % list_any - Any()
    declare_binding(csharp, list_any/1, '.Any()',
        [enumerable], [bool],
        [pure, deterministic, total, pattern(method_call), import('System.Linq')]).

% ============================================================================
% TESTS
% ============================================================================

test_csharp_bindings :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  C# Bindings Tests                    ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Initialize bindings
    format('[Test 1] Initializing C# bindings~n', []),
    init_csharp_bindings,
    format('[✓] C# bindings initialized~n~n', []),

    % Test builtin bindings exist
    format('[Test 2] Checking built-in bindings~n', []),
    (   cs_binding(length/2, '.Length', _, _, _)
    ->  format('[✓] length/2 -> .Length binding exists~n', [])
    ;   format('[✗] length/2 binding missing~n', []), fail
    ),

    % Test string bindings exist
    format('~n[Test 3] Checking string bindings~n', []),
    (   cs_binding(string_split/3, _, _, _, _)
    ->  format('[✓] string_split/3 binding exists~n', [])
    ;   format('[✗] string_split/3 binding missing~n', []), fail
    ),

    % Test math bindings with import
    format('~n[Test 4] Checking math bindings with imports~n', []),
    (   cs_binding(sqrt/2, 'Math.Sqrt', _, _, Options),
        member(import('System'), Options)
    ->  format('[✓] sqrt/2 has import(System)~n', [])
    ;   format('[✗] sqrt/2 missing import~n', []), fail
    ),

    % Count total bindings
    format('~n[Test 5] Counting total bindings~n', []),
    findall(P, cs_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('[✓] Total C# bindings: ~w~n', [Count]),

    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  All C# Bindings Tests Passed         ║~n', []),
    format('╚════════════════════════════════════════╝~n', []).
