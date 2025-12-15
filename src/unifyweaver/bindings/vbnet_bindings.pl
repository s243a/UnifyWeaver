:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% vbnet_bindings.pl - VB.NET-specific bindings
%
% This module defines bindings for VB.NET target language features.

:- module(vbnet_bindings, [
    init_vbnet_bindings/0,
    vb_binding/5,               % vb_binding(Pred, TargetName, Inputs, Outputs, Options)
    test_vbnet_bindings/0
]).

:- use_module('../core/binding_registry').

%% init_vbnet_bindings
init_vbnet_bindings :-
    register_vbnet_builtin_bindings,
    register_vbnet_string_bindings,
    register_vbnet_math_bindings,
    register_vbnet_io_bindings.

%% vb_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
vb_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(vbnet, Pred, TargetName, Inputs, Outputs, Options).

%% Core Built-in Bindings
register_vbnet_builtin_bindings :-
    % Length - Array/String length
    declare_binding(vbnet, length/2, '.Length',
        [sequence], [integer],
        [pure, deterministic, total, pattern(property)]),

    % Count - List/Collection count
    declare_binding(vbnet, count/2, '.Count',
        [collection], [integer],
        [pure, deterministic, total, pattern(property)]),

    % to_string - ToString()
    declare_binding(vbnet, to_string/2, '.ToString()',
        [any], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % CInt - Convert to Integer
    declare_binding(vbnet, parse_int/2, 'CInt',
        [string], [integer],
        [pure, deterministic, partial, effect(throws), pattern(function_call)]),

    % CDbl - Convert to Double
    declare_binding(vbnet, parse_float/2, 'CDbl',
        [string], [double],
        [pure, deterministic, partial, effect(throws), pattern(function_call)]),

    % CBool - Convert to Boolean
    declare_binding(vbnet, to_bool/2, 'CBool',
        [any], [boolean],
        [pure, deterministic, total, pattern(function_call)]).

%% String Operation Bindings
register_vbnet_string_bindings :-
    % Split
    declare_binding(vbnet, string_split/3, '.Split',
        [string, string], [array],
        [pure, deterministic, total, pattern(method_call)]),

    % Join
    declare_binding(vbnet, string_join/3, 'String.Join',
        [string, enumerable], [string],
        [pure, deterministic, total, pattern(static_call)]),

    % Replace
    declare_binding(vbnet, string_replace/4, '.Replace',
        [string, string, string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % Trim
    declare_binding(vbnet, string_trim/2, '.Trim()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % ToLower
    declare_binding(vbnet, string_lower/2, '.ToLower()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % ToUpper
    declare_binding(vbnet, string_upper/2, '.ToUpper()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % Contains
    declare_binding(vbnet, string_contains/2, '.Contains',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]).

%% Math Operation Bindings
register_vbnet_math_bindings :-
    % Math.Sqrt
    declare_binding(vbnet, sqrt/2, 'Math.Sqrt',
        [number], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % Math.Abs
    declare_binding(vbnet, abs/2, 'Math.Abs',
        [number], [number],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % Math.Pow
    declare_binding(vbnet, pow/3, 'Math.Pow',
        [number, number], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % Math.Round
    declare_binding(vbnet, round/2, 'Math.Round',
        [number], [number],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % Math.Floor
    declare_binding(vbnet, floor/2, 'Math.Floor',
        [number], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % Math.Ceiling
    declare_binding(vbnet, ceil/2, 'Math.Ceiling',
        [number], [double],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % Math.Min
    declare_binding(vbnet, min/3, 'Math.Min',
        [number, number], [number],
        [pure, deterministic, total, pattern(static_call), import('System')]),

    % Math.Max
    declare_binding(vbnet, max/3, 'Math.Max',
        [number, number], [number],
        [pure, deterministic, total, pattern(static_call), import('System')]).

%% I/O Operation Bindings
register_vbnet_io_bindings :-
    % File.Exists
    declare_binding(vbnet, file_exists/1, 'File.Exists',
        [string], [boolean],
        [effect(io), deterministic, total, pattern(static_call), import('System.IO')]),

    % File.ReadAllText
    declare_binding(vbnet, read_file/2, 'File.ReadAllText',
        [string], [string],
        [effect(io), deterministic, partial, effect(throws), pattern(static_call), import('System.IO')]),

    % File.WriteAllText
    declare_binding(vbnet, write_file/2, 'File.WriteAllText',
        [string, string], [],
        [effect(io), deterministic, partial, effect(throws), pattern(static_call), import('System.IO')]),

    % Path.Combine
    declare_binding(vbnet, path_combine/3, 'Path.Combine',
        [string, string], [string],
        [pure, deterministic, total, pattern(static_call), import('System.IO')]).

%% Tests
test_vbnet_bindings :-
    format('[VB.NET Bindings] Initializing...~n', []),
    init_vbnet_bindings,
    format('[VB.NET Bindings] Testing length/2...~n', []),
    (   vb_binding(length/2, '.Length', _, _, _)
    ->  format('[PASS] length/2 binding exists~n', [])
    ;   format('[FAIL] length/2 binding missing~n', [])
    ),
    findall(P, vb_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('[VB.NET Bindings] Total: ~w bindings~n', [Count]).
