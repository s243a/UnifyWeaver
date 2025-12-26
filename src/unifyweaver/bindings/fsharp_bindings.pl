:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% fsharp_bindings.pl - F#-specific bindings
%
% This module defines bindings for F# target language features.
% Uses F# functional idioms.

:- module(fsharp_bindings, [
    init_fsharp_bindings/0,
    fs_binding/5,               % fs_binding(Pred, TargetName, Inputs, Outputs, Options)
    fs_binding_open/2,          % fs_binding_open(Pred, Module)
    test_fsharp_bindings/0
]).

:- use_module('../core/binding_registry').

%% init_fsharp_bindings
init_fsharp_bindings :-
    register_fsharp_builtin_bindings,
    register_fsharp_string_bindings,
    register_fsharp_list_bindings,
    register_fsharp_io_bindings.

%% fs_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
fs_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(fsharp, Pred, TargetName, Inputs, Outputs, Options).

%% fs_binding_open(?Pred, ?Module)
%  Get the open module required for an F# binding.
fs_binding_open(Pred, Module) :-
    fs_binding(Pred, _, _, _, Options),
    member(open(Module), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

%% :- fs_binding(Pred, TargetName, Inputs, Outputs, Options)
%  Directive for user-defined F# bindings.
:- multifile user:term_expansion/2.

user:term_expansion(
    (:- fs_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(fsharp, Pred, TargetName, Inputs, Outputs, Options)))
).

%% Core Built-in Bindings
register_fsharp_builtin_bindings :-
    % Length - Array/List length
    declare_binding(fsharp, length/2, 'Array.length',
        [array], [int],
        [pure, deterministic, total, pattern(function_call)]),

    % List.length
    declare_binding(fsharp, list_length/2, 'List.length',
        [list], [int],
        [pure, deterministic, total, pattern(function_call)]),

    % string
    declare_binding(fsharp, to_string/2, 'string',
        [any], [string],
        [pure, deterministic, total, pattern(function_call)]),

    % int
    declare_binding(fsharp, parse_int/2, 'int',
        [string], [int],
        [pure, deterministic, partial, effect(throws), pattern(function_call)]),

    % float
    declare_binding(fsharp, parse_float/2, 'float',
        [string], [float],
        [pure, deterministic, partial, effect(throws), pattern(function_call)]).

%% String Operation Bindings
register_fsharp_string_bindings :-
    % Split
    declare_binding(fsharp, string_split/3, '.Split',
        [string, char], [array],
        [pure, deterministic, total, pattern(method_call)]),

    % String.concat
    declare_binding(fsharp, string_join/3, 'String.concat',
        [string, seq], [string],
        [pure, deterministic, total, pattern(function_call)]),

    % Replace
    declare_binding(fsharp, string_replace/4, '.Replace',
        [string, string, string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % Trim
    declare_binding(fsharp, string_trim/2, '.Trim()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % ToLower
    declare_binding(fsharp, string_lower/2, '.ToLower()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % ToUpper
    declare_binding(fsharp, string_upper/2, '.ToUpper()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % Contains
    declare_binding(fsharp, string_contains/2, '.Contains',
        [string, string], [bool],
        [pure, deterministic, total, pattern(method_call)]).

%% List/Seq Operation Bindings (Functional style)
register_fsharp_list_bindings :-
    % List.map
    declare_binding(fsharp, list_map/3, 'List.map',
        [function, list], [list],
        [pure, deterministic, total, pattern(function_call)]),

    % List.filter
    declare_binding(fsharp, list_filter/3, 'List.filter',
        [function, list], [list],
        [pure, deterministic, total, pattern(function_call)]),

    % List.fold
    declare_binding(fsharp, list_fold/4, 'List.fold',
        [function, any, list], [any],
        [pure, deterministic, total, pattern(function_call)]),

    % List.head
    declare_binding(fsharp, list_head/2, 'List.head',
        [list], [any],
        [pure, deterministic, partial, effect(throws), pattern(function_call)]),

    % List.tail
    declare_binding(fsharp, list_tail/2, 'List.tail',
        [list], [list],
        [pure, deterministic, partial, effect(throws), pattern(function_call)]),

    % Seq.ofList
    declare_binding(fsharp, seq_of_list/2, 'Seq.ofList',
        [list], [seq],
        [pure, deterministic, total, pattern(function_call)]),

    % Seq.toList
    declare_binding(fsharp, seq_to_list/2, 'Seq.toList',
        [seq], [list],
        [pure, deterministic, total, pattern(function_call)]),

    % List.sum
    declare_binding(fsharp, list_sum/2, 'List.sum',
        [list], [number],
        [pure, deterministic, total, pattern(function_call)]),

    % List.average
    declare_binding(fsharp, list_avg/2, 'List.average',
        [list], [float],
        [pure, deterministic, partial, effect(throws), pattern(function_call)]).

%% I/O Operation Bindings
register_fsharp_io_bindings :-
    % File.Exists
    declare_binding(fsharp, file_exists/1, 'File.Exists',
        [string], [bool],
        [effect(io), deterministic, total, pattern(function_call), import('System.IO')]),

    % File.ReadAllText
    declare_binding(fsharp, read_file/2, 'File.ReadAllText',
        [string], [string],
        [effect(io), deterministic, partial, effect(throws), pattern(function_call), import('System.IO')]),

    % File.WriteAllText
    declare_binding(fsharp, write_file/2, 'File.WriteAllText',
        [string, string], [unit],
        [effect(io), deterministic, partial, effect(throws), pattern(function_call), import('System.IO')]),

    % printfn
    declare_binding(fsharp, printfn/2, 'printfn',
        [string, any], [unit],
        [effect(io), deterministic, total, pattern(function_call)]).

%% Tests
test_fsharp_bindings :-
    format('[F# Bindings] Initializing...~n', []),
    init_fsharp_bindings,
    format('[F# Bindings] Testing length/2...~n', []),
    (   fs_binding(length/2, 'Array.length', _, _, _)
    ->  format('[PASS] length/2 binding exists~n', [])
    ;   format('[FAIL] length/2 binding missing~n', [])
    ),
    format('[F# Bindings] Testing List.map...~n', []),
    (   fs_binding(list_map/3, 'List.map', _, _, _)
    ->  format('[PASS] list_map/3 binding exists~n', [])
    ;   format('[FAIL] list_map/3 binding missing~n', [])
    ),
    findall(P, fs_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('[F# Bindings] Total: ~w bindings~n', [Count]).
