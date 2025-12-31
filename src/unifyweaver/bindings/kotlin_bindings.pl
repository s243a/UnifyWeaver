% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% kotlin_bindings.pl - Kotlin-specific bindings for UnifyWeaver
%
% Kotlin bindings leverage Java stdlib but use Kotlin idioms.

:- encoding(utf8).

:- module(kotlin_bindings, [
    init_kotlin_bindings/0,
    kt_binding/5,
    kt_binding_import/2,         % kt_binding_import(Pred, Import)
    test_kotlin_bindings/0
]).

:- use_module('../core/binding_registry').

%% init_kotlin_bindings
init_kotlin_bindings :-
    register_stdlib_bindings,
    register_collection_bindings,
    register_string_bindings,
    register_io_bindings,
    register_sequence_bindings.

%% kt_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
kt_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(kotlin, Pred, TargetName, Inputs, Outputs, Options).

%% kt_binding_import(?Pred, ?Import)
%  Get the import required for a Kotlin binding.
kt_binding_import(Pred, Import) :-
    kt_binding(Pred, _, _, _, Options),
    member(import(Import), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

%% :- kt_binding(Pred, TargetName, Inputs, Outputs, Options)
%  Directive for user-defined Kotlin bindings.
:- multifile user:term_expansion/2.

user:term_expansion(
    (:- kt_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(kotlin, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% STDLIB BINDINGS
% ============================================================================

register_stdlib_bindings :-
    % Type conversions
    declare_binding(kotlin, to_int/2, '.toInt()',
        [any], [int],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(kotlin, to_double/2, '.toDouble()',
        [any], [double],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(kotlin, to_string/2, '.toString()',
        [any], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    % Null safety
    declare_binding(kotlin, let/2, '.let { }',
        [nullable], [any],
        [pure, deterministic, total, pattern(scope_function)]),
    
    declare_binding(kotlin, also/2, '.also { }',
        [any], [any],
        [effect(state), deterministic, total, pattern(scope_function)]),
    
    declare_binding(kotlin, apply/2, '.apply { }',
        [any], [any],
        [effect(state), deterministic, total, pattern(scope_function)]),
    
    declare_binding(kotlin, run/2, '.run { }',
        [any], [any],
        [pure, deterministic, total, pattern(scope_function)]),
    
    declare_binding(kotlin, elvis/3, '?:',
        [nullable, any], [any],
        [pure, deterministic, total, pattern(operator)]),
    
    % Printing
    declare_binding(kotlin, println/1, 'println',
        [any], [],
        [effect(io), deterministic, total]),
    
    declare_binding(kotlin, print/1, 'print',
        [any], [],
        [effect(io), deterministic, total]).

% ============================================================================
% COLLECTION BINDINGS
% ============================================================================

register_collection_bindings :-
    % List creation
    declare_binding(kotlin, list_of/2, 'listOf',
        [vararg], [list],
        [pure, deterministic, total]),
    
    declare_binding(kotlin, mutable_list_of/2, 'mutableListOf',
        [vararg], [mutable_list],
        [pure, deterministic, total]),
    
    % Map creation
    declare_binding(kotlin, map_of/2, 'mapOf',
        [vararg], [map],
        [pure, deterministic, total]),
    
    declare_binding(kotlin, mutable_map_of/2, 'mutableMapOf',
        [vararg], [mutable_map],
        [pure, deterministic, total]),
    
    % Collection operations
    declare_binding(kotlin, size/2, '.size',
        [collection], [int],
        [pure, deterministic, total, pattern(property)]),
    
    declare_binding(kotlin, is_empty/1, '.isEmpty()',
        [collection], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, contains/2, '.contains',
        [collection, any], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, first/2, '.first()',
        [collection], [any],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(kotlin, last/2, '.last()',
        [collection], [any],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(kotlin, get/3, '[index]',
        [list, int], [any],
        [pure, deterministic, partial, pattern(index)]).

% ============================================================================
% STRING BINDINGS
% ============================================================================

register_string_bindings :-
    declare_binding(kotlin, length/2, '.length',
        [string], [int],
        [pure, deterministic, total, pattern(property)]),
    
    declare_binding(kotlin, substring/3, '.substring',
        [string, int], [string],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(kotlin, split/3, '.split',
        [string, string], [list],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, trim/2, '.trim()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, lowercase/2, '.lowercase()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, uppercase/2, '.uppercase()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, replace/4, '.replace',
        [string, string, string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, contains/2, '.contains',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, starts_with/2, '.startsWith',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, ends_with/2, '.endsWith',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% I/O BINDINGS
% ============================================================================

register_io_bindings :-
    declare_binding(kotlin, read_line/1, 'readLine()',
        [], [nullable_string],
        [effect(io), deterministic, total]),
    
    declare_binding(kotlin, buffered_reader/2, 'BufferedReader',
        [reader], [buffered_reader],
        [pure, deterministic, total,
         import('java.io.BufferedReader')]),
    
    declare_binding(kotlin, file/2, 'File',
        [string], [file],
        [pure, deterministic, total,
         import('java.io.File')]),
    
    declare_binding(kotlin, read_text/2, '.readText()',
        [file], [string],
        [effect(io), deterministic, partial, pattern(method_call)]),
    
    declare_binding(kotlin, write_text/2, '.writeText',
        [file, string], [],
        [effect(io), deterministic, partial, pattern(method_call)]).

% ============================================================================
% SEQUENCE BINDINGS (Kotlin's lazy sequences)
% ============================================================================

register_sequence_bindings :-
    declare_binding(kotlin, as_sequence/2, '.asSequence()',
        [iterable], [sequence],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, sequence_of/2, 'sequenceOf',
        [vararg], [sequence],
        [pure, deterministic, total]),
    
    declare_binding(kotlin, filter/2, '.filter { }',
        [sequence, lambda], [sequence],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, map/2, '.map { }',
        [sequence, lambda], [sequence],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, map_not_null/2, '.mapNotNull { }',
        [sequence, lambda], [sequence],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, flat_map/2, '.flatMap { }',
        [sequence, lambda], [sequence],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, take/3, '.take',
        [sequence, int], [sequence],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, drop/3, '.drop',
        [sequence, int], [sequence],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, to_list/2, '.toList()',
        [sequence], [list],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(kotlin, for_each/2, '.forEach { }',
        [sequence, lambda], [],
        [effect(state), deterministic, total, pattern(method_call)]).

% ============================================================================
% TESTS
% ============================================================================

test_kotlin_bindings :-
    format('~n=== Kotlin Bindings Tests ===~n~n'),

    format('[Test 1] Initializing Kotlin bindings~n'),
    init_kotlin_bindings,
    format('  [PASS] Kotlin bindings initialized~n'),

    format('~n[Test 2] Checking scope functions~n'),
    (   kt_binding(let/2, '.let { }', _, _, _)
    ->  format('  [PASS] let binding exists~n')
    ;   format('  [FAIL] let binding missing~n')
    ),

    format('~n[Test 3] Checking sequence bindings~n'),
    (   kt_binding(map_not_null/2, '.mapNotNull { }', _, _, _)
    ->  format('  [PASS] mapNotNull binding exists~n')
    ;   format('  [FAIL] mapNotNull binding missing~n')
    ),

    format('~n[Test 4] Counting total bindings~n'),
    findall(P, kt_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('  [INFO] Total Kotlin bindings: ~w~n', [Count]),

    format('~n=== Kotlin Bindings Tests Complete ===~n').
