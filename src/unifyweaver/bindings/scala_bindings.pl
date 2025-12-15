% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% scala_bindings.pl - Scala-specific bindings for UnifyWeaver
%
% Scala bindings leverage functional idioms and type-safe patterns.

:- encoding(utf8).

:- module(scala_bindings, [
    init_scala_bindings/0,
    scala_binding/5,
    test_scala_bindings/0
]).

:- use_module('../core/binding_registry').

%% init_scala_bindings
init_scala_bindings :-
    register_option_bindings,
    register_collection_bindings,
    register_string_bindings,
    register_lazylist_bindings,
    register_pattern_bindings.

%% scala_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
scala_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(scala, Pred, TargetName, Inputs, Outputs, Options).

% ============================================================================
% OPTION/EITHER BINDINGS
% ============================================================================

register_option_bindings :-
    declare_binding(scala, some/2, 'Some',
        [any], [option],
        [pure, deterministic, total]),
    
    declare_binding(scala, none/1, 'None',
        [], [option],
        [pure, deterministic, total]),
    
    declare_binding(scala, get_or_else/3, '.getOrElse',
        [option, any], [any],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, map_option/2, '.map',
        [option, lambda], [option],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, flat_map_option/2, '.flatMap',
        [option, lambda], [option],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, filter_option/2, '.filter',
        [option, lambda], [option],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, is_defined/1, '.isDefined',
        [option], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, is_empty/1, '.isEmpty',
        [option], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    % Either
    declare_binding(scala, right/2, 'Right',
        [any], [either],
        [pure, deterministic, total]),
    
    declare_binding(scala, left/2, 'Left',
        [any], [either],
        [pure, deterministic, total]).

% ============================================================================
% COLLECTION BINDINGS
% ============================================================================

register_collection_bindings :-
    % List creation
    declare_binding(scala, list/2, 'List',
        [vararg], [list],
        [pure, deterministic, total]),
    
    declare_binding(scala, nil/1, 'Nil',
        [], [list],
        [pure, deterministic, total]),
    
    declare_binding(scala, cons/3, '::',
        [any, list], [list],
        [pure, deterministic, total, pattern(operator)]),
    
    % Map creation
    declare_binding(scala, map_of/2, 'Map',
        [vararg], [map],
        [pure, deterministic, total]),
    
    % List operations
    declare_binding(scala, head/2, '.head',
        [list], [any],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(scala, tail/2, '.tail',
        [list], [list],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(scala, length/2, '.length',
        [list], [int],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, contains/2, '.contains',
        [list, any], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    % Higher-order
    declare_binding(scala, map/2, '.map',
        [collection, lambda], [collection],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, flat_map/2, '.flatMap',
        [collection, lambda], [collection],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, filter/2, '.filter',
        [collection, lambda], [collection],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, fold_left/3, '.foldLeft',
        [collection, any, lambda], [any],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, fold_right/3, '.foldRight',
        [collection, any, lambda], [any],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, foreach/2, '.foreach',
        [collection, lambda], [],
        [effect(state), deterministic, total, pattern(method_call)]).

% ============================================================================
% STRING BINDINGS
% ============================================================================

register_string_bindings :-
    declare_binding(scala, length/2, '.length',
        [string], [int],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, substring/3, '.substring',
        [string, int], [string],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(scala, split/3, '.split',
        [string, string], [array],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, trim/2, '.trim',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, to_lower/2, '.toLowerCase',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, to_upper/2, '.toUpperCase',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, starts_with/2, '.startsWith',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, ends_with/2, '.endsWith',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, interpolate/2, 's\"...\"',
        [vararg], [string],
        [pure, deterministic, total, pattern(interpolation)]).

% ============================================================================
% LAZYLIST BINDINGS (Scala 2.13+/3)
% ============================================================================

register_lazylist_bindings :-
    declare_binding(scala, lazy_list/2, 'LazyList',
        [vararg], [lazy_list],
        [pure, deterministic, total]),
    
    declare_binding(scala, lazy_list_from/2, 'LazyList.from',
        [iterator], [lazy_list],
        [pure, deterministic, total]),
    
    declare_binding(scala, cons_lazy/3, '#::',
        [any, lazy_list], [lazy_list],
        [pure, deterministic, total, pattern(operator)]),
    
    declare_binding(scala, take/3, '.take',
        [lazy_list, int], [lazy_list],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, drop/3, '.drop',
        [lazy_list, int], [lazy_list],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, take_while/2, '.takeWhile',
        [lazy_list, lambda], [lazy_list],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, drop_while/2, '.dropWhile',
        [lazy_list, lambda], [lazy_list],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(scala, to_list/2, '.toList',
        [lazy_list], [list],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% PATTERN MATCHING BINDINGS
% ============================================================================

register_pattern_bindings :-
    declare_binding(scala, match/2, 'match { case ... }',
        [any, cases], [any],
        [pure, deterministic, total, pattern(match)]),
    
    declare_binding(scala, case_class/2, 'case class',
        [name, fields], [type],
        [pure, deterministic, total, pattern(definition)]),
    
    declare_binding(scala, unapply/2, 'unapply',
        [any], [option],
        [pure, deterministic, total, pattern(extractor)]).

% ============================================================================
% TESTS
% ============================================================================

test_scala_bindings :-
    format('~n=== Scala Bindings Tests ===~n~n'),

    format('[Test 1] Initializing Scala bindings~n'),
    init_scala_bindings,
    format('  [PASS] Scala bindings initialized~n'),

    format('~n[Test 2] Checking Option operations~n'),
    (   scala_binding(some/2, 'Some', _, _, _)
    ->  format('  [PASS] Some binding exists~n')
    ;   format('  [FAIL] Some binding missing~n')
    ),

    format('~n[Test 3] Checking LazyList bindings~n'),
    (   scala_binding(cons_lazy/3, '#::', _, _, _)
    ->  format('  [PASS] #:: (cons) binding exists~n')
    ;   format('  [FAIL] #:: binding missing~n')
    ),

    format('~n[Test 4] Counting total bindings~n'),
    findall(P, scala_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('  [INFO] Total Scala bindings: ~w~n', [Count]),

    format('~n=== Scala Bindings Tests Complete ===~n').
