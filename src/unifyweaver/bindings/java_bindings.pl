% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% java_bindings.pl - Java-specific bindings for UnifyWeaver
%
% Provides bindings for Java standard library functions.
% Uses declare_binding/6 from binding_registry.

:- encoding(utf8).

:- module(java_bindings, [
    init_java_bindings/0,
    binding_import/1,
    java_binding/5,              % java_binding(Pred, TargetName, Inputs, Outputs, Options)
    java_binding_import/2        % java_binding_import(Pred, Import)
]).

:- use_module('../core/binding_registry').

% Track required imports
:- dynamic binding_import/1.

%% java_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%  Query Java bindings with reduced arity (Target=java implied).
java_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(java, Pred, TargetName, Inputs, Outputs, Options).

%% java_binding_import(?Pred, ?Import)
%  Get the import required for a Java binding.
java_binding_import(Pred, Import) :-
    java_binding(Pred, _, _, _, Options),
    member(import(Import), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

%% :- java_binding(Pred, TargetName, Inputs, Outputs, Options)
%  Directive for user-defined Java bindings.
:- multifile user:term_expansion/2.

user:term_expansion(
    (:- java_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(java, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_java_bindings
%  Initialize all Java bindings. Call this before using the compiler.
init_java_bindings :-
    retractall(binding_import(_)),
    register_string_bindings,
    register_math_bindings,
    register_collection_bindings,
    register_io_bindings,
    register_stream_bindings.

% ============================================================================
% STRING OPERATION BINDINGS (java.lang.String)
% ============================================================================

register_string_bindings :-
    % String.length()
    declare_binding(java, string_length/2, '.length()',
        [string], [int],
        [pure, deterministic, total, pattern(method_call)]),
    
    % String.toLowerCase()
    declare_binding(java, string_lower/2, '.toLowerCase()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    % String.toUpperCase()
    declare_binding(java, string_upper/2, '.toUpperCase()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    % String.trim()
    declare_binding(java, string_trim/2, '.trim()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    % String.contains(CharSequence)
    declare_binding(java, string_contains/2, '.contains',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    % String.startsWith(String)
    declare_binding(java, string_starts_with/2, '.startsWith',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    % String.endsWith(String)
    declare_binding(java, string_ends_with/2, '.endsWith',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    % String.split(String)
    declare_binding(java, string_split/3, '.split',
        [string, string], [array],
        [pure, deterministic, total, pattern(method_call)]),
    
    % String.replace(CharSequence, CharSequence)
    declare_binding(java, string_replace/4, '.replace',
        [string, string, string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    % String.substring(int)
    declare_binding(java, string_substring/3, '.substring',
        [string, int], [string],
        [pure, deterministic, partial, pattern(method_call)]),
    
    % String.substring(int, int)
    declare_binding(java, string_substring/4, '.substring',
        [string, int, int], [string],
        [pure, deterministic, partial, pattern(method_call)]).

% ============================================================================
% MATH OPERATION BINDINGS (java.lang.Math)
% ============================================================================

register_math_bindings :-
    % Math.abs(double)
    declare_binding(java, abs/2, 'Math.abs',
        [double], [double],
        [pure, deterministic, total]),
    
    % Math.max(double, double)
    declare_binding(java, max/3, 'Math.max',
        [double, double], [double],
        [pure, deterministic, total]),
    
    % Math.min(double, double)
    declare_binding(java, min/3, 'Math.min',
        [double, double], [double],
        [pure, deterministic, total]),
    
    % Math.sqrt(double)
    declare_binding(java, sqrt/2, 'Math.sqrt',
        [double], [double],
        [pure, deterministic, partial]),
    
    % Math.pow(double, double)
    declare_binding(java, pow/3, 'Math.pow',
        [double, double], [double],
        [pure, deterministic, total]),
    
    % Math.floor(double)
    declare_binding(java, floor/2, 'Math.floor',
        [double], [double],
        [pure, deterministic, total]),
    
    % Math.ceil(double)
    declare_binding(java, ceil/2, 'Math.ceil',
        [double], [double],
        [pure, deterministic, total]),
    
    % Math.round(double)
    declare_binding(java, round/2, 'Math.round',
        [double], [long],
        [pure, deterministic, total]),
    
    % Math.random()
    declare_binding(java, random/1, 'Math.random()',
        [], [double],
        [effect(random), deterministic, total]),
    
    % Math.PI
    declare_binding(java, pi/1, 'Math.PI',
        [], [double],
        [pure, deterministic, total]).

% ============================================================================
% COLLECTION BINDINGS (java.util.*)
% ============================================================================

register_collection_bindings :-
    % List.size()
    declare_binding(java, list_length/2, '.size()',
        [list], [int],
        [pure, deterministic, total, pattern(method_call)]),
    
    % List.get(int)
    declare_binding(java, list_get/3, '.get',
        [list, int], [object],
        [pure, deterministic, partial, pattern(method_call)]),
    
    % List.add(Object)
    declare_binding(java, list_add/2, '.add',
        [list, object], [],
        [effect(state), deterministic, total, pattern(method_call)]),
    
    % List.isEmpty()
    declare_binding(java, list_empty/1, '.isEmpty()',
        [list], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    % Map.get(Object)
    declare_binding(java, map_get/3, '.get',
        [map, object], [object],
        [pure, deterministic, total, pattern(method_call)]),
    
    % Map.put(Object, Object)
    declare_binding(java, map_put/3, '.put',
        [map, object, object], [],
        [effect(state), deterministic, total, pattern(method_call)]),
    
    % Map.containsKey(Object)
    declare_binding(java, map_contains_key/2, '.containsKey',
        [map, object], [boolean],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% I/O BINDINGS (java.io.*)
% ============================================================================

register_io_bindings :-
    % System.out.println(Object)
    declare_binding(java, println/1, 'System.out.println',
        [object], [],
        [effect(io), deterministic, total]),
    
    % System.err.println(Object)
    declare_binding(java, eprintln/1, 'System.err.println',
        [object], [],
        [effect(io), deterministic, total]).

% ============================================================================
% STREAM API BINDINGS (java.util.stream.*)
% ============================================================================

register_stream_bindings :-
    % stream.filter(Predicate)
    declare_binding(java, stream_filter/3, '.filter',
        [stream, predicate], [stream],
        [pure, deterministic, total, pattern(method_call), import('java.util.stream.*')]),
    
    % stream.map(Function)
    declare_binding(java, stream_map/3, '.map',
        [stream, function], [stream],
        [pure, deterministic, total, pattern(method_call), import('java.util.stream.*')]),
    
    % stream.collect(Collector)
    declare_binding(java, stream_collect/3, '.collect',
        [stream, collector], [object],
        [pure, deterministic, total, pattern(method_call), import('java.util.stream.*')]),
    
    % stream.reduce(identity, accumulator)
    declare_binding(java, stream_reduce/4, '.reduce',
        [stream, object, function], [object],
        [pure, deterministic, total, pattern(method_call), import('java.util.stream.*')]),
    
    % Collectors.toList()
    declare_binding(java, collectors_toList/1, 'Collectors.toList()',
        [], [collector],
        [pure, deterministic, total, import('java.util.stream.Collectors')]),
    
    % Collectors.toSet()
    declare_binding(java, collectors_toSet/1, 'Collectors.toSet()',
        [], [collector],
        [pure, deterministic, total, import('java.util.stream.Collectors')]).
