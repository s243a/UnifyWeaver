% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% jython_bindings.pl - Jython-specific bindings for UnifyWeaver
%
% Jython inherits most Python bindings but can use Java classes directly.
% This module:
%   1. Inherits from python_bindings for standard Python operations
%   2. Adds Java-specific bindings for performance/compatibility
%   3. Overrides certain bindings that work differently in Jython

:- encoding(utf8).

:- module(jython_bindings, [
    init_jython_bindings/0,
    jy_binding/5,               % Convenience: jy_binding(Pred, TargetName, Inputs, Outputs, Options)
    jy_binding_import/2,        % jy_binding_import(Pred, Import)
    test_jython_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_jython_bindings
%  Initialize all Jython bindings.
%  Inherits Python bindings then adds/overrides Jython-specific ones.
init_jython_bindings :-
    % First, inherit base Python bindings (they work in Jython)
    register_python_compatible_bindings,
    % Then add Jython-specific Java bindings
    register_java_io_bindings,
    register_java_string_bindings,
    register_java_collection_bindings,
    register_java_util_bindings,
    register_java_regex_bindings.

% ============================================================================
% CONVENIENCE PREDICATE
% ============================================================================

%% jy_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%  Query Jython bindings (Target=jython implied).
jy_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(jython, Pred, TargetName, Inputs, Outputs, Options).

%% jy_binding_import(?Pred, ?Import)
%  Get the import required for a Jython binding.
jy_binding_import(Pred, Import) :-
    jy_binding(Pred, _, _, _, Options),
    member(import(Import), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

%% :- jy_binding(Pred, TargetName, Inputs, Outputs, Options)
%  Directive for user-defined Jython bindings.
:- multifile user:term_expansion/2.

user:term_expansion(
    (:- jy_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(jython, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% PYTHON-COMPATIBLE BINDINGS
% These work the same in Jython as in Python
% ============================================================================

register_python_compatible_bindings :-
    % Core builtins
    declare_binding(jython, length/2, 'len',
        [sequence], [int],
        [pure, deterministic, total]),
    
    declare_binding(jython, to_string/2, 'str',
        [any], [string],
        [pure, deterministic, total]),
    
    declare_binding(jython, to_int/2, 'int',
        [any], [int],
        [pure, deterministic, partial]),
    
    declare_binding(jython, to_float/2, 'float',
        [any], [float],
        [pure, deterministic, partial]),
    
    % String operations (Python style)
    declare_binding(jython, string_split/2, '.split()',
        [string], [list],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, string_split/3, '.split',
        [string, string], [list],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, string_join/3, '.join',
        [string, list], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, string_replace/4, '.replace',
        [string, string, string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, string_strip/2, '.strip()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, string_lower/2, '.lower()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, string_upper/2, '.upper()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),
    
    % List operations
    declare_binding(jython, list_append/2, '.append',
        [list, any], [],
        [effect(state), deterministic, pattern(method_call)]),
    
    declare_binding(jython, list_extend/2, '.extend',
        [list, iterable], [],
        [effect(state), deterministic, pattern(method_call)]),
    
    % Dict operations
    declare_binding(jython, dict_get/3, '.get',
        [dict, any], [any],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, dict_keys/2, '.keys()',
        [dict], [dict_keys],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, dict_items/2, '.items()',
        [dict], [dict_items],
        [pure, deterministic, total, pattern(method_call)]),
    
    % JSON (Python json module works in Jython)
    declare_binding(jython, json_loads/2, 'json.loads',
        [string], [any],
        [pure, deterministic, partial, import('json')]),
    
    declare_binding(jython, json_dumps/2, 'json.dumps',
        [any], [string],
        [pure, deterministic, total, import('json')]),
    
    % Math module
    declare_binding(jython, sqrt/2, 'math.sqrt',
        [number], [float],
        [pure, deterministic, partial, import('math')]),
    
    declare_binding(jython, floor/2, 'math.floor',
        [number], [int],
        [pure, deterministic, total, import('math')]),
    
    declare_binding(jython, ceil/2, 'math.ceil',
        [number], [int],
        [pure, deterministic, total, import('math')]),
    
    declare_binding(jython, pi/1, 'math.pi',
        [], [float],
        [pure, deterministic, total, import('math')]).

% ============================================================================
% JAVA I/O BINDINGS (Override Python I/O for better JVM integration)
% ============================================================================

register_java_io_bindings :-
    % BufferedReader for efficient line reading
    declare_binding(jython, java_buffered_reader/2, 'BufferedReader',
        [reader], [buffered_reader],
        [pure, deterministic, total,
         import('java.io.BufferedReader')]),
    
    declare_binding(jython, java_input_stream_reader/2, 'InputStreamReader',
        [input_stream], [reader],
        [pure, deterministic, total,
         import('java.io.InputStreamReader')]),
    
    declare_binding(jython, java_file_reader/2, 'FileReader',
        [string], [reader],
        [effect(io), deterministic, partial,
         import('java.io.FileReader')]),
    
    declare_binding(jython, java_read_line/2, '.readLine()',
        [buffered_reader], [string],
        [effect(io), deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, java_close/1, '.close()',
        [closeable], [],
        [effect(io), deterministic, total, pattern(method_call)]),
    
    % PrintWriter for output
    declare_binding(jython, java_print_writer/2, 'PrintWriter',
        [output_stream], [print_writer],
        [pure, deterministic, total,
         import('java.io.PrintWriter')]),
    
    declare_binding(jython, java_println/2, '.println',
        [print_writer, any], [],
        [effect(io), deterministic, total, pattern(method_call)]),
    
    % System streams
    declare_binding(jython, java_system_in/1, 'System.in',
        [], [input_stream],
        [pure, deterministic, total,
         import('java.lang.System')]),
    
    declare_binding(jython, java_system_out/1, 'System.out',
        [], [print_stream],
        [pure, deterministic, total,
         import('java.lang.System')]),
    
    declare_binding(jython, java_system_err/1, 'System.err',
        [], [print_stream],
        [pure, deterministic, total,
         import('java.lang.System')]).

% ============================================================================
% JAVA STRING BINDINGS (Optional Java alternatives)
% ============================================================================

register_java_string_bindings :-
    % java.lang.String methods (use when Java interop is preferred)
    declare_binding(jython, jstring_length/2, '.length()',
        [jstring], [int],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, jstring_substring/3, '.substring',
        [jstring, int], [jstring],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(jython, jstring_substring/4, '.substring',
        [jstring, int, int], [jstring],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(jython, jstring_index_of/3, '.indexOf',
        [jstring, string], [int],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, jstring_contains/2, '.contains',
        [jstring, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    % StringBuilder for efficient string building
    declare_binding(jython, jstring_builder/1, 'StringBuilder()',
        [], [string_builder],
        [pure, deterministic, total,
         import('java.lang.StringBuilder')]),
    
    declare_binding(jython, jstring_builder/2, 'StringBuilder',
        [string], [string_builder],
        [pure, deterministic, total,
         import('java.lang.StringBuilder')]),
    
    declare_binding(jython, jstring_builder_append/2, '.append',
        [string_builder, any], [string_builder],
        [effect(state), deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, jstring_builder_to_string/2, '.toString()',
        [string_builder], [string],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% JAVA COLLECTION BINDINGS
% ============================================================================

register_java_collection_bindings :-
    % ArrayList
    declare_binding(jython, java_array_list/1, 'ArrayList()',
        [], [array_list],
        [pure, deterministic, total,
         import('java.util.ArrayList')]),
    
    declare_binding(jython, java_array_list/2, 'ArrayList',
        [collection], [array_list],
        [pure, deterministic, total,
         import('java.util.ArrayList')]),
    
    declare_binding(jython, jlist_add/2, '.add',
        [jlist, any], [boolean],
        [effect(state), deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, jlist_get/3, '.get',
        [jlist, int], [any],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(jython, jlist_size/2, '.size()',
        [jlist], [int],
        [pure, deterministic, total, pattern(method_call)]),
    
    % HashMap
    declare_binding(jython, java_hash_map/1, 'HashMap()',
        [], [hash_map],
        [pure, deterministic, total,
         import('java.util.HashMap')]),
    
    declare_binding(jython, jmap_put/3, '.put',
        [jmap, any, any], [any],
        [effect(state), deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, jmap_get/3, '.get',
        [jmap, any], [any],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, jmap_contains_key/2, '.containsKey',
        [jmap, any], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    % HashSet
    declare_binding(jython, java_hash_set/1, 'HashSet()',
        [], [hash_set],
        [pure, deterministic, total,
         import('java.util.HashSet')]),
    
    declare_binding(jython, jset_add/2, '.add',
        [jset, any], [boolean],
        [effect(state), deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, jset_contains/2, '.contains',
        [jset, any], [boolean],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% JAVA UTIL BINDINGS
% ============================================================================

register_java_util_bindings :-
    % java.util.Arrays
    declare_binding(jython, java_arrays_sort/1, 'Arrays.sort',
        [array], [],
        [effect(state), deterministic, total,
         import('java.util.Arrays')]),
    
    declare_binding(jython, java_arrays_as_list/2, 'Arrays.asList',
        [array], [list],
        [pure, deterministic, total,
         import('java.util.Arrays')]),
    
    % java.util.Collections
    declare_binding(jython, java_collections_sort/1, 'Collections.sort',
        [jlist], [],
        [effect(state), deterministic, total,
         import('java.util.Collections')]),
    
    declare_binding(jython, java_collections_reverse/1, 'Collections.reverse',
        [jlist], [],
        [effect(state), deterministic, total,
         import('java.util.Collections')]),
    
    % Optional (Java 8+)
    declare_binding(jython, java_optional_of/2, 'Optional.of',
        [any], [optional],
        [pure, deterministic, total,
         import('java.util.Optional')]),
    
    declare_binding(jython, java_optional_empty/1, 'Optional.empty()',
        [], [optional],
        [pure, deterministic, total,
         import('java.util.Optional')]).

% ============================================================================
% JAVA REGEX BINDINGS (Override Python re module for Java Pattern/Matcher)
% ============================================================================

register_java_regex_bindings :-
    % java.util.regex.Pattern
    declare_binding(jython, java_pattern_compile/2, 'Pattern.compile',
        [string], [pattern],
        [pure, deterministic, total,
         import('java.util.regex.Pattern')]),
    
    declare_binding(jython, java_pattern_matcher/3, '.matcher',
        [pattern, string], [matcher],
        [pure, deterministic, total, pattern(method_call)]),
    
    % java.util.regex.Matcher
    declare_binding(jython, java_matcher_find/1, '.find()',
        [matcher], [boolean],
        [effect(state), deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, java_matcher_matches/1, '.matches()',
        [matcher], [boolean],
        [pure, deterministic, total, pattern(method_call)]),
    
    declare_binding(jython, java_matcher_group/2, '.group()',
        [matcher], [string],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(jython, java_matcher_group/3, '.group',
        [matcher, int], [string],
        [pure, deterministic, partial, pattern(method_call)]),
    
    declare_binding(jython, java_matcher_replace_all/3, '.replaceAll',
        [matcher, string], [string],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% TESTS
% ============================================================================

test_jython_bindings :-
    format('~n=== Jython Bindings Tests ===~n~n'),

    % Initialize bindings
    format('[Test 1] Initializing Jython bindings~n'),
    init_jython_bindings,
    format('  [PASS] Jython bindings initialized~n'),

    % Test Python-compatible bindings
    format('~n[Test 2] Checking Python-compatible bindings~n'),
    (   jy_binding(length/2, 'len', _, _, _)
    ->  format('  [PASS] length/2 -> len binding exists~n')
    ;   format('  [FAIL] length/2 binding missing~n')
    ),
    
    % Test Java I/O bindings
    format('~n[Test 3] Checking Java I/O bindings~n'),
    (   jy_binding(java_buffered_reader/2, 'BufferedReader', _, _, Opts1),
        member(import('java.io.BufferedReader'), Opts1)
    ->  format('  [PASS] BufferedReader binding with import~n')
    ;   format('  [FAIL] BufferedReader binding missing~n')
    ),
    
    % Test Java collection bindings
    format('~n[Test 4] Checking Java collection bindings~n'),
    (   jy_binding(java_array_list/1, _, _, _, Opts2),
        member(import('java.util.ArrayList'), Opts2)
    ->  format('  [PASS] ArrayList binding with import~n')
    ;   format('  [FAIL] ArrayList binding missing~n')
    ),
    
    % Test Java regex bindings
    format('~n[Test 5] Checking Java regex bindings~n'),
    (   jy_binding(java_pattern_compile/2, 'Pattern.compile', _, _, Opts3),
        member(import('java.util.regex.Pattern'), Opts3)
    ->  format('  [PASS] Pattern.compile binding with import~n')
    ;   format('  [FAIL] Pattern.compile binding missing~n')
    ),
    
    % Count total bindings
    format('~n[Test 6] Counting total bindings~n'),
    findall(P, jy_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('  [INFO] Total Jython bindings: ~w~n', [Count]),

    format('~n=== Jython Bindings Tests Complete ===~n').
