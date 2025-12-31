% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% typescript_bindings.pl - TypeScript/JavaScript-specific bindings
%
% This module defines bindings for TypeScript/Node.js target language features.
% Maps Prolog predicates to JavaScript/TypeScript stdlib functions.
%
% Categories:
%   - Core Built-ins (Array, Object, etc.)
%   - String Operations (String methods)
%   - Math Operations (Math object)
%   - Array Operations (Array methods)
%   - Object Operations (Object methods)
%   - JSON Operations (JSON object)
%   - Console/I/O Operations
%   - Node.js specific (fs, path, etc.)
%   - Promise/Async Operations
%
% Runtime Targets:
%   - browser: Standard browser JavaScript
%   - node: Node.js runtime
%   - deno: Deno runtime
%   - bun: Bun runtime

:- module(typescript_bindings, [
    init_typescript_bindings/0,
    ts_binding/5,               % Convenience: ts_binding(Pred, TargetName, Inputs, Outputs, Options)
    ts_binding_import/2,        % ts_binding_import(Pred, Import) - get required import
    test_typescript_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_typescript_bindings
%
%  Initialize all TypeScript bindings. Call this before using the compiler.
%
init_typescript_bindings :-
    register_builtin_bindings,
    register_string_bindings,
    register_math_bindings,
    register_array_bindings,
    register_object_bindings,
    register_json_bindings,
    register_console_bindings,
    register_node_fs_bindings,
    register_node_path_bindings,
    register_promise_bindings,
    register_date_bindings.

% ============================================================================
% CONVENIENCE PREDICATES
% ============================================================================

%% ts_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query TypeScript bindings with reduced arity (Target=typescript implied).
%
ts_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(typescript, Pred, TargetName, Inputs, Outputs, Options).

%% ts_binding_import(?Pred, ?Import)
%
%  Get the import required for a TypeScript binding.
%
ts_binding_import(Pred, Import) :-
    ts_binding(Pred, _, _, _, Options),
    member(import(Import), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

:- multifile user:term_expansion/2.

user:term_expansion(
    (:- ts_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(typescript, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% CORE BUILT-IN BINDINGS
% ============================================================================

register_builtin_bindings :-
    % -------------------------------------------
    % typeof - get runtime type
    % -------------------------------------------
    declare_binding(typescript, typeof/2, 'typeof',
        [any], [string],
        [pure, deterministic, total]),

    % -------------------------------------------
    % instanceof - check prototype chain
    % -------------------------------------------
    declare_binding(typescript, instanceof/3, 'instanceof',
        [any, constructor], [boolean],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Boolean coercion
    % -------------------------------------------
    declare_binding(typescript, to_boolean/2, 'Boolean',
        [any], [boolean],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Number coercion
    % -------------------------------------------
    declare_binding(typescript, to_number/2, 'Number',
        [any], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, parse_int/3, 'parseInt',
        [string, int], [int],
        [pure, deterministic, total]),

    declare_binding(typescript, parse_float/2, 'parseFloat',
        [string], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, is_nan/2, 'isNaN',
        [any], [boolean],
        [pure, deterministic, total]),

    declare_binding(typescript, is_finite/2, 'isFinite',
        [any], [boolean],
        [pure, deterministic, total]),

    % -------------------------------------------
    % String coercion
    % -------------------------------------------
    declare_binding(typescript, to_string/2, 'String',
        [any], [string],
        [pure, deterministic, total]).

% ============================================================================
% STRING OPERATION BINDINGS
% ============================================================================

register_string_bindings :-
    % -------------------------------------------
    % String Properties
    % -------------------------------------------
    declare_binding(typescript, string_length/2, '.length',
        [string], [number],
        [pure, deterministic, total, pattern(property_access)]),

    % -------------------------------------------
    % String Searching
    % -------------------------------------------
    declare_binding(typescript, string_includes/3, '.includes',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_starts_with/3, '.startsWith',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_ends_with/3, '.endsWith',
        [string, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_index_of/3, '.indexOf',
        [string, string], [number],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_last_index_of/3, '.lastIndexOf',
        [string, string], [number],
        [pure, deterministic, total, pattern(method_call)]),

    % -------------------------------------------
    % String Transformation
    % -------------------------------------------
    declare_binding(typescript, string_upper/2, '.toUpperCase()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_lower/2, '.toLowerCase()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_trim/2, '.trim()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_trim_start/2, '.trimStart()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_trim_end/2, '.trimEnd()',
        [string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_pad_start/4, '.padStart',
        [string, number, string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_pad_end/4, '.padEnd',
        [string, number, string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_repeat/3, '.repeat',
        [string, number], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_replace/4, '.replace',
        [string, string, string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_replace_all/4, '.replaceAll',
        [string, string, string], [string],
        [pure, deterministic, total, pattern(method_call)]),

    % -------------------------------------------
    % String Extraction
    % -------------------------------------------
    declare_binding(typescript, string_slice/4, '.slice',
        [string, number, number], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_substring/4, '.substring',
        [string, number, number], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_char_at/3, '.charAt',
        [string, number], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_char_code_at/3, '.charCodeAt',
        [string, number], [number],
        [pure, deterministic, total, pattern(method_call)]),

    % -------------------------------------------
    % String Splitting/Joining
    % -------------------------------------------
    declare_binding(typescript, string_split/3, '.split',
        [string, string], [array],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, string_from_char_code/2, 'String.fromCharCode',
        [number], [string],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Template Literal support (generated differently)
    % -------------------------------------------
    declare_binding(typescript, string_concat/3, '+',
        [string, string], [string],
        [pure, deterministic, total, pattern(operator)]).

% ============================================================================
% MATH OPERATION BINDINGS
% ============================================================================

register_math_bindings :-
    % -------------------------------------------
    % Basic Operations
    % -------------------------------------------
    declare_binding(typescript, abs/2, 'Math.abs',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, ceil/2, 'Math.ceil',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, floor/2, 'Math.floor',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, round/2, 'Math.round',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, trunc/2, 'Math.trunc',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, sign/2, 'Math.sign',
        [number], [number],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Power and Root
    % -------------------------------------------
    declare_binding(typescript, sqrt/2, 'Math.sqrt',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, cbrt/2, 'Math.cbrt',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, pow/3, 'Math.pow',
        [number, number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, exp/2, 'Math.exp',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, expm1/2, 'Math.expm1',
        [number], [number],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Logarithmic
    % -------------------------------------------
    declare_binding(typescript, log/2, 'Math.log',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, log10/2, 'Math.log10',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, log2/2, 'Math.log2',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, log1p/2, 'Math.log1p',
        [number], [number],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Trigonometric
    % -------------------------------------------
    declare_binding(typescript, sin/2, 'Math.sin',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, cos/2, 'Math.cos',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, tan/2, 'Math.tan',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, asin/2, 'Math.asin',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, acos/2, 'Math.acos',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, atan/2, 'Math.atan',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, atan2/3, 'Math.atan2',
        [number, number], [number],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Hyperbolic
    % -------------------------------------------
    declare_binding(typescript, sinh/2, 'Math.sinh',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, cosh/2, 'Math.cosh',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, tanh/2, 'Math.tanh',
        [number], [number],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Min/Max
    % -------------------------------------------
    declare_binding(typescript, min/3, 'Math.min',
        [number, number], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, max/3, 'Math.max',
        [number, number], [number],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Random
    % -------------------------------------------
    declare_binding(typescript, random/1, 'Math.random()',
        [], [number],
        [effect(nondeterministic), deterministic, total]),

    % -------------------------------------------
    % Constants (accessed as properties)
    % -------------------------------------------
    declare_binding(typescript, math_pi/1, 'Math.PI',
        [], [number],
        [pure, deterministic, total, pattern(constant)]),

    declare_binding(typescript, math_e/1, 'Math.E',
        [], [number],
        [pure, deterministic, total, pattern(constant)]),

    declare_binding(typescript, math_ln2/1, 'Math.LN2',
        [], [number],
        [pure, deterministic, total, pattern(constant)]),

    declare_binding(typescript, math_ln10/1, 'Math.LN10',
        [], [number],
        [pure, deterministic, total, pattern(constant)]).

% ============================================================================
% ARRAY OPERATION BINDINGS
% ============================================================================

register_array_bindings :-
    % -------------------------------------------
    % Array Properties
    % -------------------------------------------
    declare_binding(typescript, array_length/2, '.length',
        [array], [number],
        [pure, deterministic, total, pattern(property_access)]),

    % -------------------------------------------
    % Array Creation
    % -------------------------------------------
    declare_binding(typescript, array_from/2, 'Array.from',
        [iterable], [array],
        [pure, deterministic, total]),

    declare_binding(typescript, array_of/2, 'Array.of',
        [variadic], [array],
        [pure, deterministic, total]),

    declare_binding(typescript, array_is_array/2, 'Array.isArray',
        [any], [boolean],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Array Mutation (impure)
    % -------------------------------------------
    declare_binding(typescript, array_push/3, '.push',
        [array, any], [number],
        [effect(mutation), deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_pop/2, '.pop()',
        [array], [any],
        [effect(mutation), deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_shift/2, '.shift()',
        [array], [any],
        [effect(mutation), deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_unshift/3, '.unshift',
        [array, any], [number],
        [effect(mutation), deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_splice/5, '.splice',
        [array, number, number, variadic], [array],
        [effect(mutation), deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_sort/2, '.sort()',
        [array], [array],
        [effect(mutation), deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_reverse/2, '.reverse()',
        [array], [array],
        [effect(mutation), deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_fill/4, '.fill',
        [array, any, number, number], [array],
        [effect(mutation), deterministic, total, pattern(method_call)]),

    % -------------------------------------------
    % Array Transformation (pure - returns new array)
    % -------------------------------------------
    declare_binding(typescript, array_map/3, '.map',
        [array, function], [array],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_filter/3, '.filter',
        [array, function], [array],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_reduce/4, '.reduce',
        [array, function, any], [any],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_reduce_right/4, '.reduceRight',
        [array, function, any], [any],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_flat/2, '.flat()',
        [array], [array],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_flat_map/3, '.flatMap',
        [array, function], [array],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_slice/4, '.slice',
        [array, number, number], [array],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_concat/3, '.concat',
        [array, array], [array],
        [pure, deterministic, total, pattern(method_call)]),

    % -------------------------------------------
    % Array Searching
    % -------------------------------------------
    declare_binding(typescript, array_find/3, '.find',
        [array, function], [any],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_find_index/3, '.findIndex',
        [array, function], [number],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_index_of/3, '.indexOf',
        [array, any], [number],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_last_index_of/3, '.lastIndexOf',
        [array, any], [number],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_includes/3, '.includes',
        [array, any], [boolean],
        [pure, deterministic, total, pattern(method_call)]),

    % -------------------------------------------
    % Array Testing
    % -------------------------------------------
    declare_binding(typescript, array_every/3, '.every',
        [array, function], [boolean],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, array_some/3, '.some',
        [array, function], [boolean],
        [pure, deterministic, total, pattern(method_call)]),

    % -------------------------------------------
    % Array Iteration
    % -------------------------------------------
    declare_binding(typescript, array_for_each/3, '.forEach',
        [array, function], [void],
        [effect(side_effect), deterministic, total, pattern(method_call)]),

    % -------------------------------------------
    % Array Joining
    % -------------------------------------------
    declare_binding(typescript, array_join/3, '.join',
        [array, string], [string],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% OBJECT OPERATION BINDINGS
% ============================================================================

register_object_bindings :-
    % -------------------------------------------
    % Object Static Methods
    % -------------------------------------------
    declare_binding(typescript, object_keys/2, 'Object.keys',
        [object], [array],
        [pure, deterministic, total]),

    declare_binding(typescript, object_values/2, 'Object.values',
        [object], [array],
        [pure, deterministic, total]),

    declare_binding(typescript, object_entries/2, 'Object.entries',
        [object], [array],
        [pure, deterministic, total]),

    declare_binding(typescript, object_from_entries/2, 'Object.fromEntries',
        [array], [object],
        [pure, deterministic, total]),

    declare_binding(typescript, object_assign/3, 'Object.assign',
        [object, object], [object],
        [effect(mutation), deterministic, total]),

    declare_binding(typescript, object_freeze/2, 'Object.freeze',
        [object], [object],
        [effect(mutation), deterministic, total]),

    declare_binding(typescript, object_is_frozen/2, 'Object.isFrozen',
        [object], [boolean],
        [pure, deterministic, total]),

    declare_binding(typescript, object_seal/2, 'Object.seal',
        [object], [object],
        [effect(mutation), deterministic, total]),

    declare_binding(typescript, object_is_sealed/2, 'Object.isSealed',
        [object], [boolean],
        [pure, deterministic, total]),

    % -------------------------------------------
    % Property checking
    % -------------------------------------------
    declare_binding(typescript, has_own_property/3, '.hasOwnProperty',
        [object, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, property_is_enumerable/3, '.propertyIsEnumerable',
        [object, string], [boolean],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% JSON OPERATION BINDINGS
% ============================================================================

register_json_bindings :-
    declare_binding(typescript, json_parse/2, 'JSON.parse',
        [string], [any],
        [pure, deterministic, partial, effect(throws)]),

    declare_binding(typescript, json_stringify/2, 'JSON.stringify',
        [any], [string],
        [pure, deterministic, total]),

    declare_binding(typescript, json_stringify_pretty/3, 'JSON.stringify',
        [any, null, number], [string],
        [pure, deterministic, total]).

% ============================================================================
% CONSOLE/I/O BINDINGS
% ============================================================================

register_console_bindings :-
    declare_binding(typescript, console_log/1, 'console.log',
        [any], [],
        [effect(io), deterministic, total]),

    declare_binding(typescript, console_error/1, 'console.error',
        [any], [],
        [effect(io), deterministic, total]),

    declare_binding(typescript, console_warn/1, 'console.warn',
        [any], [],
        [effect(io), deterministic, total]),

    declare_binding(typescript, console_info/1, 'console.info',
        [any], [],
        [effect(io), deterministic, total]),

    declare_binding(typescript, console_debug/1, 'console.debug',
        [any], [],
        [effect(io), deterministic, total]),

    declare_binding(typescript, console_table/1, 'console.table',
        [any], [],
        [effect(io), deterministic, total]),

    declare_binding(typescript, console_time/1, 'console.time',
        [string], [],
        [effect(io), deterministic, total]),

    declare_binding(typescript, console_time_end/1, 'console.timeEnd',
        [string], [],
        [effect(io), deterministic, total]).

% ============================================================================
% NODE.JS FILE SYSTEM BINDINGS
% ============================================================================

register_node_fs_bindings :-
    % Synchronous operations
    declare_binding(typescript, read_file_sync/2, 'fs.readFileSync',
        [string], [string],
        [effect(io), deterministic, partial, import('fs')]),

    declare_binding(typescript, write_file_sync/3, 'fs.writeFileSync',
        [string, string], [void],
        [effect(io), deterministic, partial, import('fs')]),

    declare_binding(typescript, exists_sync/2, 'fs.existsSync',
        [string], [boolean],
        [effect(io), deterministic, total, import('fs')]),

    declare_binding(typescript, mkdir_sync/2, 'fs.mkdirSync',
        [string], [void],
        [effect(io), deterministic, partial, import('fs')]),

    declare_binding(typescript, rmdir_sync/2, 'fs.rmdirSync',
        [string], [void],
        [effect(io), deterministic, partial, import('fs')]),

    declare_binding(typescript, unlink_sync/2, 'fs.unlinkSync',
        [string], [void],
        [effect(io), deterministic, partial, import('fs')]),

    declare_binding(typescript, readdir_sync/2, 'fs.readdirSync',
        [string], [array],
        [effect(io), deterministic, partial, import('fs')]),

    declare_binding(typescript, stat_sync/2, 'fs.statSync',
        [string], [object],
        [effect(io), deterministic, partial, import('fs')]),

    % Promise-based operations
    declare_binding(typescript, read_file/2, 'fs.promises.readFile',
        [string], [promise],
        [effect(io), deterministic, partial, import('fs'), async]),

    declare_binding(typescript, write_file/3, 'fs.promises.writeFile',
        [string, string], [promise],
        [effect(io), deterministic, partial, import('fs'), async]).

% ============================================================================
% NODE.JS PATH BINDINGS
% ============================================================================

register_node_path_bindings :-
    declare_binding(typescript, path_join/3, 'path.join',
        [string, string], [string],
        [pure, deterministic, total, import('path')]),

    declare_binding(typescript, path_resolve/2, 'path.resolve',
        [variadic], [string],
        [pure, deterministic, total, import('path')]),

    declare_binding(typescript, path_dirname/2, 'path.dirname',
        [string], [string],
        [pure, deterministic, total, import('path')]),

    declare_binding(typescript, path_basename/2, 'path.basename',
        [string], [string],
        [pure, deterministic, total, import('path')]),

    declare_binding(typescript, path_extname/2, 'path.extname',
        [string], [string],
        [pure, deterministic, total, import('path')]),

    declare_binding(typescript, path_parse/2, 'path.parse',
        [string], [object],
        [pure, deterministic, total, import('path')]),

    declare_binding(typescript, path_format/2, 'path.format',
        [object], [string],
        [pure, deterministic, total, import('path')]),

    declare_binding(typescript, path_is_absolute/2, 'path.isAbsolute',
        [string], [boolean],
        [pure, deterministic, total, import('path')]),

    declare_binding(typescript, path_normalize/2, 'path.normalize',
        [string], [string],
        [pure, deterministic, total, import('path')]),

    declare_binding(typescript, path_relative/3, 'path.relative',
        [string, string], [string],
        [pure, deterministic, total, import('path')]).

% ============================================================================
% PROMISE/ASYNC BINDINGS
% ============================================================================

register_promise_bindings :-
    declare_binding(typescript, promise_resolve/2, 'Promise.resolve',
        [any], [promise],
        [pure, deterministic, total]),

    declare_binding(typescript, promise_reject/2, 'Promise.reject',
        [any], [promise],
        [pure, deterministic, total]),

    declare_binding(typescript, promise_all/2, 'Promise.all',
        [array], [promise],
        [pure, deterministic, total]),

    declare_binding(typescript, promise_all_settled/2, 'Promise.allSettled',
        [array], [promise],
        [pure, deterministic, total]),

    declare_binding(typescript, promise_race/2, 'Promise.race',
        [array], [promise],
        [pure, deterministic, total]),

    declare_binding(typescript, promise_any/2, 'Promise.any',
        [array], [promise],
        [pure, deterministic, total]).

% ============================================================================
% DATE BINDINGS
% ============================================================================

register_date_bindings :-
    declare_binding(typescript, date_now/1, 'Date.now()',
        [], [number],
        [effect(clock), deterministic, total]),

    declare_binding(typescript, date_parse/2, 'Date.parse',
        [string], [number],
        [pure, deterministic, total]),

    declare_binding(typescript, new_date/1, 'new Date()',
        [], [date],
        [effect(clock), deterministic, total]),

    declare_binding(typescript, new_date_from/2, 'new Date',
        [number], [date],
        [pure, deterministic, total]),

    declare_binding(typescript, date_to_iso_string/2, '.toISOString()',
        [date], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, date_to_locale_string/2, '.toLocaleString()',
        [date], [string],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, date_get_time/2, '.getTime()',
        [date], [number],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, date_get_full_year/2, '.getFullYear()',
        [date], [number],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, date_get_month/2, '.getMonth()',
        [date], [number],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, date_get_date/2, '.getDate()',
        [date], [number],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, date_get_hours/2, '.getHours()',
        [date], [number],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, date_get_minutes/2, '.getMinutes()',
        [date], [number],
        [pure, deterministic, total, pattern(method_call)]),

    declare_binding(typescript, date_get_seconds/2, '.getSeconds()',
        [date], [number],
        [pure, deterministic, total, pattern(method_call)]).

% ============================================================================
% TESTING
% ============================================================================

test_typescript_bindings :-
    format('Testing TypeScript bindings...~n'),

    % Test initialization
    init_typescript_bindings,
    format('  [PASS] Bindings initialized~n'),

    % Test querying bindings
    (   ts_binding(sqrt/2, 'Math.sqrt', [number], [number], _)
    ->  format('  [PASS] sqrt/2 binding found~n')
    ;   format('  [FAIL] sqrt/2 binding not found~n')
    ),

    (   ts_binding(string_length/2, '.length', [string], [number], _)
    ->  format('  [PASS] string_length/2 binding found~n')
    ;   format('  [FAIL] string_length/2 binding not found~n')
    ),

    % Test import lookup
    (   ts_binding_import(read_file_sync/2, 'fs')
    ->  format('  [PASS] fs import found for read_file_sync~n')
    ;   format('  [FAIL] fs import not found for read_file_sync~n')
    ),

    (   ts_binding_import(path_join/3, 'path')
    ->  format('  [PASS] path import found for path_join~n')
    ;   format('  [FAIL] path import not found for path_join~n')
    ),

    % Count bindings
    findall(P, ts_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('  Total bindings registered: ~w~n', [Count]),

    format('TypeScript bindings test complete.~n').
