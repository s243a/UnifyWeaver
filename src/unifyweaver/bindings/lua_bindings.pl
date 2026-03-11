% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% lua_bindings.pl - Lua-specific bindings
%
% This module defines bindings for Lua target language features.
%
% Categories:
%   - Core Built-ins
%   - Math Operations
%   - String Operations
%   - Table Operations

:- module(lua_bindings, [
    init_lua_bindings/0,
    lua_binding/5,               % Convenience: lua_binding(Pred, TargetName, Inputs, Outputs, Options)
    test_lua_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_lua_bindings
%
%  Initialize all Lua bindings. Call this before using the compiler.
%
init_lua_bindings :-
    register_builtin_bindings,
    register_math_bindings,
    register_string_bindings,
    register_table_bindings,
    register_io_bindings,
    register_type_bindings,
    register_os_bindings.

% ============================================================================
% CONVENIENCE PREDICATES
% ============================================================================

%% lua_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query Lua bindings with reduced arity (Target=lua implied).
%
lua_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(lua, Pred, TargetName, Inputs, Outputs, Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

:- multifile user:term_expansion/2.

user:term_expansion(
    (:- lua_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(lua, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% CORE BUILT-IN BINDINGS
% ============================================================================

register_builtin_bindings :-
    % -------------------------------------------
    % Conditionals / Tests
    % -------------------------------------------
    declare_binding(lua, true/0, 'true',
        [], [],
        [pure, deterministic, total, pattern(command)]),

    declare_binding(lua, fail/0, 'false',
        [], [],
        [pure, deterministic, total, pattern(command)]),

    % -------------------------------------------
    % I/O
    % -------------------------------------------
    declare_binding(lua, print/1, 'print',
        [any], [],
        [effect(io), deterministic, total]),

    declare_binding(lua, write/1, 'io.write',
        [any], [],
        [effect(io), deterministic, total]),

    % -------------------------------------------
    % Error handling
    % -------------------------------------------
    declare_binding(lua, error/1, 'error',
        [any], [],
        [effect(error), deterministic, total]),

    declare_binding(lua, assert_true/1, 'assert',
        [any], [],
        [effect(error), deterministic, partial]),

    declare_binding(lua, pcall/3, 'pcall',
        [function, any], [any],
        [effect(error), deterministic, total, note(variadic)]),

    % -------------------------------------------
    % Module system
    % -------------------------------------------
    declare_binding(lua, require/2, 'require',
        [string], [any],
        [effect(io), deterministic, total]),

    % -------------------------------------------
    % Iteration
    % -------------------------------------------
    declare_binding(lua, pairs/2, 'pairs',
        [table], [iterator],
        [pure, deterministic, total]),

    declare_binding(lua, ipairs/2, 'ipairs',
        [table], [iterator],
        [pure, deterministic, total]),

    declare_binding(lua, next/3, 'next',
        [table, any], [any],
        [pure, deterministic, total]),

    declare_binding(lua, select/3, 'select',
        [any, any], [any],
        [pure, deterministic, total]),

    declare_binding(lua, unpack/2, 'table.unpack',
        [table], [any],
        [pure, deterministic, total]).

% ============================================================================
% MATH BINDINGS
% ============================================================================

register_math_bindings :-
    declare_binding(lua, abs/2, 'math.abs',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, floor/2, 'math.floor',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, ceil/2, 'math.ceil',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, max/3, 'math.max',
        [number, number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, min/3, 'math.min',
        [number, number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, sqrt/2, 'math.sqrt',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, log/2, 'math.log',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, log/3, 'math.log',
        [number, number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, sin/2, 'math.sin',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, cos/2, 'math.cos',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, tan/2, 'math.tan',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, exp/2, 'math.exp',
        [number], [number],
        [pure, deterministic, total]),

    declare_binding(lua, random/1, 'math.random',
        [], [number],
        [effect(random), deterministic, total]),

    declare_binding(lua, random/3, 'math.random',
        [number, number], [number],
        [effect(random), deterministic, total]),

    declare_binding(lua, randomseed/1, 'math.randomseed',
        [number], [],
        [effect(random), deterministic, total]).

% ============================================================================
% STRING BINDINGS
% ============================================================================

register_string_bindings :-
    declare_binding(lua, string_length/2, 'string.len',
        [string], [int],
        [pure, deterministic, total]),

    declare_binding(lua, string_upper/2, 'string.upper',
        [string], [string],
        [pure, deterministic, total]),

    declare_binding(lua, string_lower/2, 'string.lower',
        [string], [string],
        [pure, deterministic, total]),

    declare_binding(lua, string_sub/4, 'string.sub',
        [string, int, int], [string],
        [pure, deterministic, total]),

    declare_binding(lua, string_find/4, 'string.find',
        [string, string], [int, int],
        [pure, deterministic, partial]),

    declare_binding(lua, string_match/3, 'string.match',
        [string, string], [string],
        [pure, deterministic, partial]),

    declare_binding(lua, string_format/3, 'string.format',
        [string, any], [string],
        [pure, deterministic, total]),

    declare_binding(lua, string_rep/3, 'string.rep',
        [string, int], [string],
        [pure, deterministic, total]),

    declare_binding(lua, string_byte/2, 'string.byte',
        [string], [int],
        [pure, deterministic, total]),

    declare_binding(lua, string_char/2, 'string.char',
        [int], [string],
        [pure, deterministic, total]),

    declare_binding(lua, string_reverse/2, 'string.reverse',
        [string], [string],
        [pure, deterministic, total]),

    declare_binding(lua, string_gsub/4, 'string.gsub',
        [string, string, string], [string],
        [pure, deterministic, total]),

    declare_binding(lua, string_gmatch/3, 'string.gmatch',
        [string, string], [iterator],
        [pure, deterministic, total]).

% ============================================================================
% TABLE BINDINGS
% ============================================================================

register_table_bindings :-
    declare_binding(lua, length/2, '#',
        [any], [int],
        [pure, deterministic, total]),

    declare_binding(lua, table_insert/2, 'table.insert',
        [table, any], [],
        [effect(mutation), deterministic, total]),

    declare_binding(lua, table_insert/3, 'table.insert',
        [table, int, any], [],
        [effect(mutation), deterministic, total]),

    declare_binding(lua, table_remove/2, 'table.remove',
        [table], [any],
        [effect(mutation), deterministic, total]),

    declare_binding(lua, table_remove/3, 'table.remove',
        [table, int], [any],
        [effect(mutation), deterministic, total]),

    declare_binding(lua, table_sort/1, 'table.sort',
        [table], [],
        [effect(mutation), deterministic, total]),

    declare_binding(lua, table_sort/2, 'table.sort',
        [table, function], [],
        [effect(mutation), deterministic, total]),

    declare_binding(lua, table_concat/3, 'table.concat',
        [table, string], [string],
        [pure, deterministic, total]),

    declare_binding(lua, table_concat/2, 'table.concat',
        [table], [string],
        [pure, deterministic, total]),

    declare_binding(lua, table_move/6, 'table.move',
        [table, int, int, int, table], [table],
        [effect(mutation), deterministic, total]).

% ============================================================================
% I/O BINDINGS
% ============================================================================

register_io_bindings :-
    declare_binding(lua, io_open/3, 'io.open',
        [string, string], [any],
        [effect(io), deterministic, partial]),

    declare_binding(lua, io_read/2, 'io.read',
        [string], [string],
        [effect(io), deterministic, partial]),

    declare_binding(lua, io_write/1, 'io.write',
        [any], [],
        [effect(io), deterministic, total]),

    declare_binding(lua, io_close/1, 'io.close',
        [any], [],
        [effect(io), deterministic, total]),

    declare_binding(lua, io_lines/2, 'io.lines',
        [string], [iterator],
        [effect(io), deterministic, total]).

% ============================================================================
% TYPE CONVERSION BINDINGS
% ============================================================================

register_type_bindings :-
    declare_binding(lua, tonumber/2, 'tonumber',
        [any], [number],
        [pure, deterministic, partial]),

    declare_binding(lua, tostring/2, 'tostring',
        [any], [string],
        [pure, deterministic, total]),

    declare_binding(lua, type/2, 'type',
        [any], [string],
        [pure, deterministic, total]).

% ============================================================================
% OS BINDINGS
% ============================================================================

register_os_bindings :-
    declare_binding(lua, os_time/1, 'os.time',
        [], [number],
        [effect(time), deterministic, total]),

    declare_binding(lua, os_clock/1, 'os.clock',
        [], [number],
        [effect(time), deterministic, total]),

    declare_binding(lua, os_date/2, 'os.date',
        [string], [string],
        [effect(time), deterministic, total]),

    declare_binding(lua, os_execute/2, 'os.execute',
        [string], [any],
        [effect(process), deterministic, total]),

    declare_binding(lua, os_getenv/2, 'os.getenv',
        [string], [string],
        [effect(env), deterministic, partial]).

% ============================================================================
% TESTING
% ============================================================================

test_lua_bindings :-
    format('~n=== Lua Bindings Tests ===~n~n'),
    (   lua_binding(print/1, TargetName, _, _, _),
        TargetName == 'print'
    ->  format('  PASS: print/1 binding found~n')
    ;   format('  FAIL: print/1 binding not found~n')
    ),
    format('~n=== Tests Complete ===~n').
