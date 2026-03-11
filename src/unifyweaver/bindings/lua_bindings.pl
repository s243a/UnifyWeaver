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
    register_table_bindings.

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
        [effect(io), deterministic, total]).

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
        [pure, deterministic, total]).

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
        [pure, deterministic, total]).

% ============================================================================
% TABLE BINDINGS
% ============================================================================

register_table_bindings :-
    declare_binding(lua, length/2, '#', % length operator for tables/strings
        [any], [int],
        [pure, deterministic, total]).

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
