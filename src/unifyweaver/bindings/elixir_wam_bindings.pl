:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% elixir_wam_bindings.pl - Elixir bindings for WAM runtime transpilation
%
% Maps WAM runtime operations to Elixir expressions.
% Used by wam_elixir_target.pl.
%
% Categories:
%   - Map operations (WAM registers → Elixir %{})
%   - List operations (WAM heap/trail → Elixir lists)
%   - Arithmetic
%   - Type checks
%   - String/Atom operations

:- module(elixir_wam_bindings, [
    init_elixir_wam_bindings/0,
    elixir_wam_binding/5,        % +PrologPred, -ElixirExpr, -ArgTypes, -RetType, -Props
    elixir_wam_type_map/2         % +WAMType, -ElixirType
]).

:- use_module('../core/binding_registry').

% ============================================================================
% TYPE MAPPING
% ============================================================================

elixir_wam_type_map(assoc, '%{}').
elixir_wam_type_map(list, 'list()').
elixir_wam_type_map(value, 'any()').
elixir_wam_type_map(atom, 'String.t()').
elixir_wam_type_map(integer, 'integer()').
elixir_wam_type_map(float, 'float()').
elixir_wam_type_map(bool, 'boolean()').
elixir_wam_type_map(void, ':ok').
elixir_wam_type_map(trail_entry, '{String.t(), any()}').
elixir_wam_type_map(choice_point, 'map()').
elixir_wam_type_map(wam_state, '%WamState{}').

% ============================================================================
% BINDINGS: Map operations (WAM registers)
% ============================================================================

elixir_wam_binding(get_assoc/3,
    'Map.get(~map, ~key)',
    [map-assoc, key-atom], [value],
    [pure, pattern(function_call)]).

elixir_wam_binding(put_assoc/4,
    'Map.put(~map, ~key, ~val)',
    [map-assoc, key-atom, val-value], [assoc],
    [pure, pattern(function_call)]).

elixir_wam_binding(del_assoc/3,
    'Map.delete(~map, ~key)',
    [map-assoc, key-atom], [assoc],
    [pure, pattern(function_call)]).

elixir_wam_binding(assoc_contains/2,
    'Map.has_key?(~map, ~key)',
    [map-assoc, key-atom], [bool],
    [pure, pattern(function_call)]).

elixir_wam_binding(empty_assoc/1,
    '%{}',
    [], [assoc],
    [pure, pattern(literal)]).

% ============================================================================
% BINDINGS: List operations (WAM heap, trail, stack)
% ============================================================================

elixir_wam_binding(list_append/3,
    '~list ++ [~elem]',
    [list-list, elem-value], [list],
    [pure, pattern(operator)]).

elixir_wam_binding(list_at/3,
    'Enum.at(~list, ~idx)',
    [list-list, idx-integer], [value],
    [pure, pattern(function_call)]).

elixir_wam_binding(list_length/2,
    'length(~list)',
    [list-list], [integer],
    [pure, pattern(function_call)]).

elixir_wam_binding(list_slice/4,
    'Enum.slice(~list, ~start, ~count)',
    [list-list, start-integer, count-integer], [list],
    [pure, pattern(function_call)]).

% ============================================================================
% BINDINGS: Arithmetic
% ============================================================================

elixir_wam_binding(wam_add/3, '~a + ~b', [a-integer, b-integer], [integer],
    [pure, pattern(operator)]).
elixir_wam_binding(wam_sub/3, '~a - ~b', [a-integer, b-integer], [integer],
    [pure, pattern(operator)]).
elixir_wam_binding(wam_mul/3, '~a * ~b', [a-integer, b-integer], [integer],
    [pure, pattern(operator)]).
elixir_wam_binding(wam_div/3, 'div(~a, ~b)', [a-integer, b-integer], [integer],
    [pure, partial, pattern(function_call)]).
elixir_wam_binding(wam_mod/3, 'rem(~a, ~b)', [a-integer, b-integer], [integer],
    [pure, partial, pattern(function_call)]).

% ============================================================================
% BINDINGS: Type checks
% ============================================================================

elixir_wam_binding(is_atom_val/1, 'is_binary(~v)', [v-value], [bool],
    [pure, pattern(guard)]).
elixir_wam_binding(is_integer_val/1, 'is_integer(~v)', [v-value], [bool],
    [pure, pattern(guard)]).
elixir_wam_binding(is_unbound/1, 'match?({:unbound, _}, ~v)', [v-value], [bool],
    [pure, pattern(match)]).

% ============================================================================
% BINDINGS: String/Atom operations
% ============================================================================

elixir_wam_binding(atom_to_string/2,
    'to_string(~v)',
    [v-value], [atom],
    [pure, pattern(function_call)]).

elixir_wam_binding(string_concat/3,
    '~a <> ~b',
    [a-atom, b-atom], [atom],
    [pure, pattern(operator)]).

% ============================================================================
% INITIALIZATION
% ============================================================================

init_elixir_wam_bindings :-
    forall(
        elixir_wam_binding(Pred, Expr, Inputs, Outputs, Options),
        declare_binding(elixir_wam, Pred, Expr, Inputs, Outputs, Options)
    ).
