:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% c_wam_bindings.pl - C bindings for WAM runtime transpilation
%
% Maps WAM runtime operations to C expressions.
% Used by wam_c_target.pl.
%
% Categories:
%   - Hash map operations (WAM registers)
%   - Dynamic array operations (WAM heap/trail/stack)
%   - Arithmetic
%   - Type checks
%   - String/Memory operations

:- module(c_wam_bindings, [
    init_c_wam_bindings/0,
    c_wam_binding/5,          % +PrologPred, -CExpr, -ArgTypes, -RetType, -Props
    c_wam_type_map/2           % +WAMType, -CType
]).

:- use_module('../core/binding_registry').

% ============================================================================
% TYPE MAPPING
% ============================================================================

c_wam_type_map(assoc, 'WamRegMap*').
c_wam_type_map(list, 'WamArray*').
c_wam_type_map(value, 'WamValue*').
c_wam_type_map(atom, 'const char*').
c_wam_type_map(integer, 'int').
c_wam_type_map(float, 'double').
c_wam_type_map(bool, 'int').
c_wam_type_map(void, 'void').
c_wam_type_map(trail_entry, 'TrailEntry').
c_wam_type_map(choice_point, 'ChoicePoint*').
c_wam_type_map(wam_state, 'WamState*').
c_wam_type_map(instruction, 'WamInstr').
c_wam_type_map(size, 'int').

% ============================================================================
% BINDINGS: Hash map operations (WAM registers)
% ============================================================================

c_wam_binding(reg_get/3,
    'wam_reg_get(~state, ~key)',
    [state-wam_state, key-atom], [value],
    [pattern(function_call), description("Get register value by key")]).

c_wam_binding(reg_put/4,
    'wam_reg_put(~state, ~key, ~val)',
    [state-wam_state, key-atom, val-value], [void],
    [pattern(function_call), description("Set register value by key")]).

c_wam_binding(reg_delete/3,
    'wam_reg_delete(~state, ~key)',
    [state-wam_state, key-atom], [void],
    [pattern(function_call), description("Remove register by key")]).

% ============================================================================
% BINDINGS: Dynamic array operations (WAM heap, trail, stack)
% ============================================================================

c_wam_binding(array_push/3,
    'wam_array_push(~arr, ~val)',
    [arr-list, val-value], [void],
    [pattern(function_call), description("Append to dynamic array")]).

c_wam_binding(array_get/3,
    'wam_array_get(~arr, ~idx)',
    [arr-list, idx-integer], [value],
    [pattern(function_call), description("Get element at index")]).

c_wam_binding(array_size/2,
    '~arr->size',
    [arr-list], [size],
    [pure, pattern(field_access), description("Get array size")]).

% ============================================================================
% BINDINGS: Value constructors
% ============================================================================

c_wam_binding(make_atom/2,
    'wam_make_atom(~s)',
    [s-atom], [value],
    [pattern(function_call), description("Create atom value")]).

c_wam_binding(make_int/2,
    'wam_make_int(~n)',
    [n-integer], [value],
    [pattern(function_call), description("Create integer value")]).

c_wam_binding(make_ref/2,
    'wam_make_ref(~addr)',
    [addr-integer], [value],
    [pattern(function_call), description("Create heap reference value")]).

c_wam_binding(make_unbound/2,
    'wam_make_unbound(~name)',
    [name-atom], [value],
    [pattern(function_call), description("Create unbound variable")]).

% ============================================================================
% BINDINGS: Type checks
% ============================================================================

c_wam_binding(is_atom_val/1,
    '~v->tag == VAL_ATOM',
    [v-value], [bool],
    [pure, pattern(comparison)]).

c_wam_binding(is_integer_val/1,
    '~v->tag == VAL_INT',
    [v-value], [bool],
    [pure, pattern(comparison)]).

c_wam_binding(is_unbound/1,
    '~v->tag == VAL_UNBOUND',
    [v-value], [bool],
    [pure, pattern(comparison)]).

c_wam_binding(is_ref/1,
    '~v->tag == VAL_REF',
    [v-value], [bool],
    [pure, pattern(comparison)]).

% ============================================================================
% BINDINGS: Arithmetic
% ============================================================================

c_wam_binding(wam_add/3, '~a + ~b', [a-integer, b-integer], [integer],
    [pure, pattern(operator)]).
c_wam_binding(wam_sub/3, '~a - ~b', [a-integer, b-integer], [integer],
    [pure, pattern(operator)]).
c_wam_binding(wam_mul/3, '~a * ~b', [a-integer, b-integer], [integer],
    [pure, pattern(operator)]).
c_wam_binding(wam_div/3, '~a / ~b', [a-integer, b-integer], [integer],
    [pure, partial, pattern(operator)]).
c_wam_binding(wam_mod/3, '~a % ~b', [a-integer, b-integer], [integer],
    [pure, partial, pattern(operator)]).

% ============================================================================
% BINDINGS: String/Memory operations
% ============================================================================

c_wam_binding(str_eq/3,
    'strcmp(~a, ~b) == 0',
    [a-atom, b-atom], [bool],
    [pure, pattern(function_call), description("String equality")]).

c_wam_binding(str_dup/2,
    'strdup(~s)',
    [s-atom], [atom],
    [pattern(function_call), description("Duplicate string")]).

c_wam_binding(snprintf_val/3,
    'snprintf(~buf, sizeof(~buf), "%s", ~val)',
    [buf-atom, val-value], [void],
    [pattern(function_call), description("Format value to string")]).

% ============================================================================
% INITIALIZATION
% ============================================================================

init_c_wam_bindings :-
    forall(
        c_wam_binding(Pred, Expr, Inputs, Outputs, Options),
        declare_binding(c_wam, Pred, Expr, Inputs, Outputs, Options)
    ).
