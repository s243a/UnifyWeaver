% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% go_wam_bindings.pl - Go bindings for WAM runtime transpilation
%
% Maps Prolog builtins used by wam_runtime.pl to Go equivalents.
% These bindings enable native lowering of WAM runtime predicates
% to idiomatic Go code.
%
% Go-specific design choices vs Rust (rust_wam_bindings.pl):
%   - Assoc → map[string]Value (not HashMap, Go has built-in maps)
%   - Lists → []Value slices (not Vec, Go has built-in slices)
%   - Type checks → type switch / type assertion (not matches! macro)
%   - Univ (=../2) → Compound.Decompose() method
%   - No ownership/cloning — GC handles memory
%   - No Option/Result — use comma-ok idiom (val, ok := m[key])
%
% Categories:
%   - Assoc (library(assoc) → map[string]Value)
%   - List Operations (Prolog lists → []Value slices)
%   - Arithmetic & Comparison
%   - Term Manipulation (=../2, functor/3)
%   - Type Checks (atom/1, number/1, etc.)
%   - String/Atom Operations
%   - Format
%
% See: docs/design/WAM_GO_TRANSPILATION_SPECIFICATION.md

:- module(go_wam_bindings, [
    init_go_wam_bindings/0,
    go_wam_binding/5,          % +PrologPred, -GoExpr, -ArgTypes, -RetType, -Props
    go_wam_type_map/2          % +PrologType, -GoType
]).

:- use_module('../core/binding_registry').

% ============================================================================
% TYPE MAPPING
% ============================================================================

%% go_wam_type_map(+PrologType, -GoType)
%  Maps Prolog types used in the WAM runtime to Go types.
go_wam_type_map(assoc, 'map[string]Value').
go_wam_type_map(list, '[]Value').
go_wam_type_map(value, 'Value').
go_wam_type_map(atom, 'string').
go_wam_type_map(integer, 'int64').
go_wam_type_map(float, 'float64').
go_wam_type_map(number, 'float64').
go_wam_type_map(bool, 'bool').
go_wam_type_map(string, 'string').
go_wam_type_map(int, 'int').
go_wam_type_map(trail_entry, 'TrailEntry').
go_wam_type_map(choice_point, 'ChoicePoint').
go_wam_type_map(stack_entry, 'StackEntry').
go_wam_type_map(instruction, 'Instruction').

% ============================================================================
% BINDING DECLARATIONS
% ============================================================================

%% go_wam_binding(+Pred, -GoExpr, -ArgTypes, -RetType, -Props)
%  Declares how a Prolog builtin maps to Go for WAM transpilation.

% --- Assoc operations (library(assoc) → map[string]Value) ---

go_wam_binding(empty_assoc/1,
    'make(map[string]Value)',
    [], [assoc],
    [pure, import(none)]).

go_wam_binding(get_assoc/3,
    '~map[~key]',
    [key-atom, map-assoc], [value],
    [pure, partial, pattern(map_lookup),
     note('Use comma-ok idiom: val, ok := m[key]')]).

go_wam_binding(put_assoc/4,
    'func() map[string]Value { m := copyMap(~map); m[~key] = ~val; return m }()',
    [key-atom, map-assoc, val-value], [assoc],
    [pure, pattern(iife),
     note('Go maps are reference types — copy before mutating for immutable semantics')]).

go_wam_binding(assoc_to_list/2,
    'mapToSlice(~map)',
    [map-assoc], [list],
    [pure, pattern(function_call),
     note('Helper: func mapToSlice(m map[string]Value) []Value')]).

% --- List operations (Prolog lists → []Value slices) ---

go_wam_binding(append/3,
    'append(~list1, ~list2...)',
    [list1-list, list2-list], [list],
    [pure, pattern(builtin_call)]).

go_wam_binding(length/2,
    'len(~list)',
    [list-list], [int],
    [pure, pattern(builtin_call)]).

go_wam_binding(member/2,
    'sliceContains(~list, ~elem)',
    [elem-value, list-list], [bool],
    [pure, pattern(function_call),
     note('Helper: func sliceContains(s []Value, v Value) bool')]).

go_wam_binding(nth0/3,
    '~list[~idx]',
    [idx-int, list-list], [value],
    [pure, partial, pattern(index_expr),
     note('Panics if out of bounds — guard with len check')]).

go_wam_binding(nth1/3,
    '~list[~idx-1]',
    [idx-int, list-list], [value],
    [pure, partial, pattern(index_expr)]).

go_wam_binding(is_list/1,
    'isList(~val)',
    [val-value], [bool],
    [pure, pattern(function_call),
     note('func isList(v Value) bool { _, ok := v.(*List); return ok }')]).

go_wam_binding(last/2,
    '~list[len(~list)-1]',
    [list-list], [value],
    [pure, partial, pattern(index_expr)]).

go_wam_binding(reverse/2,
    'reverseSlice(~list)',
    [list-list], [list],
    [pure, pattern(function_call)]).

% --- Term manipulation ---

go_wam_binding('=..'/2,
    '~term.Decompose()',
    [term-value], [list],
    [pure, pattern(method_call),
     note('Returns []Value{functor, arg1, arg2, ...}')]).

go_wam_binding(functor/3,
    '~term.Functor()',
    [term-value], [atom, integer],
    [pure, pattern(method_call)]).

go_wam_binding(compound/1,
    'isCompound(~val)',
    [val-value], [bool],
    [pure, pattern(function_call),
     note('func isCompound(v Value) bool { _, ok := v.(*Compound); return ok }')]).

go_wam_binding(arg/3,
    '~term.Arg(~n)',
    [n-integer, term-value], [value],
    [pure, partial, pattern(method_call)]).

go_wam_binding(copy_term/2,
    '~term.DeepCopy()',
    [term-value], [value],
    [pure, pattern(method_call)]).

% --- Type checks ---

go_wam_binding(atom/1,
    'isAtom(~val)',
    [val-value], [bool],
    [pure, pattern(function_call),
     note('func isAtom(v Value) bool { _, ok := v.(*Atom); return ok }')]).

go_wam_binding(number/1,
    'isNumber(~val)',
    [val-value], [bool],
    [pure, pattern(function_call),
     note('Type switch: *Integer | *Float')]).

go_wam_binding(integer/1,
    'isInteger(~val)',
    [val-value], [bool],
    [pure, pattern(function_call)]).

go_wam_binding(float/1,
    'isFloat(~val)',
    [val-value], [bool],
    [pure, pattern(function_call)]).

go_wam_binding(var/1,
    'isUnbound(~val)',
    [val-value], [bool],
    [pure, pattern(function_call),
     note('func isUnbound(v Value) bool { _, ok := v.(*Unbound); return ok }')]).

go_wam_binding(nonvar/1,
    '!isUnbound(~val)',
    [val-value], [bool],
    [pure, pattern(prefix_not)]).

go_wam_binding(ground/1,
    'isGround(~val)',
    [val-value], [bool],
    [pure, pattern(function_call),
     note('Recursive: no Unbound anywhere in term')]).

% --- String/Atom operations ---

go_wam_binding(atom_string/2,
    'fmt.Sprintf("%v", ~atom)',
    [atom-value], [string],
    [pure, pattern(format_call), import('"fmt"')]).

go_wam_binding(atom_concat/3,
    '~a + ~b',
    [a-atom, b-atom], [atom],
    [pure, pattern(binary_op)]).

go_wam_binding(sub_atom/5,
    'strings.Contains(~atom, ~sub)',
    [atom-atom, sub-atom], [bool],
    [pure, partial, pattern(function_call), import('"strings"'),
     note('Simplified — full sub_atom/5 semantics need custom impl')]).

go_wam_binding(split_string/4,
    'strings.Split(~str, ~sep)',
    [str-string, sep-string], [list],
    [pure, pattern(function_call), import('"strings"')]).

go_wam_binding(atom_length/2,
    'len(~atom)',
    [atom-atom], [int],
    [pure, pattern(builtin_call)]).

go_wam_binding(number_string/2,
    'fmt.Sprintf("%g", ~num)',
    [num-number], [string],
    [pure, pattern(format_call), import('"fmt"')]).

go_wam_binding(atom_number/2,
    'parseNumber(~atom)',
    [atom-atom], [number],
    [pure, partial, pattern(function_call),
     note('Helper using strconv.ParseFloat')]).

% --- Comparison ---

go_wam_binding('=='/2,
    'valueEquals(~a, ~b)',
    [a-value, b-value], [bool],
    [pure, pattern(function_call),
     note('Structural equality — deep comparison of Value trees')]).

go_wam_binding('\\=='/2,
    '!valueEquals(~a, ~b)',
    [a-value, b-value], [bool],
    [pure, pattern(prefix_not)]).

% --- Format ---

go_wam_binding(format/2,
    'fmt.Sprintf(~fmt)',
    [fmt-string], [string],
    [impure, pattern(format_call), import('"fmt"'),
     note('Prolog format directives need translation to Go format verbs')]).

go_wam_binding(format/3,
    'fmt.Sprintf(~fmt, ~args...)',
    [fmt-string, args-list], [string],
    [impure, pattern(format_call), import('"fmt"')]).

% --- WAM-specific helpers ---

go_wam_binding(succ/2,
    '~n + 1',
    [n-integer], [integer],
    [pure, pattern(binary_op)]).

go_wam_binding(plus/3,
    '~a + ~b',
    [a-integer, b-integer], [integer],
    [pure, pattern(binary_op)]).

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_go_wam_bindings
%  Registers all WAM-specific Go bindings with the binding registry.
init_go_wam_bindings :-
    forall(
        go_wam_binding(Pred, GoExpr, ArgTypes, RetType, Props),
        (   \+ binding(go_wam, Pred, _, _, _, _)
        ->  declare_binding(go_wam, Pred, GoExpr, ArgTypes, RetType, Props)
        ;   true
        )
    ).
