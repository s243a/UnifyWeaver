% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% rust_wam_bindings.pl - Rust bindings for WAM runtime transpilation
%
% Maps Prolog builtins used by wam_runtime.pl to Rust equivalents.
% These bindings enable native lowering of WAM runtime predicates
% to idiomatic Rust code.
%
% Categories:
%   - Assoc (library(assoc) → HashMap)
%   - List Operations (Prolog lists → Vec)
%   - Arithmetic & Comparison
%   - Term Manipulation (=../2, functor/3)
%   - Type Checks (atom/1, number/1, etc.)
%   - String/Atom Operations
%
% See: docs/design/WAM_RUST_TRANSPILATION_SPECIFICATION.md

:- module(rust_wam_bindings, [
    init_rust_wam_bindings/0,
    rust_wam_binding/5,          % +PrologPred, -RustExpr, -ArgTypes, -RetType, -Props
    rust_wam_type_map/2          % +PrologType, -RustType
]).

:- use_module('../core/binding_registry').

% ============================================================================
% TYPE MAPPING
% ============================================================================

%% rust_wam_type_map(+PrologType, -RustType)
%  Maps Prolog types used in the WAM runtime to Rust types.
rust_wam_type_map(assoc, 'HashMap<String, Value>').
rust_wam_type_map(list, 'Vec<Value>').
rust_wam_type_map(value, 'Value').
rust_wam_type_map(atom, 'String').
rust_wam_type_map(integer, 'i64').
rust_wam_type_map(float, 'f64').
rust_wam_type_map(number, 'f64').
rust_wam_type_map(bool, 'bool').
rust_wam_type_map(string, 'String').
rust_wam_type_map(usize, 'usize').
rust_wam_type_map(trail_entry, 'TrailEntry').
rust_wam_type_map(choice_point, 'ChoicePoint').
rust_wam_type_map(stack_entry, 'StackEntry').
rust_wam_type_map(instruction, 'Instruction').

% ============================================================================
% BINDING DECLARATIONS
% ============================================================================

%% rust_wam_binding(+Pred, -RustExpr, -ArgTypes, -RetType, -Props)
%  Declares how a Prolog builtin maps to Rust for WAM transpilation.

% --- Assoc operations (library(assoc) → HashMap) ---

rust_wam_binding(empty_assoc/1,
    'HashMap::new()',
    [], [assoc],
    [pure, import('std::collections::HashMap')]).

rust_wam_binding(get_assoc/3,
    '~map.get(&~key).cloned()',
    [key-atom, map-assoc], [value],
    [pure, partial, pattern(method_call)]).

rust_wam_binding(put_assoc/4,
    '{ let mut m = ~map.clone(); m.insert(~key.clone(), ~val.clone()); m }',
    [key-atom, map-assoc, val-value], [assoc],
    [pure, pattern(block_expr)]).

rust_wam_binding(assoc_to_list/2,
    '~map.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<Vec<_>>()',
    [map-assoc], [list],
    [pure, pattern(method_chain)]).

% --- List operations ---

rust_wam_binding(append/3,
    '{ let mut v = ~list1.clone(); v.extend(~list2.iter().cloned()); v }',
    [list1-list, list2-list], [list],
    [pure, pattern(block_expr)]).

rust_wam_binding(length/2,
    '~list.len()',
    [list-list], [usize],
    [pure, pattern(method_call)]).

rust_wam_binding(member/2,
    '~list.iter().any(|x| x == &~elem)',
    [elem-value, list-list], [bool],
    [pure, pattern(method_chain)]).

rust_wam_binding(nth0/3,
    '~list.get(~idx).cloned()',
    [idx-usize, list-list], [value],
    [pure, partial, pattern(method_call)]).

rust_wam_binding(nth1/3,
    '~list.get(~idx - 1).cloned()',
    [idx-usize, list-list], [value],
    [pure, partial, pattern(method_call)]).

rust_wam_binding(is_list/1,
    '~val.is_list()',
    [val-value], [bool],
    [pure, pattern(method_call)]).

% --- Term manipulation ---

rust_wam_binding('=..'/2,
    '~term.univ()',
    [term-value], [list],
    [pure, pattern(method_call),
     note('Decompose: val.univ() → [functor, arg1, arg2, ...]')]).

rust_wam_binding(functor/3,
    '~term.functor()',
    [term-value], [atom, integer],
    [pure, pattern(method_call)]).

rust_wam_binding(compound/1,
    '~val.is_compound()',
    [val-value], [bool],
    [pure, pattern(method_call)]).

% --- Type checks ---

rust_wam_binding(atom/1,
    'matches!(~val, Value::Atom(_))',
    [val-value], [bool],
    [pure, pattern(matches_expr)]).

rust_wam_binding(number/1,
    '~val.is_number()',
    [val-value], [bool],
    [pure, pattern(method_call)]).

rust_wam_binding(integer/1,
    'matches!(~val, Value::Integer(_))',
    [val-value], [bool],
    [pure, pattern(matches_expr)]).

rust_wam_binding(float/1,
    'matches!(~val, Value::Float(_))',
    [val-value], [bool],
    [pure, pattern(matches_expr)]).

rust_wam_binding(var/1,
    '~val.is_unbound()',
    [val-value], [bool],
    [pure, pattern(method_call)]).

rust_wam_binding(nonvar/1,
    '!~val.is_unbound()',
    [val-value], [bool],
    [pure, pattern(prefix_not)]).

% --- String/Atom operations ---

rust_wam_binding(atom_string/2,
    '~atom.to_string()',
    [atom-atom], [string],
    [pure, pattern(method_call)]).

rust_wam_binding(atom_concat/3,
    'format!("{}{}", ~a, ~b)',
    [a-atom, b-atom], [atom],
    [pure, pattern(format_macro)]).

rust_wam_binding(sub_atom/5,
    '~atom.contains(&~sub)',
    [atom-atom, sub-atom], [bool],
    [pure, partial, pattern(method_call),
     note('Simplified — full sub_atom/5 semantics need custom impl')]).

rust_wam_binding(split_string/4,
    '~str.split(~sep).map(|s| s.trim_matches(|c: char| ~pad.contains(c)).to_string()).collect::<Vec<_>>()',
    [str-string, sep-string, pad-string], [list],
    [pure, pattern(method_chain)]).

rust_wam_binding(number_string/2,
    '~str.parse::<f64>()',
    [str-string], [number],
    [pure, partial, pattern(method_call)]).

% --- Format ---

rust_wam_binding(format/2,
    'format!(~fmt)',
    [fmt-string], [string],
    [impure, pattern(format_macro),
     note('Prolog format directives need translation to Rust format strings')]).

rust_wam_binding(format/3,
    'format!(~fmt, ~args)',
    [fmt-string, args-list], [string],
    [impure, pattern(format_macro)]).

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_rust_wam_bindings
%  Registers all WAM-specific Rust bindings with the binding registry.
init_rust_wam_bindings :-
    forall(
        rust_wam_binding(Pred, RustExpr, ArgTypes, RetType, Props),
        (   \+ binding(rust_wam, Pred, _, _, _, _)
        ->  declare_binding(rust_wam, Pred, RustExpr, ArgTypes, RetType, Props)
        ;   true
        )
    ).
