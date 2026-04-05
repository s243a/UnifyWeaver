% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% cil_wam_bindings.pl - CIL bindings for WAM runtime transpilation
%
% Maps Prolog builtins used by wam_runtime.pl to CIL equivalents.
% These bindings enable native lowering of WAM runtime predicates
% to idiomatic CIL assembly code.
%
% CIL-specific design choices vs Rust/Go/LLVM:
%   - Assoc → Value[] fixed array (not HashMap/map/tagged union array)
%   - Lists → Value[] managed arrays (GC handles deallocation)
%   - Type checks → isinst ClassName (not matches!/tag compare)
%   - Value → class hierarchy (AtomValue, IntegerValue, etc.)
%   - GC handles all memory — no arena, no ownership
%   - Register access → ldelem.ref/stelem.ref on Value[]
%   - Atoms stored as string fields — no interning table needed
%
% See: docs/design/WAM_ILASM_TRANSPILATION_SPECIFICATION.md

:- module(cil_wam_bindings, [
    init_cil_wam_bindings/0,
    cil_wam_binding/5,          % +PrologPred, -CILExpr, -ArgTypes, -RetType, -Props
    cil_wam_type_map/2,         % +PrologType, -CILType
    cil_reg_name_to_index/2     % +RegName, -Index
]).

:- use_module('../core/binding_registry').

% ============================================================================
% TYPE MAPPING
% ============================================================================

%% cil_wam_type_map(+PrologType, -CILType)
cil_wam_type_map(assoc, 'class Value[]').
cil_wam_type_map(list, 'class Value[]').
cil_wam_type_map(value, 'class Value').
cil_wam_type_map(atom, 'string').
cil_wam_type_map(integer, 'int64').
cil_wam_type_map(float, 'float64').
cil_wam_type_map(number, 'float64').
cil_wam_type_map(bool, 'bool').
cil_wam_type_map(string, 'string').
cil_wam_type_map(int, 'int32').
cil_wam_type_map(trail_entry, 'class TrailEntry').
cil_wam_type_map(choice_point, 'class ChoicePoint').
cil_wam_type_map(stack_entry, 'class StackEntry').
cil_wam_type_map(instruction, 'class Instruction').
cil_wam_type_map(wam_state, 'class WamState').

% ============================================================================
% COMPILE-TIME REGISTER INDEX MAPPING
% ============================================================================

%% cil_reg_name_to_index(+Name, -Index)
%  Maps WAM register names to fixed array indices in the Value[] array.
%  Register ABI (shared with LLVM target):
%  A1→0, A2→1, ..., A16→15, X1→16, X2→17, ..., X16→31
cil_reg_name_to_index(Name, Index) :-
    atom_string(Name, Str),
    (   string_concat("A", NumStr, Str)
    ->  number_string(Num, NumStr),
        Index is Num - 1
    ;   string_concat("X", NumStr, Str)
    ->  number_string(Num, NumStr),
        Index is Num + 15
    ;   string_concat("Y", NumStr, Str)
    ->  number_string(Num, NumStr),
        Index is Num + 15
    ;   fail
    ).

% ============================================================================
% BINDING DECLARATIONS
% ============================================================================

%% cil_wam_binding(+Pred, -CILExpr, -ArgTypes, -RetType, -Props)

% --- Register access (Value[] array) ---

cil_wam_binding(empty_assoc/1,
    'newarr Value',
    [], [assoc],
    [pure, pattern(array_new),
     note('ldc.i4 32 / newarr Value — creates 32-element register array')]).

cil_wam_binding(get_assoc/3,
    'ldelem.ref',
    [reg_idx-int32, array-assoc], [value],
    [pure, pattern(array_load),
     note('Push array, push index, ldelem.ref')]).

cil_wam_binding(put_assoc/4,
    'stelem.ref',
    [reg_idx-int32, array-assoc, val-value], [],
    [mutating, pattern(array_store),
     note('Push array, push index, push value, stelem.ref')]).

cil_wam_binding(assoc_to_list/2,
    'call class Value[] WamHelpers::RegsToList(class Value[])',
    [array-assoc], [list],
    [pure, pattern(static_call)]).

% --- List operations (managed arrays) ---

cil_wam_binding(append/3,
    'call class Value[] WamHelpers::ListAppend(class Value[], class Value[])',
    [list1-list, list2-list], [list],
    [pure, pattern(static_call)]).

cil_wam_binding(length/2,
    'ldlen',
    [list-list], [int],
    [pure, pattern(array_length),
     note('Push array, ldlen, conv.i4')]).

cil_wam_binding(member/2,
    'call bool WamHelpers::ListMember(class Value, class Value[])',
    [elem-value, list-list], [bool],
    [pure, pattern(static_call)]).

cil_wam_binding(nth0/3,
    'ldelem.ref',
    [idx-int32, list-list], [value],
    [pure, partial, pattern(array_load)]).

cil_wam_binding(nth1/3,
    'ldelem.ref',
    [idx-int32, list-list], [value],
    [pure, partial, pattern(array_load),
     note('Index adjusted: push idx, ldc.i4.1, sub, ldelem.ref')]).

cil_wam_binding(is_list/1,
    'isinst ListValue',
    [val-value], [bool],
    [pure, pattern(type_check)]).

cil_wam_binding(reverse/2,
    'call class Value[] WamHelpers::ListReverse(class Value[])',
    [list-list], [list],
    [pure, pattern(static_call)]).

% --- Arithmetic (CIL stack operations) ---

cil_wam_binding('+'/3, 'add', [integer, integer], [integer],
    [pure, pattern(stack_op)]).
cil_wam_binding('-'/3, 'sub', [integer, integer], [integer],
    [pure, pattern(stack_op)]).
cil_wam_binding('*'/3, 'mul', [integer, integer], [integer],
    [pure, pattern(stack_op)]).
cil_wam_binding('//'/3, 'div', [integer, integer], [integer],
    [pure, partial, pattern(stack_op)]).
cil_wam_binding(mod/3, 'rem', [integer, integer], [integer],
    [pure, partial, pattern(stack_op)]).

% --- Comparisons (CIL branch instructions) ---

cil_wam_binding('>'/2, 'bgt', [integer, integer], [bool],
    [pure, pattern(branch_op)]).
cil_wam_binding('<'/2, 'blt', [integer, integer], [bool],
    [pure, pattern(branch_op)]).
cil_wam_binding('>='/2, 'bge', [integer, integer], [bool],
    [pure, pattern(branch_op)]).
cil_wam_binding('=<'/2, 'ble', [integer, integer], [bool],
    [pure, pattern(branch_op)]).
cil_wam_binding('=:='/2, 'beq', [integer, integer], [bool],
    [pure, pattern(branch_op)]).
cil_wam_binding('=\\='/2, 'bne.un', [integer, integer], [bool],
    [pure, pattern(branch_op)]).

% --- Term manipulation ---

cil_wam_binding('=..'/2,
    'call class Value[] WamHelpers::Univ(class Value)',
    [term-value], [list],
    [pure, pattern(static_call)]).

cil_wam_binding(functor/3,
    'call class Value WamHelpers::Functor(class Value)',
    [term-value], [atom, integer],
    [pure, pattern(static_call)]).

cil_wam_binding(compound/1,
    'isinst CompoundValue',
    [val-value], [bool],
    [pure, pattern(type_check)]).

cil_wam_binding(copy_term/2,
    'call class Value WamHelpers::CopyTerm(class Value)',
    [term-value], [value],
    [pure, pattern(static_call)]).

% --- Type checks (isinst) ---

cil_wam_binding(atom/1,
    'isinst AtomValue',
    [val-value], [bool],
    [pure, pattern(type_check)]).

cil_wam_binding(number/1,
    'call bool WamHelpers::IsNumber(class Value)',
    [val-value], [bool],
    [pure, pattern(static_call),
     note('Checks isinst IntegerValue || isinst FloatValue')]).

cil_wam_binding(integer/1,
    'isinst IntegerValue',
    [val-value], [bool],
    [pure, pattern(type_check)]).

cil_wam_binding(float/1,
    'isinst FloatValue',
    [val-value], [bool],
    [pure, pattern(type_check)]).

cil_wam_binding(var/1,
    'callvirt instance bool Value::IsUnbound()',
    [val-value], [bool],
    [pure, pattern(virtual_call)]).

cil_wam_binding(nonvar/1,
    'callvirt instance bool Value::IsUnbound()\n    ldc.i4.0\n    ceq',
    [val-value], [bool],
    [pure, pattern(negate_call),
     note('IsUnbound then negate: push result, push 0, ceq')]).

% --- String/Atom operations (.NET BCL) ---

cil_wam_binding(atom_string/2,
    'callvirt instance string [mscorlib]System.Object::ToString()',
    [val-value], [string],
    [pure, pattern(virtual_call)]).

cil_wam_binding(atom_concat/3,
    'call string [mscorlib]System.String::Concat(string, string)',
    [a-string, b-string], [string],
    [pure, pattern(static_call)]).

cil_wam_binding(sub_atom/5,
    'callvirt instance bool [mscorlib]System.String::Contains(string)',
    [atom-string, sub-string], [bool],
    [pure, partial, pattern(virtual_call)]).

cil_wam_binding(atom_length/2,
    'callvirt instance int32 [mscorlib]System.String::get_Length()',
    [atom-string], [int],
    [pure, pattern(virtual_call)]).

% --- Value equality ---

cil_wam_binding('=='/2,
    'callvirt instance bool Value::Equals(class Value)',
    [a-value, b-value], [bool],
    [pure, pattern(virtual_call)]).

cil_wam_binding('\\=='/2,
    'callvirt instance bool Value::Equals(class Value)\n    ldc.i4.0\n    ceq',
    [a-value, b-value], [bool],
    [pure, pattern(negate_call)]).

% --- Format (String.Format) ---

cil_wam_binding(format/2,
    'call string [mscorlib]System.String::Format(string, object)',
    [fmt-string, arg-value], [string],
    [impure, pattern(static_call)]).

% --- WAM-specific helpers ---

cil_wam_binding(succ/2,
    'ldc.i8 1\n    add',
    [n-integer], [integer],
    [pure, pattern(inline_op)]).

cil_wam_binding(plus/3, 'add', [integer, integer], [integer],
    [pure, pattern(stack_op)]).

% ============================================================================
% INITIALIZATION
% ============================================================================

init_cil_wam_bindings :-
    forall(
        cil_wam_binding(Pred, CILExpr, ArgTypes, RetType, Props),
        (   \+ binding(cil_wam, Pred, _, _, _, _)
        ->  declare_binding(cil_wam, Pred, CILExpr, ArgTypes, RetType, Props)
        ;   true
        )
    ).
