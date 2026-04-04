% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% llvm_wam_bindings.pl - LLVM IR bindings for WAM runtime transpilation
%
% Maps Prolog builtins used by wam_runtime.pl to LLVM IR equivalents.
% These bindings enable native lowering of WAM runtime predicates
% to LLVM IR code.
%
% LLVM-specific design choices vs Rust/Go:
%   - Assoc → [32 x %Value] fixed register array (not HashMap/map)
%   - Lists → %Value* heap-allocated arrays (not Vec/slice)
%   - Type checks → icmp eq on tag field (not matches!/type switch)
%   - Value → %Value = { i32, i64 } tagged union (not enum/interface)
%   - No GC, no ownership — arena allocation with backtrack rewind
%   - Register access → compile-time index via getelementptr
%
% Categories:
%   - Register Access (library(assoc) → fixed array GEP)
%   - List Operations (Prolog lists → runtime helper calls)
%   - Arithmetic & Comparison (native LLVM instructions)
%   - Term Manipulation (=../2, functor/3)
%   - Type Checks (tag field icmp)
%   - String/Atom Operations (C runtime calls)
%
% See: docs/design/WAM_LLVM_TRANSPILATION_SPECIFICATION.md

:- module(llvm_wam_bindings, [
    init_llvm_wam_bindings/0,
    llvm_wam_binding/5,          % +PrologPred, -LLVMExpr, -ArgTypes, -RetType, -Props
    llvm_wam_type_map/2,         % +PrologType, -LLVMType
    reg_name_to_index/2          % +RegName, -Index
]).

:- use_module('../core/binding_registry').

% ============================================================================
% TYPE MAPPING
% ============================================================================

%% llvm_wam_type_map(+PrologType, -LLVMType)
%  Maps Prolog types used in the WAM runtime to LLVM IR types.
llvm_wam_type_map(assoc, '[32 x %Value]').
llvm_wam_type_map(list, '%Value*').
llvm_wam_type_map(value, '%Value').
llvm_wam_type_map(atom, 'i8*').
llvm_wam_type_map(integer, 'i64').
llvm_wam_type_map(float, 'double').
llvm_wam_type_map(number, 'double').
llvm_wam_type_map(bool, 'i1').
llvm_wam_type_map(string, 'i8*').
llvm_wam_type_map(int, 'i32').
llvm_wam_type_map(usize, 'i32').
llvm_wam_type_map(trail_entry, '%TrailEntry').
llvm_wam_type_map(choice_point, '%ChoicePoint').
llvm_wam_type_map(stack_entry, '%StackEntry').
llvm_wam_type_map(instruction, '%Instruction').
llvm_wam_type_map(wam_state, '%WamState*').

% ============================================================================
% COMPILE-TIME REGISTER INDEX MAPPING
% ============================================================================

%% reg_name_to_index(+Name, -Index)
%  Maps WAM register names to fixed array indices.
%  A1..A16 → 0..15, X1..X16 → 16..31
reg_name_to_index(Name, Index) :-
    atom_string(Name, Str),
    (   string_concat("A", NumStr, Str)
    ->  number_string(Num, NumStr),
        Index is Num - 1                    % A1 → 0, A2 → 1, ...
    ;   string_concat("X", NumStr, Str)
    ->  number_string(Num, NumStr),
        Index is Num + 15                   % X1 → 16, X2 → 17, ...
    ;   string_concat("Y", NumStr, Str)
    ->  number_string(Num, NumStr),
        Index is Num + 15                   % Y regs share X space
    ;   fail
    ).

% ============================================================================
% BINDING DECLARATIONS
% ============================================================================

%% llvm_wam_binding(+Pred, -LLVMExpr, -ArgTypes, -RetType, -Props)
%  Declares how a Prolog builtin maps to LLVM IR for WAM transpilation.

% --- Register access (library(assoc) → fixed array GEP + load/store) ---

llvm_wam_binding(empty_assoc/1,
    'zeroinitializer',
    [], [assoc],
    [pure, pattern(literal)]).

llvm_wam_binding(get_assoc/3,
    'call %Value @wam_get_reg(%WamState* %vm, i32 ~reg_idx)',
    [reg_idx-int], [value],
    [pure, partial, pattern(gep_load),
     note('Compile-time: reg_name_to_index maps name to i32 constant')]).

llvm_wam_binding(put_assoc/4,
    'call void @wam_set_reg(%WamState* %vm, i32 ~reg_idx, %Value ~val)',
    [reg_idx-int, val-value], [],
    [mutating, pattern(gep_store)]).

llvm_wam_binding(assoc_to_list/2,
    'call %Value* @wam_regs_to_list(%WamState* %vm, i32* %count)',
    [], [list],
    [pure, pattern(runtime_call)]).

% --- List operations (runtime helper calls) ---

llvm_wam_binding(append/3,
    'call %Value* @wam_list_append(%Value* ~list1, i32 ~len1, %Value* ~list2, i32 ~len2)',
    [list1-list, len1-int, list2-list, len2-int], [list],
    [pure, pattern(runtime_call)]).

llvm_wam_binding(length/2,
    'call i32 @wam_list_length(%Value* ~list)',
    [list-list], [int],
    [pure, pattern(runtime_call)]).

llvm_wam_binding(member/2,
    'call i1 @wam_list_member(%Value ~elem, %Value* ~list, i32 ~len)',
    [elem-value, list-list, len-int], [bool],
    [pure, pattern(runtime_call)]).

llvm_wam_binding(nth0/3,
    'getelementptr %Value, %Value* ~list, i32 ~idx',
    [idx-int, list-list], [value],
    [pure, partial, pattern(gep_load)]).

llvm_wam_binding(nth1/3,
    'getelementptr %Value, %Value* ~list, i32 (sub i32 ~idx, 1)',
    [idx-int, list-list], [value],
    [pure, partial, pattern(gep_load)]).

llvm_wam_binding(is_list/1,
    'icmp eq i32 ~tag, 4',
    [tag-int], [bool],
    [pure, pattern(tag_check)]).

llvm_wam_binding(last/2,
    'call %Value @wam_list_last(%Value* ~list, i32 ~len)',
    [list-list, len-int], [value],
    [pure, partial, pattern(runtime_call)]).

llvm_wam_binding(reverse/2,
    'call %Value* @wam_list_reverse(%Value* ~list, i32 ~len)',
    [list-list, len-int], [list],
    [pure, pattern(runtime_call)]).

% --- Arithmetic (native LLVM instructions on extracted payloads) ---

llvm_wam_binding('+'/3,
    'add i64 ~a, ~b',
    [a-integer, b-integer], [integer],
    [pure, pattern(binary_op)]).

llvm_wam_binding('-'/3,
    'sub i64 ~a, ~b',
    [a-integer, b-integer], [integer],
    [pure, pattern(binary_op)]).

llvm_wam_binding('*'/3,
    'mul i64 ~a, ~b',
    [a-integer, b-integer], [integer],
    [pure, pattern(binary_op)]).

llvm_wam_binding('//'/3,
    'sdiv i64 ~a, ~b',
    [a-integer, b-integer], [integer],
    [pure, partial, pattern(binary_op)]).

llvm_wam_binding(mod/3,
    'srem i64 ~a, ~b',
    [a-integer, b-integer], [integer],
    [pure, partial, pattern(binary_op)]).

llvm_wam_binding('+.'/3,
    'fadd double ~a, ~b',
    [a-float, b-float], [float],
    [pure, pattern(binary_op)]).

llvm_wam_binding('-.'/3,
    'fsub double ~a, ~b',
    [a-float, b-float], [float],
    [pure, pattern(binary_op)]).

llvm_wam_binding('*.'/3,
    'fmul double ~a, ~b',
    [a-float, b-float], [float],
    [pure, pattern(binary_op)]).

llvm_wam_binding('/.'/3,
    'fdiv double ~a, ~b',
    [a-float, b-float], [float],
    [pure, partial, pattern(binary_op)]).

% --- Comparisons ---

llvm_wam_binding('>'/2,
    'icmp sgt i64 ~a, ~b',
    [a-integer, b-integer], [bool],
    [pure, pattern(icmp)]).

llvm_wam_binding('<'/2,
    'icmp slt i64 ~a, ~b',
    [a-integer, b-integer], [bool],
    [pure, pattern(icmp)]).

llvm_wam_binding('>='/2,
    'icmp sge i64 ~a, ~b',
    [a-integer, b-integer], [bool],
    [pure, pattern(icmp)]).

llvm_wam_binding('=<'/2,
    'icmp sle i64 ~a, ~b',
    [a-integer, b-integer], [bool],
    [pure, pattern(icmp)]).

llvm_wam_binding('=:='/2,
    'icmp eq i64 ~a, ~b',
    [a-integer, b-integer], [bool],
    [pure, pattern(icmp)]).

llvm_wam_binding('=\\='/2,
    'icmp ne i64 ~a, ~b',
    [a-integer, b-integer], [bool],
    [pure, pattern(icmp)]).

% --- Term manipulation ---

llvm_wam_binding('=..'/2,
    'call %Value* @wam_univ(%Value ~term, i32* %arity_out)',
    [term-value], [list],
    [pure, pattern(runtime_call),
     note('Decompose: returns [functor, arg1, arg2, ...]')]).

llvm_wam_binding(functor/3,
    'call %Value @wam_functor(%Value ~term)',
    [term-value], [atom, integer],
    [pure, pattern(runtime_call)]).

llvm_wam_binding(compound/1,
    'icmp eq i32 ~tag, 3',
    [tag-int], [bool],
    [pure, pattern(tag_check)]).

llvm_wam_binding(arg/3,
    'call %Value @wam_arg(i32 ~n, %Value ~term)',
    [n-integer, term-value], [value],
    [pure, partial, pattern(runtime_call)]).

llvm_wam_binding(copy_term/2,
    'call %Value @wam_copy_term(%WamState* %vm, %Value ~term)',
    [term-value], [value],
    [pure, pattern(runtime_call)]).

% --- Type checks (tag field comparison) ---

llvm_wam_binding(atom/1,
    'icmp eq i32 ~tag, 0',
    [tag-int], [bool],
    [pure, pattern(tag_check)]).

llvm_wam_binding(number/1,
    'call i1 @value_is_number(%Value ~val)',
    [val-value], [bool],
    [pure, pattern(runtime_call),
     note('Checks tag == 1 (integer) or tag == 2 (float)')]).

llvm_wam_binding(integer/1,
    'icmp eq i32 ~tag, 1',
    [tag-int], [bool],
    [pure, pattern(tag_check)]).

llvm_wam_binding(float/1,
    'icmp eq i32 ~tag, 2',
    [tag-int], [bool],
    [pure, pattern(tag_check)]).

llvm_wam_binding(var/1,
    'icmp eq i32 ~tag, 6',
    [tag-int], [bool],
    [pure, pattern(tag_check)]).

llvm_wam_binding(nonvar/1,
    'icmp ne i32 ~tag, 6',
    [tag-int], [bool],
    [pure, pattern(tag_check)]).

llvm_wam_binding(ground/1,
    'call i1 @value_is_ground(%Value ~val)',
    [val-value], [bool],
    [pure, pattern(runtime_call),
     note('Recursive: no Unbound anywhere in term')]).

% --- String/Atom operations (C runtime) ---

llvm_wam_binding(atom_string/2,
    'call i8* @value_to_string(%Value ~val)',
    [val-value], [string],
    [pure, pattern(runtime_call)]).

llvm_wam_binding(atom_concat/3,
    'call i8* @wam_atom_concat(i8* ~a, i8* ~b)',
    [a-atom, b-atom], [atom],
    [pure, pattern(runtime_call), requires(libc)]).

llvm_wam_binding(sub_atom/5,
    'call i8* @strstr(i8* ~haystack, i8* ~needle)',
    [haystack-atom, needle-atom], [atom],
    [pure, partial, pattern(libc_call), requires(libc),
     note('Simplified — full sub_atom/5 semantics need custom impl')]).

llvm_wam_binding(atom_length/2,
    'call i64 @strlen(i8* ~atom)',
    [atom-atom], [integer],
    [pure, pattern(libc_call), requires(libc)]).

llvm_wam_binding(number_string/2,
    'call i8* @wam_number_to_string(%Value ~num)',
    [num-value], [string],
    [pure, partial, pattern(runtime_call)]).

% --- Format (C runtime snprintf) ---

llvm_wam_binding(format/2,
    'call i32 @snprintf(i8* ~buf, i64 ~len, i8* ~fmt)',
    [buf-atom, len-integer, fmt-atom], [int],
    [impure, pattern(libc_call), requires(libc),
     note('Prolog format directives need translation to printf format strings')]).

% --- Value operations ---

llvm_wam_binding('=='/2,
    'call i1 @value_equals(%Value ~a, %Value ~b)',
    [a-value, b-value], [bool],
    [pure, pattern(runtime_call)]).

llvm_wam_binding('\\=='/2,
    'xor i1 (call i1 @value_equals(%Value ~a, %Value ~b)), true',
    [a-value, b-value], [bool],
    [pure, pattern(negate_call)]).

% --- WAM-specific helpers ---

llvm_wam_binding(succ/2,
    'add i64 ~n, 1',
    [n-integer], [integer],
    [pure, pattern(binary_op)]).

llvm_wam_binding(plus/3,
    'add i64 ~a, ~b',
    [a-integer, b-integer], [integer],
    [pure, pattern(binary_op)]).

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_llvm_wam_bindings
%  Registers all WAM-specific LLVM bindings with the binding registry.
init_llvm_wam_bindings :-
    forall(
        llvm_wam_binding(Pred, LLVMExpr, ArgTypes, RetType, Props),
        (   \+ binding(llvm_wam, Pred, _, _, _, _)
        ->  declare_binding(llvm_wam, Pred, LLVMExpr, ArgTypes, RetType, Props)
        ;   true
        )
    ).
