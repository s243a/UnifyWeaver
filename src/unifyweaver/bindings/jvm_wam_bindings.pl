:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% jvm_wam_bindings.pl - JVM bindings for WAM runtime transpilation
%
% Maps WAM runtime operations to JVM bytecode sequences.
% Used by wam_jvm_target.pl for both Jamaica and Krakatau output.
%
% Categories:
%   - Assoc (WAM registers → HashMap)
%   - List Operations (WAM heap/trail → ArrayList)
%   - Arithmetic & Comparison
%   - Type Checks (instanceof)
%   - Boxing/Unboxing

:- module(jvm_wam_bindings, [
    init_jvm_wam_bindings/0,
    jvm_wam_binding/5,          % +PrologPred, -JVMExpr, -ArgTypes, -RetType, -Props
    jvm_wam_type_map/2           % +PrologType, -JVMType
]).

:- use_module('../core/binding_registry').

% ============================================================================
% TYPE MAPPING
% ============================================================================

%% jvm_wam_type_map(+WAMType, -JVMType)
jvm_wam_type_map(assoc, 'java/util/HashMap').
jvm_wam_type_map(list, 'java/util/ArrayList').
jvm_wam_type_map(value, 'Ljava/lang/Object;').
jvm_wam_type_map(atom, 'Ljava/lang/String;').
jvm_wam_type_map(integer, 'Ljava/lang/Integer;').
jvm_wam_type_map(int, 'I').
jvm_wam_type_map(bool, 'Z').
jvm_wam_type_map(void, 'V').
jvm_wam_type_map(trail_entry, 'Ljava/lang/String;').
jvm_wam_type_map(choice_point, '[Ljava/lang/Object;').
jvm_wam_type_map(instruction_array, '[I').
jvm_wam_type_map(wam_state, 'LWamState;').

% ============================================================================
% BINDINGS: HashMap operations (WAM registers)
% ============================================================================

jvm_wam_binding(get_assoc/3,
    'invokevirtual java/util/HashMap get (Ljava/lang/Object;)Ljava/lang/Object;',
    [assoc, atom], [value],
    [pattern(method_call), description("Get value from register map")]).

jvm_wam_binding(put_assoc/4,
    'invokevirtual java/util/HashMap put (Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;',
    [assoc, atom, value], [value],
    [pattern(method_call), description("Put value into register map")]).

jvm_wam_binding(del_assoc/3,
    'invokevirtual java/util/HashMap remove (Ljava/lang/Object;)Ljava/lang/Object;',
    [assoc, atom], [value],
    [pattern(method_call), description("Remove key from register map")]).

jvm_wam_binding(assoc_contains/2,
    'invokevirtual java/util/HashMap containsKey (Ljava/lang/Object;)Z',
    [assoc, atom], [bool],
    [pure, pattern(method_call), description("Check if key exists in map")]).

jvm_wam_binding(empty_assoc/1,
    'new java/util/HashMap + dup + invokespecial java/util/HashMap <init> ()V',
    [], [assoc],
    [pattern(constructor), description("Create empty HashMap")]).

% ============================================================================
% BINDINGS: ArrayList operations (WAM heap, trail, stack)
% ============================================================================

jvm_wam_binding(list_add/2,
    'invokevirtual java/util/ArrayList add (Ljava/lang/Object;)Z',
    [list, value], [bool],
    [pattern(method_call), description("Append to list")]).

jvm_wam_binding(list_get/3,
    'invokevirtual java/util/ArrayList get (I)Ljava/lang/Object;',
    [list, int], [value],
    [pattern(method_call), description("Get element at index")]).

jvm_wam_binding(list_set/3,
    'invokevirtual java/util/ArrayList set (ILjava/lang/Object;)Ljava/lang/Object;',
    [list, int, value], [value],
    [pattern(method_call), description("Set element at index")]).

jvm_wam_binding(list_size/2,
    'invokevirtual java/util/ArrayList size ()I',
    [list], [int],
    [pure, pattern(method_call), description("Get list size")]).

jvm_wam_binding(list_remove_last/1,
    'invokevirtual java/util/ArrayList remove (I)Ljava/lang/Object;',
    [list, int], [value],
    [pattern(method_call), description("Remove element at index")]).

jvm_wam_binding(empty_list/1,
    'new java/util/ArrayList + dup + invokespecial java/util/ArrayList <init> ()V',
    [], [list],
    [pattern(constructor), description("Create empty ArrayList")]).

% ============================================================================
% BINDINGS: Boxing / Unboxing
% ============================================================================

jvm_wam_binding(box_int/2,
    'invokestatic java/lang/Integer valueOf (I)Ljava/lang/Integer;',
    [int], [integer],
    [pure, pattern(function), description("Box int to Integer")]).

jvm_wam_binding(unbox_int/2,
    'invokevirtual java/lang/Integer intValue ()I',
    [integer], [int],
    [pure, pattern(function), description("Unbox Integer to int")]).

% ============================================================================
% BINDINGS: Type checks
% ============================================================================

jvm_wam_binding(is_string/1,
    'instanceof java/lang/String',
    [value], [bool],
    [pure, pattern(type_check), description("Check if value is String")]).

jvm_wam_binding(is_integer/1,
    'instanceof java/lang/Integer',
    [value], [bool],
    [pure, pattern(type_check), description("Check if value is Integer")]).

jvm_wam_binding(is_unbound/1,
    'instanceof WamUnbound',
    [value], [bool],
    [pure, pattern(type_check), description("Check if value is unbound WAM variable")]).

% ============================================================================
% BINDINGS: Arithmetic
% ============================================================================

jvm_wam_binding(wam_add/3, 'iadd', [int, int], [int],
    [pure, deterministic, pattern(binary_op)]).
jvm_wam_binding(wam_sub/3, 'isub', [int, int], [int],
    [pure, deterministic, pattern(binary_op)]).
jvm_wam_binding(wam_mul/3, 'imul', [int, int], [int],
    [pure, deterministic, pattern(binary_op)]).
jvm_wam_binding(wam_div/3, 'idiv', [int, int], [int],
    [pure, deterministic, partial, pattern(binary_op)]).
jvm_wam_binding(wam_mod/3, 'irem', [int, int], [int],
    [pure, deterministic, partial, pattern(binary_op)]).

% ============================================================================
% BINDINGS: String / Atom operations
% ============================================================================

jvm_wam_binding(atom_string/2,
    'invokevirtual java/lang/Object toString ()Ljava/lang/String;',
    [value], [atom],
    [pure, pattern(method_call), description("Convert value to string")]).

jvm_wam_binding(atom_equals/2,
    'invokevirtual java/lang/String equals (Ljava/lang/Object;)Z',
    [atom, atom], [bool],
    [pure, pattern(method_call), description("String equality check")]).

jvm_wam_binding(atom_concat/3,
    'invokevirtual java/lang/String concat (Ljava/lang/String;)Ljava/lang/String;',
    [atom, atom], [atom],
    [pure, pattern(method_call), description("Concatenate two strings")]).

% ============================================================================
% INITIALIZATION
% ============================================================================

init_jvm_wam_bindings :-
    forall(
        jvm_wam_binding(Pred, Instr, Inputs, Outputs, Options),
        (   declare_binding(jamaica_wam, Pred, Instr, Inputs, Outputs, Options),
            declare_binding(krakatau_wam, Pred, Instr, Inputs, Outputs, Options)
        )
    ).
