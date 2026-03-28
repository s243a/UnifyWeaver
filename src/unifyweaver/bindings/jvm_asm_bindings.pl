:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% jvm_asm_bindings.pl - Shared JVM Assembly Bindings
% Registers bindings for both Jamaica and Krakatau targets.
% Maps Prolog operations to JVM bytecode instructions.

:- module(jvm_asm_bindings, [init_jvm_asm_bindings/0]).

:- use_module('../core/binding_registry').

init_jvm_asm_bindings :-
    register_jvm_asm_arithmetic_bindings,
    register_jvm_asm_comparison_bindings,
    register_jvm_asm_bitwise_bindings,
    register_jvm_asm_math_bindings,
    register_jvm_asm_io_bindings,
    register_jvm_asm_string_bindings.

%% declare_dual_binding(+Pred, +Instruction, +Inputs, +Outputs, +Options)
%%   Register a binding for both jamaica and krakatau targets.
declare_dual_binding(Pred, Instruction, Inputs, Outputs, Options) :-
    declare_binding(jamaica, Pred, Instruction, Inputs, Outputs, Options),
    declare_binding(krakatau, Pred, Instruction, Inputs, Outputs, Options).

register_jvm_asm_arithmetic_bindings :-
    declare_dual_binding('+'/3, 'iadd', [int, int], [int],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_dual_binding('-'/3, 'isub', [int, int], [int],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_dual_binding('*'/3, 'imul', [int, int], [int],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_dual_binding('//'/3, 'idiv', [int, int], [int],
        [pure, deterministic, partial, pattern(binary_op)]),
    declare_dual_binding('/'/3, 'idiv', [int, int], [int],
        [pure, deterministic, partial, pattern(binary_op)]),
    declare_dual_binding('mod'/3, 'irem', [int, int], [int],
        [pure, deterministic, partial, pattern(binary_op)]),
    declare_dual_binding('neg'/2, 'ineg', [int], [int],
        [pure, deterministic, total, pattern(unary_op)]).

register_jvm_asm_comparison_bindings :-
    declare_dual_binding('>'/2, 'if_icmpgt', [int, int], [boolean],
        [pure, deterministic, total, pattern(comparison)]),
    declare_dual_binding('<'/2, 'if_icmplt', [int, int], [boolean],
        [pure, deterministic, total, pattern(comparison)]),
    declare_dual_binding('>='/2, 'if_icmpge', [int, int], [boolean],
        [pure, deterministic, total, pattern(comparison)]),
    declare_dual_binding('=<'/2, 'if_icmple', [int, int], [boolean],
        [pure, deterministic, total, pattern(comparison)]),
    declare_dual_binding('=:='/2, 'if_icmpeq', [int, int], [boolean],
        [pure, deterministic, total, pattern(comparison)]),
    declare_dual_binding('=\\='/2, 'if_icmpne', [int, int], [boolean],
        [pure, deterministic, total, pattern(comparison)]).

register_jvm_asm_bitwise_bindings :-
    declare_dual_binding('/\\'/3, 'iand', [int, int], [int],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_dual_binding('\\/'/3, 'ior', [int, int], [int],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_dual_binding('xor'/3, 'ixor', [int, int], [int],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_dual_binding('<<'/3, 'ishl', [int, int], [int],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_dual_binding('>>'/3, 'ishr', [int, int], [int],
        [pure, deterministic, total, pattern(binary_op)]).

register_jvm_asm_math_bindings :-
    declare_dual_binding('abs'/2, 'invokestatic java/lang/Math abs (I)I',
        [int], [int],
        [pure, deterministic, total, pattern(function)]),
    declare_dual_binding('max'/3, 'invokestatic java/lang/Math max (II)I',
        [int, int], [int],
        [pure, deterministic, total, pattern(function)]),
    declare_dual_binding('min'/3, 'invokestatic java/lang/Math min (II)I',
        [int, int], [int],
        [pure, deterministic, total, pattern(function)]).

register_jvm_asm_io_bindings :-
    declare_dual_binding('print_int'/2,
        'getstatic java/lang/System out Ljava/io/PrintStream; + invokevirtual java/io/PrintStream println (I)V',
        [int], [],
        [deterministic, pattern(io_op),
         description("Print integer to stdout")]),
    declare_dual_binding('print_str'/2,
        'getstatic java/lang/System out Ljava/io/PrintStream; + invokevirtual java/io/PrintStream println (Ljava/lang/Object;)V',
        ['String'], [],
        [deterministic, pattern(io_op),
         description("Print string to stdout")]).

register_jvm_asm_string_bindings :-
    declare_dual_binding('string_equals'/3,
        'invokevirtual java/lang/String equals (Ljava/lang/Object;)Z',
        ['String', 'String'], [boolean],
        [pure, deterministic, total, pattern(function),
         description("Compare two strings for equality")]),
    declare_dual_binding('string_concat'/3,
        'invokevirtual java/lang/String concat (Ljava/lang/String;)Ljava/lang/String;',
        ['String', 'String'], ['String'],
        [pure, deterministic, total, pattern(function),
         description("Concatenate two strings")]),
    declare_dual_binding('string_length'/2,
        'invokevirtual java/lang/String length ()I',
        ['String'], [int],
        [pure, deterministic, total, pattern(function),
         description("Get string length")]),
    declare_dual_binding('string_valueof'/2,
        'invokestatic java/lang/String valueOf (I)Ljava/lang/String;',
        [int], ['String'],
        [pure, deterministic, total, pattern(function),
         description("Convert int to string")]),
    declare_dual_binding('string_charat'/3,
        'invokevirtual java/lang/String charAt (I)C',
        ['String', int], [char],
        [pure, deterministic, total, pattern(function),
         description("Get character at index")]),
    declare_dual_binding('string_substring'/4,
        'invokevirtual java/lang/String substring (II)Ljava/lang/String;',
        ['String', int, int], ['String'],
        [pure, deterministic, total, pattern(function),
         description("Extract substring by start and end index")]),
    declare_dual_binding('string_tolower'/2,
        'invokevirtual java/lang/String toLowerCase ()Ljava/lang/String;',
        ['String'], ['String'],
        [pure, deterministic, total, pattern(function),
         description("Convert string to lowercase")]),
    declare_dual_binding('string_toupper'/2,
        'invokevirtual java/lang/String toUpperCase ()Ljava/lang/String;',
        ['String'], ['String'],
        [pure, deterministic, total, pattern(function),
         description("Convert string to uppercase")]).
