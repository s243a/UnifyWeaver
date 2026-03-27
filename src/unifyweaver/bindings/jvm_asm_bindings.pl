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
    register_jvm_asm_io_bindings.

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
