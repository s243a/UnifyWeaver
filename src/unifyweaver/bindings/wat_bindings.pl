:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wat_bindings.pl - WebAssembly Text Format Bindings
% Maps Prolog operations to WAT instructions.

:- module(wat_bindings, [init_wat_bindings/0]).

:- use_module('../core/binding_registry').

init_wat_bindings :-
    register_wat_arithmetic_bindings,
    register_wat_comparison_bindings,
    register_wat_bitwise_bindings,
    register_wat_conversion_bindings,
    register_wat_memory_bindings.

register_wat_arithmetic_bindings :-
    declare_binding(wat, '+'/3, 'i64.add', [i64, i64], [i64],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_binding(wat, '-'/3, 'i64.sub', [i64, i64], [i64],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_binding(wat, '*'/3, 'i64.mul', [i64, i64], [i64],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_binding(wat, '//'/3, 'i64.div_s', [i64, i64], [i64],
        [pure, deterministic, partial, pattern(binary_op)]),
    declare_binding(wat, '/'/3, 'i64.div_s', [i64, i64], [i64],
        [pure, deterministic, partial, pattern(binary_op)]),
    declare_binding(wat, 'mod'/3, 'i64.rem_s', [i64, i64], [i64],
        [pure, deterministic, partial, pattern(binary_op)]).

register_wat_comparison_bindings :-
    declare_binding(wat, '>'/2, 'i64.gt_s', [i64, i64], [i32],
        [pure, deterministic, total, pattern(comparison)]),
    declare_binding(wat, '<'/2, 'i64.lt_s', [i64, i64], [i32],
        [pure, deterministic, total, pattern(comparison)]),
    declare_binding(wat, '>='/2, 'i64.ge_s', [i64, i64], [i32],
        [pure, deterministic, total, pattern(comparison)]),
    declare_binding(wat, '=<'/2, 'i64.le_s', [i64, i64], [i32],
        [pure, deterministic, total, pattern(comparison)]),
    declare_binding(wat, '=:='/2, 'i64.eq', [i64, i64], [i32],
        [pure, deterministic, total, pattern(comparison)]),
    declare_binding(wat, '=\\='/2, 'i64.ne', [i64, i64], [i32],
        [pure, deterministic, total, pattern(comparison)]).

register_wat_bitwise_bindings :-
    declare_binding(wat, '/\\'/3, 'i64.and', [i64, i64], [i64],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_binding(wat, '\\/'/3, 'i64.or', [i64, i64], [i64],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_binding(wat, 'xor'/3, 'i64.xor', [i64, i64], [i64],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_binding(wat, '<<'/3, 'i64.shl', [i64, i64], [i64],
        [pure, deterministic, total, pattern(binary_op)]),
    declare_binding(wat, '>>'/3, 'i64.shr_s', [i64, i64], [i64],
        [pure, deterministic, total, pattern(binary_op)]).

register_wat_conversion_bindings :-
    declare_binding(wat, 'float_integer'/2, 'i64.trunc_f64_s', [f64], [i64],
        [pure, deterministic, partial, pattern(function)]),
    declare_binding(wat, 'integer_float'/2, 'f64.convert_i64_s', [i64], [f64],
        [pure, deterministic, total, pattern(function)]),
    declare_binding(wat, 'i32_wrap'/2, 'i32.wrap_i64', [i64], [i32],
        [pure, deterministic, total, pattern(function)]),
    declare_binding(wat, 'i64_extend'/2, 'i64.extend_i32_s', [i32], [i64],
        [pure, deterministic, total, pattern(function)]).

register_wat_memory_bindings :-
    declare_binding(wat, 'memory_load_i64'/2, 'i64.load', [i32], [i64],
        [deterministic, pattern(memory_op)]),
    declare_binding(wat, 'memory_store_i64'/3, 'i64.store', [i32, i64], [],
        [deterministic, pattern(memory_op)]),
    declare_binding(wat, 'memory_load_i32'/2, 'i32.load', [i32], [i32],
        [deterministic, pattern(memory_op)]),
    declare_binding(wat, 'memory_store_i32'/3, 'i32.store', [i32, i32], [],
        [deterministic, pattern(memory_op)]),
    declare_binding(wat, 'memory_size'/1, 'memory.size', [], [i32],
        [deterministic, pattern(memory_op)]),
    declare_binding(wat, 'memory_grow'/2, 'memory.grow', [i32], [i32],
        [deterministic, pattern(memory_op)]).
