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
    register_wat_memory_bindings,
    register_wat_bulk_memory_bindings,
    register_wat_simd_bindings.

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

%% Bulk memory operations (WebAssembly bulk-memory-operations proposal)
register_wat_bulk_memory_bindings :-
    declare_binding(wat, 'memory_copy'/4, 'memory.copy', [i32, i32, i32], [],
        [deterministic, pattern(bulk_memory_op),
         description("Copy N bytes from src to dest in linear memory")]),
    declare_binding(wat, 'memory_fill'/4, 'memory.fill', [i32, i32, i32], [],
        [deterministic, pattern(bulk_memory_op),
         description("Fill N bytes at dest with a byte value")]),
    declare_binding(wat, 'memory_init'/5, 'memory.init', [i32, i32, i32], [],
        [deterministic, pattern(bulk_memory_op),
         description("Copy from passive data segment to linear memory")]),
    declare_binding(wat, 'data_drop'/2, 'data.drop', [], [],
        [deterministic, pattern(bulk_memory_op),
         description("Drop a passive data segment")]).

%% SIMD bindings (128-bit packed vectors)
register_wat_simd_bindings :-
    % i64x2 operations (2 x i64 lanes)
    declare_binding(wat, 'v128_i64x2_add'/3, 'i64x2.add', [v128, v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    declare_binding(wat, 'v128_i64x2_sub'/3, 'i64x2.sub', [v128, v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    declare_binding(wat, 'v128_i64x2_mul'/3, 'i64x2.mul', [v128, v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    declare_binding(wat, 'v128_i64x2_neg'/2, 'i64x2.neg', [v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    declare_binding(wat, 'v128_i64x2_eq'/3, 'i64x2.eq', [v128, v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    % i32x4 operations (4 x i32 lanes)
    declare_binding(wat, 'v128_i32x4_add'/3, 'i32x4.add', [v128, v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    declare_binding(wat, 'v128_i32x4_sub'/3, 'i32x4.sub', [v128, v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    declare_binding(wat, 'v128_i32x4_mul'/3, 'i32x4.mul', [v128, v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    % v128 bitwise
    declare_binding(wat, 'v128_and'/3, 'v128.and', [v128, v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    declare_binding(wat, 'v128_or'/3, 'v128.or', [v128, v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    declare_binding(wat, 'v128_xor'/3, 'v128.xor', [v128, v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    declare_binding(wat, 'v128_not'/2, 'v128.not', [v128], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    % Lane extract/replace
    declare_binding(wat, 'v128_i64x2_extract'/3, 'i64x2.extract_lane', [v128], [i64],
        [pure, deterministic, total, pattern(simd_lane_op),
         description("Extract i64 from lane 0 or 1")]),
    declare_binding(wat, 'v128_i64x2_replace'/4, 'i64x2.replace_lane', [v128, i64], [v128],
        [pure, deterministic, total, pattern(simd_lane_op),
         description("Replace i64 at lane 0 or 1")]),
    % Splat (broadcast scalar to all lanes)
    declare_binding(wat, 'v128_i64x2_splat'/2, 'i64x2.splat', [i64], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    declare_binding(wat, 'v128_i32x4_splat'/2, 'i32x4.splat', [i32], [v128],
        [pure, deterministic, total, pattern(simd_op)]),
    % Load/store v128
    declare_binding(wat, 'v128_load'/2, 'v128.load', [i32], [v128],
        [deterministic, pattern(simd_memory_op)]),
    declare_binding(wat, 'v128_store'/3, 'v128.store', [i32, v128], [],
        [deterministic, pattern(simd_memory_op)]).
