:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (@s243a)
%
% cil_asm_bindings.pl — .NET CIL Assembly Bindings
% Registers bindings for the ILAsm target.
% Maps Prolog operations to CIL instructions and .NET BCL calls.

:- module(cil_asm_bindings, [init_cil_asm_bindings/0]).

:- use_module('../core/binding_registry').

init_cil_asm_bindings :-
    register_cil_arithmetic_bindings,
    register_cil_comparison_bindings,
    register_cil_math_bindings,
    register_cil_conversion_bindings,
    register_cil_string_bindings,
    register_cil_io_bindings.

% ============================================================================
% ARITHMETIC BINDINGS
% ============================================================================

register_cil_arithmetic_bindings :-
    declare_binding(ilasm, add/3, 'add', [int64, int64], [int64],
        [pure, deterministic]),
    declare_binding(ilasm, sub/3, 'sub', [int64, int64], [int64],
        [pure, deterministic]),
    declare_binding(ilasm, mul/3, 'mul', [int64, int64], [int64],
        [pure, deterministic]),
    declare_binding(ilasm, div/3, 'div', [int64, int64], [int64],
        [pure, deterministic]),
    declare_binding(ilasm, mod/3, 'rem', [int64, int64], [int64],
        [pure, deterministic]),
    declare_binding(ilasm, neg/2, 'neg', [int64], [int64],
        [pure, deterministic]).

% ============================================================================
% COMPARISON BINDINGS (as guards)
% ============================================================================

register_cil_comparison_bindings :-
    declare_binding(ilasm, gt/2, 'cgt', [int64, int64], [],
        [pure, deterministic, pattern(command)]),
    declare_binding(ilasm, lt/2, 'clt', [int64, int64], [],
        [pure, deterministic, pattern(command)]),
    declare_binding(ilasm, eq/2, 'ceq', [int64, int64], [],
        [pure, deterministic, pattern(command)]).

% ============================================================================
% MATH BINDINGS (System.Math calls)
% ============================================================================

register_cil_math_bindings :-
    declare_binding(ilasm, sqrt/2,
        'call float64 [mscorlib]System.Math::Sqrt(float64)',
        [float64], [float64], [pure, deterministic, import(system_math)]),
    declare_binding(ilasm, abs/2,
        'call int64 [mscorlib]System.Math::Abs(int64)',
        [int64], [int64], [pure, deterministic, import(system_math)]),
    declare_binding(ilasm, max_val/3,
        'call int64 [mscorlib]System.Math::Max(int64, int64)',
        [int64, int64], [int64], [pure, deterministic, import(system_math)]),
    declare_binding(ilasm, min_val/3,
        'call int64 [mscorlib]System.Math::Min(int64, int64)',
        [int64, int64], [int64], [pure, deterministic, import(system_math)]),
    declare_binding(ilasm, pow/3,
        'call float64 [mscorlib]System.Math::Pow(float64, float64)',
        [float64, float64], [float64], [pure, deterministic, import(system_math)]).

% ============================================================================
% TYPE CONVERSION BINDINGS
% ============================================================================

register_cil_conversion_bindings :-
    declare_binding(ilasm, to_int32/2, 'conv.i4', [int64], [int32],
        [pure, deterministic]),
    declare_binding(ilasm, to_int64/2, 'conv.i8', [int32], [int64],
        [pure, deterministic]),
    declare_binding(ilasm, to_float64/2, 'conv.r8', [int64], [float64],
        [pure, deterministic]),
    declare_binding(ilasm, to_string/2,
        'call string [mscorlib]System.Convert::ToString(int64)',
        [int64], [string], [pure, deterministic]).

% ============================================================================
% STRING BINDINGS
% ============================================================================

register_cil_string_bindings :-
    declare_binding(ilasm, length/2,
        'callvirt instance int32 [mscorlib]System.String::get_Length()',
        [string], [int32], [pure, deterministic]),
    declare_binding(ilasm, string_concat/3,
        'call string [mscorlib]System.String::Concat(string, string)',
        [string, string], [string], [pure, deterministic]),
    declare_binding(ilasm, to_upper/2,
        'callvirt instance string [mscorlib]System.String::ToUpper()',
        [string], [string], [pure, deterministic]),
    declare_binding(ilasm, to_lower/2,
        'callvirt instance string [mscorlib]System.String::ToLower()',
        [string], [string], [pure, deterministic]).

% ============================================================================
% I/O BINDINGS
% ============================================================================

register_cil_io_bindings :-
    declare_binding(ilasm, print/1,
        'call void [mscorlib]System.Console::WriteLine(int64)',
        [int64], [], [effect(io), deterministic]),
    declare_binding(ilasm, print_string/1,
        'call void [mscorlib]System.Console::WriteLine(string)',
        [string], [], [effect(io), deterministic]),
    declare_binding(ilasm, read_line/1,
        'call string [mscorlib]System.Console::ReadLine()',
        [], [string], [effect(io), deterministic]).
