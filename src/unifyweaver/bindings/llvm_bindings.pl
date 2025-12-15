:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% llvm_bindings.pl - LLVM IR bindings for arithmetic and comparisons
%
% Maps Prolog operations to LLVM IR instructions.

:- module(llvm_bindings, [
    init_llvm_bindings/0,
    llvm_binding/5,            % llvm_binding(Pred, Instruction, Inputs, Outputs, Options)
    llvm_type/2,               % llvm_type(PrologType, LLVMType)
    test_llvm_bindings/0
]).

:- use_module('../core/binding_registry').

%% init_llvm_bindings
init_llvm_bindings :-
    register_llvm_arithmetic_bindings,
    register_llvm_comparison_bindings,
    register_llvm_control_bindings.

%% llvm_binding(?Pred, ?Instruction, ?Inputs, ?Outputs, ?Options)
llvm_binding(Pred, Instruction, Inputs, Outputs, Options) :-
    binding(llvm, Pred, Instruction, Inputs, Outputs, Options).

%% Type mappings
llvm_type(integer, 'i64').
llvm_type(float, 'double').
llvm_type(boolean, 'i1').
llvm_type(pointer, 'i8*').
llvm_type(string, 'i8*').

%% Arithmetic Bindings
register_llvm_arithmetic_bindings :-
    % Addition
    declare_binding(llvm, '+'/3, 'add',
        [i64, i64], [i64],
        [pure, deterministic, total, pattern(binary_op)]),

    % Subtraction
    declare_binding(llvm, '-'/3, 'sub',
        [i64, i64], [i64],
        [pure, deterministic, total, pattern(binary_op)]),

    % Multiplication
    declare_binding(llvm, '*'/3, 'mul',
        [i64, i64], [i64],
        [pure, deterministic, total, pattern(binary_op)]),

    % Division (signed)
    declare_binding(llvm, '//'/3, 'sdiv',
        [i64, i64], [i64],
        [pure, deterministic, partial, pattern(binary_op)]),

    % Modulo (signed remainder)
    declare_binding(llvm, mod/3, 'srem',
        [i64, i64], [i64],
        [pure, deterministic, partial, pattern(binary_op)]),

    % Floating point addition
    declare_binding(llvm, fadd/3, 'fadd',
        [double, double], [double],
        [pure, deterministic, total, pattern(binary_op)]),

    % Floating point subtraction
    declare_binding(llvm, fsub/3, 'fsub',
        [double, double], [double],
        [pure, deterministic, total, pattern(binary_op)]),

    % Floating point multiplication
    declare_binding(llvm, fmul/3, 'fmul',
        [double, double], [double],
        [pure, deterministic, total, pattern(binary_op)]).

%% Comparison Bindings
register_llvm_comparison_bindings :-
    % Equal (signed integer)
    declare_binding(llvm, '=:='/2, 'icmp eq',
        [i64, i64], [i1],
        [pure, deterministic, total, pattern(comparison)]),

    % Not equal
    declare_binding(llvm, '=\\='/2, 'icmp ne',
        [i64, i64], [i1],
        [pure, deterministic, total, pattern(comparison)]),

    % Less than (signed)
    declare_binding(llvm, '<'/2, 'icmp slt',
        [i64, i64], [i1],
        [pure, deterministic, total, pattern(comparison)]),

    % Less than or equal (signed)
    declare_binding(llvm, '=<'/2, 'icmp sle',
        [i64, i64], [i1],
        [pure, deterministic, total, pattern(comparison)]),

    % Greater than (signed)
    declare_binding(llvm, '>'/2, 'icmp sgt',
        [i64, i64], [i1],
        [pure, deterministic, total, pattern(comparison)]),

    % Greater than or equal (signed)
    declare_binding(llvm, '>='/2, 'icmp sge',
        [i64, i64], [i1],
        [pure, deterministic, total, pattern(comparison)]).

%% Control Flow Bindings
register_llvm_control_bindings :-
    % Conditional branch
    declare_binding(llvm, if_then_else/3, 'br',
        [i1, label, label], [],
        [effect(control), pattern(branch)]),

    % Unconditional branch
    declare_binding(llvm, goto/1, 'br',
        [label], [],
        [effect(control), pattern(jump)]),

    % Return
    declare_binding(llvm, return/1, 'ret',
        [i64], [],
        [effect(control), pattern(return)]),

    % Tail call
    declare_binding(llvm, tail_call/2, 'musttail call',
        [function, args], [any],
        [effect(control), pattern(tail_call)]).

%% Tests
test_llvm_bindings :-
    format('[LLVM Bindings] Initializing...~n', []),
    init_llvm_bindings,
    format('[LLVM Bindings] Testing add binding...~n', []),
    (   llvm_binding('+'/3, 'add', _, _, _)
    ->  format('[PASS] add binding exists~n', [])
    ;   format('[FAIL] add binding missing~n', [])
    ),
    format('[LLVM Bindings] Testing icmp slt...~n', []),
    (   llvm_binding('<'/2, 'icmp slt', _, _, _)
    ->  format('[PASS] icmp slt binding exists~n', [])
    ;   format('[FAIL] icmp slt binding missing~n', [])
    ),
    findall(P, llvm_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('[LLVM Bindings] Total: ~w bindings~n', [Count]).
