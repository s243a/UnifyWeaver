:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (@s243a)
%
% cil_bytecode.pl — Shared CIL instruction generation for .NET targets.
% Analogous to jvm_bytecode.pl for JVM targets.
%
% Provides expression compilation, guard translation, and if-chain
% assembly for CIL (Common Intermediate Language) used by ILAsm.

:- module(cil_bytecode, [
    cil_expr_to_instrs/3,           % +Expr, +VarMap, -Instructions
    cil_guard_to_instrs/4,          % +Guard, +VarMap, +FalseLabel, -Instructions
    cil_if_chain/4,                 % +Branches, +N, +Prefix, -Instructions
    cil_resolve_value/3,            % +Value, +VarMap, -Instruction
    cil_cmp_op/3,                   % +PrologOp, -CILBranch, -Negated
    cil_arith_op/2,                 % +PrologOp, -CILOp
    cil_ensure_strings/2            % +Mixed, -Strings
]).

:- use_module(library(lists)).
:- use_module('clause_body_analysis').

% ============================================================================
% EXPRESSION COMPILATION
% ============================================================================

%% cil_expr_to_instrs(+Expr, +VarMap, -Instructions)
%  Compile a Prolog expression to CIL evaluation stack instructions.

%% Variable
cil_expr_to_instrs(Var, VarMap, [LoadInstr]) :-
    var(Var), !,
    lookup_var(Var, VarMap, VarName),
    format(string(LoadInstr), '    ldloc ~w', [VarName]).

%% Integer literal
cil_expr_to_instrs(N, _, [Instr]) :-
    integer(N), !,
    format(string(Instr), '    ldc.i8 ~w', [N]).

%% Float literal
cil_expr_to_instrs(F, _, [Instr]) :-
    float(F), !,
    format(string(Instr), '    ldc.r8 ~w', [F]).

%% Atom/string literal
cil_expr_to_instrs(Atom, _, [Instr]) :-
    atom(Atom), !,
    format(string(Instr), '    ldstr "~w"', [Atom]).

%% String literal
cil_expr_to_instrs(Str, _, [Instr]) :-
    string(Str), !,
    format(string(Instr), '    ldstr "~w"', [Str]).

%% Negation
cil_expr_to_instrs(-Expr, VarMap, Instrs) :-
    !,
    cil_expr_to_instrs(Expr, VarMap, Inner),
    append(Inner, ['    neg'], Instrs).

%% abs(X)
cil_expr_to_instrs(abs(Expr), VarMap, Instrs) :-
    !,
    cil_expr_to_instrs(Expr, VarMap, Inner),
    append(Inner, ['    call int64 [mscorlib]System.Math::Abs(int64)'], Instrs).

%% sqrt(X)
cil_expr_to_instrs(sqrt(Expr), VarMap, Instrs) :-
    !,
    cil_expr_to_instrs(Expr, VarMap, Inner),
    append(Inner, ['    conv.r8', '    call float64 [mscorlib]System.Math::Sqrt(float64)'], Instrs).

%% Binary arithmetic
cil_expr_to_instrs(Expr, VarMap, Instrs) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    cil_arith_op(Op, CILOp),
    !,
    cil_expr_to_instrs(Left, VarMap, LI),
    cil_expr_to_instrs(Right, VarMap, RI),
    format(string(OpInstr), '    ~w', [CILOp]),
    append(LI, RI, LIRI),
    append(LIRI, [OpInstr], Instrs).

%% Fallback: try as value
cil_expr_to_instrs(Value, VarMap, [Instr]) :-
    cil_resolve_value(Value, VarMap, Instr).

%% cil_arith_op(+PrologOp, -CILOp)
cil_arith_op(+, 'add').
cil_arith_op(-, 'sub').
cil_arith_op(*, 'mul').
cil_arith_op(/, 'div').
cil_arith_op(//, 'div').
cil_arith_op(mod, 'rem').

% ============================================================================
% GUARD COMPILATION
% ============================================================================

%% cil_guard_to_instrs(+Guard, +VarMap, +FalseLabel, -Instructions)
%  Compile a guard (comparison) to CIL branch instructions.
%  Branches to FalseLabel if the guard FAILS.

cil_guard_to_instrs(Guard, VarMap, FalseLabel, Instrs) :-
    Guard =.. [Op, Left, Right],
    cil_cmp_op(Op, BranchOp, _Negated),
    !,
    cil_expr_to_instrs(Left, VarMap, LI),
    cil_expr_to_instrs(Right, VarMap, RI),
    format(string(BrInstr), '    ~w ~w', [BranchOp, FalseLabel]),
    append(LI, RI, LIRI),
    append(LIRI, [BrInstr], Instrs).

%% cil_cmp_op(+PrologOp, -CILBranchOnFail, -Negated)
%  Maps Prolog comparison to CIL branch-on-fail instruction.
%  E.g., X > Y fails when X <= Y, so branch is 'ble'.
cil_cmp_op(>,   'ble',   false).
cil_cmp_op(<,   'bge',   false).
cil_cmp_op(>=,  'blt',   false).
cil_cmp_op(=<,  'bgt',   false).
cil_cmp_op(=:=, 'bne.un', false).
cil_cmp_op(=\=, 'beq',   false).
cil_cmp_op(==,  'bne.un', false).
cil_cmp_op(\==, 'beq',   false).

% ============================================================================
% RESOLVE VALUE
% ============================================================================

%% cil_resolve_value(+Value, +VarMap, -Instruction)
cil_resolve_value(Var, VarMap, Instr) :-
    var(Var), !,
    lookup_var(Var, VarMap, VarName),
    format(string(Instr), '    ldloc ~w', [VarName]).
cil_resolve_value(N, _, Instr) :-
    integer(N), !,
    format(string(Instr), '    ldc.i8 ~w', [N]).
cil_resolve_value(F, _, Instr) :-
    float(F), !,
    format(string(Instr), '    ldc.r8 ~w', [F]).
cil_resolve_value(Atom, _, Instr) :-
    atom(Atom), !,
    format(string(Instr), '    ldstr "~w"', [Atom]).
cil_resolve_value(Str, _, Instr) :-
    string(Str), !,
    format(string(Instr), '    ldstr "~w"', [Str]).

% ============================================================================
% IF-CHAIN ASSEMBLY
% ============================================================================

%% cil_if_chain(+Branches, +N, +Prefix, -Instructions)
%  Assemble compiled branches into CIL label-based if-chain.
%  Each branch is branch(GuardInstrs, ValueInstrs).
cil_if_chain([], _, _, ErrInstrs) :- !,
    ErrInstrs = ['    ldstr "No matching clause"',
                 '    newobj instance void [mscorlib]System.Exception::.ctor(string)',
                 '    throw'].

cil_if_chain([branch([], ValueInstrs)], _N, _Prefix, Instrs) :- !,
    %% Unconditional (no guard) — just emit value + ret
    append(ValueInstrs, ['    ret'], Instrs).

cil_if_chain([branch(GuardInstrs, ValueInstrs)|Rest], N, Prefix, Instrs) :-
    N1 is N + 1,
    format(string(NextLabel), '~w_~w', [Prefix, N1]),
    %% Replace "NEXT" placeholder in guard instrs with actual label
    maplist(cil_replace_label("NEXT", NextLabel), GuardInstrs, PatchedGuards),
    append(PatchedGuards, ValueInstrs, PreRet),
    append(PreRet, ['    ret'], BranchInstrs),
    format(string(LabelLine), '~w:', [NextLabel]),
    cil_if_chain(Rest, N1, Prefix, RestInstrs),
    append(BranchInstrs, [LabelLine|RestInstrs], Instrs).

%% cil_replace_label(+Old, +New, +Instr, -Patched)
cil_replace_label(Old, New, Instr, Patched) :-
    (   string(Instr), sub_string(Instr, _, _, _, Old)
    ->  split_string(Instr, "", "", Chars),
        split_string(Old, "", "", OldChars),
        split_string(New, "", "", NewChars),
        cil_string_replace(Chars, OldChars, NewChars, PatchedChars),
        atomic_list_concat(PatchedChars, Patched)
    ;   atom(Instr), atom_string(Instr, InstrStr), sub_string(InstrStr, _, _, _, Old)
    ->  atom_string(Old, OldStr), atom_string(New, NewStr),
        atomic_list_concat(Parts, OldStr, Instr),
        atomic_list_concat(Parts, NewStr, Patched)
    ;   Patched = Instr
    ).

cil_string_replace([], _, _, []).
cil_string_replace(String, Old, New, Result) :-
    append(Old, Rest, String), !,
    append(New, ResultRest, Result),
    cil_string_replace(Rest, Old, New, ResultRest).
cil_string_replace([H|T], Old, New, [H|Result]) :-
    cil_string_replace(T, Old, New, Result).

% ============================================================================
% UTILITY
% ============================================================================

cil_ensure_strings([], []).
cil_ensure_strings([H|T], [S|Rest]) :-
    (string(H) -> S = H ; atom_string(H, S)),
    cil_ensure_strings(T, Rest).
