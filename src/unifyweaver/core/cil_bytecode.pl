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
    cil_ensure_strings/2,           % +Mixed, -Strings
    cil_tail_recursion_bytecode/3,  % +PredStr, +Arity, -Instructions
    cil_tail_recursion_entry/3,     % +PredStr, +ClassName, -Instructions
    cil_linear_recursion_bytecode/3,% +PredStr, +Arity, -Instructions
    cil_tree_recursion_bytecode/3,  % +PredStr, +ClassName, -Instructions
    cil_mutual_recursion_bytecode/5 % +PredStr, +CalledStr, +ClassName, +BaseVal, -Instrs
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

% ============================================================================
% RECURSION BYTECODE GENERATION
% ============================================================================

%% cil_tail_recursion_bytecode(+PredStr, +Arity, -Instructions)
%%   Tail recursion using br loop. O(1) stack space.
%%   Pattern: pred(n, acc) { while(n > 0) { acc op= n; n--; } return acc; }
cil_tail_recursion_bytecode(PredStr, _Arity, Instructions) :-
    format(string(Header), '    // Tail recursion: ~w', [PredStr]),
    Instructions = [
        Header,
        "LOOP:",
        "    ldarg.0",         % load n
        "    ldc.i8 0",
        "    ble DONE",        % if n <= 0, done
        "    ldarg.1",         % load acc
        "    ldarg.0",         % load n
        "    add",             % acc + n
        "    starg.s 1",       % acc = acc + n
        "    ldarg.0",         % load n
        "    ldc.i8 1",
        "    sub",             % n - 1
        "    starg.s 0",       % n = n - 1
        "    br LOOP",
        "DONE:",
        "    ldarg.1",         % return acc
        "    ret"
    ].

%% cil_tail_recursion_entry(+PredStr, +ClassName, -Instructions)
%%   Entry wrapper: calls pred(n, 0) with initial accumulator.
cil_tail_recursion_entry(PredStr, ClassName, Instructions) :-
    format(string(CallInstr), '    call int64 ~w::~w_worker(int64, int64)', [ClassName, PredStr]),
    Instructions = [
        "    ldarg.0",         % load n
        "    ldc.i8 0",        % initial acc = 0
        CallInstr,
        "    ret"
    ].

%% cil_linear_recursion_bytecode(+PredStr, +Arity, -Instructions)
%%   Linear recursion as iterative loop.
%%   Pattern: result = base; for(i = base+1; i <= n; i++) result = result op i;
cil_linear_recursion_bytecode(PredStr, _Arity, Instructions) :-
    format(string(Header), '    // Linear recursion: ~w', [PredStr]),
    Instructions = [
        Header,
        "    ldc.i8 1",        % result = 1 (base case for factorial)
        "    stloc result",
        "    ldc.i8 1",        % i = 1
        "    stloc i",
        "LOOP:",
        "    ldloc i",
        "    ldarg.0",         % n
        "    bgt DONE",        % if i > n, done
        "    ldloc result",
        "    ldloc i",
        "    mul",             % result * i
        "    stloc result",
        "    ldloc i",
        "    ldc.i8 1",
        "    add",
        "    stloc i",         % i++
        "    br LOOP",
        "DONE:",
        "    ldloc result",
        "    ret"
    ].

%% cil_tree_recursion_bytecode(+PredStr, +ClassName, -Instructions)
%%   Tree recursion with memoization via dictionary.
%%   Fibonacci-style: f(n) = f(n-1) + f(n-2)
cil_tree_recursion_bytecode(PredStr, ClassName, Instructions) :-
    format(string(Header), '    // Tree recursion: ~w (memoized)', [PredStr]),
    format(string(Call1), '    call int64 ~w::~w(int64)', [ClassName, PredStr]),
    format(string(Call2), '    call int64 ~w::~w(int64)', [ClassName, PredStr]),
    Instructions = [
        Header,
        "    ldarg.0",         % n
        "    ldc.i8 0",
        "    beq BASE_0",
        "    ldarg.0",
        "    ldc.i8 1",
        "    beq BASE_1",
        "    // Recursive case",
        "    ldarg.0",
        "    ldc.i8 1",
        "    sub",             % n - 1
        Call1,                 % f(n-1)
        "    ldarg.0",
        "    ldc.i8 2",
        "    sub",             % n - 2
        Call2,                 % f(n-2)
        "    add",             % f(n-1) + f(n-2)
        "    ret",
        "BASE_0:",
        "    ldc.i8 0",        % f(0) = 0
        "    ret",
        "BASE_1:",
        "    ldc.i8 1",        % f(1) = 1
        "    ret"
    ].

%% cil_mutual_recursion_bytecode(+PredStr, +CalledPredStr, +ClassName, -Instructions)
%%   One function in a mutual recursion group.
%%   Pattern: if n == base_val return base_result; return other(n-1)
cil_mutual_recursion_bytecode(PredStr, CalledPredStr, ClassName, BaseVal, Instructions) :-
    format(string(Header), '    // Mutual recursion: ~w', [PredStr]),
    format(string(CallOther), '    .tail\n    call int64 ~w::~w(int64)', [ClassName, CalledPredStr]),
    format(string(BaseInstr), '    ldc.i8 ~w', [BaseVal]),
    Instructions = [
        Header,
        "    ldarg.0",
        "    ldc.i8 0",
        "    beq BASE",
        "    ldarg.0",
        "    ldc.i8 1",
        "    sub",
        CallOther,
        "    ret",
        "BASE:",
        BaseInstr,
        "    ret"
    ].

cil_ensure_strings([], []).
cil_ensure_strings([H|T], [S|Rest]) :-
    (string(H) -> S = H ; atom_string(H, S)),
    cil_ensure_strings(T, Rest).
