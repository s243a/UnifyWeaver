:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% jvm_bytecode.pl - Shared JVM Bytecode Infrastructure
% Common bytecode generation logic used by Jamaica and Krakatau targets.
% Handles instruction sequences, stack tracking, expression compilation,
% and control flow patterns independent of output syntax.

:- module(jvm_bytecode, [
    % Expression compilation to bytecode sequences
    jvm_expr_to_bytecode/4,          % +Expr, +VarMap, +VarStyle, -Instructions
    jvm_guard_to_bytecode/5,         % +Guard, +VarMap, +VarStyle, +Label, -Instructions
    jvm_arith_to_bytecode/4,         % +ArithExpr, +VarMap, +VarStyle, -Instructions

    % Instruction helpers
    jvm_load_var/4,                  % +Var, +VarMap, +VarStyle, -Instruction
    jvm_store_var/4,                 % +Var, +VarMap, +VarStyle, -Instruction
    jvm_load_const/2,                % +Value, -Instructions
    jvm_return_type/2,               % +Type, -ReturnInstr

    % Type descriptors
    jvm_type_descriptor/2,           % +JavaType, -JVMDescriptor
    jvm_method_descriptor/3,         % +ParamTypes, +ReturnType, -Descriptor

    % Control flow
    jvm_if_chain_to_bytecode/5,      % +Branches, +VarMap, +VarStyle, +LabelPrefix, -Instructions

    % Native clause lowering (shared logic)
    jvm_native_clause_body/5,        % +PredStr/Arity, +Clauses, +VarStyle, +OutputFmt, -Code

    % Stack depth estimation
    jvm_estimate_stack_depth/2       % +Instructions, -Depth
]).

:- use_module('../core/clause_body_analysis').

%% ============================================
%% VARIABLE STYLES
%% ============================================
%%
%% VarStyle controls how variables are referenced:
%%   symbolic  — Jamaica: iload varname
%%   numeric   — Krakatau: iload_N or iload N

%% jvm_load_var(+Var, +VarMap, +VarStyle, -Instruction)
jvm_load_var(Var, VarMap, VarStyle, Instr) :-
    lookup_var(Var, VarMap, Name),
    (   VarStyle = symbolic
    ->  format(string(Instr), '    iload ~w', [Name])
    ;   % numeric: convert "argN" to slot index (0-based)
        var_name_to_slot(Name, Slot),
        (   Slot < 4
        ->  format(string(Instr), '    iload_~w', [Slot])
        ;   format(string(Instr), '    iload ~w', [Slot])
        )
    ).

%% jvm_store_var(+Var, +VarMap, +VarStyle, -Instruction)
jvm_store_var(Var, VarMap, VarStyle, Instr) :-
    lookup_var(Var, VarMap, Name),
    (   VarStyle = symbolic
    ->  format(string(Instr), '    istore ~w', [Name])
    ;   var_name_to_slot(Name, Slot),
        (   Slot < 4
        ->  format(string(Instr), '    istore_~w', [Slot])
        ;   format(string(Instr), '    istore ~w', [Slot])
        )
    ).

%% var_name_to_slot(+Name, -Slot)
%%   Convert "argN" to slot index N-1 (0-based for static methods).
var_name_to_slot(Name, Slot) :-
    (   number(Name)
    ->  Slot = Name
    ;   atom_string(NameAtom, Name),
        atom_concat(arg, NumAtom, NameAtom),
        atom_number(NumAtom, N),
        Slot is N - 1
    ).

%% jvm_load_const(+Value, -Instructions)
%%   Load a constant onto the stack.
jvm_load_const(Value, Instr) :-
    integer(Value),
    !,
    (   Value >= -1, Value =< 5
    ->  format(string(Instr), '    iconst_~w', [Value])
    ;   Value >= -128, Value =< 127
    ->  format(string(Instr), '    bipush ~w', [Value])
    ;   Value >= -32768, Value =< 32767
    ->  format(string(Instr), '    sipush ~w', [Value])
    ;   format(string(Instr), '    ldc ~w', [Value])
    ).
jvm_load_const(Value, Instr) :-
    atom(Value),
    !,
    format(string(Instr), '    ldc "~w"', [Value]).
jvm_load_const(Value, Instr) :-
    number(Value),
    format(string(Instr), '    ldc ~w', [Value]).

%% jvm_return_type(+Type, -ReturnInstr)
jvm_return_type(int, "    ireturn").
jvm_return_type(long, "    lreturn").
jvm_return_type(float, "    freturn").
jvm_return_type(double, "    dreturn").
jvm_return_type(void, "    return").
jvm_return_type(ref, "    areturn").

%% ============================================
%% TYPE DESCRIPTORS
%% ============================================

jvm_type_descriptor(int, "I").
jvm_type_descriptor(long, "J").
jvm_type_descriptor(float, "F").
jvm_type_descriptor(double, "D").
jvm_type_descriptor(void, "V").
jvm_type_descriptor(boolean, "Z").
jvm_type_descriptor(byte, "B").
jvm_type_descriptor(char, "C").
jvm_type_descriptor(short, "S").
jvm_type_descriptor('String', "Ljava/lang/String;").
jvm_type_descriptor('Object', "Ljava/lang/Object;").

jvm_method_descriptor(ParamTypes, ReturnType, Descriptor) :-
    maplist(jvm_type_descriptor, ParamTypes, ParamDescs),
    atomic_list_concat(ParamDescs, '', ParamStr),
    jvm_type_descriptor(ReturnType, RetDesc),
    format(string(Descriptor), '(~w)~w', [ParamStr, RetDesc]).

%% ============================================
%% EXPRESSION COMPILATION
%% ============================================

%% jvm_expr_to_bytecode(+Expr, +VarMap, +VarStyle, -Instructions)
%%   Compile an expression to a list of JVM bytecode instructions.
%%   Result is left on the operand stack.
jvm_expr_to_bytecode(Expr, VarMap, VarStyle, Instructions) :-
    var(Expr),
    !,
    jvm_load_var(Expr, VarMap, VarStyle, Instr),
    Instructions = [Instr].
jvm_expr_to_bytecode(Expr, _VarMap, _VarStyle, Instructions) :-
    integer(Expr),
    !,
    jvm_load_const(Expr, Instr),
    Instructions = [Instr].
jvm_expr_to_bytecode(Expr, VarMap, VarStyle, Instructions) :-
    atom(Expr),
    lookup_var(Expr, VarMap, _),
    !,
    jvm_load_var(Expr, VarMap, VarStyle, Instr),
    Instructions = [Instr].
jvm_expr_to_bytecode(Expr, _VarMap, _VarStyle, Instructions) :-
    atom(Expr),
    !,
    jvm_load_const(Expr, Instr),
    Instructions = [Instr].
jvm_expr_to_bytecode(abs(Expr), VarMap, VarStyle, Instructions) :-
    !,
    jvm_expr_to_bytecode(Expr, VarMap, VarStyle, Inner),
    append(Inner,
        ["    invokestatic java/lang/Math abs (I)I"],
        Instructions).
jvm_expr_to_bytecode(-Expr, VarMap, VarStyle, Instructions) :-
    !,
    jvm_expr_to_bytecode(Expr, VarMap, VarStyle, Inner),
    append(Inner, ["    ineg"], Instructions).
jvm_expr_to_bytecode(Expr, VarMap, VarStyle, Instructions) :-
    Expr =.. [Op, Left, Right],
    jvm_arith_op(Op, Instr),
    !,
    jvm_expr_to_bytecode(Left, VarMap, VarStyle, LeftInstrs),
    jvm_expr_to_bytecode(Right, VarMap, VarStyle, RightInstrs),
    append(LeftInstrs, RightInstrs, Operands),
    append(Operands, [Instr], Instructions).

jvm_arith_op(+, "    iadd").
jvm_arith_op(-, "    isub").
jvm_arith_op(*, "    imul").
jvm_arith_op(/, "    idiv").
jvm_arith_op(//, "    idiv").
jvm_arith_op(mod, "    irem").

%% jvm_arith_to_bytecode(+ArithExpr, +VarMap, +VarStyle, -Instructions)
%%   Alias for jvm_expr_to_bytecode for arithmetic expressions.
jvm_arith_to_bytecode(Expr, VarMap, VarStyle, Instructions) :-
    jvm_expr_to_bytecode(Expr, VarMap, VarStyle, Instructions).

%% ============================================
%% GUARD COMPILATION
%% ============================================

%% jvm_guard_to_bytecode(+Guard, +VarMap, +VarStyle, +FailLabel, -Instructions)
%%   Compile a guard condition. If the guard fails, branches to FailLabel.
jvm_guard_to_bytecode(Guard, VarMap, VarStyle, FailLabel, Instructions) :-
    Guard =.. [Op, Left, Right],
    jvm_cmp_branch(Op, BranchInstr),
    !,
    jvm_expr_to_bytecode(Left, VarMap, VarStyle, LeftInstrs),
    jvm_expr_to_bytecode(Right, VarMap, VarStyle, RightInstrs),
    % Branch to fail label if condition is FALSE (negate the comparison)
    format(string(Branch), '    ~w ~w', [BranchInstr, FailLabel]),
    append(LeftInstrs, RightInstrs, Operands),
    append(Operands, [Branch], Instructions).

%% jvm_cmp_branch(+PrologOp, -NegatedBranchInstr)
%%   The negated branch: if cond is FALSE, jump to fail label.
jvm_cmp_branch(>,  "if_icmple").
jvm_cmp_branch(<,  "if_icmpge").
jvm_cmp_branch(>=, "if_icmplt").
jvm_cmp_branch(=<, "if_icmpgt").
jvm_cmp_branch(=:=, "if_icmpne").
jvm_cmp_branch(=\=, "if_icmpeq").
jvm_cmp_branch(==, "if_icmpne").
jvm_cmp_branch(\==, "if_icmpeq").

%% ============================================
%% IF/ELSE CHAIN (native clause lowering)
%% ============================================

%% jvm_if_chain_to_bytecode(+Branches, +VarMap, +VarStyle, +LabelPrefix, -Instructions)
%%   Compile branches into a JVM if/else bytecode chain.
%%   Branches: list of branch(Guards, ValueInstrs) where
%%     Guards: list of guard goals ([] = default)
%%     ValueInstrs: list of bytecode instructions for the return value
jvm_if_chain_to_bytecode([], _VarMap, _VarStyle, _Prefix, []) :- !.
jvm_if_chain_to_bytecode([branch([], ValueInstrs)], _VarMap, _VarStyle, _Prefix, Instructions) :-
    !,
    % Default branch (no guards)
    append(ValueInstrs, ["    ireturn"], Instructions).
jvm_if_chain_to_bytecode(Branches, VarMap, VarStyle, Prefix, Instructions) :-
    jvm_if_chain_numbered(Branches, 0, VarMap, VarStyle, Prefix, Instructions).

jvm_if_chain_numbered([], _N, _VarMap, _VarStyle, _Prefix, ErrInstrs) :-
    !,
    % No matching clause — throw
    ErrInstrs = [
        "    new java/lang/RuntimeException",
        "    dup",
        "    ldc \"No matching clause\"",
        "    invokespecial java/lang/RuntimeException <init> (Ljava/lang/String;)V",
        "    athrow"
    ].
jvm_if_chain_numbered([branch([], ValueInstrs)|_], _N, _VarMap, _VarStyle, _Prefix, Instructions) :-
    !,
    append(ValueInstrs, ["    ireturn"], Instructions).
jvm_if_chain_numbered([branch(Guards, ValueInstrs)|Rest], N, VarMap, VarStyle, Prefix, Instructions) :-
    N1 is N + 1,
    format(string(NextLabel), '~w_~w', [Prefix, N1]),
    format(string(EndLabel), '~w_end', [Prefix]),
    % Compile guards — each guard branches to NextLabel on failure
    maplist(jvm_guard_to_bytecode_with(VarMap, VarStyle, NextLabel), Guards, GuardInstrLists),
    append(GuardInstrLists, FlatGuards),
    % Value + return
    append(ValueInstrs, ["    ireturn"], ValueWithRet),
    % Jump over remaining branches
    format(string(GotoEnd), '    goto ~w', [EndLabel]),
    % Next label
    format(string(NextLabelDecl), '~w:', [NextLabel]),
    % Rest of chain
    jvm_if_chain_numbered(Rest, N1, VarMap, VarStyle, Prefix, RestInstrs),
    % Assemble
    append(FlatGuards, ValueWithRet, GuardAndValue),
    append(GuardAndValue, [GotoEnd, NextLabelDecl], WithLabel),
    append(WithLabel, RestInstrs, BeforeEnd),
    % Add end label only at top level (N=0)
    (   N =:= 0
    ->  format(string(EndLabelDecl), '~w:', [EndLabel]),
        append(BeforeEnd, [EndLabelDecl], Instructions)
    ;   Instructions = BeforeEnd
    ).

jvm_guard_to_bytecode_with(VarMap, VarStyle, Label, Guard, Instructions) :-
    jvm_guard_to_bytecode(Guard, VarMap, VarStyle, Label, Instructions).

%% ============================================
%% NATIVE CLAUSE BODY LOWERING
%% ============================================

%% jvm_native_clause_body(+PredStr/Arity, +Clauses, +VarStyle, +OutputFmt, -Code)
%%   Shared native clause lowering for JVM bytecode targets.
%%   OutputFmt: format(FormatPred) where FormatPred formats instruction list to string.
%%   Returns the bytecode instruction list as a newline-joined string.
jvm_native_clause_body(PredStr/Arity, Clauses, VarStyle, _OutputFmt, Code) :-
    Arity1 is Arity - 1,
    findall(branch(GuardInstrs, ValueInstrs), (
        member(Head-Body, Clauses),
        Head =.. [_|AllArgs],
        length(InputArgs, Arity1),
        append(InputArgs, [OutputArg], AllArgs),
        build_head_varmap(InputArgs, 1, VarMap),
        Body \== true,
        normalize_goals(Body, Goals),
        once(clause_guard_output_split(Goals, VarMap, Guards, Outputs)),
        % Compile guards to bytecode
        maplist(jvm_guard_goal_to_bytecode(VarMap, VarStyle), Guards, GuardInstrLists),
        append(GuardInstrLists, GuardInstrs),
        % Compile output to bytecode
        jvm_output_to_bytecode(OutputArg, Outputs, VarMap, VarStyle, ValueInstrs)
    ), Branches),
    Branches \= [],
    jvm_if_chain_to_bytecode(Branches, _, VarStyle, "CL", AllInstrs),
    maplist(ensure_string, AllInstrs, StrInstrs),
    atomic_list_concat(StrInstrs, '\n', Code).

jvm_guard_goal_to_bytecode(VarMap, VarStyle, Guard, Instructions) :-
    % Use a placeholder label — will be replaced in if_chain
    jvm_guard_to_bytecode(Guard, VarMap, VarStyle, "NEXT", Instructions).

jvm_output_to_bytecode(OutputArg, _Outputs, _VarMap, _VarStyle, Instructions) :-
    nonvar(OutputArg),
    integer(OutputArg),
    !,
    jvm_load_const(OutputArg, Instr),
    Instructions = [Instr].
jvm_output_to_bytecode(OutputArg, _Outputs, _VarMap, _VarStyle, Instructions) :-
    nonvar(OutputArg),
    atom(OutputArg),
    !,
    % Atoms become their hash code (integer) for JVM bytecode targets
    atom_codes(OutputArg, Codes),
    hash_atom_codes(Codes, 0, Hash),
    jvm_load_const(Hash, Instr),
    Instructions = [Instr].
jvm_output_to_bytecode(_OutputArg, Outputs, VarMap, VarStyle, Instructions) :-
    last(Outputs, LastGoal),
    jvm_goal_to_bytecode(LastGoal, VarMap, VarStyle, Instructions).

jvm_goal_to_bytecode(Goal, VarMap, VarStyle, Instructions) :-
    Goal = (_Var is ArithExpr),
    !,
    jvm_expr_to_bytecode(ArithExpr, VarMap, VarStyle, Instructions).
jvm_goal_to_bytecode(Goal, VarMap, VarStyle, Instructions) :-
    Goal = (_Var = RHS),
    !,
    jvm_expr_to_bytecode(RHS, VarMap, VarStyle, Instructions).
jvm_goal_to_bytecode(Goal, VarMap, VarStyle, Instructions) :-
    jvm_expr_to_bytecode(Goal, VarMap, VarStyle, Instructions).

hash_atom_codes([], Acc, Acc).
hash_atom_codes([C|Cs], Acc, Hash) :-
    Acc1 is (Acc * 31 + C) mod 2147483647,
    hash_atom_codes(Cs, Acc1, Hash).

ensure_string(S, S) :- string(S), !.
ensure_string(A, S) :- atom_string(A, S).

%% ============================================
%% STACK DEPTH ESTIMATION
%% ============================================

jvm_estimate_stack_depth(Instructions, Depth) :-
    estimate_depth(Instructions, 0, 0, Depth).

estimate_depth([], _Current, Max, Max).
estimate_depth([Instr|Rest], Current, Max, Result) :-
    instr_stack_effect(Instr, Effect),
    New is Current + Effect,
    NewMax is max(Max, New),
    estimate_depth(Rest, New, NewMax, Result).

instr_stack_effect(Instr, Effect) :-
    (   sub_string(Instr, _, _, _, "iconst")  -> Effect = 1
    ;   sub_string(Instr, _, _, _, "bipush")  -> Effect = 1
    ;   sub_string(Instr, _, _, _, "sipush")  -> Effect = 1
    ;   sub_string(Instr, _, _, _, "ldc")     -> Effect = 1
    ;   sub_string(Instr, _, _, _, "iload")   -> Effect = 1
    ;   sub_string(Instr, _, _, _, "istore")  -> Effect = -1
    ;   sub_string(Instr, _, _, _, "iadd")    -> Effect = -1
    ;   sub_string(Instr, _, _, _, "isub")    -> Effect = -1
    ;   sub_string(Instr, _, _, _, "imul")    -> Effect = -1
    ;   sub_string(Instr, _, _, _, "idiv")    -> Effect = -1
    ;   sub_string(Instr, _, _, _, "irem")    -> Effect = -1
    ;   sub_string(Instr, _, _, _, "ineg")    -> Effect = 0
    ;   sub_string(Instr, _, _, _, "ireturn") -> Effect = -1
    ;   sub_string(Instr, _, _, _, "if_icmp") -> Effect = -2
    ;   sub_string(Instr, _, _, _, "dup")     -> Effect = 1
    ;   sub_string(Instr, _, _, _, "athrow")  -> Effect = -1
    ;   sub_string(Instr, _, _, _, "new ")    -> Effect = 1
    ;   sub_string(Instr, _, _, _, "invoke")  -> Effect = 0  % approximation
    ;   Effect = 0  % labels, gotos, etc.
    ).
