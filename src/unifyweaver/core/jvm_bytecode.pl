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
    jvm_estimate_stack_depth/2,      % +Instructions, -Depth

    % Recursion pattern generators
    jvm_tail_recursion_bytecode/4,   % +PredStr, +Arity, +VarStyle, -Instructions
    jvm_linear_recursion_bytecode/4, % +PredStr, +Arity, +VarStyle, -Instructions
    jvm_tree_recursion_bytecode/4,   % +PredStr, +Arity, +VarStyle, -Instructions
    jvm_multicall_recursion_bytecode/4,  % +PredStr, +Arity, +VarStyle, -Instructions
    jvm_direct_multicall_bytecode/4,     % +PredStr, +Arity, +VarStyle, -Instructions
    jvm_mutual_recursion_bytecode/4, % +Predicates, +ClassName, +VarStyle, -Instructions
    jvm_entry_method_bytecode/4,     % +PredStr, +Arity, +VarStyle, -Instructions

    % String operations
    jvm_string_equals_bytecode/4,    % +VarA, +VarB, +VarMap, -Instructions
    jvm_string_concat_bytecode/4,    % +VarA, +VarB, +VarMap, -Instructions
    jvm_tostring_bytecode/3,         % +Var, +VarMap, -Instructions
    jvm_println_bytecode/3,          % +Var, +VarMap, -Instructions
    jvm_load_string/2                % +StringValue, -Instruction
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
jvm_native_clause_body(_PredStr/Arity, Clauses, VarStyle, _OutputFmt, Code) :-
    Arity1 is Arity - 1,
    jvm_clauses_to_branches(Clauses, Arity1, VarStyle, Branches),
    Branches \= [],
    jvm_compiled_if_chain(Branches, 0, "CL", AllInstrs),
    jvm_ensure_strings(AllInstrs, StrInstrs),
    atomic_list_concat(StrInstrs, '\n', Code).

%% jvm_compiled_if_chain(+Branches, +N, +Prefix, -Instructions)
%%   Assemble pre-compiled branches into an if/else chain with labels.
%%   Guard instructions already contain "NEXT" as placeholder label.
jvm_compiled_if_chain([], _, _, ErrInstrs) :- !,
    ErrInstrs = [
        "    new java/lang/RuntimeException",
        "    dup",
        "    ldc \"No matching clause\"",
        "    invokespecial java/lang/RuntimeException <init> (Ljava/lang/String;)V",
        "    athrow"
    ].
jvm_compiled_if_chain([branch([], ValueInstrs)], _, _, Instructions) :- !,
    append(ValueInstrs, ["    ireturn"], Instructions).
jvm_compiled_if_chain([branch(GuardInstrs, ValueInstrs)|Rest], N, Prefix, Instructions) :-
    N1 is N + 1,
    format(string(NextLabel), '~w_~w', [Prefix, N1]),
    format(string(EndLabel), '~w_end', [Prefix]),
    % Replace "NEXT" placeholder in guard instructions with actual label
    maplist(replace_next_label(NextLabel), GuardInstrs, FixedGuards),
    % Value + return
    append(ValueInstrs, ["    ireturn"], ValueWithRet),
    format(string(GotoEnd), '    goto ~w', [EndLabel]),
    format(string(NextLabelDecl), '~w:', [NextLabel]),
    jvm_compiled_if_chain(Rest, N1, Prefix, RestInstrs),
    append(FixedGuards, ValueWithRet, GuardAndValue),
    append(GuardAndValue, [GotoEnd, NextLabelDecl], WithLabel),
    append(WithLabel, RestInstrs, BeforeEnd),
    (   N =:= 0
    ->  format(string(EndLabelDecl), '~w:', [EndLabel]),
        append(BeforeEnd, [EndLabelDecl], Instructions)
    ;   Instructions = BeforeEnd
    ).

replace_next_label(Label, Instr, Fixed) :-
    (   sub_string(Instr, _, _, _, "NEXT")
    ->  split_string(Instr, "", "", _),
        string_concat(Pre, "NEXT", Instr),
        string_concat(Pre, Label, Fixed)
    ;   Fixed = Instr
    ).

jvm_clauses_to_branches([], _, _, []).
jvm_clauses_to_branches([Head-Body|Rest], Arity1, VarStyle, Result) :-
    jvm_clauses_to_branches(Rest, Arity1, VarStyle, RestBranches),
    (   Body \== true,
        Head =.. [_|AllArgs],
        length(InputArgs, Arity1),
        append(InputArgs, [OutputArg], AllArgs),
        build_head_varmap(InputArgs, 1, VarMap),
        normalize_goals(Body, Goals),
        once(clause_guard_output_split(Goals, VarMap, Guards, Outputs)),
        jvm_compile_guards(Guards, VarMap, VarStyle, GuardInstrs),
        jvm_output_to_bytecode(OutputArg, Outputs, VarMap, VarStyle, ValueInstrs)
    ->  Result = [branch(GuardInstrs, ValueInstrs)|RestBranches]
    ;   Result = RestBranches
    ).

jvm_compile_guards([], _, _, []).
jvm_compile_guards([G|Gs], VarMap, VarStyle, AllInstrs) :-
    jvm_guard_to_bytecode(G, VarMap, VarStyle, "NEXT", GInstrs),
    jvm_compile_guards(Gs, VarMap, VarStyle, RestInstrs),
    append(GInstrs, RestInstrs, AllInstrs).

jvm_ensure_strings([], []).
jvm_ensure_strings([H|T], [S|Rest]) :-
    (string(H) -> S = H ; atom_string(H, S)),
    jvm_ensure_strings(T, Rest).

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
    ;   sub_string(Instr, _, _, _, "aload")   -> Effect = 1
    ;   sub_string(Instr, _, _, _, "astore")  -> Effect = -1
    ;   sub_string(Instr, _, _, _, "iaload")  -> Effect = -1  % pops ref+idx, pushes val
    ;   sub_string(Instr, _, _, _, "iastore") -> Effect = -3
    ;   sub_string(Instr, _, _, _, "baload")  -> Effect = -1
    ;   sub_string(Instr, _, _, _, "bastore") -> Effect = -3
    ;   sub_string(Instr, _, _, _, "newarray") -> Effect = 0  % pops count, pushes ref
    ;   sub_string(Instr, _, _, _, "goto")    -> Effect = 0
    ;   Effect = 0  % labels, gotos, etc.
    ).

%% ============================================
%% STRING OPERATIONS
%% ============================================

%% jvm_load_string(+StringValue, -Instruction)
%%   Load a string constant onto the stack.
jvm_load_string(Value, Instr) :-
    format(string(Instr), '    ldc "~w"', [Value]).

%% jvm_string_equals_bytecode(+ExprA, +ExprB, +VarMap, -Instructions)
%%   Compare two strings for equality. Result: 1 (true) or 0 (false) on stack.
%%   Uses String.equals(Object) which returns boolean.
jvm_string_equals_bytecode(ExprA, ExprB, VarMap, Instructions) :-
    jvm_resolve_string_operand(ExprA, VarMap, LoadA),
    jvm_resolve_string_operand(ExprB, VarMap, LoadB),
    append(LoadA, LoadB, Operands),
    append(Operands,
        ["    invokevirtual java/lang/String equals (Ljava/lang/Object;)Z"],
        Instructions).

%% jvm_string_concat_bytecode(+ExprA, +ExprB, +VarMap, -Instructions)
%%   Concatenate two strings. Result: new String on stack.
%%   Uses String.concat(String).
jvm_string_concat_bytecode(ExprA, ExprB, VarMap, Instructions) :-
    jvm_resolve_string_operand(ExprA, VarMap, LoadA),
    jvm_resolve_string_operand(ExprB, VarMap, LoadB),
    append(LoadA, LoadB, Operands),
    append(Operands,
        ["    invokevirtual java/lang/String concat (Ljava/lang/String;)Ljava/lang/String;"],
        Instructions).

%% jvm_tostring_bytecode(+Expr, +VarMap, -Instructions)
%%   Convert an int on the stack to a String via String.valueOf(int).
jvm_tostring_bytecode(Expr, VarMap, Instructions) :-
    jvm_expr_to_bytecode(Expr, VarMap, symbolic, LoadExpr),
    append(LoadExpr,
        ["    invokestatic java/lang/String valueOf (I)Ljava/lang/String;"],
        Instructions).

%% jvm_println_bytecode(+Expr, +VarMap, -Instructions)
%%   Print a value to stdout followed by newline.
%%   Works for both int and String (uses Object overload).
jvm_println_bytecode(Expr, VarMap, Instructions) :-
    jvm_resolve_string_operand(Expr, VarMap, LoadExpr),
    Instructions0 = [
        "    getstatic java/lang/System out Ljava/io/PrintStream;"
    ],
    append(Instructions0, LoadExpr, WithValue),
    append(WithValue,
        ["    invokevirtual java/io/PrintStream println (Ljava/lang/Object;)V"],
        Instructions).

%% jvm_resolve_string_operand(+Expr, +VarMap, -Instructions)
%%   Resolve an expression to bytecode that leaves a String/Object on stack.
jvm_resolve_string_operand(Expr, _VarMap, [Instr]) :-
    atom(Expr),
    \+ lookup_var(Expr, _VarMap, _),
    !,
    jvm_load_string(Expr, Instr).
jvm_resolve_string_operand(Expr, VarMap, [Instr]) :-
    (var(Expr) ; atom(Expr)),
    lookup_var(Expr, VarMap, Name),
    !,
    format(string(Instr), '    aload ~w', [Name]).
jvm_resolve_string_operand(Expr, _VarMap, [Instr]) :-
    string(Expr),
    !,
    format(string(Instr), '    ldc "~w"', [Expr]).
jvm_resolve_string_operand(Expr, VarMap, Instructions) :-
    jvm_expr_to_bytecode(Expr, VarMap, symbolic, Instructions).

%% ============================================
%% RECURSION PATTERN GENERATORS
%% ============================================

%% jvm_tail_recursion_bytecode(+PredStr, +Arity, +VarStyle, -Instructions)
%%   Tail recursion using goto loop. O(1) stack space.
%%   Pattern: sum(n, acc) { while(n > 0) { acc += n; n--; } return acc; }
jvm_tail_recursion_bytecode(PredStr, _Arity, VarStyle, Instructions) :-
    var_ref(VarStyle, "arg1", 0, NRef),
    var_ref(VarStyle, "arg2", 1, AccRef),
    format(string(LoadN),   '    iload ~w', [NRef]),
    format(string(LoadAcc), '    iload ~w', [AccRef]),
    format(string(StoreN),  '    istore ~w', [NRef]),
    format(string(StoreAcc),'    istore ~w', [AccRef]),
    format(string(Header),  '    ;; Tail recursion: ~w', [PredStr]),
    Instructions = [
        Header,
        "LOOP:",
        LoadN,
        "    iconst_0",
        "    if_icmple DONE",
        LoadAcc, LoadN,
        "    iadd",
        StoreAcc,
        LoadN,
        "    iconst_1",
        "    isub",
        StoreN,
        "    goto LOOP",
        "DONE:",
        LoadAcc,
        "    ireturn"
    ].

%% jvm_entry_method_bytecode(+PredStr, +Arity, +VarStyle, -Instructions)
%%   Entry-point wrapper that passes initial accumulator = 0.
jvm_entry_method_bytecode(PredStr, _Arity, VarStyle, Instructions) :-
    var_ref(VarStyle, "arg1", 0, NRef),
    format(string(LoadN), '    iload ~w', [NRef]),
    format(string(Invoke), '    invokestatic ~w(II)I', [PredStr]),
    Instructions = [
        LoadN,
        "    iconst_0",
        Invoke,
        "    ireturn"
    ].

%% jvm_linear_recursion_bytecode(+PredStr, +Arity, +VarStyle, -Instructions)
%%   Linear recursion with memoization. f(n) = f(n-1) + n.
jvm_linear_recursion_bytecode(PredStr, _Arity, VarStyle, Instructions) :-
    var_ref(VarStyle, "arg1", 0, NRef),
    format(string(LoadN), '    iload ~w', [NRef]),
    format(string(SelfCall), '    invokestatic ~w(I)I', [PredStr]),
    format(string(Header), '    ;; Linear recursion: ~w (memoized)', [PredStr]),
    Instructions = [
        Header,
        "    ;; Base cases",
        LoadN, "    iconst_0", "    if_icmpgt NOT_BASE0",
        "    iconst_0", "    ireturn",
        "NOT_BASE0:",
        LoadN, "    iconst_1", "    if_icmpne COMPUTE",
        "    iconst_1", "    ireturn",
        "COMPUTE:",
        "    ;; Recursive call: f(n-1) + n",
        LoadN, "    iconst_1", "    isub",
        SelfCall,
        LoadN,
        "    iadd",
        "    ireturn"
    ].

%% jvm_tree_recursion_bytecode(+PredStr, +Arity, +VarStyle, -Instructions)
%%   Tree recursion with two calls. f(n) = f(n-1) + f(n-2).
jvm_tree_recursion_bytecode(PredStr, _Arity, VarStyle, Instructions) :-
    var_ref(VarStyle, "arg1", 0, NRef),
    format(string(LoadN), '    iload ~w', [NRef]),
    format(string(SelfCall), '    invokestatic ~w(I)I', [PredStr]),
    format(string(Header), '    ;; Tree recursion: ~w (fibonacci-like)', [PredStr]),
    Instructions = [
        Header,
        "    ;; Base cases",
        LoadN, "    iconst_0", "    if_icmpgt NOT_BASE0",
        "    iconst_0", "    ireturn",
        "NOT_BASE0:",
        LoadN, "    iconst_1", "    if_icmpne COMPUTE",
        "    iconst_1", "    ireturn",
        "COMPUTE:",
        "    ;; f(n-1) + f(n-2)",
        LoadN, "    iconst_1", "    isub",
        SelfCall,
        LoadN, "    iconst_2", "    isub",
        SelfCall,
        "    iadd",
        "    ireturn"
    ].

%% jvm_multicall_recursion_bytecode/4 — same structure as tree for multicall hook
jvm_multicall_recursion_bytecode(PredStr, Arity, VarStyle, Instructions) :-
    jvm_tree_recursion_bytecode(PredStr, Arity, VarStyle, Instructions).

%% jvm_direct_multicall_bytecode/4 — same structure, clause-body-analysis driven
jvm_direct_multicall_bytecode(PredStr, Arity, VarStyle, Instructions) :-
    jvm_tree_recursion_bytecode(PredStr, Arity, VarStyle, Instructions).

%% jvm_mutual_recursion_bytecode(+Predicates, +ClassName, +VarStyle, -MethodsMap)
%%   Mutual recursion: is_even/is_odd pattern.
%%   Returns list of method(Name, Instructions) for each predicate.
jvm_mutual_recursion_bytecode(Predicates, ClassName, VarStyle, Methods) :-
    (   Predicates = [Pred1, Pred2|_]
    ->  atom_string(Pred1, P1Str),
        atom_string(Pred2, P2Str),
        var_ref(VarStyle, "arg1", 0, NRef),
        format(string(LoadN), '    iload ~w', [NRef]),
        format(string(Call2), '    invokestatic ~w/~w(I)I', [ClassName, P2Str]),
        format(string(Call1), '    invokestatic ~w/~w(I)I', [ClassName, P1Str]),
        M1Instrs = [
            LoadN, "    iconst_0", "    if_icmpne NOT_ZERO_1",
            "    iconst_1", "    ireturn",
            "NOT_ZERO_1:",
            LoadN, "    iconst_1", "    isub",
            Call2,
            "    ireturn"
        ],
        M2Instrs = [
            LoadN, "    iconst_0", "    if_icmpne NOT_ZERO_2",
            "    iconst_0", "    ireturn",
            "NOT_ZERO_2:",
            LoadN, "    iconst_1", "    isub",
            Call1,
            "    ireturn"
        ],
        Methods = [method(P1Str, M1Instrs), method(P2Str, M2Instrs)]
    ;   Methods = []
    ).

%% var_ref(+VarStyle, +SymName, +SlotIdx, -Ref)
%%   Returns the appropriate variable reference based on style.
var_ref(symbolic, SymName, _Slot, SymName) :- !.
var_ref(numeric, _SymName, Slot, Ref) :-
    number_string(Slot, Ref).
