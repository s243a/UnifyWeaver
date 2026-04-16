:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_fsharp_lowered_emitter.pl — WAM-lowered F# emission (Phase 4+)
%
% Emits one F# function per predicate using computation expression (CE)
% style — equivalent to Haskell's Maybe-monad do-notation. Simple register
% operations are inlined as `let` bindings; complex instructions (Call,
% BuiltinCall, GetConstant, PutStructure sequences) delegate to `step` or
% `dispatchCall` from WamRuntime.
%
% Design mirrors wam_haskell_lowered_emitter.pl closely. F# differences:
%   - `option { ... }` CE (or explicit `Option.bind` chain) replaces do-notation
%   - `{ s with Field = v }` record update replaces `s { field = v }`
%   - `Map.tryFind` / `Map.add` replace `IM.lookup` / `IM.insert`
%   - `Option.defaultValue` replaces `fromMaybe`
%   - state variable chaining uses `let! sv = step ctx s instr` in CE blocks
%
% For multi-clause predicates, only clause 1 is lowered. Clause 2+ stays
% in the interpreter's instruction array for backtrack fallback (identical
% approach to the Haskell target).
%
% See: src/unifyweaver/targets/wam_haskell_lowered_emitter.pl (primary reference)

:- module(wam_fsharp_lowered_emitter, [
    wam_fsharp_lowerable/3,
    lower_predicate_to_fsharp/4
]).

:- use_module(library(lists)).

% ============================================================================
% Parsing — identical to Haskell emitter (WAM text format is target-agnostic)
% ============================================================================

parse_wam_text_fs(WamText, PCInstrs, LabelMap) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_lines_fs(Lines, 1, PCInstrs, LabelMap).

parse_lines_fs([], _, [], []).
parse_lines_fs([Line|Rest], PC, Instrs, Labels) :-
    split_string(Line, "", " \t", [Trimmed]),
    (   Trimmed == ""
    ->  parse_lines_fs(Rest, PC, Instrs, Labels)
    ;   sub_string(Trimmed, _, 1, 0, ":")
    ->  sub_string(Trimmed, 0, _, 1, LStr),
        atom_string(LAtom, LStr),
        Labels = [LAtom-PC|LabelsRest],
        parse_lines_fs(Rest, PC, Instrs, LabelsRest)
    ;   tokenize_fs(Trimmed, Instr),
        PC1 is PC + 1,
        Instrs = [pc(PC, Instr)|InstrsRest],
        parse_lines_fs(Rest, PC1, InstrsRest, Labels)
    ).

tokenize_fs(Line, Term) :-
    split_string(Line, " \t", " \t,", Tokens),
    exclude(=(""), Tokens, [MStr|Args]),
    atom_string(M, MStr),
    Term =.. [M|Args].

% ============================================================================
% Lowerability — same whitelist as Haskell emitter
% ============================================================================

%% wam_fsharp_lowerable(+PI, +WamCode, -Reason)
%  Succeeds if the predicate is safe to lower to a standalone F# function.
%  Multi-clause predicates are now lowerable:
%    - Clause 1 is lowered into an F# function
%    - Clause 2+ stays in the interpreter, reached via backtrack
%    - CallForeign resolves ambiguity for foreign calls at compile time
%    - Detected kernels are excluded upstream by wam_fsharp_partition_predicates
wam_fsharp_lowerable(_PI, WamCode, lowerable) :-
    parse_wam_text_fs(WamCode, PCInstrs, _),
    clause1_instrs_fs(PCInstrs, C1),
    forall(member(I, C1), supported_fs(I)).

clause1_instrs_fs([], []).
% Skip switch_on_constant prefix (multi-clause indexing)
clause1_instrs_fs([pc(_, Instr)|Rest], C1) :-
    functor(Instr, switch_on_constant, _), !,
    clause1_instrs_fs(Rest, C1).
clause1_instrs_fs([pc(_, try_me_else(_))|Rest], C1) :- !,
    take_to_proceed_fs(Rest, C1).
clause1_instrs_fs(PCInstrs, Instrs) :-
    maplist([pc(_, I), I]>>true, PCInstrs, Instrs).

take_to_proceed_fs([], []).
take_to_proceed_fs([pc(_, proceed)|_], [proceed]) :- !.
take_to_proceed_fs([pc(_, I)|Rest], [I|More]) :- take_to_proceed_fs(Rest, More).

supported_fs(try_me_else(_)).
supported_fs(allocate).
supported_fs(deallocate).
supported_fs(get_constant(_, _)).
supported_fs(get_variable(_, _)).
supported_fs(get_value(_, _)).
supported_fs(put_constant(_, _)).
supported_fs(put_variable(_, _)).
supported_fs(put_value(_, _)).
supported_fs(put_structure(_, _)).
supported_fs(put_list(_)).
supported_fs(set_value(_)).
supported_fs(set_constant(_)).
supported_fs(call(_, _)).
supported_fs(builtin_call(_, _)).
% begin_aggregate/end_aggregate not lowerable — need the run loop for
% backtrack-driven collection (same constraint as Haskell target).
supported_fs(cut_ite).
supported_fs(jump(_)).
supported_fs(trust_me).
supported_fs(proceed).
supported_fs(execute(_)).

% ============================================================================
% Emission
% ============================================================================

%% lower_predicate_to_fsharp(+PI, +WamCode, +Opts, -lowered(PredName, FuncName, Code))
lower_predicate_to_fsharp(PI, WamCode, Opts, lowered(PredName, FuncName, Code)) :-
    (   PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    % Sanitize predicate name for F# identifier (replace $ with _)
    atom_string(Pred, PredStr),
    split_string(PredStr, "$", "", Parts),
    atomic_list_concat(Parts, '_', SanitizedPred),
    format(atom(FuncName), 'lowered_~w_~w', [SanitizedPred, Arity]),
    % base_pc(N) offsets local PCs to match the global merged instruction array
    (   member(base_pc(BasePC), Opts) -> true ; BasePC = 1 ),
    % foreign_preds(List) — predicate keys dispatched via callForeign
    (   member(foreign_preds(ForeignPreds), Opts) -> true ; ForeignPreds = [] ),
    parse_wam_text_fs(WamCode, PCInstrs, LabelMap),
    Offset is BasePC - 1,
    offset_pcs_fs(PCInstrs, Offset, GlobalPCInstrs),
    offset_labels_fs(LabelMap, Offset, GlobalLabelMap),
    with_output_to(string(Code),
        emit_func_fs(FuncName, GlobalPCInstrs, GlobalLabelMap, ForeignPreds)).

offset_pcs_fs([], _, []).
offset_pcs_fs([pc(PC, I)|Rest], Off, [pc(GPC, I)|Rest2]) :-
    GPC is PC + Off,
    offset_pcs_fs(Rest, Off, Rest2).

offset_labels_fs([], _, []).
offset_labels_fs([L-PC|Rest], Off, [L-GPC|Rest2]) :-
    GPC is PC + Off,
    offset_labels_fs(Rest, Off, Rest2).

% ============================================================================
% Function emission
% ============================================================================

emit_func_fs(FN, PCInstrs, LabelMap, ForeignPreds) :-
    format("/// Lowered: ~w~n", [FN]),
    format("let ~w (ctx: WamContext) (s_init: WamState) : WamState option =~n", [FN]),
    % Skip switch_on_constant prefix if present
    (   PCInstrs = [pc(_, SOC)|PCInstrs1],
        functor(SOC, switch_on_constant, _)
    ->  true
    ;   PCInstrs1 = PCInstrs
    ),
    (   PCInstrs1 = [pc(_, try_me_else(LStr))|BodyPCs]
    ->  % Multi-clause: push CP for clause-2+ backtrack, try clause 1
        atom_string(LAtom, LStr),
        (   member(LAtom-AltPC, LabelMap) -> true ; AltPC = 0 ),
        format("    let s_cp =~n"),
        format("        { s_init with~n"),
        format("            WsCPs    = { CpNextPC   = ~w~n", [AltPC]),
        format("                         CpRegs     = s_init.WsRegs~n"),
        format("                         CpStack    = s_init.WsStack~n"),
        format("                         CpCP       = s_init.WsCP~n"),
        format("                         CpTrailLen = s_init.WsTrailLen~n"),
        format("                         CpHeapLen  = s_init.WsHeapLen~n"),
        format("                         CpBindings = s_init.WsBindings~n"),
        format("                         CpCutBar   = s_init.WsCutBar~n"),
        format("                         CpAggFrame = None~n"),
        format("                         CpBuiltin  = None } :: s_init.WsCPs~n"),
        format("            WsCPsLen = s_init.WsCPsLen + 1 }~n"),
        format("    let clause1 (s_c1: WamState) : WamState option =~n"),
        take_to_proceed_pc_fs(BodyPCs, Clause1PCs),
        emit_instrs_fs(Clause1PCs, "s_c1", "        ", ForeignPreds),
        format("    match clause1 s_cp with~n"),
        format("    | Some result -> Some result~n"),
        format("    | None ->~n"),
        format("        // Clause 1 failed — backtrack to clause 2+ in the interpreter~n"),
        format("        backtrack s_cp |> Option.bind (fun s_bt -> run ctx { s_bt with WsPC = s_bt.WsPC + 1 })~n")
    ;   % Single-clause: straightforward binding chain
        emit_instrs_fs(PCInstrs, "s_init", "    ", ForeignPreds)
    ).

take_to_proceed_pc_fs([], []).
take_to_proceed_pc_fs([pc(PC, proceed)|_], [pc(PC, proceed)]) :- !.
take_to_proceed_pc_fs([H|T], [H|R]) :- take_to_proceed_pc_fs(T, R).

% ============================================================================
% Instruction emission — F# binding chain style
%
% Unlike Haskell's do-notation, we use explicit Option.bind chains:
%   let sv1 = step ctx { s with WsPC = pc } instr
%   match sv1 with Some sv2 -> ... | None -> None
%
% For clarity and readability we use a `maybe` computation expression
% defined in WamRuntime (see WamRuntime preamble). Equivalent to:
%   maybe { let! sv = step ctx ... instr; ... return sv }
% ============================================================================

%% emit_instrs_fs(+PCInstrs, +CurrentStateVar, +Indent, +ForeignPreds)
emit_instrs_fs([], SV, I, _FP) :-
    % Empty tail — return the current state
    format("~wSome ~w~n", [I, SV]).
emit_instrs_fs([pc(_PC, try_me_else(ElseLabelStr))|Rest], SV, Ind, FP) :-
    atom_string(ElseLabel, ElseLabelStr),
    split_ite_blocks_fs(Rest, ElseLabel, CondInstrs, ThenInstrs, ElseInstrs, _ContInstrs),
    !,
    % Emit if-then-else as F# nested match
    format("~wmatch (~n", [Ind]),
    atom_concat(Ind, "    ", CondInd),
    emit_ite_block_fs(CondInstrs, SV, CondInd, FP),
    format("~w) with~n", [Ind]),
    fresh_sv_fs(SV, SVthen),
    format("~w| Some ~w ->~n", [Ind, SVthen]),
    atom_concat(Ind, "    ", ThenInd),
    emit_ite_block_fs(ThenInstrs, SVthen, ThenInd, FP),
    format("~w| None ->~n", [Ind]),
    atom_concat(Ind, "    ", ElseInd),
    emit_ite_block_fs(ElseInstrs, SV, ElseInd, FP).
emit_instrs_fs([pc(PC, Instr)|Rest], SV, Ind, FP) :-
    emit_one_fs(Instr, PC, SV, SVout, Ind, FP),
    (   Rest = []
    ->  % last instruction already emitted terminal/return
        true
    ;   emit_instrs_fs(Rest, SVout, Ind, FP)
    ).

emit_ite_block_fs([], SV, Ind, _FP) :-
    format("~wSome ~w~n", [Ind, SV]).
emit_ite_block_fs([pc(PC, Instr)], SV, Ind, FP) :-
    emit_one_fs(Instr, PC, SV, SVout, Ind, FP),
    (   is_terminal_instr_fs(Instr) -> true
    ;   format("~wSome ~w~n", [Ind, SVout])
    ).
emit_ite_block_fs([pc(PC, Instr)|Rest], SV, Ind, FP) :-
    Rest \= [],
    emit_one_fs(Instr, PC, SV, SVout, Ind, FP),
    emit_ite_block_fs(Rest, SVout, Ind, FP).

is_terminal_instr_fs(proceed).

split_ite_blocks_fs(Instrs, _ElseLabel, CondInstrs, ThenInstrs, ElseInstrs, ContInstrs) :-
    split_at_instr_fs(Instrs, cut_ite, CondInstrs, AfterCut),
    split_at_jump_fs(AfterCut, ThenInstrs, ContLabelStr, AfterJump),
    AfterJump = [pc(_, trust_me)|ElseAndCont],
    atom_string(ContLabel, ContLabelStr),
    split_else_cont_fs(ElseAndCont, ContLabel, ElseInstrs, ContInstrs).

split_at_instr_fs([], _, _, _) :- !, fail.
split_at_instr_fs([pc(_, Instr)|Rest], Instr, [], Rest) :- !.
split_at_instr_fs([H|T], Instr, [H|Before], After) :-
    split_at_instr_fs(T, Instr, Before, After).

split_at_jump_fs([], [], "", []) :- !, fail.
split_at_jump_fs([pc(_, jump(Label))|Rest], [], Label, Rest) :- !.
split_at_jump_fs([H|T], [H|Then], Label, Rest) :-
    split_at_jump_fs(T, Then, Label, Rest).

split_else_cont_fs(Instrs, _ContLabel, Instrs, []).

% ============================================================================
% emit_one_fs — single instruction → F# binding line
% ============================================================================

% Terminal: proceed
emit_one_fs(proceed, _, SV, SV, I, _FP) :-
    format("~wlet ret_ = ~w.WsCP~n", [I, SV]),
    format("~wif ret_ = 0 then Some { ~w with WsPC = 0 } else Some { ~w with WsPC = ret_; WsCP = 0 }~n",
           [I, SV, SV]).

% Allocate — always succeeds, inline
emit_one_fs(allocate, _, SV, SVout, I, _FP) :-
    fresh_sv_fs(SV, SVout),
    format("~wlet ~w = { ~w with~n", [I, SVout, SV]),
    format("~w               WsStack = { EfSavedCP = ~w.WsCP; EfYRegs = Map.empty } :: ~w.WsStack~n", [I, SV, SV]),
    format("~w               WsCutBar = ~w.WsCPsLen }~n", [I, SV]).

% Deallocate — can fail on empty stack, delegate to step
emit_one_fs(deallocate, PC, SV, SVout, I, _FP) :-
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } Deallocate with~n", [I, SV, PC]),
    format("~w| Some ~w ->~n", [I, SVout]).

% GetVariable Xn Ai — always succeeds, inline register copy
emit_one_fs(get_variable(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wlet ~w = { ~w with WsRegs = Map.add ~w (derefVar ~w.WsBindings (Map.tryFind ~w ~w.WsRegs |> Option.defaultValue (Atom \"\"))) ~w.WsRegs }~n",
           [I, SVout, SV, Xn, SV, Ai, SV, SV]).

% GetConstant C Ai — can fail, delegate to step
emit_one_fs(get_constant(CStr, AiStr), PC, SV, SVout, I, _FP) :-
    val_fs(CStr, FC), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (GetConstant (~w, ~w)) with~n", [I, SV, PC, FC, Ai]),
    format("~w| Some ~w ->~n", [I, SVout]).

% GetValue Xn Ai — can fail (unification), delegate to step
emit_one_fs(get_value(XnStr, AiStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (GetValue (~w, ~w)) with~n", [I, SV, PC, Xn, Ai]),
    format("~w| Some ~w ->~n", [I, SVout]).

% PutValue Xn Ai — always succeeds, inline
emit_one_fs(put_value(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wlet ~w = { ~w with WsRegs = Map.add ~w (getReg ~w ~w |> Option.defaultValue (Atom \"\")) ~w.WsRegs }~n",
           [I, SVout, SV, Ai, Xn, SV, SV]).

% PutVariable Xn Ai — always succeeds, inline (creates fresh Unbound)
emit_one_fs(put_variable(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wlet vid_~w = ~w.WsVarCounter~n", [I, SVout, SV]),
    format("~wlet var_~w = Unbound vid_~w~n", [I, SVout, SVout]),
    format("~wlet ~w = putReg ~w var_~w { ~w with WsRegs = Map.add ~w var_~w ~w.WsRegs; WsVarCounter = ~w.WsVarCounter + 1 }~n",
           [I, SVout, Xn, SVout, SV, Ai, SVout, SV, SV]).

% PutConstant C Ai — always succeeds, inline
emit_one_fs(put_constant(CStr, AiStr), _, SV, SVout, I, _FP) :-
    val_fs(CStr, FC), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wlet ~w = { ~w with WsRegs = Map.add ~w (~w) ~w.WsRegs }~n",
           [I, SVout, SV, Ai, FC, SV]).

% PutStructure, PutList, SetValue, SetConstant — delegate to step
emit_one_fs(put_structure(FnStr, AiStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(AiStr, Ai),
    parse_functor_fs(FnStr, FuncName, Arity),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (PutStructure (\"~w\", ~w, ~w)) with~n",
           [I, SV, PC, FuncName, Ai, Arity]),
    format("~w| Some ~w ->~n", [I, SVout]).

emit_one_fs(put_list(AiStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (PutList ~w) with~n", [I, SV, PC, Ai]),
    format("~w| Some ~w ->~n", [I, SVout]).

emit_one_fs(set_value(XnStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (SetValue ~w) with~n", [I, SV, PC, Xn]),
    format("~w| Some ~w ->~n", [I, SVout]).

emit_one_fs(set_constant(CStr), PC, SV, SVout, I, _FP) :-
    val_fs(CStr, FC),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (SetConstant (~w)) with~n", [I, SV, PC, FC]),
    format("~w| Some ~w ->~n", [I, SVout]).

% Call — use callForeign for known foreign preds, dispatchCall otherwise
emit_one_fs(call(PredStr, _NStr), PC, SV, SVout, I, FP) :-
    RetPC is PC + 1,
    fresh_sv_fs(SV, SVout),
    atom_string(PredAtom, PredStr),
    (   member(PredAtom, FP)
    ->  format("~wmatch callForeign ctx \"~w\" { ~w with WsCP = ~w } with~n",
               [I, PredStr, SV, RetPC])
    ;   format("~wmatch dispatchCall ctx \"~w\" { ~w with WsCP = ~w } with~n",
               [I, PredStr, SV, RetPC])
    ),
    format("~w| Some ~w ->~n", [I, SVout]).

% BuiltinCall — delegate to step
emit_one_fs(builtin_call(OpStr, NStr), PC, SV, SVout, I, _FP) :-
    escape_dq_fs(OpStr, EscOp),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (BuiltinCall (\"~w\", ~w)) with~n",
           [I, SV, PC, EscOp, NStr]),
    format("~w| Some ~w ->~n", [I, SVout]).

% CutIte — delegate to step
emit_one_fs(cut_ite, PC, SV, SVout, I, _FP) :-
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } CutIte with~n", [I, SV, PC]),
    format("~w| Some ~w ->~n", [I, SVout]).

% Jump — delegate to step
emit_one_fs(jump(LabelStr), PC, SV, SVout, I, _FP) :-
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (Jump \"~w\") with~n", [I, SV, PC, LabelStr]),
    format("~w| Some ~w ->~n", [I, SVout]).

% TrustMe — delegate to step
emit_one_fs(trust_me, PC, SV, SVout, I, _FP) :-
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } TrustMe with~n", [I, SV, PC]),
    format("~w| Some ~w ->~n", [I, SVout]).

% Execute — tail call; returns directly (no WsCP change)
emit_one_fs(execute(PredStr), _PC, SV, SV, I, FP) :-
    atom_string(PredAtom, PredStr),
    (   member(PredAtom, FP)
    ->  format("~wcallForeign ctx \"~w\" ~w~n", [I, PredStr, SV])
    ;   format("~wdispatchCall ctx \"~w\" ~w~n", [I, PredStr, SV])
    ).

% BeginAggregate — delegate to step
emit_one_fs(begin_aggregate(TypeStr, ValRegStr, ResRegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(ValRegStr, ValReg),
    reg_to_int_fs(ResRegStr, ResReg),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (BeginAggregate (\"~w\", ~w, ~w)) with~n",
           [I, SV, PC, TypeStr, ValReg, ResReg]),
    format("~w| Some ~w ->~n", [I, SVout]).

% EndAggregate — delegate to step
emit_one_fs(end_aggregate(ValRegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(ValRegStr, ValReg),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (EndAggregate ~w) with~n",
           [I, SV, PC, ValReg]),
    format("~w| Some ~w ->~n", [I, SVout]).

% ============================================================================
% Helpers
% ============================================================================

%% fresh_sv_fs(+Current, -Next)
%  Generate the next state variable name: s_init → s_0 → s_1 → s_2 ...
fresh_sv_fs(Cur, Next) :-
    atom_string(Cur, CStr),
    (   sub_string(CStr, 0, 2, _, "s_")
    ->  sub_string(CStr, 2, _, 0, NumPart),
        (   number_string(N, NumPart) -> N1 is N + 1 ; N1 = 0 ),
        format(atom(Next), 's_~w', [N1])
    ;   Next = 's_0'
    ).

%% val_fs(+Str, -FSharpExpr)
%  Convert a WAM value token to its F# Value constructor.
val_fs(Str, FS) :-
    (   number_string(N, Str), integer(N)
    ->  format(atom(FS), 'Integer ~w', [N])
    ;   number_string(F, Str), float(F)
    ->  format(atom(FS), 'Float ~w', [F])
    ;   format(atom(FS), 'Atom "~w"', [Str])
    ).

%% reg_to_int_fs(+RegStr, -Int)
%  Map register names to integer ids:
%    A1 → 1, X1 → 101, Y1 → 201
reg_to_int_fs(Reg, Int) :-
    atom_string(RegA, Reg),
    sub_atom(RegA, 0, 1, _, B),
    sub_atom(RegA, 1, _, 0, NA),
    atom_number(NA, Num),
    (   B == 'A' -> Int = Num
    ;   B == 'X' -> Int is Num + 100
    ;   B == 'Y' -> Int is Num + 200
    ;   Int = 0
    ).

%% parse_functor_fs(+FnStr, -Name, -Arity)
parse_functor_fs(FnStr, Name, Arity) :-
    atom_string(FA, FnStr),
    (   sub_atom(FA, B, 1, _, '/')
    ->  sub_atom(FA, 0, B, _, Name),
        B1 is B + 1,
        sub_atom(FA, B1, _, 0, AS),
        atom_number(AS, Arity)
    ;   Name = FA, Arity = 0
    ).

%% escape_dq_fs(+Str, -Escaped)
%  Escape backslashes in builtin call names for F# string literals.
escape_dq_fs(Str, Esc) :-
    split_string(Str, "\\", "", Parts),
    atomic_list_concat(Parts, "\\\\", E0),
    atom_string(E0, Esc).
