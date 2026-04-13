:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_haskell_lowered_emitter.pl — WAM-lowered Haskell emission (Phase 4+)
%
% Emits one Haskell function per predicate using Maybe-monad do-notation.
% Simple register operations are inlined as `let` bindings; complex
% instructions (Call, BuiltinCall, GetConstant, PutStructure sequences)
% delegate to `step` or `dispatchCall` from WamRuntime. This gives us:
%   - No per-instruction array-fetch for the hot path
%   - GHC can specialize step calls for known instruction constructors
%   - Correctness for free via step's existing semantics
%
% For multi-clause predicates, only clause 1 is lowered. Clause 2+ stays
% in the interpreter's instruction array for backtrack fallback.

:- module(wam_haskell_lowered_emitter, [
    wam_haskell_lowerable/3,
    lower_predicate_to_haskell/4
]).

:- use_module(library(lists)).

%% =====================================================================
%% Parsing
%% =====================================================================

parse_wam_text(WamText, PCInstrs, LabelMap) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_lines(Lines, 1, PCInstrs, LabelMap).

parse_lines([], _, [], []).
parse_lines([Line|Rest], PC, Instrs, Labels) :-
    split_string(Line, "", " \t", [Trimmed]),
    (   Trimmed == ""
    ->  parse_lines(Rest, PC, Instrs, Labels)
    ;   sub_string(Trimmed, _, 1, 0, ":")
    ->  sub_string(Trimmed, 0, _, 1, LStr),
        atom_string(LAtom, LStr),
        Labels = [LAtom-PC|LabelsRest],
        parse_lines(Rest, PC, Instrs, LabelsRest)
    ;   tokenize(Trimmed, Instr),
        PC1 is PC + 1,
        Instrs = [pc(PC, Instr)|InstrsRest],
        parse_lines(Rest, PC1, InstrsRest, Labels)
    ).

tokenize(Line, Term) :-
    split_string(Line, " \t", " \t,", Tokens),
    exclude(=(""), Tokens, [MStr|Args]),
    atom_string(M, MStr),
    Term =.. [M|Args].

%% =====================================================================
%% Lowerability
%% =====================================================================

wam_haskell_lowerable(_PI, WamCode, _Reason) :-
    parse_wam_text(WamCode, PCInstrs, _),
    % Multi-clause predicates (try_me_else) are now lowerable:
    %   - Clause 1 is lowered into a Haskell function
    %   - Clause 2+ stays in the interpreter, reached via backtrack
    %   - CallForeign resolves the no-handler/no-solutions ambiguity
    %     at compile time, so foreign calls from lowered functions are safe.
    %   - Detected kernels are excluded from lowering by the partition
    %     logic (they use FFI via CallForeign, lowering would be dead code).
    clause1_instrs(PCInstrs, C1),
    forall(member(I, C1), supported(I)).

clause1_instrs([], []).
% Skip switch_on_constant at the head (multi-clause indexing prefix).
% The parsed term may have variable arity depending on the dispatch table.
clause1_instrs([pc(_, Instr)|Rest], C1) :-
    functor(Instr, switch_on_constant, _), !,
    clause1_instrs(Rest, C1).
clause1_instrs([pc(_, try_me_else(_))|Rest], C1) :- !,
    take_to_proceed(Rest, C1).
clause1_instrs(PCInstrs, Instrs) :-
    maplist([pc(_, I), I]>>true, PCInstrs, Instrs).

take_to_proceed([], []).
take_to_proceed([pc(_, proceed)|_], [proceed]) :- !.
take_to_proceed([pc(_, I)|Rest], [I|More]) :- take_to_proceed(Rest, More).

supported(try_me_else(_)).
supported(allocate).
supported(deallocate).
supported(get_constant(_, _)).
supported(get_variable(_, _)).
supported(get_value(_, _)).
supported(put_constant(_, _)).
supported(put_variable(_, _)).
supported(put_value(_, _)).
supported(put_structure(_, _)).
supported(put_list(_)).
supported(set_value(_)).
supported(set_constant(_)).
supported(call(_, _)).
supported(builtin_call(_, _)).
supported(proceed).

%% =====================================================================
%% Emission
%% =====================================================================

lower_predicate_to_haskell(PI, WamCode, Opts, lowered(PredName, FuncName, Code)) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    format(atom(FuncName), 'lowered_~w_~w', [Pred, Arity]),
    % base_pc(N) offsets local PCs to match the global merged instruction array.
    ( member(base_pc(BasePC), Opts) -> true ; BasePC = 1 ),
    % foreign_preds(List) — predicate keys with CallForeign dispatch.
    ( member(foreign_preds(ForeignPreds), Opts) -> true ; ForeignPreds = [] ),
    parse_wam_text(WamCode, PCInstrs, LabelMap),
    % Offset all PCs: local PC 1 maps to global BasePC.
    Offset is BasePC - 1,
    offset_pcs(PCInstrs, Offset, GlobalPCInstrs),
    offset_labels(LabelMap, Offset, GlobalLabelMap),
    with_output_to(string(Code),
        emit_func(FuncName, GlobalPCInstrs, GlobalLabelMap, ForeignPreds)).

offset_pcs([], _, []).
offset_pcs([pc(PC, I)|Rest], Off, [pc(GPC, I)|Rest2]) :-
    GPC is PC + Off,
    offset_pcs(Rest, Off, Rest2).

offset_labels([], _, []).
offset_labels([L-PC|Rest], Off, [L-GPC|Rest2]) :-
    GPC is PC + Off,
    offset_labels(Rest, Off, Rest2).

emit_func(FN, PCInstrs, LabelMap, ForeignPreds) :-
    format("-- | Lowered: ~w~n", [FN]),
    format("~w :: WamContext -> WamState -> Maybe WamState~n", [FN]),
    % Multi-clause: push CP, try clause 1, if it fails backtrack into
    % the interpreter for clause 2+. This ensures the CP is visible
    % to the backtrack machinery even when clause 1 returns Nothing.
    % Skip switch_on_constant prefix if present (variable arity)
    (   PCInstrs = [pc(_, SOC)|PCInstrs1],
        functor(SOC, switch_on_constant, _)
    ->  true
    ;   PCInstrs1 = PCInstrs
    ),
    (   PCInstrs1 = [pc(_, try_me_else(LStr))|BodyPCs]
    ->  atom_string(LAtom, LStr),
        (   member(LAtom-AltPC, LabelMap) -> true ; AltPC = 0 ),
        format("~w !ctx s_init =~n", [FN]),
        format("  let s_cp = s_init { wsCPs = ChoicePoint~n"),
        format("        { cpNextPC = ~w, cpRegs = wsRegs s_init, cpStack = wsStack s_init~n", [AltPC]),
        format("        , cpCP = wsCP s_init, cpTrailLen = wsTrailLen s_init~n"),
        format("        , cpHeapLen = wsHeapLen s_init, cpBindings = wsBindings s_init~n"),
        format("        , cpCutBar = wsCutBar s_init, cpAggFrame = Nothing, cpBuiltin = Nothing~n"),
        format("        } : wsCPs s_init~n"),
        format("        , wsCPsLen = wsCPsLen s_init + 1 }~n"),
        format("  in case clause1 s_cp of~n"),
        format("       Just result -> Just result~n"),
        format("       Nothing -> backtrack s_cp >>= \\s_bt -> run ctx (s_bt { wsPC = wsPC s_bt + 1 })~n"),
        format("       -- ^ skip TrustMe at cpNextPC: backtrack already popped the CP~n"),
        format("  where~n"),
        format("    clause1 s_c1 = do~n"),
        take_to_proceed_pc(BodyPCs, Clause1PCs),
        emit_instrs(Clause1PCs, "s_c1", "      ", ForeignPreds)
    ;   % Single-clause: simple do-notation
        format("~w !ctx s_init = do~n", [FN]),
        emit_instrs(PCInstrs, "s_init", "  ", ForeignPreds)
    ).

take_to_proceed_pc([], []).
take_to_proceed_pc([pc(PC, proceed)|_], [pc(PC, proceed)]) :- !.
take_to_proceed_pc([H|T], [H|R]) :- take_to_proceed_pc(T, R).

%% emit_instrs(+PCInstrs, +CurrentStateVar, +IndentPrefix, +ForeignPreds)
%  Emit one line of do-notation per instruction, threading state vars.
emit_instrs([], _, _, _).
emit_instrs([pc(PC, Instr)|Rest], SV, Ind, FP) :-
    emit_one(Instr, PC, SV, SVout, Ind, FP),
    emit_instrs(Rest, SVout, Ind, FP).

%% emit_one(+Instr, +PC, +StateVarIn, -StateVarOut, +IndentPrefix, +ForeignPreds)
%  Emit do-notation for a single instruction.

% Terminal: proceed
emit_one(proceed, _, SV, SV, I, _FP) :-
    format("~wlet ret_ = wsCP ~w~n", [I, SV]),
    format("~wif ret_ == 0 then Just (~w { wsPC = 0 }) else Just (~w { wsPC = ret_, wsCP = 0 })~n",
           [I, SV, SV]).

% Allocate — always succeeds, use let
emit_one(allocate, _, SV, SVout, I, _FP) :-
    fresh_sv(SV, SVout),
    format("~wlet ~w = ~w { wsStack = EnvFrame (wsCP ~w) IM.empty : wsStack ~w, wsCutBar = wsCPsLen ~w }~n",
           [I, SVout, SV, SV, SV, SV]).

% Deallocate — use step
emit_one(deallocate, PC, SV, SVout, I, _FP) :-
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) Deallocate~n", [I, SVout, SV, PC]).

% GetVariable Xn Ai — always succeeds, inline register copy
emit_one(get_variable(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~wlet ~w = ~w { wsRegs = IM.insert ~w (derefVar (wsBindings ~w) (fromMaybe (Atom \"\") (IM.lookup ~w (wsRegs ~w)))) (wsRegs ~w) }~n",
           [I, SVout, SV, Xn, SV, Ai, SV, SV]).

% GetConstant C Ai — can fail, use step
emit_one(get_constant(CStr, AiStr), PC, SV, SVout, I, _FP) :-
    val_hs(CStr, HC), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (GetConstant (~w) ~w)~n",
           [I, SVout, SV, PC, HC, Ai]).

% GetValue — can fail (unification), use step
emit_one(get_value(XnStr, AiStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (GetValue ~w ~w)~n",
           [I, SVout, SV, PC, Xn, Ai]).

% PutValue Xn Ai — always succeeds, inline
emit_one(put_value(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~wlet ~w = ~w { wsRegs = IM.insert ~w (fromMaybe (Atom \"\") (getReg ~w ~w)) (wsRegs ~w) }~n",
           [I, SVout, SV, Ai, Xn, SV, SV]).

% PutVariable Xn Ai — always succeeds, inline (creates fresh Unbound)
emit_one(put_variable(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~wlet v_~w = Unbound (wsVarCounter ~w)~n", [I, SVout, SV]),
    format("~w    ~w = (putReg ~w v_~w ~w) { wsRegs = IM.insert ~w v_~w (wsRegs (putReg ~w v_~w ~w)), wsVarCounter = wsVarCounter ~w + 1 }~n",
           [I, SVout, Xn, SVout, SV, Ai, SVout, Xn, SVout, SV, SV]).

% PutConstant C Ai — always succeeds, inline
emit_one(put_constant(CStr, AiStr), _, SV, SVout, I, _FP) :-
    val_hs(CStr, HC), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~wlet ~w = ~w { wsRegs = IM.insert ~w (~w) (wsRegs ~w) }~n",
           [I, SVout, SV, Ai, HC, SV]).

% PutStructure, PutList, SetValue, SetConstant — delegate to step
emit_one(put_structure(FnStr, AiStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(AiStr, Ai),
    parse_functor(FnStr, FuncName, Arity),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (PutStructure \"~w\" ~w ~w)~n",
           [I, SVout, SV, PC, FuncName, Ai, Arity]).

emit_one(put_list(AiStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (PutList ~w)~n",
           [I, SVout, SV, PC, Ai]).

emit_one(set_value(XnStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (SetValue ~w)~n",
           [I, SVout, SV, PC, Xn]).

emit_one(set_constant(CStr), PC, SV, SVout, I, _FP) :-
    val_hs(CStr, HC),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (SetConstant (~w))~n",
           [I, SVout, SV, PC, HC]).

% Call — use callForeign for known foreign preds, dispatchCall otherwise
emit_one(call(PredStr, _NStr), PC, SV, SVout, I, FP) :-
    RetPC is PC + 1,
    fresh_sv(SV, SVout),
    atom_string(PredAtom, PredStr),
    (   member(PredAtom, FP)
    ->  format("~w~w <- callForeign ctx \"~w\" (~w { wsCP = ~w })~n",
               [I, SVout, PredStr, SV, RetPC])
    ;   format("~w~w <- dispatchCall ctx \"~w\" (~w { wsCP = ~w })~n",
               [I, SVout, PredStr, SV, RetPC])
    ).

% BuiltinCall — delegate to step
emit_one(builtin_call(OpStr, NStr), PC, SV, SVout, I, _FP) :-
    escape_bs(OpStr, EOp),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (BuiltinCall \"~w\" ~w)~n",
           [I, SVout, SV, PC, EOp, NStr]).

%% =====================================================================
%% Helpers
%% =====================================================================

%% fresh_sv(+Current, -Next) — generate next state variable name
fresh_sv(Cur, Next) :-
    atom_string(Cur, CStr),
    (   sub_string(CStr, 0, _, _, "s_")
    ->  sub_string(CStr, 2, _, 0, NumPart),
        (   number_string(N, NumPart) -> N1 is N + 1
        ;   N1 = 0
        ),
        format(atom(Next), 's_~w', [N1])
    ;   Next = 's_0'
    ).

val_hs(Str, Hs) :-
    (   number_string(N, Str), integer(N)
    ->  format(atom(Hs), 'Integer ~w', [N])
    ;   number_string(F, Str), float(F)
    ->  format(atom(Hs), 'Float ~w', [F])
    ;   format(atom(Hs), 'Atom "~w"', [Str])
    ).

reg_to_int(Reg, Int) :-
    atom_string(RegA, Reg),
    sub_atom(RegA, 0, 1, _, B),
    sub_atom(RegA, 1, _, 0, NA),
    atom_number(NA, Num),
    ( B == 'A' -> Int = Num ; B == 'X' -> Int is Num + 100
    ; B == 'Y' -> Int is Num + 200 ; Int = 0 ).

parse_functor(FnStr, Name, Arity) :-
    atom_string(FA, FnStr),
    (   sub_atom(FA, B, 1, _, '/')
    ->  sub_atom(FA, 0, B, _, Name),
        B1 is B + 1,
        sub_atom(FA, B1, _, 0, AS),
        atom_number(AS, Arity)
    ;   Name = FA, Arity = 0
    ).

escape_bs(Str, Esc) :-
    split_string(Str, "\\", "", Parts),
    atomic_list_concat(Parts, "\\\\", E0),
    atom_string(E0, Esc).
