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
%
% Inline optimizations (Phase 4.1): get_constant, get_value, put_structure,
% put_list, set_variable, set_value, set_constant, deallocate, and cut
% (!/0) are now inlined rather than delegated to step, matching the F#
% lowered emitter's optimization level. Phase I specialized instructions
% (PutStructureDyn, Arg, NotMemberList, etc.) are also lowerable,
% delegating to step.

:- module(wam_haskell_lowered_emitter, [
    wam_haskell_lowerable/3,
    lower_predicate_to_haskell/4
]).

:- use_module(library(lists)).
:- use_module(wam_text_parser, [wam_classify_constant_token/2]).

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
% Skip switch_on_constant* prefixes (multi-clause indexing).
% Both switch_on_constant and switch_on_constant_a2 map to
% SwitchOnConstant at runtime, so neither is part of the lowered body.
clause1_instrs(PCInstrs0, C1) :-
    strip_switch_prefixes(PCInstrs0, PCInstrs),
    PCInstrs0 \== PCInstrs, !,
    clause1_instrs(PCInstrs, C1).
clause1_instrs([pc(_, try_me_else(_))|Rest], C1) :- !,
    take_to_proceed(Rest, C1).
clause1_instrs(PCInstrs, Instrs) :-
    maplist([pc(_, I), I]>>true, PCInstrs, Instrs).

wam_switch_prefix(Instr) :-
    functor(Instr, switch_on_constant, _), !.
wam_switch_prefix(Instr) :-
    functor(Instr, switch_on_constant_a2, _), !.

strip_switch_prefixes([pc(_, Instr)|Rest0], Rest) :-
    wam_switch_prefix(Instr), !,
    strip_switch_prefixes(Rest0, Rest).
strip_switch_prefixes(PCInstrs, PCInstrs).

take_to_proceed([], []).
take_to_proceed([pc(_, proceed)|_], [proceed]) :- !.
take_to_proceed([pc(_, fail)|_], [fail]) :- !.
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
supported(set_variable(_)).
supported(set_value(_)).
supported(set_constant(_)).
supported(call(_, _)).
supported(call_foreign(_, _)).
supported(builtin_call(_, _)).
% begin_aggregate/end_aggregate are NOT lowerable — they require the run
% loop for backtrack-driven solution collection. Delegating individual
% step calls breaks because EndAggregate needs to loop.
supported(cut_ite).
supported(jump(_)).
supported(retry_me_else(_)).
supported(trust_me).  % consumed by if-then-else pattern detection in emit_instrs
supported(proceed).
supported(fail).
% Execute — tail call to another predicate
supported(execute(_)).

% Phase I specialized instructions. Emitted by the WAM compiler's
% binding-analysis pass; each mirrors the matching wam_instr_to_haskell
% text parse rule (src/unifyweaver/targets/wam_haskell_target.pl Phase I
% block) so the lowered Haskell function emits the same Instruction
% constructor the interpreter codegen does.
supported(put_structure_dyn(_, _, _)).
supported(arg(_, _, _)).
supported(not_member_list(_, _)).
supported(build_empty_set(_)).
supported(set_insert(_, _, _)).
supported(not_member_set(_, _)).
%% not_member_const_atoms is variable-arity:
%%   not_member_const_atoms(XReg, Atom1, Atom2, ..., AtomN)  (N >= 1)
%% The compound term arity is therefore >= 2 (one XReg + one or more atoms).
supported(T) :-
    compound(T),
    functor(T, not_member_const_atoms, N),
    N >= 2.

%% =====================================================================
%% Emission
%% =====================================================================

lower_predicate_to_haskell(PI, WamCode, Opts, lowered(PredName, FuncName, Code)) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    % Sanitize predicate name for Haskell: replace $ with _
    atom_string(Pred, PredStr),
    split_string(PredStr, "$", "", Parts),
    atomic_list_concat(Parts, '_', SanitizedPred),
    format(atom(FuncName), 'lowered_~w_~w', [SanitizedPred, Arity]),
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
    % Skip switch_on_constant* prefixes before deciding multi/single-clause.
    strip_switch_prefixes(PCInstrs, PCInstrs1),
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
        emit_instrs_lm(Clause1PCs, "s_c1", "      ", ForeignPreds, LabelMap)
    ;   % Single-clause: simple do-notation
        format("~w !ctx s_init = do~n", [FN]),
        emit_instrs_lm(PCInstrs1, "s_init", "  ", ForeignPreds, LabelMap)
    ).

take_to_proceed_pc([], []).
take_to_proceed_pc([pc(PC, proceed)|_], [pc(PC, proceed)]) :- !.
take_to_proceed_pc([pc(PC, fail)|_], [pc(PC, fail)]) :- !.
take_to_proceed_pc([H|T], [H|R]) :- take_to_proceed_pc(T, R).

%% =====================================================================
%% emit_instrs_lm — LabelMap-aware top-level emission
%% =====================================================================

%% emit_instrs_lm(+PCInstrs, +SV, +Indent, +ForeignPreds, +LabelMap)
%  LabelMap-aware entry point called from emit_func.
%  Threads LabelMap into split_ite_blocks_lm so that
%  split_else_cont can correctly identify the continuation target.
emit_instrs_lm([], _, _, _, _).
emit_instrs_lm([pc(_PC, try_me_else(ElseLabelStr))|Rest], SV, Ind, FP, LM) :-
    atom_string(ElseLabel, ElseLabelStr),
    split_ite_blocks_lm(Rest, ElseLabel, LM, CondInstrs, ThenInstrs, ElseInstrs, ContInstrs),
    !,
    (   ContInstrs = []
    ->  % No continuation — emit ITE directly
        format("~wcase (do~n", [Ind]),
        atom_concat(Ind, "      ", CondInd),
        emit_ite_block(CondInstrs, SV, CondInd, FP),
        format("~w  ) of~n", [Ind]),
        fresh_sv(SV, SVthen),
        format("~w    Just ~w -> do~n", [Ind, SVthen]),
        atom_concat(Ind, "      ", ThenInd),
        emit_ite_block(ThenInstrs, SVthen, ThenInd, FP),
        format("~w    Nothing -> do~n", [Ind]),
        atom_concat(Ind, "      ", ElseInd),
        emit_ite_block(ElseInstrs, SV, ElseInd, FP)
    ;   % Has continuation — bind ITE result, then emit continuation
        fresh_sv(SV, SVcont),
        format("~w~w <- case (do~n", [Ind, SVcont]),
        atom_concat(Ind, "          ", CondInd),
        emit_ite_block(CondInstrs, SV, CondInd, FP),
        format("~w      ) of~n", [Ind]),
        fresh_sv(SV, SVthen),
        format("~w        Just ~w -> do~n", [Ind, SVthen]),
        atom_concat(Ind, "          ", ThenInd),
        emit_ite_block(ThenInstrs, SVthen, ThenInd, FP),
        format("~w        Nothing -> do~n", [Ind]),
        atom_concat(Ind, "          ", ElseInd),
        emit_ite_block(ElseInstrs, SV, ElseInd, FP),
        emit_instrs_lm(ContInstrs, SVcont, Ind, FP, LM)
    ).

% fail is a terminal instruction — nothing should follow it.
emit_instrs_lm([pc(_PC, fail)|Rest], _SV, Ind, _FP, _LM) :-
    (   Rest \= []
    ->  format("~w-- WARNING: fail is not the last instruction — unreachable code follows~n", [Ind])
    ;   true
    ),
    format("~wNothing~n", [Ind]).

% execute is a tail call — it must be the last instruction.
emit_instrs_lm([pc(PC, execute(PredStr))|Rest], SV, Ind, FP, _LM) :-
    (   Rest \= []
    ->  format("~w-- WARNING: execute(~w) is not the last instruction — tail-call semantics violated~n",
               [Ind, PredStr])
    ;   true
    ),
    emit_one(execute(PredStr), PC, SV, _, Ind, FP).

emit_instrs_lm([pc(PC, Instr)|Rest], SV, Ind, FP, LM) :-
    emit_one(Instr, PC, SV, SVout, Ind, FP),
    emit_instrs_lm(Rest, SVout, Ind, FP, LM).

%% =====================================================================
%% emit_instrs — legacy no-LabelMap version (used inside ITE blocks)
%% =====================================================================

%% emit_instrs(+PCInstrs, +CurrentStateVar, +IndentPrefix, +ForeignPreds)
%  Emit one line of do-notation per instruction, threading state vars.
%  Detects if-then-else patterns (try_me_else/cut_ite/jump/trust_me)
%  and emits native Haskell case branching instead of WAM choice points.
emit_instrs([], _, _, _).
emit_instrs([pc(_PC, try_me_else(ElseLabelStr))|Rest], SV, Ind, FP) :-
    atom_string(ElseLabel, ElseLabelStr),
    split_ite_blocks(Rest, ElseLabel, CondInstrs, ThenInstrs, ElseInstrs, _ContInstrs),
    !,
    % Emit: run condition in a nested do-block, case-split on result.
    % The case expression is the last statement — it returns directly.
    format("~wcase (do~n", [Ind]),
    atom_concat(Ind, "      ", CondInd),
    % Emit condition instructions, ending with return of final state
    emit_ite_block(CondInstrs, SV, CondInd, FP),
    format("~w  ) of~n", [Ind]),
    % Then branch: condition succeeded
    fresh_sv(SV, SVthen),
    format("~w    Just ~w -> do~n", [Ind, SVthen]),
    atom_concat(Ind, "      ", ThenInd),
    emit_ite_block(ThenInstrs, SVthen, ThenInd, FP),
    % Else branch: condition failed, restore original state
    format("~w    Nothing -> do~n", [Ind]),
    atom_concat(Ind, "      ", ElseInd),
    emit_ite_block(ElseInstrs, SV, ElseInd, FP).

% fail terminal
emit_instrs([pc(_PC, fail)|Rest], _SV, Ind, _FP) :-
    (   Rest \= []
    ->  format("~w-- WARNING: fail is not the last instruction — unreachable code follows~n", [Ind])
    ;   true
    ),
    format("~wNothing~n", [Ind]).

% execute terminal
emit_instrs([pc(PC, execute(PredStr))|Rest], SV, Ind, FP) :-
    (   Rest \= []
    ->  format("~w-- WARNING: execute(~w) is not the last instruction — tail-call semantics violated~n",
               [Ind, PredStr])
    ;   true
    ),
    emit_one(execute(PredStr), PC, SV, _, Ind, FP).

emit_instrs([pc(PC, Instr)|Rest], SV, Ind, FP) :-
    emit_one(Instr, PC, SV, SVout, Ind, FP),
    emit_instrs(Rest, SVout, Ind, FP).

%% emit_ite_block(+PCInstrs, +SV, +Ind, +FP)
%  Emit instructions for an if-then-else branch block.
%  Emits do-notation and returns the final state via the last instruction.
%  If the block ends with proceed, it emits the proceed normally.
%  Otherwise it adds a `return` of the final state.
emit_ite_block([], SV, Ind, _FP) :-
    format("~wreturn ~w~n", [Ind, SV]).
emit_ite_block([pc(_PC, fail)|Rest], _SV, Ind, _FP) :-
    (   Rest \= []
    ->  format("~w-- WARNING: fail is not the last ITE instruction — unreachable code follows~n", [Ind])
    ;   true
    ),
    format("~wNothing~n", [Ind]).
emit_ite_block([pc(PC, execute(PredStr))|Rest], SV, Ind, FP) :-
    (   Rest \= []
    ->  format("~w-- WARNING: execute(~w) is not the last ITE instruction — tail-call semantics violated~n",
               [Ind, PredStr])
    ;   true
    ),
    emit_one(execute(PredStr), PC, SV, _, Ind, FP).
emit_ite_block([pc(PC, Instr)], SV, Ind, FP) :-
    % Last instruction — emit it, then return if it's not a terminal
    emit_one(Instr, PC, SV, SVout, Ind, FP),
    (   is_terminal_instr(Instr) -> true
    ;   format("~wreturn ~w~n", [Ind, SVout])
    ).
emit_ite_block([pc(PC, Instr)|Rest], SV, Ind, FP) :-
    Rest \= [],
    emit_one(Instr, PC, SV, SVout, Ind, FP),
    emit_ite_block(Rest, SVout, Ind, FP).

is_terminal_instr(proceed).
is_terminal_instr(fail).
is_terminal_instr(execute(_)).

%% =====================================================================
%% ITE block splitting
%% =====================================================================

%% split_ite_blocks_lm/7 — LabelMap-aware version
%  Used inside emit_instrs_lm so that ContInstrs (code after the ITE
%  block's continuation jump target) is correctly split off.
split_ite_blocks_lm(Instrs, _ElseLabel, LabelMap,
                     CondInstrs, ThenInstrs, ElseInstrs, ContInstrs) :-
    split_at_instr(Instrs, cut_ite, CondInstrs, AfterCut),
    split_at_jump(AfterCut, ThenInstrs, ContLabelStr, AfterJump),
    AfterJump = [pc(_, trust_me)|ElseAndCont],
    atom_string(ContLabel, ContLabelStr),
    split_else_cont(ElseAndCont, ContLabel, LabelMap, ElseInstrs, ContInstrs).

%% split_ite_blocks/6 — legacy no-LabelMap version
%  Used inside emit_instrs (4-arg). ContInstrs is always [] here.
split_ite_blocks(Instrs, _ElseLabel, CondInstrs, ThenInstrs, ElseInstrs, ContInstrs) :-
    split_at_instr(Instrs, cut_ite, CondInstrs, AfterCut),
    split_at_jump(AfterCut, ThenInstrs, _ContLabelStr, AfterJump),
    AfterJump = [pc(_, trust_me)|ElseInstrs],
    ContInstrs = [].

split_at_instr([], _, _, _) :- !, fail.
split_at_instr([pc(_, Instr)|Rest], Instr, [], Rest) :- !.
split_at_instr([H|T], Instr, [H|Before], After) :-
    split_at_instr(T, Instr, Before, After).

split_at_jump([], [], "", []) :- !, fail.
split_at_jump([pc(_, jump(Label))|Rest], [], Label, Rest) :- !.
split_at_jump([H|T], [H|Then], Label, Rest) :-
    split_at_jump(T, Then, Label, Rest).

%% split_else_cont(+ElseAndCont, +ContLabel, +LabelMap, -ElseInstrs, -ContInstrs)
%  Split the instruction sequence following trust_me into:
%    ElseInstrs — the else branch body (up to the continuation target PC)
%    ContInstrs — instructions at and after the continuation target PC
%  Uses LabelMap to resolve ContLabel to a PC, then splits there.
split_else_cont(Instrs, ContLabel, LM, Else, Cont) :-
    (   \+ member(ContLabel-_, LM)
    ->  format(user_error,
               'WARNING: split_else_cont — ContLabel ~q not in LabelMap; continuation code will be empty~n',
               [ContLabel])
    ;   true
    ),
    split_else_cont_(Instrs, ContLabel, LM, Else, Cont).

split_else_cont_([], _ContLabel, _LM, [], []).
split_else_cont_([pc(PC, Instr)|Rest], ContLabel, LM, Else, Cont) :-
    (   member(ContLabel-ContPC, LM),
        PC >= ContPC
    ->  Else = [],
        Cont = [pc(PC, Instr)|Rest]
    ;   Else = [pc(PC, Instr)|ElseRest],
        split_else_cont_(Rest, ContLabel, LM, ElseRest, Cont)
    ).

%% =====================================================================
%% emit_one — single instruction → Haskell do-notation line
%% =====================================================================

% ---- Terminal: proceed ----
emit_one(proceed, _, SV, SV, I, _FP) :-
    format("~wlet ret_ = wsCP ~w~n", [I, SV]),
    format("~wif ret_ == 0 then Just (~w { wsPC = 0 }) else Just (~w { wsPC = ret_, wsCP = 0 })~n",
           [I, SV, SV]).

% ---- Terminal: fail ----
emit_one(fail, _, _SV, _SVout, I, _FP) :-
    format("~wNothing~n", [I]).

% ---- Allocate — always succeeds, use let ----
emit_one(allocate, _, SV, SVout, I, _FP) :-
    fresh_sv(SV, SVout),
    format("~wlet ~w = ~w { wsStack = EnvFrame (wsCP ~w) IM.empty : wsStack ~w, wsCutBar = wsCPsLen ~w }~n",
           [I, SVout, SV, SV, SV, SV]).

% ---- Deallocate — inline pop of env frame ----
% Empty stack is a hard programming error (compiler bug), so `error`
% rather than Nothing. Matches F# emitter's failwith approach.
emit_one(deallocate, _PC, SV, SVout, I, _FP) :-
    fresh_sv(SV, SVout),
    format("~wlet ~w = case wsStack ~w of~n", [I, SVout, SV]),
    format("~w           (EnvFrame oldCP _ : rest) -> ~w { wsStack = rest, wsCP = oldCP }~n",
           [I, SV]),
    format("~w           _ -> error \"Deallocate: empty WsStack\"~n", [I]).

% ---- GetVariable Xn Ai — always succeeds, inline register copy ----
% Defensive: error on missing source register (matches F# failwith).
emit_one(get_variable(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~wlet ~w = putReg ~w (derefVar (wsBindings ~w) (fromMaybe (error \"GetVariable: source register not bound\") (IM.lookup ~w (wsRegs ~w)))) ~w~n",
           [I, SVout, Xn, SV, Ai, SV, SV]).

% ---- GetConstant C Ai — inline case expression ----
% Deref Ai, match on equality or bind if Unbound, else fail.
% Mirrors step's GetConstant branch without the wsPC increment.
emit_one(get_constant(CStr, AiStr), _PC, SV, SVout, I, _FP) :-
    val_hs(CStr, HC), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~wlet val_~w = derefVar (wsBindings ~w) <$> IM.lookup ~w (wsRegs ~w)~n",
           [I, SVout, SV, Ai, SV]),
    format("~w~w <- case val_~w of~n", [I, SVout, SVout]),
    format("~w  Just v | v == (~w) -> Just ~w~n", [I, HC, SV]),
    format("~w  Just (Unbound vid) ->~n", [I]),
    format("~w    Just (~w { wsRegs = IM.insert ~w (~w) (wsRegs ~w)~n",
           [I, SV, Ai, HC, SV]),
    format("~w              , wsBindings = IM.insert vid (~w) (wsBindings ~w)~n",
           [I, HC, SV]),
    format("~w              , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings ~w)) : wsTrail ~w~n",
           [I, SV, SV]),
    format("~w              , wsTrailLen = wsTrailLen ~w + 1 })~n", [I, SV]),
    format("~w  _ -> Nothing~n", [I]).

% ---- GetValue Xn Ai — inline case expression (symmetric) ----
% Deref both regs, equal -> succeed, either side Unbound -> bind, else fail.
% Includes the symmetric case where xn holds the Unbound side.
emit_one(get_value(XnStr, AiStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~wlet va_~w = derefVar (wsBindings ~w) <$> IM.lookup ~w (wsRegs ~w)~n",
           [I, SVout, SV, Ai, SV]),
    format("~w    vx_~w = getReg ~w ~w~n", [I, SVout, Xn, SV]),
    format("~w~w <- case (va_~w, vx_~w) of~n", [I, SVout, SVout, SVout]),
    format("~w  (Just a, Just x) | a == x -> Just ~w~n", [I, SV]),
    format("~w  (Just (Unbound vid), Just x) ->~n", [I]),
    format("~w    Just (~w { wsRegs = IM.insert ~w x (wsRegs ~w)~n",
           [I, SV, Ai, SV]),
    format("~w              , wsBindings = IM.insert vid x (wsBindings ~w)~n", [I, SV]),
    format("~w              , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings ~w)) : wsTrail ~w~n",
           [I, SV, SV]),
    format("~w              , wsTrailLen = wsTrailLen ~w + 1 })~n", [I, SV]),
    format("~w  (Just a, Just (Unbound vid)) ->~n", [I]),
    format("~w    Just (~w { wsBindings = IM.insert vid a (wsBindings ~w)~n",
           [I, SV, SV]),
    format("~w              , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings ~w)) : wsTrail ~w~n",
           [I, SV, SV]),
    format("~w              , wsTrailLen = wsTrailLen ~w + 1 })~n", [I, SV]),
    format("~w  _ -> Nothing~n", [I]).

% ---- PutValue Xn Ai — always succeeds, inline ----
% Defensive: error on missing source register.
emit_one(put_value(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~wlet ~w = ~w { wsRegs = IM.insert ~w (fromMaybe (error \"PutValue: source register not bound\") (getReg ~w ~w)) (wsRegs ~w) }~n",
           [I, SVout, SV, Ai, Xn, SV, SV]).

% ---- PutVariable Xn Ai — always succeeds, inline (creates fresh Unbound) ----
emit_one(put_variable(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~wlet v_~w = Unbound (wsVarCounter ~w)~n", [I, SVout, SV]),
    format("~w    ~w = (putReg ~w v_~w ~w) { wsRegs = IM.insert ~w v_~w (wsRegs (putReg ~w v_~w ~w)), wsVarCounter = wsVarCounter ~w + 1 }~n",
           [I, SVout, Xn, SVout, SV, Ai, SVout, Xn, SVout, SV, SV]).

% ---- PutConstant C Ai — always succeeds, inline ----
emit_one(put_constant(CStr, AiStr), _, SV, SVout, I, _FP) :-
    val_hs(CStr, HC), reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~wlet ~w = ~w { wsRegs = IM.insert ~w (~w) (wsRegs ~w) }~n",
           [I, SVout, SV, Ai, HC, SV]).

% ---- PutStructure — inline let (always succeeds) ----
% Directly sets wsBuilder, matching step's PutStructure semantics.
emit_one(put_structure(FnStr, AiStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int(AiStr, Ai),
    parse_functor(FnStr, FuncName, Arity),
    wam_haskell_target:intern_atom(FuncName, FnId),
    fresh_sv(SV, SVout),
    format("~wlet ~w = ~w { wsBuilder = BuildStruct ~w ~w ~w [] }~n",
           [I, SVout, SV, FnId, Ai, Arity]).

% ---- PutList — inline let (always succeeds) ----
emit_one(put_list(AiStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~wlet ~w = ~w { wsBuilder = BuildList ~w [] }~n",
           [I, SVout, SV, Ai]).

% ---- SetVariable Xn — inline with addToBuilder ----
% Creates fresh var, stores in Xn, appends to active builder.
emit_one(set_variable(XnStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn),
    fresh_sv(SV, SVout),
    format("~w~w <- let vid_ = wsVarCounter ~w~n", [I, SVout, SV]),
    format("~w           var_ = Unbound vid_~n", [I]),
    format("~w           sv_ = putReg ~w var_ (~w { wsVarCounter = wsVarCounter ~w + 1 })~n",
           [I, Xn, SV, SV]),
    format("~w       in addToBuilder var_ sv_~n", [I]).

% ---- SetValue Xn — inline with addToBuilder ----
emit_one(set_value(XnStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn),
    fresh_sv(SV, SVout),
    format("~w~w <- case getReg ~w ~w of~n", [I, SVout, Xn, SV]),
    format("~w  Just val -> addToBuilder val ~w~n", [I, SV]),
    format("~w  Nothing  -> Nothing~n", [I]).

% ---- SetConstant C — inline with addToBuilder ----
emit_one(set_constant(CStr), _PC, SV, SVout, I, _FP) :-
    val_hs(CStr, HC),
    fresh_sv(SV, SVout),
    format("~w~w <- addToBuilder (~w) ~w~n", [I, SVout, HC, SV]).

% ---- Call — use callForeign for known foreign preds, dispatchCall otherwise ----
emit_one(call(PredStr, _NStr), PC, SV, SVout, I, FP) :-
    RetPC is PC + 1,
    fresh_sv(SV, SVout),
    atom_string(PredAtom, PredStr),
    escape_dq(PredStr, EscPred),
    (   member(PredAtom, FP)
    ->  format("~w~w <- callForeign ctx \"~w\" (~w { wsCP = ~w })~n",
               [I, SVout, EscPred, SV, RetPC])
    ;   format("~w~w <- dispatchCall ctx \"~w\" (~w { wsCP = ~w })~n",
               [I, SVout, EscPred, SV, RetPC])
    ).

% ---- CallForeign — delegate to step ----
emit_one(call_foreign(PredStr, NStr), PC, SV, SVout, I, _FP) :-
    (   number_string(N, NStr) -> true ; N = 0 ),
    fresh_sv(SV, SVout),
    escape_dq(PredStr, EscPred),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (CallForeign \"~w\" ~w)~n",
           [I, SVout, SV, PC, EscPred, N]).

% ---- BuiltinCall !/0 (cut) — inline always-succeed ----
% Drop CPs above WsCutBar. Common enough to warrant bypassing step.
emit_one(builtin_call("!/0", _NStr), _PC, SV, SVout, I, _FP) :- !,
    fresh_sv(SV, SVout),
    format("~wlet drop_~w = max 0 (wsCPsLen ~w - wsCutBar ~w)~n", [I, SVout, SV, SV]),
    format("~w    ~w = ~w { wsCPs = drop drop_~w (wsCPs ~w), wsCPsLen = wsCutBar ~w }~n",
           [I, SVout, SV, SVout, SV, SV]).

% ---- BuiltinCall (general) — delegate to step ----
emit_one(builtin_call(OpStr, NStr), PC, SV, SVout, I, _FP) :-
    escape_dq(OpStr, EOp),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (BuiltinCall \"~w\" ~w)~n",
           [I, SVout, SV, PC, EOp, NStr]).

% ---- BeginAggregate — delegate to step ----
emit_one(begin_aggregate(TypeStr, ValRegStr, ResRegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(ValRegStr, ValReg),
    reg_to_int(ResRegStr, ResReg),
    escape_dq(TypeStr, EscType),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (BeginAggregate \"~w\" ~w ~w)~n",
           [I, SVout, SV, PC, EscType, ValReg, ResReg]).

% ---- EndAggregate — delegate to step ----
emit_one(end_aggregate(ValRegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(ValRegStr, ValReg),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (EndAggregate ~w)~n",
           [I, SVout, SV, PC, ValReg]).

% ---- CutIte — delegate to step ----
emit_one(cut_ite, PC, SV, SVout, I, _FP) :-
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) CutIte~n",
           [I, SVout, SV, PC]).

% ---- Jump — delegate to step ----
emit_one(jump(LabelStr), PC, SV, SVout, I, _FP) :-
    fresh_sv(SV, SVout),
    escape_dq(LabelStr, EscLabel),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (Jump \"~w\")~n",
           [I, SVout, SV, PC, EscLabel]).

% ---- RetryMeElse — delegate to step ----
emit_one(retry_me_else(LabelStr), PC, SV, SVout, I, _FP) :-
    fresh_sv(SV, SVout),
    escape_dq(LabelStr, EscLabel),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (RetryMeElse \"~w\")~n",
           [I, SVout, SV, PC, EscLabel]).

% ---- TrustMe — delegate to step ----
emit_one(trust_me, PC, SV, SVout, I, _FP) :-
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) TrustMe~n",
           [I, SVout, SV, PC]).

% ---- Execute — tail call, returns directly ----
emit_one(execute(PredStr), _PC, SV, SV, I, FP) :-
    atom_string(PredAtom, PredStr),
    escape_dq(PredStr, EscPred),
    (   member(PredAtom, FP)
    ->  format("~wcallForeign ctx \"~w\" ~w~n", [I, EscPred, SV])
    ;   format("~wdispatchCall ctx \"~w\" ~w~n", [I, EscPred, SV])
    ).

%% =====================================================================
%% Phase I — specialized instructions: delegate to step
%% =====================================================================

% PutStructureDyn — runtime-parsed functor (name+arity from registers).
emit_one(put_structure_dyn(NameRegStr, ArityRegStr, TargetRegStr),
         PC, SV, SVout, I, _FP) :-
    reg_to_int(NameRegStr, NameReg),
    reg_to_int(ArityRegStr, ArityReg),
    reg_to_int(TargetRegStr, TargetReg),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (PutStructureDyn ~w ~w ~w)~n",
           [I, SVout, SV, PC, NameReg, ArityReg, TargetReg]).

% Arg — specialized arg/3 with compile-time N.
emit_one(arg(NStr, TRegStr, ARegStr), PC, SV, SVout, I, _FP) :-
    (   number_string(N, NStr) -> true
    ;   atom_number(NStr, N) -> true
    ;   throw(error(domain_error(arg_specialization_n, NStr), emit_one/6))
    ),
    reg_to_int(TRegStr, TReg),
    reg_to_int(ARegStr, AReg),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (Arg ~w ~w ~w)~n",
           [I, SVout, SV, PC, N, TReg, AReg]).

% NotMemberList — \+ member(X, L) on a bound VList L.
emit_one(not_member_list(XRegStr, LRegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(XRegStr, XReg),
    reg_to_int(LRegStr, LReg),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (NotMemberList ~w ~w)~n",
           [I, SVout, SV, PC, XReg, LReg]).

% NotMemberConstAtoms — variable-arity: not_member_const_atoms(XReg, A1, ..., An).
emit_one(NotMemConstAtoms, PC, SV, SVout, I, _FP) :-
    compound(NotMemConstAtoms),
    functor(NotMemConstAtoms, not_member_const_atoms, N),
    N >= 2,
    arg(1, NotMemConstAtoms, XRegStr),
    reg_to_int(XRegStr, XReg),
    findall(AtomTok,
            (between(2, N, K), arg(K, NotMemConstAtoms, AtomTok)),
            AtomTokens),
    maplist([Tok, Id]>>(
        atom_string(TokA, Tok),
        wam_haskell_target:intern_atom(TokA, Id)
    ), AtomTokens, Ids),
    atomic_list_concat(Ids, ',', IdsAtom),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (NotMemberConstAtoms ~w [~w])~n",
           [I, SVout, SV, PC, XReg, IdsAtom]).

% BuildEmptySet — write VSet IS.empty into the target register.
emit_one(build_empty_set(RegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(RegStr, Reg),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (BuildEmptySet ~w)~n",
           [I, SVout, SV, PC, Reg]).

% SetInsert — elem + VSet -> VSet (with IS.insert).
emit_one(set_insert(ERegStr, InRegStr, OutRegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(ERegStr, EReg),
    reg_to_int(InRegStr, InReg),
    reg_to_int(OutRegStr, OutReg),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (SetInsert ~w ~w ~w)~n",
           [I, SVout, SV, PC, EReg, InReg, OutReg]).

% NotMemberSet — O(log N) visited-set membership check.
emit_one(not_member_set(ERegStr, SRegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(ERegStr, EReg),
    reg_to_int(SRegStr, SReg),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (NotMemberSet ~w ~w)~n",
           [I, SVout, SV, PC, EReg, SReg]).

%% =====================================================================
%% Helpers
%% =====================================================================

%% fresh_sv(+Current, -Next) — generate next state variable name
%  s_init → s_0 → s_1 → s_2 ...
%  Bug fix: validates that the numeric part actually parsed as a number
%  before incrementing. Without this, names like s_init would map to
%  s_0 → s_0 (shadowed), since failed number_string collapses to N1=0.
fresh_sv(Cur, Next) :-
    atom_string(Cur, CStr),
    (   sub_string(CStr, 0, 2, _, "s_"),
        sub_string(CStr, 2, _, 0, NumPart),
        number_string(N, NumPart)
    ->  N1 is N + 1,
        format(atom(Next), 's_~w', [N1])
    ;   Next = 's_0'
    ).

%% val_hs(+Str, -HaskellExpr)
%  Convert a WAM value token to its Haskell Value constructor.
%  Quote handling: a token with outer single quotes (e.g. '42', '+')
%  is always an atom, even if the inner content looks like a number.
val_hs(Str, Hs) :-
    val_hs_strip_quotes(Str, Inner, ForceAtom),
    (   ForceAtom == true
    ->  atom_string(InnerA, Inner),
        wam_haskell_target:intern_atom(InnerA, AtomId),
        format(atom(Hs), 'Atom ~w', [AtomId])
    ;   atom_string(InnerS, Inner),
        wam_classify_constant_token(InnerS, Class),
        (   Class = integer(N)
        ->  format(atom(Hs), 'Integer ~w', [N])
        ;   Class = float(F)
        ->  format(atom(Hs), 'Float ~w', [F])
        ;   Class = atom(Name),
            wam_haskell_target:intern_atom(Name, AtomId),
            format(atom(Hs), 'Atom ~w', [AtomId])
        )
    ).

%% val_hs_strip_quotes(+Raw, -Inner, -ForceAtom)
%  Strip a single pair of outer single quotes if present; ForceAtom
%  is true iff the quotes were present (so the caller knows to skip
%  the number-parse fallback).
val_hs_strip_quotes(S0, S, ForceAtom) :-
    string_chars(S0, Chars0),
    (   Chars0 = [''''|Rest], append(Inner, [''''], Rest)
    ->  string_chars(S, Inner),
        ForceAtom = true
    ;   S = S0,
        ForceAtom = false
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

%% escape_dq(+Str, -Escaped)
%  Escape backslashes and double quotes for Haskell string literals.
%  Handles both atom and string inputs.
escape_dq(Str0, Esc) :-
    (   string(Str0)
    ->  Str = Str0
    ;   atom(Str0)
    ->  atom_string(Str0, Str)
    ;   term_string(Str0, Str)
    ),
    split_string(Str, "\\", "", SlashParts),
    atomic_list_concat(SlashParts, "\\\\", EscSlashes),
    split_string(EscSlashes, "\"", "", QuoteParts),
    atomic_list_concat(QuoteParts, "\\\"", Esc).
