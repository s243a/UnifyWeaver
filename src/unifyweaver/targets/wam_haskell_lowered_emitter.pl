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
:- use_module(wam_ite_structurer, [structure_ite/2]).
:- use_module(wam_clause_chain, [clause_chain/2]).

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
    forall(member(I, C1), supported(I)),
    % Verify clause 1 actually folds via the shared structurer. Negation
    % (!/0 commit) and nested ITEs now structure and lower; a genuinely
    % ill-formed control block fails here and falls back to the interpreter
    % instead of failing lower_all (keeps gate <-> emit in lockstep).
    clause1_pc(PCInstrs, C1PC),
    parse_wam_text(WamCode, _, LabelMap),
    struct_stream(C1PC, LabelMap, Stream),
    structure_ite(Stream, _).

%% clause1_pc(+PCInstrs, -Clause1PCs)
%  Like clause1_instrs/2 but keeps pc(PC,Instr) form, matching exactly the
%  clause-1 slice emit_func/4 structures (strip switch prefixes, then the
%  predicate-level try_me_else, then take to proceed).
clause1_pc([], []).
clause1_pc(PCInstrs0, C1) :-
    strip_switch_prefixes(PCInstrs0, PCInstrs),
    PCInstrs0 \== PCInstrs, !,
    clause1_pc(PCInstrs, C1).
clause1_pc([pc(_, try_me_else(_))|Rest], C1) :- !,
    take_to_proceed_pc(Rest, C1).
clause1_pc(PCInstrs, PCInstrs).

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
supported(get_structure(_, _)).
supported(get_list(_)).
supported(get_nil(_)).
supported(get_integer(_, _)).
supported(unify_variable(_)).
supported(unify_value(_)).
supported(unify_constant(_)).
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
    (   emit_func_t5(FN, PCInstrs1, LabelMap, ForeignPreds)
    ->  true   % T5 first-argument dispatch (all clauses lowered natively)
    ;   PCInstrs1 = [pc(_, try_me_else(LStr))|BodyPCs]
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
        emit_clause_struct(Clause1PCs, LabelMap, "s_c1", "      ", ForeignPreds)
    ;   % Single-clause: simple do-notation
        format("~w !ctx s_init = do~n", [FN]),
        emit_clause_struct(PCInstrs1, LabelMap, "s_init", "  ", ForeignPreds)
    ).

take_to_proceed_pc([], []).
take_to_proceed_pc([pc(PC, proceed)|_], [pc(PC, proceed)]) :- !.
take_to_proceed_pc([pc(PC, fail)|_], [pc(PC, fail)]) :- !.
take_to_proceed_pc([H|T], [H|R]) :- take_to_proceed_pc(T, R).

%% =====================================================================
%% T5: multi-clause as a first-argument dispatch (wam_clause_chain)
%% =====================================================================
%
%  When the clauses discriminate on a DISTINCT first-argument constant
%  (lowering type T5 in docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md)
%  ALL clauses are lowered to native Haskell and selected by a deref-and-match
%  cascade, instead of lowering only clause 1 and reaching clauses 2+ through
%  the interpreter on backtrack. When the first argument is BOUND this is
%  deterministic dispatch with no interpreter hop; when it is UNBOUND (or the
%  register is unset) we defer to the interpreter via the same choice-point /
%  backtrack / run fallback the ordinary multi-clause path uses — that path
%  enumerates every clause, binding the variable in turn.
%
%  Each clause body is emitted in FULL (the leading `get_constant V, A1` is
%  kept): on the bound fast path it harmlessly re-matches the already-bound
%  first argument, and on the unbound fallback it is exactly what binds the
%  variable. So the only thing T5 changes is HOW the matching clause is
%  reached, never what a clause does.

%% emit_func_t5(+FN, +PCInstrs1, +LabelMap, +FP) is semidet.
%  Emits the T5 dispatch and succeeds, or fails (emitting nothing) when the
%  predicate is not a distinct-first-argument constant chain. All checks run
%  before any output so a failure leaves the stream untouched for the caller's
%  fallback emission.
emit_func_t5(FN, PCInstrs1, LabelMap, FP) :-
    PCInstrs1 = [pc(_, try_me_else(L2Str))|_],
    maplist(t5_strip_pc, PCInstrs1, PlainInstrs),
    clause_chain(PlainInstrs, chain(_Guards)),
    t5_split_clauses_pc(PCInstrs1, Slices),
    Slices = [_, _ | _],
    forall(( member(Sl, Slices), member(pc(_, I), Sl) ), supported(I)),
    maplist(t5_slice_discriminator, Slices, Discrs),
    % All checks passed — emit.
    atom_string(L2Atom, L2Str),
    ( member(L2Atom-AltPC, LabelMap) -> true ; AltPC = 0 ),
    format("~w !ctx s_init =~n", [FN]),
    format("  case derefVar (wsBindings s_init) <$> IM.lookup 1 (wsRegs s_init) of~n"),
    format("    Just (Unbound _) -> t5fallback~n"),
    format("    Just v~n"),
    t5_emit_dispatch_guards(Discrs, 1),
    format("      | otherwise -> Nothing~n"),
    format("    _ -> t5fallback~n"),
    format("  where~n"),
    % Interpreter fallback for the unbound/unset first-argument case: push a
    % choice point at clause 2's label, try clause 1 natively, and on failure
    % backtrack into the interpreter to enumerate the remaining clauses.
    format("    t5fallback =~n"),
    format("      let s_cp = s_init { wsCPs = ChoicePoint~n"),
    format("            { cpNextPC = ~w, cpRegs = wsRegs s_init, cpStack = wsStack s_init~n", [AltPC]),
    format("            , cpCP = wsCP s_init, cpTrailLen = wsTrailLen s_init~n"),
    format("            , cpHeapLen = wsHeapLen s_init, cpBindings = wsBindings s_init~n"),
    format("            , cpCutBar = wsCutBar s_init, cpAggFrame = Nothing, cpBuiltin = Nothing~n"),
    format("            } : wsCPs s_init~n"),
    format("            , wsCPsLen = wsCPsLen s_init + 1 }~n"),
    format("      in case t5clause_1 s_cp of~n"),
    format("           Just result -> Just result~n"),
    format("           Nothing -> backtrack s_cp >>= \\s_bt -> run ctx (s_bt { wsPC = wsPC s_bt + 1 })~n"),
    t5_emit_clause_defs(Slices, 1, LabelMap, FP).

t5_strip_pc(pc(_, I), I).

%% t5_split_clauses_pc(+PCInstrs1, -Slices)
%  Split the switch-stripped pc-instruction list (which opens with
%  try_me_else) at the choice-point separators into per-clause slices, each
%  trimmed to its terminal proceed/fail. Mirrors wam_clause_chain's
%  split_clauses but keeps the pc(PC,Instr) wrappers for emission.
t5_split_clauses_pc([pc(_, try_me_else(_))|Rest], [Slice|More]) :-
    t5_collect_clause_pc(Rest, Clause, After),
    take_to_proceed_pc(Clause, Slice),
    t5_split_more_pc(After, More).

t5_split_more_pc([], []).
t5_split_more_pc([pc(_, retry_me_else(_))|Rest], [Slice|More]) :- !,
    t5_collect_clause_pc(Rest, Clause, After),
    take_to_proceed_pc(Clause, Slice),
    t5_split_more_pc(After, More).
t5_split_more_pc([pc(_, trust_me)|Rest], [Slice|More]) :- !,
    t5_collect_clause_pc(Rest, Clause, After),
    take_to_proceed_pc(Clause, Slice),
    t5_split_more_pc(After, More).

t5_collect_clause_pc([], [], []).
t5_collect_clause_pc([pc(P, retry_me_else(L))|Rest], [], [pc(P, retry_me_else(L))|Rest]) :- !.
t5_collect_clause_pc([pc(P, trust_me)|Rest], [], [pc(P, trust_me)|Rest]) :- !.
t5_collect_clause_pc([Item|Rest], [Item|More], After) :-
    t5_collect_clause_pc(Rest, More, After).

%% t5_slice_discriminator(+Slice, -HaskellValueExpr)
%  Each lowerable clause opens with `get_constant V, A1`; render V as its
%  Haskell Value constructor.
t5_slice_discriminator([pc(_, get_constant(VStr, _A1))|_], HC) :-
    val_hs(VStr, HC).

%% t5_emit_dispatch_guards(+Discrs, +Index)
t5_emit_dispatch_guards([], _).
t5_emit_dispatch_guards([HC|Rest], N) :-
    format("      | v == (~w) -> t5clause_~w s_init~n", [HC, N]),
    N1 is N + 1,
    t5_emit_dispatch_guards(Rest, N1).

%% t5_emit_clause_defs(+Slices, +Index, +LabelMap, +FP)
t5_emit_clause_defs([], _, _, _).
t5_emit_clause_defs([Slice|Rest], N, LabelMap, FP) :-
    format("    t5clause_~w s_c~w = do~n", [N, N]),
    format(atom(SV), 's_c~w', [N]),
    emit_clause_struct(Slice, LabelMap, SV, "      ", FP),
    N1 is N + 1,
    t5_emit_clause_defs(Rest, N1, LabelMap, FP).

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
%% Structured ITE emission (shared nesting-aware structurer)
%% =====================================================================
%
%  The flat split_ite_blocks_lm/split_at_jump heuristic is NOT nesting
%  aware (split_at_jump stops at an inner jump) and only recognises
%  cut_ite (not the !/0 negation commit), so nested ITEs and \+ failed to
%  lower. We instead feed clause 1 through the shared wam_ite_structurer:
%  struct_stream/3 rebuilds a label-marked instruction stream (control
%  markers stay bare so the structurer can match them; data instructions
%  keep their pc(PC,_) wrapper so emit_one still has the PC), structure_ite
%  folds every (C->T;E)/\+/once block into ite(Cond,Then,Else), and
%  emit_structured/4 walks the result.
%
%  emit_structured reuses the EXACT case-(do…)of formats of the previous
%  emit_instrs_lm + emit_ite_block, so a simple or sequential ITE produces
%  byte-identical Haskell (no performance change); nested blocks recurse
%  through the same path and negation is just an ite whose commit was !/0.

%% emit_clause_struct(+ClausePCs, +LabelMap, +SV, +Ind, +FP)
emit_clause_struct(ClausePCs, LabelMap, SV, Ind, FP) :-
    struct_stream(ClausePCs, LabelMap, Stream),
    structure_ite(Stream, Structured),
    emit_structured(Structured, SV, Ind, FP).

%% struct_stream(+ClausePCs, +LabelMap, -Stream)
%  Re-insert label(L) markers at their PCs and keep ITE control markers
%  (try_me_else/jump/trust_me/cut_ite and the !/0 commit) bare so the
%  structurer matches them; every other instruction stays pc(PC,Instr).
struct_stream([], _LM, []).
struct_stream([pc(PC, Instr)|Rest], LM, Out) :-
    % LabelMap keys are atoms, but try_me_else/jump args are strings; emit
    % label markers as strings so structure_ite unifies them.
    findall(label(LStr), (member(L-PC, LM), atom_string(L, LStr)), Labels),
    struct_item(PC, Instr, Item),
    append(Labels, [Item|More], Out),
    struct_stream(Rest, LM, More).

struct_item(_PC, try_me_else(L), try_me_else(L)) :- !.
struct_item(_PC, jump(L),        jump(L))        :- !.
struct_item(_PC, trust_me,       trust_me)       :- !.
struct_item(_PC, cut_ite,        cut_ite)        :- !.
struct_item(_PC, builtin_call(Op, N), builtin_call(Op, N)) :- neg_commit_op(Op), !.
struct_item(PC, Instr, pc(PC, Instr)).

neg_commit_op("!/0").
neg_commit_op('!/0').

%% emit_structured(+Structured, +SV, +Ind, +FP)
%  Structured is a list of pc(PC,Instr) and ite(Cond,Then,Else) terms.
%  Emits a do-block body yielding (Maybe WamState).
emit_structured([], SV, Ind, _FP) :-
    format("~wreturn ~w~n", [Ind, SV]).
% Terminal: fail
emit_structured([pc(_PC, fail)|Rest], _SV, Ind, _FP) :- !,
    (   Rest \= []
    ->  format("~w-- WARNING: fail is not the last instruction — unreachable code follows~n", [Ind])
    ;   true
    ),
    format("~wNothing~n", [Ind]).
% Terminal: execute (tail call)
emit_structured([pc(PC, execute(PredStr))|Rest], SV, Ind, FP) :- !,
    (   Rest \= []
    ->  format("~w-- WARNING: execute(~w) is not the last instruction — tail-call semantics violated~n",
               [Ind, PredStr])
    ;   true
    ),
    emit_one(execute(PredStr), PC, SV, _, Ind, FP).
% If-then-else as the final expression (no continuation)
emit_structured([ite(Cond, Then, Else)], SV, Ind, FP) :- !,
    format("~wcase (do~n", [Ind]),
    atom_concat(Ind, "      ", CondInd),
    emit_structured(Cond, SV, CondInd, FP),
    format("~w  ) of~n", [Ind]),
    fresh_sv(SV, SVthen),
    format("~w    Just ~w -> do~n", [Ind, SVthen]),
    atom_concat(Ind, "      ", ThenInd),
    emit_structured(Then, SVthen, ThenInd, FP),
    format("~w    Nothing -> do~n", [Ind]),
    atom_concat(Ind, "      ", ElseInd),
    emit_structured(Else, SV, ElseInd, FP).
% If-then-else with a continuation — bind the result, then continue.
emit_structured([ite(Cond, Then, Else)|Rest], SV, Ind, FP) :- Rest \= [], !,
    fresh_sv(SV, SVcont),
    format("~w~w <- case (do~n", [Ind, SVcont]),
    atom_concat(Ind, "          ", CondInd),
    emit_structured(Cond, SV, CondInd, FP),
    format("~w      ) of~n", [Ind]),
    fresh_sv(SV, SVthen),
    format("~w        Just ~w -> do~n", [Ind, SVthen]),
    atom_concat(Ind, "          ", ThenInd),
    emit_structured(Then, SVthen, ThenInd, FP),
    format("~w        Nothing -> do~n", [Ind]),
    atom_concat(Ind, "          ", ElseInd),
    emit_structured(Else, SV, ElseInd, FP),
    emit_structured(Rest, SVcont, Ind, FP).
% Last plain instruction — emit, then return its state unless terminal.
emit_structured([pc(PC, Instr)], SV, Ind, FP) :- !,
    emit_one(Instr, PC, SV, SVout, Ind, FP),
    (   is_terminal_instr(Instr) -> true
    ;   format("~wreturn ~w~n", [Ind, SVout])
    ).
% Plain instruction with more following.
emit_structured([pc(PC, Instr)|Rest], SV, Ind, FP) :- Rest \= [], !,
    emit_one(Instr, PC, SV, SVout, Ind, FP),
    emit_structured(Rest, SVout, Ind, FP).
% A bare !/0 that was NOT an ITE commit (a user cut left by the
% structurer's pass-through). emit_one ignores the PC for !/0.
emit_structured([builtin_call(Op, N)|Rest], SV, Ind, FP) :- !,
    emit_one(builtin_call(Op, N), 0, SV, SVout, Ind, FP),
    (   Rest == [] -> format("~wreturn ~w~n", [Ind, SVout])
    ;   emit_structured(Rest, SVout, Ind, FP)
    ).
% A bare cut_ite outside any ITE block (defensive; should not occur).
emit_structured([cut_ite|Rest], SV, Ind, FP) :- !,
    (   Rest == [] -> format("~wreturn ~w~n", [Ind, SV])
    ;   emit_structured(Rest, SV, Ind, FP)
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

% ---- GetStructure F Ai — delegate to step ----
emit_one(get_structure(FnStr, AiStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(AiStr, Ai),
    % Intern the FULL functor string (e.g. "+/2"), matching the interpreter
    % codegen (wam_haskell_target's get_structure/put_structure rules). The
    % lowered and interpreted paths share one runtime intern table, so a bare
    % "+" here would get a different id than the table's "+/2" and evalArith /
    % unification would look up the wrong name.
    parse_functor(FnStr, _FuncName, Arity),
    wam_haskell_target:intern_struct_functor(FnStr, FnId),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (GetStructure ~w ~w ~w)~n",
           [I, SVout, SV, PC, FnId, Ai, Arity]).

% ---- GetList Ai — delegate to step ----
emit_one(get_list(AiStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (GetList ~w)~n",
           [I, SVout, SV, PC, Ai]).

% ---- GetNil Ai — delegate to step as GetConstant (Atom atomNil) ----
emit_one(get_nil(AiStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(AiStr, Ai),
    wam_haskell_target:intern_atom("[]", NilId),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (GetConstant (Atom ~w) ~w)~n",
           [I, SVout, SV, PC, NilId, Ai]).

% ---- GetInteger N Ai — delegate to step as GetConstant (Integer N) ----
emit_one(get_integer(NStr, AiStr), PC, SV, SVout, I, _FP) :-
    (   number_string(N, NStr) -> true ; N = 0 ),
    reg_to_int(AiStr, Ai),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (GetConstant (Integer ~w) ~w)~n",
           [I, SVout, SV, PC, N, Ai]).

% ---- UnifyVariable Xn — delegate to step ----
emit_one(unify_variable(XnStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (UnifyVariable ~w)~n",
           [I, SVout, SV, PC, Xn]).

% ---- UnifyValue Xn — delegate to step ----
emit_one(unify_value(XnStr), PC, SV, SVout, I, _FP) :-
    reg_to_int(XnStr, Xn),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (UnifyValue ~w)~n",
           [I, SVout, SV, PC, Xn]).

% ---- UnifyConstant C — delegate to step ----
emit_one(unify_constant(CStr), PC, SV, SVout, I, _FP) :-
    val_hs(CStr, HC),
    fresh_sv(SV, SVout),
    format("~w~w <- step ctx (~w { wsPC = ~w }) (UnifyConstant (~w))~n",
           [I, SVout, SV, PC, HC]).

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
    % Intern the FULL functor string (see get_structure above) so the
    % builder's functor id matches the shared runtime intern table.
    parse_functor(FnStr, _FuncName, Arity),
    wam_haskell_target:intern_struct_functor(FnStr, FnId),
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
