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
    lower_predicate_to_fsharp/4,
    fsharp_fact_table_classify/3,     % +PI, +Opts, -fact_info(Arity,Rows)
    emit_fact_table_fsharp/4          % +FuncName, +Arity, +Rows, -Code
]).

:- use_module(library(lists)).
:- use_module(wam_ite_structurer, [structure_ite/2]).
:- use_module(wam_clause_chain, [clause_chain/2]).

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
    parse_wam_text_fs(WamCode, PCInstrs, LabelMap),
    clause1_instrs_fs(PCInstrs, C1),
    forall(member(I, C1), supported_fs(I)),
    % Clause 1 must fold cleanly via the shared structurer. This enables
    % if-then-else / negation lowering (clause 1's internal try_me_else is a
    % well-formed ITE block, consumed into ite/3) while still rejecting any
    % stray choice-point markers (predicate-level try/retry/trust that are
    % not part of a clean block remain in the structured form, failing the
    % checks below, so such predicates fall back to the interpreter).
    clause1_pc_fs(PCInstrs, C1PC),
    struct_stream_fs(C1PC, LabelMap, Stream),
    structure_ite(Stream, Structured),
    \+ member(try_me_else(_), Structured),
    \+ member(trust_me, Structured),
    \+ member(retry_me_else(_), Structured).

%% clause1_pc_fs(+PCInstrs, -Clause1PCs)
%  Like clause1_instrs_fs/2 but keeps pc(PC,Instr) form, matching exactly the
%  clause-1 slice emit_func_fs structures.
clause1_pc_fs([], []).
clause1_pc_fs(PCInstrs0, C1) :-
    strip_switch_prefixes_fs(PCInstrs0, PCInstrs),
    PCInstrs0 \== PCInstrs, !,
    clause1_pc_fs(PCInstrs, C1).
clause1_pc_fs([pc(_, try_me_else(_))|Rest], C1) :- !,
    take_to_proceed_pc_fs(Rest, C1).
clause1_pc_fs(PCInstrs, PCInstrs).

% Match Rust/Clojure lowered emitters: only deterministic clause-1 bodies are
% lowered. Choicepoint-manipulating instructions should stay interpreter-driven.
is_deterministic_pred_fs(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

clause1_instrs_fs([], []).
% Skip switch_on_constant* prefixes (multi-clause indexing).  The F#
% runtime maps both switch_on_constant and switch_on_constant_a2 to
% SwitchOnConstant, so neither is part of the lowered clause body.
clause1_instrs_fs(PCInstrs0, C1) :-
    strip_switch_prefixes_fs(PCInstrs0, PCInstrs),
    PCInstrs0 \== PCInstrs, !,
    clause1_instrs_fs(PCInstrs, C1).
clause1_instrs_fs([pc(_, try_me_else(_))|Rest], C1) :- !,
    take_to_proceed_fs(Rest, C1).
clause1_instrs_fs(PCInstrs, Instrs) :-
    maplist([pc(_, I), I]>>true, PCInstrs, Instrs).

take_to_proceed_fs([], []).
% Clause bodies can terminate with either success (proceed) or explicit
% failure.  Stop at fail too so later alternative-clause instructions are not
% misclassified as part of clause 1 when checking lowerability.
take_to_proceed_fs([pc(_, proceed)|_], [proceed]) :- !.
take_to_proceed_fs([pc(_, fail)|_], [fail]) :- !.
take_to_proceed_fs([pc(_, I)|Rest], [I|More]) :- take_to_proceed_fs(Rest, More).

wam_fsharp_switch_prefix(Instr) :-
    functor(Instr, switch_on_constant, _), !.
wam_fsharp_switch_prefix(Instr) :-
    functor(Instr, switch_on_constant_a2, _), !.

strip_switch_prefixes_fs([pc(_, Instr)|Rest0], Rest) :-
    wam_fsharp_switch_prefix(Instr), !,
    strip_switch_prefixes_fs(Rest0, Rest).
strip_switch_prefixes_fs(PCInstrs, PCInstrs).

supported_fs(try_me_else(_)).
supported_fs(allocate).
supported_fs(deallocate).
supported_fs(get_constant(_, _)).
supported_fs(get_variable(_, _)).
supported_fs(get_value(_, _)).
supported_fs(get_structure(_, _)).
supported_fs(get_structure(_, _, _)).
supported_fs(get_list(_)).
supported_fs(get_nil(_)).
supported_fs(get_integer(_, _)).
supported_fs(unify_variable(_)).
supported_fs(unify_value(_)).
supported_fs(unify_constant(_)).
supported_fs(put_constant(_, _)).
supported_fs(put_variable(_, _)).
supported_fs(put_value(_, _)).
supported_fs(put_structure(_, _)).
supported_fs(put_list(_)).
supported_fs(set_variable(_)).
supported_fs(set_value(_)).
supported_fs(set_constant(_)).
supported_fs(call(_, _)).
supported_fs(call_foreign(_, _)).
supported_fs(builtin_call(_, _)).
% begin_aggregate/end_aggregate not lowerable — need the run loop for
% backtrack-driven collection (same constraint as Haskell target).
supported_fs(cut_ite).
supported_fs(jump(_)).
supported_fs(retry_me_else(_)).
supported_fs(trust_me).
supported_fs(proceed).
supported_fs(fail).
supported_fs(execute(_)).

% ----------------------------------------------------------------------
% Phase I — Haskell-only specialized instructions.  Emitted by the WAM
% compiler's binding-analysis pass; previously the lowered emitter
% silently fell back to interpreter mode when it saw any of these.
% Each clause below mirrors the matching wam_instr_to_fsharp text parse
% rule (src/unifyweaver/targets/wam_fsharp_target.pl Phase I block) so
% the lowered F# function emits the same Instruction constructor the
% interpreter codegen does.
% ----------------------------------------------------------------------
supported_fs(put_structure_dyn(_, _, _)).
supported_fs(arg(_, _, _)).
supported_fs(not_member_list(_, _)).
supported_fs(build_empty_set(_)).
supported_fs(set_insert(_, _, _)).
supported_fs(not_member_set(_, _)).
%% not_member_const_atoms is variable-arity:
%%   not_member_const_atoms(XReg, Atom1, Atom2, ..., AtomN)  (N >= 1)
%% The compound term arity is therefore >= 2 (one XReg + one or more atoms).
supported_fs(T) :-
    compound(T),
    functor(T, not_member_const_atoms, N),
    N >= 2.

% ============================================================================
% Emission
% ============================================================================

%% lower_predicate_to_fsharp(+PI, +WamCode, +Opts, -lowered(PredName, FuncName, Code))
%  T9 fact-table inline: an all-ground-facts predicate whose row count is in the
%  inline window [t9_min_rows, t9_max_rows] (defaults 64..256) lowers to a static
%  row table + first-arg index + a backtracking enumerator (factTableAttempt),
%  registered as a lowered predicate so call/execute reach it for free. Default
%  in-range; opt out with fact_table_inline(false). Checked first so it wins.
lower_predicate_to_fsharp(PI, _WamCode, Opts, lowered(PredName, FuncName, Code)) :-
    \+ member(fact_table_inline(false), Opts),
    fsharp_fact_table_classify(PI, Opts, fact_info(Arity, Rows)),
    !,
    ( PI = _M:Pred/_ -> true ; PI = Pred/_ ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    atom_string(Pred, PredStr0),
    split_string(PredStr0, "$", "", PParts),
    atomic_list_concat(PParts, '_', SanPred),
    format(atom(FuncName), 'lowered_~w_~w', [SanPred, Arity]),
    emit_fact_table_fsharp(FuncName, Arity, Rows, Code).

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
        emit_func_fs(FuncName, GlobalPCInstrs, GlobalLabelMap, ForeignPreds, Opts)).

% --- T9 fact-table inline: classification + emission ------------------------

%% fsharp_fact_table_classify(+PI, +Opts, -fact_info(Arity, Rows))
%  An all-ground-facts predicate (every clause a ground unit clause) whose row
%  count is in [t9_min_rows, t9_max_rows]. Rows are arg tuples in source order.
fsharp_fact_table_classify(PI, Opts, fact_info(Arity, Rows)) :-
    ( PI = Module:Pred/Arity -> true ; PI = Pred/Arity, Module = user ),
    Arity >= 1,
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(Module:Head, Body), Clauses),
    Clauses = [_|_],
    forall(member(_-B, Clauses), B == true),
    forall(member(H-_, Clauses), ( H =.. [_|As], forall(member(A, As), ground(A)) )),
    findall(As, ( member(H-true, Clauses), H =.. [_|As] ), Rows),
    fsharp_t9_min_rows(Opts, Min),
    fsharp_t9_max_rows(Opts, Max),
    length(Rows, NR),
    NR >= Min,
    NR =< Max.

fsharp_t9_min_rows(Opts, N) :- ( member(t9_min_rows(N), Opts) -> true ; N = 64 ).
fsharp_t9_max_rows(Opts, N) :- ( member(t9_max_rows(N), Opts) -> true ; N = 256 ).

%% emit_fact_table_fsharp(+FuncName, +Arity, +Rows, -Code)
%  Emit a static row table + first-arg index (built once at module init) and a
%  lowered predicate that derefs its args, selects candidates (index bucket for a
%  bound atomic first arg, else full scan) and drives factTableAttempt.
emit_fact_table_fsharp(FuncName, Arity, Rows, Code) :-
    maplist(fsharp_fact_row_literal, Rows, RowLits),
    atomic_list_concat(RowLits, '\n          ', RowsBody),
    format(string(Code),
'let private ~w_rows : Value list =
        [ ~w ]
let private ~w_index : Map<string, Value list> =
    ~w_rows
    |> List.choose (fun r -> match r with | VList (c0 :: _) -> (match factIndexKey c0 with Some k -> Some (k, r) | None -> None) | _ -> None)
    |> List.groupBy fst
    |> List.map (fun (k, ps) -> (k, ps |> List.map snd))
    |> Map.ofList
let ~w (_ctx: WamContext) (s: WamState) : WamState option =
    let args = [ for i in 1 .. ~w -> (match getReg i s with Some v -> derefVar s.WsBindings v | None -> Unbound -1) ]
    let cands =
        match args with
        | a1 :: _ -> (match factIndexKey a1 with Some k -> (match Map.tryFind k ~w_index with Some rs -> rs | None -> []) | None -> ~w_rows)
        | [] -> ~w_rows
    factTableAttempt args cands s.WsCP s',
        [FuncName, RowsBody, FuncName, FuncName, FuncName, Arity, FuncName, FuncName, FuncName]).

% One row -> `VList [<col>; <col>; ...]`.
fsharp_fact_row_literal(Row, Lit) :-
    maplist(fsharp_term_to_value_literal, Row, ColLits),
    atomic_list_concat(ColLits, '; ', Inner),
    format(string(Lit), 'VList [~w]', [Inner]).

% Ground Prolog term -> F# Value literal (matches the F# Value DU). Integers and
% floats are parenthesised so a leading minus is not read as subtraction.
fsharp_term_to_value_literal(T, L) :- integer(T), !, format(string(L), 'Integer (~w)', [T]).
fsharp_term_to_value_literal(T, L) :- float(T), !, format(string(L), 'Float (~w)', [T]).
fsharp_term_to_value_literal(T, L) :- is_list(T), !,
    maplist(fsharp_term_to_value_literal, T, Es), atomic_list_concat(Es, '; ', I),
    format(string(L), 'VList [~w]', [I]).
fsharp_term_to_value_literal(T, L) :- atom(T), !,
    fsharp_escape_string(T, E), format(string(L), 'Atom "~w"', [E]).
fsharp_term_to_value_literal(T, L) :- string(T), !,
    fsharp_escape_string(T, E), format(string(L), 'Atom "~w"', [E]).
fsharp_term_to_value_literal(T, L) :- compound(T), !,
    T =.. [F|As], fsharp_escape_string(F, EF),
    maplist(fsharp_term_to_value_literal, As, Es), atomic_list_concat(Es, '; ', I),
    format(string(L), 'Str ("~w", [~w])', [EF, I]).

% Escape backslash and double-quote for an F# string literal.
fsharp_escape_string(In, Out) :-
    atom_string(In, S),
    split_string(S, "\\", "", P1), atomic_list_concat(P1, '\\\\', S1),
    split_string(S1, "\"", "", P2), atomic_list_concat(P2, '\\"', Out).

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

emit_func_fs(FN, PCInstrs, LabelMap, ForeignPreds, Opts) :-
    format("/// Lowered: ~w~n", [FN]),
    format("let ~w (ctx: WamContext) (s_init: WamState) : WamState option =~n", [FN]),
    % Skip all switch_on_constant* prefixes before deciding whether this is
    % a multi-clause or single-clause lowered body.
    strip_switch_prefixes_fs(PCInstrs, PCInstrs1),
    (   emit_func_t5_fs(PCInstrs1, LabelMap, ForeignPreds, Opts)
    ->  true   % T5/T6 first-argument dispatch (all clauses lowered natively)
    ;   emit_func_t4_fs(PCInstrs1, LabelMap, ForeignPreds)
    ->  true   % T4 all-clauses inline (every clause native, no interpreter hop)
    ;   PCInstrs1 = [pc(_, try_me_else(LStr))|BodyPCs]
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
        format("                         CpB0StackLen = List.length s_init.WsB0Stack~n"),
        format("                         CpAggFrame = None~n"),
        format("                         CpBuiltin  = None } :: s_init.WsCPs~n"),
        format("            WsCPsLen = s_init.WsCPsLen + 1 }~n"),
        format("    let clause1 (s_c1: WamState) : WamState option =~n"),
        take_to_proceed_pc_fs(BodyPCs, Clause1PCs),
        emit_clause_struct_fs(Clause1PCs, LabelMap, "s_c1", "        ", ForeignPreds),
        format("    match clause1 s_cp with~n"),
        format("    | Some result -> Some result~n"),
        format("    | None ->~n"),
        format("        // Clause 1 failed — backtrack to clause 2+ in the interpreter~n"),
        format("        backtrack s_cp |> Option.bind (fun s_bt -> run ctx { s_bt with WsPC = s_bt.WsPC + 1 })~n")
    ;   % Single-clause: straightforward binding chain
        emit_clause_struct_fs(PCInstrs1, LabelMap, "s_init", "    ", ForeignPreds)
    ).

take_to_proceed_pc_fs([], []).
take_to_proceed_pc_fs([pc(PC, proceed)|_], [pc(PC, proceed)]) :- !.
take_to_proceed_pc_fs([pc(PC, fail)|_], [pc(PC, fail)]) :- !.
take_to_proceed_pc_fs([H|T], [H|R]) :- take_to_proceed_pc_fs(T, R).

% ============================================================================
% T5: multi-clause as a first-argument dispatch (wam_clause_chain)
%
%  When the clauses discriminate on a DISTINCT first-argument constant
%  (lowering type T5 in docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md)
%  ALL clauses are lowered to native F# and selected by a deref-and-match
%  cascade, instead of lowering only clause 1 and reaching clauses 2+ through
%  the interpreter on backtrack. When the first argument is BOUND this is
%  deterministic dispatch with no interpreter hop; when it is UNBOUND (or the
%  register is unset) we defer to the interpreter via the same choice-point /
%  backtrack / run fallback the ordinary multi-clause path uses.
%
%  Each clause body is emitted in FULL (the leading `get_constant V, A1` is
%  kept): on the bound fast path it harmlessly re-matches the already-bound
%  first argument, and on the unbound fallback it is exactly what binds the
%  variable. Mirrors the Haskell emitter's emit_func_t5/4.
% ============================================================================

%% emit_func_t5_fs(+PCInstrs1, +LabelMap, +FP) is semidet.
%  Emits the T5 dispatch body and succeeds, or fails (emitting nothing) when
%  the predicate is not a distinct-first-argument constant chain. All checks
%  run before any output, so a failure leaves the stream untouched for the
%  caller's ordinary multi/single-clause emission.
emit_func_t5_fs(PCInstrs1, LabelMap, FP, Opts) :-
    PCInstrs1 = [pc(_, try_me_else(L2Str))|_],
    maplist(t5_strip_pc_fs, PCInstrs1, PlainInstrs),
    clause_chain(PlainInstrs, chain(_Guards)),
    t5_split_clauses_pc_fs(PCInstrs1, Slices),
    Slices = [_, _ | _],
    forall(( member(Sl, Slices), member(pc(_, In), Sl) ), supported_fs(In)),
    maplist(t5_slice_discriminator_fs, Slices, Discrs),
    maplist(t5_slice_discr_token_fs, Slices, Tokens),
    % All checks passed — emit.
    atom_string(L2Atom, L2Str),
    ( member(L2Atom-AltPC, LabelMap) -> true ; AltPC = 0 ),
    t5_emit_clause_defs_fs(Slices, 1, LabelMap, FP),
    % Interpreter fallback for the unbound/unset first-argument case.
    format("    let t5fallback () : WamState option =~n"),
    format("        let s_cp =~n"),
    format("            { s_init with~n"),
    format("                WsCPs    = { CpNextPC   = ~w~n", [AltPC]),
    format("                             CpRegs     = s_init.WsRegs~n"),
    format("                             CpStack    = s_init.WsStack~n"),
    format("                             CpCP       = s_init.WsCP~n"),
    format("                             CpTrailLen = s_init.WsTrailLen~n"),
    format("                             CpHeapLen  = s_init.WsHeapLen~n"),
    format("                             CpBindings = s_init.WsBindings~n"),
    format("                             CpCutBar   = s_init.WsCutBar~n"),
    format("                             CpB0StackLen = List.length s_init.WsB0Stack~n"),
    format("                             CpAggFrame = None~n"),
    format("                             CpBuiltin  = None } :: s_init.WsCPs~n"),
    format("                WsCPsLen = s_init.WsCPsLen + 1 }~n"),
    format("        match t5clause_1 s_cp with~n"),
    format("        | Some result -> Some result~n"),
    format("        | None ->~n"),
    format("            backtrack s_cp |> Option.bind (fun s_bt -> run ctx { s_bt with WsPC = s_bt.WsPC + 1 })~n"),
    % Dispatch: deref the first argument once, then select. When the
    % discriminators are all atoms and there are enough of them (the T6 gate),
    % emit a native F# string `match` (which the F# compiler lowers to an
    % efficient hash/jump dispatch) instead of the linear if/elif cascade.
    (   fsharp_t6_applicable(Tokens, Opts)
    ->  t6_emit_dispatch_fs(Tokens)
    ;   format("    // T5 first-argument dispatch (if/elif cascade)~n"),
        format("    match (getReg 1 s_init |> Option.map (derefVar s_init.WsBindings)) with~n"),
        format("    | Some (Unbound _) -> t5fallback ()~n"),
        format("    | Some v ->~n"),
        t5_emit_dispatch_arms_fs(Discrs, 1),
        format("    | None -> t5fallback ()~n")
    ).

t5_strip_pc_fs(pc(_, I), I).

%% t5_slice_discr_token_fs(+Slice, -VStr) — the raw first-arg constant token.
t5_slice_discr_token_fs([pc(_, get_constant(VStr, _A1))|_], VStr).

% ============================================================================
% T6: first-argument indexing (native string match), gated above T5
%
%  When every clause discriminates on a distinct ATOM and there are at least
%  t6_min_clauses of them (default 8), the linear if/elif cascade is replaced by
%  a native F# `match` on the atom's string. F# compiles a many-branch string
%  match to a hash/jump dispatch, so this is O(1) where the cascade is O(n);
%  below the threshold the compiler would just flatten the match back to a
%  cascade, so it is gated. The gate ties its atom test to val_fs (the same
%  detector the T5 cascade uses), so the two paths agree on what an atom is.
% ============================================================================

%% fsharp_t6_applicable(+Tokens, +Opts) is semidet.
fsharp_t6_applicable(Tokens, Opts) :-
    fsharp_t6_min_clauses(Opts, Min),
    length(Tokens, N), N >= Min,
    forall(member(T, Tokens), t6_atom_token_fs(T, _)).

%% fsharp_t6_min_clauses(+Opts, -N) — threshold (default 8).
fsharp_t6_min_clauses(Opts, N) :-
    ( member(t6_min_clauses(N), Opts) -> true ; N = 8 ).

%% t6_atom_token_fs(+VStr, -EscStr) is semidet.
%  Succeeds only when VStr is an ATOM (val_fs renders it `Atom "EscStr"`), with
%  EscStr the F#-escaped name used in the string match arm — exactly the literal
%  the T5 cascade would compare against, so dispatch is identical.
t6_atom_token_fs(VStr, EscStr) :-
    val_fs(VStr, FS),
    atom_concat('Atom "', Rest, FS),
    atom_concat(EscStr, '"', Rest).

%% t6_emit_dispatch_fs(+Tokens)
t6_emit_dispatch_fs(Tokens) :-
    format("    // T6 first-argument indexing (native string match)~n"),
    format("    match (getReg 1 s_init |> Option.map (derefVar s_init.WsBindings)) with~n"),
    format("    | Some (Unbound _) -> t5fallback ()~n"),
    format("    | Some (Atom t6s) ->~n"),
    format("        (match t6s with~n"),
    t6_emit_match_arms_fs(Tokens, 1),
    format("         | _ -> None)~n"),
    format("    | Some _ -> None~n"),
    format("    | None -> t5fallback ()~n").

t6_emit_match_arms_fs([], _).
t6_emit_match_arms_fs([T|Rest], N) :-
    t6_atom_token_fs(T, EscStr),
    format("         | \"~w\" -> t5clause_~w s_init~n", [EscStr, N]),
    N1 is N + 1,
    t6_emit_match_arms_fs(Rest, N1).

% ============================================================================
% T4: multi-clause, all clauses inline (multi_clause_n)
%
%  When the clauses do NOT discriminate on a distinct first-argument constant
%  (so T5 declines) but every clause is a fully-supported deterministic body,
%  lower them ALL: each clause becomes a `WamState -> WamState option`, and the
%  function tries them in order on the SAME input state, taking the first Some.
%  F#'s immutability gives a free per-clause restore (each clause runs against
%  the unchanged s_init), so — unlike the imperative targets — no snapshot/
%  restore is needed and no choice point is pushed: the interpreter is never
%  entered for the predicate. Mirrors the Haskell emitter's emit_func_t4.
% ============================================================================

%% emit_func_t4_fs(+PCInstrs1, +LabelMap, +FP) is semidet.
%  Emits the T4 all-clauses body and succeeds, or fails (emitting nothing)
%  when the predicate is not a multi-clause predicate whose every clause is a
%  supported deterministic body. All checks run before any output.
emit_func_t4_fs(PCInstrs1, LabelMap, FP) :-
    PCInstrs1 = [pc(_, try_me_else(_))|_],
    t5_split_clauses_pc_fs(PCInstrs1, Slices),
    Slices = [_, _ | _],
    forall(member(Sl, Slices),
           ( % Reject an if-then-else soft-cut block masquerading as a
             % multi-clause head: it also opens with try_me_else and uses
             % trust_me as its else-separator, but its slices carry cut_ite /
             % jump markers a clean multi-clause body never has. ITE blocks
             % fall through to the multi_clause_1 path.
             \+ member(pc(_, try_me_else(_)), Sl),
             \+ member(pc(_, cut_ite), Sl),
             \+ member(pc(_, jump(_)), Sl),
             forall(member(pc(_, In), Sl), supported_fs(In)) )),
    % All checks passed — emit.
    format("    // T4 all-clauses inline (generated): try each clause on the input~n"),
    format("    // state; first Some wins. Immutability gives a free per-clause~n"),
    format("    // restore, so the interpreter is never entered for the predicate.~n"),
    t4_emit_clause_defs_fs(Slices, 1, LabelMap, FP),
    format("    t4clause_1 s_init~n"),
    t4_emit_chain_tail_fs(Slices, 2).

%% t4_emit_clause_defs_fs(+Slices, +Index, +LabelMap, +FP)
t4_emit_clause_defs_fs([], _, _, _).
t4_emit_clause_defs_fs([Slice|Rest], N, LabelMap, FP) :-
    format("    let t4clause_~w (s_c~w: WamState) : WamState option =~n", [N, N]),
    format(atom(SV), 's_c~w', [N]),
    emit_clause_struct_fs(Slice, LabelMap, SV, "        ", FP),
    N1 is N + 1,
    t4_emit_clause_defs_fs(Rest, N1, LabelMap, FP).

%% t4_emit_chain_tail_fs(+Slices, +Index) — emit `|> Option.orElseWith` for
%  clauses 2..N (lazy; clause 1 already emitted as the chain head).
t4_emit_chain_tail_fs(Slices, N) :-
    (   nth1(N, Slices, _)
    ->  format("    |> Option.orElseWith (fun () -> t4clause_~w s_init)~n", [N]),
        N1 is N + 1,
        t4_emit_chain_tail_fs(Slices, N1)
    ;   true
    ).

%% t5_split_clauses_pc_fs(+PCInstrs1, -Slices)
%  Split the switch-stripped pc-instruction list (opens with try_me_else) at
%  the choice-point separators into per-clause slices, each trimmed to its
%  terminal proceed/fail. Mirrors wam_clause_chain's split_clauses but keeps
%  the pc(PC,Instr) wrappers for emission.
t5_split_clauses_pc_fs([pc(_, try_me_else(_))|Rest], [Slice|More]) :-
    t5_collect_clause_pc_fs(Rest, Clause, After),
    take_to_proceed_pc_fs(Clause, Slice),
    t5_split_more_pc_fs(After, More).

t5_split_more_pc_fs([], []).
t5_split_more_pc_fs([pc(_, retry_me_else(_))|Rest], [Slice|More]) :- !,
    t5_collect_clause_pc_fs(Rest, Clause, After),
    take_to_proceed_pc_fs(Clause, Slice),
    t5_split_more_pc_fs(After, More).
t5_split_more_pc_fs([pc(_, trust_me)|Rest], [Slice|More]) :- !,
    t5_collect_clause_pc_fs(Rest, Clause, After),
    take_to_proceed_pc_fs(Clause, Slice),
    t5_split_more_pc_fs(After, More).

t5_collect_clause_pc_fs([], [], []).
t5_collect_clause_pc_fs([pc(P, retry_me_else(L))|Rest], [], [pc(P, retry_me_else(L))|Rest]) :- !.
t5_collect_clause_pc_fs([pc(P, trust_me)|Rest], [], [pc(P, trust_me)|Rest]) :- !.
t5_collect_clause_pc_fs([Item|Rest], [Item|More], After) :-
    t5_collect_clause_pc_fs(Rest, More, After).

%% t5_slice_discriminator_fs(+Slice, -FSharpValueExpr)
t5_slice_discriminator_fs([pc(_, get_constant(VStr, _A1))|_], FC) :-
    val_fs(VStr, FC).

%% t5_emit_clause_defs_fs(+Slices, +Index, +LabelMap, +FP)
t5_emit_clause_defs_fs([], _, _, _).
t5_emit_clause_defs_fs([Slice|Rest], N, LabelMap, FP) :-
    format("    let t5clause_~w (s_c~w: WamState) : WamState option =~n", [N, N]),
    format(atom(SV), 's_c~w', [N]),
    emit_clause_struct_fs(Slice, LabelMap, SV, "        ", FP),
    N1 is N + 1,
    t5_emit_clause_defs_fs(Rest, N1, LabelMap, FP).

%% t5_emit_dispatch_arms_fs(+Discrs, +Index)
%  Emit the bound-value if/elif cascade inside the `| Some v ->` arm.
t5_emit_dispatch_arms_fs([FC|Rest], 1) :- !,
    format("        if v = (~w) then t5clause_1 s_init~n", [FC]),
    t5_emit_dispatch_arms_fs(Rest, 2).
t5_emit_dispatch_arms_fs([], _) :- !,
    format("        else None~n").
t5_emit_dispatch_arms_fs([FC|Rest], N) :-
    format("        elif v = (~w) then t5clause_~w s_init~n", [FC, N]),
    N1 is N + 1,
    t5_emit_dispatch_arms_fs(Rest, N1).

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

%% is_match_instr_fs(+Instr)
%  True if emit_one_fs for this instruction ends by emitting a
%  "| Some SVout ->" arm that must have a body nested one level deeper.
%  This drives the indentation decision in emit_instrs_fs for the
%  last-instruction case (body = "Some SVout") and intermediate case
%  (body = the rest of the chain at IndInner).
is_match_instr_fs(get_constant(_, _)).
is_match_instr_fs(get_value(_, _)).
is_match_instr_fs(get_structure(_, _)).
is_match_instr_fs(get_structure(_, _, _)).
is_match_instr_fs(get_list(_)).
is_match_instr_fs(get_nil(_)).
is_match_instr_fs(get_integer(_, _)).
is_match_instr_fs(unify_variable(_)).
is_match_instr_fs(unify_value(_)).
is_match_instr_fs(unify_constant(_)).
% PutStructure / PutList moved to inline let-binding emitters; no longer
% match-emitting.
is_match_instr_fs(set_variable(_)).
is_match_instr_fs(set_value(_)).
is_match_instr_fs(set_constant(_)).
is_match_instr_fs(call(_, _)).
is_match_instr_fs(call_foreign(_, _)).
% Cut (!/0) is now an always-succeed let-binding (inlined below).
% Every other builtin_call still delegates to step as a match arm.
is_match_instr_fs(builtin_call("!/0", _)) :- !, fail.
is_match_instr_fs(builtin_call(_, _)).
is_match_instr_fs(cut_ite).
is_match_instr_fs(jump(_)).
is_match_instr_fs(retry_me_else(_)).
is_match_instr_fs(trust_me).
is_match_instr_fs(begin_aggregate(_, _, _)).
is_match_instr_fs(end_aggregate(_)).
%% Phase I — all delegate to `step` via a match arm, so they need the
%% `| Some SVout ->` continuation indent like every other match-emitting
%% instruction.
is_match_instr_fs(put_structure_dyn(_, _, _)).
is_match_instr_fs(arg(_, _, _)).
is_match_instr_fs(not_member_list(_, _)).
is_match_instr_fs(build_empty_set(_)).
is_match_instr_fs(set_insert(_, _, _)).
is_match_instr_fs(not_member_set(_, _)).
is_match_instr_fs(T) :-
    compound(T),
    functor(T, not_member_const_atoms, N),
    N >= 2.

%% emit_instrs_lm_fs(+PCInstrs, +SV, +Indent, +FP, +LabelMap)
%  LabelMap-aware entry point called from emit_func_fs.
%  Threads LabelMap into split_ite_blocks_lm_fs so that
%  split_else_cont_fs can correctly identify the continuation target.
emit_instrs_lm_fs([], SV, I, _FP, _LM) :-
    format("~wSome ~w~n", [I, SV]).
emit_instrs_lm_fs([pc(_PC, try_me_else(ElseLabelStr))|Rest], SV, Ind, FP, LM) :-
    atom_string(ElseLabel, ElseLabelStr),
    split_ite_blocks_lm_fs(Rest, ElseLabel, LM, CondInstrs, ThenInstrs, ElseInstrs, ContInstrs),
    !,
    (   ContInstrs = []
    ->  emit_ite_match_fs(SV, CondInstrs, ThenInstrs, ElseInstrs, Ind, FP)
    ;   % Bind the whole ITE result before emitting continuation code.
        % A previous version appended another `| Some ... ->` arm to the
        % inner condition match, making the continuation arm unreachable.
        format("~wmatch (~n", [Ind]),
        atom_concat(Ind, "    ", IteInd),
        emit_ite_match_fs(SV, CondInstrs, ThenInstrs, ElseInstrs, IteInd, FP),
        format("~w) with~n", [Ind]),
        fresh_sv_fs(SV, SVcont),
        format("~w| Some ~w ->~n", [Ind, SVcont]),
        atom_concat(Ind, "    ", ContInd),
        emit_instrs_lm_fs(ContInstrs, SVcont, ContInd, FP, LM),
        format("~w| None -> None~n", [Ind])
    ).

% execute/1 is a tail call — it must always be the last instruction.
% If it isn’t, the chain silently breaks because emit_one_fs emits a bare
% expression with no ‘| Some sv ->’ arm for callers to continue into.
% This dedicated clause fires before the general catch-all to:
%   (a) emit a visible F# comment warning if Rest ≠ [], so the generated
%       code fails loudly rather than silently producing wrong output, and
%   (b) emit the execute expression and stop (Rest is dropped — the user
%       gets the warning and can fix the upstream WAM generator).
emit_instrs_lm_fs([pc(_PC, fail)|Rest], _SV, Ind, _FP, _LM) :-
    (   Rest \= []
    ->  format("~w// WARNING: fail is not the last instruction — ~w instruction(s) unreachable~n",
               [Ind, Rest])
    ;   true
    ),
    format("~wNone~n", [Ind]).

emit_instrs_lm_fs([pc(PC, execute(PredStr))|Rest], SV, Ind, FP, _LM) :-
    (   Rest \= []
    ->  format("~w// WARNING: execute(~w) is not the last instruction — tail-call semantics violated; ~w instruction(s) unreachable~n",
               [Ind, PredStr, Rest])
    ;   true
    ),
    emit_one_fs(execute(PredStr), PC, SV, _, Ind, FP).

emit_instrs_lm_fs([pc(PC, Instr)|Rest], SV, Ind, FP, LM) :-
    emit_one_fs(Instr, PC, SV, SVout, Ind, FP),
    atom_concat(Ind, "    ", IndInner),
    (   Rest = []
    ->  (   is_match_instr_fs(Instr)
        ->  format("~wSome ~w~n", [IndInner, SVout]),
            format("~w| None -> None~n", [Ind])
        ;   true
        )
    ;   (   is_match_instr_fs(Instr)
        ->  emit_instrs_lm_fs(Rest, SVout, IndInner, FP, LM),
            format("~w| None -> None~n", [Ind])
        ;   emit_instrs_lm_fs(Rest, SVout, Ind, FP, LM)
        )
    ).

%% emit_ite_match_fs(+SV, +CondInstrs, +ThenInstrs, +ElseInstrs, +Ind, +FP)
%  Helper for emit_instrs_lm_fs's try_me_else clause: emits the F# nested
%  match for an if-then-else block.  Placed after the emit_instrs_lm_fs
%  clause group so the latter is contiguous.
emit_ite_match_fs(SV, CondInstrs, ThenInstrs, ElseInstrs, Ind, FP) :-
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

%% emit_instrs_fs(+PCInstrs, +CurrentStateVar, +Indent, +ForeignPreds)
%  Legacy 4-arg version used inside ITE blocks (LabelMap not needed there
%  because ITE blocks are self-contained and never themselves contain
%  nested try_me_else sequences in lowerable predicates).
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
emit_instrs_fs([pc(_PC, fail)|Rest], _SV, Ind, _FP) :-
    (   Rest \= []
    ->  format("~w// WARNING: fail is not the last instruction — ~w instruction(s) unreachable~n",
               [Ind, Rest])
    ;   true
    ),
    format("~wNone~n", [Ind]).

emit_instrs_fs([pc(PC, execute(PredStr))|Rest], SV, Ind, FP) :-
    (   Rest \= []
    ->  format("~w// WARNING: execute(~w) is not the last instruction — tail-call semantics violated; ~w instruction(s) unreachable~n",
               [Ind, PredStr, Rest])
    ;   true
    ),
    emit_one_fs(execute(PredStr), PC, SV, _, Ind, FP).

emit_instrs_fs([pc(PC, Instr)|Rest], SV, Ind, FP) :-
    emit_one_fs(Instr, PC, SV, SVout, Ind, FP),
    atom_concat(Ind, "    ", IndInner),
    (   Rest = []
    ->  % Last instruction.
        %   - match-emitting instructions leave an open "| Some SVout ->" arm;
        %     we must emit "Some SVout" indented one level deeper as its body,
        %     and close the match with "| None -> None" at Ind level so the
        %     resulting F# pattern match is exhaustive (FS0025 closure).
        %   - let-binding instructions (allocate, put_*, get_variable, proceed)
        %     already closed their output, so nothing more is needed.
        (   is_match_instr_fs(Instr)
        ->  format("~wSome ~w~n", [IndInner, SVout]),
            format("~w| None -> None~n", [Ind])
        ;   true
        )
    ;   % Intermediate instruction: continue chain.
        %   match-emitting instructions opened a "| Some SVout ->" arm;
        %   remaining instructions become its body at IndInner.
        %   After the body, close the match with "| None -> None" at Ind level.
        %   let-binding instructions stay at the same indent level.
        (   is_match_instr_fs(Instr)
        ->  emit_instrs_fs(Rest, SVout, IndInner, FP),
            format("~w| None -> None~n", [Ind])
        ;   emit_instrs_fs(Rest, SVout, Ind, FP)
        )
    ).

emit_ite_block_fs([], SV, Ind, _FP) :-
    format("~wSome ~w~n", [Ind, SV]).
emit_ite_block_fs([pc(_PC, fail)|Rest], _SV, Ind, _FP) :-
    (   Rest \= []
    ->  format("~w// WARNING: fail is not the last ITE instruction — ~w instruction(s) unreachable~n",
               [Ind, Rest])
    ;   true
    ),
    format("~wNone~n", [Ind]).
emit_ite_block_fs([pc(PC, execute(PredStr))|Rest], SV, Ind, FP) :-
    (   Rest \= []
    ->  format("~w// WARNING: execute(~w) is not the last ITE instruction — tail-call semantics violated; ~w instruction(s) unreachable~n",
               [Ind, PredStr, Rest])
    ;   true
    ),
    emit_one_fs(execute(PredStr), PC, SV, _, Ind, FP).
emit_ite_block_fs([pc(PC, Instr)], SV, Ind, FP) :-
    emit_one_fs(Instr, PC, SV, SVout, Ind, FP),
    atom_concat(Ind, "    ", IndInner),
    (   is_terminal_instr_fs(Instr) -> true
    ;   is_match_instr_fs(Instr)    ->
        format("~wSome ~w~n", [IndInner, SVout]),
        format("~w| None -> None~n", [Ind])
    ;   format("~wSome ~w~n", [Ind, SVout])
    ).
emit_ite_block_fs([pc(PC, Instr)|Rest], SV, Ind, FP) :-
    Rest \= [],
    emit_one_fs(Instr, PC, SV, SVout, Ind, FP),
    % Mirror emit_instrs_lm_fs: match-emitting instructions open a
    % '| Some SVout ->' arm, so remaining code becomes its body at IndInner.
    % After the body, close the match with '| None -> None' at Ind level so
    % the F# pattern match is exhaustive (FS0025 closure).
    atom_concat(Ind, "    ", IndInner),
    (   is_match_instr_fs(Instr)
    ->  emit_ite_block_fs(Rest, SVout, IndInner, FP),
        format("~w| None -> None~n", [Ind])
    ;   emit_ite_block_fs(Rest, SVout, Ind, FP)
    ).

is_terminal_instr_fs(proceed).
is_terminal_instr_fs(fail).
is_terminal_instr_fs(execute(_)).

% ============================================================================
% Structured ITE emission (shared nesting-aware structurer)
% ============================================================================
%
%  The flat split_ite_blocks_lm_fs/split_at_jump_fs heuristic is NOT nesting
%  aware (split_at_jump_fs stops at an inner jump) and only recognises
%  cut_ite, not the !/0 negation commit. Feed clause 1 through the shared
%  wam_ite_structurer instead: struct_stream_fs/3 rebuilds a label-marked
%  stream (control markers bare so the structurer matches them, labels as
%  strings to match try_me_else/jump args, data instructions keep pc(PC,_)
%  for emit_one_fs), structure_ite folds every block into ite(Cond,Then,Else),
%  and emit_structured_fs/4 walks it — reusing the exact F# match-arm
%  threading (is_match_instr_fs / "| None -> None") so non-ITE predicates
%  emit byte-identically and nested blocks recurse for free.

emit_clause_struct_fs(ClausePCs, LabelMap, SV, Ind, FP) :-
    struct_stream_fs(ClausePCs, LabelMap, Stream),
    structure_ite(Stream, Structured),
    emit_structured_fs(Structured, SV, Ind, FP).

struct_stream_fs([], _LM, []).
struct_stream_fs([pc(PC, Instr)|Rest], LM, Out) :-
    findall(label(LStr), (member(L-PC, LM), atom_string(L, LStr)), Labels),
    struct_item_fs(PC, Instr, Item),
    append(Labels, [Item|More], Out),
    struct_stream_fs(Rest, LM, More).

struct_item_fs(_PC, try_me_else(L), try_me_else(L)) :- !.
struct_item_fs(_PC, jump(L),        jump(L))        :- !.
struct_item_fs(_PC, trust_me,       trust_me)       :- !.
struct_item_fs(_PC, cut_ite,        cut_ite)        :- !.
struct_item_fs(_PC, builtin_call(Op, N), builtin_call(Op, N)) :- neg_commit_op_fs(Op), !.
struct_item_fs(PC, Instr, pc(PC, Instr)).

neg_commit_op_fs("!/0").
neg_commit_op_fs('!/0').

%% emit_structured_fs(+Structured, +SV, +Ind, +FP)
%  Structured is a list of pc(PC,Instr) and ite(Cond,Then,Else). Emits an
%  F# expression of type WamState option.
emit_structured_fs([], SV, I, _FP) :-
    format("~wSome ~w~n", [I, SV]).
% If-then-else with a continuation — bind the result, then continue.
emit_structured_fs([ite(C, T, E)|Rest], SV, Ind, FP) :- Rest \= [], !,
    format("~wmatch (~n", [Ind]),
    atom_concat(Ind, "    ", IteInd),
    emit_ite_match_struct_fs(SV, C, T, E, IteInd, FP),
    format("~w) with~n", [Ind]),
    fresh_sv_fs(SV, SVcont),
    format("~w| Some ~w ->~n", [Ind, SVcont]),
    atom_concat(Ind, "    ", ContInd),
    emit_structured_fs(Rest, SVcont, ContInd, FP),
    format("~w| None -> None~n", [Ind]).
% If-then-else as the final expression.
emit_structured_fs([ite(C, T, E)], SV, Ind, FP) :- !,
    emit_ite_match_struct_fs(SV, C, T, E, Ind, FP).
% Terminal: fail
emit_structured_fs([pc(_PC, fail)|Rest], _SV, Ind, _FP) :- !,
    (   Rest \= []
    ->  format("~w// WARNING: fail is not the last instruction — ~w instruction(s) unreachable~n", [Ind, Rest])
    ;   true
    ),
    format("~wNone~n", [Ind]).
% Terminal: execute (tail call)
emit_structured_fs([pc(PC, execute(PredStr))|Rest], SV, Ind, FP) :- !,
    (   Rest \= []
    ->  format("~w// WARNING: execute(~w) is not the last instruction — tail-call semantics violated; ~w instruction(s) unreachable~n",
               [Ind, PredStr, Rest])
    ;   true
    ),
    emit_one_fs(execute(PredStr), PC, SV, _, Ind, FP).
% Last plain instruction.
emit_structured_fs([pc(PC, Instr)], SV, Ind, FP) :- !,
    emit_one_fs(Instr, PC, SV, SVout, Ind, FP),
    atom_concat(Ind, "    ", IndInner),
    (   is_terminal_instr_fs(Instr) -> true
    ;   is_match_instr_fs(Instr)
    ->  format("~wSome ~w~n", [IndInner, SVout]),
        format("~w| None -> None~n", [Ind])
    ;   format("~wSome ~w~n", [Ind, SVout])
    ).
% Plain instruction with more following.
emit_structured_fs([pc(PC, Instr)|Rest], SV, Ind, FP) :- Rest \= [], !,
    emit_one_fs(Instr, PC, SV, SVout, Ind, FP),
    atom_concat(Ind, "    ", IndInner),
    (   is_match_instr_fs(Instr)
    ->  emit_structured_fs(Rest, SVout, IndInner, FP),
        format("~w| None -> None~n", [Ind])
    ;   emit_structured_fs(Rest, SVout, Ind, FP)
    ).

%% emit_ite_match_struct_fs(+SV, +Cond, +Then, +Else, +Ind, +FP)
%  F# nested match for one ite/3 block; branches recurse through
%  emit_structured_fs so nested blocks are handled.
emit_ite_match_struct_fs(SV, CondInstrs, ThenInstrs, ElseInstrs, Ind, FP) :-
    format("~wmatch (~n", [Ind]),
    atom_concat(Ind, "    ", CondInd),
    emit_structured_fs(CondInstrs, SV, CondInd, FP),
    format("~w) with~n", [Ind]),
    fresh_sv_fs(SV, SVthen),
    format("~w| Some ~w ->~n", [Ind, SVthen]),
    atom_concat(Ind, "    ", ThenInd),
    emit_structured_fs(ThenInstrs, SVthen, ThenInd, FP),
    format("~w| None ->~n", [Ind]),
    atom_concat(Ind, "    ", ElseInd),
    emit_structured_fs(ElseInstrs, SV, ElseInd, FP).

%% split_ite_blocks_fs/6 — legacy no-LabelMap version
%  Used inside emit_instrs_fs (4-arg). ContInstrs is always [] here
%  because ITE blocks inside ITE blocks are rare in lowerable predicates.
split_ite_blocks_fs(Instrs, _ElseLabel, CondInstrs, ThenInstrs, ElseInstrs, ContInstrs) :-
    split_at_instr_fs(Instrs, cut_ite, CondInstrs, AfterCut),
    split_at_jump_fs(AfterCut, ThenInstrs, _ContLabelStr, AfterJump),
    AfterJump = [pc(_, trust_me)|ElseInstrs],
    ContInstrs = [].

%% split_ite_blocks_lm_fs/7 — LabelMap-aware version
%  Used inside emit_instrs_lm_fs so that ContInstrs (code after the ITE
%  block's continuation jump target) is correctly split off.
split_ite_blocks_lm_fs(Instrs, _ElseLabel, LabelMap,
                        CondInstrs, ThenInstrs, ElseInstrs, ContInstrs) :-
    split_at_instr_fs(Instrs, cut_ite, CondInstrs, AfterCut),
    split_at_jump_fs(AfterCut, ThenInstrs, ContLabelStr, AfterJump),
    AfterJump = [pc(_, trust_me)|ElseAndCont],
    atom_string(ContLabel, ContLabelStr),
    split_else_cont_fs(ElseAndCont, ContLabel, LabelMap, ElseInstrs, ContInstrs).

split_at_instr_fs([], _, _, _) :- !, fail.
split_at_instr_fs([pc(_, Instr)|Rest], Instr, [], Rest) :- !.
split_at_instr_fs([H|T], Instr, [H|Before], After) :-
    split_at_instr_fs(T, Instr, Before, After).

split_at_jump_fs([], [], "", []) :- !, fail.
split_at_jump_fs([pc(_, jump(Label))|Rest], [], Label, Rest) :- !.
split_at_jump_fs([H|T], [H|Then], Label, Rest) :-
    split_at_jump_fs(T, Then, Label, Rest).

%% split_else_cont_fs(+ElseAndCont, +ContLabel, +LabelMap, -ElseInstrs, -ContInstrs)
%  Split the instruction sequence following trust_me into:
%    ElseInstrs — the else branch body (up to the continuation target PC)
%    ContInstrs — instructions at and after the continuation target PC
%
%  ContLabel is the atom label that the jump at the end of the then-branch
%  targets.  We use LabelMap to resolve it to a PC, then split there.
%
%  Bug fix: the old stub put all instructions in ElseInstrs and left
%  ContInstrs=[], silently dropping any continuation code.
%% split_else_cont_fs(+ElseAndCont, +ContLabel, +LabelMap, -Else, -Cont)
%  Entry point: warn if ContLabel is absent from LabelMap (defensive hardening
%  for label typos or parsing gaps), then delegate to the worker predicate.
%  Without this guard, a missing label causes member/2 to silently fail on
%  every instruction and all code ends up in ElseInstrs — the same silent
%  data-loss as the old no-op stub, just harder to notice.
split_else_cont_fs(Instrs, ContLabel, LM, Else, Cont) :-
    (   \+ member(ContLabel-_, LM)
    ->  format(user_error,
               'WARNING: split_else_cont_fs — ContLabel ~q not in LabelMap ~w; continuation code will be empty~n',
               [ContLabel, LM])
    ;   true
    ),
    split_else_cont_fs_(Instrs, ContLabel, LM, Else, Cont).

split_else_cont_fs_([], _ContLabel, _LM, [], []).
split_else_cont_fs_([pc(PC,Instr)|Rest], ContLabel, LM, Else, Cont) :-
    (   member(ContLabel-ContPC, LM),
        PC >= ContPC
    ->  % Reached or passed the continuation target — everything from here
        %  is continuation code, not else-branch code.
        Else = [],
        Cont = [pc(PC,Instr)|Rest]
    ;   Else = [pc(PC,Instr)|ElseRest],
        split_else_cont_fs_(Rest, ContLabel, LM, ElseRest, Cont)
    ).

% ============================================================================
% emit_one_fs — single instruction → F# binding line
% ============================================================================

% Terminal: proceed
emit_one_fs(proceed, _, SV, SV, I, _FP) :-
    format("~wlet ret_ = ~w.WsCP~n", [I, SV]),
    format("~wif ret_ = 0 then Some { ~w with WsPC = 0 } else Some { ~w with WsPC = ret_; WsCP = 0 }~n",
           [I, SV, SV]).

% Terminal: fail
emit_one_fs(fail, _, _SV, _SVout, I, _FP) :-
    format("~wNone~n", [I]).

% Allocate — always succeeds, inline
emit_one_fs(allocate, _, SV, SVout, I, _FP) :-
    fresh_sv_fs(SV, SVout),
    format("~wlet ~w = { ~w with~n", [I, SVout, SV]),
    format("~w               WsStack = { EfSavedCP = ~w.WsCP; EfYRegs = Map.empty; EfSavedCutBar = ~w.WsCutBar } :: ~w.WsStack~n", [I, SV, SV, SV]),
    format("~w               WsCutBar = ~w.WsCPsLen }~n", [I, SV]).

% Deallocate — inline pop of env frame.  Empty stack is a hard programming
% error (compiler bug, not a runtime failure) so failwith rather than None.
% Switched from match-emitting to let-binding: removed deallocate from
% is_match_instr_fs/1 below.
emit_one_fs(deallocate, _PC, SV, SVout, I, _FP) :-
    fresh_sv_fs(SV, SVout),
    format("~wlet ~w =~n", [I, SVout]),
    format("~w    match ~w.WsStack with~n", [I, SV]),
    format("~w    | ef :: rest -> { ~w with WsStack = rest; WsCP = ef.EfSavedCP; WsCutBar = ef.EfSavedCutBar }~n", [I, SV]),
    format("~w    | [] -> failwith \"Deallocate: empty WsStack\"~n", [I]).

% GetVariable Xn Ai — always succeeds, inline register copy
emit_one_fs(get_variable(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    % Bug fix: Atom "" was a silent wrong default; use failwith so the runtime
    % surfaces a register-not-bound error immediately rather than producing
    % a spurious Atom "" binding that propagates silently.
    format("~wlet ~w = putReg ~w (derefVar ~w.WsBindings (getReg ~w ~w |> Option.defaultWith (fun _ -> failwith \"GetVariable: source register not bound\"))) ~w~n",
           [I, SVout, Xn, SV, Ai, SV, SV]).

% GetConstant C Ai — can fail, delegate to step
% GetConstant C Ai — inline: deref reg, succeed on match (with []/VList []
% equivalence) or bind-when-Unbound, else fail.  Same logic as the
% interpreter's step branch.
emit_one_fs(get_constant(CStr, AiStr), _PC, SV, SVout, I, _FP) :-
    val_fs(CStr, FC), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wmatch (match getReg ~w ~w with~n", [I, Ai, SV]),
    format("~w       | Some v when v = (~w) -> Some ~w~n", [I, FC, SV]),
    format("~w       | Some (VList []) when (~w) = Atom \"[]\" -> Some ~w~n", [I, FC, SV]),
    format("~w       | Some (Unbound vid) ->~n", [I]),
    format("~w           let r = Array.copy ~w.WsRegs~n", [I, SV]),
    format("~w           r.[~w] <- (~w)~n", [I, Ai, FC]),
    format("~w           Some { ~w with~n", [I, SV]),
    format("~w                    WsRegs = r~n", [I]),
    format("~w                    WsBindings = Map.add vid (~w) ~w.WsBindings~n", [I, FC, SV]),
    format("~w                    WsTrail = { TrailVarId = vid; TrailOldVal = Map.tryFind vid ~w.WsBindings } :: ~w.WsTrail~n",
           [I, SV, SV]),
    format("~w                    WsTrailLen = ~w.WsTrailLen + 1 }~n", [I, SV]),
    format("~w       | _ -> None) with~n", [I]),
    format("~w| Some ~w ->~n", [I, SVout]).

% GetValue Xn Ai — inline: deref both regs, equal -> succeed, Unbound ai ->
% bind to xn's value, else fail.
emit_one_fs(get_value(XnStr, AiStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wmatch (match getReg ~w ~w, getReg ~w ~w with~n", [I, Ai, SV, Xn, SV]),
    format("~w       | Some a, Some x when a = x -> Some ~w~n", [I, SV]),
    format("~w       | Some (Unbound vid), Some x ->~n", [I]),
    format("~w           let r = Array.copy ~w.WsRegs~n", [I, SV]),
    format("~w           r.[~w] <- x~n", [I, Ai]),
    format("~w           Some { ~w with~n", [I, SV]),
    format("~w                    WsRegs = r~n", [I]),
    format("~w                    WsBindings = Map.add vid x ~w.WsBindings~n", [I, SV]),
    format("~w                    WsTrail = { TrailVarId = vid; TrailOldVal = Map.tryFind vid ~w.WsBindings } :: ~w.WsTrail~n",
           [I, SV, SV]),
    format("~w                    WsTrailLen = ~w.WsTrailLen + 1 }~n", [I, SV]),
    format("~w       | _ -> None) with~n", [I]),
    format("~w| Some ~w ->~n", [I, SVout]).

% GetStructure F Ai — inline read-mode / write-mode dispatch.  85
% occurrences in the parser smoke (the biggest single source of step
% delegations after Unify*), so worth inlining.  Mirrors step's
% GetStructure cases:
%   (1) reg holds Str matching fn/arity -> read mode (ReadArgs)
%   (2) reg holds VList cons cell AND fn = "[|]"/2 -> read mode (cons-as-list)
%   (3) reg holds Unbound, arity = 0 -> bind to Str(fn, [])
%   (4) reg holds Unbound, arity > 0 -> write mode (BuildStruct)
%   (5) otherwise -> fail (None)
% Cons-cell branch (2) is only emitted when the static fn = "[|]" and
% arity = 2; otherwise it'd be unreachable dead code (compiler warning).
emit_one_fs(get_structure(FStr, AiStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(AiStr, Ai),
    parse_functor_fs(FStr, FuncName, Arity),
    escape_dq_fs(FuncName, EscFuncName),
    fresh_sv_fs(SV, SVout),
    format("~wmatch (match getReg ~w ~w with~n", [I, Ai, SV]),
    %% (1) Same-functor Str match
    format("~w       | Some (Str (fn0, args)) when fn0 = \"~w\" && List.length args = ~w ->~n",
           [I, EscFuncName, Arity]),
    (   Arity =:= 0
    ->  format("~w           Some ~w~n", [I, SV])
    ;   format("~w           let push = pushBuilderIfActive ~w~n", [I, SV]),
        format("~w           Some { push with WsBuilder = Some (ReadArgs args) }~n", [I])
    ),
    %% (2) Cons-cell match (only when fn = "[|]" and arity = 2)
    (   FuncName == '[|]', Arity =:= 2
    ->  format("~w       | Some (VList (h :: t)) ->~n", [I]),
        format("~w           let tailVal = if List.isEmpty t then Atom \"[]\" else VList t~n", [I]),
        format("~w           let push = pushBuilderIfActive ~w~n", [I, SV]),
        format("~w           Some { push with WsBuilder = Some (ReadArgs [h; tailVal]) }~n", [I])
    ;   true
    ),
    %% (3) Unbound, arity = 0: bind to Str(fn, [])
    (   Arity =:= 0
    ->  format("~w       | Some (Unbound vid) ->~n", [I]),
        format("~w           let str = Str (\"~w\", [])~n", [I, EscFuncName]),
        format("~w           let s0 = putReg ~w str ~w~n", [I, Ai, SV]),
        format("~w           Some { s0 with~n", [I]),
        format("~w                    WsBindings = Map.add vid str ~w.WsBindings~n", [I, SV]),
        format("~w                    WsTrail = { TrailVarId = vid; TrailOldVal = Map.tryFind vid ~w.WsBindings } :: ~w.WsTrail~n",
               [I, SV, SV]),
        format("~w                    WsTrailLen = ~w.WsTrailLen + 1 }~n", [I, SV])
    %% (4) Unbound, arity > 0: write mode (BuildStruct)
    ;   format("~w       | Some (Unbound _) ->~n", [I]),
        format("~w           let push = pushBuilderIfActive ~w~n", [I, SV]),
        format("~w           Some { push with WsBuilder = Some (BuildStruct (\"~w\", ~w, ~w, [])) }~n",
               [I, EscFuncName, Ai, Arity])
    ),
    %% (5) Default fail
    format("~w       | _ -> None) with~n", [I]),
    format("~w| Some ~w ->~n", [I, SVout]).

emit_one_fs(get_structure(FnStr, ArityStr, AiStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(AiStr, Ai),
    (   number_string(Arity, ArityStr)
    ->  true
    ;   throw(error(domain_error(get_structure_arity, ArityStr), emit_one_fs/6))
    ),
    escape_dq_fs(FnStr, EscFnStr),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (GetStructure (\"~w\", ~w, ~w)) with~n",
           [I, SV, PC, EscFnStr, Arity, Ai]),
    format("~w| Some ~w ->~n", [I, SVout]).

% GetList Ai — inline 3-case dispatch (VList cons, Str "[|]" cons, Unbound).
% Read-mode for the first two (sets ReadArgs builder), write-mode for the
% Unbound case (sets BuildList builder).  pushBuilderIfActive preserves any
% outer build context the same way step does.
emit_one_fs(get_list(AiStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wmatch (let push = pushBuilderIfActive ~w in~n", [I, SV]),
    format("~w       match getReg ~w ~w with~n", [I, Ai, SV]),
    format("~w       | Some (VList (h :: t)) ->~n", [I]),
    format("~w           let tailVal = if List.isEmpty t then Atom \"[]\" else VList t~n", [I]),
    format("~w           Some { push with WsBuilder = Some (ReadArgs [h; tailVal]) }~n", [I]),
    format("~w       | Some (Str (\"[|]\", [h; t])) ->~n", [I]),
    format("~w           Some { push with WsBuilder = Some (ReadArgs [h; t]) }~n", [I]),
    format("~w       | Some (Unbound _) ->~n", [I]),
    format("~w           Some { push with WsBuilder = Some (BuildList (~w, [])) }~n", [I, Ai]),
    format("~w       | _ -> None) with~n", [I]),
    format("~w| Some ~w ->~n", [I, SVout]).

% GetNil / GetInteger are Rust-lowered aliases for GetConstant.
emit_one_fs(get_nil(AiStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (GetConstant (Atom \"[]\", ~w)) with~n", [I, SV, PC, Ai]),
    format("~w| Some ~w ->~n", [I, SVout]).

emit_one_fs(get_integer(NStr, AiStr), PC, SV, SVout, I, _FP) :-
    (   number_string(N, NStr), integer(N)
    ->  true
    ;   throw(error(domain_error(get_integer_value, NStr), emit_one_fs/6))
    ),
    reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (GetConstant (Integer ~w, ~w)) with~n", [I, SV, PC, N, Ai]),
    format("~w| Some ~w ->~n", [I, SVout]).

% Unify* — can fail, delegate to step
% UnifyVariable Xn — inline read-mode + write-mode dispatch.  Far hotter
% than the other Unify* in real workloads (87 occurrences across the parser
% smoke alone) so worth bypassing step's dispatch + WsPC record-with alloc.
% Read mode: copy next arg into Xn (always succeeds).
% Write mode (no readable arg): create fresh var, store in Xn, append to
% the active builder (BuildList / BuildStruct), which may fail if no
% builder is active.
emit_one_fs(unify_variable(XnStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn),
    fresh_sv_fs(SV, SVout),
    format("~wmatch (match readNextArg ~w with~n", [I, SV]),
    format("~w       | Some (v, sR) -> Some (putReg ~w v sR)~n", [I, Xn]),
    format("~w       | None ->~n", [I]),
    format("~w           let vid = ~w.WsVarCounter~n", [I, SV]),
    format("~w           let var = Unbound vid~n", [I]),
    format("~w           let sW = putReg ~w var { ~w with WsVarCounter = ~w.WsVarCounter + 1 }~n",
           [I, Xn, SV, SV]),
    format("~w           addToBuilder var sW) with~n", [I]),
    format("~w| Some ~w ->~n", [I, SVout]).

emit_one_fs(unify_value(XnStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (UnifyValue ~w) with~n", [I, SV, PC, Xn]),
    format("~w| Some ~w ->~n", [I, SVout]).

% UnifyConstant C — inline read-mode + write-mode dispatch.  By far the most
% frequent step-delegating instruction in compiled parser code (132 of them
% in the parser smoke alone), so the biggest win for inlining.  Read mode:
% unify constant with the next structure arg.  Write mode: append constant
% to the active builder.  Both can fail.
emit_one_fs(unify_constant(CStr), _PC, SV, SVout, I, _FP) :-
    val_fs(CStr, FC),
    fresh_sv_fs(SV, SVout),
    format("~wmatch (match readNextArg ~w with~n", [I, SV]),
    format("~w       | Some (v, sR) -> unifyVal v (~w) sR~n", [I, FC]),
    format("~w       | None -> addToBuilder (~w) ~w) with~n", [I, FC, SV]),
    format("~w| Some ~w ->~n", [I, SVout]).

% PutValue Xn Ai — always succeeds, inline
emit_one_fs(put_value(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wlet ~w = putReg ~w (getReg ~w ~w |> Option.defaultWith (fun _ -> failwith \"PutValue: source register not bound\")) ~w~n",
           [I, SVout, Ai, Xn, SV, SV]).

% PutVariable Xn Ai — always succeeds, inline (creates fresh Unbound)
emit_one_fs(put_variable(XnStr, AiStr), _, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wlet vid_~w = ~w.WsVarCounter~n", [I, SVout, SV]),
    format("~wlet var_~w = Unbound vid_~w~n", [I, SVout, SVout]),
    format("~wlet ~w = putReg ~w var_~w (putReg ~w var_~w { ~w with WsVarCounter = ~w.WsVarCounter + 1 })~n",
           [I, SVout, Xn, SVout, Ai, SVout, SV, SV]).

% PutConstant C Ai — always succeeds, inline
emit_one_fs(put_constant(CStr, AiStr), _, SV, SVout, I, _FP) :-
    val_fs(CStr, FC), reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wlet ~w = putReg ~w (~w) ~w~n",
           [I, SVout, Ai, FC, SV]).

% PutStructure, PutList, SetVariable, SetValue, SetConstant — delegate to step
% PutStructure / PutList — always-succeed let-bindings.  Both start a fresh
% build context; pushBuilderIfActive preserves any outer one.  Removed from
% is_match_instr_fs/1 below since they no longer emit match arms.
emit_one_fs(put_structure(FnStr, AiStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(AiStr, Ai),
    parse_functor_fs(FnStr, FuncName, Arity),
    escape_dq_fs(FuncName, EscFuncName),
    fresh_sv_fs(SV, SVout),
    format("~wlet push_~w = pushBuilderIfActive ~w~n", [I, SVout, SV]),
    format("~wlet ~w = { push_~w with WsBuilder = Some (BuildStruct (\"~w\", ~w, ~w, [])) }~n",
           [I, SVout, SVout, EscFuncName, Ai, Arity]).

emit_one_fs(put_list(AiStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(AiStr, Ai),
    fresh_sv_fs(SV, SVout),
    format("~wlet push_~w = pushBuilderIfActive ~w~n", [I, SVout, SV]),
    format("~wlet ~w = { push_~w with WsBuilder = Some (BuildList (~w, [])) }~n",
           [I, SVout, SVout, Ai]).

% SetVariable Xn — inline: create fresh var, store in Xn, append to active
% builder.  Always reaches addToBuilder (no None short-circuit), but
% addToBuilder itself can return None when no builder is active, so the
% emit stays match-emitting.
emit_one_fs(set_variable(XnStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn),
    fresh_sv_fs(SV, SVout),
    format("~wmatch (let vid = ~w.WsVarCounter in~n", [I, SV]),
    format("~w       let var = Unbound vid in~n", [I]),
    format("~w       let sV = putReg ~w var { ~w with WsVarCounter = ~w.WsVarCounter + 1 } in~n",
           [I, Xn, SV, SV]),
    format("~w       addToBuilder var sV) with~n", [I]),
    format("~w| Some ~w ->~n", [I, SVout]).

% SetValue Xn — inline: read Xn (failwith on unbound), append to builder.
emit_one_fs(set_value(XnStr), _PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(XnStr, Xn),
    fresh_sv_fs(SV, SVout),
    format("~wmatch (match getReg ~w ~w with~n", [I, Xn, SV]),
    format("~w       | Some v -> addToBuilder v ~w~n", [I, SV]),
    format("~w       | None   -> None) with~n", [I]),
    format("~w| Some ~w ->~n", [I, SVout]).

% SetConstant C — inline: append constant to builder.
emit_one_fs(set_constant(CStr), _PC, SV, SVout, I, _FP) :-
    val_fs(CStr, FC),
    fresh_sv_fs(SV, SVout),
    format("~wmatch addToBuilder (~w) ~w with~n", [I, FC, SV]),
    format("~w| Some ~w ->~n", [I, SVout]).

% Call — use callForeign for known foreign preds, dispatchCall otherwise
emit_one_fs(call(PredStr, _NStr), PC, SV, SVout, I, FP) :-
    RetPC is PC + 1,
    fresh_sv_fs(SV, SVout),
    atom_string(PredAtom, PredStr),
    escape_dq_fs(PredStr, EscPred),
    (   member(PredAtom, FP)
    ->  format("~wmatch callForeign ctx \"~w\" { ~w with WsCP = ~w } with~n",
               [I, EscPred, SV, RetPC])
    ;   format("~wmatch dispatchCall ctx \"~w\" { ~w with WsCP = ~w } with~n",
               [I, EscPred, SV, RetPC])
    ),
    format("~w| Some ~w ->~n", [I, SVout]).


% CallForeign — explicit foreign call opcode (delegate to step)
emit_one_fs(call_foreign(PredStr, NStr), PC, SV, SVout, I, _FP) :-
    (   number_string(N, NStr)
    ->  true
    ;   throw(error(domain_error(call_foreign_arity, NStr), emit_one_fs/6))
    ),
    fresh_sv_fs(SV, SVout),
    escape_dq_fs(PredStr, EscPred),
    format("~wmatch step ctx { ~w with WsPC = ~w } (CallForeign (\"~w\", ~w)) with~n",
           [I, SV, PC, EscPred, N]),
    format("~w| Some ~w ->~n", [I, SVout]).


% BuiltinCall !/0 (cut) — inline always-succeed: drop CPs above WsCutBar.
% Common enough (every clause with `:- !` body) and step's branch is
% trivial, so worth bypassing the step dispatch + WsPC record-with allocation.
emit_one_fs(builtin_call("!/0", _NStr), _PC, SV, SVout, I, _FP) :- !,
    fresh_sv_fs(SV, SVout),
    format("~wlet drop_~w = max 0 (~w.WsCPsLen - ~w.WsCutBar)~n", [I, SVout, SV, SV]),
    format("~wlet ~w = { ~w with WsCPs = List.skip drop_~w ~w.WsCPs; WsCPsLen = ~w.WsCutBar }~n",
           [I, SVout, SV, SVout, SV, SV]).

% BuiltinCall (general) — delegate to step
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
    escape_dq_fs(LabelStr, EscLabel),
    format("~wmatch step ctx { ~w with WsPC = ~w } (Jump \"~w\") with~n", [I, SV, PC, EscLabel]),
    format("~w| Some ~w ->~n", [I, SVout]).


emit_one_fs(retry_me_else(LabelStr), PC, SV, SVout, I, _FP) :-
    fresh_sv_fs(SV, SVout),
    escape_dq_fs(LabelStr, EscLabel),
    format("~wmatch step ctx { ~w with WsPC = ~w } (RetryMeElse \"~w\") with~n", [I, SV, PC, EscLabel]),
    format("~w| Some ~w ->~n", [I, SVout]).

% TrustMe — delegate to step
emit_one_fs(trust_me, PC, SV, SVout, I, _FP) :-
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } TrustMe with~n", [I, SV, PC]),
    format("~w| Some ~w ->~n", [I, SVout]).

% Execute — tail call; returns directly (no WsCP change)
emit_one_fs(execute(PredStr), PC, SV, SV, I, FP) :-
    atom_string(PredAtom, PredStr),
    escape_dq_fs(PredStr, EscPred),
    (   member(PredAtom, FP)
    ->  format("~wstep ctx { ~w with WsPC = ~w } (ExecuteForeign \"~w\")~n",
               [I, SV, PC, EscPred])
    ;   format("~wdispatchCall ctx \"~w\" ~w~n", [I, EscPred, SV])
    ).

% ============================================================================
% Phase I — Haskell-only specialized instructions: delegate to step.
%
% Each clause emits exactly the same shape as the BuiltinCall delegation
% above: open a `match step ctx { s with WsPC = PC } (Constructor ...) with`
% block with a `| Some SVout ->` arm that the next instruction continues
% into.  The matching is_match_instr_fs/1 entries above make sure the
% emit_instrs_lm_fs chain places the continuation at the right indent.
% ============================================================================

% PutStructureDyn — runtime-parsed functor (name+arity from registers).
emit_one_fs(put_structure_dyn(NameRegStr, ArityRegStr, TargetRegStr),
            PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(NameRegStr, NameReg),
    reg_to_int_fs(ArityRegStr, ArityReg),
    reg_to_int_fs(TargetRegStr, TargetReg),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (PutStructureDyn (~w, ~w, ~w)) with~n",
           [I, SV, PC, NameReg, ArityReg, TargetReg]),
    format("~w| Some ~w ->~n", [I, SVout]).

% Arg — specialized arg/3 with compile-time N.
emit_one_fs(arg(NStr, TRegStr, ARegStr), PC, SV, SVout, I, _FP) :-
    (   number_string(N, NStr) -> true
    ;   atom_number(NStr, N) -> true
    ;   throw(error(domain_error(arg_specialization_n, NStr), emit_one_fs/6))
    ),
    reg_to_int_fs(TRegStr, TReg),
    reg_to_int_fs(ARegStr, AReg),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (Arg (~w, ~w, ~w)) with~n",
           [I, SV, PC, N, TReg, AReg]),
    format("~w| Some ~w ->~n", [I, SVout]).

% NotMemberList — \\+ member(X, L) on a bound VList L.
emit_one_fs(not_member_list(XRegStr, LRegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(XRegStr, XReg),
    reg_to_int_fs(LRegStr, LReg),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (NotMemberList (~w, ~w)) with~n",
           [I, SV, PC, XReg, LReg]),
    format("~w| Some ~w ->~n", [I, SVout]).

% NotMemberConstAtoms — variable-arity: not_member_const_atoms(XReg, A1, ..., An).
emit_one_fs(NotMemConstAtoms, PC, SV, SVout, I, _FP) :-
    compound(NotMemConstAtoms),
    functor(NotMemConstAtoms, not_member_const_atoms, N),
    N >= 2,
    arg(1, NotMemConstAtoms, XRegStr),
    reg_to_int_fs(XRegStr, XReg),
    findall(AtomTok,
            (between(2, N, K), arg(K, NotMemConstAtoms, AtomTok)),
            AtomTokens),
    maplist([Tok, Quoted]>>(
        escape_dq_fs(Tok, EscTok),
        format(atom(Quoted), '"~w"', [EscTok])
    ), AtomTokens, QuotedAtoms),
    atomic_list_concat(QuotedAtoms, '; ', AtomsList),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (NotMemberConstAtoms (~w, [~w])) with~n",
           [I, SV, PC, XReg, AtomsList]),
    format("~w| Some ~w ->~n", [I, SVout]).

% BuildEmptySet — write VSet Set.empty into the target register.
emit_one_fs(build_empty_set(RegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(RegStr, Reg),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (BuildEmptySet ~w) with~n",
           [I, SV, PC, Reg]),
    format("~w| Some ~w ->~n", [I, SVout]).

% SetInsert — Atom + VSet -> VSet (with Set.add).
emit_one_fs(set_insert(ERegStr, InRegStr, OutRegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(ERegStr, EReg),
    reg_to_int_fs(InRegStr, InReg),
    reg_to_int_fs(OutRegStr, OutReg),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (SetInsert (~w, ~w, ~w)) with~n",
           [I, SV, PC, EReg, InReg, OutReg]),
    format("~w| Some ~w ->~n", [I, SVout]).

% NotMemberSet — O(log N) visited-set membership check.
emit_one_fs(not_member_set(ERegStr, SRegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(ERegStr, EReg),
    reg_to_int_fs(SRegStr, SReg),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (NotMemberSet (~w, ~w)) with~n",
           [I, SV, PC, EReg, SReg]),
    format("~w| Some ~w ->~n", [I, SVout]).

% BeginAggregate — delegate to step
emit_one_fs(begin_aggregate(TypeStr, ValRegStr, ResRegStr), PC, SV, SVout, I, _FP) :-
    reg_to_int_fs(ValRegStr, ValReg),
    reg_to_int_fs(ResRegStr, ResReg),
    escape_dq_fs(TypeStr, EscType),
    fresh_sv_fs(SV, SVout),
    format("~wmatch step ctx { ~w with WsPC = ~w } (BeginAggregate (\"~w\", ~w, ~w)) with~n",
           [I, SV, PC, EscType, ValReg, ResReg]),
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
%  The s_init (or any non-numeric suffix) case always maps to s_0 as the
%  first step.  Once we are in the s_N series, we simply increment N.
%  Bug fix: the old code collapsed any failed number_string/2 to N1=0,
%  which re-emitted s_0 for names like s_init → s_0 → s_0 (shadowed).
fresh_sv_fs(Cur, Next) :-
    atom_string(Cur, CStr),
    (   sub_string(CStr, 0, 2, _, "s_"),
        sub_string(CStr, 2, _, 0, NumPart),
        number_string(N, NumPart)
    ->  N1 is N + 1,
        format(atom(Next), 's_~w', [N1])
    ;   Next = 's_0'
    ).

%% val_fs(+Str, -FSharpExpr)
%  Convert a WAM value token to its F# Value constructor.
%
%  Quote handling matches wam_fsharp_target:fs_wam_value/2: a token
%  with outer single quotes (e.g. `'42'`, `'+'`) is *always* an atom,
%  even if the inner content looks like a number.  Without this the
%  lowered emitter rendered `read_term_from_atom('42', _T)` as
%  Atom "'42'" -- a 4-char atom -- so the runtime parser tokenized
%  the quotes as syntax errors and every literal-atom parser test
%  failed.
val_fs(Str, FS) :-
    val_fs_strip_quotes(Str, Inner, ForceAtom),
    (   ForceAtom == true
    ->  escape_dq_fs(Inner, EscStr),
        format(atom(FS), 'Atom "~w"', [EscStr])
    ;   number_string(N, Inner), integer(N)
    ->  format(atom(FS), 'Integer ~w', [N])
    ;   number_string(F, Inner), float(F)
    ->  format(atom(FS), 'Float ~w', [F])
    ;   escape_dq_fs(Inner, EscStr),
        format(atom(FS), 'Atom "~w"', [EscStr])
    ).

%% val_fs_strip_quotes(+Raw, -Inner, -ForceAtom)
%  Strip a single pair of outer single quotes if present; ForceAtom
%  is true iff the quotes were present (so the caller knows to skip
%  the number-parse fallback).  Mirrors
%  wam_fsharp_target:fs_strip_quoted_atom/3 but keeps the lowered
%  emitter self-contained.
val_fs_strip_quotes(S0, S, ForceAtom) :-
    string_chars(S0, Chars0),
    (   Chars0 = [''''|Rest], append(Inner, [''''], Rest)
    ->  string_chars(S, Inner),
        ForceAtom = true
    ;   S = S0,
        ForceAtom = false
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
%  Split "name/N" on the *last* "/". Functors that contain "/" themselves
%  — integer-div "//" (WAM text `///2`) and float-div "/" (`//2`) — must
%  not soft-cut on the first slash: the previous `(sub_atom(...'/') -> …)`
%  committed to Before=0, then `atom_number('//2', _)` failed and the
%  whole parse failed. That made emit_one_fs(put_structure("///2",…))
%  fail mid-body for cbi_arith under emit_mode(functions), so
%  lower_all_fs never finished (FS-FUNCTIONS-BUILTINS-LOWER). Mirrors
%  Scala/R/Lua last_slash_index / parse_functor_arity.
parse_functor_fs(FnStr, Name, Arity) :-
    atom_string(FA, FnStr),
    (   last_slash_index_fs(FA, B)
    ->  sub_atom(FA, 0, B, _, Name),
        B1 is B + 1,
        sub_atom(FA, B1, _, 0, AS),
        atom_number(AS, Arity)
    ;   Name = FA, Arity = 0
    ).

%% last_slash_index_fs(+Atom, -Index)
%  Index of the last "/" in Atom, or fails if none.
last_slash_index_fs(Atom, Index) :-
    findall(B, sub_atom(Atom, B, 1, _, '/'), Bs),
    Bs \= [],
    last(Bs, Index).

%% escape_dq_fs(+Str, -Escaped)
%  Escape backslashes and double quotes for F# string literals.
%  WAM tokens normally arrive as strings, but parsed functor names are atoms;
%  normalize first so all string-emitting call sites can share this helper.
escape_dq_fs(Str0, Esc) :-
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
