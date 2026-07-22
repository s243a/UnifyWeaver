:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_r_lowered_emitter.pl -- WAM-lowered R emission (Phase 4)
%
% Plan mirrors the Haskell hybrid path in
%   docs/design/WAM_HASKELL_LOWERED_{PHILOSOPHY,SPECIFICATION,
%                                     IMPLEMENTATION_PLAN}.md
%
% Phase 2 covered deterministic single-clause predicates and emitted
% one R function per predicate, delegating every instruction to
% WamRuntime$step. Phase 3 added two refinements:
%
%   1. Expanded lowerability. Multi-clause predicates whose first
%      clause is in the supported set became lowerable with reason
%      multi_clause_1.
%
%   2. Native per-instruction emission for the simple register ops
%      (allocate/deallocate, put_constant/put_variable/put_value,
%      get_variable). These never fail and never allocate heap, so we
%      avoid the WamRuntime$step dispatch entirely. Anything that
%      involves deref/unify/heap-build (get_constant, get_value,
%      get_structure, *_list, unify_*, set_*, builtin_call,
%      call_foreign) still delegates to step.
%
% Phase 4 broadens multi-clause lowering to multi_clause_n: when every
% clause is supported, each clause is emitted inline and later clauses
% are exposed through an iter-style retry CP.
%
% Output under emit_mode(interpreter) is unchanged; this file is only
% consulted under emit_mode(functions) / emit_mode(mixed([...])).

:- module(wam_r_lowered_emitter, [
    wam_r_lowerable/3,           % +Pred, +WamCode, -Reason
    lower_predicate_to_r/4,      % +Pred, +WamCode, +Options, -Entry
    r_lowered_func_name/2,       % +Functor/Arity, -RFuncName
    gather_pred_mode_records/2   % +PredIndicator, -Records
]).

:- use_module(library(lists)).
:- use_module('../core/binding_state_analysis').
:- use_module('../core/demand_analysis').
:- use_module(wam_ite_structurer, [structure_ite/2]).
:- use_module(wam_clause_chain, [clause_chain/2]).

% =====================================================================
% Emission plan
% =====================================================================
%
% A predicate's WAM text is summarised into a `plan(Mode, AltLabel,
% ClauseLines)` record:
%
%   - Mode is `deterministic` or `multi_clause_n`.
%   - AltLabel is retained for the older multi_clause_1 shape; phase 4
%     multi_clause_n does not need it and uses `none`.
%   - ClauseLines is either the deterministic clause's WAM-text lines
%     or, for multi_clause_n, a list of per-clause WAM-text line lists.
%
% switch_on_constant / switch_on_structure prefixes are dropped: they
% were optimisations the array path uses to short-circuit dispatch.
% Skipping them in the lowered path is correctness-preserving: each
% lowered clause tries its own head match and failures move to the next
% lowered clause.

build_emission_plan(WamCode, plan(Mode, AltLabel, ClauseLines)) :-
    atom_string(WamCode, S),
    split_string(S, "\n", "", AllLines),
    skip_to_first_real_instr(AllLines, Filtered),
    classify_clause_shape(Filtered, plan(Mode, AltLabel, ClauseLines)).

% Drop blank lines, label-only lines, and switch_on_* dispatch prefixes
% until we find the first real WAM instruction.
skip_to_first_real_instr([], []).
skip_to_first_real_instr([Line|Rest], Out) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   skippable_prefix_line(Parts)
    ->  skip_to_first_real_instr(Rest, Out)
    ;   Out = [Line|Rest]
    ).

skippable_prefix_line([]).
skippable_prefix_line([First|_]) :- sub_string(First, _, 1, 0, ":").
skippable_prefix_line(["switch_on_constant"|_]).
skippable_prefix_line(["switch_on_structure"|_]).

% First-real-instruction discriminates between deterministic and
% multi_clause_n. A try_me_else here means a standard WAM try-chain
% follows, which we split into per-clause line groups.
% T5: the clauses discriminate on a DISTINCT first-argument constant
% (lowering type T5). Takes precedence over multi_clause_n: a BOUND first
% argument dispatches in O(1) straight to the matching clause (no try-all
% loop, no choice point); an UNBOUND first argument falls back to the
% multi_clause_n try-all loop, which enumerates every clause. The payload
% carries the per-clause discriminators (for the dispatch) and the same
% per-clause line groups multi_clause_n uses (for try_clause_).
classify_clause_shape([FirstLine|Rest],
                      plan(clause_chain, none, chain_payload(Discriminators, Clauses))) :-
    wam_r_target:tokenize_wam_line(FirstLine, ["try_me_else", _AltStr]),
    r_chain_terms([FirstLine|Rest], Terms),
    clause_chain(Terms, chain(Guards)),
    findall(V, member(guard(V, _), Guards), Discriminators),
    take_multi_clause_lines(Rest, Clauses),
    length(Guards, NClauses),
    length(Clauses, NClauses),                 % guards align 1:1 with clauses
    forall(member(guard(_, Rem), Guards), r_chain_rem_supported(Rem)),
    !.
classify_clause_shape([FirstLine|Rest], plan(multi_clause_n, none, Clauses)) :-
    wam_r_target:tokenize_wam_line(FirstLine, ["try_me_else", _AltStr]), !,
    take_multi_clause_lines(Rest, Clauses).
% Soft-cut block: a single-clause if-then-else / negation / once. Its
% try_me_else is internal (preceded by the shared head-arg setup), not a
% clause separator, so the first real instruction is NOT try_me_else and
% the multi_clause_n clause above does not fire. Fold it through the shared
% structurer into ite(Cond,Then,Else) terms.
classify_clause_shape(Lines, plan(ite, none, Structured)) :-
    r_parse_terms(Lines, Terms),
    structure_ite(Terms, Structured),
    member(ite(_, _, _), Structured),
    \+ member(try_me_else(_), Structured),
    \+ member(trust_me, Structured),
    !.
classify_clause_shape(Lines, plan(deterministic, none, Lines)).

% --- Label-preserving term parse (for the shared structurer) -------------
% Each WAM line becomes a structural term the structurer understands
% (try_me_else/trust_me/jump/cut_ite/label and the !/0-commit builtin_call),
% or an opaque line(Parts) leaf that emit_line_parts/2 renders unchanged.
r_parse_terms([], []).
r_parse_terms([Line|Rest], Terms) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  r_parse_terms(Rest, Terms)
    ;   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  sub_string(First, 0, _, 1, LabelName),
        Terms = [label(LabelName)|More],
        r_parse_terms(Rest, More)
    ;   r_line_term(Parts, T),
        Terms = [T|More],
        r_parse_terms(Rest, More)
    ).

r_line_term(["try_me_else", L], try_me_else(L)) :- !.
r_line_term(["trust_me"], trust_me) :- !.
r_line_term(["jump", L], jump(L)) :- !.
r_line_term(["cut_ite"], cut_ite) :- !.
r_line_term(["builtin_call", Op, Ar], builtin_call(Op, Ar)) :- !.
r_line_term(Parts, line(Parts)).

% --- T5 clause-chain term parse (for the shared wam_clause_chain front-end) -
% Convert WAM lines into just the terms clause_chain inspects: the choice-point
% separators, the head get_constant(V, A1), and an opaque line(Parts) leaf for
% everything else. Label lines and blanks are dropped.
r_chain_terms([], []).
r_chain_terms([Line|Rest], Terms) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  r_chain_terms(Rest, Terms)
    ;   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  r_chain_terms(Rest, Terms)             % drop label lines
    ;   r_chain_term(Parts, T),
        Terms = [T|More],
        r_chain_terms(Rest, More)
    ).

r_chain_term(["try_me_else", L], try_me_else(L)) :- !.
r_chain_term(["retry_me_else", L], retry_me_else(L)) :- !.
r_chain_term(["trust_me"], trust_me) :- !.
r_chain_term(["get_constant", V, A], get_constant(V, A)) :- !.
r_chain_term(Parts, line(Parts)).

% Each clause remainder (everything after the head get_constant) must be a
% renderable line(Parts) leaf or a further get_constant.
r_chain_rem_supported([]).
r_chain_rem_supported([T|Rest]) :-
    ( T = get_constant(_, _) -> true
    ; T = line(Parts) -> parts_supported(Parts)
    ),
    r_chain_rem_supported(Rest).

% Multi-clause WAM emitted by the shared compiler has the shape:
%   try_me_else L2, clause1..., L2:, retry_me_else L3, clause2...,
%   L3:, trust_me, clause3...
% We drop labels and choice instructions, preserving each clause body
% through its `proceed` so the lowered closure returns TRUE at success.
take_multi_clause_lines(Lines0, Clauses) :-
    skip_clause_prefix(Lines0, Lines),
    take_one_clause(Lines, Clause, Rest),
    (   Clause == []
    ->  Clauses = []
    ;   Clauses = [Clause | Tail],
        take_multi_clause_lines(Rest, Tail)
    ).

skip_clause_prefix([], []).
skip_clause_prefix([Line|Rest], Out) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  skip_clause_prefix(Rest, Out)
    ;   Parts = [First|_],
        sub_string(First, _, 1, 0, ":")
    ->  skip_clause_prefix(Rest, Out)
    ;   Parts = ["retry_me_else", _]
    ->  skip_clause_prefix(Rest, Out)
    ;   Parts == ["trust_me"]
    ->  skip_clause_prefix(Rest, Out)
    ;   Out = [Line|Rest]
    ).

take_one_clause(Lines, Clause, Tail) :-
    take_one_clause_(Lines, [], RevClause, Tail),
    reverse(RevClause, Clause).

take_one_clause_([], Acc, Acc, []).
take_one_clause_([Line|Rest], Acc, Acc, [Line|Rest]) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    Acc \= [],
    clause_boundary_parts(Parts),
    !.
take_one_clause_([Line|Rest], Acc, Out, Tail) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    Acc1 = [Line|Acc],
    (   terminal_clause_parts(Parts)
    ->  Out = Acc1,
        Tail = Rest
    ;   take_one_clause_(Rest, Acc1, Out, Tail)
    ).

clause_boundary_parts([First|_]) :-
    sub_string(First, _, 1, 0, ":"), !.
clause_boundary_parts(["retry_me_else", _]).
clause_boundary_parts(["trust_me"]).

terminal_clause_parts(["proceed"]).
terminal_clause_parts(["fail"]).
terminal_clause_parts(["execute", _]).
terminal_clause_parts(["execute", _, _]).

% =====================================================================
% Lowerability
% =====================================================================

%% wam_r_lowerable(+Pred, +WamCode, -Reason) is semidet.
%  Reason is one of `deterministic` or `multi_clause_n`. Lowerability
%  is decided against the emission-plan's clause lines.
wam_r_lowerable(_PI, WamCode, Reason) :-
    catch(build_emission_plan(WamCode, plan(Mode, _, ClauseData)), _, fail),
    (   Mode == ite
    ->  forall(member(I, ClauseData), r_struct_supported(I))
    ;   emission_plan_lines(Mode, ClauseData, Lines),
        forall(member(Line, Lines), line_supported(Line))
    ),
    Reason = Mode.

emission_plan_lines(deterministic, Lines, Lines).
emission_plan_lines(multi_clause_n, Clauses, Lines) :-
    append(Clauses, Lines).
emission_plan_lines(clause_chain, chain_payload(_, Clauses), Lines) :-
    append(Clauses, Lines).

%% r_struct_supported(+StructuredInstr) -- recurse through ite/3; each leaf
%  must be an instruction emit_struct_r/2 can render.
r_struct_supported(ite(C, T, E)) :- !,
    forall(member(I, C), r_struct_supported(I)),
    forall(member(I, T), r_struct_supported(I)),
    forall(member(I, E), r_struct_supported(I)).
r_struct_supported(builtin_call(_, _)) :- !.
r_struct_supported(line(Parts)) :- !,
    ( Parts == [] -> true ; parts_supported(Parts) ).
r_struct_supported(_) :- fail.

line_supported(Line) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   Parts == [] -> true
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":") -> true
    ;   parts_supported(Parts)
    ).

parts_supported(["allocate"]).
parts_supported(["deallocate"]).
parts_supported(["proceed"]).
parts_supported(["fail"]).
parts_supported(["get_constant", _, _]).
parts_supported(["get_variable", _, _]).
parts_supported(["get_value", _, _]).
parts_supported(["get_structure", _, _]).
parts_supported(["get_list", _]).
parts_supported(["put_constant", _, _]).
parts_supported(["put_variable", _, _]).
parts_supported(["put_value", _, _]).
parts_supported(["put_structure", _, _]).
parts_supported(["put_list", _]).
parts_supported(["unify_variable", _]).
parts_supported(["unify_value", _]).
parts_supported(["unify_constant", _]).
parts_supported(["set_variable", _]).
parts_supported(["set_value", _]).
parts_supported(["set_constant", _]).
parts_supported(["call", _]).
parts_supported(["call", _, _]).
parts_supported(["execute", _]).
parts_supported(["execute", _, _]).
parts_supported(["call_foreign", _, _]).
parts_supported(["builtin_call", _, _]).
parts_supported(["arg", _, _, _]).

% =====================================================================
% Function-name generation
% =====================================================================

%% r_lowered_func_name(+Functor/Arity, -RFuncName)
%  foo/2 -> "lowered_foo_2", my_pred/3 -> "lowered_my_pred_3".
r_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, FStr),
    sanitize_r_ident(FStr, SanStr),
    format(atom(Name), 'lowered_~w_~w', [SanStr, Arity]).

sanitize_r_ident(In, Out) :-
    string_codes(In, Codes),
    maplist(r_safe_code, Codes, OutCodes),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

r_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z -> true
    ;   C >= 0'A, C =< 0'Z -> true
    ;   C >= 0'0, C =< 0'9 -> true
    ;   C =:= 0'_ -> true
    ),
    !.
r_safe_code(_, 0'_).

% =====================================================================
% Lowered-function emission
% =====================================================================

%% lower_predicate_to_r(+Pred/Arity, +WamCode, +Options, -Entry)
%  Entry = lowered(PredName, FuncName, RCode).
%
%  Options:
%    mode_comments(on)  -- prepend a `# Mode analysis:` comment block
%                          summarising the inferred binding env per
%                          clause. Off by default. Visibility-only;
%                          phase 1 of WAM-R mode-analysis integration.
%    mode_specialise(off) -- disable the analyser-driven instruction
%                          specialisations (currently: inline
%                          get_constant when the target A-register is
%                          provably bound). Phase 2 specialisations
%                          are ON by default since they're always
%                          sound when mode declarations are honest
%                          and degrade to the step-delegating path
%                          when mode info is insufficient.
%
%  See docs/design/WAM_R_MODE_ANALYSIS_PLAN.md for the roadmap.
lower_predicate_to_r(PI, WamCode, Opts,
                     lowered(PredName, FuncName, Code)) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    r_lowered_func_name(Pred/Arity, FuncName),
    build_emission_plan(WamCode, plan(Mode, _AltLabel, ClauseLines)),
    catch(gather_pred_mode_records(PI, Records), _, Records = []),
    set_lowered_mode_context(Records, Opts),
    mode_comment_header_from_records(Opts, Records, ModeHeader),
    (   Mode == deterministic
    ->  emit_deterministic_function(PredName, FuncName, ClauseLines,
                                    ModeHeader, Code)
    ;   Mode == ite
    ->  emit_ite_function(PredName, FuncName, ClauseLines, ModeHeader, Code)
    ;   Mode == multi_clause_n
    ->  emit_multi_clause_function(PredName, FuncName, ClauseLines,
                                   ModeHeader, Code)
    ;   Mode == clause_chain
    ->  ClauseLines = chain_payload(Discriminators, Clauses),
        emit_clause_chain_function(PredName, FuncName, Discriminators, Clauses,
                                   ModeHeader, Code)
    ),
    clear_lowered_mode_context.

% If-then-else / negation / once. Same wrapper as the deterministic case,
% but the body is the structured term list rendered by emit_struct_r/2.
% R's bind always trails, so undoing the trail to the pre-condition mark
% before the else branch restores any partial bindings the condition made
% (no register snapshot needed; mirrors the Rust/Lua emitters). R `{}`
% blocks do not introduce a scope, so each ite uses a counter-suffixed
% mark/cond variable to stay safe under nesting and sequencing.
emit_ite_function(PredName, FuncName, Structured, ModeHeader, Code) :-
    reset_x_reg_states,
    b_setval(wam_r_ite_ctr, 0),
    with_output_to(string(Body), emit_struct_r(Structured, "  ")),
    format(string(Code),
'~w# Lowered: ~w  (if-then-else / negation / once)
~w <- function(program, state) {
~w  invisible(TRUE)
}', [ModeHeader, PredName, FuncName, Body]).

fresh_r_ite(N) :-
    ( catch(b_getval(wam_r_ite_ctr, N0), _, N0 = 0) -> true ; N0 = 0 ),
    N is N0 + 1,
    b_setval(wam_r_ite_ctr, N).

emit_struct_r([], _).
emit_struct_r([Item|Rest], Ind) :-
    emit_struct_item_r(Item, Ind),
    emit_struct_r(Rest, Ind).

emit_struct_item_r(ite(Cond, Then, Else), Ind) :- !,
    fresh_r_ite(N),
    string_concat(Ind, "  ", I2),
    format("~w{~n", [Ind]),
    format("~w  ite_mark_~w <- length(state$trail)~n", [Ind, N]),
    format("~w  ite_cond_~w <- (function() {~n", [Ind, N]),
    emit_struct_r(Cond, I2),
    format("~w    TRUE~n", [Ind]),
    format("~w  })()~n", [Ind]),
    format("~w  if (isTRUE(ite_cond_~w)) {~n", [Ind, N]),
    emit_struct_r(Then, I2),
    format("~w  } else {~n", [Ind]),
    format("~w    WamRuntime$undo_trail_to(state, ite_mark_~w)~n", [Ind, N]),
    emit_struct_r(Else, I2),
    format("~w  }~n", [Ind]),
    format("~w}~n", [Ind]).
emit_struct_item_r(builtin_call(Op, Ar), Ind) :- !,
    emit_line_parts(["builtin_call", Op, Ar], Ind).
emit_struct_item_r(line(Parts), Ind) :- !,
    emit_line_parts(Parts, Ind).

%% set_lowered_mode_context(+Records, +Opts) is det.
%  Stashes clause 1's `:- mode/1` declaration on a non-backtrackable
%  global so emit_line_parts can consult it without threading extra
%  args through the entire emission chain. Mirrors the shared
%  compiler's wam_clause_binding_records pattern.
%
%  Specialisations are gated by `mode_specialise(off)` in Opts --
%  defaulting to ON. When OFF (or when no mode declaration exists),
%  the stashed value is `none` so the specialised emit_line_parts
%  clauses all fail and the default step-delegating path runs.
%
%  IMPORTANT: the specialisation uses the mode declaration directly,
%  not the analyser's BeforeEnv. The BeforeEnv conflates head-pattern
%  binding (a literal `alice` looks `bound`) with caller-side binding
%  (whether A_k was bound when the caller invoked us). For head-match
%  instructions like get_constant we need the latter, and only the
%  mode declaration gives it to us.
set_lowered_mode_context(Records, Opts) :-
    (   member(mode_specialise(off), Opts)
    ->  ModeDecl = none
    ;   clause1_mode_decl(Records, ModeDecl)
    ),
    b_setval(wam_r_lowered_mode_decl, ModeDecl),
    b_setval(wam_r_lowered_x_states, []).

clear_lowered_mode_context :-
    b_setval(wam_r_lowered_mode_decl, none),
    b_setval(wam_r_lowered_x_states, []).

%% clause1_mode_decl(+Records, -ModeDecl) is det.
%  Pulls the ModeDecl component of clause 1's record. Returns `none`
%  when there are no records or no declaration.
clause1_mode_decl([mode_record(1, ModeDecl, _, _) | _], ModeDecl) :- !.
clause1_mode_decl(_, none).

%% head_state_for_areg(+AiStr, -State) is semidet.
%  Returns the caller-side binding state for the A-register named by
%  AiStr (e.g. "A1" -> first head arg position). Uses the stashed
%  mode declaration:
%    +  (input)  -> bound
%    -  (output) -> unbound
%    ?  (any)    -> fails (no provable state)
%  Fails if AiStr isn't in the A-register range (1..100), if no mode
%  declaration is stashed, or if the position is out of range in the
%  mode list. Used by the get_constant specialised emit_line_parts
%  clause.
head_state_for_areg(AiStr, State) :-
    wam_r_target:reg_to_int(AiStr, AIdx),
    AIdx >= 1, AIdx =< 100,
    catch(b_getval(wam_r_lowered_mode_decl, ModeDecl), _, fail),
    is_list(ModeDecl),
    nth1(AIdx, ModeDecl, Mode),
    mode_atom_to_state(Mode, State).

mode_atom_to_state(input,  bound).
mode_atom_to_state(output, unbound).

%% x_reg_state(+XStr, -State) is semidet.
%  Returns the simple binding state tracked for X registers during
%  lowered emission. This is intentionally shallow: get_variable from
%  a known-bound A register marks the destination X register as bound;
%  get_variable from a known-unbound A register marks it unbound.
x_reg_state(XStr, State) :-
    catch(b_getval(wam_r_lowered_x_states, States), _, fail),
    memberchk(XStr-State, States).

%% set_x_reg_state(+XStr, +State) is det.
set_x_reg_state(XStr, State) :-
    catch(b_getval(wam_r_lowered_x_states, States0), _, States0 = []),
    exclude(same_x_reg(XStr), States0, States),
    b_setval(wam_r_lowered_x_states, [XStr-State | States]).

same_x_reg(XStr, K-_) :-
    K == XStr.

reset_x_reg_states :-
    b_setval(wam_r_lowered_x_states, []).

%% mode_comment_header_from_records(+Opts, +Records, -Header) is det.
%  Builds the R comment block for the predicate's mode analysis when
%  Opts contains mode_comments(on). Otherwise Header is the empty
%  string.
mode_comment_header_from_records(Opts, Records, Header) :-
    (   member(mode_comments(on), Opts),
        Records \= []
    ->  format_mode_records(Records, Lines),
        atomic_list_concat(Lines, '', Header)
    ;   Header = ""
    ).

%% gather_pred_mode_records(+PredIndicator, -Records) is det.
%  Pulls the user-asserted clauses for the predicate, runs the
%  binding-state analyser per clause, and returns one
%  `mode_record(Idx, ModeDecl, HeadVars, GoalBindings)` per clause.
%
%    - Idx        : 1-based clause number.
%    - ModeDecl   : list of mode atoms (input/output/any) from
%                   `:- mode/1`, or `none` if undeclared.
%    - HeadVars   : list of `Name-State` pairs, one per head arg
%                   position. Name is `Arg<N>` for the Nth arg
%                   position (1-based); State is the analyser's
%                   binding-state for that arg right before the
%                   first body goal (== entry to the clause body).
%                   Non-variable head args render as `Arg<N>-bound`
%                   (literal terms are always bound).
%    - GoalBindings : the raw goal_binding(Idx, Before, After) list
%                     from the analyser (renderable via
%                     format_mode_records/2).
%
%  IMPORTANT: the analyser is called on plain Prolog variables (NOT
%  numbervars-renamed), because `get_binding_state(_, NonVar, bound)`
%  treats `'$VAR'(N)` as a nonvar and would always return `bound`.
%  Head arg names in HeadVars are position-based for display only.
%
%  Module qualifiers in clause bodies are stripped (plunit wraps
%  asserted bodies in plunit_<test>:user:Goal). Failure is silent --
%  any error in the analyser causes the records to be empty so the
%  caller can degrade gracefully.
gather_pred_mode_records(PI, Records) :-
    ( PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    functor(HeadTpl, Pred, Arity),
    catch(findall(HeadCopy-BodyCopy,
                  ( user:clause(HeadTpl, RawBody),
                    wam_r_target:strip_module_qualifiers(RawBody,
                                                         StrippedBody),
                    copy_term(HeadTpl-StrippedBody,
                              HeadCopy-BodyCopy) ),
                  Clauses), _, Clauses = []),
    (   demand_analysis:read_mode_declaration(Pred, Arity, ModeDecl)
    ->  true
    ;   ModeDecl = none
    ),
    gather_clause_records(Clauses, 1, ModeDecl, Records).

gather_clause_records([], _, _, []).
gather_clause_records([Head-Body | Rest], Idx, ModeDecl,
                      [mode_record(Idx, ModeDecl, HeadVars, GBs) | Tail]) :-
    catch(binding_state_analysis:analyse_clause_bindings(Head, Body, GBs0),
          _, GBs0 = []),
    GBs = GBs0,
    head_var_states(Head, ModeDecl, GBs, HeadVars),
    Idx1 is Idx + 1,
    gather_clause_records(Rest, Idx1, ModeDecl, Tail).

%% head_var_states(+Head, +ModeDecl, +GoalBindings, -HeadVars) is det.
%  Returns one `Arg<N>-State` pair per head arg position, where State
%  is the analyser's binding state for that arg right before the first
%  body goal (the env that an optimizer would consult when deciding
%  whether to specialise a head-match instruction). Non-variable head
%  args (compound, atomic, struct, ...) always render as `bound`
%  because head unification by definition binds them. Variable head
%  args render with the analyser's actual state -- `bound` (mode +),
%  `unbound` (mode -), or `unknown` (no info).
%
%  When the body has no goals (e.g. `true` / a fact clause), the
%  analyser returns an empty GoalBindings list; we fall back to the
%  initial env from `initial_binding_env/3` so head modes still
%  surface in the visibility comment.
head_var_states(Head, ModeDecl, GBs, HeadVars) :-
    Head =.. [_ | Args],
    (   member(goal_binding(1, BeforeEnv, _), GBs)
    ->  Env = BeforeEnv
    ;   binding_state_analysis:initial_binding_env(Head, ModeDecl, Env)
    ),
    head_var_states_(Args, 1, Env, HeadVars).

head_var_states_([], _, _, []).
head_var_states_([Arg | Rest], Pos, Env, [Name-State | Tail]) :-
    format(atom(Name), 'Arg~w', [Pos]),
    head_arg_state(Arg, Env, State),
    Pos1 is Pos + 1,
    head_var_states_(Rest, Pos1, Env, Tail).

head_arg_state(Arg, _Env, bound) :-
    nonvar(Arg), !.
head_arg_state(Arg, Env, State) :-
    var(Arg),
    binding_state_analysis:get_binding_state(Env, Arg, State).

%% format_mode_records(+Records, -Lines) is det.
%  Renders the analyser output as a block of R comment lines.
%  Empty record list yields the empty string (no comment block).
format_mode_records([], [""]) :- !.
format_mode_records(Records, [Header | Body]) :-
    Header = "# Mode analysis (phase 1, visibility-only):\n",
    maplist(format_one_clause, Records, Bodies),
    flatten(Bodies, Body0),
    append(Body0, ["#\n"], Body).

format_one_clause(mode_record(Idx, ModeDecl, HeadVars, _GBs), Lines) :-
    format(string(L1), "#   clause ~w head: ~w   (mode_decl=~w)\n",
           [Idx, HeadVars, ModeDecl]),
    Lines = [L1].

emit_deterministic_function(PredName, FuncName, ClauseLines,
                            ModeHeader, Code) :-
    reset_x_reg_states,
    with_output_to(string(Body), emit_lines(ClauseLines, "  ")),
    % The clause body always ends with `proceed`, which emits
    % `return(TRUE)`. We keep an explicit `invisible(TRUE)` after the
    % body as a defensive fallback in case future codegen produces a
    % proceed-less clause; the wrapper does isTRUE() on the result.
    format(string(Code),
'~w# Lowered: ~w  (deterministic single-clause)
~w <- function(program, state) {
~w  invisible(TRUE)
}', [ModeHeader, PredName, FuncName, Body]).

% Multi-clause: every supported clause is emitted as an inline closure.
% On success, a lightweight iter CP is pushed for the next clause so
% caller-side backtracking can resume at clause N+1. On failure, the
% original entry snapshot is restored before the next inline clause is
% tried.
emit_multi_clause_function(PredName, FuncName, Clauses,
                           ModeHeader, Code) :-
    length(Clauses, NClauses),
    with_output_to(string(ClauseCode), emit_clause_dispatch(Clauses, 1)),
    format(string(Code),
'~w# Lowered: ~w  (multi-clause; all clauses inline)
~w <- function(program, state) {
  snap_regs_      <- state$regs2
  snap_cp_        <- state$cp
  snap_trail_     <- length(state$trail)
  snap_var_count_ <- state$var_counter
  snap_mode_      <- state$mode
  snap_build_     <- state$build_stack
  snap_read_      <- state$read_stack
  snap_stack_len_ <- length(state$stack)
  snap_barrier_   <- state$pending_call_barrier
  resume_pc_      <- state$pc + 1L
  restore_clause_entry_ <- function() {
    state$regs2 <- snap_regs_
    WamRuntime$undo_trail_to(state, snap_trail_)
    state$cp <- snap_cp_
    state$var_counter <- snap_var_count_
    state$mode <- snap_mode_
    state$build_stack <- snap_build_
    state$read_stack <- snap_read_
    state$pending_call_barrier <- snap_barrier_
    if (length(state$stack) > snap_stack_len_) {
      if (snap_stack_len_ == 0L) state$stack <- list()
      else state$stack <- state$stack[seq_len(snap_stack_len_)]
    }
    state$shadow_frame <- NULL
  }
  try_clause_ <- function(idx_) {
~w    FALSE
  }
  push_next_clause_ <- function(next_idx_) {
    state$cps <- c(state$cps, list(list(
      kind        = "iter",
      regs        = snap_regs_,
      trail_len   = snap_trail_,
      cp          = snap_cp_,
      var_counter = snap_var_count_,
      mode        = snap_mode_,
      build_stack = snap_build_,
      read_stack  = snap_read_,
      call_barrier = snap_barrier_,
      stack_len   = snap_stack_len_,
      resume_pc   = resume_pc_,
      retry       = function(state) {
        for (idx_ in seq.int(next_idx_, ~wL)) {
          ok_ <- isTRUE(try_clause_(idx_))
          if (ok_) {
            if (idx_ < ~wL) push_next_clause_(idx_ + 1L)
            return(TRUE)
          }
          restore_clause_entry_()
        }
        FALSE
      }
    )))
  }
  for (idx_ in seq_len(~wL)) {
    ok_ <- isTRUE(try_clause_(idx_))
    if (ok_) {
      if (idx_ < ~wL) push_next_clause_(idx_ + 1L)
      return(TRUE)
    }
    restore_clause_entry_()
  }
  FALSE
}', [ModeHeader, PredName, FuncName, ClauseCode,
      NClauses, NClauses, NClauses, NClauses]).

% T5 first-argument dispatch. Reuses the multi_clause_n machinery verbatim
% (per-clause snapshots, try_clause_, restore_clause_entry_, push_next_clause_
% and the try-all fallback loop) but inserts an O(1) dispatch prologue: a
% BOUND first argument is matched against each clause's distinct discriminator
% and the matching clause is run directly via try_clause_(k) — no try-all
% loop, no choice point (the discriminators are distinct, so it is
% deterministic). On a clause-body failure the state is restored, matching the
% loop's contract. An UNBOUND first argument (or a register that is unset)
% falls through to the multi_clause_n try-all loop, which enumerates every
% clause exactly as before.
emit_clause_chain_function(PredName, FuncName, Discriminators, Clauses,
                           ModeHeader, Code) :-
    length(Clauses, NClauses),
    with_output_to(string(ClauseCode), emit_clause_dispatch(Clauses, 1)),
    with_output_to(string(DispatchCode), emit_chain_dispatch_r(Discriminators, 1)),
    % The header keeps the multi_clause_n marker — clause_chain IS an
    % all-clauses-inline lowering (try_clause_ holds every clause) — and adds
    % the T5 note for the first-argument dispatch prologue.
    format(string(Code),
'~w# Lowered: ~w  (multi-clause; all clauses inline; T5 first-argument dispatch)
~w <- function(program, state) {
  snap_regs_      <- state$regs2
  snap_cp_        <- state$cp
  snap_trail_     <- length(state$trail)
  snap_var_count_ <- state$var_counter
  snap_mode_      <- state$mode
  snap_build_     <- state$build_stack
  snap_read_      <- state$read_stack
  snap_stack_len_ <- length(state$stack)
  snap_barrier_   <- state$pending_call_barrier
  resume_pc_      <- state$pc + 1L
  restore_clause_entry_ <- function() {
    state$regs2 <- snap_regs_
    WamRuntime$undo_trail_to(state, snap_trail_)
    state$cp <- snap_cp_
    state$var_counter <- snap_var_count_
    state$mode <- snap_mode_
    state$build_stack <- snap_build_
    state$read_stack <- snap_read_
    state$pending_call_barrier <- snap_barrier_
    if (length(state$stack) > snap_stack_len_) {
      if (snap_stack_len_ == 0L) state$stack <- list()
      else state$stack <- state$stack[seq_len(snap_stack_len_)]
    }
    state$shadow_frame <- NULL
  }
  try_clause_ <- function(idx_) {
~w    FALSE
  }
  push_next_clause_ <- function(next_idx_) {
    state$cps <- c(state$cps, list(list(
      kind        = "iter",
      regs        = snap_regs_,
      trail_len   = snap_trail_,
      cp          = snap_cp_,
      var_counter = snap_var_count_,
      mode        = snap_mode_,
      build_stack = snap_build_,
      read_stack  = snap_read_,
      call_barrier = snap_barrier_,
      stack_len   = snap_stack_len_,
      resume_pc   = resume_pc_,
      retry       = function(state) {
        for (idx_ in seq.int(next_idx_, ~wL)) {
          ok_ <- isTRUE(try_clause_(idx_))
          if (ok_) {
            if (idx_ < ~wL) push_next_clause_(idx_ + 1L)
            return(TRUE)
          }
          restore_clause_entry_()
        }
        FALSE
      }
    )))
  }
  # T5 fast path: a bound first argument selects exactly one clause.
  t5a1_ <- WamRuntime$deref(state, WamRuntime$get_reg(state, 1L))
  if (!is.null(t5a1_) && !(is.list(t5a1_) && !is.null(t5a1_$tag) && t5a1_$tag == "unbound")) {
~w    return(FALSE)
  }
  # Unbound first argument: enumerate every clause.
  for (idx_ in seq_len(~wL)) {
    ok_ <- isTRUE(try_clause_(idx_))
    if (ok_) {
      if (idx_ < ~wL) push_next_clause_(idx_ + 1L)
      return(TRUE)
    }
    restore_clause_entry_()
  }
  FALSE
}', [ModeHeader, PredName, FuncName, ClauseCode,
      NClauses, NClauses, DispatchCode, NClauses, NClauses]).

%% emit_chain_dispatch_r(+Discriminators, +Idx)
%  One `if (identical(t5a1_, <const>)) { ... try_clause_(idx) ... }` per clause.
emit_chain_dispatch_r([], _).
emit_chain_dispatch_r([V|Rest], Idx) :-
    wam_r_target:constant_to_r_term(V, CTerm),
    format("    if (identical(t5a1_, ~w)) { ok_ <- isTRUE(try_clause_(~wL)); if (!ok_) restore_clause_entry_(); return(ok_) }~n",
           [CTerm, Idx]),
    Idx1 is Idx + 1,
    emit_chain_dispatch_r(Rest, Idx1).

emit_clause_dispatch([], _).
emit_clause_dispatch([Lines|Rest], Idx) :-
    format("    if (identical(idx_, ~wL)) {~n", [Idx]),
    format("      return((function() {~n"),
    reset_x_reg_states,
    emit_lines(Lines, "        "),
    format("        invisible(FALSE)~n"),
    format("      })())~n"),
    format("    }~n"),
    Idx1 is Idx + 1,
    emit_clause_dispatch(Rest, Idx1).

emit_lines([], _).
emit_lines([Line|Rest], Ind) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  true
    ;   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  true
    ;   emit_line_parts(Parts, Ind)
    ),
    emit_lines(Rest, Ind).

% --- Control instructions ------------------------------------------------

emit_line_parts(["proceed"], I) :- !,
    format("~wreturn(TRUE)~n", [I]).
emit_line_parts(["fail"], I) :- !,
    format("~wreturn(FALSE)~n", [I]).
% Explicit is_iso/is_lax arrive as Call/Execute; route to shared helpers.
emit_line_parts(["call", "is_iso/2"], I) :- !,
    format("~wif (!isTRUE(WamRuntime$builtin_is_iso(program, state))) return(FALSE)~n", [I]).
emit_line_parts(["call", "is_lax/2"], I) :- !,
    format("~wif (!isTRUE(WamRuntime$builtin_is_lax(program, state))) return(FALSE)~n", [I]).
emit_line_parts(["execute", "is_iso/2"], I) :- !,
    format("~wif (!isTRUE(WamRuntime$builtin_is_iso(program, state))) return(FALSE)~n", [I]),
    format("~wreturn(TRUE)~n", [I]).
emit_line_parts(["execute", "is_lax/2"], I) :- !,
    format("~wif (!isTRUE(WamRuntime$builtin_is_lax(program, state))) return(FALSE)~n", [I]),
    format("~wreturn(TRUE)~n", [I]).
emit_line_parts(["call", PredArity], I) :- !,
    emit_call(PredArity, I).
emit_line_parts(["call", Pred, ArityStr], I) :- !,
    strip_arity_local(Pred, PredName),
    format(string(PredArity), "~w/~w", [PredName, ArityStr]),
    emit_call(PredArity, I).
emit_line_parts(["execute", PredArity], I) :- !,
    emit_execute(PredArity, I).
emit_line_parts(["execute", Pred, ArityStr], I) :- !,
    strip_arity_local(Pred, PredName),
    format(string(PredArity), "~w/~w", [PredName, ArityStr]),
    emit_execute(PredArity, I).

% --- Native register ops (always succeed; skip the step dispatcher) ------
%
% These are the paydirt of Phase 3's per-instruction inlining: each
% one is a direct mutation on state$regs2 / state$stack / state, no
% heap-build state machine, no deref, no unify. The R generated for
% them is the same code WamRuntime$step would have run, just inline.

% allocate / deallocate must mirror the interpreter's frame layout: Y
% registers (idx >= 201) live in the frame's `ys` environment, which
% WamRuntime$put_reg/get_reg read/write, and cut consults `cps_barrier`.
% The previous emit used a `locals` field with no `ys`, so any lowered
% predicate with an environment frame (every if-then-else, and any
% deterministic clause with permanent vars) crashed on the first Y-register
% access. Deallocate retains the popped frame as `shadow_frame` so a Y read
% after deallocate (the SWI emit's Call;Deallocate;PutValue Y_n;... shape)
% still sees the body's writes — matching the interpreter's Deallocate.
emit_line_parts(["allocate"], I) :- !,
    format("~wstate$stack <- c(state$stack, list(list(cp = state$cp, ys = new.env(parent = emptyenv(), hash = TRUE), cps_barrier = state$pending_call_barrier)))~n", [I]),
    format("~wstate$shadow_frame <- NULL~n", [I]).
emit_line_parts(["deallocate"], I) :- !,
    format("~w{ n_ <- length(state$stack); if (n_ > 0L) { frame_ <- state$stack[[n_]]; state$cp <- frame_$cp; state$shadow_frame <- frame_; state$stack <- state$stack[-n_] } }~n", [I]).
emit_line_parts(["put_constant", CStr, RegStr], I) :- !,
    wam_r_target:reg_to_int(RegStr, RegIdx),
    wam_r_target:constant_to_r_term(CStr, CTerm),
    format("~wWamRuntime$put_reg(state, ~w, ~w)~n", [I, RegIdx, CTerm]).
emit_line_parts(["put_variable", XStr, AiStr], I) :- !,
    wam_r_target:reg_to_int(XStr, XIdx),
    wam_r_target:reg_to_int(AiStr, AIdx),
    format("~w{ v_ <- Unbound(paste0(\"V\", state$var_counter)); state$var_counter <- state$var_counter + 1L; WamRuntime$put_reg(state, ~w, v_); WamRuntime$put_reg(state, ~w, v_) }~n",
           [I, XIdx, AIdx]).
emit_line_parts(["put_value", XStr, AiStr], I) :- !,
    wam_r_target:reg_to_int(XStr, XIdx),
    wam_r_target:reg_to_int(AiStr, AIdx),
    format("~wWamRuntime$put_reg(state, ~w, WamRuntime$get_reg(state, ~w))~n",
           [I, AIdx, XIdx]).
emit_line_parts(["get_variable", XStr, AiStr], I) :- !,
    wam_r_target:reg_to_int(XStr, XIdx),
    wam_r_target:reg_to_int(AiStr, AIdx),
    (   head_state_for_areg(AiStr, State)
    ->  set_x_reg_state(XStr, State)
    ;   true
    ),
    format("~wWamRuntime$put_reg(state, ~w, WamRuntime$get_reg(state, ~w))~n",
           [I, XIdx, AIdx]).

% --- Mode-driven specialisation: inline get_constant when the analyser
% proves the target A-register is bound. The default path delegates to
% WamRuntime$step which handles the unbound branch (binds A_i to the
% constant). When mode info promises A_i is already bound, the unbound
% branch is dead code -- we just need a deref + identical-equality
% check. Saves one function call (~0.5us, the dominant cost) per
% get_constant; significant on head-match-heavy workloads.
%
% Soundness: relies on the analyser's `bound` answer being correct,
% which in turn relies on `:- mode/1` declarations being honest. If a
% caller passes an unbound term where the mode says +, the inline
% form returns FALSE (treating unbound as a tag mismatch) where the
% step path would have bound it. Document; user is responsible.
emit_line_parts(["get_constant", CStr, AiStr], I) :-
    head_state_for_areg(AiStr, bound),
    !,
    wam_r_target:reg_to_int(AiStr, AIdx),
    wam_r_target:constant_to_r_term(CStr, CTerm),
    format("~w{ val_ <- WamRuntime$deref(state, WamRuntime$get_reg(state, ~w)); if (is.null(val_) || !identical(val_, ~w)) return(FALSE) }~n",
           [I, AIdx, CTerm]).

% Mode-driven specialisation for head get_value. In the common
% repeated-head-var pattern (`p(X, X)`), WAM emits get_variable for
% the first occurrence and get_value for the second. When mode decls
% prove both positions are bound, the unbound-binding branch of unify
% is dead. Atomic bound pairs can be checked by tag/value identity;
% non-atomic bound terms fall back to unify to preserve structural
% semantics.
emit_line_parts(["get_value", XStr, AiStr], I) :-
    x_reg_state(XStr, bound),
    head_state_for_areg(AiStr, bound),
    !,
    wam_r_target:reg_to_int(XStr, XIdx),
    wam_r_target:reg_to_int(AiStr, AIdx),
    format("~w{~n", [I]),
    format("~w  gv_x_ <- WamRuntime$get_reg(state, ~w)~n", [I, XIdx]),
    format("~w  gv_a_ <- WamRuntime$get_reg(state, ~w)~n", [I, AIdx]),
    format("~w  gv_x_d_ <- WamRuntime$deref(state, gv_x_)~n", [I]),
    format("~w  gv_a_d_ <- WamRuntime$deref(state, gv_a_)~n", [I]),
    format("~w  if (is.null(gv_x_d_) || is.null(gv_a_d_)) return(FALSE)~n", [I]),
    format("~w  if (!is.null(gv_x_d_$tag) && gv_x_d_$tag == \"unbound\") return(FALSE)~n", [I]),
    format("~w  if (!is.null(gv_a_d_$tag) && gv_a_d_$tag == \"unbound\") return(FALSE)~n", [I]),
    format("~w  gv_atomic_ <- c(\"atom\", \"int\", \"float\")~n", [I]),
    format("~w  if (!is.null(gv_x_d_$tag) && !is.null(gv_a_d_$tag) && gv_x_d_$tag %in% gv_atomic_ && gv_a_d_$tag %in% gv_atomic_) {~n", [I]),
    format("~w    if (!identical(gv_x_d_, gv_a_d_)) return(FALSE)~n", [I]),
    format("~w  } else if (!isTRUE(WamRuntime$unify(state, gv_x_, gv_a_))) return(FALSE)~n", [I]),
    format("~w}~n", [I]).

% --- Builtin specialisations: inline the most common BuiltinCall
% targets so they skip the WamRuntime$step -> WamRuntime$call_builtin
% function-call + switch-dispatch hop. The inline path is semantically
% identical to the slow path; it just avoids two function calls and
% two switch lookups per call. Used heavily on arith-heavy workloads
% where is/2 is the dominant builtin. is_lax/2 shares the lax helper;
% is_iso/2 routes through the shared ISO helper (no duplicated classify).

emit_line_parts(["builtin_call", "is/2", "2"], I) :- !,
    format("~wif (!isTRUE(WamRuntime$builtin_is_lax(program, state))) return(FALSE)~n",
           [I]).
emit_line_parts(["builtin_call", "is_lax/2", "2"], I) :- !,
    format("~wif (!isTRUE(WamRuntime$builtin_is_lax(program, state))) return(FALSE)~n",
           [I]).
emit_line_parts(["builtin_call", "is_iso/2", "2"], I) :- !,
    format("~wif (!isTRUE(WamRuntime$builtin_is_iso(program, state))) return(FALSE)~n",
           [I]).

% --- Default: delegate to step with the same R literal the array uses
emit_line_parts(Parts, I) :-
    wam_r_target:wam_parts_to_r(Parts, [], Lit),
    format("~wif (!isTRUE(WamRuntime$step(program, state, ~w))) return(FALSE)~n",
           [I, Lit]).

emit_call(PredArity, I) :-
    format("~w{~n", [I]),
    format("~w  saved_cp <- state$cp~n", [I]),
    format("~w  tgt <- program$labels[[\"~w\"]]~n", [I, PredArity]),
    format("~w  if (is.null(tgt)) return(FALSE)~n", [I]),
    format("~w  state$cp <- 0L~n", [I]),
    format("~w  state$pc <- as.integer(tgt)~n", [I]),
    format("~w  if (!isTRUE(WamRuntime$run(program, state))) return(FALSE)~n", [I]),
    format("~w  state$halt <- FALSE~n", [I]),
    format("~w  state$cp <- saved_cp~n", [I]),
    format("~w}~n", [I]).

emit_execute(PredArity, I) :-
    format("~w{~n", [I]),
    format("~w  tgt <- program$labels[[\"~w\"]]~n", [I, PredArity]),
    format("~w  if (is.null(tgt)) return(FALSE)~n", [I]),
    format("~w  state$pc <- as.integer(tgt)~n", [I]),
    format("~w  return(isTRUE(WamRuntime$run(program, state)))~n", [I]),
    format("~w}~n", [I]).

%% strip_arity_local(+TokenWithArity, -Name)
%  Same convention as wam_r_target's strip_arity_suffix: drop trailing
%  "/N" if present so we can rebuild "Name/N" from the explicit arity
%  token that follows in the WAM line.
strip_arity_local(Tok, Name) :-
    (   sub_string(Tok, B, 1, _, "/")
    ->  sub_string(Tok, 0, B, _, Name)
    ;   Name = Tok
    ).
