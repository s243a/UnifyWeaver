:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_javascript_lowered_emitter.pl — Tier-2 lowered JS functions.
%
% Ports the Lua lowered emitter (wam_lua_lowered_emitter.pl) to Node:
% clause-1 / T4 / T5 / T6 / ITE fast paths that emit DIRECT host functions
% (no per-goal interpreter dispatch). Unsupported shapes fall back to the
% interpreter rather than emitting wrong code.

:- module(wam_javascript_lowered_emitter, [
    wam_javascript_lowerable/3,
    wam_javascript_explain_lower/3,
    lower_predicate_to_javascript/4,
    js_lowered_func_name/2
]).

:- use_module(library(lists)).
:- use_module(wam_ite_structurer, [structure_ite/2]).
:- use_module(wam_clause_chain, [clause_chain/2]).
:- use_module(wam_text_parser, [
    wam_tokenize_line/2,
    wam_classify_constant_token/2,
    wam_constant_token_is_string/1
]).

tokenize_line(Line, Parts) :-
    wam_tokenize_line(Line, Parts).

build_emission_plan(WamCode, plan(Mode, AltLabel, ClauseLines)) :-
    atom_string(WamCode, S),
    split_string(S, "\n", "", Lines),
    skip_to_first_real_instr(Lines, Filtered0),
    % JS first-arg indexing re-emits each clause once per dispatch group
    % and appends try/retry/trust chains. Collapse back to a unique-A1
    % linear chain so T5 / deterministic classification still fires.
    normalize_js_indexed_clauses(Filtered0, Filtered),
    classify_clause_shape(Filtered, plan(Mode, AltLabel, ClauseLines)).

skip_to_first_real_instr([], []).
skip_to_first_real_instr([Line|Rest], Out) :-
    tokenize_line(Line, Parts),
    (   skippable_prefix_line(Parts)
    ->  skip_to_first_real_instr(Rest, Out)
    ;   Out = [Line|Rest]
    ).

skippable_prefix_line([]).
skippable_prefix_line([First|_]) :- sub_string(First, _, 1, 0, ":").
skippable_prefix_line(["switch_on_constant"|_]).
skippable_prefix_line(["switch_on_constant_fallthrough"|_]).
skippable_prefix_line(["switch_on_constant_a2"|_]).
skippable_prefix_line(["switch_on_constant_a2_fallthrough"|_]).
skippable_prefix_line(["switch_on_structure"|_]).
skippable_prefix_line(["switch_on_structure_a2"|_]).
skippable_prefix_line(["switch_on_term"|_]).
skippable_prefix_line(["switch_on_term_a2"|_]).
skippable_prefix_line(["try"|_]).
skippable_prefix_line(["retry"|_]).
skippable_prefix_line(["trust"|_]).

% Drop dedicated try/retry/trust dispatch ops and collapse duplicate
% first-arg clauses produced by JS indexing into a unique-A1 try-chain.
normalize_js_indexed_clauses(Lines, Out) :-
    exclude(js_index_dispatch_line, Lines, Kept),
    js_split_choice_clauses(Kept, Clauses0),
    Clauses0 = [_, _ | _],
    maplist(js_clause_a1_key, Clauses0, Keys),
    \+ member(none, Keys),
    js_first_unique_pairs(Keys, Clauses0, Unique),
    Unique \== [],
    js_first_alt_label(Kept, Alt),
    js_rebuild_try_chain(Unique, Alt, Out), !.
normalize_js_indexed_clauses(Lines, Lines).

js_first_alt_label([Line|_], Alt) :-
    tokenize_line(Line, ["try_me_else", Alt]), !.
js_first_alt_label([_|Rest], Alt) :-
    js_first_alt_label(Rest, Alt).
js_first_alt_label([], "L_js_lowered_alt").

js_index_dispatch_line(Line) :-
    tokenize_line(Line, Parts),
    Parts = [Op|_],
    member(Op, ["try", "retry", "trust"]).

js_split_choice_clauses(Lines, Clauses) :-
    js_take_choice_clause(Lines, Clause, Rest),
    (   Rest == []
    ->  (Clause == [] -> Clauses = [] ; Clauses = [Clause])
    ;   js_split_choice_clauses(Rest, More),
        (Clause == [] -> Clauses = More ; Clauses = [Clause|More])
    ).

js_take_choice_clause([], [], []).
js_take_choice_clause([Line|Rest], [], Rest) :-
    tokenize_line(Line, Parts),
    ( Parts = ["try_me_else"|_] ; Parts = ["retry_me_else"|_] ; Parts == ["trust_me"] ), !.
js_take_choice_clause([Line|Rest], [Line|More], After) :-
    js_take_choice_clause(Rest, More, After).

js_clause_a1_key(ClauseLines, Key) :-
    member(Line, ClauseLines),
    tokenize_line(Line, ["get_constant", V, A1]),
    ( A1 == "A1" ; A1 == 'A1' ), !,
    Key = V.
js_clause_a1_key(_, none).

js_first_unique_pairs(Keys, Clauses, Unique) :-
    js_first_unique_pairs_(Keys, Clauses, [], Unique).

js_first_unique_pairs_([], [], _, []).
js_first_unique_pairs_([K|Ks], [C|Cs], Seen, [C|Out]) :-
    \+ memberchk(K, Seen), !,
    js_first_unique_pairs_(Ks, Cs, [K|Seen], Out).
js_first_unique_pairs_([_|Ks], [_|Cs], Seen, Out) :-
    js_first_unique_pairs_(Ks, Cs, Seen, Out).

js_rebuild_try_chain([Only], _Alt, Only) :- !.
js_rebuild_try_chain([C1|Rest], Alt, Lines) :-
    format(string(Try), 'try_me_else ~w', [Alt]),
    append([Try|C1], Mid, Lines),
    js_rebuild_try_rest(Rest, Alt, Mid).

js_rebuild_try_rest([Last], _Alt, ["trust_me"|Last]) :- !.
js_rebuild_try_rest([C|Rest], Alt, [Retry|Lines]) :-
    format(string(Retry), 'retry_me_else ~w', [Alt]),
    append(C, Mid, Lines),
    js_rebuild_try_rest(Rest, Alt, Mid).

% T5: distinct first-argument constants. Bound A1 dispatches natively;
% unbound A1 falls back to clause-1 inline + interpreter from the alt label.
classify_clause_shape([FirstLine|Rest], plan(clause_chain, AltAtom, chain_payload(Guards, ClauseLines))) :-
    tokenize_line(FirstLine, ["try_me_else", AltStr]),
    js_chain_terms([FirstLine|Rest], Terms),
    clause_chain(Terms, chain(Guards)),
    forall(member(guard(_, Rem), Guards), js_chain_rem_supported(Rem)),
    !,
    atom_string(AltAtom, AltStr),
    take_clause1_lines(Rest, ClauseLines).
% T4: every clause is a supported deterministic body, including inner
% if-then-else (structure_ite). Used for list-recursion helpers such as
% first_char_index/4 (nil vs cons + ITE) that would otherwise become
% multi_clause_1 and keep a choice point per recursive step.
classify_clause_shape([FirstLine|Rest], plan(multi_clause_n, none, Structured)) :-
    tokenize_line(FirstLine, ["try_me_else", _AltStr]),
    js_t4_structured_clauses([FirstLine|Rest], Structured),
    !.
% T4 (flat): no inner ITE; every remaining line is a supported instr.
classify_clause_shape([FirstLine|Rest], plan(multi_clause_n, none, Clauses)) :-
    tokenize_line(FirstLine, ["try_me_else", _AltStr]),
    js_split_clause_lines([FirstLine|Rest], Clauses),
    Clauses = [_, _ | _],
    forall(member(Cl, Clauses),
           ( forall(member(Line, Cl), js_t4_clause_line_supported(Line)),
             last(Cl, LastLine), js_t4_terminal_line(LastLine) )),
    !.
classify_clause_shape([FirstLine|Rest], plan(multi_clause_1, AltAtom, ClauseLines)) :-
    tokenize_line(FirstLine, ["try_me_else", AltStr]), !,
    atom_string(AltAtom, AltStr),
    take_clause1_lines(Rest, ClauseLines).
classify_clause_shape(Lines, plan(ite, none, Structured)) :-
    js_parse_terms(Lines, Terms),
    structure_ite(Terms, Structured),
    member(ite(_, _, _), Structured),
    \+ member(try_me_else(_), Structured),
    \+ member(trust_me, Structured),
    !.
classify_clause_shape(Lines, plan(deterministic, none, Lines)).

js_parse_terms([], []).
js_parse_terms([Line|Rest], Terms) :-
    tokenize_line(Line, Parts),
    (   Parts == []
    ->  js_parse_terms(Rest, Terms)
    ;   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  sub_string(First, 0, _, 1, LabelName),
        Terms = [label(LabelName)|More],
        js_parse_terms(Rest, More)
    ;   js_line_term(Parts, T),
        Terms = [T|More],
        js_parse_terms(Rest, More)
    ).

js_line_term(["try_me_else", L], try_me_else(L)) :- !.
js_line_term(["trust_me"], trust_me) :- !.
js_line_term(["jump", L], jump(L)) :- !.
js_line_term(["cut_ite"], cut_ite) :- !.
js_line_term(["cut", Yn], cut(Yn)) :- !.
js_line_term(["builtin_call", Op, Ar], builtin_call(Op, Ar)) :- !.
js_line_term(Parts, line(Parts)).

js_chain_terms([], []).
js_chain_terms([Line|Rest], Terms) :-
    tokenize_line(Line, Parts),
    (   Parts == []
    ->  js_chain_terms(Rest, Terms)
    ;   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  js_chain_terms(Rest, Terms)
    ;   js_chain_term(Parts, T),
        Terms = [T|More],
        js_chain_terms(Rest, More)
    ).

js_chain_term(["try_me_else", L], try_me_else(L)) :- !.
js_chain_term(["retry_me_else", L], retry_me_else(L)) :- !.
js_chain_term(["trust_me"], trust_me) :- !.
js_chain_term(["get_constant", V, A], get_constant(V, A)) :- !.
js_chain_term(Parts, line(Parts)).

js_chain_rem_supported([]).
js_chain_rem_supported([T|Rest]) :-
    ( T = get_constant(_, _) -> true
    ; T = line(Parts) -> parts_supported(Parts)
    ),
    js_chain_rem_supported(Rest).

take_clause1_lines([], []).
take_clause1_lines([Line|Rest], Out) :-
    tokenize_line(Line, Parts),
    (   Parts == ["proceed"] -> Out = [Line]
    ;   Parts == ["trust_me"] -> Out = []
    ;   Out = [Line|More],
        take_clause1_lines(Rest, More)
    ).

js_prefix_switch_line(Line) :-
    tokenize_line(Line, Parts),
    Parts = [Op|_],
    member(Op, ["switch_on_constant", "switch_on_constant_fallthrough",
                "switch_on_constant_a2", "switch_on_constant_a2_fallthrough",
                "switch_on_structure", "switch_on_structure_a2",
                "switch_on_term", "switch_on_term_a2"]).

js_t4_structured_clauses(Lines, Structured) :-
    exclude(js_index_dispatch_line, Lines, K1),
    exclude(js_prefix_switch_line, K1, Kept),
    js_skip_clause_sep(Kept, Bodies),
    js_split_try_trust_bodies(Bodies, Clauses),
    Clauses = [_, _ | _],
    maplist(js_clause_to_structured, Clauses, Structured).

js_clause_to_structured(ClauseLines, Structured) :-
    js_parse_terms(ClauseLines, Terms),
    structure_ite(Terms, Structured),
    Structured \== [],
    forall(member(I, Structured), js_struct_supported(I)).

% Split a try_me_else/trust_me chain on *clause* proceed/fail, leaving
% inner ITE try_me_else blocks intact for structure_ite/2.
js_split_try_trust_bodies([], []).
js_split_try_trust_bodies(Lines, [Clause|More]) :-
    js_take_to_proceed(Lines, Clause, After0),
    Clause \== [],
    js_skip_clause_sep(After0, After1),
    js_split_try_trust_bodies(After1, More).

js_take_to_proceed([Line|Rest], [Line], Rest) :-
    tokenize_line(Line, Parts),
    ( Parts == ["proceed"] ; Parts == ["fail"] ), !.
js_take_to_proceed([Line|Rest], More, After) :-
    tokenize_line(Line, []), !,
    js_take_to_proceed(Rest, More, After).
js_take_to_proceed([Line|Rest], [Line|More], After) :-
    js_take_to_proceed(Rest, More, After).
js_take_to_proceed([], [], []).

js_skip_clause_sep([], []).
js_skip_clause_sep([Line|Rest], Out) :-
    tokenize_line(Line, Parts),
    (   Parts == []
    ->  js_skip_clause_sep(Rest, Out)
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":")
    ->  js_skip_clause_sep(Rest, Out)
    ;   ( Parts = ["trust_me"] ; Parts = ["retry_me_else"|_] ; Parts = ["try_me_else"|_] )
    ->  Out = Rest
    ;   Out = [Line|Rest]
    ).

js_split_clause_lines(AllLines, Clauses) :-
    include(js_t4_instr_line, AllLines, InstrLines),
    js_split_at_terminal(InstrLines, Clauses).

js_t4_instr_line(Line) :-
    tokenize_line(Line, Parts),
    Parts \== [],
    Parts = [F|_],
    \+ sub_string(F, _, 1, 0, ":"),
    \+ member(F, ["try_me_else", "retry_me_else", "trust_me",
                  "try", "retry", "trust",
                  "switch_on_constant", "switch_on_constant_fallthrough",
                  "switch_on_constant_a2", "switch_on_constant_a2_fallthrough",
                  "switch_on_structure", "switch_on_structure_a2",
                  "switch_on_term", "switch_on_term_a2"]).

js_split_at_terminal([], []).
js_split_at_terminal([L|Ls], [Clause|Rest]) :-
    js_take_to_terminal([L|Ls], Clause, After),
    ( After == [] -> Rest = [] ; js_split_at_terminal(After, Rest) ).

js_take_to_terminal([Line|Rest], [Line], Rest) :-
    tokenize_line(Line, Parts),
    ( Parts == ["proceed"] ; Parts == ["fail"] ), !.
js_take_to_terminal([Line|Rest], [Line|More], After) :-
    js_take_to_terminal(Rest, More, After).
js_take_to_terminal([], [], []).

js_t4_clause_line_supported(Line) :-
    tokenize_line(Line, Parts),
    ( Parts == [] -> true ; parts_supported(Parts) ),
    Parts \= ["cut_ite"|_],
    Parts \= ["jump"|_].

js_t4_terminal_line(Line) :-
    tokenize_line(Line, Parts),
    ( Parts == ["proceed"] ; Parts == ["fail"] ).

wam_javascript_lowerable(PI, WamCode, Reason) :-
    js_pi_key(PI, PredName),
    % Execute of a *different* predicate cannot be a nested Runtime.run
    % (cp=0 steals the query continuation; GP-PERF mixed-mode). Keep
    % those wrappers on the interpreter; Call of a lowered helper is fine.
    \+ js_wam_has_foreign_execute(WamCode, PredName),
    catch(build_emission_plan(WamCode, plan(Reason, _, Payload)), _, fail),
    % multi_clause_1 inlines clause 1 then Runtime.run from pc+1, which is
    % the caller's next instruction rather than the alt clause. Wrong-but-
    % fast is a failure; leave these on the interpreter until T4+ITE fits.
    Reason \== multi_clause_1,
    (   Reason == ite
    ->  forall(member(I, Payload), js_struct_supported(I))
    ;   Reason == clause_chain
    ->  Payload = chain_payload(Guards, ClauseLines),
        forall(member(guard(_, Rem), Guards), js_chain_rem_supported(Rem)),
        forall(member(Line, ClauseLines), line_supported(Line))
    ;   Reason == multi_clause_n
    ->  Payload = [C1|_],
        (   C1 = [S|_], string(S)
        ->  forall(member(Cl, Payload),
                   forall(member(Line, Cl), line_supported(Line)))
        ;   forall(member(Cl, Payload),
                   forall(member(I, Cl), js_struct_supported(I)))
        )
    ;   forall(member(Line, Payload), line_supported(Line))
    ).

js_struct_supported(ite(C, T, E)) :- !,
    forall(member(I, C), js_struct_supported(I)),
    forall(member(I, T), js_struct_supported(I)),
    forall(member(I, E), js_struct_supported(I)).
js_struct_supported(builtin_call(_, _)) :- !.
js_struct_supported(line(Parts)) :- !,
    ( Parts == [] -> true ; parts_supported(Parts) ).
js_struct_supported(_) :- fail.

line_supported(Line) :-
    tokenize_line(Line, Parts),
    (Parts == [] -> true ; Parts = [F|_], sub_string(F, _, 1, 0, ":") -> true ; parts_supported(Parts)).

parts_supported(["allocate"]).
parts_supported(["deallocate"]).
parts_supported(["get_level", _]).
parts_supported(["proceed"]).
parts_supported(["fail"]).
parts_supported(["get_constant", _, _]).
parts_supported(["get_variable", _, _]).
parts_supported(["get_value", _, _]).
parts_supported(["get_structure", _, _]).
parts_supported(["get_list", _]).
parts_supported(["get_nil", _]).
parts_supported(["get_integer", _, _]).
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

js_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, S),
    string_codes(S, Codes),
    maplist(js_safe_code, Codes, Safe),
    string_codes(SafeS, Safe),
    format(atom(Name), 'lowered_~w_~w', [SafeS, Arity]).

js_safe_code(C, C) :-
    (C >= 0'a, C =< 0'z ; C >= 0'A, C =< 0'Z ; C >= 0'0, C =< 0'9 ; C =:= 0'_), !.
js_safe_code(_, 0'_).

lower_predicate_to_javascript(PI, WamCode, Options, lowered(PredName, FuncName, Code)) :-
    (PI = _M:Pred/Arity -> true ; PI = Pred/Arity),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    js_lowered_func_name(Pred/Arity, FuncName),
    build_emission_plan(WamCode, plan(Mode, AltLabel, Payload)),
    (   Mode == deterministic
    ->  emit_deterministic_function(PredName, FuncName, Payload, Code)
    ;   Mode == ite
    ->  emit_ite_function(PredName, FuncName, Payload, Code)
    ;   Mode == clause_chain
    ->  Payload = chain_payload(Guards, Clause1Lines),
        emit_clause_chain_function(PredName, FuncName, AltLabel, Guards, Clause1Lines, Options, Code)
    ;   Mode == multi_clause_n
    ->  emit_multi_clause_n_function(PredName, FuncName, Payload, Code)
    ;   emit_multi_clause_function(PredName, FuncName, AltLabel, Payload, Code)
    ).

emit_multi_clause_n_function(PredName, FuncName, Clauses, Code) :-
    with_output_to(string(ClausesBody), emit_js_t4_payload(Clauses)),
    format(string(Code),
'// Lowered: ~w (T4 all-clauses inline)
function ~w(program, state) {
  const _t4_trail = state.trail.length;
  const _t4_regs = Runtime.copy_table(state.regs);
  const _t4_vc = state.var_counter;
  const _t4_stack = state.stack.slice();
  const _t4_ysave = (state.y_save || []).slice();
  const _t4_mode = state.mode;
  const _t4_build = state.build_stack.slice();
  const _t4_rstack = (state.read_stack || []).slice();
  const _t4_rargs = state.read_args;
  const _t4_rcur = state.read_cursor;
~w  return false;
}
', [PredName, FuncName, ClausesBody]).

emit_js_t4_payload([]).
emit_js_t4_payload([Clause|Rest]) :-
    format("  if ((function () {~n"),
    (   Clause = [S|_], string(S)
    ->  emit_lines(Clause, "    ")
    ;   emit_struct_js(Clause, "    ")
    ),
    format("    return false;~n"),
    format("  })()) return true;~n"),
    format("  while (state.trail.length > _t4_trail) { const _n = state.trail.pop(); delete state.bindings[_n]; }~n"),
    format("  state.regs = Runtime.copy_table(_t4_regs);~n"),
    format("  state.var_counter = _t4_vc;~n"),
    format("  state.stack = _t4_stack.slice();~n"),
    format("  state.y_save = _t4_ysave.slice();~n"),
    format("  state.mode = _t4_mode;~n"),
    format("  state.build_stack = _t4_build.slice();~n"),
    format("  state.read_stack = _t4_rstack.slice();~n"),
    format("  state.read_args = _t4_rargs;~n"),
    format("  state.read_cursor = _t4_rcur;~n"),
    emit_js_t4_payload(Rest).

emit_js_t4_clauses(Clauses) :-
    emit_js_t4_payload(Clauses).

emit_ite_function(PredName, FuncName, Structured, Code) :-
    with_output_to(string(Body), emit_struct_js(Structured, "  ")),
    format(string(Code),
'// Lowered: ~w (if-then-else / negation / once)
function ~w(program, state) {
~w  return true;
}
', [PredName, FuncName, Body]).

emit_struct_js([], _).
emit_struct_js([Item|Rest], Ind) :-
    emit_struct_item_js(Item, Ind),
    emit_struct_js(Rest, Ind).

emit_struct_item_js(ite(Cond, Then, Else), Ind) :- !,
    string_concat(Ind, "    ", Ind4),
    format("~w{~n", [Ind]),
    format("~w  const _ite_snap = Runtime.snapshot_machine(state);~n", [Ind]),
    format("~w  const _ite_cps = state.cps.length;~n", [Ind]),
    format("~w  const _ite_cond = (function () {~n", [Ind]),
    emit_struct_js(Cond, Ind4),
    format("~w    return true;~n", [Ind]),
    format("~w  })();~n", [Ind]),
    format("~w  if (_ite_cond) {~n", [Ind]),
    emit_struct_js(Then, Ind4),
    format("~w  } else {~n", [Ind]),
    format("~w    Runtime.restore_machine(state, _ite_snap);~n", [Ind]),
    format("~w    while (state.cps.length > _ite_cps) state.cps.pop();~n", [Ind]),
    emit_struct_js(Else, Ind4),
    format("~w  }~n", [Ind]),
    format("~w}~n", [Ind]).
emit_struct_item_js(builtin_call(Op, Ar), Ind) :- !,
    emit_line_parts(["builtin_call", Op, Ar], Ind).
emit_struct_item_js(line(Parts), Ind) :- !,
    emit_line_parts(Parts, Ind).

emit_deterministic_function(PredName, FuncName, Lines, Code) :-
    wam_javascript_target:js_string_literal(PredName, PredQ),
    js_pred_arity_from_name(PredName, Arity),
    js_ground_fact_lines(Lines), !,
    with_output_to(string(Body), emit_lines_skip_proceed(Lines, "  ")),
    format(string(Code),
'// Lowered: ~w (deterministic ground fact; interned after first success)
function ~w(program, state) {
  const _gk = ~w;
  const _gm = program.ground_memo || (program.ground_memo = Object.create(null));
  const _cached = _gm[_gk];
  if (_cached !== undefined) {
    // Trail-safety: _cached is a copy_term snapshot of a fully ground
    // answer. It contains no unbound cells, so it is never trailed;
    // unify only binds the caller''s registers (or compares ground-to-ground).
    // Sharing the interned object across calls is sound because get_structure
    // / unify in read mode do not mutate args arrays, and a later trail undo
    // only deletes caller bindings.
    for (let _i = 0; _i < _cached.length; _i++) {
      if (Runtime.unify(state, Runtime.get_reg(state, _i + 1), _cached[_i], program) !== true) return false;
    }
    return true;
  }
~w  const _snap = [];
  for (let _i = 0; _i < ~w; _i++) {
    const _ti = Runtime.deref(state, Runtime.get_reg(state, _i + 1));
    if (!Runtime.term_is_ground(state, _ti)) return true;
    _snap.push(Runtime.copy_term(state, _ti));
  }
  _gm[_gk] = _snap;
  return true;
}
', [PredName, FuncName, PredQ, Body, Arity]).
emit_deterministic_function(PredName, FuncName, Lines, Code) :-
    with_output_to(string(Body), emit_lines(Lines, "  ")),
    format(string(Code),
'// Lowered: ~w (deterministic)
function ~w(program, state) {
~w  return true;
}
', [PredName, FuncName, Body]).

emit_multi_clause_function(PredName, FuncName, AltLabel, Lines, Code) :-
    with_output_to(string(Body), emit_lines(Lines, "    ")),
    wam_javascript_target:js_string_literal(AltLabel, AltQ),
    format(string(Code),
'// Lowered: ~w (multi-clause; clause 1 inline, array fallback)
function ~w(program, state) {
  const alt_pc = program.labels[~w];
  if (alt_pc === undefined || alt_pc === null) return false;
  const _cp = {
    next_pc: alt_pc,
    regs: Runtime.copy_table(state.regs),
    cp: state.cp,
    trail_len: state.trail.length,
    var_counter: state.var_counter,
    mode: state.mode,
    build_stack: state.build_stack.slice(),
    stack: state.stack.slice(),
    read_stack: (state.read_stack || []).slice(),
    read_args: state.read_args,
    read_cursor: state.read_cursor,
    y_save: (state.y_save || []).slice()
  };
  state.cps.push(_cp);
  const ok = (function () {
~w    return false;
  })();
    if (ok === true) return true;
    if (Runtime.backtrack(state) !== true) return false;
    // backtrack already set pc to next_pc (the alt clause).
    state.halt = false;
    state.program = program;
    return Runtime.run(program, state) === true;
}
', [PredName, FuncName, AltQ, Body]).

emit_clause_chain_function(PredName, FuncName, AltLabel, Guards, Clause1Lines, Options, Code) :-
    js_t6_applicable(Guards, Options), !,
    with_output_to(string(Table), js_emit_t6_table(Guards, "    ")),
    with_output_to(string(Clause1Body), emit_lines(Clause1Lines, "      ")),
    wam_javascript_target:js_string_literal(AltLabel, AltQ),
    format(string(Code),
'// Lowered: ~w (T6 first-argument indexing)
const ~w = (function () {
  const _t6 = {
~w  };
  return function (program, state) {
    const t5a1 = Runtime.deref(state, Runtime.get_reg(state, 1));
    if (t5a1 && t5a1.tag === "atom") {
      const _f = _t6[t5a1.id];
      if (_f !== undefined) return _f(program, state);
      return false;
    }
    if (t5a1 && typeof t5a1 === "object" && t5a1.tag !== "unbound") return false;
    const alt_pc = program.labels[~w];
    if (alt_pc === undefined || alt_pc === null) return false;
    const _cp = {
      next_pc: alt_pc,
      regs: Runtime.copy_table(state.regs),
      cp: state.cp,
      trail_len: state.trail.length,
      var_counter: state.var_counter,
      mode: state.mode,
      build_stack: state.build_stack.slice(),
      stack: state.stack.slice(),
      read_stack: (state.read_stack || []).slice(),
      read_args: state.read_args,
      read_cursor: state.read_cursor,
      y_save: (state.y_save || []).slice()
    };
    state.cps.push(_cp);
    const ok = (function () {
~w      return false;
    })();
    if (ok === true) return true;
    if (Runtime.backtrack(state) !== true) return false;
    // backtrack already set pc to next_pc (the alt clause).
    state.halt = false;
    state.program = program;
    return Runtime.run(program, state) === true;
  };
})();
', [PredName, FuncName, Table, AltQ, Clause1Body]).
emit_clause_chain_function(PredName, FuncName, AltLabel, Guards, Clause1Lines, _Options, Code) :-
    with_output_to(string(Dispatch), js_emit_chain_guards(Guards, "    ")),
    with_output_to(string(Clause1Body), emit_lines(Clause1Lines, "    ")),
    wam_javascript_target:js_string_literal(AltLabel, AltQ),
    format(string(Code),
'// Lowered: ~w (T5 first-argument dispatch)
function ~w(program, state) {
  const t5a1 = Runtime.deref(state, Runtime.get_reg(state, 1));
  if (t5a1 && typeof t5a1 === "object" && t5a1.tag !== "unbound") {
~w    return false;
  }
  const alt_pc = program.labels[~w];
  if (alt_pc === undefined || alt_pc === null) return false;
  const _cp = {
    next_pc: alt_pc,
    regs: Runtime.copy_table(state.regs),
    cp: state.cp,
    trail_len: state.trail.length,
    var_counter: state.var_counter,
    mode: state.mode,
    build_stack: state.build_stack.slice(),
    stack: state.stack.slice(),
    read_stack: (state.read_stack || []).slice(),
    read_args: state.read_args,
    read_cursor: state.read_cursor,
    y_save: (state.y_save || []).slice()
  };
  state.cps.push(_cp);
  const ok = (function () {
~w    return false;
  })();
  if (ok === true) return true;
  if (Runtime.backtrack(state) !== true) return false;
  // backtrack already set pc to next_pc (the alt clause).
  state.halt = false;
  state.program = program;
  return Runtime.run(program, state) === true;
}
', [PredName, FuncName, Dispatch, AltQ, Clause1Body]).

js_t6_applicable(Guards, Options) :-
    js_t6_min_clauses(Options, Min),
    length(Guards, N), N >= Min,
    forall(member(guard(V, _), Guards), js_t6_atom_id(V, _)).

js_t6_min_clauses(Options, N) :-
    ( member(t6_min_clauses(N), Options) -> true ; N = 8 ).

js_t6_atom_id(V, Id) :-
    \+ wam_constant_token_is_string(V),
    wam_classify_constant_token(V, atom(Name)),
    wam_javascript_target:intern_js_atom(Name, Id).

js_emit_t6_table([], _).
js_emit_t6_table([guard(V, Rem)|Rest], Ind) :-
    js_t6_atom_id(V, Id),
    format("~w~w: function (program, state) {~n", [Ind, Id]),
    string_concat(Ind, "  ", Ind2),
    js_emit_chain_rem(Rem, Ind2),
    format("~w  return false;~n", [Ind]),
    format("~w},~n", [Ind]),
    js_emit_t6_table(Rest, Ind).

js_emit_chain_guards([], _).
js_emit_chain_guards([guard(V, Rem)|Rest], Ind) :-
    js_chain_eq_expr(V, "t5a1", Eq),
    format("~wif (~w) {~n", [Ind, Eq]),
    string_concat(Ind, "  ", Ind2),
    js_emit_chain_rem(Rem, Ind2),
    format("~w}~n", [Ind]),
    js_emit_chain_guards(Rest, Ind).

js_emit_chain_rem([], _).
js_emit_chain_rem([T|Rest], Ind) :-
    js_emit_chain_term(T, Ind),
    js_emit_chain_rem(Rest, Ind).

js_emit_chain_term(get_constant(C, R), Ind) :- !, emit_line_parts(["get_constant", C, R], Ind).
js_emit_chain_term(line(Parts), Ind) :- !, emit_line_parts(Parts, Ind).

js_chain_eq_expr(VStr, Var, Expr) :-
    (   wam_constant_token_is_string(VStr)
    ->  wam_classify_constant_token(VStr, atom(Name)),
        wam_javascript_target:js_string_literal(Name, Lit),
        format(string(Expr),
               '~w && ~w.tag === "string" && ~w.val === ~w',
               [Var, Var, Var, Lit])
    ;   wam_classify_constant_token(VStr, Class),
        (   Class = integer(N)
        ->  format(string(Expr), '~w && ~w.tag === "int" && ~w.val === ~w', [Var, Var, Var, N])
        ;   Class = float(F)
        ->  format(string(Expr), '~w && ~w.tag === "float" && ~w.val === ~w', [Var, Var, Var, F])
        ;   Class = atom(Name),
            wam_javascript_target:intern_js_atom(Name, Id),
            format(string(Expr), '~w && ~w.tag === "atom" && ~w.id === ~w', [Var, Var, Var, Id])
        )
    ).

emit_lines([], _).
emit_lines([Line|Rest], Ind) :-
    tokenize_line(Line, Parts),
    (   Parts == [] -> true
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":") -> true
    ;   emit_line_parts(Parts, Ind)
    ),
    emit_lines(Rest, Ind).

emit_line_parts(["proceed"], I) :- !, format("~wreturn true;~n", [I]).
emit_line_parts(["fail"], I) :- !, format("~wreturn false;~n", [I]).
emit_line_parts(["call", PredArity], I) :- !, emit_call(PredArity, I).
emit_line_parts(["call", Pred, ArityStr], I) :- !,
    strip_arity_local(Pred, Name), format(string(PA), "~w/~w", [Name, ArityStr]), emit_call(PA, I).
emit_line_parts(["execute", PredArity], I) :- !, emit_execute(PredArity, I).
emit_line_parts(["execute", Pred, ArityStr], I) :- !,
    strip_arity_local(Pred, Name), format(string(PA), "~w/~w", [Name, ArityStr]), emit_execute(PA, I).
% Allocate / Deallocate / GetLevel go through Runtime.step so the JS
% Y-register snapshot convention (not Lua's locals = {}) is preserved.
emit_line_parts(["allocate"], I) :- !,
    format("~wif (Runtime.step(program, state, I.Allocate()) !== true) return false;~n", [I]).
emit_line_parts(["deallocate"], I) :- !,
    format("~wif (Runtime.step(program, state, I.Deallocate()) !== true) return false;~n", [I]).
emit_line_parts(["get_level", Yn], I) :- !,
    wam_javascript_target:wam_parts_to_js(["get_level", Yn], [], Lit),
    format("~wif (Runtime.step(program, state, ~w) !== true) return false;~n", [I, Lit]).
emit_line_parts(["put_constant", C, R], I) :- !,
    wam_javascript_target:reg_to_int(R, RI),
    wam_javascript_target:constant_to_js_term(C, T),
    format("~wRuntime.put_reg(state, ~w, ~w);~n", [I, RI, T]).
emit_line_parts(["put_variable", X, A], I) :- !,
    wam_javascript_target:reg_to_int(X, XI),
    wam_javascript_target:reg_to_int(A, AI),
    format("~w{ const v = Runtime.new_var(state); Runtime.put_reg(state, ~w, v); Runtime.put_reg(state, ~w, v); }~n", [I, XI, AI]).
emit_line_parts(["put_value", X, A], I) :- !,
    wam_javascript_target:reg_to_int(X, XI),
    wam_javascript_target:reg_to_int(A, AI),
    format("~wRuntime.put_reg(state, ~w, Runtime.get_reg(state, ~w));~n", [I, AI, XI]).
emit_line_parts(["get_variable", X, A], I) :- !,
    wam_javascript_target:reg_to_int(X, XI),
    wam_javascript_target:reg_to_int(A, AI),
    format("~wRuntime.put_reg(state, ~w, Runtime.get_reg(state, ~w));~n", [I, XI, AI]).
emit_line_parts(Parts, I) :-
    wam_javascript_target:wam_parts_to_js(Parts, [], Lit),
    format("~wif (Runtime.step(program, state, ~w) !== true) return false;~n", [I, Lit]).

emit_call(PredArity, I) :-
    wam_javascript_target:js_string_literal(PredArity, Q),
    parse_call_pred_arity(PredArity, PredName, Arity),
    wam_javascript_target:js_string_literal(PredName, PQ),
    format("~w{~n", [I]),
    format("~w  const saved_cp = state.cp;~n", [I]),
    format("~w  const saved_pc = state.pc;~n", [I]),
    format("~w  const _lf = (program.lowered_dispatch && program.lowered_dispatch[~w]) || ((typeof lowered_dispatch !== \"undefined\") ? lowered_dispatch[~w] : undefined);~n", [I, Q, Q]),
    format("~w  let _ok = true;~n", [I]),
    format("~w  if (typeof _lf === \"function\") {~n", [I]),
    format("~w    _ok = _lf(program, state) === true;~n", [I]),
    format("~w  } else {~n", [I]),
    format("~w    const target = program.labels[~w];~n", [I, Q]),
    format("~w    if (target !== undefined && target !== null) {~n", [I]),
    format("~w      Runtime.push_y_save(state);~n", [I]),
    format("~w      state.cp = 0;~n", [I]),
    format("~w      state.pc = target;~n", [I]),
    format("~w      state.program = program;~n", [I]),
    format("~w      _ok = Runtime.run_isolated(program, state) === true;~n", [I]),
    format("~w      state.halt = false;~n", [I]),
    format("~w    } else if (Runtime.step(program, state, I.Call(~w, ~w)) !== true) {~n", [I, PQ, Arity]),
    format("~w      _ok = false;~n", [I]),
    format("~w    }~n", [I]),
    format("~w  }~n", [I]),
    format("~w  state.cp = saved_cp;~n", [I]),
    format("~w  state.pc = saved_pc;~n", [I]),
    format("~w  state.halt = false;~n", [I]),
    format("~w  if (!_ok) return false;~n", [I]),
    format("~w}~n", [I]).

emit_execute(PredArity, I) :-
    wam_javascript_target:js_string_literal(PredArity, Q),
    parse_call_pred_arity(PredArity, PredName, Arity),
    wam_javascript_target:js_string_literal(PredName, PQ),
    format("~w{~n", [I]),
    format("~w  const _lf = (program.lowered_dispatch && program.lowered_dispatch[~w]) || ((typeof lowered_dispatch !== \"undefined\") ? lowered_dispatch[~w] : undefined);~n", [I, Q, Q]),
    format("~w  if (typeof _lf === \"function\") return _lf(program, state) === true;~n", [I]),
    format("~w  const target = program.labels[~w];~n", [I, Q]),
    format("~w  if (target !== undefined && target !== null) {~n", [I]),
    format("~w    state.pc = target;~n", [I]),
    format("~w    state.program = program;~n", [I]),
    format("~w    return Runtime.run_isolated(program, state) === true;~n", [I]),
    format("~w  }~n", [I]),
    format("~w  if (Runtime.step(program, state, I.Execute(~w, ~w)) !== true) return false;~n", [I, PQ, Arity]),
    format("~w  if (state.halt) return true;~n", [I]),
    format("~w  state.program = program;~n", [I]),
    format("~w  return Runtime.run_isolated(program, state) === true;~n", [I]),
    format("~w}~n", [I]).

parse_call_pred_arity(PredArity, PredName, Arity) :-
    atom_string(PredArity, S),
    (   sub_string(S, B, 1, After, "/"),
        sub_string(S, 0, B, _, PredName0),
        sub_string(S, _, After, 0, ArStr),
        number_string(Arity, ArStr)
    ->  PredName = PredName0
    ;   PredName = S,
        Arity = 0
    ).

strip_arity_local(Tok, Name) :-
    (sub_string(Tok, B, 1, _, "/") -> sub_string(Tok, 0, B, _, Name) ; Name = Tok).

emit_lines_skip_proceed([], _).
emit_lines_skip_proceed([Line|Rest], Ind) :-
    tokenize_line(Line, Parts),
    (   Parts == ["proceed"] -> true
    ;   Parts == [] -> true
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":") -> true
    ;   emit_line_parts(Parts, Ind)
    ),
    emit_lines_skip_proceed(Rest, Ind).

js_ground_fact_lines(Lines) :-
    Lines \== [],
    forall(member(Line, Lines), js_ground_fact_line(Line)).

js_ground_fact_line(Line) :-
    tokenize_line(Line, Parts),
    (   Parts == [] -> true
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":") -> true
    ;   Parts = [Op|_],
        memberchk(Op, ["proceed",
                       "get_constant", "get_variable", "get_value",
                       "get_structure", "get_list", "get_nil", "get_integer",
                       "unify_variable", "unify_value", "unify_constant",
                       "put_constant", "put_variable", "put_value",
                       "put_structure", "put_list",
                       "set_variable", "set_value", "set_constant"])
    ).

js_pred_arity_from_name(PredName, Arity) :-
    atom_string(PredName, S),
    split_string(S, "/", "", Parts),
    last(Parts, ArStr),
    number_string(Arity, ArStr).

%% wam_javascript_explain_lower(+PI, +WamCode, -Decision)
%  Decision = lower(Reason) | fallback(Why). Never fails.
wam_javascript_explain_lower(PI, WamCode, fallback('execute of a non-self callee (nested Runtime.run would steal CP)')) :-
    js_pi_key(PI, Key),
    js_wam_has_foreign_execute(WamCode, Key), !.
wam_javascript_explain_lower(PI, WamCode, lower(Reason)) :-
    catch(wam_javascript_lowerable(PI, WamCode, Reason), _, fail), !.
wam_javascript_explain_lower(_PI, WamCode, fallback(Why)) :-
    catch(build_emission_plan(WamCode, plan(Mode, _, Payload)), E,
          (Why = error(E), !)),
    (   Mode == ite
    ->  (   member(I, Payload), \+ js_struct_supported(I)
        ->  Why = 'ite: unsupported structured instruction'
        ;   Why = 'ite: structure_ite leftover'
        )
    ;   Mode == deterministic
    ->  (   member(Line, Payload), \+ line_supported(Line)
        ->  Why = 'deterministic: unsupported instruction (jump/try/cut/...)'
        ;   Why = 'deterministic: emit failed'
        )
    ;   Mode == multi_clause_1
    ->  Why = 'multi_clause_1 would keep interpreter CPs; T4+ITE did not match'
    ;   Why = Mode
    ), !.
wam_javascript_explain_lower(_, _, fallback('not classified')).

js_pi_key(_M:Pred/Arity, Key) :- !,
    format(atom(Key), '~w/~w', [Pred, Arity]).
js_pi_key(Pred/Arity, Key) :-
    format(atom(Key), '~w/~w', [Pred, Arity]).

%% js_wam_has_foreign_execute(+WamCode, +SelfKey)
%  True when some Execute targets a predicate other than SelfKey.
%  Self-recursive last-goal Execute (T4 merge_flags_/3) is allowed.
js_wam_has_foreign_execute(WamCode, Self) :-
    atom_string(Self, SelfS),
    atom_string(WamCode, S),
    split_string(S, "\n", "", Lines),
    member(Line, Lines),
    tokenize_line(Line, Parts),
    js_execute_target(Parts, TargetS),
    TargetS \== SelfS.

js_execute_target(["execute", PA], TargetS) :-
    atom_string(PA, TargetS).
js_execute_target(["execute", Pred, ArityStr], TargetS) :-
    format(string(TargetS), "~w/~w", [Pred, ArityStr]).
