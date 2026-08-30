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
    lower_predicate_to_javascript/4,
    js_lowered_func_name/2
]).

:- use_module(library(lists)).
:- use_module(wam_ite_structurer, [structure_ite/2]).
:- use_module(wam_clause_chain, [clause_chain/2]).
:- use_module(wam_text_parser, [
    wam_tokenize_line/2,
    wam_classify_constant_token/2
]).

tokenize_line(Line, Parts) :-
    wam_tokenize_line(Line, Parts).

build_emission_plan(WamCode, plan(Mode, AltLabel, ClauseLines)) :-
    atom_string(WamCode, S),
    split_string(S, "\n", "", Lines),
    skip_to_first_real_instr(Lines, Filtered),
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
% T4: every clause is a supported deterministic body (no distinct A1 key).
% Try each clause inline with trail/register restore; never enter the interpreter.
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

js_split_clause_lines(AllLines, Clauses) :-
    include(js_t4_instr_line, AllLines, InstrLines),
    js_split_at_terminal(InstrLines, Clauses).

js_t4_instr_line(Line) :-
    tokenize_line(Line, Parts),
    Parts \== [],
    Parts = [F|_],
    \+ sub_string(F, _, 1, 0, ":"),
    \+ member(F, ["try_me_else", "retry_me_else", "trust_me",
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

wam_javascript_lowerable(_PI, WamCode, Reason) :-
    catch(build_emission_plan(WamCode, plan(Reason, _, Payload)), _, fail),
    (   Reason == ite
    ->  forall(member(I, Payload), js_struct_supported(I))
    ;   Reason == clause_chain
    ->  Payload = chain_payload(Guards, ClauseLines),
        forall(member(guard(_, Rem), Guards), js_chain_rem_supported(Rem)),
        forall(member(Line, ClauseLines), line_supported(Line))
    ;   Reason == multi_clause_n
    ->  forall(member(Cl, Payload),
               forall(member(Line, Cl), line_supported(Line)))
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
    with_output_to(string(ClausesBody), emit_js_t4_clauses(Clauses)),
    format(string(Code),
'// Lowered: ~w (T4 all-clauses inline)
function ~w(program, state) {
  const _t4_trail = state.trail.length;
  const _t4_regs = Runtime.copy_table(state.regs);
  const _t4_vc = state.var_counter;
~w  return false;
}
', [PredName, FuncName, ClausesBody]).

emit_js_t4_clauses([]).
emit_js_t4_clauses([Clause|Rest]) :-
    format("  if ((function () {~n"),
    emit_lines(Clause, "    "),
    format("    return false;~n"),
    format("  })()) return true;~n"),
    format("  while (state.trail.length > _t4_trail) { const _n = state.trail.pop(); delete state.bindings[_n]; }~n"),
    format("  state.regs = Runtime.copy_table(_t4_regs);~n"),
    format("  state.var_counter = _t4_vc;~n"),
    emit_js_t4_clauses(Rest).

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
    format("~w  const _ite_mark = state.trail.length;~n", [Ind]),
    format("~w  const _ite_cond = (function () {~n", [Ind]),
    emit_struct_js(Cond, Ind4),
    format("~w    return true;~n", [Ind]),
    format("~w  })();~n", [Ind]),
    format("~w  if (_ite_cond) {~n", [Ind]),
    emit_struct_js(Then, Ind4),
    format("~w  } else {~n", [Ind]),
    format("~w    while (state.trail.length > _ite_mark) { const _n = state.trail.pop(); delete state.bindings[_n]; }~n", [Ind]),
    emit_struct_js(Else, Ind4),
    format("~w  }~n", [Ind]),
    format("~w}~n", [Ind]).
emit_struct_item_js(builtin_call(Op, Ar), Ind) :- !,
    emit_line_parts(["builtin_call", Op, Ar], Ind).
emit_struct_item_js(line(Parts), Ind) :- !,
    emit_line_parts(Parts, Ind).

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
    read_cursor: state.read_cursor
  };
  state.cps.push(_cp);
  const ok = (function () {
~w    return false;
  })();
  if (ok === true) return true;
  if (Runtime.backtrack(state) !== true) return false;
  state.pc = state.pc + 1;
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
      read_cursor: state.read_cursor
    };
    state.cps.push(_cp);
    const ok = (function () {
~w      return false;
    })();
    if (ok === true) return true;
    if (Runtime.backtrack(state) !== true) return false;
    state.pc = state.pc + 1;
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
    read_cursor: state.read_cursor
  };
  state.cps.push(_cp);
  const ok = (function () {
~w    return false;
  })();
  if (ok === true) return true;
  if (Runtime.backtrack(state) !== true) return false;
  state.pc = state.pc + 1;
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
    wam_classify_constant_token(VStr, Class),
    (   Class = integer(N)
    ->  format(string(Expr), '~w && ~w.tag === "int" && ~w.val === ~w', [Var, Var, Var, N])
    ;   Class = float(F)
    ->  format(string(Expr), '~w && ~w.tag === "float" && ~w.val === ~w', [Var, Var, Var, F])
    ;   Class = atom(Name),
        wam_javascript_target:intern_js_atom(Name, Id),
        format(string(Expr), '~w && ~w.tag === "atom" && ~w.id === ~w', [Var, Var, Var, Id])
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
    format("~w{~n", [I]),
    format("~w  const saved_cp = state.cp;~n", [I]),
    format("~w  const _lf = (typeof lowered_dispatch !== \"undefined\") ? lowered_dispatch[~w] : undefined;~n", [I, Q]),
    format("~w  if (typeof _lf === \"function\") {~n", [I]),
    format("~w    if (_lf(program, state) !== true) return false;~n", [I]),
    format("~w  } else {~n", [I]),
    format("~w    const target = program.labels[~w];~n", [I, Q]),
    format("~w    if (target === undefined || target === null) return false;~n", [I]),
    format("~w    state.cp = 0;~n", [I]),
    format("~w    state.pc = target;~n", [I]),
    format("~w    state.program = program;~n", [I]),
    format("~w    if (Runtime.run(program, state) !== true) return false;~n", [I]),
    format("~w    state.halt = false;~n", [I]),
    format("~w  }~n", [I]),
    format("~w  state.cp = saved_cp;~n", [I]),
    format("~w}~n", [I]).

emit_execute(PredArity, I) :-
    wam_javascript_target:js_string_literal(PredArity, Q),
    format("~w{~n", [I]),
    format("~w  const _lf = (typeof lowered_dispatch !== \"undefined\") ? lowered_dispatch[~w] : undefined;~n", [I, Q]),
    format("~w  if (typeof _lf === \"function\") return _lf(program, state) === true;~n", [I]),
    format("~w  const target = program.labels[~w];~n", [I, Q]),
    format("~w  if (target === undefined || target === null) return false;~n", [I]),
    format("~w  state.pc = target;~n", [I]),
    format("~w  state.program = program;~n", [I]),
    format("~w  return Runtime.run(program, state) === true;~n", [I]),
    format("~w}~n", [I]).

strip_arity_local(Tok, Name) :-
    (sub_string(Tok, B, 1, _, "/") -> sub_string(Tok, 0, B, _, Name) ; Name = Tok).
