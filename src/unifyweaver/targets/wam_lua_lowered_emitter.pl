:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

:- module(wam_lua_lowered_emitter, [
    wam_lua_lowerable/3,
    lower_predicate_to_lua/4,
    lua_lowered_func_name/2
]).

:- use_module(library(lists)).
:- use_module(wam_ite_structurer, [structure_ite/2]).

build_emission_plan(WamCode, plan(Mode, AltLabel, ClauseLines)) :-
    atom_string(WamCode, S),
    split_string(S, "\n", "", Lines),
    skip_to_first_real_instr(Lines, Filtered),
    classify_clause_shape(Filtered, plan(Mode, AltLabel, ClauseLines)).

skip_to_first_real_instr([], []).
skip_to_first_real_instr([Line|Rest], Out) :-
    wam_lua_target:tokenize_wam_line(Line, Parts),
    (   skippable_prefix_line(Parts)
    ->  skip_to_first_real_instr(Rest, Out)
    ;   Out = [Line|Rest]
    ).

skippable_prefix_line([]).
skippable_prefix_line([First|_]) :- sub_string(First, _, 1, 0, ":").
skippable_prefix_line(["switch_on_constant"|_]).
skippable_prefix_line(["switch_on_structure"|_]).

classify_clause_shape([FirstLine|Rest], plan(multi_clause_1, AltAtom, ClauseLines)) :-
    wam_lua_target:tokenize_wam_line(FirstLine, ["try_me_else", AltStr]), !,
    atom_string(AltAtom, AltStr),
    take_clause1_lines(Rest, ClauseLines).
% Soft-cut block: an if-then-else / negation / once whose try_me_else is
% internal (preceded by the shared head-arg setup), not a clause separator.
% Fold it through the shared structurer into ite(Cond,Then,Else) terms.
classify_clause_shape(Lines, plan(ite, none, Structured)) :-
    lua_parse_terms(Lines, Terms),
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
lua_parse_terms([], []).
lua_parse_terms([Line|Rest], Terms) :-
    wam_lua_target:tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  lua_parse_terms(Rest, Terms)
    ;   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  sub_string(First, 0, _, 1, LabelName),
        Terms = [label(LabelName)|More],
        lua_parse_terms(Rest, More)
    ;   lua_line_term(Parts, T),
        Terms = [T|More],
        lua_parse_terms(Rest, More)
    ).

lua_line_term(["try_me_else", L], try_me_else(L)) :- !.
lua_line_term(["trust_me"], trust_me) :- !.
lua_line_term(["jump", L], jump(L)) :- !.
lua_line_term(["cut_ite"], cut_ite) :- !.
lua_line_term(["builtin_call", Op, Ar], builtin_call(Op, Ar)) :- !.
lua_line_term(Parts, line(Parts)).

take_clause1_lines([], []).
take_clause1_lines([Line|Rest], Out) :-
    wam_lua_target:tokenize_wam_line(Line, Parts),
    (   Parts == ["proceed"] -> Out = [Line]
    ;   Parts == ["trust_me"] -> Out = []
    ;   Out = [Line|More],
        take_clause1_lines(Rest, More)
    ).

wam_lua_lowerable(_PI, WamCode, Reason) :-
    catch(build_emission_plan(WamCode, plan(Reason, _, Payload)), _, fail),
    (   Reason == ite
    ->  forall(member(I, Payload), lua_struct_supported(I))
    ;   forall(member(Line, Payload), line_supported(Line))
    ).

%% lua_struct_supported(+StructuredInstr) — recurse through ite/3; each leaf
%  must be an instruction emit_struct_lua/2 can render.
lua_struct_supported(ite(C, T, E)) :- !,
    forall(member(I, C), lua_struct_supported(I)),
    forall(member(I, T), lua_struct_supported(I)),
    forall(member(I, E), lua_struct_supported(I)).
lua_struct_supported(builtin_call(_, _)) :- !.
lua_struct_supported(line(Parts)) :- !,
    ( Parts == [] -> true ; parts_supported(Parts) ).
lua_struct_supported(_) :- fail.

line_supported(Line) :-
    wam_lua_target:tokenize_wam_line(Line, Parts),
    (Parts == [] -> true ; Parts = [F|_], sub_string(F, _, 1, 0, ":") -> true ; parts_supported(Parts)).

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

lua_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, S),
    string_codes(S, Codes),
    maplist(lua_safe_code, Codes, Safe),
    string_codes(SafeS, Safe),
    format(atom(Name), 'lowered_~w_~w', [SafeS, Arity]).

lua_safe_code(C, C) :-
    (C >= 0'a, C =< 0'z ; C >= 0'A, C =< 0'Z ; C >= 0'0, C =< 0'9 ; C =:= 0'_), !.
lua_safe_code(_, 0'_).

lower_predicate_to_lua(PI, WamCode, _Options, lowered(PredName, FuncName, Code)) :-
    (PI = _M:Pred/Arity -> true ; PI = Pred/Arity),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    lua_lowered_func_name(Pred/Arity, FuncName),
    build_emission_plan(WamCode, plan(Mode, AltLabel, Payload)),
    (   Mode == deterministic
    ->  emit_deterministic_function(PredName, FuncName, Payload, Code)
    ;   Mode == ite
    ->  emit_ite_function(PredName, FuncName, Payload, Code)
    ;   emit_multi_clause_function(PredName, FuncName, AltLabel, Payload, Code)
    ).

% If-then-else / negation / once. Same wrapper as the deterministic case,
% but the body is the structured term list rendered by emit_struct_lua/2.
% lua's bind_var always trails, so undoing the trail to the pre-condition
% mark before the else branch restores any partial bindings the condition
% made (no register snapshot needed; mirrors the Rust emitter).
emit_ite_function(PredName, FuncName, Structured, Code) :-
    with_output_to(string(Body), emit_struct_lua(Structured, "  ")),
    format(string(Code),
'-- Lowered: ~w (if-then-else / negation / once)
local function ~w(program, state)
~w  return true
end
', [PredName, FuncName, Body]).

emit_struct_lua([], _).
emit_struct_lua([Item|Rest], Ind) :-
    emit_struct_item_lua(Item, Ind),
    emit_struct_lua(Rest, Ind).

emit_struct_item_lua(ite(Cond, Then, Else), Ind) :- !,
    string_concat(Ind, "    ", Ind4),
    format("~wdo~n", [Ind]),
    format("~w  local _ite_mark = #state.trail~n", [Ind]),
    format("~w  local _ite_cond = (function()~n", [Ind]),
    emit_struct_lua(Cond, Ind4),
    format("~w    return true~n", [Ind]),
    format("~w  end)()~n", [Ind]),
    format("~w  if _ite_cond then~n", [Ind]),
    emit_struct_lua(Then, Ind4),
    format("~w  else~n", [Ind]),
    format("~w    while #state.trail > _ite_mark do state.bindings[table.remove(state.trail)] = nil end~n", [Ind]),
    emit_struct_lua(Else, Ind4),
    format("~w  end~n", [Ind]),
    format("~wend~n", [Ind]).
emit_struct_item_lua(builtin_call(Op, Ar), Ind) :- !,
    emit_line_parts(["builtin_call", Op, Ar], Ind).
emit_struct_item_lua(line(Parts), Ind) :- !,
    emit_line_parts(Parts, Ind).

emit_deterministic_function(PredName, FuncName, Lines, Code) :-
    with_output_to(string(Body), emit_lines(Lines, "  ")),
    format(string(Code),
'-- Lowered: ~w (deterministic)
local function ~w(program, state)
~w  return true
end
', [PredName, FuncName, Body]).

emit_multi_clause_function(PredName, FuncName, AltLabel, Lines, Code) :-
    with_output_to(string(Body), emit_lines(Lines, "    ")),
    wam_lua_target:lua_string_literal(AltLabel, AltQ),
    format(string(Code),
'-- Lowered: ~w (multi-clause; clause 1 inline, array fallback)
local function ~w(program, state)
  local alt_pc = program.labels[~w]
  if alt_pc == nil then return false end
  table.insert(state.cps, {
    next_pc = alt_pc,
    regs = Runtime.copy_table(state.regs),
    cp = state.cp,
    trail_len = #state.trail,
    var_counter = state.var_counter
  })
  local ok = (function()
~w    return false
  end)()
  if ok == true then return true end
  if Runtime.backtrack(state) ~~= true then return false end
  state.pc = state.pc + 1
  state.halt = false
  return Runtime.run(program, state) == true
end
', [PredName, FuncName, AltQ, Body]).

emit_lines([], _).
emit_lines([Line|Rest], Ind) :-
    wam_lua_target:tokenize_wam_line(Line, Parts),
    (   Parts == [] -> true
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":") -> true
    ;   emit_line_parts(Parts, Ind)
    ),
    emit_lines(Rest, Ind).

emit_line_parts(["proceed"], I) :- !, format("~wdo return true end~n", [I]).
emit_line_parts(["fail"], I) :- !, format("~wdo return false end~n", [I]).
emit_line_parts(["call", PredArity], I) :- !, emit_call(PredArity, I).
emit_line_parts(["call", Pred, ArityStr], I) :- !,
    strip_arity_local(Pred, Name), format(string(PA), "~w/~w", [Name, ArityStr]), emit_call(PA, I).
emit_line_parts(["execute", PredArity], I) :- !, emit_execute(PredArity, I).
emit_line_parts(["execute", Pred, ArityStr], I) :- !,
    strip_arity_local(Pred, Name), format(string(PA), "~w/~w", [Name, ArityStr]), emit_execute(PA, I).
emit_line_parts(["allocate"], I) :- !,
    format("~wtable.insert(state.stack, {cp = state.cp, locals = {}})~n", [I]).
emit_line_parts(["deallocate"], I) :- !,
    format("~wdo local fr = table.remove(state.stack); if fr then state.cp = fr.cp end end~n", [I]).
emit_line_parts(["put_constant", C, R], I) :- !,
    wam_lua_target:reg_to_int(R, RI),
    wam_lua_target:constant_to_lua_term(C, T),
    format("~wRuntime.put_reg(state, ~w, ~w)~n", [I, RI, T]).
emit_line_parts(["put_variable", X, A], I) :- !,
    wam_lua_target:reg_to_int(X, XI),
    wam_lua_target:reg_to_int(A, AI),
    format("~wdo local v = Runtime.new_var(state); Runtime.put_reg(state, ~w, v); Runtime.put_reg(state, ~w, v) end~n", [I, XI, AI]).
emit_line_parts(["put_value", X, A], I) :- !,
    wam_lua_target:reg_to_int(X, XI),
    wam_lua_target:reg_to_int(A, AI),
    format("~wRuntime.put_reg(state, ~w, Runtime.get_reg(state, ~w))~n", [I, AI, XI]).
emit_line_parts(["get_variable", X, A], I) :- !,
    wam_lua_target:reg_to_int(X, XI),
    wam_lua_target:reg_to_int(A, AI),
    format("~wRuntime.put_reg(state, ~w, Runtime.get_reg(state, ~w))~n", [I, XI, AI]).
emit_line_parts(Parts, I) :-
    wam_lua_target:wam_parts_to_lua(Parts, [], Lit),
    format("~wif Runtime.step(program, state, ~w) ~~= true then return false end~n", [I, Lit]).

emit_call(PredArity, I) :-
    wam_lua_target:lua_string_literal(PredArity, Q),
    format("~wdo~n", [I]),
    format("~w  local saved_cp = state.cp~n", [I]),
    format("~w  local target = program.labels[~w]~n", [I, Q]),
    format("~w  if target == nil then return false end~n", [I]),
    format("~w  state.cp = 0~n", [I]),
    format("~w  state.pc = target~n", [I]),
    format("~w  if Runtime.run(program, state) ~~= true then return false end~n", [I]),
    format("~w  state.halt = false~n", [I]),
    format("~w  state.cp = saved_cp~n", [I]),
    format("~wend~n", [I]).

emit_execute(PredArity, I) :-
    wam_lua_target:lua_string_literal(PredArity, Q),
    format("~wdo~n", [I]),
    format("~w  local target = program.labels[~w]~n", [I, Q]),
    format("~w  if target == nil then return false end~n", [I]),
    format("~w  state.pc = target~n", [I]),
    format("~w  return Runtime.run(program, state) == true~n", [I]),
    format("~wend~n", [I]).

strip_arity_local(Tok, Name) :-
    (sub_string(Tok, B, 1, _, "/") -> sub_string(Tok, 0, B, _, Name) ; Name = Tok).
