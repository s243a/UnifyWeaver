:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

:- module(wam_kotlin_lowered_emitter, [
    wam_kotlin_lowerable/3,
    lower_predicate_to_kotlin/4,
    kotlin_lowered_func_name/2
]).

:- use_module(library(lists)).
:- use_module(wam_text_parser, [wam_classify_constant_token/2]).

build_emission_plan(WamCode, plan(deterministic, none, ClauseLines)) :-
    atom_string(WamCode, S),
    split_string(S, "\n", "", Lines),
    skip_to_first_real_instr(Lines, Filtered),
    deterministic_single_clause(Filtered, ClauseLines).

skip_to_first_real_instr([], []).
skip_to_first_real_instr([Line|Rest], Out) :-
    wam_kotlin_target:tokenize_wam_line(Line, Parts),
    (   skippable_prefix_line(Parts)
    ->  skip_to_first_real_instr(Rest, Out)
    ;   Out = [Line|Rest]
    ).

skippable_prefix_line([]).
skippable_prefix_line([First|_]) :- sub_string(First, _, 1, 0, ":").
skippable_prefix_line(["switch_on_constant"|_]).
skippable_prefix_line(["switch_on_constant_a2"|_]).
skippable_prefix_line(["switch_on_structure"|_]).
skippable_prefix_line(["switch_on_term"|_]).

deterministic_single_clause(Lines, Lines) :-
    \+ has_unsupported_shape(Lines),
    forall(member(Line, Lines), line_supported(Line)).

has_unsupported_shape(Lines) :-
    member(Line, Lines),
    wam_kotlin_target:tokenize_wam_line(Line, Parts),
    unsupported_shape_parts(Parts).

unsupported_shape_parts(["try_me_else"|_]).
unsupported_shape_parts(["retry_me_else"|_]).
unsupported_shape_parts(["trust_me"]).
unsupported_shape_parts(["cut_ite"]).
unsupported_shape_parts(["jump"|_]).

wam_kotlin_lowerable(_PI, WamCode, deterministic) :-
    catch(build_emission_plan(WamCode, plan(deterministic, _, Payload)), _, fail),
    forall(member(Line, Payload), line_supported(Line)).

line_supported(Line) :-
    wam_kotlin_target:tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  true
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":")
    ->  true
    ;   parts_supported(Parts)
    ).

% Structure/list + unify/set ops are lowerable. get_variable/put_variable must
% emit `run { ... }` (not a bare block) — see emit_line_parts/2 note.
parts_supported(["allocate"]).
parts_supported(["deallocate"]).
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
parts_supported(["put_nil", _]).
parts_supported(["put_integer", _, _]).
parts_supported(["unify_variable", _]).
parts_supported(["unify_value", _]).
parts_supported(["unify_constant", _]).
parts_supported(["unify_nil"]).
parts_supported(["set_variable", _]).
parts_supported(["set_value", _]).
parts_supported(["set_constant", _]).
parts_supported(["set_nil"]).
parts_supported(["set_integer", _]).

kotlin_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, S),
    string_codes(S, Codes),
    maplist(kotlin_safe_code, Codes, Safe),
    string_codes(SafeS, Safe),
    format(atom(Name), 'lowered_~w_~w', [SafeS, Arity]).

kotlin_safe_code(C, C) :-
    (C >= 0'a, C =< 0'z ; C >= 0'A, C =< 0'Z ; C >= 0'0, C =< 0'9 ; C =:= 0'_), !.
kotlin_safe_code(_, 0'_).

lower_predicate_to_kotlin(PI, WamCode, _Options, lowered(PredName, FuncName, Code)) :-
    (PI = _M:Pred/Arity -> true ; PI = Pred/Arity),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    kotlin_lowered_func_name(Pred/Arity, FuncName),
    build_emission_plan(WamCode, plan(deterministic, _, Payload)),
    emit_deterministic_function(PredName, FuncName, Payload, Code).

emit_deterministic_function(PredName, FuncName, Lines, Code) :-
    with_output_to(string(Body), emit_lines(Lines, "    ")),
    format(string(Code),
'// Lowered: ~w (deterministic single-clause)
fun ~w(state: WamState): Boolean {
~w    return true
}
', [PredName, FuncName, Body]).

emit_lines([], _).
emit_lines([Line|Rest], Ind) :-
    wam_kotlin_target:tokenize_wam_line(Line, Parts),
    (   Parts == [] -> true
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":") -> true
    ;   emit_line_parts(Parts, Ind)
    ),
    emit_lines(Rest, Ind).

emit_line_parts(["proceed"], I) :- !, format("~w// proceed~n", [I]).
emit_line_parts(["fail"], I) :- !, format("~wreturn false~n", [I]).
emit_line_parts(["allocate"], I) :- !,
    format("~wstate.allocateEnvironment()~n", [I]).
emit_line_parts(["deallocate"], I) :- !,
    format("~wstate.deallocateEnvironment()~n", [I]).
emit_line_parts(["get_constant", C, R], I) :- !,
    kotlin_constant_expr(C, Expr),
    kotlin_register_lit(R, RL),
    format("~wif (!kotlinLoGetConstant(state, ~w, ~w)) return false~n", [I, Expr, RL]).
emit_line_parts(["get_integer", C, R], I) :- !,
    kotlin_constant_expr(C, Expr),
    kotlin_register_lit(R, RL),
    format("~wif (!kotlinLoGetConstant(state, ~w, ~w)) return false~n", [I, Expr, RL]).
emit_line_parts(["get_nil", R], I) :- !,
    kotlin_register_lit(R, RL),
    format("~wif (!kotlinLoGetConstant(state, Value.Atom(\"[]\"), ~w)) return false~n", [I, RL]).
% NOTE: must use `run { ... }`, not a bare `{ ... }` block. In Kotlin a
% standalone lambda literal is NOT invoked, so get_variable/put_variable
% would silently no-op — write-mode unify_value then saw null registers and
% fabricated unbound vars (kt_make_list → [X1,X2] instead of [alpha,beta]).
emit_line_parts(["get_variable", X, A], I) :- !,
    kotlin_register_lit(X, XL), kotlin_register_lit(A, AL),
    format("~wrun {~n", [I]),
    format("~w    val v = state.deref(state.readRegister(~w)) ?: state.newVariable(~w)~n", [I, AL, XL]),
    format("~w    state.writeRegister(~w, v)~n", [I, XL]),
    format("~w    state.writeRegister(~w, v)~n", [I, AL]),
    format("~w}~n", [I]).
emit_line_parts(["get_value", X, A], I) :- !,
    kotlin_register_lit(X, XL), kotlin_register_lit(A, AL),
    format("~wif (!kotlinLoGetValue(state, ~w, ~w)) return false~n", [I, XL, AL]).
emit_line_parts(["get_structure", F, R], I) :- !,
    kotlin_register_lit(R, RL),
    kotlin_string_lit(F, FL),
    format("~wif (!state.beginStructure(~w, ~w)) return false~n", [I, FL, RL]).
emit_line_parts(["get_list", R], I) :- !,
    kotlin_register_lit(R, RL),
    format("~wif (!state.beginStructure(\"[|]/2\", ~w)) return false~n", [I, RL]).
emit_line_parts(["put_constant", C, R], I) :- !,
    kotlin_constant_expr(C, Expr),
    kotlin_register_lit(R, RL),
    format("~wstate.writeRegister(~w, ~w)~n", [I, RL, Expr]).
emit_line_parts(["put_integer", C, R], I) :- !,
    kotlin_constant_expr(C, Expr),
    kotlin_register_lit(R, RL),
    format("~wstate.writeRegister(~w, ~w)~n", [I, RL, Expr]).
emit_line_parts(["put_nil", R], I) :- !,
    kotlin_register_lit(R, RL),
    format("~wstate.writeRegister(~w, Value.Atom(\"[]\"))~n", [I, RL]).
emit_line_parts(["put_variable", X, A], I) :- !,
    kotlin_register_lit(X, XL), kotlin_register_lit(A, AL),
    format("~wrun {~n", [I]),
    format("~w    val v = state.newVariable(~w)~n", [I, XL]),
    format("~w    state.writeRegister(~w, v)~n", [I, XL]),
    format("~w    state.writeRegister(~w, v)~n", [I, AL]),
    format("~w}~n", [I]).
emit_line_parts(["put_value", X, A], I) :- !,
    kotlin_register_lit(X, XL), kotlin_register_lit(A, AL),
    format("~wstate.writeRegister(~w, state.deref(state.readRegister(~w)))~n", [I, AL, XL]).
emit_line_parts(["put_structure", F, R], I) :- !,
    kotlin_register_lit(R, RL),
    kotlin_string_lit(F, FL),
    format("~wif (!state.beginStructurePut(~w, ~w)) return false~n", [I, FL, RL]).
emit_line_parts(["put_list", R], I) :- !,
    kotlin_register_lit(R, RL),
    format("~wif (!state.beginStructurePut(\"[|]/2\", ~w)) return false~n", [I, RL]).
emit_line_parts(["set_variable", X], I) :- !,
    kotlin_register_lit(X, XL),
    format("~wif (!state.pushWriteArg(state.newVariable(~w))) return false~n", [I, XL]).
emit_line_parts(["set_value", X], I) :- !,
    kotlin_register_lit(X, XL),
    % Mirror interpreter set_value: missing register → fail (not push null).
    format("~wif (!state.pushWriteArg(state.deref(state.readRegister(~w)) ?: return false)) return false~n", [I, XL]).
emit_line_parts(["set_constant", C], I) :- !,
    kotlin_constant_expr(C, Expr),
    format("~wif (!state.pushWriteArg(~w)) return false~n", [I, Expr]).
emit_line_parts(["set_nil"], I) :- !,
    format("~wif (!state.pushWriteArg(Value.Atom(\"[]\"))) return false~n", [I]).
emit_line_parts(["set_integer", C], I) :- !,
    kotlin_constant_expr(C, Expr),
    format("~wif (!state.pushWriteArg(~w)) return false~n", [I, Expr]).
emit_line_parts(["unify_variable", X], I) :- !,
    kotlin_register_lit(X, XL),
    format("~wif (!kotlinLoUnifyVariable(state, ~w)) return false~n", [I, XL]).
emit_line_parts(["unify_value", X], I) :- !,
    kotlin_register_lit(X, XL),
    format("~wif (!kotlinLoUnifyValue(state, ~w)) return false~n", [I, XL]).
emit_line_parts(["unify_constant", C], I) :- !,
    kotlin_constant_expr(C, Expr),
    format("~wif (!kotlinLoUnifyConstant(state, ~w)) return false~n", [I, Expr]).
emit_line_parts(["unify_nil"], I) :- !,
    format("~wif (!kotlinLoUnifyConstant(state, Value.Atom(\"[]\"))) return false~n", [I]).

kotlin_register_lit(Reg, Lit) :-
    atom_string_like(Reg, S),
    escape_kotlin_string(S, Esc),
    format(string(Lit), '"~w"', [Esc]).

kotlin_string_lit(Atom, Lit) :-
    atom_string_like(Atom, S),
    escape_kotlin_string(S, Esc),
    format(string(Lit), '"~w"', [Esc]).

kotlin_constant_expr(Token, Expr) :-
    wam_classify_constant_token(Token, Class),
    (   Class = integer(N)
    ->  format(string(Expr), 'Value.IntVal(~wL)', [N])
    ;   Class = float(F)
    ->  format(string(Expr), 'Value.FloatVal(~w)', [F])
    ;   Class = atom(Name)
    ->  escape_kotlin_string(Name, Esc),
        format(string(Expr), 'Value.Atom("~w")', [Esc])
    ).

atom_string_like(Value, String) :-
    string(Value), !, String = Value.
atom_string_like(Value, String) :-
    atom(Value), !, atom_string(Value, String).
atom_string_like(Value, String) :-
    number(Value), !, number_string(Value, String).
atom_string_like(Value, String) :-
    term_string(Value, String).

escape_kotlin_string(Input, Escaped) :-
    atom_string_like(Input, S),
    string_chars(S, Chars),
    phrase(escaped_chars(Chars), OutChars),
    string_chars(Escaped, OutChars).

escaped_chars([]) --> [].
escaped_chars(['\\'|Cs]) --> ['\\','\\'], escaped_chars(Cs).
escaped_chars(['"'|Cs]) --> ['\\','"'], escaped_chars(Cs).
escaped_chars(['\n'|Cs]) --> ['\\','n'], escaped_chars(Cs).
escaped_chars(['\r'|Cs]) --> ['\\','r'], escaped_chars(Cs).
escaped_chars(['\t'|Cs]) --> ['\\','t'], escaped_chars(Cs).
escaped_chars([C|Cs]) --> [C], escaped_chars(Cs).
