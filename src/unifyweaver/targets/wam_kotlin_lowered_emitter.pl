:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

:- module(wam_kotlin_lowered_emitter, [
    wam_kotlin_lowerable/3,
    lower_predicate_to_kotlin/4,
    kotlin_lowered_func_name/2
]).

:- use_module(library(lists)).
:- use_module(wam_text_parser, [wam_classify_constant_token/2]).
:- use_module(wam_clause_chain, [clause_chain/2]).

% ============================================================================
% Emission plan (T5 clause_chain → T4 multi_clause_n → T1 deterministic)
% ============================================================================

build_emission_plan(WamCode, Plan) :-
    atom_string(WamCode, S),
    split_string(S, "\n", "", Lines),
    skip_to_first_real_instr(Lines, Filtered),
    classify_clause_shape(Filtered, Plan).

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
skippable_prefix_line(["switch_on_term_a2"|_]).

% T5: distinct first-arg get_constant discriminators.
classify_clause_shape([FirstLine|Rest],
                     plan(clause_chain, none, chain_payload(Guards))) :-
    wam_kotlin_target:tokenize_wam_line(FirstLine, ["try_me_else", _AltStr]),
    kotlin_chain_terms([FirstLine|Rest], Terms),
    clause_chain(Terms, chain(Guards)),
    forall(member(guard(_, Rem), Guards), kotlin_chain_rem_supported(Rem)),
    !.
% T4: multi-clause with only supported deterministic bodies (no mid-body call).
% Last-call `execute` is allowed (EMIT-KOTLIN-4); mid-body `call` is EMIT-KOTLIN-5.
classify_clause_shape([FirstLine|Rest], plan(multi_clause_n, none, Clauses)) :-
    wam_kotlin_target:tokenize_wam_line(FirstLine, ["try_me_else", _AltStr]),
    kotlin_split_clause_lines([FirstLine|Rest], Clauses),
    Clauses = [_, _ | _],
    forall(member(Cl, Clauses),
           ( forall(member(Line, Cl), kotlin_t4_clause_line_supported(Line)),
             last(Cl, LastLine), kotlin_t4_terminal_line(LastLine) )),
    !.
% T1: deterministic single-clause (no try_me_else / cut_ite / jump).
classify_clause_shape(Lines, plan(deterministic, none, Lines)) :-
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

% --- T5 chain term parse -------------------------------------------------
kotlin_chain_terms([], []).
kotlin_chain_terms([Line|Rest], Terms) :-
    wam_kotlin_target:tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  kotlin_chain_terms(Rest, Terms)
    ;   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  kotlin_chain_terms(Rest, Terms)
    ;   kotlin_chain_term(Parts, T),
        Terms = [T|More],
        kotlin_chain_terms(Rest, More)
    ).

kotlin_chain_term(["try_me_else", L], try_me_else(L)) :- !.
kotlin_chain_term(["retry_me_else", L], retry_me_else(L)) :- !.
kotlin_chain_term(["trust_me"], trust_me) :- !.
kotlin_chain_term(["get_constant", V, A], get_constant(V, A)) :- !.
kotlin_chain_term(Parts, line(Parts)).

kotlin_chain_rem_supported([]).
kotlin_chain_rem_supported([T|Rest]) :-
    (   T = get_constant(_, _)
    ->  true
    ;   T = line(Parts)
    ->  (   Parts = ["call"|_]
        ->  fail
        ;   parts_supported(Parts)
        )
    ),
    kotlin_chain_rem_supported(Rest).

% --- T4 clause splitting -------------------------------------------------
kotlin_split_clause_lines(AllLines, Clauses) :-
    include(kotlin_t4_instr_line, AllLines, InstrLines),
    kotlin_split_at_terminal(InstrLines, Clauses).

kotlin_t4_instr_line(Line) :-
    wam_kotlin_target:tokenize_wam_line(Line, Parts),
    Parts \== [],
    Parts = [F|_],
    \+ sub_string(F, _, 1, 0, ":"),
    \+ member(F, ["try_me_else", "retry_me_else", "trust_me",
                  "switch_on_constant", "switch_on_constant_a2",
                  "switch_on_structure", "switch_on_term",
                  "switch_on_term_a2"]).

kotlin_split_at_terminal([], []).
kotlin_split_at_terminal([L|Ls], [Clause|Rest]) :-
    kotlin_take_to_terminal([L|Ls], Clause, After),
    ( After == [] -> Rest = [] ; kotlin_split_at_terminal(After, Rest) ).

kotlin_take_to_terminal([Line|Rest], [Line], Rest) :-
    wam_kotlin_target:tokenize_wam_line(Line, Parts),
    kotlin_is_terminal_parts(Parts), !.
kotlin_take_to_terminal([Line|Rest], [Line|More], After) :-
    kotlin_take_to_terminal(Rest, More, After).
kotlin_take_to_terminal([], [], []).

kotlin_is_terminal_parts(["proceed"]).
kotlin_is_terminal_parts(["fail"]).
kotlin_is_terminal_parts(["execute"|_]).

kotlin_t4_clause_line_supported(Line) :-
    wam_kotlin_target:tokenize_wam_line(Line, Parts),
    ( Parts == [] -> true ; parts_supported(Parts) ),
    Parts \= ["cut_ite"|_],
    Parts \= ["jump"|_],
    % Mid-body call needs a continuation — EMIT-KOTLIN-5.
    \+ (Parts = ["call"|_]).

kotlin_t4_terminal_line(Line) :-
    wam_kotlin_target:tokenize_wam_line(Line, Parts),
    kotlin_is_terminal_parts(Parts).

wam_kotlin_lowerable(_PI, WamCode, Reason) :-
    catch(build_emission_plan(WamCode, plan(Reason, _, Payload)), _, fail),
    (   Reason == clause_chain
    ->  Payload = chain_payload(Guards),
        forall(member(guard(_, Rem), Guards), kotlin_chain_rem_supported(Rem))
    ;   Reason == multi_clause_n
    ->  forall(member(Cl, Payload),
               ( forall(member(Line, Cl), line_supported(Line)),
                 \+ kotlin_clause_has_call(Cl) ))
    ;   forall(member(Line, Payload), line_supported(Line)),
        \+ kotlin_clause_has_call(Payload)
    ).

kotlin_clause_has_call(Lines) :-
    member(Line, Lines),
    wam_kotlin_target:tokenize_wam_line(Line, ["call"|_]).

line_supported(Line) :-
    wam_kotlin_target:tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  true
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":")
    ->  true
    ;   Parts = ["call"|_]
    ->  fail
    ;   parts_supported(Parts)
    ).

% Structure/list + unify/set + last-call execute. Mid-body call → EMIT-KOTLIN-5.
% get_variable/put_variable must emit `run { ... }` — see emit_line_parts/2.
parts_supported(["allocate"]).
parts_supported(["deallocate"]).
parts_supported(["proceed"]).
parts_supported(["fail"]).
parts_supported(["execute", _]).
parts_supported(["execute", _, _]).
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
    build_emission_plan(WamCode, plan(Mode, _, Payload)),
    (   Mode == clause_chain
    ->  Payload = chain_payload(Guards),
        emit_clause_chain_function(PredName, FuncName, Guards, Code)
    ;   Mode == multi_clause_n
    ->  emit_multi_clause_n_function(PredName, FuncName, Payload, Code)
    ;   emit_deterministic_function(PredName, FuncName, Payload, Code)
    ).

emit_deterministic_function(PredName, FuncName, Lines, Code) :-
    with_output_to(string(Body), emit_lines(Lines, "    ", return_true)),
    format(string(Code),
'// Lowered: ~w (deterministic single-clause)
fun ~w(state: WamState, dispatch: (String, WamState) -> Boolean): Boolean {
~w}
', [PredName, FuncName, Body]).

% T4: try each clause as a local fun; restore snapshot between attempts.
% KT-HEAP-SNAPSHOT-OPT-2: when clause 1 starts with get_constant/get_nil/
% get_integer, peel that discriminant so a closed fail (ground mismatch)
% jumps to later clauses with *no* entry snapshot. Deep list recursion
% (append cons path) therefore avoids the O(depth) map copy per hop.
emit_multi_clause_n_function(PredName, FuncName, Clauses, Code) :-
    (   kotlin_t4_peelable_get_constant(Clauses, ConstTok, Reg, FirstBody, RestClauses)
    ->  emit_multi_clause_n_peeled(PredName, FuncName, ConstTok, Reg, FirstBody, RestClauses, Code)
    ;   with_output_to(string(Body), emit_kotlin_t4_clauses(Clauses, 0, _t4)),
        format(string(Code),
'// Lowered: ~w (T4 all-clauses inline)
fun ~w(state: WamState, dispatch: (String, WamState) -> Boolean): Boolean {
    val _t4 = state.snapshotForNative()
~w    return false
}
', [PredName, FuncName, Body])
    ).

% Peel leading get_constant when ≥2 clauses (need an alternative after miss).
kotlin_t4_peelable_get_constant([First|Rest], ConstTok, Reg, FirstBody, Rest) :-
    Rest = [_|_],
    First = [HeadLine|FirstBody],
    wam_kotlin_target:tokenize_wam_line(HeadLine, Parts),
    kotlin_t4_peel_head(Parts, ConstTok, Reg).

kotlin_t4_peel_head(["get_constant", C, R], C, R) :- !.
kotlin_t4_peel_head(["get_integer", C, R], C, R) :- !.
kotlin_t4_peel_head(["get_nil", R], '[]', R) :- !.

emit_multi_clause_n_peeled(PredName, FuncName, ConstTok, Reg, FirstBody, RestClauses, Code) :-
    kotlin_constant_expr(ConstTok, Expr),
    kotlin_register_lit(Reg, RL),
    with_output_to(string(FirstFun), (
        format("        fun clause_1(): Boolean {~n", []),
        format("            if (!kotlinLoGetConstant(state, ~w, ~w)) return false~n", [Expr, RL]),
        emit_lines(FirstBody, "            ", return_true),
        format("        }~n", []),
        format("        if (clause_1()) return true~n", []),
        format("        state.restoreFromSnapshot(_t4)~n", [])
    )),
    (   RestClauses = [_]
    ->  % Single remaining clause: last-clause path — no snapshot.
        with_output_to(string(RestBody), emit_kotlin_t4_last_clauses(RestClauses, 1)),
        format(string(Code),
'// Lowered: ~w (T4 peel leading get_constant — skip snap on closed fail)
fun ~w(state: WamState, dispatch: (String, WamState) -> Boolean): Boolean {
    val _peel = state.deref(state.readRegister(~w))
    val _peelHit = _peel == ~w || _peel == null || _peel is Value.Var
    if (_peelHit) {
        val _t4 = state.snapshotForNative()
~w    }
~w    return false
}
', [PredName, FuncName, RL, Expr, FirstFun, RestBody])
    ;   with_output_to(string(RestBody), emit_kotlin_t4_clauses(RestClauses, 1, _t4b)),
        format(string(Code),
'// Lowered: ~w (T4 peel leading get_constant — skip snap on closed fail)
fun ~w(state: WamState, dispatch: (String, WamState) -> Boolean): Boolean {
    val _peel = state.deref(state.readRegister(~w))
    val _peelHit = _peel == ~w || _peel == null || _peel is Value.Var
    if (_peelHit) {
        val _t4 = state.snapshotForNative()
~w    }
    val _t4b = state.snapshotForNative()
~w    return false
}
', [PredName, FuncName, RL, Expr, FirstFun, RestBody])
    ).

% Emit remaining clauses without a shared entry snapshot (last-clause / peel miss).
emit_kotlin_t4_last_clauses([], _).
emit_kotlin_t4_last_clauses([Clause|Rest], N) :-
    N1 is N + 1,
    format(atom(CName), 'clause_~w', [N1]),
    format("    fun ~w(): Boolean {~n", [CName]),
    emit_lines(Clause, "        ", return_true),
    format("    }~n", []),
    format("    if (~w()) return true~n", [CName]),
    emit_kotlin_t4_last_clauses(Rest, N1).

% SnapVar is the Kotlin local holding the restore point (_t4 / _t4b).
emit_kotlin_t4_clauses([], _, _).
emit_kotlin_t4_clauses([Clause|Rest], N, SnapVar) :-
    N1 is N + 1,
    format(atom(CName), 'clause_~w', [N1]),
    format("    fun ~w(): Boolean {~n", [CName]),
    % proceed → return true; execute → return dispatch(...); fail → return false.
    emit_lines(Clause, "        ", return_true),
    format("    }~n", []),
    format("    if (~w()) return true~n", [CName]),
    (   Rest == []
    ->  true  % last clause: no restore before return false
    ;   format("    state.restoreFromSnapshot(~w)~n", [SnapVar])
    ),
    emit_kotlin_t4_clauses(Rest, N1, SnapVar).

% T5: bound first-arg if-cascade; unbound → false (tryRun → interpreter).
emit_clause_chain_function(PredName, FuncName, Guards, Code) :-
    with_output_to(string(Dispatch), emit_kotlin_t5_guards(Guards)),
    format(string(Code),
'// Lowered: ~w (T5 first-argument dispatch)
fun ~w(state: WamState, dispatch: (String, WamState) -> Boolean): Boolean {
    val t5a1 = state.deref(state.readRegister("A1"))
    if (t5a1 is Value.Var) return false
~w    return false
}
', [PredName, FuncName, Dispatch]).

emit_kotlin_t5_guards([]).
emit_kotlin_t5_guards([guard(V, Rem)|Rest]) :-
    kotlin_constant_expr(V, Expr),
    format("    if (t5a1 == ~w) {~n", [Expr]),
    emit_kotlin_t5_rem(Rem, "        ", Ended),
    (   Ended == true
    ->  true
    ;   format("        return true~n", [])
    ),
    format("    }~n", []),
    emit_kotlin_t5_guards(Rest).

emit_kotlin_t5_rem([], _, false).
emit_kotlin_t5_rem([get_constant(C, R)|Rest], Ind, Ended) :- !,
    emit_line_parts(["get_constant", C, R], Ind),
    emit_kotlin_t5_rem(Rest, Ind, Ended).
emit_kotlin_t5_rem([line(Parts)|Rest], Ind, Ended) :- !,
    (   Parts == ["proceed"]
    ->  emit_kotlin_t5_rem(Rest, Ind, Ended)
    ;   Parts = ["execute"|_]
    ->  emit_line_parts(Parts, Ind),
        Ended = true
    ;   emit_line_parts(Parts, Ind),
        emit_kotlin_t5_rem(Rest, Ind, Ended)
    ).

% ProceedMode = comment | return_true
emit_lines([], _, _).
emit_lines([Line|Rest], Ind, ProceedMode) :-
    wam_kotlin_target:tokenize_wam_line(Line, Parts),
    (   Parts == [] -> true
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":") -> true
    ;   Parts == ["proceed"], ProceedMode == return_true
    ->  format("~wreturn true~n", [Ind])
    ;   emit_line_parts(Parts, Ind)
    ),
    emit_lines(Rest, Ind, ProceedMode).

emit_line_parts(["proceed"], I) :- !, format("~w// proceed~n", [I]).
emit_line_parts(["fail"], I) :- !, format("~wreturn false~n", [I]).
emit_line_parts(["execute", PredArity], I) :- !,
    kotlin_execute_pred_key(PredArity, Key),
    escape_kotlin_string(Key, Esc),
    format("~wreturn dispatch(\"~w\", state)~n", [I, Esc]).
emit_line_parts(["execute", Pred, ArityStr], I) :- !,
    strip_arity_token(Pred, Name),
    format(string(PA), '~w/~w', [Name, ArityStr]),
    escape_kotlin_string(PA, Esc),
    format("~wreturn dispatch(\"~w\", state)~n", [I, Esc]).
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

% execute target key: "foo/2" or already-slashed token.
kotlin_execute_pred_key(Tok, Key) :-
    atom_string_like(Tok, S),
    (   sub_string(S, _, 1, _, "/")
    ->  Key = S
    ;   Key = S
    ).

strip_arity_token(Tok, Name) :-
    atom_string_like(Tok, S),
    (   sub_string(S, B, 1, _, "/"),
        sub_string(S, 0, B, _, Name)
    ->  true
    ;   Name = S
    ).

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
