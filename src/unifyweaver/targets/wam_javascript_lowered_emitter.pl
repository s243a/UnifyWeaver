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
:- use_module(wam_ite_structurer, [is_commit/1]).
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
% Indexed single ground fact: switch_on_* re-emits the same clause
% along a try/retry/trust chain. Collapse to one deterministic body so
% the intern-once memo path still fires (GP-PERF item 2).
classify_clause_shape([FirstLine|Rest], plan(deterministic, none, Unique)) :-
    tokenize_line(FirstLine, ["try_me_else", _AltStr]),
    js_split_clause_lines([FirstLine|Rest], Clauses),
    Clauses = [C1|_],
    C1 \== [],
    forall(member(Cl, Clauses),
           ( Cl \== [],
             forall(member(Line, Cl), js_ground_fact_line(Line)) )),
    maplist(js_clause_op_fingerprint, Clauses, FPs),
    sort(FPs, [_]),
    !,
    Unique = C1.
% Indexed copies of the same non-ground deterministic body (including
% last-goal Execute wrappers). Collapse to one body so they lower.
classify_clause_shape([FirstLine|Rest], plan(deterministic, none, Unique)) :-
    tokenize_line(FirstLine, ["try_me_else", _AltStr]),
    js_split_clause_lines([FirstLine|Rest], Clauses),
    Clauses = [C1|_],
    C1 \== [],
    forall(member(Cl, Clauses),
           ( Cl \== [],
             forall(member(Line, Cl), line_supported(Line)),
             last(Cl, LastLine), js_t4_terminal_line(LastLine) )),
    maplist(js_clause_op_fingerprint, Clauses, FPs),
    sort(FPs, [_]),
    !,
    Unique = C1.
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
    js_structure_ite(Terms, Structured),
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
    js_structure_ite(Terms, Structured),
    Structured \== [],
    forall(member(I, Structured), js_struct_supported(I)).

%% js_structure_ite(+Flat, -Structured)
%  Same fold as structure_ite/2, but split_commit ignores commits that
%  sit inside a nested try_me_else (the `\+ Goal` fail-commit). The
%  shared structurer splits on the first cut/cut_ite and then fails to
%  match the leftover nested block — that is why lenient_loop/5 stayed
%  on multi_clause_1.
js_structure_ite([], []).
js_structure_ite([try_me_else(LE)|Rest0], [ite(CondS, ThenS, ElseS)|Out]) :-
    !,
    append(ThenWithJump, [label(LE), trust_me | ElseAndRest], Rest0),
    \+ member(label(LE), ThenWithJump),
    append(ThenPath, [jump(LC)], ThenWithJump),
    append(ElsePath, [label(LC) | AfterCont], ElseAndRest),
    \+ member(label(LC), ElsePath),
    js_split_commit(ThenPath, Cond, Then),
    js_structure_ite(Cond, CondS),
    js_structure_ite(Then, ThenS),
    js_structure_ite(ElsePath, ElseS),
    js_structure_ite(AfterCont, Out).
js_structure_ite([label(_)|Rest], Out) :- !,
    js_structure_ite(Rest, Out).
js_structure_ite([I|Rest], [I|Out]) :-
    js_structure_ite(Rest, Out).

js_split_commit(Path, Cond, Then) :-
    js_split_commit_(Path, 0, [], CondR, Then),
    reverse(CondR, Cond).

js_split_commit_([try_me_else(L)|R], D, Acc, Cond, Then) :- !,
    D1 is D + 1,
    js_split_commit_(R, D1, [try_me_else(L)|Acc], Cond, Then).
js_split_commit_([trust_me|R], D, Acc, Cond, Then) :-
    D > 0, !,
    D1 is D - 1,
    js_split_commit_(R, D1, [trust_me|Acc], Cond, Then).
js_split_commit_([Commit|Then], 0, Acc, Acc, Then) :-
    is_commit(Commit), !.
% A try_me_else block with NO commit on the then-path is a plain
% disjunction (A ; B), not an if-then-else. It used to be folded into
% ite(A, [], B) — i.e. compiled as (A -> true ; B) — which silently
% deleted B as a retry alternative and made A first-solution-only.
% `( Goal, write(X), nl, fail ; true )` (the standard failure-driven
% enumeration loop) then printed only the first solution. Cut-semantics
% probe D-DISJ. Refuse: js_split_commit_/5 now fails on a commit-less
% path, js_structure_ite/2 fails with it, and the predicate falls back
% to the interpreter, which has real choice points.
js_split_commit_([I|R], D, Acc, Cond, Then) :-
    js_split_commit_(R, D, [I|Acc], Cond, Then).

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
    js_clause_terminal_parts(Parts), !.
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
    js_clause_terminal_parts(Parts), !.
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
    js_clause_terminal_parts(Parts).

% Last-goal Execute is a clause terminator (no Proceed after it).
% Indexed copies of Execute-terminated wrappers (parse_args/2, wrap_tail/1)
% were classified as multi_clause_1 because splitters only saw proceed/fail.
js_clause_terminal_parts(["proceed"]).
js_clause_terminal_parts(["fail"]).
js_clause_terminal_parts(["execute"|_]).

wam_javascript_lowerable(PI, WamCode, Reason) :-
    js_pi_key(PI, _PredName),
    % Lowered JS functions are first-solution: they cannot retry a callee.
    % Predicates that enumerate via member/2 (directly or through a user
    % callee, including under \+) stay on the interpreter. findall/bagof/
    % setof/once wrap enumeration, so those inners do not taint the caller.
    \+ js_pi_needs_naked_member_cps(PI),
    catch(build_emission_plan(WamCode, plan(Reason, _, Payload)), _, fail),
    % multi_clause_1 inlines clause 1 then Runtime.run from pc+1, which is
    % the caller's next instruction rather than the alt clause. Wrong-but-
    % fast is a failure; leave these on the interpreter until T4+ITE fits.
    Reason \== multi_clause_1,
    % Every lowered plan except clause_chain returns the FIRST clause that
    % succeeds and pushes no choice point, so it is only sound when at
    % most one clause CAN succeed: either the heads are first-argument
    % mutually exclusive, or every clause but the last commits with a
    % top-level cut. `p(X) :- once(d(X)). p(9).` is neither, and a caller
    % could never reach the second solution (probes P10/P18). T5/T6
    % (clause_chain) are exempt: their unbound-A1 path pushes a real
    % interpreter choice point and hands the alternatives back.
    (   Reason == clause_chain
    ->  true
    ;   \+ js_pi_may_yield_many(PI)
    ),
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
    ->  emit_deterministic_function(PredName, FuncName, Payload, Code0)
    ;   Mode == ite
    ->  emit_ite_function(PredName, FuncName, Payload, Code0)
    ;   Mode == clause_chain
    ->  Payload = chain_payload(Guards, Clause1Lines),
        emit_clause_chain_function(PredName, FuncName, AltLabel, Guards, Clause1Lines, Options, Code0)
    ;   Mode == multi_clause_n
    ->  emit_multi_clause_n_function(PredName, FuncName, Payload, Code0)
    ;   emit_multi_clause_function(PredName, FuncName, AltLabel, Payload, Code0)
    ),
    js_inject_prof_enter(PredName, FuncName, Code0, Code).

%% Direct Call/Execute skips lowered_dispatch, so the call counter lives
%% on the inner function (one falsy check when UW_PROFILE is off).
js_inject_prof_enter(PredName, FuncName, Code0, Code) :-
    wam_javascript_target:js_string_literal(PredName, Q),
    format(string(Old), "function ~w(program, state) {", [FuncName]),
    format(string(New),
           "function ~w(program, state) {~n  if (Runtime._prof) Runtime.prof_lowered_call(~w);",
           [FuncName, Q]),
    (   sub_string(Code0, B, _, A, Old)
    ->  sub_string(Code0, 0, B, _, Pref),
        sub_string(Code0, _, A, 0, Suff),
        string_concat(Pref, New, T),
        string_concat(T, Suff, Code)
    ;   Code = Code0
    ).

emit_multi_clause_n_function(PredName, FuncName, Clauses, Code) :-
    js_t4_nil_cons_split(Clauses, NilRest, ConsClause), !,
    with_output_to(string(NilBody), emit_t4_clause_body(NilRest, "    ")),
    with_output_to(string(ConsBody), emit_t4_clause_body(ConsClause, "    ")),
    with_output_to(string(Fallback), emit_js_t4_payload(Clauses)),
    format(string(Code),
'// Lowered: ~w (T4 nil/cons dispatch; no snapshot on bound A1)
function ~w(program, state) {
  const _a1 = Runtime.deref(state, Runtime.get_reg(state, 1));
  if (Runtime.term_is_nil(program, _a1)) {
~w    return true;
  }
  if (Runtime.term_is_cons(program, _a1)) {
~w    return true;
  }
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
', [PredName, FuncName, NilBody, ConsBody, Fallback]).
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

emit_t4_clause_body(Clause, Ind) :-
    (   Clause = [S|_], string(S)
    ->  emit_lines(Clause, Ind)
    ;   emit_struct_js(Clause, Ind)
    ).

%% js_t4_nil_cons_split(+Clauses, -NilRest, -ConsClause)
%  Two-clause list recursion: clause 1 starts get_nil/get_constant [] on A1,
%  clause 2 starts (optional allocate +) get_list A1. Bound A1 needs no
%  T4 register snapshot — the heads are mutually exclusive.
js_t4_nil_cons_split(Clauses, NilRest, ConsClause) :-
    member(C1, Clauses),
    js_clause_skip_nil_a1(C1, NilRest),
    member(ConsClause, Clauses),
    js_clause_starts_list_a1(ConsClause),
    !.

js_clause_skip_nil_a1([Item|Rest], Rest) :-
    js_item_is_nil_a1(Item), !.
js_clause_skip_nil_a1([Item|Rest0], [Item|Rest]) :-
    js_item_is_skip_prefix(Item),
    js_clause_skip_nil_a1(Rest0, Rest).

js_clause_starts_list_a1([Item|_]) :-
    js_item_is_list_a1(Item), !.
js_clause_starts_list_a1([Item|Rest]) :-
    js_item_is_skip_prefix(Item),
    js_clause_starts_list_a1(Rest).

js_item_is_skip_prefix(line(["allocate"])) :- !.
js_item_is_skip_prefix(Line) :-
    string(Line), tokenize_line(Line, ["allocate"]), !.

js_item_is_nil_a1(line(["get_nil", A1])) :- js_is_a1(A1).
js_item_is_nil_a1(line(["get_constant", C, A1])) :-
    js_is_a1(A1), js_is_nil_const(C).
js_item_is_nil_a1(Line) :-
    string(Line), tokenize_line(Line, Parts),
    js_item_is_nil_a1(line(Parts)).

js_item_is_list_a1(line(["get_list", A1])) :- js_is_a1(A1).
js_item_is_list_a1(Line) :-
    string(Line), tokenize_line(Line, Parts),
    js_item_is_list_a1(line(Parts)).

js_is_a1(A1) :- A1 == "A1" ; A1 == 'A1' ; A1 == "1" ; A1 == 1.

js_is_nil_const(C) :-
    atom_string(C, S),
    ( S == "[]" ; S == "'[]'" ).

emit_js_t4_payload([]).
emit_js_t4_payload([Clause|Rest]) :-
    format("  if ((function () {~n"),
    emit_t4_clause_body(Clause, "    "),
    format("    return false;~n"),
    format("  })()) return true;~n"),
    (   Rest == []
    ->  true
    ;   format("  while (state.trail.length > _t4_trail) { const _n = state.trail.pop(); delete state.bindings[_n]; }~n"),
        format("  state.regs = Runtime.copy_table(_t4_regs);~n"),
        format("  state.var_counter = _t4_vc;~n"),
        format("  state.stack = _t4_stack.slice();~n"),
        format("  state.y_save = _t4_ysave.slice();~n"),
        format("  state.mode = _t4_mode;~n"),
        format("  state.build_stack = _t4_build.slice();~n"),
        format("  state.read_stack = _t4_rstack.slice();~n"),
        format("  state.read_args = _t4_rargs;~n"),
        format("  state.read_cursor = _t4_rcur;~n")
    ),
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

emit_struct_item_js(ite(Cond, Then, Else), Ind) :-
    js_ite_pure_test_cond(Cond), !,
    string_concat(Ind, "    ", Ind4),
    format("~w{~n", [Ind]),
    format("~w  const _ite_trail = state.trail.length;~n", [Ind]),
    format("~w  const _ite_args = Runtime.capture_a_regs(state, 8);~n", [Ind]),
    format("~w  const _ite_cond = (function () {~n", [Ind]),
    emit_struct_js(Cond, Ind4),
    format("~w    return true;~n", [Ind]),
    format("~w  })();~n", [Ind]),
    format("~w  if (_ite_cond) {~n", [Ind]),
    emit_struct_js(Then, Ind4),
    format("~w  } else {~n", [Ind]),
    format("~w    Runtime.undo_trail(state, _ite_trail);~n", [Ind]),
    format("~w    Runtime.restore_a_regs(state, _ite_args);~n", [Ind]),
    emit_struct_js(Else, Ind4),
    format("~w  }~n", [Ind]),
    format("~w}~n", [Ind]).
% Condition may unify / put_structure but does not Call or Allocate:
% restore trail, A1-A16, X101-X160, mode/build/read — not the full
% register file, stack, or Y-save (those were the T4/ITE wall).
emit_struct_item_js(ite(Cond, Then, Else), Ind) :-
    \+ js_ite_cond_needs_full_snap(Cond), !,
    string_concat(Ind, "    ", Ind4),
    format("~w{~n", [Ind]),
    format("~w  const _ite_lite = Runtime.snapshot_lite(state);~n", [Ind]),
    format("~w  const _ite_cond = (function () {~n", [Ind]),
    emit_struct_js(Cond, Ind4),
    format("~w    return true;~n", [Ind]),
    format("~w  })();~n", [Ind]),
    format("~w  if (_ite_cond) {~n", [Ind]),
    emit_struct_js(Then, Ind4),
    format("~w  } else {~n", [Ind]),
    format("~w    Runtime.restore_lite(state, _ite_lite);~n", [Ind]),
    emit_struct_js(Else, Ind4),
    format("~w  }~n", [Ind]),
    format("~w}~n", [Ind]).
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
    % ISO: entering the then-branch COMMITS to the condition. The
    % condition is a once-like scope, so its choice points must be cut
    % here. Without this, `once(d(X))` (compiled to (d(X) -> true ; fail))
    % left d/1's clause choice point live and the caller backtracked into
    % it -- probe P18 produced 1, 2, 3 where SWI gives 1. The then- and
    % else-branches' own choice points are NOT cut (they belong to the
    % enclosing clause).
    format("~w    while (state.cps.length > _ite_cps) state.cps.pop();~n", [Ind]),
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
  // Runtime.snapshot_machine (not a hand-rolled literal): it also
  // captures cut_barrier / cut_stack. Without them restore_machine
  // DELETED the barrier and emptied the stack on backtrack into this
  // choice point, so the alt clause ran with no cut barrier at all.
  const _cp = Runtime.snapshot_machine(state);
  _cp.next_pc = alt_pc;
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
    // snapshot_machine also captures cut_barrier / cut_stack; a literal
    // omitting them made restore_machine wipe the barrier stack.
    const _cp = Runtime.snapshot_machine(state);
    _cp.next_pc = alt_pc;
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
  // Runtime.snapshot_machine (not a hand-rolled literal): it also
  // captures cut_barrier / cut_stack. Without them restore_machine
  // DELETED the barrier and emptied the stack on backtrack into this
  // choice point, so the alt clause ran with no cut barrier at all.
  const _cp = Runtime.snapshot_machine(state);
  _cp.next_pc = alt_pc;
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
    ;   ( Parts = ["call", PredArity] ; Parts = ["call", Pred, ArityStr] )
    ->  (   Parts = ["call", PredArity] -> PA = PredArity
        ;   strip_arity_local(Pred, Name), format(string(PA), "~w/~w", [Name, ArityStr])
        ),
        js_y_live_across(Rest, Live),
        emit_call(PA, Ind, Live)
    ;   emit_line_parts(Parts, Ind)
    ),
    emit_lines(Rest, Ind).

emit_line_parts(["proceed"], I) :- !, format("~wreturn true;~n", [I]).
emit_line_parts(["fail"], I) :- !, format("~wreturn false;~n", [I]).
emit_line_parts(["call", PredArity], I) :- !, emit_call(PredArity, I, false).
emit_line_parts(["call", Pred, ArityStr], I) :- !,
    strip_arity_local(Pred, Name), format(string(PA), "~w/~w", [Name, ArityStr]), emit_call(PA, I, false).
emit_line_parts(["execute", PredArity], I) :- !, emit_execute(PredArity, I).
emit_line_parts(["execute", Pred, ArityStr], I) :- !,
    strip_arity_local(Pred, Name), format(string(PA), "~w/~w", [Name, ArityStr]), emit_execute(PA, I).
% Allocate / Deallocate go through Runtime.op_* so the JS Y-register
% snapshot convention (not Lua's locals = {}) is preserved, without
% allocating an I.Allocate() instruction object per call.
emit_line_parts(["allocate"], I) :- !,
    format("~wif (Runtime.op_allocate(state) !== true) return false;~n", [I]).
emit_line_parts(["deallocate"], I) :- !,
    format("~wif (Runtime.op_deallocate(state) !== true) return false;~n", [I]).
emit_line_parts(["get_level", Yn], I) :- !,
    wam_javascript_target:reg_to_int(Yn, YI),
    format("~wif (Runtime.op_get_level(state, ~w) !== true) return false;~n", [I, YI]).
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
emit_line_parts(["get_constant", C, R], I) :- !,
    wam_javascript_target:reg_to_int(R, RI),
    wam_javascript_target:constant_to_js_term(C, T),
    format("~wif (Runtime.op_get_constant(program, state, ~w, ~w) !== true) return false;~n", [I, RI, T]).
emit_line_parts(["get_nil", R], I) :- !,
    wam_javascript_target:intern_js_atom("[]", Id),
    wam_javascript_target:reg_to_int(R, RI),
    format("~wif (Runtime.op_get_constant(program, state, ~w, V.Atom(~w)) !== true) return false;~n", [I, RI, Id]).
emit_line_parts(["get_integer", NStr, R], I) :- !,
    (number_string(N, NStr) -> true ; N = NStr),
    wam_javascript_target:reg_to_int(R, RI),
    format("~wif (Runtime.op_get_constant(program, state, ~w, V.Int(~w)) !== true) return false;~n", [I, RI, N]).
emit_line_parts(["get_value", X, A], I) :- !,
    wam_javascript_target:reg_to_int(X, XI),
    wam_javascript_target:reg_to_int(A, AI),
    format("~wif (Runtime.op_get_value(program, state, ~w, ~w) !== true) return false;~n", [I, XI, AI]).
emit_line_parts(["get_list", R], I) :- !,
    wam_javascript_target:reg_to_int(R, RI),
    wam_javascript_target:intern_js_atom("[|]", Id),
    format("~wif (Runtime.op_get_list(program, state, ~w, ~w) !== true) return false;~n", [I, RI, Id]).
emit_line_parts(["get_structure", F, R], I) :- !,
    wam_javascript_target:reg_to_int(R, RI),
    wam_javascript_target:parse_functor_arity(F, Name, Arity),
    wam_javascript_target:intern_js_atom(Name, Id),
    format("~wif (Runtime.op_get_structure(program, state, ~w, ~w, ~w) !== true) return false;~n", [I, Id, RI, Arity]).
emit_line_parts(["put_structure", F, R], I) :- !,
    wam_javascript_target:reg_to_int(R, RI),
    wam_javascript_target:parse_functor_arity(F, Name, Arity),
    wam_javascript_target:intern_js_atom(Name, Id),
    format("~wif (Runtime.op_put_structure(program, state, ~w, ~w, ~w) !== true) return false;~n", [I, Id, RI, Arity]).
emit_line_parts(["put_list", R], I) :- !,
    wam_javascript_target:reg_to_int(R, RI),
    wam_javascript_target:intern_js_atom("[|]", Id),
    format("~wif (Runtime.op_put_list(program, state, ~w, ~w) !== true) return false;~n", [I, RI, Id]).
emit_line_parts(["unify_variable", X], I) :- !,
    wam_javascript_target:reg_to_int(X, XI),
    format("~wif (Runtime.op_unify_variable(state, ~w) !== true) return false;~n", [I, XI]).
emit_line_parts(["unify_value", X], I) :- !,
    wam_javascript_target:reg_to_int(X, XI),
    format("~wif (Runtime.op_unify_value(program, state, ~w) !== true) return false;~n", [I, XI]).
emit_line_parts(["unify_constant", C], I) :- !,
    wam_javascript_target:constant_to_js_term(C, T),
    format("~wif (Runtime.op_unify_constant(program, state, ~w) !== true) return false;~n", [I, T]).
emit_line_parts(["set_variable", X], I) :- !,
    wam_javascript_target:reg_to_int(X, XI),
    format("~wif (Runtime.op_unify_variable(state, ~w) !== true) return false;~n", [I, XI]).
emit_line_parts(["set_value", X], I) :- !,
    wam_javascript_target:reg_to_int(X, XI),
    format("~wif (Runtime.op_unify_value(program, state, ~w) !== true) return false;~n", [I, XI]).
emit_line_parts(["set_constant", C], I) :- !,
    wam_javascript_target:constant_to_js_term(C, T),
    format("~wif (Runtime.op_unify_constant(program, state, ~w) !== true) return false;~n", [I, T]).
emit_line_parts(["builtin_call", Pred, ArityStr], I) :- !,
    (   number(ArityStr)
    ->  Arity = ArityStr
    ;   number_string(Arity, ArityStr)
    ),
    wam_javascript_target:js_string_literal(Pred, P),
    format("~wif (Runtime.op_builtin(program, state, ~w, ~w) !== true) return false;~n", [I, P, Arity]).
emit_line_parts(Parts, I) :-
    wam_javascript_target:wam_parts_to_js(Parts, [], Lit),
    format("~wif (Runtime.step(program, state, ~w) !== true) return false;~n", [I, Lit]).

% Call of a JS WAM builtin (sub_string/5, …): first-solution, same as
% interpreter Call → try_builtin_fallback. Do not Y-save (interpreter
% Call of a builtin does not) and do not go through I.Call.
emit_call(PredArity, I, _Protect) :-
    parse_call_pred_arity(PredArity, PredName, Arity),
    js_wam_builtin_functor(PredName), !,
    wam_javascript_target:js_string_literal(PredName, PQ),
    format("~wif (Runtime.op_builtin(program, state, ~w, ~w) !== true) return false;~n",
           [I, PQ, Arity]).
emit_call(PredArity, I, Protect) :-
    parse_call_pred_arity(PredArity, PredName, Arity),
    atom_string(PredAtom, PredName),
    js_lowered_func_name(PredAtom/Arity, FuncName),
    wam_javascript_target:js_string_literal(PredArity, Q),
    wam_javascript_target:js_string_literal(PredName, PQ),
    % Direct JS call: the callee's Proceed is `return`, so Call does not
    % touch cp/pc. typeof is false when mixed([subset]) left it interpreted.
    format("~wif (typeof ~w === \"function\") {~n", [I, FuncName]),
    % Choice points a lowered callee leaves (T5/T6 unbound-A1 dispatch)
    % are not resumable from a lowered frame: their snapshot carries the
    % LOWERED CALLER's cp, so backtracking into one resumes the
    % interpreter past this call site and prints garbage (probe P01 in
    % emit_mode(mixed) gave `r(1,one) _V2 _V2`). Drop them, matching
    % Runtime.run_isolated: an inner lowered call is honestly
    % first-solution rather than silently wrong.
    format("~w  const _cpd = state.cps.length;~n", [I]),
    (   Protect == true
    ->  % Caller's Y is live across this Call (parse_args/2 after
        % default_registry). Match invoke_lowered_call: Y-snapshot +
        % mode/build/read restore so intern write-mode cannot clobber Y.
        % push_call_frame = Y-save + cut barrier (WAM call: B0 <- B), so
        % a `!` in the callee prunes only the callee's alternatives.
        format("~w  Runtime.push_call_frame(state);~n", [I]),
        format("~w  const _ok = Runtime.run_lowered_body(program, state, ~w);~n", [I, FuncName]),
        format("~w  Runtime.pop_call_frame(state);~n", [I]),
        format("~w  while (state.cps.length > _cpd) state.cps.pop();~n", [I]),
        format("~w  if (_ok !== true) return false;~n", [I])
    ;   % WAM call: B0 <- B. Without the barrier a neck cut in the callee
        % prunes the CALLER's choice points (the D44 bug shape, lowered
        % side). Two array ops; the interpreter Call pays the same.
        format("~w  Runtime.push_cut_barrier(state);~n", [I]),
        format("~w  if (~w(program, state) !== true) {~n", [I, FuncName]),
        format("~w    Runtime.pop_cut_barrier(state);~n", [I]),
        format("~w    while (state.cps.length > _cpd) state.cps.pop();~n", [I]),
        format("~w    return false;~n", [I]),
        format("~w  }~n", [I]),
        format("~w  Runtime.pop_cut_barrier(state);~n", [I]),
        format("~w  while (state.cps.length > _cpd) state.cps.pop();~n", [I])
    ),
    format("~w} else {~n", [I]),
    format("~w  const saved_cp = state.cp;~n", [I]),
    format("~w  const saved_pc = state.pc;~n", [I]),
    format("~w  const target = program.labels[~w];~n", [I, Q]),
    format("~w  let _ok = true;~n", [I]),
    format("~w  if (target !== undefined && target !== null) {~n", [I]),
    % push_call_frame (not bare push_y_save): the interpreted callee's
    % Proceed runs pop_call_frame, which pops a cut_stack entry too. With
    % only a Y-save pushed it popped THIS frame's barrier, so a `!` after
    % the call pruned the caller's choice points (probe P28).
    format("~w    const _fm = Runtime.call_frame_mark(state);~n", [I]),
    format("~w    Runtime.push_call_frame(state);~n", [I]),
    format("~w    state.cp = 0;~n", [I]),
    format("~w    state.pc = target;~n", [I]),
    format("~w    state.program = program;~n", [I]),
    format("~w    _ok = Runtime.run_isolated(program, state) === true;~n", [I]),
    format("~w    state.halt = false;~n", [I]),
    format("~w    Runtime.call_frame_release(state, _fm);~n", [I]),
    format("~w  } else if (Runtime.step(program, state, I.Call(~w, ~w)) !== true) {~n", [I, PQ, Arity]),
    format("~w    _ok = false;~n", [I]),
    format("~w  }~n", [I]),
    format("~w  state.cp = saved_cp;~n", [I]),
    format("~w  state.pc = saved_pc;~n", [I]),
    format("~w  state.halt = false;~n", [I]),
    format("~w  if (!_ok) return false;~n", [I]),
    format("~w}~n", [I]).

%% js_y_live_across(+LinesAfterCall, -Live)
%  True when a Y register is read/written after Call before the frame
%  is deallocated or the clause ends. Those Calls need a Y snapshot.
js_y_live_across([], false).
js_y_live_across([Line|Rest], Live) :-
    tokenize_line(Line, Parts),
    (   Parts == [] -> js_y_live_across(Rest, Live)
    ;   Parts = [F|_], sub_string(F, _, 1, 0, ":") -> js_y_live_across(Rest, Live)
    ;   ( Parts == ["deallocate"] ; Parts == ["proceed"]
        ; Parts == ["fail"] ; Parts = ["execute"|_] )
    ->  Live = false
    ;   js_parts_mention_y(Parts)
    ->  Live = true
    ;   js_y_live_across(Rest, Live)
    ).

js_parts_mention_y(Parts) :-
    member(P, Parts),
    (   sub_string(P, 0, 1, _, "Y")
    ;   sub_string(P, 0, 1, _, "y")
    ).

emit_execute(PredArity, I) :-
    parse_call_pred_arity(PredArity, PredName, Arity),
    wam_javascript_target:js_string_literal(PredName, PQ),
    js_wam_builtin_functor(PredName), !,
    % Last-goal Execute of a JS WAM builtin. JS return is Proceed —
    % I.Execute would proceed_to_cp and steal/halt the query.
    format("~wreturn Runtime.op_builtin(program, state, ~w, ~w) === true;~n",
           [I, PQ, Arity]).
emit_execute(PredArity, I) :-
    parse_call_pred_arity(PredArity, PredName, Arity),
    atom_string(PredAtom, PredName),
    js_lowered_func_name(PredAtom/Arity, FuncName),
    wam_javascript_target:js_string_literal(PredArity, Q),
    wam_javascript_target:js_string_literal(PredName, PQ),
    % Lowered-to-lowered Execute: JS return IS Proceed. Do not proceed_to_cp
    % and do not touch cp (the caller's continuation stays in state.cp).
    % WAM execute still does B0 <- B: enter_execute rebases the cut
    % barrier onto the tail-called predicate so its `!` cannot prune the
    % choice points the CALLER left behind (probes P01/P33/P34).
    format("~wif (typeof ~w === \"function\") {~n", [I, FuncName]),
    format("~w  Runtime.enter_execute(state);~n", [I]),
    format("~w  return ~w(program, state) === true;~n", [I, FuncName]),
    format("~w}~n", [I]),
    format("~w{~n", [I]),
    format("~w  const target = program.labels[~w];~n", [I, Q]),
    format("~w  if (target !== undefined && target !== null) {~n", [I]),
    % Interpreted user callee: run until THAT predicate Proceeds. Setting
    % cp=0 inside execute_user_isolated makes Proceed halt the isolated
    % interpreter instead of jumping the query continuation; the helper
    % then restores the saved cp so this function's `return` is Proceed.
    format("~w    return Runtime.execute_user_isolated(program, state, target) === true;~n", [I]),
    format("~w  }~n", [I]),
    format("~w  return Runtime.op_builtin(program, state, ~w, ~w) === true;~n", [I, PQ, Arity]),
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
        % Intern-once is sound only for a unit ground fact (one answer,
        % independent of inputs). get_value / get_variable alias a head
        % variable across arguments — packages(catalog(Ps,_,_,_,_,_), Ps)
        % — and must not intern the first call's registers as *the* answer.
        % Nested ground compounds (default_registry/1, memo_fact(g(a,[1,2,3])))
        % use unify_variable as a temp then get_structure/get_list on it.
        memberchk(Op, ["proceed",
                       "get_constant",
                       "get_structure", "get_list", "get_nil", "get_integer",
                       "unify_variable", "unify_value", "unify_constant",
                       "put_constant", "put_variable", "put_value",
                       "put_structure", "put_list",
                       "set_variable", "set_value", "set_constant"])
    ).

%% js_clause_op_fingerprint(+Lines, -Fingerprint)
%  Opcode sequence of a clause body, labels and empty lines stripped.
%  Used to collapse switch_on_* duplicate copies of the same ground fact
%  into one intern-once deterministic body.
js_clause_op_fingerprint(Lines, Fingerprint) :-
    findall(Op, (
        member(Line, Lines),
        tokenize_line(Line, Tokens),
        Tokens \= [],
        Tokens = [Tok|_],
        \+ sub_string(Tok, _, 1, 0, ":"),
        Op = Tok
    ), Ops),
    atomic_list_concat(Ops, '|', Fingerprint).

js_pred_arity_from_name(PredName, Arity) :-
    atom_string(PredName, S),
    split_string(S, "/", "", Parts),
    last(Parts, ArStr),
    number_string(Arity, ArStr).

%% wam_javascript_explain_lower(+PI, +WamCode, -Decision)
%  Decision = lower(Reason) | fallback(Why). Never fails.
wam_javascript_explain_lower(PI, WamCode, lower(Reason)) :-
    catch(wam_javascript_lowerable(PI, WamCode, Reason), _, fail), !.
wam_javascript_explain_lower(PI, _WamCode, fallback(Why)) :-
    js_pi_needs_naked_member_cps(PI), !,
    Why = 'naked member/2 (or callee) needs interpreter choice points'.
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

%% js_pi_needs_naked_member_cps(+PI)
%  True when this predicate (or a user callee) uses a NONDETERMINISTIC
%  BUILTIN outside findall/bagof/setof/once/aggregate_all/forall.
%  Lowering would keep only the first witness.
%
%  A lowered JS body is straight-line: when a goal fails there is no
%  machinery to retry an earlier goal's choice point, and any choice
%  point the goal left on state.cps is unreachable from the lowered
%  frame (Runtime.run_isolated now drops those rather than letting a
%  later backtrack resume the interpreter incoherently). The rule used
%  to name member/2 only; every builtin in js_nondet_builtin/2 has the
%  same shape, and `between(1, 3, X), X > 1, !` was the probe (P20) that
%  showed member/2 was not special.
js_pi_needs_naked_member_cps(PI) :-
    js_pi_functor_arity(PI, F, A),
    js_functor_needs_naked_member(F, A, []).

js_pi_functor_arity(_M:F/A, F, A) :- !.
js_pi_functor_arity(F/A, F, A).

% A CALLEE with more than one distinct clause can succeed more than once,
% so calling it leaves a choice point. A lowered body cannot resume one:
% either the callee is itself lowered and its T5/T6 unbound-A1 choice
% point snapshots the LOWERED CALLER's continuation, or it is interpreted
% and Runtime.run_isolated drops what it left. Either way the caller only
% ever sees the first solution, so the caller must stay on the
% interpreter. Visited == [] is the predicate being classified: its own
% clause count is fine (T4/T5/T6 dispatch it), only its CALLEES matter.
%
% Clauses are counted up to variant equality: a fixture that re-asserts
% the same fact must not look nondeterministic.
js_functor_needs_naked_member(F, A, Visited) :-
    Visited \== [],
    \+ memberchk(F/A, Visited),
    js_distinct_clause_count(F, A, N),
    N > 1, !.
js_functor_needs_naked_member(F, A, Visited) :-
    \+ memberchk(F/A, Visited),
    functor(Head, F, A),
    catch(clause(user:Head, Body), _, fail),
    js_body_naked_member(Body, [F/A|Visited]).

js_distinct_clause_count(F, A, N) :-
    js_distinct_clauses(F, A, Uniq),
    length(Uniq, N).

% Deduplicated but ORDER-PRESERVING: js_clauses_cut_committed/1 reads
% "every clause but the last", which is only meaningful in source order.
js_distinct_clauses(F, A, Uniq) :-
    functor(Head, F, A),
    findall(C,
            (   catch(clause(user:Head, Body), _, fail),
                copy_term(Head-Body, C),
                numbervars(C, 0, _)
            ), Cs),
    js_dedupe_ordered(Cs, [], Uniq).

js_dedupe_ordered([], _, []).
js_dedupe_ordered([C|Cs], Seen, Out) :-
    (   memberchk(C, Seen)
    ->  js_dedupe_ordered(Cs, Seen, Out)
    ;   Out = [C|More],
        js_dedupe_ordered(Cs, [C|Seen], More)
    ).

%% js_pi_may_yield_many(+PI)
%  True when more than one clause of PI can succeed for the same call, so
%  a first-solution lowering would hide answers. False (safe to lower)
%  when the heads are first-argument mutually exclusive, or when every
%  clause but the last commits with a top-level cut.
js_pi_may_yield_many(PI) :-
    js_pi_functor_arity(PI, F, A),
    js_distinct_clauses(F, A, Clauses),
    Clauses = [_, _ | _],
    \+ js_clauses_first_arg_exclusive(Clauses),
    \+ js_clauses_cut_committed(Clauses).

js_clauses_first_arg_exclusive(Clauses) :-
    findall(K, ( member(H-_, Clauses), js_first_arg_key(H, K) ), Keys),
    length(Clauses, N),
    length(Keys, N),
    sort(Keys, Sorted),
    length(Sorted, N).

js_first_arg_key(Head, Key) :-
    compound(Head),
    arg(1, Head, A1),
    nonvar(A1),
    % js_distinct_clauses/3 numbervars its output, so a clause-head
    % variable arrives as '$VAR'(N) -- still a variable for indexing.
    A1 \= '$VAR'(_),
    (   compound(A1)
    ->  functor(A1, KF, KA), Key = KF/KA
    ;   Key = c(A1)
    ).

% Every clause but the last reaches a top-level `!` (a conjunction chain
% cut, not one buried in call/1, \+ or an if-then-else branch).
js_clauses_cut_committed(Clauses) :-
    append(AllButLast, [_], Clauses),
    AllButLast \== [],
    forall(member(_-B, AllButLast), js_body_top_level_cut(B)).

js_body_top_level_cut(B) :- var(B), !, fail.
js_body_top_level_cut(!) :- !.
js_body_top_level_cut((A, B)) :- !,
    (   js_body_top_level_cut(A)
    ;   js_body_top_level_cut(B)
    ).
js_body_top_level_cut(_) :- fail.

js_body_naked_member(Goal, _) :-
    var(Goal), !, fail.
js_body_naked_member(_Mod:G, V) :-
    !,
    js_body_naked_member(G, V).
js_body_naked_member(findall(_, _, _), _) :- !, fail.
js_body_naked_member(bagof(_, _, _), _) :- !, fail.
js_body_naked_member(setof(_, _, _), _) :- !, fail.
js_body_naked_member(aggregate_all(_, _, _), _) :- !, fail.
js_body_naked_member(aggregate_all(_, _, _, _), _) :- !, fail.
js_body_naked_member(once(_), _) :- !, fail.
js_body_naked_member(forall(_, _), _) :- !, fail.
js_body_naked_member(G, _) :-
    nonvar(G),
    functor(G, F, A),
    js_nondet_builtin(F, A), !.
js_body_naked_member((A, B), V) :- !,
    (   js_body_naked_member(A, V)
    ;   js_body_naked_member(B, V)
    ).
js_body_naked_member((A; B), V) :- !,
    (   js_body_naked_member(A, V)
    ;   js_body_naked_member(B, V)
    ).
js_body_naked_member((A -> B), V) :- !,
    (   js_body_naked_member(A, V)
    ;   js_body_naked_member(B, V)
    ).
js_body_naked_member(\+ A, V) :- !,
    js_body_naked_member(A, V).
js_body_naked_member(G, V) :-
    functor(G, F, A),
    A > 0,
    \+ js_body_skip_functor(F),
    js_functor_needs_naked_member(F, A, V).

%% js_nondet_builtin(+Name, +Arity)
%  Library builtins that can leave a choice point on state.cps. A lowered
%  body cannot resume one, so any predicate reaching one of these outside
%  a commit wrapper stays on the interpreter.
%  The list is exactly the builtins that push onto state.cps in
%  runtime.js.mustache. Every other library predicate there is
%  implemented semi-deterministically (select/3, nth0/3, append/3,
%  sub_atom/5, ... return one solution and push nothing), so they cannot
%  leak a choice point into a lowered body and must NOT be listed --
%  listing them only refuses lowering that is in fact safe.
js_nondet_builtin(member, 2).
js_nondet_builtin(between, 3).

js_body_skip_functor(',').
js_body_skip_functor(';').
js_body_skip_functor('->').
js_body_skip_functor(\+).
js_body_skip_functor(!).
js_body_skip_functor(=).
js_body_skip_functor(==).
js_body_skip_functor(\==).
js_body_skip_functor(is).
js_body_skip_functor(=:=).
js_body_skip_functor(=\=).
js_body_skip_functor(<).
js_body_skip_functor(>).
js_body_skip_functor(=<).
js_body_skip_functor(>=).
js_body_skip_functor(true).
js_body_skip_functor(fail).
js_body_skip_functor(member).
js_body_skip_functor(findall).
js_body_skip_functor(bagof).
js_body_skip_functor(setof).
js_body_skip_functor(write).
js_body_skip_functor(nl).
js_body_skip_functor(sub_string).
js_body_skip_functor(sub_atom).
js_body_skip_functor(sort).
js_body_skip_functor(reverse).
js_body_skip_functor(append).
js_body_skip_functor(length).

js_pi_key(_M:Pred/Arity, Key) :- !,
    format(atom(Key), '~w/~w', [Pred, Arity]).
js_pi_key(Pred/Arity, Key) :-
    format(atom(Key), '~w/~w', [Pred, Arity]).

%% js_wam_has_builtin_execute(+WamCode, +SelfKey)
%  True when some Execute targets a JS WAM builtin (sub_string/5, …).
%  Not a fallback: emit_execute / emit_call emit Runtime.op_builtin.
%  User-predicate Execute (self or other) preserves the caller's CP.
js_wam_has_builtin_execute(WamCode, Self) :-
    atom_string(Self, SelfS),
    atom_string(WamCode, S),
    split_string(S, "\n", "", Lines),
    member(Line, Lines),
    tokenize_line(Line, Parts),
    js_execute_target(Parts, TargetS),
    TargetS \== SelfS,
    js_execute_is_wam_builtin(TargetS).

js_execute_is_wam_builtin(TargetS) :-
    js_execute_functor(TargetS, Name),
    js_wam_builtin_functor(Name).

js_execute_functor(TargetS, Name) :-
    (   sub_string(TargetS, B, 1, After, "/"),
        sub_string(TargetS, _, After, 0, ArStr),
        number_string(_, ArStr)
    ->  sub_string(TargetS, 0, B, _, Name)
    ;   Name = TargetS
    ).

% ISO / JS WAM builtins that appear as last-goal Execute in wrappers.
% User predicates that merely share a name are not expected in this tree.
js_wam_builtin_functor("sub_string").
js_wam_builtin_functor("sub_atom").
js_wam_builtin_functor(sub_string).
js_wam_builtin_functor(sub_atom).

js_execute_target(["execute", PA], TargetS) :-
    atom_string(PA, TargetS).
js_execute_target(["execute", Pred, ArityStr], TargetS) :-
    format(string(TargetS), "~w/~w", [Pred, ArityStr]).

%% js_ite_pure_test_cond(+Structured)
%  Condition is register moves + a comparison builtin (==, =:=, ...).
%  Those builtins never trail or Allocate, so ITE else restores A1-A8
%  and the trail mark instead of snapshot_machine (full register copy).
js_ite_pure_test_cond(Items) :-
    Items \== [],
    forall(member(I, Items), js_ite_pure_test_item(I)).

js_ite_pure_test_item(ite(C, T, E)) :- !,
    js_ite_pure_test_cond(C),
    ( T == [] -> true ; js_ite_pure_test_cond(T) ),
    ( E == [] -> true ; js_ite_pure_test_cond(E) ).
js_ite_pure_test_item(builtin_call(Op, _)) :-
    js_pure_compare_builtin(Op).
js_ite_pure_test_item(line(Parts)) :-
    Parts = [Op|_],
    memberchk(Op, ["put_value", "put_constant", "put_variable",
                   "get_variable", "get_value"]).

js_pure_compare_builtin(Op) :-
    atom_string(Op, S),
    memberchk(S, ["==/2", "\\==/2", "=:=/2", "=\\=/2",
                  ">/2", "</2", ">=/2", "=</2",
                  "==", "\\==", "=:=", "=\\=", ">", "<", ">=", "=<"]).

%% js_ite_cond_needs_full_snap(+Structured)
%  Call / Execute / Allocate in the condition can push CPs or Y frames;
%  those still need snapshot_machine. Unifying put_structure / =/2 do not.
js_ite_cond_needs_full_snap(Items) :-
    member(I, Items),
    js_ite_item_full_snap(I).

js_ite_item_full_snap(ite(C, T, E)) :-
    (   js_ite_cond_needs_full_snap(C)
    ;   js_ite_cond_needs_full_snap(T)
    ;   js_ite_cond_needs_full_snap(E)
    ).
js_ite_item_full_snap(line(["call"|_])).
js_ite_item_full_snap(line(["execute"|_])).
js_ite_item_full_snap(line(["allocate"])).
js_ite_item_full_snap(line(["deallocate"])).
