:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_clojure_lowered_emitter.pl — WAM-lowered Clojure emission
%
% First scaffold slice:
%   - parses WAM text into a small instruction IR
%   - classifies whether a predicate can be lowered directly
%   - emits one Clojure function per lowerable predicate
%
% This mirrors the Rust lowered-emitter shape conservatively. The first
% version is intentionally narrow and is not yet wired into
% wam_clojure_target.pl routing.

:- module(wam_clojure_lowered_emitter, [
    wam_clojure_lowerable/3,
    lower_predicate_to_clojure/4,
    is_deterministic_pred_clojure/1,
    clojure_lowered_func_name/2
]).

:- use_module(library(lists)).
:- use_module(wam_ite_structurer, [structure_ite/2]).

% =====================================================================
% Parsing
% =====================================================================

parse_wam_text(WamText, Instrs) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_lines(Lines, Instrs).

% Label-preserving parse: keeps label(Name) markers so structure_ite can
% locate the else/cont boundaries of an (C -> T ; E) / \+ / once block.
parse_wam_text_labeled(WamText, Instrs) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_lines_labeled(Lines, Instrs).

parse_lines_labeled([], []).
parse_lines_labeled([Line|Rest], Instrs) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  parse_lines_labeled(Rest, Instrs)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelStr),
            Instrs = [label(LabelStr)|More],
            parse_lines_labeled(Rest, More)
        ;   instr_from_parts(CleanParts, Instr)
        ->  Instrs = [Instr|More],
            parse_lines_labeled(Rest, More)
        ;   parse_lines_labeled(Rest, Instrs)
        )
    ).

parse_lines([], []).
parse_lines([Line|Rest], Instrs) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  parse_lines(Rest, Instrs)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  parse_lines(Rest, Instrs)
        ;   instr_from_parts(CleanParts, Instr)
        ->  Instrs = [Instr|RestInstrs],
            parse_lines(Rest, RestInstrs)
        ;   parse_lines(Rest, Instrs)
        )
    ).

instr_from_parts(["get_constant", C, Ai], get_constant(C, Ai)).
instr_from_parts(["get_variable", Xn, Ai], get_variable(Xn, Ai)).
instr_from_parts(["get_value", Xn, Ai], get_value(Xn, Ai)).
instr_from_parts(["get_structure", F, Ai], get_structure(F, Ai)).
instr_from_parts(["get_list", Ai], get_list(Ai)).
instr_from_parts(["get_nil", Ai], get_nil(Ai)).
instr_from_parts(["get_integer", N, Ai], get_integer(N, Ai)).
instr_from_parts(["unify_variable", Xn], unify_variable(Xn)).
instr_from_parts(["unify_value", Xn], unify_value(Xn)).
instr_from_parts(["unify_constant", C], unify_constant(C)).
instr_from_parts(["put_variable", Xn, Ai], put_variable(Xn, Ai)).
instr_from_parts(["put_value", Xn, Ai], put_value(Xn, Ai)).
instr_from_parts(["put_constant", C, Ai], put_constant(C, Ai)).
instr_from_parts(["put_structure", F, Ai], put_structure(F, Ai)).
instr_from_parts(["put_list", Ai], put_list(Ai)).
instr_from_parts(["set_variable", Xn], set_variable(Xn)).
instr_from_parts(["set_value", Xn], set_value(Xn)).
instr_from_parts(["set_constant", C], set_constant(C)).
instr_from_parts(["call", P, N], call(P, N)).
instr_from_parts(["execute", P], execute(P)).
instr_from_parts(["proceed"], proceed).
instr_from_parts(["fail"], fail).
instr_from_parts(["allocate"], allocate).
instr_from_parts(["deallocate"], deallocate).
instr_from_parts(["builtin_call", Op, Ar], builtin_call(Op, Ar)).
instr_from_parts(["call_foreign", Pred, Ar], call_foreign(Pred, Ar)).
instr_from_parts(["switch_on_constant"|Cases], switch_on_constant(Cases)).
instr_from_parts(["try_me_else", L], try_me_else(L)).
instr_from_parts(["retry_me_else", L], retry_me_else(L)).
instr_from_parts(["trust_me"], trust_me).
instr_from_parts(["jump", L], jump(L)).
instr_from_parts(["cut_ite"], cut_ite).

% =====================================================================
% Lowerability
% =====================================================================

wam_clojure_lowerable(PI, WamCode, Reason) :-
    (   is_list(WamCode) -> Instrs = WamCode
    ;   atom(WamCode) -> parse_wam_text(WamCode, Instrs)
    ;   atom_string(WamCode, _), parse_wam_text(WamCode, Instrs)
    ),
    clause1_instrs(Instrs, C1),
    forall(member(I, C1), clojure_supported(I)),
    (   is_deterministic_pred_clojure(Instrs)
    ->  Reason = deterministic
    ;   Reason = multi_clause_1
    ),
    ( PI = _M:_P/_A -> true ; PI = _/_A2 -> true ; true ).

clause1_instrs([], []).
clause1_instrs([try_me_else(_)|Rest], C1) :- !,
    take_to_terminal(Rest, C1).
clause1_instrs(Instrs, Instrs).

take_to_terminal([], []).
take_to_terminal([proceed|_], [proceed]) :- !.
take_to_terminal([fail|_], [fail]) :- !.
take_to_terminal([I|Rest], [I|More]) :-
    take_to_terminal(Rest, More).

is_deterministic_pred_clojure(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

clojure_supported(allocate).
clojure_supported(deallocate).
clojure_supported(get_constant(_, _)).
clojure_supported(get_variable(_, _)).
clojure_supported(get_value(_, _)).
clojure_supported(get_structure(_, _)).
clojure_supported(get_list(_)).
clojure_supported(get_nil(_)).
clojure_supported(get_integer(_, _)).
clojure_supported(unify_variable(_)).
clojure_supported(unify_value(_)).
clojure_supported(unify_constant(_)).
clojure_supported(put_constant(_, _)).
clojure_supported(put_variable(_, _)).
clojure_supported(put_value(_, _)).
clojure_supported(put_structure(_, _)).
clojure_supported(put_list(_)).
clojure_supported(set_variable(_)).
clojure_supported(set_value(_)).
clojure_supported(set_constant(_)).
clojure_supported(call(_, _)).
clojure_supported(execute(_)).
clojure_supported(proceed).
clojure_supported(fail).
clojure_supported(builtin_call(_, _)).
clojure_supported(call_foreign(_, _)).
clojure_supported(try_me_else(_)).
clojure_supported(trust_me).
clojure_supported(cut_ite).
clojure_supported(jump(_)).

% =====================================================================
% Function name generation
% =====================================================================

clojure_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, FStr),
    sanitize_clojure_ident(FStr, SanStr),
    format(atom(Name), 'lowered-~w-~w', [SanStr, Arity]).

sanitize_clojure_ident(In, Out) :-
    string_codes(In, Codes),
    maplist(clojure_safe_code, Codes, OutCodes),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

clojure_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z -> true
    ;   C >= 0'A, C =< 0'Z -> true
    ;   C >= 0'0, C =< 0'9 -> true
    ;   C =:= 0'- -> true
    ),
    !.
clojure_safe_code(0'_, 0'-) :- !.
clojure_safe_code(_, 0'-).

% =====================================================================
% Emission
% =====================================================================

lower_predicate_to_clojure(PI, WamCode, _Options, ClojureCode) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    clojure_lowered_func_name(Pred/Arity, FuncName),
    (   is_list(WamCode) -> Instrs = WamCode
    ;   parse_wam_text(WamCode, Instrs)
    ),
    clause1_instrs(Instrs, C1Instrs0),
    (   is_deterministic_pred_clojure(Instrs)
    ->  lowered_direct_prefix(C1Instrs0, allow_control, C1Instrs),
        with_output_to(string(Body), emit_instrs(C1Instrs, "  "))
    ;   clojure_structured_clause1(WamCode, Structured)
    ->  % Single-clause if-then-else / negation / once: emit native
        % branching instead of the no-op stub (which delegated to run-wam).
        with_output_to(string(Body), emit_struct_clj_body(Structured, "  "))
    ;   % Multi-clause or unstructurable: keep the no-op stub; run-wam
        % interprets the predicate from start-pc (sound fallback).
        with_output_to(string(Body), emit_instrs([], "  "))
    ),
    format(string(ClojureCode),
';; ~w — lowered from ~w/~w
(defn ~w [state]
~w)
', [FuncName, Pred, Arity, FuncName, Body]).

emit_instrs([], Indent) :-
    format("~wstate~n", [Indent]).
emit_instrs(Instrs, Indent) :-
    length(Instrs, Len),
    format("~w(let [s0 state~n", [Indent]),
    emit_instr_bindings(Instrs, 0, Indent),
    format("~w      ]~n", [Indent]),
    format("~w  s~w)~n", [Indent, Len]).

emit_instr_bindings([], _, _).
emit_instr_bindings([Instr|Rest], Index, Indent) :-
    NextIndex is Index + 1,
    format(atom(InState), 's~w', [Index]),
    emit_lowered_expr(Instr, InState, Expr),
    instr_comment(Instr, Comment),
    format("~w      ;; ~w~n", [Indent, Comment]),
    format("~w      s~w (if (= :running (:status ~w)) ~w ~w)~n",
           [Indent, NextIndex, InState, Expr, InState]),
    emit_instr_bindings(Rest, NextIndex, Indent).

% =====================================================================
% Structured ITE emission (shared nesting-aware structurer)
% =====================================================================
%
%  Clojure's flat `(let [s0 state s1 ...] sN)` threading cannot express an
%  if-then-else (there is no jump), so the previous emitter produced a
%  no-op stub for any predicate containing try_me_else and relied on
%  run-wam (the interpreter) to execute it. This adds native branching:
%  clause 1 is folded by the shared structurer into ite(Cond,Then,Else)
%  and emitted as a let whose binding for the block is
%      (if (= :running (:status sCond)) <then from sCond> <else from sPre>)
%  i.e. run the condition; on success take Then (inheriting the condition's
%  bindings), else take Else from the PRE-condition state (discarding them)
%  — the same Maybe-style discipline as the Haskell/F# backends. A failed
%  condition builtin calls runtime/backtrack with no choice point in scope,
%  which yields :status :failed (not :running), so the `if` routes to Else.

%% clojure_structured_clause1(+WamCode, -Structured) is semidet.
clojure_structured_clause1(WamCode, Structured) :-
    \+ is_list(WamCode),                       % need the text to recover labels
    parse_wam_text_labeled(WamCode, LInstrs0),
    ( LInstrs0 = [label(_)|LInstrs1] -> true ; LInstrs1 = LInstrs0 ),
    \+ ( LInstrs1 = [try_me_else(_)|_] ),       % single-clause predicates only
    take_to_terminal(LInstrs1, C1L),
    structure_ite(C1L, Structured),
    \+ member(try_me_else(_), Structured),
    \+ member(trust_me, Structured),
    \+ member(retry_me_else(_), Structured),
    once(member(ite(_, _, _), Structured)).     % must actually contain an ITE

%% emit_struct_clj_body(+Structured, +Indent)
emit_struct_clj_body(Structured, Indent) :-
    nb_setval(clj_sv_ctr, 0),
    with_output_to(string(Bindings), emit_clj_bindings(Structured, "state", FinalSV, Indent)),
    format("~w(let [~n~w~w      ]~n~w  ~w)~n", [Indent, Bindings, Indent, Indent, FinalSV]).

fresh_clj_sv(SV) :-
    nb_getval(clj_sv_ctr, N), N1 is N + 1, nb_setval(clj_sv_ctr, N1),
    format(atom(SV), 's~w', [N1]).

%% emit_clj_bindings(+Structured, +InSV, -OutSV, +Indent)
emit_clj_bindings([], InSV, InSV, _Ind).
emit_clj_bindings([ite(C, T, E)|Rest], InSV, OutSV, Ind) :- !,
    emit_clj_bindings(C, InSV, CondSV, Ind),
    fresh_clj_sv(IteSV),
    emit_clj_branch(T, CondSV, ThenExpr, Ind),
    emit_clj_branch(E, InSV, ElseExpr, Ind),
    format("~w      ;; if-then-else~n", [Ind]),
    format("~w      ~w (if (= :running (:status ~w))~n", [Ind, IteSV, CondSV]),
    format("~w            ~w~n", [Ind, ThenExpr]),
    format("~w            ~w)~n", [Ind, ElseExpr]),
    emit_clj_bindings(Rest, IteSV, OutSV, Ind).
emit_clj_bindings([Instr|Rest], InSV, OutSV, Ind) :-
    fresh_clj_sv(SV),
    emit_lowered_expr(Instr, InSV, Expr),
    instr_comment(Instr, Comment),
    format("~w      ;; ~w~n", [Ind, Comment]),
    format("~w      ~w (if (= :running (:status ~w)) ~w ~w)~n", [Ind, SV, InSV, Expr, InSV]),
    emit_clj_bindings(Rest, SV, OutSV, Ind).

%% emit_clj_branch(+Instrs, +InSV, -Expr, +Indent)
%  A branch body as a self-contained Clojure expression (nested let, or
%  just InSV when the branch is empty).
emit_clj_branch(Instrs, InSV, Expr, Ind) :-
    with_output_to(string(B), emit_clj_bindings(Instrs, InSV, FSV, Ind)),
    (   B == ""
    ->  Expr = FSV
    ;   format(atom(Expr), '(let [~n~w~w            ] ~w)', [B, Ind, FSV])
    ).

lowered_direct_prefix(Instrs, Prefix) :-
    lowered_direct_prefix(Instrs, allow_control, Prefix).

lowered_direct_prefix([], _, []).
lowered_direct_prefix([Instr|_], ControlLowering, [Instr]) :-
    lowered_terminal_direct_instr(Instr, ControlLowering),
    !.
lowered_direct_prefix([Instr|Rest], ControlLowering, [Instr|PrefixRest]) :-
    lowered_direct_instr(Instr, ControlLowering),
    !,
    lowered_direct_prefix(Rest, ControlLowering, PrefixRest).
lowered_direct_prefix(_, _, []).

lowered_terminal_direct_instr(call(_, _), allow_control).
lowered_terminal_direct_instr(execute(_), allow_control).
lowered_terminal_direct_instr(jump(_), allow_control).
lowered_terminal_direct_instr(builtin_call(Op, Arity), _) :-
    clojure_terminal_builtin(Op, Arity).
lowered_terminal_direct_instr(proceed, _).
lowered_terminal_direct_instr(fail, _).

lowered_direct_instr(call(_, _), allow_control).
lowered_direct_instr(execute(_), allow_control).
lowered_direct_instr(jump(_), allow_control).
lowered_direct_instr(Instr, _) :-
    lowered_data_instr(Instr).

lowered_data_instr(allocate).
lowered_data_instr(deallocate).
lowered_data_instr(proceed).
lowered_data_instr(fail).
lowered_data_instr(get_constant(_, _)).
lowered_data_instr(get_structure(_, _)).
lowered_data_instr(get_list(_)).
lowered_data_instr(get_integer(_, _)).
lowered_data_instr(get_nil(_)).
lowered_data_instr(unify_constant(_)).
lowered_data_instr(unify_variable(_)).
lowered_data_instr(unify_value(_)).
lowered_data_instr(builtin_call(Op, Arity)) :-
    clojure_direct_builtin(Op, Arity).
lowered_data_instr(put_constant(_, _)).
lowered_data_instr(put_nil(_)).
lowered_data_instr(get_variable(_, _)).
lowered_data_instr(put_variable(_, _)).
lowered_data_instr(get_value(_, _)).
lowered_data_instr(put_value(_, _)).
lowered_data_instr(put_structure(_, _)).
lowered_data_instr(put_list(_)).
lowered_data_instr(set_constant(_)).
lowered_data_instr(set_variable(_)).
lowered_data_instr(set_value(_)).

clojure_direct_builtin("=/2", "2").
clojure_direct_builtin("=/2", 2).
clojure_direct_builtin('=/2', "2").
clojure_direct_builtin('=/2', 2).
clojure_direct_builtin("\\=/2", "2").
clojure_direct_builtin("\\=/2", 2).
clojure_direct_builtin('\\=/2', "2").
clojure_direct_builtin('\\=/2', 2).
clojure_direct_builtin("==/2", "2").
clojure_direct_builtin("==/2", 2).
clojure_direct_builtin('==/2', "2").
clojure_direct_builtin('==/2', 2).
clojure_direct_builtin("\\==/2", "2").
clojure_direct_builtin("\\==/2", 2).
clojure_direct_builtin('\\==/2', "2").
clojure_direct_builtin('\\==/2', 2).
clojure_direct_builtin("@</2", "2").
clojure_direct_builtin("@</2", 2).
clojure_direct_builtin('@</2', "2").
clojure_direct_builtin('@</2', 2).
clojure_direct_builtin("@=</2", "2").
clojure_direct_builtin("@=</2", 2).
clojure_direct_builtin('@=</2', "2").
clojure_direct_builtin('@=</2', 2).
clojure_direct_builtin("@>/2", "2").
clojure_direct_builtin("@>/2", 2).
clojure_direct_builtin('@>/2', "2").
clojure_direct_builtin('@>/2', 2).
clojure_direct_builtin("@>=/2", "2").
clojure_direct_builtin("@>=/2", 2).
clojure_direct_builtin('@>=/2', "2").
clojure_direct_builtin('@>=/2', 2).
clojure_direct_builtin("compare/3", "3").
clojure_direct_builtin("compare/3", 3).
clojure_direct_builtin('compare/3', "3").
clojure_direct_builtin('compare/3', 3).
clojure_direct_builtin("=:=/2", "2").
clojure_direct_builtin("=:=/2", 2).
clojure_direct_builtin('=:=/2', "2").
clojure_direct_builtin('=:=/2', 2).
clojure_direct_builtin("=\\=/2", "2").
clojure_direct_builtin("=\\=/2", 2).
clojure_direct_builtin('=\\=/2', "2").
clojure_direct_builtin('=\\=/2', 2).
clojure_direct_builtin("succ/2", "2").
clojure_direct_builtin("succ/2", 2).
clojure_direct_builtin('succ/2', "2").
clojure_direct_builtin('succ/2', 2).
clojure_direct_builtin("is/2", "2").
clojure_direct_builtin("is/2", 2).
clojure_direct_builtin('is/2', "2").
clojure_direct_builtin('is/2', 2).
clojure_direct_builtin("</2", "2").
clojure_direct_builtin("</2", 2).
clojure_direct_builtin('</2', "2").
clojure_direct_builtin('</2', 2).
clojure_direct_builtin(">/2", "2").
clojure_direct_builtin(">/2", 2).
clojure_direct_builtin('>/2', "2").
clojure_direct_builtin('>/2', 2).
clojure_direct_builtin("=</2", "2").
clojure_direct_builtin("=</2", 2).
clojure_direct_builtin('=</2', "2").
clojure_direct_builtin('=</2', 2).
clojure_direct_builtin(">=/2", "2").
clojure_direct_builtin(">=/2", 2).
clojure_direct_builtin('>=/2', "2").
clojure_direct_builtin('>=/2', 2).
clojure_direct_builtin("true/0", "0").
clojure_direct_builtin("true/0", 0).
clojure_direct_builtin('true/0', "0").
clojure_direct_builtin('true/0', 0).
clojure_direct_builtin("fail/0", "0").
clojure_direct_builtin("fail/0", 0).
clojure_direct_builtin('fail/0', "0").
clojure_direct_builtin('fail/0', 0).
clojure_direct_builtin("atom/1", "1").
clojure_direct_builtin("atom/1", 1).
clojure_direct_builtin('atom/1', "1").
clojure_direct_builtin('atom/1', 1).
clojure_direct_builtin("integer/1", "1").
clojure_direct_builtin("integer/1", 1).
clojure_direct_builtin('integer/1', "1").
clojure_direct_builtin('integer/1', 1).
clojure_direct_builtin("number/1", "1").
clojure_direct_builtin("number/1", 1).
clojure_direct_builtin('number/1', "1").
clojure_direct_builtin('number/1', 1).
clojure_direct_builtin("atomic/1", "1").
clojure_direct_builtin("atomic/1", 1).
clojure_direct_builtin('atomic/1', "1").
clojure_direct_builtin('atomic/1', 1).
clojure_direct_builtin("nonvar/1", "1").
clojure_direct_builtin("nonvar/1", 1).
clojure_direct_builtin('nonvar/1', "1").
clojure_direct_builtin('nonvar/1', 1).
clojure_direct_builtin("var/1", "1").
clojure_direct_builtin("var/1", 1).
clojure_direct_builtin('var/1', "1").
clojure_direct_builtin('var/1', 1).
clojure_direct_builtin("compound/1", "1").
clojure_direct_builtin("compound/1", 1).
clojure_direct_builtin('compound/1', "1").
clojure_direct_builtin('compound/1', 1).
clojure_direct_builtin("callable/1", "1").
clojure_direct_builtin("callable/1", 1).
clojure_direct_builtin('callable/1', "1").
clojure_direct_builtin('callable/1', 1).
clojure_direct_builtin("float/1", "1").
clojure_direct_builtin("float/1", 1).
clojure_direct_builtin('float/1', "1").
clojure_direct_builtin('float/1', 1).
clojure_direct_builtin("is_list/1", "1").
clojure_direct_builtin("is_list/1", 1).
clojure_direct_builtin('is_list/1', "1").
clojure_direct_builtin('is_list/1', 1).
clojure_direct_builtin("length/2", "2").
clojure_direct_builtin("length/2", 2).
clojure_direct_builtin('length/2', "2").
clojure_direct_builtin('length/2', 2).
clojure_direct_builtin("member/2", "2").
clojure_direct_builtin("member/2", 2).
clojure_direct_builtin('member/2', "2").
clojure_direct_builtin('member/2', 2).
clojure_direct_builtin("memberchk/2", "2").
clojure_direct_builtin("memberchk/2", 2).
clojure_direct_builtin('memberchk/2', "2").
clojure_direct_builtin('memberchk/2', 2).
clojure_direct_builtin("append/3", "3").
clojure_direct_builtin("append/3", 3).
clojure_direct_builtin('append/3', "3").
clojure_direct_builtin('append/3', 3).
clojure_direct_builtin("reverse/2", "2").
clojure_direct_builtin("reverse/2", 2).
clojure_direct_builtin('reverse/2', "2").
clojure_direct_builtin('reverse/2', 2).
clojure_direct_builtin("last/2", "2").
clojure_direct_builtin("last/2", 2).
clojure_direct_builtin('last/2', "2").
clojure_direct_builtin('last/2', 2).
clojure_direct_builtin("nth0/3", "3").
clojure_direct_builtin("nth0/3", 3).
clojure_direct_builtin('nth0/3', "3").
clojure_direct_builtin('nth0/3', 3).
clojure_direct_builtin("nth1/3", "3").
clojure_direct_builtin("nth1/3", 3).
clojure_direct_builtin('nth1/3', "3").
clojure_direct_builtin('nth1/3', 3).
clojure_direct_builtin("select/3", "3").
clojure_direct_builtin("select/3", 3).
clojure_direct_builtin('select/3', "3").
clojure_direct_builtin('select/3', 3).
clojure_direct_builtin("numlist/3", "3").
clojure_direct_builtin("numlist/3", 3).
clojure_direct_builtin('numlist/3', "3").
clojure_direct_builtin('numlist/3', 3).
clojure_direct_builtin("delete/3", "3").
clojure_direct_builtin("delete/3", 3).
clojure_direct_builtin('delete/3', "3").
clojure_direct_builtin('delete/3', 3).
clojure_direct_builtin("subtract/3", "3").
clojure_direct_builtin("subtract/3", 3).
clojure_direct_builtin('subtract/3', "3").
clojure_direct_builtin('subtract/3', 3).
clojure_direct_builtin("list_to_set/2", "2").
clojure_direct_builtin("list_to_set/2", 2).
clojure_direct_builtin('list_to_set/2', "2").
clojure_direct_builtin('list_to_set/2', 2).
clojure_direct_builtin("sort/2", "2").
clojure_direct_builtin("sort/2", 2).
clojure_direct_builtin('sort/2', "2").
clojure_direct_builtin('sort/2', 2).
clojure_direct_builtin("msort/2", "2").
clojure_direct_builtin("msort/2", 2).
clojure_direct_builtin('msort/2', "2").
clojure_direct_builtin('msort/2', 2).
clojure_direct_builtin("keysort/2", "2").
clojure_direct_builtin("keysort/2", 2).
clojure_direct_builtin('keysort/2', "2").
clojure_direct_builtin('keysort/2', 2).
clojure_direct_builtin("copy_term/2", "2").
clojure_direct_builtin("copy_term/2", 2).
clojure_direct_builtin('copy_term/2', "2").
clojure_direct_builtin('copy_term/2', 2).
clojure_direct_builtin("term_variables/2", "2").
clojure_direct_builtin("term_variables/2", 2).
clojure_direct_builtin('term_variables/2', "2").
clojure_direct_builtin('term_variables/2', 2).
clojure_direct_builtin("variant/2", "2").
clojure_direct_builtin("variant/2", 2).
clojure_direct_builtin('variant/2', "2").
clojure_direct_builtin('variant/2', 2).
clojure_direct_builtin("functor/3", "3").
clojure_direct_builtin("functor/3", 3).
clojure_direct_builtin('functor/3', "3").
clojure_direct_builtin('functor/3', 3).
clojure_direct_builtin("arg/3", "3").
clojure_direct_builtin("arg/3", 3).
clojure_direct_builtin('arg/3', "3").
clojure_direct_builtin('arg/3', 3).
clojure_direct_builtin("compound_name_arity/3", "3").
clojure_direct_builtin("compound_name_arity/3", 3).
clojure_direct_builtin('compound_name_arity/3', "3").
clojure_direct_builtin('compound_name_arity/3', 3).
clojure_direct_builtin("compound_name_arguments/3", "3").
clojure_direct_builtin("compound_name_arguments/3", 3).
clojure_direct_builtin('compound_name_arguments/3', "3").
clojure_direct_builtin('compound_name_arguments/3', 3).
clojure_direct_builtin("=../2", "2").
clojure_direct_builtin("=../2", 2).
clojure_direct_builtin('=../2', "2").
clojure_direct_builtin('=../2', 2).
clojure_direct_builtin("ground/1", "1").
clojure_direct_builtin("ground/1", 1).
clojure_direct_builtin('ground/1', "1").
clojure_direct_builtin('ground/1', 1).
clojure_direct_builtin("atom_codes/2", "2").
clojure_direct_builtin("atom_codes/2", 2).
clojure_direct_builtin('atom_codes/2', "2").
clojure_direct_builtin('atom_codes/2', 2).
clojure_direct_builtin("atom_chars/2", "2").
clojure_direct_builtin("atom_chars/2", 2).
clojure_direct_builtin('atom_chars/2', "2").
clojure_direct_builtin('atom_chars/2', 2).
clojure_direct_builtin("atom_string/2", "2").
clojure_direct_builtin("atom_string/2", 2).
clojure_direct_builtin('atom_string/2', "2").
clojure_direct_builtin('atom_string/2', 2).
clojure_direct_builtin("atom_number/2", "2").
clojure_direct_builtin("atom_number/2", 2).
clojure_direct_builtin('atom_number/2', "2").
clojure_direct_builtin('atom_number/2', 2).
clojure_direct_builtin("upcase_atom/2", "2").
clojure_direct_builtin("upcase_atom/2", 2).
clojure_direct_builtin('upcase_atom/2', "2").
clojure_direct_builtin('upcase_atom/2', 2).
clojure_direct_builtin("downcase_atom/2", "2").
clojure_direct_builtin("downcase_atom/2", 2).
clojure_direct_builtin('downcase_atom/2', "2").
clojure_direct_builtin('downcase_atom/2', 2).
clojure_direct_builtin("atomic_list_concat/2", "2").
clojure_direct_builtin("atomic_list_concat/2", 2).
clojure_direct_builtin('atomic_list_concat/2', "2").
clojure_direct_builtin('atomic_list_concat/2', 2).
clojure_direct_builtin("atomic_list_concat/3", "3").
clojure_direct_builtin("atomic_list_concat/3", 3).
clojure_direct_builtin('atomic_list_concat/3', "3").
clojure_direct_builtin('atomic_list_concat/3', 3).
clojure_direct_builtin("string_to_atom/2", "2").
clojure_direct_builtin("string_to_atom/2", 2).
clojure_direct_builtin('string_to_atom/2', "2").
clojure_direct_builtin('string_to_atom/2', 2).
clojure_direct_builtin("string_codes/2", "2").
clojure_direct_builtin("string_codes/2", 2).
clojure_direct_builtin('string_codes/2', "2").
clojure_direct_builtin('string_codes/2', 2).
clojure_direct_builtin("string_chars/2", "2").
clojure_direct_builtin("string_chars/2", 2).
clojure_direct_builtin('string_chars/2', "2").
clojure_direct_builtin('string_chars/2', 2).
clojure_direct_builtin("number_codes/2", "2").
clojure_direct_builtin("number_codes/2", 2).
clojure_direct_builtin('number_codes/2', "2").
clojure_direct_builtin('number_codes/2', 2).
clojure_direct_builtin("number_chars/2", "2").
clojure_direct_builtin("number_chars/2", 2).
clojure_direct_builtin('number_chars/2', "2").
clojure_direct_builtin('number_chars/2', 2).
clojure_direct_builtin("char_code/2", "2").
clojure_direct_builtin("char_code/2", 2).
clojure_direct_builtin('char_code/2', "2").
clojure_direct_builtin('char_code/2', 2).
clojure_direct_builtin("char_type/2", "2").
clojure_direct_builtin("char_type/2", 2).
clojure_direct_builtin('char_type/2', "2").
clojure_direct_builtin('char_type/2', 2).
clojure_direct_builtin("atom_concat/3", "3").
clojure_direct_builtin("atom_concat/3", 3).
clojure_direct_builtin('atom_concat/3', "3").
clojure_direct_builtin('atom_concat/3', 3).
clojure_direct_builtin("string_concat/3", "3").
clojure_direct_builtin("string_concat/3", 3).
clojure_direct_builtin('string_concat/3', "3").
clojure_direct_builtin('string_concat/3', 3).
clojure_direct_builtin("atom_length/2", "2").
clojure_direct_builtin("atom_length/2", 2).
clojure_direct_builtin('atom_length/2', "2").
clojure_direct_builtin('atom_length/2', 2).
clojure_direct_builtin("string_length/2", "2").
clojure_direct_builtin("string_length/2", 2).
clojure_direct_builtin('string_length/2', "2").
clojure_direct_builtin('string_length/2', 2).
clojure_direct_builtin("sub_atom/5", "5").
clojure_direct_builtin("sub_atom/5", 5).
clojure_direct_builtin('sub_atom/5', "5").
clojure_direct_builtin('sub_atom/5', 5).
clojure_direct_builtin("!/0", "0").
clojure_direct_builtin("!/0", 0).
clojure_direct_builtin('!/0', "0").
clojure_direct_builtin('!/0', 0).

clojure_terminal_builtin("fail/0", "0").
clojure_terminal_builtin("fail/0", 0).
clojure_terminal_builtin('fail/0', "0").
clojure_terminal_builtin('fail/0', 0).

clojure_pred_key_direct_builtin(Pred, Op, Arity) :-
    (   string(Pred) -> PredString = Pred
    ;   atom(Pred) -> atom_string(Pred, PredString)
    ),
    (   split_string(PredString, "/", "", [Name, ArityString])
    ->  (   var(Arity) -> number_string(Arity, ArityString)
        ;   integer(Arity) -> number_string(Arity, ArityString)
        ;   string(Arity) -> Arity = ArityString
        ;   atom(Arity) -> atom_string(Arity, ArityString)
        )
    ;   integer(Arity),
        Name = PredString
    ),
    format(string(Op), "~w/~w", [Name, Arity]),
    clojure_direct_builtin(Op, Arity).

clojure_unary_guard_test("atom/1", "(runtime/atom-term? value)").
clojure_unary_guard_test('atom/1', "(runtime/atom-term? value)").
clojure_unary_guard_test("integer/1", "(integer? value)").
clojure_unary_guard_test('integer/1', "(integer? value)").
clojure_unary_guard_test("number/1", "(number? value)").
clojure_unary_guard_test('number/1', "(number? value)").
clojure_unary_guard_test("atomic/1", "(or (runtime/atom-term? value) (number? value))").
clojure_unary_guard_test('atomic/1', "(or (runtime/atom-term? value) (number? value))").
clojure_unary_guard_test("nonvar/1", "(and (not= value ::lowered-unbound) (not (runtime/logic-var? value)))").
clojure_unary_guard_test('nonvar/1', "(and (not= value ::lowered-unbound) (not (runtime/logic-var? value)))").
clojure_unary_guard_test("var/1", "(or (= value ::lowered-unbound) (runtime/logic-var? value))").
clojure_unary_guard_test('var/1', "(or (= value ::lowered-unbound) (runtime/logic-var? value))").
clojure_unary_guard_test("compound/1", "(runtime/structure-term? value)").
clojure_unary_guard_test('compound/1', "(runtime/structure-term? value)").
clojure_unary_guard_test("callable/1", "(or (runtime/atom-term? value) (runtime/structure-term? value))").
clojure_unary_guard_test('callable/1', "(or (runtime/atom-term? value) (runtime/structure-term? value))").
clojure_unary_guard_test("float/1", "(float? value)").
clojure_unary_guard_test('float/1', "(float? value)").

clojure_arithmetic_order_builtin("</2", 'arithmetic-less?').
clojure_arithmetic_order_builtin('</2', 'arithmetic-less?').
clojure_arithmetic_order_builtin(">/2", 'arithmetic-greater?').
clojure_arithmetic_order_builtin('>/2', 'arithmetic-greater?').
clojure_arithmetic_order_builtin("=</2", 'arithmetic-less-or-equal?').
clojure_arithmetic_order_builtin('=</2', 'arithmetic-less-or-equal?').
clojure_arithmetic_order_builtin(">=/2", 'arithmetic-greater-or-equal?').
clojure_arithmetic_order_builtin('>=/2', 'arithmetic-greater-or-equal?').

clojure_term_order_builtin("@</2", 'term-less?').
clojure_term_order_builtin('@</2', 'term-less?').
clojure_term_order_builtin("@=</2", 'term-less-or-equal?').
clojure_term_order_builtin('@=</2', 'term-less-or-equal?').
clojure_term_order_builtin("@>/2", 'term-greater?').
clojure_term_order_builtin('@>/2', 'term-greater?').
clojure_term_order_builtin("@>=/2", 'term-greater-or-equal?').
clojure_term_order_builtin('@>=/2', 'term-greater-or-equal?').

emit_lowered_unary_guard(TestExpr, S, Expr) :-
    format(atom(Expr),
           '(let [value (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound))] (if ~w (runtime/advance ~w) (runtime/backtrack ~w)))',
           [S, S, TestExpr, S, S]).

emit_lowered_expr(proceed, S, Expr) :-
    format(atom(Expr), '(runtime/succeed-state ~w)', [S]).
emit_lowered_expr(fail, S, Expr) :-
    format(atom(Expr), '(runtime/fail-state ~w)', [S]).
emit_lowered_expr(call(Pred, CallArity), S, Expr) :-
    clojure_pred_key_direct_builtin(Pred, Op, CallArity),
    !,
    emit_lowered_expr(builtin_call(Op, CallArity), S, Expr).
emit_lowered_expr(call(Pred, _Arity), S, Expr) :-
    clj_lowered_string_literal(Pred, PredLit),
    format(atom(Expr),
           '(if-let [target-pc (get (:labels ~w) ~w)] (-> ~w (update :stack conj (inc (:pc ~w))) (assoc :pc target-pc)) (runtime/backtrack ~w))',
           [S, PredLit, S, S, S]).
emit_lowered_expr(execute(Pred), S, Expr) :-
    clojure_pred_key_direct_builtin(Pred, Op, Arity),
    !,
    emit_lowered_expr(builtin_call(Op, Arity), S, BuiltinExpr),
    format(atom(Expr),
           '(let [next-state ~w] (if (= :running (:status next-state)) (runtime/succeed-state next-state) next-state))',
           [BuiltinExpr]).
emit_lowered_expr(execute(Pred), S, Expr) :-
    clj_lowered_string_literal(Pred, PredLit),
    format(atom(Expr),
           '(if-let [target-pc (get (:labels ~w) ~w)] (assoc ~w :pc target-pc) (runtime/backtrack ~w))',
           [S, PredLit, S, S]).
emit_lowered_expr(jump(Label), S, Expr) :-
    clj_lowered_string_literal(Label, LabelLit),
    format(atom(Expr),
           '(if-let [target-pc (get (:labels ~w) ~w)] (assoc ~w :pc target-pc) (runtime/backtrack ~w))',
           [S, LabelLit, S, S]).
emit_lowered_expr(allocate, S, Expr) :-
    format(atom(Expr),
           '(-> ~w (update :env-stack conj {}) (assoc :cut-bar (count (:choice-points ~w))) runtime/advance)',
           [S, S]).
emit_lowered_expr(deallocate, S, Expr) :-
    format(atom(Expr),
           '(-> ~w (update :env-stack #(if (seq %) (pop %) %)) runtime/advance)',
           [S]).
emit_lowered_expr(get_constant(C, Ai), S, Expr) :-
    clj_lowered_literal(C, Lit),
    format(atom(Expr),
           '(let [constant (runtime/normalize-literal-term (:intern-context ~w) ~w) current (or (runtime/reg-get-raw ~w ~q) ::lowered-unbound) current* (if (= current ::lowered-unbound) current (runtime/deref-value (:bindings ~w) current))] (cond (= current* ::lowered-unbound) (-> ~w (runtime/reg-set-raw ~q constant) runtime/advance) (runtime/logic-var? current*) (-> (runtime/bind-var ~w current* constant) runtime/advance) (runtime/interned-equal? current* constant) (runtime/advance ~w) :else (runtime/backtrack ~w)))',
           [S, Lit, S, Ai, S, S, Ai, S, S, S]).
emit_lowered_expr(get_integer(N, Ai), S, Expr) :-
    format(atom(Expr),
           '(let [constant ~w current (or (runtime/reg-get-raw ~w ~q) ::lowered-unbound) current* (if (= current ::lowered-unbound) current (runtime/deref-value (:bindings ~w) current))] (cond (= current* ::lowered-unbound) (-> ~w (runtime/reg-set-raw ~q constant) runtime/advance) (runtime/logic-var? current*) (-> (runtime/bind-var ~w current* constant) runtime/advance) (runtime/interned-equal? current* constant) (runtime/advance ~w) :else (runtime/backtrack ~w)))',
           [N, S, Ai, S, S, Ai, S, S, S]).
emit_lowered_expr(get_nil(Ai), S, Expr) :-
    emit_lowered_expr(get_constant("[]", Ai), S, Expr).
emit_lowered_expr(get_structure(F, Ai), S, Expr) :-
    clj_lowered_literal(F, Lit),
    format(atom(Expr),
           '(let [functor (runtime/normalize-literal-term (:intern-context ~w) ~w) reg-val (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w ~q) ::lowered-unbound))] (cond (and (runtime/structure-term? reg-val) (runtime/interned-equal? (:functor reg-val) functor)) (-> ~w (runtime/enter-unify-mode (:args reg-val)) runtime/advance) :else (runtime/backtrack ~w)))',
           [S, Lit, S, S, Ai, S, S]).
emit_lowered_expr(get_list(Ai), S, Expr) :-
    format(atom(Expr),
           '(let [reg-val (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w ~q) ::lowered-unbound)) list-functor (runtime/list-functor-term (:intern-context ~w))] (cond (and (runtime/structure-term? reg-val) (runtime/interned-equal? (:functor reg-val) list-functor)) (-> ~w (runtime/enter-unify-mode (:args reg-val)) runtime/advance) :else (runtime/backtrack ~w)))',
           [S, S, Ai, S, S, S]).
emit_lowered_expr(unify_constant(C), S, Expr) :-
    clj_lowered_literal(C, Lit),
    format(atom(Expr),
           '(let [constant (runtime/normalize-literal-term (:intern-context ~w) ~w) [item next-state] (runtime/pop-unify-item ~w) [ok bound-state] (runtime/unify-values next-state item constant)] (if ok (runtime/advance bound-state) (runtime/backtrack ~w)))',
           [S, Lit, S, S]).
emit_lowered_expr(unify_variable(Xn), S, Expr) :-
    format(atom(Expr),
           '(let [[item next-state] (runtime/pop-unify-item ~w)] (-> next-state (runtime/reg-set-raw ~q item) runtime/advance))',
           [S, Xn]).
emit_lowered_expr(unify_value(Xn), S, Expr) :-
    format(atom(Expr),
           '(let [[item next-state] (runtime/pop-unify-item ~w) reg-val (or (runtime/reg-get-raw next-state ~q) ::lowered-unbound) [ok bound-state] (runtime/unify-values next-state reg-val item)] (if ok (runtime/advance bound-state) (runtime/backtrack ~w)))',
           [S, Xn, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "=/2" ; Op == '=/2'),
    !,
    format(atom(Expr),
           '(let [left (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound) right (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound) [ok next-state] (runtime/unify-values ~w left right)] (if ok (runtime/advance next-state) (runtime/backtrack ~w)))',
           [S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "\\=/2" ; Op == '\\=/2'),
    !,
    format(atom(Expr),
           '(let [left (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound) right (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound)] (if (runtime/unifiable? ~w left right) (runtime/backtrack ~w) (runtime/advance ~w)))',
           [S, S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "==/2" ; Op == '==/2'),
    !,
    format(atom(Expr),
           '(let [left (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound) right (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound)] (if (runtime/term-identical? ~w left right) (runtime/advance ~w) (runtime/backtrack ~w)))',
           [S, S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "\\==/2" ; Op == '\\==/2'),
    !,
    format(atom(Expr),
           '(let [left (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound) right (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound)] (if (runtime/term-identical? ~w left right) (runtime/backtrack ~w) (runtime/advance ~w)))',
           [S, S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    clojure_term_order_builtin(Op, RuntimePred),
    !,
    format(atom(Expr),
           '(let [left (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound) right (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound)] (if (runtime/~w ~w left right) (runtime/advance ~w) (runtime/backtrack ~w)))',
           [S, S, RuntimePred, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "compare/3" ; Op == 'compare/3'),
    !,
    format(atom(Expr), '(runtime/apply-compare-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "=:=/2" ; Op == '=:=/2'),
    !,
    format(atom(Expr),
           '(let [left (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound)) right (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound))] (if (runtime/arithmetic-equal? ~w left right) (runtime/advance ~w) (runtime/backtrack ~w)))',
           [S, S, S, S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "=\\=/2" ; Op == '=\\=/2'),
    !,
    format(atom(Expr),
           '(let [left (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound)) right (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound))] (if (runtime/arithmetic-not-equal? ~w left right) (runtime/advance ~w) (runtime/backtrack ~w)))',
           [S, S, S, S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "succ/2" ; Op == 'succ/2'),
    !,
    format(atom(Expr), '(runtime/apply-succ-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "is/2" ; Op == 'is/2'),
    !,
    format(atom(Expr),
           '(let [left (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound)) expr (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound)) result (runtime/eval-arithmetic-term ~w expr)] (if (some? result) (let [[ok next-state] (runtime/unify-values ~w left result)] (if ok (runtime/advance next-state) (runtime/backtrack ~w))) (runtime/backtrack ~w)))',
           [S, S, S, S, S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    clojure_arithmetic_order_builtin(Op, RuntimePred),
    !,
    format(atom(Expr),
           '(let [left (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound)) right (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound))] (if (runtime/~w ~w left right) (runtime/advance ~w) (runtime/backtrack ~w)))',
           [S, S, S, S, RuntimePred, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "true/0" ; Op == 'true/0'),
    !,
    format(atom(Expr), '(runtime/advance ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "fail/0" ; Op == 'fail/0'),
    !,
    format(atom(Expr), '(runtime/backtrack ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "is_list/1" ; Op == 'is_list/1'),
    !,
    format(atom(Expr),
           '(let [value (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound))] (if (runtime/proper-list-term? ~w value) (runtime/advance ~w) (runtime/backtrack ~w)))',
           [S, S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "length/2" ; Op == 'length/2'),
    !,
    format(atom(Expr), '(runtime/apply-length-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "member/2" ; Op == 'member/2'),
    !,
    format(atom(Expr),
           '(let [elem (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound)) list-value (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound))] (runtime/apply-member-solution ~w (inc (:pc ~w)) elem list-value))',
           [S, S, S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "memberchk/2" ; Op == 'memberchk/2'),
    !,
    format(atom(Expr),
           '(let [elem (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound)) list-value (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound))] (runtime/apply-memberchk-solution ~w elem list-value))',
           [S, S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "append/3" ; Op == 'append/3'),
    !,
    format(atom(Expr),
           '(let [left (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound)) right (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A2") ::lowered-unbound)) out (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A3") ::lowered-unbound))] (runtime/apply-append-solution ~w (inc (:pc ~w)) left right out))',
           [S, S, S, S, S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "reverse/2" ; Op == 'reverse/2'),
    !,
    format(atom(Expr), '(runtime/apply-reverse-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "last/2" ; Op == 'last/2'),
    !,
    format(atom(Expr), '(runtime/apply-last-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "nth0/3" ; Op == 'nth0/3'),
    !,
    format(atom(Expr), '(runtime/apply-nth0-solution ~w (inc (:pc ~w)))', [S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "nth1/3" ; Op == 'nth1/3'),
    !,
    format(atom(Expr), '(runtime/apply-nth1-solution ~w (inc (:pc ~w)))', [S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "select/3" ; Op == 'select/3'),
    !,
    format(atom(Expr), '(runtime/apply-select-solution ~w (inc (:pc ~w)))', [S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "numlist/3" ; Op == 'numlist/3'),
    !,
    format(atom(Expr), '(runtime/apply-numlist-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "delete/3" ; Op == 'delete/3'),
    !,
    format(atom(Expr), '(runtime/apply-delete-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "subtract/3" ; Op == 'subtract/3'),
    !,
    format(atom(Expr), '(runtime/apply-subtract-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "list_to_set/2" ; Op == 'list_to_set/2'),
    !,
    format(atom(Expr), '(runtime/apply-list-to-set-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "sort/2" ; Op == 'sort/2'),
    !,
    format(atom(Expr), '(runtime/apply-sort-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "msort/2" ; Op == 'msort/2'),
    !,
    format(atom(Expr), '(runtime/apply-msort-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "keysort/2" ; Op == 'keysort/2'),
    !,
    format(atom(Expr), '(runtime/apply-keysort-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "copy_term/2" ; Op == 'copy_term/2'),
    !,
    format(atom(Expr), '(runtime/apply-copy-term-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "term_variables/2" ; Op == 'term_variables/2'),
    !,
    format(atom(Expr), '(runtime/apply-term-variables-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "variant/2" ; Op == 'variant/2'),
    !,
    format(atom(Expr), '(runtime/apply-variant-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "functor/3" ; Op == 'functor/3'),
    !,
    format(atom(Expr), '(runtime/apply-functor-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "arg/3" ; Op == 'arg/3'),
    !,
    format(atom(Expr), '(runtime/apply-arg-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "compound_name_arity/3" ; Op == 'compound_name_arity/3'),
    !,
    format(atom(Expr), '(runtime/apply-compound-name-arity-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "compound_name_arguments/3" ; Op == 'compound_name_arguments/3'),
    !,
    format(atom(Expr), '(runtime/apply-compound-name-arguments-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "=../2" ; Op == '=../2'),
    !,
    format(atom(Expr), '(runtime/apply-univ-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "ground/1" ; Op == 'ground/1'),
    !,
    format(atom(Expr),
           '(let [value (runtime/deref-value (:bindings ~w) (or (runtime/reg-get-raw ~w "A1") ::lowered-unbound))] (if (runtime/ground-term? ~w value) (runtime/advance ~w) (runtime/backtrack ~w)))',
           [S, S, S, S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (   Op == "atom_codes/2" ; Op == 'atom_codes/2'
    ;   Op == "atom_chars/2" ; Op == 'atom_chars/2'
    ;   Op == "string_codes/2" ; Op == 'string_codes/2'
    ;   Op == "string_chars/2" ; Op == 'string_chars/2'
    ;   Op == "number_codes/2" ; Op == 'number_codes/2'
    ;   Op == "number_chars/2" ; Op == 'number_chars/2'
    ),
    !,
    format(atom(Expr), '(runtime/apply-text-conversion-solution ~w "~w")', [S, Op]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (   Op == "atom_string/2" ; Op == 'atom_string/2'
    ;   Op == "string_to_atom/2" ; Op == 'string_to_atom/2'
    ),
    !,
    format(atom(Expr), '(runtime/apply-atom-string-solution ~w "~w")', [S, Op]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "atom_number/2" ; Op == 'atom_number/2'),
    !,
    format(atom(Expr), '(runtime/apply-atom-number-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (   Op == "upcase_atom/2" ; Op == 'upcase_atom/2'
    ;   Op == "downcase_atom/2" ; Op == 'downcase_atom/2'
    ),
    !,
    format(atom(Expr), '(runtime/apply-atom-case-solution ~w "~w")', [S, Op]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (   Op == "atomic_list_concat/2" ; Op == 'atomic_list_concat/2'
    ;   Op == "atomic_list_concat/3" ; Op == 'atomic_list_concat/3'
    ),
    !,
    format(atom(Expr), '(runtime/apply-atomic-list-concat-solution ~w "~w")', [S, Op]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "char_code/2" ; Op == 'char_code/2'),
    !,
    format(atom(Expr), '(runtime/apply-char-code-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "char_type/2" ; Op == 'char_type/2'),
    !,
    format(atom(Expr), '(runtime/apply-char-type-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (   Op == "atom_concat/3" ; Op == 'atom_concat/3'
    ;   Op == "string_concat/3" ; Op == 'string_concat/3'
    ),
    !,
    format(atom(Expr), '(runtime/apply-atom-concat-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (   Op == "atom_length/2" ; Op == 'atom_length/2'
    ;   Op == "string_length/2" ; Op == 'string_length/2'
    ),
    !,
    format(atom(Expr), '(runtime/apply-atom-length-solution ~w)', [S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "sub_atom/5" ; Op == 'sub_atom/5'),
    !,
    format(atom(Expr), '(runtime/apply-sub-atom-solution ~w (inc (:pc ~w)))', [S, S]).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    clojure_unary_guard_test(Op, TestExpr),
    !,
    emit_lowered_unary_guard(TestExpr, S, Expr).
emit_lowered_expr(builtin_call(Op, Arity), S, Expr) :-
    clojure_direct_builtin(Op, Arity),
    (Op == "!/0" ; Op == '!/0'),
    !,
    format(atom(Expr),
           '(-> ~w (update :choice-points #(vec (take (:cut-bar ~w) %))) runtime/advance)',
           [S, S]).
emit_lowered_expr(put_constant(C, Ai), S, Expr) :-
    clj_lowered_literal(C, Lit),
    format(atom(Expr),
           '(let [constant (runtime/normalize-literal-term (:intern-context ~w) ~w)] (-> ~w (runtime/reg-set-raw ~q constant) runtime/advance))',
           [S, Lit, S, Ai]).
emit_lowered_expr(put_nil(Ai), S, Expr) :-
    emit_lowered_expr(put_constant("[]", Ai), S, Expr).
emit_lowered_expr(get_variable(Xn, Ai), S, Expr) :-
    format(atom(Expr),
           '(-> ~w (runtime/reg-set-raw ~q (runtime/reg-get-raw ~w ~q)) runtime/advance)',
           [S, Xn, S, Ai]).
emit_lowered_expr(put_variable(Xn, Ai), S, Expr) :-
    format(atom(Expr),
           '(let [[fresh next-state] (runtime/fresh-var ~w)] (-> next-state (runtime/reg-set-raw ~q fresh) (runtime/reg-set-raw ~q fresh) runtime/advance))',
           [S, Xn, Ai]).
emit_lowered_expr(get_value(Xn, Ai), S, Expr) :-
    format(atom(Expr),
           '(let [left (or (runtime/reg-get-raw ~w ~q) ::lowered-unbound) right (or (runtime/reg-get-raw ~w ~q) ::lowered-unbound) [ok next-state] (runtime/unify-values ~w left right)] (if ok (runtime/advance next-state) (runtime/backtrack ~w)))',
           [S, Xn, S, Ai, S, S]).
emit_lowered_expr(put_value(Xn, Ai), S, Expr) :-
    format(atom(Expr),
           '(let [val (runtime/deref-value (:bindings ~w) (runtime/reg-get-raw ~w ~q))] (-> ~w (runtime/reg-set-raw ~q val) runtime/advance))',
           [S, S, Xn, S, Ai]).
emit_lowered_expr(put_structure(F, Ai), S, Expr) :-
    clj_lowered_literal(F, Lit),
    format(atom(Expr),
           '(let [functor (runtime/normalize-literal-term (:intern-context ~w) ~w) arity (runtime/functor-arity (:intern-context ~w) functor)] (-> ~w (runtime/push-build-frame ~q functor arity) runtime/advance))',
           [S, Lit, S, S, Ai]).
emit_lowered_expr(put_list(Ai), S, Expr) :-
    format(atom(Expr),
           '(-> ~w (runtime/push-build-frame ~q (runtime/list-functor-term (:intern-context ~w)) 2) runtime/advance)',
           [S, Ai, S]).
emit_lowered_expr(set_constant(C), S, Expr) :-
    clj_lowered_literal(C, Lit),
    format(atom(Expr),
           '(let [constant (runtime/normalize-literal-term (:intern-context ~w) ~w)] (-> ~w (runtime/append-build-arg constant) runtime/finalize-complete-builds runtime/advance))',
           [S, Lit, S]).
emit_lowered_expr(set_variable(Xn), S, Expr) :-
    format(atom(Expr),
           '(let [[fresh next-state] (runtime/fresh-var ~w)] (-> next-state (runtime/reg-set-raw ~q fresh) (runtime/append-build-arg fresh) runtime/finalize-complete-builds runtime/advance))',
           [S, Xn]).
emit_lowered_expr(set_value(Xn), S, Expr) :-
    format(atom(Expr),
           '(let [val (runtime/deref-value (:bindings ~w) (runtime/reg-get-raw ~w ~q))] (-> ~w (runtime/append-build-arg val) runtime/finalize-complete-builds runtime/advance))',
           [S, S, Xn, S]).
emit_lowered_expr(_Instr, S, Expr) :-
    format(atom(Expr), '(runtime/step ~w)', [S]).

clj_lowered_literal(Value, Literal) :-
    (   number(Value)
    ->  format(atom(Literal), '~w', [Value])
    ;   clj_lowered_atom_text(Value, Text),
        clj_lowered_string_literal(Text, Literal)
    ).

clj_lowered_atom_text(Value, Text) :-
    (   string(Value) -> S0 = Value ; atom_string(Value, S0) ),
    clj_unquote_wam_atom_token(S0, S1),
    clj_strip_quoted_numeric_marker(S1, Text).

clj_unquote_wam_atom_token("''", "") :- !.
clj_unquote_wam_atom_token(S0, S) :-
    sub_string(S0, 0, 1, _, "'"),
    sub_string(S0, _, 1, 0, "'"),
    !,
    sub_string(S0, 1, _, 1, Inner),
    string_codes(Inner, Codes0),
    clj_unescape_wam_codes(Codes0, Codes),
    string_codes(S, Codes).
clj_unquote_wam_atom_token(S, S).

clj_unescape_wam_codes([], []).
clj_unescape_wam_codes([92, C|Rest], [C|More]) :- !,
    clj_unescape_wam_codes(Rest, More).
clj_unescape_wam_codes([C|Rest], [C|More]) :-
    clj_unescape_wam_codes(Rest, More).

clj_strip_quoted_numeric_marker(S0, S) :-
    string_codes(S0, [1|Rest]),
    !,
    string_codes(S, Rest).
clj_strip_quoted_numeric_marker(S, S).

clj_lowered_string_literal(Value, Literal) :-
    (   string(Value) -> S0 = Value ; atom_string(Value, S0) ),
    clj_escape_string(S0, Escaped),
    format(atom(Literal), '"~w"', [Escaped]).

clj_escape_string(S0, Escaped) :-
    string_codes(S0, Codes),
    maplist(clj_escape_code, Codes, Parts),
    atomics_to_string(Parts, "", Escaped).

clj_escape_code(92, "\\\\") :- !.
clj_escape_code(34, "\\\"") :- !.
clj_escape_code(10, "\\n") :- !.
clj_escape_code(13, "\\r") :- !.
clj_escape_code(9, "\\t") :- !.
clj_escape_code(Code, Escaped) :-
    (   Code < 32
    ->  format(string(Escaped), '\\u~|~`0t~16r~4+', [Code])
    ;   char_code(Char, Code),
        string_chars(Escaped, [Char])
    ).

instr_comment(proceed, "proceed").
instr_comment(fail, "fail").
instr_comment(allocate, "allocate").
instr_comment(deallocate, "deallocate").
instr_comment(get_constant(C, Ai), Comment) :-
    format(atom(Comment), 'get-constant ~w, ~w', [C, Ai]).
instr_comment(get_variable(Xn, Ai), Comment) :-
    format(atom(Comment), 'get-variable ~w, ~w', [Xn, Ai]).
instr_comment(get_value(Xn, Ai), Comment) :-
    format(atom(Comment), 'get-value ~w, ~w', [Xn, Ai]).
instr_comment(get_structure(F, Ai), Comment) :-
    format(atom(Comment), 'get-structure ~w, ~w', [F, Ai]).
instr_comment(get_list(Ai), Comment) :-
    format(atom(Comment), 'get-list ~w', [Ai]).
instr_comment(get_nil(Ai), Comment) :-
    format(atom(Comment), 'get-nil ~w', [Ai]).
instr_comment(get_integer(N, Ai), Comment) :-
    format(atom(Comment), 'get-integer ~w, ~w', [N, Ai]).
instr_comment(put_constant(C, Ai), Comment) :-
    format(atom(Comment), 'put-constant ~w, ~w', [C, Ai]).
instr_comment(put_variable(Xn, Ai), Comment) :-
    format(atom(Comment), 'put-variable ~w, ~w', [Xn, Ai]).
instr_comment(put_value(Xn, Ai), Comment) :-
    format(atom(Comment), 'put-value ~w, ~w', [Xn, Ai]).
instr_comment(put_structure(F, Ai), Comment) :-
    format(atom(Comment), 'put-structure ~w, ~w', [F, Ai]).
instr_comment(put_list(Ai), Comment) :-
    format(atom(Comment), 'put-list ~w', [Ai]).
instr_comment(unify_variable(Xn), Comment) :-
    format(atom(Comment), 'unify-variable ~w', [Xn]).
instr_comment(unify_value(Xn), Comment) :-
    format(atom(Comment), 'unify-value ~w', [Xn]).
instr_comment(unify_constant(C), Comment) :-
    format(atom(Comment), 'unify-constant ~w', [C]).
instr_comment(set_variable(Xn), Comment) :-
    format(atom(Comment), 'set-variable ~w', [Xn]).
instr_comment(set_value(Xn), Comment) :-
    format(atom(Comment), 'set-value ~w', [Xn]).
instr_comment(set_constant(C), Comment) :-
    format(atom(Comment), 'set-constant ~w', [C]).
instr_comment(call(P, N), Comment) :-
    format(atom(Comment), 'call ~w/~w', [P, N]).
instr_comment(execute(P), Comment) :-
    format(atom(Comment), 'execute ~w', [P]).
instr_comment(builtin_call(Op, Ar), Comment) :-
    format(atom(Comment), 'builtin-call ~w/~w', [Op, Ar]).
instr_comment(call_foreign(P, Ar), Comment) :-
    format(atom(Comment), 'call-foreign ~w/~w', [P, Ar]).
instr_comment(try_me_else(Label), Comment) :-
    format(atom(Comment), 'try_me_else ~w', [Label]).
instr_comment(trust_me, "trust_me").
instr_comment(cut_ite, "cut_ite").
instr_comment(jump(Label), Comment) :-
    format(atom(Comment), 'jump ~w', [Label]).
instr_comment(Instr, Comment) :-
    format(atom(Comment), 'TODO: lowered emission for ~w', [Instr]).
