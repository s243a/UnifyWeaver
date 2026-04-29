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

% =====================================================================
% Parsing
% =====================================================================

parse_wam_text(WamText, Instrs) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_lines(Lines, Instrs).

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
    clause1_instrs(Instrs, C1Instrs),
    with_output_to(string(Body), emit_instrs(C1Instrs, "  ")),
    format(string(ClojureCode),
';; ~w — lowered from ~w/~w
(defn ~w [state]
~w)
', [FuncName, Pred, Arity, FuncName, Body]).

emit_instrs([], _).
emit_instrs([Instr|Rest], Indent) :-
    emit_one(Instr, Indent),
    emit_instrs(Rest, Indent).

emit_one(proceed, I) :-
    format("~w(runtime/succeed-state state)~n", [I]).

emit_one(fail, I) :-
    format("~w(runtime/fail-state state)~n", [I]).

emit_one(allocate, I) :-
    format("~w;; allocate — lowered tier keeps runtime-managed env frames~n", [I]),
    format("~w(runtime/step state)~n", [I]).

emit_one(deallocate, I) :-
    format("~w;; deallocate — lowered tier keeps runtime-managed env frames~n", [I]),
    format("~w(runtime/step state)~n", [I]).

emit_one(get_constant(C, Ai), I) :-
    format("~w;; get-constant ~w, ~w~n", [I, C, Ai]),
    format("~w(let [instr {:op :get-constant :constant ~q :reg ~q}]~n", [I, C, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(get_variable(Xn, Ai), I) :-
    format("~w;; get-variable ~w, ~w~n", [I, Xn, Ai]),
    format("~w(let [instr {:op :get-variable :var ~q :reg ~q}]~n", [I, Xn, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(get_value(Xn, Ai), I) :-
    format("~w;; get-value ~w, ~w~n", [I, Xn, Ai]),
    format("~w(let [instr {:op :get-value :var ~q :reg ~q}]~n", [I, Xn, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(get_structure(F, Ai), I) :-
    format("~w;; get-structure ~w, ~w~n", [I, F, Ai]),
    format("~w(let [instr {:op :get-structure :functor ~q :reg ~q}]~n", [I, F, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(get_list(Ai), I) :-
    format("~w;; get-list ~w~n", [I, Ai]),
    format("~w(let [instr {:op :get-list :reg ~q}]~n", [I, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(get_nil(Ai), I) :-
    format("~w;; get-nil ~w~n", [I, Ai]),
    format("~w(let [instr {:op :get-constant :constant \"[]\" :reg ~q}]~n", [I, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(get_integer(N, Ai), I) :-
    format("~w;; get-integer ~w, ~w~n", [I, N, Ai]),
    format("~w(let [instr {:op :get-constant :constant ~w :reg ~q}]~n", [I, N, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(put_constant(C, Ai), I) :-
    format("~w;; put-constant ~w, ~w~n", [I, C, Ai]),
    format("~w(let [instr {:op :put-constant :constant ~q :reg ~q}]~n", [I, C, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(put_variable(Xn, Ai), I) :-
    format("~w;; put-variable ~w, ~w~n", [I, Xn, Ai]),
    format("~w(let [instr {:op :put-variable :var ~q :reg ~q}]~n", [I, Xn, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(put_value(Xn, Ai), I) :-
    format("~w;; put-value ~w, ~w~n", [I, Xn, Ai]),
    format("~w(let [instr {:op :put-value :var ~q :reg ~q}]~n", [I, Xn, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(put_structure(F, Ai), I) :-
    format("~w;; put-structure ~w, ~w~n", [I, F, Ai]),
    format("~w(let [instr {:op :put-structure :functor ~q :reg ~q}]~n", [I, F, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(put_list(Ai), I) :-
    format("~w;; put-list ~w~n", [I, Ai]),
    format("~w(let [instr {:op :put-list :reg ~q}]~n", [I, Ai]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(unify_variable(Xn), I) :-
    format("~w;; unify-variable ~w~n", [I, Xn]),
    format("~w(let [instr {:op :unify-variable :var ~q}]~n", [I, Xn]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(unify_value(Xn), I) :-
    format("~w;; unify-value ~w~n", [I, Xn]),
    format("~w(let [instr {:op :unify-value :var ~q}]~n", [I, Xn]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(unify_constant(C), I) :-
    format("~w;; unify-constant ~w~n", [I, C]),
    format("~w(let [instr {:op :unify-constant :constant ~q}]~n", [I, C]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(set_variable(Xn), I) :-
    format("~w;; set-variable ~w~n", [I, Xn]),
    format("~w(let [instr {:op :set-variable :var ~q}]~n", [I, Xn]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(set_value(Xn), I) :-
    format("~w;; set-value ~w~n", [I, Xn]),
    format("~w(let [instr {:op :set-value :var ~q}]~n", [I, Xn]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(set_constant(C), I) :-
    format("~w;; set-constant ~w~n", [I, C]),
    format("~w(let [instr {:op :set-constant :constant ~q}]~n", [I, C]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(call(P, N), I) :-
    format("~w;; call ~w/~w — initial scaffold delegates through runtime invoke path~n", [I, P, N]),
    format("~w(let [instr {:op :call :pred ~q :arity ~w}]~n", [I, P, N]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(execute(P), I) :-
    format("~w;; execute ~w — tail-call form stays runtime-mediated in the first slice~n", [I, P]),
    format("~w(let [instr {:op :execute :pred ~q}]~n", [I, P]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(builtin_call(Op, Ar), I) :-
    format("~w;; builtin-call ~w/~w~n", [I, Op, Ar]),
    format("~w(let [instr {:op :builtin-call :pred ~q :arity ~w}]~n", [I, Op, Ar]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(call_foreign(P, Ar), I) :-
    format("~w;; call-foreign ~w/~w~n", [I, P, Ar]),
    format("~w(let [instr {:op :call-foreign :pred ~q :arity ~w}]~n", [I, P, Ar]),
    format("~w  (runtime/step (assoc state :instr instr)))~n", [I]).

emit_one(try_me_else(_), _) :- !.
emit_one(trust_me, _) :- !.
emit_one(cut_ite, _) :- !.
emit_one(jump(_), _) :- !.

emit_one(Instr, I) :-
    format("~w;; TODO: lowered emission for ~w~n", [I, Instr]).
