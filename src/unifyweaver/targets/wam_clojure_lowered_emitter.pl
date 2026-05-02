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
    clause1_instrs(Instrs, C1Instrs0),
    (   is_deterministic_pred_clojure(Instrs)
    ->  ControlLowering = allow_control
    ;   ControlLowering = runtime_control
    ),
    lowered_direct_prefix(C1Instrs0, ControlLowering, C1Instrs),
    with_output_to(string(Body), emit_instrs(C1Instrs, "  ")),
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
clojure_direct_builtin("true/0", "0").
clojure_direct_builtin("true/0", 0).
clojure_direct_builtin('true/0', "0").
clojure_direct_builtin('true/0', 0).
clojure_direct_builtin("!/0", "0").
clojure_direct_builtin("!/0", 0).
clojure_direct_builtin('!/0', "0").
clojure_direct_builtin('!/0', 0).

emit_lowered_expr(proceed, S, Expr) :-
    format(atom(Expr), '(runtime/succeed-state ~w)', [S]).
emit_lowered_expr(fail, S, Expr) :-
    format(atom(Expr), '(runtime/fail-state ~w)', [S]).
emit_lowered_expr(call(Pred, _Arity), S, Expr) :-
    clj_lowered_string_literal(Pred, PredLit),
    format(atom(Expr),
           '(if-let [target-pc (get (:labels ~w) ~w)] (-> ~w (update :stack conj (inc (:pc ~w))) (assoc :pc target-pc)) (runtime/backtrack ~w))',
           [S, PredLit, S, S, S]).
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
    (Op == "true/0" ; Op == 'true/0'),
    !,
    format(atom(Expr), '(runtime/advance ~w)', [S]).
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
    ;   format(atom(Literal), '~q', [Value])
    ).

clj_lowered_string_literal(Value, Literal) :-
    (   string(Value) -> S0 = Value ; atom_string(Value, S0) ),
    split_string(S0, "\\", "", BackslashParts),
    atomics_to_string(BackslashParts, "\\\\", S1),
    split_string(S1, "\"", "", QuoteParts),
    atomics_to_string(QuoteParts, "\\\"", Escaped),
    format(atom(Literal), '"~w"', [Escaped]).

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
