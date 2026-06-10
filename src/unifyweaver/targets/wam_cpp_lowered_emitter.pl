:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_cpp_lowered_emitter.pl — WAM-lowered C++ emission
%
% Emits one C++ function per deterministic predicate (or per clause-1 of a
% multi-clause predicate). Simple register operations are inlined as direct
% C++ statements; complex instructions delegate to vm methods declared in
% wam_runtime.h.
%
% For multi-clause predicates (try_me_else), only clause 1 is lowered.
% Clause 2+ stays in the interpreter's instruction array for backtrack,
% mirroring the wam_rust_lowered_emitter / wam_haskell_lowered_emitter design.
%
% Modelled on wam_rust_lowered_emitter.pl (the closest systems-language
% sibling) and the hybrid pattern shared with wam_haskell, wam_lua, wam_r,
% wam_go, wam_clojure, wam_scala, wam_fsharp, wam_elixir.

:- module(wam_cpp_lowered_emitter, [
    wam_cpp_lowerable/3,
    lower_predicate_to_cpp/4,
    is_deterministic_pred_cpp/1,
    cpp_lowered_func_name/2,
    parse_wam_text/2
]).

:- use_module(library(lists)).
:- use_module(wam_ite_structurer, [structure_ite/2, split_commit/3, is_commit/1]).
:- use_module(wam_text_parser, [wam_text_to_items/2, wam_classify_constant_token/2]).
:- use_module(wam_clause_chain, [clause_chain/2]).
% Inlined escape helper to avoid a circular import with wam_cpp_target.
% Keeps this module standalone-loadable.

% =====================================================================
% Parsing — accept either an instruction list or a WAM-text blob.
% Delegates to the shared wam_text_parser module so this module gets
% the quote-aware tokenizer + ",/N" handling for free (previously
% used a split_string-based parser that mis-tokenised any predicate
% with a quoted atom argument or a conjunction-as-data, silently
% returning empty instruction lists and disabling lowering for those
% predicates).
% =====================================================================

parse_wam_text(WamText, Instrs) :-
    wam_text_to_items(WamText, Items),
    exclude(is_label_item, Items, Instrs).

is_label_item(label(_)).

% --- Label-preserving parse + if-then-else structuring ---------------
% wam_text_to_items already emits label(Name) items; parse_wam_text drops
% them. For (C -> T ; E) / \+ / once we need the labels, so keep them and
% fold each block into ite(Cond,Then,Else) via the shared structurer. The
% previous emitter no-op'd try_me_else/cut_ite/jump/trust_me, dropping the
% structure and emitting cond+then+else as one flat conjunction.

cpp_base_instrs(WamCode, Instrs) :-
    ( is_list(WamCode) -> Instrs = WamCode ; parse_wam_text(WamCode, Instrs) ).

%% cpp_structured_clause1(+WamCode, -Structured) is semidet.
%  Re-parse keeping labels, drop leading indexing/label items, take clause 1
%  and fold its ITE block(s). Succeeds only for a single-clause predicate
%  whose every block is consumed.
cpp_structured_clause1(WamCode, Structured) :-
    ( is_list(WamCode) -> LInstrs0 = WamCode ; wam_text_to_items(WamCode, LInstrs0) ),
    strip_leading_index_labels(LInstrs0, LInstrs),
    \+ ( LInstrs = [try_me_else(_)|_] ),   % not predicate-level multi-clause
    take_to_proceed(LInstrs, C1L),
    structure_ite(C1L, Structured),
    \+ member(try_me_else(_), Structured),
    \+ member(trust_me, Structured),
    \+ member(retry_me_else(_), Structured).

strip_leading_index_labels([switch_on_constant(_)|T], R) :- !, strip_leading_index_labels(T, R).
strip_leading_index_labels([switch_on_constant_a2(_)|T], R) :- !, strip_leading_index_labels(T, R).
strip_leading_index_labels([switch_on_structure(_)|T], R) :- !, strip_leading_index_labels(T, R).
strip_leading_index_labels([switch_on_term(_)|T], R) :- !, strip_leading_index_labels(T, R).
strip_leading_index_labels([label(_)|T], R) :- !, strip_leading_index_labels(T, R).
strip_leading_index_labels(L, L).

%% cpp_supported_structured(+StructuredInstr)
cpp_supported_structured(ite(C, T, E)) :- !,
    forall(member(I, C), cpp_supported_structured(I)),
    forall(member(I, T), cpp_supported_structured(I)),
    forall(member(I, E), cpp_supported_structured(I)).
cpp_supported_structured(I) :- cpp_supported(I).

% =====================================================================
% Lowerability
% =====================================================================

%% wam_cpp_lowerable(+Pred/Arity, +WamCode, -Reason)
%  True if the predicate can be lowered to a direct C++ function.
%  Reason is `clause_chain`, `deterministic`, `multi_clause_1` or
%  `ite_lowered`.
wam_cpp_lowerable(PI, WamCode, Reason) :-
    cpp_base_instrs(WamCode, Instrs),
    clause1_instrs(Instrs, C1),
    (   % T5: multi-clause predicate discriminating on a distinct
        % first-argument constant lowers to a bound-checked if-cascade over
        % ALL clauses (no interpreter hop for clauses 2+ when A1 is bound).
        % Takes precedence over multi_clause_1.
        cpp_clause_chain_lowerable(Instrs, _Guards)
    ->  Reason = clause_chain
    ;   % No internal ITE: existing deterministic / multi-clause path.
        \+ member(try_me_else(_), C1),
        forall(member(I, C1), cpp_supported(I))
    ->  ( is_deterministic_pred_cpp(Instrs)
        ->  Reason = deterministic
        ;   % T4: every clause is a clean supported deterministic body — lower
            % them ALL inline (tried in order with restore-between), so the
            % function never returns to the interpreter for clauses 2+. Takes
            % precedence over multi_clause_1 (clause-1 only).
            cpp_all_clauses_lowerable(Instrs)
        ->  Reason = multi_clause_n
        ;   Reason = multi_clause_1
        )
    ;   % Clause-1 has an inner choice point: lower only if it is a pure
        % (C -> T ; E) / \+ / once block whose pieces are all supported.
        \+ ( Instrs = [try_me_else(_)|_] ),   % single-clause predicate
        cpp_structured_clause1(WamCode, Structured),
        forall(member(I, Structured), cpp_supported_structured(I)),
        Reason = ite_lowered
    ),
    ( PI = _M:_P/_A -> true ; PI = _/_A2 -> true ; true ).

%% cpp_clause_chain_lowerable(+Instrs, -Guards) is semidet.
%  True when the predicate is a distinct-first-argument-constant clause chain
%  (T5) and every clause's remainder is a supported deterministic body. The
%  C++ parser keeps a leading switch_on_* indexing prefix, so strip it before
%  handing the try_me_else/retry_me_else/trust_me chain to the shared
%  front-end. Guards is the front-end's guard(Const, Remainder) list.
cpp_clause_chain_lowerable(Instrs, Guards) :-
    cpp_strip_switch_prefix(Instrs, Stripped),
    clause_chain(Stripped, chain(Guards)),
    forall(member(guard(_, Rem), Guards),
           ( is_deterministic_pred_cpp(Rem),
             forall(member(I, Rem), cpp_supported(I)) )).

cpp_strip_switch_prefix([switch_on_constant(_)|Rest], S) :- !, cpp_strip_switch_prefix(Rest, S).
cpp_strip_switch_prefix([switch_on_constant_a2(_)|Rest], S) :- !, cpp_strip_switch_prefix(Rest, S).
cpp_strip_switch_prefix([switch_on_structure(_)|Rest], S) :- !, cpp_strip_switch_prefix(Rest, S).
cpp_strip_switch_prefix([switch_on_term(_)|Rest], S) :- !, cpp_strip_switch_prefix(Rest, S).
cpp_strip_switch_prefix(L, L).

clause1_instrs([], []).
% Strip leading indexing instructions (switch_on_*) — they are dispatch
% helpers ahead of try_me_else, not part of any clause body.
clause1_instrs([switch_on_constant(_)|Rest], C1) :- !, clause1_instrs(Rest, C1).
clause1_instrs([switch_on_constant_a2(_)|Rest], C1) :- !, clause1_instrs(Rest, C1).
clause1_instrs([switch_on_structure(_)|Rest], C1) :- !, clause1_instrs(Rest, C1).
clause1_instrs([switch_on_term(_)|Rest], C1) :- !, clause1_instrs(Rest, C1).
clause1_instrs([try_me_else(_)|Rest], C1) :- !,
    take_to_proceed(Rest, C1).
clause1_instrs(Instrs, Instrs).

take_to_proceed([], []).
take_to_proceed([proceed|_], [proceed]) :- !.
take_to_proceed([I|Rest], [I|More]) :- take_to_proceed(Rest, More).

%% cpp_split_clauses(+Instrs, -Clauses) is semidet.
%  Split a multi-clause predicate (opens with try_me_else) at the choice-point
%  separators into per-clause instruction lists (T4 / multi_clause_n).
cpp_split_clauses([try_me_else(_)|Rest], [Clause|More]) :-
    cpp_collect_clause(Rest, Clause, After),
    cpp_split_more(After, More).

cpp_split_more([], []).
cpp_split_more([retry_me_else(_)|Rest], [Clause|More]) :- !,
    cpp_collect_clause(Rest, Clause, After),
    cpp_split_more(After, More).
cpp_split_more([trust_me|Rest], [Clause|More]) :- !,
    cpp_collect_clause(Rest, Clause, After),
    cpp_split_more(After, More).

cpp_collect_clause([], [], []).
cpp_collect_clause([retry_me_else(L)|Rest], [], [retry_me_else(L)|Rest]) :- !.
cpp_collect_clause([trust_me|Rest], [], [trust_me|Rest]) :- !.
cpp_collect_clause([I|Rest], [I|More], After) :-
    cpp_collect_clause(Rest, More, After).

%% cpp_all_clauses_lowerable(+Instrs) is semidet.
%  True when EVERY clause is a clean supported deterministic body (no inner
%  choice point, ends in a terminal) — the T4 (multi_clause_n) condition. The
%  C++ parser keeps a leading switch_on_* indexing prefix, so strip it first.
cpp_all_clauses_lowerable(Instrs0) :-
    cpp_strip_switch_prefix(Instrs0, Instrs),
    cpp_split_clauses(Instrs, Clauses),
    Clauses = [_, _ | _],
    forall(member(Cl, Clauses),
           ( \+ member(try_me_else(_), Cl),
             forall(member(I, Cl), cpp_supported(I)),
             last(Cl, Last), cpp_clause_terminal(Last) )).

cpp_clause_terminal(proceed).
cpp_clause_terminal(fail).
cpp_clause_terminal(execute(_)).

%% is_deterministic_pred_cpp(+Instrs)
%  True if the instruction list has no choice point instructions.
is_deterministic_pred_cpp(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

cpp_supported(allocate).
cpp_supported(deallocate).
cpp_supported(get_constant(_, _)).
cpp_supported(get_variable(_, _)).
cpp_supported(get_value(_, _)).
cpp_supported(get_structure(_, _)).
cpp_supported(get_list(_)).
cpp_supported(get_nil(_)).
cpp_supported(get_integer(_, _)).
cpp_supported(unify_variable(_)).
cpp_supported(unify_value(_)).
cpp_supported(unify_constant(_)).
cpp_supported(put_constant(_, _)).
cpp_supported(put_variable(_, _)).
cpp_supported(put_value(_, _)).
cpp_supported(put_structure(_, _)).
cpp_supported(put_list(_)).
cpp_supported(set_variable(_)).
cpp_supported(set_value(_)).
cpp_supported(set_constant(_)).
cpp_supported(call(_, _)).
cpp_supported(execute(_)).
cpp_supported(proceed).
cpp_supported(fail).
cpp_supported(builtin_call(_, _)).
cpp_supported(call_foreign(_, _)).
cpp_supported(try_me_else(_)).
cpp_supported(trust_me).
cpp_supported(cut_ite).
cpp_supported(cut(_)).         % Y-level soft cut (ite_use_y_level mode)
cpp_supported(get_level(_)).   % captures the cut level in the clause prefix
cpp_supported(jump(_)).
cpp_supported(begin_aggregate(_, _, _)).
cpp_supported(end_aggregate(_)).
% Indexing instructions: lowered emitter doesn''t inline these, but it
% must accept them as supported so a predicate containing them remains
% lowerable. The actual dispatch lives in the interpreter''s step().
cpp_supported(switch_on_constant(_)).
cpp_supported(switch_on_constant_a2(_)).
cpp_supported(switch_on_structure(_)).
cpp_supported(switch_on_term(_)).

% =====================================================================
% Function name generation
% =====================================================================

%% cpp_lowered_func_name(+Functor/Arity, -CppFuncName)
%  foo/2 -> "lowered_foo_2", my_pred/3 -> "lowered_my_pred_3"
cpp_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, FStr),
    sanitize_cpp_ident(FStr, SanStr),
    format(atom(Name), 'lowered_~w_~w', [SanStr, Arity]).

sanitize_cpp_ident(In, Out) :-
    string_codes(In, Codes),
    maplist(cpp_safe_code, Codes, OutCodes),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

cpp_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z
    ;   C >= 0'A, C =< 0'Z
    ;   C >= 0'0, C =< 0'9
    ;   C =:= 0'_
    ),
    !.
cpp_safe_code(_, 0'_).

% =====================================================================
% Emission
% =====================================================================

%% lower_predicate_to_cpp(+Pred/Arity, +WamCode, +Options, -CppLines)
%  Emit a C++ function for the predicate. CppLines is [Header, Body, Footer].
lower_predicate_to_cpp(PI, WamCode, Options, CppLines) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    cpp_lowered_func_name(Pred/Arity, FuncName),
    cpp_base_instrs(WamCode, Instrs),
    clause1_instrs(Instrs, C1Instrs),
    (   member(foreign_pred_keys(ForeignPreds0), Options)
    ->  maplist(foreign_key_string, ForeignPreds0, ForeignPreds)
    ;   ForeignPreds = []
    ),
    nb_setval(cpp_ite_ctr, 0),
    (   % T5 first-argument-constant dispatch takes precedence.
        cpp_clause_chain_lowerable(Instrs, Guards)
    ->  emit_clause_chain_cpp(FuncName, Pred, Arity, Guards, ForeignPreds, CppLines)
    ;   % T4: a multi-clause predicate whose clauses are all supported
        % deterministic bodies (but not a distinct-first-arg chain) — lower
        % every clause inline, tried in order with a restore between attempts.
        \+ is_deterministic_pred_cpp(Instrs),
        cpp_all_clauses_lowerable(Instrs)
    ->  emit_multi_clause_n_cpp(FuncName, Pred, Arity, Instrs, ForeignPreds, CppLines)
    ;   % Emit the plain clause-1 when it has no inner choice point; otherwise
        % fold its ITE block(s) into structured form (ite/3) first.
        (   \+ member(try_me_else(_), C1Instrs)
        ->  EmitInstrs = C1Instrs
        ;   cpp_structured_clause1(WamCode, EmitInstrs)
        ),
        with_output_to(string(Body), emit_instrs(EmitInstrs, "    ", ForeignPreds)),
        format(string(Header),
'// ~w — lowered from ~w/~w
bool ~w(WamState* vm) {', [FuncName, Pred, Arity, FuncName]),
        format(string(Footer), '}', []),
        CppLines = [Header, Body, Footer]
    ).

%% emit_clause_chain_cpp(+FuncName, +Pred, +Arity, +Guards, +FK, -CppLines)
%  Emit T5: read+deref the first argument once; if it is still unbound, defer
%  to the entry wrapper's interpreter fallback (the unbound case is genuinely
%  nondeterministic). Otherwise dispatch with an if-cascade comparing the
%  bound value against each clause's distinct discriminator and running that
%  clause's remainder (which self-terminates: its `proceed` emits
%  `return true`). A bound value matching no clause returns false (the
%  predicate fails; the wrapper's fresh re-run also fails — sound, since the
%  discriminators are distinct).
emit_clause_chain_cpp(FuncName, Pred, Arity, Guards, ForeignPreds, CppLines) :-
    format(string(Header),
'// ~w — lowered from ~w/~w (T5 first-argument dispatch)
bool ~w(WamState* vm) {', [FuncName, Pred, Arity, FuncName]),
    with_output_to(string(Body),
        ( % Per-guard dispatch compares the first argument in place (no
          % `Value t5a1 = get_reg(...)` copy, no temporary Value::Atom per
          % guard). An unbound / non-matching first arg matches no guard and
          % returns false, deferring to the interpreter fallback as before.
          emit_cpp_guards(Guards, ForeignPreds),
          format("    return false;~n") )),
    format(string(Footer), '}', []),
    CppLines = [Header, Body, Footer].

emit_cpp_guards([], _).
emit_cpp_guards([guard(V, Rem) | Rest], ForeignPreds) :-
    wam_classify_constant_token(V, Class),
    ( Class = atom(Name)
    ->  local_escape_cpp_string(Name, Esc),
        format("    if (vm->match_reg_atom(\"A1\", \"~w\") == 1) {~n", [Esc])
    ;   cpp_val_literal(V, CppVal),
        format("    if (vm->get_reg(\"A1\") == ~w) {~n", [CppVal])
    ),
    emit_instrs(Rem, "        ", ForeignPreds),
    format("    }~n"),
    emit_cpp_guards(Rest, ForeignPreds).

%% emit_multi_clause_n_cpp(+FuncName, +Pred, +Arity, +Instrs, +FK, -CppLines)
%  Emit T4: capture the clause-entry state, then try every clause inline as an
%  immediately-invoked lambda (its `proceed` returns true, failures return
%  false), restoring the entry state between attempts. The first clause that
%  succeeds wins (first-solution / deterministic-prefix); the function never
%  returns to the interpreter for clauses 2+, unlike multi_clause_1.
emit_multi_clause_n_cpp(FuncName, Pred, Arity, Instrs0, ForeignPreds, CppLines) :-
    cpp_strip_switch_prefix(Instrs0, Instrs),
    cpp_split_clauses(Instrs, Clauses),
    format(string(Header),
'// ~w — lowered from ~w/~w (T4 all-clauses inline)
bool ~w(WamState* vm) {', [FuncName, Pred, Arity, FuncName]),
    with_output_to(string(Body),
        ( format("    auto _t4 = vm->lo_clause_snapshot();~n"),
          emit_cpp_clauses(Clauses, ForeignPreds),
          format("    return false;~n") )),
    format(string(Footer), '}', []),
    CppLines = [Header, Body, Footer].

emit_cpp_clauses([], _).
emit_cpp_clauses([Cl | Rest], ForeignPreds) :-
    format("    if ([&]() -> bool {~n"),
    emit_instrs(Cl, "        ", ForeignPreds),
    format("    }()) return true;~n"),
    format("    vm->lo_restore_clause(_t4);~n"),
    emit_cpp_clauses(Rest, ForeignPreds).

%% emit_instrs(+Instrs, +Indent, +ForeignPreds)
emit_instrs([], _, _).
emit_instrs([Instr|Rest], Ind, ForeignPreds) :-
    emit_one(Instr, Ind, ForeignPreds),
    emit_instrs(Rest, Ind, ForeignPreds).

% Foreign-routed call/execute take precedence over the generic clauses.
emit_one(call(PredStr, NStr), I, ForeignPreds) :-
    member(PredStr, ForeignPreds), !,
    format("~w// call ~w via foreign kernel~n", [I, PredStr]),
    format("~wif (!vm->step(Instruction::CallForeign(\"~w\", ~w))) return false;~n",
           [I, PredStr, NStr]).
emit_one(execute(PredStr), I, ForeignPreds) :-
    member(PredStr, ForeignPreds),
    sub_atom(PredStr, _, 1, ArityLen, '/'),
    sub_atom(PredStr, _, ArityLen, 0, ArityAtom),
    atom_number(ArityAtom, Arity), !,
    format("~w// execute ~w via foreign kernel~n", [I, PredStr]),
    format("~wreturn vm->step(Instruction::CallForeign(\"~w\", ~w));~n",
           [I, PredStr, Arity]).
% If-then-else (structured; see wam_ite_structurer). The condition runs in
% an immediately-invoked lambda so its `return false` means "condition
% failed"; inside then/else, `return false` returns from the lowered
% function. cpp's get_reg reflects cell state, so unwinding the trail
% before the else branch restores any partial bindings the condition made
% (cell-based, like the existing \=/2 builtin).
emit_one(ite(Cond, Then, Else), I, ForeignPreds) :- !,
    nb_getval(cpp_ite_ctr, N0), N is N0 + 1, nb_setval(cpp_ite_ctr, N),
    string_concat(I, "    ", I2),
    format("~w{~n", [I]),
    format("~w    std::size_t _ite_mark~w = vm->trail.size();~n", [I, N]),
    format("~w    bool _ite_cond~w = [&]() -> bool {~n", [I, N]),
    emit_instrs(Cond, I2, ForeignPreds),
    format("~w        return true;~n", [I]),
    format("~w    }();~n", [I]),
    format("~w    if (_ite_cond~w) {~n", [I, N]),
    emit_instrs(Then, I2, ForeignPreds),
    format("~w    } else {~n", [I]),
    format("~w        vm->unwind_trail_to(_ite_mark~w);~n", [I, N]),
    emit_instrs(Else, I2, ForeignPreds),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).
emit_one(Instr, I, _) :-
    emit_one(Instr, I).

foreign_key_string(Key, String) :-
    (   string(Key)
    ->  String = Key
    ;   atom_string(Key, String)
    ).

% --- Terminal instructions ---

emit_one(proceed, I) :-
    format("~wreturn true;~n", [I]).

emit_one(fail, I) :-
    format("~wreturn false;~n", [I]).

% --- Head unification (get_*) ---

% get_constant — the hot head-match. An ATOM constant is compared against the
% register in place (vm->match_reg_atom), avoiding the per-comparison std::string
% allocations the old `get_reg() == Value::Atom("...")` form paid (the get_reg
% Value copy and the temporary Value::Atom). Integer/float keep the Value
% comparison (no allocation).
emit_one(get_constant(CStr, AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    wam_classify_constant_token(CStr, Class),
    ( Class = atom(Name)
    ->  local_escape_cpp_string(Name, Esc),
        format("~w// get_constant ~w, ~w~n", [I, CStr, AiStr]),
        format("~w{~n", [I]),
        format("~w    int _m = vm->match_reg_atom(\"~w\", \"~w\");~n", [I, Ai, Esc]),
        format("~w    if (_m < 0) {~n", [I]),
        format("~w        vm->trail_binding(\"~w\");~n", [I, Ai]),
        format("~w        vm->put_reg(\"~w\", Value::Atom(\"~w\"));~n", [I, Ai, Esc]),
        format("~w    } else if (_m == 0) {~n", [I]),
        format("~w        return false;~n", [I]),
        format("~w    }~n", [I]),
        format("~w}~n", [I])
    ;   cpp_val_literal(CStr, CppVal),
        format("~w// get_constant ~w, ~w~n", [I, CStr, AiStr]),
        format("~w{~n", [I]),
        format("~w    Value _a = vm->get_reg(\"~w\");~n", [I, Ai]),
        format("~w    if (_a.is_unbound()) {~n", [I]),
        format("~w        vm->trail_binding(\"~w\");~n", [I, Ai]),
        format("~w        vm->put_reg(\"~w\", ~w);~n", [I, Ai, CppVal]),
        format("~w    } else if (!(_a == ~w)) {~n", [I, CppVal]),
        format("~w        return false;~n", [I]),
        format("~w    }~n", [I]),
        format("~w}~n", [I])
    ).

emit_one(get_integer(NStr, AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// get_integer ~w, ~w~n", [I, NStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    Value _a = vm->get_reg(\"~w\");~n", [I, Ai]),
    format("~w    if (_a.is_unbound()) {~n", [I]),
    format("~w        vm->trail_binding(\"~w\");~n", [I, Ai]),
    format("~w        vm->put_reg(\"~w\", Value::Integer(~w));~n", [I, Ai, NStr]),
    format("~w    } else if (!(_a == Value::Integer(~w))) {~n", [I, NStr]),
    format("~w        return false;~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_nil(AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// get_nil ~w~n", [I, AiStr]),
    format("~w{~n", [I]),
    format("~w    int _m = vm->match_reg_atom(\"~w\", \"[]\");~n", [I, Ai]),
    format("~w    if (_m < 0) {~n", [I]),
    format("~w        vm->trail_binding(\"~w\");~n", [I, Ai]),
    format("~w        vm->put_reg(\"~w\", Value::Atom(\"[]\"));~n", [I, Ai]),
    format("~w    } else if (_m == 0) {~n", [I]),
    format("~w        return false;~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_variable(XnStr, AiStr), I) :-
    cpp_reg_name(XnStr, Xn), cpp_reg_name(AiStr, Ai),
    format("~w// get_variable ~w, ~w~n", [I, XnStr, AiStr]),
    % Repoint Xn to share Ai's cell (matches interpreter GetVariable);
    % put_reg would mutate Xn's cell and corrupt any register aliasing it.
    format("~wvm->set_cell(\"~w\", vm->get_cell(\"~w\"));~n", [I, Xn, Ai]).

emit_one(get_value(XnStr, AiStr), I) :-
    cpp_reg_name(XnStr, Xn), cpp_reg_name(AiStr, Ai),
    format("~w// get_value ~w, ~w~n", [I, XnStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    Value va = vm->get_reg(\"~w\");~n", [I, Ai]),
    format("~w    Value vx = vm->get_reg(\"~w\");~n", [I, Xn]),
    format("~w    if (!vm->unify(va, vx)) return false;~n", [I]),
    format("~w}~n", [I]).

emit_one(get_structure(FStr, AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// get_structure ~w, ~w (delegate to step)~n", [I, FStr, AiStr]),
    format("~wif (!vm->step(Instruction::GetStructure(\"~w\", \"~w\"))) return false;~n",
           [I, FStr, Ai]).

emit_one(get_list(AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// get_list ~w (delegate to step)~n", [I, AiStr]),
    format("~wif (!vm->step(Instruction::GetList(\"~w\"))) return false;~n", [I, Ai]).

% --- Body construction (put_*) ---

emit_one(put_constant(CStr, AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    cpp_val_literal(CStr, CppVal),
    format("~w// put_constant ~w, ~w~n", [I, CStr, AiStr]),
    % Repoint Ai to a fresh cell (matches interpreter PutConstant); mutating
    % would corrupt any register aliasing Ai (e.g. via a prior put_variable).
    format("~wvm->assign_reg(\"~w\", ~w);~n", [I, Ai, CppVal]).

emit_one(put_variable(XnStr, AiStr), I) :-
    cpp_reg_name(XnStr, Xn), cpp_reg_name(AiStr, Ai),
    format("~w// put_variable ~w, ~w~n", [I, XnStr, AiStr]),
    % Both registers must alias ONE fresh cell so the variable has a single
    % identity (binding via the cell is seen through either register). Two
    % put_reg copies would create two independent cells.
    format("~wvm->put_variable_reg(\"~w\", \"~w\");~n", [I, Xn, Ai]).

emit_one(put_value(XnStr, AiStr), I) :-
    cpp_reg_name(XnStr, Xn), cpp_reg_name(AiStr, Ai),
    format("~w// put_value ~w, ~w~n", [I, XnStr, AiStr]),
    % Repoint Ai to share Xn's cell (matches interpreter PutValue).
    format("~wvm->set_cell(\"~w\", vm->get_cell(\"~w\"));~n", [I, Ai, Xn]).

emit_one(put_structure(FStr, AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// put_structure ~w, ~w (delegate to step)~n", [I, FStr, AiStr]),
    format("~wif (!vm->step(Instruction::PutStructure(\"~w\", \"~w\"))) return false;~n",
           [I, FStr, Ai]).

emit_one(put_list(AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// put_list ~w (delegate to step)~n", [I, AiStr]),
    format("~wif (!vm->step(Instruction::PutList(\"~w\"))) return false;~n", [I, Ai]).

% --- Unify instructions (delegate to step) ---

emit_one(unify_variable(XnStr), I) :-
    cpp_reg_name(XnStr, Xn),
    format("~w// unify_variable ~w (delegate to step)~n", [I, XnStr]),
    format("~wif (!vm->step(Instruction::UnifyVariable(\"~w\"))) return false;~n", [I, Xn]).

emit_one(unify_value(XnStr), I) :-
    cpp_reg_name(XnStr, Xn),
    format("~w// unify_value ~w (delegate to step)~n", [I, XnStr]),
    format("~wif (!vm->step(Instruction::UnifyValue(\"~w\"))) return false;~n", [I, Xn]).

emit_one(unify_constant(CStr), I) :-
    cpp_val_literal(CStr, CppVal),
    format("~w// unify_constant ~w (delegate to step)~n", [I, CStr]),
    format("~wif (!vm->step(Instruction::UnifyConstant(~w))) return false;~n", [I, CppVal]).

% --- Set instructions (delegate to step) ---

emit_one(set_variable(XnStr), I) :-
    cpp_reg_name(XnStr, Xn),
    format("~w// set_variable ~w (delegate to step)~n", [I, XnStr]),
    format("~wif (!vm->step(Instruction::SetVariable(\"~w\"))) return false;~n", [I, Xn]).

emit_one(set_value(XnStr), I) :-
    cpp_reg_name(XnStr, Xn),
    format("~w// set_value ~w (delegate to step)~n", [I, XnStr]),
    format("~wif (!vm->step(Instruction::SetValue(\"~w\"))) return false;~n", [I, Xn]).

emit_one(set_constant(CStr), I) :-
    cpp_val_literal(CStr, CppVal),
    format("~w// set_constant ~w (delegate to step)~n", [I, CStr]),
    format("~wif (!vm->step(Instruction::SetConstant(~w))) return false;~n", [I, CppVal]).

% --- Environment instructions ---

emit_one(allocate, I) :-
    format("~w// allocate~n", [I]),
    format("~wvm->step(Instruction::Allocate());~n", [I]).

emit_one(deallocate, I) :-
    format("~w// deallocate~n", [I]),
    format("~wvm->step(Instruction::Deallocate());~n", [I]).

% --- Control instructions ---

emit_one(call(PredStr, _NStr), I) :-
    format("~w// call ~w~n", [I, PredStr]),
    format("~w{~n", [I]),
    format("~w    std::size_t saved_cp = vm->cp;~n", [I]),
    format("~w    auto it = vm->labels.find(\"~w\");~n", [I, PredStr]),
    format("~w    if (it == vm->labels.end()) return false;~n", [I]),
    format("~w    vm->pc = it->second;~n", [I]),
    format("~w    if (!vm->run()) return false;~n", [I]),
    format("~w    vm->cp = saved_cp;~n", [I]),
    format("~w}~n", [I]).

emit_one(execute(PredStr), I) :-
    format("~w// execute ~w (tail call)~n", [I, PredStr]),
    format("~w{~n", [I]),
    format("~w    auto it = vm->labels.find(\"~w\");~n", [I, PredStr]),
    format("~w    if (it == vm->labels.end()) return false;~n", [I]),
    format("~w    vm->pc = it->second;~n", [I]),
    format("~w    return vm->run();~n", [I]),
    format("~w}~n", [I]).

emit_one(builtin_call(OpStr, NStr), I) :-
    format("~w// builtin_call ~w ~w~n", [I, OpStr, NStr]),
    local_escape_cpp_string(OpStr, EscOp),
    format("~wif (!vm->step(Instruction::BuiltinCall(\"~w\", ~w))) return false;~n",
           [I, EscOp, NStr]).

emit_one(call_foreign(PredStr, ArStr), I) :-
    format("~w// call_foreign ~w ~w~n", [I, PredStr, ArStr]),
    format("~wif (!vm->step(Instruction::CallForeign(\"~w\", ~w))) return false;~n",
           [I, PredStr, ArStr]).

% --- Choicepoint / ITE related (consumed during lowering) ---

emit_one(try_me_else(_), _) :- !.
emit_one(trust_me, _) :- !.
emit_one(cut_ite, _) :- !.
emit_one(cut(_), _) :- !.        % commit consumed by the structurer
emit_one(get_level(_), _) :- !.  % cut-level capture: no-op in the lowered if/else
emit_one(jump(_), _) :- !.

% Aggregate ops delegate to vm->step() (the interpreter handles the
% findall / aggregate_all driver loop).
emit_one(begin_aggregate(KStr, VStr, RStr), I) :-
    local_escape_cpp_string(KStr, EK),
    local_escape_cpp_string(VStr, EV),
    local_escape_cpp_string(RStr, ER),
    format("~w// begin_aggregate ~w ~w ~w~n", [I, KStr, VStr, RStr]),
    format("~wif (!vm->step(Instruction::BeginAggregate(\"~w\", \"~w\", \"~w\"))) return false;~n",
           [I, EK, EV, ER]).
emit_one(end_aggregate(VStr), I) :-
    local_escape_cpp_string(VStr, EV),
    format("~w// end_aggregate ~w~n", [I, VStr]),
    format("~wif (!vm->step(Instruction::EndAggregate(\"~w\"))) return false;~n",
           [I, EV]).

% --- Fallback ---

emit_one(Instr, I) :-
    format("~w// TODO: lowered emission for ~w~n", [I, Instr]).

% =====================================================================
% Helpers
% =====================================================================

%% cpp_reg_name(+RegStr, -Name)
%  Pass through register name (A1, X2, Y3 etc.) — used for get_reg/put_reg.
cpp_reg_name(RegStr, Name) :-
    atom_string(RegA, RegStr),
    atom_string(RegA, Name).

%% local_escape_cpp_string(+In, -Out)
%  Inlined copy of wam_cpp_target:escape_cpp_string/2 to keep this module
%  loadable without a back-import to wam_cpp_target.
local_escape_cpp_string(In, Out) :-
    atom_string(In, S),
    split_string(S, "\\", "", Parts),
    local_join(Parts, "\\\\", Escaped1),
    split_string(Escaped1, "\"", "", Parts2),
    local_join(Parts2, "\\\"", Out).

local_join([], _, "").
local_join([X], _, X).
local_join([X, Y|Rest], Sep, Result) :-
    local_join([Y|Rest], Sep, Tail),
    string_concat(X, Sep, XSep),
    string_concat(XSep, Tail, Result).

%% cpp_val_literal(+Str, -CppLiteral)
%  Convert a WAM constant token to a C++ Value literal. Atom-vs-number
%  disambiguation goes through wam_classify_constant_token/2 — a
%  bare token `5` is the integer 5; a quoted token `'5'` (with outer
%  apostrophes preserved by the WAM-text tokenizer) is the atom `'5'`.
cpp_val_literal(Str, CppVal) :-
    wam_classify_constant_token(Str, Class),
    (   Class = integer(N)
    ->  format(atom(CppVal), 'Value::Integer(~w)', [N])
    ;   Class = float(F)
    ->  format(atom(CppVal), 'Value::Float(~w)', [F])
    ;   Class = atom(Name),
        format(atom(CppVal), 'Value::Atom("~w")', [Name])
    ).
