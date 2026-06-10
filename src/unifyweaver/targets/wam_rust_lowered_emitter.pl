:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_rust_lowered_emitter.pl — WAM-lowered Rust emission
%
% Emits one Rust function per deterministic predicate.
% Simple register operations are inlined as direct Rust code; complex
% instructions delegate to vm methods from state.rs.
%
% For multi-clause predicates (try_me_else), only clause 1 is lowered.
% Clause 2+ stays in the interpreter's instruction array for backtrack.
%
% Modelled on wam_go_lowered_emitter.pl (475 lines) and
% wam_fsharp_lowered_emitter.pl (646 lines).

:- module(wam_rust_lowered_emitter, [
    wam_rust_lowerable/3,
    lower_predicate_to_rust/4,
    is_deterministic_pred_rust/1,
    rust_lowered_func_name/2
]).

:- use_module(library(lists)).
:- use_module(wam_ite_structurer, [structure_ite/2, split_commit/3, is_commit/1]).
:- use_module(wam_clause_chain, [clause_chain/2]).
:- use_module(wam_rust_target, [escape_rust_string/2]).
:- use_module(wam_text_parser, [wam_classify_constant_token/2]).

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
instr_from_parts(["try_me_else", L], try_me_else(L)).
instr_from_parts(["retry_me_else", L], retry_me_else(L)).
instr_from_parts(["trust_me"], trust_me).
instr_from_parts(["jump", L], jump(L)).
instr_from_parts(["cut_ite"], cut_ite).

% =====================================================================
% Label-preserving parse + if-then-else structuring
% =====================================================================
%
%  The base parser drops label lines, so the boundaries of an
%  (C -> T ; E) / \+ / once block are lost. The previous emitter simply
%  no-op'd try_me_else/cut_ite/jump/trust_me, which dropped the structure
%  entirely and emitted the condition, then- and else-branches as one flat
%  conjunction (e.g. unifying the output with BOTH branch values), so the
%  lowered function always failed.
%
%  parse_wam_text_labeled keeps label(Name) markers (and cut_ite) so
%  structure_ite can fold each block into an ite(Cond,Then,Else) term.
%  Rust's get_reg derefs through the binding table, so trail save/undo
%  alone restores a failed condition's bindings — no register snapshot is
%  needed (unlike the Go backend). Mirrors the shared wam_ite_structurer.

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
            Instrs = [label(LabelStr)|More],   % keep as string to match try_me_else/jump args
            parse_lines_labeled(Rest, More)
        ;   instr_from_parts(CleanParts, Instr)
        ->  Instrs = [Instr|More],
            parse_lines_labeled(Rest, More)
        ;   parse_lines_labeled(Rest, Instrs)
        )
    ).

% structure_ite/2, split_commit/3 and is_commit/1 are shared across the
% lowered backends — see wam_ite_structurer.pl.

%% rust_base_instrs(+WamCode, -Instrs)  — base (label-stripped) parse or list.
rust_base_instrs(WamCode, Instrs) :-
    ( is_list(WamCode) -> Instrs = WamCode ; parse_wam_text(WamCode, Instrs) ).

%% rust_structured_clause1(+WamCode, -Structured) is semidet.
rust_structured_clause1(WamCode, Structured) :-
    ( is_list(WamCode) -> LInstrs = WamCode ; parse_wam_text_labeled(WamCode, LInstrs) ),
    \+ ( LInstrs = [try_me_else(_)|_] ),   % not predicate-level multi-clause
    take_to_proceed(LInstrs, C1L),
    structure_ite(C1L, Structured),
    \+ member(try_me_else(_), Structured),
    \+ member(trust_me, Structured),
    \+ member(retry_me_else(_), Structured).

%% rust_supported_structured(+StructuredInstr)
rust_supported_structured(ite(C, T, E)) :- !,
    forall(member(I, C), rust_supported_structured(I)),
    forall(member(I, T), rust_supported_structured(I)),
    forall(member(I, E), rust_supported_structured(I)).
rust_supported_structured(I) :- rust_supported(I).

% =====================================================================
% Lowerability
% =====================================================================

%% wam_rust_lowerable(+Pred/Arity, +WamCode, -Reason)
%  True if the predicate can be lowered to a direct Rust function.
wam_rust_lowerable(PI, WamCode, Reason) :-
    rust_base_instrs(WamCode, Instrs),
    clause1_instrs(Instrs, C1),
    (   % T5: multi-clause predicate discriminating on a distinct
        % first-argument constant lowers to a bound-checked if-cascade over
        % ALL clauses (no interpreter hop for clauses 2+ when A1 is bound).
        % Takes precedence over multi_clause_1.
        rust_clause_chain_lowerable(Instrs, _Guards)
    ->  Reason = clause_chain
    ;   % No internal ITE: existing deterministic / multi-clause path.
        \+ member(try_me_else(_), C1),
        forall(member(I, C1), rust_supported(I))
    ->  ( is_deterministic_pred_rust(Instrs)
        ->  Reason = deterministic
        ;   % T4: every clause is a clean supported deterministic body — lower
            % them ALL inline (tried in order with restore-between), so the
            % function never returns to the interpreter for clauses 2+. Takes
            % precedence over multi_clause_1 (clause-1 only).
            rust_all_clauses_lowerable(Instrs)
        ->  Reason = multi_clause_n
        ;   Reason = multi_clause_1
        )
    ;   % Clause-1 has an inner choice point: lower only if it is a pure
        % (C -> T ; E) / \+ / once block whose pieces are all supported.
        \+ ( Instrs = [try_me_else(_)|_] ),   % single-clause predicate
        rust_structured_clause1(WamCode, Structured),
        forall(member(I, Structured), rust_supported_structured(I)),
        Reason = ite_lowered
    ),
    ( PI = _M:_P/_A -> true ; PI = _/_A2 -> true ; true ).

clause1_instrs([], []).
clause1_instrs([try_me_else(_)|Rest], C1) :- !,
    take_to_proceed(Rest, C1).
clause1_instrs(Instrs, Instrs).

take_to_proceed([], []).
take_to_proceed([proceed|_], [proceed]) :- !.
take_to_proceed([I|Rest], [I|More]) :- take_to_proceed(Rest, More).

%% rust_split_clauses(+Instrs, -Clauses) is semidet.
%  Split a multi-clause predicate (opens with try_me_else) at the choice-point
%  separators into per-clause instruction lists (T4 / multi_clause_n).
rust_split_clauses([try_me_else(_)|Rest], [Clause|More]) :-
    rust_collect_clause(Rest, Clause, After),
    rust_split_more(After, More).

rust_split_more([], []).
rust_split_more([retry_me_else(_)|Rest], [Clause|More]) :- !,
    rust_collect_clause(Rest, Clause, After),
    rust_split_more(After, More).
rust_split_more([trust_me|Rest], [Clause|More]) :- !,
    rust_collect_clause(Rest, Clause, After),
    rust_split_more(After, More).

rust_collect_clause([], [], []).
rust_collect_clause([retry_me_else(L)|Rest], [], [retry_me_else(L)|Rest]) :- !.
rust_collect_clause([trust_me|Rest], [], [trust_me|Rest]) :- !.
rust_collect_clause([I|Rest], [I|More], After) :-
    rust_collect_clause(Rest, More, After).

%% rust_all_clauses_lowerable(+Instrs) is semidet.
%  True when EVERY clause is a clean supported deterministic body (no inner
%  choice point, ends in a terminal) — the T4 (multi_clause_n) condition.
rust_all_clauses_lowerable(Instrs) :-
    rust_split_clauses(Instrs, Clauses),
    Clauses = [_, _ | _],
    forall(member(Cl, Clauses),
           ( \+ member(try_me_else(_), Cl),
             forall(member(I, Cl), rust_supported(I)),
             last(Cl, Last), rust_clause_terminal(Last) )).

rust_clause_terminal(proceed).
rust_clause_terminal(fail).
rust_clause_terminal(execute(_)).

%% rust_clause_chain_lowerable(+Instrs, -Guards) is semidet.
%  True when the predicate is a distinct-first-argument-constant clause
%  chain (T5) and every clause's remainder is a supported deterministic
%  body. Guards is the shared front-end's guard(Const, Remainder) list.
rust_clause_chain_lowerable(Instrs, Guards) :-
    clause_chain(Instrs, chain(Guards)),
    forall(member(guard(_, Rem), Guards),
           ( is_deterministic_pred_rust(Rem),
             forall(member(I, Rem), rust_supported(I)) )).

%% is_deterministic_pred_rust(+Instrs)
%  True if the instruction list has no choice point instructions.
is_deterministic_pred_rust(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

rust_supported(allocate).
rust_supported(deallocate).
rust_supported(get_constant(_, _)).
rust_supported(get_variable(_, _)).
rust_supported(get_value(_, _)).
rust_supported(get_structure(_, _)).
rust_supported(get_list(_)).
rust_supported(get_nil(_)).
rust_supported(get_integer(_, _)).
rust_supported(unify_variable(_)).
rust_supported(unify_value(_)).
rust_supported(unify_constant(_)).
rust_supported(put_constant(_, _)).
rust_supported(put_variable(_, _)).
rust_supported(put_value(_, _)).
rust_supported(put_structure(_, _)).
rust_supported(put_list(_)).
rust_supported(set_variable(_)).
rust_supported(set_value(_)).
rust_supported(set_constant(_)).
rust_supported(call(_, _)).
rust_supported(execute(_)).
rust_supported(proceed).
rust_supported(fail).
rust_supported(builtin_call(_, _)).
rust_supported(call_foreign(_, _)).
rust_supported(try_me_else(_)).
rust_supported(trust_me).
rust_supported(cut_ite).
rust_supported(jump(_)).

% =====================================================================
% Function name generation
% =====================================================================

%% rust_lowered_func_name(+Functor/Arity, -RustFuncName)
%  Generates a valid Rust function name.
%  foo/2 -> "lowered_foo_2", my_pred/3 -> "lowered_my_pred_3"
rust_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, FStr),
    sanitize_rust_ident(FStr, SanStr),
    format(atom(Name), 'lowered_~w_~w', [SanStr, Arity]).

sanitize_rust_ident(In, Out) :-
    string_codes(In, Codes),
    maplist(rust_safe_code, Codes, OutCodes),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

rust_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z -> true
    ;   C >= 0'A, C =< 0'Z -> true
    ;   C >= 0'0, C =< 0'9 -> true
    ;   C =:= 0'_ -> true
    ),
    !.
rust_safe_code(_, 0'_).

% =====================================================================
% Emission
% =====================================================================

%% lower_predicate_to_rust(+Pred/Arity, +WamCode, +Options, -RustLines)
%  Emit a Rust function for the predicate.
lower_predicate_to_rust(PI, WamCode, Options, RustLines) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    rust_lowered_func_name(Pred/Arity, FuncName),
    rust_base_instrs(WamCode, Instrs),
    clause1_instrs(Instrs, C1Instrs),
    (   member(foreign_pred_keys(ForeignPreds0), Options)
    ->  maplist(foreign_key_string, ForeignPreds0, ForeignPreds)
    ;   ForeignPreds = []
    ),
    nb_setval(rust_ite_ctr, 0),
    (   % T5 first-argument-constant dispatch takes precedence.
        rust_clause_chain_lowerable(Instrs, Guards)
    ->  emit_clause_chain_rust(FuncName, Pred, Arity, Guards, ForeignPreds, RustLines)
    ;   % T4: a multi-clause predicate whose clauses are all supported
        % deterministic bodies (but not a distinct-first-arg chain) — lower
        % every clause inline, tried in order with a restore between attempts.
        \+ is_deterministic_pred_rust(Instrs),
        rust_all_clauses_lowerable(Instrs)
    ->  emit_multi_clause_n_rust(FuncName, Pred, Arity, Instrs, ForeignPreds, RustLines)
    ;   % Emit the plain clause-1 when it has no inner choice point;
        % otherwise fold its ITE block(s) into structured form (ite/3).
        (   \+ member(try_me_else(_), C1Instrs)
        ->  EmitInstrs = C1Instrs
        ;   rust_structured_clause1(WamCode, EmitInstrs)
        ),
        with_output_to(string(Body), emit_instrs(EmitInstrs, "    ", ForeignPreds)),
        format(string(Header),
'// ~w — lowered from ~w/~w
pub fn ~w(vm: &mut WamState) -> bool {', [FuncName, Pred, Arity, FuncName]),
        format(string(Footer), '}', []),
        RustLines = [Header, Body, Footer]
    ).

%% emit_clause_chain_rust(+FuncName, +Pred, +Arity, +Guards, +FK, -RustLines)
%  Emit T5: deref the first argument once; if it is still unbound, defer to
%  the entry wrapper's interpreter fallback (the unbound case is genuinely
%  nondeterministic). Otherwise dispatch with an if-cascade comparing the
%  bound value against each clause's distinct discriminator and running that
%  clause's remainder (which self-terminates: its `proceed` emits
%  `return true`). A bound value matching no clause returns false (the
%  predicate fails; the wrapper's fresh re-run also fails — sound).
emit_clause_chain_rust(FuncName, Pred, Arity, Guards, ForeignPreds, RustLines) :-
    format(string(Header),
'// ~w — lowered from ~w/~w (T5 first-argument dispatch)
pub fn ~w(vm: &mut WamState) -> bool {', [FuncName, Pred, Arity, FuncName]),
    with_output_to(string(Body),
        ( % The per-guard dispatch compares the first argument in place (no
          % `let t5a1 = get_reg(...)` clone, no `Value::Atom("...")` per guard).
          % An unbound / non-matching first arg matches no guard and returns
          % false, deferring to the entry wrapper's interpreter fallback exactly
          % as the old explicit is_unbound() check did.
          emit_rust_guards(Guards, ForeignPreds),
          format("    false~n") )),
    format(string(Footer), '}', []),
    RustLines = [Header, Body, Footer].

emit_rust_guards([], _).
emit_rust_guards([guard(V, Rem) | Rest], ForeignPreds) :-
    wam_classify_constant_token(V, Class),
    ( Class = atom(Name)
    ->  escape_rust_string(Name, Esc),
        format("    if vm.match_reg_atom(\"A1\", \"~w\") == Some(true) {~n", [Esc])
    ;   rust_val_literal(V, RustVal),
        format("    if vm.get_reg(\"A1\").map_or(false, |__v| __v == ~w) {~n", [RustVal])
    ),
    emit_instrs(Rem, "        ", ForeignPreds),
    format("    }~n"),
    emit_rust_guards(Rest, ForeignPreds).

%% emit_multi_clause_n_rust(+FuncName, +Pred, +Arity, +Instrs, +FK, -RustLines)
%  Emit T4: capture the clause-entry state, then try every clause inline as an
%  immediately-invoked closure (its `proceed` returns true, its failures return
%  false), restoring the entry state between attempts. The first clause that
%  succeeds wins (first-solution / deterministic-prefix semantics); on total
%  failure it returns false. The interpreter is never entered for the
%  predicate — clauses 2+ run natively, unlike multi_clause_1.
emit_multi_clause_n_rust(FuncName, Pred, Arity, Instrs, ForeignPreds, RustLines) :-
    rust_split_clauses(Instrs, Clauses),
    format(string(Header),
'// ~w — lowered from ~w/~w (T4 all-clauses inline)
pub fn ~w(vm: &mut WamState) -> bool {', [FuncName, Pred, Arity, FuncName]),
    with_output_to(string(Body),
        ( format("    let _t4 = vm.lo_clause_snapshot();~n"),
          emit_rust_clauses(Clauses, ForeignPreds),
          format("    false~n") )),
    format(string(Footer), '}', []),
    RustLines = [Header, Body, Footer].

emit_rust_clauses([], _).
emit_rust_clauses([Cl | Rest], ForeignPreds) :-
    format("    if (|vm: &mut WamState| -> bool {~n"),
    emit_instrs(Cl, "        ", ForeignPreds),
    format("    })(vm) { return true; }~n"),
    format("    vm.lo_restore_clause(&_t4);~n"),
    emit_rust_clauses(Rest, ForeignPreds).

%% emit_instrs(+Instrs, +Indent, +ForeignPreds)
%  Emit Rust code for a list of instructions.
emit_instrs([], _, _).
emit_instrs([Instr|Rest], Ind, ForeignPreds) :-
    emit_one(Instr, Ind, ForeignPreds),
    emit_instrs(Rest, Ind, ForeignPreds).

emit_one(call(PredStr, NStr), I, ForeignPreds) :-
    member(PredStr, ForeignPreds), !,
    format("~w// call ~w via foreign kernel~n", [I, PredStr]),
    format("~wif !vm.step(&Instruction::CallForeign(\"~w\".to_string(), ~w)) { return false; }~n",
           [I, PredStr, NStr]).
emit_one(execute(PredStr), I, ForeignPreds) :-
    member(PredStr, ForeignPreds),
    sub_atom(PredStr, _, 1, ArityLen, '/'),
    sub_atom(PredStr, _, ArityLen, 0, ArityAtom),
    atom_number(ArityAtom, Arity), !,
    format("~w// execute ~w via foreign kernel~n", [I, PredStr]),
    format("~wreturn vm.step(&Instruction::CallForeign(\"~w\".to_string(), ~w));~n",
           [I, PredStr, Arity]).
% --- If-then-else (structured; see wam_ite_structurer) ---
% The condition runs in an immediately-invoked closure so its `return
% false` means "condition failed"; inside then/else, `return false`
% returns from the lowered function. Rust's get_reg derefs through the
% binding table, so unwinding the trail before the else branch restores
% any partial bindings the condition made (no register snapshot needed).
emit_one(ite(Cond, Then, Else), I, ForeignPreds) :- !,
    nb_getval(rust_ite_ctr, N0), N is N0 + 1, nb_setval(rust_ite_ctr, N),
    string_concat(I, "    ", I2),
    format("~wlet _ite_mark~w = vm.trail.len();~n", [I, N]),
    format("~wlet _ite_cond~w = (|vm: &mut WamState| -> bool {~n", [I, N]),
    emit_instrs(Cond, I2, ForeignPreds),
    format("~w    true~n", [I]),
    format("~w})(vm);~n", [I]),
    format("~wif _ite_cond~w {~n", [I, N]),
    emit_instrs(Then, I2, ForeignPreds),
    format("~w} else {~n", [I]),
    format("~w    vm.unwind_trail_to(_ite_mark~w);~n", [I, N]),
    emit_instrs(Else, I2, ForeignPreds),
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

% get_constant — the hot head-match. For an ATOM constant we compare the
% register against the &str in place (vm.match_reg_atom), avoiding the two heap
% allocations the old `get_reg() != Value::Atom("...".to_string())` form paid
% per comparison. Integer/other constants keep the (allocation-free, Copy)
% Value comparison.
emit_one(get_constant(CStr, AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    wam_classify_constant_token(CStr, Class),
    ( Class = atom(Name)
    ->  escape_rust_string(Name, Esc),
        format("~w// get_constant ~w, ~w~n", [I, CStr, AiStr]),
        format("~w{~n", [I]),
        format("~w    match vm.match_reg_atom(\"~w\", \"~w\") {~n", [I, Ai, Esc]),
        format("~w        Some(true) => {}~n", [I]),
        format("~w        Some(false) => return false,~n", [I]),
        format("~w        None => {~n", [I]),
        format("~w            vm.trail_binding(\"~w\");~n", [I, Ai]),
        format("~w            vm.put_reg(\"~w\", Value::Atom(\"~w\".to_string()));~n", [I, Ai, Esc]),
        format("~w        }~n", [I]),
        format("~w    }~n", [I]),
        format("~w}~n", [I])
    ;   rust_val_literal(CStr, RustVal),
        format("~w// get_constant ~w, ~w~n", [I, CStr, AiStr]),
        format("~w{~n", [I]),
        format("~w    let _a = vm.get_reg(\"~w\").unwrap_or(Value::Uninit);~n", [I, Ai]),
        format("~w    if _a.is_unbound() {~n", [I]),
        format("~w        vm.trail_binding(\"~w\");~n", [I, Ai]),
        format("~w        vm.put_reg(\"~w\", ~w);~n", [I, Ai, RustVal]),
        format("~w    } else if _a != ~w {~n", [I, RustVal]),
        format("~w        return false;~n", [I]),
        format("~w    }~n", [I]),
        format("~w}~n", [I])
    ).

emit_one(get_integer(NStr, AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// get_integer ~w, ~w~n", [I, NStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    let _a = vm.get_reg(\"~w\").unwrap_or(Value::Uninit);~n", [I, Ai]),
    format("~w    if _a.is_unbound() {~n", [I]),
    format("~w        vm.trail_binding(\"~w\");~n", [I, Ai]),
    format("~w        vm.put_reg(\"~w\", Value::Integer(~w));~n", [I, Ai, NStr]),
    format("~w    } else if _a != Value::Integer(~w) {~n", [I, NStr]),
    format("~w        return false;~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_nil(AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// get_nil ~w~n", [I, AiStr]),
    format("~w{~n", [I]),
    format("~w    match vm.match_reg_atom(\"~w\", \"[]\") {~n", [I, Ai]),
    format("~w        Some(true) => {}~n", [I]),
    format("~w        Some(false) => return false,~n", [I]),
    format("~w        None => {~n", [I]),
    format("~w            vm.trail_binding(\"~w\");~n", [I, Ai]),
    format("~w            vm.put_reg(\"~w\", Value::Atom(\"[]\".to_string()));~n", [I, Ai]),
    format("~w        }~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_variable(XnStr, AiStr), I) :-
    rust_reg_name(XnStr, Xn), rust_reg_name(AiStr, Ai),
    format("~w// get_variable ~w, ~w~n", [I, XnStr, AiStr]),
    format("~wif let Some(v) = vm.get_reg(\"~w\") { vm.put_reg(\"~w\", v); }~n", [I, Ai, Xn]).

emit_one(get_value(XnStr, AiStr), I) :-
    rust_reg_name(XnStr, Xn), rust_reg_name(AiStr, Ai),
    format("~w// get_value ~w, ~w~n", [I, XnStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    let va = vm.get_reg(\"~w\").unwrap_or(Value::Uninit);~n", [I, Ai]),
    format("~w    let vx = vm.get_reg(\"~w\").unwrap_or(Value::Uninit);~n", [I, Xn]),
    format("~w    if !vm.unify(&va, &vx) { return false; }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_structure(FStr, AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// get_structure ~w, ~w (delegate to step)~n", [I, FStr, AiStr]),
    format("~wif !vm.step(&Instruction::GetStructure(\"~w\".to_string(), \"~w\".to_string())) { return false; }~n", [I, FStr, Ai]).

emit_one(get_list(AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// get_list ~w (delegate to step)~n", [I, AiStr]),
    format("~wif !vm.step(&Instruction::GetList(\"~w\".to_string())) { return false; }~n", [I, Ai]).

% --- Body construction (put_*) ---

emit_one(put_constant(CStr, AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    rust_val_literal(CStr, RustVal),
    format("~w// put_constant ~w, ~w~n", [I, CStr, AiStr]),
    format("~wvm.put_reg(\"~w\", ~w);~n", [I, Ai, RustVal]).

emit_one(put_variable(XnStr, AiStr), I) :-
    rust_reg_name(XnStr, Xn), rust_reg_name(AiStr, Ai),
    format("~w// put_variable ~w, ~w~n", [I, XnStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    let v = Value::Unbound(format!(\"_V{}\", vm.var_counter));~n", [I]),
    format("~w    vm.var_counter += 1;~n", [I]),
    format("~w    vm.put_reg(\"~w\", v.clone());~n", [I, Xn]),
    format("~w    vm.put_reg(\"~w\", v);~n", [I, Ai]),
    format("~w}~n", [I]).

emit_one(put_value(XnStr, AiStr), I) :-
    rust_reg_name(XnStr, Xn), rust_reg_name(AiStr, Ai),
    format("~w// put_value ~w, ~w~n", [I, XnStr, AiStr]),
    format("~wif let Some(v) = vm.get_reg(\"~w\") { vm.put_reg(\"~w\", v); }~n", [I, Xn, Ai]).

emit_one(put_structure(FStr, AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// put_structure ~w, ~w (delegate to step)~n", [I, FStr, AiStr]),
    format("~wif !vm.step(&Instruction::PutStructure(\"~w\".to_string(), \"~w\".to_string())) { return false; }~n", [I, FStr, Ai]).

emit_one(put_list(AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// put_list ~w (delegate to step)~n", [I, AiStr]),
    format("~wif !vm.step(&Instruction::PutList(\"~w\".to_string())) { return false; }~n", [I, Ai]).

% --- Unify instructions (delegate to step) ---

emit_one(unify_variable(XnStr), I) :-
    rust_reg_name(XnStr, Xn),
    format("~w// unify_variable ~w (delegate to step)~n", [I, XnStr]),
    format("~wif !vm.step(&Instruction::UnifyVariable(\"~w\".to_string())) { return false; }~n", [I, Xn]).

emit_one(unify_value(XnStr), I) :-
    rust_reg_name(XnStr, Xn),
    format("~w// unify_value ~w (delegate to step)~n", [I, XnStr]),
    format("~wif !vm.step(&Instruction::UnifyValue(\"~w\".to_string())) { return false; }~n", [I, Xn]).

emit_one(unify_constant(CStr), I) :-
    rust_val_literal(CStr, RustVal),
    format("~w// unify_constant ~w (delegate to step)~n", [I, CStr]),
    format("~wif !vm.step(&Instruction::UnifyConstant(~w)) { return false; }~n", [I, RustVal]).

% --- Set instructions (delegate to step) ---

emit_one(set_variable(XnStr), I) :-
    rust_reg_name(XnStr, Xn),
    format("~w// set_variable ~w (delegate to step)~n", [I, XnStr]),
    format("~wif !vm.step(&Instruction::SetVariable(\"~w\".to_string())) { return false; }~n", [I, Xn]).

emit_one(set_value(XnStr), I) :-
    rust_reg_name(XnStr, Xn),
    format("~w// set_value ~w (delegate to step)~n", [I, XnStr]),
    format("~wif !vm.step(&Instruction::SetValue(\"~w\".to_string())) { return false; }~n", [I, Xn]).

emit_one(set_constant(CStr), I) :-
    rust_val_literal(CStr, RustVal),
    format("~w// set_constant ~w (delegate to step)~n", [I, CStr]),
    format("~wif !vm.step(&Instruction::SetConstant(~w)) { return false; }~n", [I, RustVal]).

% --- Environment instructions ---

emit_one(allocate, I) :-
    format("~w// allocate~n", [I]),
    format("~wvm.step(&Instruction::Allocate);~n", [I]).

emit_one(deallocate, I) :-
    format("~w// deallocate~n", [I]),
    format("~wvm.step(&Instruction::Deallocate);~n", [I]).

% --- Control instructions ---

emit_one(call(PredStr, _NStr), I) :-
    format("~w// call ~w~n", [I, PredStr]),
    format("~w{~n", [I]),
    format("~w    let saved_cp = vm.cp;~n", [I]),
    format("~w    if let Some(&pc) = vm.labels.get(\"~w\") {~n", [I, PredStr]),
    format("~w        vm.pc = pc;~n", [I]),
    format("~w        if !vm.run() { return false; }~n", [I]),
    format("~w    } else { return false; }~n", [I]),
    format("~w    vm.cp = saved_cp;~n", [I]),
    format("~w}~n", [I]).

emit_one(execute(PredStr), I) :-
    format("~w// execute ~w (tail call)~n", [I, PredStr]),
    format("~wif let Some(&pc) = vm.labels.get(\"~w\") {~n", [I, PredStr]),
    format("~w    vm.pc = pc;~n", [I]),
    format("~w    return vm.run();~n", [I]),
    format("~w}~n", [I]),
    format("~wreturn false;~n", [I]).

emit_one(builtin_call(OpStr, NStr), I) :-
    format("~w// builtin_call ~w ~w~n", [I, OpStr, NStr]),
    escape_rust_string(OpStr, EscOp),
    format("~wif !vm.step(&Instruction::BuiltinCall(\"~w\".to_string(), ~w)) { return false; }~n", [I, EscOp, NStr]).

emit_one(call_foreign(PredStr, ArStr), I) :-
    format("~w// call_foreign ~w ~w~n", [I, PredStr, ArStr]),
    format("~wif !vm.step(&Instruction::CallForeign(\"~w\".to_string(), ~w)) { return false; }~n", [I, PredStr, ArStr]).

% --- Choicepoint / ITE related (consumed during lowering) ---

emit_one(try_me_else(_), _) :- !.
emit_one(trust_me, _) :- !.
emit_one(cut_ite, _) :- !.
emit_one(jump(_), _) :- !.

% --- Fallback ---

emit_one(Instr, I) :-
    format("~w// TODO: lowered emission for ~w~n", [I, Instr]).

% =====================================================================
% Helpers
% =====================================================================

%% rust_reg_name(+RegStr, -Name)
%  Pass through register name (A1, X2, Y3 etc.) — used for get_reg/put_reg calls.
rust_reg_name(RegStr, Name) :-
    atom_string(RegA, RegStr),
    atom_string(RegA, Name).

%% rust_val_literal(+Str, -RustLiteral)
%  Convert a WAM constant to a Rust value literal.
%
%  Uses the shared wam_classify_constant_token/2 so quoted atoms like
%  `'42'` are classified as atoms (with the outer quotes stripped to
%  `42`) instead of being formatted verbatim with the quote characters
%  baked into the F# string.  This mirrors the F# target's #2422 fix
%  and the existing approach in the Haskell / Go / C++ / Elixir /
%  Python lowered emitters, all of which already delegate to this
%  classifier.
rust_val_literal(Str, RustVal) :-
    wam_classify_constant_token(Str, Class),
    (   Class = integer(N)
    ->  format(atom(RustVal), 'Value::Integer(~w)', [N])
    ;   Class = float(F)
    ->  format(atom(RustVal), 'Value::Float(~w)', [F])
    ;   Class = atom(Name),
        format(atom(RustVal), 'Value::Atom("~w".to_string())', [Name])
    ).
