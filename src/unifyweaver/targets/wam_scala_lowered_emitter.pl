:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_scala_lowered_emitter.pl — WAM-lowered Scala emission
%
% Emits one Scala function per deterministic predicate, giving the Scala
% hybrid WAM target the per-predicate native fast-path that the other
% hybrid WAM targets (Haskell, Rust, C++, F#, Go, Clojure, Lua, Elixir,
% R) already ship.  This closes the "single biggest gap" called out in
% docs/WAM_TARGET_ROADMAP.md:
%
%   "Scala ... WAM-instruction lowering only; no wam_scala_lowered_emitter.pl;
%    step-loop interpreter ... add per-predicate native fast-path emitter
%    (single biggest gap)".
%
% Architecture — modelled on wam_rust_lowered_emitter.pl, because the
% Scala runtime (like Rust's) is *imperative / mutable*: WamState is
% mutated in place and `step` returns Unit, unlike the Haskell/F#
% functional runtimes whose `step` returns an Option.  Each lowered
% predicate becomes:
%
%     def lowered_<pred>_<arity>(s: WamState, program: WamProgram): Boolean
%
% Simple register operations (put_constant, put_value, get_variable, ...)
% are inlined as direct mutations of `s`.  Failure-capable structure and
% unification operations delegate to small `lo*` helper methods added to
% WamRuntime (loGetConstant, loGetStructure, loUnifyVariable, ...), each
% of which returns a Boolean and never calls `backtrack` — the lowered
% function decides what to do on failure.  Deterministic builtins
% (=/2, is/2, comparisons, type checks, !/0) route through loBuiltin/2.
%
% Determinism contract (identical to Rust/Clojure):
%   - Only clause 1 of a predicate is lowered.  For a multi-clause
%     predicate, clause 1 runs inline; on failure the function pushes a
%     choice point for clause 2+ and backtracks into the interpreter.
%   - A predicate is lowerable only if clause 1 is deterministic (no
%     try/retry/trust *inside* clause 1) and every clause-1 instruction
%     is supported.
%   - Multi-clause predicates whose clause-1 body calls a *user*
%     predicate are NOT lowered (keeps the choice-point stack clean for
%     the backtrack-into-clause-2 fallback).  Single-clause bodies may
%     call sub-predicates; the sub-call is delegated to the interpreter
%     and its first solution is taken (deterministic-prefix semantics).
%
% See: src/unifyweaver/targets/wam_rust_lowered_emitter.pl  (primary model)
%      src/unifyweaver/targets/wam_fsharp_lowered_emitter.pl (multi-clause shape)

:- module(wam_scala_lowered_emitter, [
    wam_scala_lowerable/3,        % +PI, +WamCode, -Reason
    lower_predicate_to_scala/4,   % +PI, +WamCode, +Options, -lowered(PredName, FuncName, Code)
    is_deterministic_pred_scala/1,
    scala_lowered_func_name/2     % +Pred/Arity, -Name
]).

:- use_module(library(lists)).
:- use_module('../targets/wam_scala_target', [
       scala_lowered_constant_term/2,
       scala_lowered_functor_arity/3,
       scala_lowered_reg_index/2,
       scala_lowered_intern_atom/2
   ]).

% =====================================================================
% Parsing — WAM text → instruction terms (target-agnostic format)
% =====================================================================

parse_wam_text_scala(WamText, Instrs) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_lines_scala(Lines, Instrs).

parse_lines_scala([], []).
parse_lines_scala([Line|Rest], Instrs) :-
    split_string(Line, " \t,", " \t,", Parts0),
    delete(Parts0, "", Parts),
    (   Parts == []
    ->  parse_lines_scala(Rest, Instrs)
    ;   Parts = [First|_],
        sub_string(First, _, 1, 0, ":")
    ->  parse_lines_scala(Rest, Instrs)      % label line — skip
    ;   instr_from_parts_scala(Parts, Instr)
    ->  Instrs = [Instr|More],
        parse_lines_scala(Rest, More)
    ;   parse_lines_scala(Rest, Instrs)      % unrecognised — skip
    ).

% Mirrors the token shapes wam_scala_target.pl already recognises.
instr_from_parts_scala(["get_constant", C, Ai],   get_constant(C, Ai)).
instr_from_parts_scala(["get_variable", Xn, Ai],  get_variable(Xn, Ai)).
instr_from_parts_scala(["get_value", Xn, Ai],     get_value(Xn, Ai)).
instr_from_parts_scala(["get_structure", F, Ai],  get_structure(F, Ai)).
instr_from_parts_scala(["get_list", Ai],          get_list(Ai)).
instr_from_parts_scala(["put_constant", C, Ai],   put_constant(C, Ai)).
instr_from_parts_scala(["put_variable", Xn, Ai],  put_variable(Xn, Ai)).
instr_from_parts_scala(["put_value", Xn, Ai],     put_value(Xn, Ai)).
instr_from_parts_scala(["put_structure", F, Ai],  put_structure(F, Ai)).
instr_from_parts_scala(["put_list", Ai],          put_list(Ai)).
instr_from_parts_scala(["set_variable", Xn],      set_variable(Xn)).
instr_from_parts_scala(["set_value", Xn],         set_value(Xn)).
instr_from_parts_scala(["set_constant", C],       set_constant(C)).
instr_from_parts_scala(["unify_variable", Xn],    unify_variable(Xn)).
instr_from_parts_scala(["unify_value", Xn],       unify_value(Xn)).
instr_from_parts_scala(["unify_constant", C],     unify_constant(C)).
instr_from_parts_scala(["allocate"],              allocate).
instr_from_parts_scala(["deallocate"],            deallocate).
instr_from_parts_scala(["proceed"],               proceed).
instr_from_parts_scala(["fail"],                  fail).
instr_from_parts_scala(["try_me_else", L],        try_me_else(L)).
instr_from_parts_scala(["retry_me_else", L],      retry_me_else(L)).
instr_from_parts_scala(["trust_me"],              trust_me).
instr_from_parts_scala(["jump", L],               jump(L)).
% call/execute come in 2- or 3-token forms (name/arity in one token, or
% name and arity split).  Normalise both to call(Name, Arity).
instr_from_parts_scala(["call", PredArity],       call(Name, Arity)) :-
    parse_pred_arity(PredArity, Name, Arity).
instr_from_parts_scala(["call", Pred, ArityStr],  call(Name, Arity)) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix_scala(Pred, Name).
instr_from_parts_scala(["execute", PredArity],    execute(Name, Arity)) :-
    parse_pred_arity(PredArity, Name, Arity).
instr_from_parts_scala(["execute", Pred, ArityStr], execute(Name, Arity)) :-
    number_string(Arity, ArityStr),
    strip_arity_suffix_scala(Pred, Name).
instr_from_parts_scala(["builtin_call", Pred, ArityStr], builtin_call(Pred, Arity)) :-
    number_string(Arity, ArityStr).
instr_from_parts_scala(["call_foreign", Pred, ArityStr], call_foreign(Pred, Arity)) :-
    number_string(Arity, ArityStr).

parse_pred_arity(PredArity, Name, Arity) :-
    ( strip_arity_suffix_scala(PredArity, Name0), Name0 \== PredArity
    ->  Name = Name0,
        sub_string(PredArity, B, 1, _, "/"),
        B1 is B + 1, sub_string(PredArity, B1, _, 0, AS),
        number_string(Arity, AS)
    ;   Name = PredArity, Arity = 0
    ).

strip_arity_suffix_scala(Pred, Name) :-
    (   sub_string(Pred, B, 1, _, "/")
    ->  sub_string(Pred, 0, B, _, Name)
    ;   Name = Pred
    ).

% =====================================================================
% Lowerability analysis
% =====================================================================

%% wam_scala_lowerable(+PI, +WamCode, -Reason) is semidet.
%  Succeeds if the predicate's clause 1 is a deterministic body composed
%  entirely of supported instructions.  Reason records whether the
%  predicate is single- or multi-clause (informational only — the entry
%  wrapper's interpreter fallback makes both shapes sound for boolean
%  queries regardless of clause-1 sub-calls).
wam_scala_lowerable(_PI, WamCode, Reason) :-
    parse_wam_text_scala(WamCode, Instrs),
    clause1_instrs_scala(Instrs, C1, MultiClause),
    is_deterministic_pred_scala(C1),
    forall(member(I, C1), scala_supported(I)),
    ( MultiClause == true -> Reason = multi_clause_1 ; Reason = single_clause ).

%% clause1_instrs_scala(+Instrs, -Clause1, -MultiClause)
%  Extracts clause 1.  MultiClause is `true` when the predicate opens
%  with a try_me_else chain, `false` otherwise.
clause1_instrs_scala([try_me_else(_)|Rest], C1, true) :- !,
    take_to_proceed_scala(Rest, C1).
clause1_instrs_scala(Instrs, Instrs, false).

take_to_proceed_scala([], []).
take_to_proceed_scala([proceed|_], [proceed]) :- !.
take_to_proceed_scala([fail|_], [fail]) :- !.
take_to_proceed_scala([I|Rest], [I|More]) :- take_to_proceed_scala(Rest, More).

%% is_deterministic_pred_scala(+Instrs)
%  True if the clause-1 instruction list has no choice-point ops.
is_deterministic_pred_scala(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

% Supported clause-1 instruction whitelist.
scala_supported(get_constant(_, _)).
scala_supported(get_variable(_, _)).
scala_supported(get_value(_, _)).
scala_supported(get_structure(_, _)).
scala_supported(get_list(_)).
scala_supported(put_constant(_, _)).
scala_supported(put_variable(_, _)).
scala_supported(put_value(_, _)).
scala_supported(put_structure(_, _)).
scala_supported(put_list(_)).
scala_supported(set_variable(_)).
scala_supported(set_value(_)).
scala_supported(set_constant(_)).
scala_supported(unify_variable(_)).
scala_supported(unify_value(_)).
scala_supported(unify_constant(_)).
scala_supported(allocate).
scala_supported(deallocate).
scala_supported(proceed).
scala_supported(fail).
scala_supported(call(_, _)).
scala_supported(execute(_, _)).
% Only deterministic builtins may be lowered.  Nondeterministic builtins
% (member/2, between/3, sort/2, findall, ...) keep the predicate in the
% interpreter.
scala_supported(builtin_call(Op, _)) :-
    deterministic_builtin(Op).

deterministic_builtin("=/2").
deterministic_builtin("true/0").
deterministic_builtin("fail/0").
deterministic_builtin("!/0").
deterministic_builtin("is/2").
deterministic_builtin("=:=/2").
deterministic_builtin("=\\=/2").
deterministic_builtin("</2").
deterministic_builtin(">/2").
deterministic_builtin("=</2").
deterministic_builtin(">=/2").
deterministic_builtin("var/1").
deterministic_builtin("nonvar/1").
deterministic_builtin("atom/1").
deterministic_builtin("number/1").
deterministic_builtin("integer/1").
deterministic_builtin("float/1").
deterministic_builtin("atomic/1").

% =====================================================================
% Function name generation
% =====================================================================

%% scala_lowered_func_name(+Functor/Arity, -Name)
%  foo/2 -> 'lowered_foo_2',  my$pred/3 -> 'lowered_my_pred_3'
scala_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, FStr),
    sanitize_scala_ident(FStr, San),
    format(atom(Name), 'lowered_~w_~w', [San, Arity]).

sanitize_scala_ident(In, Out) :-
    string_codes(In, Codes),
    maplist(scala_safe_code, Codes, OutCodes),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

scala_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z -> true
    ;   C >= 0'A, C =< 0'Z -> true
    ;   C >= 0'0, C =< 0'9 -> true
    ;   C =:= 0'_
    ), !.
scala_safe_code(_, 0'_).

% =====================================================================
% Emission
% =====================================================================

%% lower_predicate_to_scala(+PI, +WamCode, +Options, -lowered(PredName, FuncName, Code))
%  PredName is the "pred/arity" key (used to register the lowered entry).
%  FuncName is the generated Scala identifier.
%  Code is the Scala source for the function definition(s).
lower_predicate_to_scala(PI, WamCode, Options, lowered(PredName, FuncName, Code)) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    scala_lowered_func_name(Pred/Arity, FuncName),
    parse_wam_text_scala(WamCode, Instrs),
    clause1_instrs_scala(Instrs, C1, MultiClause),
    (   member(foreign_pred_keys(FK0), Options)
    ->  maplist(foreign_key_atom, FK0, ForeignKeys)
    ;   ForeignKeys = []
    ),
    with_output_to(string(Code),
        emit_function_scala(FuncName, C1, MultiClause, ForeignKeys)).

foreign_key_atom(K, A) :- ( atom(K) -> A = K ; atom_string(A, K) ).

%% emit_function_scala(+FuncName, +Clause1, +MultiClause, +ForeignKeys)
%  Both single- and multi-clause predicates emit the same shape: the
%  function runs clause 1 inline on the supplied (already arg-loaded)
%  state and returns whether clause 1 produced a solution.  The entry
%  wrapper (generated in wam_scala_target.pl) is responsible for the
%  interpreter fallback when this returns false:
%
%      if (lowered_p_a(s, prog)) true
%      else WamRuntime.runPredicate(prog, startPc, args)   // fresh state
%
%  This is sound for boolean top-level queries: a `true` is always a real
%  solution (clause 1 genuinely succeeded), and a `false` defers to the
%  complete step-loop interpreter, so first-argument indexing, clause 2+,
%  and backtracking into nondeterministic sub-goals are all handled there.
%  `MultiClause` is accepted for signature symmetry but no longer changes
%  the emission — the entry-wrapper fallback subsumes the old replay/CP
%  juggling and is correct regardless of the predicate's indexing prefix.
emit_function_scala(FuncName, C1, _MultiClause, FK) :-
    format("  /** ~w — clause-1 fast path (generated); entry wrapper handles fallback. */~n", [FuncName]),
    format("  def ~w(s: WamState, program: WamProgram): Boolean = {~n", [FuncName]),
    emit_body_scala(C1, "    ", FK),
    % Only emit the trailing `true` when control can fall off the end of
    % the body. An execute (tail call) or fail emits its own `return`, so
    % a trailing `true` there would be unreachable dead code.
    (   body_falls_through(C1)
    ->  format("    true~n", [])
    ;   true
    ),
    format("  }~n", []).

%% body_falls_through(+Clause1Instrs)
%  False when the clause ends in a return-emitting terminal (execute/fail).
body_falls_through(Instrs) :-
    ( last(Instrs, Last) -> true ; Last = proceed ),
    \+ ( Last = execute(_, _) ),
    Last \== fail.

%% emit_body_scala(+Instrs, +Indent, +ForeignKeys)
emit_body_scala([], _, _).
emit_body_scala([I|Rest], Ind, FK) :-
    emit_one_scala(I, Ind, FK),
    emit_body_scala(Rest, Ind, FK).

% --- Terminal ---
emit_one_scala(proceed, I, _) :-
    format("~w// proceed (clause complete)~n", [I]).
emit_one_scala(fail, I, _) :-
    format("~wreturn false~n", [I]).

% --- Head unification (get_*) ---
emit_one_scala(get_constant(CStr, AiStr), I, _) :-
    scala_lowered_reg_index(AiStr, Ai),
    scala_lowered_constant_term(CStr, Term),
    format("~wif (!WamRuntime.loGetConstant(s, ~w, ~w)) return false~n", [I, Term, Ai]).
emit_one_scala(get_variable(XnStr, AiStr), I, _) :-
    scala_lowered_reg_index(XnStr, Xn), scala_lowered_reg_index(AiStr, Ai),
    format("~wWamRuntime.setReg(s, ~w, s.regs(~w))~n", [I, Xn, Ai]).
emit_one_scala(get_value(XnStr, AiStr), I, _) :-
    scala_lowered_reg_index(XnStr, Xn), scala_lowered_reg_index(AiStr, Ai),
    format("~wif (!WamRuntime.loGetValue(s, ~w, ~w)) return false~n", [I, Xn, Ai]).
emit_one_scala(get_structure(FStr, AiStr), I, _) :-
    scala_lowered_reg_index(AiStr, Ai),
    scala_lowered_functor_arity(FStr, FName, FArity),
    scala_lowered_intern_atom(FName, FId),
    format("~wif (!WamRuntime.loGetStructure(s, program, ~w, ~w, ~w)) return false~n",
           [I, FId, Ai, FArity]).
emit_one_scala(get_list(AiStr), I, _) :-
    scala_lowered_reg_index(AiStr, Ai),
    scala_lowered_intern_atom("[|]", ConsId),
    format("~wif (!WamRuntime.loGetList(s, program, ~w, ~w)) return false~n", [I, Ai, ConsId]).

% --- Body construction (put_*) ---
emit_one_scala(put_constant(CStr, AiStr), I, _) :-
    scala_lowered_reg_index(AiStr, Ai),
    scala_lowered_constant_term(CStr, Term),
    format("~wWamRuntime.setReg(s, ~w, ~w)~n", [I, Ai, Term]).
emit_one_scala(put_variable(XnStr, AiStr), I, _) :-
    scala_lowered_reg_index(XnStr, Xn), scala_lowered_reg_index(AiStr, Ai),
    format("~w{ val v = WamRuntime.freshVar(s); WamRuntime.setReg(s, ~w, v); s.regs(~w) = v }~n",
           [I, Xn, Ai]).
emit_one_scala(put_value(XnStr, AiStr), I, _) :-
    scala_lowered_reg_index(XnStr, Xn), scala_lowered_reg_index(AiStr, Ai),
    format("~ws.regs(~w) = WamRuntime.deref(s.bindings, WamRuntime.getReg(s, ~w))~n", [I, Ai, Xn]).
emit_one_scala(put_structure(FStr, AiStr), I, _) :-
    scala_lowered_reg_index(AiStr, Ai),
    scala_lowered_functor_arity(FStr, FName, FArity),
    scala_lowered_intern_atom(FName, FId),
    format("~wWamRuntime.loPutStructure(s, ~w, ~w, ~w)~n", [I, FId, Ai, FArity]).
emit_one_scala(put_list(AiStr), I, _) :-
    scala_lowered_reg_index(AiStr, Ai),
    scala_lowered_intern_atom("[|]", ConsId),
    format("~wWamRuntime.loPutList(s, ~w, ~w)~n", [I, Ai, ConsId]).

% --- Set (write-mode, build frame) ---
emit_one_scala(set_variable(XnStr), I, _) :-
    scala_lowered_reg_index(XnStr, Xn),
    format("~wif (!WamRuntime.loSetVariable(s, ~w)) return false~n", [I, Xn]).
emit_one_scala(set_value(XnStr), I, _) :-
    scala_lowered_reg_index(XnStr, Xn),
    format("~wif (!WamRuntime.loSetValue(s, ~w)) return false~n", [I, Xn]).
emit_one_scala(set_constant(CStr), I, _) :-
    scala_lowered_constant_term(CStr, Term),
    format("~wif (!WamRuntime.loSetConstant(s, ~w)) return false~n", [I, Term]).

% --- Unify (read/write-mode) ---
emit_one_scala(unify_variable(XnStr), I, _) :-
    scala_lowered_reg_index(XnStr, Xn),
    format("~wif (!WamRuntime.loUnifyVariable(s, ~w)) return false~n", [I, Xn]).
emit_one_scala(unify_value(XnStr), I, _) :-
    scala_lowered_reg_index(XnStr, Xn),
    format("~wif (!WamRuntime.loUnifyValue(s, ~w)) return false~n", [I, Xn]).
emit_one_scala(unify_constant(CStr), I, _) :-
    scala_lowered_constant_term(CStr, Term),
    format("~wif (!WamRuntime.loUnifyConstant(s, ~w)) return false~n", [I, Term]).

% --- Environment ---
emit_one_scala(allocate, I, _) :-
    format("~wWamRuntime.loAllocate(s)~n", [I]).
emit_one_scala(deallocate, I, _) :-
    format("~wWamRuntime.loDeallocate(s)~n", [I]).

% --- Builtins (deterministic allowlist) ---
emit_one_scala(builtin_call(OpStr, _N), I, _) :-
    scala_string_lit(OpStr, OpLit),
    format("~wif (!WamRuntime.loBuiltin(s, program, ~w)) return false~n", [I, OpLit]).

% --- Control: sub-calls delegate to the interpreter ---
% loCall / loExecute resolve the target through program.dispatch and run
% the interpreter, so they transparently handle both user predicates and
% foreign-predicate stubs (which also have a dispatch entry).
emit_one_scala(call(NameStr, Arity), I, _) :-
    scala_string_lit(NameStr, NameLit),
    format("~wif (!WamRuntime.loCall(s, program, ~w, ~w)) return false~n", [I, NameLit, Arity]).
emit_one_scala(execute(NameStr, Arity), I, _) :-
    scala_string_lit(NameStr, NameLit),
    format("~wreturn WamRuntime.loExecute(s, program, ~w, ~w)~n", [I, NameLit, Arity]).

% --- Choice/structural markers consumed during clause-1 lowering ---
emit_one_scala(try_me_else(_), _, _) :- !.
emit_one_scala(retry_me_else(_), _, _) :- !.
emit_one_scala(trust_me, _, _) :- !.
emit_one_scala(jump(_), _, _) :- !.

% --- Fallback (should not be reached for lowerable predicates) ---
emit_one_scala(Instr, I, _) :-
    format("~w// TODO: unlowered instruction ~w~n", [I, Instr]).

%% scala_string_lit(+Raw, -Quoted) — Scala double-quoted string literal.
scala_string_lit(Raw, Quoted) :-
    atom_string(Raw, S),
    string_chars(S, Chars),
    maplist(scala_escape_char, Chars, EscLists),
    append(EscLists, EscChars),
    string_chars(Body, EscChars),
    format(string(Quoted), '"~w"', [Body]).

scala_escape_char('\\', ['\\', '\\']) :- !.
scala_escape_char('"',  ['\\', '"'])  :- !.
scala_escape_char(C, [C]).
