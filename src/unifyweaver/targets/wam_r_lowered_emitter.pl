:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_r_lowered_emitter.pl -- WAM-lowered R emission (Phase 2)
%
% Plan mirrors the Haskell hybrid path in
%   docs/design/WAM_HASKELL_LOWERED_{PHILOSOPHY,SPECIFICATION,
%                                     IMPLEMENTATION_PLAN}.md
%
% Phase 2 (this file): real lowering for deterministic single-clause
%                      predicates whose body shape is in the supported
%                      set. Each lowered predicate becomes one R
%                      function that delegates per-instruction work to
%                      WamRuntime$step (so semantics stay identical to
%                      the array path) but handles control-flow
%                      instructions natively (Call/Execute/Proceed).
%
% Phase 3+ would inline simple register ops, threaded across calls to
% other lowered predicates; that's deferred. The Phase-2 design keeps
% the surface area small while wiring up the emit-mode plumbing and
% exercising the project-writer integration end-to-end.

:- module(wam_r_lowered_emitter, [
    wam_r_lowerable/3,           % +Pred, +WamCode, -Reason
    lower_predicate_to_r/4,      % +Pred, +WamCode, +Options, -Entry
    r_lowered_func_name/2        % +Functor/Arity, -RFuncName
]).

:- use_module(library(lists)).

% =====================================================================
% WAM-text parsing
% =====================================================================
% We re-parse the WAM-assembly text into a list of structured terms so
% the lowerability check can pattern-match on instruction shape. Lines
% that are blank or contain a label header (foo/2:) are dropped.

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
        sub_string(First, _, 1, 0, ":")
    ->  parse_lines(Rest, Instrs)
    ;   instr_from_parts(CleanParts, Instr)
    ->  Instrs = [Instr|RestInstrs],
        parse_lines(Rest, RestInstrs)
    ;   parse_lines(Rest, Instrs)
    ).

instr_from_parts(["allocate"], allocate).
instr_from_parts(["deallocate"], deallocate).
instr_from_parts(["proceed"], proceed).
instr_from_parts(["fail"], fail).
instr_from_parts(["get_constant", C, Ai], get_constant(C, Ai)).
instr_from_parts(["get_variable", X, Ai], get_variable(X, Ai)).
instr_from_parts(["get_value", X, Ai], get_value(X, Ai)).
instr_from_parts(["get_structure", F, Ai], get_structure(F, Ai)).
instr_from_parts(["get_list", Ai], get_list(Ai)).
instr_from_parts(["put_constant", C, Ai], put_constant(C, Ai)).
instr_from_parts(["put_variable", X, Ai], put_variable(X, Ai)).
instr_from_parts(["put_value", X, Ai], put_value(X, Ai)).
instr_from_parts(["put_structure", F, Ai], put_structure(F, Ai)).
instr_from_parts(["put_list", Ai], put_list(Ai)).
instr_from_parts(["unify_variable", X], unify_variable(X)).
instr_from_parts(["unify_value", X], unify_value(X)).
instr_from_parts(["unify_constant", C], unify_constant(C)).
instr_from_parts(["set_variable", X], set_variable(X)).
instr_from_parts(["set_value", X], set_value(X)).
instr_from_parts(["set_constant", C], set_constant(C)).
instr_from_parts(["call", Pred, N], call(Pred, N)).
instr_from_parts(["call", PredArity], call(PredArity)).
instr_from_parts(["execute", Pred, N], execute(Pred, N)).
instr_from_parts(["execute", PredArity], execute(PredArity)).
instr_from_parts(["call_foreign", Pred, N], call_foreign(Pred, N)).
instr_from_parts(["builtin_call", Op, N], builtin_call(Op, N)).
instr_from_parts(["try_me_else", L], try_me_else(L)).
instr_from_parts(["retry_me_else", L], retry_me_else(L)).
instr_from_parts(["trust_me"], trust_me).
instr_from_parts(["jump", L], jump(L)).
instr_from_parts(["cut_ite"], cut_ite).
instr_from_parts(["switch_on_constant" | _], switch_on_constant).
instr_from_parts(["switch_on_structure" | _], switch_on_structure).

% =====================================================================
% Lowerability
% =====================================================================
%
% A predicate is Phase-2-lowerable if:
%   (1) all its instructions are in the supported set,
%   (2) it has no choice-point instructions (try_me_else / retry_me_else
%       / trust_me) -- multi-clause predicates stay on the array path
%       so the standard backtracking machinery handles them, and
%   (3) it contains no switch_on_* dispatch (those imply multi-clause).
%
% Reason is unified with `deterministic` on success; future phases can
% add more granular reasons.

wam_r_lowerable(_PI, WamCode, deterministic) :-
    parse_wam_text(WamCode, Instrs),
    \+ member(try_me_else(_),   Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me,         Instrs),
    \+ member(switch_on_constant,  Instrs),
    \+ member(switch_on_structure, Instrs),
    forall(member(I, Instrs), supported_op(I)).

supported_op(allocate).
supported_op(deallocate).
supported_op(proceed).
supported_op(get_constant(_, _)).
supported_op(get_variable(_, _)).
supported_op(get_value(_, _)).
supported_op(get_structure(_, _)).
supported_op(get_list(_)).
supported_op(put_constant(_, _)).
supported_op(put_variable(_, _)).
supported_op(put_value(_, _)).
supported_op(put_structure(_, _)).
supported_op(put_list(_)).
supported_op(unify_variable(_)).
supported_op(unify_value(_)).
supported_op(unify_constant(_)).
supported_op(set_variable(_)).
supported_op(set_value(_)).
supported_op(set_constant(_)).
supported_op(call(_, _)).
supported_op(call(_)).
supported_op(execute(_, _)).
supported_op(execute(_)).
supported_op(call_foreign(_, _)).
supported_op(builtin_call(_, _)).

% =====================================================================
% Function-name generation
% =====================================================================

%% r_lowered_func_name(+Functor/Arity, -RFuncName)
%  foo/2 -> "lowered_foo_2", my_pred/3 -> "lowered_my_pred_3".
r_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, FStr),
    sanitize_r_ident(FStr, SanStr),
    format(atom(Name), 'lowered_~w_~w', [SanStr, Arity]).

sanitize_r_ident(In, Out) :-
    string_codes(In, Codes),
    maplist(r_safe_code, Codes, OutCodes),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

r_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z -> true
    ;   C >= 0'A, C =< 0'Z -> true
    ;   C >= 0'0, C =< 0'9 -> true
    ;   C =:= 0'_ -> true
    ),
    !.
r_safe_code(_, 0'_).

% =====================================================================
% Per-instruction emission
% =====================================================================
%
% The lowered function body is a sequence of R statements, one per WAM
% instruction. Most instructions delegate to WamRuntime$step using the
% same R Instruction-literal that the array path uses; we re-tokenize
% and feed wam_r_target's wam_parts_to_r/3 to keep the literal shape
% identical. Control instructions are emitted natively:
%   proceed       -> return(TRUE)
%   call P/N      -> save cp, set pc to label, run, restore cp
%   execute P/N   -> tail call (set pc, return WamRuntime$run(...))

%% lower_predicate_to_r(+Pred/Arity, +WamCode, +Options, -Entry)
%  Entry = lowered(PredName, FuncName, RCode).
lower_predicate_to_r(PI, WamCode, _Opts,
                     lowered(PredName, FuncName, Code)) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    r_lowered_func_name(Pred/Arity, FuncName),
    atom_string(WamCode, Str),
    split_string(Str, "\n", "", Lines),
    with_output_to(string(Body), emit_lines(Lines, "  ")),
    format(string(Code),
'# Lowered: ~w  (Phase-2 deterministic single-clause)
~w <- function(program, state) {
~w  return(TRUE)
}', [PredName, FuncName, Body]).

emit_lines([], _).
emit_lines([Line|Rest], Ind) :-
    wam_r_target:tokenize_wam_line(Line, Parts),
    (   Parts == []
    ->  true
    ;   Parts = [First|_], sub_string(First, _, 1, 0, ":")
    ->  true
    ;   emit_line_parts(Parts, Ind)
    ),
    emit_lines(Rest, Ind).

% --- Control instructions: special emission --------------------------

emit_line_parts(["proceed"], I) :-
    format("~wreturn(TRUE)~n", [I]).
emit_line_parts(["call", PredArity], I) :-
    emit_call(PredArity, I).
emit_line_parts(["call", Pred, ArityStr], I) :-
    strip_arity_local(Pred, PredName),
    format(string(PredArity), "~w/~w", [PredName, ArityStr]),
    emit_call(PredArity, I).
emit_line_parts(["execute", PredArity], I) :-
    emit_execute(PredArity, I).
emit_line_parts(["execute", Pred, ArityStr], I) :-
    strip_arity_local(Pred, PredName),
    format(string(PredArity), "~w/~w", [PredName, ArityStr]),
    emit_execute(PredArity, I).
% --- Default: delegate to step with the same R literal the array uses
emit_line_parts(Parts, I) :-
    wam_r_target:wam_parts_to_r(Parts, [], Lit),
    format("~wif (!isTRUE(WamRuntime$step(program, state, ~w))) return(FALSE)~n",
           [I, Lit]).

emit_call(PredArity, I) :-
    format("~w{~n", [I]),
    format("~w  saved_cp <- state$cp~n", [I]),
    format("~w  tgt <- program$labels[[\"~w\"]]~n", [I, PredArity]),
    format("~w  if (is.null(tgt)) return(FALSE)~n", [I]),
    format("~w  state$cp <- 0L~n", [I]),
    format("~w  state$pc <- as.integer(tgt)~n", [I]),
    format("~w  if (!isTRUE(WamRuntime$run(program, state))) return(FALSE)~n", [I]),
    format("~w  state$halt <- FALSE~n", [I]),
    format("~w  state$cp <- saved_cp~n", [I]),
    format("~w}~n", [I]).

emit_execute(PredArity, I) :-
    format("~w{~n", [I]),
    format("~w  tgt <- program$labels[[\"~w\"]]~n", [I, PredArity]),
    format("~w  if (is.null(tgt)) return(FALSE)~n", [I]),
    format("~w  state$pc <- as.integer(tgt)~n", [I]),
    format("~w  return(isTRUE(WamRuntime$run(program, state)))~n", [I]),
    format("~w}~n", [I]).

%% strip_arity_local(+TokenWithArity, -Name)
%  Same convention as wam_r_target's strip_arity_suffix: drop trailing
%  "/N" if present so we can rebuild "Name/N" from the explicit arity
%  token that follows in the WAM line.
strip_arity_local(Tok, Name) :-
    (   sub_string(Tok, B, 1, _, "/")
    ->  sub_string(Tok, 0, B, _, Name)
    ;   Name = Tok
    ).
