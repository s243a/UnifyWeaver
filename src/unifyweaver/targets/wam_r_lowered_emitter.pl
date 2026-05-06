:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_r_lowered_emitter.pl -- WAM-lowered R emission (Phase 1: stub)
%
% Plan mirrors the Haskell hybrid path in
%   docs/design/WAM_HASKELL_LOWERED_{PHILOSOPHY,SPECIFICATION,
%                                     IMPLEMENTATION_PLAN}.md
%
% Phase 1 (this file): wam_r_lowerable/3 always fails, so every
%                      predicate routes to the interpreter array path.
%                      lower_predicate_to_r/4 is provided for symmetry
%                      and emits a placeholder TODO function.
% Phase 2+: replace the stub with a real whitelist over the WAM
%           instruction stream and emit native R functions per
%           lowerable predicate. See the Rust/Haskell/Go lowered
%           emitters for prior art.

:- module(wam_r_lowered_emitter, [
    wam_r_lowerable/3,           % +Pred, +WamCode, -Reason
    lower_predicate_to_r/4,      % +Pred, +WamCode, +Options, -Entry
    r_lowered_func_name/2        % +Functor/Arity, -RFuncName
]).

:- use_module(library(lists)).

% =====================================================================
% Lowerability (Phase 1: always fails)
% =====================================================================

%% wam_r_lowerable(+Pred/Arity, +WamCode, -Reason) is semidet.
%  Phase 1 stub: always fails. Phase 2+ will return a Reason term
%  describing why the predicate is lowerable (e.g. deterministic,
%  multi_clause_1).
wam_r_lowerable(_PI, _WamCode, _Reason) :- fail.

% =====================================================================
% Function-name generation
% =====================================================================

%% r_lowered_func_name(+Functor/Arity, -RFuncName)
%  foo/2 -> "lowered_foo_2", my_pred/3 -> "lowered_my_pred_3".
%  R identifiers can contain "." but we use "_" to avoid clashing with
%  R's S3 method dispatch convention (e.g. print.foo).
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
% Emission (Phase 1 placeholder)
% =====================================================================

%% lower_predicate_to_r(+Pred/Arity, +WamCode, +Options, -Entry)
%  Entry = lowered(PredName, FuncName, RCode).
%
%  Phase 1: emits a TODO placeholder. Lowering is gated by
%  wam_r_lowerable/3 which always fails, so this code is unreachable
%  in practice. It's here to make the contract explicit and to give
%  Phase 2 a starting point.
lower_predicate_to_r(PI, _WamCode, _Opts,
                     lowered(PredName, FuncName, Code)) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    r_lowered_func_name(Pred/Arity, FuncName),
    format(string(Code),
'# Lowered: ~w
# Phase 1 stub -- lowering is not yet implemented for this target.
# See wam_r_lowered_emitter.pl, modelled on
#   wam_haskell_lowered_emitter.pl (Maybe-monad style) and
#   wam_rust_lowered_emitter.pl   (delegate-to-step style).
~w <- function(ctx, state) {
  stop("R lowered emitter not yet implemented")
}
', [PredName, FuncName]).
