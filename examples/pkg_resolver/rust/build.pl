:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% build.pl -- compile examples/pkg_resolver/resolver.pl into a Rust WAM
% project. Clauses are re-asserted into `user` (the WAM compiler reads
% user:clause/2). Directives are skipped.
%
%   swipl -q -g main -t halt build.pl -- ../resolver.pl uw_resolve_wam

:- use_module('../../../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).

load_into_user(File, Preds) :-
    setup_call_cleanup(open(File, read, S), load_terms(S, [], Acc), close(S)),
    sort(Acc, Preds).

load_terms(S, Acc, Preds) :-
    read_term(S, T, []),
    (   T == end_of_file
    ->  Preds = Acc
    ;   T = (:- _)
    ->  load_terms(S, Acc, Preds)
    ;   pred_of_term(T, PA),
        assertz(user:T),
        load_terms(S, [PA|Acc], Preds)
    ).

pred_of_term((Head :- _), P/A) :- !, functor(Head, P, A).
pred_of_term(Head, P/A) :- functor(Head, P, A).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [Src, OutDir|_]
    ->  true
    ;   Src = '../resolver.pl',
        OutDir = 'uw_resolve_wam'
    ),
    load_into_user(Src, Preds0),
    findall(user:(P/A), member(P/A, Preds0), Preds),
    length(Preds, N),
    format("build.pl: compiling ~w predicates from ~w~n", [N, Src]),
    write_wam_rust_project(Preds,
        [ module_name('uw_resolve_wam'),
          wam_fallback(true),
          no_kernels(true),
          emit_mode(interpreter) ],
        OutDir),
    format("build.pl: wrote Rust WAM project under ~w~n", [OutDir]).
