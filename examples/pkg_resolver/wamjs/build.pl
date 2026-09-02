:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% build.pl -- compile examples/pkg_resolver/resolver.pl into a JS WAM project
% (mixed emit). Clauses are re-asserted into `user` (WAM compiler reads
% user:clause/2). Directives are skipped.

:- use_module('../../../src/unifyweaver/targets/wam_javascript_target',
              [write_wam_javascript_project/3]).

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
        OutDir = '.'
    ),
    load_into_user(Src, Preds),
    length(Preds, N),
    format("build.pl: compiling ~w predicates from ~w~n", [N, Src]),
    forall(member(P/A, Preds), format("  ~w/~w~n", [P, A])),
    write_wam_javascript_project(Preds, [emit_mode(mixed)], OutDir),
    format("build.pl: wrote JS WAM project under ~w/js/~n", [OutDir]).
