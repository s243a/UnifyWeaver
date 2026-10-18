:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% build_diff.pl -- compile examples/pkg_resolver/resolver.pl into a C++ WAM
% project (interpreter emit mode, emit_main(false)) for the differential driver.
% Clauses are re-asserted into `user` (WAM compiler reads user:clause/2).
% Directives are skipped.

:- use_module('../../../src/unifyweaver/targets/wam_cpp_target',
              [write_wam_cpp_project/3]).

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
    (   Argv = [Src0, Out0|_]
    ->  Src = Src0, OutDir = Out0
    ;   Argv = [Out0|_]
    ->  Src = '../resolver.pl',
        (Out0 == '.' -> OutDir = 'diff' ; OutDir = Out0)
    ;   Src = '../resolver.pl',
        OutDir = 'diff'
    ),
    load_into_user(Src, Preds0),
    maplist(qualify_user, Preds0, Preds),
    length(Preds, N),
    format("build_diff.pl: compiling ~w predicates from ~w~n", [N, Src]),
    write_wam_cpp_project(Preds,
        [emit_mode(interpreter),
         emit_main(false),
         include_stdlib([predsort]),
         module_name('uw-pkg-resolver-diff')],
        OutDir),
    format("build_diff.pl: wrote C++ WAM diff project under ~w/cpp/~n", [OutDir]).

qualify_user(P/A, user:P/A).
