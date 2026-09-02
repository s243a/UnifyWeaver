:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% build.pl -- compile resolver.pl + resolver_store.pl with D43 indexed
% fact sources for store_pkg/2, store_dep/2, store_conflict/2, store_revdep/2.

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

store_prefix(StoreDir, Name, Prefix) :-
    atom_concat(StoreDir, '/', T),
    atom_concat(T, Name, Prefix).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [SrcStore, OutDir, StoreDir|_]
    ->  true
    ;   SrcStore = '../resolver_store.pl',
        OutDir = '.',
        StoreDir = '../store/.out/corpus'
    ),
    file_directory_name(SrcStore, SrcDir),
    atom_concat(SrcDir, '/resolver.pl', SrcRes),
    load_into_user(SrcRes, PredsR),
    load_into_user(SrcStore, PredsS),
    append(PredsR, PredsS, Preds0),
    StorePreds = [store_pkg/2, store_dep/2, store_conflict/2, store_revdep/2],
    append(Preds0, StorePreds, Preds1),
    sort(Preds1, Preds),
    store_prefix(StoreDir, pkg, PkgP),
    store_prefix(StoreDir, dep, DepP),
    store_prefix(StoreDir, conflict, ConfP),
    store_prefix(StoreDir, revdep, RevP),
    Sources = [
        source(store_pkg/2, indexed(PkgP)),
        source(store_dep/2, indexed(DepP)),
        source(store_conflict/2, indexed(ConfP)),
        source(store_revdep/2, indexed(RevP))
    ],
    length(Preds, N),
    format("wamjs_store build.pl: compiling ~w predicates; stores under ~w~n", [N, StoreDir]),
    write_wam_javascript_project(Preds,
        [emit_mode(mixed), javascript_wam_fact_sources(Sources)],
        OutDir),
    format("wamjs_store build.pl: wrote JS WAM project under ~w/js/~n", [OutDir]).
