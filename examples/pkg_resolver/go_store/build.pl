:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% build.pl -- compile resolver.pl + resolver_store.pl through wam_go
% (prefer_wam(true)) with D43 store-backed fact sources for
% store_pkg/2, store_dep/2, store_conflict/2, store_revdep/2,
% store_provides/2. Mirrors examples/pkg_resolver/wamjs_store/build.pl
% for the Go lane.
%
% Declaration-level switch (UW_STORE_BACKEND or 4th argv):
%   indexed  — source(P/2, indexed(Prefix))   default; Prefix.data+.idx
%   lmdb     — source(P/2, lmdb(Dir))          opt-in; loud error on Go
% Same 10 queries, same shared adapter, no silent fallback.

:- use_module('../../../src/unifyweaver/targets/wam_go_target',
              [write_wam_go_project/3]).

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

backend_kind(Kind) :-
    current_prolog_flag(argv, Argv),
    (   Argv = [_, _, _, Raw|_]
    ->  downcase_atom(Raw, Kind0)
    ;   getenv('UW_STORE_BACKEND', Raw)
    ->  downcase_atom(Raw, Kind0)
    ;   Kind0 = indexed
    ),
    (   memberchk(Kind0, [indexed, lmdb])
    ->  Kind = Kind0
    ;   format(user_error,
               "go_store build.pl: unknown UW_STORE_BACKEND=~w (indexed|lmdb)~n",
               [Kind0]),
        halt(2)
    ).

store_sources(indexed, StoreDir, Sources) :-
    store_prefix(StoreDir, pkg, PkgP),
    store_prefix(StoreDir, dep, DepP),
    store_prefix(StoreDir, conflict, ConfP),
    store_prefix(StoreDir, revdep, RevP),
    store_prefix(StoreDir, provide, ProvP),
    Sources = [
        source(store_pkg/2, indexed(PkgP)),
        source(store_dep/2, indexed(DepP)),
        source(store_conflict/2, indexed(ConfP)),
        source(store_revdep/2, indexed(RevP)),
        source(store_provides/2, indexed(ProvP))
    ].
store_sources(lmdb, StoreDir, Sources) :-
    store_prefix(StoreDir, 'lmdb/pkg', PkgD),
    store_prefix(StoreDir, 'lmdb/dep', DepD),
    store_prefix(StoreDir, 'lmdb/conflict', ConfD),
    store_prefix(StoreDir, 'lmdb/revdep', RevD),
    store_prefix(StoreDir, 'lmdb/provide', ProvD),
    Sources = [
        source(store_pkg/2, lmdb(PkgD)),
        source(store_dep/2, lmdb(DepD)),
        source(store_conflict/2, lmdb(ConfD)),
        source(store_revdep/2, lmdb(RevD)),
        source(store_provides/2, lmdb(ProvD))
    ].

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [SrcStore, OutDir, StoreDir|_]
    ->  true
    ;   SrcStore = '../resolver_store.pl',
        OutDir = '.',
        StoreDir = '../store/.out/corpus'
    ),
    backend_kind(Kind),
    file_directory_name(SrcStore, SrcDir),
    atom_concat(SrcDir, '/resolver.pl', SrcRes),
    load_into_user(SrcRes, PredsR),
    load_into_user(SrcStore, PredsS),
    append(PredsR, PredsS, Preds0),
    StorePreds = [store_pkg/2, store_dep/2, store_conflict/2, store_revdep/2, store_provides/2],
    append(Preds0, StorePreds, Preds1),
    sort(Preds1, Preds2),
    maplist(qualify_user, Preds2, Preds),
    store_sources(Kind, StoreDir, Sources),
    length(Preds, N),
    format("go_store build.pl: compiling ~w predicates; backend=~w stores under ~w~n",
           [N, Kind, StoreDir]),
    write_wam_go_project(Preds,
        [prefer_wam(true),
         module_name('uw-pkg-resolver-store'),
         package_name(wam),
         go_wam_fact_sources(Sources)],
        OutDir),
    format("go_store build.pl: wrote Go WAM project under ~w/~n", [OutDir]).

qualify_user(P/A, user:P/A).
