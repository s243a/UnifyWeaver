:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_resolver_store.pl -- store adapter vs term-catalog identity.
% Existing test_resolver.pl scenarios are not modified.
%
%   swipl -q -g test_resolver_store -t halt examples/pkg_resolver/test_resolver_store.pl

:- module(test_resolver_store, [test_resolver_store/0]).

:- use_module(library(plunit)).
:- use_module(resolver).
:- use_module(resolver_store).
:- use_module(test_resolver, [scenario_catalog/2, corpus_case/4]).

test_resolver_store :-
    run_tests(pkg_resolver_store),
    format("pkg_resolver store-backed corpus: tests finished~n", []).

term_result(resolve, Cat, Args, ok(Sel)) :-
    resolve(Cat, Args, Sel), !.
term_result(resolve, _, _, fail).
term_result(resolve_layered, Cat, Args, ok(Sel)) :-
    resolve_layered(Cat, Args, Sel), !.
term_result(resolve_layered, _, _, fail).
term_result(explain_blocked, Cat, Args, ok(List)) :-
    explain_blocked_list(Cat, Args, List).
term_result(layer_closure, Cat, Args, ok(Layer)) :-
    layer_closure(Cat, Args, Layer), !.
term_result(layer_closure, _, _, fail).
term_result(removal_orphans, Cat, Args, ok(Orphans)) :-
    removal_orphans(Cat, Args, Orphans).
term_result(safe_upgrade, Cat, [Pkg, Ver], ok(V)) :-
    safe_upgrade(Cat, Pkg, Ver, V).
term_result(upgrade_set, Cat, [Pkg, Ver], R) :-
    upgrade_set_result(Cat, Pkg, Ver, R).
term_result(freeze_audit, Cat, _, ok(A)) :-
    freeze_audit(Cat, A).
term_result(dependents, Cat, Args, ok(Ds)) :-
    dependents(Cat, Args, Ds).
term_result(dependents_installed, Cat, Args, ok(Ds)) :-
    dependents_installed(Cat, Args, Ds).

store_result(resolve, Env, Args, ok(Sel)) :-
    resolve_store(Env, Args, Sel), !.
store_result(resolve, _, _, fail).
store_result(resolve_layered, Env, Args, ok(Sel)) :-
    resolve_layered_store(Env, Args, Sel), !.
store_result(resolve_layered, _, _, fail).
store_result(explain_blocked, Env, Args, ok(List)) :-
    explain_blocked_list_store(Env, Args, List).
store_result(layer_closure, Env, Args, ok(Layer)) :-
    layer_closure_store(Env, Args, Layer), !.
store_result(layer_closure, _, _, fail).
store_result(removal_orphans, Env, Args, ok(Orphans)) :-
    removal_orphans_store(Env, Args, Orphans).
store_result(safe_upgrade, Env, [Pkg, Ver], ok(V)) :-
    safe_upgrade_store(Env, Pkg, Ver, V).
store_result(upgrade_set, Env, [Pkg, Ver], R) :-
    upgrade_set_result_store(Env, Pkg, Ver, R).
store_result(freeze_audit, Env, _, ok(A)) :-
    freeze_audit_store(Env, A).
store_result(dependents, Env, Args, ok(Ds)) :-
    dependents_store(Env, Args, Ds).
store_result(dependents_installed, Env, Args, ok(Ds)) :-
    dependents_installed_store(Env, Args, Ds).

prepare_store(CatName, Env) :-
    store_clear,
    scenario_catalog(CatName, Cat),
    assert_catalog_store(CatName, Cat),
    env_from_catalog(CatName, Cat, Env).

:- begin_tests(pkg_resolver_store).

test(store_matches_term_corpus) :-
    findall(Id-Term-Store, (
        corpus_case(Id, CatName, Query, Args),
        scenario_catalog(CatName, Cat),
        prepare_store(CatName, Env),
        term_result(Query, Cat, Args, Term),
        store_result(Query, Env, Args, Store),
        Term \== Store
    ), Bad),
    (   Bad == []
    ->  findall(Id, corpus_case(Id, _, _, _), All),
        length(All, N),
        format("store-backed corpus: ~w/~w identical to term-catalog~n", [N, N])
    ;   format("store mismatches: ~q~n", [Bad]),
        assertion(Bad == [])
    ).

test(store_env_split_ignores_big_lists) :-
    scenario_catalog(linear, Cat),
    prepare_store(linear, Env),
    % Env has no packages/depends; resolution still closes a→b→c via the store.
    resolve_store(Env, [a], Sel),
    resolve(Cat, [a], Term),
    assertion(Sel == Term),
    !.

:- end_tests(pkg_resolver_store).
