:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_pruning_probes.pl -- hazard + guard probes for RESOLVER_PRUNING_DESIGN.
%
% CE1-CE6 pin the six pre-existing semantic hazards (H1-H6) recorded in
% docs/proposals/RESOLVER_PRUNING_DESIGN.md §3.2. They are *preservation*
% probes: they succeed on the pre-pruning resolver and must still succeed
% after the guards land. A red bar here means a guard changed answers.
%
% A1-A11 are the §6 review assertions for the guards that were implemented.
%
%   swipl -q -g test_pruning_probes -t halt \
%       examples/pkg_resolver/test_pruning_probes.pl

:- module(test_pruning_probes, [
    test_pruning_probes/0,
    ce1/0, ce2/0, ce3/0, ce4/0, ce5/0, ce6/0
]).

:- use_module(library(plunit)).
:- use_module(library(time)).
:- use_module(resolver).
:- use_module(test_resolver, [scenario_catalog/2]).

test_pruning_probes :-
    run_tests(pruning_probes),
    format("pkg_resolver pruning probes: tests finished~n", []).

v1(v(1,0,0)).
v2(v(2,0,0)).

% ---------------------------------------------------------------------------
% CE1-CE6 -- design doc §6.1, verbatim shapes
% ---------------------------------------------------------------------------

% CE1 -- pending order decides the first solution (rejected family R1).
ce1 :-
    v1(V1), v2(V2),
    Ps = [package(app,V1), package(b,V2), package(b,V1), package(c,V2),
          package(c,V1), package(d,V2), package(d,V1)],
    Dd = [depends(b,V2,d,eq(V1)), depends(c,V2,d,eq(V2))],
    resolve(catalog(Ps, [depends(app,V1,b,any), depends(app,V1,c,any)|Dd],
                    [], [], [], []), [app], S1),
    resolve(catalog(Ps, [depends(app,V1,c,any), depends(app,V1,b,any)|Dd],
                    [], [], [], []), [app], S2),
    S1 \== S2,
    S1 == [app-V1, b-V2, c-V1, d-V1],
    S2 == [app-V1, b-V1, c-V2, d-V2].

% CE2 -- skipping a from_base re-expansion changes the answer (R2).
ce2 :-
    v1(V1), v2(V2),
    Ps = [package(base0,V1), package(a,V1), package(b,V2), package(b,V1),
          package(x,V2), package(x,V1), package(d,V2), package(d,V1)],
    Dd = [depends(b,V2,d,eq(V1)), depends(x,V2,d,gte(V1))],
    Ds1 = [depends(base0,V1,a,any), depends(base0,V1,b,any),
           depends(a,V1,base0,any), depends(a,V1,x,any)|Dd],
    Ds2 = [depends(base0,V1,a,any), depends(base0,V1,b,any),
           depends(a,V1,x,any)|Dd],
    resolve_layered(catalog(Ps, Ds1, [], [base0-V1], [], []), [base0], S1),
    \+ resolve_layered(catalog(Ps, Ds2, [], [base0-V1], [], []), [base0], _),
    S1 == [a-V1, b-V2, d-V1, x-V2].

% CE3 -- two versions of one name via the provider path (H2).
ce3 :-
    v1(V1), v2(V2),
    Ps = [package(app,V1), package(mawk,V2), package(mawk,V1)],
    Ds = [depends(app,V1,mawk,eq(V1)), depends(app,V1,awk,any)],
    Pr = [provides(mawk,V2,awk)],
    resolve(catalog(Ps, Ds, [], [], [], [], [], [], [], Pr), [app], S1),
    S1 == [app-V1, mawk-V1, mawk-V2],
    resolve_layered(catalog(Ps, Ds, [], [mawk-V1], [], [], [], [], [], Pr),
                    [app], S2),
    S2 == [app-V1, mawk-V2].

% CE4 -- held ceiling bypassed by a provider; explain disagrees
% (H3, and the G4 provides_mentions/2 hypothesis).
ce4 :-
    ce4_catalog(Cat),
    v1(V1), v2(V2),
    resolve_layered(Cat, [app], S), S == [app-V1, bar-V1],
    explain_blocked_list(Cat, app, L),
    L == [blocked(foo, needs(gte(V2)), base_has(V1))].

ce4_catalog(Cat) :-
    v1(V1), v2(V2),
    Ps = [package(app,V1), package(bar,V1), package(foo,V2)],
    Ds = [depends(app,V1,bar,any), depends(app,V1,foo,gte(V2))],
    Cat = catalog(Ps, Ds, [], [foo-V1], [], [], [], [], [],
                  [provides(bar,V1,foo,V2)]).

% CE5 -- cyclic held deps do not terminate in layered mode (H4).
ce5 :-
    v1(V1),
    Cat = catalog([package(a,V1), package(b,V1)],
                  [depends(a,V1,b,any), depends(b,V1,a,any)],
                  [], [a-V1, b-V1], [], []),
    catch(call_with_time_limit(2, resolve_layered(Cat, [a], _)),
          time_limit_exceeded, true).

% CE6 -- layered commits to the highest candidate (H1).
ce6 :-
    v1(V1), v2(V2),
    Ps = [package(b,V2), package(b,V1), package(x,V2), package(x,V1),
          package(d,V2), package(d,V1)],
    Ds = [depends(b,V2,d,eq(V1)), depends(x,V2,d,eq(V2))],
    Cat = catalog(Ps, Ds, [], [zz-V1], [], []),
    resolve(Cat, [b,x], S1), S1 == [b-V2, d-V1, x-V1],
    \+ resolve_layered(Cat, [b,x], _).

% ---------------------------------------------------------------------------
% H5 -- named-layer providers are unreachable (dead clause). Preserved.
% ---------------------------------------------------------------------------

% A layer named other than `base` holding a provider of a virtual: the
% resolve must NOT find it (the second layer_provider/5 clause calls
% lookup_held/3 with the package name unbound, which never matches).
ce_h5 :-
    v1(V1),
    Ps = [package(app,V1), package(bar,V1)],
    Ds = [depends(app,V1,foo,any)],
    Cat = catalog(Ps, Ds, [], [], [], [],
                  [layer(extra, [bar-V1])], [], [], [provides(bar,V1,foo)]),
    % bar is held in a named layer and provides foo; H5 says the layer
    % provider path is dead, so the resolve falls through to the catalog
    % provider path and *selects* bar rather than using it in place.
    resolve_layered(Cat, [app], S),
    S == [app-V1, bar-V1].

% ---------------------------------------------------------------------------
% §6 review assertions (A1-A11) for the guards that landed
% ---------------------------------------------------------------------------

% A1 -- indexed collect_deps ≡ the linear scan, as lists (order,
% multiplicity), including duplicate rows, rows differing only in the
% constraint, and deb/3 keys that are version_lt-equal but not ==.
a1_catalog(Cat) :-
    v1(V1), v2(V2),
    D10  = deb(0, [s([],1), s([46],0)], []),
    D10b = deb(0, [s([],1), s([46],0)], [s([],0)]),   % ~= D10 under version_lt
    Ps = [package(a,V1), package(a,V1), package(a,V2),
          package(z,D10), package(z,D10b)],
    Ds = [depends(a,V1,b,any),          % duplicate row pair
          depends(a,V1,b,any),
          depends(a,V1,c,eq(V1)),       % same target, different constraint
          depends(a,V1,c,eq(V2)),
          depends(a,V2,d,any),
          depends(z,D10,p,any),         % deb keys must stay separate
          depends(z,D10b,q,any),
          depends(a,V1,alternatives([dep(m,any), dep(n,any)]), any)],
    Cat = catalog(Ps, Ds, [], [], [], []).

% A2 -- the indexed version list is the catalog-order version list of a
% name, duplicates included, before satisfies/2 filtering.
% (Checked observationally through candidates_high_first via resolve.)

% ---------------------------------------------------------------------------
:- begin_tests(pruning_probes).

test(ce1_pending_order_decides_first_solution) :- ce1.
test(ce2_from_base_reexpansion_is_load_bearing) :- ce2.
test(ce3_two_versions_one_name_via_provider) :- ce3.
test(ce4_held_ceiling_bypassed_by_provider) :- ce4.
test(ce5_cyclic_held_deps_do_not_terminate) :- ce5.
test(ce6_layered_commits_to_highest_candidate) :- ce6.
test(ce_h5_named_layer_providers_unreachable) :- ce_h5.

% A1: deps by (Name,Ver) come back in list order with multiplicity.
test(a1_collect_deps_order_and_multiplicity) :-
    a1_catalog(Cat),
    v1(V1), v2(V2),
    resolver:collect_deps(Cat, a, V1, R1),
    assertion(R1 == [req(b,any), req(b,any), req(c,eq(V1)), req(c,eq(V2)),
                     req(alternatives([dep(m,any), dep(n,any)]), any)]),
    resolver:collect_deps(Cat, a, V2, R2),
    assertion(R2 == [req(d,any)]),
    resolver:collect_deps(Cat, nosuch, V1, R3),
    assertion(R3 == []).

% A1 (deb keys): two deb/3 versions that compare equal under version_lt but
% are not == must key separately.
test(a1_deb_keys_stay_separate) :-
    a1_catalog(Cat),
    D10  = deb(0, [s([],1), s([46],0)], []),
    D10b = deb(0, [s([],1), s([46],0)], [s([],0)]),
    assertion(\+ resolver:version_lt(D10, D10b)),
    assertion(\+ resolver:version_lt(D10b, D10)),
    assertion(D10 \== D10b),
    resolver:collect_deps(Cat, z, D10, RA),
    resolver:collect_deps(Cat, z, D10b, RB),
    assertion(RA == [req(p,any)]),
    assertion(RB == [req(q,any)]).

% A2: candidate enumeration keeps catalog-order-then-descending semantics
% with duplicate package rows preserved through the filter.
test(a2_candidate_versions_with_duplicates) :-
    a1_catalog(Cat),
    v1(V1), v2(V2),
    findall(V, resolver:candidates_high_first(Cat, a, any, V), Vs),
    assertion(Vs == [V2, V1]).

% A3: the index wrapper never escapes into any answer term.
test(a3_icat_never_escapes) :-
    a1_catalog(Cat),
    v1(V1),
    resolve(Cat, [], S0), assertion(S0 == []),
    ( resolve(Cat, [req(a, eq(V1))], _) -> true ; true ),
    forall(( member(G, [resolve(Cat,[a],X), resolve_layered(Cat,[a],X)]),
             catch(call(G), _, fail) ),
           assertion(\+ contains_icat(X))).

% A4: index_catalog/2 (when present) is det and leaves no choice point.
test(a4_index_catalog_deterministic) :-
    (   current_predicate(resolver:index_catalog/2)
    ->  forall(probe_catalog(C),
               ( findall(x, resolver:index_catalog(C, _), L),
                 assertion(L == [x]),
                 resolver:index_catalog(C, _),
                 deterministic(Det),
                 assertion(Det == true) ))
    ;   true
    ).

% A5: sort/2 with position tags is order-preserving within a key (the
% position-tagged form has no keysort-stability assumption; assert the
% property the design relies on anyway).
test(a5_positional_sort_is_stable_by_construction) :-
    sort([(b-1)-0-x, (a-1)-1-y, (b-1)-2-z, (a-1)-3-w], S),
    assertion(S == [(a-1)-1-y, (a-1)-3-w, (b-1)-0-x, (b-1)-2-z]).

% A6: standard order on Name-Ver keys, v/3 and deb/3 including s([],0)
% padding and ~ segments.
test(a6_standard_order_on_keys) :-
    D0   = deb(0, [s([],1)], []),
    Drc  = deb(0, [s([],1), s([126,114,99],1)], []),
    sort([b-D0, a-Drc, a-D0], S),
    assertion(S == [a-D0, a-Drc, b-D0]),
    sort([foo-v(2,0,0), foo-v(1,0,0), bar-v(1,0,0)], S2),
    assertion(S2 == [bar-v(1,0,0), foo-v(1,0,0), foo-v(2,0,0)]).

% A7 (G2): no_acc_conflicts/4 sees Pkg and Ver ground at the moved-up site.
% pick_need/8 binds both on every arm; provider_candidate/5 checks
% package_in/3 so a provides row with no package row cannot leak a var.
test(a7_pick_need_always_binds_pkg_and_ver) :-
    v1(V1),
    Ps = [package(app,V1), package(bar,V1)],
    Ds = [depends(app,V1,foo,any)],
    Cat = catalog(Ps, Ds, [], [], [], [], [], [], [],
                  [provides(bar,V1,foo), provides(ghost,V1,foo)]),
    forall(resolver:pick_need(classic, Cat, foo, any, [], P, V, _),
           ( assertion(ground(P)), assertion(ground(V)) )),
    findall(P-V, resolver:pick_need(classic, Cat, foo, any, [], P, V, _), L),
    assertion(L == [bar-V1]).

% A9/A10/A11 (G4): the no_doomed_req/3 hypotheses. Only meaningful when G4
% landed; the CE4 catalog is the pin -- a G4 without provides_mentions/2
% fails this.
test(a9_a10_held_name_satisfied_by_provider_still_resolves) :-
    ce4_catalog(Cat), v1(V1),
    resolve_layered(Cat, [app], S),
    assertion(S == [app-V1, bar-V1]).

% A10 variant: a versioned virtual equal to a held name, satisfied through
% already_provided/4 rather than through pick_need/8.
test(a10_versioned_virtual_equal_to_held_name) :-
    v1(V1), v2(V2),
    Ps = [package(app,V1), package(bar,V1), package(foo,V2)],
    Ds = [depends(app,V1,bar,any), depends(app,V1,foo,gte(V2))],
    Cat = catalog(Ps, Ds, [], [foo-V1], [], [], [], [], [],
                  [provides(bar,V1,foo,V2)]),
    resolve_layered(Cat, [app], S),
    assertion(S == [app-V1, bar-V1]).

% A11: nothing in the search mutates the base -- base_ver/3 answers the
% same before and after a resolve.
test(a11_base_is_immutable_across_resolve) :-
    ce4_catalog(Cat), v1(V1),
    resolver:base_ver(Cat, foo, B0),
    resolve_layered(Cat, [app], _),
    resolver:base_ver(Cat, foo, B1),
    assertion(B0 == V1), assertion(B1 == V1).

% A9 third pairing (flagged in §6): the SAME atom used as a real package
% name, a held name, and a virtual name at once. Two shapes, both pinned as
% the resolver behaves today.
%
%   (i) provider NOT yet in Acc: pick_need/8's base_ver arm commits and the
%       ceiling fails -- the real package/2 rows for foo-2.0 and the
%       provides row are both unreachable. The query FAILS. This is the
%       case G4's no_doomed_req/3 would fire on, and it is sound to fire:
%       the branch has no solution either way.
%  (ii) provider already in Acc: already_provided/4 runs before pick_need/8
%       and the query succeeds. G4 must NOT fire here -- and does not,
%       because provides_mentions/2 sees the `foo` virtual row.
test(a9_pkg_held_virtual_provider_not_selected_fails) :-
    v1(V1), v2(V2),
    Ps = [package(app,V1), package(foo,V1), package(foo,V2), package(bar,V1)],
    Ds = [depends(app,V1,foo,gte(V2))],
    Cat = catalog(Ps, Ds, [], [foo-V1], [], [], [], [], [],
                  [provides(bar,V1,foo,V2)]),
    assertion(\+ resolve_layered(Cat, [app], _)).

test(a9_pkg_held_virtual_provider_selected_succeeds) :-
    v1(V1), v2(V2),
    Ps = [package(app,V1), package(foo,V1), package(foo,V2), package(bar,V1)],
    Ds = [depends(app,V1,bar,any), depends(app,V1,foo,gte(V2))],
    Cat = catalog(Ps, Ds, [], [foo-V1], [], [], [], [], [],
                  [provides(bar,V1,foo,V2)]),
    resolve_layered(Cat, [app], S),
    assertion(S == [app-V1, bar-V1]).

% A1 (exhaustive form): on EVERY corpus catalog, for every depends-row head
% and every package name, the indexed collect_deps/4 returns exactly the
% list the linear scan returns -- same elements, same order, same
% multiplicity. This is the assertion the whole G1 entailment rests on.
test(a1_index_equals_scan_on_every_corpus_catalog) :-
    forall(probe_catalog(Cat),
           ( forced_index(Cat, ICat),
             forall(dep_head(Cat, N, V),
                    ( resolver:collect_deps(Cat, N, V, A),
                      resolver:collect_deps(ICat, N, V, B),
                      assertion(A == B) )),
             forall(pkg_name(Cat, PN),
                    ( resolver:collect_deps(Cat, PN, v(9,9,9), A2),
                      resolver:collect_deps(ICat, PN, v(9,9,9), B2),
                      assertion(A2 == B2) )),
             resolver:collect_deps(Cat, no_such_name_xyz, v(0,0,0), A3),
             resolver:collect_deps(ICat, no_such_name_xyz, v(0,0,0), B3),
             assertion(A3 == []), assertion(B3 == []) )).

% A2 (exhaustive form): the indexed candidate enumeration is the same
% sequence, including choice points and duplicates, on every corpus
% catalog, for every package name crossed with every constraint that
% appears in that catalog.
test(a2_candidates_identical_on_every_corpus_catalog) :-
    forall(probe_catalog(Cat),
           ( forced_index(Cat, ICat),
             forall(( pkg_name(Cat, N), constraint_in(Cat, C) ),
                    ( findall(X, resolver:candidates_high_first(Cat, N, C, X), L1),
                      findall(X, resolver:candidates_high_first(ICat, N, C, X), L2),
                      assertion(L1 == L2) )) )).

% A3 (exhaustive form): every accessor answers identically through the
% wrapper, so no clause of the search can see a different catalog.
test(a3_every_accessor_delegates) :-
    forall(probe_catalog(Cat),
           ( forced_index(Cat, ICat),
             forall(member(Acc, [packages, depends_list, conflicts_list,
                                 base_list, installed_list, requested_list,
                                 layers_list, excluded_list, alias_list,
                                 provides_list]),
                    ( G1 =.. [Acc, Cat, X1], G2 =.. [Acc, ICat, X2],
                      resolver:G1, resolver:G2,
                      assertion(X1 == X2) )) )).

% A3 (escape form): the top-level answers of both worlds are identical and
% carry no icat/3 anywhere.
test(a3_resolve_agrees_and_icat_never_escapes) :-
    forall(probe_catalog(Cat),
           forall(pkg_name(Cat, N),
                  ( forced_index(Cat, ICat),
                    ( resolver:resolve_pending(classic, Cat, [req(N,any)], [], A)
                    ->  RA = sel(A) ; RA = fail ),
                    ( resolver:resolve_pending(classic, ICat, [req(N,any)], [], B)
                    ->  RB = sel(B) ; RB = fail ),
                    assertion(RA == RB),
                    assertion(\+ contains_icat(RB)) ))).

:- end_tests(pruning_probes).

probe_catalog(Cat) :- a1_catalog(Cat).
probe_catalog(Cat) :- ce4_catalog(Cat).
probe_catalog(catalog([], [], [], [], [], [])).
probe_catalog(Cat) :- scenario_catalog(_, Cat).

% Build the icat/3 wrapper unconditionally, ignoring the size threshold, so
% the index path is exercised on catalogs of every size. Mirrors
% resolver:index_catalog/2's indexing branch exactly.
forced_index(Cat, icat(Cat, DepT, PkgT)) :-
    resolver:depends_list(Cat, Ds),
    resolver:key_dep_rows(Ds, 0, KDs),
    sort(KDs, SDs),
    resolver:group_keyed(SDs, GDs),
    resolver:list_to_tree(GDs, DepT),
    resolver:packages(Cat, Ps),
    resolver:key_pkg_rows(Ps, 0, KPs),
    sort(KPs, SPs),
    resolver:group_keyed(SPs, GPs),
    resolver:list_to_tree(GPs, PkgT).

% Every (Name, Ver) that appears as a depends-row head in Cat, plus a name
% that appears in no row at all.
dep_head(Cat, N, V) :-
    resolver:depends_list(Cat, Ds),
    member(depends(N, V, _, _), Ds).

pkg_name(Cat, N) :-
    resolver:packages(Cat, Ps),
    member(package(N, _), Ps).

constraint_in(Cat, C) :-
    resolver:depends_list(Cat, Ds),
    member(depends(_, _, _, C), Ds),
    C \= alternatives(_).
constraint_in(_Cat, any).

contains_icat(T) :- compound(T), functor(T, icat, 3), !.
contains_icat(T) :- compound(T), arg(_, T, A), contains_icat(A), !.
