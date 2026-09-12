:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% resolver_store_snapshot.pl -- snapshot-aware, memory-deduped P2 store adapter.
%
% A mechanical port of the frozen resolver_store.pl control flow onto a split
% schema:
%
%   store_pkg      : SnapId|Name    -> Ver                      (per-snapshot; membership)
%   pool_dep       : PoolId|Name    -> Ver#Dep#Constraint       (pooled)
%   pool_conflict  : PoolId|Name    -> Ver#Other                (pooled)
%   pool_provides  : PoolId|Virtual -> Pkg#Ver#VirtualVer       (pooled)
%   pool_revdep    : PoolId|Name    -> Dependent#Ver#Constraint (pooled)
%
% The machine-local environment carries TWO ids (one for membership, one for
% the pool) instead of resolver_store.pl's single CatId:
%
%   env_snap(SnapId, PoolId, Base, Installed, Requested, Layers, Excluded, Aliases)
%
% store_pkg lookups seek by SnapId|Name (store_key_snap/3); pool_* lookups seek
% by PoolId|Name (store_key_pool/3). Every predicate below is a *_snap copy of
% the corresponding *_store predicate in resolver_store.pl with those two
% substitutions applied; dependents_snap/3 (and its transitive users) add a
% membership guard on each pooled revdep hit -- the one place the pooled schema
% needs new logic. At N=1 with PoolId == SnapId == CatId the two schemas are
% isomorphic, which is how the differential harness validates this port.
%
% Packing helpers are REUSED from resolver_store (exported); the small
% unexported wire decoders (pack_conflict/3, unpack_conflict/3, unpack_dep/4,
% unpack_rev/4, unpack_provide/4 and their leaf helpers) are COPIED verbatim --
% the wire format is frozen by the shared JS codec contract.

:- module(resolver_store_snapshot, [
    resolve_snap/3,
    resolve_layered_snap/3,
    explain_blocked_snap/3,
    explain_blocked_list_snap/3,
    layer_closure_snap/3,
    removal_orphans_snap/3,
    safe_upgrade_snap/4,
    upgrade_set_snap/4,
    upgrade_set_result_snap/4,
    freeze_audit_snap/2,
    dependents_snap/3,
    dependents_installed_snap/3,
    store_clear_snap/0,
    load_p2_snap/2
]).

:- use_module(resolver, [satisfies/2, version_lt/2]).
:- use_module(resolver_store, [
    pack_ver/2,
    unpack_ver/2,
    pack_constraint/2,
    unpack_constraint/2,
    pack_key/3,
    pack_dep/4,
    pack_rev/4,
    pack_provide/4
]).
:- use_module(library(http/json)).

:- dynamic store_pkg/2.
:- dynamic pool_dep/2.
:- dynamic pool_conflict/2.
:- dynamic pool_revdep/2.
:- dynamic pool_provides/2.

% ---------------------------------------------------------------------------
% Copied wire decoders (unexported in resolver_store; frozen by JS codec)
% ---------------------------------------------------------------------------

unpack_dep(Packed0, Ver, Dep, C) :-
    to_atom(Packed0, Packed),
    split_string(Packed, '#', '', Parts),
    (   Parts = [VA, DepS, CA],
        atom_string(DepA, DepS),
        atom_concat('ALTS/', Body, DepA)
    ->          unpack_ver(VA, Ver),
        unpack_constraint(CA, C),
        split_string(Body, '+', '', Cells),
        unpack_alt_cells(Cells, Alts),
        Dep = alternatives(Alts)
    ;   Parts = [VA, DepS, CA],
        unpack_ver(VA, Ver),
        to_atom(DepS, Dep),
        unpack_constraint(CA, C)
    ).

unpack_alt_cells([], []).
unpack_alt_cells([Cell|Rest], [Alt|Alts]) :-
    unpack_alt_cell(Cell, Alt),
    unpack_alt_cells(Rest, Alts).

unpack_alt_cell(Cell, dep(D, C)) :-
    split_string(Cell, '=', '', [DS, CS]),
    to_atom(DS, D),
    unpack_constraint(CS, C).

pack_conflict(Ver, Other, Atom) :-
    pack_ver(Ver, VA),
    atom_concat(VA, '#', T),
    atom_concat(T, Other, Atom).

unpack_conflict(Packed0, Ver, Other) :-
    to_atom(Packed0, Packed),
    split_string(Packed, '#', '', [VA, OtherS]),
    unpack_ver(VA, Ver),
    to_atom(OtherS, Other).

unpack_rev(Packed0, Name, Ver, C) :-
    to_atom(Packed0, Packed),
    split_string(Packed, '#', '', [NameS, VA, CA]),
    to_atom(NameS, Name),
    unpack_ver(VA, Ver),
    unpack_constraint(CA, C).

unpack_provide(Packed0, P, V, VVer) :-
    to_atom(Packed0, Packed),
    split_string(Packed, '#', '', [PS, VAS, VVS]),
    to_atom(PS, P),
    unpack_ver(VAS, V),
    (   VVS == "-"
    ->  VVer = unversioned
    ;   unpack_ver(VVS, VVer)
    ).

provide_satisfies_snap(unversioned, any).
provide_satisfies_snap(VV, C) :-
    VV \== unversioned,
    satisfies(VV, C).

to_atom(S, A) :- atom(S), !, A = S.
to_atom(S, A) :- atom_string(A, S).

tight_constraint(C) :-
    C \== any.

% ---------------------------------------------------------------------------
% Env accessors (8-ary; two ids)
% ---------------------------------------------------------------------------

env_snapid(env_snap(S, _, _, _, _, _, _, _), S).
env_poolid(env_snap(_, P, _, _, _, _, _, _), P).
env_base(env_snap(_, _, B, _, _, _, _, _), B).
env_installed(env_snap(_, _, _, I, _, _, _, _), I).
env_requested(env_snap(_, _, _, _, R, _, _, _), R).
env_layers(env_snap(_, _, _, _, _, L, _, _), L).
env_excluded(env_snap(_, _, _, _, _, _, E, _), E).
env_aliases(env_snap(_, _, _, _, _, _, _, A), A).

base_ver_env(Env, Name, Ver) :-
    env_base(Env, Bs),
    env_layers(Env, Ls),
    append(Bs, Ls, All),
    lookup_held(All, Name, Ver).

base_name_env(Env, Name) :-
    base_ver_env(Env, Name, _).

installed_ver_env(Env, Name, Ver) :-
    env_installed(Env, Is),
    member(Name-Ver, Is).

excluded_name_env(Env, Name) :-
    env_excluded(Env, Es),
    member(Name, Es).

lookup_held([H|T], Name, Ver) :-
    (   item_ver(H, Name, V0)
    ->  Ver = V0
    ;   lookup_held(T, Name, Ver)
    ).

item_ver(N-V, Name, V) :-
    N == Name.
item_ver(base(N-V, _R), Name, V) :-
    N == Name.
item_ver(layer(_L, Pkgs), Name, V) :-
    lookup_held(Pkgs, Name, V).

canonicalize_name_env(Env, In, Out) :-
    env_aliases(Env, As),
    alias_lookup(As, In, Out).

alias_lookup([], N, N).
alias_lookup([alias(A, Canon)|Rest], In, Out) :-
    (   In == A
    ->  Out = Canon
    ;   alias_lookup(Rest, In, Out)
    ).

request_to_req_env(Env, R, req(Name, C)) :-
    (   R = req(Raw, C)
    ->  canonicalize_name_env(Env, Raw, Name)
    ;   canonicalize_name_env(Env, R, Name),
        C = any
    ).

map_requests_env(_Env, [], []).
map_requests_env(Env, [R|Rs], [Q|Qs]) :-
    request_to_req_env(Env, R, Q),
    map_requests_env(Env, Rs, Qs).

selected_ver([H|Rest], Name, Ver) :-
    (   H = Name-Ver
    ->  true
    ;   selected_ver(Rest, Name, Ver)
    ).

% ---------------------------------------------------------------------------
% Bound store lookups: SnapId|Name for store_pkg, PoolId|Name for pool_*
% ---------------------------------------------------------------------------

store_key_snap(Env, Name, Key) :-
    env_snapid(Env, Id),
    pack_key(Id, Name, Key).

store_key_pool(Env, Name, Key) :-
    env_poolid(Env, Id),
    pack_key(Id, Name, Key).

package_in_store_snap(Env, Name, Ver) :-
    store_key_snap(Env, Name, Key),
    store_pkg(Key, Packed),
    unpack_ver(Packed, Ver).

candidates_high_first_snap(Env, Name, C, Ver) :-
    candidate_versions_snap(Env, Name, C, Desc),
    member(Ver, Desc).

candidate_versions_snap(Env, Name, C, Desc) :-
    (   excluded_name_env(Env, Name)
    ->  Desc = []
    ;   store_key_snap(Env, Name, Key),
        findall(V, (
            store_pkg(Key, Packed),
            unpack_ver(Packed, V),
            satisfies(V, C)
        ), Vs),
        sort_versions_desc_snap(Vs, Desc)
    ).

sort_versions_desc_snap(Vs, Desc) :-
    (   maplist(is_v3_snap, Vs)
    ->  sort(Vs, Asc), reverse(Asc, Desc)
    ;   predsort(cmp_ver_snap, Vs, Asc), reverse(Asc, Desc)
    ).
is_v3_snap(v(_, _, _)).
cmp_ver_snap(<, A, B) :- version_lt(A, B), !.
cmp_ver_snap(>, A, B) :- version_lt(B, A), !.
cmp_ver_snap(=, _, _).

collect_deps_snap(Env, Name, Ver, Reqs) :-
    store_key_pool(Env, Name, Key),
    findall(Req, (
        pool_dep(Key, Packed),
        unpack_dep(Packed, Ver0, Dep, C),
        Ver0 == Ver,
        dep_to_req_snap(Dep, C, Req)
    ), Reqs).

dep_to_req_snap(alternatives(Alts), _C, req(alternatives(Alts), any)) :- !.
dep_to_req_snap(D, C, req(D, C)).

conflicts_in_snap(Env, Name, Ver, Other) :-
    store_key_pool(Env, Name, Key),
    pool_conflict(Key, Packed),
    unpack_conflict(Packed, Ver0, Other),
    Ver0 == Ver.

no_acc_conflicts_snap(_Env, _Name, _Ver, []).
no_acc_conflicts_snap(Env, Name, Ver, [Other-OtherVer|Rest]) :-
    \+ (conflicts_in_snap(Env, Name, Ver, Other)),
    \+ (conflicts_in_snap(Env, Other, OtherVer, Name)),
    no_acc_conflicts_snap(Env, Name, Ver, Rest).

% ---------------------------------------------------------------------------
% Load pool JSONL into pool_*, membership pkg JSONL into store_pkg
% ---------------------------------------------------------------------------

store_clear_snap :-
    retractall(store_pkg(_, _)),
    retractall(pool_dep(_, _)),
    retractall(pool_conflict(_, _)),
    retractall(pool_revdep(_, _)),
    retractall(pool_provides(_, _)).

% PoolDir supplies dep/conflict/revdep/provide.jsonl; PkgDir supplies pkg.jsonl.
% At N=1 both are the same store directory.
load_p2_snap(PoolDir, PkgDir) :-
    store_clear_snap,
    atom_concat(PkgDir, '/pkg.jsonl', PkgF),
    atom_concat(PoolDir, '/dep.jsonl', DepF),
    atom_concat(PoolDir, '/conflict.jsonl', ConfF),
    atom_concat(PoolDir, '/revdep.jsonl', RevF),
    atom_concat(PoolDir, '/provide.jsonl', PrF),
    load_pairs(PkgF, store_pkg),
    load_pairs(DepF, pool_dep),
    load_pairs(ConfF, pool_conflict),
    load_pairs(RevF, pool_revdep),
    load_pairs(PrF, pool_provides).

load_pairs(Path, Pred) :-
    exists_file(Path),
    !,
    setup_call_cleanup(open(Path, read, S), load_pair_lines(S, Pred), close(S)).
load_pairs(_Path, _Pred).

load_pair_lines(S, Pred) :-
    read_line_to_string(S, Line),
    (   Line == end_of_file
    ->  true
    ;   (   Line == ""
        ->  true
        ;   atom_string(Atom, Line),
            atom_json_term(Atom, Term, [value_string_as(atom)]),
            Term = [K, V],
            Fact =.. [Pred, K, V],
            assertz(Fact)
        ),
        load_pair_lines(S, Pred)
    ).

% ---------------------------------------------------------------------------
% Queries (mirror of resolver_store; lookups switched to snap/pool)
% ---------------------------------------------------------------------------

resolve_snap(Env, Requests, Selection) :-
    map_requests_env(Env, Requests, Pending),
    resolve_pending_snap(classic, Env, Pending, [], Acc),
    !,
    sort(Acc, Selection).

resolve_layered_snap(Env, Requests, Selection) :-
    map_requests_env(Env, Requests, Pending),
    resolve_pending_snap(layered, Env, Pending, [], Acc),
    !,
    sort(Acc, Selection).

resolve_pending_snap(Mode, Env, Pending, Acc, Sel) :-
    resolve_pending_snap(Mode, Env, Pending, Acc, st(0, []), Sel).

resolve_pending_snap(_Mode, _Env, [], Acc, _St, Acc).
resolve_pending_snap(Mode, Env, [Item|Rest], Acc, St, Sel) :-
    (   Item = done(Pkg, Ver, Gen)
    ->  St = st(GenNow, [a(Pkg, Ver, Gen)|Active1]),
        resolve_pending_snap(Mode, Env, Rest, Acc, st(GenNow, Active1), Sel)
    ;   Item = req(Name, C),
        (   Name = alternatives(Alts)
        ->  resolve_alternatives_snap(Mode, Env, Alts, Rest, Acc, St, Sel)
        ;   selected_ver(Acc, Name, Ver)
        ->  satisfies(Ver, C),
            resolve_pending_snap(Mode, Env, Rest, Acc, St, Sel)
        ;   already_provided_snap(Env, Acc, Name, C)
        ->  resolve_pending_snap(Mode, Env, Rest, Acc, St, Sel)
        ;   pick_need_snap(Mode, Env, Name, C, Pkg, Ver, Origin),
            (   Origin = from_base
            ->  St = st(Gen, Active),
                (   active_member(Active, Pkg, Ver, Gen)
                ->  resolve_pending_snap(Mode, Env, Rest, Acc, St, Sel)
                ;   collect_deps_snap(Env, Pkg, Ver, DepReqs),
                    append(DepReqs, [done(Pkg, Ver, Gen)|Rest], More),
                    resolve_pending_snap(Mode, Env, More, Acc,
                                          st(Gen, [a(Pkg, Ver, Gen)|Active]), Sel)
                )
            ;   no_acc_conflicts_snap(Env, Pkg, Ver, Acc),
                collect_deps_snap(Env, Pkg, Ver, DepReqs),
                append(DepReqs, Rest, More),
                St = st(Gen, Active),
                Gen1 is Gen + 1,
                resolve_pending_snap(Mode, Env, More, [Pkg-Ver|Acc],
                                      st(Gen1, Active), Sel)
            )
        )
    ).

active_member([a(P, V, G)|Rest], Pkg, Ver, Gen) :-
    (   G < Gen
    ->  fail
    ;   P == Pkg, V == Ver
    ->  true
    ;   active_member(Rest, Pkg, Ver, Gen)
    ).

resolve_alternatives_snap(Mode, Env, Alts, Rest, Acc, St, Sel) :-
    (   first_alt_already_snap(Mode, Env, Acc, Alts)
    ->  resolve_pending_snap(Mode, Env, Rest, Acc, St, Sel)
    ;   member(dep(N, C), Alts),
        resolve_pending_snap(Mode, Env, [req(N, C)|Rest], Acc, St, Sel)
    ).

first_alt_already_snap(_Mode, Env, Acc, Alts) :-
    member(dep(N, C), Alts),
    already_satisfied_snap(Env, Acc, N, C),
    !.
first_alt_already_snap(layered, Env, _Acc, Alts) :-
    member(dep(N, C), Alts),
    layer_satisfies_snap(Env, N, C),
    !.

already_satisfied_snap(_Env, Acc, Name, C) :-
    selected_ver(Acc, Name, Ver),
    satisfies(Ver, C).
already_satisfied_snap(Env, Acc, Name, C) :-
    already_provided_snap(Env, Acc, Name, C).

already_provided_snap(Env, Acc, Name, C) :-
    member(P-PV, Acc),
    provides_sat_snap(Env, P, PV, Name, C).

layer_satisfies_snap(Env, Name, C) :-
    base_ver_env(Env, Name, BV),
    satisfies(BV, C).
layer_satisfies_snap(Env, Name, C) :-
    base_holds_env(Env, Holds),
    member(hold(P, PV, _), Holds),
    provides_sat_snap(Env, P, PV, Name, C).
layer_satisfies_snap(Env, Name, C) :-
    env_layers(Env, Ls),
    member(layer(_, Pkgs), Ls),
    lookup_held(Pkgs, P, PV),
    (   P == Name, satisfies(PV, C)
    ->  true
    ;   provides_sat_snap(Env, P, PV, Name, C)
    ).

provides_sat_snap(Env, Pkg, Ver, Virtual, C) :-
    store_key_pool(Env, Virtual, Key),
    pool_provides(Key, Packed),
    unpack_provide(Packed, P0, V0, VVer),
    P0 == Pkg,
    V0 == Ver,
    provide_satisfies_snap(VVer, C).

provider_candidate_snap(Env, Virtual, C, Pkg, Ver) :-
    store_key_pool(Env, Virtual, Key),
    pool_provides(Key, Packed),
    unpack_provide(Packed, Pkg, Ver, VVer),
    \+ excluded_name_env(Env, Pkg),
    provide_satisfies_snap(VVer, C),
    package_in_store_snap(Env, Pkg, Ver).

layer_provider_snap(Env, Virtual, C, Pkg, Ver) :-
    base_holds_env(Env, Holds),
    member(hold(Pkg, Ver, _), Holds),
    provides_sat_snap(Env, Pkg, Ver, Virtual, C).
layer_provider_snap(Env, Virtual, C, Pkg, Ver) :-
    env_layers(Env, Ls),
    member(layer(_, Pkgs), Ls),
    lookup_held(Pkgs, Pkg, Ver),
    provides_sat_snap(Env, Pkg, Ver, Virtual, C).

pick_need_snap(classic, Env, Name, C, Name, Ver, from_catalog) :-
    candidates_high_first_snap(Env, Name, C, Ver).
pick_need_snap(classic, Env, Name, C, Pkg, Ver, from_catalog) :-
    provider_candidate_snap(Env, Name, C, Pkg, Ver).
pick_need_snap(layered, Env, Name, C, Pkg, Ver, Origin) :-
    (   base_ver_env(Env, Name, BV)
    ->  satisfies(BV, C),
        Pkg = Name,
        Ver = BV,
        Origin = from_base
    ;   layer_provider_snap(Env, Name, C, Pkg, Ver)
    ->  Origin = from_base
    ;   candidate_versions_snap(Env, Name, C, Desc),
        Desc = [_|_]
    ->  member(Ver, Desc),
        Pkg = Name,
        Origin = from_catalog
    ;   provider_candidate_snap(Env, Name, C, Pkg, Ver),
        Origin = from_catalog
    ).

pick_snap(classic, Env, Name, C, Ver, from_catalog) :-
    candidates_high_first_snap(Env, Name, C, Ver).
pick_snap(layered, Env, Name, C, Ver, Origin) :-
    (   base_ver_env(Env, Name, BV)
    ->  satisfies(BV, C),
        Ver = BV,
        Origin = from_base
    ;   candidates_high_first_snap(Env, Name, C, Ver),
        Origin = from_catalog
    ).

explain_blocked_snap(Env, Request, Blocked) :-
    request_to_req_env(Env, Request, Req),
    blocked_from_snap(Env, Req, [], Blocked).

explain_blocked_list_snap(Env, Request, List) :-
    request_to_req_env(Env, Request, Req),
    blocked_acc_snap(Env, Req, [], [], Acc),
    sort(Acc, List),
    !.

blocked_from_snap(Env, req(alternatives(Alts), _), Seen, Blocked) :-
    !,
    alt_reasons_snap(Env, Alts, Seen, Rs),
    Blocked = blocked(alternatives(Rs)).
blocked_from_snap(Env, req(Name, C), _Seen, Blocked) :-
    base_ver_env(Env, Name, BV),
    \+ satisfies(BV, C),
    Blocked = blocked(Name, needs(C), base_has(BV)).
blocked_from_snap(Env, req(Name, C), _Seen, Blocked) :-
    virtual_provider_ceilings_snap(Env, Name, C, Reasons),
    Reasons \== [],
    Blocked = blocked(Name, needs(C), providers(Reasons)).
blocked_from_snap(Env, req(Name, C), Seen, Blocked) :-
    \+ seen_name(Seen, Name),
    walk_pkg_for_blocked_snap(Env, Name, C, Pkg, Ver),
    collect_deps_snap(Env, Pkg, Ver, DepReqs),
    member(Dep, DepReqs),
    blocked_from_snap(Env, Dep, [Name|Seen], Blocked).

blocked_acc_snap(Env, req(alternatives(Alts), _), Seen, Acc0, Acc) :-
    !,
    alt_reasons_snap(Env, Alts, Seen, Rs),
    Acc = [blocked(alternatives(Rs))|Acc0].
blocked_acc_snap(Env, req(Name, C), Seen, Acc0, Acc) :-
    (   base_ver_env(Env, Name, BV),
        \+ satisfies(BV, C)
    ->  Acc1 = [blocked(Name, needs(C), base_has(BV))|Acc0]
    ;   virtual_provider_ceilings_snap(Env, Name, C, Reasons),
        Reasons \== []
    ->  Acc1 = [blocked(Name, needs(C), providers(Reasons))|Acc0]
    ;   Acc1 = Acc0
    ),
    (   seen_name(Seen, Name)
    ->  Acc = Acc1
    ;   walk_pkg_for_blocked_snap(Env, Name, C, Pkg, Ver)
    ->  collect_deps_snap(Env, Pkg, Ver, DepReqs),
        blocked_acc_list_snap(Env, DepReqs, [Name|Seen], Acc1, Acc)
    ;   Acc = Acc1
    ).

walk_pkg_for_blocked_snap(Env, Name, C, Name, Ver) :-
    layered_walk_ver_snap(Env, Name, C, Ver).
walk_pkg_for_blocked_snap(Env, Name, C, Pkg, Ver) :-
    layer_provider_snap(Env, Name, C, Pkg, Ver).
walk_pkg_for_blocked_snap(Env, Name, C, Pkg, Ver) :-
    provider_candidate_snap(Env, Name, C, Pkg, Ver).

virtual_provider_ceilings_snap(Env, Virtual, C, Reasons) :-
    \+ package_in_name_snap(Env, Virtual),
    findall(blocked(P, needs(C), base_has(BV)), (
        store_key_pool(Env, Virtual, Key),
        pool_provides(Key, Packed),
        unpack_provide(Packed, P, BV, VVer),
        base_ver_env(Env, P, BV),
        \+ provide_satisfies_snap(VVer, C)
    ), Reasons).

package_in_name_snap(Env, Name) :-
    store_key_snap(Env, Name, Key),
    store_pkg(Key, _).

alt_reasons_snap(_Env, [], _Seen, []).
alt_reasons_snap(Env, [dep(N, C)|Rest], Seen, [alt(N, Reason)|Rs]) :-
    (   explain_alt_snap(Env, N, C, Seen, Reason)
    ->  true
    ;   Reason = unsatisfiable
    ),
    alt_reasons_snap(Env, Rest, Seen, Rs).

explain_alt_snap(Env, N, C, Seen, Reason) :-
    (   blocked_from_snap(Env, req(N, C), Seen, Reason)
    ->  true
    ;   \+ pick_need_snap(layered, Env, N, C, _, _, _)
    ->  Reason = unsatisfiable
    ).

blocked_acc_list_snap(_Env, [], _Seen, Acc, Acc).
blocked_acc_list_snap(Env, [Dep|Rest], Seen, Acc0, Acc) :-
    blocked_acc_snap(Env, Dep, Seen, Acc0, Acc1),
    blocked_acc_list_snap(Env, Rest, Seen, Acc1, Acc).

seen_name([H|Rest], Name) :-
    (   H == Name
    ->  true
    ;   seen_name(Rest, Name)
    ).

layered_walk_ver_snap(Env, Name, C, Ver) :-
    (   base_ver_env(Env, Name, BV)
    ->  satisfies(BV, C),
        Ver = BV
    ;   candidates_high_first_snap(Env, Name, C, Ver)
    ),
    !.

layer_closure_snap(Env, Request, Layer) :-
    resolve_layered_snap(Env, [Request], Sel),
    topo_sort_sel_snap(Env, Sel, Layer),
    !.

topo_sort_sel_snap(_Env, [], []) :- !.
topo_sort_sel_snap(Env, Sel, Layer) :-
    sort(Sel, Sorted),
    names_of(Sorted, Names),
    topo_all_snap(Env, Names, Sel, [], _Seen, [], Acc),
    reverse(Acc, Layer).

names_of([], []).
names_of([N-_|Rest], [N|Ns]) :-
    names_of(Rest, Ns).

topo_all_snap(_Env, [], _Sel, Seen, Seen, Acc, Acc).
topo_all_snap(Env, [N|Ns], Sel, Seen0, Seen, Acc0, Acc) :-
    topo_one_snap(Env, N, Sel, Seen0, Seen1, Acc0, Acc1),
    topo_all_snap(Env, Ns, Sel, Seen1, Seen, Acc1, Acc).

topo_one_snap(_Env, Name, _Sel, Seen, Seen, Acc, Acc) :-
    member(Name, Seen),
    !.
topo_one_snap(Env, Name, Sel, Seen0, Seen, Acc0, Acc) :-
    (   member(Name-Ver, Sel)
    ->  findall(D, (
            collect_deps_snap(Env, Name, Ver, Reqs),
            member(Req, Reqs),
            follow_req_sel_snap(Req, Sel, D)
        ), Ds0),
        sort(Ds0, Ds),
        topo_all_snap(Env, Ds, Sel, [Name|Seen0], Seen1, Acc0, Acc1),
        Acc = [Name-Ver|Acc1],
        Seen = Seen1
    ;   Seen = [Name|Seen0],
        Acc = Acc0
    ).

follow_req_sel_snap(req(alternatives(Alts), _), Sel, D) :-
    member(dep(N, _), Alts),
    member(N-_, Sel),
    D = N.
follow_req_sel_snap(req(D, _), Sel, D) :-
    atom(D),
    member(D-_, Sel).

removal_orphans_snap(Env, Pkg0, Orphans) :-
    canonicalize_name_env(Env, Pkg0, Pkg),
    env_installed(Env, Inst),
    (   installed_ver_env(Env, Pkg, Ver)
    ->  true
    ;   Ver = none
    ),
    (   Ver == none
    ->  Orphans = []
    ;   inst_closure_names_snap(Env, Inst, Pkg, Ver, Closure),
        env_requested(Env, Reqs0),
        exclude_name(Pkg, Reqs0, Reqs1),
        needed_names_snap(Env, Inst, Reqs1, Needed),
        findall(N-V, (
            member(N-V, Inst),
            N \== Pkg,
            member(N, Closure),
            \+ member(N, Needed),
            \+ base_name_env(Env, N)
        ), Or0),
        sort(Or0, Orphans)
    ),
    !.

exclude_name(_N, [], []).
exclude_name(N, [N|Rs], Out) :- !,
    exclude_name(N, Rs, Out).
exclude_name(N, [R|Rs], [R|Out]) :-
    exclude_name(N, Rs, Out).

inst_closure_names_snap(Env, Inst, Name, Ver, Names) :-
    inst_walk_snap([Name-Ver], Env, Inst, [], [], Names).

inst_walk_snap([], _Env, _Inst, _Seen, Acc, Acc).
inst_walk_snap([Name-Ver|Rest], Env, Inst, Seen, Acc0, Acc) :-
    (   member(Name, Seen)
    ->  inst_walk_snap(Rest, Env, Inst, Seen, Acc0, Acc)
    ;   collect_deps_snap(Env, Name, Ver, Reqs),
        findall(D-DV, (
            member(Req, Reqs),
            follow_req_inst_snap(Req, Inst, D, DV)
        ), Kids),
        append(Kids, Rest, More),
        inst_walk_snap(More, Env, Inst, [Name|Seen], [Name|Acc0], Acc)
    ).

follow_req_inst_snap(req(alternatives(Alts), _), Inst, D, DV) :-
    member(dep(N, _), Alts),
    member(N-DV, Inst),
    D = N.
follow_req_inst_snap(req(D, _), Inst, D, DV) :-
    atom(D),
    member(D-DV, Inst).

needed_names_snap(_Env, _Inst, [], []).
needed_names_snap(Env, Inst, Roots, Needed) :-
    roots_to_pairs(Roots, Inst, Pairs),
    inst_walk_snap(Pairs, Env, Inst, [], [], Needed).

roots_to_pairs([], _Inst, []).
roots_to_pairs([N|Ns], Inst, [N-V|Ps]) :-
    member(N-V, Inst),
    !,
    roots_to_pairs(Ns, Inst, Ps).
roots_to_pairs([_|Ns], Inst, Ps) :-
    roots_to_pairs(Ns, Inst, Ps).

base_holds_env(Env, Holds) :-
    env_base(Env, Bs),
    scan_base_holds(Bs, [], Acc),
    sort(Acc, Holds).

scan_base_holds([], Acc, Acc).
scan_base_holds([H|T], Acc0, Acc) :-
    (   H = layer(base, Pkgs)
    ->  scan_base_holds(Pkgs, Acc0, Acc1)
    ;   H = layer(_, _)
    ->  Acc1 = Acc0
    ;   H = base(N-V, R)
    ->  Acc1 = [hold(N, V, R)|Acc0]
    ;   H = N-V
    ->  Acc1 = [hold(N, V, blanket)|Acc0]
    ;   Acc1 = Acc0
    ),
    scan_base_holds(T, Acc1, Acc).

base_reason_env(Env, Name, Reason) :-
    base_holds_env(Env, Holds),
    hold_reason(Holds, Name, Reason).

hold_reason([hold(N, _V, R)|T], Name, Reason) :-
    (   N == Name
    ->  Reason = R
    ;   hold_reason(T, Name, Reason)
    ).

safe_upgrade_snap(Env, Pkg0, NewVer, Verdict) :-
    canonicalize_name_env(Env, Pkg0, Pkg),
    (   \+ package_in_store_snap(Env, Pkg, NewVer)
    ->  Verdict = no_candidate
    ;   \+ base_reason_env(Env, Pkg, _)
    ->  Verdict = no_candidate
    ;   base_reason_env(Env, Pkg, Reason),
        safe_upgrade_reason_snap(Env, Pkg, NewVer, Reason, Verdict)
    ),
    !.

safe_upgrade_reason_snap(_Env, _Pkg, _NewVer, modified, unsafe(modified)).
safe_upgrade_reason_snap(_Env, _Pkg, _NewVer, footprint, safe(cost(footprint))).
safe_upgrade_reason_snap(_Env, _Pkg, _NewVer, blanket, safe(cost(blanket))).
safe_upgrade_reason_snap(_Env, _Pkg, _NewVer, layer_shadow, safe(cost(layer_shadow))).
safe_upgrade_reason_snap(Env, Pkg, NewVer, abi_anchor, coordinated(Set)) :-
    upgrade_set_result_snap(Env, Pkg, NewVer, ok(Set)).

upgrade_set_snap(Env, Pkg, NewVer, Set) :-
    upgrade_set_result_snap(Env, Pkg, NewVer, ok(Set)),
    !.

upgrade_set_result_snap(Env, Pkg0, NewVer, Result) :-
    canonicalize_name_env(Env, Pkg0, Pkg),
    (   package_in_store_snap(Env, Pkg, NewVer)
    ->  close_moving_snap(Env, [Pkg-NewVer], Result)
    ;   Result = no_candidate
    ),
    !.

close_moving_snap(Env, Acc, Result) :-
    base_holds_env(Env, Holds),
    first_broken_snap(Holds, Env, Acc, Broken),
    (   Broken = none
    ->  sort(Acc, Sorted),
        Result = ok(Sorted)
    ;   Broken = broken(N, V, C),
        (   pick_repair_snap(Env, N, Acc, NewV)
        ->  close_moving_snap(Env, [N-NewV|Acc], Result)
        ;   Result = blocked(N, needs(C), base_has(V))
        )
    ).

first_broken_snap([], _Env, _Acc, none).
first_broken_snap([hold(N, V, _R)|Rest], Env, Acc, Broken) :-
    (   selected_ver(Acc, N, _)
    ->  first_broken_snap(Rest, Env, Acc, Broken)
    ;   dep_breaks_moving_snap(Env, N, V, Acc, C)
    ->  Broken = broken(N, V, C)
    ;   first_broken_snap(Rest, Env, Acc, Broken)
    ).

dep_breaks_moving_snap(Env, N, V, Acc, C) :-
    collect_deps_snap(Env, N, V, Reqs),
    member(Req, Reqs),
    dep_breaks_need_snap(Acc, Req, C).

dep_breaks_need_snap(Acc, req(alternatives(Alts), _), COut) :-
    member(dep(D, COut), Alts),
    selected_ver(Acc, D, MV),
    \+ satisfies(MV, COut),
    \+ (member(dep(D2, C2), Alts), selected_ver(Acc, D2, MV2), satisfies(MV2, C2)).
dep_breaks_need_snap(Acc, req(D, C), C) :-
    atom(D),
    selected_ver(Acc, D, MV),
    \+ satisfies(MV, C).

pick_repair_snap(Env, Name, Acc, NewV) :-
    candidates_high_first_snap(Env, Name, any, NewV),
    repairs_moving_snap(Env, Name, NewV, Acc).

repairs_moving_snap(Env, Name, NewV, Acc) :-
    collect_deps_snap(Env, Name, NewV, Reqs),
    reqs_ok_moving(Reqs, Acc).

reqs_ok_moving([], _).
reqs_ok_moving([req(alternatives(Alts), _)|Rest], Acc) :-
    !,
    (   member(dep(D, C), Alts),
        selected_ver(Acc, D, MV)
    ->  satisfies(MV, C)
    ;   true
    ),
    reqs_ok_moving(Rest, Acc).
reqs_ok_moving([req(D, C)|Rest], Acc) :-
    (   selected_ver(Acc, D, MV)
    ->  satisfies(MV, C)
    ;   true
    ),
    reqs_ok_moving(Rest, Acc).

freeze_audit_snap(Env, Audit) :-
    base_holds_env(Env, Holds),
    audit_holds_snap(Holds, Env, [], Acc),
    sort(Acc, Audit),
    !.

audit_holds_snap([], _Env, Acc, Acc).
audit_holds_snap([hold(N, _V, R)|Rest], Env, Acc0, Acc) :-
    (   R == blanket
    ->  (   tight_base_revdep_snap(Env, N)
        ->  Item = audit(N, suggest(abi_anchor))
        ;   Item = audit(N, over_frozen)
        )
    ;   Item = audit(N, held(R))
    ),
    audit_holds_snap(Rest, Env, [Item|Acc0], Acc).

% Pooled revdep hits get a membership guard (package_in_store_snap) so stale
% cross-snapshot dependents cannot leak in. At N=1 the guard is a no-op (every
% revdep-sourced dependent is a published package present in store_pkg).
tight_base_revdep_snap(Env, Pkg) :-
    store_key_pool(Env, Pkg, Key),
    pool_revdep(Key, Packed),
    unpack_rev(Packed, N, V, C),
    N \== Pkg,
    tight_constraint(C),
    package_in_store_snap(Env, N, V),
    base_ver_env(Env, N, BV),
    V == BV.
tight_base_revdep_snap(Env, Pkg) :-
    base_ver_env(Env, Pkg, PV),
    env_poolid(Env, Id),
    atom_concat(Id, '|', Prefix),
    pool_provides(K, Packed),
    atom_concat(Prefix, Virtual, K),
    unpack_provide(Packed, P0, V0, VVer),
    P0 == Pkg,
    V0 == PV,
    store_key_pool(Env, Virtual, VK),
    pool_revdep(VK, RPacked),
    unpack_rev(RPacked, N, V, C),
    N \== Pkg,
    package_in_store_snap(Env, N, V),
    base_ver_env(Env, N, BV),
    V == BV,
    provide_satisfies_snap(VVer, C).

dependents_snap(Env, Pkg0, Deps) :-
    canonicalize_name_env(Env, Pkg0, Pkg),
    store_key_pool(Env, Pkg, Key),
    findall(N-V, (
        pool_revdep(Key, Packed),
        unpack_rev(Packed, N, V, _C),
        package_in_store_snap(Env, N, V)
    ), Acc),
    sort(Acc, Deps),
    !.

dependents_installed_snap(Env, Pkg0, Deps) :-
    dependents_snap(Env, Pkg0, All),
    keep_installed_or_base_env(All, Env, [], Acc),
    sort(Acc, Deps),
    !.

keep_installed_or_base_env([], _Env, Acc, Acc).
keep_installed_or_base_env([N-V|Rest], Env, Acc0, Acc) :-
    (   installed_or_base_env(Env, N, V)
    ->  Acc1 = [N-V|Acc0]
    ;   Acc1 = Acc0
    ),
    keep_installed_or_base_env(Rest, Env, Acc1, Acc).

installed_or_base_env(Env, N, V) :-
    installed_ver_env(Env, N, V).
installed_or_base_env(Env, N, V) :-
    base_ver_env(Env, N, BV),
    V == BV.
