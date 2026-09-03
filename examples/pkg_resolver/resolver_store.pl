:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% resolver_store.pl -- P2 store-backed adapter. The catalog-as-term API in
% resolver.pl is untouched. The big three (packages / depends / conflicts)
% plus a precomputed reverse-deps index are P/2 fact-source predicates
% (D43 indexed(Prefix) on wamjs; the same JSONL P/2 rows on SWI). The
% machine-local environment stays a term:
%
%   env(CatId, Base, Installed, Requested, Layers, Excluded, Aliases)
%
% Store keys are `CatId|Name` so many catalogs can share one index (corpus)
% without scanning. Lookups always bind the key (seek, not scan).
%
% Packed a2 cells (atoms; D43 stores are scalars only):
%   store_pkg      a2 = Major.Minor.Patch
%   store_dep      a2 = Ver#Dep#Constraint
%   store_conflict a2 = Ver#Other
%   store_revdep   a2 = Name#Ver#Constraint   (dependents of the key)

:- module(resolver_store, [
    resolve_store/3,
    resolve_layered_store/3,
    explain_blocked_store/3,
    explain_blocked_list_store/3,
    layer_closure_store/3,
    removal_orphans_store/3,
    safe_upgrade_store/4,
    upgrade_set_store/4,
    upgrade_set_result_store/4,
    freeze_audit_store/2,
    dependents_store/3,
    dependents_installed_store/3,
    env_from_catalog/3,
    store_clear/0,
    assert_catalog_store/2,
    load_p2_jsonl/1,
    pack_ver/2,
    unpack_ver/2,
    pack_constraint/2,
    unpack_constraint/2,
    pack_key/3
]).

:- use_module(resolver, [satisfies/2, version_lt/2]).
:- use_module(library(http/json)).

:- dynamic store_pkg/2.
:- dynamic store_dep/2.
:- dynamic store_conflict/2.
:- dynamic store_revdep/2.

% ---------------------------------------------------------------------------
% Packing (shared with the JS builder — keep in lockstep)
% ---------------------------------------------------------------------------

pack_key(CatId, Name, Key) :-
    atom_concat(CatId, '|', T),
    atom_concat(T, Name, Key).

pack_ver(v(A, B, C), Atom) :-
    number_string(A, SA),
    number_string(B, SB),
    number_string(C, SC),
    atom_concat(SA, '.', T1),
    atom_concat(T1, SB, T2),
    atom_concat(T2, '.', T3),
    atom_concat(T3, SC, Atom).

unpack_ver(Packed0, v(A, B, C)) :-
    to_atom(Packed0, Packed),
    split_string(Packed, '.', '', [SA, SB, SC]),
    number_string(A, SA),
    number_string(B, SB),
    number_string(C, SC).

pack_constraint(any, any).
pack_constraint(eq(V), Atom) :-
    pack_ver(V, VA), atom_concat('eq:', VA, Atom).
pack_constraint(gte(V), Atom) :-
    pack_ver(V, VA), atom_concat('gte:', VA, Atom).
pack_constraint(lt(V), Atom) :-
    pack_ver(V, VA), atom_concat('lt:', VA, Atom).
pack_constraint(range(Lo, Hi), Atom) :-
    pack_ver(Lo, LA), pack_ver(Hi, HA),
    atom_concat('range:', LA, T),
    atom_concat(T, ':', T2),
    atom_concat(T2, HA, Atom).

unpack_constraint(Packed0, C) :-
    to_atom(Packed0, Packed),
    unpack_constraint_atom(Packed, C).

unpack_constraint_atom(any, any) :- !.
unpack_constraint_atom(Packed, eq(V)) :-
    atom_concat('eq:', Rest, Packed), !, unpack_ver(Rest, V).
unpack_constraint_atom(Packed, gte(V)) :-
    atom_concat('gte:', Rest, Packed), !, unpack_ver(Rest, V).
unpack_constraint_atom(Packed, lt(V)) :-
    atom_concat('lt:', Rest, Packed), !, unpack_ver(Rest, V).
unpack_constraint_atom(Packed, range(Lo, Hi)) :-
    atom_concat('range:', Rest, Packed), !,
    split_string(Rest, ':', '', [LA, HA]),
    unpack_ver(LA, Lo), unpack_ver(HA, Hi).

pack_dep(Ver, Dep, C, Atom) :-
    pack_ver(Ver, VA),
    pack_constraint(C, CA),
    atom_concat(VA, '#', T1),
    atom_concat(T1, Dep, T2),
    atom_concat(T2, '#', T3),
    atom_concat(T3, CA, Atom).

unpack_dep(Packed0, Ver, Dep, C) :-
    to_atom(Packed0, Packed),
    split_string(Packed, '#', '', [VA, DepS, CA]),
    unpack_ver(VA, Ver),
    to_atom(DepS, Dep),
    unpack_constraint(CA, C).

pack_conflict(Ver, Other, Atom) :-
    pack_ver(Ver, VA),
    atom_concat(VA, '#', T),
    atom_concat(T, Other, Atom).

unpack_conflict(Packed0, Ver, Other) :-
    to_atom(Packed0, Packed),
    split_string(Packed, '#', '', [VA, OtherS]),
    unpack_ver(VA, Ver),
    to_atom(OtherS, Other).

pack_rev(Name, Ver, C, Atom) :-
    pack_ver(Ver, VA),
    pack_constraint(C, CA),
    atom_concat(Name, '#', T1),
    atom_concat(T1, VA, T2),
    atom_concat(T2, '#', T3),
    atom_concat(T3, CA, Atom).

unpack_rev(Packed0, Name, Ver, C) :-
    to_atom(Packed0, Packed),
    split_string(Packed, '#', '', [NameS, VA, CA]),
    to_atom(NameS, Name),
    unpack_ver(VA, Ver),
    unpack_constraint(CA, C).

to_atom(S, A) :- atom(S), !, A = S.
to_atom(S, A) :- atom_string(A, S).

tight_constraint(C) :-
    C \== any.

% ---------------------------------------------------------------------------
% Env accessors (tiny, term-side)
% ---------------------------------------------------------------------------

env_id(env(Id, _, _, _, _, _, _), Id).
env_base(env(_, B, _, _, _, _, _), B).
env_installed(env(_, _, I, _, _, _, _), I).
env_requested(env(_, _, _, R, _, _, _), R).
env_layers(env(_, _, _, _, L, _, _), L).
env_excluded(env(_, _, _, _, _, E, _), E).
env_aliases(env(_, _, _, _, _, _, A), A).

env_from_catalog(Id, catalog(_Ps, _Ds, _Cs, B, I, R),
                 env(Id, B, I, R, [], [], [])).
env_from_catalog(Id, catalog(_Ps, _Ds, _Cs, B, I, R, L, E, A),
                 env(Id, B, I, R, L, E, A)).

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
% Bound store lookups (seek by CatId|Name)
% ---------------------------------------------------------------------------

store_key(Env, Name, Key) :-
    env_id(Env, Id),
    pack_key(Id, Name, Key).

package_in_store(Env, Name, Ver) :-
    store_key(Env, Name, Key),
    store_pkg(Key, Packed),
    unpack_ver(Packed, Ver).

candidates_high_first_store(Env, Name, C, Ver) :-
    \+ excluded_name_env(Env, Name),
    store_key(Env, Name, Key),
    findall(V, (
        store_pkg(Key, Packed),
        unpack_ver(Packed, V),
        satisfies(V, C)
    ), Vs),
    sort(Vs, Asc),
    reverse(Asc, Desc),
    member(Ver, Desc).

collect_deps_store(Env, Name, Ver, Reqs) :-
    store_key(Env, Name, Key),
    findall(req(Dep, C), (
        store_dep(Key, Packed),
        unpack_dep(Packed, Ver0, Dep, C),
        Ver0 == Ver
    ), Reqs).

conflicts_in_store(Env, Name, Ver, Other) :-
    store_key(Env, Name, Key),
    store_conflict(Key, Packed),
    unpack_conflict(Packed, Ver0, Other),
    Ver0 == Ver.

no_acc_conflicts_store(_Env, _Name, _Ver, []).
no_acc_conflicts_store(Env, Name, Ver, [Other-OtherVer|Rest]) :-
    \+ (conflicts_in_store(Env, Name, Ver, Other)),
    \+ (conflicts_in_store(Env, Other, OtherVer, Name)),
    no_acc_conflicts_store(Env, Name, Ver, Rest).

% ---------------------------------------------------------------------------
% Load a term catalog into the P/2 store (SWI oracle / tests)
% ---------------------------------------------------------------------------

store_clear :-
    retractall(store_pkg(_, _)),
    retractall(store_dep(_, _)),
    retractall(store_conflict(_, _)),
    retractall(store_revdep(_, _)).

assert_catalog_store(Id, catalog(Ps, Ds, Cs, B, I, R)) :-
    assert_catalog_store(Id, catalog(Ps, Ds, Cs, B, I, R, [], [], [])).
assert_catalog_store(Id, catalog(Ps, Ds, Cs, _B, _I, _R, _L, _E, _A)) :-
    forall(member(package(N, V), Ps),
           (   pack_key(Id, N, K), pack_ver(V, VA),
               assertz(store_pkg(K, VA))
           )),
    forall(member(depends(N, V, D, C), Ds),
           (   pack_key(Id, N, KN), pack_dep(V, D, C, DA),
               assertz(store_dep(KN, DA)),
               pack_key(Id, D, KD), pack_rev(N, V, C, RA),
               assertz(store_revdep(KD, RA))
           )),
    forall(member(conflicts(N, V, O), Cs),
           (   pack_key(Id, N, K), pack_conflict(V, O, CA),
               assertz(store_conflict(K, CA))
           )).

load_p2_jsonl(Dir) :-
    store_clear,
    atom_concat(Dir, '/pkg.jsonl', PkgF),
    atom_concat(Dir, '/dep.jsonl', DepF),
    atom_concat(Dir, '/conflict.jsonl', ConfF),
    atom_concat(Dir, '/revdep.jsonl', RevF),
    load_pairs(PkgF, store_pkg),
    load_pairs(DepF, store_dep),
    load_pairs(ConfF, store_conflict),
    load_pairs(RevF, store_revdep).

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
% Queries (same 10 as resolver.pl; store lookups instead of list members)
% ---------------------------------------------------------------------------

resolve_store(Env, Requests, Selection) :-
    map_requests_env(Env, Requests, Pending),
    resolve_pending_store(classic, Env, Pending, [], Acc),
    !,
    sort(Acc, Selection).

resolve_layered_store(Env, Requests, Selection) :-
    map_requests_env(Env, Requests, Pending),
    resolve_pending_store(layered, Env, Pending, [], Acc),
    !,
    sort(Acc, Selection).

resolve_pending_store(_Mode, _Env, [], Acc, Acc).
resolve_pending_store(Mode, Env, [req(Name, C)|Rest], Acc, Sel) :-
    (   selected_ver(Acc, Name, Ver)
    ->  satisfies(Ver, C),
        resolve_pending_store(Mode, Env, Rest, Acc, Sel)
    ;   pick_store(Mode, Env, Name, C, Ver, Origin),
        collect_deps_store(Env, Name, Ver, DepReqs),
        append(DepReqs, Rest, More),
        (   Origin = from_base
        ->  resolve_pending_store(Mode, Env, More, Acc, Sel)
        ;   no_acc_conflicts_store(Env, Name, Ver, Acc),
            resolve_pending_store(Mode, Env, More, [Name-Ver|Acc], Sel)
        )
    ).

pick_store(classic, Env, Name, C, Ver, from_catalog) :-
    candidates_high_first_store(Env, Name, C, Ver).
pick_store(layered, Env, Name, C, Ver, Origin) :-
    (   base_ver_env(Env, Name, BV)
    ->  satisfies(BV, C),
        Ver = BV,
        Origin = from_base
    ;   candidates_high_first_store(Env, Name, C, Ver),
        Origin = from_catalog
    ).

explain_blocked_store(Env, Request, Blocked) :-
    request_to_req_env(Env, Request, Req),
    blocked_from_store(Env, Req, [], Blocked).

explain_blocked_list_store(Env, Request, List) :-
    request_to_req_env(Env, Request, Req),
    blocked_acc_store(Env, Req, [], [], Acc),
    sort(Acc, List),
    !.

blocked_from_store(Env, req(Name, C), Seen, Blocked) :-
    \+ seen_name(Seen, Name),
    base_ver_env(Env, Name, BV),
    \+ satisfies(BV, C),
    Blocked = blocked(Name, needs(C), base_has(BV)).
blocked_from_store(Env, req(Name, C), Seen, Blocked) :-
    \+ seen_name(Seen, Name),
    layered_walk_ver_store(Env, Name, C, Ver),
    collect_deps_store(Env, Name, Ver, DepReqs),
    member(Dep, DepReqs),
    blocked_from_store(Env, Dep, [Name|Seen], Blocked).

blocked_acc_store(_Env, req(Name, _C), Seen, Acc, Acc) :-
    seen_name(Seen, Name), !.
blocked_acc_store(Env, req(Name, C), Seen, Acc0, Acc) :-
    (   base_ver_env(Env, Name, BV),
        \+ satisfies(BV, C)
    ->  Acc1 = [blocked(Name, needs(C), base_has(BV))|Acc0]
    ;   Acc1 = Acc0
    ),
    (   layered_walk_ver_store(Env, Name, C, Ver)
    ->  collect_deps_store(Env, Name, Ver, DepReqs),
        blocked_acc_list_store(Env, DepReqs, [Name|Seen], Acc1, Acc)
    ;   Acc = Acc1
    ).

blocked_acc_list_store(_Env, [], _Seen, Acc, Acc).
blocked_acc_list_store(Env, [Dep|Rest], Seen, Acc0, Acc) :-
    blocked_acc_store(Env, Dep, Seen, Acc0, Acc1),
    blocked_acc_list_store(Env, Rest, Seen, Acc1, Acc).

seen_name([H|Rest], Name) :-
    (   H == Name
    ->  true
    ;   seen_name(Rest, Name)
    ).

layered_walk_ver_store(Env, Name, C, Ver) :-
    (   base_ver_env(Env, Name, BV)
    ->  satisfies(BV, C),
        Ver = BV
    ;   candidates_high_first_store(Env, Name, C, Ver)
    ),
    !.

layer_closure_store(Env, Request, Layer) :-
    resolve_layered_store(Env, [Request], Sel),
    topo_sort_sel_store(Env, Sel, Layer),
    !.

topo_sort_sel_store(_Env, [], []) :- !.
topo_sort_sel_store(Env, Sel, Layer) :-
    sort(Sel, Sorted),
    names_of(Sorted, Names),
    topo_all_store(Env, Names, Sel, [], _Seen, [], Acc),
    reverse(Acc, Layer).

names_of([], []).
names_of([N-_|Rest], [N|Ns]) :-
    names_of(Rest, Ns).

topo_all_store(_Env, [], _Sel, Seen, Seen, Acc, Acc).
topo_all_store(Env, [N|Ns], Sel, Seen0, Seen, Acc0, Acc) :-
    topo_one_store(Env, N, Sel, Seen0, Seen1, Acc0, Acc1),
    topo_all_store(Env, Ns, Sel, Seen1, Seen, Acc1, Acc).

topo_one_store(_Env, Name, _Sel, Seen, Seen, Acc, Acc) :-
    member(Name, Seen),
    !.
topo_one_store(Env, Name, Sel, Seen0, Seen, Acc0, Acc) :-
    (   member(Name-Ver, Sel)
    ->  findall(D, (
            collect_deps_store(Env, Name, Ver, Reqs),
            member(req(D, _), Reqs)
        ), Ds0),
        sort(Ds0, Ds),
        topo_all_store(Env, Ds, Sel, [Name|Seen0], Seen1, Acc0, Acc1),
        Acc = [Name-Ver|Acc1],
        Seen = Seen1
    ;   Seen = [Name|Seen0],
        Acc = Acc0
    ).

removal_orphans_store(Env, Pkg0, Orphans) :-
    canonicalize_name_env(Env, Pkg0, Pkg),
    env_installed(Env, Inst),
    (   installed_ver_env(Env, Pkg, Ver)
    ->  true
    ;   Ver = none
    ),
    (   Ver == none
    ->  Orphans = []
    ;   inst_closure_names_store(Env, Inst, Pkg, Ver, Closure),
        env_requested(Env, Reqs0),
        exclude_name(Pkg, Reqs0, Reqs1),
        needed_names_store(Env, Inst, Reqs1, Needed),
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

inst_closure_names_store(Env, Inst, Name, Ver, Names) :-
    inst_walk_store([Name-Ver], Env, Inst, [], [], Names).

inst_walk_store([], _Env, _Inst, _Seen, Acc, Acc).
inst_walk_store([Name-Ver|Rest], Env, Inst, Seen, Acc0, Acc) :-
    (   member(Name, Seen)
    ->  inst_walk_store(Rest, Env, Inst, Seen, Acc0, Acc)
    ;   collect_deps_store(Env, Name, Ver, Reqs),
        findall(D-DV, (
            member(req(D, _), Reqs),
            member(D-DV, Inst)
        ), Kids),
        append(Kids, Rest, More),
        inst_walk_store(More, Env, Inst, [Name|Seen], [Name|Acc0], Acc)
    ).

needed_names_store(_Env, _Inst, [], []).
needed_names_store(Env, Inst, Roots, Needed) :-
    roots_to_pairs(Roots, Inst, Pairs),
    inst_walk_store(Pairs, Env, Inst, [], [], Needed).

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

safe_upgrade_store(Env, Pkg0, NewVer, Verdict) :-
    canonicalize_name_env(Env, Pkg0, Pkg),
    (   \+ package_in_store(Env, Pkg, NewVer)
    ->  Verdict = no_candidate
    ;   \+ base_reason_env(Env, Pkg, _)
    ->  Verdict = no_candidate
    ;   base_reason_env(Env, Pkg, Reason),
        safe_upgrade_reason_store(Env, Pkg, NewVer, Reason, Verdict)
    ),
    !.

safe_upgrade_reason_store(_Env, _Pkg, _NewVer, modified, unsafe(modified)).
safe_upgrade_reason_store(_Env, _Pkg, _NewVer, footprint, safe(cost(footprint))).
safe_upgrade_reason_store(_Env, _Pkg, _NewVer, blanket, safe(cost(blanket))).
safe_upgrade_reason_store(_Env, _Pkg, _NewVer, layer_shadow, safe(cost(layer_shadow))).
safe_upgrade_reason_store(Env, Pkg, NewVer, abi_anchor, coordinated(Set)) :-
    upgrade_set_result_store(Env, Pkg, NewVer, ok(Set)).

upgrade_set_store(Env, Pkg, NewVer, Set) :-
    upgrade_set_result_store(Env, Pkg, NewVer, ok(Set)),
    !.

upgrade_set_result_store(Env, Pkg0, NewVer, Result) :-
    canonicalize_name_env(Env, Pkg0, Pkg),
    (   package_in_store(Env, Pkg, NewVer)
    ->  close_moving_store(Env, [Pkg-NewVer], Result)
    ;   Result = no_candidate
    ),
    !.

close_moving_store(Env, Acc, Result) :-
    base_holds_env(Env, Holds),
    first_broken_store(Holds, Env, Acc, Broken),
    (   Broken = none
    ->  sort(Acc, Sorted),
        Result = ok(Sorted)
    ;   Broken = broken(N, V, C),
        (   pick_repair_store(Env, N, Acc, NewV)
        ->  close_moving_store(Env, [N-NewV|Acc], Result)
        ;   Result = blocked(N, needs(C), base_has(V))
        )
    ).

first_broken_store([], _Env, _Acc, none).
first_broken_store([hold(N, V, _R)|Rest], Env, Acc, Broken) :-
    (   selected_ver(Acc, N, _)
    ->  first_broken_store(Rest, Env, Acc, Broken)
    ;   dep_breaks_moving_store(Env, N, V, Acc, C)
    ->  Broken = broken(N, V, C)
    ;   first_broken_store(Rest, Env, Acc, Broken)
    ).

dep_breaks_moving_store(Env, N, V, Acc, C) :-
    collect_deps_store(Env, N, V, Reqs),
    member(req(D, C), Reqs),
    selected_ver(Acc, D, MV),
    \+ satisfies(MV, C).

pick_repair_store(Env, Name, Acc, NewV) :-
    candidates_high_first_store(Env, Name, any, NewV),
    repairs_moving_store(Env, Name, NewV, Acc).

repairs_moving_store(Env, Name, NewV, Acc) :-
    collect_deps_store(Env, Name, NewV, Reqs),
    reqs_ok_moving(Reqs, Acc).

reqs_ok_moving([], _).
reqs_ok_moving([req(D, C)|Rest], Acc) :-
    (   selected_ver(Acc, D, MV)
    ->  satisfies(MV, C)
    ;   true
    ),
    reqs_ok_moving(Rest, Acc).

freeze_audit_store(Env, Audit) :-
    base_holds_env(Env, Holds),
    audit_holds_store(Holds, Env, [], Acc),
    sort(Acc, Audit),
    !.

audit_holds_store([], _Env, Acc, Acc).
audit_holds_store([hold(N, _V, R)|Rest], Env, Acc0, Acc) :-
    (   R == blanket
    ->  (   tight_base_revdep_store(Env, N)
        ->  Item = audit(N, suggest(abi_anchor))
        ;   Item = audit(N, over_frozen)
        )
    ;   Item = audit(N, held(R))
    ),
    audit_holds_store(Rest, Env, [Item|Acc0], Acc).

tight_base_revdep_store(Env, Pkg) :-
    store_key(Env, Pkg, Key),
    store_revdep(Key, Packed),
    unpack_rev(Packed, N, V, C),
    N \== Pkg,
    tight_constraint(C),
    base_ver_env(Env, N, BV),
    V == BV.

dependents_store(Env, Pkg0, Deps) :-
    canonicalize_name_env(Env, Pkg0, Pkg),
    store_key(Env, Pkg, Key),
    findall(N-V, (
        store_revdep(Key, Packed),
        unpack_rev(Packed, N, V, _C)
    ), Acc),
    sort(Acc, Deps),
    !.

dependents_installed_store(Env, Pkg0, Deps) :-
    dependents_store(Env, Pkg0, All),
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
