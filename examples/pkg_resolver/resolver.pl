:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% resolver.pl -- uw-resolve P0. Pure relations over a frozen catalog term.
%
% Catalog:
%   catalog(Packages, Depends, Conflicts, Base, Installed, Requested)
%     Packages  = [package(Name, v(M,I,P)), ...]
%     Depends   = [depends(Name, Ver, DepName, Constraint), ...]
%     Conflicts = [conflicts(Name, Ver, OtherName), ...]
%     Base/Installed = [Name-Ver, ...]
%     Requested = [Name, ...]          % manual/root installs (removal_orphans)
%
% Constraints: any | eq(V) | gte(V) | lt(V) | range(Lo, Hi)
%   Lo inclusive, Hi exclusive. Versions compared lexicographically.
%   Debian epoch/tilde semantics are deferred.
%
% Queries (Catalog is data; no assert/retract):
%   resolve(+Cat, +Requests, -Selection)
%   resolve_layered(+Cat, +Requests, -Selection)
%   explain_blocked(+Cat, +Request, -Blocked)     % nondet
%   layer_closure(+Cat, +Request, -Layer)
%   removal_orphans(+Cat, +Pkg, -Orphans)
%
% Determinism: resolve* commit to the FIRST search solution (prefer a
% base-satisfied dependency, else highest version). Selections are sorted
% by Name-Ver. explain_blocked enumerates; callers findall+sort at the
% API edge. removal_orphans returns a sorted list.
%
%   swipl -q -g test_resolver -t halt examples/pkg_resolver/test_resolver.pl

:- module(resolver, [
    resolve/3,
    resolve_layered/3,
    explain_blocked/3,
    explain_blocked_list/3,
    layer_closure/3,
    removal_orphans/3,
    satisfies/2,
    version_lt/2
]).

% ---------------------------------------------------------------------------
% Catalog accessors
% ---------------------------------------------------------------------------

packages(catalog(Ps, _, _, _, _, _), Ps).
depends_list(catalog(_, Ds, _, _, _, _), Ds).
conflicts_list(catalog(_, _, Cs, _, _, _), Cs).
base_list(catalog(_, _, _, Bs, _, _), Bs).
installed_list(catalog(_, _, _, _, Is, _), Is).
requested_list(catalog(_, _, _, _, _, Rs), Rs).

package_in(Cat, Name, Ver) :-
    packages(Cat, Ps),
    member(package(Name, Ver), Ps).

depends_in(Cat, Name, Ver, Dep, C) :-
    depends_list(Cat, Ds),
    member(depends(Name, Ver, Dep, C), Ds).

conflicts_in(Cat, Name, Ver, Other) :-
    conflicts_list(Cat, Cs),
    member(conflicts(Name, Ver, Other), Cs).

base_ver(Cat, Name, Ver) :-
    base_list(Cat, Bs),
    member(Name-Ver, Bs).

base_name(Cat, Name) :-
    base_ver(Cat, Name, _).

installed_ver(Cat, Name, Ver) :-
    installed_list(Cat, Is),
    member(Name-Ver, Is).

% ---------------------------------------------------------------------------
% Versions and constraints
% ---------------------------------------------------------------------------

version_lt(v(A, B, C), v(D, E, F)) :-
    (   A < D
    ->  true
    ;   A =:= D, B < E
    ->  true
    ;   A =:= D, B =:= E, C < F
    ).

satisfies(_Ver, any).
satisfies(Ver, eq(E)) :- Ver = E.
satisfies(Ver, gte(G)) :- \+ version_lt(Ver, G).
satisfies(Ver, lt(H)) :- version_lt(Ver, H).
satisfies(Ver, range(Lo, Hi)) :-
    \+ version_lt(Ver, Lo),
    version_lt(Ver, Hi).

% Highest version first (standard order on v/3 is lexicographic).
% Collected without findall so the JS WAM interpreter does not have to
% run a lowered helper inside BeginAggregate (that path drops answers).
candidates_high_first(Cat, Name, C, Ver) :-
    packages(Cat, Ps),
    matching_versions(Ps, Name, C, Vs),
    sort(Vs, Asc),
    reverse(Asc, Desc),
    member(Ver, Desc).

matching_versions([], _Name, _C, []).
matching_versions([package(N, V)|Rest], Name, C, Out) :-
    (   N == Name,
        satisfies(V, C)
    ->  Out = [V|Vs]
    ;   Out = Vs
    ),
    matching_versions(Rest, Name, C, Vs).

% ---------------------------------------------------------------------------
% Requests
% ---------------------------------------------------------------------------

request_to_req(R, req(Name, C)) :-
    (   R = req(Name, C)
    ->  true
    ;   Name = R,
        C = any
    ).

map_requests([], []).
map_requests([R|Rs], [Q|Qs]) :-
    request_to_req(R, Q),
    map_requests(Rs, Qs).

selected_ver([H|Rest], Name, Ver) :-
    (   H = Name-Ver
    ->  true
    ;   selected_ver(Rest, Name, Ver)
    ).

member_selected(Acc, Name, Ver) :-
    member(Name-Ver, Acc).

% ---------------------------------------------------------------------------
% Conflicts: either direction, against a selected OtherName (any version).
% ---------------------------------------------------------------------------

acc_conflicts(Cat, Name, Ver, Acc) :-
    member_selected(Acc, Other, OtherVer),
    (   conflicts_in(Cat, Name, Ver, Other)
    ;   conflicts_in(Cat, Other, OtherVer, Name)
    ).

% Deterministic scan — no leftover CPs, so a later failure can still
% retry candidate versions. Used by resolve_pending instead of \+/1.
no_acc_conflicts(_Cat, _Name, _Ver, []).
no_acc_conflicts(Cat, Name, Ver, [Other-OtherVer|Rest]) :-
    \+ conflicts_in(Cat, Name, Ver, Other),
    \+ conflicts_in(Cat, Other, OtherVer, Name),
    no_acc_conflicts(Cat, Name, Ver, Rest).

% ---------------------------------------------------------------------------
% Classic resolve — first solution, highest version, may "upgrade" a base pkg.
% ---------------------------------------------------------------------------

resolve(Cat, Requests, Selection) :-
    map_requests(Requests, Pending),
    resolve_pending(classic, Cat, Pending, [], Acc),
    !,
    sort(Acc, Selection).

% ---------------------------------------------------------------------------
% Layered resolve — never re-select or upgrade a base package.
% ---------------------------------------------------------------------------

resolve_layered(Cat, Requests, Selection) :-
    map_requests(Requests, Pending),
    resolve_pending(layered, Cat, Pending, [], Acc),
    !,
    sort(Acc, Selection).

resolve_pending(_Mode, _Cat, [], Acc, Acc).
resolve_pending(Mode, Cat, [req(Name, C)|Rest], Acc, Sel) :-
    (   selected_ver(Acc, Name, Ver)
    ->  satisfies(Ver, C),
        resolve_pending(Mode, Cat, Rest, Acc, Sel)
    ;   pick(Mode, Cat, Name, C, Acc, Ver, Origin),
        collect_deps(Cat, Name, Ver, DepReqs),
        append(DepReqs, Rest, More),
        (   Origin = from_base
        ->  resolve_pending(Mode, Cat, More, Acc, Sel)
        ;   no_acc_conflicts(Cat, Name, Ver, Acc),
            resolve_pending(Mode, Cat, More, [Name-Ver|Acc], Sel)
        )
    ).

collect_deps(Cat, Name, Ver, Reqs) :-
    depends_list(Cat, Ds),
    matching_deps(Ds, Name, Ver, Reqs).

matching_deps([], _Name, _Ver, []).
matching_deps([depends(N, V, D, C)|Rest], Name, Ver, Out) :-
    (   N == Name,
        V == Ver
    ->  Out = [req(D, C)|Rs]
    ;   Out = Rs
    ),
    matching_deps(Rest, Name, Ver, Rs).

% Classic: catalog versions, highest first. Backtracks.
pick(classic, Cat, Name, C, _Acc, Ver, from_catalog) :-
    candidates_high_first(Cat, Name, C, Ver).

% Layered: a base package that satisfies is used in-place (not selected).
% A base package that does not satisfy is a hard fail (no upgrade).
% Names not in the base are chosen from the catalog, highest first.
pick(layered, Cat, Name, C, _Acc, Ver, Origin) :-
    (   base_ver(Cat, Name, BV)
    ->  satisfies(BV, C),
        Ver = BV,
        Origin = from_base
    ;   candidates_high_first(Cat, Name, C, Ver),
        Origin = from_catalog
    ).

% ---------------------------------------------------------------------------
% explain_blocked — same layered pick, failure branch yields blocked/3.
% ---------------------------------------------------------------------------

explain_blocked(Cat, Request, Blocked) :-
    request_to_req(Request, Req),
    blocked_from(Cat, Req, [], Blocked).

explain_blocked_list(Cat, Request, List) :-
    request_to_req(Request, Req),
    blocked_acc(Cat, Req, [], [], Acc),
    sort(Acc, List),
    !.

blocked_from(Cat, req(Name, C), Seen, Blocked) :-
    \+ seen_name(Seen, Name),
    base_ver(Cat, Name, BV),
    \+ satisfies(BV, C),
    Blocked = blocked(Name, needs(C), base_has(BV)).
blocked_from(Cat, req(Name, C), Seen, Blocked) :-
    \+ seen_name(Seen, Name),
    layered_walk_ver(Cat, Name, C, Ver),
    collect_deps(Cat, Name, Ver, DepReqs),
    member(Dep, DepReqs),
    blocked_from(Cat, Dep, [Name|Seen], Blocked).

% Deterministic walk used by explain_blocked_list/3 (no findall).
blocked_acc(_Cat, req(Name, _C), Seen, Acc, Acc) :-
    seen_name(Seen, Name), !.
blocked_acc(Cat, req(Name, C), Seen, Acc0, Acc) :-
    (   base_ver(Cat, Name, BV),
        \+ satisfies(BV, C)
    ->  Acc1 = [blocked(Name, needs(C), base_has(BV))|Acc0]
    ;   Acc1 = Acc0
    ),
    (   layered_walk_ver(Cat, Name, C, Ver)
    ->  collect_deps(Cat, Name, Ver, DepReqs),
        blocked_acc_list(Cat, DepReqs, [Name|Seen], Acc1, Acc)
    ;   Acc = Acc1
    ).

blocked_acc_list(_Cat, [], _Seen, Acc, Acc).
blocked_acc_list(Cat, [Dep|Rest], Seen, Acc0, Acc) :-
    blocked_acc(Cat, Dep, Seen, Acc0, Acc1),
    blocked_acc_list(Cat, Rest, Seen, Acc1, Acc).

seen_name([H|Rest], Name) :-
    (   H == Name
    ->  true
    ;   seen_name(Rest, Name)
    ).

% The version layered resolution would use, if any (no conflict check —
% version-ceiling explanations are independent of sibling conflicts).
layered_walk_ver(Cat, Name, C, Ver) :-
    (   base_ver(Cat, Name, BV)
    ->  satisfies(BV, C),
        Ver = BV
    ;   candidates_high_first(Cat, Name, C, Ver)
    ),
    !.

% ---------------------------------------------------------------------------
% layer_closure — non-base closure, dependencies before dependents.
% ---------------------------------------------------------------------------

layer_closure(Cat, Request, Layer) :-
    resolve_layered(Cat, [Request], Sel),
    topo_sort_sel(Cat, Sel, Layer),
    !.

topo_sort_sel(_Cat, [], []) :- !.
topo_sort_sel(Cat, Sel, Layer) :-
    sort(Sel, Sorted),
    names_of(Sorted, Names),
    topo_all(Cat, Names, Sel, [], _Seen, [], Acc),
    reverse(Acc, Layer).

names_of([], []).
names_of([N-_|Rest], [N|Ns]) :-
    names_of(Rest, Ns).

topo_all(_Cat, [], _Sel, Seen, Seen, Acc, Acc).
topo_all(Cat, [N|Ns], Sel, Seen0, Seen, Acc0, Acc) :-
    topo_one(Cat, N, Sel, Seen0, Seen1, Acc0, Acc1),
    topo_all(Cat, Ns, Sel, Seen1, Seen, Acc1, Acc).

topo_one(_Cat, Name, _Sel, Seen, Seen, Acc, Acc) :-
    member(Name, Seen),
    !.
topo_one(Cat, Name, Sel, Seen0, Seen, Acc0, Acc) :-
    (   member(Name-Ver, Sel)
    ->  findall(D, depends_in(Cat, Name, Ver, D, _), Ds0),
        sort(Ds0, Ds),
        topo_all(Cat, Ds, Sel, [Name|Seen0], Seen1, Acc0, Acc1),
        Acc = [Name-Ver|Acc1],
        Seen = Seen1
    ;   Seen = [Name|Seen0],
        Acc = Acc0
    ).

% ---------------------------------------------------------------------------
% removal_orphans — PPM-style trim. Base packages are never orphans.
% ---------------------------------------------------------------------------

removal_orphans(Cat, Pkg, Orphans) :-
    installed_list(Cat, Inst),
    (   installed_ver(Cat, Pkg, Ver)
    ->  true
    ;   Ver = none
    ),
    (   Ver == none
    ->  Orphans = []
    ;   inst_closure_names(Cat, Inst, Pkg, Ver, Closure),
        requested_list(Cat, Reqs0),
        exclude_name(Pkg, Reqs0, Reqs1),
        needed_names(Cat, Inst, Reqs1, Needed),
        findall(N-V, (
            member(N-V, Inst),
            N \== Pkg,
            member(N, Closure),
            \+ member(N, Needed),
            \+ base_name(Cat, N)
        ), Or0),
        sort(Or0, Orphans)
    ),
    !.

exclude_name(_N, [], []).
exclude_name(N, [N|Rs], Out) :- !,
    exclude_name(N, Rs, Out).
exclude_name(N, [R|Rs], [R|Out]) :-
    exclude_name(N, Rs, Out).

inst_closure_names(Cat, Inst, Name, Ver, Names) :-
    inst_walk([Name-Ver], Cat, Inst, [], [], Names).

inst_walk([], _Cat, _Inst, _Seen, Acc, Acc).
inst_walk([Name-Ver|Rest], Cat, Inst, Seen, Acc0, Acc) :-
    (   member(Name, Seen)
    ->  inst_walk(Rest, Cat, Inst, Seen, Acc0, Acc)
    ;   findall(D-DV, (
            depends_in(Cat, Name, Ver, D, _),
            member(D-DV, Inst)
        ), Kids),
        append(Kids, Rest, More),
        inst_walk(More, Cat, Inst, [Name|Seen], [Name|Acc0], Acc)
    ).

needed_names(_Cat, _Inst, [], []).
needed_names(Cat, Inst, Roots, Needed) :-
    roots_to_pairs(Roots, Inst, Pairs),
    inst_walk(Pairs, Cat, Inst, [], [], Needed).

roots_to_pairs([], _Inst, []).
roots_to_pairs([N|Ns], Inst, [N-V|Ps]) :-
    member(N-V, Inst),
    !,
    roots_to_pairs(Ns, Inst, Ps).
roots_to_pairs([_|Ns], Inst, Ps) :-
    roots_to_pairs(Ns, Inst, Ps).
