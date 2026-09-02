:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% resolver.pl -- uw-resolve P0.5. Pure relations over a frozen catalog term.
%
% Catalog (P0, still accepted):
%   catalog(Packages, Depends, Conflicts, Base, Installed, Requested)
%
% Catalog (P0.5 extras; Layers / Excluded / Aliases):
%   catalog(Packages, Depends, Conflicts, Base, Installed, Requested,
%           Layers, Excluded, Aliases)
%
% Base entries (back-compat):
%   Name-Ver                         % Reason = blanket
%   base(Name-Ver, Reason)           % Reason ∈ layer_shadow | abi_anchor
%                                    %          | modified | footprint | blanket
%   layer(LayerName, [Entry...])     % also accepted in Base; base = layer `base`
% Layers: [layer(Name, [Entry...]), ...]
% Excluded: [Name, ...]              % candidate-generation blacklist only
% Aliases:  [alias(Alias, Canonical), ...]   % request edge only
%
% Constraints: any | eq(V) | gte(V) | lt(V) | range(Lo, Hi)
% Debian epoch/tilde, Provides/virtual, GP-LMDB catalog, CLI: deferred.
%
%   swipl -q -g test_resolver -t halt examples/pkg_resolver/test_resolver.pl

:- module(resolver, [
    resolve/3,
    resolve_layered/3,
    explain_blocked/3,
    explain_blocked_list/3,
    layer_closure/3,
    removal_orphans/3,
    safe_upgrade/4,
    upgrade_set/4,
    upgrade_set_result/4,
    freeze_audit/2,
    dependents/3,
    dependents_installed/3,
    satisfies/2,
    version_lt/2
]).

% ---------------------------------------------------------------------------
% Catalog accessors — catalog/6 (P0) and catalog/9 (P0.5)
% ---------------------------------------------------------------------------

packages(catalog(Ps, _, _, _, _, _), Ps).
packages(catalog(Ps, _, _, _, _, _, _, _, _), Ps).
depends_list(catalog(_, Ds, _, _, _, _), Ds).
depends_list(catalog(_, Ds, _, _, _, _, _, _, _), Ds).
conflicts_list(catalog(_, _, Cs, _, _, _), Cs).
conflicts_list(catalog(_, _, Cs, _, _, _, _, _, _), Cs).
base_list(catalog(_, _, _, Bs, _, _), Bs).
base_list(catalog(_, _, _, Bs, _, _, _, _, _), Bs).
installed_list(catalog(_, _, _, _, Is, _), Is).
installed_list(catalog(_, _, _, _, Is, _, _, _, _), Is).
requested_list(catalog(_, _, _, _, _, Rs), Rs).
requested_list(catalog(_, _, _, _, _, Rs, _, _, _), Rs).
layers_list(catalog(_, _, _, _, _, _), []).
layers_list(catalog(_, _, _, _, _, _, Ls, _, _), Ls).
excluded_list(catalog(_, _, _, _, _, _), []).
excluded_list(catalog(_, _, _, _, _, _, _, Es, _), Es).
alias_list(catalog(_, _, _, _, _, _), []).
alias_list(catalog(_, _, _, _, _, _, _, _, As), As).

package_in(Cat, Name, Ver) :-
    packages(Cat, Ps),
    member(package(Name, Ver), Ps).

depends_in(Cat, Name, Ver, Dep, C) :-
    depends_list(Cat, Ds),
    member(depends(Name, Ver, Dep, C), Ds).

conflicts_in(Cat, Name, Ver, Other) :-
    conflicts_list(Cat, Cs),
    member(conflicts(Name, Ver, Other), Cs).

% A package in ANY loaded layer (base list + named layers), first match.
% Bare Name-Ver is the P0 shape; base/2 and nested layer/2 are P0.5.
base_ver(Cat, Name, Ver) :-
    base_list(Cat, Bs),
    layers_list(Cat, Ls),
    append(Bs, Ls, All),
    lookup_held(All, Name, Ver).

base_name(Cat, Name) :-
    base_ver(Cat, Name, _).

installed_ver(Cat, Name, Ver) :-
    installed_list(Cat, Is),
    member(Name-Ver, Is).

excluded_name(Cat, Name) :-
    excluded_list(Cat, Es),
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

tight_constraint(C) :-
    C \== any.

% Highest version first. Excluded names produce no candidates (blacklist
% filters generation only — never removal).
candidates_high_first(Cat, Name, C, Ver) :-
    \+ excluded_name(Cat, Name),
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
% Requests (aliases applied here only; catalog names stay canonical)
% ---------------------------------------------------------------------------

canonicalize_name(Cat, In, Out) :-
    alias_list(Cat, As),
    alias_lookup(As, In, Out).

alias_lookup([], N, N).
alias_lookup([alias(A, Canon)|Rest], In, Out) :-
    (   In == A
    ->  Out = Canon
    ;   alias_lookup(Rest, In, Out)
    ).

request_to_req(Cat, R, req(Name, C)) :-
    (   R = req(Raw, C)
    ->  canonicalize_name(Cat, Raw, Name)
    ;   canonicalize_name(Cat, R, Name),
        C = any
    ).

map_requests(_Cat, [], []).
map_requests(Cat, [R|Rs], [Q|Qs]) :-
    request_to_req(Cat, R, Q),
    map_requests(Cat, Rs, Qs).

selected_ver([H|Rest], Name, Ver) :-
    (   H = Name-Ver
    ->  true
    ;   selected_ver(Rest, Name, Ver)
    ).

member_selected(Acc, Name, Ver) :-
    member(Name-Ver, Acc).

% ---------------------------------------------------------------------------
% Conflicts
% ---------------------------------------------------------------------------

acc_conflicts(Cat, Name, Ver, Acc) :-
    member_selected(Acc, Other, OtherVer),
    (   conflicts_in(Cat, Name, Ver, Other)
    ;   conflicts_in(Cat, Other, OtherVer, Name)
    ).

no_acc_conflicts(_Cat, _Name, _Ver, []).
no_acc_conflicts(Cat, Name, Ver, [Other-OtherVer|Rest]) :-
    \+ conflicts_in(Cat, Name, Ver, Other),
    \+ conflicts_in(Cat, Other, OtherVer, Name),
    no_acc_conflicts(Cat, Name, Ver, Rest).

% ---------------------------------------------------------------------------
% Classic / layered resolve
% ---------------------------------------------------------------------------

resolve(Cat, Requests, Selection) :-
    map_requests(Cat, Requests, Pending),
    resolve_pending(classic, Cat, Pending, [], Acc),
    !,
    sort(Acc, Selection).

resolve_layered(Cat, Requests, Selection) :-
    map_requests(Cat, Requests, Pending),
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

pick(classic, Cat, Name, C, _Acc, Ver, from_catalog) :-
    candidates_high_first(Cat, Name, C, Ver).

% Layered: a loaded-layer package that satisfies is used in-place
% (never re-selected). A loaded-layer package that does not satisfy is
% a hard fail. Names not in any loaded layer come from the catalog.
pick(layered, Cat, Name, C, _Acc, Ver, Origin) :-
    (   base_ver(Cat, Name, BV)
    ->  satisfies(BV, C),
        Ver = BV,
        Origin = from_base
    ;   candidates_high_first(Cat, Name, C, Ver),
        Origin = from_catalog
    ).

% ---------------------------------------------------------------------------
% explain_blocked
% ---------------------------------------------------------------------------

explain_blocked(Cat, Request, Blocked) :-
    request_to_req(Cat, Request, Req),
    blocked_from(Cat, Req, [], Blocked).

explain_blocked_list(Cat, Request, List) :-
    request_to_req(Cat, Request, Req),
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

layered_walk_ver(Cat, Name, C, Ver) :-
    (   base_ver(Cat, Name, BV)
    ->  satisfies(BV, C),
        Ver = BV
    ;   candidates_high_first(Cat, Name, C, Ver)
    ),
    !.

% ---------------------------------------------------------------------------
% layer_closure
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
% removal_orphans — exclusion is NOT consulted (Pkg bug we refuse to copy)
% ---------------------------------------------------------------------------

removal_orphans(Cat, Pkg0, Orphans) :-
    canonicalize_name(Cat, Pkg0, Pkg),
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

% ---------------------------------------------------------------------------
% Base holds with reasons (P0 Name-Ver ⇒ blanket). Named layers other
% than `base` are not freeze-audited.
% ---------------------------------------------------------------------------

base_holds(Cat, Holds) :-
    base_list(Cat, Bs),
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

base_reason(Cat, Name, Reason) :-
    base_holds(Cat, Holds),
    hold_reason(Holds, Name, Reason).

hold_reason([hold(N, _V, R)|T], Name, Reason) :-
    (   N == Name
    ->  Reason = R
    ;   hold_reason(T, Name, Reason)
    ).

% ---------------------------------------------------------------------------
% safe_upgrade / upgrade_set
% ---------------------------------------------------------------------------

safe_upgrade(Cat, Pkg0, NewVer, Verdict) :-
    canonicalize_name(Cat, Pkg0, Pkg),
    (   \+ package_in(Cat, Pkg, NewVer)
    ->  Verdict = no_candidate
    ;   \+ base_reason(Cat, Pkg, _)
    ->  Verdict = no_candidate
    ;   base_reason(Cat, Pkg, Reason),
        safe_upgrade_reason(Cat, Pkg, NewVer, Reason, Verdict)
    ),
    !.

safe_upgrade_reason(_Cat, _Pkg, _NewVer, modified, unsafe(modified)).
safe_upgrade_reason(_Cat, _Pkg, _NewVer, footprint, safe(cost(footprint))).
safe_upgrade_reason(_Cat, _Pkg, _NewVer, blanket, safe(cost(blanket))).
safe_upgrade_reason(_Cat, _Pkg, _NewVer, layer_shadow, safe(cost(layer_shadow))).
safe_upgrade_reason(Cat, Pkg, NewVer, abi_anchor, coordinated(Set)) :-
    upgrade_set_result(Cat, Pkg, NewVer, ok(Set)).

upgrade_set(Cat, Pkg, NewVer, Set) :-
    upgrade_set_result(Cat, Pkg, NewVer, ok(Set)),
    !.

upgrade_set_result(Cat, Pkg0, NewVer, Result) :-
    canonicalize_name(Cat, Pkg0, Pkg),
    (   package_in(Cat, Pkg, NewVer)
    ->  close_moving(Cat, [Pkg-NewVer], Result)
    ;   Result = no_candidate
    ),
    !.

close_moving(Cat, Acc, Result) :-
    base_holds(Cat, Holds),
    first_broken(Holds, Cat, Acc, Broken),
    (   Broken = none
    ->  sort(Acc, Sorted),
        Result = ok(Sorted)
    ;   Broken = broken(N, V, C),
        (   pick_repair(Cat, N, Acc, NewV)
        ->  close_moving(Cat, [N-NewV|Acc], Result)
        ;   Result = blocked(N, needs(C), base_has(V))
        )
    ).

first_broken([], _Cat, _Acc, none).
first_broken([hold(N, V, _R)|Rest], Cat, Acc, Broken) :-
    (   selected_ver(Acc, N, _)
    ->  first_broken(Rest, Cat, Acc, Broken)
    ;   dep_breaks_moving(Cat, N, V, Acc, C)
    ->  Broken = broken(N, V, C)
    ;   first_broken(Rest, Cat, Acc, Broken)
    ).

dep_breaks_moving(Cat, N, V, Acc, C) :-
    depends_list(Cat, Ds),
    dep_breaks(Ds, N, V, Acc, C).

dep_breaks([depends(HN, HV, D, C)|Rest], N, V, Acc, COut) :-
    (   HN == N,
        HV == V,
        selected_ver(Acc, D, MV),
        \+ satisfies(MV, C)
    ->  COut = C
    ;   dep_breaks(Rest, N, V, Acc, COut)
    ).

pick_repair(Cat, Name, Acc, NewV) :-
    candidates_high_first(Cat, Name, any, NewV),
    repairs_moving(Cat, Name, NewV, Acc).

repairs_moving(Cat, Name, NewV, Acc) :-
    collect_deps(Cat, Name, NewV, Reqs),
    reqs_ok_moving(Reqs, Acc).

reqs_ok_moving([], _).
reqs_ok_moving([req(D, C)|Rest], Acc) :-
    (   selected_ver(Acc, D, MV)
    ->  satisfies(MV, C)
    ;   true
    ),
    reqs_ok_moving(Rest, Acc).

% ---------------------------------------------------------------------------
% freeze_audit
% ---------------------------------------------------------------------------

freeze_audit(Cat, Audit) :-
    base_holds(Cat, Holds),
    audit_holds(Holds, Cat, [], Acc),
    sort(Acc, Audit),
    !.

audit_holds([], _Cat, Acc, Acc).
audit_holds([hold(N, _V, R)|Rest], Cat, Acc0, Acc) :-
    (   R == blanket
    ->  (   tight_base_revdep(Cat, N)
        ->  Item = audit(N, suggest(abi_anchor))
        ;   Item = audit(N, over_frozen)
        )
    ;   Item = audit(N, held(R))
    ),
    audit_holds(Rest, Cat, [Item|Acc0], Acc).

tight_base_revdep(Cat, Pkg) :-
    base_holds(Cat, Holds),
    tight_rev_in(Holds, Cat, Pkg).

tight_rev_in([hold(N, V, _R)|Rest], Cat, Pkg) :-
    (   N \== Pkg,
        depends_in(Cat, N, V, Pkg, C),
        tight_constraint(C)
    ->  true
    ;   tight_rev_in(Rest, Cat, Pkg)
    ).

% ---------------------------------------------------------------------------
% dependents / dependents_installed  (what-needs)
% ---------------------------------------------------------------------------

dependents(Cat, Pkg0, Deps) :-
    canonicalize_name(Cat, Pkg0, Pkg),
    depends_list(Cat, Ds),
    direct_on(Ds, Pkg, [], Acc),
    sort(Acc, Deps),
    !.

dependents_installed(Cat, Pkg0, Deps) :-
    dependents(Cat, Pkg0, All),
    keep_installed_or_base(All, Cat, [], Acc),
    sort(Acc, Deps),
    !.

direct_on([], _Pkg, Acc, Acc).
direct_on([depends(N, V, D, _C)|Rest], Pkg, Acc0, Acc) :-
    (   D == Pkg
    ->  Acc1 = [N-V|Acc0]
    ;   Acc1 = Acc0
    ),
    direct_on(Rest, Pkg, Acc1, Acc).

keep_installed_or_base([], _Cat, Acc, Acc).
keep_installed_or_base([N-V|Rest], Cat, Acc0, Acc) :-
    (   installed_or_base(Cat, N, V)
    ->  Acc1 = [N-V|Acc0]
    ;   Acc1 = Acc0
    ),
    keep_installed_or_base(Rest, Cat, Acc1, Acc).

installed_or_base(Cat, N, V) :-
    installed_ver(Cat, N, V).
installed_or_base(Cat, N, V) :-
    base_ver(Cat, N, BV),
    V == BV.
