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
% Catalog (P3 extras; Provides — virtual packages):
%   catalog(Packages, Depends, Conflicts, Base, Installed, Requested,
%           Layers, Excluded, Aliases, Provides)
%
% Base entries (back-compat):
%   Name-Ver                         % Reason = blanket
%   base(Name-Ver, Reason)           % Reason ∈ layer_shadow | abi_anchor
%                                    %          | modified | footprint | blanket
%   layer(LayerName, [Entry...])     % also accepted in Base; base = layer `base`
% Layers: [layer(Name, [Entry...]), ...]
% Excluded: [Name, ...]              % candidate-generation blacklist only
% Aliases:  [alias(Alias, Canonical), ...]   % request edge only
% Provides (P3):
%   provides(Name, Ver, Virtual)              % unversioned (satisfies `any` only)
%   provides(Name, Ver, Virtual, VirtualVer)  % versioned
% Depends third argument may be alternatives([dep(Name, C)|...]).
%
% Versions: v(M,I,P) (P0, unchanged) | deb(Epoch, UpSegs, RevSegs) (P3)
%   Seg = s(OrderCodes, DigitInt) — pre-segmented at the ingestion edge.
% Constraints: any | eq(V) | gte(V) | lt(V) | range(Lo, Hi)
%              | lte(V) | gt(V)   % Debian <= and >> at the ingestion edge
% Debian relation map (ingestion): >= → gte, <= → lte, >> → gt, << → lt, = → eq.
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

% Each accessor gains one delegating icat/3 clause (G1a). icat/3 is an
% internal per-call wrapper built by index_catalog/2 at the API edge; it
% never escapes resolver.pl and is never accepted as input.
packages(catalog(Ps, _, _, _, _, _), Ps).
packages(catalog(Ps, _, _, _, _, _, _, _, _), Ps).
packages(catalog(Ps, _, _, _, _, _, _, _, _, _), Ps).
packages(icat(Cat, _, _), Ps) :- packages(Cat, Ps).
depends_list(catalog(_, Ds, _, _, _, _), Ds).
depends_list(catalog(_, Ds, _, _, _, _, _, _, _), Ds).
depends_list(catalog(_, Ds, _, _, _, _, _, _, _, _), Ds).
depends_list(icat(Cat, _, _), Ds) :- depends_list(Cat, Ds).
conflicts_list(catalog(_, _, Cs, _, _, _), Cs).
conflicts_list(catalog(_, _, Cs, _, _, _, _, _, _), Cs).
conflicts_list(catalog(_, _, Cs, _, _, _, _, _, _, _), Cs).
conflicts_list(icat(Cat, _, _), Cs) :- conflicts_list(Cat, Cs).
base_list(catalog(_, _, _, Bs, _, _), Bs).
base_list(catalog(_, _, _, Bs, _, _, _, _, _), Bs).
base_list(catalog(_, _, _, Bs, _, _, _, _, _, _), Bs).
base_list(icat(Cat, _, _), Bs) :- base_list(Cat, Bs).
installed_list(catalog(_, _, _, _, Is, _), Is).
installed_list(catalog(_, _, _, _, Is, _, _, _, _), Is).
installed_list(catalog(_, _, _, _, Is, _, _, _, _, _), Is).
installed_list(icat(Cat, _, _), Is) :- installed_list(Cat, Is).
requested_list(catalog(_, _, _, _, _, Rs), Rs).
requested_list(catalog(_, _, _, _, _, Rs, _, _, _), Rs).
requested_list(catalog(_, _, _, _, _, Rs, _, _, _, _), Rs).
requested_list(icat(Cat, _, _), Rs) :- requested_list(Cat, Rs).
layers_list(catalog(_, _, _, _, _, _), []).
layers_list(catalog(_, _, _, _, _, _, Ls, _, _), Ls).
layers_list(catalog(_, _, _, _, _, _, Ls, _, _, _), Ls).
layers_list(icat(Cat, _, _), Ls) :- layers_list(Cat, Ls).
excluded_list(catalog(_, _, _, _, _, _), []).
excluded_list(catalog(_, _, _, _, _, _, _, Es, _), Es).
excluded_list(catalog(_, _, _, _, _, _, _, Es, _, _), Es).
excluded_list(icat(Cat, _, _), Es) :- excluded_list(Cat, Es).
alias_list(catalog(_, _, _, _, _, _), []).
alias_list(catalog(_, _, _, _, _, _, _, _, As), As).
alias_list(catalog(_, _, _, _, _, _, _, _, As, _), As).
alias_list(icat(Cat, _, _), As) :- alias_list(Cat, As).
provides_list(catalog(_, _, _, _, _, _), []).
provides_list(catalog(_, _, _, _, _, _, _, _, _), []).
provides_list(catalog(_, _, _, _, _, _, _, _, _, Pr), Pr).
provides_list(icat(Cat, _, _), Pr) :- provides_list(Cat, Pr).

dep_index(icat(_, DepT, _), DepT).
pkg_index(icat(_, _, PkgT), PkgT).

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
version_lt(deb(E1, U1, R1), deb(E2, U2, R2)) :-
    (   E1 < E2
    ->  true
    ;   E1 =:= E2, segs_lt(U1, U2)
    ->  true
    ;   E1 =:= E2, \+ segs_lt(U1, U2), \+ segs_lt(U2, U1), segs_lt(R1, R2)
    ).

% Debian Policy §5.6.12 segment walk. Missing part = empty (s([],0) pad).
% ~ (code 126) sorts before everything, including the empty string.
segs_lt([], []) :- !, fail.
segs_lt(A, B) :-
    pad_head(A, A1),
    pad_head(B, B1),
    segs_lt_1(A1, B1).

pad_head([], [s([], 0)]) :- !.
pad_head(Segs, Segs).

segs_lt_1([s(O1, N1)|T1], [s(O2, N2)|T2]) :-
    (   order_lt(O1, O2)
    ->  true
    ;   O1 == O2, N1 < N2
    ->  true
    ;   O1 == O2, N1 =:= N2,
        segs_lt(T1, T2)
    ).

order_lt([], []) :- !, fail.
order_lt([], [C|_]) :-
    order_val(C, V),
    0 < V.
order_lt([C|_], []) :-
    order_val(C, V),
    V < 0.
order_lt([A|As], [B|Bs]) :-
    order_val(A, VA),
    order_val(B, VB),
    (   VA < VB
    ->  true
    ;   VA =:= VB,
        order_lt(As, Bs)
    ).

order_val(126, -1) :- !.          % ~
order_val(C, C) :-                % letters before non-letters
    C >= 65, C =< 90, !.
order_val(C, C) :-
    C >= 97, C =< 122, !.
order_val(C, V) :-
    V is C + 256.

satisfies(_Ver, any).
satisfies(Ver, eq(E)) :- Ver = E.
satisfies(Ver, gte(G)) :- \+ version_lt(Ver, G).
satisfies(Ver, lte(G)) :- \+ version_lt(G, Ver).
satisfies(Ver, lt(H)) :- version_lt(Ver, H).
satisfies(Ver, gt(H)) :- version_lt(H, Ver).
satisfies(Ver, range(Lo, Hi)) :-
    \+ version_lt(Ver, Lo),
    version_lt(Ver, Hi).

tight_constraint(C) :-
    C \== any.

is_v3(v(_, _, _)).

% v/3 keeps the historical sort/2 path (byte-identical candidate order).
% deb/3 uses version_lt/2 via predsort.
sort_versions_desc(Vs, Desc) :-
    (   maplist(is_v3, Vs)
    ->  sort(Vs, Asc),
        reverse(Asc, Desc)
    ;   predsort(cmp_ver, Vs, Asc),
        reverse(Asc, Desc)
    ).

cmp_ver(<, A, B) :- version_lt(A, B), !.
cmp_ver(>, A, B) :- version_lt(B, A), !.
cmp_ver(=, _, _).

% Highest version first. Excluded names produce no candidates (blacklist
% filters generation only — never removal).
candidates_high_first(Cat, Name, C, Ver) :-
    \+ excluded_name(Cat, Name),
    matching_versions_in(Cat, Name, C, Vs),
    sort_versions_desc(Vs, Desc),
    member(Ver, Desc).

% Indexed when the catalog is wrapped (G1b), the historical full scan
% otherwise. Both compute the same list: the catalog-order versions of
% Name that satisfy C, duplicates included.
matching_versions_in(Cat, Name, C, Vs) :-
    (   pkg_index(Cat, T)
    ->  (   tree_lookup(T, Name, All)
        ->  filter_satisfies(All, C, Vs)
        ;   Vs = []
        )
    ;   packages(Cat, Ps),
        matching_versions(Ps, Name, C, Vs)
    ).

filter_satisfies([], _C, []).
filter_satisfies([V|Vs], C, Out) :-
    (   satisfies(V, C)
    ->  Out = [V|Os]
    ;   Out = Os
    ),
    filter_satisfies(Vs, C, Os).

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
% Per-call catalog index (G1a + G1b)
%
% resolve/3 and resolve_layered/3 wrap the catalog in icat/3, which carries
% two balanced lookup trees: depends rows grouped by Name-Ver, and package
% versions grouped by Name. Every accessor has a delegating icat/3 clause,
% so the rest of the file cannot tell the difference; collect_deps/4 and
% matching_versions_in/4 take the tree when it is there and the historical
% full scan when it is not.
%
% Answer-preserving because both forms compute the same list in the same
% order: rows are tagged with their catalog position before sort/2, so no
% two keys compare equal, nothing is dropped, and within one key the rows
% stay in catalog order. See docs/proposals/RESOLVER_PRUNING_DESIGN.md §2.
%
% Pure: no assert/retract. The index is a term built per call and dropped
% with it. icat/3 never escapes and is never accepted as input.
% ---------------------------------------------------------------------------

index_catalog(Cat, Out) :-
    depends_list(Cat, Ds),
    packages(Cat, Ps),
    (   worth_indexing(Ds, Ps)
    ->  key_dep_rows(Ds, 0, KDs),
        sort(KDs, SDs),
        group_keyed(SDs, GDs),
        list_to_tree(GDs, DepT),
        key_pkg_rows(Ps, 0, KPs),
        sort(KPs, SPs),
        group_keyed(SPs, GPs),
        list_to_tree(GPs, PkgT),
        Out = icat(Cat, DepT, PkgT)
    ;   Out = Cat
    ).

% Size threshold. Both branches are the same relation, so this is a cost
% decision, not a semantic one: the index costs ~45 interpreted
% instructions per catalog row to build and pays for itself after about
% two scans, so tiny catalogs are left on the historical scan path.
index_threshold(64).

worth_indexing(Ds, Ps) :-
    index_threshold(N),
    (   long_enough(Ds, N)
    ->  true
    ;   long_enough(Ps, N)
    ).

% length(L, Len), Len >= N without walking past N cells.
long_enough([_|T], N) :-
    (   N =< 1
    ->  true
    ;   N1 is N - 1,
        long_enough(T, N1)
    ).

% (Name-Ver)-Pos-Req: Pos is unique, so sort/2 on this shape drops nothing
% and orders equal keys by ascending Pos, i.e. catalog-list order.
key_dep_rows([], _I, []).
key_dep_rows([depends(N, V, D, C)|Rest], I, [(N-V)-I-Req|Ks]) :-
    dep_to_req(D, C, Req),
    I1 is I + 1,
    key_dep_rows(Rest, I1, Ks).

key_pkg_rows([], _I, []).
key_pkg_rows([package(N, V)|Rest], I, [N-I-V|Ks]) :-
    I1 is I + 1,
    key_pkg_rows(Rest, I1, Ks).

group_keyed([], []).
group_keyed([K-_-X|Rest], [K-[X|Xs]|Gs]) :-
    same_key(Rest, K, Xs, Rest1),
    group_keyed(Rest1, Gs).

same_key([], _K, [], []).
same_key([K2-I-X|Rest], K, Xs, Rest1) :-
    (   K2 == K
    ->  Xs = [X|Xs1],
        same_key(Rest, K, Xs1, Rest1)
    ;   Xs = [],
        Rest1 = [K2-I-X|Rest]
    ).

list_to_tree(Pairs, Tree) :-
    length(Pairs, N),
    build_tree(N, Pairs, Tree, []).

build_tree(N, Pairs, Tree, Rest) :-
    (   N =:= 0
    ->  Tree = t,
        Rest = Pairs
    ;   NL is (N - 1) // 2,
        NR is N - 1 - NL,
        build_tree(NL, Pairs, L, [K-V|Mid]),
        build_tree(NR, Mid, R, Rest),
        Tree = t(L, K, V, R)
    ).

% Fails on the empty tree `t` and on a key that is not present.
tree_lookup(t(L, K, V, R), Key, Val) :-
    compare(Ord, Key, K),
    (   Ord = (=)
    ->  Val = V
    ;   Ord = (<)
    ->  tree_lookup(L, Key, Val)
    ;   tree_lookup(R, Key, Val)
    ).

% ---------------------------------------------------------------------------
% Classic / layered resolve
% ---------------------------------------------------------------------------

resolve(Cat0, Requests, Selection) :-
    index_catalog(Cat0, Cat),
    map_requests(Cat, Requests, Pending),
    resolve_pending(classic, Cat, Pending, [], Acc),
    !,
    sort(Acc, Selection).

resolve_layered(Cat0, Requests, Selection) :-
    index_catalog(Cat0, Cat),
    map_requests(Cat, Requests, Pending),
    resolve_pending(layered, Cat, Pending, [], Acc),
    !,
    sort(Acc, Selection).

resolve_pending(_Mode, _Cat, [], Acc, Acc).
resolve_pending(Mode, Cat, [req(Name, C)|Rest], Acc, Sel) :-
    (   Name = alternatives(Alts)
    ->  resolve_alternatives(Mode, Cat, Alts, Rest, Acc, Sel)
    ;   selected_ver(Acc, Name, Ver)
    ->  % commit: a second version of the same name is never added (diamond)
        satisfies(Ver, C),
        resolve_pending(Mode, Cat, Rest, Acc, Sel)
    ;   already_provided(Cat, Acc, Name, C)
    ->  resolve_pending(Mode, Cat, Rest, Acc, Sel)
    ;   pick_need(Mode, Cat, Name, C, Acc, Pkg, Ver, Origin),
        % G2: the conflict test moved ahead of the expansion. It is a test
        % on ground arguments (pick_need/8 binds Pkg and Ver on every arm;
        % Acc is ground), and collect_deps/4 and append/3 are det and always
        % succeed, so permuting it earlier changes neither the solutions nor
        % their order nor the choice points -- it only skips the two goals
        % on the failing path. See RESOLVER_PRUNING_DESIGN.md §2, G2.
        (   Origin = from_base
        ->  collect_deps(Cat, Pkg, Ver, DepReqs),
            append(DepReqs, Rest, More),
            resolve_pending(Mode, Cat, More, Acc, Sel)
        ;   no_acc_conflicts(Cat, Pkg, Ver, Acc),
            collect_deps(Cat, Pkg, Ver, DepReqs),
            append(DepReqs, Rest, More),
            resolve_pending(Mode, Cat, More, [Pkg-Ver|Acc], Sel)
        )
    ).

% Alternatives: first already-satisfied (Acc or, in layered mode, a loaded
% layer) wins without re-selection. Otherwise try each dep/2 in listed
% order, backtracking into later alternatives when a choice dead-ends.
resolve_alternatives(Mode, Cat, Alts, Rest, Acc, Sel) :-
    (   first_alt_already(Mode, Cat, Acc, Alts)
    ->  resolve_pending(Mode, Cat, Rest, Acc, Sel)
    ;   member(dep(N, C), Alts),
        resolve_pending(Mode, Cat, [req(N, C)|Rest], Acc, Sel)
    ).

first_alt_already(_Mode, Cat, Acc, Alts) :-
    member(dep(N, C), Alts),
    already_satisfied(Cat, Acc, N, C),
    !.
first_alt_already(layered, Cat, _Acc, Alts) :-
    member(dep(N, C), Alts),
    layer_satisfies(Cat, N, C),
    !.

already_satisfied(_Cat, Acc, Name, C) :-
    selected_ver(Acc, Name, Ver),
    satisfies(Ver, C).
already_satisfied(Cat, Acc, Name, C) :-
    already_provided(Cat, Acc, Name, C).

already_provided(Cat, Acc, Name, C) :-
    member(P-PV, Acc),
    provides_sat(Cat, P, PV, Name, C).

layer_satisfies(Cat, Name, C) :-
    base_ver(Cat, Name, BV),
    satisfies(BV, C).
layer_satisfies(Cat, Name, C) :-
    base_holds(Cat, Holds),
    member(hold(P, PV, _), Holds),
    provides_sat(Cat, P, PV, Name, C).
layer_satisfies(Cat, Name, C) :-
    layers_list(Cat, Ls),
    member(layer(_, Pkgs), Ls),
    lookup_held(Pkgs, P, PV),
    (   P == Name, satisfies(PV, C)
    ->  true
    ;   provides_sat(Cat, P, PV, Name, C)
    ).

collect_deps(Cat, Name, Ver, Reqs) :-
    (   dep_index(Cat, T)
    ->  (   tree_lookup(T, Name-Ver, Reqs0)
        ->  Reqs = Reqs0
        ;   Reqs = []
        )
    ;   depends_list(Cat, Ds),
        matching_deps(Ds, Name, Ver, Reqs)
    ).

matching_deps([], _Name, _Ver, []).
matching_deps([depends(N, V, D, C)|Rest], Name, Ver, Out) :-
    (   N == Name,
        V == Ver
    ->  dep_to_req(D, C, Req),
        Out = [Req|Rs]
    ;   Out = Rs
    ),
    matching_deps(Rest, Name, Ver, Rs).

dep_to_req(alternatives(Alts), _C, req(alternatives(Alts), any)) :- !.
dep_to_req(D, C, req(D, C)).

% pick/7 kept for the real-package-only path (classic/layered, no provides).
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

% pick_need/8 — real package preferred; else a provider (never the virtual
% name itself). Providers are tried in catalog-list order (stable).
pick_need(classic, Cat, Name, C, _Acc, Name, Ver, from_catalog) :-
    candidates_high_first(Cat, Name, C, Ver).
pick_need(classic, Cat, Name, C, _Acc, Pkg, Ver, from_catalog) :-
    provider_candidate(Cat, Name, C, Pkg, Ver).
pick_need(layered, Cat, Name, C, _Acc, Pkg, Ver, Origin) :-
    (   base_ver(Cat, Name, BV)
    ->  satisfies(BV, C),
        Pkg = Name,
        Ver = BV,
        Origin = from_base
    ;   layer_provider(Cat, Name, C, Pkg, Ver)
    ->  Origin = from_base
    ;   candidates_high_first(Cat, Name, C, Ver)
    ->  Pkg = Name,
        Origin = from_catalog
    ;   provider_candidate(Cat, Name, C, Pkg, Ver),
        Origin = from_catalog
    ).

layer_provider(Cat, Virtual, C, Pkg, Ver) :-
    base_holds(Cat, Holds),
    member(hold(Pkg, Ver, _), Holds),
    provides_sat(Cat, Pkg, Ver, Virtual, C).
layer_provider(Cat, Virtual, C, Pkg, Ver) :-
    layers_list(Cat, Ls),
    member(layer(_, Pkgs), Ls),
    lookup_held(Pkgs, Pkg, Ver),
    provides_sat(Cat, Pkg, Ver, Virtual, C).

provider_candidate(Cat, Virtual, C, Pkg, Ver) :-
    provides_list(Cat, Prs),
    member(Row, Prs),
    provide_row(Row, Pkg, Ver, Virtual, VVer),
    \+ excluded_name(Cat, Pkg),
    provide_satisfies(VVer, C),
    package_in(Cat, Pkg, Ver).

provide_row(provides(P, V, Virt), P, V, Virt, unversioned).
provide_row(provides(P, V, Virt, VV), P, V, Virt, VV).

% Unversioned Provides satisfy only unversioned (`any`) dependencies
% (Debian Policy). Versioned Provides compare VirtualVer against C.
provide_satisfies(unversioned, any).
provide_satisfies(VV, C) :-
    VV \== unversioned,
    satisfies(VV, C).

provides_sat(Cat, Pkg, Ver, Virtual, C) :-
    provides_list(Cat, Prs),
    member(Row, Prs),
    provide_row(Row, Pkg, Ver, Virtual, VVer),
    provide_satisfies(VVer, C).

provides_for(Cat, Virtual, Pkg, Ver, VVer) :-
    provides_list(Cat, Prs),
    member(Row, Prs),
    provide_row(Row, Pkg, Ver, Virtual, VVer).

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

blocked_from(Cat, req(alternatives(Alts), _), Seen, Blocked) :-
    !,
    alt_reasons(Cat, Alts, Seen, Rs),
    Blocked = blocked(alternatives(Rs)).
blocked_from(Cat, req(Name, C), Seen, Blocked) :-
    \+ seen_name(Seen, Name),
    base_ver(Cat, Name, BV),
    \+ satisfies(BV, C),
    Blocked = blocked(Name, needs(C), base_has(BV)).
blocked_from(Cat, req(Name, C), Seen, Blocked) :-
    \+ seen_name(Seen, Name),
    virtual_provider_ceilings(Cat, Name, C, Reasons),
    Reasons \== [],
    Blocked = blocked(Name, needs(C), providers(Reasons)).
blocked_from(Cat, req(Name, C), Seen, Blocked) :-
    \+ seen_name(Seen, Name),
    walk_pkg_for_blocked(Cat, Name, C, Pkg, Ver),
    collect_deps(Cat, Pkg, Ver, DepReqs),
    member(Dep, DepReqs),
    blocked_from(Cat, Dep, [Name|Seen], Blocked).

blocked_acc(_Cat, req(Name, _C), Seen, Acc, Acc) :-
    atom(Name),
    seen_name(Seen, Name), !.
blocked_acc(Cat, req(alternatives(Alts), _), Seen, Acc0, Acc) :-
    !,
    alt_reasons(Cat, Alts, Seen, Rs),
    Acc = [blocked(alternatives(Rs))|Acc0].
blocked_acc(Cat, req(Name, C), Seen, Acc0, Acc) :-
    (   base_ver(Cat, Name, BV),
        \+ satisfies(BV, C)
    ->  Acc1 = [blocked(Name, needs(C), base_has(BV))|Acc0]
    ;   virtual_provider_ceilings(Cat, Name, C, Reasons),
        Reasons \== []
    ->  Acc1 = [blocked(Name, needs(C), providers(Reasons))|Acc0]
    ;   Acc1 = Acc0
    ),
    (   walk_pkg_for_blocked(Cat, Name, C, Pkg, Ver)
    ->  collect_deps(Cat, Pkg, Ver, DepReqs),
        blocked_acc_list(Cat, DepReqs, [Name|Seen], Acc1, Acc)
    ;   Acc = Acc1
    ).

walk_pkg_for_blocked(Cat, Name, C, Name, Ver) :-
    layered_walk_ver(Cat, Name, C, Ver).
walk_pkg_for_blocked(Cat, Name, C, Pkg, Ver) :-
    layer_provider(Cat, Name, C, Pkg, Ver).
walk_pkg_for_blocked(Cat, Name, C, Pkg, Ver) :-
    provider_candidate(Cat, Name, C, Pkg, Ver).

% Held providers of Virtual whose provided version (or unversioned
% Provides against a versioned dep) does not satisfy C.
virtual_provider_ceilings(Cat, Virtual, C, Reasons) :-
    \+ package_in_name(Cat, Virtual),
    findall(blocked(P, needs(C), base_has(BV)), (
        provides_for(Cat, Virtual, P, BV, VVer),
        base_ver(Cat, P, BV),
        \+ provide_satisfies(VVer, C)
    ), Reasons).

package_in_name(Cat, Name) :-
    packages(Cat, Ps),
    member(package(Name, _), Ps).

alt_reasons(_Cat, [], _Seen, []).
alt_reasons(Cat, [dep(N, C)|Rest], Seen, [alt(N, Reason)|Rs]) :-
    (   explain_alt(Cat, N, C, Seen, Reason)
    ->  true
    ;   Reason = unsatisfiable
    ),
    alt_reasons(Cat, Rest, Seen, Rs).

explain_alt(Cat, N, C, Seen, Reason) :-
    (   blocked_from(Cat, req(N, C), Seen, Reason)
    ->  true
    ;   \+ pick_need(layered, Cat, N, C, [], _, _, _)
    ->  Reason = unsatisfiable
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
    ->  findall(D, follow_dep_name(Cat, Name, Ver, Sel, D), Ds0),
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
    ;       findall(D-DV, (
            follow_dep_name(Cat, Name, Ver, Inst, D),
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
        dep_breaks_need(Acc, D, C, CBroken)
    ->  COut = CBroken
    ;   dep_breaks(Rest, N, V, Acc, COut)
    ).

dep_breaks_need(Acc, alternatives(Alts), _C, COut) :-
    !,
    member(dep(D, COut), Alts),
    selected_ver(Acc, D, MV),
    \+ satisfies(MV, COut),
    \+ (member(dep(D2, C2), Alts), selected_ver(Acc, D2, MV2), satisfies(MV2, C2)).
dep_breaks_need(Acc, D, C, C) :-
    selected_ver(Acc, D, MV),
    \+ satisfies(MV, C).

pick_repair(Cat, Name, Acc, NewV) :-
    candidates_high_first(Cat, Name, any, NewV),
    repairs_moving(Cat, Name, NewV, Acc).

repairs_moving(Cat, Name, NewV, Acc) :-
    collect_deps(Cat, Name, NewV, Reqs),
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
        dep_targets(Cat, N, V, Target, C),
        (   Target == Pkg,
            tight_constraint(C)
        ;   base_ver(Cat, Pkg, PV),
            provides_sat(Cat, Pkg, PV, Target, C)
        )
    ->  true
    ;   tight_rev_in(Rest, Cat, Pkg)
    ).

dep_targets(Cat, N, V, Target, C) :-
    depends_in(Cat, N, V, Raw, C0),
    (   Raw = alternatives(Alts)
    ->  member(dep(Target, C), Alts)
    ;   Target = Raw, C = C0
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
    (   dep_mentions(D, Pkg)
    ->  Acc1 = [N-V|Acc0]
    ;   Acc1 = Acc0
    ),
    direct_on(Rest, Pkg, Acc1, Acc).

dep_mentions(alternatives(Alts), Pkg) :-
    !,
    member(dep(Pkg, _), Alts).
dep_mentions(D, Pkg) :-
    D == Pkg.

% Follow a dep to a name that is present in Sel (real or selected provider).
follow_dep_name(Cat, Name, Ver, Sel, D) :-
    depends_in(Cat, Name, Ver, Raw, C),
    follow_raw_dep(Raw, C, Sel, D).

follow_raw_dep(alternatives(Alts), _C, Sel, D) :-
    !,
    member(dep(N, _), Alts),
    member(N-_, Sel),
    D = N.
follow_raw_dep(D, _C, Sel, D) :-
    atom(D),
    member(D-_, Sel).

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
