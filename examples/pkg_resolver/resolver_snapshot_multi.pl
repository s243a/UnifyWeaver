:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% resolver_snapshot_multi.pl -- cross-snapshot "newest-compatible" driver.
%
% Answers the owner's §2g/§2h headline multi-snapshot question:
%
%   "The newest version of a package P that is compatible with the currently
%    locked set of packages, considering all versions available across N
%    repository snapshots."
%
%   newest_compatible_snap(EnvSnap, Pkg, Constraint, Result)
%
% ---------------------------------------------------------------------------
% Design (thin driver ABOVE the frozen resolver -- no resolution reinvented)
% ---------------------------------------------------------------------------
%
% The validated snapshot resolver resolver_store_snapshot.pl already selects
% candidate versions NEWEST-FIRST with backtracking: candidate_versions_snap
% sorts descending (sort_versions_desc_snap) and resolve_layered_snap tries the
% next-older version whenever a newer one cannot be fully resolved. So "newest
% compatible" is nothing more than an ordinary layered resolve over the UNION
% of versions, with the locked set held fixed:
%
%  * UNION MEMBERSHIP. The caller loads EVERY pool package version into
%    store_pkg keyed by the PoolId, and we run the resolve with SnapId := PoolId
%    (env_snap(PoolId, PoolId, ...)). Then store_key_snap and store_key_pool
%    both resolve to the PoolId, so candidate_versions_snap sees every version
%    across all snapshots (the union) and pool_dep/pool_conflict are the shared
%    pool. (See load_union_snap/1.)
%
%  * THE LOCKED SET. Represented as PINNED requests req(L, eq(LV)) -- one per
%    locked package -- resolved from the union catalog so each lands in the
%    resolver's Acc accumulator.
%
%    Why pinned requests and NOT base holds? The task's design memo suggests
%    carrying the locked set in Base (base_holds). That IS enough for the
%    DEPENDENCY-vs-locked case (pick_need_snap's from_base branch requires the
%    candidate's dep constraint to be satisfied by the held version). But it is
%    NOT enough for the CONFLICT-TABLE-vs-locked case: the frozen control flow's
%    no_acc_conflicts_snap only checks the candidate against packages in Acc,
%    and from_base packages are never placed in Acc (verified against the frozen
%    resolver_store.pl: no_acc_conflicts_store scans Acc only; from_base skips
%    it). Pinning each locked package as req(L, eq(LV)) puts it in Acc, so BOTH
%    directions of conflict (candidate declares conflict with L, or L declares
%    conflict with candidate) are caught, while eq(LV) keeps it immovable -- the
%    locked version is the only one that can be chosen. Locked requests are
%    ordered BEFORE the target so the lock is established before any candidate
%    dependency could otherwise pull a different version of the same name.
%
%  * THE DRIVER. Run resolve_layered_snap(UnionEnv, LockedReqs ++
%    [req(Pkg,Constraint) | Existing], Sel). resolve_layered_snap commits (via
%    its internal cut) to the FIRST complete resolution, and because the target
%    is explored newest-first with backtracking, that first success carries the
%    newest compatible version of Pkg. Read Pkg's version V out of Sel:
%       success -> Result = compatible(V, Sel)
%       failure -> Result = no_candidate
%    A plain resolve therefore suffices; no manual version enumeration is
%    needed. Termination is guaranteed by the finite version set -- when every
%    version conflicts, backtracking exhausts the (small, locked-bounded)
%    candidate list and fails fast.
%
% This module NEVER edits the frozen resolver.pl / resolver_store.pl /
% resolver_store_snapshot.pl. It reuses their exported helpers and copies the
% one tiny frozen wire encoder (pack_conflict/3) that resolver_store does not
% export -- exactly as resolver_store_snapshot.pl itself does.

:- module(resolver_snapshot_multi, [
    newest_compatible_snap/4,
    load_union_snap/1,
    % test-support asserters (build the union pool by hand for M0 scenarios)
    ms_clear/0,
    ms_pkg/3,
    ms_dep/4,
    ms_conflict/4,
    ms_provide/4
]).

:- use_module(resolver_store, [
    pack_key/3,
    pack_ver/2,
    pack_dep/4,
    pack_provide/4
]).
:- use_module(resolver_store_snapshot, [
    resolve_layered_snap/3,
    load_p2_snap/2,
    store_clear_snap/0
]).

% ---------------------------------------------------------------------------
% Driver
% ---------------------------------------------------------------------------

newest_compatible_snap(Env, Pkg, Constraint, Result) :-
    Env = env_snap(_SnapId, PoolId, Base, Installed, Requested, Layers, Excluded, Aliases),
    % Locked set: everything held in Base and in the layers -> pinned reqs.
    locked_pairs(Base, LB),
    locked_pairs(Layers, LL),
    append(LB, LL, Locked),
    locked_reqs(Locked, LockedReqs),
    append(LockedReqs, [req(Pkg, Constraint) | Requested], Reqs),
    % Union env: SnapId := PoolId, and the locked set has moved out of Base /
    % Layers into pinned requests, so both are empty here.
    UnionEnv = env_snap(PoolId, PoolId, [], Installed, [], [], Excluded, Aliases),
    (   resolve_layered_snap(UnionEnv, Reqs, Sel),
        member(Pkg-V, Sel)
    ->  Result = compatible(V, Sel)
    ;   Result = no_candidate
    ).

% Extract Name-Ver pairs from a base/layer hold list (mirrors the shapes
% resolver_store_snapshot's scan_base_holds understands: N-V, base(N-V,R),
% layer(_, Pkgs)).
locked_pairs([], []).
locked_pairs([H|T], Out) :-
    (   H = base(N-V, _R)
    ->  Ps = [N-V]
    ;   H = layer(_L, Pkgs)
    ->  locked_pairs(Pkgs, Ps)
    ;   H = (N-V), atom(N)
    ->  Ps = [N-V]
    ;   Ps = []
    ),
    locked_pairs(T, Rest),
    append(Ps, Rest, Out).

locked_reqs([], []).
locked_reqs([N-V|T], [req(N, eq(V))|R]) :-
    locked_reqs(T, R).

% ---------------------------------------------------------------------------
% Union store loader (reuse the validated JSONL loader)
% ---------------------------------------------------------------------------
%
% For a materialized store, the union pkg.jsonl (every version across snapshots,
% keyed by the PoolId) lives beside the pooled dep/conflict/revdep/provide.jsonl
% in the same directory, so PoolDir == PkgDir. The caller then builds the env
% with SnapId == PoolId.
load_union_snap(PoolDir) :-
    load_p2_snap(PoolDir, PoolDir).

% ---------------------------------------------------------------------------
% Test-support asserters -- build a tiny union pool directly, keeping the
% snapshot facts inside resolver_store_snapshot's module (no double-define).
% ---------------------------------------------------------------------------

ms_clear :-
    store_clear_snap.

% store_pkg is the union membership: several versions may share one PoolId|Name.
ms_pkg(PoolId, Name, Ver) :-
    pack_key(PoolId, Name, Key),
    pack_ver(Ver, VA),
    assertz(resolver_store_snapshot:store_pkg(Key, VA)).

% pool_dep : PoolId|Name -> Ver#Dep#Constraint  (Dep an atom or alternatives/1)
ms_dep(PoolId, Name, Ver, dep(Dep, C)) :-
    pack_key(PoolId, Name, Key),
    pack_dep(Ver, Dep, C, Packed),
    assertz(resolver_store_snapshot:pool_dep(Key, Packed)).

% pool_conflict : PoolId|Name -> Ver#Other  (pack_conflict/3 is unexported in
% resolver_store; the wire format is frozen by the shared JS codec, so this
% 3-line encoder is copied verbatim -- same as resolver_store_snapshot does).
ms_conflict(PoolId, Name, Ver, Other) :-
    pack_key(PoolId, Name, Key),
    pack_conflict(Ver, Other, Packed),
    assertz(resolver_store_snapshot:pool_conflict(Key, Packed)).

% pool_provides : PoolId|Virtual -> Pkg#Ver#VirtualVer
ms_provide(PoolId, Virtual, Pkg, VerSpec) :-
    VerSpec = pv(Ver, VVer),
    pack_key(PoolId, Virtual, Key),
    pack_provide(Pkg, Ver, VVer, Packed),
    assertz(resolver_store_snapshot:pool_provides(Key, Packed)).

pack_conflict(Ver, Other, Atom) :-
    pack_ver(Ver, VA),
    atom_concat(VA, '#', T),
    atom_concat(T, Other, Atom).
