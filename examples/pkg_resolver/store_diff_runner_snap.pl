:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% store_diff_runner_snap.pl -- SWI differential runner for the snapshot-aware
% (deduped) store adapter resolver_store_snapshot.pl. Mirrors
% store_diff_runner.pl exactly, but builds an 8-ary env_snap (SnapId == PoolId
% == the row's catalog_id at N=1) and dispatches to the *_snap predicates. At
% N=1 the emitted result stream must be byte-identical to the frozen runner's.
%
%   STORE_DIR=... swipl -q -g main -t halt examples/pkg_resolver/store_diff_runner_snap.pl \
%     < cases.jsonl > snap.jsonl

:- module(store_diff_runner_snap, [main/0]).

:- use_module(library(http/json)).
:- use_module(resolver_store_snapshot).

main :-
    prompt(_, ''),
    getenv('STORE_DIR', Dir),
    % N=1 default: pool + membership from the one frozen store dir. N>1: point
    % UW_POOL_DIR at the shared pool dir (dep/conflict/revdep/provide) while
    % STORE_DIR holds the per-snapshot membership pkg.jsonl.
    ( getenv('UW_POOL_DIR', PoolDir) -> true ; PoolDir = Dir ),
    load_p2_snap(PoolDir, Dir),
    read_line_to_string(user_input, Line),
    process_lines(Line).

process_lines(end_of_file) :- !.
process_lines("") :-
    read_line_to_string(user_input, Next),
    process_lines(Next).
process_lines(Line) :-
    atom_json_dict(Line, Row, [value_string_as(atom)]),
    json_to_env(Row, Env),
    get_dict(query, Row, Q0),
    json_atom(Q0, Q),
    run_query(Q, Env, Row.args, Exp),
    json_write_dict(current_output, Exp, [width(0), true(true), false(false)]),
    nl,
    read_line_to_string(user_input, Next),
    process_lines(Next).

json_to_env(Row, env_snap(SnapId, PoolId, B, I, R, L, E, A)) :-
    (   get_dict(catalog_id, Row, Id0) -> json_atom(Id0, Id)
    ;   get_dict(env, Row, Env0), get_dict(catalog_id, Env0, Id0) -> json_atom(Id0, Id)
    ;   Id = default
    ),
    % N=1 default: both ids are the catalog id. N>1: UW_SNAP_ID selects the
    % membership snapshot, UW_POOL_ID the shared package pool.
    ( getenv('UW_SNAP_ID', SnapId) -> true ; SnapId = Id ),
    ( getenv('UW_POOL_ID', PoolId) -> true ; PoolId = Id ),
    (   get_dict(env, Row, D) -> true ; D = Row ),
    (   get_dict(base, D, B0) -> maplist(json_hold, B0, B) ; B = [] ),
    (   get_dict(installed, D, I0) -> maplist(json_pair, I0, I) ; I = [] ),
    (   get_dict(requested, D, R0) -> maplist(json_atom, R0, R) ; R = [] ),
    (   get_dict(layers, D, L0) -> maplist(json_layer, L0, L) ; L = [] ),
    (   get_dict(excluded, D, E0) -> maplist(json_atom, E0, E) ; E = [] ),
    (   get_dict(aliases, D, A0) -> maplist(json_alias, A0, A) ; A = [] ).

json_hold([N, V], Name-Ver) :-
    json_atom(N, Name), json_ver(V, Ver).
json_hold([N, V, R], base(Name-Ver, Reason)) :-
    json_atom(N, Name), json_ver(V, Ver), json_atom(R, Reason).
json_pair([N, V], Name-Ver) :-
    json_atom(N, Name), json_ver(V, Ver).
json_layer(D, layer(Name, Pkgs)) :-
    is_dict(D),
    json_atom(D.name, Name),
    maplist(json_hold, D.packages, Pkgs).
json_alias([A, C], alias(Alias, Canon)) :-
    json_atom(A, Alias), json_atom(C, Canon).
json_ver([X, Y, Z], v(X, Y, Z)) :- !.
json_ver(D, deb(E, Up, Rev)) :-
    is_dict(D),
    get_dict(deb, D, [E, UpJ, RevJ]),
    maplist(json_seg, UpJ, Up),
    maplist(json_seg, RevJ, Rev).

json_seg([Order, N], s(Codes, N)) :-
    (   atom(Order)
    ->  atom_codes(Order, Codes)
    ;   string_codes(Order, Codes)
    ).
json_atom(A, A) :- atom(A), !.
json_atom(S, A) :- string(S), atom_string(A, S).

run_query(resolve, Env, Args, Exp) :-
    json_to_reqs(Args, Reqs),
    (   resolve_snap(Env, Reqs, Sel) -> sel_json(Sel, Js), Exp = _{ok: Js} ; Exp = _{fail: true} ).
run_query(resolve_layered, Env, Args, Exp) :-
    json_to_reqs(Args, Reqs),
    (   resolve_layered_snap(Env, Reqs, Sel) -> sel_json(Sel, Js), Exp = _{ok: Js} ; Exp = _{fail: true} ).
run_query(explain_blocked, Env, Args, Exp) :-
    json_to_req(Args, Req),
    explain_blocked_list_snap(Env, Req, List),
    blocked_list_json(List, Js),
    Exp = _{ok: Js}.
run_query(layer_closure, Env, Args, Exp) :-
    json_to_req(Args, Req),
    (   layer_closure_snap(Env, Req, Layer) -> sel_json(Layer, Js), Exp = _{ok: Js} ; Exp = _{fail: true} ).
run_query(removal_orphans, Env, Args, Exp) :-
    json_to_pkg(Args, Pkg),
    removal_orphans_snap(Env, Pkg, Orphans),
    sel_json(Orphans, Js),
    Exp = _{ok: Js}.
run_query(safe_upgrade, Env, Args, Exp) :-
    json_to_pkg_ver(Args, Pkg, Ver),
    safe_upgrade_snap(Env, Pkg, Ver, Verdict),
    verdict_json(Verdict, Js),
    Exp = _{ok: Js}.
run_query(upgrade_set, Env, Args, Exp) :-
    json_to_pkg_ver(Args, Pkg, Ver),
    upgrade_set_result_snap(Env, Pkg, Ver, R),
    upgrade_json(R, Exp).
run_query(freeze_audit, Env, _, Exp) :-
    freeze_audit_snap(Env, Audit),
    maplist(audit_json, Audit, Js),
    Exp = _{ok: Js}.
run_query(dependents, Env, Args, Exp) :-
    json_to_pkg(Args, Pkg),
    dependents_snap(Env, Pkg, Deps),
    sel_json(Deps, Js),
    Exp = _{ok: Js}.
run_query(dependents_installed, Env, Args, Exp) :-
    json_to_pkg(Args, Pkg),
    dependents_installed_snap(Env, Pkg, Deps),
    sel_json(Deps, Js),
    Exp = _{ok: Js}.

json_to_reqs(List, Reqs) :- maplist(json_to_req, List, Reqs).
json_to_req(D, req(Name, C)) :-
    is_dict(D), !, json_atom(D.req, Name), json_constraint(D.constraint, C).
json_to_req(A, Name) :- json_atom(A, Name).
json_to_pkg(A, Name) :- json_atom(A, Name).
json_to_pkg_ver([P, V], Pkg, Ver) :- json_atom(P, Pkg), json_ver(V, Ver).

json_constraint(any, any) :- !.
json_constraint("any", any) :- !.
json_constraint(D, C) :-
    is_dict(D), json_atom(D.op, Op), json_constraint_op(Op, D, C).
json_constraint_op(eq, D, eq(V)) :- json_ver(D.v, V).
json_constraint_op(gte, D, gte(V)) :- json_ver(D.v, V).
json_constraint_op(lte, D, lte(V)) :- json_ver(D.v, V).
json_constraint_op(gt, D, gt(V)) :- json_ver(D.v, V).
json_constraint_op(lt, D, lt(V)) :- json_ver(D.v, V).
json_constraint_op(range, D, range(Lo, Hi)) :- json_ver(D.lo, Lo), json_ver(D.hi, Hi).

sel_json([], []).
sel_json([N-V|Rest], [[NS, VJ]|Js]) :-
    atom_string(N, NS), ver_json(V, VJ), sel_json(Rest, Js).
ver_json(v(A, B, C), [A, B, C]) :- !.
ver_json(deb(E, Up, Rev), _{deb: [E, UJ, RJ]}) :-
    maplist(seg_json, Up, UJ),
    maplist(seg_json, Rev, RJ).

seg_json(s(Codes, N), [S, N]) :-
    string_codes(S, Codes).

blocked_list_json([], []).
blocked_list_json([blocked(N, needs(C), base_has(V))|Rest],
                  [_{name: NS, needs: CJ, base_has: VJ}|Js]) :-
    atom(N),
    !,
    atom_string(N, NS), constraint_json(C, CJ), ver_json(V, VJ),
    blocked_list_json(Rest, Js).
blocked_list_json([blocked(N, needs(C), providers(Ps))|Rest],
                  [_{name: NS, needs: CJ, providers: PJ}|Js]) :-
    !,
    atom_string(N, NS), constraint_json(C, CJ),
    blocked_list_json(Ps, PJ),
    blocked_list_json(Rest, Js).
blocked_list_json([blocked(alternatives(Rs))|Rest],
                  [_{alternatives: AJ}|Js]) :-
    !,
    maplist(alt_reason_json, Rs, AJ),
    blocked_list_json(Rest, Js).

alt_reason_json(alt(N, unsatisfiable), _{dep: NS, reason: "unsatisfiable"}) :-
    atom_string(N, NS).
alt_reason_json(alt(N, B), _{dep: NS, reason: BJ}) :-
    atom_string(N, NS),
    blocked_list_json([B], [BJ]).

constraint_json(any, any).
constraint_json(eq(V), _{op: "eq", v: J}) :- ver_json(V, J).
constraint_json(gte(V), _{op: "gte", v: J}) :- ver_json(V, J).
constraint_json(lte(V), _{op: "lte", v: J}) :- ver_json(V, J).
constraint_json(gt(V), _{op: "gt", v: J}) :- ver_json(V, J).
constraint_json(lt(V), _{op: "lt", v: J}) :- ver_json(V, J).
constraint_json(range(Lo, Hi), _{op: "range", lo: LJ, hi: HJ}) :-
    ver_json(Lo, LJ), ver_json(Hi, HJ).

verdict_json(safe(cost(R)), _{cost: RS, verdict: "safe"}) :- atom_string(R, RS).
verdict_json(coordinated(Set), _{set: SJ, verdict: "coordinated"}) :- sel_json(Set, SJ).
verdict_json(unsafe(modified), _{reason: "modified", verdict: "unsafe"}).
verdict_json(no_candidate, _{verdict: "no_candidate"}).

upgrade_json(ok(Set), _{ok: Js}) :- sel_json(Set, Js).
upgrade_json(no_candidate, _{fail: true}).
upgrade_json(blocked(N, needs(C), base_has(V)), _{ok: _{blocked: BJ}}) :-
    blocked_list_json([blocked(N, needs(C), base_has(V))], [BJ]).

audit_json(audit(N, over_frozen), _{kind: "over_frozen", name: NS}) :- atom_string(N, NS).
audit_json(audit(N, suggest(R)), _{kind: "suggest", name: NS, reason: RS}) :-
    atom_string(N, NS), atom_string(R, RS).
audit_json(audit(N, held(R)), _{kind: "held", name: NS, reason: RS}) :-
    atom_string(N, NS), atom_string(R, RS).
