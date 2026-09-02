:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% store_diff_runner.pl -- SWI oracle for store-backed resolution.
% Reads JSONL {env, query, args} ; writes one result object per line.
% The P/2 JSONL under STORE_DIR must already be loaded... we load it here.
%
%   STORE_DIR=... swipl -q -g main -t halt examples/pkg_resolver/store_diff_runner.pl \
%     < cases.jsonl > swi.jsonl

:- module(store_diff_runner, [main/0]).

:- use_module(library(http/json)).
:- use_module(resolver_store).

main :-
    prompt(_, ''),
    getenv('STORE_DIR', Dir),
    load_p2_jsonl(Dir),
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

json_to_env(Row, env(Id, B, I, R, L, E, A)) :-
    (   get_dict(catalog_id, Row, Id0) -> json_atom(Id0, Id)
    ;   get_dict(env, Row, Env0), get_dict(catalog_id, Env0, Id0) -> json_atom(Id0, Id)
    ;   Id = default
    ),
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
json_ver([X, Y, Z], v(X, Y, Z)).
json_atom(A, A) :- atom(A), !.
json_atom(S, A) :- string(S), atom_string(A, S).

run_query(resolve, Env, Args, Exp) :-
    json_to_reqs(Args, Reqs),
    (   resolve_store(Env, Reqs, Sel) -> sel_json(Sel, Js), Exp = _{ok: Js} ; Exp = _{fail: true} ).
run_query(resolve_layered, Env, Args, Exp) :-
    json_to_reqs(Args, Reqs),
    (   resolve_layered_store(Env, Reqs, Sel) -> sel_json(Sel, Js), Exp = _{ok: Js} ; Exp = _{fail: true} ).
run_query(explain_blocked, Env, Args, Exp) :-
    json_to_req(Args, Req),
    explain_blocked_list_store(Env, Req, List),
    blocked_list_json(List, Js),
    Exp = _{ok: Js}.
run_query(layer_closure, Env, Args, Exp) :-
    json_to_req(Args, Req),
    (   layer_closure_store(Env, Req, Layer) -> sel_json(Layer, Js), Exp = _{ok: Js} ; Exp = _{fail: true} ).
run_query(removal_orphans, Env, Args, Exp) :-
    json_to_pkg(Args, Pkg),
    removal_orphans_store(Env, Pkg, Orphans),
    sel_json(Orphans, Js),
    Exp = _{ok: Js}.
run_query(safe_upgrade, Env, Args, Exp) :-
    json_to_pkg_ver(Args, Pkg, Ver),
    safe_upgrade_store(Env, Pkg, Ver, Verdict),
    verdict_json(Verdict, Js),
    Exp = _{ok: Js}.
run_query(upgrade_set, Env, Args, Exp) :-
    json_to_pkg_ver(Args, Pkg, Ver),
    upgrade_set_result_store(Env, Pkg, Ver, R),
    upgrade_json(R, Exp).
run_query(freeze_audit, Env, _, Exp) :-
    freeze_audit_store(Env, Audit),
    maplist(audit_json, Audit, Js),
    Exp = _{ok: Js}.
run_query(dependents, Env, Args, Exp) :-
    json_to_pkg(Args, Pkg),
    dependents_store(Env, Pkg, Deps),
    sel_json(Deps, Js),
    Exp = _{ok: Js}.
run_query(dependents_installed, Env, Args, Exp) :-
    json_to_pkg(Args, Pkg),
    dependents_installed_store(Env, Pkg, Deps),
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
json_constraint_op(lt, D, lt(V)) :- json_ver(D.v, V).
json_constraint_op(range, D, range(Lo, Hi)) :- json_ver(D.lo, Lo), json_ver(D.hi, Hi).

sel_json([], []).
sel_json([N-V|Rest], [[NS, VJ]|Js]) :-
    atom_string(N, NS), ver_json(V, VJ), sel_json(Rest, Js).
ver_json(v(A, B, C), [A, B, C]).

blocked_list_json([], []).
blocked_list_json([blocked(N, needs(C), base_has(V))|Rest],
                  [_{name: NS, needs: CJ, base_has: VJ}|Js]) :-
    atom_string(N, NS), constraint_json(C, CJ), ver_json(V, VJ),
    blocked_list_json(Rest, Js).

constraint_json(any, any).
constraint_json(eq(V), _{op: "eq", v: J}) :- ver_json(V, J).
constraint_json(gte(V), _{op: "gte", v: J}) :- ver_json(V, J).
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
