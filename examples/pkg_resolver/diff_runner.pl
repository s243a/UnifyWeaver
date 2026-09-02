:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% diff_runner.pl -- SWI oracle for the pkg_resolver differential.
% Reads JSONL cases (catalog + query + args) on stdin; writes one JSON
% result object per line (same shape as dump_corpus expected / the
% wamjs shim).
%
%   swipl -q -g main -t halt examples/pkg_resolver/diff_runner.pl \
%     < cases.jsonl > swi.jsonl

:- module(pkg_diff_runner, [main/0]).

:- use_module(library(http/json)).
:- use_module(resolver).

main :-
    prompt(_, ''),
    read_line_to_string(user_input, Line),
    process_lines(Line).

process_lines(end_of_file) :- !.
process_lines("") :-
    read_line_to_string(user_input, Next),
    process_lines(Next).
process_lines(Line) :-
    atom_json_dict(Line, Row, [value_string_as(atom)]),
    run_row(Row, Exp),
    json_write_dict(current_output, Exp, [width(0), true(true), false(false)]),
    nl,
    read_line_to_string(user_input, Next),
    process_lines(Next).

run_row(Row, Exp) :-
    json_to_catalog(Row.catalog, Cat),
    get_dict(query, Row, Q0),
    json_atom(Q0, Q),
    run_query(Q, Cat, Row.args, Exp).

run_query(resolve, Cat, Args, Exp) :-
    json_to_reqs(Args, Reqs),
    (   resolve(Cat, Reqs, Sel)
    ->  sel_json(Sel, Js), Exp = _{ok: Js}
    ;   Exp = _{fail: true}
    ).
run_query(resolve_layered, Cat, Args, Exp) :-
    json_to_reqs(Args, Reqs),
    (   resolve_layered(Cat, Reqs, Sel)
    ->  sel_json(Sel, Js), Exp = _{ok: Js}
    ;   Exp = _{fail: true}
    ).
run_query(explain_blocked, Cat, Args, Exp) :-
    json_to_req(Args, Req),
    explain_blocked_list(Cat, Req, List),
    blocked_list_json(List, Js),
    Exp = _{ok: Js}.
run_query(layer_closure, Cat, Args, Exp) :-
    json_to_req(Args, Req),
    (   layer_closure(Cat, Req, Layer)
    ->  sel_json(Layer, Js), Exp = _{ok: Js}
    ;   Exp = _{fail: true}
    ).
run_query(removal_orphans, Cat, Args, Exp) :-
    json_to_pkg(Args, Pkg),
    removal_orphans(Cat, Pkg, Orphans),
    sel_json(Orphans, Js),
    Exp = _{ok: Js}.
run_query(safe_upgrade, Cat, Args, Exp) :-
    json_to_pkg_ver(Args, Pkg, Ver),
    safe_upgrade(Cat, Pkg, Ver, Verdict),
    verdict_json(Verdict, Js),
    Exp = _{ok: Js}.
run_query(upgrade_set, Cat, Args, Exp) :-
    json_to_pkg_ver(Args, Pkg, Ver),
    upgrade_set_result(Cat, Pkg, Ver, R),
    upgrade_json(R, Exp).
run_query(freeze_audit, Cat, _, Exp) :-
    freeze_audit(Cat, Audit),
    maplist(audit_json, Audit, Js),
    Exp = _{ok: Js}.
run_query(dependents, Cat, Args, Exp) :-
    json_to_pkg(Args, Pkg),
    dependents(Cat, Pkg, Deps),
    sel_json(Deps, Js),
    Exp = _{ok: Js}.
run_query(dependents_installed, Cat, Args, Exp) :-
    json_to_pkg(Args, Pkg),
    dependents_installed(Cat, Pkg, Deps),
    sel_json(Deps, Js),
    Exp = _{ok: Js}.

json_to_catalog(D, Cat) :-
    maplist(json_pkg, D.packages, Ps),
    maplist(json_dep, D.depends, Ds),
    maplist(json_conf, D.conflicts, Cs),
    maplist(json_hold, D.base, Bs),
    maplist(json_pair, D.installed, Is),
    maplist(json_atom, D.requested, Rs),
    (   get_dict(layers, D, L0) -> maplist(json_layer, L0, Ls) ; Ls = [] ),
    (   get_dict(excluded, D, E0) -> maplist(json_atom, E0, Es) ; Es = [] ),
    (   get_dict(aliases, D, A0) -> maplist(json_alias, A0, As) ; As = [] ),
    (   Ls == [], Es == [], As == []
    ->  Cat = catalog(Ps, Ds, Cs, Bs, Is, Rs)
    ;   Cat = catalog(Ps, Ds, Cs, Bs, Is, Rs, Ls, Es, As)
    ).

json_pkg([N, V], package(Name, Ver)) :-
    json_atom(N, Name), json_ver(V, Ver).
json_dep([N, V, D, C], depends(Name, Ver, Dep, Con)) :-
    json_atom(N, Name), json_ver(V, Ver), json_atom(D, Dep),
    json_constraint(C, Con).
json_conf([N, V, O], conflicts(Name, Ver, Other)) :-
    json_atom(N, Name), json_ver(V, Ver), json_atom(O, Other).
json_pair([N, V], Name-Ver) :-
    json_atom(N, Name), json_ver(V, Ver).
json_hold([N, V], Name-Ver) :-
    json_atom(N, Name), json_ver(V, Ver).
json_hold([N, V, R], base(Name-Ver, Reason)) :-
    json_atom(N, Name), json_ver(V, Ver), json_atom(R, Reason).
json_layer(D, layer(Name, Pkgs)) :-
    is_dict(D),
    json_atom(D.name, Name),
    maplist(json_hold, D.packages, Pkgs).
json_alias([A, C], alias(Alias, Canon)) :-
    json_atom(A, Alias), json_atom(C, Canon).

json_to_pkg_ver([P, V], Pkg, Ver) :-
    json_atom(P, Pkg), json_ver(V, Ver).

json_ver([A, B, C], v(A, B, C)).

json_atom(A, A) :- atom(A), !.
json_atom(S, A) :- string(S), atom_string(A, S).

json_constraint(any, any) :- !.
json_constraint("any", any) :- !.
json_constraint(D, C) :-
    is_dict(D),
    json_atom(D.op, Op),
    json_constraint_op(Op, D, C).

json_constraint_op(eq, D, eq(V)) :- json_ver(D.v, V).
json_constraint_op(gte, D, gte(V)) :- json_ver(D.v, V).
json_constraint_op(lt, D, lt(V)) :- json_ver(D.v, V).
json_constraint_op(range, D, range(Lo, Hi)) :-
    json_ver(D.lo, Lo), json_ver(D.hi, Hi).

json_to_reqs(List, Reqs) :-
    maplist(json_to_req, List, Reqs).

json_to_req(D, req(Name, C)) :-
    is_dict(D),
    !,
    json_atom(D.req, Name),
    json_constraint(D.constraint, C).
json_to_req(A, Name) :-
    json_atom(A, Name).

json_to_pkg(A, Name) :- json_atom(A, Name).

sel_json([], []).
sel_json([N-V|Rest], [[NS, VJ]|Js]) :-
    atom_string(N, NS), ver_json(V, VJ),
    sel_json(Rest, Js).

ver_json(v(A, B, C), [A, B, C]).

blocked_list_json([], []).
blocked_list_json([blocked(N, needs(C), base_has(V))|Rest],
                  [_{name: NS, needs: CJ, base_has: VJ}|Js]) :-
    atom_string(N, NS),
    constraint_json(C, CJ),
    ver_json(V, VJ),
    blocked_list_json(Rest, Js).

verdict_json(safe(cost(R)), _{cost: RS, verdict: "safe"}) :-
    atom_string(R, RS).
verdict_json(coordinated(Set), _{set: SJ, verdict: "coordinated"}) :-
    sel_json(Set, SJ).
verdict_json(unsafe(modified), _{reason: "modified", verdict: "unsafe"}).
verdict_json(no_candidate, _{verdict: "no_candidate"}).

upgrade_json(ok(Set), _{ok: Js}) :-
    sel_json(Set, Js).
upgrade_json(no_candidate, _{fail: true}).
upgrade_json(blocked(N, needs(C), base_has(V)), _{ok: _{blocked: BJ}}) :-
    blocked_list_json([blocked(N, needs(C), base_has(V))], [BJ]).

audit_json(audit(N, over_frozen), _{kind: "over_frozen", name: NS}) :-
    atom_string(N, NS).
audit_json(audit(N, suggest(R)), _{kind: "suggest", name: NS, reason: RS}) :-
    atom_string(N, NS), atom_string(R, RS).
audit_json(audit(N, held(R)), _{kind: "held", name: NS, reason: RS}) :-
    atom_string(N, NS), atom_string(R, RS).

constraint_json(any, any).
constraint_json(eq(V), _{op: "eq", v: J}) :- ver_json(V, J).
constraint_json(gte(V), _{op: "gte", v: J}) :- ver_json(V, J).
constraint_json(lt(V), _{op: "lt", v: J}) :- ver_json(V, J).
constraint_json(range(Lo, Hi), _{op: "range", lo: LJ, hi: HJ}) :-
    ver_json(Lo, LJ), ver_json(Hi, HJ).
