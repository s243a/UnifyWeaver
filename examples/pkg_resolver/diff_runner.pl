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

json_to_catalog(D, catalog(Ps, Ds, Cs, Bs, Is, Rs)) :-
    maplist(json_pkg, D.packages, Ps),
    maplist(json_dep, D.depends, Ds),
    maplist(json_conf, D.conflicts, Cs),
    maplist(json_pair, D.base, Bs),
    maplist(json_pair, D.installed, Is),
    maplist(json_atom, D.requested, Rs).

json_pkg([N, V], package(Name, Ver)) :-
    json_atom(N, Name), json_ver(V, Ver).
json_dep([N, V, D, C], depends(Name, Ver, Dep, Con)) :-
    json_atom(N, Name), json_ver(V, Ver), json_atom(D, Dep),
    json_constraint(C, Con).
json_conf([N, V, O], conflicts(Name, Ver, Other)) :-
    json_atom(N, Name), json_ver(V, Ver), json_atom(O, Other).
json_pair([N, V], Name-Ver) :-
    json_atom(N, Name), json_ver(V, Ver).

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

constraint_json(any, any).
constraint_json(eq(V), _{op: "eq", v: J}) :- ver_json(V, J).
constraint_json(gte(V), _{op: "gte", v: J}) :- ver_json(V, J).
constraint_json(lt(V), _{op: "lt", v: J}) :- ver_json(V, J).
constraint_json(range(Lo, Hi), _{op: "range", lo: LJ, hi: HJ}) :-
    ver_json(Lo, LJ), ver_json(Hi, HJ).
