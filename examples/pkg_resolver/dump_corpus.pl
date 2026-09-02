:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% dump_corpus.pl -- write the contract corpus as JSONL (catalog + query +
% SWI expected result) for the wamjs runner.
%
%   swipl -q -g dump_corpus -t halt examples/pkg_resolver/dump_corpus.pl

:- module(dump_corpus, [dump_corpus/0]).

:- use_module(library(http/json)).
:- use_module(resolver).
:- use_module(test_resolver, [scenario_catalog/2, corpus_case/4]).

dump_corpus :-
    findall(Row, corpus_json_row(Row), Rows),
    forall(member(Row, Rows),
           (   json_write_dict(current_output, Row, [width(0), true(true), false(false)]),
               nl
           )).

corpus_json_row(_{id:Id, catalog:CatJ, query:QAtom, args:ArgsJ, expected:Exp}) :-
    corpus_case(IdAtom, CatName, Query, Args),
    atom_string(IdAtom, Id),
    atom_string(Query, QAtom),
    scenario_catalog(CatName, Cat),
    catalog_json(Cat, CatJ),
    args_json(Query, Args, ArgsJ),
    run_query(Query, Cat, Args, Exp).

run_query(resolve, Cat, Args, Exp) :-
    (   resolve(Cat, Args, Sel)
    ->  sel_json(Sel, Js), Exp = _{ok: Js}
    ;   Exp = _{fail: true}
    ).
run_query(resolve_layered, Cat, Args, Exp) :-
    (   resolve_layered(Cat, Args, Sel)
    ->  sel_json(Sel, Js), Exp = _{ok: Js}
    ;   Exp = _{fail: true}
    ).
run_query(explain_blocked, Cat, Args, Exp) :-
    explain_blocked_list(Cat, Args, List),
    blocked_list_json(List, Js),
    Exp = _{ok: Js}.
run_query(layer_closure, Cat, Args, Exp) :-
    (   layer_closure(Cat, Args, Layer)
    ->  sel_json(Layer, Js), Exp = _{ok: Js}
    ;   Exp = _{fail: true}
    ).
run_query(removal_orphans, Cat, Args, Exp) :-
    removal_orphans(Cat, Args, Orphans),
    sel_json(Orphans, Js),
    Exp = _{ok: Js}.

args_json(resolve, Reqs, Js) :- reqs_json(Reqs, Js).
args_json(resolve_layered, Reqs, Js) :- reqs_json(Reqs, Js).
args_json(explain_blocked, Req, Js) :- req_json(Req, Js).
args_json(layer_closure, Req, Js) :- req_json(Req, Js).
args_json(removal_orphans, Pkg, Js) :- atom_string(Pkg, Js).

reqs_json([], []).
reqs_json([R|Rs], [J|Js]) :-
    req_json(R, J),
    reqs_json(Rs, Js).

req_json(req(Name, C), _{req: N, constraint: CJ}) :-
    atom_string(Name, N),
    constraint_json(C, CJ).
req_json(Name, N) :-
    atom(Name),
    atom_string(Name, N).

catalog_json(catalog(Ps, Ds, Cs, Bs, Is, Rs),
             _{packages: PJ, depends: DJ, conflicts: CJ,
               base: BJ, installed: IJ, requested: RJ}) :-
    maplist(pkg_json, Ps, PJ),
    maplist(dep_json, Ds, DJ),
    maplist(conf_json, Cs, CJ),
    maplist(pair_json, Bs, BJ),
    maplist(pair_json, Is, IJ),
    maplist(atom_string, Rs, RJ).

pkg_json(package(N, V), [NS, VJ]) :-
    atom_string(N, NS), ver_json(V, VJ).
dep_json(depends(N, V, D, C), [NS, VJ, DS, CJ]) :-
    atom_string(N, NS), ver_json(V, VJ), atom_string(D, DS),
    constraint_json(C, CJ).
conf_json(conflicts(N, V, O), [NS, VJ, OS]) :-
    atom_string(N, NS), ver_json(V, VJ), atom_string(O, OS).
pair_json(N-V, [NS, VJ]) :-
    atom_string(N, NS), ver_json(V, VJ).

ver_json(v(A, B, C), [A, B, C]).

constraint_json(any, any).
constraint_json(eq(V), _{op: "eq", v: J}) :- ver_json(V, J).
constraint_json(gte(V), _{op: "gte", v: J}) :- ver_json(V, J).
constraint_json(lt(V), _{op: "lt", v: J}) :- ver_json(V, J).
constraint_json(range(Lo, Hi), _{op: "range", lo: LJ, hi: HJ}) :-
    ver_json(Lo, LJ), ver_json(Hi, HJ).

sel_json([], []).
sel_json([N-V|Rest], [[NS, VJ]|Js]) :-
    atom_string(N, NS), ver_json(V, VJ),
    sel_json(Rest, Js).

blocked_list_json([], []).
blocked_list_json([blocked(N, needs(C), base_has(V))|Rest],
                  [_{name: NS, needs: CJ, base_has: VJ}|Js]) :-
    atom_string(N, NS),
    constraint_json(C, CJ),
    ver_json(V, VJ),
    blocked_list_json(Rest, Js).
