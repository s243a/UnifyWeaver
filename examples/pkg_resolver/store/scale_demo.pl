:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% scale_demo.pl -- load the 5k catalog as a term AND as P/2 facts; time
% the same resolve_layered query both ways (load cost included).
%
%   STORE_DIR=... swipl -q -g scale_demo -t halt examples/pkg_resolver/store/scale_demo.pl -- DIR

:- module(scale_demo, [scale_demo/0]).

:- use_module(library(http/json)).
:- use_module('../resolver').
:- use_module('../resolver_store').

scale_demo :-
    current_prolog_flag(argv, Argv),
    (   Argv = [Dir|_] -> true ; getenv('STORE_DIR', Dir) ),
    atom_concat(Dir, '/rich.jsonl', Rich),
    atom_concat(Dir, '/probe.json', ProbeF),
    setup_call_cleanup(open(ProbeF, read, S), json_read_dict(S, Probe, [value_string_as(atom)]), close(S)),
    probe_request(Probe, Req),
    statistics(cputime, T0),
    load_rich_catalog(Rich, Cat),
    statistics(cputime, T1),
    (   resolve_layered(Cat, [Req], TermSel) -> true ; TermSel = fail ),
    statistics(cputime, T2),
    format("swi_term_load_s ~3f~n", [T1 - T0]),
    format("swi_term_resolve_s ~3f~n", [T2 - T1]),
    format("swi_term_total_s ~3f~n", [T2 - T0]),
    format("swi_term_result ~q~n", [TermSel]),
    statistics(cputime, U0),
    load_p2_jsonl(Dir),
    statistics(cputime, U1),
    env_from_probe(Probe, Env),
    (   resolve_layered_store(Env, [Req], StoreSel) -> true ; StoreSel = fail ),
    statistics(cputime, U2),
    format("swi_store_load_s ~3f~n", [U1 - U0]),
    format("swi_store_resolve_s ~3f~n", [U2 - U1]),
    format("swi_store_total_s ~3f~n", [U2 - U0]),
    format("swi_store_result ~q~n", [StoreSel]),
    (   TermSel == StoreSel
    ->  format("swi_term_store_match true~n", [])
    ;   format("swi_term_store_match false~n", []),
        halt(1)
    ).

probe_request(Probe, Req) :-
    get_dict(args, Probe, Args),
    (   is_list(Args) -> Args = [Req0] ; Req0 = Args ),
    json_atom(Req0, Req).

env_from_probe(Probe, env(Id, B, I, R, L, E, A)) :-
    get_dict(env, Probe, D),
    (   get_dict(catalog_id, D, Id0) -> json_atom(Id0, Id) ; Id = s5k ),
    (   get_dict(base, D, B0) -> maplist(json_pair, B0, B) ; B = [] ),
    (   get_dict(installed, D, I0) -> maplist(json_pair, I0, I) ; I = [] ),
    (   get_dict(requested, D, R0) -> maplist(json_atom, R0, R) ; R = [] ),
    L = [], E = [], A = [].

json_pair([N, V], Name-Ver) :- json_atom(N, Name), json_ver(V, Ver).
json_ver([X, Y, Z], v(X, Y, Z)).
json_atom(A, A) :- atom(A), !.
json_atom(S, A) :- string(S), atom_string(A, S).

load_rich_catalog(Path, catalog(Ps, Ds, Cs, B, I, R, [], [], [])) :-
    setup_call_cleanup(open(Path, read, S), read_rich(S, [], Ps, [], Ds, [], Cs), close(S)),
    B = [p0-v(0, 0, 0)],
    I = [],
    R = [].

read_rich(S, Ps0, Ps, Ds0, Ds, Cs0, Cs) :-
    read_line_to_string(S, Line),
    (   Line == end_of_file
    ->  reverse(Ps0, Ps), reverse(Ds0, Ds), reverse(Cs0, Cs)
    ;   Line == ""
    ->  read_rich(S, Ps0, Ps, Ds0, Ds, Cs0, Cs)
    ;   atom_json_dict(Line, Row, [value_string_as(atom)]),
        acc_rich(Row, Ps0, Ps1, Ds0, Ds1, Cs0, Cs1),
        read_rich(S, Ps1, Ps, Ds1, Ds, Cs1, Cs)
    ).

acc_rich(Row, Ps, [package(N, V)|Ps], Ds, Ds, Cs, Cs) :-
    get_dict(kind, Row, package), !,
    json_atom(Row.name, N), json_ver(Row.ver, V).
acc_rich(Row, Ps, Ps, Ds, [depends(N, V, D, C)|Ds], Cs, Cs) :-
    get_dict(kind, Row, depends), !,
    json_atom(Row.name, N), json_ver(Row.ver, V),
    json_atom(Row.dep, D), json_constraint(Row.constraint, C).
acc_rich(Row, Ps, Ps, Ds, Ds, Cs, [conflicts(N, V, O)|Cs]) :-
    get_dict(kind, Row, conflicts), !,
    json_atom(Row.name, N), json_ver(Row.ver, V), json_atom(Row.other, O).
acc_rich(_, Ps, Ps, Ds, Ds, Cs, Cs).

json_constraint(any, any) :- !.
json_constraint("any", any) :- !.
json_constraint(D, C) :-
    is_dict(D), json_atom(D.op, Op),
    (   Op == gte -> json_ver(D.v, V), C = gte(V)
    ;   Op == eq -> json_ver(D.v, V), C = eq(V)
    ;   Op == lt -> json_ver(D.v, V), C = lt(V)
    ;   Op == range -> json_ver(D.lo, Lo), json_ver(D.hi, Hi), C = range(Lo, Hi)
    ).
