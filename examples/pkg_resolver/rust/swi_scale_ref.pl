:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% swi_scale_ref.pl -- SWI reference leg for B3, reading the SAME single-case
% JSON the Rust shim reads (examples/pkg_resolver/rust/scale_to_case.mjs), so
% the two legs see byte-identical catalogs at every truncation point. Load
% (term construction) and resolve are timed separately, matching the shim's
% load_ms / resolve_ms.
%
%   swipl -q -g main -t halt swi_scale_ref.pl -- case_N.json

:- use_module(library(http/json)).
:- use_module('../resolver').

main :-
    current_prolog_flag(argv, [File|_]),
    setup_call_cleanup(open(File, read, S),
                       json_read_dict(S, Row, [value_string_as(atom)]),
                       close(S)),
    statistics(walltime, _),
    T0 is cputime,
    catalog_of(Row.catalog, Cat),
    reqs_of(Row.args, Reqs),
    T1 is cputime,
    (   resolve_layered(Cat, Reqs, Sel)
    ->  length(Sel, N)
    ;   N = fail
    ),
    T2 is cputime,
    L is (T1 - T0) * 1000,
    R is (T2 - T1) * 1000,
    format("load_ms=~3f~n", [L]),
    format("resolve_ms=~3f~n", [R]),
    format("selection_size=~w~n", [N]).

catalog_of(D, catalog(Ps, Ds, Cs, B, I, Rq, L, E, A)) :-
    maplist(pkg_of, D.packages, Ps),
    maplist(dep_of, D.depends, Ds),
    maplist(conf_of, D.conflicts, Cs),
    maplist(hold_of, D.base, B),
    maplist(pair_of, D.installed, I),
    maplist(atom_of, D.requested, Rq),
    maplist(layer_of, D.layers, L),
    maplist(atom_of, D.excluded, E),
    maplist(alias_of, D.aliases, A).

pkg_of([N, V], package(NA, Ver)) :- atom_of(N, NA), ver_of(V, Ver).
dep_of([N, V, Dp, C], depends(NA, Ver, DA, Cn)) :-
    atom_of(N, NA), ver_of(V, Ver), atom_of(Dp, DA), constraint_of(C, Cn).
conf_of([N, V, O], conflicts(NA, Ver, OA)) :-
    atom_of(N, NA), ver_of(V, Ver), atom_of(O, OA).
pair_of([N, V], NA-Ver) :- atom_of(N, NA), ver_of(V, Ver).
hold_of([N, V], NA-Ver) :- atom_of(N, NA), ver_of(V, Ver).
hold_of([N, V, R], base(NA-Ver, RA)) :-
    atom_of(N, NA), ver_of(V, Ver), atom_of(R, RA).
layer_of(D, layer(NA, Ps)) :-
    atom_of(D.name, NA), maplist(hold_of, D.packages, Ps).
alias_of([A, C], alias(AA, CA)) :- atom_of(A, AA), atom_of(C, CA).

ver_of([X, Y, Z], v(X, Y, Z)).
atom_of(A, A) :- atom(A), !.
atom_of(S, A) :- atom_string(A, S).

constraint_of(any, any) :- !.
constraint_of(D, C) :-
    atom_of(D.op, Op),
    (   Op == range
    ->  ver_of(D.lo, Lo), ver_of(D.hi, Hi), C = range(Lo, Hi)
    ;   ver_of(D.v, V), C =.. [Op, V]
    ).

reqs_of([], []).
reqs_of([R|Rs], [Q|Qs]) :- req_of(R, Q), reqs_of(Rs, Qs).
req_of(R, req(NA, C)) :-
    is_dict(R), !,
    atom_of(R.req, NA),
    ( get_dict(constraint, R, C0) -> constraint_of(C0, C) ; C = any ).
req_of(R, A) :- atom_of(R, A).
