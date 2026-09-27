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
run_query(safe_upgrade, Cat, [Pkg, Ver], Exp) :-
    safe_upgrade(Cat, Pkg, Ver, Verdict),
    verdict_json(Verdict, Js),
    Exp = _{ok: Js}.
run_query(upgrade_set, Cat, [Pkg, Ver], Exp) :-
    upgrade_set_result(Cat, Pkg, Ver, R),
    upgrade_json(R, Exp).
run_query(freeze_audit, Cat, _, Exp) :-
    freeze_audit(Cat, Audit),
    maplist(audit_json, Audit, Js),
    Exp = _{ok: Js}.
run_query(dependents, Cat, Args, Exp) :-
    dependents(Cat, Args, Deps),
    sel_json(Deps, Js),
    Exp = _{ok: Js}.
run_query(dependents_installed, Cat, Args, Exp) :-
    dependents_installed(Cat, Args, Deps),
    sel_json(Deps, Js),
    Exp = _{ok: Js}.

args_json(resolve, Reqs, Js) :- reqs_json(Reqs, Js).
args_json(resolve_layered, Reqs, Js) :- reqs_json(Reqs, Js).
args_json(explain_blocked, Req, Js) :- req_json(Req, Js).
args_json(layer_closure, Req, Js) :- req_json(Req, Js).
args_json(removal_orphans, Pkg, Js) :- atom_string(Pkg, Js).
args_json(safe_upgrade, [Pkg, Ver], [NS, VJ]) :-
    atom_string(Pkg, NS), ver_json(Ver, VJ).
args_json(upgrade_set, [Pkg, Ver], [NS, VJ]) :-
    atom_string(Pkg, NS), ver_json(Ver, VJ).
args_json(freeze_audit, _, []).
args_json(dependents, Pkg, Js) :- atom_string(Pkg, Js).
args_json(dependents_installed, Pkg, Js) :- atom_string(Pkg, Js).

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
    maplist(hold_json, Bs, BJ),
    maplist(pair_json, Is, IJ),
    maplist(atom_string, Rs, RJ).
catalog_json(catalog(Ps, Ds, Cs, Bs, Is, Rs, Ls, Es, As),
             _{packages: PJ, depends: DJ, conflicts: CJ,
               base: BJ, installed: IJ, requested: RJ,
               layers: LJ, excluded: EJ, aliases: AJ}) :-
    maplist(pkg_json, Ps, PJ),
    maplist(dep_json, Ds, DJ),
    maplist(conf_json, Cs, CJ),
    maplist(hold_json, Bs, BJ),
    maplist(pair_json, Is, IJ),
    maplist(atom_string, Rs, RJ),
    maplist(layer_json, Ls, LJ),
    maplist(atom_string, Es, EJ),
    maplist(alias_json, As, AJ).
catalog_json(catalog(Ps, Ds, Cs, Bs, Is, Rs, Ls, Es, As, Pr),
             _{packages: PJ, depends: DJ, conflicts: CJ,
               base: BJ, installed: IJ, requested: RJ,
               layers: LJ, excluded: EJ, aliases: AJ, provides: PrJ}) :-
    maplist(pkg_json, Ps, PJ),
    maplist(dep_json, Ds, DJ),
    maplist(conf_json, Cs, CJ),
    maplist(hold_json, Bs, BJ),
    maplist(pair_json, Is, IJ),
    maplist(atom_string, Rs, RJ),
    maplist(layer_json, Ls, LJ),
    maplist(atom_string, Es, EJ),
    maplist(alias_json, As, AJ),
    maplist(provide_json, Pr, PrJ).

provide_json(provides(N, V, Virt), [NS, VJ, VS]) :-
    atom_string(N, NS), ver_json(V, VJ), atom_string(Virt, VS).
provide_json(provides(N, V, Virt, VV), [NS, VJ, VS, VVJ]) :-
    atom_string(N, NS), ver_json(V, VJ), atom_string(Virt, VS), ver_json(VV, VVJ).

pkg_json(package(N, V), [NS, VJ]) :-
    atom_string(N, NS), ver_json(V, VJ).
dep_json(depends(N, V, alternatives(Alts), C), [NS, VJ, _{alternatives: AJ}, CJ]) :-
    !,
    atom_string(N, NS), ver_json(V, VJ),
    maplist(alt_json, Alts, AJ),
    constraint_json(C, CJ).
dep_json(depends(N, V, D, C), [NS, VJ, DS, CJ]) :-
    atom_string(N, NS), ver_json(V, VJ), atom_string(D, DS),
    constraint_json(C, CJ).

alt_json(dep(D, C), _{constraint: CJ, dep: DS}) :-
    atom_string(D, DS), constraint_json(C, CJ).
conf_json(conflicts(N, V, O), [NS, VJ, OS]) :-
    atom_string(N, NS), ver_json(V, VJ), atom_string(O, OS).
pair_json(N-V, [NS, VJ]) :-
    atom_string(N, NS), ver_json(V, VJ).
hold_json(N-V, [NS, VJ]) :-
    atom_string(N, NS), ver_json(V, VJ).
hold_json(base(N-V, R), [NS, VJ, RS]) :-
    atom_string(N, NS), ver_json(V, VJ), atom_string(R, RS).
layer_json(layer(N, Pkgs), _{name: NS, packages: PJ}) :-
    atom_string(N, NS),
    maplist(hold_json, Pkgs, PJ).
alias_json(alias(A, C), [AS, CS]) :-
    atom_string(A, AS), atom_string(C, CS).

ver_json(v(A, B, C), [A, B, C]) :- !.
ver_json(deb(E, Up, Rev), _{deb: [E, UJ, RJ]}) :-
    maplist(seg_json, Up, UJ),
    maplist(seg_json, Rev, RJ).

seg_json(s(Codes, N), [S, N]) :-
    string_codes(S, Codes).

constraint_json(any, any).
constraint_json(eq(V), _{op: "eq", v: J}) :- ver_json(V, J).
constraint_json(gte(V), _{op: "gte", v: J}) :- ver_json(V, J).
constraint_json(lte(V), _{op: "lte", v: J}) :- ver_json(V, J).
constraint_json(gt(V), _{op: "gt", v: J}) :- ver_json(V, J).
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
    atom(N),
    !,
    atom_string(N, NS),
    constraint_json(C, CJ),
    ver_json(V, VJ),
    blocked_list_json(Rest, Js).
blocked_list_json([blocked(N, needs(C), providers(Ps))|Rest],
                  [_{name: NS, needs: CJ, providers: PJ}|Js]) :-
    !,
    atom_string(N, NS),
    constraint_json(C, CJ),
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
