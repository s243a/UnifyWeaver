:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% dump_store_data.pl -- write corpus P/2 JSONL + query rows for the store adapter.
%
%   swipl -q -g dump_store_data -t halt examples/pkg_resolver/dump_store_data.pl -- DIR
%
% Writes DIR/pkg.jsonl DIR/dep.jsonl DIR/conflict.jsonl DIR/revdep.jsonl
%        DIR/rich.jsonl DIR/cases.jsonl DIR/envs.jsonl

:- module(dump_store_data, [dump_store_data/0]).

:- use_module(library(http/json)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(resolver).
:- use_module(resolver_store).
:- use_module(test_resolver, [scenario_catalog/2, corpus_case/4]).

dump_store_data :-
    current_prolog_flag(argv, Argv),
    (   Argv = [Dir|_] -> true ; Dir = 'examples/pkg_resolver/store/.out/corpus' ),
    make_directory_path(Dir),
    atom_concat(Dir, '/pkg.jsonl', PkgF),
    atom_concat(Dir, '/dep.jsonl', DepF),
    atom_concat(Dir, '/conflict.jsonl', ConfF),
    atom_concat(Dir, '/revdep.jsonl', RevF),
    atom_concat(Dir, '/rich.jsonl', RichF),
    atom_concat(Dir, '/cases.jsonl', CaseF),
    atom_concat(Dir, '/envs.jsonl', EnvF),
    setup_call_cleanup(open(PkgF, write, PS),
    setup_call_cleanup(open(DepF, write, DS),
    setup_call_cleanup(open(ConfF, write, CS),
    setup_call_cleanup(open(RevF, write, RS),
    setup_call_cleanup(open(RichF, write, RichS),
        dump_all_catalogs(PS, DS, CS, RS, RichS),
    close(RichS)), close(RS)), close(CS)), close(DS)), close(PS)),
    setup_call_cleanup(open(EnvF, write, ES), dump_envs(ES), close(ES)),
    setup_call_cleanup(open(CaseF, write, KS), dump_cases(KS), close(KS)),
    format("dump_store_data: wrote P/2 + cases under ~w~n", [Dir]).

dump_all_catalogs(PS, DS, CS, RS, RichS) :-
    findall(Name, scenario_catalog(Name, _), Names0),
    sort(Names0, Names),
    forall(member(Name, Names),
           (   scenario_catalog(Name, Cat),
               dump_one_catalog(Name, Cat, PS, DS, CS, RS, RichS)
           )).

dump_one_catalog(Id, catalog(Ps, Ds, Cs, B, I, R), PS, DS, CS, RS, RichS) :-
    dump_one_catalog(Id, catalog(Ps, Ds, Cs, B, I, R, [], [], []), PS, DS, CS, RS, RichS).
dump_one_catalog(Id, catalog(Ps, Ds, Cs, B, I, R, Ls, Es, As), PS, DS, CS, RS, RichS) :-
    atom_string(Id, IdS),
    forall(member(package(N, V), Ps),
           (   pack_key(Id, N, K), pack_ver(V, VA),
               write_pair(PS, K, VA),
               ver_json(V, VJ), atom_string(N, NS),
               json_write_dict(RichS, _{kind: "package", catalog: IdS, name: NS, ver: VJ},
                               [width(0)]), nl(RichS)
           )),
    forall(member(depends(N, V, D, C), Ds),
           (   pack_key(Id, N, KN), pack_dep_local(V, D, C, DA),
               write_pair(DS, KN, DA),
               pack_key(Id, D, KD), pack_rev_local(N, V, C, RA),
               write_pair(RS, KD, RA),
               ver_json(V, VJ), atom_string(N, NS), atom_string(D, DepS),
               constraint_json(C, CJ),
               json_write_dict(RichS,
                   _{kind: "depends", catalog: IdS, name: NS, ver: VJ, dep: DepS, constraint: CJ},
                   [width(0)]), nl(RichS)
           )),
    forall(member(conflicts(N, V, O), Cs),
           (   pack_key(Id, N, K), pack_conflict_local(V, O, CA),
               write_pair(CS, K, CA),
               ver_json(V, VJ), atom_string(N, NS), atom_string(O, OS),
               json_write_dict(RichS,
                   _{kind: "conflicts", catalog: IdS, name: NS, ver: VJ, other: OS},
                   [width(0)]), nl(RichS)
           )),
    dump_env_rich(IdS, B, I, R, Ls, Es, As, RichS).

pack_dep_local(V, D, C, DA) :-
    resolver_store:pack_ver(V, VA),
    resolver_store:pack_constraint(C, CA),
    atom_concat(VA, '#', T1), atom_concat(T1, D, T2),
    atom_concat(T2, '#', T3), atom_concat(T3, CA, DA).

pack_rev_local(N, V, C, RA) :-
    resolver_store:pack_ver(V, VA),
    resolver_store:pack_constraint(C, CA),
    atom_concat(N, '#', T1), atom_concat(T1, VA, T2),
    atom_concat(T2, '#', T3), atom_concat(T3, CA, RA).

pack_conflict_local(V, O, CA) :-
    resolver_store:pack_ver(V, VA),
    atom_concat(VA, '#', T), atom_concat(T, O, CA).

write_pair(S, K, V) :-
    atom_string(K, KS), atom_string(V, VS),
    json_write(S, [KS, VS], [width(0)]), nl(S).

dump_env_rich(IdS, B, I, R, Ls, Es, As, RichS) :-
    forall(member(H, B), dump_hold_rich(IdS, "base", H, RichS)),
    forall(member(layer(LN, Pkgs), Ls),
           forall(member(H, Pkgs), dump_layer_hold_rich(IdS, LN, H, RichS))),
    forall(member(N-V, I),
           (   ver_json(V, VJ), atom_string(N, NS),
               json_write_dict(RichS, _{kind: "installed", catalog: IdS, name: NS, ver: VJ}, [width(0)]),
               nl(RichS)
           )),
    forall(member(N, R),
           (   atom_string(N, NS),
               json_write_dict(RichS, _{kind: "requested", catalog: IdS, name: NS}, [width(0)]),
               nl(RichS)
           )),
    forall(member(N, Es),
           (   atom_string(N, NS),
               json_write_dict(RichS, _{kind: "excluded", catalog: IdS, name: NS}, [width(0)]),
               nl(RichS)
           )),
    forall(member(alias(A, C), As),
           (   atom_string(A, AS), atom_string(C, CS),
               json_write_dict(RichS, _{kind: "alias", catalog: IdS, alias: AS, canonical: CS}, [width(0)]),
               nl(RichS)
           )).

dump_hold_rich(IdS, Kind, N-V, RichS) :-
    ver_json(V, VJ), atom_string(N, NS),
    json_write_dict(RichS, _{kind: Kind, catalog: IdS, name: NS, ver: VJ, reason: "blanket"}, [width(0)]),
    nl(RichS).
dump_hold_rich(IdS, Kind, base(N-V, Reason), RichS) :-
    ver_json(V, VJ), atom_string(N, NS), atom_string(Reason, RS),
    json_write_dict(RichS, _{kind: Kind, catalog: IdS, name: NS, ver: VJ, reason: RS}, [width(0)]),
    nl(RichS).
dump_hold_rich(IdS, _Kind, layer(LN, Pkgs), RichS) :-
    forall(member(H, Pkgs), dump_layer_hold_rich(IdS, LN, H, RichS)).

dump_layer_hold_rich(IdS, LN, N-V, RichS) :-
    ver_json(V, VJ), atom_string(N, NS), atom_string(LN, LS),
    json_write_dict(RichS, _{kind: "layer", catalog: IdS, layer: LS, name: NS, ver: VJ}, [width(0)]),
    nl(RichS).
dump_layer_hold_rich(IdS, LN, base(N-V, Reason), RichS) :-
    ver_json(V, VJ), atom_string(N, NS), atom_string(LN, LS), atom_string(Reason, RS),
    json_write_dict(RichS, _{kind: "layer", catalog: IdS, layer: LS, name: NS, ver: VJ, reason: RS}, [width(0)]),
    nl(RichS).

dump_envs(ES) :-
    findall(Name, scenario_catalog(Name, _), Names0),
    sort(Names0, Names),
    forall(member(Name, Names),
           (   scenario_catalog(Name, Cat),
               env_from_catalog(Name, Cat, Env),
               env_json(Env, J),
               json_write_dict(ES, J, [width(0), true(true), false(false)]),
               nl(ES)
           )).

env_json(env(Id, B, I, R, L, E, A),
         _{catalog_id: IdS, base: BJ, installed: IJ, requested: RJ,
           layers: LJ, excluded: EJ, aliases: AJ}) :-
    atom_string(Id, IdS),
    maplist(hold_json, B, BJ),
    maplist(pair_json, I, IJ),
    maplist(atom_string, R, RJ),
    maplist(layer_json, L, LJ),
    maplist(atom_string, E, EJ),
    maplist(alias_json, A, AJ).

dump_cases(KS) :-
    forall(corpus_case(IdAtom, CatName, Query, Args),
           dump_one_case(KS, IdAtom, CatName, Query, Args)).

dump_one_case(KS, IdAtom, CatName, Query, Args) :-
    scenario_catalog(CatName, Cat),
    env_from_catalog(CatName, Cat, Env),
    env_json(Env, EJ),
    atom_string(IdAtom, Id),
    atom_string(Query, QAtom),
    atom_string(CatName, CatS),
    args_json_local(Query, Args, ArgsJ),
    run_term_local(Query, Cat, Args, Exp),
    json_write_dict(KS,
        _{id: Id, catalog_id: CatS, env: EJ, query: QAtom, args: ArgsJ, expected: Exp},
        [width(0), true(true), false(false)]),
    nl(KS).

args_json_local(resolve, Reqs, Js) :- reqs_json(Reqs, Js).
args_json_local(resolve_layered, Reqs, Js) :- reqs_json(Reqs, Js).
args_json_local(explain_blocked, Req, Js) :- req_json(Req, Js).
args_json_local(layer_closure, Req, Js) :- req_json(Req, Js).
args_json_local(removal_orphans, Pkg, Js) :- atom_string(Pkg, Js).
args_json_local(safe_upgrade, [Pkg, Ver], [NS, VJ]) :-
    atom_string(Pkg, NS), ver_json(Ver, VJ).
args_json_local(upgrade_set, [Pkg, Ver], [NS, VJ]) :-
    atom_string(Pkg, NS), ver_json(Ver, VJ).
args_json_local(freeze_audit, _, []).
args_json_local(dependents, Pkg, Js) :- atom_string(Pkg, Js).
args_json_local(dependents_installed, Pkg, Js) :- atom_string(Pkg, Js).

reqs_json([], []).
reqs_json([R|Rs], [J|Js]) :- req_json(R, J), reqs_json(Rs, Js).

req_json(req(Name, C), _{req: N, constraint: CJ}) :-
    atom_string(Name, N), constraint_json(C, CJ).
req_json(Name, N) :- atom(Name), atom_string(Name, N).

run_term_local(resolve, Cat, Args, Exp) :-
    (   resolve(Cat, Args, Sel) -> sel_json(Sel, Js), Exp = _{ok: Js} ; Exp = _{fail: true} ).
run_term_local(resolve_layered, Cat, Args, Exp) :-
    (   resolve_layered(Cat, Args, Sel) -> sel_json(Sel, Js), Exp = _{ok: Js} ; Exp = _{fail: true} ).
run_term_local(explain_blocked, Cat, Args, Exp) :-
    explain_blocked_list(Cat, Args, List), blocked_list_json(List, Js), Exp = _{ok: Js}.
run_term_local(layer_closure, Cat, Args, Exp) :-
    (   layer_closure(Cat, Args, Layer) -> sel_json(Layer, Js), Exp = _{ok: Js} ; Exp = _{fail: true} ).
run_term_local(removal_orphans, Cat, Args, Exp) :-
    removal_orphans(Cat, Args, Orphans), sel_json(Orphans, Js), Exp = _{ok: Js}.
run_term_local(safe_upgrade, Cat, [Pkg, Ver], Exp) :-
    safe_upgrade(Cat, Pkg, Ver, Verdict), verdict_json(Verdict, Js), Exp = _{ok: Js}.
run_term_local(upgrade_set, Cat, [Pkg, Ver], Exp) :-
    upgrade_set_result(Cat, Pkg, Ver, R), upgrade_json(R, Exp).
run_term_local(freeze_audit, Cat, _, Exp) :-
    freeze_audit(Cat, Audit), maplist(audit_json, Audit, Js), Exp = _{ok: Js}.
run_term_local(dependents, Cat, Args, Exp) :-
    dependents(Cat, Args, Deps), sel_json(Deps, Js), Exp = _{ok: Js}.
run_term_local(dependents_installed, Cat, Args, Exp) :-
    dependents_installed(Cat, Args, Deps), sel_json(Deps, Js), Exp = _{ok: Js}.

ver_json(v(A, B, C), [A, B, C]).
pair_json(N-V, [NS, VJ]) :- atom_string(N, NS), ver_json(V, VJ).
hold_json(N-V, [NS, VJ]) :- atom_string(N, NS), ver_json(V, VJ).
hold_json(base(N-V, R), [NS, VJ, RS]) :- atom_string(N, NS), ver_json(V, VJ), atom_string(R, RS).
layer_json(layer(N, Pkgs), _{name: NS, packages: PJ}) :-
    atom_string(N, NS), maplist(hold_json, Pkgs, PJ).
alias_json(alias(A, C), [AS, CS]) :- atom_string(A, AS), atom_string(C, CS).

constraint_json(any, any).
constraint_json(eq(V), _{op: "eq", v: J}) :- ver_json(V, J).
constraint_json(gte(V), _{op: "gte", v: J}) :- ver_json(V, J).
constraint_json(lt(V), _{op: "lt", v: J}) :- ver_json(V, J).
constraint_json(range(Lo, Hi), _{op: "range", lo: LJ, hi: HJ}) :-
    ver_json(Lo, LJ), ver_json(Hi, HJ).

sel_json([], []).
sel_json([N-V|Rest], [[NS, VJ]|Js]) :-
    atom_string(N, NS), ver_json(V, VJ), sel_json(Rest, Js).

blocked_list_json([], []).
blocked_list_json([blocked(N, needs(C), base_has(V))|Rest],
                  [_{name: NS, needs: CJ, base_has: VJ}|Js]) :-
    atom_string(N, NS), constraint_json(C, CJ), ver_json(V, VJ),
    blocked_list_json(Rest, Js).

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
