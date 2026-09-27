:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% derive.pl -- the SWI side of the CLI contract corpus.
%
% Nothing under `cli/generated/` is hand-written. This program derives all of
% it from the two Prolog specs the CLI composes:
%
%   generated/pkg_registry.json      <- cli/pkg_schema.pl   (the command grammar)
%   generated/catalogs/*.json        <- cli/examples/*.pl   (the example catalogs)
%   generated/expected.json          <- resolver.pl run over those catalogs,
%                                       rendered into the CLI's documented
%                                       `--json` document, plus the exit code
%
% `test_pkg_cli.mjs` then runs `pkg.mjs` as a subprocess and compares. So the
% expectations are SWI's answers, regenerable at any time -- not typed by hand.
%
%   swipl -q -g derive -t halt examples/pkg_resolver/cli/derive.pl

:- module(pkg_cli_derive, [derive/0]).

:- use_module(library(http/json)).
:- use_module(library(filesex)).
:- use_module('../resolver.pl').
:- use_module('pkg_schema.pl').
% imported with an empty list: both catalogs export example_catalog/2, so the
% calls below stay explicitly module-qualified.
:- use_module('examples/teaching.pl', []).
:- use_module('examples/frozen_base.pl', []).

:- prolog_load_context(directory, D), assertz(here_dir(D)).

catalog_of(teaching,    C) :- catalog_teaching:example_catalog(teaching, C).
catalog_of(frozen_base, C) :- catalog_frozen_base:example_catalog(frozen_base, C).

% ===========================================================================
% The corpus: every command, on both catalogs.
% argv here EXCLUDES `--catalog <file>`; the JS runner appends it.
% ===========================================================================

cli_case(t_resolve,              teaching,    ["resolve", "editor"]).
cli_case(t_install_plan,         teaching,    ["install-plan", "editor"]).
cli_case(t_layer_spelling,       teaching,    ["layer", "editor"]).
cli_case(t_why_blocked,          teaching,    ["why-blocked", "editor"]).
cli_case(t_deps,                 teaching,    ["deps", "editor"]).
cli_case(t_what_needs,           teaching,    ["what-needs", "libc"]).
cli_case(t_what_needs_installed, teaching,    ["what-needs", "libc", "--installed"]).
cli_case(t_orphans,              teaching,    ["orphans", "theme"]).
cli_case(t_audit,                teaching,    ["audit"]).
cli_case(t_why_frozen,           teaching,    ["why-frozen", "libc"]).
cli_case(t_safe_upgrade,         teaching,    ["safe-upgrade", "libc", "2.0.0"]).

cli_case(f_resolve,              frozen_base, ["resolve", "firefox"]).
cli_case(f_resolve_multi,        frozen_base, ["resolve", "firefox", "mplayer"]).
cli_case(f_resolve_excluded,     frozen_base, ["resolve", "systemd"]).
cli_case(f_install_plan,         frozen_base, ["install-plan", "firefox"]).
cli_case(f_install_plan_layer,   frozen_base, ["install-plan", "gcc"]).
cli_case(f_install_plan_multi,   frozen_base, ["install-plan", "firefox", "mplayer"]).
cli_case(f_why_blocked_alias,    frozen_base, ["why-blocked", "firefox-esr"]).
cli_case(f_why_blocked_clear,    frozen_base, ["why-blocked", "mplayer"]).
cli_case(f_deps,                 frozen_base, ["deps", "firefox"]).
cli_case(f_deps_gtk,             frozen_base, ["deps", "gtk"]).
cli_case(f_what_needs,           frozen_base, ["what-needs", "glibc"]).
cli_case(f_what_needs_installed, frozen_base, ["what-needs", "glibc", "--installed"]).
cli_case(f_orphans,              frozen_base, ["orphans", "firefox"]).
cli_case(f_audit,                frozen_base, ["audit"]).
cli_case(f_why_frozen_anchor,    frozen_base, ["why-frozen", "glibc"]).
cli_case(f_why_frozen_suggest,   frozen_base, ["why-frozen", "gtk"]).
cli_case(f_why_frozen_over,      frozen_base, ["why-frozen", "pango"]).
cli_case(f_why_frozen_absent,    frozen_base, ["why-frozen", "nss"]).
cli_case(f_safe_upgrade_coord,   frozen_base, ["safe-upgrade", "glibc", "2.35.0"]).
cli_case(f_safe_upgrade_unsafe,  frozen_base, ["safe-upgrade", "busybox", "1.35.0"]).
cli_case(f_safe_upgrade_alias,   frozen_base, ["safe-upgrade", "urxvt", "9.22.0"]).
cli_case(f_safe_upgrade_none,    frozen_base, ["safe-upgrade", "nss", "3.90.0"]).

% ===========================================================================
% Driver
% ===========================================================================

derive :-
    out_dir(Dir),
    atom_concat(Dir, '/catalogs', CatDir),
    make_directory_path(Dir),
    make_directory_path(CatDir),
    write_registry(Dir),
    aggregate_all(count, catalog_name(_), NCat),
    forall(catalog_name(N), write_catalog(CatDir, N)),
    write_expected(Dir),
    format("registry: 1  catalogs: ~w~n", [NCat]).

catalog_name(teaching).
catalog_name(frozen_base).

out_dir(Dir) :-
    here_dir(Here),
    atom_concat(Here, '/generated', Dir).

% --- the registry ---------------------------------------------------------
% json_write/3 over json([Key=Value,...]) preserves declaration order, which
% SWI dicts would not; cliArgs.mjs reads the object with Object.keys().

write_registry(Dir) :-
    atom_concat(Dir, '/pkg_registry.json', Path),
    pkg_registry(Reg),
    registry_json(Reg, J),
    setup_call_cleanup(
        open(Path, write, S, [encoding(utf8)]),
        ( json_write(S, J, [width(78)]), nl(S) ),
        close(S)).

registry_json(Reg, json(Pairs)) :-
    maplist(registry_entry_json, Reg, Pairs).

registry_entry_json(Name-schema(Opts, Pos), Key=json([options=json(OJ), positionals=Pos])) :-
    atom_string(Key, Name),
    maplist(option_json, Opts, OJ).
registry_entry_json(Name-group(Actions), Key=json([actions=json(AJ)])) :-
    atom_string(Key, Name),
    maplist(registry_entry_json, Actions, AJ).

option_json(Name-Kind, Key=Kind) :-
    atom_string(Key, Name).

% --- the catalogs ---------------------------------------------------------

write_catalog(CatDir, Name) :-
    catalog_of(Name, Cat),
    catalog_json(Cat, J),
    format(atom(Path), '~w/~w.json', [CatDir, Name]),
    setup_call_cleanup(
        open(Path, write, S, [encoding(utf8)]),
        ( json_write_dict(S, J, [width(78)]), nl(S) ),
        close(S)).

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

pkg_json(package(N, V), [NS, VJ]) :- atom_string(N, NS), ver_list(V, VJ).
dep_json(depends(N, V, D, C), [NS, VJ, DS, CJ]) :-
    atom_string(N, NS), ver_list(V, VJ), atom_string(D, DS), cat_constraint_json(C, CJ).
conf_json(conflicts(N, V, O), [NS, VJ, OS]) :-
    atom_string(N, NS), ver_list(V, VJ), atom_string(O, OS).
pair_json(N-V, [NS, VJ]) :- atom_string(N, NS), ver_list(V, VJ).
hold_json(N-V, [NS, VJ]) :- atom_string(N, NS), ver_list(V, VJ).
hold_json(base(N-V, R), [NS, VJ, RS]) :-
    atom_string(N, NS), ver_list(V, VJ), atom_string(R, RS).
layer_json(layer(N, Pkgs), _{name: NS, packages: PJ}) :-
    atom_string(N, NS), maplist(hold_json, Pkgs, PJ).
alias_json(alias(A, C), [AS, CS]) :- atom_string(A, AS), atom_string(C, CS).

ver_list(v(A, B, C), [A, B, C]).

% the shim's catalog-side constraint encoding
cat_constraint_json(any, any).
cat_constraint_json(eq(V),  _{op: "eq",  v: J}) :- ver_list(V, J).
cat_constraint_json(gte(V), _{op: "gte", v: J}) :- ver_list(V, J).
cat_constraint_json(lt(V),  _{op: "lt",  v: J}) :- ver_list(V, J).
cat_constraint_json(range(Lo, Hi), _{op: "range", lo: LJ, hi: HJ}) :-
    ver_list(Lo, LJ), ver_list(Hi, HJ).

% ===========================================================================
% The expected CLI documents
% ===========================================================================

write_expected(Dir) :-
    atom_concat(Dir, '/expected.json', Path),
    findall(Case, expected_case(Case), Cases),
    length(Cases, N),
    setup_call_cleanup(
        open(Path, write, S, [encoding(utf8)]),
        ( json_write_dict(S, _{cases: Cases}, [width(78)]), nl(S) ),
        close(S)),
    format("cases: ~w~n", [N]).

expected_case(_{id: IdS, catalog: CatS, argv: Argv, exit: Exit, json: Doc}) :-
    cli_case(Id, CatName, Argv),
    atom_string(Id, IdS),
    atom_string(CatName, CatS),
    catalog_of(CatName, Cat),
    argv_doc(Argv, Cat, Doc),
    get_dict(status, Doc, Status),
    exit_for_status(Status, Exit).

exit_for_status("ok", 0).
exit_for_status("clear", 0).
exit_for_status("blocked", 1).
exit_for_status("fail", 1).
exit_for_status("not-frozen", 1).

% --- argv -> the documented JSON doc ---------------------------------------
% Same dispatch the CLI performs, over resolver.pl directly.

argv_doc([Cmd0|Rest], Cat, Doc) :-
    ( Cmd0 == "layer" -> Cmd = "install-plan" ; Cmd = Cmd0 ),
    partition_flags(Rest, Names, Flags),
    command_doc(Cmd, Cat, Names, Flags, Doc).

partition_flags([], [], []).
partition_flags([T|Ts], Names, Flags) :-
    (   string_concat("--", F, T)
    ->  Flags = [F|Fs], partition_flags(Ts, Names, Fs)
    ;   Names = [N|Ns], atom_string(N, T), partition_flags(Ts, Ns, Flags)
    ).

command_doc("resolve", Cat, Names, _F, Doc) :-
    maplist(atom_string, Names, ReqS),
    (   resolve(Cat, Names, Sel)
    ->  sel_json(Sel, SJ),
        Doc = _{command: "resolve", status: "ok", requests: ReqS, selection: SJ}
    ;   Doc = _{command: "resolve", status: "fail", requests: ReqS,
                reason: "no_solution"}
    ).

command_doc("install-plan", Cat, Names, _F, Doc) :-
    maplist(atom_string, Names, ReqS),
    (   resolve_layered(Cat, Names, Sel)
    ->  sel_json(Sel, SJ),
        maplist(manifest_json(Cat), Names, MJ),
        Doc = _{command: "install-plan", status: "ok", requests: ReqS,
                selection: SJ, manifests: MJ}
    ;   Doc = _{command: "install-plan", status: "fail", requests: ReqS,
                reason: "no_solution", manifests: []}
    ).

command_doc("why-blocked", Cat, [Name], _F,
            _{command: "why-blocked", status: Status, request: NS, blocked: BJ}) :-
    atom_string(Name, NS),
    explain_blocked_list(Cat, Name, List),
    maplist(blocked_json, List, BJ),
    ( List == [] -> Status = "clear" ; Status = "blocked" ).

command_doc("deps", Cat, [Name], _F,
            _{command: "deps", status: "ok", package: NS, depends: DJ}) :-
    atom_string(Name, NS),
    catalog_depends(Cat, Ds),
    findall(k(V, D, C), member(depends(Name, V, D, C), Ds), Rows0),
    msort(Rows0, Rows),
    maplist(dep_row_json, Rows, DJ).

command_doc("what-needs", Cat, [Name], Flags,
            _{command: "what-needs", status: "ok", package: NS,
              installed_only: Only, dependents: DJ}) :-
    atom_string(Name, NS),
    (   memberchk("installed", Flags)
    ->  Only = true,  dependents_installed(Cat, Name, Deps)
    ;   Only = false, dependents(Cat, Name, Deps)
    ),
    sel_json(Deps, DJ).

command_doc("orphans", Cat, [Name], _F,
            _{command: "orphans", status: "ok", package: NS, orphans: OJ}) :-
    atom_string(Name, NS),
    removal_orphans(Cat, Name, Orphans),
    sel_json(Orphans, OJ).

command_doc("why-frozen", Cat, [Name], _F, Doc) :-
    atom_string(Name, NS),
    freeze_audit(Cat, Audit),
    (   member(audit(Name, Payload), Audit)
    ->  audit_kind_reason(Payload, Kind, Reason),
        Doc = _{command: "why-frozen", status: "ok", package: NS,
                kind: Kind, reason: Reason}
    ;   Doc = _{command: "why-frozen", status: "not-frozen", package: NS,
                kind: null, reason: null}
    ).

command_doc("audit", Cat, [], _F,
            _{command: "audit", status: "ok", audit: AJ}) :-
    freeze_audit(Cat, Audit),
    maplist(audit_json, Audit, AJ).

command_doc("safe-upgrade", Cat, [Name, VerAtom], _F, Doc) :-
    atom_string(Name, NS),
    parse_ver(VerAtom, Ver),
    ver_string(Ver, VS),
    safe_upgrade(Cat, Name, Ver, Verdict),
    verdict_json(Verdict, Status, RJ),
    Doc = _{command: "safe-upgrade", status: Status, package: NS,
            version: VS, result: RJ}.

catalog_depends(catalog(_, Ds, _, _, _, _), Ds).
catalog_depends(catalog(_, Ds, _, _, _, _, _, _, _), Ds).

manifest_json(Cat, Name, _{request: NS, order: OJ}) :-
    atom_string(Name, NS),
    layer_closure(Cat, Name, Layer),
    sel_json(Layer, OJ).

sel_json([], []).
sel_json([N-V|Rest], [_{name: NS, version: VS}|Js]) :-
    atom_string(N, NS), ver_string(V, VS),
    sel_json(Rest, Js).

blocked_json(blocked(N, needs(C), base_has(V)),
             _{name: NS, needs: CJ, base_has: VS}) :-
    atom_string(N, NS), doc_constraint_json(C, CJ), ver_string(V, VS).

dep_row_json(k(V, D, C), _{version: VS, dep: DS, constraint: CJ}) :-
    ver_string(V, VS), atom_string(D, DS), doc_constraint_json(C, CJ).

audit_json(audit(N, Payload), _{name: NS, kind: Kind, reason: Reason}) :-
    atom_string(N, NS),
    audit_kind_reason(Payload, Kind, Reason).

audit_kind_reason(over_frozen, "over_frozen", null).
audit_kind_reason(suggest(R), "suggest", RS) :- atom_string(R, RS).
audit_kind_reason(held(R), "held", RS) :- atom_string(R, RS).

verdict_json(safe(cost(R)), "ok", _{verdict: "safe", cost: RS}) :-
    atom_string(R, RS).
verdict_json(coordinated(Set), "ok", _{verdict: "coordinated", set: SJ}) :-
    sel_json(Set, SJ).
verdict_json(unsafe(R), "fail", _{verdict: "unsafe", reason: RS}) :-
    atom_string(R, RS).
verdict_json(no_candidate, "fail", _{verdict: "no_candidate"}).

% the CLI's own doc-level constraint encoding (uniform tagged object)
doc_constraint_json(any, _{op: "any"}).
doc_constraint_json(eq(V),  _{op: "eq",  version: VS}) :- ver_string(V, VS).
doc_constraint_json(gte(V), _{op: "gte", version: VS}) :- ver_string(V, VS).
doc_constraint_json(lt(V),  _{op: "lt",  version: VS}) :- ver_string(V, VS).
doc_constraint_json(range(Lo, Hi), _{op: "range", lo: LS, hi: HS}) :-
    ver_string(Lo, LS), ver_string(Hi, HS).

ver_string(v(A, B, C), S) :- format(atom(A0), '~w.~w.~w', [A, B, C]), atom_string(A0, S).

parse_ver(Atom, v(A, B, C)) :-
    atomic_list_concat(Parts, '.', Atom),
    Parts = [PA, PB, PC],
    atom_number(PA, A), atom_number(PB, B), atom_number(PC, C).
