:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_abi.pl -- verify the redesigned symbol-level ABI lane.
%
%   swipl -q -g run -t halt examples/pkg_resolver/abi/test_abi.pl -- <out-dir>
%
% <out-dir> is run_abi_verify.sh's ./.out: it holds the real-data store
% (<out>/store), the fixture stores (<out>/fx/store_*) and the declared
% dependency ground truth (<out>/declared_depends.txt).
%
% Sections (each check names the review point it proves; see REVIEW_NOTES.md):
%   A. real data     -- libc6 / libselinux1 .symbols + /bin/ls on this machine
%   B. version axes  -- Debian package axis via the frozen deb/3 machinery
%   C. model         -- in-memory store fixtures for the epistemic contract
%   D. ELF fixtures  -- gcc-built libraries ingested through the real pipeline

:- use_module(abi_resolve).
:- use_module('../debian/deb_parse', [parse_deb_version/2]).

:- dynamic pass_count/1, fail_count/1, skip_count/1.
pass_count(0).
fail_count(0).
skip_count(0).

check(Name, Goal) :-
    (   catch(Goal, E, (print_message(error, E), fail))
    ->  format("  PASS  ~w~n", [Name]), bump(pass_count)
    ;   format("  FAIL  ~w~n", [Name]), bump(fail_count)
    ).

skip(Name) :-
    format("  SKIP  ~w~n", [Name]), bump(skip_count).

bump(C) :- G0 =.. [C, N], retract(G0), N1 is N + 1, G1 =.. [C, N1], assertz(G1).

report(Label, Val) :- format("        ~w = ~q~n", [Label, Val]).

out_dir(Dir) :-
    ( current_prolog_flag(argv, [D | _]), D \== [] -> Dir = D ; Dir = './.out' ).

run :-
    out_dir(Out),
    format("~n== ABI lane verification (out: ~w) ==~n", [Out]),
    section_real(Out),
    section_axes,
    section_model,
    section_elf_fixtures(Out),
    pass_count(P), fail_count(F), skip_count(S),
    format("~n== ~w passed, ~w failed, ~w skipped ==~n", [P, F, S]),
    ( F =:= 0 -> true ; halt(1) ).

% ===========================================================================
% A. Real data
% ===========================================================================

section_real(Out) :-
    atom_concat(Out, '/store', Store),
    format("~n-- A. real data (~w) --~n", [Store]),
    load_abi_store(Store),
    aggregate_all(count, symprov(_, _, _, _), NP),
    aggregate_all(count, symprov('libc.so.6', _, _, _), NLibc),
    aggregate_all(count, symreq(_, _, _, _, _), NR),
    aggregate_all(count, symreq(_, _, none, none, _), NU),
    format("        store: symprov=~w (libc.so.6=~w) symreq=~w (~w unversioned)~n", [NP, NLibc, NR, NU]),
    Bin = '/bin/ls', So = 'libc.so.6',

    check('A1 requirement evidence for /bin/ls is complete (#2)',
          req_evidence(Bin, readelf, complete, _)),
    check('A2 provider evidence for libc.so.6 is complete, from .symbols (#2)',
          prov_evidence(So, symbols, _, complete)),
    check('A3 NEEDED sonames ingested and checked (#2)',
          ( needed(Bin, So), needed(Bin, 'libselinux.so.1') )),
    check('A4 exact identity: __libc_start_main@GLIBC_2.34 attributed to libc.so.6 via version index (#1,#3)',
          symreq(Bin, '__libc_start_main', 'GLIBC_2.34', So, 'GLOBAL')),
    check('A5 libselinux requirement attributed to libselinux.so.1, not libc (#3)',
          symreq(Bin, freecon, 'LIBSELINUX_1.0', 'libselinux.so.1', _)),
    check('A6 unversioned (weak) requirements preserved, not dropped (#2)',
          symreq(Bin, '__gmon_start__', none, none, 'WEAK')),
    check('A7 no bare-name provider rows: every symprov row carries a node (#1)',
          \+ ( symprov(_, _, N, _), ( N == '' ; N == none ) )),

    % Curated floors on the Debian package axis == dpkg-shlibdeps' declared deps.
    check('A8 abi_floor(/bin/ls, libc.so.6) == 2.34 (#4)',
          ( abi_floor(Bin, So, F1), report(floor_libc, F1), F1 == '2.34' )),
    check('A9 abi_floor(/bin/ls, libselinux.so.1) == 3.1~ (tilde preserved) (#4)',
          ( abi_floor(Bin, 'libselinux.so.1', F2), report(floor_selinux, F2), F2 == '3.1~' )),
    atom_concat(Out, '/declared_depends.txt', DepFile),
    (   exists_file(DepFile)
    ->  read_file_to_string(DepFile, DepS, []),
        check('A10 floors equal the package''s declared Pre-Depends/Depends (ground truth) (#4)',
              ( abi_floor(Bin, So, FL), abi_floor(Bin, 'libselinux.so.1', FS),
                format(atom(E1), "libc6 (>= ~w)", [FL]),
                format(atom(E2), "libselinux1 (>= ~w)", [FS]),
                sub_string(DepS, _, _, _, E1), sub_string(DepS, _, _, _, E2) ))
    ;   skip('A10 declared_depends.txt not present')
    ),

    % Verdicts over the ACTUAL release axis.
    release_axis(So, Axis),
    report(libc_release_axis, Axis),
    check('A11 release axis is non-empty and made of real package versions (#5)',
          ( Axis = [_|_], forall(member(A, Axis), (rel_term(A, deb(_, _, _)))) )),
    check('A12 verdict at the .symbols evidence release is compatible(exact)',
          ( prov_evidence(So, symbols, R0, complete), release(So, R0, R0A),
            abi_verdict(Bin, So, R0A, V), report(verdict_at_evidence_release, R0A-V),
            V == compatible(exact) )),
    check('A13 abi_range over the real axis: both ends carry compatible verdicts (#5)',
          ( abi_range(Bin, So, RR), RR = range(Min, Max, Pairs),
            report(range, Min-Max),
            memberchk(Min-compatible(_), Pairs), memberchk(Max-compatible(_), Pairs) )),
    check('A14 older hypothetical release 2.31-0ubuntu9.9 -> incompatible(below_floor __libc_start_main@GLIBC_2.34) (#4,#5)',
          ( abi_verdict(Bin, So, '2.31-0ubuntu9.9', V31), report(verdict_2_31, V31),
            V31 = incompatible(L31),
            memberchk(below_floor('__libc_start_main'@'GLIBC_2.34', '2.34'), L31) )),
    check('A15 valid lower bound: axis extended below the floor -> min is the first COMPATIBLE release, not the lowest (#5)',
          ( abi_range(Bin, So, ['2.31-0ubuntu9.9', '2.34-0ubuntu3' | Axis], none, RX),
            RX = range(MinX, MaxX, PX), report(extended_range, MinX-MaxX),
            MinX == '2.34-0ubuntu3', last(Axis, MaxX),
            memberchk('2.31-0ubuntu9.9'-incompatible(_), PX) )),
    check('A16 beyond the evidence release -> compatible(extrapolated), never exact',
          ( abi_verdict(Bin, So, '99:1.0', VX), VX == compatible(extrapolated) )),

    % Exact node identity on real data: same symbol, other node -> hard veto.
    check('A17 getenv@GLIBC_2.99 (symbol present, node absent) -> incompatible(missing) (#1)',
          with_extra_req(Bin, getenv, 'GLIBC_2.99', So,
                         ( abi_verdict(Bin, So, '2.35-0ubuntu3', VG), report(verdict_getenv_2_99, VG),
                           VG = incompatible(LG), memberchk(missing(getenv@'GLIBC_2.99'), LG) ))),
    check('A18 pthread_setname_np@GLIBC_2.12 (non-default node, real) -> provided; @GLIBC_2.13 -> missing (#1)',
          ( with_extra_req(Bin, pthread_setname_np, 'GLIBC_2.12', So,
                           abi_verdict(Bin, So, '2.35-0ubuntu3', compatible(_))),
            with_extra_req(Bin, pthread_setname_np, 'GLIBC_2.13', So,
                           abi_verdict(Bin, So, '2.35-0ubuntu3', incompatible(_))) )),

    check('A19 soname mismatch is a hard veto: offering libc.so.7 for a libc.so.6 NEEDED (#2)',
          ( abi_verdict(Bin, 'libc.so.7', '2.35-0ubuntu3', VS), report(verdict_libc7, VS),
            VS == incompatible([soname_mismatch(offered('libc.so.7'), needed('libc.so.6'))]) )),
    check('A20 a soname the binary does not need -> not_needed',
          abi_verdict(Bin, 'libz.so.1', '1.0', not_needed('libz.so.1'))),
    check('A21 binary with no requirement evidence -> unknown, not compatible (#2)',
          abi_verdict('/nonexistent/bin', So, '2.35-0ubuntu3', unknown([no_requires_evidence(_)]))),
    check('A22 weak unversioned refs never veto (statuses are weak_unresolved)',
          ( prov_evidence(So, symbols, R0b, complete),
            findall(S, ( req_status(Bin, So, R0b, none, S), S \= provided(_, _) ), Odd),
            report(non_provided_statuses, Odd),
            forall(member(O, Odd), O = weak_unresolved(_)) )),

    % Hypothetical in-soname removal: the max is refined downward, min stays valid.
    last(Axis, Top), Axis = [Bottom|_],
    check('A23 drop getenv@GLIBC_2.2.5 at the newest release -> max refined below it (or no_candidate on a 1-release axis)',
          ( abi_range(Bin, So, drop(getenv, 'GLIBC_2.2.5', Top), RD), report(range_drop_getenv, RD),
            (   RD = range(_, MaxD, _) -> rel_term(MaxD, MD), rel_term(Top, TD), rel_lt(MD, TD)
            ;   RD = no_candidate(_), Axis = [_] ) )),
    check('A24 drop __libc_start_main@GLIBC_2.34 at the oldest release -> no_candidate (squeeze)',
          ( abi_range(Bin, So, drop('__libc_start_main', 'GLIBC_2.34', Bottom), RD2),
            RD2 = no_candidate(_) )).

with_extra_req(Bin, Sym, Node, So, Goal) :-
    setup_call_cleanup(assertz(abi_resolve:symreq(Bin, Sym, Node, So, 'GLOBAL')),
                       Goal,
                       retract(abi_resolve:symreq(Bin, Sym, Node, So, 'GLOBAL'))).

% ===========================================================================
% B. Version axes (Debian package axis via the frozen deb/3 comparison)
% ===========================================================================

section_axes :-
    format("~n-- B. version axes --~n", []),
    check('B1 3.1~ < 3.1 (tilde sorts first) (#4)',        deb_lt('3.1~', '3.1')),
    check('B2 3.1~ < 3.1~rc1 (#4)',                          deb_lt('3.1~', '3.1~rc1')),
    check('B3 epoch: 1:2.3-1 > 3.1, not 0 (#4)',             deb_lt('3.1', '1:2.3-1')),
    check('B4 revision: 2.35 < 2.35-0ubuntu3 < 2.35-0ubuntu3.15 (#4)',
          ( deb_lt('2.35', '2.35-0ubuntu3'), deb_lt('2.35-0ubuntu3', '2.35-0ubuntu3.15') )),
    check('B5 four components kept: 2.2.5 < 2.2.5.1 < 2.2.6 (#4)',
          ( deb_lt('2.2.5', '2.2.5.1'), deb_lt('2.2.5.1', '2.2.6') )),
    check('B6 minimum 0 (private) is below everything',      deb_lt('0', '0.0.1')),
    check('B7 non-deb release ids are labels that only match themselves',
          ( rel_term('file:/tmp/x.so', label(_)), rel_le(label(a), label(a)), \+ rel_le(label(a), label(b)),
            \+ rel_le(label(a), deb(0, [s([], 1)], [])) )),
    check('B8 ELF nodes are never ordered numerically: GLIBC_2.34 vs GLIBC_2.4 is identity only (#1,#4)',
          ( \+ catch(rel_lt('GLIBC_2.4', 'GLIBC_2.34'), _, fail),
            rel_term('GLIBC_2.34', label('GLIBC_2.34')) )).

deb_lt(A, B) :-
    rel_term(A, DA), rel_term(B, DB),
    DA = deb(_, _, _), DB = deb(_, _, _),
    rel_lt(DA, DB), \+ rel_lt(DB, DA).

% ===========================================================================
% C. Model fixtures (in-memory stores)
% ===========================================================================

fx_clear :- abi_store_clear.
fx(Fact) :- assertz(abi_resolve:Fact).
fx_since(A, since(D, A)) :- parse_deb_version(A, D).
fx_rel(A, R) :- rel_term(A, R).
fx_releases(So, Atoms) :- forall(member(A, Atoms), ( fx_rel(A, R), fx(release(So, R, A)) )).

section_model :-
    format("~n-- C. model fixtures --~n", []),

    % C1: Astra #1 -- foo@LIB_1 required, only foo@LIB_2 exported (same soname).
    fx_clear,
    fx_since('1.0', S1), fx_rel('2.0-1', R2),
    fx(symprov('libfoo.so.1', foo, 'LIB_2', S1)),
    fx(symprov('libfoo.so.1', foo_legacy, 'LIB_1', S1)),
    fx(prov_evidence('libfoo.so.1', symbols, R2, complete)),
    fx(symreq(usefoo, foo, 'LIB_1', 'libfoo.so.1', 'GLOBAL')),
    fx(needed(usefoo, 'libfoo.so.1')),
    fx(req_evidence(usefoo, readelf, complete, usefoo)),
    check('C1 foo@LIB_1 required vs foo@LIB_2 exported -> incompatible([missing(foo@LIB_1)]) (#1)',
          ( abi_verdict(usefoo, 'libfoo.so.1', '2.0-1', V1), report(verdict, V1),
            V1 == incompatible([missing(foo@'LIB_1')]) )),
    fx(symprov('libfoo.so.1', foo, 'LIB_1', S1)),
    check('C1b adding the exact foo@LIB_1 row -> compatible(exact) (#1)',
          abi_verdict(usefoo, 'libfoo.so.1', '2.0-1', compatible(exact))),

    % C2: Astra #5 -- the range(1,5)-with-floor-5 bug.
    fx_clear,
    fx_since('1.0', Sa), fx_since('5.0', Sb), fx_rel('5.0', R5),
    fx(symprov('libr.so.1', a, 'L', Sa)),
    fx(symprov('libr.so.1', b, 'L', Sb)),
    fx(prov_evidence('libr.so.1', symbols, R5, complete)),
    fx(symreq(bin, a, 'L', 'libr.so.1', 'GLOBAL')),
    fx(symreq(bin, b, 'L', 'libr.so.1', 'GLOBAL')),
    fx(needed(bin, 'libr.so.1')),
    fx(req_evidence(bin, readelf, complete, bin)),
    fx_releases('libr.so.1', ['1.0', '2.0', '3.0', '4.0', '5.0']),
    check('C2 releases 1..5, a since 1, b since 5 -> range(5.0, 5.0), NOT range(1,5) (#5)',
          ( abi_range(bin, 'libr.so.1', RR), report(range, RR), RR = range('5.0', '5.0', _) )),
    check('C2b verdict at 1.0 is incompatible(below_floor(b@L, 5.0)) (#5)',
          abi_verdict(bin, 'libr.so.1', '1.0', incompatible([below_floor(b@'L', '5.0')]))),
    fx_releases('libr.so.1', ['6.0']),
    check('C2c a release newer than the evidence (6.0) is compatible(extrapolated); range(5.0, 6.0) (#5)',
          ( abi_verdict(bin, 'libr.so.1', '6.0', compatible(extrapolated)),
            abi_range(bin, 'libr.so.1', range('5.0', '6.0', _)) )),
    check('C2d releases adding no symbols (2.0..4.0) are evaluated on the axis, each incompatible (#5)',
          ( abi_range(bin, 'libr.so.1', range(_, _, P)),
            memberchk('3.0'-incompatible(_), P), length(P, 6) )),

    % C3: readelf-only provider evidence (at R0): older releases are UNKNOWN.
    fx_clear,
    fx_rel('2.0', R20),
    fx(symprov('libe.so.1', s, 'N', at(R20))),
    fx(prov_evidence('libe.so.1', elf, R20, complete)),
    fx(symreq(bin, s, 'N', 'libe.so.1', 'GLOBAL')),
    fx(needed(bin, 'libe.so.1')),
    fx(req_evidence(bin, readelf, complete, bin)),
    check('C3 at(2.0) evidence: 1.0 -> unknown, 2.0 -> compatible(exact), 3.0 -> compatible(extrapolated) (#2)',
          ( abi_verdict(bin, 'libe.so.1', '1.0', unknown([unknown(s@'N', evidence_release(_))])),
            abi_verdict(bin, 'libe.so.1', '2.0', compatible(exact)),
            abi_verdict(bin, 'libe.so.1', '3.0', compatible(extrapolated)) )),
    fx_releases('libe.so.1', ['1.0', '2.0']),
    check('C3b range over [1.0, 2.0] with at(2.0) evidence -> range(2.0, 2.0); 1.0 reported unknown (#2,#5)',
          ( abi_range(bin, 'libe.so.1', range('2.0', '2.0', P3)), memberchk('1.0'-unknown(_), P3) )),

    % C4: Astra #2 -- incomplete evidence is never a false compat or false veto.
    fx_clear,
    fx(symreq(bin, s, 'N', 'libe.so.1', 'GLOBAL')),
    fx(needed(bin, 'libe.so.1')),
    fx(req_evidence(bin, readelf, complete, bin)),
    check('C4 needed soname with NO provider evidence -> unknown([no_provider_evidence]) (#2)',
          abi_verdict(bin, 'libe.so.1', '1.0', unknown([no_provider_evidence('libe.so.1')]))),
    fx_clear,
    fx(req_evidence(bin2, readelf, missing_file, bin2)),
    fx_rel('1.0', R10), fx(prov_evidence('libe.so.1', elf, R10, complete)),
    check('C4b requirement evidence missing_file -> unknown([requires_evidence(missing_file, _)]) (#2)',
          abi_verdict(bin2, 'libe.so.1', '1.0', unknown([requires_evidence(missing_file, _)]))),
    fx_clear,
    fx(symreq(bin, plain_fn, none, none, 'GLOBAL')),
    fx(needed(bin, 'liba.so.1')), fx(needed(bin, 'libb.so.1')),
    fx(req_evidence(bin, readelf, complete, bin)),
    fx(prov_evidence('liba.so.1', elf, R10, complete)),
    check('C4c unversioned obligation, one NEEDED lib lacks evidence -> unknown, not missing (#2)',
          abi_verdict(bin, 'liba.so.1', '1.0', unknown([unknown(plain_fn, no_provider_evidence('libb.so.1'))]))),
    fx(prov_evidence('libb.so.1', elf, R10, complete)),
    check('C4d unversioned obligation, all evidence complete, nobody exports it -> incompatible([missing(plain_fn)]) (#2)',
          abi_verdict(bin, 'liba.so.1', '1.0', incompatible([missing(plain_fn)]))),
    fx(symprov('libb.so.1', plain_fn, 'Base', at(R10))),
    check('C4e unversioned obligation satisfied by ANY exported node of another NEEDED lib -> compatible (#2)',
          abi_verdict(bin, 'liba.so.1', '1.0', compatible(exact))),

    % C5: non-numeric nodes are first-class identities.
    fx_clear,
    fx(symprov('libpub.so.1', pub_fn, 'PUBLIC', at(R10))),
    fx(prov_evidence('libpub.so.1', elf, R10, complete)),
    fx(symreq(bin, pub_fn, 'PUBLIC', 'libpub.so.1', 'GLOBAL')),
    fx(needed(bin, 'libpub.so.1')),
    fx(req_evidence(bin, readelf, complete, bin)),
    check('C5 pub_fn@PUBLIC (no numeric bound) is matched exactly -> compatible (#1,#2)',
          abi_verdict(bin, 'libpub.so.1', '1.0', compatible(exact))),
    check('C5b pub_fn@PUBLIC_2 -> incompatible([missing(pub_fn@PUBLIC_2)]) (#1)',
          with_extra_req(bin, pub_fn, 'PUBLIC_2', 'libpub.so.1',
                         abi_verdict(bin, 'libpub.so.1', '1.0', incompatible([missing(pub_fn@'PUBLIC_2')])))),

    % C6: a weak reference to an absent symbol is not a veto.
    fx(symreq(bin, maybe_fn, 'PUBLIC', 'libpub.so.1', 'WEAK')),
    check('C6 weak requirement of an absent symbol -> still compatible (weak_unresolved)',
          ( abi_verdict(bin, 'libpub.so.1', '1.0', compatible(exact)),
            req_status(bin, 'libpub.so.1', R10, none, weak_unresolved(maybe_fn@'PUBLIC')) )).

% ===========================================================================
% D. ELF fixtures built by run_abi_verify.sh (real ingest pipeline)
% ===========================================================================

section_elf_fixtures(Out) :-
    format("~n-- D. ELF fixtures (~w/fx) --~n", [Out]),
    atom_concat(Out, '/fx', FX),
    atom_concat(FX, '/BUILT', Built),
    (   exists_file(Built)
    ->  elf_fixture_checks(FX)
    ;   (   getenv('ABI_ALLOW_SKIP', '1')
        ->  skip('D* ELF fixtures not built (no gcc); ABI_ALLOW_SKIP=1')
        ;   check('D0 ELF fixtures were built (gcc available)', fail)
        )
    ),
    symbols_fixture_checks(FX).

elf_fixture_checks(FX) :-
    atom_concat(FX, '/usefoo', UseFoo),
    atom_concat(FX, '/usecommon', UseCommon),
    atom_concat(FX, '/usepub', UsePub),

    atom_concat(FX, '/store_foo_v2', StV2), load_abi_store(StV2),
    check('D1 [ELF] usefoo vs libfoo v2 (foo@LIB_2 only, LIB_1 node present) -> incompatible([missing(foo@LIB_1)]); loader agrees (#1)',
          ( abi_verdict(UseFoo, 'libfoo.so.1', '2.0-1', V), report(verdict, V),
            V == incompatible([missing(foo@'LIB_1')]) )),
    check('D1b [ELF] v2 store has foo@LIB_2 and foo_legacy@LIB_1 as exact identities (#1)',
          ( symprov('libfoo.so.1', foo, 'LIB_2', at(_)), symprov('libfoo.so.1', foo_legacy, 'LIB_1', at(_)),
            \+ symprov('libfoo.so.1', foo, 'LIB_1', _) )),
    atom_concat(FX, '/store_foo_v1', StV1), load_abi_store(StV1),
    check('D2 [ELF] usefoo vs libfoo v1 (foo@LIB_1) -> compatible(exact); range(1.0-1, 1.0-1) (#1,#5)',
          ( abi_verdict(UseFoo, 'libfoo.so.1', '1.0-1', compatible(exact)),
            abi_range(UseFoo, 'libfoo.so.1', range('1.0-1', '1.0-1', _)) )),

    atom_concat(FX, '/store_common', StC), load_abi_store(StC),
    check('D3 [ELF] COMMON_1 in two libs: alpha_fn -> libalpha.so.1, beta_fn -> libbeta.so.1 (version-index join) (#3)',
          ( symreq(UseCommon, alpha_fn, 'COMMON_1', 'libalpha.so.1', 'GLOBAL'),
            symreq(UseCommon, beta_fn,  'COMMON_1', 'libbeta.so.1',  'GLOBAL'),
            \+ symreq(UseCommon, alpha_fn, _, 'libbeta.so.1', _),
            \+ symreq(UseCommon, beta_fn,  _, 'libalpha.so.1', _) )),
    check('D3b [ELF] both COMMON_1 providers verdict compatible(exact) (#3)',
          ( abi_verdict(UseCommon, 'libalpha.so.1', '1.0-1', compatible(exact)),
            abi_verdict(UseCommon, 'libbeta.so.1',  '1.0-1', compatible(exact)) )),

    atom_concat(FX, '/store_pub', StP), load_abi_store(StP),
    check('D4 [ELF] pub_fn@PUBLIC (non-numeric node) and plain_fn (unversioned) both ingested as obligations (#2)',
          ( symreq(UsePub, pub_fn, 'PUBLIC', 'libpub.so.1', 'GLOBAL'),
            symreq(UsePub, plain_fn, none, none, 'GLOBAL') )),
    check('D4b [ELF] unversioned export recorded as @Base; verdicts compatible(exact) (#2)',
          ( symprov('libplain.so.1', plain_fn, 'Base', at(_)),
            abi_verdict(UsePub, 'libpub.so.1', '1.0-1', compatible(exact)),
            abi_verdict(UsePub, 'libplain.so.1', '1.0-1', compatible(exact)) )),
    atom_concat(FX, '/store_pub_nolibpub', StPn), load_abi_store(StPn),
    check('D5 [ELF] libpub evidence absent -> unknown([no_provider_evidence(libpub.so.1)]), not compatible (#2)',
          abi_verdict(UsePub, 'libpub.so.1', '1.0-1', unknown([no_provider_evidence('libpub.so.1')]))),
    atom_concat(FX, '/store_pub_noplain', StPp), load_abi_store(StPp),
    check('D5b [ELF] unversioned plain_fn with libplain evidence absent -> unknown, not missing (#2)',
          abi_verdict(UsePub, 'libpub.so.1', '1.0-1', unknown([unknown(plain_fn, no_provider_evidence('libplain.so.1'))]))),

    atom_concat(FX, '/store_missing', StM), load_abi_store(StM),
    atom_concat(FX, '/does-not-exist', Missing),
    check('D6 [ELF] missing ELF -> req_evidence missing_file recorded; verdict unknown (#2)',
          ( req_evidence(Missing, readelf, missing_file, _),
            \+ symreq(Missing, _, _, _, _),
            abi_verdict(Missing, 'libc.so.6', '1.0', unknown([requires_evidence(missing_file, _)])) )).

symbols_fixture_checks(FX) :-
    atom_concat(FX, '/simple', StS), load_abi_store(StS),
    check('D7 [.symbols] simple cases: 2 sonames, |/* lines, comments, private 0, tilde + epoch minimums (#7)',
          ( aggregate_all(count, symprov(_, _, _, _), 5),
            symprov('libsimple.so.1', simple_old, 'SIMPLE_1.0', since(deb(1, _, _), '1:0.9-2')),
            symprov('libsimple.so.1', simple_new, 'SIMPLE_1.2', since(_, '1.2~rc1')),
            symprov('libsimple.so.1', '_private_thing', 'SIMPLE_PRIVATE', since(_, '0')),
            symprov('libsimple-extra.so.0', extra_fn, 'Base', since(_, '1.1')),
            prov_evidence('libsimple.so.1', symbols, _, complete),
            prov_evidence('libsimple-extra.so.0', symbols, _, complete) )),
    atom_concat(FX, '/tmpl_arch_amd64', StA), load_abi_store(StA),
    check('D8 [.symbols] (arch=..) processed with --arch amd64: amd64 rows kept, !amd64 dropped, (optional) kept (#7)',
          ( aggregate_all(count, symprov(_, _, _, _), 4),
            symprov('libtmpl.so.1', only_amd64, 'Base', _), symprov('libtmpl.so.1', bits64, 'Base', _),
            symprov('libtmpl.so.1', maybe_fn, 'Base', _), symprov('libtmpl.so.1', plain_fn, 'Base', _),
            \+ symprov('libtmpl.so.1', not_amd64, _, _) )),
    forall(member(T, [tmpl_symver, tmpl_cxx, tmpl_arch]),
           ( atomic_list_concat([FX, '/', T, '/symprov.jsonl'], F),
             format(atom(Name), "D9 [.symbols] ~w rejected loudly: no store written (#7)", [T]),
             check(Name, \+ exists_file(F)) )).
