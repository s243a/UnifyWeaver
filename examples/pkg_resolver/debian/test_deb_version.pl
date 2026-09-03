:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_deb_version.pl -- curated Debian version table (UNGATED) plus a
% gated dpkg --compare-versions cross-check (≥2000 seeded pairs).
%
%   swipl -q -g test_deb_version -t halt examples/pkg_resolver/debian/test_deb_version.pl

:- module(test_deb_version, [test_deb_version/0, dpkg_arm_ran/0]).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(apply), [maplist/2]).
:- use_module('../resolver', [version_lt/2, satisfies/2]).
:- use_module(deb_parse, [parse_deb_version/2]).

test_deb_version :-
    nb_setval(dpkg_arm_ran, false),
    run_tests([deb_version_curated, deb_version_dpkg]),
    (   nb_current(dpkg_arm_ran, true)
    ->  format("dpkg arm: RAN~n", [])
    ;   format("dpkg arm: SKIPPED (dpkg --compare-versions not available)~n", [])
    ).

dpkg_arm_ran :-
    nb_current(dpkg_arm_ran, true).

curated_pair("1.0~rc1", lt, "1.0").
curated_pair("1.0~~", lt, "1.0~").
curated_pair("1.0~", lt, "1.0").
curated_pair("2:1.0", gt, "1:99.0").
curated_pair("1:1.0", lt, "2:0.1").
curated_pair("1.0-1", lt, "1.0-2").
curated_pair("1.0", lt, "1.0-1").
curated_pair("1.0a", lt, "1.0b").
curated_pair("1.0+dfsg", lt, "1.0+dfsg1").
curated_pair("1.0+dfsg1", lt, "1.0+dfsg2").
curated_pair("0.1", lt, "0.10").
curated_pair("1.0~alpha", lt, "1.0~beta").
curated_pair("1.0~beta2", lt, "1.0").
curated_pair("3.11.2-6", lt, "3.11.2-6+deb12u1").
curated_pair("1.2.3", eq, "1.2.3").
curated_pair("0:1.0", eq, "1.0").
curated_pair("1.0-0", eq, "1.0").   % dpkg: missing revision ≡ 0
curated_pair("2.36-9", lt, "2.38-1").
curated_pair("1~~", lt, "1~").
curated_pair("1.0+b1", lt, "1.0+b2").

pair_rel(A, lt, B) :-
    parse_deb_version(A, DA),
    parse_deb_version(B, DB),
    version_lt(DA, DB),
    \+ version_lt(DB, DA).
pair_rel(A, gt, B) :-
    pair_rel(B, lt, A).
pair_rel(A, eq, B) :-
    parse_deb_version(A, DA),
    parse_deb_version(B, DB),
    \+ version_lt(DA, DB),
    \+ version_lt(DB, DA).

:- begin_tests(deb_version_curated).

test(curated_table) :-
    findall(A-Rel-B, curated_pair(A, Rel, B), Pairs),
    assertion(Pairs \== []),
    maplist(check_pair, Pairs).

check_pair(A-Rel-B) :-
    (   pair_rel(A, Rel, B)
    ->  true
    ;   format(user_error, "CURATED FAIL ~w ~w ~w~n", [A, Rel, B]),
        fail
    ).

test(constraint_forms) :-
    parse_deb_version("1.0~rc1", R),
    parse_deb_version("1.0", V),
    assertion(satisfies(V, gte(R))),
    assertion(satisfies(R, lt(V))),
    assertion(satisfies(V, lte(V))),
    assertion(satisfies(V, gt(R))),
    assertion(\+ satisfies(R, gt(V))),
    assertion(satisfies(V, eq(V))).

:- end_tests(deb_version_curated).

dpkg_available :-
    catch(
        (   process_create(path(dpkg), ['--compare-versions', '1.0~rc1', 'lt', '1.0'],
                [process(Pid)]),
            process_wait(Pid, exit(0))
        ),
        _,
        fail
    ).

:- begin_tests(deb_version_dpkg).

test(dpkg_curated_and_random, [condition(dpkg_available)]) :-
    nb_setval(dpkg_arm_ran, true),
    findall(A-Rel-B, curated_pair(A, Rel, B), Curated),
    maplist(check_dpkg_pair, Curated),
    random_pairs(2000, 0xdeb00c1e, Pairs),
    maplist(check_dpkg_pair, Pairs),
    length(Curated, NC),
    length(Pairs, N),
    format("dpkg cross-check: ~w curated + ~w random pairs, 0 mismatches~n",
           [NC, N]).

check_dpkg_pair(A-Rel-B) :-
    dpkg_rel(A, DpkgRel, B),
    (   pair_rel(A, Rel, B),
        (   Rel == DpkgRel
        ;   Rel == eq, DpkgRel == eq
        )
    ->  true
    ;   pair_rel(A, DpkgRel, B)
    ->  true
    ;   format(user_error, "DPKG MISMATCH ~w  swi=~w dpkg=~w  ~w~n",
               [A, Rel, DpkgRel, B]),
        fail
    ).

:- end_tests(deb_version_dpkg).

dpkg_rel(A, Rel, B) :-
    dpkg_exit(A, 'lt', B, L),
    dpkg_exit(A, 'gt', B, G),
    (   L =:= 0 -> Rel = lt
    ;   G =:= 0 -> Rel = gt
    ;   Rel = eq
    ).

dpkg_exit(A, Op, B, Code) :-
    process_create(path(dpkg), ['--compare-versions', A, Op, B],
        [process(Pid)]),
    process_wait(Pid, exit(Code)).

% Seeded mulberry32-style generator, same family as gen_catalogs.mjs.
random_pairs(N, Seed, Pairs) :-
    random_pairs_loop(N, Seed, [], Acc),
    reverse(Acc, Pairs).

random_pairs_loop(0, _S, Acc, Acc) :- !.
random_pairs_loop(N, S0, Acc, Out) :-
    next_rand(S0, S1, R1),
    next_rand(S1, S2, R2),
    next_rand(S2, S3, R3),
    synth_ver(R1, R2, A),
    synth_ver(R2, R3, B),
    N1 is N - 1,
    dpkg_rel(A, DRel, B),
    random_pairs_loop(N1, S3, [A-DRel-B|Acc], Out).

next_rand(A0, A, U) :-
    A1 is (A0 + 0x6d2b79f5) /\ 0xffffffff,
    T0 is A1 xor (A1 >> 15),
    T1 is (T0 * (1 \/ A1)) /\ 0xffffffff,
    T2 is T1 xor (T1 >> 7),
    T3 is (T2 * (61 \/ T1)) /\ 0xffffffff,
    T4 is T3 xor T2,
    A is T4,
    U is float((T4 xor (T4 >> 14)) /\ 0xffffffff) / 4294967296.0.

synth_ver(R1, R2, S) :-
    Epoch is floor(R1 * 3),
    Maj is floor(R2 * 5),
    Min is floor((R1 + R2) * 4) mod 4,
    (   R1 < 0.25
    ->  format(string(S), "~w.~w~w", [Maj, Min, "~rc1"])
    ;   R1 < 0.4
    ->  format(string(S), "~w.~w+dfsg~w", [Maj, Min, Min])
    ;   R1 < 0.55
    ->  format(string(S), "~w:~w.~w-~w", [Epoch, Maj, Min, Min])
    ;   R1 < 0.7
    ->  format(string(S), "~w.~w-~w", [Maj, Min, Min])
    ;   R1 < 0.85
    ->  format(string(S), "~w.~w~~", [Maj, Min])
    ;   format(string(S), "~w.~w", [Maj, Min])
    ).
