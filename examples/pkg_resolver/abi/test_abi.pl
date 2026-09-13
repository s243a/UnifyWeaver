:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_abi.pl -- verify the symbol-level ABI lane on THIS machine's real data.
% The store must already be built (see run_abi_verify.sh) into a directory
% passed as the first program argument (default: ./.out/store).
%
%   swipl -q -g run -t halt examples/pkg_resolver/abi/test_abi.pl -- <store-dir>
%
% Asserts (the PoC's numbers, libc6 / /bin/ls on Ubuntu 22.04):
%   abi_min(/bin/ls, libc.so.6)             == 2.34
%   abi_compatible(/bin/ls, libc.so.6, 2.35) is TRUE (0 missing)
%   newest_abi_compatible(no drop)          == compatible(2.35), range(2.34,2.35)
%   simulated removal of getenv at 2.30     -> cap 2.29, min 2.34 > max 2.29
%                                           -> no_candidate (the min>max squeeze)

:- use_module(abi_resolve).

:- dynamic pass_count/1, fail_count/1.
pass_count(0).
fail_count(0).

check(Name, Goal) :-
    (   catch(Goal, E, (print_message(error, E), fail))
    ->  format("  PASS  ~w~n", [Name]),
        retract(pass_count(P)), P1 is P + 1, assertz(pass_count(P1))
    ;   format("  FAIL  ~w~n", [Name]),
        retract(fail_count(F)), F1 is F + 1, assertz(fail_count(F1))
    ).

store_dir(Dir) :-
    ( current_prolog_flag(argv, [D | _]), D \== [] -> Dir = D ; Dir = './.out/store' ).

run :-
    store_dir(Dir),
    format("~n== ABI lane verification (store: ~w) ==~n", [Dir]),
    load_abi_store(Dir),
    aggregate_all(count, abi_resolve:symprov(_,_), NP),
    aggregate_all(count, abi_resolve:symreq(_,_), NR),
    aggregate_all(count, ( abi_resolve:symprov(K,_), atom_concat('libc.so.6|', _, K) ), NLibc),
    format("  store: symprov=~w (libc.so.6=~w) symreq=~w~n~n", [NP, NLibc, NR]),

    Bin = '/bin/ls', So = 'libc.so.6',
    soname_candidates(So, Cands),
    length(Cands, NC),
    format("  libc.so.6 version axis: ~w distinct versions~n", [NC]),

    check('abi_min(/bin/ls, libc.so.6) == 2.34',
          ( abi_min(Bin, So, Min), report('    abi_min', Min), Min == '2.34' )),

    check('abi_compatible(/bin/ls, libc.so.6, 2.35) is TRUE',
          abi_compatible(Bin, So, '2.35')),

    check('0 missing symbols at 2.35',
          ( missing_syms(Bin, So, '2.35', none, M), length(M, LM),
            report('    missing@2.35', LM), LM =:= 0 )),

    check('newest_abi_compatible (no drop) == compatible(2.35)',
          ( newest_abi_compatible(Bin, So, Cands, R),
            report('    newest', R), R == compatible('2.35') )),

    check('abi_range (no drop) == range(2.34, 2.35)',
          ( abi_range(Bin, So, Cands, RR),
            report('    range', RR), RR == range('2.34', '2.35') )),

    % Simulated ABI break: remove getenv at 2.30 (a symbol /bin/ls needs).
    check('/bin/ls needs getenv',
          req_sym(Bin, So, getenv, _)),

    check('removal cap: newest version still exporting getenv == 2.29',
          ( newest_providing(So, getenv, Cands, drop(getenv, '2.30'), Cap),
            report('    getenv cap', Cap), Cap == compatible('2.29') )),

    check('simulated removal -> no_candidate (min 2.34 > max 2.29 squeeze)',
          ( abi_range(Bin, So, Cands, drop(getenv, '2.30'), RD),
            report('    range(drop)', RD), RD = no_candidate(_) )),

    pass_count(P), fail_count(F),
    format("~n== ~w passed, ~w failed ==~n", [P, F]),
    ( F =:= 0 -> true ; halt(1) ).

report(Label, Val) :- format("~w = ~w~n", [Label, Val]).
