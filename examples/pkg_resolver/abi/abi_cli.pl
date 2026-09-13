:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% abi_cli.pl -- a small callable driver over the ABI resolver.
%
%   swipl -q -g main -t halt examples/pkg_resolver/abi/abi_cli.pl -- \
%         <store-dir> <cmd> <args...>
%
% Commands:
%   min    <binary> <soname>                 derived verneed floor
%   newest <binary> <soname> [DropSym DropAt] newest ABI-compatible version
%   range  <binary> <soname> [DropSym DropAt] [min, max] or no_candidate
%   compat <binary> <soname> <version>        containment check at a version

:- use_module(abi_resolve).

main :-
    current_prolog_flag(argv, Argv),
    ( Argv = [Dir, Cmd | Args] -> true ; usage, halt(2) ),
    load_abi_store(Dir),
    run(Cmd, Args).

usage :-
    format(user_error,
           "usage: abi_cli.pl -- <store-dir> min|newest|range|compat <args>~n", []).

drop_of([], none).
drop_of([Sym, At], drop(SymA, At)) :- atom_string(SymA, Sym).

run(min, [Bin, So]) :- !,
    ( abi_min(Bin, So, Min) -> format("min ~w~n", [Min]) ; format("min none~n", []) ).
run(newest, [Bin, So | DropArgs]) :- !,
    soname_candidates(So, Cands),
    drop_of(DropArgs, Drop),
    newest_abi_compatible(Bin, So, Cands, Drop, R),
    format("newest ~w~n", [R]).
run(range, [Bin, So | DropArgs]) :- !,
    soname_candidates(So, Cands),
    drop_of(DropArgs, Drop),
    abi_range(Bin, So, Cands, Drop, R),
    format("range ~w~n", [R]).
run(compat, [Bin, So, V]) :- !,
    ( abi_compatible(Bin, So, V)
    ->  format("compatible ~w yes~n", [V])
    ;   missing_syms(Bin, So, V, none, M), length(M, N),
        format("compatible ~w no (missing ~w)~n", [V, N])
    ).
run(_, _) :- usage, halt(2).
