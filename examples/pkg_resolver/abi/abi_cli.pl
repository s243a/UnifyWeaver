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
%   verdict <binary> <soname> <release> [DropSym DropNode DropAt]
%       compatible(exact|extrapolated) | incompatible([...]) | unknown([...]) | not_needed(_)
%   status  <binary> <soname> <release>   one line per requirement (provided / missing / ...)
%   floor   <binary> <soname>             curated lower bound implied by `.symbols` (= dpkg-shlibdeps' dep)
%   axis    <soname>                      the ingested release candidates, ascending
%   range   <binary> <soname> [DropSym DropNode DropAt]
%       range(Min, Max, ...) over the release axis (both ends compatible), or
%       no_candidate / unknown / no_releases

:- use_module(abi_resolve).

main :-
    current_prolog_flag(argv, Argv),
    ( Argv = [Dir, Cmd | Args] -> true ; usage, halt(2) ),
    load_abi_store(Dir),
    atom_string(CmdA, Cmd),
    run(CmdA, Args).

usage :-
    format(user_error,
           "usage: abi_cli.pl -- <store-dir> verdict|status|floor|axis|range <args>~n", []).

drop_of([], none).
drop_of([Sym, Node, At], drop(Sym, Node, At)).

run(verdict, [Bin, So, Rel | DropArgs]) :- !,
    drop_of(DropArgs, Drop),
    abi_verdict(Bin, So, Rel, Drop, V),
    format("verdict ~w ~w ~w: ~q~n", [Bin, So, Rel, V]).
run(status, [Bin, So, Rel]) :- !,
    rel_term(Rel, R),
    forall(req_status(Bin, So, R, none, S), format("  ~q~n", [S])).
run(floor, [Bin, So]) :- !,
    (   abi_floor(Bin, So, F)
    ->  format("floor ~w ~w: ~w~n", [Bin, So, F])
    ;   format("floor ~w ~w: none (a requirement has no since() provider row)~n", [Bin, So])
    ).
run(axis, [So]) :- !,
    release_axis(So, Rels),
    format("axis ~w: ~w~n", [So, Rels]).
run(range, [Bin, So | DropArgs]) :- !,
    drop_of(DropArgs, Drop),
    abi_range(Bin, So, Drop, R),
    (   R = range(Min, Max, Pairs)
    ->  format("range ~w ~w: [~w, ~w]~n", [Bin, So, Min, Max]),
        forall(member(A-V, Pairs), format("  ~w: ~q~n", [A, V]))
    ;   R =.. [Kind, Pairs], is_list(Pairs)
    ->  format("range ~w ~w: ~w~n", [Bin, So, Kind]),
        forall(member(A-V, Pairs), format("  ~w: ~q~n", [A, V]))
    ;   format("range ~w ~w: ~q~n", [Bin, So, R])
    ).
run(_, _) :- usage, halt(2).
