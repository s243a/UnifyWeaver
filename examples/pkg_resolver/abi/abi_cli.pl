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
%       compatible(exact|curated|extrapolated) | incompatible([...]) | unknown([...]) | not_needed(_)
%   status  <binary> <soname> <release>   one line per requirement (provided / missing / ...)
%   floor   <binary> <soname>             curated lower bound implied by `.symbols` (= dpkg-shlibdeps' dep)
%   axis    <soname>                      the ingested release candidates, ascending
%   range   <binary> <soname> [DropSym DropNode DropAt]
%       range(Min, Max, ...) over the release axis (both ends compatible), or
%       no_candidate / unknown / no_releases
%   report  <binary>
%       one floor/axis/range block per NEEDED soname (ascending soname order);
%       a one-shot summary of the whole dependency set instead of repeating
%       floor/axis/range per soname by hand.
%   explain <binary> <soname> <release> [DropSym DropNode DropAt]
%       like verdict, but prints one human-readable line per reason in an
%       incompatible([...]) / unknown([...]) verdict (compatible: header only).

:- use_module(abi_resolve).

main :-
    current_prolog_flag(argv, Argv),
    ( Argv = [Dir, Cmd | Args] -> true ; usage, halt(2) ),
    load_abi_store(Dir),
    atom_string(CmdA, Cmd),
    run(CmdA, Args).

usage :-
    format(user_error,
           "usage: abi_cli.pl -- <store-dir> verdict|status|floor|axis|range|report|explain <args>~n", []).

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
run(report, [Bin]) :- !,
    findall(So, needed(Bin, So), Sos0), sort(Sos0, Sos),
    (   Sos == []
    ->  format("report ~w: no NEEDED sonames evidenced~n", [Bin])
    ;   forall(member(So, Sos), report_one(Bin, So))
    ).
run(explain, [Bin, So, Rel | DropArgs]) :- !,
    drop_of(DropArgs, Drop),
    abi_verdict(Bin, So, Rel, Drop, V),
    (   V = incompatible(Rs) -> Kind = incompatible
    ;   V = unknown(Rs)      -> Kind = unknown
    ;   Rs = [], Kind = V
    ),
    format("explain ~w ~w ~w: ~w~n", [Bin, So, Rel, Kind]),
    forall(member(R, Rs), (explain_line(R, T), format("  ~w~n", [T]))).
run(_, _) :- usage, halt(2).

% explain_line(Reason, Text): a human-readable line for one verdict reason.
explain_line(missing(S@N), T) :- !,
    format(atom(T), "symbol ~w (version node ~w) is absent at this release", [S, N]).
explain_line(missing(S@N, hypothetical_drop), T) :- !,
    format(atom(T), "symbol ~w@~w removed by the hypothetical drop", [S, N]).
explain_line(missing(S@N, observed_absent(Src, R1)), T) :- !,
    format(atom(T), "symbol ~w@~w absent at later release ~w (via ~w), so absent here", [S, N, R1, Src]).
explain_line(below_floor(S@N, Min), T) :- !,
    format(atom(T), "symbol ~w (version node ~w) first appears in release ~w (below_floor)", [S, N, Min]).
explain_line(unknown(S@N, Why), T) :- !,
    format(atom(T), "symbol ~w@~w: ~w", [S, N, Why]).
explain_line(soname_mismatch(offered(O), needed(Nd)), T) :- !,
    format(atom(T), "soname mismatch: offered ~w, needed ~w", [O, Nd]).
explain_line(X, T) :-
    format(atom(T), "~q", [X]).

% report_one(Bin, So): the floor/axis/range block for one NEEDED soname.
report_one(Bin, So) :-
    (   abi_floor(Bin, So, F)
    ->  true
    ;   F = none
    ),
    release_axis(So, Rels),
    abi_range(Bin, So, R),
    format("~w:~n", [So]),
    format("  floor: ~w~n", [F]),
    format("  axis:  ~w~n", [Rels]),
    format("  range: ~q~n", [R]).
