:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% abi_resolve.pl -- symbol-level ABI-compatibility resolver, the fine-grained
% generalization of the package resolver's coarse `provides`. It reuses the
% SAME interval-store idea one level down: instead of a package having a
% version tenure, each exported SYMBOL has a validity interval [intro, inf)
% WITHIN a soname (a library version V provides sym iff intro(sym) =< V, until
% a soname bump / removal). Given a binary's referenced versioned symbols
% (verneed floor) and a library's exported-symbol intervals, it computes the
% MIN and MAX (newest) compatible library version via symbol-set containment.
%
% Frozen resolver.pl / resolver_store.pl are NOT edited; version comparison is
% delegated to resolver:version_lt/2.
%
% Store rows (P/2 JSONL, load_p2_jsonl shape), produced by ingest_symbols.mjs:
%   symprov(Key, Val)  Key = 'SoName|Sym'   Val = 'Intro#inf'   (interval)
%   symreq(Key, Val)   Key = 'Binary|Sym'   Val = 'SoName#Ver'  (verneed)
%   needed(Binary, SoName)                                       (DT_NEEDED)

:- module(abi_resolve, [
    load_abi_store/1,
    abi_store_clear/0,
    provides_at/3,
    symprov_intro/3,
    req_sym/4,
    abi_min/3,
    abi_compatible/3,
    abi_compatible/4,
    missing_syms/5,
    soname_candidates/2,
    newest_providing/5,
    newest_abi_compatible/4,
    newest_abi_compatible/5,
    abi_range/4,
    abi_range/5
]).

:- use_module('../resolver', [version_lt/2]).
:- use_module(library(http/json)).

:- dynamic symprov/2.
:- dynamic symreq/2.
:- dynamic needed/2.

% ---------------------------------------------------------------------------
% Store loading (mirrors resolver_store:load_p2_jsonl / load_pairs)
% ---------------------------------------------------------------------------

abi_store_clear :-
    retractall(symprov(_, _)),
    retractall(symreq(_, _)),
    retractall(needed(_, _)).

load_abi_store(Dir) :-
    abi_store_clear,
    load_pairs(Dir, 'symprov.jsonl', symprov),
    load_pairs(Dir, 'symreq.jsonl',  symreq),
    load_pairs(Dir, 'needed.jsonl',  needed).

load_pairs(Dir, File, Pred) :-
    atomic_list_concat([Dir, '/', File], Path),
    (   exists_file(Path)
    ->  setup_call_cleanup(open(Path, read, S),
                           load_pair_lines(S, Pred),
                           close(S))
    ;   true
    ).

load_pair_lines(S, Pred) :-
    read_line_to_string(S, Line),
    (   Line == end_of_file
    ->  true
    ;   (   Line == ""
        ->  true
        ;   atom_string(Atom, Line),
            atom_json_term(Atom, [K, V], [value_string_as(atom)]),
            Fact =.. [Pred, K, V],
            assertz(Fact)
        ),
        load_pair_lines(S, Pred)
    ).

% ---------------------------------------------------------------------------
% Version comparison (delegated to frozen resolver:version_lt/2)
% ---------------------------------------------------------------------------
% Dotted numeric versions ("2.34", "2.2.5") -> v(A,B,C), padded with 0 so the
% 2- and 3-component glibc tags compare correctly.

ver_term(V, v(A, B, C)) :-
    (   atom(V) -> atom_string(V, S) ; V = S ),
    split_string(S, ".", "", Parts0),
    exclude(==(""), Parts0, Parts),
    nums_pad(Parts, A, B, C).

nums_pad(Parts, A, B, C) :-
    ( nth0(0, Parts, P0) -> to_num(P0, A) ; A = 0 ),
    ( nth0(1, Parts, P1) -> to_num(P1, B) ; B = 0 ),
    ( nth0(2, Parts, P2) -> to_num(P2, C) ; C = 0 ).

to_num(S, N) :- ( number(S) -> N = S ; number_string(N0, S) -> N = N0 ; N = 0 ).

ver_lt(A, B) :- ver_term(A, TA), ver_term(B, TB), resolver:version_lt(TA, TB).
ver_le(A, B) :- \+ ver_lt(B, A).

% max version of a non-empty list (highest wins).
max_ver([V|Vs], Max) :- foldl(max_ver_1, Vs, V, Max).
max_ver_1(V, Acc, Out) :- ( ver_lt(Acc, V) -> Out = V ; Out = Acc ).

% ---------------------------------------------------------------------------
% Store accessors
% ---------------------------------------------------------------------------

% intro(SoName|Sym) -- the symbol's introduced version (interval floor).
symprov_intro(SoName, Sym, Intro) :-
    symprov(Key, Val),
    split_key(Key, SoName, Sym),
    split_hash(Val, Intro, _To).

% req_sym(Binary, SoName, Sym, Ver) -- a versioned symbol the binary needs.
req_sym(Binary, SoName, Sym, Ver) :-
    symreq(Key, Val),
    split_key(Key, Binary, Sym),
    split_hash(Val, SoName, Ver).

split_key(Key, A, B) :-
    ( atom(Key) -> atom_string(Key, S) ; S = Key ),
    sub_string(S, Before, _, After, "|"),
    !,
    sub_string(S, 0, Before, _, AS),
    sub_string(S, _, After, 0, BS),
    atom_string(A, AS),
    atom_string(B, BS).

split_hash(Val, A, B) :-
    ( atom(Val) -> atom_string(Val, S) ; S = Val ),
    split_string(S, "#", "", [AS, BS | _]),
    atom_string(A, AS),
    atom_string(B, BS).

% ---------------------------------------------------------------------------
% Core ABI predicates
% ---------------------------------------------------------------------------

% provides_at(SoName, Sym, V): library version V exports Sym, i.e. it was
% introduced at or before V (to = inf within a soname).
provides_at(SoName, Sym, V) :-
    symprov_intro(SoName, Sym, Intro),
    ver_le(Intro, V).

% provides_at with a simulated removal: drop(DropSym, At) removes DropSym for
% every V >= At (a soname-era ABI break, for testing the max).
provides_at_drop(SoName, Sym, V, none) :- !,
    provides_at(SoName, Sym, V).
provides_at_drop(_SoName, Sym, V, drop(Sym, At)) :-
    ver_le(At, V), !,
    fail.
provides_at_drop(SoName, Sym, V, _Drop) :-
    provides_at(SoName, Sym, V).

% abi_min(Binary, SoName, Min): the verneed floor -- the highest required
% version among the binary's required symbols of that soname. Hard/derived.
abi_min(Binary, SoName, Min) :-
    findall(Ver, req_sym(Binary, SoName, _Sym, Ver), Vers),
    Vers \== [],
    max_ver(Vers, Min).

% missing_syms(Binary, SoName, V, Drop, Missing): required symbols of SoName
% NOT provided at library version V (containment failures = hard vetoes).
missing_syms(Binary, SoName, V, Drop, Missing) :-
    findall(Sym,
            ( req_sym(Binary, SoName, Sym, _),
              \+ provides_at_drop(SoName, Sym, V, Drop)
            ),
            Missing0),
    sort(Missing0, Missing).

% abi_compatible(Binary, SoName, V): every required symbol of SoName is
% provided at library version V (set containment).
abi_compatible(Binary, SoName, V) :-
    abi_compatible(Binary, SoName, V, none).

abi_compatible(Binary, SoName, V, Drop) :-
    missing_syms(Binary, SoName, V, Drop, []).

% soname_candidates(SoName, Descending): the soname's version axis = the
% distinct intro versions in the store, highest first.
soname_candidates(SoName, Desc) :-
    findall(Intro, symprov_intro(SoName, _Sym, Intro), Intros0),
    sort(Intros0, Uniq),
    predsort(cmp_ver_desc, Uniq, Desc).

cmp_ver_desc(Order, A, B) :-
    ( ver_lt(A, B) -> Order = (>)
    ; ver_lt(B, A) -> Order = (<)
    ; Order = (=)
    ).

% newest_providing(SoName, Sym, Candidates, Drop, Result): the newest V in
% Candidates (descending) that still exports Sym under Drop -> compatible(V);
% else no_candidate. With Drop = drop(Sym, At) this is the "effective max cap"
% a symbol removal imposes (the newest version just below the removal).
newest_providing(_SoName, _Sym, [], _Drop, no_candidate) :- !.
newest_providing(SoName, Sym, [V|Vs], Drop, Result) :-
    (   provides_at_drop(SoName, Sym, V, Drop)
    ->  Result = compatible(V)
    ;   newest_providing(SoName, Sym, Vs, Drop, Result)
    ).

% newest_abi_compatible(Binary, SoName, Candidates, Result):
% newest V in Candidates (descending) with abi_compatible -> compatible(V);
% else no_candidate.
newest_abi_compatible(Binary, SoName, Candidates, Result) :-
    newest_abi_compatible(Binary, SoName, Candidates, none, Result).

newest_abi_compatible(_Binary, _SoName, [], _Drop, no_candidate) :- !.
newest_abi_compatible(Binary, SoName, [V|Vs], Drop, Result) :-
    (   abi_compatible(Binary, SoName, V, Drop)
    ->  Result = compatible(V)
    ;   newest_abi_compatible(Binary, SoName, Vs, Drop, Result)
    ).

% abi_range(Binary, SoName, Candidates, Result): combine the derived MIN
% (verneed floor) with the newest compatible MAX. If min > max (a removed
% symbol squeezed the range) -> no_candidate.
abi_range(Binary, SoName, Candidates, Result) :-
    abi_range(Binary, SoName, Candidates, none, Result).

abi_range(Binary, SoName, Candidates, Drop, Result) :-
    (   abi_min(Binary, SoName, Min) -> true ; Min = none ),
    newest_abi_compatible(Binary, SoName, Candidates, Drop, Newest),
    (   Newest = compatible(Max)
    ->  (   Min == none
        ->  Result = range(none, Max)
        ;   ver_le(Min, Max)
        ->  Result = range(Min, Max)
        ;   Result = no_candidate(min_gt_max(Min, Max))
        )
    ;   Result = no_candidate(no_compatible_version)
    ).
