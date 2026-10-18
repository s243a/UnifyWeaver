:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% swi_oracle.pl -- print the SWI grounded answer for one named smoke
% case. Uses the frozen resolver + test_resolver corpus, not the C
% wrappers. Output shape matches driver.c:
%   STATUS ok
%   TERM <write_canonical>
% or
%   STATUS fail
%
%   swipl -q -g main -t halt examples/pkg_resolver/c/swi_oracle.pl -- CASE_ID

:- use_module('../resolver').
:- use_module('../test_resolver', [scenario_catalog/2, corpus_case/4]).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [Id0|_]
    ->  true
    ;   format(user_error, "swi_oracle.pl: missing CASE_ID~n", []),
        halt(2)
    ),
    (   atom(Id0)
    ->  Id = Id0
    ;   atom_string(Id, Id0)
    ),
    (   corpus_case(Id, CatName, resolve, Args)
    ->  true
    ;   format(user_error, "swi_oracle.pl: unknown resolve case ~q~n", [Id]),
        halt(2)
    ),
    scenario_catalog(CatName, Cat),
    format("CASE ~w~n", [Id]),
    (   resolve(Cat, Args, Sel)
    ->  format("STATUS ok~n", []),
        format("TERM ", []),
        write_canonical(Sel),
        nl
    ;   format("STATUS fail~n", [])
    ).
