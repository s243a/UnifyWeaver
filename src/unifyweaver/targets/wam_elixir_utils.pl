:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_elixir_utils.pl - Shared utilities for WAM-to-Elixir transpilation

:- module(wam_elixir_utils, [
    reg_id/2,       % +Reg, -Id
    clean_comma/2,  % +String, -CleanString
    is_label_part/1, % +String
    camel_case/2,   % +String, -CamelString
    parse_arity/2   % +Functor, -Arity
]).

%% is_label_part(+String)
% True if the string is a WAM label (ends in ':' and is not a comment).
is_label_part(Str) :-
    sub_string(Str, _, 1, 0, ":"),
    \+ sub_string(Str, 0, 1, _, "%").

%% camel_case(+String, -CamelString)
%  Converts snake_case to CamelCase for Elixir module names.
camel_case(Str, Camel) :-
    atom_string(Str, S),
    split_string(S, "_/", "_/", Parts),
    include([P]>>(P \= "" , P \= "_", P \= "/"), Parts, FinalParts),
    maplist(capitalize_string, FinalParts, Caps),
    atomic_list_concat(Caps, Camel).

capitalize_string("", "") :- !.
capitalize_string(Str, Cap) :-
    string_chars(Str, [First|Rest]),
    upcase_atom(First, UpFirst),
    string_chars(Cap, [UpFirst|Rest]).

%% reg_id(+Reg, -Id)
% Maps string/atom WAM register names to integer IDs for Elixir.
% Y-registers are offset by 100 to avoid collision with X/A registers (1-99).
reg_id(Reg, Id) :-
    (atom(Reg) -> RegAtom = Reg ; atom_string(RegAtom, Reg)),
    (   sub_atom(RegAtom, 0, 1, _, 'A') -> sub_atom(RegAtom, 1, _, 0, Num), atom_number(Num, Id)
    ;   sub_atom(RegAtom, 0, 1, _, 'X') -> sub_atom(RegAtom, 1, _, 0, Num), atom_number(Num, Id)
    ;   sub_atom(RegAtom, 0, 1, _, 'Y') -> sub_atom(RegAtom, 1, _, 0, Num), atom_number(Num, N), Id is N + 100
    ;   Id = Reg
    ).

%% clean_comma(+String, -CleanString)
clean_comma(S, Clean) :-
    (   sub_string(S, _, 1, 0, ",")
    ->  sub_string(S, 0, _, 1, Clean)
    ;   Clean = S
    ).

%% parse_arity(+Functor, -Arity)
%  Extracts the integer arity from a "name/arity" functor string.
%  Falls back to 0 for malformed or arity-less functors. Matches the
%  semantics of the generated Elixir parse_functor_arity/1 exactly so
%  codegen-time pre-computation gives the same result as runtime parse.
parse_arity(Functor, Arity) :-
    (   split_string(Functor, "/", "", [_, ArityStr]),
        catch(number_string(Arity, ArityStr), _, false)
    ->  true
    ;   Arity = 0
    ).
