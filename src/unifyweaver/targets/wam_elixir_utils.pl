:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_elixir_utils.pl - Shared utilities for WAM-to-Elixir transpilation

:- module(wam_elixir_utils, [
    reg_id/2,       % +Reg, -Id
    clean_comma/2,  % +String, -CleanString
    is_label_part/1 % +String
]).

%% is_label_part(+String)
% True if the string is a WAM label (ends in ':' and is not a comment).
is_label_part(Str) :-
    sub_string(Str, _, 1, 0, ":"),
    \+ sub_string(Str, 0, 1, _, "%").

%% reg_id(+Reg, -Id)
% Maps string/atom WAM register names to integer IDs for Elixir.
% Y-registers are offset by 100 to avoid collision with X/A registers (1-99).
reg_id(Reg, Id) :-
    (   sub_atom(Reg, 0, 1, _, 'A') -> sub_atom(Reg, 1, _, 0, Num), atom_number(Num, Id)
    ;   sub_atom(Reg, 0, 1, _, 'X') -> sub_atom(Reg, 1, _, 0, Num), atom_number(Num, Id)
    ;   sub_atom(Reg, 0, 1, _, 'Y') -> sub_atom(Reg, 1, _, 0, Num), atom_number(Num, N), Id is N + 100
    ;   Id = Reg
    ).

%% clean_comma(+String, -CleanString)
clean_comma(S, Clean) :-
    (   sub_string(S, _, 1, 0, ",")
    ->  sub_string(S, 0, _, 1, Clean)
    ;   Clean = S
    ).
