% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% diff_runner.pl -- the PROLOG side of the differential harness.
%
% Protocol: reads argv-lines on stdin (one line per case, tokens separated by
% spaces; lines that carry no tokens are skipped), runs cli_args:parse_args/2,
% and prints one JSON object per line in exactly the shape diff_runner.mjs
% prints for the oracle:
%
%   {"ok":{"positional":[...],"flags":{...}}}
%   {"error":"<message>"}
%
% The JSON is hand-rolled (no library(http/json)) so the emitted shape is under
% direct control and the file stays dependency-free for the transpile steps.
%
%   swipl -q -g main -t halt examples/cli_args/diff_runner.pl < lines.txt > prolog.jsonl

:- module(diff_runner, [main/0]).

:- use_module(cli_args).

main :-
    read_line_to_string(user_input, Line),
    process_lines(Line).

process_lines(end_of_file) :- !.
process_lines(Line) :-
    split_string(Line, " ", "", Raw),
    exclude(==(""), Raw, Tokens),
    (   Tokens == []
    ->  true
    ;   parse_args(Tokens, Result),
        emit_result(Result),
        nl
    ),
    read_line_to_string(user_input, Next),
    process_lines(Next).

emit_result(ok(Positional, Flags)) :-
    write('{"ok":{"positional":'),
    emit_string_array(Positional),
    write(',"flags":'),
    emit_flags(Flags),
    write('}}').
emit_result(error(Message)) :-
    write('{"error":'),
    emit_json_string(Message),
    write('}').

emit_string_array(List) :-
    write('['),
    emit_string_items(List, first),
    write(']').

emit_string_items([], _).
emit_string_items([X|Xs], Position) :-
    (   Position == first
    ->  true
    ;   write(',')
    ),
    emit_json_string(X),
    emit_string_items(Xs, rest).

emit_flags(Flags) :-
    write('{'),
    emit_flag_items(Flags, first),
    write('}').

emit_flag_items([], _).
emit_flag_items([Key-Value|Rest], Position) :-
    (   Position == first
    ->  true
    ;   write(',')
    ),
    emit_json_string(Key),
    write(':'),
    emit_json_value(Value),
    emit_flag_items(Rest, rest).

emit_json_value(true) :- !, write(true).
emit_json_value(false) :- !, write(false).
emit_json_value(Value) :- emit_json_string(Value).

emit_json_string(Value) :-
    string_chars(Value, Chars),
    write('"'),
    emit_json_chars(Chars),
    write('"').

emit_json_chars([]).
emit_json_chars([C|Cs]) :-
    emit_json_char(C),
    emit_json_chars(Cs).

emit_json_char(C) :-
    char_code(C, Code),
    (   Code =:= 0'"
    ->  write('\\"')
    ;   Code =:= 0'\\
    ->  write('\\\\')
    ;   Code =:= 0'\n
    ->  write('\\n')
    ;   Code =:= 0'\r
    ->  write('\\r')
    ;   Code =:= 0'\t
    ->  write('\\t')
    ;   Code < 0x20
    ->  format("\\u~|~`0t~16r~4+", [Code])
    ;   write(C)
    ).
