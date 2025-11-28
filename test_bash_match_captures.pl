:- encoding(utf8).
% Test bash match predicate with capture groups using BASH_REMATCH

:- use_module('src/unifyweaver/core/stream_compiler').

% Test data - log lines with timestamps and levels
log_line('2025-01-15 10:30:45 ERROR timeout occurred').
log_line('2025-01-15 10:31:12 WARNING slow response').
log_line('2025-01-15 10:32:01 INFO operation successful').
log_line('2025-01-15 10:33:22 ERROR connection failed').

% Test 1: Extract timestamp and level (two capture groups)
parse_log(Line, Time, Level) :-
    log_line(Line),
    match(Line, '([0-9-]+ [0-9:]+) ([A-Z]+)', auto, [Time, Level]).

% Test 2: Extract just the timestamp (single capture group)
parse_timestamp(Line, Time) :-
    log_line(Line),
    match(Line, '([0-9-]+ [0-9:]+)', auto, [Time]).

% Test 3: Extract error message (capture after fixed text)
parse_error_msg(Line, Msg) :-
    log_line(Line),
    match(Line, 'ERROR (.+)', auto, [Msg]).

% Compile tests
test_two_captures :-
    write('=== Test 1: Two Capture Groups (Time + Level) ==='), nl, nl,
    compile_predicate(parse_log/3, [], Code),
    write(Code), nl.

test_single_capture :-
    write('=== Test 2: Single Capture Group (Timestamp) ==='), nl, nl,
    compile_predicate(parse_timestamp/2, [], Code),
    write(Code), nl.

test_message_capture :-
    write('=== Test 3: Capture After Fixed Text ==='), nl, nl,
    compile_predicate(parse_error_msg/2, [], Code),
    write(Code), nl.

run_all :-
    write('======================================'), nl,
    write('BASH MATCH CAPTURE GROUP TESTS'), nl,
    write('======================================'), nl, nl,
    test_two_captures,
    nl, nl,
    test_single_capture,
    nl, nl,
    test_message_capture,
    write('======================================'), nl,
    write('All bash capture tests completed!'), nl,
    write('======================================'), nl.

% Usage:
% ?- consult('test_bash_match_captures.pl').
% ?- run_all.
