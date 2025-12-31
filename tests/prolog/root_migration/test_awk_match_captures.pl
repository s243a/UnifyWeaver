:- encoding(utf8).
% Test AWK match predicate with capture groups

:- use_module('src/unifyweaver/targets/awk_target').

% Test data - log lines
log('2025-01-15 10:30:45 ERROR: timeout').
log('2025-01-15 10:31:22 WARNING: slow response').
log('2025-01-15 10:32:10 ERROR: connection failed').

% Test 1: Parse timestamp and level from log line
parse_log(Line, Time, Level) :-
    log(Line),
    match(Line, '([0-9-]+ [0-9:]+) ([A-Z]+)', ere, [Time, Level]).

% Test 2: Extract just error logs with timestamp
parse_error(Line, Time) :-
    log(Line),
    match(Line, '([0-9-]+ [0-9:]+) ERROR', ere, [Time]).

% Compile tests
test_parse_log :-
    write('=== Test: Parse Log with Captures ==='), nl, nl,
    awk_target:compile_predicate_to_awk(parse_log/3, [], AwkCode),
    write(AwkCode), nl, nl.

test_parse_error :-
    write('=== Test: Parse Error with Single Capture ==='), nl, nl,
    awk_target:compile_predicate_to_awk(parse_error/2, [], AwkCode),
    write(AwkCode), nl, nl.

run_all :-
    test_parse_log,
    test_parse_error,
    write('All capture group tests completed!'), nl.

% Usage:
% ?- consult('test_awk_match_captures.pl').
% ?- run_all.
