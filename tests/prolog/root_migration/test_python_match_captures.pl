:- encoding(utf8).
% Test Python match predicate with capture groups

:- use_module('src/unifyweaver/targets/python_target').

% Test 1: Extract timestamp and level from log line
parse_log(Record, Time, Level) :-
    get_dict(line, Record, Line),
    match(Line, '([0-9-]+ [0-9:]+) ([A-Z]+)', python, [Time, Level]),
    Record = _{line: Line, time: Time, level: Level}.

% Test 2: Extract just the timestamp
parse_timestamp(Record, Time) :-
    get_dict(line, Record, Line),
    match(Line, '([0-9-]+ [0-9:]+)', python, [Time]),
    Record = _{line: Line, timestamp: Time}.

% Test 3: Extract IP address from log
parse_ip(Record, IP) :-
    get_dict(line, Record, Line),
    match(Line, '([0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+)', python, [IP]),
    Record = _{line: Line, ip: IP}.

% Compile tests
test_parse_log :-
    write('=== Test: Parse Log with Two Captures ==='), nl, nl,
    python_target:compile_predicate_to_python(parse_log/3, [], Code),
    write(Code), nl.

test_parse_timestamp :-
    write('=== Test: Parse Timestamp with Single Capture ==='), nl, nl,
    python_target:compile_predicate_to_python(parse_timestamp/2, [], Code),
    write(Code), nl.

test_parse_ip :-
    write('=== Test: Parse IP with Single Capture ==='), nl, nl,
    python_target:compile_predicate_to_python(parse_ip/2, [], Code),
    write(Code), nl.

run_all :-
    test_parse_log,
    nl, nl,
    test_parse_timestamp,
    nl, nl,
    test_parse_ip,
    write('All Python capture group tests completed!'), nl.

% Usage:
% ?- consult('test_python_match_captures.pl').
% ?- run_all.
