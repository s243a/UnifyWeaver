:- encoding(utf8).
% Regression tests for Python target
% Tests basic functionality to ensure match predicate addition didn't break existing features

:- use_module('src/unifyweaver/targets/python_target').

% Test 1: Basic filtering without match
filter_info(Record) :-
    get_dict(level, Record, Level),
    Level = info.

% Test 2: Dict construction
make_record(Line, Record) :-
    get_dict(message, Record, Line),
    Record = _{message: Line, processed: true}.

% Test 3: Match boolean (new feature)
filter_errors(Record) :-
    get_dict(message, Record, Line),
    match(Line, 'ERROR', python).

% Test 4: Match with captures (new feature)
parse_log(Record, Time, Level) :-
    get_dict(line, Record, Line),
    match(Line, '([0-9-]+ [0-9:]+) ([A-Z]+)', python, [Time, Level]),
    Record = _{line: Line, time: Time, level: Level}.

% Run tests
test_basic_filter :-
    write('=== Test 1: Basic Filtering (no match) ==='), nl, nl,
    python_target:compile_predicate_to_python(filter_info/1, [], Code),
    write(Code), nl, nl.

test_dict_construction :-
    write('=== Test 2: Dict Construction ==='), nl, nl,
    python_target:compile_predicate_to_python(make_record/2, [], Code),
    write(Code), nl, nl.

test_match_boolean :-
    write('=== Test 3: Match Boolean (NEW) ==='), nl, nl,
    python_target:compile_predicate_to_python(filter_errors/1, [], Code),
    write(Code), nl, nl.

test_match_captures :-
    write('=== Test 4: Match with Captures (NEW) ==='), nl, nl,
    python_target:compile_predicate_to_python(parse_log/3, [], Code),
    write(Code), nl, nl.

run_all :-
    write('======================================'), nl,
    write('PYTHON TARGET REGRESSION TESTS'), nl,
    write('======================================'), nl, nl,
    test_basic_filter,
    test_dict_construction,
    test_match_boolean,
    test_match_captures,
    write('======================================'), nl,
    write('All Python regression tests completed!'), nl,
    write('======================================'), nl.

% Usage:
% ?- consult('test_python_regression.pl').
% ?- run_all.
