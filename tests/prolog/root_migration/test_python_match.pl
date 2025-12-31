:- encoding(utf8).
% Test Python match predicate with regex support

:- use_module('src/unifyweaver/targets/python_target').

% Test data - log lines
log(error, 'ERROR: timeout in connection').
log(warning, 'WARNING: slow response').
log(error, 'ERROR: database connection failed').
log(info, 'INFO: request completed').

% Test 1: Boolean match with auto type (default Python regex)
error_line(Line) :-
    log(error, Line),
    match(Line, 'ERROR').

% Test 2: Explicit Python regex type
timeout_error(Line) :-
    log(error, Line),
    match(Line, 'ERROR.*timeout', python).

% Test 3: Pattern with special characters
db_error(Line) :-
    log(error, Line),
    match(Line, 'database.*failed', python).

% Test 4: Match with PCRE type
pcre_match(Line) :-
    log(_, Line),
    match(Line, 'completed', pcre).

% Compile tests
test_auto_match :-
    write('=== Test: Auto Match (default Python regex) ==='), nl, nl,
    python_target:compile_predicate_to_python(error_line/1, [], PythonCode),
    write(PythonCode), nl, nl.

test_python_match :-
    write('=== Test: Explicit Python Match ==='), nl, nl,
    python_target:compile_predicate_to_python(timeout_error/1, [], PythonCode),
    write(PythonCode), nl, nl.

test_db_error :-
    write('=== Test: DB Error Match ==='), nl, nl,
    python_target:compile_predicate_to_python(db_error/1, [], PythonCode),
    write(PythonCode), nl, nl.

test_pcre_match :-
    write('=== Test: PCRE Match ==='), nl, nl,
    python_target:compile_predicate_to_python(pcre_match/1, [], PythonCode),
    write(PythonCode), nl, nl.

run_all :-
    test_auto_match,
    test_python_match,
    test_db_error,
    test_pcre_match,
    write('All Python match tests completed!'), nl.

% Usage:
% ?- consult('test_python_match.pl').
% ?- run_all.
