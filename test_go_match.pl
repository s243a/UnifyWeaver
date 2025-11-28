:- encoding(utf8).
% Test Go target match predicate support

:- use_module('src/unifyweaver/targets/go_target').

% Test data - log lines
log('ERROR: timeout occurred').
log('WARNING: slow response').
log('INFO: operation successful').
log('ERROR: connection failed').

% Test 1: Boolean match - filter error logs
error_log(Line) :-
    log(Line),
    match(Line, 'ERROR').

% Test 2: Boolean match with pattern
timeout_error(Line) :-
    log(Line),
    match(Line, 'ERROR.*timeout', auto).

test_boolean_match :-
    write('=== Test: Boolean Match - error_log ==='), nl,
    go_target:compile_predicate_to_go(error_log/1, [], Code),
    write(Code), nl.

test_pattern_match :-
    write('=== Test: Pattern Match - timeout_error ==='), nl,
    go_target:compile_predicate_to_go(timeout_error/1, [], Code),
    write(Code), nl.

test_write_error_log :-
    write('=== Test: Write error_log to file ==='), nl,
    go_target:compile_predicate_to_go(error_log/1, [], Code),
    go_target:write_go_program(Code, 'error_log.go').

run_all_tests :-
    test_boolean_match,
    test_pattern_match,
    test_write_error_log,
    write('All match tests completed!'), nl.

% Usage:
% ?- consult('test_go_match.pl').
% ?- run_all_tests.
