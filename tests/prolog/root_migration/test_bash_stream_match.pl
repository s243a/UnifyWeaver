:- encoding(utf8).
% Test bash match predicate with stream compiler

:- use_module('src/unifyweaver/core/stream_compiler').

% Test data - simple log lines
log('ERROR: timeout occurred').
log('WARNING: slow response').
log('INFO: operation successful').
log('ERROR: connection failed').

% Test 1: Boolean match - filter ERROR lines
error_lines(Line) :-
    log(Line),
    match(Line, 'ERROR').

% Test 2: Match with explicit type
warning_lines(Line) :-
    log(Line),
    match(Line, 'WARNING', auto).

% Test 3: Match at start of line
starts_with_error(Line) :-
    log(Line),
    match(Line, '^ERROR').

% Compile tests
test_error_match :-
    write('=== Test: Boolean Match for ERROR ==='), nl, nl,
    compile_predicate(error_lines/1, [], BashCode),
    write(BashCode), nl.

test_warning_match :-
    write('=== Test: Match with Explicit Type ==='), nl, nl,
    compile_predicate(warning_lines/1, [], BashCode),
    write(BashCode), nl.

test_anchored_match :-
    write('=== Test: Anchored Pattern Match ==='), nl, nl,
    compile_predicate(starts_with_error/1, [], BashCode),
    write(BashCode), nl.

run_all :-
    test_error_match,
    nl, nl,
    test_warning_match,
    nl, nl,
    test_anchored_match,
    write('All bash stream match tests completed!'), nl.

% Usage:
% ?- consult('test_bash_stream_match.pl').
% ?- run_all.
