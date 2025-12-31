:- encoding(utf8).
% Test AWK match predicate with regex support

:- use_module('src/unifyweaver/targets/awk_target').

% Test data - log lines
log(error, 'ERROR: timeout in connection').
log(warning, 'WARNING: slow response').
log(error, 'ERROR: database connection failed').
log(info, 'INFO: request completed').

% Test 1: Boolean match with auto type (default ERE)
error_line(Line) :-
    log(error, Line),
    match(Line, 'ERROR').

% Test 2: Explicit ERE type
timeout_error(Line) :-
    log(error, Line),
    match(Line, 'ERROR.*timeout', ere).

% Test 3: Pattern with special characters
db_error(Line) :-
    log(error, Line),
    match(Line, 'database.*failed', ere).

% Test 4: Match with BRE type
simple_match(Line) :-
    log(_, Line),
    match(Line, 'request', bre).

% Test 5: Match with awk type
awk_match(Line) :-
    log(_, Line),
    match(Line, 'completed', awk).

% Compile tests
test_auto_match :-
    write('=== Test: Auto Match (default ERE) ==='), nl, nl,
    awk_target:compile_predicate_to_awk(error_line/1, [], AwkCode),
    write(AwkCode), nl, nl.

test_ere_match :-
    write('=== Test: Explicit ERE Match ==='), nl, nl,
    awk_target:compile_predicate_to_awk(timeout_error/1, [], AwkCode),
    write(AwkCode), nl, nl.

test_db_error :-
    write('=== Test: DB Error Match ==='), nl, nl,
    awk_target:compile_predicate_to_awk(db_error/1, [], AwkCode),
    write(AwkCode), nl, nl.

test_bre_match :-
    write('=== Test: BRE Match ==='), nl, nl,
    awk_target:compile_predicate_to_awk(simple_match/1, [], AwkCode),
    write(AwkCode), nl, nl.

test_awk_match :-
    write('=== Test: AWK Match ==='), nl, nl,
    awk_target:compile_predicate_to_awk(awk_match/1, [], AwkCode),
    write(AwkCode), nl, nl.

% Test unsupported regex type (should fail)
test_pcre_fail :-
    write('=== Test: PCRE Match (should fail) ==='), nl, nl,
    catch(
        (   log(_, Line),
            match(Line, 'test', pcre)
        ),
        Error,
        (   write('Caught error (expected): '), write(Error), nl
        )
    ).

run_all :-
    test_auto_match,
    test_ere_match,
    test_db_error,
    test_bre_match,
    test_awk_match,
    write('All match tests completed!'), nl.

% Usage:
% ?- consult('test_awk_match.pl').
% ?- run_all.
