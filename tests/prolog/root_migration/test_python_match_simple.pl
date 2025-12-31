:- encoding(utf8).
% Simple test for Python match predicate - test translation directly

:- use_module('src/unifyweaver/targets/python_target').

% Simple test: Use get_dict to extract from record, then match
filter_errors(Record) :-
    get_dict(message, Record, Line),
    match(Line, 'ERROR', python).

% Test with pattern matching
filter_timeout(Record) :-
    get_dict(message, Record, Line),
    match(Line, 'ERROR.*timeout', python),
    Line = Result.

% Compile and inspect
test_simple :-
    write('=== Test: Simple Python Match ==='), nl, nl,
    python_target:compile_predicate_to_python(filter_errors/1, [], Code),
    write(Code), nl.

test_pattern :-
    write('=== Test: Pattern Match ==='), nl, nl,
    python_target:compile_predicate_to_python(filter_timeout/1, [], Code),
    write(Code), nl.

run_all :-
    test_simple,
    nl, nl,
    test_pattern.

% Usage:
% ?- consult('test_python_match_simple.pl').
% ?- run_all.
