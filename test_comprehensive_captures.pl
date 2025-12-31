:- encoding(utf8).
% Comprehensive capture group tests

:- use_module(library(filesex)).
:- use_module('src/unifyweaver/targets/go_target').

ensure_output_dir :- make_directory_path('output_test').

% Test 1: Single capture group
extract_time(Line, Time) :-
    match(Line, '([0-9-]+ [0-9:]+)', auto, [Time]).

% Test 2: Two capture groups (already tested)
extract_time_level(Line, Time, Level) :-
    match(Line, '([0-9-]+ [0-9:]+) ([A-Z]+)', auto, [Time, Level]).

% Test 3: Three capture groups
extract_date_time_level(Line, Date, Time, Level) :-
    match(Line, '([0-9-]+) ([0-9:]+) ([A-Z]+)', auto, [Date, Time, Level]).

% Test 4: Capture after fixed text
extract_error_msg(Line, Msg) :-
    match(Line, 'ERROR (.+)', auto, [Msg]).

% Compile tests
test_single :-
    write('=== Test 1: Single Capture Group ==='), nl,
    ensure_output_dir,
    go_target:compile_predicate_to_go(extract_time/2, [], Code1),
    go_target:write_go_program(Code1, 'output_test/extract_time.go'),
    write('output_test/extract_time.go written'), nl, nl.

test_two :-
    write('=== Test 2: Two Capture Groups ==='), nl,
    ensure_output_dir,
    go_target:compile_predicate_to_go(extract_time_level/3, [], Code2),
    go_target:write_go_program(Code2, 'output_test/extract_time_level.go'),
    write('output_test/extract_time_level.go written'), nl, nl.

test_three :-
    write('=== Test 3: Three Capture Groups ==='), nl,
    ensure_output_dir,
    go_target:compile_predicate_to_go(extract_date_time_level/4, [], Code3),
    go_target:write_go_program(Code3, 'output_test/extract_date_time_level.go'),
    write('output_test/extract_date_time_level.go written'), nl, nl.

test_partial :-
    write('=== Test 4: Partial Match (ERROR only) ==='), nl,
    ensure_output_dir,
    go_target:compile_predicate_to_go(extract_error_msg/2, [], Code4),
    go_target:write_go_program(Code4, 'output_test/extract_error_msg.go'),
    write('output_test/extract_error_msg.go written'), nl, nl.

run_all :-
    test_single,
    test_two,
    test_three,
    test_partial,
    write('All programs generated successfully!'), nl.

% Usage:
% ?- consult('test_comprehensive_captures.pl').
% ?- run_all.
