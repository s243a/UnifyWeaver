:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_dynamic_sources.pl - Tests for dynamic source system

:- module(test_dynamic_sources, [
    test_dynamic_sources/0
]).

:- use_module('dynamic_source_compiler').
:- use_module('../sources/awk_source').

test_dynamic_sources :-
    writeln('=== Testing Dynamic Sources ==='),

    % Test 1: AWK plugin is registered
    write('Test 1 - AWK plugin registered: '),
    list_source_types(Types),
    (   member(awk, Types)
    ->  writeln('PASS')
    ;   writeln('FAIL'), fail
    ),

    % Test 2: Register AWK source predicate
    write('Test 2 - Register AWK source predicate: '),
    register_dynamic_source(user_data/2,
        type(awk, [awk_command('{print $1":"$2}'), input_file('/etc/passwd')]),
        []),
    (   is_dynamic_source(user_data/2)
    ->  writeln('PASS')
    ;   writeln('FAIL'), fail
    ),

    % Test 3: Compile AWK source
    write('Test 3 - Compile AWK source: '),
    compile_dynamic_source(user_data/2, [], BashCode),
    (   sub_string(BashCode, _, _, _, 'user_data()')
    ->  writeln('PASS')
    ;   writeln('FAIL'), fail
    ),

    % Test 4: Generated code contains AWK command
    write('Test 4 - Generated code contains AWK: '),
    (   sub_string(BashCode, _, _, _, 'awk')
    ->  writeln('PASS')
    ;   writeln('FAIL'), fail
    ),

    % Test 5: Validate config with awk_command
    write('Test 5 - Validate config: '),
    (   validate_config([awk_command('test')])
    ->  writeln('PASS')
    ;   writeln('FAIL'), fail
    ),

    % Clean up
    retractall(dynamic_source_def(user_data/2, _, _)),

    writeln('=== Dynamic Sources Tests Complete ===').
