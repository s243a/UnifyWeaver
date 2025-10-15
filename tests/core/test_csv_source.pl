:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_csv_source.pl - Tests for CSV source plugin

:- module(test_csv_source, [
    test_csv_source/0,
    test_csv_with_headers/0,
    test_csv_manual_columns/0,
    test_tsv_support/0
]).

:- use_module('../../src/unifyweaver/sources/csv_source').
:- use_module(library(lists)).

%% Main test predicate
test_csv_source :-
    format('Testing CSV source plugin...~n', []),
    test_csv_with_headers,
    test_csv_manual_columns,
    test_tsv_support,
    format('✅ All CSV source tests passed~n', []).

%% Test CSV with header auto-detection
test_csv_with_headers :-
    format('  Testing CSV with headers...~n', []),
    
    % Create test CSV file
    TestFile = 'test_headers.csv',
    TestData = 'name,age,city\nalice,25,nyc\nbob,30,sf\ncharlie,35,la\n',
    write_test_file(TestFile, TestData),
    
    % Test compilation
    catch(
        (   compile_source(csv, users/3, [
                csv_file(TestFile), 
                has_header(true)
            ], [], Code),
            
            % Verify generated code contains expected elements
            sub_atom(Code, _, _, _, 'users()'),
            sub_atom(Code, _, _, _, 'awk'),
            sub_atom(Code, _, _, _, 'name, age, city'),
            
            format('    ✅ Header auto-detection works~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ),
    
    % Clean up
    delete_test_file(TestFile).

%% Test CSV with manual columns
test_csv_manual_columns :-
    format('  Testing CSV with manual columns...~n', []),
    
    % Create test CSV file without headers
    TestFile = 'test_manual.csv',
    TestData = 'alice,25,nyc\nbob,30,sf\ncharlie,35,la\n',
    write_test_file(TestFile, TestData),
    
    % Test compilation
    catch(
        (   compile_source(csv, people/3, [
                csv_file(TestFile),
                columns([name, age, city])
            ], [], Code),
            
            % Verify generated code
            sub_atom(Code, _, _, _, 'people()'),
            sub_atom(Code, _, _, _, 'awk'),
            sub_atom(Code, _, _, _, 'name, age, city'),
            
            format('    ✅ Manual columns work~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ),
    
    % Clean up
    delete_test_file(TestFile).

%% Test TSV support
test_tsv_support :-
    format('  Testing TSV support...~n', []),
    
    % Create test TSV file
    TestFile = 'test_data.tsv',
    TestData = 'name\tage\tcity\nalice\t25\tnyc\nbob\t30\tsf\n',
    write_test_file(TestFile, TestData),
    
    % Test compilation
    catch(
        (   compile_source(csv, tsv_data/3, [
                csv_file(TestFile),
                delimiter('\t'),
                has_header(true)
            ], [], Code),
            
            % Verify TSV-specific code
            sub_atom(Code, _, _, _, 'tsv_data()'),
            sub_atom(Code, _, _, _, '\\t'),  % Tab delimiter
            
            format('    ✅ TSV support works~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ),
    
    % Clean up
    delete_test_file(TestFile).

%% Helper predicates for test file management
write_test_file(File, Data) :-
    open(File, write, Stream),
    write(Stream, Data),
    close(Stream).

delete_test_file(File) :-
    (   exists_file(File)
    ->  delete_file(File)
    ;   true
    ).
