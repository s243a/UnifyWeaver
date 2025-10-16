:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_python_source.pl - Tests for Python embedded source plugin

:- module(test_python_source, [
    test_python_source/0,
    test_python_inline/0,
    test_sqlite_integration/0,
    test_python_file/0
]).

:- use_module('../../src/unifyweaver/sources/python_source').
:- use_module(library(lists)).

%% Main test predicate
test_python_source :-
    format('Testing Python source plugin...~n', []),
    test_python_inline,
    test_sqlite_integration,
    test_python_file,
    format('✅ All Python source tests passed~n', []).

%% Test inline Python code with heredoc pattern
test_python_inline :-
    format('  Testing Python inline code...~n', []),
    
    % Test basic inline Python
    catch(
        (   python_source:compile_source(hello/1, [
                python_inline('print("hello:world")')
            ], [], Code),
            
            % Check heredoc pattern is used
            sub_atom(Code, _, _, _, '/dev/fd/3'),
            sub_atom(Code, _, _, _, 'PYTHON'),
            sub_atom(Code, _, _, _, 'print("hello:world")'),
            sub_atom(Code, _, _, _, 'hello()'),
            
            format('    ✅ Heredoc pattern works~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ),
    
    % Test Python with stdin processing
    catch(
        (   python_source:compile_source(transform/2, [
                python_inline('
import sys
for line in sys.stdin:
    parts = line.strip().split(":")
    if len(parts) >= 2:
        print(f"{parts[0]}:{parts[1].upper()}")
'),
                input_mode(stdin)
            ], [], Code2),
            
            % Check stdin template is used
            sub_atom(Code2, _, _, _, 'sys.stdin'),
            sub_atom(Code2, _, _, _, 'transform()'),
            
            format('    ✅ Stdin processing works~n', [])
        ),
        Error2,
        (   format('    ❌ Error: ~w~n', [Error2]),
            fail
        )
    ).

%% Test SQLite integration
test_sqlite_integration :-
    format('  Testing SQLite integration...~n', []),
    
    % Test SQLite query compilation
    catch(
        (   python_source:compile_source(test_users/3, [
                sqlite_query('SELECT name, email, age FROM users WHERE active = 1'),
                database('test.db')
            ], [], Code),
            
            % Check SQLite-specific code generation
            sub_atom(Code, _, _, _, 'sqlite3'),
            sub_atom(Code, _, _, _, 'test.db'),
            sub_atom(Code, _, _, _, 'SELECT name, email, age FROM users'),
            sub_atom(Code, _, _, _, 'test_users()'),
            
            % Check error handling
            sub_atom(Code, _, _, _, 'SQLite error'),
            sub_atom(Code, _, _, _, 'sys.stderr'),
            
            format('    ✅ SQLite integration works~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ).

%% Test Python file integration
test_python_file :-
    format('  Testing Python file integration...~n', []),
    
    % Create test Python file
    TestFile = 'test_script.py',
    TestCode = 'import sys\nprint("file:loaded")\nfor arg in sys.argv[1:]:\n    print(f"arg:{arg}")\n',
    write_test_file(TestFile, TestCode),
    
    % Test compilation
    catch(
        (   python_source:compile_source(script_runner/2, [
                python_file(TestFile)
            ], [], Code),
            
            % Check file-based code generation
            sub_atom(Code, _, _, _, 'script_runner()'),
            sub_atom(Code, _, _, _, 'import sys'),
            sub_atom(Code, _, _, _, 'file:loaded'),
            
            format('    ✅ Python file integration works~n', [])
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
