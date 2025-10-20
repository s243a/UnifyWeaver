:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% bash_executor.pl - Native bash script execution for Linux/WSL/Docker
% Provides direct bash execution for platforms that support it natively

:- module(bash_executor, [
    execute_bash/2,              % +BashCode, -Output
    execute_bash/3,              % +BashCode, +Input, -Output
    execute_bash_file/2,         % +FilePath, -Output
    execute_bash_file/3,         % +FilePath, +Input, -Output
    can_execute_natively/0,
    write_and_execute_bash/2,    % +BashCode, -Output
    write_and_execute_bash/3     % +BashCode, +Input, -Output
]).

:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('platform_detection').

%% ============================================
%% CAPABILITY CHECK
%% ============================================

%% can_execute_natively/0
%  True if we can execute bash scripts natively
can_execute_natively :-
    can_execute_bash_directly.

%% ============================================
%% BASH EXECUTION
%% ============================================

%% execute_bash(+BashCode, -Output)
%  Execute bash code directly and capture output
%  BashCode is a string/atom containing bash code
%  Output is the stdout as a string
execute_bash(BashCode, Output) :-
    execute_bash(BashCode, '', Output).

%% execute_bash(+BashCode, +Input, -Output)
%  Execute bash code with stdin input
%  BashCode is a string/atom containing bash code
%  Input is stdin as a string
%  Output is the stdout as a string
execute_bash(BashCode, Input, Output) :-
    % Ensure we can execute natively
    (   can_execute_natively
    ->  true
    ;   throw(error(platform_error,
                   context(execute_bash/3,
                          'Cannot execute bash natively on this platform')))
    ),

    % Execute bash with code
    setup_call_cleanup(
        process_create(path(bash), ['-c', BashCode],
                      [stdin(pipe(In)), stdout(pipe(Out)), stderr(std),
                       process(PID)]),
        (   % Write input to stdin
            (   Input \= ''
            ->  write(In, Input)
            ;   true
            ),
            close(In),
            % Read output
            read_string(Out, _, Output),
            close(Out),
            % Wait for process to finish
            process_wait(PID, exit(ExitCode))
        ),
        true
    ),

    % Check exit code
    (   ExitCode = 0
    ->  true
    ;   format(atom(Msg), 'Bash script exited with code ~w', [ExitCode]),
        throw(error(execution_error(ExitCode), context(execute_bash/3, Msg)))
    ).

%% execute_bash_file(+FilePath, -Output)
%  Execute a bash script file and capture output
execute_bash_file(FilePath, Output) :-
    execute_bash_file(FilePath, '', Output).

%% execute_bash_file(+FilePath, +Input, -Output)
%  Execute a bash script file with stdin input
execute_bash_file(FilePath, Input, Output) :-
    % Ensure we can execute natively
    (   can_execute_natively
    ->  true
    ;   throw(error(platform_error,
                   context(execute_bash_file/3,
                          'Cannot execute bash natively on this platform')))
    ),

    % Check file exists
    (   exists_file(FilePath)
    ->  true
    ;   format(atom(Msg), 'Bash script file not found: ~w', [FilePath]),
        throw(error(existence_error(file, FilePath), context(execute_bash_file/3, Msg)))
    ),

    % Make file executable
    process_create(path(chmod), ['+x', FilePath], []),

    % Execute file
    setup_call_cleanup(
        process_create(path(bash), [FilePath],
                      [stdin(pipe(In)), stdout(pipe(Out)), stderr(std),
                       process(PID)]),
        (   % Write input to stdin
            (   Input \= ''
            ->  write(In, Input)
            ;   true
            ),
            close(In),
            % Read output
            read_string(Out, _, Output),
            close(Out),
            % Wait for process to finish
            process_wait(PID, exit(ExitCode))
        ),
        true
    ),

    % Check exit code
    (   ExitCode = 0
    ->  true
    ;   format(atom(Msg), 'Bash script exited with code ~w', [ExitCode]),
        throw(error(execution_error(ExitCode), context(execute_bash_file/3, Msg)))
    ).

%% ============================================
%% CONVENIENCE PREDICATES
%% ============================================

%% write_and_execute_bash(+BashCode, -Output)
%  Write bash code to a temporary file and execute it
%  Useful for longer scripts
write_and_execute_bash(BashCode, Output) :-
    write_and_execute_bash(BashCode, '', Output).

%% write_and_execute_bash(+BashCode, +Input, -Output)
%  Write bash code to a temporary file with input and execute it
write_and_execute_bash(BashCode, Input, Output) :-
    % Create temporary file
    tmp_file_stream(text, TmpFile, Stream),
    write(Stream, BashCode),
    close(Stream),

    % Execute and clean up
    catch(
        execute_bash_file(TmpFile, Input, Output),
        Error,
        (   delete_file(TmpFile),
            throw(Error)
        )
    ),
    delete_file(TmpFile).

%% ============================================
%% TESTING
%% ============================================

test_bash_executor :-
    format('~n=== Bash Executor Test ===~n', []),

    % Test 1: Simple command
    format('~nTest 1: Simple echo command~n', []),
    execute_bash('echo "Hello from bash!"', Output1),
    format('  Output: ~w', [Output1]),

    % Test 2: Command with input
    format('~nTest 2: Command with stdin input~n', []),
    execute_bash('cat', 'test input', Output2),
    format('  Output: ~w', [Output2]),

    % Test 3: Pipeline
    format('~nTest 3: Bash pipeline~n', []),
    execute_bash('echo "line1\nline2\nline3" | grep "line2"', Output3),
    format('  Output: ~w', [Output3]),

    % Test 4: Multiple commands
    format('~nTest 4: Multiple commands~n', []),
    execute_bash('X=42; echo "The answer is $X"', Output4),
    format('  Output: ~w', [Output4]),

    % Test 5: File execution
    format('~nTest 5: Execute from file~n', []),
    write_and_execute_bash('#!/bin/bash\necho "Executed from file"', Output5),
    format('  Output: ~w', [Output5]),

    format('~n=== All Tests Passed ===~n', []).
