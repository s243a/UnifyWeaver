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
:- use_module(library(filesex)).
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

    % Make file executable (ignore errors on Windows where chmod might not work)
    catch(
        process_create(path(chmod), ['+x', FilePath], [stderr(null)]),
        _,
        true  % Ignore chmod failures - bash will execute the file anyway
    ),

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
    % Create temporary file with .sh extension
    % NOTE: Can't use tmp_file/2 - SWI-Prolog auto-cleanup conflicts with external bash access
    % Create file with timestamp-based name instead
    get_time(TimeStamp),

    detect_execution_mode(ExecMode),
    build_temp_paths(ExecMode, TimeStamp, temp_paths(TmpFileWin, TmpFileBash)),

    format('DEBUG write_and_execute: Creating file: ~q~n', [TmpFileWin]),

    create_temp_script(ExecMode, TmpFileWin, TmpFileBash, BashCode),

    % Give Windows/filesystem a moment to sync
    sleep(0.1),

    format('DEBUG write_and_execute: File written, checking existence...~n', []),
    (   exists_file(TmpFileWin)
    ->  format('DEBUG write_and_execute: File EXISTS after write~n', [])
    ;   format('DEBUG write_and_execute: File DOES NOT EXIST after write!~n', [])
    ),

    % Test if bash can see and read the file before trying to execute
    format('DEBUG: Testing bash access to file...~n', []),
    (   nonvar(TmpFileBash),
        TmpFileBash \= none
    ->  BashTestPath = TmpFileBash
    ;   convert_to_cygwin_path(TmpFileWin, BashTestPath)
    ),
    format('DEBUG: Bash test path (atom): ~q~n', [BashTestPath]),

    % Test 1: Can bash cat the file?
    format('DEBUG: Running: bash -c "cat "$1"" -- bash_executor ~q~n', [BashTestPath]),
    catch(
        (setup_call_cleanup(
            process_create(path(bash), ['-c', 'cat "$1"', 'bash_executor', BashTestPath],
                          [stdout(pipe(CatOut)), stderr(pipe(CatErr)), process(CatPID)]),
            (read_string(CatOut, _, CatResult),
             read_string(CatErr, _, CatError),
             close(CatOut),
             close(CatErr),
             process_wait(CatPID, exit(CatExit))),
            true
         ),
         format('DEBUG: Cat exit code: ~w~n', [CatExit]),
         format('DEBUG: Cat stdout: ~q~n', [CatResult]),
         (CatError \= '' -> format('DEBUG: Cat stderr: ~q~n', [CatError]) ; true)),
        CatErr,
        format('DEBUG: Cat command error: ~w~n', [CatErr])
    ),

    % Execute and clean up (use special version for temp files)
    TempSpec = temp_paths(TmpFileWin, TmpFileBash),
    catch(
        execute_bash_tempfile(TempSpec, Input, Output),
        Error,
        (   delete_temp_file(TempSpec),
            throw(Error)
        )
    ),
    delete_temp_file(TempSpec).

%% execute_bash_tempfile(+TmpFile, +Input, -Output)
%  Execute a temp file, converting path for Cygwin if needed
execute_bash_tempfile(TempSpec, Input, Output) :-
    (   can_execute_natively
    ->  true
    ;   throw(error(platform_error,
                   context(execute_bash_tempfile/3,
                          'Cannot execute bash natively on this platform')))
    ),
    resolve_temp_spec(TempSpec, _TmpFileWin, BashPath),
    format('DEBUG: Bash execution path: ~q~n', [BashPath]),
    catch(
        process_create(path(chmod), ['+x', BashPath], [stderr(null)]),
        _,
        true
    ),
    setup_call_cleanup(
        process_create(path(bash), [BashPath],
                      [stdin(pipe(In)), stdout(pipe(Out)), stderr(std),
                       process(PID)]),
        (   (   Input \= ''
            ->  write(In, Input)
            ;   true
            ),
            close(In),
            read_string(Out, _, Output),
            close(Out),
            process_wait(PID, exit(ExitCode))
        ),
        true
    ),
    (   ExitCode = 0
    ->  true
    ;   format(atom(Msg), 'Bash script exited with code ~w', [ExitCode]),
        throw(error(execution_error(ExitCode),
                    context(execute_bash_tempfile/3, Msg)))
    ).

create_temp_script(powershell_cygwin, TmpFileWin, TmpFileBash, BashCode) :-
    normalize_path_atom(TmpFileWin, WinTarget),
    ensure_parent_directory(WinTarget),
    (   nonvar(TmpFileBash),
        TmpFileBash \= none
    ->  normalize_path_atom(TmpFileBash, BashTarget),
        write_script_via_cygwin(BashTarget, BashCode)
    ;   write_script_via_windows(WinTarget, BashCode)
    ),
    !.
create_temp_script(_ExecMode, TmpFileWin, _TmpFileBash, BashCode) :-
    normalize_path_atom(TmpFileWin, Target),
    ensure_parent_directory(Target),
    write_script_via_windows(Target, BashCode).

write_script_via_windows(Target, BashCode) :-
    open(Target, write, Stream, [encoding(utf8)]),
    write(Stream, BashCode),
    flush_output(Stream),
    close(Stream).

write_script_via_cygwin(BashTarget, BashCode) :-
    setup_call_cleanup(
        process_create(path(bash), ['-lc', 'cat > "$1"', 'bash_executor', BashTarget],
                      [stdin(pipe(In)), stdout(null), stderr(pipe(Err)), process(PID)]),
        (   write(In, BashCode),
            close(In),
            read_string(Err, _, ErrMsg),
            close(Err),
            process_wait(PID, exit(ExitCode))
        ),
        true
    ),
    (   ExitCode = 0
    ->  true
    ;   format(atom(Msg), 'Failed to write temp script via bash (exit ~w): ~s', [ExitCode, ErrMsg]),
        throw(error(execution_error(ExitCode), context(write_script_via_cygwin/2, Msg)))
    ).

delete_temp_file(temp_paths(TmpFileWin, _)) :-
    delete_temp_file_path(TmpFileWin).
delete_temp_file(Path) :-
    delete_temp_file_path(Path).

delete_temp_file_path(Path) :-
    normalize_path_atom(Path, PathAtom),
    catch(delete_file(PathAtom), _, true).

resolve_temp_spec(temp_paths(TmpFileWin, TmpFileBash), TmpFileAtom, BashPathAtom) :-
    normalize_path_atom(TmpFileWin, TmpFileAtom),
    (   nonvar(TmpFileBash),
        TmpFileBash \= none
    ->  normalize_path_atom(TmpFileBash, BashPathAtom)
    ;   convert_to_cygwin_path(TmpFileAtom, BashPathAtom)
    ).
resolve_temp_spec(Path, TmpFileAtom, BashPathAtom) :-
    normalize_path_atom(Path, TmpFileAtom),
    convert_to_cygwin_path(TmpFileAtom, BashPathAtom).

ensure_parent_directory(PathAtom) :-
    file_directory_name(PathAtom, DirAtom),
    (   DirAtom \== PathAtom,
        DirAtom \= '.',
        DirAtom \= ''
    ->  catch(make_directory_path(DirAtom), _, true)
    ;   true
    ).

normalize_path_atom(Path, Atom) :-
    (   atom(Path)
    ->  Atom = Path
    ;   atom_string(Atom, Path)
    ).

%% convert_to_cygwin_path(+WindowsPath, -CygwinPath)
%  Convert Windows path to Cygwin-compatible path using cygpath
cygwin_tmp_directory(WindowsPath) :-
    find_cygpath(CygpathExe),
    catch(
        (   setup_call_cleanup(
                process_create(CygpathExe, ['-w', '/tmp'],
                              [stdout(pipe(Out)), stderr(null), process(PID)]),
                (   read_string(Out, _, RawDir),
                    close(Out),
                    process_wait(PID, exit(ExitCode))
                ),
                true
            ),
            ExitCode = 0
        ),
        _,
        fail
    ),
    split_string(RawDir, "\n\r", " \t", [DirStr|_]),
    normalize_windows_path(DirStr, WindowsPath),
    !.

normalize_windows_path(Input, Output) :-
    split_string(Input, "\\", "", Parts),
    atomics_to_string(Parts, '/', Output).

convert_to_cygwin_path(WindowsPath, CygwinPath) :-
    % Convert to string and normalise separators
    (   atom(WindowsPath)
    ->  atom_string(WindowsPath, WinPathRaw)
    ;   WinPathRaw = WindowsPath
    ),
    normalize_windows_path(WinPathRaw, WinPathStr),

    (   convert_with_cygpath(WinPathStr, CygwinPath)
    ->  true
    ;   fallback_cygwin_path(WinPathStr, CygwinPath)
    ).

convert_with_cygpath(WinPathStr, CygwinPath) :-
    find_cygpath(CygpathExe),
    catch(
        (   setup_call_cleanup(
                process_create(CygpathExe, ['-u', WinPathStr],
                              [stdout(pipe(Out)), stderr(null), process(PID)]),
                (   read_string(Out, _, CygwinPathRaw),
                    close(Out),
                    process_wait(PID, exit(ExitCode))
                ),
                true
            ),
            ExitCode = 0
        ),
        _,
        fail
    ),
    split_string(CygwinPathRaw, "\n\r", " \t", [CygPathStr0|_]),
    CygPathStr0 \= '',
    atom_string(CygwinPath, CygPathStr0).

fallback_cygwin_path(WinPathStr, CygwinPath) :-
    (   sub_string(WinPathStr, 0, 1, _, "/")
    ->  atom_string(CygwinPath, WinPathStr)
    ;   sub_string(WinPathStr, 1, 1, _, ":")
    ->  sub_string(WinPathStr, 0, 1, _, DriveLetter),
        string_lower(DriveLetter, DriveLower),
        sub_string(WinPathStr, 2, _, 0, Rest0),
        (   sub_string(Rest0, 0, 1, _, "/")
        ->  sub_string(Rest0, 1, _, 0, Rest)
        ;   Rest = Rest0
        ),
        atomics_to_string(['/cygdrive/', DriveLower, '/', Rest], CygPathStr),
        atom_string(CygwinPath, CygPathStr)
    ;   atom_string(CygwinPath, WinPathStr)
    ).


get_temp_env_directory(TempDirStr) :-
    (   getenv('TEMP', TempRaw)
    ->  true
    ;   getenv('TMP', TempRaw)
    ->  true
    ;   TempRaw = '/tmp'
    ),
    (   atom(TempRaw)
    ->  atom_string(TempRaw, TempStr0)
    ;   TempStr0 = TempRaw
    ),
    normalize_windows_path(TempStr0, TempDirStr).

select_temp_paths(powershell_cygwin, TimeStamp, TmpFileWin, TmpFileBash) :-
    get_temp_env_directory(EnvDir),
    (   cygwin_tmp_directory(CygTmpDir)
    ->  WinDir = CygTmpDir,
        BashDir = '/tmp'
    ;   WinDir = EnvDir,
        BashDir = none
    ),
    build_temp_paths(WinDir, BashDir, TimeStamp, TmpFileWin, TmpFileBash).
select_temp_paths(_, TimeStamp, TmpFileWin, TmpFileBash) :-
    get_temp_env_directory(EnvDir),
    build_temp_paths(EnvDir, none, TimeStamp, TmpFileWin, TmpFileBash).

build_temp_paths(ExecMode, TimeStamp, temp_paths(TmpFileWin, TmpFileBash)) :-
    select_temp_paths(ExecMode, TimeStamp, TmpFileWin, TmpFileBash).

build_temp_paths(WinDir, BashDir, TimeStamp, TmpFileWin, TmpFileBash) :-
    format(atom(FileName), 'plbash_~w.sh', [TimeStamp]),
    join_dir_file(WinDir, FileName, WinPathStr),
    atom_string(TmpFileWin, WinPathStr),
    (   BashDir = none
    ->  TmpFileBash = none
    ;   join_dir_file(BashDir, FileName, BashPathStr),
        atom_string(TmpFileBash, BashPathStr)
    ).

join_dir_file(Dir, File, PathStr) :-
    (   atom(Dir)
    ->  atom_string(Dir, DirStr0)
    ;   DirStr0 = Dir
    ),
    strip_trailing_slash(DirStr0, DirStr),
    (   DirStr = ''
    ->  PathStr = File
    ;   atomics_to_string([DirStr, '/', File], PathStr)
    ).

strip_trailing_slash(Source, Result) :-
    (   sub_string(Source, _, 1, 0, "/")
    ->  sub_string(Source, 0, _, 1, Trimmed),
        strip_trailing_slash(Trimmed, Result)
    ;   Result = Source
    ).

%% find_cygpath(-CygpathExe)
%  Find cygpath executable, checking standard Cygwin locations
find_cygpath(CygpathExe) :-
    % Try path first
    catch(process_which(path(cygpath), CygpathExe), _, fail),
    !.
find_cygpath('C:/cygwin64/bin/cygpath.exe') :-
    exists_file('C:/cygwin64/bin/cygpath.exe'),
    !.
find_cygpath('C:/cygwin/bin/cygpath.exe') :-
    exists_file('C:/cygwin/bin/cygpath.exe'),
    !.
find_cygpath('/usr/bin/cygpath') :-
    exists_file('/usr/bin/cygpath'),
    !.
% If not found, fail and use fallback
find_cygpath(_) :- fail.

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
