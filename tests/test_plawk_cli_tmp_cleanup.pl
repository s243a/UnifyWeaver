:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% The plawk CLI leaves nothing in the temp directory.
%
% Every `plawk build` / `plawk run` creates $TMP/plawk_cli_<pid> (the intermediate
% plawk_build.ll, the `run` binary, the LMDB schema probe). It was never removed:
% a successful build deleted the .ll but left the empty directory, and every early
% exit (parse error, compile decline, clang failure) left the .ll too. A few days
% of test sweeps accumulated 42,107 of them (~6 GB). The directory is now removed
% at halt by whoever created it, on every exit path.
%
% Each case runs the CLI with TMP (which sets SWI-Prolog's tmp_dir) pointing at a
% fresh directory, and checks that the directory is empty afterwards.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1, delete_directory_and_contents/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_cli_tmp_cleanup).

test(successful_build_leaves_nothing, [condition(clang_available)]) :-
    cli_leaves_nothing(build, "{ print $1 }\n", 0),
    !.

test(parse_error_leaves_nothing) :-
    cli_leaves_nothing(build, "{ print $1 \n", 2),
    !.

test(compile_decline_leaves_nothing) :-
    cli_leaves_nothing(build, "BEGIN { x = $1; print x }\n", 3),
    !.

test(run_leaves_nothing, [condition(clang_available)]) :-
    cli_leaves_nothing(run, "{ print $1 }\n", 0),
    !.

:- end_tests(plawk_cli_tmp_cleanup).

% --- helpers ---------------------------------------------------------------

scratch(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    current_prolog_flag(pid, Pid),
    format(atom(Dir), '~w/uw_plawk_cli_tmp_cleanup_~w', [Tmp, Pid]),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir).

cli_leaves_nothing(Mode, Src, ExpectedStatus) :-
    scratch(Base),
    directory_file_path(Base, 'tmp', CliTmp),
    make_directory_path(CliTmp),
    directory_file_path(Base, 'prog.plawk', Prog),
    directory_file_path(Base, 'prog_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    (   Mode == build
    ->  Args = ['examples/plawk/bin/plawk', build, Prog, '-o', Bin]
    ;   Args = ['examples/plawk/bin/plawk', run, Prog]
    ),
    format(atom(TmpEnv), '~w', [CliTmp]),
    process_create(path(swipl), Args,
        [environment(['TMP'=TmpEnv]), stdin(null), stdout(null), stderr(null),
         process(Pid)]),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus),
    directory_files(CliTmp, Entries0),
    subtract(Entries0, ['.', '..'], Entries),
    assertion(Entries == []),
    delete_directory_and_contents(Base).
