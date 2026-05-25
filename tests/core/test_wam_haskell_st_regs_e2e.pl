% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_haskell_st_regs_e2e.pl - End-to-end test for ST mutable registers
%
% Generates a Haskell WAM project with a simple predicate, replaces
% Main.hs with a driver that calls both `run` (IntMap) and
% `runMutableRegs` (STArray), and verifies they produce identical results.
%
% Prerequisites: ghc, cabal, LANG=C.UTF-8

:- encoding(utf8).
:- use_module('../../src/unifyweaver/targets/wam_haskell_target',
              [write_wam_haskell_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3,
                                  copy_file/2]).
:- use_module(library(process)).

%% Define a tiny test predicate
:- dynamic user:st_probe/1.
user:st_probe(42).

run_cmd(Prog, Args, Dir, ExitCode, OutText) :-
    setup_call_cleanup(
        process_create(path(Prog), Args,
                       [cwd(Dir),
                        stdout(pipe(Out)),
                        stderr(pipe(Err)),
                        process(PID)]),
        ( read_string(Out, _, OutStr),
          read_string(Err, _, ErrStr),
          process_wait(PID, exit(ExitCode)),
          string_concat(OutStr, ErrStr, OutText)
        ),
        ( catch(close(Out), _, true), catch(close(Err), _, true) )).

main :-
    ProjectDir = '/tmp/uw_haskell_st_regs_e2e',
    catch(delete_directory_and_contents(ProjectDir), _, true),

    format('Generating Haskell project...~n'),
    write_wam_haskell_project(
        [user:st_probe/1],
        [no_kernels(true), module_name('uw-st-regs-e2e'), use_hashmap(false)],
        ProjectDir),

    %% Replace Main.hs with our test driver
    directory_file_path(ProjectDir, 'src', SrcDir),
    directory_file_path(SrcDir, 'Main.hs', MainPath),
    copy_file('tests/fixtures/haskell_st_regs_driver.hs', MainPath),

    %% Build
    format('Building (cabal)...~n'),
    run_cmd(cabal, ['build'], ProjectDir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('Build OK.~n')
    ;   format('BUILD FAILED:~n~w~n', [BuildOut]), halt(1)
    ),

    %% Run
    format('Running E2E test...~n~n'),
    run_cmd(cabal, ['run'], ProjectDir, RunExit, RunOut),
    format('~w~n', [RunOut]),
    halt(RunExit).
