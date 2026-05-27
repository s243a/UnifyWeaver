% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_haskell_st_regs_bench.pl - ST register benchmark on effective-distance
%
% Generates a Haskell effective-distance project, writes a driver that
% runs both run (IntMap) and runMutableRegs (STArray) and compares
% correctness + timing.
%
% Prerequisites: ghc, cabal, LANG=C.UTF-8

:- encoding(utf8).
:- use_module('../../src/unifyweaver/targets/wam_haskell_target',
              [write_wam_haskell_project/3]).
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3,
                                  copy_file/2]).
:- use_module(library(process)).
:- use_module(library(option)).

%% effective_distance workload predicates
:- discontiguous article_category/2.
:- discontiguous category_parent/2.
:- discontiguous root_category/1.

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
    FactsPath = 'data/benchmark/dev/facts.pl',
    ProjectDir = '/tmp/uw_haskell_st_bench',
    catch(delete_directory_and_contents(ProjectDir), _, true),

    %% Load facts
    format('Loading facts from ~w...~n', [FactsPath]),
    load_files(FactsPath, [silent(true)]),

    %% Load effective_distance workload
    load_files('examples/benchmark/effective_distance.pl', [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    %% Generate project
    format('Generating Haskell project...~n'),
    Predicates = [
        user:dimension_n/1,
        user:max_depth/1,
        user:category_ancestor/4
    ],
    write_wam_haskell_project(Predicates,
        [no_kernels(true), module_name('wam-haskell-bench'), use_hashmap(false)],
        ProjectDir),

    %% Replace Main.hs with benchmark driver
    directory_file_path(ProjectDir, 'src', SrcDir),
    directory_file_path(SrcDir, 'Main.hs', MainPath),
    copy_file('tests/fixtures/haskell_st_regs_bench_driver.hs', MainPath),

    %% Build
    format('Building (cabal)...~n'),
    run_cmd(cabal, ['build'], ProjectDir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('Build OK.~n')
    ;   format('BUILD FAILED:~n~w~n', [BuildOut]), halt(1)
    ),

    %% Run
    format('Running benchmark...~n~n'),
    atom_string(ProjectDir, ProjDirStr),
    run_cmd(cabal, ['run', 'wam-haskell-bench', '--',
                    '/home/user/UnifyWeaver/data/benchmark/dev', '7'],
            ProjectDir, RunExit, RunOut),
    format('~w~n', [RunOut]),
    halt(RunExit).
