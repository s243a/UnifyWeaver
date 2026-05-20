:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_fsharp_dotnet_smoke.pl — Real `dotnet build` smoke for the
% WAM-to-F# transpilation target.
%
% Why this exists
% ---------------
% The companion file tests/test_wam_fsharp_target.pl is entirely
% codegen-pattern-based: it asserts string fragments are present in
% the emitted F# source but never invokes the F# compiler. That
% caught zero of the two latent bugs fixed in the parent commit
% (multi-line record-update syntax in GetList; fs_wam_value/2
% crashing on atom input from fs_parse_switch_entries).
%
% This smoke test closes that gap by:
%   1. Generating a real F# project via write_wam_fsharp_project/3
%      for two representative shapes:
%        a. An empty-predicates project (covers WamTypes.fs +
%           WamRuntime.fs + Predicates.fs + Lowered.fs scaffolding
%           and the Program.fs benchmark driver).
%        b. A two-fact predicate (covers the multi-clause /
%           switch-on-constant codegen path that triggered bug #2).
%   2. Running `dotnet build` on each and asserting "Build succeeded."
%      with no errors.
%
% Skip behaviour
% --------------
% If `dotnet --version` fails (the .NET SDK is not installed) the
% test prints a [SKIP] diagnostic and exits 0. CI environments
% without a .NET toolchain can still run the rest of the F# WAM
% suite via tests/test_wam_fsharp_target.pl.
%
% Cleanup
% -------
% Build directories are removed on success. Set the environment
% variable WAM_FSHARP_SMOKE_KEEP=1 to keep them for inspection.
%
% Usage:
%   swipl -q -g run_tests -t halt tests/core/test_wam_fsharp_dotnet_smoke.pl

:- use_module('../../src/unifyweaver/targets/wam_fsharp_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).

:- dynamic test_failed/0.
:- dynamic test_skipped/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

skip(Test, Reason) :-
    format('[SKIP] ~w: ~w~n', [Test, Reason]),
    (   test_skipped -> true ; assert(test_skipped) ).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% ========================================================================
%% Toolchain detection
%% ========================================================================

dotnet_available :-
    catch(
        (   process_create(path(dotnet), ['--version'],
                [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0))
        ),
        _, fail).

%% ========================================================================
%% Paths
%% ========================================================================

tmp_root(Root) :-
    (   getenv('TMPDIR', R0), R0 \== ''
    ->  Root = R0
    ;   Root = '/tmp'
    ).

smoke_root(Dir) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_fsharp_dotnet_smoke', Dir).

%% ========================================================================
%% Project generation + build
%% ========================================================================

clean_dir(Dir) :-
    (   exists_directory(Dir)
    ->  process_create(path(rm), ['-rf', Dir],
            [stdout(null), stderr(null), process(Pid)]),
        process_wait(Pid, _)
    ;   true
    ).

%% setenv on the parent so the child inherits HOME / PATH / etc. and just
%% sees our additions.  Avoids the InvalidOperationException from NuGet
%% when HOME is missing.
setup_dotnet_env :-
    setenv('DOTNET_CLI_TELEMETRY_OPTOUT', '1'),
    setenv('DOTNET_NOLOGO', '1').

run_dotnet_build(Dir, ExitCode, Output) :-
    setup_dotnet_env,
    setup_call_cleanup(
        process_create(path(dotnet),
            ['build', '--nologo', '-v', 'minimal'],
            [cwd(Dir),
             stdout(pipe(Out)), stderr(pipe(Err)),
             process(Pid)]),
        (   read_string(Out, _, OutText),
            read_string(Err, _, ErrText),
            process_wait(Pid, exit(ExitCode)),
            atomic_list_concat([OutText, '\n', ErrText], Output)
        ),
        (   catch(close(Out), _, true),
            catch(close(Err), _, true)
        )
    ).

keep_build_dirs :-
    catch(getenv('WAM_FSHARP_SMOKE_KEEP', '1'), _, fail), !.
keep_build_dirs :- fail.

maybe_clean(Dir) :-
    (   keep_build_dirs
    ->  format('  (keeping build dir: ~w)~n', [Dir])
    ;   clean_dir(Dir)
    ).

%% ------------------------------------------------------------------------
%% Smoke 1: empty-predicates project builds
%% ------------------------------------------------------------------------

test_dotnet_build_empty_project :-
    Test = 'WAM-FSharp dotnet: empty-predicates project builds',
    smoke_root(Root),
    directory_file_path(Root, empty, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    write_wam_fsharp_project([],
        [no_kernels(true), module_name('uw_fsharp_smoke_empty')],
        Dir),
    run_dotnet_build(Dir, ExitCode, Output),
    (   ExitCode == 0,
        sub_string(Output, _, _, _, "Build succeeded")
    ->  pass(Test),
        maybe_clean(Dir)
    ;   format('---- dotnet build output ----~n~w~n----~n', [Output]),
        fail_test(Test,
            format_atom('dotnet build failed (exit ~w) for empty project', [ExitCode]))
    ).

%% ------------------------------------------------------------------------
%% Smoke 2: multi-fact predicate project builds
%% ------------------------------------------------------------------------

:- dynamic user:parent_smoke/2.

setup_parent_facts :-
    retractall(user:parent_smoke(_, _)),
    assertz(user:parent_smoke(tom, bob)),
    assertz(user:parent_smoke(bob, ann)),
    assertz(user:parent_smoke(ann, eve)).

test_dotnet_build_multi_fact_project :-
    Test = 'WAM-FSharp dotnet: multi-fact predicate project builds',
    setup_parent_facts,
    smoke_root(Root),
    directory_file_path(Root, multi, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    catch(
        write_wam_fsharp_project([parent_smoke/2],
            [no_kernels(true), module_name('uw_fsharp_smoke_multi')],
            Dir),
        E,
        (   format('  ERROR during write_wam_fsharp_project: ~q~n', [E]),
            fail_test(Test, 'write_wam_fsharp_project crashed'),
            !,
            fail
        )
    ),
    run_dotnet_build(Dir, ExitCode, Output),
    (   ExitCode == 0,
        sub_string(Output, _, _, _, "Build succeeded")
    ->  pass(Test),
        maybe_clean(Dir)
    ;   format('---- dotnet build output ----~n~w~n----~n', [Output]),
        fail_test(Test,
            format_atom('dotnet build failed (exit ~w) for multi-fact project', [ExitCode]))
    ).

%% Helper for inline format → atom
format_atom(Format, Args, Atom) :-
    format(atom(Atom), Format, Args).
format_atom(Format, Args) :-
    format_atom(Format, Args, _).

%% ========================================================================
%% Runner
%% ========================================================================

run_tests :-
    format('~n========================================~n'),
    format('WAM-FSharp dotnet build smoke~n'),
    format('========================================~n~n'),
    (   dotnet_available
    ->  test_dotnet_build_empty_project,
        test_dotnet_build_multi_fact_project
    ;   skip('WAM-FSharp dotnet smoke', 'dotnet not on PATH — install .NET SDK to run')
    ),
    format('~n========================================~n'),
    (   test_failed
    ->  format('Tests FAILED~n'), halt(1)
    ;   (   test_skipped
        ->  format('Tests SKIPPED~n')
        ;   format('All tests passed~n')
        )
    ).
