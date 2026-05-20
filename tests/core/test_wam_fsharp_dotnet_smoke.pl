:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_fsharp_dotnet_smoke.pl — Real `dotnet build` + `dotnet run` smokes
% for the WAM-to-F# transpilation target.
%
% Why this exists
% ---------------
% The companion file tests/test_wam_fsharp_target.pl is entirely
% codegen-pattern-based: it asserts string fragments are present in
% the emitted F# source but never invokes the F# compiler. That
% caught zero of the two latent bugs fixed in PR #2340 (multi-line
% record-update syntax in GetList; fs_wam_value/2 crashing on atom
% input from fs_parse_switch_entries).
%
% This smoke test closes that gap with two layers:
%
%   1. Build smokes (test_dotnet_build_*):
%      - empty-predicates project: scaffolding compiles
%      - multi-fact predicate: switch-on-constant path compiles
%
%   2. Run smoke (test_dotnet_run_smoke):
%      - generate a project, OVERWRITE Program.fs with the fixture
%        at tests/fixtures/wam_fsharp_smoke/Smoke.fs
%      - build, then `dotnet run`
%      - parse stdout for `RESULT N/M`, assert N == M and N > 0
%
% The fixture drives the runtime through a few representative
% scenarios that exercise step semantics, backtracking, compareValue,
% derefVar, unifyVal, and the enumerateParBranches helper without
% depending on the codegen pipeline (Instructions are constructed
% inline).
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

%% ------------------------------------------------------------------------
%% Run smoke: overwrite Program.fs with the fixture and exercise the
%% runtime via `dotnet run`.  The fixture (tests/fixtures/wam_fsharp_smoke
%% /Smoke.fs) is independent of which predicates were generated — it
%% constructs Instruction arrays inline — so we can use the empty-
%% predicates project as the build base.
%% ------------------------------------------------------------------------

run_dotnet_run(Dir, ExitCode, Output) :-
    setup_dotnet_env,
    setup_call_cleanup(
        process_create(path(dotnet),
            ['run', '--nologo', '-v', 'quiet', '--no-build'],
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

repo_root_for_smoke(Root) :-
    source_file(repo_root_for_smoke(_), This),
    file_directory_name(This, CoreDir),
    file_directory_name(CoreDir, TestsDir),
    file_directory_name(TestsDir, Root).

smoke_fixture_path(Path) :-
    repo_root_for_smoke(Root),
    directory_file_path(Root,
        'tests/fixtures/wam_fsharp_smoke/Smoke.fs', Path).

copy_file_overwrite(Src, Dst) :-
    setup_call_cleanup(
        open(Src, read, In, [type(binary)]),
        setup_call_cleanup(
            open(Dst, write, Out, [type(binary)]),
            copy_stream_data(In, Out),
            close(Out)
        ),
        close(In)
    ).

%% Parse `RESULT N/M` from the smoke output.  Returns N, M.
parse_smoke_result(Output, Passes, Total) :-
    split_string(Output, "\n", "", Lines),
    member(Line, Lines),
    string_concat("RESULT ", Rest, Line),
    split_string(Rest, "/", "", [PStr, TStr|_]),
    number_string(Passes, PStr),
    number_string(Total, TStr), !.

test_dotnet_run_smoke :-
    Test = 'WAM-FSharp dotnet: runtime smoke (build + run + assert RESULT N/N)',
    smoke_root(Root),
    directory_file_path(Root, runsmoke, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    %% Generate the project (use the empty-predicates shape — the fixture
    %% doesn't depend on Predicates.fs content).
    write_wam_fsharp_project([],
        [no_kernels(true), module_name('uw_fsharp_runsmoke')],
        Dir),
    %% Overwrite Program.fs with the fixture.
    smoke_fixture_path(FixturePath),
    (   exists_file(FixturePath)
    ->  true
    ;   fail_test(Test,
            format_atom('Fixture not found: ~w', [FixturePath])), !, fail
    ),
    directory_file_path(Dir, 'Program.fs', ProgPath),
    copy_file_overwrite(FixturePath, ProgPath),
    %% Build + run.
    run_dotnet_build(Dir, BuildExit, BuildOutput),
    (   BuildExit == 0
    ->  true
    ;   format('---- dotnet build output ----~n~w~n----~n', [BuildOutput]),
        fail_test(Test, 'fixture build failed'), !, fail
    ),
    run_dotnet_run(Dir, RunExit, RunOutput),
    (   parse_smoke_result(RunOutput, Passes, Total)
    ->  (   RunExit == 0,
            Passes =:= Total,
            Passes > 0
        ->  format('  RESULT ~w/~w~n', [Passes, Total]),
            pass(Test),
            maybe_clean(Dir)
        ;   format('---- dotnet run output ----~n~w~n----~n', [RunOutput]),
            fail_test(Test,
                format_atom('runtime smoke failed: ~w/~w pass, exit ~w', [Passes, Total, RunExit]))
        )
    ;   format('---- dotnet run output ----~n~w~n----~n', [RunOutput]),
        fail_test(Test, 'no RESULT line in smoke stdout')
    ).

%% ------------------------------------------------------------------------
%% Phase-I lowered emitter build smoke: hand-roll a WAM body containing
%% every Phase-I instruction, drive lower_predicate_to_fsharp/4 to emit
%% an F# function, splice it into Lowered.fs, and confirm the project
%% builds.  Without lowered-emitter coverage these instructions would
%% silently fall back to interpreter; with coverage they emit step-
%% delegation chains that must compile clean F#.
%% ------------------------------------------------------------------------

test_dotnet_build_phase_i_lowered :-
    Test = 'WAM-FSharp dotnet: Phase-I lowered emitter compiles',
    smoke_root(Root),
    directory_file_path(Root, phase_i, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    %% Generate the empty-predicates project as a base.
    write_wam_fsharp_project([],
        [no_kernels(true), module_name('uw_fsharp_phase_i_lowered')],
        Dir),
    %% Drive the lowered emitter on a body exercising every Phase-I op.
    Wam = 'p_phase_i/3:\n  build_empty_set A1\n  put_structure_dyn A1 A2 A3\n  arg 2 A1 A3\n  not_member_list A1 A2\n  set_insert A1 A2 A3\n  not_member_set A1 A2\n  not_member_const_atoms A1 foo bar baz\n  proceed',
    catch(
        wam_fsharp_target:lower_predicate_to_fsharp(p_phase_i/3, Wam,
            [base_pc(1), foreign_preds([])],
            lowered(_, FuncName, Code)),
        LowerErr,
        (   format('  lower_predicate_to_fsharp error: ~q~n', [LowerErr]),
            fail_test(Test, 'lower_predicate_to_fsharp failed'), !, fail
        )
    ),
    %% Splice into Lowered.fs (overwrites the auto-generated empty stub).
    format(atom(LoweredContent),
'module Lowered

open WamTypes
open WamRuntime

~w

let loweredPredicates : Map<string, WamContext -> WamState -> WamState option> =
    Map.ofList [ ("p_phase_i/3", ~w) ]
', [Code, FuncName]),
    directory_file_path(Dir, 'Lowered.fs', LoweredPath),
    setup_call_cleanup(
        open(LoweredPath, write, S, [encoding(utf8)]),
        write(S, LoweredContent),
        close(S)
    ),
    run_dotnet_build(Dir, ExitCode, Output),
    (   ExitCode == 0,
        sub_string(Output, _, _, _, "Build succeeded")
    ->  pass(Test),
        maybe_clean(Dir)
    ;   format('---- dotnet build output ----~n~w~n----~n', [Output]),
        fail_test(Test,
            format_atom('dotnet build failed (exit ~w) for Phase-I lowered', [ExitCode]))
    ).

%% ------------------------------------------------------------------------
%% Phase-I lowered emitter RUNTIME smoke.  Extends the build smoke above
%% with an actual `dotnet run` step that drives each lowered function
%% with crafted register inputs and asserts on the resulting WamState.
%% Closes the loop on PR #2343 (which only verified the lowered code
%% compiles, not that it runs correctly).
%%
%% Predicate naming convention (matches lower_predicate_to_fsharp''s
%% sanitizer): phase_i_arg/3 -> lowered_phase_i_arg_3, etc.  The fixture
%% Driver.fs (tests/fixtures/wam_fsharp_phase_i_lowered_smoke/) calls
%% these functions by their predictable names.
%% ------------------------------------------------------------------------

phase_i_lowered_scenarios([
    phase_i_arg/3   - 'phase_i_arg/3:\n  arg 2 A1 X1\n  proceed',
    phase_i_nml/2   - 'phase_i_nml/2:\n  not_member_list A1 A2\n  proceed',
    phase_i_vset/2  - 'phase_i_vset/2:\n  build_empty_set X1\n  set_insert A1 X1 X2\n  not_member_set A2 X2\n  proceed',
    phase_i_nmca/1  - 'phase_i_nmca/1:\n  not_member_const_atoms A1 foo bar baz\n  proceed',
    phase_i_psd/3   - 'phase_i_psd/3:\n  put_structure_dyn A1 A2 A3\n  proceed'
]).

phase_i_driver_fixture(Path) :-
    repo_root_for_smoke(Root),
    directory_file_path(Root,
        'tests/fixtures/wam_fsharp_phase_i_lowered_smoke/Driver.fs', Path).

test_dotnet_run_phase_i_lowered :-
    Test = 'WAM-FSharp dotnet: Phase-I lowered runtime smoke (build + run + assert)',
    smoke_root(Root),
    directory_file_path(Root, phase_i_run, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    write_wam_fsharp_project([],
        [no_kernels(true), module_name('uw_fsharp_phase_i_run')],
        Dir),
    %% Lower each scenario and concatenate the emitted F# functions.
    phase_i_lowered_scenarios(Scenarios),
    findall(Code,
            (   member(PI - Wam, Scenarios),
                wam_fsharp_target:lower_predicate_to_fsharp(PI, Wam,
                    [base_pc(1), foreign_preds([])],
                    lowered(_, _, Code))
            ),
            Codes),
    atomic_list_concat(Codes, '\n\n', AllCode),
    format(atom(LoweredContent),
'module Lowered

open WamTypes
open WamRuntime

~w

let loweredPredicates : Map<string, WamContext -> WamState -> WamState option> =
    Map.empty
', [AllCode]),
    directory_file_path(Dir, 'Lowered.fs', LoweredPath),
    setup_call_cleanup(
        open(LoweredPath, write, S, [encoding(utf8)]),
        write(S, LoweredContent),
        close(S)
    ),
    %% Overwrite Program.fs with the runtime driver fixture.
    phase_i_driver_fixture(FixturePath),
    (   exists_file(FixturePath)
    ->  true
    ;   fail_test(Test,
            format_atom('Driver fixture not found: ~w', [FixturePath])), !, fail
    ),
    directory_file_path(Dir, 'Program.fs', ProgPath),
    copy_file_overwrite(FixturePath, ProgPath),
    %% Build + run.
    run_dotnet_build(Dir, BuildExit, BuildOutput),
    (   BuildExit == 0
    ->  true
    ;   format('---- dotnet build output ----~n~w~n----~n', [BuildOutput]),
        fail_test(Test, 'Phase-I lowered runtime build failed'), !, fail
    ),
    run_dotnet_run(Dir, RunExit, RunOutput),
    (   parse_smoke_result(RunOutput, Passes, Total)
    ->  (   RunExit == 0,
            Passes =:= Total,
            Passes > 0
        ->  format('  RESULT ~w/~w~n', [Passes, Total]),
            pass(Test),
            maybe_clean(Dir)
        ;   format('---- dotnet run output ----~n~w~n----~n', [RunOutput]),
            fail_test(Test,
                format_atom('Phase-I lowered runtime failed: ~w/~w pass, exit ~w', [Passes, Total, RunExit]))
        )
    ;   format('---- dotnet run output ----~n~w~n----~n', [RunOutput]),
        fail_test(Test, 'no RESULT line in Phase-I lowered run output')
    ).

%% ========================================================================
%% Runner
%% ========================================================================

run_tests :-
    format('~n========================================~n'),
    format('WAM-FSharp dotnet build smoke~n'),
    format('========================================~n~n'),
    (   dotnet_available
    ->  test_dotnet_build_empty_project,
        test_dotnet_build_multi_fact_project,
        test_dotnet_run_smoke,
        test_dotnet_build_phase_i_lowered,
        test_dotnet_run_phase_i_lowered
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
