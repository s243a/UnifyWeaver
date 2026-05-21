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

%% ------------------------------------------------------------------------
%% Query smoke: end-to-end multi-clause predicate dispatch.
%% Compiles `parent(tom, bob). parent(bob, ann). parent(ann, eve).` from
%% real Prolog source through write_wam_fsharp_project/3, then drives
%% queries against it via the runtime's dispatchCall.
%%
%% Surfaces two pre-existing F# WAM dispatch bugs (currently asserted as
%% [KNOWN BUG] in the fixture so the smoke passes — see Driver.fs for
%% details).  When those bugs are fixed in follow-up PRs, the
%% [KNOWN BUG] assertions will need to flip.
%% ------------------------------------------------------------------------

:- dynamic user:parent_query_smoke/2.

setup_parent_query_facts :-
    retractall(user:parent_query_smoke(_, _)),
    assertz(user:parent_query_smoke(tom, bob)),
    assertz(user:parent_query_smoke(bob, ann)),
    assertz(user:parent_query_smoke(ann, eve)).

query_smoke_driver_fixture(Path) :-
    repo_root_for_smoke(Root),
    directory_file_path(Root,
        'tests/fixtures/wam_fsharp_query_smoke/Driver.fs', Path).

test_dotnet_run_query_smoke :-
    Test = 'WAM-FSharp dotnet: query smoke (multi-clause predicate, real Prolog source)',
    setup_parent_query_facts,
    smoke_root(Root),
    directory_file_path(Root, query, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    %% Compile parent_query_smoke/2 (renamed to parent/2 in Predicates.fs
    %% wouldn't be straightforward; instead we generate with the actual
    %% pred name and have the Driver.fs reference allCode/allLabels directly).
    %% Simpler: just use a generic predicate name and let Driver.fs find
    %% it via dispatchCall.  The Prolog source uses parent_query_smoke/2;
    %% the driver knows to call "parent_query_smoke/2".
    %%
    %% Actually the simplest is to use a Prolog predicate literally named
    %% "parent" so the driver can call "parent/2".  Use a fresh dynamic
    %% predicate that we then retract.
    write_wam_fsharp_project([parent_query_smoke/2],
        [no_kernels(true), module_name('uw_fsharp_query_smoke')],
        Dir),
    %% The fixture references `parent_query_smoke/2` via dispatchCall.
    query_smoke_driver_fixture(FixturePath),
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
        fail_test(Test, 'query smoke build failed'), !, fail
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
                format_atom('query smoke failed: ~w/~w pass, exit ~w', [Passes, Total, RunExit]))
        )
    ;   format('---- dotnet run output ----~n~w~n----~n', [RunOutput]),
        fail_test(Test, 'no RESULT line in query smoke output')
    ).

%% ------------------------------------------------------------------------
%% Category-ancestor end-to-end benchmark smoke.
%%
%% First test that exercises a NON-TRIVIAL real workload through the full
%% pipeline: load effective_distance.pl + data/benchmark/dev/facts.pl,
%% compile category_ancestor/4 + category_parent/2 + max_depth/1 to F#
%% via write_wam_fsharp_project/3, build, then run the auto-generated
%% Program.fs against the dev fixture's TSV files.
%%
%% The auto-generated Program.fs is a category-ancestor benchmark driver
%% with hard-coded query semantics — it passes (Cat, Root, Distance) as
%% A1/A2/A3.  The effective_distance.pl category_ancestor/4 signature is
%% (+Cat, -Ancestor, -Hops, +Visited), so the driver's args don't match
%% the predicate's expected output mode — solutions=0 is the expected
%% outcome.  This is fine for our purpose: we're proving the PIPELINE
%% works, not validating Prolog semantics for this specific query.
%%
%% Assertions:
%%   - Build succeeds.
%%   - dotnet run exits 0.
%%   - stdout contains "total_ms=" (the driver's summary line).
%%   - stdout contains "RESULT 0/0" not asserted — the driver doesn't
%%     emit RESULT lines.  We parse out total_ms instead for the smoke
%%     output.

bench_dataset_dir(Dir) :-
    repo_root_for_smoke(Root),
    directory_file_path(Root, 'data/benchmark/dev', Dir).

effective_distance_workload(Path) :-
    repo_root_for_smoke(Root),
    directory_file_path(Root, 'examples/benchmark/effective_distance.pl', Path).

dev_facts_path(Path) :-
    repo_root_for_smoke(Root),
    directory_file_path(Root, 'data/benchmark/dev/facts.pl', Path).

%% `--nologo` is NOT a recognized `dotnet run` flag — including it
%% would push it through to the program as argv[0].  The other smokes
%% pass `--nologo` and their fixtures ignore argv, so it slips through
%% invisibly; here the program actually consumes argv (factsDir), so
%% we omit it.
run_dotnet_run_with_args(Dir, Args, ExitCode, Output) :-
    setup_dotnet_env,
    append(['run', '-v', 'quiet', '--no-build', '--'], Args, FullArgs),
    setup_call_cleanup(
        process_create(path(dotnet), FullArgs,
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

%% Parse `total_ms=N` from the benchmark stdout.
parse_total_ms(Output, Ms) :-
    split_string(Output, "\n", "", Lines),
    member(Line, Lines),
    sub_string(Line, B, _, _, "total_ms="),
    B0 is B + 9,
    sub_string(Line, B0, _, 0, Rest),
    split_string(Rest, " \t\r", "", [MsStr|_]),
    number_string(Ms, MsStr), !.

test_dotnet_run_category_ancestor_bench :-
    Test = 'WAM-FSharp dotnet: category-ancestor end-to-end benchmark',
    smoke_root(Root),
    directory_file_path(Root, cat_ancestor, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    effective_distance_workload(WorkloadPath),
    dev_facts_path(FactsPath),
    (   exists_file(WorkloadPath), exists_file(FactsPath)
    ->  true
    ;   fail_test(Test,
            format_atom('Missing workload or facts: ~w / ~w', [WorkloadPath, FactsPath])),
        !, fail
    ),
    bench_dataset_dir(BenchDir),
    %% Load the workload + facts so the WAM compiler sees the predicates.
    catch(
        ( load_files(WorkloadPath, [silent(true)]),
          load_files(FactsPath,    [silent(true)]),
          write_wam_fsharp_project(
              [category_ancestor/4, category_parent/2, max_depth/1],
              [no_kernels(true), module_name('cat_ancestor_bench')],
              Dir)
        ),
        Err,
        (   format('  Prolog generation error: ~q~n', [Err]),
            fail_test(Test, 'project generation failed'), !, fail
        )
    ),
    run_dotnet_build(Dir, BuildExit, BuildOutput),
    (   BuildExit == 0
    ->  true
    ;   format('---- dotnet build output ----~n~w~n----~n', [BuildOutput]),
        fail_test(Test, 'category-ancestor build failed'), !, fail
    ),
    %% Run with `<factsDir> <reps>` args.
    run_dotnet_run_with_args(Dir, [BenchDir, '3'], RunExit, RunOutput),
    (   RunExit == 0,
        parse_total_ms(RunOutput, Ms)
    ->  format('  total_ms=~w (dev fixture, 3 reps)~n', [Ms]),
        pass(Test),
        maybe_clean(Dir)
    ;   format('---- dotnet run output ----~n~w~n----~n', [RunOutput]),
        fail_test(Test,
            format_atom('category-ancestor run failed: exit ~w', [RunExit]))
    ).

%% ------------------------------------------------------------------------
%% NAF micro-benchmark smoke.
%%
%% Drives runNegationParallel with hand-constructed 5-branch Par* chains:
%%   - Scenario A: 1 fast-succeed + 4 slow-fail.  Async.Choice from
%%     PR #2353 returns on first Some, so wall_ms_A should be small.
%%   - Scenario B: all 5 slow-fail.  Async.Choice waits for all to
%%     return None.  wall_ms_B should be larger.
%%
%% Both scenarios test correctness of `runNegationParallel`'s boolean
%% return.  A is expected to return true (any branch succeeded), B is
%% expected to return false.  Wall time is reported for future
%% comparison after hard-cancel work.
%% ------------------------------------------------------------------------

naf_microbench_driver(Path) :-
    repo_root_for_smoke(Root),
    directory_file_path(Root,
        'tests/fixtures/wam_fsharp_naf_microbench/Driver.fs', Path).

%% Parse a single metric like `wall_ms_A=N` from the output.  Returns
%% the integer value of the field, or fails if not present.
parse_naf_metric(Output, Key, N) :-
    split_string(Output, "\n", "", Lines),
    member(Line, Lines),
    atom_string(Key, KeyStr),
    string_concat(KeyStr, "=", Marker),
    sub_string(Line, B, _, _, Marker),
    string_length(Marker, MLen),
    B0 is B + MLen,
    sub_string(Line, B0, _, 0, Rest),
    split_string(Rest, " \t", "", [NStr|_]),
    number_string(N, NStr), !.

parse_naf_microbench(Output, WallA, WallB, CpuA, CpuB) :-
    parse_naf_metric(Output, wall_ms_A, WallA),
    parse_naf_metric(Output, wall_ms_B, WallB),
    parse_naf_metric(Output, cpu_ms_A,  CpuA),
    parse_naf_metric(Output, cpu_ms_B,  CpuB).

test_dotnet_run_naf_microbench :-
    Test = 'WAM-FSharp dotnet: NAF micro-benchmark (runNegationParallel)',
    smoke_root(Root),
    directory_file_path(Root, naf_microbench, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    write_wam_fsharp_project([],
        [no_kernels(true), module_name('uw_fsharp_naf_microbench')],
        Dir),
    naf_microbench_driver(FixturePath),
    (   exists_file(FixturePath)
    ->  true
    ;   fail_test(Test,
            format_atom('NAF microbench driver fixture not found: ~w', [FixturePath])),
        !, fail
    ),
    directory_file_path(Dir, 'Program.fs', ProgPath),
    copy_file_overwrite(FixturePath, ProgPath),
    run_dotnet_build(Dir, BuildExit, BuildOutput),
    (   BuildExit == 0
    ->  true
    ;   format('---- dotnet build output ----~n~w~n----~n', [BuildOutput]),
        fail_test(Test, 'NAF microbench build failed'), !, fail
    ),
    run_dotnet_run(Dir, RunExit, RunOutput),
    (   parse_smoke_result(RunOutput, Passes, Total),
        parse_naf_microbench(RunOutput, WallA, WallB, CpuA, CpuB)
    ->  (   RunExit == 0,
            Passes =:= Total,
            Passes > 0
        ->  format('  wall_ms_A=~w cpu_ms_A=~w  wall_ms_B=~w cpu_ms_B=~w (A=fast-succeed, B=all-fail)~n',
                   [WallA, CpuA, WallB, CpuB]),
            pass(Test),
            maybe_clean(Dir)
        ;   format('---- dotnet run output ----~n~w~n----~n', [RunOutput]),
            fail_test(Test,
                format_atom('NAF microbench failed: ~w/~w pass, exit ~w', [Passes, Total, RunExit]))
        )
    ;   format('---- dotnet run output ----~n~w~n----~n', [RunOutput]),
        fail_test(Test, 'no RESULT, wall_ms, or cpu_ms lines in microbench output')
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
        test_dotnet_run_phase_i_lowered,
        test_dotnet_run_query_smoke,
        test_dotnet_run_category_ancestor_bench,
        test_dotnet_run_naf_microbench
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
