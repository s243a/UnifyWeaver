% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_strategy_autoselect_e2e.pl - End-to-end test for
% the cost-model-driven auto-selection of bidirectional upgrade in
% the F# WAM target.
%
% Phase 5b deliverable. Companion to test_wam_fsharp_bidirectional_e2e.pl
% (which exercises the EXPLICIT kernel_mode(bidirectional) path).
% This test exercises the AUTO-SELECT path: the caller does NOT pass
% kernel_mode(bidirectional) but passes workload signals that trigger
% the cost model's prefer_bidirectional_csr_present rule.
%
% Verifies the cost-model auto-selection works end-to-end through the
% Prolog pipeline. Asserts on the generated F# files (kernel template
% emitted = bidirectional, not category_ancestor) but DOES NOT run
% dotnet build, because the existing F# code-gen pipeline has a
% pre-existing bug where Program.fs's kernel call site is not updated
% when the kernel template is upgraded (same bug breaks the existing
% test_wam_fsharp_bidirectional_e2e.pl). Once that bug is fixed in a
% separate PR, the dotnet-build step can be added here.
%
% Run: swipl -g main -t halt tests/core/test_wam_fsharp_strategy_autoselect_e2e.pl
% Prerequisites: python3 + lmdb (NO .NET required — build step is
% explicitly skipped)

:- encoding(utf8).
:- use_module('../../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).
:- use_module(library(process)).
:- use_module(library(readutil), [read_file_to_string/3]).

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

%% Assert the same category_ancestor/4 user predicate the bidir-e2e
%% test uses, so kernel detection picks it up.
:- assertz((
    user:category_ancestor(Cat, Parent, 1, Visited) :-
        category_parent(Cat, Parent),
        \+ member(Parent, Visited)
)).
:- assertz((
    user:category_ancestor(Cat, Ancestor, Hops, Visited) :-
        max_depth(MaxD), length(Visited, D), D < MaxD, !,
        category_parent(Cat, Mid),
        \+ member(Mid, Visited),
        category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1
)).
:- assertz(user:max_depth(10)).

main :-
    LmdbDir = '/tmp/uw_fsharp_autoselect_e2e_lmdb',
    CsrDir  = '/tmp/uw_fsharp_autoselect_e2e_csr',
    ProjectDir = '/tmp/uw_fsharp_autoselect_e2e_proj',
    catch(delete_directory_and_contents(LmdbDir), _, true),
    catch(delete_directory_and_contents(CsrDir), _, true),
    catch(delete_directory_and_contents(ProjectDir), _, true),

    %% Step 1: Generate LMDB fixture
    format('Step 1: Generating LMDB fixture (50 parents x 4 children)...~n'),
    run_cmd(python3,
            ['examples/benchmark/generate_synthetic_phase1_lmdb.py',
             LmdbDir, '--parents', '50', '--children-per-parent', '4', '--refresh'],
            '.', PyExit, _),
    (PyExit == 0 -> true ; format('LMDB gen FAILED~n'), halt(1)),

    %% Step 2: Build CSR
    format('Step 2: Building reverse CSR...~n'),
    run_cmd(python3,
            ['examples/benchmark/build_reverse_csr_artifact.py',
             LmdbDir, CsrDir, '--refresh'],
            '.', CsrExit, CsrOut),
    (CsrExit == 0 -> true ; format('CSR build FAILED: ~w~n', [CsrOut]), halt(1)),

    %% Step 3: Generate F# project WITHOUT kernel_mode(bidirectional).
    %% Include workload signals that trigger prefer_bidirectional_csr_present:
    %%   csr_path(CsrDir)            -> csr_available(true) (via build_workload_signals)
    %%   query_pattern(single_pair)  -> direct workload signal
    %%   cardinality(large)          -> direct workload signal
    %%
    %% Cost model should select per_query(bidirectional) with weighted
    %% score 3.0 via prefer_bidirectional_csr_present. The F# WAM
    %% target then upgrades category_ancestor -> bidirectional_ancestor.
    format('Step 3: Generating F# project with cost-model auto-select (NO kernel_mode option)...~n'),
    make_directory_path(ProjectDir),
    catch(
        write_wam_fsharp_project(
            [category_ancestor/4],
            [
                lmdb_path(LmdbDir),
                csr_path(CsrDir),                  % implies csr_available(true)
                query_pattern(single_pair),        % cost-model signal
                cardinality(large),                % cost-model signal
                module_name('uw_autoselect_e2e')
            ],
            ProjectDir),
        GenError,
        ( format('F# project generation FAILED: ~w~n', [GenError]), halt(1) )
    ),

    %% Step 4: Verify the cost-model selected bidirectional. The
    %% generated WamRuntime.fs should contain nativeKernel_bidirectional_ancestor
    %% (proves the upgrade happened) and should NOT contain
    %% nativeKernel_category_ancestor (proves the unupgraded kernel
    %% template was displaced).
    format('Step 4: Verifying auto-select chose bidirectional...~n'),
    directory_file_path(ProjectDir, 'WamRuntime.fs', WamRuntimePath),
    read_file_to_string(WamRuntimePath, WamRuntimeSource, []),

    %% Positive assertion: bidirectional kernel template emitted
    (   sub_string(WamRuntimeSource, _, _, _, "nativeKernel_bidirectional_ancestor")
    ->  format('  [PASS] WamRuntime.fs contains nativeKernel_bidirectional_ancestor~n')
    ;   format('  [FAIL] WamRuntime.fs does NOT contain nativeKernel_bidirectional_ancestor~n'),
        format('         (cost-model did NOT auto-select bidirectional)~n'),
        halt(1)
    ),

    %% Negative assertion: un-upgraded category_ancestor kernel
    %% template is not also emitted. (If both were present, the
    %% upgrade would be ambiguous.)
    (   sub_string(WamRuntimeSource, _, _, _, "let nativeKernel_category_ancestor")
    ->  format('  [FAIL] WamRuntime.fs ALSO contains let nativeKernel_category_ancestor~n'),
        format('         (upgrade was supposed to displace it)~n'),
        halt(1)
    ;   format('  [PASS] WamRuntime.fs does NOT contain unupgraded category_ancestor kernel template~n')
    ),

    %% Step 5: NOTE — dotnet build step is intentionally SKIPPED.
    %%
    %% The companion test_wam_fsharp_bidirectional_e2e.pl runs the
    %% build and currently fails with:
    %%   Program.fs(260,24): error FS0039: The value or constructor
    %%   'nativeKernel_category_ancestor' is not defined.
    %%
    %% This is a pre-existing bug in the F# code-gen pipeline where
    %% Program.fs's kernel call site uses the unupgraded kernel name
    %% even when the kernel template was upgraded. The bug exists
    %% independent of the strategy-selector work (verified by running
    %% the existing test on the commit before Phase 5a merged — same
    %% error).
    %%
    %% Once that bug is fixed in a separate PR, this test should be
    %% extended to run the dotnet build and execute the resulting
    %% binary against the LMDB fixture, matching the bidir-e2e test's
    %% pattern.
    format('Step 5: Dotnet build SKIPPED (pre-existing Program.fs bug — see test header)~n'),

    %% Cleanup
    catch(delete_directory_and_contents(LmdbDir), _, true),
    catch(delete_directory_and_contents(CsrDir), _, true),
    catch(delete_directory_and_contents(ProjectDir), _, true),

    format('~n========================================~n'),
    format('Auto-select e2e PASSED~n'),
    format('Cost-model auto-selected bidirectional upgrade without explicit kernel_mode option~n'),
    format('========================================~n').
