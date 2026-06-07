% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_strategy_autoselect_e2e.pl - End-to-end test for
% the cost-model-driven auto-selection path AND the safe-by-default
% bidirectional emission policy.
%
% Phase 5b deliverable + bidirectional-not-default fix verification.
% Companion to test_wam_fsharp_bidirectional_e2e.pl (which exercises
% the OPT-IN bidirectional emission path via
% allow_bidirectional_kernel_swap(true)).
%
% This test exercises the AUTO-SELECT path with the safe default:
% the caller passes workload signals that trigger the cost-model's
% prefer_bidirectional_csr_present rule, but does NOT set
% allow_bidirectional_kernel_swap(true). The cost-model still
% recommends bidirectional (logged in the trace + the suppression
% message), but the actual emission stays as category_ancestor —
% which matches program.fs.mustache's hardcoded benchmark loop and
% lets the dotnet build SUCCEED.
%
% This is the "safe-by-default" behaviour the fix introduces:
% advisory cost-model decisions don't break the build.
%
% Run: swipl -g main -t halt tests/core/test_wam_fsharp_strategy_autoselect_e2e.pl
% Prerequisites: python3 + lmdb + .NET 8 SDK (build step IS run
% as part of the verification)

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

    %% Step 3: Generate F# project WITHOUT kernel_mode(bidirectional)
    %% AND WITHOUT allow_bidirectional_kernel_swap(true). Include
    %% workload signals that would trigger the cost-model's
    %% prefer_bidirectional_csr_present rule:
    %%   csr_path(CsrDir)            -> csr_available(true) (via build_workload_signals)
    %%   query_pattern(single_pair)  -> direct workload signal
    %%   cardinality(large)          -> direct workload signal
    %%
    %% Cost model selects per_query(bidirectional) (advisory) but
    %% the F# WAM target SUPPRESSES the kernel-kind swap by default
    %% (because allow_bidirectional_kernel_swap(true) is not set).
    %% The emitted kernel stays as category_ancestor, which is
    %% consistent with program.fs.mustache's hardcoded benchmark
    %% loop — so the dotnet build succeeds.
    format('Step 3: Generating F# project with cost-model preferring bidirectional (default-safe path)...~n'),
    format('         The [WAM-FSharp] suppression log line above is the expected behaviour.~n'),
    make_directory_path(ProjectDir),
    catch(
        write_wam_fsharp_project(
            [category_ancestor/4],
            [
                lmdb_path(LmdbDir),
                csr_path(CsrDir),                  % implies csr_available(true)
                query_pattern(single_pair),        % cost-model signal
                cardinality(large),                % cost-model signal
                %% NO allow_bidirectional_kernel_swap(true) —
                %% testing the safe default
                module_name('uw_autoselect_e2e')
            ],
            ProjectDir),
        GenError,
        ( format('F# project generation FAILED: ~w~n', [GenError]), halt(1) )
    ),

    %% Step 4: Verify the SAFE-DEFAULT path. The generated WamRuntime.fs
    %% should contain nativeKernel_category_ancestor (proves the
    %% suppression worked) and should NOT contain
    %% nativeKernel_bidirectional_ancestor (proves the upgrade was
    %% NOT applied).
    format('Step 4: Verifying safe-default emission (category_ancestor, NOT bidirectional)...~n'),
    directory_file_path(ProjectDir, 'WamRuntime.fs', WamRuntimePath),
    read_file_to_string(WamRuntimePath, WamRuntimeSource, []),

    %% Positive assertion: category_ancestor kernel template emitted
    (   sub_string(WamRuntimeSource, _, _, _, "nativeKernel_category_ancestor")
    ->  format('  [PASS] WamRuntime.fs contains nativeKernel_category_ancestor (safe default)~n')
    ;   format('  [FAIL] WamRuntime.fs does NOT contain nativeKernel_category_ancestor~n'),
        format('         (suppression did NOT keep the safe default emission)~n'),
        halt(1)
    ),

    %% Negative assertion: bidirectional kernel template NOT emitted
    (   sub_string(WamRuntimeSource, _, _, _, "let nativeKernel_bidirectional_ancestor")
    ->  format('  [FAIL] WamRuntime.fs contains let nativeKernel_bidirectional_ancestor~n'),
        format('         (the kernel-kind swap was supposed to be suppressed by default)~n'),
        halt(1)
    ;   format('  [PASS] WamRuntime.fs does NOT contain bidirectional kernel template (correctly suppressed)~n')
    ),

    %% Step 5: Actually run dotnet build — with the safe default
    %% (category_ancestor in both WamRuntime.fs and Program.fs),
    %% the build should succeed.
    format('Step 5: Running dotnet build (should succeed with safe default)...~n'),
    run_cmd(dotnet,
            ['build', '--nologo', '-v', 'minimal', '-c', 'Release'],
            ProjectDir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('  [PASS] dotnet build succeeded — safe default produces a buildable F# project~n')
    ;   format('  [FAIL] dotnet build FAILED:~n~w~n', [BuildOut]),
        halt(1)
    ),

    %% Cleanup
    catch(delete_directory_and_contents(LmdbDir), _, true),
    catch(delete_directory_and_contents(CsrDir), _, true),
    catch(delete_directory_and_contents(ProjectDir), _, true),

    format('~n========================================~n'),
    format('Auto-select + safe-default e2e PASSED~n'),
    format('Cost-model preferred bidirectional but the swap was suppressed by default;~n'),
    format('the emitted F# project built successfully with category_ancestor.~n'),
    format('========================================~n').
