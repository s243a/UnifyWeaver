:- encoding(utf8).
%% run_wam_fsharp_tests.pl — aggregate runner for the F# WAM target tests.
%%
%% The files use incompatible top-level runners and shared dynamic fixtures,
%% so this script loads each in an isolated subprocess with an explicit goal
%% and aggregates pass/fail.
%%
%% Tests run:
%%   tests/test_wam_fsharp_target.pl
%%     - Source-level codegen assertions.  Runs everywhere swipl is
%%       available; no external toolchain required.
%%   tests/test_wam_td3_contract_parity.pl
%%     - fleet TD3 dist+ oracle/structural checks plus executable F#/C/Rust
%%       retry, binding-mode, cycle, and no-kernels coverage.
%%   tests/test_wam_tpd4_contract_parity.pl
%%     - fleet TPD4 shortest-positive-parents oracle/structural checks plus
%%       executable F#/C/Rust/Elixir coverage; LLVM remains capability-gated.
%%   tests/test_wam_tspd5_contract_parity.pl
%%     - fleet TSPD5 shortest-positive correlated step/parent oracle/structural
%%       checks plus executable F#/C/Rust/Elixir/Go coverage; LLVM remains
%%       capability-gated.
%%   focused Rust, Go, Scala, R, Haskell, Elixir, and LLVM TD3/TPD4/TSPD5 regressions
%%     - exercise transactional correlated-tuple binding, the R aggregate
%%       path, Elixir ordinary retries, and LLVM exact paired streaming /
%%       range safety.
%%       Runtime legs skip only when their external compiler is unavailable.
%%   tests/core/test_wam_fsharp_dotnet_smoke.pl
%%     - dotnet build + run smokes (8 cases including category-ancestor
%%       end-to-end + NAF micro-benchmark).  Each sub-case skips
%%       gracefully if `dotnet` isn't on PATH.
%%   tests/core/test_wam_lmdb_cross_target_conformance.pl
%%     - builds a tiny int-native LMDB fixture (needs python3 + lmdb) and
%%       asserts every available target's LMDB category_ancestor benchmark
%%       agrees with the oracle.  F# eager/lazy/cached run when `dotnet` is
%%       present; the Rust cursor leg is opt-in (UW_CONFORMANCE_RUST=1 + cargo).
%%       All legs skip gracefully when their toolchain is absent.  This is the
%%       guard for the LMDB-mode id-space divergence class (the Rust s2i bug).
%%
%% Exit code: 0 iff every spawned test exits 0.
%%
%% Usage:
%%   swipl -q -g main -t halt tests/run_wam_fsharp_tests.pl
%%
%% Invoked indirectly by the top-level run_all_tests.pl runner.

:- use_module(library(process)).

repo_root(Root) :-
    source_file(repo_root(_), This),
    file_directory_name(This, TestsDir),
    file_directory_name(TestsDir, Root).

%% (RelativePath, HumanLabel, Goal).  Order matters — codegen first (fast),
%% then dotnet smokes (slower, may skip on toolchain-less machines).
fsharp_test('tests/test_wam_fsharp_target.pl',
            'F# WAM codegen tests', run_tests).
fsharp_test('tests/core/test_wam_fsharp_kernel_gate_tc.pl',
            'F# WAM kernel capability gate + native transitive_closure2',
            run_tests).
fsharp_test('tests/test_wam_td3_contract_parity.pl',
            'TD3 strict dist+ fleet contract + F# native kernel', run_tests).
fsharp_test('tests/test_wam_tpd4_contract_parity.pl',
            'TPD4 shortest-positive-parents fleet contract + F# native kernel',
            run_tests).
fsharp_test('tests/test_wam_tspd5_contract_parity.pl',
            'TSPD5 shortest-positive correlated step/parent fleet contract + F# native kernel',
            run_tests).
fsharp_test('tests/test_wam_rust_foreign_tuple_aliases.pl',
            'Rust correlated foreign-tuple alias/retry cleanup', run_tests).
fsharp_test('tests/test_wam_go_foreign_tuple_aliases.pl',
            'Go correlated foreign-tuple alias/retry cleanup', run_tests).
fsharp_test('tests/test_wam_scala_foreign_tuple_aliases.pl',
            'Scala correlated foreign-tuple alias/retry cleanup', run_tests).
fsharp_test('tests/test_wam_r_generator.pl',
            'R TPD4/TSPD5 aggregate, binding-mode, alias, and retry contract',
            run_tests([wam_r_generator:kernel_tpd4_e2e_rscript,
                       wam_r_generator:kernel_tspd5_e2e_rscript])).
fsharp_test('tests/test_wam_scala_kernels.pl',
            'TD3/TPD4/TSPD5 Scala kernel structural + runtime parity',
            ( run_tests(wam_scala_kernels_structural:
                        distance_kernel_emits_handler_and_stub),
              run_tests(wam_scala_kernels_structural:
                        parent_distance_kernel_emits_handler_and_stub),
              run_tests(wam_scala_kernels_structural:
                        step_parent_distance_kernel_emits_handler_and_stub),
              run_tests(wam_scala_kernels_runtime:
                        transitive_distance_rejects_non_atom_source),
              run_tests(wam_scala_kernels_runtime:
                        transitive_parent_distance_rejects_non_atom_nodes),
              run_tests(wam_scala_kernels_runtime:
                        transitive_step_parent_distance_parity),
              run_tests(wam_scala_kernels_runtime:
                        transitive_step_parent_distance_rejects_non_atom_nodes) )).
fsharp_test('tests/test_wam_haskell_target.pl',
            'TD3/TPD4/TSPD5 Haskell multi-output foreign retry regressions',
            (test_multi_output_foreign_filters_bound_results,
             test_multi_output_foreign_retry_filters_and_pops,
             test_same_kind_kernel_body_deduplicated,
             test_tpd4_generated_ghc_smoke,
             test_tspd5_generated_ghc_smoke)).
fsharp_test('tests/test_wam_elixir_target.pl',
            'TD3/TPD4/TSPD5 Elixir bound modes + ordinary retry stream regressions',
            (test_graph_kernel_transitive_distance_uses_distplus_bfs,
             test_kernel_dispatch_emits_transitive_distance_module,
             test_kernel_dispatch_transitive_distance_e2e,
             test_graph_kernel_transitive_parent_distance_uses_bfs_parent_sets,
             test_kernel_dispatch_emits_transitive_parent_distance_module,
             test_kernel_dispatch_transitive_parent_distance_e2e,
             test_graph_kernel_transitive_step_parent_distance_emitted_in_runtime,
             test_graph_kernel_tspd_uses_bfs_correlated_contract,
             test_kernel_dispatch_emits_transitive_step_parent_distance_module,
             test_shared_detector_finds_transitive_step_parent_distance)).
fsharp_test('tests/core/test_wam_llvm_reach_execution.pl',
            'TD3 LLVM bound-distance + range-safety execution', test_all).
fsharp_test('tests/core/test_wam_llvm_td3_stream_execution.pl',
            'TD3 LLVM exact paired-stream execution', test_all).
fsharp_test('tests/core/test_wam_fsharp_dotnet_smoke.pl',
            'F# WAM dotnet runtime smoke', run_tests).
fsharp_test('tests/core/test_wam_lmdb_cross_target_conformance.pl',
            'WAM LMDB cross-target conformance (F# eager/lazy/cached vs oracle; Rust opt-in)',
            run_tests).

run_one(RelPath, Label, Goal, Status) :-
    repo_root(Root),
    atom_concat(Root, '/', Root1),
    atom_concat(Root1, RelPath, AbsPath),
    format('~n──── ~w ────~n', [Label]),
    format('     ~w~n', [RelPath]),
    catch(
        (   term_to_atom(Goal, GoalAtom),
            process_create(path(swipl),
                ['-q', '-l', AbsPath, '-g', GoalAtom, '-t', 'halt'],
                [stdout(std), stderr(std), process(Pid)]),
            process_wait(Pid, exit(EC))
        ),
        Err,
        (   format('     [orchestrator error] ~q~n', [Err]),
            EC = 99
        )),
    (   EC == 0
    ->  Status = pass,
        format('[orchestrator PASS] ~w (exit=0)~n', [Label])
    ;   Status = fail(EC),
        format('[orchestrator FAIL] ~w (exit=~w)~n', [Label, EC])
    ).

main :-
    format('~n========================================~n', []),
    format('F# WAM target test suite~n', []),
    format('========================================~n', []),
    findall(test(P, L, G), fsharp_test(P, L, G), Tests),
    length(Tests, Total),
    run_all(Tests, [], Results),
    summarize(Results, Total).

run_all([], Acc, Reversed) :-
    reverse(Acc, Reversed).
run_all([test(P, L, G)|Rest], Acc, Out) :-
    run_one(P, L, G, Status),
    run_all(Rest, [Status|Acc], Out).

summarize(Results, Total) :-
    include(==(pass), Results, Passes),
    length(Passes, OK),
    Fails is Total - OK,
    format('~n========================================~n', []),
    format('F# WAM suite: ~w/~w passed (~w failed)~n', [OK, Total, Fails]),
    format('========================================~n', []),
    (   OK =:= Total -> halt(0) ; halt(1) ).
