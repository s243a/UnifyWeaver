:- encoding(utf8).
%% run_wam_fsharp_tests.pl — aggregate runner for the F# WAM target tests.
%%
%% Each F# WAM test file uses `:- initialization(run_tests, main)` and
%% `halt(0/1)` on completion, so they can't be `use_module`'d in-process.
%% This script spawns each as a subprocess and aggregates pass/fail.
%%
%% Tests run:
%%   tests/test_wam_fsharp_target.pl
%%     - Source-level codegen assertions.  Runs everywhere swipl is
%%       available; no external toolchain required.
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

%% (RelativePath, HumanLabel).  Order matters — codegen first (fast),
%% then dotnet smokes (slower, may skip on toolchain-less machines).
fsharp_test('tests/test_wam_fsharp_target.pl',
            'F# WAM codegen tests').
fsharp_test('tests/core/test_wam_fsharp_kernel_gate_tc.pl',
            'F# WAM kernel capability gate + native transitive_closure2').
fsharp_test('tests/core/test_wam_fsharp_dotnet_smoke.pl',
            'F# WAM dotnet runtime smoke').
fsharp_test('tests/core/test_wam_lmdb_cross_target_conformance.pl',
            'WAM LMDB cross-target conformance (F# eager/lazy/cached vs oracle; Rust opt-in)').

run_one(RelPath, Label, Status) :-
    repo_root(Root),
    atom_concat(Root, '/', Root1),
    atom_concat(Root1, RelPath, AbsPath),
    format('~n──── ~w ────~n', [Label]),
    format('     ~w~n', [RelPath]),
    catch(
        (   process_create(path(swipl),
                ['-q', '-g', 'run_tests', '-t', 'halt', AbsPath],
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
    findall(P-L, fsharp_test(P, L), Tests),
    length(Tests, Total),
    run_all(Tests, [], Results),
    summarize(Results, Total).

run_all([], Acc, Reversed) :-
    reverse(Acc, Reversed).
run_all([P-L|Rest], Acc, Out) :-
    run_one(P, L, Status),
    run_all(Rest, [Status|Acc], Out).

summarize(Results, Total) :-
    include(==(pass), Results, Passes),
    length(Passes, OK),
    Fails is Total - OK,
    format('~n========================================~n', []),
    format('F# WAM suite: ~w/~w passed (~w failed)~n', [OK, Total, Fails]),
    format('========================================~n', []),
    (   OK =:= Total -> halt(0) ; halt(1) ).
