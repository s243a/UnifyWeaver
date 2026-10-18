:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_go_effective_distance_bench.pl — BENCH-GO driver contract
%
% Closes the BENCH-GO card in docs/WAM_FLEET_GAP_TASKS.md. The Go
% effective-distance generator already wired the benchmark facts into
% the emitted project; what was missing was rep support in the driver
% and a harness block that actually invoked this generator (the
% cross-target script called generate_wam_go_optimized_benchmark.pl and
% only built it).
%
% This test pins the driver contract so the harness keeps working:
% [factsDir] [reps] arguments, median-of-reps timing, and the stderr
% metric block the results doc quotes. It generates and builds the
% project but does not run the scale-300 query — that belongs to
% examples/benchmark/run_wam_cross_target_benchmark.sh, whose numbers
% land in docs/design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md.
%
% Skipped when the scale-300 fixture is absent; the build step is
% skipped when `go` isn't on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).

:- begin_tests(wam_go_effective_distance_bench).

bench_facts_file('data/benchmark/300/facts.pl').

bench_generator('examples/benchmark/generate_wam_go_effective_distance_benchmark.pl').

test(go_ed_generator_emits_reps_aware_driver) :-
    bench_facts_file(Facts),
    (   \+ exists_file(Facts)
    ->  format('~n[skip] ~w missing — skipping BENCH-GO driver check~n', [Facts])
    ;   get_time(T),
        format(atom(TmpDir), 'tmp_wam_go_ed_~w', [T]),
        setup_call_cleanup(
            make_directory_path(TmpDir),
            go_ed_check(TmpDir),
            ( exists_directory(TmpDir)
            ->  delete_directory_and_contents(TmpDir)
            ;   true ))
    ).

go_ed_check(TmpDir) :-
    bench_facts_file(Facts),
    bench_generator(Generator),
    process_create(path(swipl),
                   ['-q', '-s', Generator, '--', Facts, TmpDir,
                    accumulated, kernels_on],
                   [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, Exit),
    assertion(Exit == exit(0)),

    directory_file_path(TmpDir, 'main.go', MainPath),
    assertion(exists_file(MainPath)),
    read_file_to_string(MainPath, Main, []),

    % [factsDir] [reps] argument handling, mirroring the Rust driver so
    % the cross-target harness invokes every target the same way.
    assertion(sub_string(Main, _, _, _, 'reps := 1')),
    assertion(sub_string(Main, _, _, _, 'strconv.Atoi(os.Args[2])')),
    assertion(sub_string(Main, _, _, _, 'strconv.Atoi(os.Args[1])')),

    % Median over reps, not a single run.
    assertion(sub_string(Main, _, _, _, 'func medianMs(samples []int64) int64')),
    assertion(sub_string(Main, _, _, _, 'medianMs(querySamples)')),
    assertion(sub_string(Main, _, _, _, 'medianMs(totalSamples)')),

    % The stderr metric block quoted by the results doc.
    forall(member(Metric, ["mode=", "kernel_mode=", "reps=", "load_ms=",
                           "query_ms=", "aggregation_ms=", "total_ms=",
                           "seed_count=", "tuple_count=", "article_count="]),
           assertion(sub_string(Main, _, _, _, Metric))),

    % The facts are compiled in, which is why load_ms is 0 in the doc.
    assertion(sub_string(Main, _, _, _, 'benchmarkArticleCategories')),
    assertion(sub_string(Main, _, _, _, 'benchmarkRoots')),

    go_ed_build(TmpDir).

go_ed_build(TmpDir) :-
    (   catch(process_create(path(go), ['version'],
                             [stdout(null), stderr(null)]), _, fail)
    ->  format(string(BuildCmd), "cd ~w && go build ./... 2>&1", [TmpDir]),
        process_create(path(sh), ['-c', BuildCmd],
                       [stdout(pipe(Out)), process(Pid)]),
        read_string(Out, _, BuildOutput),
        process_wait(Pid, Exit),
        (   Exit == exit(0)
        ->  true
        ;   format('~nGo build output:~n~s~n', [BuildOutput])
        ),
        assertion(Exit == exit(0))
    ;   format('~n[skip] go not on PATH — skipping BENCH-GO build check~n')
    ).

:- end_tests(wam_go_effective_distance_bench).
