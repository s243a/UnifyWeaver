% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_lmdb_bench.pl - F# LMDB materialisation mode benchmark
%
% Generates a synthetic Phase 1 LMDB fixture, builds an F# project that
% loads it in three modes (eager, lazy, cached), runs a simple ancestor
% traversal kernel from multiple seeds, and reports load_ms + query_ms +
% total_ms for each mode. Results are directly comparable to the Rust/
% Haskell numbers in docs/design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md.
%
% Prerequisites: python3 + lmdb package, .NET 8 SDK, LANG=C.UTF-8

:- encoding(utf8).
:- use_module('../../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).
:- use_module(library(process)).

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
    LmdbDir = '/tmp/uw_fsharp_lmdb_bench_fixture',
    ProjectDir = '/tmp/uw_fsharp_lmdb_bench',
    catch(delete_directory_and_contents(LmdbDir), _, true),
    catch(delete_directory_and_contents(ProjectDir), _, true),

    %% Generate synthetic fixture: 1000 parents, 8 children each = 8000 edges
    format('Generating LMDB fixture (1000 parents x 8 children = 8000 edges)...~n'),
    run_cmd(python3,
            ['examples/benchmark/generate_synthetic_phase1_lmdb.py',
             LmdbDir, '--parents', '1000', '--children-per-parent', '8', '--refresh'],
            '.', PyExit, _),
    (PyExit == 0 -> true ; format('LMDB gen failed~n'), halt(1)),

    %% Generate F# project with LMDB
    make_directory_path(ProjectDir),
    write_wam_fsharp_project([], [no_kernels(true), module_name('uw_fs_lmdb_bench'), lmdb_path(LmdbDir)], ProjectDir),

    %% Write benchmark driver
    atom_string(LmdbDir, LmdbDirStr),
    string_concat(LmdbDirStr, "\"
    let env = openEnv lmdbPath

    // Seeds: all 8000 children (1001..9000)
    let seeds = [| 1001 .. 9000 |]

    // --- Eager mode ---
    let swEager = Stopwatch.StartNew()
    let eagerMap = loadCategoryParent env
    let eagerLoadMs = swEager.Elapsed.TotalMilliseconds
    let eagerSrc = WamTypes.EagerLookupSource(eagerMap) :> WamTypes.ILookupSource

    let swEagerQ = Stopwatch.StartNew()
    let mutable eagerTotal = 0
    for seed in seeds do
        eagerTotal <- eagerTotal + List.length (eagerSrc.Lookup(seed))
    swEagerQ.Stop()
    let eagerQueryMs = swEagerQ.Elapsed.TotalMilliseconds
    swEager.Stop()

    // --- Lazy mode ---
    let swLazy = Stopwatch.StartNew()
    let lazySrc = LmdbCursorLookup(env, \"category_parent\") :> WamTypes.ILookupSource
    let lazyLoadMs = swLazy.Elapsed.TotalMilliseconds

    let swLazyQ = Stopwatch.StartNew()
    let mutable lazyTotal = 0
    for seed in seeds do
        lazyTotal <- lazyTotal + List.length (lazySrc.Lookup(seed))
    swLazyQ.Stop()
    let lazyQueryMs = swLazyQ.Elapsed.TotalMilliseconds
    swLazy.Stop()

    // --- Cached mode ---
    let swCached = Stopwatch.StartNew()
    let cachedSrc = CachedLookupSource(LmdbCursorLookup(env, \"category_parent\")) :> WamTypes.ILookupSource
    let cachedLoadMs = swCached.Elapsed.TotalMilliseconds

    let swCachedQ = Stopwatch.StartNew()
    let mutable cachedTotal = 0
    for seed in seeds do
        cachedTotal <- cachedTotal + List.length (cachedSrc.Lookup(seed))
    swCachedQ.Stop()
    let cachedQueryMs = swCachedQ.Elapsed.TotalMilliseconds
    swCached.Stop()

    // Second pass on cached to measure cache-hit performance
    let swCachedHit = Stopwatch.StartNew()
    let mutable cachedHitTotal = 0
    for seed in seeds do
        cachedHitTotal <- cachedHitTotal + List.length (cachedSrc.Lookup(seed))
    swCachedHit.Stop()
    let cachedHitMs = swCachedHit.Elapsed.TotalMilliseconds

    env.Dispose()
    swTotal.Stop()

    printfn \"\"
    printfn \"F# LMDB Benchmark (1000 parents x 8 children = 8000 edges, %d seeds)\" seeds.Length
    printfn \"========================================================================\"
    printfn \"Mode       load_ms   query_ms   total_ms   result\"
    printfn \"eager      %7.2f   %8.2f   %8.2f   %d\" eagerLoadMs eagerQueryMs (eagerLoadMs + eagerQueryMs) eagerTotal
    printfn \"lazy       %7.2f   %8.2f   %8.2f   %d\" lazyLoadMs lazyQueryMs (lazyLoadMs + lazyQueryMs) lazyTotal
    printfn \"cached     %7.2f   %8.2f   %8.2f   %d\" cachedLoadMs cachedQueryMs (cachedLoadMs + cachedQueryMs) cachedTotal
    printfn \"cached-hit %7.2f   %8.2f   %8.2f   %d\" 0.0 cachedHitMs cachedHitMs cachedHitTotal
    printfn \"\"
    printfn \"total_wall_ms = %.2f\" swTotal.Elapsed.TotalMilliseconds

    // Sanity: all modes should produce same result
    if eagerTotal = lazyTotal && lazyTotal = cachedTotal && cachedTotal = cachedHitTotal then
        printfn \"RESULT OK (all modes agree)\"
        0
    else
        printfn \"RESULT MISMATCH\"
        1
", DriverTail),
    string_concat("module Program

open System.Diagnostics
open LmdbFactSource
open WamTypes

[<EntryPoint>]
let main _argv =
    let swTotal = Stopwatch.StartNew()
    let lmdbPath = \"", DriverTail, DriverCode),

    directory_file_path(ProjectDir, 'Program.fs', ProgPath),
    open(ProgPath, write, OW, [encoding(utf8)]),
    write(OW, DriverCode),
    close(OW),

    %% Build (Release for realistic perf)
    format('Building (Release)...~n'),
    run_cmd(dotnet, ['build', '--nologo', '-v', 'minimal', '-c', 'Release'], ProjectDir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('Build OK.~n')
    ;   format('--- build output ---~n~w~n----~nBUILD FAILED~n', [BuildOut]),
        halt(1)
    ),

    %% Run
    format('Running benchmark...~n'),
    run_cmd(dotnet, ['run', '--no-build', '-c', 'Release'], ProjectDir, RunExit, RunOut),
    format('~w~n', [RunOut]),
    halt(RunExit).
