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

    // --- Dict mode (skip Map, O(1) Dictionary) ---
    let swDict = Stopwatch.StartNew()
    let dictData = loadDupsortRelationDict env \"category_parent\"
    let dictLoadMs = swDict.Elapsed.TotalMilliseconds
    let dictSrc = WamTypes.DictLookupSource(dictData) :> WamTypes.ILookupSource

    let swDictQ = Stopwatch.StartNew()
    let mutable dictTotal = 0
    for seed in seeds do
        dictTotal <- dictTotal + List.length (dictSrc.Lookup(seed))
    swDictQ.Stop()
    let dictQueryMs = swDictQ.Elapsed.TotalMilliseconds

    // --- Eager mode (Map) ---
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

    // --- Two-level cache (L1 per-thread + L2 bounded shared) ---
    let swTwoLevel = Stopwatch.StartNew()
    let twoLevelSrc = TwoLevelCachedLookupSource(
                          LmdbCursorLookup(env, \"category_parent\"),
                          l1Capacity = 4096,
                          maxL2Entries = 65536) :> WamTypes.ILookupSource
    let twoLevelLoadMs = swTwoLevel.Elapsed.TotalMilliseconds

    let swTwoLevelQ = Stopwatch.StartNew()
    let mutable twoLevelTotal = 0
    for seed in seeds do
        twoLevelTotal <- twoLevelTotal + List.length (twoLevelSrc.Lookup(seed))
    swTwoLevelQ.Stop()
    let twoLevelQueryMs = swTwoLevelQ.Elapsed.TotalMilliseconds

    // Second pass: warm L1+L2
    let swTwoLevelHit = Stopwatch.StartNew()
    let mutable twoLevelHitTotal = 0
    for seed in seeds do
        twoLevelHitTotal <- twoLevelHitTotal + List.length (twoLevelSrc.Lookup(seed))
    swTwoLevelHit.Stop()
    let twoLevelHitMs = swTwoLevelHit.Elapsed.TotalMilliseconds

    env.Dispose()
    swTotal.Stop()

    printfn \"\"
    printfn \"F# LMDB Benchmark (1000 parents x 8 children = 8000 edges, %d seeds)\" seeds.Length
    printfn \"========================================================================\"
    printfn \"Mode       load_ms   query_ms   total_ms   result\"
    printfn \"dict       %7.2f   %8.2f   %8.2f   %d\" dictLoadMs dictQueryMs (dictLoadMs + dictQueryMs) dictTotal
    printfn \"eager      %7.2f   %8.2f   %8.2f   %d\" eagerLoadMs eagerQueryMs (eagerLoadMs + eagerQueryMs) eagerTotal
    printfn \"lazy       %7.2f   %8.2f   %8.2f   %d\" lazyLoadMs lazyQueryMs (lazyLoadMs + lazyQueryMs) lazyTotal
    printfn \"cached     %7.2f   %8.2f   %8.2f   %d\" cachedLoadMs cachedQueryMs (cachedLoadMs + cachedQueryMs) cachedTotal
    printfn \"cached-hit %7.2f   %8.2f   %8.2f   %d\" 0.0 cachedHitMs cachedHitMs cachedHitTotal
    printfn \"2level     %7.2f   %8.2f   %8.2f   %d\" twoLevelLoadMs twoLevelQueryMs (twoLevelLoadMs + twoLevelQueryMs) twoLevelTotal
    printfn \"2level-hit %7.2f   %8.2f   %8.2f   %d\" 0.0 twoLevelHitMs twoLevelHitMs twoLevelHitTotal
    printfn \"\"
    printfn \"total_wall_ms = %.2f\" swTotal.Elapsed.TotalMilliseconds

    // Sanity: all modes should produce same result
    if dictTotal = eagerTotal && eagerTotal = lazyTotal && lazyTotal = cachedTotal && cachedTotal = cachedHitTotal && cachedHitTotal = twoLevelTotal && twoLevelTotal = twoLevelHitTotal then
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
