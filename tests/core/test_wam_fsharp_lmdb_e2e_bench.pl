% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_lmdb_e2e_bench.pl - Full WAM+LMDB E2E benchmark
%
% Runs a depth-bounded DFS (category_ancestor pattern) against LMDB-
% backed facts and compares query_ms to in-memory Map baseline.
% This confirms whether the published 11 ms F# query number holds
% when facts come from LMDB instead of pre-loaded TSV.
%
% Fixture: 750 parents × 8 children = 6000 edges (matching scale 300)
% Workload: DFS from 100 seeds, max_depth=10
%
% Prerequisites: python3 + lmdb, .NET 8 SDK, LANG=C.UTF-8

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
    LmdbDir = '/tmp/uw_fsharp_lmdb_e2e_fixture',
    ProjectDir = '/tmp/uw_fsharp_lmdb_e2e_bench',
    catch(delete_directory_and_contents(LmdbDir), _, true),
    catch(delete_directory_and_contents(ProjectDir), _, true),

    %% Generate fixture: 750 parents × 8 children = 6000 edges (≈ scale 300)
    format('Generating LMDB fixture (750 parents x 8 children = 6000 edges)...~n'),
    run_cmd(python3,
            ['examples/benchmark/generate_synthetic_phase1_lmdb.py',
             LmdbDir, '--parents', '750', '--children-per-parent', '8', '--refresh'],
            '.', PyExit, _),
    (PyExit == 0 -> true ; format('LMDB gen failed~n'), halt(1)),

    %% Generate F# project with LMDB
    make_directory_path(ProjectDir),
    write_wam_fsharp_project([], [no_kernels(true), module_name('uw_fs_lmdb_e2e'), lmdb_path(LmdbDir)], ProjectDir),

    %% Write E2E benchmark driver
    atom_string(LmdbDir, LmdbDirStr),

    DriverHead = "module Program

open System
open System.Diagnostics
open LmdbFactSource
open WamTypes

/// Depth-bounded BFS counting reachable ancestors within max_depth.
/// Each lookup call goes through the ILookupSource interface —
/// this is the hot path we're benchmarking.
let bfsReachable (lookupParents: int -> int list) (start: int) (maxDepth: int) : int =
    let mutable visited = Set.singleton start
    let mutable frontier = [start]
    let mutable depth = 0
    while depth < maxDepth && not (List.isEmpty frontier) do
        let mutable nextFrontier = []
        for node in frontier do
            let parents = lookupParents node
            for p in parents do
                if not (Set.contains p visited) then
                    visited <- Set.add p visited
                    nextFrontier <- p :: nextFrontier
        frontier <- nextFrontier
        depth <- depth + 1
    Set.count visited - 1  // exclude start node

[<EntryPoint>]
let main _argv =
    let lmdbPath = \"",

    DriverTail = "\"
    let env = openEnv lmdbPath
    let maxDepth = 10
    let seeds = [| 751 .. 6750 |]  // all 6000 children
    let nWarmup = 5
    let nRounds = 3

    let mapData = loadCategoryParent env
    let mapLookup (key: int) = Map.tryFind key mapData |> Option.defaultValue []

    for _ in 1 .. nWarmup do
        for seed in seeds do bfsReachable mapLookup seed maxDepth |> ignore
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect()

    let mapTimings = Array.zeroCreate nRounds
    let mutable mapHits = 0
    for r in 0 .. nRounds - 1 do
        let sw = Stopwatch.StartNew()
        for seed in seeds do mapHits <- mapHits + bfsReachable mapLookup seed maxDepth
        sw.Stop()
        mapTimings.[r] <- sw.Elapsed.TotalMilliseconds
    Array.sortInPlace mapTimings
    let mapMedian = mapTimings.[nRounds / 2]

    let cachedSrc = TwoLevelCachedLookupSource(LmdbCursorLookup(env, \"category_parent\")) :> ILookupSource
    let cachedLookup = cachedSrc.Lookup
    for _ in 1 .. nWarmup do
        for seed in seeds do bfsReachable cachedLookup seed maxDepth |> ignore
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect()

    let cachedTimings = Array.zeroCreate nRounds
    let mutable cachedHits = 0
    for r in 0 .. nRounds - 1 do
        let sw = Stopwatch.StartNew()
        for seed in seeds do cachedHits <- cachedHits + bfsReachable cachedLookup seed maxDepth
        sw.Stop()
        cachedTimings.[r] <- sw.Elapsed.TotalMilliseconds
    Array.sortInPlace cachedTimings
    let cachedMedian = cachedTimings.[nRounds / 2]

    let lazySrc = LmdbCursorLookup(env, \"category_parent\") :> ILookupSource
    let lazyLookup = lazySrc.Lookup
    let lazyTimings = Array.zeroCreate nRounds
    let mutable lazyHits = 0
    for r in 0 .. nRounds - 1 do
        let sw = Stopwatch.StartNew()
        for seed in seeds do lazyHits <- lazyHits + bfsReachable lazyLookup seed maxDepth
        sw.Stop()
        lazyTimings.[r] <- sw.Elapsed.TotalMilliseconds
    Array.sortInPlace lazyTimings
    let lazyMedian = lazyTimings.[nRounds / 2]

    env.Dispose()
    printfn \"\"
    printfn \"F# WAM+LMDB E2E Benchmark (6000 edges, 100 seeds, max_depth=10)\"
    printfn \"================================================================\"
    printfn \"Mode            query_ms (median of %d)   hits\" nRounds
    printfn \"Map (baseline)  %8.2f                  %d\" mapMedian mapHits
    printfn \"LMDB cached     %8.2f                  %d\" cachedMedian cachedHits
    printfn \"LMDB lazy       %8.2f                  %d\" lazyMedian lazyHits
    printfn \"Rounds: Map=%A  Cached=%A  Lazy=%A\" mapTimings cachedTimings lazyTimings
    if mapHits = cachedHits && cachedHits = lazyHits then
        printfn \"RESULT OK (all modes agree on %d hits)\" (mapHits / nRounds)
        0
    else
        printfn \"RESULT MISMATCH\"
        1
",

    string_concat(DriverHead, LmdbDirStr, P1),
    string_concat(P1, DriverTail, DriverFull),

    directory_file_path(ProjectDir, 'Program.fs', ProgPath),
    open(ProgPath, write, OW, [encoding(utf8)]),
    write(OW, DriverFull),
    close(OW),

    %% Build (Release)
    format('Building (Release)...~n'),
    run_cmd(dotnet, ['build', '--nologo', '-v', 'minimal', '-c', 'Release'], ProjectDir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('Build OK.~n')
    ;   format('BUILD FAILED:~n~w~n', [BuildOut]), halt(1)
    ),

    %% Run
    format('Running benchmark...~n~n'),
    run_cmd(dotnet, ['run', '--no-build', '-c', 'Release'], ProjectDir, RunExit, RunOut),
    format('~w~n', [RunOut]),
    halt(RunExit).
