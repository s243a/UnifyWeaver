% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_parallel_seeds.pl - Parallel seed execution benchmark
%
% Benchmarks sequential vs parallel seed dispatch using a BFS kernel
% on a deep synthetic tree (depth=15, branching=3). This creates heavy
% per-seed work that benefits from multi-core execution.
%
% The TwoLevelCachedLookupSource is verified for thread-safety: L1 is
% ThreadLocal (per-thread), L2 is ConcurrentDictionary (lock-free).
%
% Prerequisites: .NET 8 SDK

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
    ProjectDir = '/tmp/uw_fsharp_parallel_proj',
    LmdbDir = '/tmp/uw_fsharp_parallel_lmdb',
    catch(delete_directory_and_contents(ProjectDir), _, true),
    catch(delete_directory_and_contents(LmdbDir), _, true),

    %% Create a minimal LMDB fixture so LmdbFactSource.fs is included
    %% (we need TwoLevelCachedLookupSource for the parallel test)
    run_cmd(python3,
            ['examples/benchmark/generate_synthetic_phase1_lmdb.py',
             LmdbDir, '--parents', '10', '--children-per-parent', '2', '--refresh'],
            '.', PyExit, _),
    (PyExit == 0 -> true ; format('LMDB gen failed~n'), halt(1)),

    make_directory_path(ProjectDir),
    write_wam_fsharp_project([], [no_kernels(true), module_name('uw_fs_parallel'), lmdb_path(LmdbDir)], ProjectDir),

    DriverCode = "module Program

open System
open System.Diagnostics
open System.Threading.Tasks
open WamTypes

/// Dictionary-backed lookup source for in-memory graph.
type DictGraphSource(graph: Collections.Generic.Dictionary<int, int list>) =
    interface ILookupSource with
        member _.Lookup(key) =
            let ok, vs = graph.TryGetValue(key)
            if ok then vs else []

/// Build a multi-level tree graph where each node at level L has
/// `branching` children at level L+1. Returns (graph, leaves).
/// This creates deep per-seed work: BFS from a leaf traverses
/// all levels up to root.
let buildTree (depth: int) (branching: int) =
    let graph = Collections.Generic.Dictionary<int, int list>()
    let mutable nextId = 1
    let root = 0
    // parentEdges: child -> parent (for upward BFS)
    let parentEdges = Collections.Generic.Dictionary<int, int list>()
    let rec build parentId level =
        if level >= depth then []
        else
            let children = [
                for _ in 1 .. branching do
                    let childId = nextId
                    nextId <- nextId + 1
                    // child -> [parent] (upward edge)
                    parentEdges.[childId] <- [parentId]
                    yield childId
            ]
            // parent -> children (downward edge)
            graph.[parentId] <- children
            // Recurse for each child
            let leaves = children |> List.collect (fun c -> build c (level + 1))
            if level = depth - 1 then children
            else leaves
    let leaves = build root 1
    parentEdges, leaves, nextId

/// BFS counting reachable ancestors (upward traversal).
let bfsAncestors (lookup: int -> int list) (start: int) (maxDepth: int) : int =
    let mutable visited = Set.singleton start
    let mutable frontier = [start]
    let mutable depth = 0
    while depth < maxDepth && not (List.isEmpty frontier) do
        let mutable nextFrontier = []
        for node in frontier do
            let parents = lookup node
            for p in parents do
                if not (Set.contains p visited) then
                    visited <- Set.add p visited
                    nextFrontier <- p :: nextFrontier
        frontier <- nextFrontier
        depth <- depth + 1
    Set.count visited - 1

[<EntryPoint>]
let main _argv =
    let treeDepth = 12
    let branching = 3
    let maxBfsDepth = 15

    // Build tree
    let parentEdges, leaves, totalNodes = buildTree treeDepth branching
    let seeds = leaves |> List.toArray
    let nSeeds = min 2000 seeds.Length  // cap seeds for reasonable runtime
    let seeds = seeds.[..nSeeds-1]

    // Wrap in TwoLevelCachedLookupSource for thread-safe parallel access
    let innerSrc = DictGraphSource(parentEdges) :> ILookupSource
    let cachedSrc = LmdbFactSource.TwoLevelCachedLookupSource(innerSrc, l2CapacitySpec = \"medium\") :> ILookupSource
    let lookup = cachedSrc.Lookup

    let cores = Environment.ProcessorCount
    printfn \"Tree: depth=%d branching=%d nodes=%d\" treeDepth branching totalNodes
    printfn \"Seeds: %d (leaves)  MaxBfsDepth: %d  Cores: %d\" nSeeds maxBfsDepth cores

    // Warmup
    for i in 0 .. min 100 (nSeeds-1) do
        bfsAncestors lookup seeds.[i] maxBfsDepth |> ignore

    let nRounds = 7

    // Sequential
    let seqTimings = Array.zeroCreate nRounds
    let mutable seqTotal = 0
    for r in 0 .. nRounds - 1 do
        let sw = Stopwatch.StartNew()
        let mutable hits = 0
        for seed in seeds do
            hits <- hits + bfsAncestors lookup seed maxBfsDepth
        sw.Stop()
        seqTimings.[r] <- sw.Elapsed.TotalMilliseconds
        seqTotal <- hits
    Array.sortInPlace seqTimings

    // Parallel (Array.Parallel.map)
    let parTimings = Array.zeroCreate nRounds
    let mutable parTotal = 0
    for r in 0 .. nRounds - 1 do
        let sw = Stopwatch.StartNew()
        let results = seeds |> Array.Parallel.map (fun seed -> bfsAncestors lookup seed maxBfsDepth)
        sw.Stop()
        parTimings.[r] <- sw.Elapsed.TotalMilliseconds
        parTotal <- Array.sum results
    Array.sortInPlace parTimings

    // Parallel.For with explicit parallelism
    let parOpts = ParallelOptions(MaxDegreeOfParallelism = cores)
    let par2Timings = Array.zeroCreate nRounds
    let mutable par2Total = 0
    for r in 0 .. nRounds - 1 do
        let results = Array.zeroCreate seeds.Length
        let sw = Stopwatch.StartNew()
        Parallel.For(0, seeds.Length, parOpts, fun i ->
            results.[i] <- bfsAncestors lookup seeds.[i] maxBfsDepth
        ) |> ignore
        sw.Stop()
        par2Timings.[r] <- sw.Elapsed.TotalMilliseconds
        par2Total <- Array.sum results
    Array.sortInPlace par2Timings

    let median (a: float array) = a.[a.Length / 2]
    let seqMs = median seqTimings
    let parMs = median parTimings
    let par2Ms = median par2Timings
    let speedup1 = seqMs / parMs
    let speedup2 = seqMs / par2Ms

    printfn \"\"
    printfn \"F# Parallel Seed Execution Benchmark\"
    printfn \"=====================================\"
    printfn \"Mode                 median_ms  speedup  hits\"
    printfn \"Sequential           %8.2f    1.00x   %d\" seqMs seqTotal
    printfn \"Array.Parallel.map   %8.2f    %.2fx   %d\" parMs speedup1 parTotal
    printfn \"Parallel.For(N=%d)   %8.2f    %.2fx   %d\" cores par2Ms speedup2 par2Total
    printfn \"Rounds: Seq=%A\" seqTimings
    printfn \"        Par=%A\" parTimings
    printfn \"        Par2=%A\" par2Timings

    if seqTotal = parTotal && parTotal = par2Total then
        printfn \"RESULT OK (all modes agree on %d hits, speedup=%.2fx)\" seqTotal (max speedup1 speedup2)
        0
    else
        printfn \"RESULT MISMATCH (seq=%d par=%d par2=%d)\" seqTotal parTotal par2Total
        1
",

    directory_file_path(ProjectDir, 'Program.fs', ProgPath),
    open(ProgPath, write, OW, [encoding(utf8)]),
    write(OW, DriverCode),
    close(OW),

    %% Build
    format('Building (Release)...~n'),
    run_cmd(dotnet, ['build', '--nologo', '-v', 'minimal', '-c', 'Release'], ProjectDir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('Build OK.~n')
    ;   format('BUILD FAILED:~n~w~n', [BuildOut]), halt(1)
    ),

    %% Run
    format('Running parallel benchmark...~n~n'),
    run_cmd(dotnet, ['run', '--no-build', '-c', 'Release'], ProjectDir, RunExit, RunOut),
    format('~w~n', [RunOut]),
    halt(RunExit).
