% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_csr_bench.pl - CSR vs LMDB reverse lookup benchmark
%
% Generates a fixture, builds both CSR and LMDB artifacts, then
% benchmarks CSR (raw + cached) against LMDB cursor for reverse
% child lookup. Also validates correctness by comparing results.
%
% Prerequisites: python3 + lmdb, .NET 8 SDK

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
    LmdbDir = '/tmp/uw_fsharp_csr_bench_lmdb',
    CsrDir  = '/tmp/uw_fsharp_csr_bench_csr',
    ProjectDir = '/tmp/uw_fsharp_csr_bench_proj',
    catch(delete_directory_and_contents(LmdbDir), _, true),
    catch(delete_directory_and_contents(CsrDir), _, true),
    catch(delete_directory_and_contents(ProjectDir), _, true),

    %% Generate fixture: 500 parents x 6 children = 3000 edges
    format('Generating fixture (500 parents x 6 children = 3000 edges)...~n'),
    run_cmd(python3,
            ['examples/benchmark/generate_synthetic_phase1_lmdb.py',
             LmdbDir, '--parents', '500', '--children-per-parent', '6', '--refresh'],
            '.', PyExit, _),
    (PyExit == 0 -> true ; format('LMDB gen failed~n'), halt(1)),

    %% Build CSR
    format('Building CSR artifact...~n'),
    run_cmd(python3,
            ['examples/benchmark/build_reverse_csr_artifact.py',
             LmdbDir, CsrDir, '--refresh'],
            '.', CsrExit, CsrOut),
    (CsrExit == 0 -> true ; format('CSR build failed: ~w~n', [CsrOut]), halt(1)),

    %% Generate F# project with both LMDB and CSR
    make_directory_path(ProjectDir),
    write_wam_fsharp_project([], [
        no_kernels(true),
        module_name('uw_fs_csr_bench'),
        lmdb_path(LmdbDir),
        csr_path(CsrDir)
    ], ProjectDir),

    atom_string(LmdbDir, LmdbDirStr),
    atom_string(CsrDir, CsrDirStr),

    DriverHead = "module Program

open System
open System.Diagnostics
open LmdbFactSource
open CsrReader
open WamTypes

[<EntryPoint>]
let main _argv =
    let lmdbPath = \"",
    DriverMid1 = "\"
    let csrPath = \"",
    DriverTail = "\"
    let env = openEnv lmdbPath
    use csr = openCsr csrPath
    let nRounds = 5
    let allParents = csr.Parents()
    printfn \"Parents: %d  Edges: %d\" csr.ParentCount csr.EdgeCount

    // Mode 1: CSR raw (no cache)
    let csrSrc = csr :> ILookupSource
    let csrTimings = Array.zeroCreate nRounds
    let mutable csrHits = 0
    for r in 0 .. nRounds - 1 do
        let sw = Stopwatch.StartNew()
        let mutable hits = 0
        for parent in allParents do
            hits <- hits + (csrSrc.Lookup parent).Length
        sw.Stop()
        csrTimings.[r] <- sw.Elapsed.TotalMilliseconds
        csrHits <- hits
    Array.sortInPlace csrTimings

    // Mode 2: CSR + TwoLevelCachedLookupSource
    let cachedCsr = TwoLevelCachedLookupSource(csrSrc, l2CapacitySpec = \"dev\") :> ILookupSource
    // Warmup
    for parent in allParents do cachedCsr.Lookup parent |> ignore
    let cachedTimings = Array.zeroCreate nRounds
    let mutable cachedHits = 0
    for r in 0 .. nRounds - 1 do
        let sw = Stopwatch.StartNew()
        let mutable hits = 0
        for parent in allParents do
            hits <- hits + (cachedCsr.Lookup parent).Length
        sw.Stop()
        cachedTimings.[r] <- sw.Elapsed.TotalMilliseconds
        cachedHits <- hits
    Array.sortInPlace cachedTimings

    // Mode 3: LMDB cursor (category_child DUPSORT)
    let lmdbSrc = LmdbCursorLookup(env, \"category_child\") :> ILookupSource
    let lmdbTimings = Array.zeroCreate nRounds
    let mutable lmdbHits = 0
    for r in 0 .. nRounds - 1 do
        let sw = Stopwatch.StartNew()
        let mutable hits = 0
        for parent in allParents do
            hits <- hits + (lmdbSrc.Lookup parent).Length
        sw.Stop()
        lmdbTimings.[r] <- sw.Elapsed.TotalMilliseconds
        lmdbHits <- hits
    Array.sortInPlace lmdbTimings

    env.Dispose()

    let median (a: float array) = a.[a.Length / 2]
    printfn \"\"
    printfn \"CSR vs LMDB Reverse Lookup Benchmark\"
    printfn \"=====================================\"
    printfn \"Mode            median_ms   hits\"
    printfn \"CSR raw         %8.2f    %d\" (median csrTimings) csrHits
    printfn \"CSR cached      %8.2f    %d\" (median cachedTimings) cachedHits
    printfn \"LMDB cursor     %8.2f    %d\" (median lmdbTimings) lmdbHits
    printfn \"Rounds: CSR=%A  Cached=%A  LMDB=%A\" csrTimings cachedTimings lmdbTimings

    if csrHits = cachedHits && cachedHits = lmdbHits then
        printfn \"RESULT OK (all modes agree on %d children)\" csrHits
        0
    else
        printfn \"RESULT MISMATCH (csr=%d cached=%d lmdb=%d)\" csrHits cachedHits lmdbHits
        1
",

    string_concat(DriverHead, LmdbDirStr, P1),
    string_concat(P1, DriverMid1, P2),
    string_concat(P2, CsrDirStr, P3),
    string_concat(P3, DriverTail, DriverFull),

    directory_file_path(ProjectDir, 'Program.fs', ProgPath),
    open(ProgPath, write, OW, [encoding(utf8)]),
    write(OW, DriverFull),
    close(OW),

    %% Build
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
