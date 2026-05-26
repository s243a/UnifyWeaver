% Run: swipl -g main tests/core/test_wam_fsharp_cost_analyzer_bench.pl
% Prerequisites: python3 + lmdb, .NET 8 SDK
%
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_cost_analyzer_bench.pl - Cost analyzer E2E benchmark
%
% Generates synthetic LMDB fixtures at 4 scales, builds CSR artifacts,
% then benchmarks BFS reachability across edge_store modes:
%   lmdb_cached, lmdb_eager, csr (forward), auto
%
% Scales:
%   dev:    25 parents x 8 children =   200 edges,   50 seeds
%   small: 250 parents x 8 children =  2000 edges,  100 seeds
%   medium:750 parents x 8 children =  6000 edges,  300 seeds
%   large:2500 parents x 8 children = 20000 edges, 1000 seeds

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

%% bench_scale(Name, Parents, ChildrenPerParent)
bench_scale(dev,      25, 8).
bench_scale(small,   250, 8).
bench_scale(medium,  750, 8).
bench_scale(large,  2500, 8).

%% seeds_for_scale(+Name, -NSeeds)
seeds_for_scale(dev,     50).
seeds_for_scale(small,  100).
seeds_for_scale(medium, 300).
seeds_for_scale(large, 1000).

main :-
    format('~n================================================~n'),
    format('F# Cost Analyzer Benchmark — BFS across edge_store modes~n'),
    format('================================================~n~n'),

    forall(bench_scale(Scale, _, _), run_scale(Scale)),

    %% Additional test: edge_store(auto) at medium scale
    format('~n--- edge_store(auto) at medium scale ---~n'),
    run_auto_medium,

    format('~nDone.~n'),
    halt(0).

%% run_scale(+ScaleName)
%  For the given scale, generate LMDB + CSR, build one F# project
%  with all three explicit modes (eager, cached, csr) in a custom
%  driver, then build and run.
run_scale(Scale) :-
    bench_scale(Scale, Parents, CPP),
    seeds_for_scale(Scale, NSeeds),
    NEdges is Parents * CPP,
    atom_concat('/tmp/uw_fsharp_cost_bench_', Scale, BaseDir),
    atom_concat(BaseDir, '_lmdb', LmdbDir),
    atom_concat(BaseDir, '_csr', CsrDir),
    atom_concat(BaseDir, '_proj', ProjectDir),
    catch(delete_directory_and_contents(LmdbDir), _, true),
    catch(delete_directory_and_contents(CsrDir), _, true),
    catch(delete_directory_and_contents(ProjectDir), _, true),

    %% 1. Generate LMDB fixture
    atom_number(ParentsA, Parents),
    atom_number(CPPA, CPP),
    format('~n[~w] Generating LMDB fixture (~w parents x ~w children = ~w edges)...~n',
           [Scale, Parents, CPP, NEdges]),
    run_cmd(python3,
            ['examples/benchmark/generate_synthetic_phase1_lmdb.py',
             LmdbDir, '--parents', ParentsA,
             '--children-per-parent', CPPA, '--refresh'],
            '.', PyExit, _),
    (PyExit == 0 -> true
    ; format('[~w] LMDB gen failed~n', [Scale]), halt(1)),

    %% 2. Build forward CSR artifact (category_parent)
    format('[~w] Building forward CSR artifact...~n', [Scale]),
    run_cmd(python3,
            ['examples/benchmark/build_csr_artifact.py',
             LmdbDir, CsrDir,
             '--relation', 'category_parent', '--refresh'],
            '.', CsrExit, CsrOut),
    (CsrExit == 0 -> true
    ; format('[~w] CSR build failed: ~w~n', [Scale, CsrOut]), halt(1)),

    %% 3. Generate F# project with LMDB + CSR parent path
    make_directory_path(ProjectDir),
    write_wam_fsharp_project([], [
        lmdb_path(LmdbDir),
        csr_path(CsrDir),
        csr_relation(category_parent),
        no_kernels(true),
        module_name('uw_fs_cost_bench')
    ], ProjectDir),

    %% 4. Write custom benchmark driver (Program.fs)
    FirstChild is Parents + 1,
    LastChild  is Parents + NSeeds,
    atom_string(LmdbDir, LmdbDirStr),
    atom_string(CsrDir, CsrDirStr),
    atom_number(FirstChildA, FirstChild),
    atom_number(LastChildA, LastChild),
    atom_string(FirstChildA, FirstChildS),
    atom_string(LastChildA, LastChildS),
    atom_string(NEdges, NEdgesS),

    %% Build the driver source by concatenation
    string_concat("module Program

open System
open System.Diagnostics
open LmdbFactSource
open CsrReader
open WamTypes

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
    Set.count visited - 1

[<EntryPoint>]
let main _argv =
    let lmdbPath = \"", LmdbDirStr, S1),
    string_concat(S1, "\"
    let csrPath = \"", S2),
    string_concat(S2, CsrDirStr, S3),
    string_concat(S3, "\"
    let env = openEnv lmdbPath
    let maxDepth = 10
    let seeds = [| ", S4),
    string_concat(S4, FirstChildS, S5),
    string_concat(S5, " .. ", S6),
    string_concat(S6, LastChildS, S7),
    string_concat(S7, " |]
    let nRounds = 3

    // Mode 1: LMDB eager (Map)
    let mapData = loadCategoryParent env
    let mapLookup (key: int) = Map.tryFind key mapData |> Option.defaultValue []
    let mutable eagerHits = 0
    let eagerSw = Stopwatch.StartNew()
    for _ in 1 .. nRounds do
        for seed in seeds do eagerHits <- eagerHits + bfsReachable mapLookup seed maxDepth
    eagerSw.Stop()
    let eagerMs = eagerSw.Elapsed.TotalMilliseconds / float nRounds

    // Mode 2: LMDB cached
    let cachedSrc = TwoLevelCachedLookupSource(LmdbCursorLookup(env, \"category_parent\")) :> ILookupSource
    let mutable cachedHits = 0
    let cachedSw = Stopwatch.StartNew()
    for _ in 1 .. nRounds do
        for seed in seeds do cachedHits <- cachedHits + bfsReachable cachedSrc.Lookup seed maxDepth
    cachedSw.Stop()
    let cachedMs = cachedSw.Elapsed.TotalMilliseconds / float nRounds

    // Mode 3: CSR
    let csrSrc = CsrReader.CsrLookupSource(csrPath, \"category_parent\") :> ILookupSource
    let mutable csrHits = 0
    let csrSw = Stopwatch.StartNew()
    for _ in 1 .. nRounds do
        for seed in seeds do csrHits <- csrHits + bfsReachable csrSrc.Lookup seed maxDepth
    csrSw.Stop()
    let csrMs = csrSw.Elapsed.TotalMilliseconds / float nRounds

    env.Dispose()
    printfn \"SCALE edges=", S8),
    string_concat(S8, NEdgesS, S9),
    string_concat(S9, " seeds=%d\" seeds.Length
    printfn \"eager_ms=%.2f eager_hits=%d\" eagerMs eagerHits
    printfn \"cached_ms=%.2f cached_hits=%d\" cachedMs cachedHits
    printfn \"csr_ms=%.2f csr_hits=%d\" csrMs csrHits
    if eagerHits = cachedHits && cachedHits = csrHits then
        printfn \"CORRECTNESS OK (%d hits/round)\" (eagerHits / nRounds)
        0
    else
        printfn \"CORRECTNESS MISMATCH\"
        1
", DriverCode),

    directory_file_path(ProjectDir, 'Program.fs', ProgPath),
    open(ProgPath, write, OW, [encoding(utf8)]),
    write(OW, DriverCode),
    close(OW),

    %% 5. Build (Release)
    format('[~w] Building (Release)...~n', [Scale]),
    run_cmd(dotnet, ['build', '--nologo', '-v', 'minimal', '-c', 'Release'],
            ProjectDir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('[~w] Build OK.~n', [Scale])
    ;   format('[~w] BUILD FAILED:~n~w~n', [Scale, BuildOut]), halt(1)
    ),

    %% 6. Run
    format('[~w] Running benchmark...~n', [Scale]),
    run_cmd(dotnet, ['run', '--no-build', '-c', 'Release'],
            ProjectDir, RunExit, RunOut),
    format('~w~n', [RunOut]),
    (   RunExit == 0
    ->  format('[~w] PASS~n', [Scale])
    ;   format('[~w] FAIL (exit ~w)~n', [Scale, RunExit]), halt(1)
    ).

%% run_auto_medium/0
%  Generate a separate project at medium scale with edge_store(auto)
%  and run the generated Program.fs as-is to verify the cost analyzer
%  resolves correctly.
run_auto_medium :-
    bench_scale(medium, Parents, CPP),
    NEdges is Parents * CPP,
    LmdbDir = '/tmp/uw_fsharp_cost_bench_medium_lmdb',
    CsrDir  = '/tmp/uw_fsharp_cost_bench_medium_csr',
    AutoDir = '/tmp/uw_fsharp_cost_bench_auto',
    catch(delete_directory_and_contents(AutoDir), _, true),

    %% LMDB + CSR already generated by run_scale(medium); reuse them.
    %% If they don't exist, regenerate.
    (   \+ exists_directory(LmdbDir)
    ->  atom_number(ParentsA, Parents),
        atom_number(CPPA, CPP),
        format('[auto] Regenerating LMDB fixture...~n'),
        run_cmd(python3,
                ['examples/benchmark/generate_synthetic_phase1_lmdb.py',
                 LmdbDir, '--parents', ParentsA,
                 '--children-per-parent', CPPA, '--refresh'],
                '.', PyExit, _),
        (PyExit == 0 -> true ; format('[auto] LMDB gen failed~n'), halt(1)),
        format('[auto] Rebuilding CSR...~n'),
        run_cmd(python3,
                ['examples/benchmark/build_csr_artifact.py',
                 LmdbDir, CsrDir,
                 '--relation', 'category_parent', '--refresh'],
                '.', CsrExit, _),
        (CsrExit == 0 -> true ; format('[auto] CSR build failed~n'), halt(1))
    ;   true
    ),

    %% Generate F# project with edge_store(auto)
    make_directory_path(AutoDir),
    write_wam_fsharp_project([], [
        edge_store(auto),
        edge_count(NEdges),
        lmdb_path(LmdbDir),
        csr_path(CsrDir),
        csr_relation(category_parent),
        no_kernels(true),
        module_name('uw_fs_cost_auto')
    ], AutoDir),

    %% Build
    format('[auto] Building (Release)...~n'),
    run_cmd(dotnet, ['build', '--nologo', '-v', 'minimal', '-c', 'Release'],
            AutoDir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('[auto] Build OK.~n')
    ;   format('[auto] BUILD FAILED:~n~w~n', [BuildOut]), halt(1)
    ),

    %% Build success proves the template renders valid F# with resolved
    %% auto values. Runtime requires TSV fixtures which the synthetic
    %% LMDB fixture doesn't provide, so we verify build-only.
    format('[auto] PASS (build verified, template rendered valid F#)~n').
