% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_lmdb_scale_sweep.pl - Scale sweep across fixture sizes
%
% Runs the LMDB materialisation mode benchmark at multiple fixture sizes
% to find the lazy↔eager crossover point and measure how each mode
% scales. Feeds the lmdb_materialisation(auto) cost-model resolver.
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
    LmdbDir = '/tmp/uw_fsharp_lmdb_sweep_fixture',
    ProjectDir = '/tmp/uw_fsharp_lmdb_sweep',
    catch(delete_directory_and_contents(ProjectDir), _, true),

    %% Build the F# project once (first fixture size).
    %% Subsequent runs only rewrite Program.fs and do incremental build.
    Scales = [50, 200, 500, 1000, 2000, 5000],
    ChildrenPerParent = 8,

    format('~n========================================~n'),
    format('F# LMDB Scale Sweep — FULL demand (all edges queried)~n'),
    format('========================================~n~n'),
    format('~w ~w ~w ~w ~w ~w ~w ~w ~w~n',
           ['Parents', 'Edges', 'DictLoad', 'DictQ', 'EagerLoad', 'EagerQ',
            'LazyQ', '2LvlCold', '2LvlWarm']),
    format('------- ----- -------- ----- --------- ------ ----- -------- --------~n'),

    forall(member(NParents, Scales),
           run_one_scale(NParents, ChildrenPerParent, LmdbDir, ProjectDir, full)),

    format('~n========================================~n'),
    format('F# LMDB Scale Sweep — PARTIAL demand (100 seeds, varying graph)~n'),
    format('This shows the lazy/cached advantage at large scale.~n'),
    format('========================================~n~n'),
    format('~w ~w ~w ~w ~w ~w ~w ~w ~w~n',
           ['Parents', 'Edges', 'DictLoad', 'DictQ', 'EagerLoad', 'EagerQ',
            'LazyQ', '2LvlCold', '2LvlWarm']),
    format('------- ----- -------- ----- --------- ------ ----- -------- --------~n'),

    forall(member(NParents, Scales),
           run_one_scale(NParents, ChildrenPerParent, LmdbDir, ProjectDir, partial)).

run_one_scale(NParents, CPP, LmdbDir, ProjectDir, DemandMode) :-
    NEdges is NParents * CPP,
    FirstChild is NParents + 1,
    (   DemandMode = full
    ->  LastChild is NParents + NEdges
    ;   % partial: query only 100 seeds from the middle of the range
        NSeedsPartial is min(100, NEdges),
        MidStart is NParents + 1 + (NEdges - NSeedsPartial) // 2,
        FirstChild2 is MidStart,
        LastChild is FirstChild2 + NSeedsPartial - 1
    ),
    (   DemandMode = partial, NEdges >= 100
    ->  true  % use partial range
    ;   true
    ),
    (   DemandMode = partial
    ->  SeedFirst = FirstChild2
    ;   SeedFirst = FirstChild
    ),
    atom_number(NParentsA, NParents),
    atom_number(CPPA, CPP),

    %% Generate fixture
    catch(delete_directory_and_contents(LmdbDir), _, true),
    run_cmd(python3,
            ['examples/benchmark/generate_synthetic_phase1_lmdb.py',
             LmdbDir, '--parents', NParentsA,
             '--children-per-parent', CPPA, '--refresh'],
            '.', PyExit, _),
    (PyExit == 0 -> true ; format('LMDB gen failed for ~w~n', [NParents]), halt(1)),

    %% Generate or update F# project
    (   \+ exists_directory(ProjectDir)
    ->  make_directory_path(ProjectDir),
        write_wam_fsharp_project([], [no_kernels(true), module_name('uw_fs_lmdb_sweep'), lmdb_path(LmdbDir)], ProjectDir)
    ;   true
    ),

    %% Write Program.fs with this fixture's path and seed range
    atom_string(LmdbDir, LmdbDirStr),
    atom_number(FirstChildA, SeedFirst),
    atom_number(LastChildA, LastChild),
    atom_string(FirstChildA, FirstChildS),
    atom_string(LastChildA, LastChildS),

    string_concat(LmdbDirStr, "\"
    let env = openEnv lmdbPath
    let seeds = [| ", Seg1),
    string_concat(Seg1, FirstChildS, Seg2),
    string_concat(Seg2, " .. ", Seg3),
    string_concat(Seg3, LastChildS, Seg4),
    string_concat(Seg4, " |]

    let sw = Diagnostics.Stopwatch.StartNew()
    let dictData = loadDupsortRelationDict env \"category_parent\"
    let dictLoadMs = sw.Elapsed.TotalMilliseconds
    let dictSrc = WamTypes.DictLookupSource(dictData) :> WamTypes.ILookupSource
    let swQ = Diagnostics.Stopwatch.StartNew()
    for seed in seeds do dictSrc.Lookup(seed) |> ignore
    swQ.Stop()
    let dictQMs = swQ.Elapsed.TotalMilliseconds

    let sw2 = Diagnostics.Stopwatch.StartNew()
    let eagerMap = loadCategoryParent env
    let eagerLoadMs = sw2.Elapsed.TotalMilliseconds
    let eagerSrc = WamTypes.EagerLookupSource(eagerMap) :> WamTypes.ILookupSource
    let swQ2 = Diagnostics.Stopwatch.StartNew()
    for seed in seeds do eagerSrc.Lookup(seed) |> ignore
    swQ2.Stop()
    let eagerQMs = swQ2.Elapsed.TotalMilliseconds

    let lazySrc = LmdbCursorLookup(env, \"category_parent\") :> WamTypes.ILookupSource
    let swQ3 = Diagnostics.Stopwatch.StartNew()
    for seed in seeds do lazySrc.Lookup(seed) |> ignore
    swQ3.Stop()
    let lazyQMs = swQ3.Elapsed.TotalMilliseconds

    let tlSrc = TwoLevelCachedLookupSource(LmdbCursorLookup(env, \"category_parent\")) :> WamTypes.ILookupSource
    let swQ4 = Diagnostics.Stopwatch.StartNew()
    for seed in seeds do tlSrc.Lookup(seed) |> ignore
    swQ4.Stop()
    let tlColdMs = swQ4.Elapsed.TotalMilliseconds
    let swQ5 = Diagnostics.Stopwatch.StartNew()
    for seed in seeds do tlSrc.Lookup(seed) |> ignore
    swQ5.Stop()
    let tlWarmMs = swQ5.Elapsed.TotalMilliseconds

    env.Dispose()
    printfn \"%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\" (float seeds.Length) dictLoadMs dictQMs eagerLoadMs eagerQMs lazyQMs tlColdMs tlWarmMs
    0
", DriverTail),
    string_concat("module Program
open System
open LmdbFactSource
open WamTypes
[<EntryPoint>]
let main _argv =
    let lmdbPath = \"", DriverTail, DriverCode),

    directory_file_path(ProjectDir, 'Program.fs', ProgPath),
    open(ProgPath, write, OW, [encoding(utf8)]),
    write(OW, DriverCode),
    close(OW),

    %% Build (incremental after first)
    run_cmd(dotnet, ['build', '--nologo', '-v', 'quiet', '-c', 'Release'],
            ProjectDir, BuildExit, BuildOut),
    (   BuildExit == 0 -> true
    ;   format('BUILD FAILED at ~w parents:~n~w~n', [NParents, BuildOut]), halt(1)
    ),

    %% Run
    run_cmd(dotnet, ['run', '--no-build', '-c', 'Release'],
            ProjectDir, _RunExit, RunOut),
    %% Parse CSV output line
    split_string(RunOut, "\n", "\r\n ", Lines),
    include(\=(""), Lines, NonEmpty),
    last(NonEmpty, CsvLine),
    split_string(CsvLine, ",", " ", Fields),
    (   Fields = [EdgesS, DictLoadS, DictQS, EagerLoadS, EagerQS, LazyQS, TlColdS, TlWarmS]
    ->  format('~7w ~5w ~8w ~5w ~9w ~6w ~5w ~8w ~8w~n',
               [NParents, NEdges, DictLoadS, DictQS, EagerLoadS, EagerQS, LazyQS, TlColdS, TlWarmS])
    ;   format('~7w ~5w  (parse error: ~w)~n', [NParents, NEdges, CsvLine])
    ).
