% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_lowered_bench.pl — F# WAM lowered emitter benchmark
%
% Generates the SAME workload as two separate F# projects -- one under
% emit_mode(interpreter), one under emit_mode(functions) -- builds both,
% runs each driver, and compares wall-clock execution time.  Used to
% justify (or refute) flipping emit_mode(functions) to the default.
%
% Workload: a parser-heavy predicate
%
%   fs_bench :- read_term_from_atom('foo(a, b+c, [1,2,3])', _T).
%
% called N = 100_000 times in a tight loop inside the F# driver.  The
% input exercises atom-head + compound + infix op + list literal, which
% means many instructions per call -- exactly where instruction dispatch
% overhead would show up.
%
% Each project is built once (build time is NOT measured); only the
% runPredicate loop is timed via System.Diagnostics.Stopwatch.

:- encoding(utf8).
:- use_module('../../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).
:- use_module(library(process)).
:- use_module(library(readutil), [read_string/5]).

:- dynamic user:fs_bench/0.
:- dynamic user:fs_bench_simple/0.

run_dotnet(Args, Dir, ExitCode, Out) :-
    process_create(path(dotnet), Args,
        [cwd(Dir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OS),
    read_string(E, _, ES),
    close(O), close(E),
    process_wait(Pid, exit(ExitCode)),
    atomic_list_concat([OS, '\n', ES], Out).

bench_driver(NIter, PredKey, DriverCode) :-
    format(string(DriverCode), "module Program

open System.Diagnostics
open WamTypes
open WamRuntime
open Predicates
open Lowered

let mkContext () =
    let foreignPreds : string list = []
    let resolvedCode =
        resolveCallInstrs allLabels foreignPreds (Array.toList allCode)
        |> List.toArray
    { WcCode              = resolvedCode
      WcLabels            = allLabels
      WcForeignFacts      = Map.empty
      WcFfiFacts          = Map.empty
      WcFfiWeightedFacts  = Map.empty
      WcAtomIntern        = Map.empty
      WcAtomDeintern      = Map.empty
      WcForeignConfig     = Map.empty
      WcLoweredPredicates = loweredPredicates
      WcLookupSources   = Map.empty
      WcCancellationToken = None }

let mkState () : WamState =
    { WsPC         = 0
      WsRegs       = Array.create MaxRegs (Unbound -1)
      WsStack      = []
      WsHeap       = []
      WsHeapLen    = 0
      WsTrail      = []
      WsTrailLen   = 0
      WsCP         = 0
      WsCPs        = []
      WsCPsLen     = 0
      WsBindings   = Map.empty
      WsCutBar     = 0
      WsVarCounter = 0
      WsBuilder    = None
      WsBuilderStack = []
      WsAggAccum   = []
      WsB0Stack    = []
      WsCatchers   = [] }

let runPredicate (ctx: WamContext) (predKey: string) =
    let s = mkState ()
    match dispatchCall ctx predKey s with
    | Some s1 ->
        match run ctx s1 with
        | Some _ -> true
        | None   -> false
    | None -> false

[<EntryPoint>]
let main _argv =
    let ctx = mkContext ()

    // Warm-up: tier-up JIT + first-time costs out of the timed region.
    let predKey = \"~w\"
    let nIter = ~w
    // Multiple warm-up rounds so any tiered-JIT promotion settles before
    // measurement, and so the GC's per-allocation cost has steady state.
    for _ in 1 .. 5 do
        for _ in 1 .. min nIter 1000 do
            ignore (runPredicate ctx predKey)
    System.GC.Collect()
    System.GC.WaitForPendingFinalizers()
    System.GC.Collect()

    // Three timed rounds; report the MEDIAN to dampen JIT / GC variance.
    let rounds = 3
    let timings = Array.zeroCreate rounds
    let mutable totalOk = 0
    for r in 0 .. rounds - 1 do
        let sw = Stopwatch.StartNew()
        let mutable ok = 0
        for _ in 1 .. nIter do
            if runPredicate ctx predKey then ok <- ok + 1
        sw.Stop()
        timings.[r] <- sw.ElapsedMilliseconds
        totalOk <- totalOk + ok
    Array.sortInPlace timings
    let median = timings.[rounds / 2]
    let expectedOk = nIter * rounds
    printfn \"ELAPSED_MS %d\" median
    printfn \"ROUNDS %A\" timings
    printfn \"ITER %d\" nIter
    printfn \"OK %d\" totalOk
    if totalOk <> expectedOk then 1 else 0
", [PredKey, NIter]).

run_bench(Workload, EmitMode, Dir, ModuleName, NIter, ElapsedMs) :-
    setup_workload(Workload, PredKey, ProjectOptions),
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir),
    workload_predicates(Workload, Predicates),
    write_wam_fsharp_project(
        Predicates,
        [no_kernels(true),
         emit_mode(EmitMode),
         module_name(ModuleName)
         | ProjectOptions],
        Dir),
    format('Generated [~w / ~w] at ~w~n', [Workload, EmitMode, Dir]),
    directory_file_path(Dir, 'Program.fs', ProgPath),
    bench_driver(NIter, PredKey, DriverCode),
    open(ProgPath, write, OW, [encoding(utf8)]),
    write(OW, DriverCode),
    close(OW),
    %% Release build for fair measurement -- Debug builds disable optimisations.
    format('Building [~w / ~w] (Release)...~n', [Workload, EmitMode]),
    run_dotnet(['build', '-c', 'Release', '--nologo', '-v', 'quiet'],
               Dir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('Build OK.~n')
    ;   format('--- build output ---~n~w~n----~n', [BuildOut]),
        halt(1)
    ),
    format('Running [~w / ~w]...~n', [Workload, EmitMode]),
    run_dotnet(['run', '-c', 'Release', '--no-build', '--nologo'],
               Dir, RunExit, RunOut),
    (   RunExit == 0 -> true
    ;   format('--- run output ---~n~w~n----~n', [RunOut]),
        halt(1)
    ),
    (   split_string(RunOut, "\n", "", Lines),
        member(Line, Lines),
        split_string(Line, " ", "", ["ELAPSED_MS", MsStr]),
        number_string(ElapsedMs, MsStr)
    ->  format('[~w / ~w] ~w ms for ~w iterations~n', [Workload, EmitMode, ElapsedMs, NIter])
    ;   format('Could not parse ELAPSED_MS from output:~n~w~n', [RunOut]),
        halt(1)
    ).

%% setup_workload(+Workload, -PredKey, -ProjectOptions)
%  Assert the workload's body into user: and return the predicate key
%  (matches what dispatchCall takes) plus any extra options to pass to
%  write_wam_fsharp_project (e.g. runtime_parser(compiled) for parser).
setup_workload(parser_heavy, "fs_bench/0", [runtime_parser(compiled)]) :-
    retractall(user:fs_bench),
    assertz((user:fs_bench :-
        read_term_from_atom('foo(a, b+c, [1,2,3])', _T))).
setup_workload(fully_lowered, "fs_bench_simple/0", []) :-
    retractall(user:fs_bench_simple),
    %% A predicate that fully lowers (no parser library calls).
    %% Exercises unify, ==, deallocate, proceed -- all inlined now.
    assertz((user:fs_bench_simple :-
        X = foo(a, b, c, [1,2,3]),
        X == foo(a, b, c, [1,2,3]))).

workload_predicates(parser_heavy, [user:fs_bench/0]).
workload_predicates(fully_lowered, [user:fs_bench_simple/0]).

bench_pair(Workload, NIter, MsInterp, MsFunc) :-
    atom_concat('/tmp/uw_fs_bench_', Workload, DirBase),
    atom_concat(DirBase, '_interp', DirI),
    atom_concat(DirBase, '_func', DirF),
    atom_concat('uw_fs_bench_', Workload, ModBase),
    atom_concat(ModBase, '_interp', ModI),
    atom_concat(ModBase, '_func', ModF),
    run_bench(Workload, interpreter, DirI, ModI, NIter, MsInterp),
    run_bench(Workload, functions,   DirF, ModF, NIter, MsFunc).

report_workload(Workload, NIter, MsInterp, MsFunc) :-
    format('~n--- ~w ---~n', [Workload]),
    format('Iterations    : ~w~n', [NIter]),
    format('interpreter ms: ~w~n', [MsInterp]),
    format('functions   ms: ~w~n', [MsFunc]),
    (   MsFunc > 0
    ->  Ratio is MsInterp / MsFunc,
        format('Speedup       : ~3fx (interpreter / functions)~n', [Ratio])
    ;   true
    ),
    (   MsInterp > 0
    ->  PctSaved is 100 * (MsInterp - MsFunc) / MsInterp,
        format('Time saved    : ~2f percent~n', [PctSaved])
    ;   true
    ).

main :-
    NIterParser = 10000,
    NIterSimple = 200000,
    bench_pair(parser_heavy,  NIterParser, MsPI, MsPF),
    bench_pair(fully_lowered, NIterSimple, MsSI, MsSF),
    format('~n========================================~n'),
    format('F# WAM lowered-emitter benchmark~n'),
    format('========================================~n'),
    report_workload(parser_heavy,  NIterParser, MsPI, MsPF),
    report_workload(fully_lowered, NIterSimple, MsSI, MsSF),
    format('========================================~n').
