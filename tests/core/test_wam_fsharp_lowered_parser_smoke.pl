% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_lowered_parser_smoke.pl
% F# WAM lowered emitter + compiled-parser end-to-end test.
%
% Exercises a focused subset of `read_term_from_atom/2` cases under
% emit_mode(functions) with runtime_parser(compiled).  Companion to
% test_wam_fsharp_lowered_smoke.pl (which covers runtime builtins under
% lowered mode); this one specifically catches regressions in the
% interaction between lowered predicates and the WAM interpreter's
% `run` loop when calling non-lowered helpers (e.g. tokenize_loop/3,
% take_digits/3) through dispatchCall.
%
% The cases below are deliberately the simple-shape parser inputs
% (`'42'`, `'foo'`, `'a'`, `'(a)'`, `'-3'`) -- these are the ones that
% surfaced the dispatchCall WsCP-propagation bug: when a lowered
% predicate calls dispatchCall to enter an interpreted helper, the
% helper's Proceed otherwise propagated WsCP into WsPC and the
% interpreter loop kept running the LOWERED caller's WAM continuation
% in addition to the F# continuation -- producing double-execution
% state corruption.

:- encoding(utf8).
:- use_module('../../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).
:- use_module(library(process)).
:- use_module(library(readutil), [read_string/5]).

:- dynamic user:fs_lp_int/0.
:- dynamic user:fs_lp_foo/0.
:- dynamic user:fs_lp_a/0.
:- dynamic user:fs_lp_paren/0.
:- dynamic user:fs_lp_minus/0.
:- dynamic user:fs_lp_p_a/0.
:- dynamic user:fs_lp_list/0.
:- dynamic user:fs_lp_plus/0.

run_dotnet(Args, Dir, ExitCode, Out) :-
    process_create(path(dotnet), Args,
        [cwd(Dir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OS),
    read_string(E, _, ES),
    close(O), close(E),
    process_wait(Pid, exit(ExitCode)),
    atomic_list_concat([OS, '\n', ES], Out).

main :-
    retractall(user:fs_lp_int),
    retractall(user:fs_lp_foo),
    retractall(user:fs_lp_a),
    retractall(user:fs_lp_paren),
    retractall(user:fs_lp_minus),
    retractall(user:fs_lp_p_a),
    retractall(user:fs_lp_list),
    retractall(user:fs_lp_plus),

    assertz((user:fs_lp_int   :- read_term_from_atom('42', _T))),
    assertz((user:fs_lp_foo   :- read_term_from_atom('foo', _T))),
    assertz((user:fs_lp_a     :- read_term_from_atom('a', _T))),
    assertz((user:fs_lp_paren :- read_term_from_atom('(a)', _T))),
    assertz((user:fs_lp_minus :- read_term_from_atom('-3', _T))),
    assertz((user:fs_lp_p_a   :- read_term_from_atom('p(a)', _T))),
    assertz((user:fs_lp_list  :- read_term_from_atom('[1,2,3]', _T))),
    assertz((user:fs_lp_plus  :- read_term_from_atom('1+2', _T))),

    Dir = '/tmp/uw_fsharp_lowered_parser_repro',
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir),

    write_wam_fsharp_project(
        [user:fs_lp_int/0, user:fs_lp_foo/0, user:fs_lp_a/0,
         user:fs_lp_paren/0, user:fs_lp_minus/0, user:fs_lp_p_a/0,
         user:fs_lp_list/0, user:fs_lp_plus/0],
        [no_kernels(true),
         emit_mode(functions),
         runtime_parser(compiled),
         module_name('uw_fs_lowered_parser_repro')],
        Dir),
    format('Project generated at ~w~n', [Dir]),

    directory_file_path(Dir, 'Program.fs', ProgPath),
    DriverCode = "module Program

open WamTypes
open WamRuntime
open Predicates
open Lowered

let mutable passes = 0
let mutable fails = 0

let assertTrue (name: string) (cond: bool) =
    if cond then
        passes <- passes + 1
        printfn \"[PASS] %s\" name
    else
        fails <- fails + 1
        printfn \"[FAIL] %s\" name

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
      WsB0Stack    = [] }

let runPredicate (predKey: string) =
    let ctx = mkContext ()
    let s = mkState ()
    match dispatchCall ctx predKey s with
    | Some s1 ->
        match run ctx s1 with
        | Some _ -> true
        | None   -> false
    | None -> false

[<EntryPoint>]
let main _argv =
    assertTrue \"read_term_from_atom('42', T)\"      (runPredicate \"fs_lp_int/0\")
    assertTrue \"read_term_from_atom('foo', T)\"     (runPredicate \"fs_lp_foo/0\")
    assertTrue \"read_term_from_atom('a', T)\"       (runPredicate \"fs_lp_a/0\")
    assertTrue \"read_term_from_atom('(a)', T)\"     (runPredicate \"fs_lp_paren/0\")
    assertTrue \"read_term_from_atom('-3', T)\"      (runPredicate \"fs_lp_minus/0\")
    assertTrue \"read_term_from_atom('p(a)', T)\"    (runPredicate \"fs_lp_p_a/0\")
    assertTrue \"read_term_from_atom('[1,2,3]', T)\" (runPredicate \"fs_lp_list/0\")
    assertTrue \"read_term_from_atom('1+2', T)\"     (runPredicate \"fs_lp_plus/0\")

    printfn \"RESULT %d/%d\" passes (passes + fails)
    if fails > 0 then 1 else 0
",
    open(ProgPath, write, OW, [encoding(utf8)]),
    write(OW, DriverCode),
    close(OW),

    format('Building...~n'),
    run_dotnet(['build', '--nologo', '-v', 'minimal'], Dir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('Build OK.~n')
    ;   format('--- build output ---~n~w~n----~n', [BuildOut]),
        format('BUILD FAILED~n'),
        halt(1)
    ),

    format('Running...~n'),
    run_dotnet(['run', '--no-build', '--nologo'], Dir, RunExit, RunOut),
    format('--- run output (exit=~w) ---~n~w~n----~n', [RunExit, RunOut]).
