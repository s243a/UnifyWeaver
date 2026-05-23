% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_parser_smoke.pl — F# WAM parser end-to-end test
%
% Compiles the prolog_term_parser library to F#, builds with dotnet,
% then drives `read_term_from_atom/2` from a hand-written F# driver
% against a battery of inputs.  Grew out of the issue #2400 follow-up
% work to catch parser-runtime regressions that pattern-only codegen
% tests miss.  Each PASS/FAIL line summarises one input; the final
% RESULT line is "PASSED/TOTAL".  Exit code 0 if all pass, 1
% otherwise.
%
% Skip behaviour: if `dotnet` isn''t on PATH the build fails and the
% test exits 1.  Run with the dotnet SDK installed.

:- encoding(utf8).
:- use_module('../../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).
:- use_module(library(process)).
:- use_module(library(readutil), [read_string/5]).

:- dynamic user:fs_parser_demo/0.
:- dynamic user:fs_parser_p_a/0.
:- dynamic user:fs_simple/0.
:- dynamic user:fs_parser_int/0.
:- dynamic user:fs_parser_foo/0.
:- dynamic user:fs_parser_a/0.
:- dynamic user:fs_parser_var/0.
:- dynamic user:fs_parser_paren_a/0.
:- dynamic user:fs_parser_minus/0.
:- dynamic user:fs_parser_list_123/0.
:- dynamic user:fs_parser_plus/0.
:- dynamic user:fs_parser_nested/0.
:- dynamic user:fs_parser_three_args/0.
:- dynamic user:fs_parser_mul_plus/0.

run_dotnet(Args, Dir, ExitCode, Out) :-
    process_create(path(dotnet), Args,
        [cwd(Dir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OS),
    read_string(E, _, ES),
    close(O), close(E),
    process_wait(Pid, exit(ExitCode)),
    atomic_list_concat([OS, '\n', ES], Out).

main :-
    %% A predicate that exercises the parser via read_term_from_atom.
    %% The unify on the second line is what would have failed in Python
    %% (and which my Python PR #2399 fixed). Test whether F# has the
    %% same bug.
    retractall(user:fs_parser_demo),
    retractall(user:fs_parser_p_a),
    retractall(user:fs_parser_list_123),
    retractall(user:fs_parser_plus),
    retractall(user:fs_parser_nested),
    retractall(user:fs_parser_three_args),
    retractall(user:fs_parser_mul_plus),
    retractall(user:fs_simple),
    assertz((user:fs_simple :- true)),
    assertz((user:fs_parser_int :-
        read_term_from_atom('42', _T))),
    assertz((user:fs_parser_foo :-
        read_term_from_atom('foo', _T))),
    assertz((user:fs_parser_a :-
        read_term_from_atom('a', _T))),
    assertz((user:fs_parser_var :-
        read_term_from_atom('X', _T))),
    assertz((user:fs_parser_paren_a :-
        read_term_from_atom('(a)', _T))),
    assertz((user:fs_parser_minus :-
        read_term_from_atom('-3', _T))),
    assertz((user:fs_parser_demo :-
        read_term_from_atom('p(a,b)', _T))),
    assertz((user:fs_parser_p_a :-
        read_term_from_atom('p(a)', _T))),
    assertz((user:fs_parser_list_123 :-
        read_term_from_atom('[1,2,3]', _T))),
    assertz((user:fs_parser_plus :-
        read_term_from_atom('1+2', _T))),
    assertz((user:fs_parser_nested :-
        read_term_from_atom('p(q(a))', _T))),
    assertz((user:fs_parser_three_args :-
        read_term_from_atom('foo(a,b,c)', _T))),
    assertz((user:fs_parser_mul_plus :-
        read_term_from_atom('2*3+4', _T))),

    Dir = '/tmp/uw_fsharp_parser_repro',
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir),

    %% Generate F# project with compiled parser.
    write_wam_fsharp_project(
        [user:fs_simple/0,
         user:fs_parser_int/0, user:fs_parser_foo/0,
         user:fs_parser_a/0, user:fs_parser_var/0,
         user:fs_parser_paren_a/0, user:fs_parser_minus/0,
         user:fs_parser_demo/0, user:fs_parser_p_a/0,
         user:fs_parser_list_123/0, user:fs_parser_plus/0,
         user:fs_parser_nested/0, user:fs_parser_three_args/0,
         user:fs_parser_mul_plus/0],
        [no_kernels(true),
         module_name('uw_fs_parser_repro'),
         runtime_parser(compiled)],
        Dir),
    format('Project generated at ~w~n', [Dir]),

    %% Write a custom Driver.fs that calls both predicates.
    directory_file_path(Dir, 'Program.fs', ProgPath),
    DriverCode = "module Program

open WamTypes
open WamRuntime
open Predicates

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
      WcLoweredPredicates = Map.empty
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
        eprintfn \"  [dispatchCall %s] WsPC=%d\" predKey s1.WsPC
        // dispatchCall only sets WsPC; run the WAM loop to actually execute.
        match run ctx s1 with
        | Some sf ->
            eprintfn \"  [run %s] succeeded; final WsPC=%d, WsCPsLen=%d\" predKey sf.WsPC sf.WsCPsLen
            true
        | None ->
            eprintfn \"  [run %s] returned None\" predKey
            false
    | None ->
        eprintfn \"  [dispatchCall %s] returned None\" predKey
        false

[<EntryPoint>]
let main _argv =
    // Dump labels containing our predicate names
    let ctx = mkContext ()
    eprintfn \"All labels containing 'fs_parser':\"
    ctx.WcLabels |> Map.iter (fun k v ->
        if k.Contains(\"fs_\") || k.Contains(\"parser_demo\") || k.Contains(\"parser_p_a\") then
            eprintfn \"  %s -> %d\" k v)
    eprintfn \"Total labels: %d\" (Map.count ctx.WcLabels)

    // Sanity: trivial true predicate
    assertTrue \"fs_simple :- true\"
               (runPredicate \"fs_simple/0\")

    // c7ed4ae claimed '42' works after that PR
    assertTrue \"read_term_from_atom('42', T)\"
               (runPredicate \"fs_parser_int/0\")

    // c7ed4ae claimed 'foo' works
    assertTrue \"read_term_from_atom('foo', T)\"
               (runPredicate \"fs_parser_foo/0\")

    // Single-letter atom
    assertTrue \"read_term_from_atom('a', T)\"
               (runPredicate \"fs_parser_a/0\")

    // Single variable
    assertTrue \"read_term_from_atom('X', T)\"
               (runPredicate \"fs_parser_var/0\")

    // Parenthesized atom — exercises tk_lparen / tk_rparen without compound
    assertTrue \"read_term_from_atom('(a)', T)\"
               (runPredicate \"fs_parser_paren_a/0\")

    // Negative number — prefix operator
    assertTrue \"read_term_from_atom('-3', T)\"
               (runPredicate \"fs_parser_minus/0\")

    // 1-arg compound
    assertTrue \"read_term_from_atom('p(a)', T)\"
               (runPredicate \"fs_parser_p_a/0\")

    // The test case: 2-arg compound
    assertTrue \"read_term_from_atom('p(a,b)', T)\"
               (runPredicate \"fs_parser_demo/0\")

    // List literal — exercises tk_lbracket + list_build
    assertTrue \"read_term_from_atom('[1,2,3]', T)\"
               (runPredicate \"fs_parser_list_123/0\")

    // Infix operator — exercises parse_op_loop's infix branch
    assertTrue \"read_term_from_atom('1+2', T)\"
               (runPredicate \"fs_parser_plus/0\")

    // Nested compound — recursion through parse_atom_head
    assertTrue \"read_term_from_atom('p(q(a))', T)\"
               (runPredicate \"fs_parser_nested/0\")

    // 3-arg compound — wider parse_args recursion
    assertTrue \"read_term_from_atom('foo(a,b,c)', T)\"
               (runPredicate \"fs_parser_three_args/0\")

    // Operator precedence — exercises parse_op_loop's prec/assoc logic
    assertTrue \"read_term_from_atom('2*3+4', T)\"
               (runPredicate \"fs_parser_mul_plus/0\")

    printfn \"RESULT %d/%d\" passes (passes + fails)
    if fails > 0 then 1 else 0
",
    open(ProgPath, write, OW),
    write(OW, DriverCode),
    close(OW),

    %% Build.
    format('Building...~n'),
    run_dotnet(['build', '--nologo', '-v', 'minimal'], Dir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('Build OK.~n')
    ;   format('--- build output ---~n~w~n----~n', [BuildOut]),
        format('BUILD FAILED~n'),
        halt(1)
    ),

    %% Run.
    format('Running...~n'),
    run_dotnet(['run', '--no-build', '--nologo'], Dir, RunExit, RunOut),
    format('--- run output (exit=~w) ---~n~w~n----~n', [RunExit, RunOut]).
