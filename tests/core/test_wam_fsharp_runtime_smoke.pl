% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_runtime_smoke.pl — F# WAM runtime predicates smoke test
%
% Compiles a battery of small predicates that exercise the runtime
% builtins (member/2, append/3, length/2, is/2, =../2, functor/3,
% arg/3, findall/3, type checks) to F#, builds with dotnet, then
% drives each predicate from a hand-written F# driver.  Companion to
% test_wam_fsharp_parser_smoke.pl, which covers read_term_from_atom.
%
% Each PASS/FAIL line summarises one input; the final RESULT line is
% "PASSED/TOTAL".  Exit code 0 if all pass, 1 otherwise.

:- encoding(utf8).
:- use_module('../../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).
:- use_module(library(process)).
:- use_module(library(readutil), [read_string/5]).

:- dynamic user:fs_rt_append/0.
:- dynamic user:fs_rt_member_first/0.
:- dynamic user:fs_rt_member_middle/0.
:- dynamic user:fs_rt_member_last/0.
:- dynamic user:fs_rt_length/0.
:- dynamic user:fs_rt_is_arith/0.
:- dynamic user:fs_rt_lt/0.
:- dynamic user:fs_rt_gt/0.
:- dynamic user:fs_rt_univ_decompose/0.
:- dynamic user:fs_rt_univ_compose/0.
:- dynamic user:fs_rt_functor/0.
:- dynamic user:fs_rt_arg/0.
:- dynamic user:fs_rt_atom_type/0.
:- dynamic user:fs_rt_integer_type/0.
:- dynamic user:fs_rt_var_type/0.
:- dynamic user:fs_rt_nonvar_type/0.
:- dynamic user:fs_rt_findall_member/0.
:- dynamic user:fs_rt_findall_single/0.
:- dynamic user:fs_rt_findall_length/0.

run_dotnet(Args, Dir, ExitCode, Out) :-
    process_create(path(dotnet), Args,
        [cwd(Dir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, OS),
    read_string(E, _, ES),
    close(O), close(E),
    process_wait(Pid, exit(ExitCode)),
    atomic_list_concat([OS, '\n', ES], Out).

main :-
    retractall(user:fs_rt_append),
    retractall(user:fs_rt_member_first),
    retractall(user:fs_rt_member_middle),
    retractall(user:fs_rt_member_last),
    retractall(user:fs_rt_length),
    retractall(user:fs_rt_is_arith),
    retractall(user:fs_rt_lt),
    retractall(user:fs_rt_gt),
    retractall(user:fs_rt_univ_decompose),
    retractall(user:fs_rt_univ_compose),
    retractall(user:fs_rt_functor),
    retractall(user:fs_rt_arg),
    retractall(user:fs_rt_atom_type),
    retractall(user:fs_rt_integer_type),
    retractall(user:fs_rt_var_type),
    retractall(user:fs_rt_nonvar_type),
    retractall(user:fs_rt_findall_member),
    retractall(user:fs_rt_findall_single),
    retractall(user:fs_rt_findall_length),

    %% Each test is a 0-arity predicate whose body succeeds when the
    %% builtin under test behaves correctly.  Each uses == (structural
    %% equality) to confirm the bound value matches the expected one,
    %% so passing means "builtin returned the right answer", not just
    %% "builtin didn't crash".
    assertz((user:fs_rt_append :-
        append([1,2], [3,4], X), X == [1,2,3,4])),
    assertz((user:fs_rt_member_first :-
        member(1, [1,2,3]))),
    assertz((user:fs_rt_member_middle :-
        member(2, [1,2,3]))),
    assertz((user:fs_rt_member_last :-
        member(3, [1,2,3]))),
    assertz((user:fs_rt_length :-
        length([a,b,c], L), L == 3)),
    assertz((user:fs_rt_is_arith :-
        X is 2 + 3 * 4, X == 14)),
    assertz((user:fs_rt_lt :-
        2 < 3)),
    assertz((user:fs_rt_gt :-
        3 > 2)),
    assertz((user:fs_rt_univ_decompose :-
        foo(a,b) =.. L, L == [foo,a,b])),
    assertz((user:fs_rt_univ_compose :-
        T =.. [bar, 1, 2], T == bar(1,2))),
    assertz((user:fs_rt_functor :-
        functor(foo(a,b), F, N), F == foo, N == 2)),
    assertz((user:fs_rt_arg :-
        arg(2, foo(a,b,c), X), X == b)),
    assertz((user:fs_rt_atom_type :-
        atom(foo))),
    assertz((user:fs_rt_integer_type :-
        integer(42))),
    assertz((user:fs_rt_var_type :-
        var(_X))),
    assertz((user:fs_rt_nonvar_type :-
        nonvar(foo))),
    assertz((user:fs_rt_findall_member :-
        findall(X, member(X, [1,2,3]), L), L == [1,2,3])),
    assertz((user:fs_rt_findall_single :-
        findall(X, X = 42, L), L == [42])),
    assertz((user:fs_rt_findall_length :-
        findall(X, member(X, [1,2,3]), L), length(L, N), N == 3)),

    Dir = '/tmp/uw_fsharp_runtime_repro',
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir),

    %% Generate F# project.  No runtime parser needed.
    write_wam_fsharp_project(
        [user:fs_rt_append/0,
         user:fs_rt_member_first/0,
         user:fs_rt_member_middle/0,
         user:fs_rt_member_last/0,
         user:fs_rt_length/0,
         user:fs_rt_is_arith/0,
         user:fs_rt_lt/0,
         user:fs_rt_gt/0,
         user:fs_rt_univ_decompose/0,
         user:fs_rt_univ_compose/0,
         user:fs_rt_functor/0,
         user:fs_rt_arg/0,
         user:fs_rt_atom_type/0,
         user:fs_rt_integer_type/0,
         user:fs_rt_var_type/0,
         user:fs_rt_nonvar_type/0,
         user:fs_rt_findall_member/0,
         user:fs_rt_findall_single/0,
         user:fs_rt_findall_length/0],
        [no_kernels(true),
         module_name('uw_fs_runtime_repro')],
        Dir),
    format('Project generated at ~w~n', [Dir]),

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
        match run ctx s1 with
        | Some _ -> true
        | None   -> false
    | None -> false

[<EntryPoint>]
let main _argv =
    assertTrue \"append([1,2], [3,4], [1,2,3,4])\"          (runPredicate \"fs_rt_append/0\")
    assertTrue \"member(1, [1,2,3])\"                        (runPredicate \"fs_rt_member_first/0\")
    assertTrue \"member(2, [1,2,3])\"                        (runPredicate \"fs_rt_member_middle/0\")
    assertTrue \"member(3, [1,2,3])\"                        (runPredicate \"fs_rt_member_last/0\")
    assertTrue \"length([a,b,c], 3)\"                        (runPredicate \"fs_rt_length/0\")
    assertTrue \"X is 2+3*4, X == 14\"                       (runPredicate \"fs_rt_is_arith/0\")
    assertTrue \"2 < 3\"                                      (runPredicate \"fs_rt_lt/0\")
    assertTrue \"3 > 2\"                                      (runPredicate \"fs_rt_gt/0\")
    assertTrue \"foo(a,b) =.. [foo,a,b]\"                    (runPredicate \"fs_rt_univ_decompose/0\")
    assertTrue \"T =.. [bar,1,2], T == bar(1,2)\"             (runPredicate \"fs_rt_univ_compose/0\")
    assertTrue \"functor(foo(a,b), foo, 2)\"                  (runPredicate \"fs_rt_functor/0\")
    assertTrue \"arg(2, foo(a,b,c), b)\"                      (runPredicate \"fs_rt_arg/0\")
    assertTrue \"atom(foo)\"                                  (runPredicate \"fs_rt_atom_type/0\")
    assertTrue \"integer(42)\"                                (runPredicate \"fs_rt_integer_type/0\")
    assertTrue \"var(_X)\"                                    (runPredicate \"fs_rt_var_type/0\")
    assertTrue \"nonvar(foo)\"                                (runPredicate \"fs_rt_nonvar_type/0\")
    assertTrue \"findall(X, member(X,[1,2,3]), [1,2,3])\"             (runPredicate \"fs_rt_findall_member/0\")
    assertTrue \"findall(X, X=42, [42])\"                              (runPredicate \"fs_rt_findall_single/0\")
    assertTrue \"findall(X, member(X,[1,2,3]), L), length(L, 3)\"      (runPredicate \"fs_rt_findall_length/0\")

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
