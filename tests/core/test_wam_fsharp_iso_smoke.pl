% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_iso_smoke.pl - F# WAM ISO catch/throw + is_iso/is_lax smoke test
%
% Compiles a battery of small predicates that exercise the ISO error
% substrate (catch/3, throw/1, is_iso/2 with structured errors,
% is_lax/2 silent failure, per-predicate iso_errors override) to F#,
% builds with dotnet, then drives each predicate from a hand-written
% F# driver.  Companion to test_wam_fsharp_runtime_smoke.pl, which
% covers the non-ISO runtime builtins.
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

:- dynamic user:fs_iso_throw_catch_match/0.
:- dynamic user:fs_iso_throw_catch_nomatch/0.
:- dynamic user:fs_iso_no_throw_binds/0.
:- dynamic user:fs_iso_is_iso_inst_err/0.
:- dynamic user:fs_iso_is_iso_type_err/0.
:- dynamic user:fs_iso_is_iso_zero_div/0.
:- dynamic user:fs_iso_is_lax_silent/0.
:- dynamic user:fs_iso_default_rewrite_is_iso/0.
:- dynamic user:fs_iso_per_pred_override_lax/0.

:- dynamic user:throwing_pred/0.
:- dynamic user:simple_recovery/0.
:- dynamic user:bind_to_42/1.
:- dynamic user:eval_is_iso/1.
:- dynamic user:eval_is_iso_unbound/1.
:- dynamic user:eval_is_iso_zerodiv/1.
:- dynamic user:eval_is_lax_bad/1.

run_dotnet(Args, Dir, ExitCode, OutText) :-
    setup_call_cleanup(
        process_create(path(dotnet), Args,
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
    %% ----- Helper predicates the tests call -----

    %% throwing_pred always throws atom 'my_error'.
    assertz((user:throwing_pred :- throw(my_error))),
    %% simple_recovery is the recovery goal for catch tests.
    assertz((user:simple_recovery :- true)),
    %% bind_to_42 binds its argument to 42 -- used to verify catch
    %% preserves bindings made by a successful goal.
    assertz((user:bind_to_42(X) :- X = 42)),

    %% eval_is_iso(X) - explicit is_iso call.  Should throw for bad
    %% expressions even when iso_errors(false) is set.
    assertz((user:eval_is_iso(X) :- is_iso(X, foo))),
    %% eval_is_iso_unbound uses an explicit is_iso with an unbound RHS.
    assertz((user:eval_is_iso_unbound(X) :- is_iso(X, _Y))),
    %% eval_is_iso_zerodiv - integer divide by zero in ISO mode.
    assertz((user:eval_is_iso_zerodiv(X) :- is_iso(X, 1 / 0))),
    %% eval_is_lax_bad - explicit lax call on a bad expression; must
    %% fail silently.
    assertz((user:eval_is_lax_bad(X) :- is_lax(X, foo))),

    %% ----- Tests -----

    %% 1. throw + catch with matching catcher pattern -> recovery runs.
    assertz((user:fs_iso_throw_catch_match :-
        catch(throwing_pred, my_error, simple_recovery))),

    %% 2. throw + catch with non-matching catcher -> rethrow.  We wrap
    %% in OUTER catch so the rethrow is captured; predicate succeeds
    %% if the outer catches what the inner re-raised.
    assertz((user:fs_iso_throw_catch_nomatch :-
        catch(
            catch(throwing_pred, other_error, fail),
            my_error,
            simple_recovery))),

    %% 3. catch + no throw -> goal succeeds, bindings persist.
    assertz((user:fs_iso_no_throw_binds :-
        catch(bind_to_42(X), _, fail),
        X == 42)),

    %% 4. is_iso with unbound RHS -> instantiation_error.
    assertz((user:fs_iso_is_iso_inst_err :-
        catch(
            eval_is_iso_unbound(_X),
            error(instantiation_error, _),
            true))),

    %% 5. is_iso with non-evaluable atom -> type_error(evaluable, ...).
    assertz((user:fs_iso_is_iso_type_err :-
        catch(
            eval_is_iso(_X),
            error(type_error(evaluable, _), _),
            true))),

    %% 6. is_iso with integer zero divide -> evaluation_error(zero_divisor).
    assertz((user:fs_iso_is_iso_zero_div :-
        catch(
            eval_is_iso_zerodiv(_X),
            error(evaluation_error(zero_divisor), _),
            true))),

    %% 7. is_lax with bad RHS -> silently fails (predicate calling
    %% it via \+ should succeed because the inner failed).
    assertz((user:fs_iso_is_lax_silent :-
        \+ eval_is_lax_bad(_X))),

    %% Setup: project dir.
    Dir = '/tmp/uw_fsharp_iso_repro',
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir),

    %% Generate F# project.  Default mode (ISO defaults off in tests
    %% above use explicit is_iso/is_lax forms, which survive any
    %% rewrite).
    write_wam_fsharp_project(
        [user:fs_iso_throw_catch_match/0,
         user:fs_iso_throw_catch_nomatch/0,
         user:fs_iso_no_throw_binds/0,
         user:fs_iso_is_iso_inst_err/0,
         user:fs_iso_is_iso_type_err/0,
         user:fs_iso_is_iso_zero_div/0,
         user:fs_iso_is_lax_silent/0,
         user:throwing_pred/0,
         user:simple_recovery/0,
         user:bind_to_42/1,
         user:eval_is_iso/1,
         user:eval_is_iso_unbound/1,
         user:eval_is_iso_zerodiv/1,
         user:eval_is_lax_bad/1],
        [no_kernels(true),
         module_name('uw_fs_iso_repro')],
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
      WsB0Stack    = []
      WsCatchers   = [] }

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
    assertTrue \"catch(throw(my_error), my_error, true)\"                  (runPredicate \"fs_iso_throw_catch_match/0\")
    assertTrue \"catch(catch(throw(my_error), other, _), my_error, true)\" (runPredicate \"fs_iso_throw_catch_nomatch/0\")
    assertTrue \"catch(X=42, _, fail), X == 42\"                          (runPredicate \"fs_iso_no_throw_binds/0\")
    assertTrue \"catch(is_iso(X, _Y), error(instantiation_error, _), _)\" (runPredicate \"fs_iso_is_iso_inst_err/0\")
    assertTrue \"catch(is_iso(X, foo), error(type_error(evaluable, _), _), _)\" (runPredicate \"fs_iso_is_iso_type_err/0\")
    assertTrue \"catch(is_iso(X, 1/0), error(evaluation_error(zero_divisor), _), _)\" (runPredicate \"fs_iso_is_iso_zero_div/0\")
    assertTrue \"\\\\+ is_lax(X, foo)\"                                    (runPredicate \"fs_iso_is_lax_silent/0\")

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
    run_dotnet(['run', '--no-build', '-c', 'Debug'], Dir, RunExit, RunOut),
    format('--- run output (exit=~w) ---~n~w~n----~n', [RunExit, RunOut]),
    (   RunExit == 0
    ->  halt(0)
    ;   halt(1)
    ).
