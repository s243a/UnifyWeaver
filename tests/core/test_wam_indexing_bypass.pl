:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_indexing_bypass.pl — Cross-target regression for the
% SwitchOnTerm + RetryMeElse + outer-CP corruption bug
% (https://github.com/s243a/UnifyWeaver/issues/2400).
%
% The bug
% -------
% When a predicate has `switch_on_term` indexed dispatch AND the
% dispatched-to clause is not clause 1 of the linear chain, the
% target clause begins with `retry_me_else`. With no leading
% `try_me_else` having fired for THIS predicate, the existing top
% choice point belongs to an outer caller. The runtime's
% `retry_me_else` modifies that outer CP — corrupting its saved
% regs/trail/heap — instead of synthesizing a fresh CP for the
% current predicate. Subsequent backtrack restores the corrupted
% snapshot, producing wrong results.
%
% Test predicate `bypass_demo/0`
% ------------------------------
%     choice(c).             %% creates the outer CP
%     choice(b).
%     classify([]).          %% mixed first-arg types → switch_on_term
%     classify([X|_]) :- X == a.
%     classify([X|_]) :- X == b.
%     bypass_demo :- choice(R), classify([R]), R == b.
%
% Expected (correct) behavior
% ---------------------------
% `choice/1` binds R to c on first try, classify([c]) fails (clause 2
% wants 'a', clause 3 wants 'b'), backtrack into choice picks the
% second clause R=b, classify([b]) succeeds via clause 3, R == b
% succeeds. `bypass_demo` succeeds.
%
% Buggy behavior
% --------------
% choice(c) pushes an outer CP. classify's switch_on_term dispatches
% [c] to L_classify_1_2 (clause 2's chain entry). Clause 2's
% retry_me_else modifies the outer CP (saving classify's clause 3
% PC as next_clause) but the outer CP's CpRegs/CpTrailLen/etc. are
% all wrong. Clause 2 body fails (c ≠ a). Backtrack uses the
% corrupted CP — restoring to a state mixing outer's regs with
% classify's continuation. The remaining backtrack chain finds no
% valid solution. `bypass_demo` fails.
%
% Status before fix
% -----------------
% F#: confirmed fails (BYPASS_DEMO_RESULT=false). This file is the
%     regression test for issue #2400.
% Python: passes — but only because Python's `switch_on_term_pc`
%     dispatch is itself broken. The emitter writes
%     `("switch_on_term", "<lvar>", "<const-tbl>", "<lstruct>",
%     "<list-label>")` (5 fields) while the runtime resolver
%     destructures `_, lv, lc, ll, ls = instr` treating each
%     positional argument as a label string. The const-table is
%     parsed as `lc` (a single label), so `labels.get(...)` returns
%     -1 for every type-specific target except the struct slot.
%     Result: switch_on_term_pc always falls through to the
%     subsequent `try_me_else_pc`, the bypass never fires, and the
%     linear chain handles every case. Bug present in principle,
%     masked by a deeper layering issue. (Tracked separately from
%     #2400; not blocking the fix here.)
% Haskell / Go / R / Scala: not yet exercised end-to-end here.
%     TODO subtests follow the same shape but are skipped pending
%     toolchain wiring.
%
% Usage
% -----
%   swipl -q -g run_tests -t halt tests/core/test_wam_indexing_bypass.pl

:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).

%% wam_python_target's write_file/2 opens output streams without
%% specifying encoding, so they pick up the process locale.  Force
%% UTF-8 here so non-ASCII bytes in the emitted Python source
%% (the runtime parser library uses unicode escape glyphs in some
%% comments / strings) don't trip the writer.
:- set_prolog_flag(encoding, utf8).
:- set_stream(user_output, encoding(utf8)).
:- set_stream(user_error,  encoding(utf8)).

:- dynamic test_failed/0.
:- dynamic test_skipped/0.

pass(Test)            :- format('[PASS] ~w~n', [Test]).
fail_test(Test, Why)  :- format('[FAIL] ~w: ~w~n', [Test, Why]),
                         (test_failed -> true ; assert(test_failed)).
skip(Test, Why)       :- format('[SKIP] ~w: ~w~n', [Test, Why]),
                         (test_skipped -> true ; assert(test_skipped)).

%% ========================================================================
%% Toolchain detection
%% ========================================================================

tool_runs(Tool, Args) :-
    catch(
        (   process_create(path(Tool), Args,
                [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0))
        ),
        _, fail).

dotnet_available  :- tool_runs(dotnet,  ['--version']).
python_available  :- tool_runs(python3, ['--version']).

tmp_root(Root) :-
    (   getenv('TMPDIR', R0), R0 \== ''
    ->  Root = R0
    ;   Root = '/tmp'
    ).

bypass_root(Dir) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_wam_indexing_bypass', Dir).

clean_dir(Dir) :-
    (   exists_directory(Dir)
    ->  process_create(path(rm), ['-rf', Dir],
            [stdout(null), stderr(null), process(Pid)]),
        process_wait(Pid, _)
    ;   true
    ).

keep_dirs :- catch(getenv('WAM_BYPASS_KEEP', '1'), _, fail), !.
keep_dirs :- fail.

maybe_clean(Dir) :-
    (   keep_dirs -> true ; clean_dir(Dir) ).

setup_dotnet_env :-
    setenv('DOTNET_CLI_TELEMETRY_OPTOUT', '1'),
    setenv('DOTNET_NOLOGO', '1').

%% ========================================================================
%% Shared fixture predicates
%% ========================================================================

:- dynamic user:choice/1.
:- dynamic user:classify/1.
:- dynamic user:bypass_demo/0.

assert_bypass_predicates :-
    retractall(user:choice(_)),
    retractall(user:classify(_)),
    retractall(user:bypass_demo),
    assertz((user:choice(c) :- true)),
    assertz((user:choice(b) :- true)),
    assertz((user:classify([]) :- true)),
    assertz((user:classify([X|_]) :- X == a)),
    assertz((user:classify([X|_]) :- X == b)),
    assertz((user:bypass_demo :-
        user:choice(R), user:classify([R]), R == b)).

teardown_bypass_predicates :-
    retractall(user:choice(_)),
    retractall(user:classify(_)),
    retractall(user:bypass_demo).

%% ========================================================================
%% F# subtest
%% ========================================================================

:- use_module('../../src/unifyweaver/targets/wam_fsharp_target').

fsharp_driver_program("module Program
open System
open WamTypes
open WamRuntime
open Predicates

[<EntryPoint>]
let main _argv =
    let foreignPreds = []
    let resolvedCode =
        resolveCallInstrs allLabels foreignPreds (Array.toList allCode)
        |> List.toArray
    let ctx =
        { WcCode = resolvedCode
          WcLabels = allLabels
          WcForeignFacts = Map.empty
          WcFfiFacts = Map.empty
          WcFfiWeightedFacts = Map.empty
          WcAtomIntern = Map.empty
          WcAtomDeintern = Map.empty
          WcForeignConfig = Map.empty
          WcLoweredPredicates = Map.empty
          WcCancellationToken = None }
    let s0 =
        { WsPC = 0
          WsRegs = Array.create MaxRegs (Unbound -1)
          WsStack = []
          WsHeap = []
          WsHeapLen = 0
          WsTrail = []
          WsTrailLen = 0
          WsCP = 0
          WsCPs = []
          WsCPsLen = 0
          WsBindings = Map.empty
          WsCutBar = 0
          WsVarCounter = 0
          WsBuilder = None
          WsBuilderStack = []
          WsAggAccum = [] }
    let ok =
        match dispatchCall ctx \"bypass_demo/0\" s0 with
        | Some _ -> true
        | None   -> false
    printfn \"BYPASS_DEMO_RESULT=%b\" ok
    if ok then 0 else 1
").

write_fsharp_project(Dir) :-
    bypass_root(Root),
    directory_file_path(Root, fsharp, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    assert_bypass_predicates,
    catch(
        wam_fsharp_target:write_wam_fsharp_project(
            [user:bypass_demo/0, user:choice/1, user:classify/1],
            [no_kernels(true), module_name('uw_bypass_demo')],
            Dir),
        E,
        (teardown_bypass_predicates, throw(E))),
    teardown_bypass_predicates,
    %% Overwrite the auto-generated benchmark Program.fs with our
    %% minimal driver that runs `bypass_demo/0` and prints the
    %% sentinel `BYPASS_DEMO_RESULT=true|false`.
    fsharp_driver_program(Prog),
    directory_file_path(Dir, 'Program.fs', PFile),
    setup_call_cleanup(
        open(PFile, write, Out, [encoding(utf8)]),
        write(Out, Prog),
        close(Out)).

run_dotnet_build(Dir, ExitCode, Output) :-
    setup_dotnet_env,
    setup_call_cleanup(
        process_create(path(dotnet),
            ['build', '--nologo', '-v', 'quiet'],
            [cwd(Dir),
             stdout(pipe(Out)), stderr(pipe(Err)),
             process(Pid)]),
        (   read_string(Out, _, OutText),
            read_string(Err, _, ErrText),
            process_wait(Pid, exit(ExitCode)),
            atomic_list_concat([OutText, '\n', ErrText], Output)
        ),
        (   catch(close(Out), _, true),
            catch(close(Err), _, true)
        )).

run_dotnet_run(Dir, ExitCode, Output) :-
    setup_dotnet_env,
    setup_call_cleanup(
        process_create(path(dotnet),
            ['run', '--nologo', '-v', 'quiet', '--no-build'],
            [cwd(Dir),
             stdout(pipe(Out)), stderr(pipe(Err)),
             process(Pid)]),
        (   read_string(Out, _, OutText),
            read_string(Err, _, ErrText),
            process_wait(Pid, exit(ExitCode)),
            atomic_list_concat([OutText, '\n', ErrText], Output)
        ),
        (   catch(close(Out), _, true),
            catch(close(Err), _, true)
        )).

test_fsharp_bypass :-
    Test = 'F# WAM: bypass_demo succeeds',
    (   \+ dotnet_available
    ->  skip(Test, 'dotnet not on PATH')
    ;   write_fsharp_project(Dir),
        run_dotnet_build(Dir, BuildEC, BuildOut),
        (   BuildEC == 0
        ->  run_dotnet_run(Dir, RunEC, RunOut),
            (   sub_string(RunOut, _, _, _, "BYPASS_DEMO_RESULT=true"), RunEC == 0
            ->  pass(Test),
                maybe_clean(Dir)
            ;   format('---- F# run output ----~n~w~n----~n', [RunOut]),
                fail_test(Test,
                    format_atom('expected BYPASS_DEMO_RESULT=true; exit=~w', [RunEC]))
            )
        ;   format('---- F# build output ----~n~w~n----~n', [BuildOut]),
            fail_test(Test,
                format_atom('dotnet build failed (exit ~w)', [BuildEC]))
        )
    ).

format_atom(Format, Args, Atom) :- format(atom(Atom), Format, Args).
format_atom(Format, Args) :- format_atom(Format, Args, _).

%% ========================================================================
%% Python subtest
%% ========================================================================

:- use_module('../../src/unifyweaver/targets/wam_python_target').

python_driver_program("#!/usr/bin/env python3
import predicates as p
import wam_runtime as wr

code, labels = wr.load_program(p.build_program())
state = wr.WamState()
ok = wr.run_wam(code, labels, 'bypass_demo/0', state)
print('BYPASS_DEMO_RESULT={}'.format('true' if ok else 'false'))
import sys
sys.exit(0 if ok else 1)
").

write_python_project(Dir) :-
    bypass_root(Root),
    directory_file_path(Root, python, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    assert_bypass_predicates,
    catch(
        wam_python_target:write_wam_python_project(
            [user:bypass_demo/0, user:choice/1, user:classify/1],
            [],
            Dir),
        E,
        (teardown_bypass_predicates, throw(E))),
    teardown_bypass_predicates,
    python_driver_program(Prog),
    directory_file_path(Dir, 'driver.py', DFile),
    setup_call_cleanup(
        open(DFile, write, Out, [encoding(utf8)]),
        write(Out, Prog),
        close(Out)).

run_python(Dir, ExitCode, Output) :-
    setup_call_cleanup(
        process_create(path(python3), ['driver.py'],
            [cwd(Dir),
             stdout(pipe(Out)), stderr(pipe(Err)),
             process(Pid)]),
        (   read_string(Out, _, OutText),
            read_string(Err, _, ErrText),
            process_wait(Pid, exit(ExitCode)),
            atomic_list_concat([OutText, '\n', ErrText], Output)
        ),
        (   catch(close(Out), _, true),
            catch(close(Err), _, true)
        )).

test_python_bypass :-
    Test = 'Python WAM: bypass_demo succeeds',
    (   \+ python_available
    ->  skip(Test, 'python3 not on PATH')
    ;   write_python_project(Dir),
        run_python(Dir, RunEC, RunOut),
        (   sub_string(RunOut, _, _, _, "BYPASS_DEMO_RESULT=true"), RunEC == 0
        ->  pass(Test),
            maybe_clean(Dir)
        ;   format('---- Python run output ----~n~w~n----~n', [RunOut]),
            fail_test(Test,
                format_atom('expected BYPASS_DEMO_RESULT=true; exit=~w', [RunEC]))
        )
    ).

%% ========================================================================
%% Other targets — stubs.  Step 2 of the rollout exercises F# + Python
%% to validate the design; Haskell / Go / R / Scala subtests follow
%% the same shape and are added once the F# emitter+runtime fix is
%% confirmed working.
%% ========================================================================

test_haskell_bypass :-
    skip('Haskell WAM: bypass_demo succeeds', 'TODO: pending step 5 (bulk port)').

test_go_bypass :-
    skip('Go WAM: bypass_demo succeeds', 'TODO: pending step 5 (bulk port)').

test_r_bypass :-
    skip('R WAM: bypass_demo succeeds', 'TODO: pending step 5 (bulk port)').

test_scala_bypass :-
    skip('Scala WAM: bypass_demo succeeds', 'TODO: pending step 5 (bulk port)').

%% ========================================================================
%% Runner
%% ========================================================================

run_tests :-
    retractall(test_failed),
    retractall(test_skipped),
    Tests = [
        test_fsharp_bypass,
        test_python_bypass,
        test_haskell_bypass,
        test_go_bypass,
        test_r_bypass,
        test_scala_bypass
    ],
    forall(member(T, Tests),
        catch(call(T), E,
            (format('[ERROR] ~w threw: ~q~n', [T, E]),
             assert(test_failed)))),
    (   test_failed
    ->  format('~n[RESULT] FAIL: at least one subtest failed~n'),
        halt(1)
    ;   format('~n[RESULT] PASS (skips allowed)~n'),
        halt(0)
    ).
