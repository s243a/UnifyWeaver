:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_kernel_gate_tc.pl
%
% FS-HYBRID-KERNEL-GATE-TC:
%   1) Capability-gate unsupported detected kernels (no undefined
%      nativeKernel_* / FS0039).
%   2) Native foreign acceleration for transitive_closure2 — not
%      "F# transitive closure from scratch" (generic WAM tc/2 already
%      works; category/bidirectional/reachableToRoot* already exist).
%
% Run: swipl -q -g run_tests -t halt tests/core/test_wam_fsharp_kernel_gate_tc.pl
% Requires: swipl; dotnet for execute/build cases (skipped if absent).

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3,
               wam_fsharp_native_kernel_kind/1,
               wam_fsharp_native_kernel_supported/1]).
:- use_module('../../src/unifyweaver/core/recursive_kernel_detection',
              [detect_recursive_kernel/4]).

:- dynamic user:tc_edge/2.
:- dynamic user:tc/2.
:- dynamic user:td_edge/2.
:- dynamic user:td/3.
:- dynamic user:tpd_edge/2.
:- dynamic user:tpd/4.
:- dynamic user:tspd_edge/2.
:- dynamic user:tspd/5.
:- dynamic user:w_edge/3.
:- dynamic user:wsp/3.
:- dynamic user:a_edge/3.
:- dynamic user:astar/4.
:- dynamic user:direct_dist/4.
:- dynamic user:dimensionality/1.

dotnet_available :-
    catch(
        ( process_create(path(dotnet), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, _) ),
        _, fail).

tmp_proj(Name, Dir) :-
    format(atom(Dir), '/tmp/uw_fs_gate_~w', [Name]),
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir).

run_dotnet_build(Dir, Exit, Out) :-
    setup_call_cleanup(
        process_create(path(dotnet),
            ['build', '--nologo', '-v', 'q', '-c', 'Release'],
            [cwd(Dir), stdout(pipe(SO)), stderr(pipe(SE)), process(Pid)]),
        ( read_string(SO, _, S1), read_string(SE, _, S2),
          process_wait(Pid, exit(Exit)),
          string_concat(S1, S2, Out) ),
        ( catch(close(SO), _, true), catch(close(SE), _, true) )).

run_dotnet_run(Dir, Exit, Out) :-
    setup_call_cleanup(
        process_create(path(dotnet),
            ['run', '--no-build', '-c', 'Release', '--no-launch-profile'],
            [cwd(Dir),
             environment(['DOTNET_NOLOGO'='1', 'DOTNET_ROLL_FORWARD'='Major']),
             stdout(pipe(SO)), stderr(pipe(SE)), process(Pid)]),
        ( read_string(SO, _, S1), read_string(SE, _, S2),
          process_wait(Pid, exit(Exit)),
          string_concat(S1, S2, Out) ),
        ( catch(close(SO), _, true), catch(close(SE), _, true) )).

% Bodies must be unqualified so detect_recursive_kernel/4 sees
% edge(X,Y) / edge(X,Z), Pred(Z,Y) — not user:edge(...).
assert_tc_program :-
    retractall(user:tc_edge(_, _)),
    retractall(user:tc(_, _)),
    % Graph: a->b->c->d, plus cycle c->a, and unreachable z.
    assertz(user:tc_edge(a, b)),
    assertz(user:tc_edge(b, c)),
    assertz(user:tc_edge(c, d)),
    assertz(user:tc_edge(c, a)),
    assertz((user:tc(X, Y) :- tc_edge(X, Y))),
    assertz((user:tc(X, Y) :- tc_edge(X, Z), tc(Z, Y))).

assert_td_program :-
    retractall(user:td_edge(_, _)),
    retractall(user:td(_, _, _)),
    assertz(user:td_edge(a, b)),
    assertz(user:td_edge(b, c)),
    assertz((user:td(X, Y, 1) :- td_edge(X, Y))),
    assertz((user:td(X, Y, D) :- td_edge(X, Z), td(Z, Y, D1), D is D1 + 1)).

assert_tpd_program :-
    retractall(user:tpd_edge(_, _)),
    retractall(user:tpd(_, _, _, _)),
    assertz(user:tpd_edge(a, b)),
    assertz(user:tpd_edge(b, c)),
    assertz((user:tpd(X, Y, X, 1) :- tpd_edge(X, Y))),
    assertz((user:tpd(X, Y, P, D) :-
                tpd_edge(X, Z), tpd(Z, Y, P, D1), D is D1 + 1)).

assert_tspd_program :-
    retractall(user:tspd_edge(_, _)),
    retractall(user:tspd(_, _, _, _, _)),
    assertz(user:tspd_edge(a, b)),
    assertz(user:tspd_edge(b, c)),
    % Base: step==target, parent==source. Rec: head step is the edge Mid.
    assertz((user:tspd(X, Y, Y, X, 1) :- tspd_edge(X, Y))),
    assertz((user:tspd(X, Y, Z, P, D) :-
                tspd_edge(X, Z), tspd(Z, Y, _, P, D1), D is D1 + 1)).

assert_wsp_program :-
    retractall(user:w_edge(_, _, _)),
    retractall(user:wsp(_, _, _)),
    assertz(user:w_edge(a, b, 1.0)),
    assertz(user:w_edge(b, c, 2.0)),
    assertz((user:wsp(X, Y, W) :- w_edge(X, Y, W))),
    assertz((user:wsp(X, Y, Total) :-
                w_edge(X, Z, W), wsp(Z, Y, Rest), Total is Rest + W)).

assert_astar_program :-
    retractall(user:a_edge(_, _, _)),
    retractall(user:astar(_, _, _, _)),
    retractall(user:direct_dist(_, _, _, _)),
    retractall(user:dimensionality(_)),
    assertz(user:dimensionality(2)),
    assertz(user:direct_dist(a, c, 2, 1.0)),
    assertz(user:a_edge(a, b, 1.0)),
    assertz(user:a_edge(b, c, 1.0)),
    assertz((user:astar(X, Y, Dim, W) :- a_edge(X, Y, W), dimensionality(Dim))),
    assertz((user:astar(X, Y, Dim, Total) :-
                a_edge(X, Z, W), astar(Z, Y, Dim, Rest),
                Total is Rest + W)).

write_native_tc_driver(ProgPath) :-
    % Custom Program.fs: wires foreignPreds + WcFfiFacts for edge, then
    % exercises callForeign (native path) — not the empty-foreignPreds
    % conformance_main driver.
    Driver = "module Program

open System
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

let mkAtoms () =
    // Stable intern ids for the fixture graph.
    let pairs = [(\"a\", 1); (\"b\", 2); (\"c\", 3); (\"d\", 4); (\"z\", 5)]
    Map.ofList pairs, Map.ofList (pairs |> List.map (fun (s, i) -> (i, s)))

let mkEdgeFacts (intern: Map<string, int>) =
    let edges = [(\"a\", \"b\"); (\"b\", \"c\"); (\"c\", \"d\"); (\"c\", \"a\")]
    let grouped =
        edges
        |> List.map (fun (f, t) -> Map.find f intern, Map.find t intern)
        |> List.groupBy fst
        |> List.map (fun (k, vs) -> k, vs |> List.map snd)
    Map.ofList [(\"tc_edge\", Map.ofList grouped)]

let mkContext () =
    let foreignPreds = [\"tc/2\"]
    let resolvedCode =
        resolveCallInstrs allLabels foreignPreds (Array.toList allCode)
        |> List.toArray
    let intern, deintern = mkAtoms ()
    { WcCode              = resolvedCode
      WcLabels            = allLabels
      WcForeignFacts      = Map.empty
      WcFfiFacts          = mkEdgeFacts intern
      WcFfiWeightedFacts  = Map.empty
      WcAtomIntern        = intern
      WcAtomDeintern      = deintern
      WcForeignConfig     = Map.empty
      WcLoweredPredicates = Map.empty
      WcLookupSources     = Map.empty
      WcCancellationToken = None }

let mkState (regs: Value array) : WamState =
    { WsPC         = 0
      WsRegs       = regs
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

let collectTargets (ctx: WamContext) (source: string) (boundTarget: string option) : string list =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom source
    match boundTarget with
    | Some t -> regs.[2] <- Atom t
    | None   -> regs.[2] <- Unbound 100
    let s0 = mkState regs
    match callForeign ctx \"tc/2\" s0 with
    | None -> []
    | Some s1 ->
        let readTarget (s: WamState) =
            match getReg 2 s with
            | Some (Atom a) -> a
            | _ -> \"?\"
        // First solution from callForeign; further solutions via FFIStreamRetry backtrack.
        let rec gather (s: WamState) (acc: string list) =
            let tgt = readTarget s
            match backtrack s with
            | Some s2 -> gather s2 (tgt :: acc)
            | None -> List.rev (tgt :: acc)
        gather s1 []

[<EntryPoint>]
let main _argv =
    let ctx = mkContext ()
    // foreignPreds=[\"tc/2\"] + callForeign proves the native executeForeign path
    // (conformance_main leaves foreignPreds empty and cannot show this).

    let fromA = collectTargets ctx \"a\" None |> List.sort
    assertTrue \"direct+multi-hop from a\" (fromA = [\"b\"; \"c\"; \"d\"])

    let fromC = collectTargets ctx \"c\" None |> List.sort
    assertTrue \"cycle-safe from c\" (fromC = [\"a\"; \"b\"; \"d\"])

    let fromD = collectTargets ctx \"d\" None
    assertTrue \"sink d has no targets\" (fromD = [])

    let boundOk = collectTargets ctx \"a\" (Some \"c\")
    assertTrue \"bound Target c succeeds\" (boundOk = [\"c\"])

    let boundNo = collectTargets ctx \"a\" (Some \"z\")
    assertTrue \"bound unreachable z fails\" (boundNo = [])

    printfn \"RESULT %d/%d\" passes (passes + fails)
    if fails > 0 then 1 else 0
",
    setup_call_cleanup(
        open(ProgPath, write, Out, [encoding(utf8)]),
        write(Out, Driver),
        close(Out)).

%% ---------- characterization of existing layers ----------

:- begin_tests(wam_fsharp_kernel_gate_tc).

test(capability_allowlist_includes_shipped_and_tc2) :-
    wam_fsharp_native_kernel_kind(category_ancestor),
    wam_fsharp_native_kernel_kind(bidirectional_ancestor),
    wam_fsharp_native_kernel_kind(transitive_closure2),
    \+ wam_fsharp_native_kernel_kind(transitive_distance3),
    \+ wam_fsharp_native_kernel_kind(weighted_shortest_path3).

test(existing_closure_layers_present) :-
    % Characterization: do not invent TC support from scratch.
    % Tests are invoked from the repo root (see docs/TESTING.md).
    Tw = 'templates/targets/fsharp_wam/',
    atom_concat(Tw, 'kernel_category_ancestor.fs.mustache', Cat),
    atom_concat(Tw, 'kernel_bidirectional_ancestor.fs.mustache', Bidir),
    atom_concat(Tw, 'lmdb_fact_source.fs.mustache', Lmdb),
    atom_concat(Tw, 'kernel_transitive_closure.fs.mustache', Tc),
    read_file_to_string(Cat, CatS, []),
    read_file_to_string(Bidir, BidirS, []),
    read_file_to_string(Lmdb, LmdbS, []),
    read_file_to_string(Tc, TcS, []),
    assertion(sub_string(CatS, _, _, _, "nativeKernel_category_ancestor")),
    assertion(sub_string(BidirS, _, _, _, "nativeKernel_bidirectional_ancestor")),
    assertion(sub_string(LmdbS, _, _, _, "reachableToRootVia")),
    assertion(sub_string(LmdbS, _, _, _, "reachableToRoot")),
    assertion(sub_string(TcS, _, _, _, "let nativeKernel_transitive_closure")),
    % TC2 must not define the demand-pruning helpers (comments may mention them).
    assertion(\+ sub_string(TcS, _, _, _, "let reachableToRoot")),
    !.

test(detector_fires_tc2_and_gate_accepts) :-
    assert_tc_program,
    findall(tc(X,Y)-Body, clause(user:tc(X, Y), Body), Clauses),
    detect_recursive_kernel(tc, 2, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(transitive_closure2, _, _)),
    assertion(wam_fsharp_native_kernel_supported(Kernel)),
    !.

test(wam_fallback_control_builds,
     [condition(dotnet_available)]) :-
    assert_tc_program,
    tmp_proj(wam_tc, Dir),
    write_wam_fsharp_project(
        [user:tc/2, user:tc_edge/2],
        [no_kernels(true), module_name('uw_fs_wam_tc'), conformance_main(true)],
        Dir),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    read_file_to_string(RT, RTS, []),
    assertion(\+ sub_string(RTS, _, _, _, "nativeKernel_transitive_closure")),
    run_dotnet_build(Dir, Exit, Out),
    assertion(Exit =:= 0),
    ( Exit =:= 0 -> true ; format(user_error, '~w~n', [Out]), fail ),
    !.

test(native_tc2_codegen_and_build,
     [condition(dotnet_available)]) :-
    assert_tc_program,
    tmp_proj(native_tc, Dir),
    write_wam_fsharp_project(
        [user:tc/2, user:tc_edge/2],
        [module_name('uw_fs_native_tc')],
        Dir),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    directory_file_path(Dir, 'Program.fs', Prog),
    read_file_to_string(RT, RTS, []),
    assertion(sub_string(RTS, _, _, _, "nativeKernel_transitive_closure")),
    assertion(sub_string(RTS, _, _, _, "| \"tc/2\" ->")),
    assertion(\+ sub_string(RTS, _, _, _, "template not found")),
    assertion(\+ sub_string(RTS, _, _, _, "no F# template available")),
    % Bound-Target filter present in executeForeign stream binder.
    assertion(sub_string(RTS, _, _, _, "WcAtomIntern")),
    write_native_tc_driver(Prog),
    run_dotnet_build(Dir, Exit, Out),
    assertion(Exit =:= 0),
    ( Exit =:= 0 -> true
    ; format(user_error, 'native_tc2 build failed:~n~w~n', [Out]), fail
    ),
    run_dotnet_run(Dir, RunExit, RunOut),
    assertion(RunExit =:= 0),
    assertion(sub_string(RunOut, _, _, _, "RESULT ")),
    (   sub_string(RunOut, _, _, _, "RESULT 5/5")
    ->  true
    ;   format(user_error, 'native_tc2 run:~n~w~n', [RunOut]), fail
    ),
    !.

%% Unsupported detected patterns must compile via WAM fallback.
fallback_kind_builds(Kind, Setup, Preds, ForbiddenNative) :-
    call(Setup),
    format(atom(Name), 'fb_~w', [Kind]),
    tmp_proj(Name, Dir),
    write_wam_fsharp_project(
        Preds,
        [module_name(Name), conformance_main(true)],
        Dir),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    read_file_to_string(RT, RTS, []),
    assertion(\+ sub_string(RTS, _, _, _, ForbiddenNative)),
    assertion(\+ sub_string(RTS, _, _, _, "template not found")),
    run_dotnet_build(Dir, Exit, Out),
    assertion(Exit =:= 0),
    ( Exit =:= 0 -> true
    ; format(user_error, '~w fallback build failed:~n~w~n', [Kind, Out]), fail
    ),
    !.

test(fallback_transitive_distance3, [condition(dotnet_available)]) :-
    fallback_kind_builds(transitive_distance3, assert_td_program,
                         [user:td/3, user:td_edge/2],
                         "nativeKernel_transitive_distance").

test(fallback_transitive_parent_distance4, [condition(dotnet_available)]) :-
    fallback_kind_builds(transitive_parent_distance4, assert_tpd_program,
                         [user:tpd/4, user:tpd_edge/2],
                         "nativeKernel_transitive_parent_distance").

test(fallback_transitive_step_parent_distance5, [condition(dotnet_available)]) :-
    fallback_kind_builds(transitive_step_parent_distance5, assert_tspd_program,
                         [user:tspd/5, user:tspd_edge/2],
                         "nativeKernel_transitive_step_parent_distance").

test(fallback_weighted_shortest_path3, [condition(dotnet_available)]) :-
    fallback_kind_builds(weighted_shortest_path3, assert_wsp_program,
                         [user:wsp/3, user:w_edge/3],
                         "nativeKernel_weighted_shortest_path").

test(fallback_astar_shortest_path4, [condition(dotnet_available)]) :-
    fallback_kind_builds(astar_shortest_path4, assert_astar_program,
                         [user:astar/4, user:a_edge/3, user:direct_dist/4, user:dimensionality/1],
                         "nativeKernel_astar_shortest_path").

:- end_tests(wam_fsharp_kernel_gate_tc).
