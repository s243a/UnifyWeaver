:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_td3_contract_parity.pl — fleet-wide transitive_distance3
% contract (dist+) parity suite.
%
% Contract: docs/design/WAM_TRANSITIVE_DISTANCE3_CONTRACT.md
% Oracle:   tests/fixtures/td3_contract_oracle.pl
%
% Usage (from repo root):
%   swipl -q -g run_tests -t halt tests/test_wam_td3_contract_parity.pl

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module(library(readutil)).
:- use_module(library(process)).
:- use_module(library(debug), [assertion/1]).

:- use_module('fixtures/td3_contract_oracle', [
    td3_oracle_pairs/3,
    td3_oracle_distance/4,
    td3_fixture/3,
    td3_fixture_expected/3
]).

:- use_module('../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3,
               wam_fsharp_native_kernel_kind/1,
               wam_fsharp_native_kernel_supported/1]).
:- use_module('../src/unifyweaver/targets/wam_c_target',
              [write_wam_c_project/3]).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3,
               compile_wam_runtime_to_rust/2]).
:- use_module('../src/unifyweaver/core/recursive_kernel_detection',
              [detect_recursive_kernel/4]).

:- dynamic user:td_edge/2.
:- dynamic user:td/3.
:- dynamic user:td_tail/2.
:- dynamic user:td_after/2.
:- dynamic user:td_cut/2.
:- dynamic user:td_call_after/2.
:- dynamic user:td_parent/2.
:- dynamic user:tc_distance/3.

dotnet_available :-
    catch(
        ( process_create(path(dotnet), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

gcc_available :-
    catch(
        ( process_create(path(gcc), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

tmp_dir(Tag, Dir) :-
    get_time(T),
    Stamp is round(T * 1000000),
    format(atom(Dir), '/tmp/uw_td3_~w_~w', [Tag, Stamp]),
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir).

read_file_string(Path, String) :-
    read_file_to_string(Path, String, []).

assert_td_cycle_program :-
    retractall(user:td_edge(_, _)),
    retractall(user:td(_, _, _)),
    retractall(user:td_tail(_, _)),
    retractall(user:td_after(_, _)),
    retractall(user:td_cut(_, _)),
    retractall(user:td_call_after(_, _)),
    assertz(user:td_edge(a, b)),
    assertz(user:td_edge(b, c)),
    assertz(user:td_edge(c, d)),
    assertz(user:td_edge(c, a)),
    assertz((user:td(X, Y, 1) :- td_edge(X, Y))),
    assertz((user:td(X, Y, D) :- td_edge(X, Z), td(Z, Y, D1), D is D1 + 1)),
    % Force both interpreter foreign opcodes around the two-output stream.
    % td_tail/2 compiles its last goal to ExecuteForeign; td_call_after/2
    % retains a continuation and therefore compiles to CallForeign.
    assertz((user:td_tail(Y, D) :- td(a, Y, D))),
    assertz((user:td_after(Y, D) :- td_tail(Y, D), Y \== b)),
    assertz((user:td_cut(Y, D) :- td_tail(Y, D), !)),
    assertz((user:td_call_after(Y, D) :- td(a, Y, D), Y \== b)).

assert_td_chain_program :-
    retractall(user:td_edge(_, _)),
    retractall(user:td(_, _, _)),
    assertz(user:td_edge(a, b)),
    assertz(user:td_edge(b, c)),
    assertz(user:td_edge(c, d)),
    assertz((user:td(X, Y, 1) :- td_edge(X, Y))),
    assertz((user:td(X, Y, D) :- td_edge(X, Z), td(Z, Y, D1), D is D1 + 1)).

assert_c_td_program :-
    retractall(user:td_parent(_, _)),
    retractall(user:tc_distance(_, _, _)),
    assertz(user:td_parent(a, b)),
    assertz(user:td_parent(b, c)),
    assertz(user:td_parent(c, d)),
    assertz(user:td_parent(c, a)),
    assertz((user:tc_distance(X, Y, 1) :- td_parent(X, Y))),
    assertz((user:tc_distance(X, Y, D) :-
                td_parent(X, Z), tc_distance(Z, Y, D0), D is D0 + 1)).

% ============================================================
% 1. Oracle vs literal expectations
% ============================================================

:- begin_tests(td3_oracle).

test(literal_expectations_are_complete) :-
    forall(
        ( td3_fixture(Name, _Edges, Sources),
          member(Src, Sources)
        ),
        assertion(td3_fixture_expected(Name, Src, _))
    ).

test(oracle_matches_literal_expectations) :-
    forall(
        ( td3_fixture(Name, Edges, Sources),
          member(Src, Sources),
          td3_fixture_expected(Name, Src, Expected)
        ),
        ( td3_oracle_pairs(Edges, Src, Got),
          assertion(Got == Expected)
        )
    ).

test(bound_distance_lookup) :-
    td3_oracle_distance([a-b, b-c, c-t, a-t], a, t, 1),
    td3_oracle_distance([a-a, a-b], a, a, 1),
    td3_oracle_distance([a-b, b-a], a, a, 2),
    \+ td3_oracle_distance([a-b, b-c], a, a, _).

test(acyclic_source_not_emitted) :-
    td3_fixture_expected(chain, a, Pairs),
    \+ memberchk(a-_, Pairs).

:- end_tests(td3_oracle).

% ============================================================
% 2. Structural dist+ pattern checks
% ============================================================

:- begin_tests(td3_structural_distplus).

test(fsharp_mustache_distplus) :-
    read_file_string(
        'templates/targets/fsharp_wam/kernel_transitive_distance.fs.mustache', S),
    assertion(sub_string(S, _, _, _, "dist+")),
    assertion(sub_string(S, _, _, _, "let nativeKernel_transitive_distance")),
    assertion(sub_string(S, _, _, _, "let visited = System.Collections.Generic.HashSet<int>()")),
    assertion(sub_string(S, _, _, _, "let mutable frontier = [(source, 0)]")),
    assertion(\+ sub_string(S, _, _, _, "visited.Add(source)")).

test(fsharp_allowlist_includes_td3) :-
    assertion(wam_fsharp_native_kernel_kind(transitive_distance3)),
    assertion(wam_fsharp_native_kernel_supported(
        recursive_kernel(transitive_distance3, probe/0, []))),
    assertion(wam_fsharp_native_kernel_kind(weighted_shortest_path3)),
    assertion(\+ wam_fsharp_native_kernel_kind(astar_shortest_path4)).

test(haskell_mustache_distplus) :-
    read_file_string(
        'templates/targets/haskell_wam/kernel_transitive_distance.hs.mustache', S),
    assertion(sub_string(S, _, _, _, "dist+")),
    assertion(sub_string(S, _, _, _, "go [(source, 0)] IS.empty []")),
    assertion(\+ sub_string(S, _, _, _, "Don't emit the source")).

test(rust_bfs_not_per_path) :-
    compile_wam_runtime_to_rust([], Code),
    atom_string(Code, S),
    assertion(sub_string(S, _, _, _, "collect_native_transitive_distance_results")),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_TRANSITIVE_DISTANCE3_CONTRACT.md")),
    assertion(sub_string(S, _, _, _, "VecDeque<(String, i64)>")),
    assertion(sub_string(S, _, _, _, "let mut seen: HashSet<String> = HashSet::new();")).

test(go_does_not_seed_visited_with_source) :-
    read_file_string('src/unifyweaver/targets/wam_go_target.pl', S),
    assertion(sub_string(S, _, _, _, "collectNativeTransitiveDistanceResults")),
    assertion(sub_string(S, _, _, _,
        "// dist+ (docs/design/WAM_TRANSITIVE_DISTANCE3_CONTRACT.md)")),
    assertion(sub_string(S, _, _, _, "visited := make(map[string]bool)")).

test(scala_does_not_seed_seen_with_source) :-
    read_file_string('src/unifyweaver/targets/wam_scala_target.pl', S),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_TRANSITIVE_DISTANCE3_CONTRACT.md")),
    assertion(sub_string(S, _, _, _,
        "val seen = scala.collection.mutable.HashSet[WamTerm]()")).

test(c_collects_and_streams_pairs) :-
    read_file_string('src/unifyweaver/targets/wam_c_target.pl', S),
    assertion(sub_string(S, _, _, _, "wam_collect_transitive_distance")),
    assertion(sub_string(S, _, _, _, "wam_bind_foreign_pair_stream")),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_TRANSITIVE_DISTANCE3_CONTRACT.md")).

test(r_does_not_seed_visited) :-
    read_file_string('templates/targets/r_wam/runtime.R.mustache', S),
    assertion(sub_string(S, _, _, _, "WamRuntime$transitive_distance3")),
    assertion(sub_string(S, _, _, _, "do NOT seed")).

test(llvm_stream_and_bound_self_are_distplus) :-
    read_file_string('templates/targets/llvm_wam/state.ll.mustache', S),
    assertion(sub_string(S, _, _, _, "wam_td3_rplus_distance")),
    assertion(sub_string(S, _, _, _, "do NOT mark start visited")),
    assertion(sub_string(S, _, _, _, "result_reg=255")),
    assertion(sub_string(S, _, _, _,
        "%ids_in_range = and i1 %start_in_range, %target_in_range")),
    assertion(sub_string(S, _, _, _,
        "br i1 %start_in_range, label %check_distance_filter, label %stream_fail_early")),
    findall(I,
        sub_string(S, I, _, _, "%q_slots = add i64 %n, 1"),
        QueueCapacityFixes),
    % TD3 rplus + TD3 stream + the two existing TC2 R+ traversals.
    assertion(QueueCapacityFixes = [_, _, _, _]).

test(elixir_bfs_not_per_path) :-
    read_file_string('src/unifyweaver/targets/wam_elixir_target.pl', S),
    assertion(sub_string(S, _, _, _, "WamRuntime.GraphKernel.TransitiveDistance")),
    assertion(sub_string(S, _, _, _, "Finite BFS")),
    assertion(sub_string(S, _, _, _, "MapSet.new()")).

:- end_tests(td3_structural_distplus).

% ============================================================
% 3. Native dispatch proof
% ============================================================

:- begin_tests(td3_native_dispatch).

test(detector_fires_transitive_distance3) :-
    assert_td_cycle_program,
    findall(td(X, Y, D)-Body, clause(user:td(X, Y, D), Body), Clauses),
    detect_recursive_kernel(td, 3, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(transitive_distance3, _, _)).

test(fsharp_native_td3_dispatch_emitted) :-
    assert_td_cycle_program,
    tmp_dir(fs_dispatch, Dir),
    write_wam_fsharp_project(
        [user:td/3, user:td_edge/2],
        [module_name('uw_td3_dispatch')],
        Dir),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    read_file_string(RT, RTS),
    assertion(sub_string(RTS, _, _, _, "nativeKernel_transitive_distance")),
    assertion(sub_string(RTS, _, _, _, "| \"td/3\" ->")),
    assertion(sub_string(RTS, _, _, _, "FFIStreamRetry")),
    assertion(sub_string(RTS, _, _, _, "Integer i -> i = rv_")),
    !.

test(rust_foreign_kind_registered_for_td3) :-
    assert_td_cycle_program,
    tmp_dir(rs_dispatch, Dir),
    write_wam_rust_project(
        [user:td/3, user:td_edge/2],
        [module_name('uw_td3_rs_dispatch'), foreign_lowering(true)],
        Dir),
    directory_file_path(Dir, 'src/lib.rs', Lib),
    read_file_string(Lib, S),
    assertion(sub_string(S, _, _, _, "transitive_distance3")),
    assertion(sub_string(S, _, _, _, "register_foreign_native_kind(\"td/3\"")),
    !.

test(c_registers_td3_handler) :-
    assert_c_td_program,
    tmp_dir(c_dispatch, Dir),
    write_wam_c_project([user:tc_distance/3], [], Dir),
    directory_file_path(Dir, 'lib.c', Lib),
    read_file_string(Lib, LibS),
    assertion(sub_string(LibS, _, _, _,
        "wam_register_transitive_distance_kernel")),
    directory_file_path(Dir, 'wam_runtime.c', RT),
    read_file_string(RT, S),
    assertion(sub_string(S, _, _, _, "wam_collect_transitive_distance")),
    !.

:- end_tests(td3_native_dispatch).

% ============================================================
% 4. Executable smokes
% ============================================================

:- begin_tests(td3_executable).

test(fsharp_cycle_distplus_e2e, [condition(dotnet_available)]) :-
    assert_td_cycle_program,
    tmp_dir(fs_e2e, Dir),
    once(write_wam_fsharp_project(
        [ user:td/3, user:td_edge/2,
          user:td_tail/2, user:td_after/2, user:td_cut/2,
          user:td_call_after/2
        ],
        [module_name('uw_td3_e2e')],
        Dir)),
    directory_file_path(Dir, 'Program.fs', Prog),
    td3_write_fsharp_driver(Prog),
    run_dotnet_build(Dir, BuildExit, BuildOut),
    ( BuildExit =:= 0 -> true
    ; format(user_error, 'fsharp td3 e2e build:~n~w~n', [BuildOut]),
      assertion(BuildExit =:= 0), fail
    ),
    run_dotnet_run(Dir, RunExit, RunOut),
    ( RunExit =:= 0 -> true
    ; format(user_error, 'fsharp td3 e2e run:~n~w~n', [RunOut]),
      assertion(RunExit =:= 0), fail
    ),
    assertion(sub_string(RunOut, _, _, _, "OK stream_from_a")),
    assertion(sub_string(RunOut, _, _, _, "OK bound_source_cycle")),
    assertion(sub_string(RunOut, _, _, _, "OK pairing_retry")),
    assertion(sub_string(RunOut, _, _, _, "OK aliased_outputs_fail")),
    assertion(sub_string(RunOut, _, _, _, "OK execute_foreign_retry")),
    assertion(sub_string(RunOut, _, _, _, "OK call_foreign_retry")),
    assertion(sub_string(RunOut, _, _, _, "OK cut_after_foreign")),
    !.

test(c_td3_stream_and_bound, [condition(gcc_available)]) :-
    assert_c_td_program,
    tmp_dir(c_e2e, Dir),
    write_wam_c_project([user:tc_distance/3], [], Dir),
    directory_file_path(Dir, 'wam_runtime.c', RuntimePath),
    directory_file_path(Dir, 'lib.c', LibPath),
    directory_file_path(Dir, 'main.c', MainPath),
    directory_file_path(Dir, 'td3_c_smoke', ExePath),
    td3_c_cycle_main(MainCode),
    setup_call_cleanup(
        open(MainPath, write, Out, [encoding(utf8)]),
        write(Out, MainCode),
        close(Out)),
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    format(atom(Cmd),
        'gcc -O0 -std=c11 -I~w ~w ~w ~w -o ~w 2>~w/gcc.err',
        [IncludeDir, RuntimePath, LibPath, MainPath, ExePath, Dir]),
    shell(Cmd, GccExit),
    ( GccExit =:= 0 -> true
    ; directory_file_path(Dir, 'gcc.err', Err),
      ( exists_file(Err) -> read_file_string(Err, E),
        format(user_error, 'gcc failed:~n~w~n', [E]) ; true ),
      fail
    ),
    shell(ExePath, RunExit),
    assertion(RunExit =:= 0),
    !.

test(rust_collect_distplus_unit, [condition(cargo_available)]) :-
    assert_td_cycle_program,
    tmp_dir(rs_e2e, Dir),
    write_wam_rust_project(
        [user:td/3, user:td_edge/2],
        [module_name('uw_td3_rs'), foreign_lowering(true)],
        Dir),
    td3_append_rust_unit(Dir),
    format(atom(Cmd),
        'cd ~w && cargo test --quiet td3_distplus_contract -- --nocapture >~w/cargo.out 2>~w/cargo.err',
        [Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; directory_file_path(Dir, 'cargo.err', ErrPath),
      directory_file_path(Dir, 'cargo.out', OutPath),
      ( exists_file(ErrPath) -> read_file_string(ErrPath, Err) ; Err = "" ),
      ( exists_file(OutPath) -> read_file_string(OutPath, Out) ; Out = "" ),
      format(user_error, 'rust td3 unit failed:~n~w~n~w~n', [Out, Err]),
      fail
    ),
    !.

:- end_tests(td3_executable).

% ============================================================
% 5. Acyclic no_kernels fallback
% ============================================================

:- begin_tests(td3_no_kernels_acyclic).

test(fsharp_no_kernels_fallback_builds, [condition(dotnet_available)]) :-
    assert_td_chain_program,
    tmp_dir(fs_nk, Dir),
    once(write_wam_fsharp_project(
        [user:td/3, user:td_edge/2],
        [no_kernels(true), module_name('uw_td3_nk'), conformance_main(true)],
        Dir)),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    read_file_string(RT, RTS),
    assertion(\+ sub_string(RTS, _, _, _, "nativeKernel_transitive_distance")),
    run_dotnet_build(Dir, Exit, Out),
    ( Exit =:= 0 -> true
    ; format(user_error, 'fsharp no-kernels build:~n~w~n', [Out]),
      assertion(Exit =:= 0), fail
    ),
    !.

:- end_tests(td3_no_kernels_acyclic).

% ============================================================
% Helpers
% ============================================================

run_dotnet_build(Dir, Exit, Out) :-
    setup_call_cleanup(
        process_create(path(dotnet),
            ['build', '--nologo', '-v', 'q', '-c', 'Release'],
            [cwd(Dir),
             environment([
                 'DOTNET_NOLOGO'='1',
                 'DOTNET_ROLL_FORWARD'='Major'
             ]),
             stdout(pipe(SO)), stderr(pipe(SE)), process(Pid)]),
        ( read_string(SO, _, S1), read_string(SE, _, S2),
          process_wait(Pid, Status),
          dotnet_status_exit(Status, Exit),
          string_concat(S1, S2, Out) ),
        ( catch(close(SO), _, true), catch(close(SE), _, true) )).

run_dotnet_run(Dir, Exit, Out) :-
    setup_call_cleanup(
        process_create(path(dotnet),
            ['run', '--no-build', '-c', 'Release', '--no-launch-profile', '--'],
            [cwd(Dir),
             environment([
                 'DOTNET_NOLOGO'='1',
                 'DOTNET_ROLL_FORWARD'='Major'
             ]),
             stdout(pipe(SO)), stderr(pipe(SE)), process(Pid)]),
        ( read_string(SO, _, S1), read_string(SE, _, S2),
          process_wait(Pid, Status),
          dotnet_status_exit(Status, Exit),
          string_concat(S1, S2, Out) ),
        ( catch(close(SO), _, true), catch(close(SE), _, true) )).

dotnet_status_exit(exit(Code), Code).
dotnet_status_exit(killed(Signal), Code) :-
    Code is 128 + Signal.

td3_write_fsharp_driver(ProgPath) :-
    Driver =
"module Program

open System
open WamTypes
open WamRuntime
open Predicates

let mutable passes = 0
let mutable fails = 0

let assertTrue (name: string) (cond: bool) =
    if cond then
        passes <- passes + 1
        printfn \"OK %s\" name
    else
        fails <- fails + 1
        printfn \"FAIL %s\" name

let mkAtoms () =
    let pairs = [(\"a\", 1); (\"b\", 2); (\"c\", 3); (\"d\", 4)]
    Map.ofList pairs, Map.ofList (pairs |> List.map (fun (s, i) -> (i, s)))

let mkEdgeFacts (intern: Map<string, int>) =
    let edges = [(\"a\", \"b\"); (\"b\", \"c\"); (\"c\", \"d\"); (\"c\", \"a\")]
    let grouped =
        edges
        |> List.map (fun (f, t) -> Map.find f intern, Map.find t intern)
        |> List.groupBy fst
        |> List.map (fun (k, vs) -> k, vs |> List.map snd)
    Map.ofList [(\"td_edge\", Map.ofList grouped)]

let mkContext () =
    let foreignPreds = [\"td/3\"]
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

let collectPairs (ctx: WamContext) (source: string)
                 (boundT: string option) (boundD: int option) =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom source
    match boundT with
    | Some t -> regs.[2] <- Atom t
    | None -> regs.[2] <- Unbound 100
    match boundD with
    | Some d -> regs.[3] <- Integer d
    | None -> regs.[3] <- Unbound 101
    match callForeign ctx \"td/3\" (mkState regs) with
    | None -> []
    | Some s1 ->
        let readPair (s: WamState) =
            let t =
                match getReg 2 s with
                | Some (Atom a) -> a
                | _ -> \"?\"
            let d =
                match getReg 3 s with
                | Some (Integer i) -> i
                | _ -> -1
            t, d
        let rec gather (s: WamState) (acc: (string * int) list) =
            let p = readPair s
            match backtrack s with
            | Some s2 -> gather s2 (p :: acc)
            | None -> List.rev (p :: acc)
        gather s1 []

let runPairWrapper (ctx: WamContext) (pred: string) : WamState option =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Unbound 200
    regs.[2] <- Unbound 201
    dispatchCall ctx pred (mkState regs)

let readWrapperPair (s: WamState) =
    // Continuation code is free to reuse A1/A2. Read the original top-level
    // output variables through their stable ids, as a Prolog caller would.
    match derefVar s.WsBindings (Unbound 200),
          derefVar s.WsBindings (Unbound 201) with
    | Atom t, Integer d -> Some (t, d)
    | _ -> None

let aliasedOutputsFail (ctx: WamContext) =
    // td(a, X, X) cannot unify an atom Target and integer Distance into X.
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom \"a\"
    regs.[2] <- Unbound 150
    regs.[3] <- Unbound 150
    callForeign ctx \"td/3\" (mkState regs) |> Option.isNone

[<EntryPoint>]
let main _argv =
    let ctx = mkContext ()
    // cycle_exit-like: a->b->c->d + c->a
    let fromA = collectPairs ctx \"a\" None None |> List.sort
    assertTrue \"stream_from_a\"
        (fromA = [(\"a\", 3); (\"b\", 1); (\"c\", 2); (\"d\", 3)])

    let boundSrc = collectPairs ctx \"a\" (Some \"a\") None
    assertTrue \"bound_source_cycle\" (boundSrc = [(\"a\", 3)])

    let boundDist = collectPairs ctx \"a\" None (Some 1)
    assertTrue \"bound_distance_1\" (boundDist = [(\"b\", 1)])

    let boundBoth = collectPairs ctx \"a\" (Some \"c\") (Some 2)
    assertTrue \"bound_both\" (boundBoth = [(\"c\", 2)])

    let boundMismatch = collectPairs ctx \"a\" (Some \"c\") (Some 1)
    assertTrue \"bound_mismatch_fails\" (boundMismatch = [])

    // Pairing on every retry: distances must match targets.
    let pairs = collectPairs ctx \"a\" None None
    let pairingOk =
        pairs |> List.forall (fun (t, d) ->
            match t, d with
            | \"b\", 1 | \"c\", 2 | \"a\", 3 | \"d\", 3 -> true
            | _ -> false)
    assertTrue \"pairing_retry\" (pairingOk && List.length pairs = 4)
    assertTrue \"aliased_outputs_fail\" (aliasedOutputsFail ctx)

    // Exercise the generated interpreter opcodes, not just callForeign.
    // td_after/2 rejects b@1 after an ExecuteForeign tail call and must
    // retry the paired stream into the outer continuation as c@2.
    let hasExecuteForeign =
        ctx.WcCode
        |> Array.exists (function ExecuteForeign \"td/3\" -> true | _ -> false)
    let executeRetry = runPairWrapper ctx \"td_after/2\"
    assertTrue \"execute_foreign_retry\"
        (hasExecuteForeign &&
         (executeRetry |> Option.exists (fun s ->
             readWrapperPair s = Some (\"c\", 2) &&
             s.WsCP = 0 && s.WsCutBar = 0 && List.isEmpty s.WsB0Stack &&
             s.WsTrailLen = List.length s.WsTrail)))

    // The non-tail wrapper proves CallForeign retains the same Target/Distance
    // pairing when its continuation rejects the first result.
    let hasCallForeign =
        ctx.WcCode
        |> Array.exists (function CallForeign (\"td/3\", 3) -> true | _ -> false)
    let callRetry = runPairWrapper ctx \"td_call_after/2\"
    assertTrue \"call_foreign_retry\"
        (hasCallForeign &&
         (callRetry |> Option.exists (fun s ->
             readWrapperPair s = Some (\"c\", 2) &&
             s.WsCP = 0 && s.WsCutBar = 0 && List.isEmpty s.WsB0Stack &&
             s.WsTrailLen = List.length s.WsTrail)))

    // A real WAM cut after the first foreign yield must discard the remaining
    // FFIStreamRetry choice point. Merely taking List.tryHead after collecting
    // the whole stream would not test cut/barrier cleanup.
    let cutResult = runPairWrapper ctx \"td_cut/2\"
    assertTrue \"cut_after_foreign\"
        (cutResult |> Option.exists (fun s ->
            readWrapperPair s = Some (\"b\", 1) &&
            s.WsCP = 0 && s.WsCutBar = 0 && List.isEmpty s.WsB0Stack &&
            s.WsCPsLen = 0 && List.isEmpty s.WsCPs &&
            s.WsTrailLen = List.length s.WsTrail &&
            Option.isNone (backtrack s)))

    let fromD = collectPairs ctx \"d\" None None
    assertTrue \"sink_d\" (fromD = [])

    printfn \"RESULT %d/%d\" passes (passes + fails)
    if fails > 0 then 1 else 0
",
    setup_call_cleanup(
        open(ProgPath, write, Out, [encoding(utf8)]),
        write(Out, Driver),
        close(Out)).

td3_c_cycle_main(Code) :-
    Code =
'#include "wam_runtime.h"
#include <string.h>

void setup_tc_distance_3(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

static void load_cycle(WamState *state) {
    wam_register_transitive_edge(state, "a", "b");
    wam_register_transitive_edge(state, "a", "b");
    wam_register_transitive_edge(state, "b", "c");
    wam_register_transitive_edge(state, "c", "d");
    wam_register_transitive_edge(state, "c", "a");
}

int main(void) {
    WamState state;

    /* Bound Source via cycle: td(a,a,D) → D=3 */
    wam_state_init(&state);
    setup_tc_distance_3(&state);
    setup_detected_wam_c_kernels(&state);
    load_cycle(&state);
    {
        WamValue args[3] = { val_atom("a"), val_atom("a"), val_unbound("D") };
        int rc = wam_run_predicate(&state, "tc_distance/3", args, 3);
        if (rc != 0 || state.P != WAM_HALT ||
            state.A[2].tag != VAL_INT || state.A[2].data.integer != 3) {
            wam_free_state(&state);
            return 11;
        }
    }
    wam_free_state(&state);

    /* Bound both: td(a,c,2) succeeds once */
    wam_state_init(&state);
    setup_tc_distance_3(&state);
    setup_detected_wam_c_kernels(&state);
    load_cycle(&state);
    {
        WamValue args[3] = { val_atom("a"), val_atom("c"), val_int(2) };
        int rc = wam_run_predicate(&state, "tc_distance/3", args, 3);
        if (rc != 0 || state.P != WAM_HALT) {
            wam_free_state(&state);
            return 12;
        }
    }
    wam_free_state(&state);

    /* Drive the handler directly, then force ordinary WAM backtracking
       through every paired stream result. The duplicate a->b edge must
       not duplicate (b,1), and Target/Distance must stay paired. */
    wam_state_init(&state);
    setup_tc_distance_3(&state);
    setup_detected_wam_c_kernels(&state);
    load_cycle(&state);
    {
        int fail_pc = state.code_size;
        state.code_size += 2;
        state.code = realloc(state.code,
                             sizeof(Instruction) * (size_t)state.code_size);
        if (!state.code) return 20;
        memset(&state.code[fail_pc], 0, sizeof(Instruction) * 2u);
        state.code[fail_pc].tag = INSTR_BUILTIN_CALL;
        state.code[fail_pc].as.pred.pred = "__td3_retry_fail/0";
        state.code[fail_pc].as.pred.arity = 0;
        state.code[fail_pc + 1].tag = INSTR_PROCEED;

        state.P = fail_pc;
        state.CP = WAM_HALT;
        state.A[0] = val_atom("a");
        state.A[1] = val_unbound("T");
        state.A[2] = val_unbound("D");
        if (!wam_execute_foreign_predicate(
                &state, "tc_distance/3", 3)) {
            wam_free_state(&state);
            return 21;
        }

        const char *expected_targets[4] = { "b", "c", "d", "a" };
        const long expected_distances[4] = { 1, 2, 3, 3 };
        int seen[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < 4; i++) {
            if (i > 0) {
                state.P = fail_pc;
                if (wam_run(&state) != 0) {
                    wam_free_state(&state);
                    return 22 + i;
                }
            }
            WamValue *target = wam_deref_ptr(&state, &state.A[1]);
            WamValue *distance = wam_deref_ptr(&state, &state.A[2]);
            if (target->tag != VAL_ATOM || distance->tag != VAL_INT) {
                wam_free_state(&state);
                return 30 + i;
            }
            int match = -1;
            for (int j = 0; j < 4; j++) {
                if (strcmp(target->data.atom, expected_targets[j]) == 0 &&
                    distance->data.integer == expected_distances[j]) {
                    match = j;
                    break;
                }
            }
            if (match < 0 || seen[match]) {
                wam_free_state(&state);
                return 34 + i;
            }
            seen[match] = 1;
        }
        state.P = fail_pc;
        if (wam_run(&state) != WAM_HALT || state.B != 0) {
            wam_free_state(&state);
            return 40;
        }
        for (int i = 0; i < 4; i++) {
            if (!seen[i]) {
                wam_free_state(&state);
                return 41 + i;
            }
        }
    }
    wam_free_state(&state);

    /* Sink d fails */
    wam_state_init(&state);
    setup_tc_distance_3(&state);
    setup_detected_wam_c_kernels(&state);
    load_cycle(&state);
    {
        WamValue args[3] = {
            val_atom("d"), val_unbound("T"), val_unbound("D")
        };
        int rc = wam_run_predicate(&state, "tc_distance/3", args, 3);
        if (rc == 0 && state.P == WAM_HALT) {
            wam_free_state(&state);
            return 17;
        }
    }
    wam_free_state(&state);
    return 0;
}
'.

td3_append_rust_unit(Dir) :-
    directory_file_path(Dir, 'src/lib.rs', Path),
    read_file_string(Path, Existing),
    Unit =
"

#[cfg(test)]
mod td3_distplus_contract {
    use super::*;
    use std::collections::HashMap;
    use state::WamState;

    #[test]
    fn unequal_paths_pick_shortest() {
        let mut vm = WamState::new(Vec::new(), HashMap::new());
        vm.register_indexed_atom_fact2_pairs(
            \"td_edge\",
            &[(\"a\", \"b\"), (\"b\", \"c\"), (\"c\", \"t\"), (\"a\", \"t\")],
        );
        let mut nodes = Vec::new();
        vm.collect_native_transitive_distance_results(\"a\", \"td_edge\", &mut nodes);
        nodes.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            nodes,
            vec![
                (\"b\".to_string(), 1),
                (\"c\".to_string(), 2),
                (\"t\".to_string(), 1),
            ]
        );
    }

    #[test]
    fn cycle_emits_source_at_min_positive() {
        let mut vm = WamState::new(Vec::new(), HashMap::new());
        vm.register_indexed_atom_fact2_pairs(
            \"td_edge\",
            &[(\"a\", \"b\"), (\"b\", \"c\"), (\"c\", \"d\"), (\"c\", \"a\")],
        );
        let mut nodes = Vec::new();
        vm.collect_native_transitive_distance_results(\"a\", \"td_edge\", &mut nodes);
        nodes.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            nodes,
            vec![
                (\"a\".to_string(), 3),
                (\"b\".to_string(), 1),
                (\"c\".to_string(), 2),
                (\"d\".to_string(), 3),
            ]
        );
    }

    #[test]
    fn self_loop_distance_one() {
        let mut vm = WamState::new(Vec::new(), HashMap::new());
        vm.register_indexed_atom_fact2_pairs(
            \"td_edge\",
            &[(\"a\", \"a\"), (\"a\", \"b\")],
        );
        let mut nodes = Vec::new();
        vm.collect_native_transitive_distance_results(\"a\", \"td_edge\", &mut nodes);
        nodes.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            nodes,
            vec![(\"a\".to_string(), 1), (\"b\".to_string(), 1)]
        );
    }
}
",
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        ( write(Out, Existing), write(Out, Unit) ),
        close(Out)).
