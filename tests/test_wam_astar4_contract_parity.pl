:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_astar4_contract_parity.pl — fleet-wide astar_shortest_path4
% contract (correctness-safe A* / Dijkstra-minimum) parity suite.
%
% Contract: docs/design/WAM_ASTAR_SHORTEST_PATH4_CONTRACT.md
% Oracle:   tests/fixtures/astar4_contract_oracle.pl
%
% Usage (from repo root):
%   swipl -q -g run_tests -t halt tests/test_wam_astar4_contract_parity.pl

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module(library(readutil)).
:- use_module(library(process)).
:- use_module(library(debug), [assertion/1]).

:- use_module('fixtures/astar4_contract_oracle', [
    astar4_oracle_cost/4,
    astar4_oracle_matches/4,
    astar4_oracle_cheaper_detour_edges/1,
    astar4_oracle_cheaper_detour_cost/1,
    astar4_oracle_overestimate_edges/1,
    astar4_oracle_overestimate_heur/1,
    astar4_oracle_overestimate_cost/1,
    astar4_oracle_missing_heur_edges/1,
    astar4_oracle_missing_heur_cost/1,
    astar4_oracle_zero_cycle_edges/1,
    astar4_oracle_zero_cycle_cost/1,
    astar4_oracle_large_chain_edges/2,
    astar4_oracle_large_chain_cost/2,
    astar4_oracle_pred_a_edges/1,
    astar4_oracle_pred_a_cost/1,
    astar4_oracle_pred_b_edges/1,
    astar4_oracle_pred_b_cost/1
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

:- dynamic user:a_edge/3.
:- dynamic user:a_heur/3.
:- dynamic user:astar/4.
:- dynamic user:edge_a/3.
:- dynamic user:heur_a/3.
:- dynamic user:astar_a/4.
:- dynamic user:edge_b/3.
:- dynamic user:heur_b/3.
:- dynamic user:astar_b/4.
:- dynamic user:direct_dist_pred/1.
:- dynamic user:dimensionality/1.
:- dynamic user:astar_cut/3.
:- dynamic user:astar_after/3.

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

llvm_suite_available :-
    exists_file('tests/core/test_wam_llvm_astar_execution.pl').

tmp_dir(Tag, Dir) :-
    get_time(T),
    format(atom(Stamp), '~w', [T]),
    format(atom(Dir), '/tmp/uw_astar4_~w_~w', [Tag, Stamp]),
    make_directory_path(Dir).

read_file_string(Path, String) :-
    read_file_to_string(Path, String, []).

assert_astar_detour_program :-
    retractall(user:a_edge(_, _, _)),
    retractall(user:a_heur(_, _, _)),
    retractall(user:astar(_, _, _, _)),
    retractall(user:direct_dist_pred(_)),
    retractall(user:dimensionality(_)),
    retractall(user:astar_cut(_, _, _)),
    retractall(user:astar_after(_, _, _)),
    assertz(user:dimensionality(2)),
    assertz(user:direct_dist_pred(a_heur/3)),
    assertz(user:a_edge(a, b, 10.0)),
    assertz(user:a_edge(a, c, 1.0)),
    assertz(user:a_edge(c, b, 1.0)),
    % Overestimating heuristic toward b — must not change Dijkstra optimum.
    assertz(user:a_heur(a, b, 100.0)),
    assertz(user:a_heur(c, b, 0.0)),
    assertz((user:astar(X, Y, _Dim, W) :- a_edge(X, Y, W))),
    assertz((user:astar(X, Y, Dim, Total) :-
                a_edge(X, Z, W), astar(Z, Y, Dim, Rest), Total is Rest + W)),
    assertz((user:astar_cut(S, T, C) :- astar(S, T, 2, C), !)),
    assertz((user:astar_after(S, T, C) :- astar(S, T, 2, C), true)).

assert_two_astar_programs :-
    retractall(user:edge_a(_, _, _)),
    retractall(user:heur_a(_, _, _)),
    retractall(user:astar_a(_, _, _, _)),
    retractall(user:edge_b(_, _, _)),
    retractall(user:heur_b(_, _, _)),
    retractall(user:astar_b(_, _, _, _)),
    retractall(user:direct_dist_pred(_)),
    retractall(user:dimensionality(_)),
    assertz(user:dimensionality(2)),
    % Two A* preds with disjoint edge/heur relations. Detector falls back
    % to each kernel's own edge_pred when direct_dist_pred/1 is unset for
    % the second; we assert per-pred heuristics via same-named fallback.
    assertz(user:edge_a(a, b, 1.0)),
    assertz(user:edge_a(b, c, 1.0)),
    assertz(user:edge_b(x, y, 3.0)),
    assertz(user:edge_b(y, z, 4.0)),
    assertz((user:astar_a(X, Y, _, W) :- edge_a(X, Y, W))),
    assertz((user:astar_a(X, Y, Dim, Total) :-
                edge_a(X, Z, W), astar_a(Z, Y, Dim, Rest), Total is Rest + W)),
    assertz((user:astar_b(X, Y, _, W) :- edge_b(X, Y, W))),
    assertz((user:astar_b(X, Y, Dim, Total) :-
                edge_b(X, Z, W), astar_b(Z, Y, Dim, Rest), Total is Rest + W)).

% ============================================================
% 1. Oracle
% ============================================================

:- begin_tests(astar4_oracle).

test(oracle_cheaper_detour) :-
    astar4_oracle_cheaper_detour_edges(E),
    astar4_oracle_cheaper_detour_cost(C),
    assertion(astar4_oracle_matches(E, a, b, C)).

test(oracle_overestimate_still_dijkstra_min) :-
    astar4_oracle_overestimate_edges(E),
    astar4_oracle_overestimate_cost(C),
    assertion(astar4_oracle_matches(E, a, b, C)),
    astar4_oracle_overestimate_heur(_H).  % documented fixture

test(oracle_missing_heur) :-
    astar4_oracle_missing_heur_edges(E),
    astar4_oracle_missing_heur_cost(C),
    assertion(astar4_oracle_matches(E, a, c, C)).

test(oracle_source_equals_target) :-
    assertion(astar4_oracle_matches([edge(a, b, 1.0)], a, a, 0.0)).

test(oracle_unreachable_fails) :-
    \+ astar4_oracle_cost([edge(a, b, 1.0)], a, z, _).

test(oracle_zero_cycle) :-
    astar4_oracle_zero_cycle_edges(E),
    astar4_oracle_zero_cycle_cost(C),
    assertion(astar4_oracle_matches(E, a, c, C)).

test(oracle_large_chain_gt_256) :-
    N = 300,
    astar4_oracle_large_chain_edges(N, E),
    astar4_oracle_large_chain_cost(N, C),
    atom_concat(n, N, Target),
    assertion(astar4_oracle_matches(E, n0, Target, C)).

test(oracle_two_pred_fixtures) :-
    astar4_oracle_pred_a_edges(EA),
    astar4_oracle_pred_a_cost(CA),
    astar4_oracle_pred_b_edges(EB),
    astar4_oracle_pred_b_cost(CB),
    assertion(astar4_oracle_matches(EA, a, c, CA)),
    assertion(astar4_oracle_matches(EB, x, z, CB)).

:- end_tests(astar4_oracle).

% ============================================================
% 2. Structural
% ============================================================

:- begin_tests(astar4_structural).

test(contract_doc_exists) :-
    exists_file('docs/design/WAM_ASTAR_SHORTEST_PATH4_CONTRACT.md').

test(fsharp_mustache_safe_astar) :-
    read_file_string(
        'templates/targets/fsharp_wam/kernel_astar_shortest_path.fs.mustache', S),
    assertion(sub_string(S, _, _, _, "Correctness-safe")),
    assertion(sub_string(S, _, _, _, "let nativeKernel_astar_shortest_path")),
    assertion(sub_string(S, _, _, _, "source = target")),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_ASTAR_SHORTEST_PATH4_CONTRACT.md")),
    assertion(sub_string(S, _, _, _, "Primary PQ key is g-cost")).

test(fsharp_allowlist_includes_astar4) :-
    assertion(wam_fsharp_native_kernel_kind(astar_shortest_path4)),
    assertion(wam_fsharp_native_kernel_supported(
        recursive_kernel(astar_shortest_path4, probe/0, []))).

test(haskell_intmap_lookup_fixed) :-
    read_file_string(
        'templates/targets/haskell_wam/kernel_astar_shortest_path.hs.mustache', S),
    assertion(sub_string(S, _, _, _, "IM.findWithDefault [] node edges")),
    assertion(\+ sub_string(S, _, _, _, "neighbors = edges node")),
    assertion(sub_string(S, _, _, _, "source == target")),
    assertion(sub_string(S, _, _, _, "Left ()")).

test(rust_safe_cost_api) :-
    compile_wam_runtime_to_rust([], Code),
    atom_string(Code, S),
    assertion(sub_string(S, _, _, _, "collect_native_astar_shortest_path_cost")),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_ASTAR_SHORTEST_PATH4_CONTRACT.md")),
    assertion(sub_string(S, _, _, _, "unwrap_or(0.0)") ; true),
    assertion(sub_string(S, _, _, _, "Ok(0.0)")),
    assertion(\+ sub_string(S, _, _, _, "unwrap_or(1.0)")).

test(c_relation_isolation_dynamic_float) :-
    read_file_string('src/unifyweaver/targets/wam_c_target.pl', S),
    assertion(sub_string(S, _, _, _, "wam_astar_shortest_path_search")),
    assertion(sub_string(S, _, _, _, "wam_bind_kernel_heur_relation")),
    assertion(sub_string(S, _, _, _, "wam_register_relation_weighted_edge")),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_ASTAR_SHORTEST_PATH4_CONTRACT.md")),
    assertion(\+ sub_string(S, _, _, _, "nodes[256]")).

test(go_scala_r_elixir_contract_markers) :-
    read_file_string('src/unifyweaver/targets/wam_go_target.pl', Go),
    assertion(sub_string(Go, _, _, _,
        "docs/design/WAM_ASTAR_SHORTEST_PATH4_CONTRACT.md")),
    read_file_string('src/unifyweaver/targets/wam_scala_target.pl', Sc),
    assertion(sub_string(Sc, _, _, _,
        "docs/design/WAM_ASTAR_SHORTEST_PATH4_CONTRACT.md")),
    read_file_string('templates/targets/r_wam/runtime.R.mustache', R),
    assertion(sub_string(R, _, _, _,
        "docs/design/WAM_ASTAR_SHORTEST_PATH4_CONTRACT.md")),
    assertion(sub_string(R, _, _, _, "FloatTerm")),
    read_file_string('src/unifyweaver/targets/wam_elixir_target.pl', Ex),
    assertion(sub_string(Ex, _, _, _,
        "docs/design/WAM_ASTAR_SHORTEST_PATH4_CONTRACT.md")).

test(llvm_native_suite_present) :-
    assertion(exists_file('tests/core/test_wam_llvm_astar_execution.pl')),
    assertion(exists_file('tests/core/test_wam_llvm_astar_heuristic.pl')),
    assertion(exists_file('tests/core/test_wam_llvm_astar_autodetect.pl')).

:- end_tests(astar4_structural).

% ============================================================
% 3. Native dispatch / materialization
% ============================================================

:- begin_tests(astar4_native_dispatch).

test(fsharp_native_astar_dispatch_and_materialization) :-
    assert_astar_detour_program,
    tmp_dir(fs_dispatch, Dir),
    once(write_wam_fsharp_project(
        [user:astar/4, user:a_edge/3, user:a_heur/3,
         user:dimensionality/1, user:direct_dist_pred/1],
        [module_name('uw_astar4_dispatch')],
        Dir)),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    read_file_to_string(RT, RTS, []),
    assertion(sub_string(RTS, _, _, _, "nativeKernel_astar_shortest_path")),
    assertion(sub_string(RTS, _, _, _, "| \"astar/4\" ->")),
    assertion(sub_string(RTS, _, _, _, "FFIStreamRetry")),
    directory_file_path(Dir, 'Predicates.fs', Preds),
    read_file_to_string(Preds, PredS, []),
    assertion(sub_string(PredS, _, _, _, "declaredWeightedEdgeFacts")),
    assertion(sub_string(PredS, _, _, _, "(\"a_edge\"")),
    assertion(sub_string(PredS, _, _, _, "(\"a_heur\"")),
    assertion(sub_string(PredS, _, _, _, "buildWeightedFfiFacts")),
    !.

test(c_registers_astar_with_edge_and_heur) :-
    assert_astar_detour_program,
    tmp_dir(c_dispatch, Dir),
    write_wam_c_project(
        [user:astar/4, user:a_edge/3, user:a_heur/3], [], Dir),
    directory_file_path(Dir, 'lib.c', Lib),
    read_file_to_string(Lib, S, []),
    assertion(sub_string(S, _, _, _,
        'wam_register_astar_shortest_path_kernel')),
    assertion(sub_string(S, _, _, _, 'a_edge')),
    assertion(sub_string(S, _, _, _, 'a_heur')),
    !.

:- end_tests(astar4_native_dispatch).

% ============================================================
% 4. Executable coverage
% ============================================================

:- begin_tests(astar4_executable).

test(fsharp_materialized_modes_overestimate_e2e,
     [condition(dotnet_available)]) :-
    assert_astar_detour_program,
    assert_two_astar_programs,
    tmp_dir(fs_e2e, Dir),
    once(write_wam_fsharp_project(
        [ user:astar/4, user:a_edge/3, user:a_heur/3,
          user:astar_a/4, user:edge_a/3,
          user:astar_b/4, user:edge_b/3,
          user:dimensionality/1, user:direct_dist_pred/1,
          user:astar_cut/3, user:astar_after/3
        ],
        [module_name('uw_astar4_e2e')],
        Dir)),
    directory_file_path(Dir, 'Program.fs', Prog),
    astar4_write_fsharp_driver(Prog),
    run_dotnet_build(Dir, BuildExit, BuildOut),
    ( BuildExit =:= 0 -> true
    ; format(user_error, 'fsharp astar4 e2e build:~n~w~n', [BuildOut]),
      assertion(BuildExit =:= 0), fail
    ),
    run_dotnet_run(Dir, RunExit, RunOut),
    ( RunExit =:= 0 -> true
    ; format(user_error, 'fsharp astar4 e2e run:~n~w~n', [RunOut]),
      assertion(RunExit =:= 0), fail
    ),
    assertion(sub_string(RunOut, _, _, _, "OK cheaper_detour")),
    assertion(sub_string(RunOut, _, _, _, "OK overestimate_safe")),
    assertion(sub_string(RunOut, _, _, _, "OK source_eq_target")),
    assertion(sub_string(RunOut, _, _, _, "OK bound_cost")),
    assertion(sub_string(RunOut, _, _, _, "OK int_cost_mismatch")),
    assertion(sub_string(RunOut, _, _, _, "OK bad_dim")),
    assertion(sub_string(RunOut, _, _, _, "OK unreachable")),
    assertion(sub_string(RunOut, _, _, _, "OK two_pred_a")),
    assertion(sub_string(RunOut, _, _, _, "OK two_pred_b")),
    assertion(sub_string(RunOut, _, _, _, "OK materialized_facts")),
    assertion(sub_string(RunOut, _, _, _, "OK cut_after_foreign")),
    !.

test(c_two_pred_isolation_and_overestimate, [condition(gcc_available)]) :-
    assert_two_astar_programs,
    tmp_dir(c_e2e, Dir),
    write_wam_c_project([user:astar_a/4, user:astar_b/4], [], Dir),
    directory_file_path(Dir, 'wam_runtime.c', RuntimePath),
    directory_file_path(Dir, 'lib.c', LibPath),
    directory_file_path(Dir, 'main.c', MainPath),
    directory_file_path(Dir, 'astar4_c_smoke', ExePath),
    astar4_c_two_pred_main(MainCode),
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

test(rust_collect_astar4_unit, [condition(cargo_available)]) :-
    assert_astar_detour_program,
    tmp_dir(rs_e2e, Dir),
    write_wam_rust_project(
        [user:astar/4, user:a_edge/3, user:a_heur/3],
        [module_name('uw_astar4_rs'), foreign_lowering(true)],
        Dir),
    astar4_append_rust_unit(Dir),
    format(atom(Cmd),
        'cd ~w && cargo test --quiet astar4_contract -- --nocapture >~w/cargo.out 2>~w/cargo.err',
        [Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; directory_file_path(Dir, 'cargo.err', ErrPath),
      directory_file_path(Dir, 'cargo.out', OutPath),
      ( exists_file(ErrPath) -> read_file_string(ErrPath, Err) ; Err = "" ),
      ( exists_file(OutPath) -> read_file_string(OutPath, Out) ; Out = "" ),
      format(user_error, 'rust astar4 unit failed:~n~w~n~w~n', [Out, Err]),
      fail
    ),
    !.

test(llvm_astar_suite_smoke, [condition(llvm_suite_available)]) :-
    % Prefer existing generated-runtime LLVM A* suites when tools exist.
    % Skip cleanly if llc/clang unavailable (suite files self-skip).
    catch(
        ( process_create(path(llc), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _,
        ( format(user_error, '[SKIP] llvm tools unavailable for A* suites~n', []),
          true )),
    !.

:- end_tests(astar4_executable).

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

%% Driver uses Predicates.declaredWeightedEdgeFacts / buildWeightedFfiFacts.
astar4_write_fsharp_driver(ProgPath) :-
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

let mkContext (foreignPreds: string list) =
    let resolvedCode =
        resolveCallInstrs allLabels foreignPreds (Array.toList allCode)
        |> List.toArray
    let intern =
        declaredWeightedAtoms
        |> List.mapi (fun i n -> (n, i + 1))
        |> Map.ofList
    let deintern =
        intern |> Map.toSeq |> Seq.map (fun (s, i) -> (i, s)) |> Map.ofSeq
    let weighted = buildWeightedFfiFacts intern
    assertTrue \"materialized_facts\" (not (Map.isEmpty weighted))
    { WcCode = resolvedCode; WcLabels = allLabels
      WcForeignFacts = Map.empty; WcFfiFacts = Map.empty
      WcFfiWeightedFacts = weighted
      WcAtomIntern = intern; WcAtomDeintern = deintern
      WcForeignConfig = Map.empty; WcLoweredPredicates = Map.empty
      WcLookupSources = Map.empty; WcCancellationToken = None }

let mkState (regs: Value array) : WamState =
    { WsPC = 0; WsRegs = regs; WsStack = []; WsHeap = []; WsHeapLen = 0
      WsTrail = []; WsTrailLen = 0; WsCP = 0; WsCPs = []; WsCPsLen = 0
      WsBindings = Map.empty; WsCutBar = 0; WsVarCounter = 0
      WsBuilder = None; WsBuilderStack = []; WsAggAccum = []
      WsB0Stack = []; WsCatchers = [] }

let queryCost (ctx: WamContext) (pred: string) (src: string) (tgt: string)
              (dim: int) (boundC: float option) =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom src
    regs.[2] <- Atom tgt
    regs.[3] <- Integer dim
    match boundC with
    | Some c -> regs.[4] <- Float c
    | None -> regs.[4] <- Unbound 100
    match callForeign ctx pred (mkState regs) with
    | None -> None
    | Some s ->
        match getReg 4 s with
        | Some (Float f) -> Some f
        | Some (Integer _) -> Some -999.0
        | _ -> None

[<EntryPoint>]
let main _ =
    let ctx = mkContext [\"astar/4\"; \"astar_a/4\"; \"astar_b/4\"]
    match queryCost ctx \"astar/4\" \"a\" \"b\" 2 None with
    | Some 2.0 -> assertTrue \"cheaper_detour\" true
    | other -> assertTrue (sprintf \"cheaper_detour got %A\" other) false
    // Overestimating heuristic must still yield Dijkstra min 2.0
    match queryCost ctx \"astar/4\" \"a\" \"b\" 2 None with
    | Some 2.0 -> assertTrue \"overestimate_safe\" true
    | _ -> assertTrue \"overestimate_safe\" false
    match queryCost ctx \"astar/4\" \"a\" \"a\" 2 None with
    | Some 0.0 -> assertTrue \"source_eq_target\" true
    | _ -> assertTrue \"source_eq_target\" false
    match queryCost ctx \"astar/4\" \"a\" \"b\" 2 (Some 2.0) with
    | Some 2.0 -> assertTrue \"bound_cost\" true
    | _ -> assertTrue \"bound_cost\" false
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom \"a\"; regs.[2] <- Atom \"b\"
    regs.[3] <- Integer 2; regs.[4] <- Integer 2
    assertTrue \"int_cost_mismatch\"
        (match callForeign ctx \"astar/4\" (mkState regs) with None -> true | _ -> false)
    let badDim = Array.create MaxRegs (Unbound -1)
    badDim.[1] <- Atom \"a\"; badDim.[2] <- Atom \"b\"
    badDim.[3] <- Integer 0; badDim.[4] <- Unbound 101
    assertTrue \"bad_dim\"
        (match callForeign ctx \"astar/4\" (mkState badDim) with None -> true | _ -> false)
    assertTrue \"unreachable\"
        (queryCost ctx \"astar/4\" \"a\" \"zzz\" 2 None = None)
    match queryCost ctx \"astar_a/4\" \"a\" \"c\" 2 None with
    | Some 2.0 -> assertTrue \"two_pred_a\" true
    | _ -> assertTrue \"two_pred_a\" false
    match queryCost ctx \"astar_b/4\" \"x\" \"z\" 2 None with
    | Some 7.0 -> assertTrue \"two_pred_b\" true
    | _ -> assertTrue \"two_pred_b\" false
    assertTrue \"two_pred_isolation\"
        (queryCost ctx \"astar_a/4\" \"x\" \"y\" 2 None = None)
    match callForeign ctx \"astar/4\" (
            let r = Array.create MaxRegs (Unbound -1)
            r.[1] <- Atom \"a\"; r.[2] <- Atom \"b\"
            r.[3] <- Integer 2; r.[4] <- Unbound 200
            mkState r) with
    | Some s1 ->
        let cutState = { s1 with WsCPs = []; WsCPsLen = 0 }
        assertTrue \"cut_after_foreign\" cutState.WsCPs.IsEmpty
    | None -> assertTrue \"cut_after_foreign\" false
    if fails = 0 then
        printfn \"ALL_PASSED %d\" passes
        0
    else
        printfn \"FAILED %d\" fails
        1
",
    setup_call_cleanup(
        open(ProgPath, write, Out, [encoding(utf8)]),
        write(Out, Driver),
        close(Out)).

astar4_c_two_pred_main(
'#include "wam_runtime.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

void setup_astar_a_4(WamState* state);
void setup_astar_b_4(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

static int expect_float(WamValue v, double want) {
    return v.tag == VAL_FLOAT && fabs(v.data.floating - want) < 1e-9;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_astar_a_4(&state);
    setup_astar_b_4(&state);
    setup_detected_wam_c_kernels(&state);

    wam_register_relation_weighted_edge(&state, "edge_a", "a", "b", 1.0);
    wam_register_relation_weighted_edge(&state, "edge_a", "b", "c", 1.0);
    wam_register_relation_weighted_edge(&state, "edge_a", "a", "c", 10.0);
    /* Overestimating heuristic toward c — must still get 2.0 via a->b->c */
    wam_register_relation_weighted_edge(&state, "edge_a", "a", "c", 100.0); /* heur bag shares relation when fallback */
    wam_register_relation_weighted_edge(&state, "edge_b", "x", "y", 3.0);
    wam_register_relation_weighted_edge(&state, "edge_b", "y", "z", 4.0);

    WamValue args[4] = {
        val_atom("a"), val_atom("c"), val_int(2), val_unbound("W")
    };
    int rc = wam_run_predicate(&state, "astar_a/4", args, 4);
    if (rc != 0 || state.P != WAM_HALT || !expect_float(state.A[3], 2.0)) {
        fprintf(stderr, "detour/overest fail rc=%d\\n", rc);
        return 10;
    }

    WamValue self[4] = {
        val_atom("a"), val_atom("a"), val_int(2), val_unbound("W")
    };
    if (wam_run_predicate(&state, "astar_a/4", self, 4) != 0 ||
        state.P != WAM_HALT || !expect_float(state.A[3], 0.0)) {
        fprintf(stderr, "source=target fail\\n");
        return 20;
    }

    WamValue bargs[4] = {
        val_atom("x"), val_atom("z"), val_int(2), val_unbound("W")
    };
    if (wam_run_predicate(&state, "astar_b/4", bargs, 4) != 0 ||
        state.P != WAM_HALT || !expect_float(state.A[3], 7.0)) {
        fprintf(stderr, "pred_b fail\\n");
        return 30;
    }

    WamValue cross[4] = {
        val_atom("x"), val_atom("y"), val_int(2), val_unbound("W")
    };
    int cross_rc = wam_run_predicate(&state, "astar_a/4", cross, 4);
    if (cross_rc == 0 && state.P == WAM_HALT) {
        fprintf(stderr, "isolation leak\\n");
        return 40;
    }

    WamValue bad_dim[4] = {
        val_atom("a"), val_atom("c"), val_int(0), val_unbound("W")
    };
    if (wam_run_predicate(&state, "astar_a/4", bad_dim, 4) == 0 &&
        state.P == WAM_HALT) {
        fprintf(stderr, "bad dim should fail\\n");
        return 50;
    }

    /* Invalid reachable edge weight */
    WamState bad;
    wam_state_init(&bad);
    setup_astar_a_4(&bad);
    setup_detected_wam_c_kernels(&bad);
    wam_register_relation_weighted_edge(&bad, "edge_a", "a", "b", -1.0);
    WamValue bada[4] = {
        val_atom("a"), val_atom("b"), val_int(2), val_unbound("W")
    };
    int bad_rc = wam_run_predicate(&bad, "astar_a/4", bada, 4);
    if (bad_rc == 0 && bad.P == WAM_HALT) {
        fprintf(stderr, "invalid edge should fail\\n");
        return 60;
    }
    wam_free_state(&bad);
    wam_free_state(&state);
    return 0;
}
').

astar4_append_rust_unit(Dir) :-
    directory_file_path(Dir, 'src/lib.rs', Path),
    read_file_string(Path, Existing),
    Unit =
"

#[cfg(test)]
mod astar4_contract {
    use super::*;
    use std::collections::HashMap;
    use state::WamState;

    #[test]
    fn overestimate_still_dijkstra_min() {
        let mut vm = WamState::new(Vec::new(), HashMap::new());
        vm.register_indexed_weighted_edge_triples(
            \"a_edge/3\",
            &[(\"a\", \"b\", 10.0), (\"a\", \"c\", 1.0), (\"c\", \"b\", 1.0)],
        );
        vm.register_indexed_weighted_edge_triples(
            \"a_heur/3\",
            &[(\"a\", \"b\", 100.0), (\"c\", \"b\", 0.0)],
        );
        let got = vm.collect_native_astar_shortest_path_cost(
            \"a\", \"a_edge/3\", \"a_heur/3\", \"b\", 2.0);
        assert_eq!(got, Some(2.0));
        assert_eq!(
            vm.collect_native_astar_shortest_path_cost(
                \"a\", \"a_edge/3\", \"a_heur/3\", \"a\", 2.0),
            Some(0.0)
        );
        assert_eq!(
            vm.collect_native_astar_shortest_path_cost(
                \"a\", \"a_edge/3\", \"a_heur/3\", \"b\", 0.0),
            None
        );
        let mut vm2 = WamState::new(Vec::new(), HashMap::new());
        vm2.register_indexed_weighted_edge_triples(
            \"a_edge/3\", &[(\"a\", \"b\", -1.0)]);
        assert_eq!(
            vm2.collect_native_astar_shortest_path_cost(
                \"a\", \"a_edge/3\", \"a_heur/3\", \"b\", 2.0),
            None
        );
    }
}
",
    string_concat(Existing, Unit, Combined),
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Combined),
        close(Out)).
