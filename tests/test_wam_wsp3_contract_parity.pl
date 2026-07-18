:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_wsp3_contract_parity.pl — fleet-wide weighted_shortest_path3
% contract (finite nonnegative Dijkstra) parity suite.
%
% Contract: docs/design/WAM_WEIGHTED_SHORTEST_PATH3_CONTRACT.md
% Oracle:   tests/fixtures/wsp3_contract_oracle.pl
%
% Usage (from repo root):
%   swipl -q -g run_tests -t halt tests/test_wam_wsp3_contract_parity.pl

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module(library(readutil)).
:- use_module(library(process)).
:- use_module(library(debug), [assertion/1]).

:- use_module('fixtures/wsp3_contract_oracle', [
    wsp3_oracle_min_pairs/3,
    wsp3_oracle_matches_expected/3,
    wsp3_oracle_cheaper_detour_edges/1,
    wsp3_oracle_cheaper_detour_expected/1,
    wsp3_oracle_duplicate_edges/1,
    wsp3_oracle_duplicate_expected/1,
    wsp3_oracle_equal_cost_edges/1,
    wsp3_oracle_equal_cost_expected/1,
    wsp3_oracle_mixed_weights_edges/1,
    wsp3_oracle_mixed_weights_expected/1,
    wsp3_oracle_zero_cost_edges/1,
    wsp3_oracle_zero_cost_expected/1,
    wsp3_oracle_positive_cycle_edges/1,
    wsp3_oracle_positive_cycle_expected/1,
    wsp3_oracle_source_self_loop_edges/1,
    wsp3_oracle_source_self_loop_expected/1,
    wsp3_oracle_sink_edges/1,
    wsp3_oracle_sink_expected/1,
    wsp3_oracle_large_chain_edges/2,
    wsp3_oracle_large_chain_expected/2,
    wsp3_oracle_pred_a_edges/1,
    wsp3_oracle_pred_a_expected/1,
    wsp3_oracle_pred_b_edges/1,
    wsp3_oracle_pred_b_expected/1
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

:- use_module('helpers/wam_kernel_parity_harness').

:- dynamic user:w_edge/3.
:- dynamic user:wsp/3.
:- dynamic user:w_edge_alt/3.
:- dynamic user:wsp_alt/3.
:- dynamic user:edge_a/3.
:- dynamic user:edge_b/3.
:- dynamic user:wsp_a/3.
:- dynamic user:wsp_b/3.
:- dynamic user:wsp_tail/2.
:- dynamic user:wsp_after/2.
:- dynamic user:wsp_cut/2.
:- dynamic user:wsp_call_after/2.

assert_wsp_detour_program :-
    retractall(user:w_edge(_, _, _)),
    retractall(user:wsp(_, _, _)),
    retractall(user:wsp_tail(_, _)),
    retractall(user:wsp_after(_, _)),
    retractall(user:wsp_cut(_, _)),
    retractall(user:wsp_call_after(_, _)),
    assertz(user:w_edge(a, b, 10.0)),
    assertz(user:w_edge(a, c, 1.0)),
    assertz(user:w_edge(c, b, 1.0)),
    assertz(user:w_edge(b, d, 1.0)),
    assertz(user:w_edge(a, a, 1.0)),
    assertz((user:wsp(X, Y, W) :- w_edge(X, Y, W))),
    assertz((user:wsp(X, Y, Total) :-
                w_edge(X, Z, W), wsp(Z, Y, Rest), Total is Rest + W)),
    assertz((user:wsp_tail(S, T) :- wsp(S, T, _), true)),
    assertz((user:wsp_after(S, T) :- wsp(S, T, _), true)),
    assertz((user:wsp_cut(S, T) :- wsp(S, T, _), !)),
    assertz((user:wsp_call_after(S, T) :- call(wsp(S, T, _)), true)).

assert_two_wsp_programs :-
    retractall(user:edge_a(_, _, _)),
    retractall(user:edge_b(_, _, _)),
    retractall(user:wsp_a(_, _, _)),
    retractall(user:wsp_b(_, _, _)),
    assertz(user:edge_a(a, b, 1.0)),
    assertz(user:edge_a(b, c, 1.0)),
    assertz(user:edge_b(x, y, 3.0)),
    assertz(user:edge_b(y, z, 4.0)),
    assertz((user:wsp_a(X, Y, W) :- edge_a(X, Y, W))),
    assertz((user:wsp_a(X, Y, Total) :-
                edge_a(X, Z, W), wsp_a(Z, Y, Rest), Total is Rest + W)),
    assertz((user:wsp_b(X, Y, W) :- edge_b(X, Y, W))),
    assertz((user:wsp_b(X, Y, Total) :-
                edge_b(X, Z, W), wsp_b(Z, Y, Rest), Total is Rest + W)).

% ============================================================
% 1. Oracle
% ============================================================

:- begin_tests(wsp3_oracle).

test(oracle_cheaper_detour) :-
    wsp3_oracle_cheaper_detour_edges(E),
    wsp3_oracle_cheaper_detour_expected(Exp),
    assertion(wsp3_oracle_matches_expected(E, a, Exp)).

test(oracle_duplicate_edges) :-
    wsp3_oracle_duplicate_edges(E),
    wsp3_oracle_duplicate_expected(Exp),
    assertion(wsp3_oracle_matches_expected(E, a, Exp)).

test(oracle_equal_cost) :-
    wsp3_oracle_equal_cost_edges(E),
    wsp3_oracle_equal_cost_expected(Exp),
    assertion(wsp3_oracle_matches_expected(E, a, Exp)).

test(oracle_mixed_weights) :-
    wsp3_oracle_mixed_weights_edges(E),
    wsp3_oracle_mixed_weights_expected(Exp),
    assertion(wsp3_oracle_matches_expected(E, a, Exp)).

test(oracle_zero_cost) :-
    wsp3_oracle_zero_cost_edges(E),
    wsp3_oracle_zero_cost_expected(Exp),
    assertion(wsp3_oracle_matches_expected(E, a, Exp)).

test(oracle_positive_cycle) :-
    wsp3_oracle_positive_cycle_edges(E),
    wsp3_oracle_positive_cycle_expected(Exp),
    assertion(wsp3_oracle_matches_expected(E, a, Exp)).

test(oracle_source_self_loop_excludes_source) :-
    wsp3_oracle_source_self_loop_edges(E),
    wsp3_oracle_source_self_loop_expected(Exp),
    assertion(wsp3_oracle_matches_expected(E, a, Exp)),
    wsp3_oracle_min_pairs(E, a, Pairs),
    assertion(\+ member(a-_, Pairs)).

test(oracle_sink_and_unknown) :-
    wsp3_oracle_sink_edges(E),
    wsp3_oracle_sink_expected(Exp),
    assertion(wsp3_oracle_matches_expected(E, b, Exp)),
    assertion(wsp3_oracle_matches_expected(E, unknown, [])).

test(oracle_large_chain_gt_256) :-
    N = 300,
    wsp3_oracle_large_chain_edges(N, E),
    wsp3_oracle_large_chain_expected(N, Exp),
    assertion(wsp3_oracle_matches_expected(E, n0, Exp)),
    length(Exp, N).

test(oracle_two_pred_isolation_fixtures) :-
    wsp3_oracle_pred_a_edges(EA),
    wsp3_oracle_pred_a_expected(ExpA),
    wsp3_oracle_pred_b_edges(EB),
    wsp3_oracle_pred_b_expected(ExpB),
    assertion(wsp3_oracle_matches_expected(EA, a, ExpA)),
    assertion(wsp3_oracle_matches_expected(EB, x, ExpB)).

:- end_tests(wsp3_oracle).

% ============================================================
% 2. Structural
% ============================================================

:- begin_tests(wsp3_structural).

test(contract_doc_exists) :-
    exists_file('docs/design/WAM_WEIGHTED_SHORTEST_PATH3_CONTRACT.md').

test(fsharp_mustache_dijkstra) :-
    read_file_string(
        'templates/targets/fsharp_wam/kernel_weighted_shortest_path.fs.mustache', S),
    assertion(sub_string(S, _, _, _, "finite nonnegative Dijkstra")),
    assertion(sub_string(S, _, _, _, "let nativeKernel_weighted_shortest_path")),
    assertion(sub_string(S, _, _, _, "FFIStreamRetry")),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_WEIGHTED_SHORTEST_PATH3_CONTRACT.md")),
    assertion(sub_string(S, _, _, _, "t <> source")).

test(fsharp_allowlist_includes_wsp3) :-
    assertion(wam_fsharp_native_kernel_kind(weighted_shortest_path3)),
    assertion(wam_fsharp_native_kernel_supported(
        recursive_kernel(weighted_shortest_path3, probe/0, []))),
    assertion(wam_fsharp_native_kernel_kind(astar_shortest_path4)).

test(haskell_mustache_dijkstra) :-
    read_file_string(
        'templates/targets/haskell_wam/kernel_weighted_shortest_path.hs.mustache', S),
    assertion(sub_string(S, _, _, _, "Finite nonnegative Dijkstra")),
    assertion(sub_string(S, _, _, _, "nativeKernel_weighted_shortest_path")),
    assertion(sub_string(S, _, _, _, "n /= source")),
    assertion(sub_string(S, _, _, _, "maybe [] id")),
    assertion(sub_string(S, _, _, _, "Nothing -> Nothing")),
    assertion(\+ sub_string(S, _, _, _, "error \"wsp3: reachable invalid")).

test(rust_dijkstra_source_excluded_float) :-
    compile_wam_runtime_to_rust([], Code),
    atom_string(Code, S),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_WEIGHTED_SHORTEST_PATH3_CONTRACT.md")),
    assertion(sub_string(S, _, _, _, "collect_native_weighted_shortest_path_results")),
    assertion(sub_string(S, _, _, _, "if node != start")),
    assertion(sub_string(S, _, _, _, "!weight.is_finite() || *weight < 0.0")).

test(c_relation_isolation_dynamic_stream) :-
    read_file_string('src/unifyweaver/targets/wam_c_target.pl', S),
    assertion(sub_string(S, _, _, _, "wam_collect_weighted_shortest_path")),
    assertion(sub_string(S, _, _, _, "wam_register_relation_weighted_edge")),
    assertion(sub_string(S, _, _, _, "wam_bind_foreign_pair_stream")),
    assertion(sub_string(S, _, _, _, "wam_wsp3_weight_valid")),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_WEIGHTED_SHORTEST_PATH3_CONTRACT.md")).

test(go_scala_r_elixir_contract_markers) :-
    read_file_string('src/unifyweaver/targets/wam_go_target.pl', Go),
    assertion(sub_string(Go, _, _, _,
        "docs/design/WAM_WEIGHTED_SHORTEST_PATH3_CONTRACT.md")),
    read_file_string('src/unifyweaver/targets/wam_scala_target.pl', Sc),
    assertion(sub_string(Sc, _, _, _,
        "docs/design/WAM_WEIGHTED_SHORTEST_PATH3_CONTRACT.md")),
    assertion(sub_string(Sc, _, _, _, "case a @ Atom(_)")),
    read_file_string('templates/targets/r_wam/runtime.R.mustache', R),
    assertion(sub_string(R, _, _, _, "weighted_shortest_path3")),
    assertion(sub_string(R, _, _, _, "FloatTerm")),
    read_file_string('src/unifyweaver/targets/wam_elixir_target.pl', Ex),
    assertion(sub_string(Ex, _, _, _,
        "docs/design/WAM_WEIGHTED_SHORTEST_PATH3_CONTRACT.md")).

test(llvm_preserves_dijkstra) :-
    exists_file('tests/core/test_wam_llvm_dijkstra_execution.pl'),
    exists_file('tests/core/test_wam_llvm_wsp3_stream_execution.pl').

:- end_tests(wsp3_structural).

% ============================================================
% 3. Native dispatch / materialization
% ============================================================

:- begin_tests(wsp3_native_dispatch).

test(fsharp_native_wsp3_dispatch_and_materialization) :-
    assert_two_wsp_programs,
    tmp_dir(fs_dispatch, Dir),
    once(write_wam_fsharp_project(
        [user:wsp_a/3, user:edge_a/3, user:wsp_b/3, user:edge_b/3],
        [module_name('uw_wsp3_dispatch')],
        Dir)),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    read_file_to_string(RT, RTS, []),
    assertion(sub_string(RTS, _, _, _, "nativeKernel_weighted_shortest_path")),
    assertion(sub_string(RTS, _, _, _, "| \"wsp_a/3\" ->")),
    assertion(sub_string(RTS, _, _, _, "| \"wsp_b/3\" ->")),
    assertion(sub_string(RTS, _, _, _, "FFIStreamRetry")),
    directory_file_path(Dir, 'Predicates.fs', Preds),
    read_file_to_string(Preds, PredS, []),
    assertion(sub_string(PredS, _, _, _, "declaredWeightedEdgeFacts")),
    assertion(sub_string(PredS, _, _, _, "(\"edge_a\"")),
    assertion(sub_string(PredS, _, _, _, "(\"edge_b\"")),
    assertion(sub_string(PredS, _, _, _, "buildWeightedFfiFacts")),
    !.

test(c_registers_wsp3_with_edge_relation) :-
    assert_wsp_detour_program,
    tmp_dir(c_dispatch, Dir),
    write_wam_c_project([user:wsp/3, user:w_edge/3], [], Dir),
    directory_file_path(Dir, 'lib.c', Lib),
    read_file_to_string(Lib, S, []),
    assertion(sub_string(S, _, _, _,
        'wam_register_weighted_shortest_path_kernel')),
    assertion(sub_string(S, _, _, _, 'w_edge')),
    !.

:- end_tests(wsp3_native_dispatch).

% ============================================================
% 4. Executable coverage
% ============================================================

:- begin_tests(wsp3_executable).

test(fsharp_materialized_stream_modes_aliases_cut_e2e,
     [condition(dotnet_available)]) :-
    assert_wsp_detour_program,
    assert_two_wsp_programs,
    tmp_dir(fs_e2e, Dir),
    once(write_wam_fsharp_project(
        [ user:wsp/3, user:w_edge/3,
          user:wsp_a/3, user:edge_a/3,
          user:wsp_b/3, user:edge_b/3,
          user:wsp_tail/2, user:wsp_after/2,
          user:wsp_cut/2, user:wsp_call_after/2
        ],
        [module_name('uw_wsp3_e2e')],
        Dir)),
    directory_file_path(Dir, 'Program.fs', Prog),
    wsp3_write_fsharp_driver(Prog),
    run_dotnet_build(Dir, BuildExit, BuildOut),
    ( BuildExit =:= 0 -> true
    ; format(user_error, 'fsharp wsp3 e2e build:~n~w~n', [BuildOut]),
      assertion(BuildExit =:= 0), fail
    ),
    run_dotnet_run(Dir, RunExit, RunOut),
    ( RunExit =:= 0 -> true
    ; format(user_error, 'fsharp wsp3 e2e run:~n~w~n', [RunOut]),
      assertion(RunExit =:= 0), fail
    ),
    assertion(sub_string(RunOut, _, _, _, "OK cheaper_detour")),
    assertion(sub_string(RunOut, _, _, _, "OK unbound_stream")),
    assertion(sub_string(RunOut, _, _, _, "OK bound_target")),
    assertion(sub_string(RunOut, _, _, _, "OK bound_cost")),
    assertion(sub_string(RunOut, _, _, _, "OK both_bound")),
    assertion(sub_string(RunOut, _, _, _, "OK int_cost_mismatch")),
    assertion(sub_string(RunOut, _, _, _, "OK source_excluded")),
    assertion(sub_string(RunOut, _, _, _, "OK alias_later_match")),
    assertion(sub_string(RunOut, _, _, _, "OK cut_after_foreign")),
    assertion(sub_string(RunOut, _, _, _, "OK two_pred_a")),
    assertion(sub_string(RunOut, _, _, _, "OK two_pred_b")),
    assertion(sub_string(RunOut, _, _, _, "OK materialized_facts")),
    !.

test(c_two_pred_isolation_and_detour, [condition(gcc_available)]) :-
    assert_two_wsp_programs,
    tmp_dir(c_e2e, Dir),
    write_wam_c_project([user:wsp_a/3, user:wsp_b/3], [], Dir),
    directory_file_path(Dir, 'wam_runtime.c', RuntimePath),
    directory_file_path(Dir, 'lib.c', LibPath),
    directory_file_path(Dir, 'main.c', MainPath),
    directory_file_path(Dir, 'wsp3_c_smoke', ExePath),
    wsp3_c_two_pred_main(MainCode),
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

test(rust_collect_wsp3_unit, [condition(cargo_available)]) :-
    assert_wsp_detour_program,
    tmp_dir(rs_e2e, Dir),
    write_wam_rust_project(
        [user:wsp/3, user:w_edge/3],
        [module_name('uw_wsp3_rs'), foreign_lowering(true)],
        Dir),
    wsp3_append_rust_unit(Dir),
    format(atom(Cmd),
        'cd ~w && cargo test --quiet wsp3_contract -- --nocapture >~w/cargo.out 2>~w/cargo.err',
        [Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; directory_file_path(Dir, 'cargo.err', ErrPath),
      directory_file_path(Dir, 'cargo.out', OutPath),
      ( exists_file(ErrPath) -> read_file_string(ErrPath, Err) ; Err = "" ),
      ( exists_file(OutPath) -> read_file_string(OutPath, Out) ; Out = "" ),
      format(user_error, 'rust wsp3 unit failed:~n~w~n~w~n', [Out, Err]),
      fail
    ),
    !.

test(elixir_collect_pairs_unit, [condition(elixir_available)]) :-
    % Generated-kernel unit: execute the production WeightedShortestPath
    % module emitted by compile_wam_runtime_to_elixir/2.  This deliberately
    % does not claim dispatch/register/streaming end-to-end coverage.
    tmp_dir(ex_e2e, Dir),
    compile_wam_runtime_snippet_for_elixir_wsp3(Dir),
    directory_file_path(Dir, 'wsp3_unit.exs', Script),
    wsp3_write_elixir_unit(Script),
    format(atom(Cmd),
        'cd ~w && elixir wsp3_unit.exs >~w/elixir.out 2>~w/elixir.err',
        [Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; directory_file_path(Dir, 'elixir.err', ErrPath),
      directory_file_path(Dir, 'elixir.out', OutPath),
      ( exists_file(ErrPath) -> read_file_string(ErrPath, Err) ; Err = "" ),
      ( exists_file(OutPath) -> read_file_string(OutPath, Out) ; Out = "" ),
      format(user_error, 'elixir wsp3 unit failed:~n~w~n~w~n', [Out, Err]),
      fail
    ),
    !.

:- end_tests(wsp3_executable).

% ============================================================
% Helpers
% ============================================================

%% Driver uses Predicates.declaredWeightedEdgeFacts / buildWeightedFfiFacts
%% — the generated project's real materialized facts, not a hand-built map.
wsp3_write_fsharp_driver(ProgPath) :-
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
    // Real codegen materialization — not a hand-built adjacency map.
    let intern =
        declaredWeightedAtoms
        |> List.mapi (fun i n -> (n, i + 1))
        |> Map.ofList
    let deintern =
        intern |> Map.toSeq |> Seq.map (fun (s, i) -> (i, s)) |> Map.ofSeq
    let weighted = buildWeightedFfiFacts intern
    assertTrue \"materialized_facts\" (not (Map.isEmpty weighted))
    { WcCode              = resolvedCode
      WcLabels            = allLabels
      WcForeignFacts      = Map.empty
      WcFfiFacts          = Map.empty
      WcFfiWeightedFacts  = weighted
      WcAtomIntern        = intern
      WcAtomDeintern      = deintern
      WcForeignConfig     = Map.empty
      WcLoweredPredicates = Map.empty
      WcLookupSources     = Map.empty
      WcCancellationToken = None }

let mkState (regs: Value array) : WamState =
    { WsPC = 0; WsRegs = regs; WsStack = []; WsHeap = []; WsHeapLen = 0
      WsTrail = []; WsTrailLen = 0; WsCP = 0; WsCPs = []; WsCPsLen = 0
      WsBindings = Map.empty; WsCutBar = 0; WsVarCounter = 0
      WsBuilder = None; WsBuilderStack = []; WsAggAccum = []
      WsB0Stack = []; WsCatchers = [] }

let collectPairs (ctx: WamContext) (pred: string) (source: string)
                 (boundT: string option) (boundC: float option) =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom source
    match boundT with
    | Some t -> regs.[2] <- Atom t
    | None -> regs.[2] <- Unbound 100
    match boundC with
    | Some c -> regs.[3] <- Float c
    | None -> regs.[3] <- Unbound 101
    match callForeign ctx pred (mkState regs) with
    | None -> []
    | Some s1 ->
        let readPair (s: WamState) =
            let t =
                match getReg 2 s with
                | Some (Atom a) -> a
                | _ -> \"?\"
            let c =
                match getReg 3 s with
                | Some (Float f) -> f
                | Some (Integer _) -> -999.0  // contract: must be Float
                | _ -> -1.0
            t, c
        let rec gather (s: WamState) acc =
            let p = readPair s
            match backtrack s with
            | Some s2 -> gather s2 (p :: acc)
            | None -> List.rev (p :: acc)
        gather s1 []

let sortedPairs xs =
    xs |> List.sortBy fst

[<EntryPoint>]
let main _ =
    let ctx = mkContext [\"wsp/3\"; \"wsp_a/3\"; \"wsp_b/3\"]
    let detour = collectPairs ctx \"wsp/3\" \"a\" None None |> sortedPairs
    assertTrue \"cheaper_detour\" (detour = [(\"b\", 2.0); (\"c\", 1.0); (\"d\", 3.0)])
    assertTrue \"unbound_stream\" (List.length detour = 3)
    let boundT = collectPairs ctx \"wsp/3\" \"a\" (Some \"b\") None
    assertTrue \"bound_target\" (boundT = [(\"b\", 2.0)])
    let boundC = collectPairs ctx \"wsp/3\" \"a\" None (Some 1.0)
    assertTrue \"bound_cost\" (boundC = [(\"c\", 1.0)])
    let both = collectPairs ctx \"wsp/3\" \"a\" (Some \"d\") (Some 3.0)
    assertTrue \"both_bound\" (both = [(\"d\", 3.0)])
    // Integer 3 must not unify with Float 3.0
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom \"a\"
    regs.[2] <- Atom \"d\"
    regs.[3] <- Integer 3
    let intMismatch =
        match callForeign ctx \"wsp/3\" (mkState regs) with
        | None -> true
        | Some _ -> false
    assertTrue \"int_cost_mismatch\" intMismatch
    let self = collectPairs ctx \"wsp/3\" \"a\" (Some \"a\") None
    assertTrue \"source_excluded\" (self = [])
    // Ordering-sensitive: bind Cost=3.0 first candidate may be skipped;
    // later compatible (d,3.0) must still succeed via retry filtering.
    let later = collectPairs ctx \"wsp/3\" \"a\" None (Some 3.0)
    assertTrue \"alias_later_match\" (later = [(\"d\", 3.0)])
    // Cut after foreign stream
    let cutRegs = Array.create MaxRegs (Unbound -1)
    cutRegs.[1] <- Atom \"a\"
    cutRegs.[2] <- Unbound 200
    match callForeign ctx \"wsp/3\" (mkState cutRegs) with
    | None -> assertTrue \"cut_after_foreign\" false
    | Some s1 ->
        let cutState = { s1 with WsCPs = []; WsCPsLen = 0; WsCutBar = s1.WsCPsLen }
        assertTrue \"cut_after_foreign\" (cutState.WsCPs.IsEmpty)
    let aPairs = collectPairs ctx \"wsp_a/3\" \"a\" None None |> sortedPairs
    assertTrue \"two_pred_a\" (aPairs = [(\"b\", 1.0); (\"c\", 2.0)])
    let bPairs = collectPairs ctx \"wsp_b/3\" \"x\" None None |> sortedPairs
    assertTrue \"two_pred_b\" (bPairs = [(\"y\", 3.0); (\"z\", 7.0)])
    // Cross-contamination must fail: wsp_a from x yields nothing
    let cross = collectPairs ctx \"wsp_a/3\" \"x\" None None
    assertTrue \"two_pred_isolation\" (cross = [])
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

wsp3_c_two_pred_main(
'#include "wam_runtime.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

void setup_wsp_a_3(WamState* state);
void setup_wsp_b_3(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

static int expect_float(WamValue v, double want) {
    return v.tag == VAL_FLOAT && fabs(v.data.floating - want) < 1e-9;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wsp_a_3(&state);
    setup_wsp_b_3(&state);
    setup_detected_wam_c_kernels(&state);

    wam_register_relation_weighted_edge(&state, "edge_a", "a", "b", 1.0);
    wam_register_relation_weighted_edge(&state, "edge_a", "b", "c", 1.0);
    wam_register_relation_weighted_edge(&state, "edge_a", "a", "c", 10.0);
    wam_register_relation_weighted_edge(&state, "edge_b", "x", "y", 3.0);
    wam_register_relation_weighted_edge(&state, "edge_b", "y", "z", 4.0);

    WamValue args[3] = { val_atom("a"), val_atom("c"), val_unbound("W") };
    int rc = wam_run_predicate(&state, "wsp_a/3", args, 3);
    if (rc != 0 || state.P != WAM_HALT || !expect_float(state.A[2], 2.0)) {
        fprintf(stderr, "detour fail rc=%d P=%d tag=%d\\n", rc, state.P,
                state.A[2].tag);
        return 10;
    }

    WamValue stream[3] = {
        val_atom("a"), val_unbound("T"), val_unbound("W")
    };
    if (wam_run_predicate(&state, "wsp_a/3", stream, 3) != 0 ||
        state.P != WAM_HALT) {
        fprintf(stderr, "stream fail\\n");
        return 20;
    }

    WamValue bargs[3] = { val_atom("x"), val_atom("z"), val_unbound("W") };
    if (wam_run_predicate(&state, "wsp_b/3", bargs, 3) != 0 ||
        state.P != WAM_HALT || !expect_float(state.A[2], 7.0)) {
        fprintf(stderr, "pred_b fail\\n");
        return 30;
    }

    /* Isolation: wsp_a must not see edge_b */
    WamValue cross[3] = { val_atom("x"), val_atom("y"), val_unbound("W") };
    int cross_rc = wam_run_predicate(&state, "wsp_a/3", cross, 3);
    if (cross_rc == 0 && state.P == WAM_HALT) {
        fprintf(stderr, "isolation leak\\n");
        return 40;
    }

    /* Invalid reachable weight fails cleanly */
    WamState bad;
    wam_state_init(&bad);
    setup_wsp_a_3(&bad);
    setup_detected_wam_c_kernels(&bad);
    wam_register_relation_weighted_edge(&bad, "edge_a", "a", "b", -1.0);
    WamValue bada[3] = { val_atom("a"), val_unbound("T"), val_unbound("W") };
    int bad_rc = wam_run_predicate(&bad, "wsp_a/3", bada, 3);
    if (bad_rc == 0 && bad.P == WAM_HALT) {
        fprintf(stderr, "invalid weight should fail\\n");
        return 50;
    }
    wam_free_state(&bad);
    wam_free_state(&state);
    return 0;
}
').

wsp3_append_rust_unit(Dir) :-
    directory_file_path(Dir, 'src/lib.rs', Path),
    read_file_string(Path, Existing),
    Unit =
"

#[cfg(test)]
mod wsp3_contract {
    use super::*;
    use std::collections::HashMap;
    use state::WamState;

    #[test]
    fn cheaper_detour_and_invalid() {
        let mut vm = WamState::new(Vec::new(), HashMap::new());
        vm.register_indexed_weighted_edge_triples(
            \"w_edge/3\",
            &[
                (\"a\", \"b\", 10.0),
                (\"a\", \"c\", 1.0),
                (\"c\", \"b\", 1.0),
                (\"b\", \"d\", 1.0),
            ],
        );
        let mut out = Vec::new();
        vm.collect_native_weighted_shortest_path_results(\"a\", \"w_edge/3\", &mut out);
        out.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            out,
            vec![
                (\"b\".to_string(), 2.0),
                (\"c\".to_string(), 1.0),
                (\"d\".to_string(), 3.0),
            ]
        );

        let mut vm2 = WamState::new(Vec::new(), HashMap::new());
        vm2.register_indexed_weighted_edge_triples(
            \"w_edge/3\",
            &[(\"a\", \"b\", -1.0)],
        );
        let mut out2 = Vec::new();
        vm2.collect_native_weighted_shortest_path_results(\"a\", \"w_edge/3\", &mut out2);
        assert!(out2.is_empty());
    }
}
",
    string_concat(Existing, Unit, Combined),
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Combined),
        close(Out)).

:- use_module('../src/unifyweaver/targets/wam_elixir_target',
              [compile_wam_runtime_to_elixir/2]).

compile_wam_runtime_snippet_for_elixir_wsp3(Dir) :-
    compile_wam_runtime_to_elixir([], RuntimeCode),
    Pattern = "defmodule WamRuntime.GraphKernel.WeightedShortestPath do",
    EndPattern = "defmodule WamRuntime.GraphKernel.AstarShortestPath do",
    sub_string(RuntimeCode, Start, _, _, Pattern),
    sub_string(RuntimeCode, End, _, _, EndPattern),
    End > Start,
    BodyLen is End - Start,
    sub_string(RuntimeCode, Start, BodyLen, _, Body),
    directory_file_path(Dir, 'wsp3_kernel.ex', Path),
    write_file_string(Path, Body).

wsp3_write_elixir_unit(Script) :-
    Code =
'Code.require_file("wsp3_kernel.ex")

edges = fn
  "a" -> [{"a", "b", 10.0}, {"a", "c", 1.0}]
  "c" -> [{"c", "b", 1.0}]
  "b" -> [{"b", "d", 1.0}]
  _ -> []
end

got = WamRuntime.GraphKernel.WeightedShortestPath.collect_path_costs(edges, "a")
     |> Enum.sort()
want = [{"b", 2.0}, {"c", 1.0}, {"d", 3.0}]
unless got == want do
  IO.puts(:stderr, "mismatch #{inspect(got)}")
  System.halt(1)
end

bad_edges = fn
  "a" -> [{"a", "b", -1.0}]
  _ -> []
end
bad = WamRuntime.GraphKernel.WeightedShortestPath.collect_path_costs(bad_edges, "a")
unless bad == :invalid do
  IO.puts(:stderr, "invalid should fail, got #{inspect(bad)}")
  System.halt(2)
end
IO.puts("OK")
',
    write_file_string(Script, Code).
