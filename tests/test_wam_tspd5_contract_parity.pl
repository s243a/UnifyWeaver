:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_tspd5_contract_parity.pl — fleet-wide transitive_step_parent_distance5
% contract (shortest-positive correlated step/parent) parity suite.
%
% Contract: docs/design/WAM_TRANSITIVE_STEP_PARENT_DISTANCE5_CONTRACT.md
% Oracle:   tests/fixtures/tspd5_contract_oracle.pl
%
% Usage (from repo root):
%   swipl -q -g run_tests -t halt tests/test_wam_tspd5_contract_parity.pl

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module(library(readutil)).
:- use_module(library(process)).
:- use_module(library(debug), [assertion/1]).

:- use_module('fixtures/tspd5_contract_oracle', [
    tspd5_oracle_quads/3,
    tspd5_oracle_has/6,
    tspd5_fixture/3,
    tspd5_fixture_expected/3
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

:- dynamic user:tspd_edge/2.
:- dynamic user:tspd/5.
:- dynamic user:tspd_tail/4.
:- dynamic user:tspd_after/4.
:- dynamic user:tspd_cut/4.
:- dynamic user:tspd_call_after/4.
:- dynamic user:edge_a/2.
:- dynamic user:edge_b/2.
:- dynamic user:tspd_a/5.
:- dynamic user:tspd_b/5.

assert_tspd_cycle_program :-
    retractall(user:tspd_edge(_, _)),
    retractall(user:tspd(_, _, _, _, _)),
    retractall(user:tspd_tail(_, _, _, _)),
    retractall(user:tspd_after(_, _, _, _)),
    retractall(user:tspd_cut(_, _, _, _)),
    retractall(user:tspd_call_after(_, _, _, _)),
    assertz(user:tspd_edge(a, b)),
    assertz(user:tspd_edge(b, c)),
    assertz(user:tspd_edge(c, d)),
    assertz(user:tspd_edge(c, a)),
    assertz((user:tspd(X, Y, Y, X, 1) :- tspd_edge(X, Y))),
    assertz((user:tspd(X, Y, S, P, D) :-
                tspd_edge(X, S), tspd(S, Y, _, P, D1), D is D1 + 1)),
    assertz((user:tspd_tail(Y, S, P, D) :- tspd(a, Y, S, P, D))),
    assertz((user:tspd_after(Y, S, P, D) :- tspd_tail(Y, S, P, D), Y \== b)),
    assertz((user:tspd_cut(Y, S, P, D) :- tspd_tail(Y, S, P, D), !)),
    assertz((user:tspd_call_after(Y, S, P, D) :- tspd(a, Y, S, P, D), Y \== b)).

assert_tspd_diamond_program :-
    retractall(user:tspd_edge(_, _)),
    retractall(user:tspd(_, _, _, _, _)),
    assertz(user:tspd_edge(a, b)),
    assertz(user:tspd_edge(a, c)),
    assertz(user:tspd_edge(b, p)),
    assertz(user:tspd_edge(c, q)),
    assertz(user:tspd_edge(p, t)),
    assertz(user:tspd_edge(q, t)),
    assertz((user:tspd(X, Y, Y, X, 1) :- tspd_edge(X, Y))),
    assertz((user:tspd(X, Y, S, P, D) :-
                tspd_edge(X, S), tspd(S, Y, _, P, D1), D is D1 + 1)).

assert_tspd_chain_program :-
    retractall(user:tspd_edge(_, _)),
    retractall(user:tspd(_, _, _, _, _)),
    assertz(user:tspd_edge(a, b)),
    assertz(user:tspd_edge(b, c)),
    assertz(user:tspd_edge(c, d)),
    assertz((user:tspd(X, Y, Y, X, 1) :- tspd_edge(X, Y))),
    assertz((user:tspd(X, Y, S, P, D) :-
                tspd_edge(X, S), tspd(S, Y, _, P, D1), D is D1 + 1)).

assert_c_tspd_two_pred_program :-
    retractall(user:edge_a(_, _)),
    retractall(user:edge_b(_, _)),
    retractall(user:tspd_a(_, _, _, _, _)),
    retractall(user:tspd_b(_, _, _, _, _)),
    assertz(user:edge_a(a, b)),
    assertz(user:edge_b(x, y)),
    assertz((user:tspd_a(X, Y, Y, X, 1) :- edge_a(X, Y))),
    assertz((user:tspd_a(X, Y, S, P, D) :-
                edge_a(X, S), tspd_a(S, Y, _, P, D1), D is D1 + 1)),
    assertz((user:tspd_b(X, Y, Y, X, 1) :- edge_b(X, Y))),
    assertz((user:tspd_b(X, Y, S, P, D) :-
                edge_b(X, S), tspd_b(S, Y, _, P, D1), D is D1 + 1)).

% ============================================================
% 1. Oracle vs literal expectations
% ============================================================

:- begin_tests(tspd5_oracle).

test(literal_expectations_are_complete) :-
    forall(
        ( tspd5_fixture(Name, _Edges, Sources),
          member(Src, Sources)
        ),
        assertion(tspd5_fixture_expected(Name, Src, _))
    ).

test(oracle_matches_literal_expectations) :-
    forall(
        ( tspd5_fixture(Name, Edges, Sources),
          member(Src, Sources),
          tspd5_fixture_expected(Name, Src, Expected)
        ),
        ( tspd5_oracle_quads(Edges, Src, Got),
          assertion(Got == Expected)
        )
    ).

test(correlated_diamond_never_cross_products) :-
    tspd5_fixture_expected(correlated_diamond, a, Quads),
    assertion(memberchk(tspd(t, b, p, 3), Quads)),
    assertion(memberchk(tspd(t, c, q, 3), Quads)),
    assertion(\+ memberchk(tspd(t, b, q, 3), Quads)),
    assertion(\+ memberchk(tspd(t, c, p, 3), Quads)),
    tspd5_oracle_has([a-b, a-c, b-p, c-q, p-t, q-t], a, t, b, p, 3),
    tspd5_oracle_has([a-b, a-c, b-p, c-q, p-t, q-t], a, t, c, q, 3),
    \+ tspd5_oracle_has([a-b, a-c, b-p, c-q, p-t, q-t], a, t, b, q, 3),
    \+ tspd5_oracle_has([a-b, a-c, b-p, c-q, p-t, q-t], a, t, c, p, 3).

test(self_loop_and_cycle_finite) :-
    tspd5_oracle_has([a-a, a-b], a, a, a, a, 1),
    tspd5_oracle_has([a-b, b-a], a, a, b, b, 2),
    tspd5_oracle_quads([a-b, b-a], a, Quads),
    assertion(length(Quads, 2)),
    \+ tspd5_oracle_has([a-b, b-c], a, a, _, _, _).

test(direct_beats_longer) :-
    tspd5_oracle_has([a-b, b-c, c-t, a-t], a, t, t, a, 1),
    \+ tspd5_oracle_has([a-b, b-c, c-t, a-t], a, t, b, c, 3).

:- end_tests(tspd5_oracle).

% ============================================================
% 2. Structural correlated-pair pattern checks
% ============================================================

:- begin_tests(tspd5_structural).

test(fsharp_mustache_correlated_pairs) :-
    read_file_string(
        'templates/targets/fsharp_wam/kernel_transitive_step_parent_distance.fs.mustache', S),
    assertion(sub_string(S, _, _, _, "shortest-positive correlated step/parent")),
    assertion(sub_string(S, _, _, _, "let nativeKernel_transitive_step_parent_distance")),
    assertion(sub_string(S, _, _, _, "HashSet<int * int>")),
    assertion(sub_string(S, _, _, _, "let mutable frontier = [(source, 0)]")),
    assertion(sub_string(S, _, _, _, "never cross-product")),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_TRANSITIVE_STEP_PARENT_DISTANCE5_CONTRACT.md")).

test(fsharp_allowlist_includes_tspd5) :-
    assertion(wam_fsharp_native_kernel_kind(transitive_step_parent_distance5)),
    assertion(wam_fsharp_native_kernel_supported(
        recursive_kernel(transitive_step_parent_distance5, probe/0, []))).

test(haskell_mustache_correlated_pairs) :-
    read_file_string(
        'templates/targets/haskell_wam/kernel_transitive_step_parent_distance.hs.mustache', S),
    assertion(sub_string(S, _, _, _, "shortest-positive correlated step/parent")),
    assertion(sub_string(S, _, _, _, "nativeKernel_transitive_step_parent_distance")),
    assertion(sub_string(S, _, _, _, "go [(source, 0)] IM.empty IM.empty")),
    assertion(sub_string(S, _, _, _, "never an independent Step")),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_TRANSITIVE_STEP_PARENT_DISTANCE5_CONTRACT.md")).

test(rust_bfs_correlated_pairs_not_dfs) :-
    compile_wam_runtime_to_rust([], Code),
    atom_string(Code, S),
    StartPattern = "pub fn collect_native_transitive_step_parent_distance_results(",
    EndPattern = "pub fn collect_native_weighted_shortest_path_results(",
    sub_string(S, Start, _, _, StartPattern),
    sub_string(S, End, _, _, EndPattern),
    End > Start,
    BodyLen is End - Start,
    sub_string(S, Start, BodyLen, _, Body),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_TRANSITIVE_STEP_PARENT_DISTANCE5_CONTRACT.md")),
    assertion(sub_string(Body, _, _, _, "HashMap<String, HashSet<(String, String)>>")),
    assertion(sub_string(Body, _, _, _, "VecDeque<(String, i64)>")),
    assertion(\+ sub_string(Body, _, _, _,
        "let mut stack: Vec<(String, i64)> = vec![(start.to_string(), 0)];")),
    assertion(\+ sub_string(Body, _, _, _, "collect_legacy")),
    !.

test(go_correlated_pair_sets) :-
    read_file_string('src/unifyweaver/targets/wam_go_target.pl', S),
    Pattern = "func (vm *WamState) collectNativeTransitiveStepParentDistanceResults",
    EndPattern = "func (vm *WamState) collectNativeCategoryAncestorHops",
    sub_string(S, Start, _, _, Pattern),
    sub_string(S, End, _, _, EndPattern),
    End > Start,
    BodyLen is End - Start,
    sub_string(S, Start, BodyLen, _, Body),
    assertion(sub_string(Body, _, _, _,
        "docs/design/WAM_TRANSITIVE_STEP_PARENT_DISTANCE5_CONTRACT.md")),
    assertion(sub_string(Body, _, _, _, "pairSets := make(map[string]map[[2]string]bool)")),
    assertion(sub_string(Body, _, _, _, "never an independent Step")),
    assertion(\+ sub_string(Body, _, _, _,
        "visited := map[string]bool{source: true}")),
    !.

test(scala_correlated_pair_sets) :-
    read_file_string('src/unifyweaver/targets/wam_scala_target.pl', S),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_TRANSITIVE_STEP_PARENT_DISTANCE5_CONTRACT.md")),
    assertion(sub_string(S, _, _, _,
        "LinkedHashSet[(WamTerm, WamTerm)]")),
    assertion(sub_string(S, _, _, _,
        "val pairs = scala.collection.mutable.LinkedHashMap[WamTerm, scala.collection.mutable.LinkedHashSet[(WamTerm, WamTerm)]]()")).

test(c_relation_isolation_and_quad_stream_253) :-
    read_file_string('src/unifyweaver/targets/wam_c_target.pl', S),
    assertion(sub_string(S, _, _, _, "wam_collect_transitive_step_parent_distance")),
    assertion(sub_string(S, _, _, _, "wam_register_relation_edge")),
    assertion(sub_string(S, _, _, _, "wam_bind_foreign_quad_stream")),
    assertion(sub_string(S, _, _, _, "result_reg == 253")),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_TRANSITIVE_STEP_PARENT_DISTANCE5_CONTRACT.md")).

test(r_correlated_pairs_no_source_seed) :-
    read_file_string('templates/targets/r_wam/runtime.R.mustache', S),
    assertion(sub_string(S, _, _, _, "WamRuntime$transitive_step_parent_distance5")),
    assertion(sub_string(S, _, _, _, "Do NOT seed Source into dist")),
    assertion(sub_string(S, _, _, _, "pairs_env")),
    assertion(sub_string(S, _, _, _, "never an independent Step")).

test(elixir_bfs_correlated_quads) :-
    read_file_string('src/unifyweaver/targets/wam_elixir_target.pl', S),
    assertion(sub_string(S, _, _, _, "WamRuntime.GraphKernel.TransitiveStepParentDistance")),
    assertion(sub_string(S, _, _, _, "shortest-positive correlated")),
    assertion(sub_string(S, _, _, _, "def collect_quads(")),
    assertion(sub_string(S, _, _, _, "Never cross-product")),
    assertion(sub_string(S, _, _, _, ":queue.in({start, 0}")).

test(llvm_remains_capability_gated) :-
    read_file_string('docs/WAM_LLVM_STATUS.md', S),
    assertion(sub_string(S, _, _, _, "transitive_step_parent_distance5")),
    read_file_string('templates/targets/llvm_wam/state.ll.mustache', LL),
    assertion(\+ sub_string(LL, _, _, _, "nativeKernel_transitive_step_parent_distance")),
    assertion(\+ sub_string(LL, _, _, _, "wam_tspd5")).

:- end_tests(tspd5_structural).

% ============================================================
% 3. Native dispatch proof
% ============================================================

:- begin_tests(tspd5_native_dispatch).

test(detector_fires_transitive_step_parent_distance5) :-
    assert_tspd_cycle_program,
    findall(tspd(X, Y, S, P, D)-Body,
            clause(user:tspd(X, Y, S, P, D), Body), Clauses),
    detect_recursive_kernel(tspd, 5, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(transitive_step_parent_distance5, _, _)).

test(fsharp_native_tspd5_dispatch_emitted) :-
    assert_tspd_cycle_program,
    tmp_dir(fs_dispatch, Dir),
    write_wam_fsharp_project(
        [user:tspd/5, user:tspd_edge/2],
        [module_name('uw_tspd5_dispatch')],
        Dir),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    read_file_string(RT, RTS),
    assertion(sub_string(RTS, _, _, _, "nativeKernel_transitive_step_parent_distance")),
    assertion(sub_string(RTS, _, _, _, "| \"tspd/5\" ->")),
    assertion(sub_string(RTS, _, _, _, "FFIStreamRetry")),
    !.

test(rust_foreign_kind_registered_for_tspd5) :-
    assert_tspd_cycle_program,
    tmp_dir(rs_dispatch, Dir),
    write_wam_rust_project(
        [user:tspd/5, user:tspd_edge/2],
        [module_name('uw_tspd5_rs_dispatch'), foreign_lowering(true)],
        Dir),
    directory_file_path(Dir, 'src/lib.rs', Lib),
    read_file_string(Lib, S),
    assertion(sub_string(S, _, _, _, "transitive_step_parent_distance5")),
    assertion(sub_string(S, _, _, _, "register_foreign_native_kind(\"tspd/5\"")),
    !.

test(c_registers_tspd5_with_edge_pred_and_relation_apis) :-
    assert_c_tspd_two_pred_program,
    tmp_dir(c_dispatch, Dir),
    write_wam_c_project([user:tspd_a/5, user:tspd_b/5], [], Dir),
    directory_file_path(Dir, 'lib.c', Lib),
    read_file_string(Lib, LibS),
    assertion(sub_string(LibS, _, _, _,
        "wam_register_transitive_step_parent_distance_kernel(state, \"tspd_a/5\", \"edge_a\")")),
    assertion(sub_string(LibS, _, _, _,
        "wam_register_transitive_step_parent_distance_kernel(state, \"tspd_b/5\", \"edge_b\")")),
    directory_file_path(Dir, 'wam_runtime.c', RT),
    read_file_string(RT, S),
    assertion(sub_string(S, _, _, _, "wam_collect_transitive_step_parent_distance")),
    assertion(sub_string(S, _, _, _, "wam_register_relation_edge")),
    !.

:- end_tests(tspd5_native_dispatch).

% ============================================================
% 4. Executable smokes
% ============================================================

:- begin_tests(tspd5_executable).

test(fsharp_stream_diamond_alias_cut_e2e, [condition(dotnet_available)]) :-
    assert_tspd_cycle_program,
    tmp_dir(fs_e2e, Dir),
    once(write_wam_fsharp_project(
        [ user:tspd/5, user:tspd_edge/2,
          user:tspd_tail/4, user:tspd_after/4, user:tspd_cut/4,
          user:tspd_call_after/4
        ],
        [module_name('uw_tspd5_e2e')],
        Dir)),
    directory_file_path(Dir, 'Program.fs', Prog),
    tspd5_write_fsharp_driver(Prog),
    run_dotnet_build(Dir, BuildExit, BuildOut),
    ( BuildExit =:= 0 -> true
    ; format(user_error, 'fsharp tspd5 e2e build:~n~w~n', [BuildOut]),
      assertion(BuildExit =:= 0), fail
    ),
    run_dotnet_run(Dir, RunExit, RunOut),
    ( RunExit =:= 0 -> true
    ; format(user_error, 'fsharp tspd5 e2e run:~n~w~n', [RunOut]),
      assertion(RunExit =:= 0), fail
    ),
    assertion(sub_string(RunOut, _, _, _, "OK stream_from_a")),
    assertion(sub_string(RunOut, _, _, _, "OK correlated_diamond")),
    assertion(sub_string(RunOut, _, _, _, "OK no_cross_product")),
    assertion(sub_string(RunOut, _, _, _, "OK alias_later_match")),
    assertion(sub_string(RunOut, _, _, _, "OK cut_after_foreign")),
    !.

test(c_two_pred_isolation_and_diamond, [condition(gcc_available)]) :-
    assert_c_tspd_two_pred_program,
    tmp_dir(c_e2e, Dir),
    write_wam_c_project([user:tspd_a/5, user:tspd_b/5], [], Dir),
    directory_file_path(Dir, 'wam_runtime.c', RuntimePath),
    directory_file_path(Dir, 'lib.c', LibPath),
    directory_file_path(Dir, 'main.c', MainPath),
    directory_file_path(Dir, 'tspd5_c_smoke', ExePath),
    tspd5_c_two_pred_main(MainCode),
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

test(rust_collect_tspd5_unit, [condition(cargo_available)]) :-
    assert_tspd_diamond_program,
    tmp_dir(rs_e2e, Dir),
    write_wam_rust_project(
        [user:tspd/5, user:tspd_edge/2],
        [module_name('uw_tspd5_rs'), foreign_lowering(true)],
        Dir),
    tspd5_append_rust_unit(Dir),
    format(atom(Cmd),
        'cd ~w && cargo test --quiet tspd5_contract -- --nocapture >~w/cargo.out 2>~w/cargo.err',
        [Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; directory_file_path(Dir, 'cargo.err', ErrPath),
      directory_file_path(Dir, 'cargo.out', OutPath),
      ( exists_file(ErrPath) -> read_file_string(ErrPath, Err) ; Err = "" ),
      ( exists_file(OutPath) -> read_file_string(OutPath, Out) ; Out = "" ),
      format(user_error, 'rust tspd5 unit failed:~n~w~n~w~n', [Out, Err]),
      fail
    ),
    !.

% Generated-kernel unit; it does not claim dispatch/register end-to-end coverage.
test(elixir_collect_quads_unit, [condition(elixir_available)]) :-
    tmp_dir(ex_e2e, Dir),
    compile_wam_runtime_snippet_for_elixir_tspd5(Dir),
    directory_file_path(Dir, 'tspd5_unit.exs', Script),
    tspd5_write_elixir_unit(Script),
    format(atom(Cmd),
        'cd ~w && elixir tspd5_unit.exs >~w/elixir.out 2>~w/elixir.err',
        [Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; directory_file_path(Dir, 'elixir.err', ErrPath),
      directory_file_path(Dir, 'elixir.out', OutPath),
      ( exists_file(ErrPath) -> read_file_string(ErrPath, Err) ; Err = "" ),
      ( exists_file(OutPath) -> read_file_string(OutPath, Out) ; Out = "" ),
      format(user_error, 'elixir tspd5 unit failed:~n~w~n~w~n', [Out, Err]),
      fail
    ),
    !.

:- end_tests(tspd5_executable).

% ============================================================
% 5. Acyclic no_kernels fallback
% ============================================================

:- begin_tests(tspd5_no_kernels_acyclic).

test(fsharp_no_kernels_omits_native_body, [condition(dotnet_available)]) :-
    assert_tspd_chain_program,
    tmp_dir(fs_nk, Dir),
    once(write_wam_fsharp_project(
        [user:tspd/5, user:tspd_edge/2],
        [no_kernels(true), module_name('uw_tspd5_nk'), conformance_main(true)],
        Dir)),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    read_file_string(RT, RTS),
    assertion(\+ sub_string(RTS, _, _, _, "nativeKernel_transitive_step_parent_distance")),
    run_dotnet_build(Dir, Exit, Out),
    ( Exit =:= 0 -> true
    ; format(user_error, 'fsharp no-kernels build:~n~w~n', [Out]),
      assertion(Exit =:= 0), fail
    ),
    !.

:- end_tests(tspd5_no_kernels_acyclic).

% ============================================================
% Helpers
% ============================================================

tspd5_write_fsharp_driver(ProgPath) :-
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

let mkAtoms names =
    let pairs = names |> List.mapi (fun i n -> (n, i + 1))
    Map.ofList pairs, Map.ofList (pairs |> List.map (fun (s, i) -> (i, s)))

let mkEdgeFacts (intern: Map<string, int>) edges =
    let grouped =
        edges
        |> List.map (fun (f, t) -> Map.find f intern, Map.find t intern)
        |> List.groupBy fst
        |> List.map (fun (k, vs) -> k, vs |> List.map snd)
    Map.ofList [(\"tspd_edge\", Map.ofList grouped)]

let mkContext edges atomNames =
    let foreignPreds = [\"tspd/5\"]
    let resolvedCode =
        resolveCallInstrs allLabels foreignPreds (Array.toList allCode)
        |> List.toArray
    let intern, deintern = mkAtoms atomNames
    { WcCode              = resolvedCode
      WcLabels            = allLabels
      WcForeignFacts      = Map.empty
      WcFfiFacts          = mkEdgeFacts intern edges
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

let collectQuads (ctx: WamContext) (source: string)
                 (boundT: string option) (boundS: string option)
                 (boundP: string option) (boundD: int option) =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom source
    match boundT with
    | Some t -> regs.[2] <- Atom t
    | None -> regs.[2] <- Unbound 100
    match boundS with
    | Some s -> regs.[3] <- Atom s
    | None -> regs.[3] <- Unbound 101
    match boundP with
    | Some p -> regs.[4] <- Atom p
    | None -> regs.[4] <- Unbound 102
    match boundD with
    | Some d -> regs.[5] <- Integer d
    | None -> regs.[5] <- Unbound 103
    match callForeign ctx \"tspd/5\" (mkState regs) with
    | None -> []
    | Some s1 ->
        let readQuad (s: WamState) =
            let t =
                match getReg 2 s with
                | Some (Atom a) -> a
                | _ -> \"?\"
            let step =
                match getReg 3 s with
                | Some (Atom a) -> a
                | _ -> \"?\"
            let p =
                match getReg 4 s with
                | Some (Atom a) -> a
                | _ -> \"?\"
            let d =
                match getReg 5 s with
                | Some (Integer i) -> i
                | _ -> -1
            t, step, p, d
        let rec gather (s: WamState) (acc: (string * string * string * int) list) =
            let q = readQuad s
            match backtrack s with
            | Some s2 -> gather s2 (q :: acc)
            | None -> List.rev (q :: acc)
        gather s1 []

let runQuadWrapper (ctx: WamContext) (pred: string) : WamState option =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Unbound 200
    regs.[2] <- Unbound 201
    regs.[3] <- Unbound 202
    regs.[4] <- Unbound 203
    dispatchCall ctx pred (mkState regs)

let readWrapperQuad (s: WamState) =
    match derefVar s.WsBindings (Unbound 200),
          derefVar s.WsBindings (Unbound 201),
          derefVar s.WsBindings (Unbound 202),
          derefVar s.WsBindings (Unbound 203) with
    | Atom t, Atom step, Atom p, Integer d -> Some (t, step, p, d)
    | _ -> None

// Target=Parent alias: (b,b,a,1) is incompatible; self-loop (a,a,a,1)
// succeeds later (Target=Step=Parent on the self-loop).
let aliasedTargetParentLater (ctx: WamContext) =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom \"a\"
    regs.[2] <- Unbound 160
    regs.[3] <- Unbound 161
    regs.[4] <- Unbound 160
    regs.[5] <- Integer 1
    match callForeign ctx \"tspd/5\" (mkState regs) with
    | Some s ->
        derefVar s.WsBindings (Unbound 160) = Atom \"a\" &&
        getReg 2 s = Some (Atom \"a\") &&
        getReg 3 s = Some (Atom \"a\") &&
        getReg 4 s = Some (Atom \"a\") &&
        Option.isNone (backtrack s)
    | None -> false

[<EntryPoint>]
let main _argv =
    let cycleEdges = [(\"a\", \"b\"); (\"b\", \"c\"); (\"c\", \"d\"); (\"c\", \"a\")]
    let ctx = mkContext cycleEdges [\"a\"; \"b\"; \"c\"; \"d\"]
    let fromA = collectQuads ctx \"a\" None None None None |> List.sort
    assertTrue \"stream_from_a\"
        (fromA = [(\"a\", \"b\", \"c\", 3); (\"b\", \"b\", \"a\", 1);
                  (\"c\", \"b\", \"b\", 2); (\"d\", \"b\", \"c\", 3)])

    let diamondEdges =
        [(\"a\", \"b\"); (\"a\", \"c\"); (\"b\", \"p\"); (\"c\", \"q\"); (\"p\", \"t\"); (\"q\", \"t\")]
    let dctx = mkContext diamondEdges [\"a\"; \"b\"; \"c\"; \"p\"; \"q\"; \"t\"]
    let diamond = collectQuads dctx \"a\" (Some \"t\") None None None |> List.sort
    assertTrue \"correlated_diamond\"
        (diamond = [(\"t\", \"b\", \"p\", 3); (\"t\", \"c\", \"q\", 3)])
    assertTrue \"no_cross_product\"
        (not (List.contains (\"t\", \"b\", \"q\", 3) diamond) &&
         not (List.contains (\"t\", \"c\", \"p\", 3) diamond))

    let aliasCtx = mkContext [(\"a\", \"b\"); (\"a\", \"a\")] [\"a\"; \"b\"]
    assertTrue \"alias_later_match\" (aliasedTargetParentLater aliasCtx)

    let cutResult = runQuadWrapper ctx \"tspd_cut/4\"
    assertTrue \"cut_after_foreign\"
        (cutResult |> Option.exists (fun s ->
            readWrapperQuad s = Some (\"b\", \"b\", \"a\", 1) &&
            s.WsCP = 0 && s.WsCutBar = 0 && List.isEmpty s.WsB0Stack &&
            s.WsCPsLen = 0 && List.isEmpty s.WsCPs &&
            s.WsTrailLen = List.length s.WsTrail &&
            Option.isNone (backtrack s)))

    printfn \"RESULT %d/%d\" passes (passes + fails)
    if fails > 0 then 1 else 0
",
    setup_call_cleanup(
        open(ProgPath, write, Out, [encoding(utf8)]),
        write(Out, Driver),
        close(Out)).

tspd5_c_two_pred_main(Code) :-
    Code =
'#include "wam_runtime.h"
#include <string.h>

void setup_tspd_a_5(WamState* state);
void setup_tspd_b_5(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

static int expect_quad(WamState *state, const char *pred,
                       const char *source, const char *target,
                       const char *step, const char *parent, int distance) {
    WamValue args[5] = {
        val_atom(source),
        val_atom(target),
        val_unbound("Step"),
        val_unbound("Parent"),
        val_unbound("Distance")
    };
    int rc = wam_run_predicate(state, pred, args, 5);
    if (rc != 0 || state->P != WAM_HALT) return 1;
    WamValue *step_v = wam_deref_ptr(state, &state->A[2]);
    WamValue *parent_v = wam_deref_ptr(state, &state->A[3]);
    WamValue *dist_v = wam_deref_ptr(state, &state->A[4]);
    if (step_v->tag != VAL_ATOM || parent_v->tag != VAL_ATOM ||
        dist_v->tag != VAL_INT) {
        return 2;
    }
    if (strcmp(step_v->data.atom, step) != 0 ||
        strcmp(parent_v->data.atom, parent) != 0 ||
        dist_v->data.integer != distance) {
        return 3;
    }
    return 0;
}

static int expect_fail(WamState *state, const char *pred,
                       const char *source, const char *target) {
    WamValue args[5] = {
        val_atom(source),
        val_atom(target),
        val_unbound("Step"),
        val_unbound("Parent"),
        val_unbound("Distance")
    };
    int rc = wam_run_predicate(state, pred, args, 5);
    return (rc == WAM_HALT) ? 0 : 1;
}

static int run_isolation(void) {
    WamState state;
    int rc = 0;
    wam_state_init(&state);
    setup_tspd_a_5(&state);
    setup_tspd_b_5(&state);
    setup_detected_wam_c_kernels(&state);

    /* Distinct relations — never the global transitive edge bag. */
    wam_register_relation_edge(&state, "edge_a", "a", "b");
    wam_register_relation_edge(&state, "edge_a", "b", "c");
    wam_register_relation_edge(&state, "edge_b", "x", "y");

    if (expect_quad(&state, "tspd_a/5", "a", "c", "b", "b", 2) != 0) {
        rc = 10; goto done;
    }
    /* tspd_a must not see edge_b. */
    if (expect_fail(&state, "tspd_a/5", "x", "y") != 0) {
        rc = 11; goto done;
    }
    if (expect_quad(&state, "tspd_b/5", "x", "y", "y", "x", 1) != 0) {
        rc = 12; goto done;
    }
    /* tspd_b must not see edge_a. */
    if (expect_fail(&state, "tspd_b/5", "a", "b") != 0) {
        rc = 13; goto done;
    }

done:
    wam_free_state(&state);
    return rc;
}

static int run_correlated_diamond(void) {
    WamState state;
    int rc = 0;
    wam_state_init(&state);
    setup_tspd_a_5(&state);
    setup_detected_wam_c_kernels(&state);

    wam_register_relation_edge(&state, "edge_a", "a", "b");
    wam_register_relation_edge(&state, "edge_a", "a", "c");
    wam_register_relation_edge(&state, "edge_a", "b", "p");
    wam_register_relation_edge(&state, "edge_a", "c", "q");
    wam_register_relation_edge(&state, "edge_a", "p", "t");
    wam_register_relation_edge(&state, "edge_a", "q", "t");

    /* Bound step/parent selects one correlated pair. */
    {
        WamValue args[5] = {
            val_atom("a"),
            val_atom("t"),
            val_atom("b"),
            val_atom("p"),
            val_int(3)
        };
        int r = wam_run_predicate(&state, "tspd_a/5", args, 5);
        if (r != 0 || state.P != WAM_HALT) { rc = 20; goto done; }
    }
    {
        WamValue args[5] = {
            val_atom("a"),
            val_atom("t"),
            val_atom("c"),
            val_atom("q"),
            val_int(3)
        };
        int r = wam_run_predicate(&state, "tspd_a/5", args, 5);
        if (r != 0 || state.P != WAM_HALT) { rc = 21; goto done; }
    }
    /* Cross-products must fail. */
    {
        WamValue args[5] = {
            val_atom("a"),
            val_atom("t"),
            val_atom("b"),
            val_atom("q"),
            val_int(3)
        };
        int r = wam_run_predicate(&state, "tspd_a/5", args, 5);
        if (r != WAM_HALT) { rc = 22; goto done; }
    }
    {
        WamValue args[5] = {
            val_atom("a"),
            val_atom("t"),
            val_atom("c"),
            val_atom("p"),
            val_int(3)
        };
        int r = wam_run_predicate(&state, "tspd_a/5", args, 5);
        if (r != WAM_HALT) { rc = 23; goto done; }
    }

done:
    wam_free_state(&state);
    return rc;
}

int main(void) {
    int rc = run_isolation();
    if (rc != 0) return rc;
    rc = run_correlated_diamond();
    if (rc != 0) return rc;
    return 0;
}
'.

tspd5_append_rust_unit(Dir) :-
    directory_file_path(Dir, 'src/lib.rs', Path),
    read_file_string(Path, Existing),
    Unit =
"

#[cfg(test)]
mod tspd5_contract {
    use super::*;
    use std::collections::HashMap;
    use state::WamState;

    #[test]
    fn correlated_diamond_emits_pairs_not_cross_product() {
        let mut vm = WamState::new(Vec::new(), HashMap::new());
        vm.register_indexed_atom_fact2_pairs(
            \"tspd_edge\",
            &[(\"a\", \"b\"), (\"a\", \"c\"), (\"b\", \"p\"), (\"c\", \"q\"), (\"p\", \"t\"), (\"q\", \"t\")],
        );
        let mut nodes = Vec::new();
        vm.collect_native_transitive_step_parent_distance_results(
            \"a\", \"tspd_edge\", &mut nodes);
        nodes.sort();
        assert_eq!(
            nodes,
            vec![
                (\"b\".to_string(), \"b\".to_string(), \"a\".to_string(), 1),
                (\"c\".to_string(), \"c\".to_string(), \"a\".to_string(), 1),
                (\"p\".to_string(), \"b\".to_string(), \"b\".to_string(), 2),
                (\"q\".to_string(), \"c\".to_string(), \"c\".to_string(), 2),
                (\"t\".to_string(), \"b\".to_string(), \"p\".to_string(), 3),
                (\"t\".to_string(), \"c\".to_string(), \"q\".to_string(), 3),
            ]
        );
        assert!(!nodes.iter().any(|q| {
            q == &(\"t\".to_string(), \"b\".to_string(), \"q\".to_string(), 3)
        }));
        assert!(!nodes.iter().any(|q| {
            q == &(\"t\".to_string(), \"c\".to_string(), \"p\".to_string(), 3)
        }));
    }
}
",
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        ( write(Out, Existing), write(Out, Unit) ),
        close(Out)).

:- use_module('../src/unifyweaver/targets/wam_elixir_target',
              [compile_wam_runtime_to_elixir/2]).

compile_wam_runtime_snippet_for_elixir_tspd5(Dir) :-
    compile_wam_runtime_to_elixir([], RuntimeCode),
    Pattern = "defmodule WamRuntime.GraphKernel.TransitiveStepParentDistance do",
    EndPattern = "defmodule WamRuntime.GraphKernel.WeightedShortestPath do",
    sub_string(RuntimeCode, Start, _, _, Pattern),
    sub_string(RuntimeCode, End, _, _, EndPattern),
    End > Start,
    BodyLen is End - Start,
    sub_string(RuntimeCode, Start, BodyLen, _, Body),
    directory_file_path(Dir, 'tspd5_kernel.ex', Path),
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Body),
        close(Out)).

tspd5_write_elixir_unit(Script) :-
    Code =
'Code.require_file("tspd5_kernel.ex")

neighbors = fn
  "a" -> [{"a", "b"}, {"a", "c"}]
  "b" -> [{"b", "p"}]
  "c" -> [{"c", "q"}]
  "p" -> [{"p", "t"}]
  "q" -> [{"q", "t"}]
  _ -> []
end

quads = WamRuntime.GraphKernel.TransitiveStepParentDistance.collect_quads(neighbors, "a")
expected = [
  {"b", "b", "a", 1},
  {"c", "c", "a", 1},
  {"p", "b", "b", 2},
  {"q", "c", "c", 2},
  {"t", "b", "p", 3},
  {"t", "c", "q", 3}
]
unless quads == expected do
  IO.puts("FAIL diamond #{inspect(quads)}")
  System.halt(1)
end

if Enum.any?(quads, &(&1 == {"t", "b", "q", 3})) or
     Enum.any?(quads, &(&1 == {"t", "c", "p", 3})) do
  IO.puts("FAIL cross-product #{inspect(quads)}")
  System.halt(1)
end

IO.puts("OK elixir_tspd5")
',
    setup_call_cleanup(
        open(Script, write, Out, [encoding(utf8)]),
        write(Out, Code),
        close(Out)).
