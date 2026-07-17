:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_tpd4_contract_parity.pl — fleet-wide transitive_parent_distance4
% contract (shortest-positive parents) parity suite.
%
% Contract: docs/design/WAM_TRANSITIVE_PARENT_DISTANCE4_CONTRACT.md
% Oracle:   tests/fixtures/tpd4_contract_oracle.pl
%
% Usage (from repo root):
%   swipl -q -g run_tests -t halt tests/test_wam_tpd4_contract_parity.pl

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module(library(readutil)).
:- use_module(library(process)).
:- use_module(library(debug), [assertion/1]).

:- use_module('fixtures/tpd4_contract_oracle', [
    tpd4_oracle_triples/3,
    tpd4_oracle_has/5,
    tpd4_fixture/3,
    tpd4_fixture_expected/3
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

:- dynamic user:tpd_edge/2.
:- dynamic user:tpd/4.
:- dynamic user:tpd_tail/3.
:- dynamic user:tpd_after/3.
:- dynamic user:tpd_cut/3.
:- dynamic user:tpd_call_after/3.
:- dynamic user:pd_parent/2.
:- dynamic user:pd/4.

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

elixir_available :-
    catch(
        ( process_create(path(elixir), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

tmp_dir(Tag, Dir) :-
    get_time(T),
    Stamp is round(T * 1000000),
    format(atom(Dir), '/tmp/uw_tpd4_~w_~w', [Tag, Stamp]),
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir).

read_file_string(Path, String) :-
    read_file_to_string(Path, String, []).

assert_tpd_cycle_program :-
    retractall(user:tpd_edge(_, _)),
    retractall(user:tpd(_, _, _, _)),
    retractall(user:tpd_tail(_, _, _)),
    retractall(user:tpd_after(_, _, _)),
    retractall(user:tpd_cut(_, _, _)),
    retractall(user:tpd_call_after(_, _, _)),
    assertz(user:tpd_edge(a, b)),
    assertz(user:tpd_edge(b, c)),
    assertz(user:tpd_edge(c, d)),
    assertz(user:tpd_edge(c, a)),
    assertz((user:tpd(X, Y, X, 1) :- tpd_edge(X, Y))),
    assertz((user:tpd(X, Y, P, D) :-
                tpd_edge(X, Z), tpd(Z, Y, P, D1), D is D1 + 1)),
    assertz((user:tpd_tail(Y, P, D) :- tpd(a, Y, P, D))),
    assertz((user:tpd_after(Y, P, D) :- tpd_tail(Y, P, D), Y \== b)),
    assertz((user:tpd_cut(Y, P, D) :- tpd_tail(Y, P, D), !)),
    assertz((user:tpd_call_after(Y, P, D) :- tpd(a, Y, P, D), Y \== b)).

assert_tpd_diamond_program :-
    retractall(user:tpd_edge(_, _)),
    retractall(user:tpd(_, _, _, _)),
    assertz(user:tpd_edge(a, b)),
    assertz(user:tpd_edge(a, c)),
    assertz(user:tpd_edge(b, d)),
    assertz(user:tpd_edge(c, d)),
    assertz((user:tpd(X, Y, X, 1) :- tpd_edge(X, Y))),
    assertz((user:tpd(X, Y, P, D) :-
                tpd_edge(X, Z), tpd(Z, Y, P, D1), D is D1 + 1)).

assert_tpd_chain_program :-
    retractall(user:tpd_edge(_, _)),
    retractall(user:tpd(_, _, _, _)),
    assertz(user:tpd_edge(a, b)),
    assertz(user:tpd_edge(b, c)),
    assertz(user:tpd_edge(c, d)),
    assertz((user:tpd(X, Y, X, 1) :- tpd_edge(X, Y))),
    assertz((user:tpd(X, Y, P, D) :-
                tpd_edge(X, Z), tpd(Z, Y, P, D1), D is D1 + 1)).

assert_c_tpd_program :-
    retractall(user:pd_parent(_, _)),
    retractall(user:pd(_, _, _, _)),
    assertz(user:pd_parent(a, b)),
    assertz(user:pd_parent(b, c)),
    assertz(user:pd_parent(c, d)),
    assertz(user:pd_parent(c, a)),
    assertz((user:pd(X, Y, X, 1) :- pd_parent(X, Y))),
    assertz((user:pd(X, Y, P, D) :-
                pd_parent(X, Z), pd(Z, Y, P, D0), D is D0 + 1)).

% ============================================================
% 1. Oracle vs literal expectations
% ============================================================

:- begin_tests(tpd4_oracle).

test(literal_expectations_are_complete) :-
    forall(
        ( tpd4_fixture(Name, _Edges, Sources),
          member(Src, Sources)
        ),
        assertion(tpd4_fixture_expected(Name, Src, _))
    ).

test(oracle_matches_literal_expectations) :-
    forall(
        ( tpd4_fixture(Name, Edges, Sources),
          member(Src, Sources),
          tpd4_fixture_expected(Name, Src, Expected)
        ),
        ( tpd4_oracle_triples(Edges, Src, Got),
          assertion(Got == Expected)
        )
    ).

test(equal_diamond_emits_both_parents) :-
    tpd4_fixture_expected(equal_diamond, a, Triples),
    assertion(memberchk(tpd(d, b, 2), Triples)),
    assertion(memberchk(tpd(d, c, 2), Triples)).

test(unequal_paths_select_shorter) :-
    tpd4_oracle_has([a-b, b-c, c-t, a-t], a, t, a, 1),
    \+ tpd4_oracle_has([a-b, b-c, c-t, a-t], a, t, c, 3).

test(self_loop_and_cycle_source) :-
    tpd4_oracle_has([a-a, a-b], a, a, a, 1),
    tpd4_oracle_has([a-b, b-a], a, a, b, 2),
    \+ tpd4_oracle_has([a-b, b-c], a, a, _, _).

test(acyclic_source_not_emitted) :-
    tpd4_fixture_expected(chain, a, Triples),
    \+ memberchk(tpd(a, _, _), Triples).

:- end_tests(tpd4_oracle).

% ============================================================
% 2. Structural shortest-positive-parent pattern checks
% ============================================================

:- begin_tests(tpd4_structural).

test(fsharp_mustache_parent_sets) :-
    read_file_string(
        'templates/targets/fsharp_wam/kernel_transitive_parent_distance.fs.mustache', S),
    assertion(sub_string(S, _, _, _, "shortest-positive parents")),
    assertion(sub_string(S, _, _, _, "let nativeKernel_transitive_parent_distance")),
    assertion(sub_string(S, _, _, _, "Dictionary<int, int>()")),
    assertion(sub_string(S, _, _, _, "HashSet<int>")),
    assertion(sub_string(S, _, _, _, "let mutable frontier = [(source, 0)]")),
    assertion(\+ sub_string(S, _, _, _, "visited.Add(source)")).

test(fsharp_allowlist_includes_tpd4) :-
    assertion(wam_fsharp_native_kernel_kind(transitive_parent_distance4)),
    assertion(wam_fsharp_native_kernel_supported(
        recursive_kernel(transitive_parent_distance4, probe/0, []))).

test(haskell_mustache_parent_sets) :-
    read_file_string(
        'templates/targets/haskell_wam/kernel_transitive_parent_distance.hs.mustache', S),
    assertion(sub_string(S, _, _, _, "shortest-positive parents")),
    assertion(sub_string(S, _, _, _, "IS.singleton node")),
    assertion(sub_string(S, _, _, _, "go [(source, 0)] IM.empty IM.empty")),
    assertion(\+ sub_string(S, _, _, _, "Don't emit the source")).

test(rust_bfs_parent_sets_not_dfs) :-
    compile_wam_runtime_to_rust([], Code),
    atom_string(Code, S),
    StartPattern = "pub fn collect_native_transitive_parent_distance_results(",
    EndPattern = "pub fn collect_native_transitive_step_parent_distance_results(",
    sub_string(S, Start, _, _, StartPattern),
    sub_string(S, End, _, _, EndPattern),
    End > Start,
    BodyLen is End - Start,
    sub_string(S, Start, BodyLen, _, Body),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_TRANSITIVE_PARENT_DISTANCE4_CONTRACT.md")),
    assertion(sub_string(Body, _, _, _, "HashMap<String, HashSet<String>>")),
    assertion(sub_string(Body, _, _, _, "VecDeque<(String, i64)>")),
    assertion(\+ sub_string(Body, _, _, _,
        "let mut stack: Vec<(String, i64)> = vec![(start.to_string(), 0)];")),
    !.

test(go_parent_sets_no_source_seed) :-
    read_file_string('src/unifyweaver/targets/wam_go_target.pl', S),
    Pattern = "func (vm *WamState) collectNativeTransitiveParentDistanceResults",
    EndPattern = "func (vm *WamState) collectNativeTransitiveStepParentDistanceResults",
    sub_string(S, Start, _, _, Pattern),
    sub_string(S, End, _, _, EndPattern),
    End > Start,
    BodyLen is End - Start,
    sub_string(S, Start, BodyLen, _, Body),
    assertion(sub_string(Body, _, _, _,
        "docs/design/WAM_TRANSITIVE_PARENT_DISTANCE4_CONTRACT.md")),
    assertion(sub_string(Body, _, _, _, "parents := make(map[string]map[string]bool)")),
    assertion(\+ sub_string(Body, _, _, _,
        "visited := map[string]bool{source: true}")),
    assertion(\+ sub_string(Body, _, _, _, "dist := map[string]int{source: 0}")),
    !.

test(scala_parent_sets_no_source_seed) :-
    read_file_string('src/unifyweaver/targets/wam_scala_target.pl', S),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_TRANSITIVE_PARENT_DISTANCE4_CONTRACT.md")),
    assertion(sub_string(S, _, _, _, "LinkedHashSet[WamTerm]")),
    % Anchor to the TPD4 handler emission (TSPD5 uses correlated pair sets).
    assertion(sub_string(S, _, _, _,
        "val dist = scala.collection.mutable.LinkedHashMap[WamTerm, Int]()")),
    assertion(sub_string(S, _, _, _,
        "parents(nb) = scala.collection.mutable.LinkedHashSet(node)")).

test(c_collects_and_streams_triples) :-
    read_file_string('src/unifyweaver/targets/wam_c_target.pl', S),
    assertion(sub_string(S, _, _, _, "wam_collect_transitive_parent_distance")),
    assertion(sub_string(S, _, _, _, "wam_bind_foreign_triple_stream")),
    assertion(sub_string(S, _, _, _, "result_reg == 254")),
    assertion(sub_string(S, _, _, _,
        "docs/design/WAM_TRANSITIVE_PARENT_DISTANCE4_CONTRACT.md")).

test(r_parent_sets_no_source_seed) :-
    read_file_string('templates/targets/r_wam/runtime.R.mustache', S),
    assertion(sub_string(S, _, _, _, "WamRuntime$transitive_parent_distance4")),
    assertion(sub_string(S, _, _, _, "Do NOT seed Source into dist")),
    assertion(sub_string(S, _, _, _, "parents_env")).

test(elixir_bfs_parent_sets) :-
    read_file_string('src/unifyweaver/targets/wam_elixir_target.pl', S),
    assertion(sub_string(S, _, _, _, "WamRuntime.GraphKernel.TransitiveParentDistance")),
    assertion(sub_string(S, _, _, _, "shortest-positive parents")),
    assertion(sub_string(S, _, _, _, "compatible_triples")),
    assertion(sub_string(S, _, _, _, "stream_triples")),
    assertion(sub_string(S, _, _, _, "MapSet.new([node])")).

test(llvm_remains_capability_gated) :-
    read_file_string('docs/WAM_LLVM_STATUS.md', S),
    assertion(sub_string(S, _, _, _, "transitive_parent_distance4")),
    read_file_string('templates/targets/llvm_wam/state.ll.mustache', LL),
    assertion(\+ sub_string(LL, _, _, _, "nativeKernel_transitive_parent_distance")),
    assertion(\+ sub_string(LL, _, _, _, "wam_tpd4")).

:- end_tests(tpd4_structural).

% ============================================================
% 3. Native dispatch proof
% ============================================================

:- begin_tests(tpd4_native_dispatch).

test(detector_fires_transitive_parent_distance4) :-
    assert_tpd_cycle_program,
    findall(tpd(X, Y, P, D)-Body, clause(user:tpd(X, Y, P, D), Body), Clauses),
    detect_recursive_kernel(tpd, 4, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(transitive_parent_distance4, _, _)).

test(fsharp_native_tpd4_dispatch_emitted) :-
    assert_tpd_cycle_program,
    tmp_dir(fs_dispatch, Dir),
    write_wam_fsharp_project(
        [user:tpd/4, user:tpd_edge/2],
        [module_name('uw_tpd4_dispatch')],
        Dir),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    read_file_string(RT, RTS),
    assertion(sub_string(RTS, _, _, _, "nativeKernel_transitive_parent_distance")),
    assertion(sub_string(RTS, _, _, _, "| \"tpd/4\" ->")),
    assertion(sub_string(RTS, _, _, _, "FFIStreamRetry")),
    !.

test(rust_foreign_kind_registered_for_tpd4) :-
    assert_tpd_cycle_program,
    tmp_dir(rs_dispatch, Dir),
    write_wam_rust_project(
        [user:tpd/4, user:tpd_edge/2],
        [module_name('uw_tpd4_rs_dispatch'), foreign_lowering(true)],
        Dir),
    directory_file_path(Dir, 'src/lib.rs', Lib),
    read_file_string(Lib, S),
    assertion(sub_string(S, _, _, _, "transitive_parent_distance4")),
    assertion(sub_string(S, _, _, _, "register_foreign_native_kind(\"tpd/4\"")),
    !.

test(c_registers_tpd4_handler) :-
    assert_c_tpd_program,
    tmp_dir(c_dispatch, Dir),
    write_wam_c_project([user:pd/4], [], Dir),
    directory_file_path(Dir, 'lib.c', Lib),
    read_file_string(Lib, LibS),
    assertion(sub_string(LibS, _, _, _,
        "wam_register_transitive_parent_distance_kernel")),
    directory_file_path(Dir, 'wam_runtime.c', RT),
    read_file_string(RT, S),
    assertion(sub_string(S, _, _, _, "wam_collect_transitive_parent_distance")),
    !.

:- end_tests(tpd4_native_dispatch).

% ============================================================
% 4. Executable smokes
% ============================================================

:- begin_tests(tpd4_executable).

test(fsharp_cycle_and_diamond_e2e, [condition(dotnet_available)]) :-
    assert_tpd_cycle_program,
    tmp_dir(fs_e2e, Dir),
    once(write_wam_fsharp_project(
        [ user:tpd/4, user:tpd_edge/2,
          user:tpd_tail/3, user:tpd_after/3, user:tpd_cut/3,
          user:tpd_call_after/3
        ],
        [module_name('uw_tpd4_e2e')],
        Dir)),
    directory_file_path(Dir, 'Program.fs', Prog),
    tpd4_write_fsharp_driver(Prog),
    run_dotnet_build(Dir, BuildExit, BuildOut),
    ( BuildExit =:= 0 -> true
    ; format(user_error, 'fsharp tpd4 e2e build:~n~w~n', [BuildOut]),
      assertion(BuildExit =:= 0), fail
    ),
    run_dotnet_run(Dir, RunExit, RunOut),
    ( RunExit =:= 0 -> true
    ; format(user_error, 'fsharp tpd4 e2e run:~n~w~n', [RunOut]),
      assertion(RunExit =:= 0), fail
    ),
    assertion(sub_string(RunOut, _, _, _, "OK stream_from_a")),
    assertion(sub_string(RunOut, _, _, _, "OK equal_diamond_parents")),
    assertion(sub_string(RunOut, _, _, _, "OK pairing_retry")),
    assertion(sub_string(RunOut, _, _, _, "OK all_bound_modes")),
    assertion(sub_string(RunOut, _, _, _, "OK alias_later_match")),
    assertion(sub_string(RunOut, _, _, _, "OK invalid_inputs")),
    assertion(sub_string(RunOut, _, _, _, "OK aliased_outputs_fail")),
    assertion(sub_string(RunOut, _, _, _, "OK execute_foreign_retry")),
    assertion(sub_string(RunOut, _, _, _, "OK call_foreign_retry")),
    assertion(sub_string(RunOut, _, _, _, "OK cut_after_foreign")),
    !.

test(c_tpd4_stream_and_bound, [condition(gcc_available)]) :-
    assert_c_tpd_program,
    tmp_dir(c_e2e, Dir),
    write_wam_c_project([user:pd/4], [], Dir),
    directory_file_path(Dir, 'wam_runtime.c', RuntimePath),
    directory_file_path(Dir, 'lib.c', LibPath),
    directory_file_path(Dir, 'main.c', MainPath),
    directory_file_path(Dir, 'tpd4_c_smoke', ExePath),
    tpd4_c_cycle_main(MainCode),
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

test(rust_collect_tpd4_unit, [condition(cargo_available)]) :-
    assert_tpd_diamond_program,
    tmp_dir(rs_e2e, Dir),
    write_wam_rust_project(
        [user:tpd/4, user:tpd_edge/2],
        [module_name('uw_tpd4_rs'), foreign_lowering(true)],
        Dir),
    tpd4_append_rust_unit(Dir),
    format(atom(Cmd),
        'cd ~w && cargo test --quiet tpd4_contract -- --nocapture >~w/cargo.out 2>~w/cargo.err',
        [Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; directory_file_path(Dir, 'cargo.err', ErrPath),
      directory_file_path(Dir, 'cargo.out', OutPath),
      ( exists_file(ErrPath) -> read_file_string(ErrPath, Err) ; Err = "" ),
      ( exists_file(OutPath) -> read_file_string(OutPath, Out) ; Out = "" ),
      format(user_error, 'rust tpd4 unit failed:~n~w~n~w~n', [Out, Err]),
      fail
    ),
    !.

test(elixir_collect_triples_unit, [condition(elixir_available)]) :-
    tmp_dir(ex_e2e, Dir),
    compile_wam_runtime_snippet_for_elixir_tpd4(Dir),
    directory_file_path(Dir, 'tpd4_unit.exs', Script),
    tpd4_write_elixir_unit(Script),
    format(atom(Cmd),
        'cd ~w && elixir tpd4_unit.exs >~w/elixir.out 2>~w/elixir.err',
        [Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; directory_file_path(Dir, 'elixir.err', ErrPath),
      directory_file_path(Dir, 'elixir.out', OutPath),
      ( exists_file(ErrPath) -> read_file_string(ErrPath, Err) ; Err = "" ),
      ( exists_file(OutPath) -> read_file_string(OutPath, Out) ; Out = "" ),
      format(user_error, 'elixir tpd4 unit failed:~n~w~n~w~n', [Out, Err]),
      fail
    ),
    !.

:- end_tests(tpd4_executable).

% ============================================================
% 5. Acyclic no_kernels fallback
% ============================================================

:- begin_tests(tpd4_no_kernels_acyclic).

test(fsharp_no_kernels_fallback_builds, [condition(dotnet_available)]) :-
    assert_tpd_chain_program,
    tmp_dir(fs_nk, Dir),
    once(write_wam_fsharp_project(
        [user:tpd/4, user:tpd_edge/2],
        [no_kernels(true), module_name('uw_tpd4_nk'), conformance_main(true)],
        Dir)),
    directory_file_path(Dir, 'WamRuntime.fs', RT),
    read_file_string(RT, RTS),
    assertion(\+ sub_string(RTS, _, _, _, "nativeKernel_transitive_parent_distance")),
    run_dotnet_build(Dir, Exit, Out),
    ( Exit =:= 0 -> true
    ; format(user_error, 'fsharp no-kernels build:~n~w~n', [Out]),
      assertion(Exit =:= 0), fail
    ),
    !.

:- end_tests(tpd4_no_kernels_acyclic).

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

tpd4_write_fsharp_driver(ProgPath) :-
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
    Map.ofList [(\"tpd_edge\", Map.ofList grouped)]

let mkContext edges atomNames =
    let foreignPreds = [\"tpd/4\"]
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

let collectTriples (ctx: WamContext) (source: string)
                   (boundT: string option) (boundP: string option)
                   (boundD: int option) =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom source
    match boundT with
    | Some t -> regs.[2] <- Atom t
    | None -> regs.[2] <- Unbound 100
    match boundP with
    | Some p -> regs.[3] <- Atom p
    | None -> regs.[3] <- Unbound 101
    match boundD with
    | Some d -> regs.[4] <- Integer d
    | None -> regs.[4] <- Unbound 102
    match callForeign ctx \"tpd/4\" (mkState regs) with
    | None -> []
    | Some s1 ->
        let readTriple (s: WamState) =
            let t =
                match getReg 2 s with
                | Some (Atom a) -> a
                | _ -> \"?\"
            let p =
                match getReg 3 s with
                | Some (Atom a) -> a
                | _ -> \"?\"
            let d =
                match getReg 4 s with
                | Some (Integer i) -> i
                | _ -> -1
            t, p, d
        let rec gather (s: WamState) (acc: (string * string * int) list) =
            let tr = readTriple s
            match backtrack s with
            | Some s2 -> gather s2 (tr :: acc)
            | None -> List.rev (tr :: acc)
        gather s1 []

let runTripleWrapper (ctx: WamContext) (pred: string) : WamState option =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Unbound 200
    regs.[2] <- Unbound 201
    regs.[3] <- Unbound 202
    dispatchCall ctx pred (mkState regs)

let readWrapperTriple (s: WamState) =
    match derefVar s.WsBindings (Unbound 200),
          derefVar s.WsBindings (Unbound 201),
          derefVar s.WsBindings (Unbound 202) with
    | Atom t, Atom p, Integer d -> Some (t, p, d)
    | _ -> None

let aliasedOutputsFail (ctx: WamContext) =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom \"a\"
    regs.[2] <- Unbound 150
    regs.[3] <- Unbound 151
    regs.[4] <- Unbound 150
    callForeign ctx \"tpd/4\" (mkState regs) |> Option.isNone

let aliasedTargetParent (ctx: WamContext) =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom \"a\"
    regs.[2] <- Unbound 160
    regs.[3] <- Unbound 160
    regs.[4] <- Integer 1
    match callForeign ctx \"tpd/4\" (mkState regs) with
    | Some s ->
        derefVar s.WsBindings (Unbound 160) = Atom \"a\" &&
        getReg 2 s = Some (Atom \"a\") &&
        getReg 3 s = Some (Atom \"a\") &&
        Option.isNone (backtrack s)
    | None -> false

let invalidSourceFails (ctx: WamContext) (source: Value) =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- source
    regs.[2] <- Unbound 170
    regs.[3] <- Unbound 171
    regs.[4] <- Unbound 172
    callForeign ctx \"tpd/4\" (mkState regs) |> Option.isNone

let invalidDistanceTypeFails (ctx: WamContext) =
    let regs = Array.create MaxRegs (Unbound -1)
    regs.[1] <- Atom \"a\"
    regs.[2] <- Unbound 180
    regs.[3] <- Unbound 181
    regs.[4] <- Atom \"not_a_distance\"
    callForeign ctx \"tpd/4\" (mkState regs) |> Option.isNone

[<EntryPoint>]
let main _argv =
    let cycleEdges = [(\"a\", \"b\"); (\"b\", \"c\"); (\"c\", \"d\"); (\"c\", \"a\")]
    let ctx = mkContext cycleEdges [\"a\"; \"b\"; \"c\"; \"d\"]
    let fromA = collectTriples ctx \"a\" None None None |> List.sort
    assertTrue \"stream_from_a\"
        (fromA = [(\"a\", \"c\", 3); (\"b\", \"a\", 1); (\"c\", \"b\", 2); (\"d\", \"c\", 3)])

    let boundSrc = collectTriples ctx \"a\" (Some \"a\") None None
    assertTrue \"bound_source_cycle\" (boundSrc = [(\"a\", \"c\", 3)])

    let diamondEdges = [(\"a\", \"b\"); (\"a\", \"c\"); (\"b\", \"d\"); (\"c\", \"d\")]
    let dctx = mkContext diamondEdges [\"a\"; \"b\"; \"c\"; \"d\"]
    let diamond = collectTriples dctx \"a\" (Some \"d\") None None |> List.sort
    assertTrue \"equal_diamond_parents\"
        (diamond = [(\"d\", \"b\", 2); (\"d\", \"c\", 2)])

    let pairingOk =
        fromA |> List.forall (fun (t, p, d) ->
            match t, p, d with
            | \"b\", \"a\", 1 | \"c\", \"b\", 2 | \"a\", \"c\", 3 | \"d\", \"c\", 3 -> true
            | _ -> false)
    assertTrue \"pairing_retry\" (pairingOk && List.length fromA = 4)

    let allBoundModes =
        collectTriples ctx \"a\" (Some \"a\") None None = [(\"a\", \"c\", 3)] &&
        (collectTriples ctx \"a\" None (Some \"c\") None |> List.sort) =
            [(\"a\", \"c\", 3); (\"d\", \"c\", 3)] &&
        (collectTriples ctx \"a\" None None (Some 3) |> List.sort) =
            [(\"a\", \"c\", 3); (\"d\", \"c\", 3)] &&
        collectTriples ctx \"a\" (Some \"a\") (Some \"c\") None =
            [(\"a\", \"c\", 3)] &&
        collectTriples ctx \"a\" (Some \"a\") None (Some 3) =
            [(\"a\", \"c\", 3)] &&
        (collectTriples ctx \"a\" None (Some \"c\") (Some 3) |> List.sort) =
            [(\"a\", \"c\", 3); (\"d\", \"c\", 3)] &&
        collectTriples ctx \"a\" (Some \"a\") (Some \"c\") (Some 3) =
            [(\"a\", \"c\", 3)]
    assertTrue \"all_bound_modes\" allBoundModes

    let aliasCtx = mkContext [(\"a\", \"b\"); (\"a\", \"a\")] [\"a\"; \"b\"]
    assertTrue \"alias_later_match\" (aliasedTargetParent aliasCtx)
    assertTrue \"invalid_inputs\"
        (invalidSourceFails ctx (Unbound 190) &&
         invalidSourceFails ctx (Integer 7) &&
         invalidDistanceTypeFails ctx &&
         collectTriples ctx \"a\" None None (Some 0) = [] &&
         collectTriples ctx \"a\" None None (Some -1) = [] &&
         collectTriples ctx \"missing\" None None None = [])
    assertTrue \"aliased_outputs_fail\" (aliasedOutputsFail ctx)

    let hasExecuteForeign =
        ctx.WcCode
        |> Array.exists (function ExecuteForeign \"tpd/4\" -> true | _ -> false)
    let executeRetry = runTripleWrapper ctx \"tpd_after/3\"
    assertTrue \"execute_foreign_retry\"
        (hasExecuteForeign &&
         (executeRetry |> Option.exists (fun s ->
             readWrapperTriple s = Some (\"c\", \"b\", 2) &&
             s.WsCP = 0 && s.WsCutBar = 0 && List.isEmpty s.WsB0Stack &&
             s.WsTrailLen = List.length s.WsTrail)))

    let hasCallForeign =
        ctx.WcCode
        |> Array.exists (function CallForeign (\"tpd/4\", 4) -> true | _ -> false)
    let callRetry = runTripleWrapper ctx \"tpd_call_after/3\"
    assertTrue \"call_foreign_retry\"
        (hasCallForeign &&
         (callRetry |> Option.exists (fun s ->
             readWrapperTriple s = Some (\"c\", \"b\", 2) &&
             s.WsCP = 0 && s.WsCutBar = 0 && List.isEmpty s.WsB0Stack &&
             s.WsTrailLen = List.length s.WsTrail)))

    let cutResult = runTripleWrapper ctx \"tpd_cut/3\"
    assertTrue \"cut_after_foreign\"
        (cutResult |> Option.exists (fun s ->
            readWrapperTriple s = Some (\"b\", \"a\", 1) &&
            s.WsCP = 0 && s.WsCutBar = 0 && List.isEmpty s.WsB0Stack &&
            s.WsCPsLen = 0 && List.isEmpty s.WsCPs &&
            s.WsTrailLen = List.length s.WsTrail &&
            Option.isNone (backtrack s)))

    let fromD = collectTriples ctx \"d\" None None None
    assertTrue \"sink_d\" (fromD = [])

    printfn \"RESULT %d/%d\" passes (passes + fails)
    if fails > 0 then 1 else 0
",
    setup_call_cleanup(
        open(ProgPath, write, Out, [encoding(utf8)]),
        write(Out, Driver),
        close(Out)).

tpd4_c_cycle_main(Code) :-
    Code =
'#include "wam_runtime.h"
#include <string.h>

void setup_pd_4(WamState* state);
void setup_detected_wam_c_kernels(WamState* state);

typedef struct {
    const char *target;
    const char *parent;
    int distance;
} ExpectedTriple;

static const ExpectedTriple contract_triples[] = {
    { "b", "a", 1 },
    { "c", "a", 1 },
    { "d", "b", 2 },
    { "d", "c", 2 },
    { "a", "d", 3 }
};

static void load_contract_graph(WamState *state) {
    wam_register_transitive_edge(state, "a", "b");
    wam_register_transitive_edge(state, "a", "b");
    wam_register_transitive_edge(state, "a", "c");
    wam_register_transitive_edge(state, "b", "d");
    wam_register_transitive_edge(state, "c", "d");
    wam_register_transitive_edge(state, "d", "a");
}

static void load_alias_graph(WamState *state, bool include_self_loop) {
    wam_register_transitive_edge(state, "a", "b");
    if (include_self_loop) {
        wam_register_transitive_edge(state, "a", "a");
    }
}

static int install_retry_program(WamState *state) {
    int fail_pc = state->code_size;
    int next_size = fail_pc + 4;
    Instruction *next = realloc(
        state->code, sizeof(Instruction) * (size_t)next_size);
    if (!next) return -1;
    state->code = next;
    state->code_size = next_size;
    memset(&state->code[fail_pc], 0, sizeof(Instruction) * 4u);

    /* Every requested retry first runs a guaranteed failing instruction.
     * wam_run then restores the native stream CP and resumes it normally. */
    state->code[fail_pc].tag = INSTR_BUILTIN_CALL;
    state->code[fail_pc].as.pred.pred = "__tpd4_retry_fail/0";
    state->code[fail_pc].as.pred.arity = 0;
    state->code[fail_pc + 1].tag = INSTR_PROCEED;

    /* An outer sentinel restores the pre-call bindings after the foreign
     * stream is exhausted, pops itself, and returns cleanly. */
    state->code[fail_pc + 2].tag = INSTR_TRUST_ME;
    state->code[fail_pc + 3].tag = INSTR_PROCEED;
    return fail_pc;
}

static int triple_index(const char *target, const char *parent, int distance) {
    for (int i = 0; i < 5; i++) {
        const ExpectedTriple *e = &contract_triples[i];
        if (strcmp(target, e->target) == 0 &&
            strcmp(parent, e->parent) == 0 &&
            distance == e->distance) {
            return i;
        }
    }
    return -1;
}

static bool selected_by_mode(int mask, const ExpectedTriple *e) {
    if ((mask & 1) && strcmp(e->target, "d") != 0) return false;
    if ((mask & 2) && strcmp(e->parent, "b") != 0) return false;
    if ((mask & 4) && e->distance != 2) return false;
    return true;
}

static bool base_mode_regs_restored(WamState *state, int mask) {
    WamValue *target = wam_deref_ptr(state, &state->A[1]);
    WamValue *parent = wam_deref_ptr(state, &state->A[2]);
    WamValue *dist = wam_deref_ptr(state, &state->A[3]);
    bool target_ok = (mask & 1)
        ? target->tag == VAL_ATOM && strcmp(target->data.atom, "d") == 0
        : val_is_unbound(*target);
    bool parent_ok = (mask & 2)
        ? parent->tag == VAL_ATOM && strcmp(parent->data.atom, "b") == 0
        : val_is_unbound(*parent);
    bool dist_ok = (mask & 4)
        ? dist->tag == VAL_INT && dist->data.integer == 2
        : val_is_unbound(*dist);
    return target_ok && parent_ok && dist_ok;
}

static int run_bound_mode(int mask) {
    WamState state;
    int rc = 0;
    int seen[5] = { 0, 0, 0, 0, 0 };
    int expected_count = 0;
    wam_state_init(&state);
    setup_pd_4(&state);
    setup_detected_wam_c_kernels(&state);
    load_contract_graph(&state);

    int fail_pc = install_retry_program(&state);
    if (fail_pc < 0) { rc = 1; goto done; }
    state.P = fail_pc;
    state.CP = WAM_HALT;
    state.A[0] = val_atom("a");
    state.A[1] = (mask & 1) ? val_atom("d") : val_unbound("T");
    state.A[2] = (mask & 2) ? val_atom("b") : val_unbound("P");
    state.A[3] = (mask & 4) ? val_int(2) : val_unbound("D");

    for (int i = 0; i < 5; i++) {
        if (selected_by_mode(mask, &contract_triples[i])) expected_count++;
    }

    /* This outer ordinary CP is the query-level retry boundary. */
    push_choice_point(&state, fail_pc + 2, 4);
    if (!wam_execute_foreign_predicate(&state, "pd/4", 4)) {
        rc = 2; goto done;
    }

    for (int n = 0; n < expected_count; n++) {
        if (n > 0) {
            state.P = fail_pc;
            if (wam_run(&state) != 0) { rc = 3; goto done; }
        }
        WamValue *target = wam_deref_ptr(&state, &state.A[1]);
        WamValue *parent = wam_deref_ptr(&state, &state.A[2]);
        WamValue *dist = wam_deref_ptr(&state, &state.A[3]);
        if (target->tag != VAL_ATOM || parent->tag != VAL_ATOM ||
            dist->tag != VAL_INT) {
            rc = 4; goto done;
        }
        int index = triple_index(target->data.atom, parent->data.atom,
                                 dist->data.integer);
        if (index < 0 || seen[index] ||
            !selected_by_mode(mask, &contract_triples[index])) {
            rc = 5; goto done;
        }
        seen[index] = 1;
    }

    for (int i = 0; i < 5; i++) {
        if (seen[i] != (selected_by_mode(mask, &contract_triples[i]) ? 1 : 0)) {
            rc = 6; goto done;
        }
    }

    /* One more failure exhausts the foreign CP.  The outer sentinel then
     * restores the pre-call registers/trail and removes itself. */
    state.P = fail_pc;
    if (wam_run(&state) != 0 || state.P != WAM_HALT ||
        state.B != 0 || state.TR != 0 ||
        !base_mode_regs_restored(&state, mask)) {
        rc = 7; goto done;
    }

done:
    wam_free_state(&state);
    return rc;
}

static int run_alias_later_match(void) {
    WamState state;
    int rc = 0;
    wam_state_init(&state);
    setup_pd_4(&state);
    setup_detected_wam_c_kernels(&state);
    load_alias_graph(&state, true);

    int fail_pc = install_retry_program(&state);
    if (fail_pc < 0) { rc = 1; goto done; }
    state.P = fail_pc;
    state.CP = WAM_HALT;
    state.A[0] = val_atom("a");
    WamValue shared = wam_make_ref(&state);
    state.A[1] = shared;
    state.A[2] = shared;
    state.A[3] = val_int(1);
    push_choice_point(&state, fail_pc + 2, 4);

    /* (b,a,1) is incompatible with T=P, but the later self-loop
     * (a,a,1) must still be selected. */
    if (!wam_execute_foreign_predicate(&state, "pd/4", 4)) {
        rc = 2; goto done;
    }
    WamValue *target = wam_deref_ptr(&state, &state.A[1]);
    WamValue *parent = wam_deref_ptr(&state, &state.A[2]);
    if (target->tag != VAL_ATOM || parent->tag != VAL_ATOM ||
        strcmp(target->data.atom, "a") != 0 ||
        strcmp(parent->data.atom, "a") != 0 || state.B != 1) {
        rc = 3; goto done;
    }

    state.P = fail_pc;
    if (wam_run(&state) != 0 || state.B != 0 || state.TR != 0 ||
        state.A[1].tag != VAL_REF || state.A[2].tag != VAL_REF ||
        state.A[1].data.ref_addr != state.A[2].data.ref_addr ||
        !val_is_unbound(*wam_deref_ptr(&state, &state.A[1]))) {
        rc = 4; goto done;
    }

done:
    wam_free_state(&state);
    return rc;
}

static int run_alias_no_match_clean(void) {
    WamState state;
    int rc = 0;
    wam_state_init(&state);
    setup_pd_4(&state);
    setup_detected_wam_c_kernels(&state);
    load_alias_graph(&state, false);
    state.P = 0;
    state.A[0] = val_atom("a");
    WamValue shared = wam_make_ref(&state);
    state.A[1] = shared;
    state.A[2] = shared;
    state.A[3] = val_int(1);

    if (wam_execute_foreign_predicate(&state, "pd/4", 4) ||
        state.B != 0 || state.TR != 0 ||
        !val_is_unbound(*wam_deref_ptr(&state, &state.A[1]))) {
        rc = 1;
    }
    wam_free_state(&state);
    return rc;
}

int main(void) {
    /* The mode loop includes the complete free stream: duplicate edge
     * suppression, both equal-shortest parents, cycle-to-Source, exact
     * tuple pairing, ordinary retries, and clean exhaustion. */
    for (int mask = 0; mask < 8; mask++) {
        int rc = run_bound_mode(mask);
        if (rc != 0) return 20 + mask * 8 + rc;
    }
    if (run_alias_later_match() != 0) return 100;
    if (run_alias_no_match_clean() != 0) return 101;
    return 0;
}
'.

tpd4_append_rust_unit(Dir) :-
    directory_file_path(Dir, 'src/lib.rs', Path),
    read_file_string(Path, Existing),
    Unit =
"

#[cfg(test)]
mod tpd4_contract {
    use super::*;
    use std::collections::HashMap;
    use state::WamState;

    #[test]
    fn equal_diamond_emits_both_parents() {
        let mut vm = WamState::new(Vec::new(), HashMap::new());
        vm.register_indexed_atom_fact2_pairs(
            \"tpd_edge\",
            &[(\"a\", \"b\"), (\"a\", \"c\"), (\"b\", \"d\"), (\"c\", \"d\"), (\"a\", \"b\")],
        );
        let mut nodes = Vec::new();
        vm.collect_native_transitive_parent_distance_results(\"a\", \"tpd_edge\", &mut nodes);
        nodes.sort();
        assert_eq!(
            nodes,
            vec![
                (\"b\".to_string(), \"a\".to_string(), 1),
                (\"c\".to_string(), \"a\".to_string(), 1),
                (\"d\".to_string(), \"b\".to_string(), 2),
                (\"d\".to_string(), \"c\".to_string(), 2),
            ]
        );
    }

    #[test]
    fn cycle_emits_source_with_parent() {
        let mut vm = WamState::new(Vec::new(), HashMap::new());
        vm.register_indexed_atom_fact2_pairs(
            \"tpd_edge\",
            &[(\"a\", \"b\"), (\"b\", \"a\")],
        );
        let mut nodes = Vec::new();
        vm.collect_native_transitive_parent_distance_results(\"a\", \"tpd_edge\", &mut nodes);
        nodes.sort();
        assert_eq!(
            nodes,
            vec![
                (\"a\".to_string(), \"b\".to_string(), 2),
                (\"b\".to_string(), \"a\".to_string(), 1),
            ]
        );
    }
}
",
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        ( write(Out, Existing), write(Out, Unit) ),
        close(Out)).

:- use_module('../src/unifyweaver/targets/wam_elixir_target',
              [compile_wam_runtime_to_elixir/2]).

compile_wam_runtime_snippet_for_elixir_tpd4(Dir) :-
    compile_wam_runtime_to_elixir([], RuntimeCode),
    Pattern = "defmodule WamRuntime.GraphKernel.TransitiveParentDistance do",
    EndPattern = "defmodule WamRuntime.GraphKernel.TransitiveStepParentDistance do",
    sub_string(RuntimeCode, Start, _, _, Pattern),
    sub_string(RuntimeCode, End, _, _, EndPattern),
    End > Start,
    BodyLen is End - Start,
    sub_string(RuntimeCode, Start, BodyLen, _, Body),
    directory_file_path(Dir, 'tpd4_kernel.ex', Path),
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Body),
        close(Out)).

tpd4_write_elixir_unit(Script) :-
    Code =
'Code.require_file("tpd4_kernel.ex")

neighbors = fn
  "a" -> [{"a", "b"}, {"a", "c"}]
  "b" -> [{"b", "d"}]
  "c" -> [{"c", "d"}]
  _ -> []
end

triples = WamRuntime.GraphKernel.TransitiveParentDistance.collect_triples(neighbors, "a")
expected = [{"b", "a", 1}, {"c", "a", 1}, {"d", "b", 2}, {"d", "c", 2}]
unless triples == expected do
  IO.puts("FAIL diamond #{inspect(triples)}")
  System.halt(1)
end

cycle_n = fn
  "a" -> [{"a", "b"}]
  "b" -> [{"b", "a"}]
  _ -> []
end
cycle = WamRuntime.GraphKernel.TransitiveParentDistance.collect_triples(cycle_n, "a")
unless cycle == [{"a", "b", 2}, {"b", "a", 1}] do
  IO.puts("FAIL cycle #{inspect(cycle)}")
  System.halt(1)
end

IO.puts("OK elixir_tpd4")
',
    setup_call_cleanup(
        open(Script, write, Out, [encoding(utf8)]),
        write(Out, Code),
        close(Out)).
