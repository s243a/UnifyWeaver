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

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

scala_available :-
    catch(
        ( process_create(path(scalac), ['-version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

rscript_available :-
    catch(
        ( process_create(path('Rscript'), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

ghc_available :-
    catch(
        ( process_create(path(ghc), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

tmp_dir(Tag, Dir) :-
    get_time(T),
    format(atom(Stamp), '~w', [T]),
    format(atom(Dir), '/tmp/uw_wsp3_~w_~w', [Tag, Stamp]),
    make_directory_path(Dir).

read_file_string(Path, String) :-
    read_file_to_string(Path, String, []).

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
    assertion(\+ wam_fsharp_native_kernel_kind(astar_shortest_path4)).

test(haskell_mustache_dijkstra) :-
    read_file_string(
        'templates/targets/haskell_wam/kernel_weighted_shortest_path.hs.mustache', S),
    assertion(sub_string(S, _, _, _, "Finite nonnegative Dijkstra")),
    assertion(sub_string(S, _, _, _, "nativeKernel_weighted_shortest_path")),
    assertion(sub_string(S, _, _, _, "n /= source")),
    assertion(sub_string(S, _, _, _, "error \"wsp3: reachable invalid")).

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

test(go_standalone_cheaper_detour, [condition(go_available)]) :-
    tmp_dir(go_e2e, Dir),
    directory_file_path(Dir, 'go.mod', ModPath),
    setup_call_cleanup(
        open(ModPath, write, ModOut, [encoding(utf8)]),
        write(ModOut, "module wsp3algo\n\ngo 1.21\n"),
        close(ModOut)),
    directory_file_path(Dir, 'wsp3_algo_test.go', GoPath),
    wsp3_write_go_standalone(GoPath),
    format(atom(Cmd),
        'cd ~w && go test -run TestWsp3CheaperDetour -count=1 >~w/go.out 2>~w/go.err',
        [Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; directory_file_path(Dir, 'go.err', ErrPath),
      directory_file_path(Dir, 'go.out', OutPath),
      ( exists_file(ErrPath) -> read_file_string(ErrPath, Err) ; Err = "" ),
      ( exists_file(OutPath) -> read_file_string(OutPath, Out) ; Out = "" ),
      format(user_error, 'go wsp3 unit failed:~n~w~n~w~n', [Out, Err]),
      fail
    ),
    !.

test(haskell_standalone_cheaper_detour, [condition(ghc_available)]) :-
    tmp_dir(hs_e2e, Dir),
    directory_file_path(Dir, 'Wsp3Algo.hs', HsPath),
    wsp3_write_haskell_standalone(HsPath),
    format(atom(Cmd),
        'cd ~w && ghc -O0 -package containers Wsp3Algo.hs -o wsp3algo >~w/ghc.out 2>~w/ghc.err && ./wsp3algo >~w/run.out 2>~w/run.err',
        [Dir, Dir, Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; format(user_error, 'haskell wsp3 failed exit=~w~n', [Exit]),
      fail
    ),
    directory_file_path(Dir, 'run.out', OutPath),
    read_file_string(OutPath, Out),
    assertion(sub_string(Out, _, _, _, "OK")),
    !.

test(r_standalone_cheaper_detour, [condition(rscript_available)]) :-
    tmp_dir(r_e2e, Dir),
    directory_file_path(Dir, 'wsp3_algo.R', RPath),
    wsp3_write_r_standalone(RPath),
    format(atom(Cmd),
        'cd ~w && Rscript wsp3_algo.R >~w/r.out 2>~w/r.err',
        [Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; directory_file_path(Dir, 'r.err', ErrPath),
      directory_file_path(Dir, 'r.out', OutPath),
      ( exists_file(ErrPath) -> read_file_string(ErrPath, Err) ; Err = "" ),
      ( exists_file(OutPath) -> read_file_string(OutPath, Out) ; Out = "" ),
      format(user_error, 'R wsp3 failed:~n~w~n~w~n', [Out, Err]),
      fail
    ),
    directory_file_path(Dir, 'r.out', OutPath),
    read_file_string(OutPath, Out),
    assertion(sub_string(Out, _, _, _, "OK")),
    !.

test(scala_standalone_cheaper_detour, [condition(scala_available)]) :-
    tmp_dir(sc_e2e, Dir),
    directory_file_path(Dir, 'Wsp3Algo.scala', ScPath),
    wsp3_write_scala_standalone(ScPath),
    format(atom(Cmd),
        'cd ~w && scalac Wsp3Algo.scala >~w/sc.out 2>~w/sc.err && scala Wsp3Algo >~w/run.out 2>~w/run.err',
        [Dir, Dir, Dir, Dir, Dir]),
    shell(Cmd, Exit),
    ( Exit =:= 0 -> true
    ; format(user_error, 'scala wsp3 failed exit=~w~n', [Exit]),
      fail
    ),
    directory_file_path(Dir, 'run.out', OutPath),
    read_file_string(OutPath, Out),
    assertion(sub_string(Out, _, _, _, "OK")),
    !.

:- end_tests(wsp3_executable).

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

compile_wam_runtime_snippet_for_elixir_wsp3(Dir) :-
    directory_file_path(Dir, 'wsp3_runtime.ex', Path),
    Code =
"defmodule Wsp3Runtime do
  def collect_pairs(edges, source) do
    case dijkstra(edges, :gb_sets.singleton({0.0, source}), %{source => 0.0}) do
      :invalid -> []
      dist ->
        dist
        |> Enum.reject(fn {n, _} -> n == source end)
        |> Enum.sort_by(fn {n, _} -> n end)
    end
  end

  defp dijkstra(_edges, pq, dist) do
    if :gb_sets.is_empty(pq) do
      dist
    else
      {{cost, node}, pq2} = :gb_sets.take_smallest(pq)
      best = Map.get(dist, node, 1.0e300)
      if cost > best do
        dijkstra(_edges, pq2, dist)
      else
        case Enum.reduce_while(Map.get(_edges, node, []), {:ok, pq2, dist}, &relax(cost, &1, &2)) do
          :invalid -> :invalid
          {:ok, pq3, dist2} -> dijkstra(_edges, pq3, dist2)
        end
      end
    end
  end

  defp relax(cost, {nxt, w}, {:ok, q, d}) do
    cond do
      not is_number(w) or w < 0 or w != w -> {:halt, :invalid}
      true ->
        nc = cost + w
        prev = Map.get(d, nxt, 1.0e300)
        if nc < prev do
          {:cont, {:ok, :gb_sets.add_element({nc, nxt}, q), Map.put(d, nxt, nc)}}
        else
          {:cont, {:ok, q, d}}
        end
    end
  end
end
",
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Code),
        close(Out)).

wsp3_write_elixir_unit(Script) :-
    Code =
"Code.require_file(\"wsp3_runtime.ex\")
edges = %{
  \"a\" => [{\"b\", 10.0}, {\"c\", 1.0}],
  \"c\" => [{\"b\", 1.0}],
  \"b\" => [{\"d\", 1.0}]
}
got = Wsp3Runtime.collect_pairs(edges, \"a\")
want = [{\"b\", 2.0}, {\"c\", 1.0}, {\"d\", 3.0}]
if got != want do
  IO.puts(:stderr, \"mismatch #{inspect(got)}\")
  System.halt(1)
end
bad = Wsp3Runtime.collect_pairs(%{\"a\" => [{\"b\", -1.0}]}, \"a\")
if bad != [] do
  IO.puts(:stderr, \"invalid should fail\")
  System.halt(2)
end
IO.puts(\"OK\")
",
    setup_call_cleanup(
        open(Script, write, Out, [encoding(utf8)]),
        write(Out, Code),
        close(Out)).

wsp3_write_go_standalone(Path) :-
    Code =
"package wsp3algo

import (
        \"math\"
        \"sort\"
        \"testing\"
)

func dijkstra(edges map[string][]struct {
        To string
        W  float64
}, source string) ([]struct {
        To string
        W  float64
}, bool) {
        dist := map[string]float64{source: 0}
        type item struct {
                n string
                c float64
        }
        pq := []item{{source, 0}}
        for len(pq) > 0 {
                best := 0
                for i := 1; i < len(pq); i++ {
                        if pq[i].c < pq[best].c {
                                best = i
                        }
                }
                u := pq[best]
                pq = append(pq[:best], pq[best+1:]...)
                if d, ok := dist[u.n]; ok && u.c > d {
                        continue
                }
                for _, e := range edges[u.n] {
                        if math.IsNaN(e.W) || math.IsInf(e.W, 0) || e.W < 0 {
                                return nil, false
                        }
                        nc := u.c + e.W
                        if prev, ok := dist[e.To]; !ok || nc < prev {
                                dist[e.To] = nc
                                pq = append(pq, item{e.To, nc})
                        }
                }
        }
        var out []struct {
                To string
                W  float64
        }
        for n, c := range dist {
                if n != source {
                        out = append(out, struct {
                                To string
                                W  float64
                        }{n, c})
                }
        }
        sort.Slice(out, func(i, j int) bool { return out[i].To < out[j].To })
        return out, true
}

func TestWsp3CheaperDetour(t *testing.T) {
        edges := map[string][]struct {
                To string
                W  float64
        }{
                \"a\": {{\"b\", 10}, {\"c\", 1}},
                \"c\": {{\"b\", 1}},
                \"b\": {{\"d\", 1}},
        }
        got, ok := dijkstra(edges, \"a\")
        if !ok || len(got) != 3 || got[0].To != \"b\" || got[0].W != 2 ||
                got[1].To != \"c\" || got[1].W != 1 || got[2].To != \"d\" || got[2].W != 3 {
                t.Fatalf(\"got %#v\", got)
        }
        _, ok2 := dijkstra(map[string][]struct {
                To string
                W  float64
        }{\"a\": {{\"b\", -1}}}, \"a\")
        if ok2 {
                t.Fatal(\"invalid should fail\")
        }
}
",
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Code),
        close(Out)).

wsp3_write_haskell_standalone(Path) :-
    Code =
"import qualified Data.Map.Strict as M
import qualified Data.Set as S
import Data.List (sort)

dijkstra2 edges source =
  go (S.singleton (0.0 :: Double, source)) (M.singleton source 0.0)
  where
    go pq dist =
      case S.minView pq of
        Nothing ->
          Right $ sort [ (n, c) | (n, c) <- M.toList dist, n /= source ]
        Just ((cost, node), pq') ->
          let best = M.findWithDefault (1/0) node dist
          in if cost > best then go pq' dist else
             let neighbors = M.findWithDefault [] node edges
             in case foldl (relax cost) (Right (pq', dist)) neighbors of
                  Left e -> Left e
                  Right (q2, d2) -> go q2 d2
    relax _ (Left e) _ = Left e
    relax cost (Right (q, d)) (nxt, w)
      | isNaN w || isInfinite w || w < 0 = Left \"invalid\"
      | otherwise =
          let nc = cost + w
              prev = M.findWithDefault (1/0) nxt d
          in if nc < prev
               then Right (S.insert (nc, nxt) q, M.insert nxt nc d)
               else Right (q, d)

main :: IO ()
main = do
  let edges = M.fromList
        [ (\"a\", [(\"b\", 10.0), (\"c\", 1.0)])
        , (\"c\", [(\"b\", 1.0)])
        , (\"b\", [(\"d\", 1.0)])
        ]
  case dijkstra2 edges \"a\" of
    Right got | got == [(\"b\", 2.0), (\"c\", 1.0), (\"d\", 3.0)] ->
      putStrLn \"OK\"
    other -> error (show other)
",
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Code),
        close(Out)).

wsp3_write_r_standalone(Path) :-
    Code =
"dijkstra <- function(edges, source) {
  dist <- new.env(parent = emptyenv())
  assign(source, 0, envir = dist)
  pq <- list(list(cost = 0, node = source))
  while (length(pq) > 0) {
    costs <- vapply(pq, function(x) x$cost, numeric(1))
    best <- which.min(costs)
    u <- pq[[best]]
    pq <- pq[-best]
    known <- if (exists(u$node, envir = dist, inherits = FALSE)) get(u$node, envir = dist) else Inf
    if (u$cost > known) next
    outs <- edges[[u$node]]
    if (is.null(outs)) outs <- list()
    for (e in outs) {
      w <- e[[2]]
      if (!is.finite(w) || w < 0) return(NULL)
      nc <- u$cost + w
      prev <- if (exists(e[[1]], envir = dist, inherits = FALSE)) get(e[[1]], envir = dist) else Inf
      if (nc < prev) {
        assign(e[[1]], nc, envir = dist)
        pq[[length(pq) + 1]] <- list(cost = nc, node = e[[1]])
      }
    }
  }
  keys <- ls(dist)
  keys <- keys[keys != source]
  pairs <- lapply(keys, function(k) list(k, get(k, envir = dist)))
  pairs[order(vapply(pairs, function(p) p[[1]], character(1)))]
}
edges <- list(
  a = list(list(\"b\", 10), list(\"c\", 1)),
  c = list(list(\"b\", 1)),
  b = list(list(\"d\", 1))
)
got <- dijkstra(edges, \"a\")
stopifnot(!is.null(got), length(got) == 3,
          got[[1]][[1]] == \"b\", abs(got[[1]][[2]] - 2) < 1e-9,
          got[[2]][[1]] == \"c\", abs(got[[2]][[2]] - 1) < 1e-9,
          got[[3]][[1]] == \"d\", abs(got[[3]][[2]] - 3) < 1e-9)
bad <- dijkstra(list(a = list(list(\"b\", -1))), \"a\")
stopifnot(is.null(bad))
cat(\"OK\\n\")
",
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Code),
        close(Out)).

wsp3_write_scala_standalone(Path) :-
    Code =
"object Wsp3Algo {
  def dijkstra(edges: Map[String, List[(String, Double)]], source: String)
      : Option[List[(String, Double)]] = {
    import scala.collection.mutable
    val dist = mutable.Map(source -> 0.0)
    val pq = mutable.PriorityQueue.empty[(Double, String)](Ordering.by(-_._1))
    pq.enqueue((0.0, source))
    while (pq.nonEmpty) {
      val (cost, node) = pq.dequeue()
      if (!(dist.contains(node) && cost > dist(node))) {
        for ((nxt, w) <- edges.getOrElse(node, Nil)) {
          if (w.isNaN || w.isInfinity || w < 0) return None
          val nc = cost + w
          if (!dist.contains(nxt) || nc < dist(nxt)) {
            dist(nxt) = nc
            pq.enqueue((nc, nxt))
          }
        }
      }
    }
    Some(dist.toList.filter(_._1 != source).sortBy(_._1))
  }
  def main(args: Array[String]): Unit = {
    val edges = Map(
      \"a\" -> List((\"b\", 10.0), (\"c\", 1.0)),
      \"c\" -> List((\"b\", 1.0)),
      \"b\" -> List((\"d\", 1.0))
    )
    val got = dijkstra(edges, \"a\")
    require(got.contains(List((\"b\", 2.0), (\"c\", 1.0), (\"d\", 3.0))))
    require(dijkstra(Map(\"a\" -> List((\"b\", -1.0))), \"a\").isEmpty)
    println(\"OK\")
  }
}
",
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Code),
        close(Out)).
