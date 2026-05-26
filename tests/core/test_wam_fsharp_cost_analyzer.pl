:- encoding(utf8).
% Tests for F# WAM cost analyzer resolver chain.
% Run: swipl -q -g run_tests -t halt tests/core/test_wam_fsharp_cost_analyzer.pl

:- prolog_load_context(directory, Dir),
   atom_concat(Dir, '/../../src/unifyweaver/targets/wam_fsharp_target', Mod),
   use_module(Mod, [
       resolve_auto_edge_store_fs/2,
       resolve_auto_lmdb_materialisation_fs/2,
       resolve_auto_lmdb_cache_tier_fs/2,
       resolve_auto_csr_index_backend_fs/2,
       resolve_fsharp_cost_options/2
   ]).

:- dynamic test_count/2.

run_tests :-
    retractall(test_count(_, _)),
    assertz(test_count(pass, 0)),
    assertz(test_count(fail, 0)),
    writeln('=== F# WAM Cost Analyzer Tests ==='),

    % -- Edge store resolver tests --

    run_test("edge_store(auto): dynamic graph -> lmdb_cached",
        ( resolve_auto_edge_store_fs(
            [edge_store(auto), edge_count(1000), graph_mutability('dynamic')], R),
          member(lmdb_materialisation(cached), R) )),

    run_test("edge_store(auto): small static, few queries -> lmdb_cached",
        ( resolve_auto_edge_store_fs(
            [edge_store(auto), edge_count(1000),
             expected_query_count(10), expected_lookups_per_query(50)], R),
          member(lmdb_materialisation(cached), R) )),

    run_test("edge_store(auto): large static, many queries -> lmdb_eager",
        ( resolve_auto_edge_store_fs(
            [edge_store(auto), edge_count(100000),
             expected_query_count(10000), expected_lookups_per_query(50)], R),
          member(lmdb_materialisation(eager), R) )),

    run_test("edge_store(auto): cost model considers preprocessing",
        ( resolve_auto_edge_store_fs(
            [edge_store(auto), edge_count(6000),
             expected_query_count(300), expected_lookups_per_query(50),
             edge_store_verbose(true)], R),
          member(lmdb_materialisation(M), R),
          format(user_error, '    -> resolved to ~w~n', [M]) )),

    run_test("edge_store(auto): needs_reverse but csr costly -> lmdb_cached",
        ( resolve_auto_edge_store_fs(
            [edge_store(auto), edge_count(6000),
             expected_query_count(300), expected_lookups_per_query(50),
             needs_reverse(true)], R),
          member(lmdb_materialisation(cached), R) )),

    run_test("edge_store not auto -> pass through",
        ( resolve_auto_edge_store_fs(
            [edge_store(csr), edge_count(1000)], R),
          member(edge_store(csr), R) )),

    run_test("edge_store absent -> pass through",
        ( resolve_auto_edge_store_fs([edge_count(1000)], R),
          \+ member(edge_store(_), R) )),

    % -- LMDB materialisation resolver tests --

    run_test("materialisation(auto): large dataset, constrained memory -> cached",
        ( resolve_auto_lmdb_materialisation_fs(
            [lmdb_materialisation(auto), fact_count(500000),
             demand_set_estimate(500000), memory_budget(1000)], R),
          member(lmdb_materialisation(cached), R) )),

    run_test("materialisation(auto): small facts, many queries -> eager",
        ( resolve_auto_lmdb_materialisation_fs(
            [lmdb_materialisation(auto), fact_count(5000),
             expected_query_count(100), memory_budget(1000000000)], R),
          member(lmdb_materialisation(eager), R) )),

    run_test("materialisation(auto): workload_segregated -> lazy when large",
        ( resolve_auto_lmdb_materialisation_fs(
            [lmdb_materialisation(auto), fact_count(1000000),
             demand_set_estimate(1000000),
             memory_budget(100), workload_segregated(true)], R),
          member(lmdb_materialisation(lazy), R) )),

    run_test("materialisation explicit cached -> pass through",
        ( resolve_auto_lmdb_materialisation_fs(
            [lmdb_materialisation(cached), fact_count(50000)], R),
          member(lmdb_materialisation(cached), R) )),

    run_test("materialisation explicit eager -> pass through",
        ( resolve_auto_lmdb_materialisation_fs(
            [lmdb_materialisation(eager)], R),
          member(lmdb_materialisation(eager), R) )),

    % -- L2 capacity resolver tests --

    run_test("l2_capacity(auto) with cached -> computed in [256, 65536]",
        ( resolve_auto_lmdb_cache_tier_fs(
            [lmdb_l2_capacity(auto), lmdb_materialisation(cached),
             fact_count(10000)], R),
          member(lmdb_l2_capacity(Cap), R),
          Cap >= 256, Cap =< 65536 )),

    run_test("l2_capacity(auto) with eager -> 0",
        ( resolve_auto_lmdb_cache_tier_fs(
            [lmdb_l2_capacity(auto), lmdb_materialisation(eager)], R),
          member(lmdb_l2_capacity(0), R) )),

    run_test("l2_capacity(auto) with lazy -> 0",
        ( resolve_auto_lmdb_cache_tier_fs(
            [lmdb_l2_capacity(auto), lmdb_materialisation(lazy)], R),
          member(lmdb_l2_capacity(0), R) )),

    run_test("l2_capacity explicit 1024 -> pass through",
        ( resolve_auto_lmdb_cache_tier_fs(
            [lmdb_l2_capacity(1024), lmdb_materialisation(cached)], R),
          member(lmdb_l2_capacity(1024), R) )),

    % -- CSR index backend resolver tests --

    run_test("csr_index_backend without csr_path -> pass through",
        ( resolve_auto_csr_index_backend_fs(
            [csr_index_backend(auto)], R),
          member(csr_index_backend(auto), R) )),

    % -- Composed chain tests --

    run_test("full chain: all auto -> all resolved",
        ( resolve_fsharp_cost_options(
            [lmdb_materialisation(auto), lmdb_l2_capacity(auto),
             fact_count(10000), memory_budget(1000000000)], R),
          member(lmdb_materialisation(Mode), R), Mode \= auto,
          member(lmdb_l2_capacity(Cap), R), integer(Cap) )),

    run_test("full chain: no auto values -> pass through unchanged",
        ( resolve_fsharp_cost_options(
            [lmdb_materialisation(cached), lmdb_l2_capacity(512)], R),
          member(lmdb_materialisation(cached), R),
          member(lmdb_l2_capacity(512), R) )),

    % -- Summary --
    test_count(pass, P),
    test_count(fail, F),
    format('~n=== Results: ~w/~w passed ===~n', [P, P + F]),
    (F > 0 -> halt(1) ; true).

run_test(Name, Goal0) :-
    copy_term(Goal0, Goal),
    (   catch(Goal, Err,
              (format("~w: ERROR: ~w~n", [Name, Err]),
               retract(test_count(fail, F0)), F1 is F0 + 1,
               assertz(test_count(fail, F1))))
    ->  format("~w: PASS~n", [Name]),
        retract(test_count(pass, P)), P1 is P + 1,
        assertz(test_count(pass, P1))
    ;   format("~w: FAIL~n", [Name]),
        retract(test_count(fail, F)), F1 is F + 1,
        assertz(test_count(fail, F1))
    ).
