:- encoding(utf8).
%% Unit tests for recurrence_inputs.pl:
%%   build_recurrence_term/3 + build_workload_signals/2 +
%%   kernel_kind_default_properties/2.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_recurrence_inputs.pl

:- use_module('../../src/unifyweaver/core/recurrence_inputs').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("recurrence_inputs Unit Tests~n"),
    format("========================================~n~n"),
    findall(Test, test(Test), Tests),
    length(Tests, Total),
    run_all(Tests, 0, Passed),
    format("~n========================================~n"),
    (   Passed =:= Total
    ->  format("All ~w tests passed~n", [Total])
    ;   Failed is Total - Passed,
        format("~w of ~w tests FAILED~n", [Failed, Total]),
        format("Tests FAILED~n"),
        halt(1)
    ),
    format("========================================~n").

run_all([], Passed, Passed).
run_all([Test|Rest], Acc, Passed) :-
    (   catch(call(Test), Error,
            (format("[FAIL] ~w: ~w~n", [Test, Error]), fail))
    ->  format("[PASS] ~w~n", [Test]),
        Acc1 is Acc + 1,
        run_all(Rest, Acc1, Passed)
    ;   run_all(Rest, Acc, Passed)
    ).

%% ========================================================================
%% Test declarations
%% ========================================================================

%% build_recurrence_term
test(test_build_recurrence_term_combinatorial_kernel).
test(test_build_recurrence_term_numeric_kernel).
test(test_build_recurrence_term_unknown_kernel_defaults).
test(test_build_recurrence_term_with_extra_properties).
test(test_build_recurrence_term_extra_overrides_defaults).
test(test_build_recurrence_term_drops_directions_admissible).
test(test_build_recurrence_term_drops_expected_cardinality).

%% kernel_kind_default_properties
test(test_default_properties_category_ancestor).
test(test_default_properties_weighted_shortest_path3).
test(test_default_properties_unknown_kernel).

%% build_workload_signals — intent options
test(test_workload_kernel_mode).
test(test_workload_strategy).
test(test_workload_force_search_algorithm).

%% build_workload_signals — declared-data options
test(test_workload_csr_path_implies_csr_available).
test(test_workload_csr_available).
test(test_workload_cardinality).
test(test_workload_determinism).
test(test_workload_unique).
test(test_workload_query_pattern).
test(test_workload_query_frequency).
test(test_workload_graph_mutability).
test(test_workload_heuristic_predicate_implies_available).
test(test_workload_heuristic_predicate_available).
test(test_workload_calibration_signals).

%% build_workload_signals — inferred-data options
test(test_workload_csr_buildable).

%% build_workload_signals — edge cases
test(test_workload_empty).
test(test_workload_drops_unknown_options).
test(test_workload_mixed_intents_data).

%% ========================================================================
%% Tests: build_recurrence_term
%% ========================================================================

test_build_recurrence_term_combinatorial_kernel :-
    build_recurrence_term(
        recursive_kernel(transitive_closure2, my_pred/2, []),
        [],
        Recurrence),
    Recurrence = recurrence(transitive_closure2, my_pred/2, Props),
    member(value_domain(combinatorial), Props),
    member(monotone(true), Props).

test_build_recurrence_term_numeric_kernel :-
    build_recurrence_term(
        recursive_kernel(weighted_shortest_path3, my_pred/3, []),
        [],
        Recurrence),
    Recurrence = recurrence(weighted_shortest_path3, my_pred/3, Props),
    member(value_domain(numeric), Props),
    member(monotone(true), Props).

test_build_recurrence_term_unknown_kernel_defaults :-
    build_recurrence_term(
        recursive_kernel(quantum_kernel, my_pred/2, []),
        [],
        Recurrence),
    Recurrence = recurrence(quantum_kernel, my_pred/2, Props),
    %% Catch-all defaults: combinatorial + monotone(true).
    member(value_domain(combinatorial), Props),
    member(monotone(true), Props).

test_build_recurrence_term_with_extra_properties :-
    build_recurrence_term(
        recursive_kernel(transitive_closure2, my_pred/2, []),
        [numeric_contraction_rate(0.04)],
        Recurrence),
    Recurrence = recurrence(_, _, Props),
    member(numeric_contraction_rate(0.04), Props),
    %% Defaults still present alongside the extra.
    member(value_domain(combinatorial), Props).

test_build_recurrence_term_extra_overrides_defaults :-
    %% Override the default monotone(true) with monotone(false).
    build_recurrence_term(
        recursive_kernel(transitive_closure2, my_pred/2, []),
        [monotone(false)],
        Recurrence),
    Recurrence = recurrence(_, _, Props),
    member(monotone(false), Props),
    %% Original monotone(true) should NOT be present.
    \+ member(monotone(true), Props).

test_build_recurrence_term_drops_directions_admissible :-
    %% Even if extra properties include directions_admissible, it
    %% should be dropped per the SPEC's removed-property note.
    build_recurrence_term(
        recursive_kernel(transitive_closure2, my_pred/2, []),
        [directions_admissible([forward, backward])],
        Recurrence),
    Recurrence = recurrence(_, _, Props),
    \+ member(directions_admissible(_), Props).

test_build_recurrence_term_drops_expected_cardinality :-
    %% expected_cardinality duplicates cardinality(C) workload signal;
    %% SPEC says drop it.
    build_recurrence_term(
        recursive_kernel(transitive_closure2, my_pred/2, []),
        [expected_cardinality(large)],
        Recurrence),
    Recurrence = recurrence(_, _, Props),
    \+ member(expected_cardinality(_), Props).

%% ========================================================================
%% Tests: kernel_kind_default_properties
%% ========================================================================

test_default_properties_category_ancestor :-
    kernel_kind_default_properties(category_ancestor, Props),
    member(value_domain(combinatorial), Props),
    member(monotone(true), Props),
    member(has_combinatorial_loop_break(true), Props).

test_default_properties_weighted_shortest_path3 :-
    kernel_kind_default_properties(weighted_shortest_path3, Props),
    member(value_domain(numeric), Props),
    member(monotone(true), Props),
    %% Dijkstra: no visited-set, has_combinatorial_loop_break absent.
    \+ member(has_combinatorial_loop_break(_), Props).

test_default_properties_unknown_kernel :-
    kernel_kind_default_properties(definitely_not_a_kernel, Props),
    member(value_domain(combinatorial), Props),
    member(monotone(true), Props).

%% ========================================================================
%% Tests: build_workload_signals — intent options
%% ========================================================================

test_workload_kernel_mode :-
    build_workload_signals([kernel_mode(bidirectional)], W),
    W == [kernel_mode(bidirectional)].

test_workload_strategy :-
    build_workload_signals([strategy(per_query(unidirectional))], W),
    W == [strategy(per_query(unidirectional))].

test_workload_force_search_algorithm :-
    build_workload_signals([force_search_algorithm(bfs)], W),
    W == [force_search_algorithm(bfs)].

%% ========================================================================
%% Tests: build_workload_signals — declared-data options
%% ========================================================================

%% csr_path implies BOTH csr_path AND csr_available(true).
test_workload_csr_path_implies_csr_available :-
    build_workload_signals([csr_path('/foo/bar.lmdb')], W),
    member(csr_path('/foo/bar.lmdb'), W),
    member(csr_available(true), W).

test_workload_csr_available :-
    build_workload_signals([csr_available(true)], W),
    W == [csr_available(true)].

test_workload_cardinality :-
    build_workload_signals([cardinality(large)], W),
    W == [cardinality(large)].

test_workload_determinism :-
    build_workload_signals([determinism(semidet)], W),
    W == [determinism(semidet)].

test_workload_unique :-
    build_workload_signals([unique(true)], W),
    W == [unique(true)].

test_workload_query_pattern :-
    build_workload_signals([query_pattern(single_pair)], W),
    W == [query_pattern(single_pair)].

test_workload_query_frequency :-
    build_workload_signals([query_frequency(high)], W),
    W == [query_frequency(high)].

test_workload_graph_mutability :-
    build_workload_signals([graph_mutability(static)], W),
    W == [graph_mutability(static)].

%% heuristic_predicate implies heuristic_predicate_available(true).
test_workload_heuristic_predicate_implies_available :-
    build_workload_signals([heuristic_predicate(my_heuristic/2)], W),
    member(heuristic_predicate(my_heuristic/2), W),
    member(heuristic_predicate_available(true), W).

test_workload_heuristic_predicate_available :-
    build_workload_signals([heuristic_predicate_available(true)], W),
    W == [heuristic_predicate_available(true)].

test_workload_calibration_signals :-
    build_workload_signals([b_eff(15.0), branching_d(4.5), contraction_r(0.04)], W),
    member(b_eff(15.0), W),
    member(branching_d(4.5), W),
    member(contraction_r(0.04), W).

%% ========================================================================
%% Tests: build_workload_signals — inferred-data options
%% ========================================================================

test_workload_csr_buildable :-
    build_workload_signals([csr_buildable(true)], W),
    W == [csr_buildable(true)].

%% ========================================================================
%% Tests: build_workload_signals — edge cases
%% ========================================================================

test_workload_empty :-
    build_workload_signals([], W),
    W == [].

%% Unknown options (target-specific keys) are silently dropped.
test_workload_drops_unknown_options :-
    build_workload_signals([
        lmdb_path('/some/path'),
        target(fsharp_wam),
        no_kernels(false),
        kernel_mode(bidirectional)       % this one IS recognised
    ], W),
    W == [kernel_mode(bidirectional)].

test_workload_mixed_intents_data :-
    build_workload_signals([
        kernel_mode(bidirectional),
        cardinality(large),
        query_pattern(single_pair),
        csr_buildable(true)
    ], W),
    %% Order is preserved per the translate_option order.
    W = [kernel_mode(bidirectional), cardinality(large),
         query_pattern(single_pair), csr_buildable(true)].
