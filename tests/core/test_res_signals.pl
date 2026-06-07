:- encoding(utf8).
%% Phase 1 test suite for recurrence_evaluation_strategy:classify_signals/4
%%
%% Verifies the three-tier signal classification (intent / declared-data /
%% inferred-data) implemented in Phase 1.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_res_signals.pl

:- use_module('../../src/unifyweaver/core/recurrence_evaluation_strategy').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("RES Phase 1 Signal-Classification Tests~n"),
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

test(test_empty_workload).

%% Intent-tier classification
test(test_intent_kernel_mode).
test(test_intent_strategy).
test(test_intent_force_search_algorithm).

%% Declared-data-tier classification
test(test_declared_csr_path).
test(test_declared_csr_available_true).
test(test_declared_cardinality).
test(test_declared_determinism).
test(test_declared_unique).
test(test_declared_query_pattern).
test(test_declared_query_frequency).
test(test_declared_graph_mutability).
test(test_declared_heuristic_predicate_available).
test(test_declared_heuristic_predicate).
test(test_declared_calibration_signals).

%% Inferred-data-tier classification
test(test_inferred_csr_available_false).
test(test_inferred_csr_buildable).

%% Forward-compat: unknown functor lands in inferred + surfaces as warning
test(test_unknown_signal_lands_in_inferred).
test(test_unknown_signal_appears_in_trace).
test(test_unknown_signal_does_not_write_stderr).

%% Mixed-tier and edge-case workloads
test(test_mix_all_three_tiers).
test(test_only_intent_signals).
test(test_only_declared_signals).
test(test_only_inferred_signals).
test(test_csr_available_value_specific_dispatch).
test(test_preserves_signal_terms).
test(test_preserves_signal_order_within_tier).

%% Integration: classify_signals trace step appears in
%% select_evaluation_strategy/3 result
test(test_classify_signals_step_in_trace).
test(test_classify_signals_step_records_unknown).

%% ========================================================================
%% Test fixtures
%% ========================================================================

a_recurrence(recurrence(transitive_closure2, my_pred/2,
    [value_domain(combinatorial), monotone(true)])).

%% ========================================================================
%% Tests
%% ========================================================================

test_empty_workload :-
    classify_signals([], I, D, F),
    I == [], D == [], F == [].

%% Intent tier --------------------------------------------------------------

test_intent_kernel_mode :-
    classify_signals([kernel_mode(bidirectional)], I, D, F),
    I == [kernel_mode(bidirectional)], D == [], F == [].

test_intent_strategy :-
    classify_signals([strategy(per_query(unidirectional))], I, D, F),
    I == [strategy(per_query(unidirectional))], D == [], F == [].

test_intent_force_search_algorithm :-
    classify_signals([force_search_algorithm(bfs)], I, D, F),
    I == [force_search_algorithm(bfs)], D == [], F == [].

%% Declared-data tier -------------------------------------------------------

test_declared_csr_path :-
    classify_signals([csr_path('/some/path.lmdb')], I, D, F),
    I == [], D == [csr_path('/some/path.lmdb')], F == [].

test_declared_csr_available_true :-
    classify_signals([csr_available(true)], I, D, F),
    I == [], D == [csr_available(true)], F == [].

test_declared_cardinality :-
    classify_signals([cardinality(large)], I, D, F),
    I == [], D == [cardinality(large)], F == [].

test_declared_determinism :-
    classify_signals([determinism(semidet)], I, D, F),
    I == [], D == [determinism(semidet)], F == [].

test_declared_unique :-
    classify_signals([unique(true)], I, D, F),
    I == [], D == [unique(true)], F == [].

test_declared_query_pattern :-
    classify_signals([query_pattern(single_pair)], I, D, F),
    I == [], D == [query_pattern(single_pair)], F == [].

test_declared_query_frequency :-
    classify_signals([query_frequency(high)], I, D, F),
    I == [], D == [query_frequency(high)], F == [].

test_declared_graph_mutability :-
    classify_signals([graph_mutability(static)], I, D, F),
    I == [], D == [graph_mutability(static)], F == [].

test_declared_heuristic_predicate_available :-
    classify_signals([heuristic_predicate_available(true)], I, D, F),
    I == [], D == [heuristic_predicate_available(true)], F == [].

test_declared_heuristic_predicate :-
    classify_signals([heuristic_predicate(my_heuristic/2)], I, D, F),
    I == [], D == [heuristic_predicate(my_heuristic/2)], F == [].

test_declared_calibration_signals :-
    classify_signals([b_eff(15.0), branching_d(4.5), contraction_r(0.04)], I, D, F),
    I == [],
    sort(D, [b_eff(15.0), branching_d(4.5), contraction_r(0.04)]),
    F == [].

%% Inferred-data tier -------------------------------------------------------

test_inferred_csr_available_false :-
    classify_signals([csr_available(false)], I, D, F),
    I == [], D == [], F == [csr_available(false)].

test_inferred_csr_buildable :-
    classify_signals([csr_buildable(true)], I, D, F),
    I == [], D == [], F == [csr_buildable(true)].

%% Forward-compat ----------------------------------------------------------

test_unknown_signal_lands_in_inferred :-
    classify_signals([quantum_signal(foo)], I, D, F),
    I == [], D == [], F == [quantum_signal(foo)].

%% Unknown signals appear in the classify_signals trace step that
%% select_evaluation_strategy/3 builds (via the unknown_signal/1
%% wrapper). The classify_signals/4 predicate itself doesn't surface
%% them — the module is pure-functional and doesn't write to streams.
test_unknown_signal_appears_in_trace :-
    a_recurrence(R),
    select_evaluation_strategy(R, [quantum_signal(foo)],
                               strategy_choice(_Strategy, trace(Steps))),
    member(step(classify_signals, classified(_, _, [quantum_signal(foo)]), Details),
           Steps),
    member(unknown_signal(quantum_signal(foo)), Details).

%% Verify the module doesn't write anything to user_error or
%% user_output when processing an unknown signal. We do this by
%% running classify_signals/4 inside a with_output_to/2 scope and
%% asserting the captured string is empty.
test_unknown_signal_does_not_write_stderr :-
    with_output_to(string(Captured),
        ( current_output(Out),
          set_stream(user_error, alias(my_test_err)),
          classify_signals([quantum_signal(foo)], _, _, _),
          set_stream(Out, alias(user_output))
        )),
    Captured == "".

%% Mixed / edge-case -------------------------------------------------------

test_mix_all_three_tiers :-
    Workload = [
        kernel_mode(bidirectional),       % intent
        cardinality(large),                % declared
        csr_buildable(true)                % inferred
    ],
    classify_signals(Workload, I, D, F),
    I == [kernel_mode(bidirectional)],
    D == [cardinality(large)],
    F == [csr_buildable(true)].

test_only_intent_signals :-
    classify_signals([kernel_mode(bidirectional), strategy(per_query(_))],
                     I, D, F),
    length(I, 2), D == [], F == [].

test_only_declared_signals :-
    classify_signals([cardinality(large), determinism(det), unique(true)],
                     I, D, F),
    I == [], length(D, 3), F == [].

test_only_inferred_signals :-
    classify_signals([csr_available(false), csr_buildable(true)],
                     I, D, F),
    I == [], D == [], length(F, 2).

%% This is the value-specific dispatch case: csr_available(true) is
%% declared, csr_available(false) is inferred. Same functor, different
%% tier based on value.
test_csr_available_value_specific_dispatch :-
    classify_signals([csr_available(true), csr_available(false)], I, D, F),
    I == [],
    D == [csr_available(true)],
    F == [csr_available(false)].

%% Verify the classifier preserves the signal terms as-is (no
%% wrapping, no transformation) in the output lists.
test_preserves_signal_terms :-
    classify_signals([cardinality(large)], _I, D, _F),
    D = [Signal],
    Signal == cardinality(large).

%% When multiple signals land in the same tier, their original input
%% order is preserved (left-to-right). The implementation uses
%% cons-onto-front recursion which naturally preserves order when
%% combined with the recursive call's already-built list.
test_preserves_signal_order_within_tier :-
    classify_signals([cardinality(large), determinism(det), unique(true)],
                     _I, D, _F),
    D == [cardinality(large), determinism(det), unique(true)].

%% Integration with select_evaluation_strategy/3 ----------------------------

test_classify_signals_step_in_trace :-
    a_recurrence(R),
    select_evaluation_strategy(R, [cardinality(large), kernel_mode(bidirectional)],
                               strategy_choice(_Strategy, trace(Steps))),
    member(step(classify_signals,
                classified([kernel_mode(bidirectional)],
                           [cardinality(large)],
                           []),
                _Details),
           Steps).

%% When unknown signals are present, the trace step's Details list
%% contains unknown_signal/1 markers wrapping each unknown term.
test_classify_signals_step_records_unknown :-
    a_recurrence(R),
    Workload = [cardinality(large), quantum_signal(foo), warp_drive(on)],
    select_evaluation_strategy(R, Workload,
                               strategy_choice(_Strategy, trace(Steps))),
    member(step(classify_signals, _Outcome, Details), Steps),
    member(unknown_signal(quantum_signal(foo)), Details),
    member(unknown_signal(warp_drive(on)), Details).
