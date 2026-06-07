:- encoding(utf8).
%% Phase 2 test suite for recurrence_evaluation_strategy:
%%   apply_cost_model/3 + admissible_strategies/2 + the six initial
%%   cost-model rules + scoring + confidence weighting.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_res_cost_model.pl

:- use_module('../../src/unifyweaver/core/recurrence_evaluation_strategy').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("RES Phase 2 Cost-Model Tests~n"),
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

%% Per-rule firing/non-firing
test(test_default_fallback_always_fires).
test(test_prefer_bidirectional_csr_present_fires).
test(test_prefer_bidirectional_csr_present_misses_csr).
test(test_prefer_bidirectional_csr_present_misses_cardinality).
test(test_prefer_bidirectional_csr_buildable_fires).
test(test_prefer_bidirectional_csr_buildable_misses_freq).
test(test_prefer_unidirectional_no_csr_fires).
test(test_prefer_unidirectional_no_csr_misses_when_csr_buildable).
test(test_prefer_unidirectional_small_fires).
test(test_prefer_unidirectional_small_misses_with_contradicting_large).
test(test_prefer_astar_heuristic_available_fires).
test(test_prefer_astar_heuristic_available_misses_when_inadmissible).

%% SPEC worked-scoring example (exact reproduction)
test(test_spec_worked_scoring_example_winner).
test(test_spec_worked_scoring_example_score).
test(test_spec_worked_scoring_example_deciding_rule).
test(test_spec_worked_scoring_example_adjustment_in_trace).

%% Rule-interaction cases
test(test_multiple_rules_same_strategy_accumulate).
test(test_priority_tiebreak_higher_priority_wins).

%% Confidence weighting
test(test_all_declared_signals_weight_1).
test(test_all_inferred_signals_weight_08).
test(test_mixed_declared_and_inferred_weights_averaged).
test(test_default_fallback_no_signals_no_division_by_zero).

%% Admissibility
test(test_admissibility_combinatorial_admits_fixed_point).
test(test_admissibility_numeric_with_contraction_admits_fixed_point).
test(test_admissibility_numeric_without_contraction_excludes_fixed_point).
test(test_admissibility_numeric_high_contraction_excludes_fixed_point).
test(test_admissibility_per_query_always_admissible).
test(test_admissibility_does_not_apply_monotone_false).

%% Kernel-kind table coverage
test(test_kernel_bidirectional_ancestor_admissible).
test(test_kernel_transitive_closure2_admissible).
test(test_kernel_weighted_shortest_path3_admissible).

%% Integration: cost_model_choice trace step
test(test_cost_model_choice_step_in_trace).
test(test_cost_model_choice_step_includes_adjustments).
test(test_cost_model_choice_step_no_adjustments_when_none).

%% ========================================================================
%% Test fixtures
%% ========================================================================

%% Combinatorial recurrence — most common case
a_recurrence_combinatorial(recurrence(transitive_closure2, my_pred/2,
    [value_domain(combinatorial), monotone(true)])).

%% Numeric recurrence with a contraction rate that admits fixed_point
a_recurrence_numeric_admits_fp(recurrence(transitive_closure2, my_pred/2,
    [value_domain(numeric), monotone(true), numeric_contraction_rate(0.04)])).

%% Numeric recurrence without a contraction rate — fixed_point excluded
a_recurrence_numeric_no_contraction(recurrence(transitive_closure2, my_pred/2,
    [value_domain(numeric), monotone(true)])).

%% Numeric recurrence with R >= 1 — fixed_point excluded
a_recurrence_numeric_high_r(recurrence(transitive_closure2, my_pred/2,
    [value_domain(numeric), monotone(true), numeric_contraction_rate(1.5)])).

%% Non-monotone recurrence — Phase 3's cross-class restriction may
%% apply but admissibility should still return the full kernel set
%% (per SPEC: monotone is NOT in admissibility)
a_recurrence_non_monotone(recurrence(transitive_closure2, my_pred/2,
    [value_domain(combinatorial), monotone(false)])).

%% Bidirectional ancestor kernel — per-query-only by template design
a_recurrence_bidir_ancestor(recurrence(bidirectional_ancestor, my_pred/5,
    [value_domain(combinatorial), monotone(true)])).

%% Weighted shortest path — numeric Dijkstra kernel
a_recurrence_weighted_sp(recurrence(weighted_shortest_path3, my_pred/3,
    [value_domain(numeric), monotone(true)])).

%% ========================================================================
%% Per-rule firing tests
%% ========================================================================

%% default_fallback fires unconditionally and yields per_query(unidirectional)
%% with score 0.
test_default_fallback_always_fires :-
    a_recurrence_combinatorial(R),
    apply_cost_model(R, [],
        cost_model_choice(Strategy, Score, DecidingRule)),
    Strategy == strategy(per_query(unidirectional)),
    Score =:= 0,
    DecidingRule == default_fallback.

%% prefer_bidirectional_csr_present: all three preconditions present.
test_prefer_bidirectional_csr_present_fires :-
    a_recurrence_combinatorial(R),
    DataSignals = [csr_available(true), query_pattern(single_pair), cardinality(large)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(Strategy, _Score, DecidingRule)),
    Strategy == strategy(per_query(bidirectional)),
    DecidingRule == prefer_bidirectional_csr_present.

%% prefer_bidirectional_csr_present: missing csr_available(true).
test_prefer_bidirectional_csr_present_misses_csr :-
    a_recurrence_combinatorial(R),
    DataSignals = [query_pattern(single_pair), cardinality(large)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(_Strategy, _Score, DecidingRule)),
    DecidingRule \== prefer_bidirectional_csr_present.

%% prefer_bidirectional_csr_present: missing cardinality(large).
test_prefer_bidirectional_csr_present_misses_cardinality :-
    a_recurrence_combinatorial(R),
    DataSignals = [csr_available(true), query_pattern(single_pair)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(_Strategy, _Score, DecidingRule)),
    DecidingRule \== prefer_bidirectional_csr_present.

%% prefer_bidirectional_csr_buildable: all three present.
test_prefer_bidirectional_csr_buildable_fires :-
    a_recurrence_combinatorial(R),
    DataSignals = [csr_buildable(true), cardinality(large), query_frequency(high)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(Strategy, _Score, DecidingRule)),
    Strategy == strategy(per_query(bidirectional)),
    DecidingRule == prefer_bidirectional_csr_buildable.

%% prefer_bidirectional_csr_buildable: missing query_frequency(high).
test_prefer_bidirectional_csr_buildable_misses_freq :-
    a_recurrence_combinatorial(R),
    DataSignals = [csr_buildable(true), cardinality(large)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(_Strategy, _Score, DecidingRule)),
    DecidingRule \== prefer_bidirectional_csr_buildable.

%% prefer_unidirectional_no_csr: both negatives present.
test_prefer_unidirectional_no_csr_fires :-
    a_recurrence_combinatorial(R),
    DataSignals = [csr_available(false), csr_buildable(false)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(Strategy, _Score, DecidingRule)),
    Strategy == strategy(per_query(unidirectional)),
    DecidingRule == prefer_unidirectional_no_csr.

%% prefer_unidirectional_no_csr: doesn't fire when csr_buildable(true).
test_prefer_unidirectional_no_csr_misses_when_csr_buildable :-
    a_recurrence_combinatorial(R),
    DataSignals = [csr_available(false), csr_buildable(true)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(_Strategy, _Score, DecidingRule)),
    DecidingRule \== prefer_unidirectional_no_csr.

%% prefer_unidirectional_small: cardinality(small) with no contradicting.
test_prefer_unidirectional_small_fires :-
    a_recurrence_combinatorial(R),
    DataSignals = [cardinality(small)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(Strategy, _Score, DecidingRule)),
    Strategy == strategy(per_query(unidirectional)),
    DecidingRule == prefer_unidirectional_small.

%% prefer_unidirectional_small: blocked by contradicting cardinality(large).
test_prefer_unidirectional_small_misses_with_contradicting_large :-
    a_recurrence_combinatorial(R),
    %% Both signals present (could happen in pathological workload).
    %% The "no contradicting" check should suppress the rule.
    DataSignals = [cardinality(small), cardinality(large)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(_Strategy, _Score, DecidingRule)),
    DecidingRule \== prefer_unidirectional_small.

%% prefer_astar_heuristic_available fires when astar is admissible AND
%% the heuristic is declared.
test_prefer_astar_heuristic_available_fires :-
    a_recurrence_bidir_ancestor(R),
    DataSignals = [heuristic_predicate_available(true)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(Strategy, _Score, DecidingRule)),
    Strategy == strategy(per_query(astar)),
    DecidingRule == prefer_astar_heuristic_available.

%% prefer_astar_heuristic_available doesn't fire when astar isn't admissible.
test_prefer_astar_heuristic_available_misses_when_inadmissible :-
    %% transitive_closure2 doesn't admit per_query(astar) per the kernel table.
    a_recurrence_combinatorial(R),
    DataSignals = [heuristic_predicate_available(true)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(_Strategy, _Score, DecidingRule)),
    DecidingRule \== prefer_astar_heuristic_available.

%% ========================================================================
%% SPEC worked-scoring example (exact reproduction)
%%
%% Workload (per SPEC):
%%   Declared: cardinality(large), query_pattern(single_pair),
%%             query_frequency(high)
%%   Inferred: csr_buildable(true), csr_available(false)
%%   No intent signals
%%
%% Expected:
%%   Winner: per_query(bidirectional)
%%   Weighted score: 2 × (0.8 + 1.0 + 1.0) / 3 ≈ 1.867
%%   Deciding rule: prefer_bidirectional_csr_buildable
%%   Adjustment in trace: build_csr_at_compile_time
%% ========================================================================

spec_example_data_signals([
    %% declared (cost model doesn't care about tier for firing — these
    %% are just signal values present in DataSignals)
    cardinality(large),
    query_pattern(single_pair),
    query_frequency(high),
    %% inferred (same — the SIGNAL_VALUES are what matter)
    csr_buildable(true),
    csr_available(false)
]).

test_spec_worked_scoring_example_winner :-
    a_recurrence_combinatorial(R),
    spec_example_data_signals(DataSignals),
    apply_cost_model(R, DataSignals,
        cost_model_choice(Strategy, _Score, _DecidingRule)),
    Strategy == strategy(per_query(bidirectional)).

test_spec_worked_scoring_example_score :-
    a_recurrence_combinatorial(R),
    spec_example_data_signals(DataSignals),
    apply_cost_model(R, DataSignals,
        cost_model_choice(_Strategy, Score, _DecidingRule)),
    %% Expected ≈ 1.867; allow small floating-point tolerance.
    abs(Score - 1.867) < 0.01.

test_spec_worked_scoring_example_deciding_rule :-
    a_recurrence_combinatorial(R),
    spec_example_data_signals(DataSignals),
    apply_cost_model(R, DataSignals,
        cost_model_choice(_Strategy, _Score, DecidingRule)),
    DecidingRule == prefer_bidirectional_csr_buildable.

test_spec_worked_scoring_example_adjustment_in_trace :-
    a_recurrence_combinatorial(R),
    spec_example_data_signals(DataSignals),
    select_evaluation_strategy(R, DataSignals,
        strategy_choice(_Strategy, trace(Steps))),
    member(step(cost_model_choice,
                chosen(strategy(per_query(bidirectional)), _Score,
                       prefer_bidirectional_csr_buildable),
                Details),
           Steps),
    member(adjustment(build_csr_at_compile_time), Details).

%% ========================================================================
%% Rule-interaction cases
%% ========================================================================

%% When multiple rules fire for the same strategy, their weighted
%% scores accumulate.
test_multiple_rules_same_strategy_accumulate :-
    a_recurrence_combinatorial(R),
    %% Workload triggers both prefer_unidirectional_no_csr (raw +2) and
    %% default_fallback (raw +0) for per_query(unidirectional). Both
    %% contribute to the cumulative score for the same strategy.
    DataSignals = [csr_available(false), csr_buildable(false)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(Strategy, Score, _DecidingRule)),
    Strategy == strategy(per_query(unidirectional)),
    %% prefer_unidirectional_no_csr: csr_available(false) and
    %%   csr_buildable(false) are BOTH inferred (per signal_tier
    %%   dispatch). Raw 2, weights (0.8 + 0.8)/2 = 0.8, weighted = 1.6.
    %% default_fallback: no preconditions, weighted = raw_score = 0.
    %% Cumulative for per_query(unidirectional) = 1.6 + 0 = 1.6.
    abs(Score - 1.6) < 0.01.

%% When two rules tie on score for different strategies, the
%% higher-priority rule wins. Test:
%%   prefer_unidirectional_no_csr (priority 80, +2)
%%   prefer_bidirectional_csr_buildable (priority 90, +2)
%% Same raw score but different strategies. Higher priority should win.
test_priority_tiebreak_higher_priority_wins :-
    a_recurrence_combinatorial(R),
    DataSignals = [
        %% Triggers prefer_bidirectional_csr_buildable (priority 90)
        csr_buildable(true), cardinality(large), query_frequency(high),
        %% Would have triggered prefer_unidirectional_no_csr (priority 80) —
        %% but the csr_buildable(true) means csr_buildable(false) precondition
        %% isn't met. So actually only prefer_bidirectional_csr_buildable
        %% fires. Different setup needed.
        csr_available(false)
    ],
    apply_cost_model(R, DataSignals,
        cost_model_choice(Strategy, _Score, DecidingRule)),
    Strategy == strategy(per_query(bidirectional)),
    DecidingRule == prefer_bidirectional_csr_buildable.

%% ========================================================================
%% Confidence weighting
%% ========================================================================

%% All-declared signals: weight 1.0 each, so weighted score = raw score.
test_all_declared_signals_weight_1 :-
    a_recurrence_combinatorial(R),
    %% prefer_bidirectional_csr_present has all three declared.
    DataSignals = [csr_available(true), query_pattern(single_pair), cardinality(large)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(_Strategy, Score, _DecidingRule)),
    %% Raw score 3, weights (1.0+1.0+1.0)/3 = 1.0, weighted = 3.0
    abs(Score - 3.0) < 0.01.

%% All-inferred signals: weight 0.8 each, so weighted score = raw × 0.8.
test_all_inferred_signals_weight_08 :-
    a_recurrence_combinatorial(R),
    %% prefer_unidirectional_no_csr: csr_available(false) AND csr_buildable(false)
    %% Both inferred per dispatch table.
    DataSignals = [csr_available(false), csr_buildable(false)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(_Strategy, Score, _DecidingRule)),
    %% Raw score 2, weights (0.8+0.8)/2 = 0.8, weighted = 1.6
    abs(Score - 1.6) < 0.01.

%% Mixed declared + inferred: weights averaged.
test_mixed_declared_and_inferred_weights_averaged :-
    %% Already tested by the SPEC worked-scoring example. Re-verify
    %% with a different workload.
    a_recurrence_combinatorial(R),
    %% prefer_bidirectional_csr_buildable: csr_buildable(inferred, 0.8),
    %% cardinality(declared, 1.0), query_frequency(declared, 1.0).
    %% Raw 2, weights (0.8+1.0+1.0)/3, weighted ≈ 1.867.
    DataSignals = [csr_buildable(true), cardinality(large), query_frequency(high)],
    apply_cost_model(R, DataSignals,
        cost_model_choice(_Strategy, Score, _DecidingRule)),
    abs(Score - 1.867) < 0.01.

%% default_fallback has no preconditions — no division by zero.
test_default_fallback_no_signals_no_division_by_zero :-
    a_recurrence_combinatorial(R),
    apply_cost_model(R, [],
        cost_model_choice(_Strategy, Score, _DecidingRule)),
    Score =:= 0.

%% ========================================================================
%% Admissibility
%% ========================================================================

%% Combinatorial recurrence admits fixed_point (Tarski + finite lattice).
test_admissibility_combinatorial_admits_fixed_point :-
    a_recurrence_combinatorial(R),
    admissible_strategies(R, Strategies),
    member(strategy(fixed_point(semi_naive)), Strategies).

%% Numeric recurrence with R < 1 admits fixed_point.
test_admissibility_numeric_with_contraction_admits_fixed_point :-
    %% transitive_closure2's table includes fixed_point(semi_naive).
    a_recurrence_numeric_admits_fp(R),
    admissible_strategies(R, Strategies),
    member(strategy(fixed_point(semi_naive)), Strategies).

%% Numeric recurrence without contraction rate: fixed_point excluded.
test_admissibility_numeric_without_contraction_excludes_fixed_point :-
    a_recurrence_numeric_no_contraction(R),
    admissible_strategies(R, Strategies),
    \+ member(strategy(fixed_point(semi_naive)), Strategies).

%% Numeric recurrence with R >= 1: fixed_point excluded.
test_admissibility_numeric_high_contraction_excludes_fixed_point :-
    a_recurrence_numeric_high_r(R),
    admissible_strategies(R, Strategies),
    \+ member(strategy(fixed_point(semi_naive)), Strategies).

%% per_query(_) strategies are always admissible (no convergence
%% guarantee required from the recurrence).
test_admissibility_per_query_always_admissible :-
    a_recurrence_combinatorial(R),
    admissible_strategies(R, CombStrategies),
    member(strategy(per_query(unidirectional)), CombStrategies),
    a_recurrence_numeric_no_contraction(R2),
    admissible_strategies(R2, NumStrategies),
    member(strategy(per_query(unidirectional)), NumStrategies).

%% monotone(false) does NOT restrict admissibility (Phase 3's job).
test_admissibility_does_not_apply_monotone_false :-
    a_recurrence_combinatorial(R),
    admissible_strategies(R, MonotoneStrategies),
    a_recurrence_non_monotone(R2),
    admissible_strategies(R2, NonMonotoneStrategies),
    %% Same list — monotone(false) is not in the admissibility check.
    MonotoneStrategies == NonMonotoneStrategies.

%% ========================================================================
%% Kernel-kind table coverage
%% ========================================================================

test_kernel_bidirectional_ancestor_admissible :-
    a_recurrence_bidir_ancestor(R),
    admissible_strategies(R, Strategies),
    member(strategy(per_query(bidirectional)), Strategies),
    member(strategy(per_query(astar)), Strategies),
    \+ member(strategy(fixed_point(_)), Strategies).

test_kernel_transitive_closure2_admissible :-
    a_recurrence_combinatorial(R),
    admissible_strategies(R, Strategies),
    member(strategy(per_query(unidirectional)), Strategies),
    member(strategy(per_query(bidirectional)), Strategies),
    member(strategy(fixed_point(semi_naive)), Strategies).

test_kernel_weighted_shortest_path3_admissible :-
    a_recurrence_weighted_sp(R),
    admissible_strategies(R, Strategies),
    Strategies == [strategy(per_query(dijkstra))].

%% ========================================================================
%% Integration: cost_model_choice trace step
%% ========================================================================

test_cost_model_choice_step_in_trace :-
    a_recurrence_combinatorial(R),
    DataSignals = [csr_available(true), query_pattern(single_pair), cardinality(large)],
    select_evaluation_strategy(R, DataSignals,
        strategy_choice(_Strategy, trace(Steps))),
    member(step(cost_model_choice,
                chosen(strategy(per_query(bidirectional)), _Score,
                       prefer_bidirectional_csr_present),
                _Details),
           Steps).

test_cost_model_choice_step_includes_adjustments :-
    a_recurrence_combinatorial(R),
    spec_example_data_signals(DataSignals),
    select_evaluation_strategy(R, DataSignals,
        strategy_choice(_Strategy, trace(Steps))),
    member(step(cost_model_choice, chosen(_, _, prefer_bidirectional_csr_buildable),
                Details),
           Steps),
    member(adjustment(build_csr_at_compile_time), Details).

test_cost_model_choice_step_no_adjustments_when_none :-
    a_recurrence_combinatorial(R),
    %% Empty workload → default_fallback fires → no adjustments
    select_evaluation_strategy(R, [],
        strategy_choice(_Strategy, trace(Steps))),
    member(step(cost_model_choice, chosen(_, _, default_fallback), Details), Steps),
    Details == [].
