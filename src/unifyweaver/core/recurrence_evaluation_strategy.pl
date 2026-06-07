:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% recurrence_evaluation_strategy.pl — pick an evaluation strategy for
% a detected recursive kernel.
%
% Given a Recurrence term (constructed from a detected kernel +
% clause analysis + algorithm manifest) and a Workload signal list,
% return the strategy the target should use to emit code, plus a
% structured reasoning trace.
%
% Phase status: Phase 0 (skeleton) + Phase 1 (signal classification)
% + Phase 2 (cost-model rules + admissibility) landed.
% classify_signals/4, apply_cost_model/3, admissible_strategies/2
% are real; resolve_against_intent/5 and the renderers remain stubs.
% The cost model produces real strategy preferences from the six
% initial rules; the trace contains a real step(cost_model_choice,
% ...) entry. The Phase 0 step(stub, ...) marker is still prepended
% to the trace for the Phase 5 leak-detector to assert against.
% Phases 3-4 will replace the remaining stubs.
%
% This baseline lets the F# WAM target integration (Phase 5) call
% against the module from day one, and lets the stub-leak assertion
% in Phase 5 tests fail loud if real selections still produce stub
% steps after Phases 1-4 land.
%
% Determinism contract: every exported predicate is `det`. None
% fails; the only errors are structural (no_admissible_strategy/1
% from select_evaluation_strategy/3, and type errors from the
% valid_* checkers).
%
% Cross-references:
%   - docs/design/RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md
%   - docs/design/RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md
%   - docs/design/RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md
%
% The trace step name `stub` is the explicit tag Phase 5 tests
% assert against to catch leak-through of unimplemented helpers.

:- module(recurrence_evaluation_strategy, [
    %% Main entry point
    select_evaluation_strategy/3,        % +Recurrence, +Workload, -StrategyAndTrace

    %% Pipeline helpers (called by select_evaluation_strategy/3,
    %% also exported for direct test access and future composability)
    classify_signals/4,                  % +Workload, -Intent, -DeclaredData, -InferredData
    apply_cost_model/3,                  % +Recurrence, +DataSignals, -CostModelChoice
    resolve_against_intent/5,            % +IntentSignals, +CostModelChoice, +Recurrence,
                                         %   -Strategy, -Trace

    %% Trace renderers (called by target adapters, NOT by the
    %% selector itself - the module is pure-functional)
    render_trace_for_stderr/2,           % +Trace, -StderrLines
    format_trace_for_comment/3,          % +Trace, +CommentPrefix, -CommentString

    %% Introspection
    admissible_strategies/2,             % +Recurrence, -StrategyList
    strategy_pretty/2,                   % +Strategy, -String

    %% Type-check helpers (exported so tests can verify the contracts
    %% without re-implementing them)
    valid_recurrence/1,                  % +Recurrence
    valid_workload/1,                    % +Workload
    valid_strategy/1                     % +Strategy (checks the outer
                                         % strategy/1 wrapper, not just
                                         % the inner Mode)
]).

:- use_module(library(error), [must_be/2, domain_error/2, type_error/2]).
:- use_module(library(lists), [append/3, sum_list/2, member/2]).
:- use_module(library(apply), [maplist/3, include/3, foldl/4]).
%% predsort/3 is autoloadable; no explicit import needed.

%% ============================================================
%% Public API — Phase 0 stubs
%% ============================================================

%% select_evaluation_strategy(+Recurrence, +Workload, -StrategyAndTrace)
%
% Pre-wires the full pipeline:
%
%   classify_signals(Workload, I, D, F)
%   append(D, F, DataSignals)
%   apply_cost_model(Recurrence, DataSignals, CostModelChoice)
%   resolve_against_intent(I, CostModelChoice, Recurrence, Strategy, Trace)
%
% Phase 0: every helper returns a stub; the baseline strategy emerges
% as `per_query(unidirectional)` with a `step(stub, ...)` trace entry.
%
% Phases 1-4 replace the helpers; this entry point's structure does
% not change.
select_evaluation_strategy(Recurrence, Workload, strategy_choice(Strategy, Trace)) :-
    valid_recurrence(Recurrence),
    valid_workload(Workload),
    %% Phase 1: real classification + collect unknowns for the trace step.
    classify_signals_internal(Workload, IntentSignals, DeclaredData, InferredData,
                              Unknowns),
    append(DeclaredData, InferredData, DataSignals),
    %% Phase 2: real cost-model evaluation.
    apply_cost_model(Recurrence, DataSignals, CostModelChoice),
    resolve_against_intent(IntentSignals, CostModelChoice, Recurrence, Strategy, Trace0),
    %% Build the classify_signals trace step from the classification result.
    build_classify_signals_step(IntentSignals, DeclaredData, InferredData, Unknowns,
                                ClassifyStep),
    %% Build the cost_model_choice trace step with adjustments in Details.
    build_cost_model_choice_step(CostModelChoice, CostModelChoiceStep),
    Trace0 = trace(ResolveSteps),
    %% Phase 0: prepend the stub marker so the stub-leak assertion can
    %% detect partial implementation while phases 3-4 are still in
    %% progress. Removed once all phases land.
    Trace  = trace([step(stub, not_yet_implemented(phase_0), []),
                    ClassifyStep,
                    CostModelChoiceStep
                    | ResolveSteps]).

build_cost_model_choice_step(
        cost_model_choice(Strategy, Score, DecidingRule),
        step(cost_model_choice,
             chosen(Strategy, Score, DecidingRule),
             Details)) :-
    rule_adjustments(DecidingRule, Adjustments),
    (   Adjustments == []
    ->  Details = []
    ;   maplist(wrap_adjustment, Adjustments, AdjustmentTerms),
        Details = AdjustmentTerms
    ).

wrap_adjustment(Adjustment, adjustment(Adjustment)).

build_classify_signals_step(Intent, Declared, Inferred, Unknowns,
                            step(classify_signals,
                                 classified(Intent, Declared, Inferred),
                                 Details)) :-
    (   Unknowns == []
    ->  Details = []
    ;   maplist(wrap_unknown_signal, Unknowns, UnknownDetails),
        Details = UnknownDetails
    ).

wrap_unknown_signal(Signal, unknown_signal(Signal)).

%% classify_signals(+Workload, -IntentSignals, -DeclaredData, -InferredData)
%
% Phase 1: sort each workload term into one of three tiers using the
% signal_tier/3 dispatch table. Unknown signal functors land in
% InferredData (with their unknown status surfaced via the trace
% step that select_evaluation_strategy/3 builds — the unknown-signal
% warning is NOT visible from this predicate alone, by design: the
% module is pure-functional and does not write to streams).
%
% Public arity-4 version drops the unknown-signal list; use the
% internal /5 version from inside this module when the warning list
% is needed.
classify_signals(Workload, IntentSignals, DeclaredData, InferredData) :-
    classify_signals_internal(Workload, IntentSignals, DeclaredData, InferredData,
                              _Unknowns).

%% classify_signals_internal(+Workload, -Intent, -Declared, -Inferred,
%%                            -Unknowns)
%
% Five-argument internal version that also returns the list of
% workload terms whose functor was not in the dispatch table. These
% terms were classified into InferredData per the forward-compat
% policy; Unknowns is the subset surfaced as warnings in the trace.
classify_signals_internal([], [], [], [], []).
classify_signals_internal([Signal|Rest], Intent, Declared, Inferred, Unknowns) :-
    (   signal_tier(Signal, Tier, _Source)
    ->  classify_signals_internal(Rest, Intent0, Declared0, Inferred0, Unknowns),
        add_to_tier(Tier, Signal, Intent0, Declared0, Inferred0,
                    Intent, Declared, Inferred)
    ;   %% Unknown functor: lands in Inferred per forward-compat policy,
        %% AND added to Unknowns so the caller can surface a warning.
        classify_signals_internal(Rest, Intent, Declared, Inferred0, Unknowns0),
        Inferred = [Signal|Inferred0],
        Unknowns = [Signal|Unknowns0]
    ).

add_to_tier(intent,        S, I, D, F, [S|I], D, F).
add_to_tier(declared_data, S, I, D, F, I, [S|D], F).
add_to_tier(inferred_data, S, I, D, F, I, D, [S|F]).

%% ============================================================
%% Signal dispatch table — signal_tier(+SignalTerm, -Tier, -Source)
%%
%% Tier is one of: intent | declared_data | inferred_data
%% Source is metadata describing where this signal typically comes
%% from (caller / manifest / relation_policy / calibration /
%% static_analysis / inferred_default). The source is not consumed
%% in Phase 1; Phase 2's confidence-weighting layer and the trace
%% renderer may use it.
%%
%% Pattern matching is by clause head — value-specific dispatch for
%% functors whose tier depends on value (csr_available) is expressed
%% as separate clauses rather than a single clause with a guard.
%%
%% Adding a new signal type means adding a clause here.
%% ============================================================

%% Intent signals
signal_tier(kernel_mode(_),                  intent, caller).
signal_tier(strategy(_),                     intent, manifest).
signal_tier(force_search_algorithm(_),       intent, caller).

%% Declared-data signals
signal_tier(csr_path(_),                     declared_data, caller).
signal_tier(csr_available(true),             declared_data, caller).
signal_tier(cardinality(_),                  declared_data, relation_policy).
signal_tier(determinism(_),                  declared_data, relation_policy).
signal_tier(unique(_),                       declared_data, relation_policy).
signal_tier(graph_mutability(_),             declared_data, manifest).
signal_tier(heuristic_predicate_available(_), declared_data, caller).
signal_tier(heuristic_predicate(_),          declared_data, caller).
signal_tier(b_eff(_),                        declared_data, calibration).
signal_tier(branching_d(_),                  declared_data, calibration).
signal_tier(contraction_r(_),                declared_data, calibration).

%% Inferred-data signals
%%
%% csr_available(false) is the default inference when no csr_path was
%% supplied; csr_buildable(_) and the second variants of query_pattern /
%% query_frequency come from static analysis when not declared in
%% manifest.
signal_tier(csr_available(false),            inferred_data, inferred_default).
signal_tier(csr_buildable(_),                inferred_data, static_analysis).

%% Dual-source signals — query_pattern and query_frequency can be
%% either declared (manifest) or inferred (static analysis). Phase 1
%% defaults them to declared_data; build_workload_signals/2 in
%% recurrence_inputs.pl (Phase 5) can wrap inferred occurrences
%% differently if it needs the inferred tier. For now, both arrive
%% from manifest-style sources.
signal_tier(query_pattern(_),                declared_data, manifest).
signal_tier(query_frequency(_),              declared_data, manifest).

%% apply_cost_model(+Recurrence, +DataSignals, -CostModelChoice)
%
% Phase 2: evaluate every cost_model_rule/6 against (Recurrence,
% DataSignals); each firing rule contributes a weighted score to its
% chosen strategy; the strategy with the highest cumulative score
% wins; tiebreak by deciding-rule priority, then lexicographic rule
% name.
%
% Returns cost_model_choice(Strategy, Score, DecidingRule) per SPEC.
% Adjustments for the deciding rule are looked up separately by
% the trace-builder in select_evaluation_strategy/3 (see
% rule_adjustments/2).
apply_cost_model(Recurrence, DataSignals, CostModelChoice) :-
    findall(firing(RuleName, Priority, WeightedScore, Strategy),
            ( cost_model_rule(RuleName, Priority, RawScore, Strategy,
                              Preconditions, AdditionalCheck),
              rule_fires(Recurrence, DataSignals, Preconditions, AdditionalCheck),
              weighted_score_for(RawScore, Preconditions, WeightedScore)
            ),
            Firings),
    pick_winning_strategy(Firings, Strategy, CumulativeScore, DecidingRule),
    CostModelChoice = cost_model_choice(Strategy, CumulativeScore, DecidingRule).

%% rule_fires(+Recurrence, +DataSignals, +Preconditions, +AdditionalCheck)
%
% A rule fires when every signal in Preconditions is present in
% DataSignals (member-check; tier-irrelevant per SPEC) AND the
% AdditionalCheck goal succeeds.
rule_fires(Recurrence, DataSignals, Preconditions, AdditionalCheck) :-
    forall(member(Signal, Preconditions),
           member(Signal, DataSignals)),
    additional_check(AdditionalCheck, Recurrence, DataSignals).

%% additional_check(+CheckTerm, +Recurrence, +DataSignals)
%
% Hook for non-member-list checks specific rules need.
additional_check(true, _Recurrence, _DataSignals).
additional_check(no_contradicting_cardinality, _Recurrence, DataSignals) :-
    \+ member(cardinality(large), DataSignals),
    \+ member(cardinality(medium), DataSignals).
additional_check(admits_strategy(Strategy), Recurrence, _DataSignals) :-
    admissible_strategies(Recurrence, Admissible),
    member(Strategy, Admissible).

%% weighted_score_for(+RawScore, +Preconditions, -WeightedScore)
%
% weighted_score = raw_score × (sum of per-signal weights) / N
% where each signal contributes 1.0 if declared, 0.8 if inferred.
% Special case: N=0 (no matching signals — e.g. default_fallback)
% yields weighted_score = raw_score (treat as weight 1.0).
weighted_score_for(RawScore, [], RawScore) :- !.
weighted_score_for(RawScore, Preconditions, WeightedScore) :-
    maplist(signal_weight, Preconditions, Weights),
    sum_list(Weights, WeightSum),
    length(Preconditions, N),
    WeightedScore is RawScore * WeightSum / N.

%% signal_weight(+Signal, -Weight)
%
% Look up the signal's tier in the dispatch table and return the
% per-tier weight constant. Intent signals don't normally appear
% in cost-model preconditions, but if they do they're weighted 1.0
% (user-stated intent is high-confidence).
signal_weight(Signal, Weight) :-
    (   signal_tier(Signal, declared_data, _)
    ->  Weight = 1.0
    ;   signal_tier(Signal, inferred_data, _)
    ->  Weight = 0.8
    ;   signal_tier(Signal, intent, _)
    ->  Weight = 1.0
    ;   %% Unknown signal — treat as inferred (lower confidence)
        Weight = 0.8
    ).

%% pick_winning_strategy(+Firings, -Strategy, -CumulativeScore, -DecidingRule)
%
% Sum weighted scores per strategy, pick the highest. Tiebreak: the
% strategy whose highest-priority firing rule has the higher priority
% wins; on priority-tie, lexicographic rule name.
%
% Firings is a list of firing(RuleName, Priority, WeightedScore, Strategy).
pick_winning_strategy([], _, _, _) :-
    %% Should never happen — default_fallback always fires. If we
    %% reach here, there's a config bug.
    throw(error(no_firing_cost_model_rules,
                context(apply_cost_model/3, 'default_fallback should always fire'))).
pick_winning_strategy(Firings, WinningStrategy, WinningScore, DecidingRule) :-
    %% Group firings by strategy, sum scores.
    findall(Strategy, member(firing(_, _, _, Strategy), Firings), AllStrategies),
    sort(AllStrategies, UniqueStrategies),
    findall(strategy_total(S, TotalScore, BestRule, BestPriority),
            ( member(S, UniqueStrategies),
              findall(WeightedScore-RuleName-Priority,
                      member(firing(RuleName, Priority, WeightedScore, S), Firings),
                      Contributions),
              foldl(sum_contribution, Contributions, 0, TotalScore),
              best_rule_for_strategy(Contributions, BestRule, BestPriority)
            ),
            StrategyTotals),
    %% Sort by: score desc, then priority desc, then name asc.
    predsort(strategy_total_compare, StrategyTotals, Sorted),
    Sorted = [strategy_total(WinningStrategy, WinningScore, DecidingRule, _) | _].

sum_contribution(WeightedScore-_-_, Acc, NewAcc) :-
    NewAcc is Acc + WeightedScore.

best_rule_for_strategy(Contributions, BestRule, BestPriority) :-
    %% Sort contributions by priority desc, then name asc; take first.
    predsort(contribution_compare, Contributions, [_-BestRule-BestPriority | _]).

%% predsort/3 comparator for contributions: higher priority first,
%% then lexicographic name.
contribution_compare(Order, _-Name1-P1, _-Name2-P2) :-
    (   P1 > P2 -> Order = (<)
    ;   P1 < P2 -> Order = (>)
    ;   compare(Order0, Name1, Name2),
        ( Order0 == (=) -> Order = (=) ; Order = Order0 )
    ).

%% predsort/3 comparator for strategy_totals: higher score first,
%% then higher priority of deciding rule, then lex rule name.
strategy_total_compare(Order,
        strategy_total(_, S1, R1, P1),
        strategy_total(_, S2, R2, P2)) :-
    (   S1 > S2 -> Order = (<)
    ;   S1 < S2 -> Order = (>)
    ;   P1 > P2 -> Order = (<)
    ;   P1 < P2 -> Order = (>)
    ;   compare(Order0, R1, R2),
        ( Order0 == (=) -> Order = (=) ; Order = Order0 )
    ).

%% resolve_against_intent(+IntentSignals, +CostModelChoice, +Recurrence,
%%                       -Strategy, -Trace)
%
% Phase 0 stub: returns the cost-model's chosen strategy unchanged
% with a single-step trace recording that resolution was skipped.
% Phase 3 implements the six-step hierarchy.
resolve_against_intent(_IntentSignals,
                       cost_model_choice(Strategy, _Score, _Rule),
                       _Recurrence,
                       Strategy,
                       trace([step(no_intent, applied,
                                   [phase_0_stub_skipped_resolution])])).

%% render_trace_for_stderr(+Trace, -Lines)
%
% Phase 0 stub: returns an empty list (no rendering).
% Phase 4 implements the real renderer.
render_trace_for_stderr(_Trace, []).

%% format_trace_for_comment(+Trace, +CommentPrefix, -CommentString)
%
% Phase 0 stub: returns an empty string.
% Phase 4 implements the real comment renderer with per-line
% CommentPrefix application.
format_trace_for_comment(_Trace, _CommentPrefix, "").

%% admissible_strategies(+Recurrence, -StrategyList)
%
% Phase 2: returns the strategies admissible for this recurrence
% based on two conditions (per SPEC §Admissibility):
%
%   1. KernelKind permits the strategy — read from
%      kernel_kind_strategies/2 static table.
%   2. Termination guarantee permits the strategy. For per_query(_)
%      strategies, no convergence guarantee is required from the
%      recurrence (per-query termination is the visited-set's job).
%      For fixed_point(_) strategies:
%        - value_domain(combinatorial) — admissible regardless of
%          monotonicity (finite-state iteration always halts; the
%          semantic meaning depends on monotonicity but termination
%          does not — see philosophy doc).
%        - value_domain(numeric) — admissible iff
%          numeric_contraction_rate(R) with R < 1.0.
%
% NOTE: the monotone(false) cross-class restriction is NOT applied
% here. It depends on intent context (which class the caller is
% asking for) and is applied in Phase 3's conflict-resolution
% machine.
admissible_strategies(Recurrence, StrategyList) :-
    Recurrence = recurrence(KernelKind, _Pred, Properties),
    kernel_kind_strategies(KernelKind, KernelPermitted),
    include(strategy_termination_ok(Properties), KernelPermitted, StrategyList).

%% strategy_termination_ok(+Properties, +Strategy)
%
% True if the strategy's termination guarantee is satisfied by the
% recurrence's properties.
strategy_termination_ok(_Properties, strategy(per_query(_))) :- !.
strategy_termination_ok(Properties, strategy(fixed_point(_))) :- !,
    (   memberchk(value_domain(combinatorial), Properties)
    ->  true
    ;   memberchk(value_domain(numeric), Properties),
        memberchk(numeric_contraction_rate(R), Properties),
        number(R), R < 1.0
    ).
strategy_termination_ok(_Properties, strategy(cached)) :- !.
strategy_termination_ok(_Properties, strategy(hybrid(_))) :- !.

%% ============================================================
%% Kernel-kind-to-strategies static table
%%
%% Maps each KernelKind (from recursive_kernel_detection.pl's
%% detector registry) to the strategies its kernel template permits.
%%
%% MAINTENANCE NOTE: when a new detector is added to
%% recursive_kernel_detection.pl, the corresponding entry MUST be
%% added here. This dual-source-of-truth is documented in the
%% implementation plan §Phase 3 hidden-scope notes.
%%
%% See SPEC §Admissibility for the per-kernel rationale.
%% ============================================================

%% bidirectional_ancestor — per-query-only by template design
%% (single-pair source+target lookup; no fixed_point provision)
kernel_kind_strategies(bidirectional_ancestor,
    [strategy(per_query(bidirectional)), strategy(per_query(astar))]).

%% category_ancestor — combinatorial; admits unidirectional and
%% upgrade-to-bidirectional via maybe_upgrade_bidirectional/2
kernel_kind_strategies(category_ancestor,
    [strategy(per_query(unidirectional)), strategy(per_query(bidirectional))]).

%% transitive_closure2 — combinatorial; fixed_point structurally
%% permitted (Datalog-style); per_query upgrades available
kernel_kind_strategies(transitive_closure2,
    [strategy(per_query(unidirectional)),
     strategy(per_query(bidirectional)),
     strategy(fixed_point(semi_naive))]).

%% transitive_distance3 — combinatorial BFS-with-distance
kernel_kind_strategies(transitive_distance3,
    [strategy(per_query(unidirectional)), strategy(per_query(bidirectional))]).

%% transitive_parent_distance4 — combinatorial; per-query only
kernel_kind_strategies(transitive_parent_distance4,
    [strategy(per_query(unidirectional))]).

%% transitive_step_parent_distance5 — combinatorial; per-query only
kernel_kind_strategies(transitive_step_parent_distance5,
    [strategy(per_query(unidirectional))]).

%% weighted_shortest_path3 — numeric; Dijkstra per-query (template
%% requires non-negative weights). fixed_point not in registry yet
%% (would need Bellman-Ford or value-iteration template — see
%% philosophy doc gap notes).
kernel_kind_strategies(weighted_shortest_path3,
    [strategy(per_query(dijkstra))]).

%% astar_shortest_path4 — numeric; A* per-query (heuristic-driven)
kernel_kind_strategies(astar_shortest_path4,
    [strategy(per_query(astar))]).

%% ============================================================
%% Cost-model rules (Phase 2)
%%
%% Each rule:
%%   cost_model_rule(RuleName, Priority, RawScore, Strategy,
%%                   Preconditions, AdditionalCheck)
%%
%% Preconditions: list of signal terms that must ALL be in DataSignals
%%   (tier-irrelevant per SPEC — confidence weighting handles tier).
%% AdditionalCheck: extra goal (one of true / no_contradicting_cardinality /
%%   admits_strategy(_)). Most rules use 'true'.
%%
%% Adjustments (for rules that need them) are looked up via
%% rule_adjustments/2.
%%
%% Adding a rule means adding a clause here (NOT multifile —
%% load-order-dependent tiebreaking is fragile per SPEC).
%% ============================================================

cost_model_rule(prefer_bidirectional_csr_present, 100, 3,
    strategy(per_query(bidirectional)),
    [csr_available(true), query_pattern(single_pair), cardinality(large)],
    true).

cost_model_rule(prefer_bidirectional_csr_buildable, 90, 2,
    strategy(per_query(bidirectional)),
    [csr_buildable(true), cardinality(large), query_frequency(high)],
    true).

cost_model_rule(prefer_unidirectional_no_csr, 80, 2,
    strategy(per_query(unidirectional)),
    [csr_available(false), csr_buildable(false)],
    true).

cost_model_rule(prefer_unidirectional_small, 70, 1,
    strategy(per_query(unidirectional)),
    [cardinality(small)],
    no_contradicting_cardinality).

cost_model_rule(prefer_astar_heuristic_available, 60, 1,
    strategy(per_query(astar)),
    [heuristic_predicate_available(true)],
    admits_strategy(strategy(per_query(astar)))).

cost_model_rule(default_fallback, 1, 0,
    strategy(per_query(unidirectional)),
    [],
    true).

%% rule_adjustments(+RuleName, -Adjustments)
%
% Returns the list of adjustment atoms the rule's preferred strategy
% requires as side-effects (e.g. build CSR at compile time before
% the kernel call). Rules without adjustments return [].
rule_adjustments(prefer_bidirectional_csr_buildable, [build_csr_at_compile_time]).
rule_adjustments(_, []) :- !.

%% strategy_pretty(+Strategy, -String)
%
% Phase 0 stub: returns the strategy term rendered with format_to_atom.
% Phase 4 implements the pretty-printer alongside the renderers.
strategy_pretty(Strategy, String) :-
    format(string(String), "~w", [Strategy]).

%% ============================================================
%% Type-check helpers
%%
%% These ARE implemented for real in Phase 0 — Phase 1+ helpers
%% rely on them at every call site, and the smoke test exercises
%% them directly.
%% ============================================================

%% valid_recurrence(+Recurrence) is det.
%
% A valid Recurrence is `recurrence(KernelKind, Pred/Arity, Properties)`
% where KernelKind is an atom, Pred/Arity is a predicate indicator,
% and Properties is a list.
%
% Throws type_error or domain_error on malformed input; succeeds
% silently on valid input.
valid_recurrence(recurrence(KernelKind, Pred/Arity, Properties)) :- !,
    must_be(atom, KernelKind),
    must_be(atom, Pred),
    must_be(nonneg, Arity),
    must_be(list, Properties).
valid_recurrence(Other) :-
    type_error(recurrence_term, Other).

%% valid_workload(+Workload) is det.
%
% A valid Workload is a list of signal terms. Phase 0 does not
% inspect individual signals; Phase 1 adds the dispatch-table-based
% per-signal validation.
valid_workload(Workload) :-
    must_be(list, Workload).

%% valid_strategy(+Strategy) is det.
%
% A valid Strategy is `strategy(Mode)` where Mode is one of:
%
%   per_query(SearchAlgo)
%   fixed_point(IterAlgo)
%   cached
%   hybrid(_)
%
% IMPORTANT: this checks the OUTER `strategy/1` wrapper, not just
% the inner Mode. A bare Mode term (e.g. `per_query(bidirectional)`)
% without the outer wrapper is rejected — that was an explicit
% Phase 0 success criterion flagged in review.
valid_strategy(strategy(Mode)) :- !,
    valid_mode(Mode).
valid_strategy(Other) :-
    type_error(strategy_wrapper, Other).

valid_mode(per_query(Algo)) :- !,
    must_be(atom, Algo).
valid_mode(fixed_point(Algo)) :- !,
    must_be(atom, Algo).
valid_mode(cached) :- !.
valid_mode(hybrid(Components)) :- !,
    must_be(list, Components).
valid_mode(Other) :-
    domain_error(strategy_mode, Other).
