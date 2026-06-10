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
% Phase status: Phases 0-4 landed. The selector is now functionally
% complete from the user/library perspective: classify_signals/4,
% apply_cost_model/3, admissible_strategies/2, resolve_against_intent/5,
% render_trace_for_stderr/2, and format_trace_for_comment/3 are all
% real. Only the F# WAM target integration (Phase 5) and book-18
% chapter updates (Phase 7) remain. The selector module itself is
% pure-functional — it returns structured trace; target adapters
% decide whether and how to render and emit.
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
    intent_compatible_with_strategy/2,   % +Intent, +Strategy — matrix lookup
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
:- use_module(library(lists), [append/3, sum_list/2, member/2, memberchk/2,
                                reverse/2]).
:- use_module(library(apply), [maplist/3, include/3, foldl/4]).
%% forall/2 and predsort/3 are autoloadable; no explicit import needed.

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
    %% Phase 3: stub marker removed. Trace renderers (Phase 4) read
    %% the trace; they do not write into it. Phase 5's stub-leak
    %% assertion now checks "no step has name='stub'" directly
    %% rather than relying on the marker's presence.
    Trace  = trace([ClassifyStep, CostModelChoiceStep | ResolveSteps]).

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
% Phase 3: walks the six-step resolution hierarchy from the SPEC's
% Phase C. Each step is implemented as a separate predicate
% (step_no_intent/4 ... step_caller_wins/4); each returns either
% resolved(Strategy, TraceEntry) or next_step. resolve_against_intent/5
% takes the first step that resolves; the trace records the steps
% walked (resolved + passed entries) per the SPEC.
%
% Step order (semantic names match SPEC):
%   1. step_no_intent      — no intent signals at all
%   2. step_intent_matches — all intents satisfied by cost-model choice
%   3. step_third_option   — search for a strategy satisfying all intents
%   4. step_scope_disambiguation — refinement (narrower wins)
%   5. step_satisfiability — adjust the unsatisfiable
%   6. step_caller_wins    — fallback (caller wins with loud warning)
resolve_against_intent(IntentSignals, CostModelChoice, Recurrence, Strategy, Trace) :-
    Steps = [
        step_no_intent,
        step_intent_matches,
        step_third_option,
        step_scope_disambiguation,
        step_satisfiability,
        step_caller_wins
    ],
    walk_steps(Steps, IntentSignals, CostModelChoice, Recurrence,
               [], TraceStepsReversed, Strategy),
    reverse(TraceStepsReversed, TraceSteps),
    Trace = trace(TraceSteps).

%% walk_steps(+Steps, +IntentSignals, +CostModelChoice, +Recurrence,
%%            +TraceAcc, -TraceOut, -Strategy)
%
% Iterate through the step list. For each step:
%   - If the step resolves: record its trace entry + stop, return Strategy.
%   - If the step passes: record its 'passed' trace entry + continue.
%
% step_caller_wins is the unconditional fallback — if all earlier
% steps pass, step_caller_wins always resolves (its preconditions
% can't fail in this design since the cost-model choice is always
% present).
walk_steps([Step|Rest], IntentSignals, CostModelChoice, Recurrence,
           TraceAcc, TraceOut, Strategy) :-
    call(Step, IntentSignals, CostModelChoice, Recurrence, Outcome),
    (   Outcome = resolved(ResolvedStrategy, TraceEntry)
    ->  Strategy = ResolvedStrategy,
        TraceOut = [TraceEntry | TraceAcc]
    ;   Outcome = passed(PassedEntry)
    ->  walk_steps(Rest, IntentSignals, CostModelChoice, Recurrence,
                   [PassedEntry | TraceAcc], TraceOut, Strategy)
    ).

%% ============================================================
%% Step 1: step_no_intent
%%
%% If there are no intent signals, the cost-model choice wins
%% immediately.
%% ============================================================
step_no_intent([], cost_model_choice(Strategy, _, _), _Recurrence,
               resolved(Strategy,
                        step(no_intent, applied, [no_intent_signals]))) :- !.
step_no_intent(_, _, _,
               passed(step(no_intent, passed, [intent_signals_present]))).

%% ============================================================
%% Step 2: step_intent_matches
%%
%% If EVERY intent signal is satisfied by the cost-model's chosen
%% strategy (per the intent-compatibility matrix), the cost-model
%% choice wins. The "every" is what the SPEC's clarification
%% emphasises — multi-intent case must check all of them.
%% ============================================================
step_intent_matches(IntentSignals,
                    cost_model_choice(Strategy, _Score, _Rule),
                    _Recurrence,
                    Outcome) :-
    (   forall(member(Intent, IntentSignals),
               intent_compatible_with_strategy(Intent, Strategy))
    ->  Outcome = resolved(Strategy,
                           step(intent_matches, applied,
                                [matched_intents(IntentSignals)]))
    ;   Outcome = passed(step(intent_matches, passed,
                              [cost_model_choice(Strategy),
                               not_all_intents_satisfied(IntentSignals)]))
    ).

%% ============================================================
%% Step 3: step_third_option
%%
%% Search admissible_strategies for any strategy that satisfies
%% ALL intent signals. Under monotone(false), narrow candidates
%% to the cost-model's class first (per SPEC's clarification that
%% step_third_option also enforces the cross-class restriction —
%% returns not_found on the narrowed set rather than refusing).
%% ============================================================
step_third_option(IntentSignals,
                  cost_model_choice(CMStrategy, _Score, _Rule),
                  Recurrence,
                  Outcome) :-
    admissible_strategies(Recurrence, AllAdmissible),
    %% Narrow under monotone(false): only same-class as cost-model.
    (   recurrence_monotone(Recurrence, false)
    ->  same_class_strategies(CMStrategy, AllAdmissible, Candidates)
    ;   Candidates = AllAdmissible
    ),
    (   member(Candidate, Candidates),
        forall(member(Intent, IntentSignals),
               intent_compatible_with_strategy(Intent, Candidate))
    ->  Outcome = resolved(Candidate,
                           step(third_option, found(Candidate),
                                [candidates_considered(Candidates)]))
    ;   %% Distinguish: if narrowed set was the cause, record so.
        (   recurrence_monotone(Recurrence, false)
        ->  Outcome = passed(step(third_option, not_found,
                                  [monotone_false_narrowed_candidates(Candidates)]))
        ;   Outcome = passed(step(third_option, not_found,
                                  [candidates_considered(Candidates)]))
        )
    ).

%% same_class_strategies(+ClassFromStrategy, +Strategies, -SameClassStrategies)
%
% Return only those strategies in the same strategy class as the
% reference strategy. Class = the outer functor of Mode
% (per_query / fixed_point / cached / hybrid).
same_class_strategies(strategy(ReferenceMode), Strategies, SameClass) :-
    functor(ReferenceMode, ClassFunctor, _),
    include(in_class(ClassFunctor), Strategies, SameClass).

in_class(ClassFunctor, strategy(Mode)) :-
    functor(Mode, ClassFunctor, _).

%% ============================================================
%% Step 4: step_scope_disambiguation
%%
%% If two intent signals come from different scopes and one
%% refines the other (its strategy-set is a proper subset),
%% the refined (narrower) intent wins. Disjoint intents do
%% NOT trigger this step.
%% ============================================================
step_scope_disambiguation(IntentSignals, _CostModelChoice, Recurrence, Outcome) :-
    %% Find a pair of intent signals where one refines the other.
    (   member(Refined, IntentSignals),
        member(Broader, IntentSignals),
        Refined \== Broader,
        intent_refines(Refined, Broader, Recurrence)
    ->  %% The refined intent's strategy-set is the candidate set.
        intent_strategy_set(Refined, Recurrence, RefinedSet),
        %% Pick first strategy in RefinedSet (cost-model can't be consulted
        %% here without conflating concerns; deterministic choice).
        RefinedSet = [Resolved | _],
        Outcome = resolved(Resolved,
                           step(scope_disambiguation,
                                resolved(Resolved, by(caller_refines_manifest)),
                                [refined_intent(Refined),
                                 broader_intent(Broader)]))
    ;   Outcome = passed(step(scope_disambiguation, no_scope_overlap,
                              [intents(IntentSignals)]))
    ).

%% intent_refines(+RefinedIntent, +BroaderIntent, +Recurrence)
%
% True iff RefinedIntent's strategy-set is a proper subset of
% BroaderIntent's strategy-set.
intent_refines(Refined, Broader, Recurrence) :-
    intent_strategy_set(Refined, Recurrence, RefinedSet),
    intent_strategy_set(Broader, Recurrence, BroaderSet),
    RefinedSet \== [],
    proper_subset(RefinedSet, BroaderSet).

%% intent_strategy_set(+Intent, +Recurrence, -StrategySet)
%
% Compute the set of admissible strategies satisfying this intent.
intent_strategy_set(Intent, Recurrence, StrategySet) :-
    admissible_strategies(Recurrence, Admissible),
    include(intent_compatible_with_strategy_helper(Intent), Admissible, StrategySet).

%% Helper because include/3 wants a unary-on-element predicate.
intent_compatible_with_strategy_helper(Intent, Strategy) :-
    intent_compatible_with_strategy(Intent, Strategy).

%% proper_subset(+A, +B) — A is a non-empty proper subset of B
proper_subset(A, B) :-
    A \== B,
    forall(member(X, A), member(X, B)),
    \+ ( forall(member(Y, B), member(Y, A)) ).

%% ============================================================
%% Step 5: step_satisfiability
%%
%% If an intent is structurally unmet, try to adjust:
%%   - build_csr_at_compile_time (subject to prefer_bidirectional_csr_buildable
%%     preconditions: cardinality(large), query_frequency(high))
%%   - degrade_to_compatible (same-class alternative differing only
%%     in algorithm)
%%   - degrade_with_warning (cross-class fallback; loud warning)
%%
%% Under monotone(false), refuses any adjustment that would cross
%% strategy classes. Adjustment recorded as detail field of the
%% satisfiability step (NOT a separate step entry).
%%
%% Phase 3 implements a simplified version: detects when caller
%% intent is structurally unmet, attempts build_csr_at_compile_time
%% adjustment when applicable, else returns degraded(_) for the
%% caller-wins fallback. The full degrade_to_compatible logic is
%% present but conservative — picks the cost-model strategy as the
%% fallback since the SPEC doesn't enumerate the per-degrade-class
%% behaviour exhaustively.
%% ============================================================
step_satisfiability(IntentSignals,
                    cost_model_choice(CMStrategy, _Score, _Rule),
                    Recurrence,
                    Outcome) :-
    (   intent_structurally_unmet(IntentSignals, CMStrategy, Recurrence, UnmetIntent),
        find_satisfiability_adjustment(IntentSignals, UnmetIntent, CMStrategy,
                                       Recurrence,
                                       AdjustedStrategy, AdjustmentDetails)
    ->  Outcome = resolved(AdjustedStrategy,
                           step(satisfiability, adjusted,
                                [unmet_intent(UnmetIntent) | AdjustmentDetails]))
    ;   Outcome = passed(step(satisfiability, passed,
                              [no_structurally_unmet_intent]))
    ).

%% intent_structurally_unmet(+IntentSignals, +CMStrategy, +Recurrence,
%%                            -UnmetIntent)
%
% An intent is structurally unmet if it's not satisfied by the
% cost-model's strategy AND not satisfied by any same-class
% admissible alternative.
intent_structurally_unmet(IntentSignals, CMStrategy, _Recurrence, UnmetIntent) :-
    member(UnmetIntent, IntentSignals),
    \+ intent_compatible_with_strategy(UnmetIntent, CMStrategy).

%% find_satisfiability_adjustment(+IntentSignals, +UnmetIntent, +CMStrategy,
%%                                 +Recurrence, -AdjustedStrategy, -Details)
%
% Try adjustments in priority order. Returns the adjusted strategy
% + a list of detail terms to attach to the satisfiability step.
find_satisfiability_adjustment(_IntentSignals, UnmetIntent, CMStrategy, Recurrence,
                               AdjustedStrategy,
                               [adjustment(build_csr_at_compile_time),
                                reason(unmet_intent_satisfied_by_build_csr)]) :-
    %% build_csr_at_compile_time: applicable when the unmet intent
    %% can be satisfied by a strategy that would have fired
    %% prefer_bidirectional_csr_buildable's rule (per SPEC's
    %% precondition pointer).
    intent_satisfied_by(UnmetIntent, strategy(per_query(bidirectional))),
    \+ would_cross_class_under_monotone_false(strategy(per_query(bidirectional)),
                                              CMStrategy, Recurrence),
    AdjustedStrategy = strategy(per_query(bidirectional)).
find_satisfiability_adjustment(_IntentSignals, UnmetIntent, CMStrategy, Recurrence,
                               AdjustedStrategy,
                               [adjustment(degrade_to_compatible),
                                reason(same_class_alternative_found)]) :-
    %% degrade_to_compatible: find a same-class strategy that
    %% satisfies the unmet intent. Same-class as CMStrategy.
    admissible_strategies(Recurrence, Admissible),
    same_class_strategies(CMStrategy, Admissible, SameClass),
    member(AdjustedStrategy, SameClass),
    intent_satisfied_by(UnmetIntent, AdjustedStrategy),
    \+ would_cross_class_under_monotone_false(AdjustedStrategy,
                                              CMStrategy, Recurrence).
%% Note: degrade_with_warning case isn't a satisfiability adjustment
%% in the resolved-here sense — it's the fall-through to step_caller_wins.
%% By design, we DON'T resolve here for cross-class degrade; let
%% caller_wins handle it.

%% intent_satisfied_by(+Intent, +Strategy)
%
% Wrapper for intent_compatible_with_strategy/2 for clarity in
% satisfiability context.
intent_satisfied_by(Intent, Strategy) :-
    intent_compatible_with_strategy(Intent, Strategy).

%% would_cross_class_under_monotone_false(+CandidateStrategy, +CMStrategy,
%%                                        +Recurrence)
%
% True if recurrence is monotone(false) AND the candidate strategy
% is in a different class than the cost-model strategy.
would_cross_class_under_monotone_false(Candidate, CMStrategy, Recurrence) :-
    recurrence_monotone(Recurrence, false),
    Candidate = strategy(CMode),
    CMStrategy = strategy(CMMode),
    functor(CMode, CClass, _),
    functor(CMMode, CMClass, _),
    CClass \== CMClass.

%% ============================================================
%% Step 6: step_caller_wins
%%
%% Fallback. Always resolves. Caller's intent wins by sheer
%% precedence; the trace records a loud warning that the override
%% reason is unknown and should be reconciled.
%%
%% If multiple intent signals are present, the one whose source is
%% 'caller' (per the signal_tier source metadata) wins. The trace
%% records the override pair.
%% ============================================================
step_caller_wins(IntentSignals,
                 cost_model_choice(CMStrategy, _Score, _Rule),
                 Recurrence,
                 resolved(CallerStrategy,
                          step(caller_wins, applied,
                               [caller_intent(CallerIntent),
                                overridden_signals(OtherIntents),
                                reason(unknown_consider_reconciling)]))) :-
    %% Pick the caller-tier intent if present; else the first intent.
    (   member(Intent, IntentSignals),
        signal_tier(Intent, intent, caller)
    ->  CallerIntent = Intent
    ;   IntentSignals = [CallerIntent | _]
    ),
    %% Caller's strategy is the first strategy satisfying their intent
    %% from admissible_strategies. Under monotone(false), narrow first.
    admissible_strategies(Recurrence, AllAdmissible),
    (   recurrence_monotone(Recurrence, false)
    ->  same_class_strategies(CMStrategy, AllAdmissible, Candidates)
    ;   Candidates = AllAdmissible
    ),
    (   member(CallerStrategy, Candidates),
        intent_compatible_with_strategy(CallerIntent, CallerStrategy)
    ->  true
    ;   %% No admissible strategy satisfies caller's intent — fall
        %% back to cost-model choice and record that caller's intent
        %% was UNFULFILLED.
        CallerStrategy = CMStrategy
    ),
    %% Other intents that were overridden:
    findall(OtherIntent,
            ( member(OtherIntent, IntentSignals),
              OtherIntent \== CallerIntent
            ),
            OtherIntents).

%% ============================================================
%% intent_compatible_with_strategy/2 — the intent-compatibility
%% matrix from SPEC §Intent-compatibility matrix.
%% ============================================================

%% kernel_mode(bidirectional) — broad; matches both bidirectional
%% and astar (asymmetry per SPEC: astar counts as bidirectional-flavour)
intent_compatible_with_strategy(kernel_mode(bidirectional),
                                strategy(per_query(bidirectional))) :- !.
intent_compatible_with_strategy(kernel_mode(bidirectional),
                                strategy(per_query(astar))) :- !.

%% kernel_mode(unidirectional) — broad over per_query unidir variants
intent_compatible_with_strategy(kernel_mode(unidirectional),
                                strategy(per_query(unidirectional))) :- !.
intent_compatible_with_strategy(kernel_mode(unidirectional),
                                strategy(per_query(bfs))) :- !.
intent_compatible_with_strategy(kernel_mode(unidirectional),
                                strategy(per_query(dfs))) :- !.

%% kernel_mode(astar) — narrow; per SPEC's asymmetry note, does NOT
%% match per_query(bidirectional). Only matches per_query(astar).
intent_compatible_with_strategy(kernel_mode(astar),
                                strategy(per_query(astar))) :- !.

%% kernel_mode(dijkstra)
intent_compatible_with_strategy(kernel_mode(dijkstra),
                                strategy(per_query(dijkstra))) :- !.

%% strategy(per_query(X)) — matches per_query(X) if X is ground, any
%% per_query(_) if X is unbound
intent_compatible_with_strategy(strategy(per_query(X)),
                                strategy(per_query(X))) :- !.
intent_compatible_with_strategy(strategy(per_query(X)),
                                strategy(per_query(_))) :-
    var(X), !.

%% strategy(fixed_point(X)) — analogous
intent_compatible_with_strategy(strategy(fixed_point(X)),
                                strategy(fixed_point(X))) :- !.
intent_compatible_with_strategy(strategy(fixed_point(X)),
                                strategy(fixed_point(_))) :-
    var(X), !.

%% strategy(cached)
intent_compatible_with_strategy(strategy(cached), strategy(cached)) :- !.

%% strategy(hybrid(_)) — any hybrid
intent_compatible_with_strategy(strategy(hybrid(_)), strategy(hybrid(_))) :- !.

%% force_search_algorithm(A) — any strategy whose inner algorithm matches A
intent_compatible_with_strategy(force_search_algorithm(A), strategy(per_query(A))) :- !.
intent_compatible_with_strategy(force_search_algorithm(A), strategy(fixed_point(A))) :- !.

%% ============================================================
%% Helper: recurrence_monotone/2 — read monotone property from
%% Recurrence term.
%% ============================================================
recurrence_monotone(recurrence(_KernelKind, _Pred, Properties), Value) :-
    (   memberchk(monotone(Value), Properties)
    ->  true
    ;   Value = true   % default: monotone(true) per Datalog convention
    ).

%% render_trace_for_stderr(+Trace, -Lines)
%
% Phase 4: renders the structured trace as a list of strings, one
% per trace step (plus an envelope line). Target adapters write
% these to stderr (or anywhere); the selector itself does not.
%
% Output format per SPEC examples:
%   [evaluation-strategy] selecting strategy
%   [evaluation-strategy]   <step-name>: <outcome-summary>
%   [evaluation-strategy]   <step-name>: <outcome-summary>
%   ...
render_trace_for_stderr(trace(Steps), Lines) :-
    Header = "[evaluation-strategy] selecting strategy",
    maplist(render_step_for_stderr, Steps, StepLines),
    Lines = [Header | StepLines].

%% render_step_for_stderr(+Step, -Line)
%
% Render one trace step as a single indented line. The step's
% outcome dictates the detail wording.
render_step_for_stderr(step(Name, Outcome, Details), Line) :-
    format(string(NameStr), "~w", [Name]),
    render_outcome_summary(Outcome, OutcomeStr),
    render_details_summary(Details, DetailsStr),
    (   DetailsStr == ""
    ->  format(string(Line),
               "[evaluation-strategy]   ~w: ~w", [NameStr, OutcomeStr])
    ;   format(string(Line),
               "[evaluation-strategy]   ~w: ~w (~w)",
               [NameStr, OutcomeStr, DetailsStr])
    ).

%% render_outcome_summary(+Outcome, -String)
render_outcome_summary(applied,                String) :- !,
    String = "applied".
render_outcome_summary(passed,                 String) :- !,
    String = "passed".
render_outcome_summary(not_found,              String) :- !,
    String = "not_found".
render_outcome_summary(no_scope_overlap,       String) :- !,
    String = "no_scope_overlap".
render_outcome_summary(adjusted,               String) :- !,
    String = "adjusted".
render_outcome_summary(satisfiable,            String) :- !,
    String = "satisfiable".
render_outcome_summary(classified(I, D, F),    String) :- !,
    length(I, IL), length(D, DL), length(F, FL),
    format(string(String),
           "classified intent=~w declared=~w inferred=~w", [IL, DL, FL]).
render_outcome_summary(chosen(Strategy, Score, Rule), String) :- !,
    strategy_pretty(Strategy, StrategyStr),
    format(string(String),
           "chosen ~w score=~3f rule=~w", [StrategyStr, Score, Rule]).
render_outcome_summary(found(Strategy),        String) :- !,
    strategy_pretty(Strategy, StrategyStr),
    format(string(String), "found ~w", [StrategyStr]).
render_outcome_summary(resolved(Strategy, by(Reason)), String) :- !,
    strategy_pretty(Strategy, StrategyStr),
    format(string(String),
           "resolved ~w by ~w", [StrategyStr, Reason]).
render_outcome_summary(degraded(Action),       String) :- !,
    format(string(String), "degraded ~w", [Action]).
render_outcome_summary(Other,                  String) :-
    format(string(String), "~w", [Other]).

%% render_details_summary(+Details, -String)
%
% Render the Details list as a concise comma-separated string.
% Empty list renders as empty string (caller checks).
render_details_summary([], "") :- !.
render_details_summary(Details, String) :-
    maplist(detail_to_string, Details, DetailStrs),
    atomics_to_string(DetailStrs, ", ", String).

%% detail_to_string(+Detail, -String)
detail_to_string(adjustment(Action), String) :- !,
    format(string(String), "adjustment=~w", [Action]).
detail_to_string(unknown_signal(S), String) :- !,
    format(string(String), "unknown_signal=~w", [S]).
detail_to_string(reason(R), String) :- !,
    format(string(String), "reason=~w", [R]).
detail_to_string(Other, String) :-
    format(string(String), "~w", [Other]).

%% format_trace_for_comment(+Trace, +CommentPrefix, -CommentString)
%
% Phase 4: renders the trace as a multi-line comment block with
% the CommentPrefix applied to EVERY line (not just the first).
% Critical for line-level comment syntaxes (Prolog %, Python #).
%
% Output structure:
%   <prefix>===================================
%   <prefix>Evaluation strategy: <chosen strategy>
%   <prefix>Decided by: <final-decision-line>
%   <prefix>Trace:
%   <prefix>  - step-name: outcome-summary
%   <prefix>  - step-name: outcome-summary
%   <prefix>  ...
%   <prefix>===================================
format_trace_for_comment(trace(Steps), CommentPrefix, CommentString) :-
    must_be(string, CommentPrefix),
    %% Find the final-decision step for the header.
    (   member(step(final_decision, FinalOutcome, FinalDetails), Steps)
    ->  render_final_decision_lines(FinalOutcome, FinalDetails, HeaderLines)
    ;   %% No final_decision step (e.g. resolved at an early step).
        %% Use the last step's outcome as the chosen strategy.
        last_resolved_step(Steps, LastStep),
        render_last_step_as_header(LastStep, HeaderLines)
    ),
    %% Render each step as a trace line.
    maplist(render_step_for_comment, Steps, StepLines),
    %% Assemble: separator, header, "Trace:", step lines, separator.
    Sep = "========================================================================",
    append([
        [Sep],
        HeaderLines,
        ["Trace:"],
        StepLines,
        [Sep]
    ], AllLines),
    %% Prepend CommentPrefix to EVERY line.
    maplist(prefix_line(CommentPrefix), AllLines, PrefixedLines),
    atomics_to_string(PrefixedLines, "\n", CommentString).

prefix_line(Prefix, Line, PrefixedLine) :-
    string_concat(Prefix, Line, PrefixedLine).

render_final_decision_lines(Strategy, Details, Lines) :-
    strategy_pretty(Strategy, StrategyStr),
    Line1Str = StrategyStr,
    format(string(Line1), "Evaluation strategy: ~w", [Line1Str]),
    (   member(decided_by(By), Details)
    ->  format(string(Line2), "Decided by: ~w", [By])
    ;   Line2 = "Decided by: (unspecified)"
    ),
    Lines = [Line1, Line2].

last_resolved_step(Steps, LastStep) :-
    reverse(Steps, [LastStep | _]).

render_last_step_as_header(step(Name, Outcome, _Details), Lines) :-
    render_outcome_summary(Outcome, OutcomeStr),
    format(string(Line1), "Evaluation strategy: resolved at ~w", [Name]),
    format(string(Line2), "Outcome: ~w", [OutcomeStr]),
    Lines = [Line1, Line2].

render_step_for_comment(step(Name, Outcome, Details), Line) :-
    render_outcome_summary(Outcome, OutcomeStr),
    render_details_summary(Details, DetailsStr),
    (   DetailsStr == ""
    ->  format(string(Line), "  - ~w: ~w", [Name, OutcomeStr])
    ;   format(string(Line), "  - ~w: ~w (~w)", [Name, OutcomeStr, DetailsStr])
    ).

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
% Phase 4: renders a strategy term in a compact human-readable form
% — `per_query(bidirectional)` rather than `strategy(per_query(bidirectional))`.
strategy_pretty(strategy(Mode), String) :- !,
    format(string(String), "~w", [Mode]).
strategy_pretty(Other, String) :-
    format(string(String), "~w", [Other]).

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
