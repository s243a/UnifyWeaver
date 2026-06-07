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
% Phase 0 status: SKELETON. The public API is wired (call chain
% classify -> concat -> cost-model -> resolve), but every helper
% returns a `not_yet_implemented(Phase)` sentinel. The baseline
% strategy is `per_query(unidirectional)` and the trace contains a
% single `step(stub, not_yet_implemented(phase_0), [])` entry.
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
:- use_module(library(lists), [append/3]).

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
    classify_signals(Workload, IntentSignals, DeclaredData, InferredData),
    append(DeclaredData, InferredData, DataSignals),
    apply_cost_model(Recurrence, DataSignals, CostModelChoice),
    resolve_against_intent(IntentSignals, CostModelChoice, Recurrence, Strategy, Trace0),
    %% Phase 0: prepend the stub marker so the stub-leak assertion can
    %% detect partial implementation.
    Trace0 = trace(Steps0),
    Trace  = trace([step(stub, not_yet_implemented(phase_0), []) | Steps0]).

%% classify_signals(+Workload, -IntentSignals, -DeclaredData, -InferredData)
%
% Phase 0 stub: returns three empty lists regardless of input.
% Phase 1 implements the real dispatch table.
classify_signals(_Workload, [], [], []).

%% apply_cost_model(+Recurrence, +DataSignals, -CostModelChoice)
%
% Phase 0 stub: returns the default-fallback choice unconditionally.
% Phase 2 implements the rule registry and scoring.
apply_cost_model(_Recurrence, _DataSignals,
    cost_model_choice(strategy(per_query(unidirectional)), 0, default_fallback)).

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
% Phase 0 stub: returns the single-element list with the baseline
% strategy. Phase 2 implements the kernel-kind-to-strategies table
% and termination-guarantee filtering.
admissible_strategies(_Recurrence, [strategy(per_query(unidirectional))]).

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
