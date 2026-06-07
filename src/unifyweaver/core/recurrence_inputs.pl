:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% recurrence_inputs.pl — target-agnostic helpers for constructing
% the Recurrence + Workload inputs to recurrence_evaluation_strategy:
% select_evaluation_strategy/3.
%
% This module is used by every WAM target (F#, Haskell, C, ...) to
% turn (detected kernel, caller options, manifest, relation-policy)
% into the structured inputs the strategy selector expects.
%
% TARGET-AGNOSTIC INVARIANT: this module must NOT import or reference
% any target-specific module or atom (no 'fsharp', 'haskell', 'c_target',
% etc.). Enforced by two layers:
%   - tests/core/test_recurrence_inputs_isolated.pl — load isolation
%   - tests/core/test_recurrence_inputs_grep.pl     — grep tripwire
%
% Cross-references:
%   - docs/design/RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md
%     §Helper module for input construction
%   - docs/design/RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md
%     §Phase 5

:- module(recurrence_inputs, [
    build_recurrence_term/3,             % +DetectedKernel, +ExtraProperties,
                                         %   -Recurrence
    build_workload_signals/2,            % +CallerOptions, -Workload

    %% Introspection
    kernel_kind_default_properties/2     % +KernelKind, -DefaultProperties
]).

:- use_module(library(error), [must_be/2]).
:- use_module(library(lists), [member/2, append/3, memberchk/2]).
:- use_module(library(apply), [foldl/4, maplist/3]).

%% ============================================================
%% build_recurrence_term(+DetectedKernel, +ExtraProperties,
%%                       -Recurrence)
%%
%% DetectedKernel is `recursive_kernel(KernelKind, Pred/Arity,
%% ConfigOps)` per recursive_kernel_detection.pl.
%%
%% ExtraProperties is a list of recurrence-property terms that
%% override the defaults from kernel_kind_default_properties/2.
%% Typically populated from algorithm_manifest:manifest_optimization_options/2
%% by the target adapter; pass [] when none.
%%
%% Result Recurrence is `recurrence(KernelKind, Pred/Arity, Properties)`.
%%
%% Removed-from-spec keys are dropped: directions_admissible was
%% explicitly removed from the SPEC's properties table and is now
%% derived internally by the selector. If a ConfigOps entry uses
%% that key, it is silently dropped here (this matches the SPEC's
%% "build_recurrence_term/3 drops directions_admissible" note).
%% ============================================================

build_recurrence_term(recursive_kernel(KernelKind, Pred/Arity, ConfigOps),
                      ExtraProperties,
                      recurrence(KernelKind, Pred/Arity, Properties)) :-
    must_be(atom, KernelKind),
    must_be(atom, Pred),
    must_be(nonneg, Arity),
    must_be(list, ConfigOps),
    must_be(list, ExtraProperties),
    kernel_kind_default_properties(KernelKind, Defaults),
    %% Merge: ExtraProperties take precedence over Defaults; drop
    %% removed-from-spec keys (directions_admissible).
    merge_properties(ExtraProperties, Defaults, Merged0),
    drop_removed_properties(Merged0, Properties).

%% merge_properties(+Override, +Defaults, -Merged)
%
% For each property functor in Override, replace the same-functor
% entry from Defaults. Override entries with no Defaults counterpart
% are appended.
merge_properties(Overrides, Defaults, Merged) :-
    foldl(merge_one_property, Overrides, Defaults, Merged0),
    %% Add any Override entries whose functor wasn't in Defaults.
    findall(O,
            ( member(O, Overrides),
              functor(O, F, A),
              \+ ( member(D, Merged0), functor(D, F, A) )
            ),
            NewOverrides),
    append(Merged0, NewOverrides, Merged).

merge_one_property(Override, Defaults0, Defaults) :-
    functor(Override, F, A),
    (   select_same_functor(F, A, Defaults0, _, Rest)
    ->  Defaults = [Override | Rest]
    ;   Defaults = Defaults0
    ).

select_same_functor(F, A, [P|Rest], P, Rest) :-
    functor(P, F, A), !.
select_same_functor(F, A, [P|Rest], Found, [P|NewRest]) :-
    select_same_functor(F, A, Rest, Found, NewRest).

%% drop_removed_properties(+Properties0, -Properties)
%
% Drop properties whose keys were removed from the SPEC.
drop_removed_properties(Properties0, Properties) :-
    findall(P,
            ( member(P, Properties0),
              functor(P, F, _),
              \+ removed_property(F)
            ),
            Properties).

removed_property(directions_admissible).
removed_property(expected_cardinality).  % duplicated cardinality(C); see SPEC

%% ============================================================
%% kernel_kind_default_properties(+KernelKind, -DefaultProperties)
%%
%% Per-kernel default Recurrence properties. Most kernels are
%% combinatorial monotone(true) with combinatorial loop-breaking
%% (visited-set in clause body). Numeric kernels override the
%% value_domain. Future Bellman-Ford-style kernels would override
%% with iteration_bound(N) once that property is added (see
%% philosophy doc gap notes).
%%
%% MAINTENANCE NOTE: keep these defaults aligned with the kernel
%% definitions in recursive_kernel_detection.pl. When a new
%% detector is added there, add a defaults entry here.
%% ============================================================

%% Combinatorial Datalog-shape kernels
kernel_kind_default_properties(category_ancestor,
    [value_domain(combinatorial), monotone(true),
     has_combinatorial_loop_break(true)]).
kernel_kind_default_properties(transitive_closure2,
    [value_domain(combinatorial), monotone(true),
     has_combinatorial_loop_break(true)]).
kernel_kind_default_properties(transitive_distance3,
    [value_domain(combinatorial), monotone(true),
     has_combinatorial_loop_break(true)]).
kernel_kind_default_properties(transitive_parent_distance4,
    [value_domain(combinatorial), monotone(true),
     has_combinatorial_loop_break(true)]).
kernel_kind_default_properties(transitive_step_parent_distance5,
    [value_domain(combinatorial), monotone(true),
     has_combinatorial_loop_break(true)]).

%% bidirectional_ancestor (upgrade target from category_ancestor) —
%% same defaults as category_ancestor.
kernel_kind_default_properties(bidirectional_ancestor,
    [value_domain(combinatorial), monotone(true),
     has_combinatorial_loop_break(true)]).

%% Numeric weighted kernels — value_domain(numeric). Dijkstra's
%% safety on cyclic graphs comes from priority-queue monotonicity,
%% NOT from visited-set; has_combinatorial_loop_break is omitted
%% per the SPEC's "absent vs false" distinction.
kernel_kind_default_properties(weighted_shortest_path3,
    [value_domain(numeric), monotone(true)]).
kernel_kind_default_properties(astar_shortest_path4,
    [value_domain(numeric), monotone(true)]).

%% Catch-all: any unknown KernelKind gets minimal defaults
%% (combinatorial + monotone). Safer than failing — lets the
%% selector compute admissibility from kernel_kind_strategies/2
%% (which will return [] for truly unknown kernels and throw).
kernel_kind_default_properties(_,
    [value_domain(combinatorial), monotone(true)]).

%% ============================================================
%% build_workload_signals(+CallerOptions, -Workload)
%%
%% Translates a flat list of caller-provided Options into a
%% Workload signal list per the SPEC's signal vocabulary.
%%
%% Caller-option key → workload-signal mapping:
%%   kernel_mode(M)                   → kernel_mode(M)
%%   strategy(S)                      → strategy(S)
%%   force_search_algorithm(A)        → force_search_algorithm(A)
%%   csr_path(P)                      → csr_path(P) + csr_available(true)
%%                                      (csr_path implies CSR is present)
%%   csr_available(B)                 → csr_available(B)
%%   cardinality(C)                   → cardinality(C)
%%   determinism(D)                   → determinism(D)
%%   unique(B)                        → unique(B)
%%   query_pattern(P)                 → query_pattern(P)
%%   query_frequency(F)               → query_frequency(F)
%%   graph_mutability(M)              → graph_mutability(M)
%%   heuristic_predicate(P/A)         → heuristic_predicate_available(true)
%%                                      + heuristic_predicate(P/A)
%%   heuristic_predicate_available(B) → heuristic_predicate_available(B)
%%   b_eff(F), branching_d(F),
%%       contraction_r(F)             → passed through
%%   csr_buildable(B)                 → csr_buildable(B)
%%
%% Other options (target-specific, unrecognised) are silently dropped
%% from the workload — they're not signals the selector knows about
%% and would pollute classify_signals/4 with unknown_signal warnings.
%% ============================================================

build_workload_signals(CallerOptions, Workload) :-
    must_be(list, CallerOptions),
    foldl(translate_option, CallerOptions, [], WorkloadReversed),
    reverse(WorkloadReversed, Workload).

reverse([], []).
reverse([H|T], R) :- reverse_acc(T, [H], R).
reverse_acc([], Acc, Acc).
reverse_acc([H|T], Acc, R) :- reverse_acc(T, [H|Acc], R).

%% translate_option(+Option, +Acc, -NewAcc)
translate_option(kernel_mode(M),                 Acc, [kernel_mode(M) | Acc]).
translate_option(strategy(S),                    Acc, [strategy(S) | Acc]).
translate_option(force_search_algorithm(A),      Acc, [force_search_algorithm(A) | Acc]).

%% csr_path implies csr_available(true) — add both signals.
translate_option(csr_path(P), Acc, [csr_available(true), csr_path(P) | Acc]).
translate_option(csr_available(B),               Acc, [csr_available(B) | Acc]).

translate_option(cardinality(C),                 Acc, [cardinality(C) | Acc]).
translate_option(determinism(D),                 Acc, [determinism(D) | Acc]).
translate_option(unique(B),                      Acc, [unique(B) | Acc]).
translate_option(query_pattern(P),               Acc, [query_pattern(P) | Acc]).
translate_option(query_frequency(F),             Acc, [query_frequency(F) | Acc]).
translate_option(graph_mutability(M),            Acc, [graph_mutability(M) | Acc]).

%% heuristic_predicate(P/A) implies heuristic_predicate_available(true).
translate_option(heuristic_predicate(PA),
                 Acc,
                 [heuristic_predicate_available(true), heuristic_predicate(PA) | Acc]).
translate_option(heuristic_predicate_available(B),
                 Acc,
                 [heuristic_predicate_available(B) | Acc]).

translate_option(b_eff(F),                       Acc, [b_eff(F) | Acc]).
translate_option(branching_d(F),                 Acc, [branching_d(F) | Acc]).
translate_option(contraction_r(F),               Acc, [contraction_r(F) | Acc]).
translate_option(csr_buildable(B),               Acc, [csr_buildable(B) | Acc]).

%% Catch-all: silently drop unrecognised options (target-specific
%% keys that the selector doesn't model).
translate_option(_, Acc, Acc).
