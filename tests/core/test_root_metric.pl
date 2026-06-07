:- encoding(utf8).
%% Test suite for root_metric.pl (Phase 0: directive validation + normalisation)
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_root_metric.pl

:- use_module('../../src/unifyweaver/core/root_metric').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner (same shape as test_demand_analysis.pl)
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("Root Metric Tests~n"),
    format("========================================~n~n"),
    findall(Name, clause(test(Name), _), Tests),
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
    (   catch(test(Test), Error,
            (format("[FAIL] ~w: ~w~n", [Test, Error]), fail))
    ->  Acc1 is Acc + 1, run_all(Rest, Acc1, Passed)
    ;   run_all(Rest, Acc, Passed)
    ).

pass(Name) :- format("[PASS] ~w~n", [Name]).

%% Helper: a valid min-distance opt list.
min_opts([ edge(parent/2, up), boundary('Root', 0),
           combine(succ), aggregate(min),
           max_depth(10), cycles(bounded), materialize(ingest) ]).

%% ========================================================================
%% Canonical instances normalise
%% ========================================================================

test(min_dist_normalises) :-
    min_opts(Opts),
    normalize_root_metric(min_dist_to_root/3, Opts, Spec),
    root_metric_aggregate(Spec, min),
    root_metric_combine(Spec, plus(1)),      % succ -> plus(1)
    root_metric_edge(Spec, parent/2, up),
    root_metric_boundary(Spec, 'Root', 0),
    root_metric_max_depth(Spec, 10),
    root_metric_materialize(Spec, ingest),
    pass(min_dist_normalises).

test(max_dist_normalises) :-
    normalize_root_metric(max_dist_to_root/3,
        [ edge(parent/2, up), boundary('Root', 0),
          combine(succ), aggregate(max), max_depth(10) ], Spec),
    root_metric_aggregate(Spec, max),
    root_metric_combine(Spec, plus(1)),
    pass(max_dist_normalises).

test(flux_normalises) :-
    normalize_root_metric(effective_distance/3,
        [ edge(parent/2, up), boundary('Root', 1.0),
          combine(scale(0.2)), aggregate(sum), max_depth(10) ], Spec),
    root_metric_aggregate(Spec, sum),
    root_metric_combine(Spec, scale(0.2)),
    root_metric_boundary(Spec, 'Root', 1.0),
    pass(flux_normalises).

%% ========================================================================
%% Defaults
%% ========================================================================

test(defaults_applied) :-
    normalize_root_metric(min_dist_to_root/3,
        [ edge(parent/2, up), boundary('Root', 0),
          combine(succ), aggregate(min) ], Spec),
    root_metric_max_depth(Spec, 10),         % default
    root_metric_cycles(Spec, bounded),       % default
    root_metric_materialize(Spec, kernel),   % default
    pass(defaults_applied).

%% ========================================================================
%% Rejections (validate_root_metric is semidet, must fail)
%% ========================================================================

test(reject_unknown_aggregate) :-
    \+ validate_root_metric(m/3,
        [ edge(parent/2, up), boundary('R', 0), combine(succ), aggregate(median) ]),
    pass(reject_unknown_aggregate).

test(reject_bad_combine) :-
    \+ validate_root_metric(m/3,
        [ edge(parent/2, up), boundary('R', 0), combine(times(2)), aggregate(min) ]),
    pass(reject_bad_combine).

test(reject_scale_out_of_range) :-
    \+ validate_root_metric(m/3,
        [ edge(parent/2, up), boundary('R', 1.0), combine(scale(1.5)), aggregate(sum) ]),
    pass(reject_scale_out_of_range).

test(reject_negative_depth) :-
    \+ validate_root_metric(m/3,
        [ edge(parent/2, up), boundary('R', 0), combine(succ), aggregate(min),
          max_depth(-1) ]),
    pass(reject_negative_depth).

test(reject_bad_direction) :-
    \+ validate_root_metric(m/3,
        [ edge(parent/2, sideways), boundary('R', 0), combine(succ), aggregate(min) ]),
    pass(reject_bad_direction).

test(reject_missing_edge) :-
    \+ validate_root_metric(m/3, [ boundary('R', 0), combine(succ), aggregate(min) ]),
    pass(reject_missing_edge).

test(reject_unsupported_pair) :-
    % min with a multiplicative combine has no semiring entry
    \+ validate_root_metric(m/3,
        [ edge(parent/2, up), boundary('R', 0), combine(scale(0.5)), aggregate(min) ]),
    pass(reject_unsupported_pair).

%% ========================================================================
%% Semiring table
%% ========================================================================

test(semiring_lookup) :-
    root_metric_semiring(min, plus(1), semiring(min, plus, inf, 0)),
    root_metric_semiring(max, plus(1), semiring(max, plus, neg_inf, 0)),
    root_metric_semiring(sum, scale(0.2), semiring(sum, scale, 0, 1)),
    combine_class(plus(1), additive),
    combine_class(scale(0.2), multiplicative),
    pass(semiring_lookup).

%% ========================================================================
%% Directive registration (idempotent re-register)
%% ========================================================================

%% ========================================================================
%% Cycle policy: per-path visited (simple-path)
%% ========================================================================

test(accept_visited_cycles) :-
    normalize_root_metric(min_dist_to_root/3,
        [ edge(parent/2, up), boundary('R', 0), combine(succ), aggregate(min),
          cycles(visited) ], Spec),
    root_metric_cycles(Spec, visited),
    pass(accept_visited_cycles).

test(reject_bad_cycles) :-
    \+ validate_root_metric(m/3,
        [ edge(parent/2, up), boundary('R', 0), combine(succ), aggregate(min),
          cycles(spiral) ]),
    pass(reject_bad_cycles).

%% ========================================================================
%% Presets + default metric
%% ========================================================================

test(preset_min_dist_fills_algorithm) :-
    % user supplies only graph-specific edge + boundary
    normalize_root_metric(min_dist_to_root/3,
        [ edge(parent/2, up), boundary('Root', 0), preset(min_dist) ], Spec),
    root_metric_aggregate(Spec, min),
    root_metric_combine(Spec, plus(1)),
    root_metric_max_depth(Spec, 10),
    pass(preset_min_dist_fills_algorithm).

test(preset_explicit_override) :-
    % explicit max_depth overrides the preset's default
    normalize_root_metric(min_dist_to_root/3,
        [ edge(parent/2, up), boundary('Root', 0), preset(min_dist), max_depth(4) ], Spec),
    root_metric_max_depth(Spec, 4),
    pass(preset_explicit_override).

test(reject_bad_preset) :-
    \+ validate_root_metric(m/3,
        [ edge(parent/2, up), boundary('R', 0), preset(nonesuch) ]),
    pass(reject_bad_preset).

test(default_root_metric_is_min_dist) :-
    default_root_metric(parent/2, up, 'Root', Spec),
    root_metric_aggregate(Spec, min),
    root_metric_combine(Spec, plus(1)),
    root_metric_edge(Spec, parent/2, up),
    root_metric_boundary(Spec, 'Root', 0),
    pass(default_root_metric_is_min_dist).

test(register_and_lookup) :-
    min_opts(Opts),
    root_metric(min_dist_to_root/3, Opts),
    registered_root_metric(min_dist_to_root/3, Spec),
    root_metric_aggregate(Spec, min),
    % re-register replaces, no duplicate
    root_metric(min_dist_to_root/3, Opts),
    findall(S, registered_root_metric(min_dist_to_root/3, S), All),
    length(All, 1),
    pass(register_and_lookup).
