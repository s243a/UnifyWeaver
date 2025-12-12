% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% pipeline_validation.pl - Compile-time validation for enhanced pipeline stages
%
% This module provides validation predicates for enhanced pipeline chaining,
% ensuring stage compatibility and catching errors early before code generation.
%
% Supported stage types:
%   - Pred/Arity          - Standard predicate stage
%   - fan_out(Stages)     - Broadcast to multiple parallel stages
%   - merge               - Combine results from fan_out
%   - route_by(Pred, Routes) - Conditional routing
%   - filter_by(Pred)     - Filter by predicate

:- module(pipeline_validation, [
    % Main validation API
    validate_pipeline/2,           % validate_pipeline(+Stages, -Errors)
    validate_pipeline/3,           % validate_pipeline(+Stages, +Options, -Result)

    % Individual stage validation
    validate_stage/2,              % validate_stage(+Stage, -Errors)
    validate_stage_type/2,         % validate_stage_type(+Stage, -Type)

    % Semantic validation
    validate_stage_sequence/2,     % validate_stage_sequence(+Stages, -Warnings)
    validate_fan_out/2,            % validate_fan_out(+FanOutStage, -Errors)
    validate_route_by/2,           % validate_route_by(+RouteByStage, -Errors)

    % Utility predicates
    is_valid_stage/1,              % is_valid_stage(+Stage)
    stage_type/2,                  % stage_type(+Stage, -Type)
    format_validation_error/2,     % format_validation_error(+Error, -Message)
    format_validation_warning/2,   % format_validation_warning(+Warning, -Message)

    % Testing
    test_pipeline_validation/0
]).

:- use_module(library(lists)).

% ============================================================================
% MAIN VALIDATION API
% ============================================================================

%% validate_pipeline(+Stages, -Errors) is det.
%
%  Validates a list of pipeline stages, returning any errors found.
%  An empty error list indicates the pipeline is valid.
%
%  @param Stages List of pipeline stages
%  @param Errors List of error terms (empty if valid)
validate_pipeline(Stages, Errors) :-
    validate_pipeline(Stages, [], result(Errors, _)).

%% validate_pipeline(+Stages, +Options, -Result) is det.
%
%  Validates a pipeline with options, returning both errors and warnings.
%
%  Options:
%    - strict(Bool)       : Treat warnings as errors (default: false)
%    - check_arities(Bool): Validate predicate arities exist (default: false)
%
%  Result is result(Errors, Warnings)
%
%  @param Stages  List of pipeline stages
%  @param Options Validation options
%  @param Result  result(Errors, Warnings) term
validate_pipeline(Stages, Options, result(FinalErrors, Warnings)) :-
    % Check for empty pipeline
    ( Stages == [] ->
        EmptyErrors = [error(empty_pipeline, 'Pipeline cannot be empty')]
    ;
        EmptyErrors = []
    ),

    % Validate each stage
    validate_all_stages(Stages, 1, StageErrors),

    % Validate stage sequence (warnings)
    validate_stage_sequence(Stages, Warnings),

    % Combine errors
    append(EmptyErrors, StageErrors, AllErrors),

    % Handle strict mode
    ( option(strict(true), Options) ->
        % Convert warnings to errors in strict mode
        maplist(warning_to_error, Warnings, WarningErrors),
        append(AllErrors, WarningErrors, FinalErrors)
    ;
        FinalErrors = AllErrors
    ).

%% warning_to_error(+Warning, -Error) is det.
warning_to_error(warning(Type, Msg), error(Type, Msg)).

%% validate_all_stages(+Stages, +Index, -Errors) is det.
validate_all_stages([], _, []).
validate_all_stages([Stage|Rest], Index, AllErrors) :-
    validate_stage_at_index(Stage, Index, Errors),
    Index1 is Index + 1,
    validate_all_stages(Rest, Index1, RestErrors),
    append(Errors, RestErrors, AllErrors).

%% validate_stage_at_index(+Stage, +Index, -Errors) is det.
validate_stage_at_index(Stage, Index, Errors) :-
    validate_stage(Stage, StageErrors),
    % Add index context to errors
    maplist(add_index_context(Index), StageErrors, Errors).

%% add_index_context(+Index, +Error, -ContextualError) is det.
add_index_context(Index, error(Type, Msg), error(Type, ContextMsg)) :-
    format(atom(ContextMsg), 'Stage ~w: ~w', [Index, Msg]).

% ============================================================================
% INDIVIDUAL STAGE VALIDATION
% ============================================================================

%% validate_stage(+Stage, -Errors) is det.
%
%  Validates a single pipeline stage.
%
%  @param Stage  A pipeline stage term
%  @param Errors List of errors (empty if valid)
validate_stage(Stage, Errors) :-
    ( is_valid_stage(Stage) ->
        validate_stage_specific(Stage, Errors)
    ;
        format(atom(Msg), 'Invalid stage type: ~w', [Stage]),
        Errors = [error(invalid_stage, Msg)]
    ).

%% is_valid_stage(+Stage) is semidet.
%
%  Succeeds if Stage is a recognized pipeline stage type.
is_valid_stage(Pred/Arity) :-
    atom(Pred),
    integer(Arity),
    Arity >= 0.
is_valid_stage(fan_out(Stages)) :-
    is_list(Stages).
is_valid_stage(merge).
is_valid_stage(route_by(Pred, Routes)) :-
    atom(Pred),
    is_list(Routes).
is_valid_stage(filter_by(Pred)) :-
    atom(Pred).

%% stage_type(+Stage, -Type) is det.
%
%  Determines the type of a pipeline stage.
stage_type(Pred/Arity, predicate) :-
    atom(Pred), integer(Arity), !.
stage_type(fan_out(_), fan_out) :- !.
stage_type(merge, merge) :- !.
stage_type(route_by(_, _), route_by) :- !.
stage_type(filter_by(_), filter_by) :- !.
stage_type(_, unknown).

%% validate_stage_type(+Stage, -Type) is det.
%
%  Validates stage and returns its type (or 'invalid' if invalid).
validate_stage_type(Stage, Type) :-
    ( is_valid_stage(Stage) ->
        stage_type(Stage, Type)
    ;
        Type = invalid
    ).

% ============================================================================
% STAGE-SPECIFIC VALIDATION
% ============================================================================

%% validate_stage_specific(+Stage, -Errors) is det.
validate_stage_specific(fan_out(Stages), Errors) :-
    !,
    validate_fan_out(fan_out(Stages), Errors).
validate_stage_specific(route_by(Pred, Routes), Errors) :-
    !,
    validate_route_by(route_by(Pred, Routes), Errors).
validate_stage_specific(filter_by(Pred), Errors) :-
    !,
    validate_filter_by(filter_by(Pred), Errors).
validate_stage_specific(Pred/Arity, Errors) :-
    !,
    validate_predicate_stage(Pred/Arity, Errors).
validate_stage_specific(merge, []) :- !.
validate_stage_specific(_, []).

%% validate_fan_out(+FanOutStage, -Errors) is det.
%
%  Validates a fan_out stage:
%    - Must have at least one sub-stage
%    - All sub-stages must be valid
validate_fan_out(fan_out([]), [error(empty_fan_out, 'fan_out requires at least one sub-stage')]) :- !.
validate_fan_out(fan_out(Stages), Errors) :-
    validate_fan_out_stages(Stages, 1, Errors).

validate_fan_out_stages([], _, []).
validate_fan_out_stages([Stage|Rest], Index, AllErrors) :-
    ( is_valid_stage(Stage) ->
        validate_stage_specific(Stage, StageErrors),
        maplist(prefix_fan_out_error(Index), StageErrors, PrefixedErrors)
    ;
        format(atom(Msg), 'fan_out sub-stage ~w is invalid: ~w', [Index, Stage]),
        PrefixedErrors = [error(invalid_fan_out_stage, Msg)]
    ),
    Index1 is Index + 1,
    validate_fan_out_stages(Rest, Index1, RestErrors),
    append(PrefixedErrors, RestErrors, AllErrors).

prefix_fan_out_error(Index, error(Type, Msg), error(Type, PrefixedMsg)) :-
    format(atom(PrefixedMsg), 'fan_out[~w]: ~w', [Index, Msg]).

%% validate_route_by(+RouteByStage, -Errors) is det.
%
%  Validates a route_by stage:
%    - Must have at least one route
%    - Each route must be (Condition, Stage) format
%    - Each route's stage must be valid
validate_route_by(route_by(_, []), [error(empty_routes, 'route_by requires at least one route')]) :- !.
validate_route_by(route_by(Pred, Routes), Errors) :-
    ( atom(Pred) ->
        PredErrors = []
    ;
        format(atom(Msg), 'route_by predicate must be an atom: ~w', [Pred]),
        PredErrors = [error(invalid_route_predicate, Msg)]
    ),
    validate_routes(Routes, 1, RouteErrors),
    append(PredErrors, RouteErrors, Errors).

validate_routes([], _, []).
validate_routes([Route|Rest], Index, AllErrors) :-
    validate_single_route(Route, Index, Errors),
    Index1 is Index + 1,
    validate_routes(Rest, Index1, RestErrors),
    append(Errors, RestErrors, AllErrors).

validate_single_route((Condition, Stage), Index, Errors) :-
    !,
    ( is_valid_stage(Stage) ->
        validate_stage_specific(Stage, StageErrors),
        maplist(prefix_route_error(Index, Condition), StageErrors, Errors)
    ;
        format(atom(Msg), 'route ~w (~w): invalid stage ~w', [Index, Condition, Stage]),
        Errors = [error(invalid_route_stage, Msg)]
    ).
validate_single_route(Route, Index, [error(invalid_route_format, Msg)]) :-
    format(atom(Msg), 'route ~w must be (Condition, Stage) format, got: ~w', [Index, Route]).

prefix_route_error(Index, Condition, error(Type, Msg), error(Type, PrefixedMsg)) :-
    format(atom(PrefixedMsg), 'route[~w,~w]: ~w', [Index, Condition, Msg]).

%% validate_filter_by(+FilterByStage, -Errors) is det.
%
%  Validates a filter_by stage:
%    - Predicate must be an atom
validate_filter_by(filter_by(Pred), Errors) :-
    ( atom(Pred) ->
        Errors = []
    ;
        format(atom(Msg), 'filter_by predicate must be an atom: ~w', [Pred]),
        Errors = [error(invalid_filter_predicate, Msg)]
    ).

%% validate_predicate_stage(+PredStage, -Errors) is det.
%
%  Validates a Pred/Arity stage:
%    - Pred must be an atom
%    - Arity must be a non-negative integer
validate_predicate_stage(Pred/Arity, Errors) :-
    ( atom(Pred) -> PredOk = true ; PredOk = false ),
    ( integer(Arity), Arity >= 0 -> ArityOk = true ; ArityOk = false ),
    ( PredOk == true, ArityOk == true ->
        Errors = []
    ;
        ( PredOk == false ->
            format(atom(Msg1), 'Predicate name must be an atom: ~w', [Pred]),
            E1 = [error(invalid_predicate_name, Msg1)]
        ;
            E1 = []
        ),
        ( ArityOk == false ->
            format(atom(Msg2), 'Arity must be a non-negative integer: ~w', [Arity]),
            E2 = [error(invalid_arity, Msg2)]
        ;
            E2 = []
        ),
        append(E1, E2, Errors)
    ).

% ============================================================================
% SEMANTIC VALIDATION (Warnings)
% ============================================================================

%% validate_stage_sequence(+Stages, -Warnings) is det.
%
%  Validates the sequence of stages for semantic issues.
%  Returns warnings (not errors) for issues that may be intentional.
%
%  Checks:
%    - merge should typically follow fan_out
%    - Multiple consecutive fan_out without merge
validate_stage_sequence(Stages, Warnings) :-
    check_merge_without_fan_out(Stages, MergeWarnings),
    check_fan_out_without_merge(Stages, FanOutWarnings),
    append(MergeWarnings, FanOutWarnings, Warnings).

%% check_merge_without_fan_out(+Stages, -Warnings) is det.
%
%  Warns if merge appears without a preceding fan_out.
check_merge_without_fan_out(Stages, Warnings) :-
    check_merge_helper(Stages, false, Warnings).

check_merge_helper([], _, []).
check_merge_helper([fan_out(_)|Rest], _, Warnings) :-
    !,
    check_merge_helper(Rest, true, Warnings).
check_merge_helper([merge|Rest], HadFanOut, Warnings) :-
    !,
    ( HadFanOut == false ->
        Warnings = [warning(merge_without_fan_out,
            'merge stage without preceding fan_out - results may be unexpected')|RestWarnings]
    ;
        Warnings = RestWarnings
    ),
    check_merge_helper(Rest, false, RestWarnings).
check_merge_helper([_|Rest], HadFanOut, Warnings) :-
    check_merge_helper(Rest, HadFanOut, Warnings).

%% check_fan_out_without_merge(+Stages, -Warnings) is det.
%
%  Warns if fan_out is not followed by merge before pipeline ends.
check_fan_out_without_merge(Stages, Warnings) :-
    check_fan_out_helper(Stages, Warnings).

check_fan_out_helper([], []).
check_fan_out_helper([fan_out(_)|Rest], Warnings) :-
    !,
    ( has_merge_before_end(Rest) ->
        check_fan_out_helper(Rest, Warnings)
    ;
        Warnings = [warning(fan_out_without_merge,
            'fan_out stage without subsequent merge - parallel results may be nested')|RestWarnings],
        check_fan_out_helper(Rest, RestWarnings)
    ).
check_fan_out_helper([_|Rest], Warnings) :-
    check_fan_out_helper(Rest, Warnings).

%% has_merge_before_end(+Stages) is semidet.
%
%  Succeeds if there's a merge stage before the next fan_out or end.
has_merge_before_end([merge|_]) :- !.
has_merge_before_end([fan_out(_)|_]) :- !, fail.
has_merge_before_end([_|Rest]) :- has_merge_before_end(Rest).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% format_validation_error(+Error, -Message) is det.
%
%  Formats an error term into a human-readable message.
format_validation_error(error(Type, Msg), Message) :-
    format(atom(Message), 'ERROR [~w]: ~w', [Type, Msg]).

%% format_validation_warning(+Warning, -Message) is det.
%
%  Formats a warning term into a human-readable message.
format_validation_warning(warning(Type, Msg), Message) :-
    format(atom(Message), 'WARNING [~w]: ~w', [Type, Msg]).

% ============================================================================
% TESTING
% ============================================================================

%% test_pipeline_validation is det.
%
%  Runs all pipeline validation tests.
test_pipeline_validation :-
    format('~n=== Pipeline Validation Tests ===~n~n', []),
    run_all_validation_tests,
    format('~n=== All Pipeline Validation Tests Passed ===~n', []).

run_all_validation_tests :-
    test_empty_pipeline,
    test_valid_simple_pipeline,
    test_valid_enhanced_pipeline,
    test_invalid_stage_type,
    test_empty_fan_out,
    test_empty_routes,
    test_invalid_route_format,
    test_merge_without_fan_out_warning,
    test_fan_out_without_merge_warning,
    test_nested_fan_out,
    test_complex_valid_pipeline.

%% Test: Empty pipeline
test_empty_pipeline :-
    format('  Test: empty pipeline...', []),
    validate_pipeline([], Errors),
    ( member(error(empty_pipeline, _), Errors) ->
        format(' PASSED~n', [])
    ;
        format(' FAILED~n', []),
        throw(test_failed(empty_pipeline))
    ).

%% Test: Valid simple pipeline
test_valid_simple_pipeline :-
    format('  Test: valid simple pipeline...', []),
    validate_pipeline([parse/1, transform/2, output/1], Errors),
    ( Errors == [] ->
        format(' PASSED~n', [])
    ;
        format(' FAILED: ~w~n', [Errors]),
        throw(test_failed(valid_simple_pipeline))
    ).

%% Test: Valid enhanced pipeline
test_valid_enhanced_pipeline :-
    format('  Test: valid enhanced pipeline...', []),
    Pipeline = [
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1]),
        merge,
        route_by(has_error, [(true, error_handler/1), (false, success/1)]),
        output/1
    ],
    validate_pipeline(Pipeline, Errors),
    ( Errors == [] ->
        format(' PASSED~n', [])
    ;
        format(' FAILED: ~w~n', [Errors]),
        throw(test_failed(valid_enhanced_pipeline))
    ).

%% Test: Invalid stage type
test_invalid_stage_type :-
    format('  Test: invalid stage type...', []),
    validate_pipeline([parse/1, "not_a_stage", output/1], Errors),
    ( member(error(invalid_stage, _), Errors) ->
        format(' PASSED~n', [])
    ;
        format(' FAILED~n', []),
        throw(test_failed(invalid_stage_type))
    ).

%% Test: Empty fan_out
test_empty_fan_out :-
    format('  Test: empty fan_out...', []),
    validate_pipeline([parse/1, fan_out([]), output/1], Errors),
    ( member(error(empty_fan_out, _), Errors) ->
        format(' PASSED~n', [])
    ;
        format(' FAILED~n', []),
        throw(test_failed(empty_fan_out))
    ).

%% Test: Empty routes
test_empty_routes :-
    format('  Test: empty routes...', []),
    validate_pipeline([parse/1, route_by(check, []), output/1], Errors),
    ( member(error(empty_routes, _), Errors) ->
        format(' PASSED~n', [])
    ;
        format(' FAILED~n', []),
        throw(test_failed(empty_routes))
    ).

%% Test: Invalid route format
test_invalid_route_format :-
    format('  Test: invalid route format...', []),
    validate_pipeline([parse/1, route_by(check, [not_a_tuple]), output/1], Errors),
    ( member(error(invalid_route_format, _), Errors) ->
        format(' PASSED~n', [])
    ;
        format(' FAILED~n', []),
        throw(test_failed(invalid_route_format))
    ).

%% Test: Merge without fan_out warning
test_merge_without_fan_out_warning :-
    format('  Test: merge without fan_out warning...', []),
    validate_pipeline([parse/1, merge, output/1], [], result(_, Warnings)),
    ( member(warning(merge_without_fan_out, _), Warnings) ->
        format(' PASSED~n', [])
    ;
        format(' FAILED~n', []),
        throw(test_failed(merge_without_fan_out_warning))
    ).

%% Test: Fan_out without merge warning
test_fan_out_without_merge_warning :-
    format('  Test: fan_out without merge warning...', []),
    validate_pipeline([parse/1, fan_out([a/1, b/1]), output/1], [], result(_, Warnings)),
    ( member(warning(fan_out_without_merge, _), Warnings) ->
        format(' PASSED~n', [])
    ;
        format(' FAILED~n', []),
        throw(test_failed(fan_out_without_merge_warning))
    ).

%% Test: Nested fan_out validation
test_nested_fan_out :-
    format('  Test: nested fan_out validation...', []),
    Pipeline = [
        parse/1,
        fan_out([
            validate/1,
            fan_out([enrich/1, audit/1])
        ]),
        merge,
        output/1
    ],
    validate_pipeline(Pipeline, Errors),
    % Nested fan_out is technically valid (will have its own warning)
    ( Errors == [] ->
        format(' PASSED~n', [])
    ;
        format(' FAILED: ~w~n', [Errors]),
        throw(test_failed(nested_fan_out))
    ).

%% Test: Complex valid pipeline
test_complex_valid_pipeline :-
    format('  Test: complex valid pipeline...', []),
    Pipeline = [
        parse/1,
        filter_by(is_valid),
        fan_out([
            transform_a/1,
            transform_b/1,
            audit_log/1
        ]),
        merge,
        route_by(priority, [
            (high, fast_track/1),
            (medium, standard/1),
            (low, batch/1)
        ]),
        filter_by(not_empty),
        output/1
    ],
    validate_pipeline(Pipeline, Errors),
    ( Errors == [] ->
        format(' PASSED~n', [])
    ;
        format(' FAILED: ~w~n', [Errors]),
        throw(test_failed(complex_valid_pipeline))
    ).
