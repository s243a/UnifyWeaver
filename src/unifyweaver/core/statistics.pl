:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 UnifyWeaver Contributors
%
% statistics.pl - Database Statistics for Query Optimization
%
% This module manages cardinality and selectivity statistics for predicates,
% allowing the optimizer to make better decisions about join ordering.

:- module(statistics, [
    declare_stats/2,        % declare_stats(+Pred, +Stats)
    get_stats/2,            % get_stats(+Pred, -Stats)
    clear_stats/1,          % clear_stats(+Pred)
    
    estimate_cost/3         % estimate_cost(+Goal, +BoundVars, -Cost)
]).

:- use_module(library(lists)).

% ============================================================================
% STORAGE
% ============================================================================

% stored_stats(Pred, stats{cardinality: C, fields: [field(Name, Selectivity), ...]})
:- dynamic stored_stats/2.

% ============================================================================
% PUBLIC API
% ============================================================================

%% declare_stats(+Pred, +Stats)
%
%  Stats is a dict: stats{cardinality: N, fields: [field(Name, Selectivity), ...]}
%  Selectivity is a float [0.0, 1.0] representing the fraction of records
%  matching a single value (1/UniqueValues).
declare_stats(Pred, Stats) :-
    retractall(stored_stats(Pred, _)),
    assertz(stored_stats(Pred, Stats)).

%% get_stats(+Pred, -Stats)
get_stats(Pred, Stats) :-
    stored_stats(Pred, Stats).

%% clear_stats(+Pred)
clear_stats(Pred) :-
    retractall(stored_stats(Pred, _)).

%% estimate_cost(+Goal, +BoundVars, -Cost)
%
%  Estimate the execution cost of a goal given currently bound variables.
%  Lower cost is better.
%
%  Cost Heuristics:
%  - Unifications/Comparisons: Cost 1 (very cheap)
%  - Generators with stats: Cardinality * Selectivity of bound fields
%  - Generators without stats: 10000 / (1 + BoundVarCount)
estimate_cost(Goal, BoundVars, Cost) :-
    classify_goal_type(Goal, Type),
    estimate_by_type(Type, Goal, BoundVars, Cost).

% ============================================================================
% INTERNAL HELPERS
% ============================================================================

classify_goal_type(_ = _, unification) :- !.
classify_goal_type(Goal, comparison) :-
    compound(Goal),
    functor(Goal, Op, 2),
    member(Op, [>, <, >=, =<, =:=, =\=]), !.
classify_goal_type(Goal, assignment) :-
    compound(Goal),
    functor(Goal, is, 2), !.
classify_goal_type(json_record(_), generator) :- !.
classify_goal_type(json_get(_, _), generator) :- !.
classify_goal_type(json_get(_, _, _), generator) :- !.
classify_goal_type(member(_, _), generator) :- !.
classify_goal_type(_, generic).

estimate_by_type(unification, _, _, 1).
estimate_by_type(comparison, _, _, 2).
estimate_by_type(assignment, _, _, 3).

estimate_by_type(generator, json_record(Fields), BoundVars, Cost) :-
    % Try to get stats for the "logical" predicate being accessed.
    % In UnifyWeaver, json_record is often used within a rule.
    % We might need to pass the Predicate indicator down.
    % For now, if we can't find specific stats, use a heuristic.
    (   find_matching_stats(json_record(Fields), BoundVars, Stats)
    ->  calculate_cost_from_stats(Fields, BoundVars, Stats, Cost)
    ;   % Heuristic: more bound variables = lower cost
        count_bound_fields(Fields, BoundVars, N),
        Cost is 10000 / (1 + N)
    ).

estimate_by_type(generator, _, BoundVars, Cost) :-
    % Generic generator heuristic
    term_variables(BoundVars, Vars),
    length(Vars, N),
    Cost is 5000 / (1 + N).

estimate_by_type(generic, _, _, 1000).

%% find_matching_stats(+Goal, +BoundVars, -Stats)
%  TODO: Implement lookup based on metadata or surrounding context.
find_matching_stats(_, _, _) :- fail.

calculate_cost_from_stats(_Fields, _BoundVars, stats{cardinality: C, fields: _FStats}, Cost) :-
    % Simple implementation: start with total cardinality
    % TODO: Multiply by selectivity of each bound field
    Cost = C.

count_bound_fields(Fields, BoundVars, Count) :-
    findall(F, (member(F-V, Fields), var_member(V, BoundVars)), Bound),
    length(Bound, Count).

var_member(V, [H|_]) :- V == H, !.
var_member(V, [_|T]) :- var_member(V, T).
