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
    load_stats/1,           % load_stats(+Path)
    
    estimate_cost/4         % estimate_cost(+Goal, +BoundVars, +Pred, -Cost)
]).

:- use_module(library(lists)).
:- use_module(library(http/json)).

% ============================================================================
% STORAGE
% ============================================================================

% stored_stats(Pred, stats{cardinality: C, fields: [field(Name, Selectivity), ...]})
:- dynamic stored_stats/2.

% ============================================================================
% PUBLIC API
% ============================================================================

%% load_stats(+Path)
%  Load statistics from a JSON file.
load_stats(Path) :-
    open(Path, read, Stream),
    json_read_dict(Stream, JSON),
    close(Stream),
    % Parse JSON and assert stats
    (   get_dict(predicates, JSON, Preds)
    ->  forall(get_dict(Key, Preds, Val), assert_stat_from_json(Key, Val))
    ;   true
    ).

assert_stat_from_json(PredKey, StatsDict) :-
    % Parse "user/2" string to term
    atom_string(PredAtom, PredKey),
    term_to_atom(Pred, PredAtom),
    
    Cardinality = StatsDict.cardinality,
    (   get_dict(fields, StatsDict, Fields)
    ->  dict_pairs(Fields, _, Pairs),
        findall(field(F, S), 
            (member(F-FData, Pairs), S = FData.selectivity), 
            FieldStats)
    ;   FieldStats = []
    ),
    
    declare_stats(Pred, stats{cardinality: Cardinality, fields: FieldStats}),
    format('Loaded stats for ~w (N=~w)~n', [Pred, Cardinality]).

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

%% estimate_cost(+Goal, +BoundVars, +Pred, -Cost)
%
%  Estimate the execution cost of a goal given currently bound variables.
%  Lower cost is better.
%
%  Cost Heuristics:
%  - Unifications/Comparisons: Cost 1 (very cheap)
%  - Generators with stats: Cardinality * Selectivity of bound fields
%  - Generators without stats: 10000 / (1 + BoundVarCount)
estimate_cost(Goal, BoundVars, Pred, Cost) :-
    classify_goal_type(Goal, Type),
    estimate_by_type(Type, Goal, BoundVars, Pred, Cost).

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

estimate_by_type(unification, _, _, _, 1).
estimate_by_type(comparison, _, _, _, 2).
estimate_by_type(assignment, _, _, _, 3).

estimate_by_type(generator, json_record(Fields), BoundVars, Pred, Cost) :-
    % Try to get stats for the predicate being accessed.
    (   find_matching_stats(Pred, Stats)
    ->  calculate_cost_from_stats(Fields, BoundVars, Stats, Cost)
    ;   % Heuristic: more bound variables = lower cost
        count_bound_fields(Fields, BoundVars, N),
        Cost is 10000 / (1 + N)
    ).

estimate_by_type(generator, _, BoundVars, _, Cost) :-
    % Generic generator heuristic
    term_variables(BoundVars, Vars),
    length(Vars, N),
    Cost is 5000 / (1 + N).

estimate_by_type(generic, _, _, _, 1000).

%% find_matching_stats(+Pred, -Stats)
find_matching_stats(Pred, Stats) :-
    get_stats(Pred, Stats).

calculate_cost_from_stats(Fields, BoundVars, stats{cardinality: C, fields: FStats}, Cost) :-
    % 1. Start with total cardinality
    % 2. Multiply by selectivity of each bound field
    % 3. Field stats is a list of field(Name, Selectivity)
    
    % Find all bound fields
    findall(F, (member(F-V, Fields), var_member(V, BoundVars)), BoundFieldNames),
    
    % Calculate combined selectivity
    foldl(multiply_selectivity(FStats), BoundFieldNames, 1.0, Selectivity),
    
    % Cost = Cardinality * Selectivity
    Cost is C * Selectivity.

multiply_selectivity(FStats, FieldName, Acc, Result) :-
    atom_string(FieldName, FieldStr),
    (   member(field(FieldStr, S), FStats)
    ->  Result is Acc * S
    ;   % Default selectivity if unknown (e.g., 0.1)
        Result is Acc * 0.1
    ).

count_bound_fields(Fields, BoundVars, Count) :-
    findall(F, (member(F-V, Fields), var_member(V, BoundVars)), Bound),
    length(Bound, Count).

var_member(V, [H|_]) :- V == H, !.
var_member(V, [_|T]) :- var_member(V, T).
