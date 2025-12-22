:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 UnifyWeaver Contributors
%
% optimizer.pl - Query Optimization and Goal Reordering
%
% This module implements heuristic-based join optimization.
% It reorders goals in a rule body to minimize intermediate result sizes
% by prioritizing filters and selective goals, while respecting variable dependencies.

:- module(optimizer, [
    optimize_clause/4,      % optimize_clause(+Head, +Body, +Options, -OptimizedBody)
    reorder_goals/3         % reorder_goals(+Goals, +BoundVars, -OptimizedGoals)
]).

:- use_module(library(lists)).
:- use_module(library(ordsets)).

% ============================================================================
% MAIN ENTRY POINT
% ============================================================================

%% optimize_clause(+Head, +Body, +Options, -OptimizedBody)
%  Optimized version that respects the 'unordered' constraint.
%  Only reorders goals if unordered(true) is set.
optimize_clause(Head, Body, Options, OptimizedBody) :-
    (   member(unordered(true), Options)
    ->  format('  Optimizer: Reordering goals (unordered=true)~n'),
        % Extract goals from conjunction
        comma_list(Body, Goals),
        
        % For reordering, we currently start with no variables bound.
        % Future: if Head is called with bound args (Mode analysis), 
        % we could seed BoundVars with Head variables.
        reorder_goals(Goals, [], OptimizedGoals),
        
        % Reassemble
        comma_list(OptimizedBody, OptimizedGoals)
    ;   format('  Optimizer: Skipping reordering (unordered=false)~n'),
        OptimizedBody = Body
    ).

%% comma_list(?Comma, ?List)
%  Convert between conjunction (A, B, C) and List [A, B, C]
comma_list(Goal, [Goal]) :- \+ Goal = (_,_), !.
comma_list((A,B), [A|Rest]) :- comma_list(B, Rest).

% ============================================================================
% REORDERING LOGIC
% ============================================================================

%% reorder_goals(+Goals, +BoundVars, -OptimizedGoals)
%  Iteratively select the 'best' next goal that is executable.
%  An executable goal has sufficient bound arguments (mode analysis).
%  For UnifyWeaver:
%  - Comparisons (>, <, =) require ground inputs usually.
%  - Generators (parent, json_record) bind variables.
reorder_goals([], _, []).
reorder_goals(Goals, BoundVars, [BestGoal|RestOptimized]) :-
    % 1. Identify candidate goals (dependencies satisfied)
    include(is_ready(BoundVars), Goals, ReadyGoals),
    
    (   ReadyGoals = []
    ->  % No goal is ready? Topological sort failure or implicit dependencies?
        % Fallback: take the first one (standard Prolog order)
        % to avoid failure, though it might be inefficient/error.
        Goals = [BestGoal|Remaining],
        format('WARNING: No goals ready given bound vars ~w. Taking ~w.~n', [BoundVars, BestGoal])
    ;   % 2. Select best candidate
        select_best_goal(ReadyGoals, BoundVars, BestGoal),
        select(BestGoal, Goals, Remaining)
    ),
    
    % 3. Update bound vars
    term_variables(BestGoal, NewVars),
    append(BoundVars, NewVars, CombinedVars),
    sort(CombinedVars, NextBoundVars),
    
    % 4. Recurse
    reorder_goals(Remaining, NextBoundVars, RestOptimized).

% ============================================================================
% READINESS CHECK
% ============================================================================

%% is_ready(+BoundVars, +Goal)
%  Check if a goal can be executed given current bindings.
is_ready(BoundVars, Goal) :-
    classify_goal(Goal, Type),
    check_dependencies(Type, Goal, BoundVars).

classify_goal(Goal, comparison) :-
    compound(Goal),
    functor(Goal, Op, 2),
    member(Op, [>, <, >=, =<, =:=, =\=]), !.
classify_goal(Goal, assignment) :-
    compound(Goal),
    functor(Goal, is, 2), !.
classify_goal(json_record(_), generator) :- !.
classify_goal(json_get(_, _), generator) :- !.
classify_goal(json_get(_, _, _), generator) :- !.
classify_goal(_, generic).

check_dependencies(comparison, Goal, BoundVars) :-
    term_variables(Goal, Vars),
    var_subset(Vars, BoundVars).  % All vars must be bound for comparison

check_dependencies(assignment, Goal, BoundVars) :-
    Goal =.. [is, _Var, Expr],
    term_variables(Expr, ExprVars),
    var_subset(ExprVars, BoundVars). % Expression inputs must be bound

check_dependencies(generator, _, _) :- true. % Generators can usually start fresh (e.g. iterate all)
check_dependencies(generic, _, _) :- true.   % Assume standard predicates can run (maybe inefficiently)

%% var_subset(+Subset, +Set)
%  Check if all variables in Subset are present in Set (using identity ==)
var_subset([], _).
var_subset([V|Rest], Set) :-
    var_member(V, Set),
    var_subset(Rest, Set).

var_member(V, [H|_]) :- V == H, !.
var_member(V, [_|T]) :- var_member(V, T).

% ============================================================================
% HEURISTICS
% ============================================================================

%% select_best_goal(+Candidates, +BoundVars, -Best)
%  Pick the best goal from ready candidates.
%  Priority:
%  1. Filters (comparison) - Cheap, reduces set size
%  2. Assignments (is/2) - Computes values needed later
%  3. Generators with MOST bound arguments (more selective)
%  4. Generators with FEWEST bound arguments
select_best_goal([Goal], _, Goal) :- !.
select_best_goal([G1, G2 | Rest], BoundVars, Best) :-
    compare_goals(Order, G1, G2, BoundVars),
    (   Order = > -> Winner = G2 ; Winner = G1 ),
    select_best_goal([Winner|Rest], BoundVars, Best).

compare_goals(<, G1, _, _) :- classify_goal(G1, comparison), !.
compare_goals(>, _, G2, _) :- classify_goal(G2, comparison), !.

compare_goals(<, G1, _, _) :- classify_goal(G1, assignment), !.
compare_goals(>, _, G2, _) :- classify_goal(G2, assignment), !.

% Prefer goals with MORE bound variables (Selectivity Heuristic)
compare_goals(Order, G1, G2, BoundVars) :-
    count_bound_vars(G1, BoundVars, N1),
    count_bound_vars(G2, BoundVars, N2),
    compare(Order, N2, N1). % Higher N is better (< means G1 is better/smaller cost?) 
    % Wait, compare/3: compare(Order, Term1, Term2). < if T1 < T2.
    % We want the "Best" to be the "Smallest" in sorted list?
    % No, we are selecting one.
    % Let's define < as "Better".
    % If N1 > N2, G1 is Better. So G1 < G2.
    % So compare(Order, N2, N1) works: if N2 < N1, returns >. Wait.
    
    % Let's be explicit:
    % if N1 > N2 -> G1 better -> return <
    % if N1 < N2 -> G2 better -> return >

count_bound_vars(Goal, BoundVars, Count) :-
    term_variables(Goal, Vars),
    include(flip_var_member(BoundVars), Vars, Bound),
    length(Bound, Count).

flip_var_member(Set, V) :- var_member(V, Set).
