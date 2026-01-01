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
    reorder_goals/4         % reorder_goals(+Goals, +BoundVars, +Pred, -OptimizedGoals)
]).

:- use_module(library(lists)).
:- use_module(library(ordsets)).
:- use_module(statistics).

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
        
        % Get predicate indicator for stats lookup
        functor(Head, Name, Arity),
        Pred = Name/Arity,
        
        % For reordering, we currently start with no variables bound.
        reorder_goals(Goals, [], Pred, OptimizedGoals),
        
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

%% reorder_goals(+Goals, +BoundVars, +Pred, -OptimizedGoals)
reorder_goals([], _, _, []).
reorder_goals(Goals, BoundVars, Pred, [BestGoal|RestOptimized]) :-
    % 1. Identify candidate goals (dependencies satisfied)
    ready_goals(BoundVars, Goals, ReadyGoals),
    
    (   ReadyGoals = []
    ->  % Fallback
        Goals = [BestGoal|Remaining],
        format('WARNING: No goals ready given bound vars ~w. Taking ~w.~n', [BoundVars, BestGoal])
    ;   % 2. Select best candidate
        select_best_goal(ReadyGoals, BoundVars, Pred, BestGoal),
        select(BestGoal, Goals, Remaining)
    ),
    
    % 3. Update bound vars
    term_variables(BestGoal, NewVars),
    append(BoundVars, NewVars, CombinedVars),
    sort(CombinedVars, NextBoundVars),
    
    % 4. Recurse
    reorder_goals(Remaining, NextBoundVars, Pred, RestOptimized).

% ============================================================================
% DEPENDENCY CHECKING
% ============================================================================

%% ready_goals(+BoundVars:list, +Goals:list, -ReadyGoals:list) is det.
%
%  Filter the goals whose dependencies are satisfied given the currently bound
%  variables.
ready_goals(_BoundVars, [], []).
ready_goals(BoundVars, [Goal|Rest], [Goal|ReadyRest]) :-
    is_ready(BoundVars, Goal),
    !,
    ready_goals(BoundVars, Rest, ReadyRest).
ready_goals(BoundVars, [_Goal|Rest], ReadyRest) :-
    ready_goals(BoundVars, Rest, ReadyRest).

%% is_ready(+BoundVars:list, +Goal) is semidet.
%
%  True if Goal can be executed given the variables currently bound.
%  Generators (predicate calls) are always considered ready because they can
%  introduce new variables. Built-ins that act as filters/constraints require
%  their operands to be bound.
is_ready(BoundVars, Goal0) :-
    strip_module(Goal0, _, Goal),
    (   Goal = (\+ Inner)
    ->  goal_vars(Inner, Vars),
        all_vars_bound(Vars, BoundVars)
    ;   Goal = is(_Var, Expr)
    ->  goal_vars(Expr, Vars),
        all_vars_bound(Vars, BoundVars)
    ;   functor(Goal, '=', 2)
    ->  goal_vars(Goal, Vars),
        all_vars_bound(Vars, BoundVars)
    ;   Goal = dif(_Left, _Right)
    ->  goal_vars(Goal, Vars),
        all_vars_bound(Vars, BoundVars)
    ;   goal_to_bash_comparison(Goal)
    ->  goal_vars(Goal, Vars),
        all_vars_bound(Vars, BoundVars)
    ;   is_match_goal(Goal, Var)
    ->  (   var(Var)
        ->  var_memberchk(Var, BoundVars)
        ;   true
        )
    ;   true
    ).

goal_vars(Goal, Vars) :-
    term_variables(Goal, Vars).

all_vars_bound([], _BoundVars).
all_vars_bound([Var|Rest], BoundVars) :-
    var_memberchk(Var, BoundVars),
    all_vars_bound(Rest, BoundVars).

var_memberchk(Var, [Var0|_]) :-
    Var == Var0,
    !.
var_memberchk(Var, [_|Rest]) :-
    var_memberchk(Var, Rest).

goal_to_bash_comparison(Goal) :-
    compound(Goal),
    functor(Goal, Op, 2),
    member(Op, ['>', '<', '>=', '=<', '=:=', '=\\=', '\\=', '\\==']).

is_match_goal(match(Var, _Pattern), Var) :- !.
is_match_goal(match(Var, _Pattern, _Type), Var) :- !.
is_match_goal(match(Var, _Pattern, _Type, _Groups), Var).

% ============================================================================
% HEURISTICS
% ============================================================================

%% select_best_goal(+Candidates, +BoundVars, +Pred, -Best)
%  Pick the best goal from ready candidates using cost estimation.
select_best_goal([Goal], _, _, Goal) :- !.
select_best_goal([G1, G2 | Rest], BoundVars, Pred, Best) :-
    compare_goals(Order, G1, G2, BoundVars, Pred),
    (   Order = > -> Winner = G2 ; Winner = G1 ),
    select_best_goal([Winner|Rest], BoundVars, Pred, Best).

%% compare_goals(?Order, +G1, +G2, +BoundVars, +Pred)
%  Compare two goals based on their estimated execution cost.
compare_goals(Order, G1, G2, BoundVars, Pred) :-
    statistics:estimate_cost(G1, BoundVars, Pred, C1),
    statistics:estimate_cost(G2, BoundVars, Pred, C2),
    compare(Order, C1, C2).
