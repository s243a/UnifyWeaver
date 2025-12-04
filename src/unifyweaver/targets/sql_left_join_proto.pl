:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% sql_left_join_proto.pl - Prototype LEFT JOIN via Disjunction
% Extends sql_target.pl with LEFT JOIN detection and compilation

:- module(sql_left_join_proto, [
    detect_left_join_pattern/4,     % +Body, -LeftGoals, -RightGoal, -Fallback
    is_left_join_disjunction/1,     % +Goal
    extract_null_bindings/2,        % +Fallback, -NullVars
    compile_left_join_clause/5      % +LeftGoals, +RightGoal, +Fallback, +Options, -SQL
]).

:- use_module(library(lists)).

%% ============================================
%% PATTERN DETECTION
%% ============================================

%% detect_left_join_pattern(+Body, -LeftGoals, -RightGoal, -Fallback)
%  Detect the pattern: LeftGoals, (RightGoal ; Fallback)
%
detect_left_join_pattern(Body, LeftGoals, RightGoal, Fallback) :-
    % Match pattern: (Left, (Right ; Fallback))
    Body = (LeftPart, Disjunction),
    Disjunction = (RightGoal ; Fallback),

    % Convert left part to list of goals
    conjunction_to_list(LeftPart, LeftGoals),

    % Validate this is actually a LEFT JOIN pattern
    validate_left_join_pattern(LeftGoals, RightGoal, Fallback).

%% conjunction_to_list(+Conjunction, -Goals)
%  Convert conjunction to list of goals
%
conjunction_to_list((A, B), [A|Rest]) :- !,
    conjunction_to_list(B, Rest).
conjunction_to_list(Goal, [Goal]).

%% validate_left_join_pattern(+LeftGoals, +RightGoal, +Fallback)
%  Validate that this pattern represents a valid LEFT JOIN
%
validate_left_join_pattern(LeftGoals, RightGoal, Fallback) :-
    % 1. Right goal must be a table access
    is_table_goal(RightGoal),

    % 2. Right goal must use variables bound in left goals
    extract_bound_vars(LeftGoals, BoundVars),
    goal_uses_vars(RightGoal, BoundVars),

    % 3. Fallback must bind right-table variables to null
    extract_null_bindings(Fallback, _NullVars).

%% is_table_goal(+Goal)
%  Check if goal accesses a table
%
is_table_goal(Goal) :-
    Goal =.. [TableName|_],
    atom(TableName),
    % Check if this is a known table (would integrate with sql_table_def/2)
    TableName \= '=',
    TableName \= ';',
    TableName \= ','.

%% extract_bound_vars(+Goals, -BoundVars)
%  Extract variables that are bound by goals
%
extract_bound_vars(Goals, BoundVars) :-
    findall(Var,
            (member(Goal, Goals),
             Goal =.. [_|Args],
             member(Arg, Args),
             var(Arg),
             Var = Arg),
            BoundVars).

%% goal_uses_vars(+Goal, +Vars)
%  Check if goal uses any of the given variables
%
goal_uses_vars(Goal, Vars) :-
    Goal =.. [_|Args],
    member(Arg, Args),
    var(Arg),
    member(Arg, Vars), !.

%% extract_null_bindings(+Fallback, -NullVars)
%  Extract variables bound to null in fallback
%
extract_null_bindings(Fallback, NullVars) :-
    fallback_to_list(Fallback, FallbackGoals),
    findall(Var,
            (member(Goal, FallbackGoals),
             Goal = (Var = null),
             var(Var)),
            NullVars).

%% fallback_to_list(+Fallback, -Goals)
%  Convert fallback (potentially a conjunction) to list
%
fallback_to_list((A, B), [A|Rest]) :- !,
    fallback_to_list(B, Rest).
fallback_to_list(Goal, [Goal]).

%% is_left_join_disjunction(+Goal)
%  Quick check if a goal might be a LEFT JOIN disjunction
%
is_left_join_disjunction((RightGoal ; Fallback)) :-
    is_table_goal(RightGoal),
    contains_null_binding(Fallback).

%% contains_null_binding(+Goal)
%  Check if goal contains X = null pattern
%
contains_null_binding((A, B)) :- !,
    (contains_null_binding(A) ; contains_null_binding(B)).
contains_null_binding(_ = null).

%% ============================================
%% SQL GENERATION
%% ============================================

%% compile_left_join_clause(+LeftGoals, +RightGoal, +Fallback, +Options, -SQL)
%  Compile LEFT JOIN pattern to SQL
%
compile_left_join_clause(LeftGoals, RightGoal, Fallback, Options, SQL) :-
    % 1. Generate FROM clause from left goals
    generate_from_clause_for_left(LeftGoals, FromClause),

    % 2. Generate LEFT JOIN clause
    generate_left_join_clause(LeftGoals, RightGoal, JoinClause),

    % 3. Generate SELECT clause
    extract_null_bindings(Fallback, NullVars),
    generate_select_for_left_join(LeftGoals, RightGoal, NullVars, SelectClause),

    % 4. Get WHERE constraints (if any)
    extract_where_constraints(LeftGoals, WhereClause),

    % 5. Combine into SQL
    combine_left_join_sql(SelectClause, FromClause, JoinClause, WhereClause, Options, SQL).

%% generate_from_clause_for_left(+Goals, -FromClause)
%  Generate FROM clause from left table goals
%
generate_from_clause_for_left([Goal|_], FromClause) :-
    Goal =.. [TableName|_],
    format(string(FromClause), 'FROM ~w', [TableName]).

%% generate_left_join_clause(+LeftGoals, +RightGoal, -JoinClause)
%  Generate LEFT JOIN ON ... clause
%
generate_left_join_clause(LeftGoals, RightGoal, JoinClause) :-
    % Extract table name from right goal
    RightGoal =.. [RightTable|RightArgs],

    % Find shared variables (join condition)
    extract_bound_vars(LeftGoals, LeftVars),
    findall(JoinCond,
            (nth1(Idx, RightArgs, Arg),
             var(Arg),
             member(Arg, LeftVars),
             get_left_column(LeftGoals, Arg, LeftCol),
             get_right_column(RightTable, Idx, RightCol),
             format(string(JoinCond), '~w = ~w', [LeftCol, RightCol])),
            JoinConditions),

    % Combine conditions
    (   JoinConditions = []
    ->  format(string(JoinClause), 'LEFT JOIN ~w', [RightTable])
    ;   atomic_list_concat(JoinConditions, ' AND ', JoinCondStr),
        format(string(JoinClause), 'LEFT JOIN ~w ON ~w', [RightTable, JoinCondStr])
    ).

%% get_left_column(+Goals, +Var, -Column)
%  Get qualified column name for variable in left goals
%
get_left_column([Goal|_], Var, Column) :-
    Goal =.. [TableName|Args],
    nth1(Idx, Args, Arg),
    Arg == Var, !,
    get_column_name(TableName, Idx, ColName),
    format(atom(Column), '~w.~w', [TableName, ColName]).
get_left_column([_|Rest], Var, Column) :-
    get_left_column(Rest, Var, Column).

%% get_right_column(+Table, +Position, -Column)
%  Get qualified column name for position in table
%
get_right_column(TableName, Position, Column) :-
    get_column_name(TableName, Position, ColName),
    format(atom(Column), '~w.~w', [TableName, ColName]).

%% get_column_name(+Table, +Position, -Name)
%  Get column name from schema (stub - would integrate with sql_table_def)
%
get_column_name(_Table, 1, id).
get_column_name(_Table, 2, customer_id).
get_column_name(_Table, 2, name).
get_column_name(_Table, 3, product).
get_column_name(_Table, 3, region).
get_column_name(_Table, 4, amount).

%% generate_select_for_left_join(+LeftGoals, +RightGoal, +NullVars, -SelectClause)
%  Generate SELECT clause including NULL-able columns
%
generate_select_for_left_join(LeftGoals, RightGoal, NullVars, SelectClause) :-
    % Extract all output variables (simplified)
    extract_output_vars(LeftGoals, LeftOutVars),
    extract_output_vars([RightGoal], RightOutVars),
    append(LeftOutVars, RightOutVars, AllVars),

    % Generate column list
    findall(Col,
            (member(V, AllVars),
             (member(V, NullVars) -> Col = 'NULL' ; Col = V)),
            Columns),

    atomic_list_concat(Columns, ', ', ColStr),
    format(string(SelectClause), 'SELECT ~w', [ColStr]).

extract_output_vars(Goals, Vars) :-
    findall(V,
            (member(Goal, Goals),
             Goal =.. [_|Args],
             member(V, Args),
             var(V)),
            Vars).

%% extract_where_constraints(+Goals, -WhereClause)
%  Extract WHERE constraints from left goals
%
extract_where_constraints(Goals, WhereClause) :-
    findall(Constraint,
            (member(Goal, Goals),
             Goal = (Var = Value),
             \+ var(Value),
             format(string(Constraint), '~w = \'~w\'', [Var, Value])),
            Constraints),
    (   Constraints = []
    ->  WhereClause = ''
    ;   atomic_list_concat(Constraints, ' AND ', ConstraintStr),
        format(string(WhereClause), 'WHERE ~w', [ConstraintStr])
    ).

%% combine_left_join_sql(+Select, +From, +Join, +Where, +Options, -SQL)
%  Combine clauses into final SQL
%
combine_left_join_sql(SelectClause, FromClause, JoinClause, WhereClause, Options, SQL) :-
    % Get view name
    (   member(view_name(ViewName), Options)
    ->  true
    ;   ViewName = 'left_join_view'
    ),

    % Get format
    (   member(format(Format), Options)
    ->  true
    ;   Format = view
    ),

    % Build query
    (   WhereClause = ''
    ->  format(string(Query), '~w~n~w~n~w', [SelectClause, FromClause, JoinClause])
    ;   format(string(Query), '~w~n~w~n~w~n~w', [SelectClause, FromClause, JoinClause, WhereClause])
    ),

    % Wrap in view if requested
    (   Format = view
    ->  format(string(SQL), '-- View: ~w~nCREATE VIEW IF NOT EXISTS ~w AS~n~w;~n~n',
               [ViewName, ViewName, Query])
    ;   format(string(SQL), '~w;~n', [Query])
    ).
