:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% sql_target.pl - SQL Target for UnifyWeaver
% Generates SQL queries (VIEWs, CTEs) from Prolog predicates
% Supports SQLite, PostgreSQL, MySQL, and other SQL databases

:- module(sql_target, [
    compile_predicate_to_sql/3,     % +Predicate, +Options, -SQLCode
    compile_set_operation/4,        % +SetOp, +Predicates, +Options, -SQLCode
    write_sql_file/2,               % +SQLCode, +FilePath
    sql_table/2                     % +TableName, +Columns (directive)
]).

:- use_module(library(lists)).
:- use_module(library(filesex)).

% Suppress singleton warnings
:- style_check(-singleton).

%% ============================================
%% SCHEMA DECLARATIONS
%% ============================================

:- dynamic sql_table_def/2.

%% sql_table(+TableName, +Columns)
%  Define a SQL table schema
%  Usage: :- sql_table(person, [name-text, age-integer, city-text]).
%
sql_table(TableName, Columns) :-
    retractall(sql_table_def(TableName, _)),
    assertz(sql_table_def(TableName, Columns)),
    format('SQL table schema defined: ~w ~w~n', [TableName, Columns]).

%% get_table_schema(+TableName, -Columns)
%  Retrieve table schema
%
get_table_schema(TableName, Columns) :-
    sql_table_def(TableName, Columns), !.
get_table_schema(TableName, _) :-
    format('ERROR: Table schema not found: ~w~n', [TableName]),
    fail.

%% ============================================
%% PUBLIC API
%% ============================================

%% compile_predicate_to_sql(+Predicate, +Options, -SQLCode)
%  Compile a Prolog predicate to SQL code
%
%  @arg Predicate Predicate indicator (Name/Arity) or clause
%  @arg Options List of options
%  @arg SQLCode Generated SQL code as string
%
%  Options:
%  - view_name(Name) - Override view name (default: predicate name)
%  - dialect(sqlite|postgres|mysql) - SQL dialect (default: sqlite)
%  - format(view|cte|select) - Output format (default: view)
%
compile_predicate_to_sql(Predicate, Options, SQLCode) :-
    % Get predicate name and arity
    (   Predicate = Name/Arity
    ->  true
    ;   functor(Predicate, Name, Arity)
    ),

    % Get all clauses for this predicate from user module
    % IMPORTANT: Collect both Head and Body together to preserve variable sharing!
    % Create fresh Head inside findall so variables are properly shared
    findall(head_body(H, B), (functor(H, Name, Arity), user:clause(H, B)), HeadBodyPairs),

    (   HeadBodyPairs = []
    ->  format('WARNING: No clauses found for ~w/~w~n', [Name, Arity]),
        SQLCode = ''
    ;   % Extract just the bodies (but we'll use heads too)
        findall(HB, member(HB, HeadBodyPairs), Pairs),
        % Compile clauses to SQL
        compile_clauses_to_sql(Name, Arity, Pairs, Options, SQLCode)
    ).

%% compile_clauses_to_sql(+Name, +Arity, +HeadBodyPairs, +Options, -SQLCode)
%  Compile multiple clauses to SQL with UNION
%
compile_clauses_to_sql(Name, Arity, HeadBodyPairs, Options, SQLCode) :-
    (   HeadBodyPairs = [head_body(Head, Body)]
    ->  % Single clause - direct compilation
        compile_single_clause(Name, Arity, Body, Head, Options, SQLCode)
    ;   % Multiple clauses - use UNION
        compile_union_clauses(Name, Arity, HeadBodyPairs, Options, SQLCode)
    ).

%% compile_union_clauses(+Name, +Arity, +HeadBodyPairs, +Options, -SQLCode)
%  Compile multiple clauses using UNION
%
compile_union_clauses(Name, Arity, HeadBodyPairs, Options, SQLCode) :-
    % Compile each clause to a SELECT statement
    findall(SelectSQL,
            (   member(head_body(Head, Body), HeadBodyPairs),
                compile_clause_to_select(Name, Arity, Body, Head, SelectSQL)
            ),
            SelectStatements),

    % Determine UNION type (default: UNION for distinct)
    (   member(union_all(true), Options)
    ->  UnionSep = '\nUNION ALL\n'
    ;   UnionSep = '\nUNION\n'
    ),

    % Join with UNION
    atomic_list_concat(SelectStatements, UnionSep, UnionQuery),

    % Get output format and view name
    (   member(format(Format), Options)
    ->  true
    ;   Format = view
    ),
    (   member(view_name(ViewName), Options)
    ->  true
    ;   ViewName = Name
    ),

    % Format as view or standalone SELECT
    format_union_sql(Format, ViewName, UnionQuery, SQLCode).

%% compile_clause_to_select(+Name, +Arity, +Body, +Head, -SelectSQL)
%  Compile a single clause to a SELECT statement (without CREATE VIEW wrapper)
%
compile_clause_to_select(Name, Arity, Body, Head, SelectSQL) :-
    % Check if this is an aggregation clause
    (   is_group_by_clause(Body)
    ->  compile_aggregation_to_select(Name, Arity, Body, Head, SelectSQL)
    ;   % Regular clause
        parse_clause_body(Body, ParsedGoals),
        separate_goals(ParsedGoals, TableGoals, Constraints),
        Head =.. [Name|Args],
        generate_select_clause(Args, TableGoals, SelectClause),
        generate_from_clause(TableGoals, FromClause),
        generate_where_clause(Constraints, Args, TableGoals, WhereClause),

        % Combine into standalone SELECT
        (   WhereClause = ''
        ->  format(string(SelectSQL), '~w\n~w', [SelectClause, FromClause])
        ;   format(string(SelectSQL), '~w\n~w\n~w', [SelectClause, FromClause, WhereClause])
        )
    ).

%% compile_aggregation_to_select(+Name, +Arity, +Body, +Head, -SelectSQL)
%  Compile aggregation clause to SELECT statement
%
compile_aggregation_to_select(Name, Arity, Body, Head, SelectSQL) :-
    extract_group_by_spec(Body, GroupField, Goal, AggOp, Result),
    extract_having_constraints(Body, HavingConstraints),
    parse_clause_body(Goal, ParsedGoals),
    separate_goals(ParsedGoals, TableGoals, WhereConstraints),
    Head =.. [Name|Args],
    generate_aggregation_select(Args, GroupField, AggOp, Result, TableGoals, SelectClause),
    generate_from_clause(TableGoals, FromClause),
    generate_where_clause(WhereConstraints, Args, TableGoals, WhereClause),
    find_var_column(GroupField, [], TableGoals, GroupColumn),
    format(string(GroupByClause), 'GROUP BY ~w', [GroupColumn]),
    generate_having_clause(HavingConstraints, Result, HavingClause),

    % Combine into standalone SELECT
    (   WhereClause = '', HavingClause = ''
    ->  format(string(SelectSQL), '~w\n~w\n~w', [SelectClause, FromClause, GroupByClause])
    ;   WhereClause \= '', HavingClause = ''
    ->  format(string(SelectSQL), '~w\n~w\n~w\n~w', [SelectClause, FromClause, WhereClause, GroupByClause])
    ;   WhereClause = '', HavingClause \= ''
    ->  format(string(SelectSQL), '~w\n~w\n~w\n~w', [SelectClause, FromClause, GroupByClause, HavingClause])
    ;   format(string(SelectSQL), '~w\n~w\n~w\n~w\n~w', [SelectClause, FromClause, WhereClause, GroupByClause, HavingClause])
    ).

%% format_union_sql(+Format, +ViewName, +UnionQuery, -SQL)
%  Format UNION query as view or standalone
%
format_union_sql(view, ViewName, UnionQuery, SQL) :-
    format(string(SQL), '-- View: ~w~nCREATE VIEW IF NOT EXISTS ~w AS~n~w;~n~n',
           [ViewName, ViewName, UnionQuery]).
format_union_sql(select, _, UnionQuery, SQL) :-
    format(string(SQL), '~w;~n', [UnionQuery]).

%% compile_set_operation(+SetOp, +Predicates, +Options, -SQLCode)
%  Compile set operations (INTERSECT, EXCEPT) over multiple predicates
%  SetOp: intersect, except (or minus)
%  Predicates: List of predicate specs (Name/Arity)
%  Options: Same as compile_predicate_to_sql/3
%
compile_set_operation(SetOp, Predicates, Options, SQLCode) :-
    % Validate set operation
    (   member(SetOp, [intersect, except, minus])
    ->  true
    ;   format('ERROR: Unknown set operation: ~w~n', [SetOp]),
        fail
    ),

    % Normalize MINUS to EXCEPT
    (   SetOp = minus
    ->  ActualOp = except
    ;   ActualOp = SetOp
    ),

    % Compile each predicate to a standalone SELECT statement
    findall(SelectSQL,
            (   member(Pred, Predicates),
                Pred = Name/Arity,
                % Get the predicate clauses
                functor(Goal, Name, Arity),
                findall(head_body(Goal, Body),
                        clause(Goal, Body),
                        HeadBodyPairs),
                (   HeadBodyPairs = []
                ->  format('ERROR: No clauses found for ~w~n', [Pred]),
                    fail
                ;   HeadBodyPairs = [head_body(Head, Body)]
                ->  % Single clause - compile to SELECT
                    compile_clause_to_select(Name, Arity, Body, Head, SelectSQL)
                ;   % Multiple clauses - need UNION first
                    compile_union_clauses(Name, Arity, HeadBodyPairs, [format(select)], UnionSQL),
                    % Wrap in parentheses for set operation
                    format(string(SelectSQL), '(~w)', [UnionSQL])
                )
            ),
            SelectStatements),

    % Ensure we have at least 2 statements
    (   length(SelectStatements, Len),
        Len < 2
    ->  format('ERROR: Set operations require at least 2 predicates~n', []),
        fail
    ;   true
    ),

    % Determine operator keyword
    (   ActualOp = intersect
    ->  OpKeyword = 'INTERSECT'
    ;   OpKeyword = 'EXCEPT'
    ),

    % Build separator with operator
    format(string(OpSep), '~n~w~n', [OpKeyword]),

    % Join statements with operator
    atomic_list_concat(SelectStatements, OpSep, SetQuery),

    % Get output format and view name
    (   member(format(Format), Options)
    ->  true
    ;   Format = view
    ),
    (   member(view_name(ViewName), Options)
    ->  true
    ;   % Generate view name from operation and predicates
        Predicates = [FirstPred|_],
        FirstPred = FirstName/_,
        format(atom(ViewName), '~w_~w', [FirstName, ActualOp])
    ),

    % Format as view or standalone SELECT
    format_set_operation_sql(Format, ViewName, SetQuery, SQLCode).

%% format_set_operation_sql(+Format, +ViewName, +SetQuery, -SQL)
%  Format set operation query as view or standalone
%
format_set_operation_sql(view, ViewName, SetQuery, SQL) :-
    format(string(SQL), '-- View: ~w~nCREATE VIEW IF NOT EXISTS ~w AS~n~w;~n~n',
           [ViewName, ViewName, SetQuery]).
format_set_operation_sql(select, _, SetQuery, SQL) :-
    format(string(SQL), '~w;~n', [SetQuery]).

%% compile_single_clause(+Name, +Arity, +Body, +Head, +Options, -SQLCode)
%  Compile a single Prolog clause to SQL
%
compile_single_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    % Check if this is an aggregation clause
    (   is_group_by_clause(Body)
    ->  compile_aggregation_clause(Name, Arity, Body, Head, Options, SQLCode)
    % Check if this is a FULL OUTER JOIN clause (check before RIGHT/LEFT)
    ;   is_full_outer_join_clause(Body)
    ->  compile_full_outer_join_clause(Name, Arity, Body, Head, Options, SQLCode)
    % Check if this is a RIGHT JOIN clause
    ;   is_right_join_clause(Body)
    ->  compile_right_join_clause(Name, Arity, Body, Head, Options, SQLCode)
    % Check if this is a LEFT JOIN clause (disjunction pattern)
    ;   is_left_join_clause(Body)
    ->  compile_left_join_clause(Name, Arity, Body, Head, Options, SQLCode)
    ;   % Regular clause - existing logic
        % Parse the clause body
        parse_clause_body(Body, ParsedGoals),

        % Extract table references and constraints
        separate_goals(ParsedGoals, TableGoals, Constraints),

        % Generate SELECT clause from head arguments
        Head =.. [Name|Args],
        generate_select_clause(Args, TableGoals, SelectClause),

        % Generate FROM clause
        generate_from_clause(TableGoals, FromClause),

        % Generate WHERE clause
        generate_where_clause(Constraints, Args, TableGoals, WhereClause),

        % Get output format
        (   member(format(Format), Options)
        ->  true
        ;   Format = view
        ),

        % Get view name
        (   member(view_name(ViewName), Options)
        ->  true
        ;   ViewName = Name
        ),

        % Combine into SQL
        format_sql(Format, ViewName, SelectClause, FromClause, WhereClause, SQLCode)
    ).

%% ============================================
%% AGGREGATION SUPPORT (GROUP BY)
%% ============================================

%% is_group_by_clause(+Body)
%  Check if clause body contains group_by
%
is_group_by_clause(Body) :-
    extract_group_by_spec(Body, _, _, _, _), !.

%% extract_group_by_spec(+Body, -GroupField, -Goal, -AggOp, -Result)
%  Extract group_by components from clause body
%
extract_group_by_spec(group_by(GroupField, Goal, AggOp), GroupField, Goal, AggOp, null) :- !.
extract_group_by_spec(group_by(GroupField, Goal, AggOp, Result), GroupField, Goal, AggOp, Result) :- !.
extract_group_by_spec((group_by(GroupField, Goal, AggOp), _Rest), GroupField, Goal, AggOp, null) :- !.
extract_group_by_spec((group_by(GroupField, Goal, AggOp, Result), _Rest), GroupField, Goal, AggOp, Result) :- !.

%% extract_having_constraints(+Body, -Constraints)
%  Extract HAVING constraints from after group_by
%
extract_having_constraints(group_by(_, _, _), []) :- !.
extract_having_constraints(group_by(_, _, _, _), []) :- !.
extract_having_constraints((group_by(_, _, _), Rest), Constraints) :- !,
    parse_clause_body(Rest, Constraints).
extract_having_constraints((group_by(_, _, _, _), Rest), Constraints) :- !,
    parse_clause_body(Rest, Constraints).
extract_having_constraints(_, []).

%% compile_aggregation_clause(+Name, +Arity, +Body, +Head, +Options, -SQLCode)
%  Compile a group_by clause to SQL with GROUP BY
%
compile_aggregation_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    % Extract group_by specification
    extract_group_by_spec(Body, GroupField, Goal, AggOp, Result),

    % Extract HAVING constraints
    extract_having_constraints(Body, HavingConstraints),

    % Parse the inner goal
    parse_clause_body(Goal, ParsedGoals),
    separate_goals(ParsedGoals, TableGoals, WhereConstraints),

    % Get head arguments
    Head =.. [Name|Args],

    % Generate SELECT clause with aggregation
    generate_aggregation_select(Args, GroupField, AggOp, Result, TableGoals, SelectClause),

    % Generate FROM clause
    generate_from_clause(TableGoals, FromClause),

    % Generate WHERE clause
    generate_where_clause(WhereConstraints, Args, TableGoals, WhereClause),

    % Generate GROUP BY clause
    find_var_column(GroupField, [], TableGoals, GroupColumn),
    format(string(GroupByClause), 'GROUP BY ~w', [GroupColumn]),

    % Generate HAVING clause
    generate_having_clause(HavingConstraints, Result, HavingClause),

    % Get output format and view name
    (   member(format(Format), Options)
    ->  true
    ;   Format = view
    ),
    (   member(view_name(ViewName), Options)
    ->  true
    ;   ViewName = Name
    ),

    % Combine into SQL
    format_aggregation_sql(Format, ViewName, SelectClause, FromClause, WhereClause,
                          GroupByClause, HavingClause, SQLCode).

%% generate_aggregation_select(+Args, +GroupField, +AggOp, +Result, +TableGoals, -SelectClause)
%  Generate SELECT clause with aggregation function
%
generate_aggregation_select(Args, GroupField, AggOp, Result, TableGoals, SelectClause) :-
    % Find group column
    find_var_column(GroupField, [], TableGoals, GroupColumn),

    % Map aggregation operation to SQL
    agg_op_to_sql(AggOp, AggFunc),

    % Determine what we're aggregating
    (   Result = null
    ->  % Simple aggregation like count(*)
        (   AggOp = count
        ->  format(string(AggExpr), '~w(*)', [AggFunc])
        ;   % Need to infer column from table goals
            infer_agg_column(TableGoals, AggColumn),
            format(string(AggExpr), '~w(~w)', [AggFunc, AggColumn])
        )
    ;   % Explicit result variable - find its column
        find_var_column(Result, [], TableGoals, ResultColumn),
        format(string(AggExpr), '~w(~w)', [AggFunc, ResultColumn])
    ),

    % Generate SELECT items
    (   Args = [GroupField, Result]
    ->  % Grouping column and aggregation result
        format(string(SelectClause), 'SELECT ~w, ~w', [GroupColumn, AggExpr])
    ;   Args = [GroupField]
    ->  % Just grouping column
        format(string(SelectClause), 'SELECT ~w', [GroupColumn])
    ;   Args = [Result]
    ->  % Just aggregation result
        format(string(SelectClause), 'SELECT ~w', [AggExpr])
    ;   % Default: group column and aggregation
        format(string(SelectClause), 'SELECT ~w, ~w', [GroupColumn, AggExpr])
    ).

%% agg_op_to_sql(+AggOp, -SQLFunc)
%  Map Prolog aggregation operation to SQL function
%
agg_op_to_sql(count, 'COUNT').
agg_op_to_sql(sum, 'SUM').
agg_op_to_sql(avg, 'AVG').
agg_op_to_sql(max, 'MAX').
agg_op_to_sql(min, 'MIN').

%% infer_agg_column(+TableGoals, -Column)
%  Infer which column to aggregate (for simple cases)
%
infer_agg_column([Goal|_], Column) :-
    Goal =.. [TableName|Args],
    get_table_schema(TableName, Schema),
    % Find first numeric column
    nth1(Pos, Args, Arg),
    var(Arg),
    nth1(Pos, Schema, Column-Type),
    member(Type, [integer, number, real]), !.
infer_agg_column([_|Rest], Column) :-
    infer_agg_column(Rest, Column).

%% generate_having_clause(+Constraints, +Result, -HavingClause)
%  Generate HAVING clause from constraints
%
generate_having_clause([], _, '') :- !.
generate_having_clause(Constraints, Result, HavingClause) :-
    findall(Condition,
            (member(C, Constraints), having_constraint_to_sql(C, Result, Condition)),
            Conditions),
    (   Conditions = []
    ->  HavingClause = ''
    ;   atomic_list_concat(Conditions, ' AND ', ConditionsStr),
        format(string(HavingClause), 'HAVING ~w', [ConditionsStr])
    ).

%% having_constraint_to_sql(+Constraint, +Result, -SQLCondition)
%  Convert HAVING constraint to SQL
%
having_constraint_to_sql(Result >= Value, Result, Condition) :- !,
    format(string(Condition), 'COUNT(*) >= ~w', [Value]).
having_constraint_to_sql(Result > Value, Result, Condition) :- !,
    format(string(Condition), 'COUNT(*) > ~w', [Value]).
having_constraint_to_sql(Result =< Value, Result, Condition) :- !,
    format(string(Condition), 'COUNT(*) <= ~w', [Value]).
having_constraint_to_sql(Result < Value, Result, Condition) :- !,
    format(string(Condition), 'COUNT(*) < ~w', [Value]).
having_constraint_to_sql(Result =:= Value, Result, Condition) :- !,
    format(string(Condition), 'COUNT(*) = ~w', [Value]).

%% format_aggregation_sql(+Format, +ViewName, +Select, +From, +Where, +GroupBy, +Having, -SQL)
%  Format SQL with GROUP BY and optional HAVING
%
format_aggregation_sql(view, ViewName, Select, From, Where, GroupBy, Having, SQL) :-
    (   Where = '', Having = ''
    ->  format(string(SQL), '-- View: ~w~nCREATE VIEW IF NOT EXISTS ~w AS~n~w~n~w~n~w;~n~n',
               [ViewName, ViewName, Select, From, GroupBy])
    ;   Where \= '', Having = ''
    ->  format(string(SQL), '-- View: ~w~nCREATE VIEW IF NOT EXISTS ~w AS~n~w~n~w~n~w~n~w;~n~n',
               [ViewName, ViewName, Select, From, Where, GroupBy])
    ;   Where = '', Having \= ''
    ->  format(string(SQL), '-- View: ~w~nCREATE VIEW IF NOT EXISTS ~w AS~n~w~n~w~n~w~n~w;~n~n',
               [ViewName, ViewName, Select, From, GroupBy, Having])
    ;   % Both WHERE and HAVING
        format(string(SQL), '-- View: ~w~nCREATE VIEW IF NOT EXISTS ~w AS~n~w~n~w~n~w~n~w~n~w;~n~n',
               [ViewName, ViewName, Select, From, Where, GroupBy, Having])
    ).

format_aggregation_sql(select, _, Select, From, Where, GroupBy, Having, SQL) :-
    (   Where = '', Having = ''
    ->  format(string(SQL), '~w~n~w~n~w;~n', [Select, From, GroupBy])
    ;   Where \= '', Having = ''
    ->  format(string(SQL), '~w~n~w~n~w~n~w;~n', [Select, From, Where, GroupBy])
    ;   Where = '', Having \= ''
    ->  format(string(SQL), '~w~n~w~n~w~n~w;~n', [Select, From, GroupBy, Having])
    ;   % Both WHERE and HAVING
        format(string(SQL), '~w~n~w~n~w~n~w~n~w;~n', [Select, From, Where, GroupBy, Having])
    ).

%% ============================================
%% LEFT JOIN SUPPORT (Phase 3)
%% ============================================

%% is_right_join_clause(+Body)
%  Check if clause body contains RIGHT JOIN pattern
%  Pattern: (LeftGoal ; Fallback), RightGoals
%  Disjunction at the START (before regular tables)
%
is_right_join_clause(Body) :-
    % Body must start with a disjunction
    Body = ((LeftGoal ; Fallback), _Rest),
    % Fallback must contain null bindings
    contains_null_binding(Fallback),
    % LeftGoal must be a table goal (not another disjunction)
    LeftGoal =.. [TableName|_],
    \+ TableName = (','),
    \+ TableName = (';').

%% is_full_outer_join_clause(+Body)
%  Check if clause body contains FULL OUTER JOIN pattern
%  Pattern: (TableA ; FallbackA), (TableB ; FallbackB)
%  Both sides are disjunctions with null bindings
%
is_full_outer_join_clause(Body) :-
    Body = ((LeftGoal ; LeftFallback), (RightGoal ; RightFallback)),
    % Both fallbacks must contain null bindings
    contains_null_binding(LeftFallback),
    contains_null_binding(RightFallback),
    % Both must be table goals
    LeftGoal =.. [LeftTable|_],
    RightGoal =.. [RightTable|_],
    \+ LeftTable = (','),
    \+ LeftTable = (';'),
    \+ RightTable = (','),
    \+ RightTable = (';').

%% is_left_join_clause(+Body)
%  Check if clause body contains LEFT JOIN pattern
%  Pattern: LeftGoals, (RightGoal ; Fallback)
%  The disjunction may be nested within conjunctions
%
is_left_join_clause(Body) :-
    % Must NOT be RIGHT or FULL OUTER (those take precedence)
    \+ is_right_join_clause(Body),
    \+ is_full_outer_join_clause(Body),
    % Search for a disjunction with null binding anywhere in the body
    find_disjunction_in_conjunction(Body, (_RightGoal ; Fallback)),
    contains_null_binding(Fallback).

%% find_disjunction_in_conjunction(+Body, -Disjunction)
%  Find a disjunction nested within conjunctions
%
find_disjunction_in_conjunction((A, B), Disj) :-
    % Check if A is a disjunction
    (   A = (_ ; _)
    ->  Disj = A
    % Otherwise recurse into B
    ;   find_disjunction_in_conjunction(B, Disj)
    ).
find_disjunction_in_conjunction((A ; B), (A ; B)).

%% split_at_disjunction(+Body, -Before, -Disjunction)
%  Split body into parts before and at the disjunction
%  For (A, (B, (C ; D))), returns Before=(A, B), Disjunction=(C ; D)
%
split_at_disjunction((A, B), Before, Disj) :-
    (   B = (_ ; _)
    ->  % Found it: B is the disjunction
        Before = A,
        Disj = B
    ;   % Recurse into B
        split_at_disjunction(B, BeforeB, Disj),
        (   BeforeB = true
        ->  Before = A
        ;   Before = (A, BeforeB)
        )
    ).
split_at_disjunction((A ; B), true, (A ; B)).

%% extract_all_disjunctions(+Body, -LeftGoals, -Disjunctions)
%  Extract all disjunctions in order, with goals before first disjunction
%  For (A, (B ; C), (D ; E)), returns LeftGoals=A, Disjunctions=[(B;C), (D;E)]
%
extract_all_disjunctions(Body, LeftPart, Disjunctions) :-
    extract_disjs_iter(Body, [], LeftPart, Disjunctions).

%% extract_disjs_iter(+Body, +Acc, -LeftGoals, -Disjunctions)
%  Iteratively extract disjunctions
%
extract_disjs_iter((A, B), Acc, LeftGoals, Disjs) :-
    (   A = (_ ; _)
    ->  % Found first disjunction
        (   Acc = []
        ->  LeftGoals = true
        ;   Acc = [Single]
        ->  LeftGoals = Single
        ;   reverse(Acc, ReversedAcc),
            list_to_conjunction(ReversedAcc, LeftGoals)
        ),
        collect_all_disjunctions((A, B), Disjs)
    ;   % Not a disjunction yet, accumulate
        extract_disjs_iter(B, [A|Acc], LeftGoals, Disjs)
    ).
extract_disjs_iter((A ; B), [], true, [(A ; B)]) :- !.
extract_disjs_iter((A ; B), Acc, LeftGoals, Disjs) :-
    % Reached a disjunction with accumulated left goals
    Acc \= [],
    (   Acc = [Single]
    ->  LeftGoals = Single
    ;   reverse(Acc, ReversedAcc),
        list_to_conjunction(ReversedAcc, LeftGoals)
    ),
    collect_all_disjunctions((A ; B), Disjs).
extract_disjs_iter(Goal, Acc, LeftGoals, []) :-
    % No disjunction found
    Goal \= (_ ; _),
    reverse([Goal|Acc], ReversedAcc),
    list_to_conjunction(ReversedAcc, LeftGoals).

%% collect_all_disjunctions(+Body, -Disjunctions)
%  Collect all disjunctions from remaining body
%
collect_all_disjunctions((A, B), [A|Rest]) :-
    A = (_ ; _), !,
    collect_all_disjunctions(B, Rest).
collect_all_disjunctions((A ; B), [(A ; B)]) :- !.
collect_all_disjunctions(_, []).

%% list_to_conjunction(+List, -Conjunction)
%  Convert list to conjunction
%
list_to_conjunction([], true).
list_to_conjunction([Goal], Goal) :- !.
list_to_conjunction([H|T], (H, Rest)) :-
    list_to_conjunction(T, Rest).

%% process_left_joins(+Disjunctions, +InitialLeftGoals, -JoinClauses, -AllRightGoals, -AllNullVars)
%  Process each disjunction iteratively to generate JOIN clauses
%
process_left_joins([], _, [], [], []).
process_left_joins([Disj|Rest], LeftGoals, [JoinClause|RestJoins], [RightGoal|RestRightGoals], AllNullVars) :-
    % Extract pattern
    Disj = (RightGoal ; Fallback),

    % Extract NULL bindings
    extract_null_bindings(Fallback, NullVars),

    % Generate LEFT JOIN clause
    generate_left_join_sql(LeftGoals, RightGoal, JoinClause),

    % Add RightGoal to LeftGoals for next iteration
    append(LeftGoals, [RightGoal], NewLeftGoals),

    % Recurse
    process_left_joins(Rest, NewLeftGoals, RestJoins, RestRightGoals, RestNullVars),

    % Accumulate NULL vars
    append(NullVars, RestNullVars, AllNullVars).

%% contains_null_binding(+Goal)
%  Check if goal contains X = null pattern
%
contains_null_binding((A, B)) :- !,
    (contains_null_binding(A) ; contains_null_binding(B)).
contains_null_binding(_ = null).

%% compile_left_join_clause(+Name, +Arity, +Body, +Head, +Options, -SQLCode)
%  Compile LEFT JOIN pattern to SQL
%  Supports multiple sequential disjunctions (nested LEFT JOINs)
%
compile_left_join_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    % Extract ALL disjunctions from body
    extract_all_disjunctions(Body, LeftPart, Disjunctions),

    % Verify we have at least one disjunction
    Disjunctions \= [],

    % Convert left part to list of goals
    conjunction_to_list(LeftPart, LeftGoals),

    % Parse head arguments
    Head =.. [Name|HeadArgs],

    % Separate left goals into tables and constraints
    separate_goals(LeftGoals, LeftTableGoals, LeftConstraints),

    % Generate FROM clause from left tables
    generate_from_clause(LeftTableGoals, FromClause),

    % Process all disjunctions iteratively to generate JOIN clauses
    process_left_joins(Disjunctions, LeftTableGoals, JoinClauses, AllRightGoals, AllNullVars),

    % Combine all JOIN clauses
    atomic_list_concat(JoinClauses, '\n', AllJoins),

    % Generate SELECT clause (include all tables and NULL-able columns)
    generate_select_for_nested_joins(HeadArgs, LeftTableGoals, AllRightGoals, AllNullVars, SelectClause),

    % Generate WHERE clause from constraints
    generate_where_clause(LeftConstraints, HeadArgs, LeftTableGoals, WhereClause),

    % Get output format and view name
    (   member(format(Format), Options)
    ->  true
    ;   Format = view
    ),
    (   member(view_name(ViewName), Options)
    ->  true
    ;   ViewName = Name
    ),

    % Combine into final SQL
    combine_left_join_sql(Format, ViewName, SelectClause, FromClause, AllJoins, WhereClause, SQLCode).

%% conjunction_to_list(+Conjunction, -Goals)
%  Convert conjunction to list of goals
%
conjunction_to_list(true, []) :- !.  % Empty list for true
conjunction_to_list((A, B), [A|Rest]) :- !,
    conjunction_to_list(B, Rest).
conjunction_to_list(Goal, [Goal]).

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

%% generate_left_join_sql(+LeftTableGoals, +RightGoal, -JoinClause)
%  Generate LEFT JOIN ON ... clause
%
generate_left_join_sql(LeftTableGoals, RightGoal, JoinClause) :-
    % Extract table name from right goal
    RightGoal =.. [RightTable|RightArgs],

    % Find shared variables (join keys)
    findall(Cond,
            (nth1(RightPos, RightArgs, RightArg),
             var(RightArg),
             % Find this variable in left goals
             member(LeftGoal, LeftTableGoals),
             LeftGoal =.. [LeftTable|LeftArgs],
             nth1(LeftPos, LeftArgs, LeftArg),
             LeftArg == RightArg,  % Same variable
             % Get column names
             get_column_name_from_schema(LeftTable, LeftPos, LeftCol),
             get_column_name_from_schema(RightTable, RightPos, RightCol),
             % Format condition
             format(atom(Cond), '~w.~w = ~w.~w', [RightTable, RightCol, LeftTable, LeftCol])
            ),
            JoinConditions),

    % Build JOIN clause
    (   JoinConditions = []
    ->  format(atom(JoinClause), 'LEFT JOIN ~w', [RightTable])
    ;   atomic_list_concat(JoinConditions, ' AND ', JoinCondStr),
        format(atom(JoinClause), 'LEFT JOIN ~w ON ~w', [RightTable, JoinCondStr])
    ).

%% extract_bound_vars_from_goals(+Goals, -BoundVars)
%  Extract variables from goals
%
extract_bound_vars_from_goals(Goals, BoundVars) :-
    findall(Var,
            (member(Goal, Goals),
             Goal =.. [_|Args],
             member(Arg, Args),
             var(Arg),
             Var = Arg),
            BoundVars).

%% member_with_position(+Var, +VarList, +Goals, -Column)
%  Find variable in goals and get its column name
%
member_with_position(Var, _VarList, Goals, Column) :-
    member(Goal, Goals),
    Goal =.. [TableName|Args],
    nth1(Pos, Args, Arg),
    Arg == Var, !,
    get_column_name_from_schema(TableName, Pos, ColName),
    format(atom(Column), '~w', [ColName]).

%% get_column_name_from_schema(+Table, +Position, -Name)
%  Get column name from table schema
%
get_column_name_from_schema(TableName, Position, ColName) :-
    % Look up schema
    sql_table_def(TableName, Columns),
    nth1(Position, Columns, ColSpec),
    (   ColSpec = (Name-_Type)
    ->  ColName = Name
    ;   ColName = ColSpec
    ), !.
get_column_name_from_schema(_Table, Position, ColName) :-
    % Fallback: generic column name
    format(atom(ColName), 'col~w', [Position]).

%% generate_select_for_left_join(+HeadArgs, +LeftGoals, +RightGoal, +NullVars, -SelectClause)
%  Generate SELECT clause for LEFT JOIN
%
generate_select_for_left_join(HeadArgs, LeftGoals, RightGoal, NullVars, SelectClause) :-
    % Extract right table info
    RightGoal =.. [RightTable|RightArgs],

    % Generate column list for head arguments
    findall(ColExpr,
            (nth1(Idx, HeadArgs, Arg),
             var(Arg),
             % Try to find in left goals first
             (   find_column_in_goals(Arg, LeftGoals, ColExpr)
             ->  true
             % Then try right goal
             ;   (   nth1(RightPos, RightArgs, RightArgMatch),
                     RightArgMatch == Arg,
                     get_column_name_from_schema(RightTable, RightPos, ColName),
                     format(atom(ColExpr), '~w.~w', [RightTable, ColName])
                 ->  true
                 % Fallback to unknown
                 ;   ColExpr = 'unknown'
                 )
             )),
            Columns),

    atomic_list_concat(Columns, ', ', ColStr),
    format(atom(SelectClause), 'SELECT ~w', [ColStr]).

%% generate_select_for_nested_joins(+HeadArgs, +LeftGoals, +RightGoals, +NullVars, -SelectClause)
%  Generate SELECT clause for nested LEFT JOINs (multiple right tables)
%
generate_select_for_nested_joins(HeadArgs, LeftGoals, RightGoals, NullVars, SelectClause) :-
    % Combine left and right goals
    append(LeftGoals, RightGoals, AllGoals),

    % Generate column list for head arguments
    findall(ColExpr,
            (nth1(Idx, HeadArgs, Arg),
             var(Arg),
             % Try to find in all goals (left + all right tables)
             (   find_column_in_goals(Arg, AllGoals, ColExpr)
             ->  true
             % Fallback to unknown
             ;   ColExpr = 'unknown'
             )),
            Columns),

    atomic_list_concat(Columns, ', ', ColStr),
    format(atom(SelectClause), 'SELECT ~w', [ColStr]).

%% member_check(+Var, +List)
%  Check if variable is in list (using ==)
%
member_check(Var, [H|_]) :- Var == H, !.
member_check(Var, [_|T]) :- member_check(Var, T).

%% find_column_in_goals(+Var, +Goals, -ColumnExpr)
%  Find column expression for variable in goals
%
find_column_in_goals(Var, Goals, ColumnExpr) :-
    member(Goal, Goals),
    Goal =.. [TableName|Args],
    nth1(Pos, Args, Arg),
    Arg == Var, !,
    get_column_name_from_schema(TableName, Pos, ColName),
    format(atom(ColumnExpr), '~w.~w', [TableName, ColName]).

%% combine_left_join_sql(+Format, +ViewName, +Select, +From, +Join, +Where, -SQL)
%  Combine clauses into final SQL
%
combine_left_join_sql(Format, ViewName, SelectClause, FromClause, JoinClause, WhereClause, SQL) :-
    % Build query
    (   WhereClause = ''
    ->  format(atom(Query), '~w~n~w~n~w', [SelectClause, FromClause, JoinClause])
    ;   format(atom(Query), '~w~n~w~n~w~n~w', [SelectClause, FromClause, JoinClause, WhereClause])
    ),

    % Wrap in view if requested
    (   Format = view
    ->  format(string(SQL), '-- View: ~w~nCREATE VIEW IF NOT EXISTS ~w AS~n~w;~n~n',
               [ViewName, ViewName, Query])
    ;   format(string(SQL), '~w;~n', [Query])
    ).

%% ============================================
%% RIGHT JOIN SUPPORT (Phase 3d)
%% ============================================

%% extract_leading_disjunctions(+Body, -LeadingDisjunctions, -RemainingGoals)
%  Extract all leading disjunctions (for RIGHT JOIN chains)
%  Pattern: (A ; ...), (B ; ...), C, D → LeadingDisjs = [(A;...), (B;...)], Remaining = [C, D]
%
extract_leading_disjunctions(Body, LeadingDisjunctions, RemainingGoals) :-
    extract_leading_disjs_iter(Body, [], LeadingDisjunctions, RemainingGoals).

extract_leading_disjs_iter((Disj, Rest), Acc, AllDisjs, Remaining) :-
    Disj = (_ ; _),  % Is a disjunction
    !,
    extract_leading_disjs_iter(Rest, [Disj|Acc], AllDisjs, Remaining).
extract_leading_disjs_iter((Disj ; _) = FinalDisj, Acc, AllDisjs, []) :-
    % Final goal is a disjunction
    !,
    reverse([FinalDisj|Acc], AllDisjs).
extract_leading_disjs_iter(Rest, Acc, AllDisjs, Remaining) :-
    % Hit a non-disjunction
    Acc \= [],
    reverse(Acc, AllDisjs),
    conjunction_to_list(Rest, Remaining).

%% process_right_joins(+Goals, +AccTables, -JoinClauses, -AllTables, -AllNullVars)
%  Process goals iteratively to generate RIGHT JOIN clauses
%  Goals can be disjunctions (table ; null) or regular table goals
%
process_right_joins([], _, [], [], []).
process_right_joins([Goal|Rest], AccTables, [JoinClause|RestJoins], [Table|RestTables], AllNullVars) :-
    % Check if Goal is a disjunction
    (   Goal = (TableGoal ; Fallback)
    ->  % Disjunction - extract table and NULL bindings
        extract_null_bindings(Fallback, NullVars),
        Table = TableGoal
    ;   % Regular table goal
        NullVars = [],
        Table = Goal
    ),

    % Generate RIGHT JOIN for this table
    generate_right_join_sql(AccTables, Table, JoinClause),

    % Add this table to accumulated
    append(AccTables, [Table], NewAccTables),

    % Recurse
    process_right_joins(Rest, NewAccTables, RestJoins, RestTables, RestNullVars),

    % Accumulate NULL vars
    append(NullVars, RestNullVars, AllNullVars).

%% compile_right_join_clause(+Name, +Arity, +Body, +Head, +Options, -SQLCode)
%  Compile RIGHT JOIN pattern to SQL
%  Pattern: (LeftGoal ; Fallback), RightGoals (supports chains with multiple disjunctions)
%
compile_right_join_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    % Extract ALL leading disjunctions and remaining goals
    extract_leading_disjunctions(Body, LeadingDisjunctions, RemainingGoals),

    % Verify we have at least one leading disjunction
    LeadingDisjunctions \= [],

    % First disjunction becomes FROM table
    LeadingDisjunctions = [FirstDisj|RestDisjs],
    FirstDisj = (FirstTable ; FirstFallback),
    extract_null_bindings(FirstFallback, FirstNullVars),

    % Generate FROM clause
    generate_from_clause([FirstTable], FromClause),

    % Combine remaining disjunctions with regular goals
    append(RestDisjs, RemainingGoals, AllRightGoals),

    % Process all right goals to generate RIGHT JOINs
    (   AllRightGoals = []
    ->  AllJoins = '',
        AllRightTables = [],
        RestNullVars = []
    ;   process_right_joins(AllRightGoals, [FirstTable], JoinClauses, AllRightTables, RestNullVars),
        atomic_list_concat(JoinClauses, '\n', AllJoins)
    ),

    % Combine NULL vars
    append(FirstNullVars, RestNullVars, AllNullVars),

    % Parse head arguments
    Head =.. [Name|HeadArgs],

    % Generate SELECT clause
    AllTableGoals = [FirstTable|AllRightTables],
    generate_select_for_nested_joins(HeadArgs, [FirstTable], AllRightTables, AllNullVars, SelectClause),

    % Generate WHERE clause (no constraints in simple RIGHT JOIN chains)
    WhereClause = '',

    % Get output format
    (   member(format(Format), Options)
    ->  true
    ;   Format = view
    ),
    (   member(view_name(ViewName), Options)
    ->  true
    ;   ViewName = Name
    ),

    % Combine into final SQL
    combine_left_join_sql(Format, ViewName, SelectClause, FromClause, AllJoins, WhereClause, SQLCode).

%% generate_right_join_chain(+AccTables, +RightTables, -JoinClauses)
%  Generate a chain of RIGHT JOIN clauses
%
generate_right_join_chain(_, [], []).
generate_right_join_chain(AccTables, [RightTable|Rest], [JoinClause|RestClauses]) :-
    % Generate RIGHT JOIN for this table
    generate_right_join_sql(AccTables, RightTable, JoinClause),
    % Add this table to accumulated tables
    append(AccTables, [RightTable], NewAccTables),
    % Recursively process remaining tables
    generate_right_join_chain(NewAccTables, Rest, RestClauses).

%% generate_right_join_sql(+LeftTables, +RightGoal, -JoinClause)
%  Generate RIGHT JOIN clause
%
generate_right_join_sql(LeftTables, RightGoal, JoinClause) :-
    RightGoal =.. [RightTableName|RightArgs],

    % Find shared variables (join keys)
    findall(Cond,
            (nth1(RightPos, RightArgs, RightArg),
             var(RightArg),
             % Find this variable in left tables
             member(LeftTable, LeftTables),
             LeftTable =.. [LeftTableName|LeftArgs],
             nth1(LeftPos, LeftArgs, LeftArg),
             LeftArg == RightArg,  % Same variable
             % Get column names
             get_column_name_from_schema(LeftTableName, LeftPos, LeftCol),
             get_column_name_from_schema(RightTableName, RightPos, RightCol),
             % Format condition
             format(atom(Cond), '~w.~w = ~w.~w',
                    [RightTableName, RightCol, LeftTableName, LeftCol])
            ),
            JoinConditions),

    % Build RIGHT JOIN clause
    (   JoinConditions = []
    ->  format(atom(JoinClause), 'RIGHT JOIN ~w', [RightTableName])
    ;   atomic_list_concat(JoinConditions, ' AND ', JoinCondStr),
        format(atom(JoinClause), 'RIGHT JOIN ~w ON ~w', [RightTableName, JoinCondStr])
    ).

%% ============================================
%% FULL OUTER JOIN SUPPORT (Phase 3d)
%% ============================================

%% compile_full_outer_join_clause(+Name, +Arity, +Body, +Head, +Options, -SQLCode)
%  Compile FULL OUTER JOIN pattern to SQL
%  Pattern: (LeftGoal ; LeftFallback), (RightGoal ; RightFallback)
%
compile_full_outer_join_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    % Extract pattern
    Body = ((LeftGoal ; LeftFallback), (RightGoal ; RightFallback)),

    % Extract NULL bindings
    extract_null_bindings(LeftFallback, LeftNullVars),
    extract_null_bindings(RightFallback, RightNullVars),
    append(LeftNullVars, RightNullVars, AllNullVars),

    % Parse head arguments
    Head =.. [Name|HeadArgs],

    % Get table names
    LeftGoal =.. [LeftTableName|LeftArgs],
    RightGoal =.. [RightTableName|RightArgs],

    % Generate FROM clause (left table)
    generate_from_clause([LeftGoal], FromClause),

    % Generate FULL OUTER JOIN
    generate_full_outer_join_sql(LeftGoal, RightGoal, JoinClause),

    % Generate SELECT clause
    generate_select_for_nested_joins(HeadArgs, [LeftGoal], [RightGoal], AllNullVars, SelectClause),

    % No WHERE clause for simple FULL OUTER
    WhereClause = '',

    % Get output format
    (   member(format(Format), Options)
    ->  true
    ;   Format = view
    ),
    (   member(view_name(ViewName), Options)
    ->  true
    ;   ViewName = Name
    ),

    % Combine into final SQL
    combine_left_join_sql(Format, ViewName, SelectClause, FromClause, JoinClause, WhereClause, SQLCode).

%% generate_full_outer_join_sql(+LeftGoal, +RightGoal, -JoinClause)
%  Generate FULL OUTER JOIN clause
%
generate_full_outer_join_sql(LeftGoal, RightGoal, JoinClause) :-
    LeftGoal =.. [LeftTableName|LeftArgs],
    RightGoal =.. [RightTableName|RightArgs],

    % Find shared variables
    findall(Cond,
            (nth1(RightPos, RightArgs, RightArg),
             var(RightArg),
             nth1(LeftPos, LeftArgs, LeftArg),
             LeftArg == RightArg,
             get_column_name_from_schema(LeftTableName, LeftPos, LeftCol),
             get_column_name_from_schema(RightTableName, RightPos, RightCol),
             format(atom(Cond), '~w.~w = ~w.~w',
                    [RightTableName, RightCol, LeftTableName, LeftCol])
            ),
            JoinConditions),

    % Build FULL OUTER JOIN clause
    (   JoinConditions = []
    ->  format(atom(JoinClause), 'FULL OUTER JOIN ~w', [RightTableName])
    ;   atomic_list_concat(JoinConditions, ' AND ', JoinCondStr),
        format(atom(JoinClause), 'FULL OUTER JOIN ~w ON ~w', [RightTableName, JoinCondStr])
    ).

%% ============================================
%% REGULAR QUERY SUPPORT
%% ============================================

%% parse_clause_body(+Body, -Goals)
%  Parse clause body into list of goals
%
parse_clause_body(true, []) :- !.
parse_clause_body((Goal1, Goal2), Goals) :- !,
    parse_clause_body(Goal1, Goals1),
    parse_clause_body(Goal2, Goals2),
    append(Goals1, Goals2, Goals).
parse_clause_body(Goal, [Goal]).

%% separate_goals(+Goals, -TableGoals, -Constraints)
%  Separate table lookups from constraints
%
separate_goals([], [], []).
separate_goals([Goal|Rest], [Goal|TableGoals], Constraints) :-
    is_table_goal(Goal), !,
    separate_goals(Rest, TableGoals, Constraints).
separate_goals([Constraint|Rest], TableGoals, [Constraint|Constraints]) :-
    separate_goals(Rest, TableGoals, Constraints).

%% is_table_goal(+Goal)
%  Check if a goal is a table lookup
%
is_table_goal(Goal) :-
    functor(Goal, TableName, _),
    sql_table_def(TableName, _), !.

%% generate_select_clause(+Args, +TableGoals, -SelectClause)
%  Generate SELECT clause from head arguments
%
generate_select_clause(Args, TableGoals, SelectClause) :-
    generate_select_items(Args, TableGoals, Items),
    atomic_list_concat(Items, ', ', ItemsStr),
    format(string(SelectClause), 'SELECT ~w', [ItemsStr]).

%% generate_select_items(+Args, +TableGoals, -Items)
%  Generate SELECT items with column names from table goals
%
generate_select_items([], _, []).
generate_select_items([Arg|Rest], TableGoals, [Item|Items]) :-
    (   var(Arg)
    ->  % Find column name for this variable
        (   find_var_column(Arg, [], TableGoals, ColumnName)
        ->  Item = ColumnName
        ;   Item = 'unknown'
        )
    ;   atom(Arg)
    ->  format(string(Item), "'~w'", [Arg])
    ;   number(Arg)
    ->  Item = Arg
    ;   format(string(Item), "'~w'", [Arg])
    ),
    generate_select_items(Rest, TableGoals, Items).

%% generate_from_clause(+TableGoals, -FromClause)
%  Generate FROM clause from table goals
%
generate_from_clause([], '') :- !.
generate_from_clause([Goal|Rest], FromClause) :-
    functor(Goal, TableName, _),
    (   Rest = []
    ->  format(string(FromClause), 'FROM ~w', [TableName])
    ;   % Multiple tables - generate JOINs (future)
        generate_from_clause_multi([Goal|Rest], FromClause)
    ).

%% generate_from_clause_multi(+Goals, -FromClause)
%  Generate FROM clause with INNER JOINs based on shared variables
%
generate_from_clause_multi([FirstGoal|RestGoals], FromClause) :-
    functor(FirstGoal, FirstTable, _),
    % Find join conditions between tables
    find_join_conditions([FirstGoal|RestGoals], JoinSpecs),
    % Generate JOIN clauses
    (   JoinSpecs = []
    ->  % No shared variables - use CROSS JOIN
        findall(TN, member(G, [FirstGoal|RestGoals]), (functor(G, TN, _)), Tables),
        atomic_list_concat(Tables, ', ', TablesStr),
        format(string(FromClause), 'FROM ~w', [TablesStr])
    ;   % Generate INNER JOINs
        generate_join_clause(FirstTable, RestGoals, JoinSpecs, FromClause)
    ).

%% find_join_conditions(+Goals, -JoinSpecs)
%  Find shared variables between table goals
%  Returns list of join_spec(Table1, Col1, Table2, Col2)
%
find_join_conditions(Goals, JoinSpecs) :-
    find_all_join_pairs(Goals, AllSpecs),
    % Remove duplicates (A-B and B-A are same join)
    remove_duplicate_joins(AllSpecs, JoinSpecs).

%% find_all_join_pairs(+Goals, -JoinSpecs)
%  Find all join pairs between goals
%
find_all_join_pairs([], []).
find_all_join_pairs([G1|Rest], JoinSpecs) :-
    G1 =.. [T1|Args1],
    findall(join_spec(T1, C1, T2, C2),
            (   member(G2, Rest),
                G2 =.. [T2|Args2],
                % Find position in Args1 that has a variable
                nth1(Pos1, Args1, Var1),
                var(Var1),
                % Check if same variable instance appears in Args2
                nth1(Pos2, Args2, Var2),
                Var1 == Var2,  % Same variable instance (by identity)
                % Get column names
                get_table_schema(T1, Schema1),
                get_table_schema(T2, Schema2),
                nth1(Pos1, Schema1, C1-_),
                nth1(Pos2, Schema2, C2-_)
            ),
            Joins1),
    find_all_join_pairs(Rest, Joins2),
    append(Joins1, Joins2, JoinSpecs).

%% remove_duplicate_joins(+Specs, -UniqueSpecs)
%  Remove duplicate join specifications
%
remove_duplicate_joins([], []).
remove_duplicate_joins([join_spec(T1, C1, T2, C2)|Rest], [join_spec(T1, C1, T2, C2)|Unique]) :-
    % Remove reverse join
    \+ member(join_spec(T2, C2, T1, C1), Rest),
    remove_duplicate_joins(Rest, Unique).
remove_duplicate_joins([join_spec(T1, C1, T2, C2)|Rest], Unique) :-
    % Skip if reverse exists
    member(join_spec(T2, C2, T1, C1), Rest),
    remove_duplicate_joins(Rest, Unique).

%% generate_join_clause(+FirstTable, +RestGoals, +JoinSpecs, -FromClause)
%  Generate FROM clause with INNER JOIN (iterative for chains)
%
generate_join_clause(FirstTable, RestGoals, JoinSpecs, FromClause) :-
    % Generate joins iteratively, accumulating joined tables
    generate_join_clauses_iter(RestGoals, [FirstTable], JoinSpecs, JoinClauses),
    atomic_list_concat(JoinClauses, '\n', JoinsStr),
    format(string(FromClause), 'FROM ~w\n~w', [FirstTable, JoinsStr]).

%% generate_join_clauses_iter(+RemainingGoals, +AccTables, +JoinSpecs, -JoinClauses)
%  Iteratively generate JOIN clauses, accumulating joined tables
%
generate_join_clauses_iter([], _, _, []).
generate_join_clauses_iter([G|RestGoals], AccTables, JoinSpecs, [JoinClause|RestJoins]) :-
    functor(G, TableName, _),
    % Find join condition with ANY previously joined table
    (   find_join_to_accumulated(TableName, AccTables, JoinSpecs, LeftTable, Col1, Col2)
    ->  format(string(JoinClause), 'INNER JOIN ~w ON ~w.~w = ~w.~w',
              [TableName, LeftTable, Col1, TableName, Col2])
    ;   % No join condition - CROSS JOIN
        format(string(JoinClause), 'CROSS JOIN ~w', [TableName])
    ),
    % Add this table to accumulated
    append(AccTables, [TableName], NewAccTables),
    % Recurse
    generate_join_clauses_iter(RestGoals, NewAccTables, JoinSpecs, RestJoins).

%% find_join_to_accumulated(+TableName, +AccTables, +JoinSpecs, -LeftTable, -Col1, -Col2)
%  Find join between TableName and any accumulated table
%
find_join_to_accumulated(TableName, AccTables, JoinSpecs, LeftTable, Col1, Col2) :-
    member(LeftTable, AccTables),
    (   member(join_spec(LeftTable, Col1, TableName, Col2), JoinSpecs)
    ->  true
    ;   member(join_spec(TableName, Col2, LeftTable, Col1), JoinSpecs)
    ).

%% generate_where_clause(+Constraints, +HeadArgs, +TableGoals, -WhereClause)
%  Generate WHERE clause from constraints and table goal constants
%
generate_where_clause(Constraints, HeadArgs, TableGoals, WhereClause) :-
    % Get conditions from explicit constraints
    findall(Condition,
            (member(C, Constraints), constraint_to_sql(C, HeadArgs, TableGoals, Condition)),
            ConstraintConditions),

    % Get conditions from constant arguments in table goals
    findall(Condition,
            (member(Goal, TableGoals), table_goal_to_conditions(Goal, Condition)),
            TableConditions),

    % Combine all conditions
    append(ConstraintConditions, TableConditions, AllConditions),

    (   AllConditions = []
    ->  WhereClause = ''
    ;   atomic_list_concat(AllConditions, ' AND ', ConditionsStr),
        format(string(WhereClause), 'WHERE ~w', [ConditionsStr])
    ).

%% constraint_to_sql(+Constraint, +HeadArgs, +TableGoals, -SQLCondition)
%  Convert Prolog constraint to SQL condition
%
constraint_to_sql(Var = Value, HeadArgs, TableGoals, Condition) :-
    var(Var),
    (   atom(Value)
    ->  find_var_column(Var, HeadArgs, TableGoals, Column),
        format(string(Condition), "~w = '~w'", [Column, Value])
    ;   number(Value)
    ->  find_var_column(Var, HeadArgs, TableGoals, Column),
        format(string(Condition), '~w = ~w', [Column, Value])
    ;   % String literal
        find_var_column(Var, HeadArgs, TableGoals, Column),
        format(string(Condition), "~w = '~w'", [Column, Value])
    ).

constraint_to_sql(Var >= Value, HeadArgs, TableGoals, Condition) :-
    var(Var),
    find_var_column(Var, HeadArgs, TableGoals, Column),
    format(string(Condition), '~w >= ~w', [Column, Value]).

constraint_to_sql(Var > Value, HeadArgs, TableGoals, Condition) :-
    var(Var),
    find_var_column(Var, HeadArgs, TableGoals, Column),
    format(string(Condition), '~w > ~w', [Column, Value]).

constraint_to_sql(Var =< Value, HeadArgs, TableGoals, Condition) :-
    var(Var),
    find_var_column(Var, HeadArgs, TableGoals, Column),
    format(string(Condition), '~w <= ~w', [Column, Value]).

constraint_to_sql(Var < Value, HeadArgs, TableGoals, Condition) :-
    var(Var),
    find_var_column(Var, HeadArgs, TableGoals, Column),
    format(string(Condition), '~w < ~w', [Column, Value]).

constraint_to_sql(Var =:= Value, HeadArgs, TableGoals, Condition) :-
    var(Var),
    find_var_column(Var, HeadArgs, TableGoals, Column),
    format(string(Condition), '~w = ~w', [Column, Value]).

constraint_to_sql(Var =\= Value, HeadArgs, TableGoals, Condition) :-
    var(Var),
    find_var_column(Var, HeadArgs, TableGoals, Column),
    format(string(Condition), '~w != ~w', [Column, Value]).

%% table_goal_to_conditions(+Goal, -Condition)
%  Extract WHERE conditions from constant arguments in table goals
%
table_goal_to_conditions(Goal, Condition) :-
    Goal =.. [TableName|Args],
    get_table_schema(TableName, Schema),
    table_args_to_conditions(Args, Schema, 1, Condition).

%% table_args_to_conditions(+Args, +Schema, +Position, -Condition)
%  Generate condition for each constant argument
%
table_args_to_conditions([], _, _, _) :- fail.  % No more args
table_args_to_conditions([Arg|Rest], Schema, Pos, Condition) :-
    (   \+ var(Arg),  % Constant or string
        Arg \= '_'    % Not anonymous variable
    ->  % Generate condition for this constant
        nth1(Pos, Schema, ColumnName-_Type),
        (   atom(Arg)
        ->  format(string(Condition), "~w = '~w'", [ColumnName, Arg])
        ;   number(Arg)
        ->  format(string(Condition), '~w = ~w', [ColumnName, Arg])
        ;   % String
            format(string(Condition), "~w = '~w'", [ColumnName, Arg])
        )
    ;   % Variable - try next argument
        Pos1 is Pos + 1,
        table_args_to_conditions(Rest, Schema, Pos1, Condition)
    ).

%% find_var_column(+Var, +HeadArgs, +TableGoals, -ColumnName)
%  Find the SQL column name for a Prolog variable
%  Returns table.column when multiple tables involved
%
find_var_column(Var, HeadArgs, TableGoals, ColumnName) :-
    % Find which table goal contains this variable
    member(Goal, TableGoals),
    Goal =.. [TableName|Args],
    nth1(Position, Args, Arg),
    Arg == Var, !,
    % Get column name from schema
    get_table_schema(TableName, Schema),
    nth1(Position, Schema, ColName-_Type),
    % Qualify with table name if multiple tables
    (   length(TableGoals, N), N > 1
    ->  format(atom(ColumnName), '~w.~w', [TableName, ColName])
    ;   ColumnName = ColName
    ).

%% format_sql(+Format, +ViewName, +Select, +From, +Where, -SQL)
%  Format the final SQL output
%
format_sql(view, ViewName, Select, From, Where, SQL) :-
    (   Where = ''
    ->  format(string(SQL), '-- View: ~w~nCREATE VIEW IF NOT EXISTS ~w AS~n~w~n~w;~n~n',
               [ViewName, ViewName, Select, From])
    ;   format(string(SQL), '-- View: ~w~nCREATE VIEW IF NOT EXISTS ~w AS~n~w~n~w~n~w;~n~n',
               [ViewName, ViewName, Select, From, Where])
    ).

format_sql(select, _, Select, From, Where, SQL) :-
    (   Where = ''
    ->  format(string(SQL), '~w~n~w;~n', [Select, From])
    ;   format(string(SQL), '~w~n~w~n~w;~n', [Select, From, Where])
    ).

format_sql(cte, ViewName, Select, From, Where, SQL) :-
    (   Where = ''
    ->  format(string(SQL), 'WITH ~w AS (~n  ~w~n  ~w~n)', [ViewName, Select, From])
    ;   format(string(SQL), 'WITH ~w AS (~n  ~w~n  ~w~n  ~w~n)', [ViewName, Select, From, Where])
    ).

%% compile_multiple_clauses(+Name, +Arity, +Clauses, +Head, +Options, -SQLCode)
%  Handle multiple clauses with UNION (future)
%
compile_multiple_clauses(Name, Arity, Clauses, Head, Options, SQLCode) :-
    % For now, just use the first clause
    Clauses = [FirstClause|_],
    compile_single_clause(Name, Arity, FirstClause, Head, Options, SQLCode).

%% ============================================
%% FILE I/O
%% ============================================

%% write_sql_file(+SQLCode, +FilePath)
%  Write SQL code to file
%
write_sql_file(SQLCode, FilePath) :-
    open(FilePath, write, Stream, [encoding(utf8)]),
    format(Stream, '-- Generated by UnifyWeaver SQL Target~n', []),
    format(Stream, '-- Timestamp: ~w~n~n', [timestamp]),
    format(Stream, '~w', [SQLCode]),
    close(Stream),
    format('SQL written to: ~w~n', [FilePath]).
