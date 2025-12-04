:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% sql_target.pl - SQL Target for UnifyWeaver
% Generates SQL queries (VIEWs, CTEs) from Prolog predicates
% Supports SQLite, PostgreSQL, MySQL, and other SQL databases

:- module(sql_target, [
    compile_predicate_to_sql/3,     % +Predicate, +Options, -SQLCode
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
%  Compile multiple clauses to SQL
%
compile_clauses_to_sql(Name, Arity, HeadBodyPairs, Options, SQLCode) :-
    % For now, we only support single-clause predicates (Phase 1)
    (   HeadBodyPairs = [head_body(Head, Body)]
    ->  compile_single_clause(Name, Arity, Body, Head, Options, SQLCode)
    ;   length(HeadBodyPairs, N),
        format('WARNING: Multiple clauses not yet supported. Found ~w clauses for ~w/~w~n', [N, Name, Arity]),
        % Use UNION for multiple clauses (future)
        HeadBodyPairs = [head_body(FirstHead, FirstBody)|_],
        compile_single_clause(Name, Arity, FirstBody, FirstHead, Options, SQLCode)
    ).

%% compile_single_clause(+Name, +Arity, +Body, +Head, +Options, -SQLCode)
%  Compile a single Prolog clause to SQL
%
compile_single_clause(Name, Arity, Body, Head, Options, SQLCode) :-
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
    format_sql(Format, ViewName, SelectClause, FromClause, WhereClause, SQLCode).

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
%  Generate FROM clause with multiple tables (simple CROSS JOIN for now)
%
generate_from_clause_multi(Goals, FromClause) :-
    findall(TableName, (member(G, Goals), functor(G, TableName, _)), Tables),
    atomic_list_concat(Tables, ', ', TablesStr),
    format(string(FromClause), 'FROM ~w', [TablesStr]).

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
%
find_var_column(Var, HeadArgs, TableGoals, ColumnName) :-
    % Find which table goal contains this variable
    member(Goal, TableGoals),
    Goal =.. [TableName|Args],
    nth1(Position, Args, Arg),
    Arg == Var, !,
    % Get column name from schema
    get_table_schema(TableName, Schema),
    nth1(Position, Schema, ColumnName-_Type).

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
