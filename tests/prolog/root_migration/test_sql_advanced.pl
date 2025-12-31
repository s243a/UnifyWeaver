:- encoding(utf8).
% Test SQL Advanced Features
% Tests for DISTINCT, CASE WHEN, Scalar Subqueries, Aliases, CTEs

:- use_module('src/unifyweaver/targets/sql_target').

%% ============================================
%% TABLE SCHEMAS
%% ============================================

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, status-text]).
:- sql_table(departments, [id-integer, name-text, budget-integer]).
:- sql_table(orders, [id-integer, customer_id-integer, amount-integer]).
:- sql_table(customers, [id-integer, name-text, city-text]).

%% ============================================
%% TEST 1: DISTINCT
%% ============================================

unique_departments(Dept) :-
    employees(_, _, Dept, _, _),
    sql_distinct.

test1 :-
    format('~n=== Test 1: DISTINCT ===~n'),
    format('Pattern: sql_distinct~n~n'),
    compile_predicate_to_sql(unique_departments/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 2: DISTINCT with ORDER BY
%% ============================================

unique_departments_sorted(Dept) :-
    employees(_, _, Dept, _, _),
    sql_distinct,
    sql_order_by(Dept).

test2 :-
    format('~n=== Test 2: DISTINCT with ORDER BY ===~n'),
    format('Pattern: sql_distinct, sql_order_by(Dept)~n~n'),
    compile_predicate_to_sql(unique_departments_sorted/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 3: CTEs (WITH clause)
%% ============================================

% Helper predicate for CTE
high_earners(Name, Salary) :-
    employees(_, Name, _, Salary, _),
    Salary > 50000.

% Main query using CTE
test3 :-
    format('~n=== Test 3: CTE (WITH clause) ===~n'),
    format('Pattern: compile_with_cte([cte(high_earners, high_earners/2)], ...)~n~n'),
    compile_with_cte(
        [cte(high_earners, high_earners/2)],
        high_earners/2,
        [],
        SQL
    ),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 4: Multiple CTEs
%% ============================================

% Second helper for CTEs
low_earners(Name, Salary) :-
    employees(_, Name, _, Salary, _),
    Salary =< 30000.

test4 :-
    format('~n=== Test 4: Multiple CTEs ===~n'),
    format('Pattern: compile_with_cte([cte(high_earners, ...), cte(low_earners, ...)], ...)~n~n'),
    compile_with_cte(
        [cte(high_earners, high_earners/2), cte(low_earners, low_earners/2)],
        high_earners/2,
        [],
        SQL
    ),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 5: DISTINCT with WHERE
%% ============================================

unique_cities_active(City) :-
    customers(_, _, City),
    sql_distinct.

test5 :-
    format('~n=== Test 5: DISTINCT with simple query ===~n'),
    compile_predicate_to_sql(unique_cities_active/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 6: DISTINCT with Window Function
%% ============================================

ranked_unique_depts(Dept, Rank) :-
    employees(_, _, Dept, Salary, _),
    sql_distinct,
    rank(Rank, [order_by(Salary, desc)]).

test6 :-
    format('~n=== Test 6: DISTINCT with Window Function ===~n'),
    format('Pattern: sql_distinct + rank(...)~n~n'),
    compile_predicate_to_sql(ranked_unique_depts/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 7: CTE with View Name
%% ============================================

test7 :-
    format('~n=== Test 7: CTE with View Name ===~n'),
    compile_with_cte(
        [cte(top_employees, high_earners/2)],
        high_earners/2,
        [view_name(employee_report)],
        SQL
    ),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 8: DISTINCT output as SELECT
%% ============================================

test8 :-
    format('~n=== Test 8: DISTINCT output as SELECT ===~n'),
    compile_predicate_to_sql(unique_departments/1, [format(select)], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% RUN ALL TESTS
%% ============================================

test_all :-
    format('~n========================================~n'),
    format('  SQL Advanced Features Tests~n'),
    format('  DISTINCT, CTEs~n'),
    format('========================================~n'),
    test1,
    test2,
    test3,
    test4,
    test5,
    test6,
    test7,
    test8,
    format('~n========================================~n'),
    format('  All tests completed!~n'),
    format('========================================~n').

%% Entry point
main :- test_all.

:- initialization(main, main).
