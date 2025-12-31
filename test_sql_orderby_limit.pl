:- encoding(utf8).
% Test SQL ORDER BY and LIMIT/OFFSET support
% Tests for sql_order_by, sql_limit, sql_offset predicates

:- use_module('src/unifyweaver/targets/sql_target').

%% ============================================
%% TABLE SCHEMAS
%% ============================================

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer]).
:- sql_table(products, [id-integer, name-text, price-real, stock-integer]).

%% ============================================
%% TEST 1: Simple ORDER BY (ascending)
%% ============================================

employees_by_name(Name, Salary) :-
    employees(_, Name, _, Salary),
    sql_order_by(Name).

test1 :-
    format('~n=== Test 1: Simple ORDER BY (ascending) ===~n'),
    format('Pattern: sql_order_by(Name)~n~n'),
    compile_predicate_to_sql(employees_by_name/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 2: ORDER BY descending
%% ============================================

employees_by_salary_desc(Name, Salary) :-
    employees(_, Name, _, Salary),
    sql_order_by(Salary, desc).

test2 :-
    format('~n=== Test 2: ORDER BY descending ===~n'),
    format('Pattern: sql_order_by(Salary, desc)~n~n'),
    compile_predicate_to_sql(employees_by_salary_desc/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 3: Multiple ORDER BY columns
%% ============================================

employees_by_dept_salary(Name, Dept, Salary) :-
    employees(_, Name, Dept, Salary),
    sql_order_by(Dept, asc),
    sql_order_by(Salary, desc).

test3 :-
    format('~n=== Test 3: Multiple ORDER BY columns ===~n'),
    format('Pattern: sql_order_by(Dept, asc), sql_order_by(Salary, desc)~n~n'),
    compile_predicate_to_sql(employees_by_dept_salary/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 4: Simple LIMIT
%% ============================================

top_5_employees(Name, Salary) :-
    employees(_, Name, _, Salary),
    sql_order_by(Salary, desc),
    sql_limit(5).

test4 :-
    format('~n=== Test 4: Simple LIMIT ===~n'),
    format('Pattern: sql_order_by(Salary, desc), sql_limit(5)~n~n'),
    compile_predicate_to_sql(top_5_employees/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 5: LIMIT with OFFSET (pagination)
%% ============================================

employees_page_2(Name, Salary) :-
    employees(_, Name, _, Salary),
    sql_order_by(Name),
    sql_limit(10),
    sql_offset(10).

test5 :-
    format('~n=== Test 5: LIMIT with OFFSET (pagination) ===~n'),
    format('Pattern: sql_limit(10), sql_offset(10)~n~n'),
    compile_predicate_to_sql(employees_page_2/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 6: ORDER BY with WHERE clause
%% ============================================

high_earners_sorted(Name, Salary) :-
    employees(_, Name, _, Salary),
    Salary > 50000,
    sql_order_by(Salary, desc).

test6 :-
    format('~n=== Test 6: ORDER BY with WHERE clause ===~n'),
    format('Pattern: Salary > 50000, sql_order_by(Salary, desc)~n~n'),
    compile_predicate_to_sql(high_earners_sorted/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 7: ORDER BY + LIMIT + WHERE
%% ============================================

top_3_high_earners(Name, Salary) :-
    employees(_, Name, _, Salary),
    Salary > 40000,
    sql_order_by(Salary, desc),
    sql_limit(3).

test7 :-
    format('~n=== Test 7: ORDER BY + LIMIT + WHERE ===~n'),
    format('Pattern: Salary > 40000, sql_order_by(Salary, desc), sql_limit(3)~n~n'),
    compile_predicate_to_sql(top_3_high_earners/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 8: OFFSET only (skip first N)
%% ============================================

skip_first_5(Name, Salary) :-
    employees(_, Name, _, Salary),
    sql_order_by(Name),
    sql_offset(5).

test8 :-
    format('~n=== Test 8: OFFSET only ===~n'),
    format('Pattern: sql_order_by(Name), sql_offset(5)~n~n'),
    compile_predicate_to_sql(skip_first_5/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 9: ORDER BY with window function
%% ============================================

ranked_employees_sorted(Name, Salary, Rank) :-
    employees(_, Name, _, Salary),
    rank(Rank, [order_by(Salary, desc)]),
    sql_order_by(Salary, desc),
    sql_limit(10).

test9 :-
    format('~n=== Test 9: ORDER BY with window function ===~n'),
    format('Pattern: rank(...), sql_order_by(Salary, desc), sql_limit(10)~n~n'),
    compile_predicate_to_sql(ranked_employees_sorted/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 10: Output as SELECT (not VIEW)
%% ============================================

test10 :-
    format('~n=== Test 10: Output as SELECT (not VIEW) ===~n'),
    format('Pattern: Using [format(select)] option~n~n'),
    compile_predicate_to_sql(top_5_employees/2, [format(select)], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% RUN ALL TESTS
%% ============================================

test_all :-
    format('~n========================================~n'),
    format('  SQL ORDER BY + LIMIT/OFFSET Tests~n'),
    format('========================================~n'),
    test1,
    test2,
    test3,
    test4,
    test5,
    test6,
    test7,
    test8,
    test9,
    test10,
    format('~n========================================~n'),
    format('  All tests completed!~n'),
    format('========================================~n').

%% Entry point
main :- test_all.

:- initialization(main, main).
