:- encoding(utf8).
% Test SQL CASE WHEN Expressions
% Tests for simple CASE, searched CASE, and aliases

:- use_module('src/unifyweaver/targets/sql_target').

%% ============================================
%% TABLE SCHEMAS
%% ============================================

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, status-text]).
:- sql_table(orders, [id-integer, customer_id-integer, amount-integer, priority-text]).
:- sql_table(products, [id-integer, name-text, price-real, category-text]).

%% ============================================
%% TEST 1: Simple CASE (value mapping)
%% ============================================

employee_status_label(Name, Status, StatusLabel) :-
    employees(_, Name, _, _, Status),
    StatusLabel = sql_case(Status, [
        active-'Active Employee',
        inactive-'Inactive',
        terminated-'Terminated'
    ], 'Unknown').

test1 :-
    format('~n=== Test 1: Simple CASE (value mapping) ===~n'),
    format('Pattern: sql_case(Column, [val1-result1, ...], Else)~n~n'),
    compile_predicate_to_sql(employee_status_label/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 2: Searched CASE (condition-based)
%% ============================================

salary_tier(Name, Salary, Tier) :-
    employees(_, Name, _, Salary, _),
    Tier = sql_case([
        when(Salary > 100000, 'Executive'),
        when(Salary > 50000, 'Senior'),
        when(Salary > 30000, 'Mid-Level')
    ], 'Entry-Level').

test2 :-
    format('~n=== Test 2: Searched CASE (condition-based) ===~n'),
    format('Pattern: sql_case([when(Cond, Result), ...], Else)~n~n'),
    compile_predicate_to_sql(salary_tier/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 3: Column with alias
%% ============================================

employee_with_alias(EmployeeName, Dept) :-
    employees(_, Name, Dept, _, _),
    EmployeeName = sql_as(Name, employee_name).

test3 :-
    format('~n=== Test 3: Column with alias ===~n'),
    format('Pattern: sql_as(Column, alias)~n~n'),
    compile_predicate_to_sql(employee_with_alias/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 4: CASE with alias
%% ============================================

priority_order(Id, Amount, PriorityLabel) :-
    orders(Id, _, Amount, Priority),
    PriorityLabel = sql_as(
        sql_case(Priority, [
            high-'Urgent',
            medium-'Normal',
            low-'Low Priority'
        ], 'Unclassified'),
        priority_label
    ).

test4 :-
    format('~n=== Test 4: CASE with alias ===~n'),
    format('Pattern: sql_as(sql_case(...), alias)~n~n'),
    compile_predicate_to_sql(priority_order/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 5: Multiple CASE expressions
%% ============================================

employee_classification(Name, Salary, SalaryTier, StatusLabel) :-
    employees(_, Name, _, Salary, Status),
    SalaryTier = sql_case([
        when(Salary > 80000, 'High'),
        when(Salary > 40000, 'Medium')
    ], 'Low'),
    StatusLabel = sql_case(Status, [
        active-'A',
        inactive-'I'
    ], 'X').

test5 :-
    format('~n=== Test 5: Multiple CASE expressions ===~n'),
    compile_predicate_to_sql(employee_classification/4, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 6: CASE with less-than condition
%% ============================================

price_category(Name, Price, Category) :-
    products(_, Name, Price, _),
    Category = sql_case([
        when(Price < 10, 'Budget'),
        when(Price < 50, 'Standard'),
        when(Price < 100, 'Premium')
    ], 'Luxury').

test6 :-
    format('~n=== Test 6: CASE with less-than condition ===~n'),
    compile_predicate_to_sql(price_category/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 7: CASE with equality condition
%% ============================================

dept_location(Name, Dept, Location) :-
    employees(_, Name, Dept, _, _),
    Location = sql_case([
        when(Dept = engineering, 'Building A'),
        when(Dept = sales, 'Building B'),
        when(Dept = hr, 'Building C')
    ], 'Remote').

test7 :-
    format('~n=== Test 7: CASE with equality condition ===~n'),
    compile_predicate_to_sql(dept_location/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 8: Simple CASE with ELSE NULL
%% ============================================

status_code(Name, Status, Code) :-
    employees(_, Name, _, _, Status),
    Code = sql_case(Status, [
        active-1,
        inactive-0
    ], null).

test8 :-
    format('~n=== Test 8: Simple CASE with ELSE NULL ===~n'),
    compile_predicate_to_sql(status_code/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 9: Output as SELECT
%% ============================================

test9 :-
    format('~n=== Test 9: Output as SELECT ===~n'),
    compile_predicate_to_sql(salary_tier/3, [format(select)], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 10: CASE with no ELSE (defaults to none)
%% ============================================

bonus_eligible(Name, Salary, Bonus) :-
    employees(_, Name, _, Salary, _),
    Bonus = sql_case([
        when(Salary > 60000, 'Yes')
    ], none).

test10 :-
    format('~n=== Test 10: CASE with no ELSE ===~n'),
    compile_predicate_to_sql(bonus_eligible/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% RUN ALL TESTS
%% ============================================

test_all :-
    format('~n========================================~n'),
    format('  SQL CASE WHEN Expression Tests~n'),
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
