:- encoding(utf8).
% Test SQL Subqueries (Phase 4)
% Tests for IN, NOT IN, EXISTS subquery support

:- use_module('src/unifyweaver/targets/sql_target').

%% ============================================
%% TABLE SCHEMAS
%% ============================================

:- sql_table(employees, [id-integer, name-text, dept_id-integer, salary-integer]).
:- sql_table(departments, [id-integer, name-text, budget-integer, active-integer]).
:- sql_table(customers, [id-integer, name-text, city-text]).
:- sql_table(orders, [id-integer, customer_id-integer, product-text, amount-integer]).
:- sql_table(projects, [id-integer, name-text, dept_id-integer]).
:- sql_table(assignments, [id-integer, emp_id-integer, project_id-integer]).

%% ============================================
%% TEST 1: Simple IN Subquery
%% ============================================

% Subquery predicate: high budget department IDs
high_budget_depts(Id) :-
    departments(Id, _, Budget, _),
    Budget > 100000.

% Main predicate: employees in high-budget departments
high_budget_employees(Name) :-
    employees(_, Name, DeptId, _),
    in_query(DeptId, high_budget_depts/1).

test1 :-
    format('~n=== Test 1: Simple IN Subquery ===~n'),
    format('Pattern: in_query(DeptId, high_budget_depts/1)~n~n'),
    compile_predicate_to_sql(high_budget_employees/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 2: NOT IN Subquery
%% ============================================

% Subquery predicate: inactive department IDs
inactive_depts(Id) :-
    departments(Id, _, _, Active),
    Active = 0.

% Main predicate: employees NOT in inactive departments
active_dept_employees(Name) :-
    employees(_, Name, DeptId, _),
    not_in_query(DeptId, inactive_depts/1).

test2 :-
    format('~n=== Test 2: NOT IN Subquery ===~n'),
    format('Pattern: not_in_query(DeptId, inactive_depts/1)~n~n'),
    compile_predicate_to_sql(active_dept_employees/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 3: EXISTS Subquery (Correlated)
%% ============================================

% Customers who have placed orders
customers_with_orders(Name) :-
    customers(CustId, Name, _),
    exists(orders(_, CustId, _, _)).

test3 :-
    format('~n=== Test 3: EXISTS Subquery (Correlated) ===~n'),
    format('Pattern: exists(orders(_, CustId, _, _))~n~n'),
    compile_predicate_to_sql(customers_with_orders/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 4: NOT EXISTS Subquery
%% ============================================

% Customers who have NOT placed orders
customers_without_orders(Name) :-
    customers(CustId, Name, _),
    not_exists(orders(_, CustId, _, _)).

test4 :-
    format('~n=== Test 4: NOT EXISTS Subquery ===~n'),
    format('Pattern: not_exists(orders(_, CustId, _, _))~n~n'),
    compile_predicate_to_sql(customers_without_orders/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 5: Multiple Subqueries
%% ============================================

% Employees in high-budget depts who have project assignments
elite_employees(Name) :-
    employees(EmpId, Name, DeptId, _),
    in_query(DeptId, high_budget_depts/1),
    exists(assignments(_, EmpId, _)).

test5 :-
    format('~n=== Test 5: Multiple Subqueries (IN + EXISTS) ===~n'),
    format('Pattern: in_query(...) AND exists(...)~n~n'),
    compile_predicate_to_sql(elite_employees/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 6: IN Subquery with Multiple Conditions
%% ============================================

% Departments with both high budget AND active
prime_depts(Id) :-
    departments(Id, _, Budget, Active),
    Budget > 50000,
    Active = 1.

% Employees in prime departments
prime_dept_employees(Name, Salary) :-
    employees(_, Name, DeptId, Salary),
    in_query(DeptId, prime_depts/1).

test6 :-
    format('~n=== Test 6: IN Subquery with Multiple Conditions ===~n'),
    format('Subquery has: Budget > 50000 AND Active = 1~n~n'),
    compile_predicate_to_sql(prime_dept_employees/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% RUN ALL TESTS
%% ============================================

test_all :-
    format('~n========================================~n'),
    format('  SQL Subquery Tests (Phase 4)~n'),
    format('========================================~n'),
    test1,
    test2,
    test3,
    test4,
    test5,
    test6,
    format('~n========================================~n'),
    format('  All tests completed!~n'),
    format('========================================~n').

%% Entry point
main :- test_all.

:- initialization(main, main).
