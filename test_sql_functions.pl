:- encoding(utf8).
% Test SQL Functions (NULL handling, String, Date, BETWEEN, LIKE)
% Comprehensive test suite for new SQL function support

:- use_module('src/unifyweaver/targets/sql_target').

%% ============================================
%% TABLE SCHEMAS
%% ============================================

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, hire_date-text, email-text]).
:- sql_table(products, [id-integer, name-text, price-real, category-text, description-text]).
:- sql_table(orders, [id-integer, customer_id-integer, order_date-text, ship_date-text, amount-real]).

%% ============================================
%% TEST 1: COALESCE
%% ============================================

employee_with_default_dept(Name, Dept) :-
    employees(_, Name, D, _, _, _),
    Dept = sql_coalesce([D, 'Unknown']).

test1 :-
    format('~n=== Test 1: COALESCE ===~n'),
    format('Pattern: sql_coalesce([D, \'Unknown\'])~n~n'),
    compile_predicate_to_sql(employee_with_default_dept/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 2: NULLIF
%% ============================================

product_non_zero_price(Name, Price) :-
    products(_, Name, P, _, _),
    Price = sql_nullif(P, 0).

test2 :-
    format('~n=== Test 2: NULLIF ===~n'),
    format('Pattern: sql_nullif(P, 0)~n~n'),
    compile_predicate_to_sql(product_non_zero_price/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 3: IFNULL
%% ============================================

order_with_default_amount(Id, Amount) :-
    orders(Id, _, _, _, A),
    Amount = sql_ifnull(A, 0).

test3 :-
    format('~n=== Test 3: IFNULL ===~n'),
    format('Pattern: sql_ifnull(A, 0)~n~n'),
    compile_predicate_to_sql(order_with_default_amount/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 4: CONCAT (string concatenation)
%% ============================================

employee_full_info(FullInfo) :-
    employees(_, Name, Dept, _, _, _),
    FullInfo = sql_concat([Name, ' - ', Dept]).

test4 :-
    format('~n=== Test 4: CONCAT (string concatenation) ===~n'),
    format('Pattern: sql_concat([Name, \' - \', Dept])~n~n'),
    compile_predicate_to_sql(employee_full_info/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 5: UPPER/LOWER
%% ============================================

employee_name_cases(Name, UpperName, LowerName) :-
    employees(_, Name, _, _, _, _),
    UpperName = sql_upper(Name),
    LowerName = sql_lower(Name).

test5 :-
    format('~n=== Test 5: UPPER/LOWER ===~n'),
    format('Pattern: sql_upper(Name), sql_lower(Name)~n~n'),
    compile_predicate_to_sql(employee_name_cases/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 6: SUBSTRING
%% ============================================

product_code(Name, Code) :-
    products(_, Name, _, _, _),
    Code = sql_substring(Name, 1, 3).

test6 :-
    format('~n=== Test 6: SUBSTRING ===~n'),
    format('Pattern: sql_substring(Name, 1, 3)~n~n'),
    compile_predicate_to_sql(product_code/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 7: TRIM/LTRIM/RTRIM
%% ============================================

trimmed_description(Name, Trimmed) :-
    products(_, Name, _, _, Desc),
    Trimmed = sql_trim(Desc).

test7 :-
    format('~n=== Test 7: TRIM ===~n'),
    format('Pattern: sql_trim(Desc)~n~n'),
    compile_predicate_to_sql(trimmed_description/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 8: LENGTH
%% ============================================

name_with_length(Name, Len) :-
    employees(_, Name, _, _, _, _),
    Len = sql_length(Name).

test8 :-
    format('~n=== Test 8: LENGTH ===~n'),
    format('Pattern: sql_length(Name)~n~n'),
    compile_predicate_to_sql(name_with_length/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 9: REPLACE
%% ============================================

sanitized_email(Name, Email) :-
    employees(_, Name, _, _, _, E),
    Email = sql_replace(E, '@', ' at ').

test9 :-
    format('~n=== Test 9: REPLACE ===~n'),
    format('Pattern: sql_replace(E, \'@\', \' at \')~n~n'),
    compile_predicate_to_sql(sanitized_email/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 10: DATE
%% ============================================

order_date_only(Id, DateOnly) :-
    orders(Id, _, OrderDate, _, _),
    DateOnly = sql_date(OrderDate).

test10 :-
    format('~n=== Test 10: DATE ===~n'),
    format('Pattern: sql_date(OrderDate)~n~n'),
    compile_predicate_to_sql(order_date_only/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 11: DATE_ADD
%% ============================================

order_due_date(Id, DueDate) :-
    orders(Id, _, OrderDate, _, _),
    DueDate = sql_date_add(OrderDate, 7, days).

test11 :-
    format('~n=== Test 11: DATE_ADD ===~n'),
    format('Pattern: sql_date_add(OrderDate, 7, days)~n~n'),
    compile_predicate_to_sql(order_due_date/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 12: DATE_DIFF
%% ============================================

shipping_days(Id, Days) :-
    orders(Id, _, OrderDate, ShipDate, _),
    Days = sql_date_diff(ShipDate, OrderDate).

test12 :-
    format('~n=== Test 12: DATE_DIFF ===~n'),
    format('Pattern: sql_date_diff(ShipDate, OrderDate)~n~n'),
    compile_predicate_to_sql(shipping_days/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 13: EXTRACT (date parts)
%% ============================================

hire_year_month(Name, Year, Month) :-
    employees(_, Name, _, _, HireDate, _),
    Year = sql_extract(year, HireDate),
    Month = sql_extract(month, HireDate).

test13 :-
    format('~n=== Test 13: EXTRACT (date parts) ===~n'),
    format('Pattern: sql_extract(year, HireDate), sql_extract(month, HireDate)~n~n'),
    compile_predicate_to_sql(hire_year_month/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 14: STRFTIME
%% ============================================

formatted_date(Id, Formatted) :-
    orders(Id, _, OrderDate, _, _),
    Formatted = sql_strftime('%Y-%m-%d', OrderDate).

test14 :-
    format('~n=== Test 14: STRFTIME ===~n'),
    format('Pattern: sql_strftime(\'%Y-%m-%d\', OrderDate)~n~n'),
    compile_predicate_to_sql(formatted_date/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 15: BETWEEN (numeric)
%% ============================================

mid_salary_employees(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    sql_between(Salary, 50000, 100000).

test15 :-
    format('~n=== Test 15: BETWEEN (numeric) ===~n'),
    format('Pattern: sql_between(Salary, 50000, 100000)~n~n'),
    compile_predicate_to_sql(mid_salary_employees/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 16: BETWEEN (dates)
%% ============================================

q1_orders(Id, OrderDate) :-
    orders(Id, _, OrderDate, _, _),
    sql_between(OrderDate, '2024-01-01', '2024-03-31').

test16 :-
    format('~n=== Test 16: BETWEEN (dates) ===~n'),
    format('Pattern: sql_between(OrderDate, \'2024-01-01\', \'2024-03-31\')~n~n'),
    compile_predicate_to_sql(q1_orders/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 17: NOT BETWEEN
%% ============================================

extreme_salaries(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    sql_not_between(Salary, 40000, 80000).

test17 :-
    format('~n=== Test 17: NOT BETWEEN ===~n'),
    format('Pattern: sql_not_between(Salary, 40000, 80000)~n~n'),
    compile_predicate_to_sql(extreme_salaries/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 18: LIKE (starts with)
%% ============================================

employees_starting_with_j(Name) :-
    employees(_, Name, _, _, _, _),
    sql_like(Name, 'J%').

test18 :-
    format('~n=== Test 18: LIKE (starts with) ===~n'),
    format('Pattern: sql_like(Name, \'J%\')~n~n'),
    compile_predicate_to_sql(employees_starting_with_j/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 19: LIKE (contains)
%% ============================================

products_with_pro(Name) :-
    products(_, Name, _, _, _),
    sql_like(Name, '%Pro%').

test19 :-
    format('~n=== Test 19: LIKE (contains) ===~n'),
    format('Pattern: sql_like(Name, \'%Pro%\')~n~n'),
    compile_predicate_to_sql(products_with_pro/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 20: NOT LIKE
%% ============================================

non_test_products(Name) :-
    products(_, Name, _, _, _),
    sql_not_like(Name, 'Test%').

test20 :-
    format('~n=== Test 20: NOT LIKE ===~n'),
    format('Pattern: sql_not_like(Name, \'Test%\')~n~n'),
    compile_predicate_to_sql(non_test_products/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 21: GLOB (SQLite-specific)
%% ============================================

glob_products(Name) :-
    products(_, Name, _, _, _),
    sql_glob(Name, '*Pro*').

test21 :-
    format('~n=== Test 21: GLOB (SQLite) ===~n'),
    format('Pattern: sql_glob(Name, \'*Pro*\')~n~n'),
    compile_predicate_to_sql(glob_products/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 22: IN (list)
%% ============================================

engineering_depts(Name, Dept) :-
    employees(_, Name, Dept, _, _, _),
    sql_in(Dept, [engineering, 'r&d', development]).

test22 :-
    format('~n=== Test 22: IN (list) ===~n'),
    format('Pattern: sql_in(Dept, [engineering, \'r&d\', development])~n~n'),
    compile_predicate_to_sql(engineering_depts/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 23: NOT IN (list)
%% ============================================

non_admin_employees(Name, Dept) :-
    employees(_, Name, Dept, _, _, _),
    sql_not_in(Dept, [admin, hr, management]).

test23 :-
    format('~n=== Test 23: NOT IN (list) ===~n'),
    format('Pattern: sql_not_in(Dept, [admin, hr, management])~n~n'),
    compile_predicate_to_sql(non_admin_employees/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 24: IS NULL
%% ============================================

unshipped_orders(Id) :-
    orders(Id, _, _, ShipDate, _),
    sql_is_null(ShipDate).

test24 :-
    format('~n=== Test 24: IS NULL ===~n'),
    format('Pattern: sql_is_null(ShipDate)~n~n'),
    compile_predicate_to_sql(unshipped_orders/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 25: IS NOT NULL
%% ============================================

shipped_orders(Id, ShipDate) :-
    orders(Id, _, _, ShipDate, _),
    sql_is_not_null(ShipDate).

test25 :-
    format('~n=== Test 25: IS NOT NULL ===~n'),
    format('Pattern: sql_is_not_null(ShipDate)~n~n'),
    compile_predicate_to_sql(shipped_orders/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 26: Nested functions
%% ============================================

upper_trimmed_name(Name, Result) :-
    employees(_, Name, _, _, _, _),
    Result = sql_upper(sql_trim(Name)).

test26 :-
    format('~n=== Test 26: Nested functions ===~n'),
    format('Pattern: sql_upper(sql_trim(Name))~n~n'),
    compile_predicate_to_sql(upper_trimmed_name/2, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 27: Combined functions with alias
%% ============================================

employee_display(DisplayName) :-
    employees(_, Name, Dept, _, _, _),
    DisplayName = sql_as(sql_concat([sql_upper(Name), ' (', Dept, ')']), display_name).

test27 :-
    format('~n=== Test 27: Combined functions with alias ===~n'),
    format('Pattern: sql_as(sql_concat([sql_upper(Name), \' (\', Dept, \')\']), display_name)~n~n'),
    compile_predicate_to_sql(employee_display/1, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 28: Output as SELECT
%% ============================================

test28 :-
    format('~n=== Test 28: Output as SELECT ===~n'),
    compile_predicate_to_sql(mid_salary_employees/2, [format(select)], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% RUN ALL TESTS
%% ============================================

test_all :-
    format('~n========================================~n'),
    format('  SQL Functions Tests~n'),
    format('========================================~n'),
    test1, test2, test3, test4, test5,
    test6, test7, test8, test9, test10,
    test11, test12, test13, test14, test15,
    test16, test17, test18, test19, test20,
    test21, test22, test23, test24, test25,
    test26, test27, test28,
    format('~n========================================~n'),
    format('  All tests completed!~n'),
    format('========================================~n').

%% Entry point
main :- test_all.

:- initialization(main, main).
