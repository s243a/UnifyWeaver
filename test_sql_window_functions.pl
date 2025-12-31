:- encoding(utf8).
% Test SQL Window Functions (Phase 5a)
% Tests for ROW_NUMBER, RANK, DENSE_RANK, NTILE ranking functions

:- use_module('src/unifyweaver/targets/sql_target').

%% ============================================
%% TABLE SCHEMAS
%% ============================================

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer]).
:- sql_table(sales, [id-integer, region-text, date-text, amount-integer]).

%% ============================================
%% TEST 1: Simple ROW_NUMBER
%% ============================================

employee_numbered(Name, Salary, RowNum) :-
    employees(_, Name, _, Salary),
    row_number(RowNum, [order_by(Salary, desc)]).

test1 :-
    format('~n=== Test 1: Simple ROW_NUMBER ===~n'),
    format('Pattern: row_number(RowNum, [order_by(Salary, desc)])~n~n'),
    compile_predicate_to_sql(employee_numbered/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 2: RANK with PARTITION BY
%% ============================================

employee_dept_rank(Name, Dept, Salary, Rank) :-
    employees(_, Name, Dept, Salary),
    rank(Rank, [partition_by(Dept), order_by(Salary, desc)]).

test2 :-
    format('~n=== Test 2: RANK with PARTITION BY ===~n'),
    format('Pattern: rank(Rank, [partition_by(Dept), order_by(Salary, desc)])~n~n'),
    compile_predicate_to_sql(employee_dept_rank/4, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 3: DENSE_RANK
%% ============================================

employee_dense_rank(Name, Salary, DenseRank) :-
    employees(_, Name, _, Salary),
    dense_rank(DenseRank, [order_by(Salary, desc)]).

test3 :-
    format('~n=== Test 3: DENSE_RANK ===~n'),
    format('Pattern: dense_rank(DenseRank, [order_by(Salary, desc)])~n~n'),
    compile_predicate_to_sql(employee_dense_rank/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 4: NTILE (Quartiles)
%% ============================================

employee_quartile(Name, Salary, Quartile) :-
    employees(_, Name, _, Salary),
    ntile(4, Quartile, [order_by(Salary, desc)]).

test4 :-
    format('~n=== Test 4: NTILE (Quartiles) ===~n'),
    format('Pattern: ntile(4, Quartile, [order_by(Salary, desc)])~n~n'),
    compile_predicate_to_sql(employee_quartile/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 5: Multiple Window Functions
%% ============================================

employee_full_ranking(Name, Salary, RowNum, Rank, Quartile) :-
    employees(_, Name, _, Salary),
    row_number(RowNum, [order_by(Salary, desc)]),
    rank(Rank, [order_by(Salary, desc)]),
    ntile(4, Quartile, [order_by(Salary, desc)]).

test5 :-
    format('~n=== Test 5: Multiple Window Functions ===~n'),
    format('Pattern: row_number + rank + ntile in same query~n~n'),
    compile_predicate_to_sql(employee_full_ranking/5, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 6: ROW_NUMBER with WHERE Clause
%% ============================================

high_salary_numbered(Name, Salary, RowNum) :-
    employees(_, Name, _, Salary),
    Salary > 50000,
    row_number(RowNum, [order_by(Salary, desc)]).

test6 :-
    format('~n=== Test 6: ROW_NUMBER with WHERE Clause ===~n'),
    format('Pattern: Salary > 50000 AND row_number(...)~n~n'),
    compile_predicate_to_sql(high_salary_numbered/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 7: RANK with Multiple PARTITION BY Columns
%% ============================================

sales_region_rank(Region, Date, Amount, Rank) :-
    sales(_, Region, Date, Amount),
    rank(Rank, [partition_by(Region), order_by(Amount, desc)]).

test7 :-
    format('~n=== Test 7: RANK with PARTITION BY ===~n'),
    format('Pattern: rank(Rank, [partition_by(Region), order_by(Amount, desc)])~n~n'),
    compile_predicate_to_sql(sales_region_rank/4, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 8: Output as SELECT (not VIEW)
%% ============================================

test8 :-
    format('~n=== Test 8: Output as SELECT (not VIEW) ===~n'),
    format('Pattern: Using [format(select)] option~n~n'),
    compile_predicate_to_sql(employee_numbered/3, [format(select)], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% RUN ALL TESTS
%% ============================================

test_all :-
    format('~n========================================~n'),
    format('  SQL Window Functions Tests (Phase 5a)~n'),
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
