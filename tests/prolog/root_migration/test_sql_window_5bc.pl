:- encoding(utf8).
% Test SQL Window Functions Phase 5b (Aggregates) and 5c (Value Functions)
% Tests for window_sum, window_avg, window_count, window_min, window_max, lag, lead

:- use_module('src/unifyweaver/targets/sql_target').

%% ============================================
%% TABLE SCHEMAS
%% ============================================

:- sql_table(sales, [id-integer, region-text, date-text, amount-integer]).
:- sql_table(stock_prices, [id-integer, symbol-text, date-text, price-real]).
:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer]).

%% ============================================
%% PHASE 5b: WINDOW AGGREGATES
%% ============================================

%% TEST 1: Running Sum (window_sum)
sales_running_total(Date, Amount, RunningTotal) :-
    sales(_, _, Date, Amount),
    window_sum(Amount, RunningTotal, [order_by(Date)]).

test1 :-
    format('~n=== Test 1: Running Sum (window_sum) ===~n'),
    format('Pattern: window_sum(Amount, RunningTotal, [order_by(Date)])~n~n'),
    compile_predicate_to_sql(sales_running_total/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% TEST 2: Running Average with PARTITION BY (window_avg)
regional_running_avg(Region, Date, Amount, RunningAvg) :-
    sales(_, Region, Date, Amount),
    window_avg(Amount, RunningAvg, [partition_by(Region), order_by(Date)]).

test2 :-
    format('~n=== Test 2: Running Average with PARTITION BY (window_avg) ===~n'),
    format('Pattern: window_avg(Amount, RunningAvg, [partition_by(Region), order_by(Date)])~n~n'),
    compile_predicate_to_sql(regional_running_avg/4, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% TEST 3: Running Count (window_count)
sales_running_count(Date, Amount, RunningCount) :-
    sales(_, _, Date, Amount),
    window_count(RunningCount, [order_by(Date)]).

test3 :-
    format('~n=== Test 3: Running Count (window_count) ===~n'),
    format('Pattern: window_count(RunningCount, [order_by(Date)])~n~n'),
    compile_predicate_to_sql(sales_running_count/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% TEST 4: Running Min (window_min)
sales_running_min(Date, Amount, RunningMin) :-
    sales(_, _, Date, Amount),
    window_min(Amount, RunningMin, [order_by(Date)]).

test4 :-
    format('~n=== Test 4: Running Min (window_min) ===~n'),
    format('Pattern: window_min(Amount, RunningMin, [order_by(Date)])~n~n'),
    compile_predicate_to_sql(sales_running_min/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% TEST 5: Running Max (window_max)
sales_running_max(Date, Amount, RunningMax) :-
    sales(_, _, Date, Amount),
    window_max(Amount, RunningMax, [order_by(Date)]).

test5 :-
    format('~n=== Test 5: Running Max (window_max) ===~n'),
    format('Pattern: window_max(Amount, RunningMax, [order_by(Date)])~n~n'),
    compile_predicate_to_sql(sales_running_max/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% PHASE 5c: VALUE FUNCTIONS (LAG/LEAD)
%% ============================================

%% TEST 6: LAG - Previous Value
price_with_previous(Symbol, Date, Price, PrevPrice) :-
    stock_prices(_, Symbol, Date, Price),
    lag(Price, 1, PrevPrice, [partition_by(Symbol), order_by(Date)]).

test6 :-
    format('~n=== Test 6: LAG - Previous Value ===~n'),
    format('Pattern: lag(Price, 1, PrevPrice, [partition_by(Symbol), order_by(Date)])~n~n'),
    compile_predicate_to_sql(price_with_previous/4, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% TEST 7: LEAD - Next Value
price_with_next(Symbol, Date, Price, NextPrice) :-
    stock_prices(_, Symbol, Date, Price),
    lead(Price, 1, NextPrice, [partition_by(Symbol), order_by(Date)]).

test7 :-
    format('~n=== Test 7: LEAD - Next Value ===~n'),
    format('Pattern: lead(Price, 1, NextPrice, [partition_by(Symbol), order_by(Date)])~n~n'),
    compile_predicate_to_sql(price_with_next/4, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% TEST 8: LAG with offset 2 (two rows back)
price_two_days_ago(Symbol, Date, Price, PrevPrice2) :-
    stock_prices(_, Symbol, Date, Price),
    lag(Price, 2, PrevPrice2, [partition_by(Symbol), order_by(Date)]).

test8 :-
    format('~n=== Test 8: LAG with offset 2 ===~n'),
    format('Pattern: lag(Price, 2, PrevPrice2, [partition_by(Symbol), order_by(Date)])~n~n'),
    compile_predicate_to_sql(price_two_days_ago/4, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% COMBINED TESTS
%% ============================================

%% TEST 9: Multiple Window Aggregates
sales_full_analysis(Date, Amount, RunningSum, RunningAvg, RunningCount) :-
    sales(_, _, Date, Amount),
    window_sum(Amount, RunningSum, [order_by(Date)]),
    window_avg(Amount, RunningAvg, [order_by(Date)]),
    window_count(RunningCount, [order_by(Date)]).

test9 :-
    format('~n=== Test 9: Multiple Window Aggregates ===~n'),
    format('Pattern: window_sum + window_avg + window_count~n~n'),
    compile_predicate_to_sql(sales_full_analysis/5, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% TEST 10: LAG and LEAD together
price_with_neighbors(Symbol, Date, Price, PrevPrice, NextPrice) :-
    stock_prices(_, Symbol, Date, Price),
    lag(Price, 1, PrevPrice, [partition_by(Symbol), order_by(Date)]),
    lead(Price, 1, NextPrice, [partition_by(Symbol), order_by(Date)]).

test10 :-
    format('~n=== Test 10: LAG and LEAD together ===~n'),
    format('Pattern: lag(...) + lead(...)~n~n'),
    compile_predicate_to_sql(price_with_neighbors/5, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% TEST 11: Window aggregate with WHERE clause
high_sales_running_total(Date, Amount, RunningTotal) :-
    sales(_, _, Date, Amount),
    Amount > 1000,
    window_sum(Amount, RunningTotal, [order_by(Date)]).

test11 :-
    format('~n=== Test 11: Window Aggregate with WHERE Clause ===~n'),
    format('Pattern: Amount > 1000 AND window_sum(...)~n~n'),
    compile_predicate_to_sql(high_sales_running_total/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% TEST 12: Mix ranking (5a) with aggregates (5b)
employee_rank_and_running(Name, Salary, Rank, RunningSum) :-
    employees(_, Name, _, Salary),
    rank(Rank, [order_by(Salary, desc)]),
    window_sum(Salary, RunningSum, [order_by(Salary, desc)]).

test12 :-
    format('~n=== Test 12: Mix Ranking with Aggregates ===~n'),
    format('Pattern: rank(...) + window_sum(...)~n~n'),
    compile_predicate_to_sql(employee_rank_and_running/4, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% RUN ALL TESTS
%% ============================================

test_all :-
    format('~n========================================~n'),
    format('  SQL Window Functions Tests~n'),
    format('  Phase 5b (Aggregates) + 5c (LAG/LEAD)~n'),
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
    test11,
    test12,
    format('~n========================================~n'),
    format('  All tests completed!~n'),
    format('========================================~n').

%% Entry point
main :- test_all.

:- initialization(main, main).
