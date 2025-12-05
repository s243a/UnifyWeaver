:- encoding(utf8).
% Test SQL Window Frame Specification (Phase 5d)
% Tests for ROWS/RANGE BETWEEN ... AND ...

:- use_module('src/unifyweaver/targets/sql_target').

%% ============================================
%% TABLE SCHEMAS
%% ============================================

:- sql_table(sales, [id-integer, date-text, amount-integer]).
:- sql_table(stock_prices, [id-integer, symbol-text, date-text, price-real]).
:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer]).

%% ============================================
%% TEST 1: ROWS BETWEEN N PRECEDING AND CURRENT ROW
%% (Rolling 3-day sum including current)
%% ============================================

rolling_3_sum(Date, Amount, RollingSum) :-
    sales(_, Date, Amount),
    window_sum(Amount, RollingSum, [
        order_by(Date),
        frame(rows, preceding(2), current_row)
    ]).

test1 :-
    format('~n=== Test 1: ROWS BETWEEN 2 PRECEDING AND CURRENT ROW ===~n'),
    format('Pattern: frame(rows, preceding(2), current_row)~n~n'),
    compile_predicate_to_sql(rolling_3_sum/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 2: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
%% (Cumulative sum from start)
%% ============================================

cumulative_sum(Date, Amount, CumSum) :-
    sales(_, Date, Amount),
    window_sum(Amount, CumSum, [
        order_by(Date),
        frame(rows, unbounded_preceding, current_row)
    ]).

test2 :-
    format('~n=== Test 2: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW ===~n'),
    format('Pattern: frame(rows, unbounded_preceding, current_row)~n~n'),
    compile_predicate_to_sql(cumulative_sum/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 3: ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
%% (Centered moving average - 3 rows)
%% ============================================

centered_avg(Date, Amount, CenterAvg) :-
    sales(_, Date, Amount),
    window_avg(Amount, CenterAvg, [
        order_by(Date),
        frame(rows, preceding(1), following(1))
    ]).

test3 :-
    format('~n=== Test 3: ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING ===~n'),
    format('Pattern: frame(rows, preceding(1), following(1))~n~n'),
    compile_predicate_to_sql(centered_avg/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 4: ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
%% (Sum from current to end)
%% ============================================

remaining_sum(Date, Amount, RemSum) :-
    sales(_, Date, Amount),
    window_sum(Amount, RemSum, [
        order_by(Date),
        frame(rows, current_row, unbounded_following)
    ]).

test4 :-
    format('~n=== Test 4: ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING ===~n'),
    format('Pattern: frame(rows, current_row, unbounded_following)~n~n'),
    compile_predicate_to_sql(remaining_sum/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 5: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
%% ============================================

range_cumulative(Date, Amount, CumSum) :-
    sales(_, Date, Amount),
    window_sum(Amount, CumSum, [
        order_by(Date),
        frame(range, unbounded_preceding, current_row)
    ]).

test5 :-
    format('~n=== Test 5: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW ===~n'),
    format('Pattern: frame(range, unbounded_preceding, current_row)~n~n'),
    compile_predicate_to_sql(range_cumulative/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 6: Frame with PARTITION BY
%% ============================================

partitioned_rolling_avg(Symbol, Date, Price, RollingAvg) :-
    stock_prices(_, Symbol, Date, Price),
    window_avg(Price, RollingAvg, [
        partition_by(Symbol),
        order_by(Date),
        frame(rows, preceding(4), current_row)
    ]).

test6 :-
    format('~n=== Test 6: Frame with PARTITION BY ===~n'),
    format('Pattern: partition_by(Symbol), order_by(Date), frame(rows, preceding(4), current_row)~n~n'),
    compile_predicate_to_sql(partitioned_rolling_avg/4, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 7: Window function without frame (should work as before)
%% ============================================

simple_rank(Name, Salary, Rank) :-
    employees(_, Name, _, Salary),
    rank(Rank, [order_by(Salary, desc)]).

test7 :-
    format('~n=== Test 7: Window function without frame (backward compat) ===~n'),
    format('Pattern: No frame specified~n~n'),
    compile_predicate_to_sql(simple_rank/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 8: ROWS UNBOUNDED PRECEDING (single bound shorthand)
%% ============================================

running_total_short(Date, Amount, RunTotal) :-
    sales(_, Date, Amount),
    window_sum(Amount, RunTotal, [
        order_by(Date),
        frame(rows, unbounded_preceding)
    ]).

test8 :-
    format('~n=== Test 8: ROWS UNBOUNDED PRECEDING (single bound) ===~n'),
    format('Pattern: frame(rows, unbounded_preceding)~n~n'),
    compile_predicate_to_sql(running_total_short/3, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 9: Multiple window functions with different frames
%% ============================================

multi_frame_analysis(Date, Amount, Roll3, CumSum) :-
    sales(_, Date, Amount),
    window_sum(Amount, Roll3, [
        order_by(Date),
        frame(rows, preceding(2), current_row)
    ]),
    window_sum(Amount, CumSum, [
        order_by(Date),
        frame(rows, unbounded_preceding, current_row)
    ]).

test9 :-
    format('~n=== Test 9: Multiple window functions with different frames ===~n'),
    format('Pattern: Two window_sum with different frames~n~n'),
    compile_predicate_to_sql(multi_frame_analysis/4, [], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 10: Output as SELECT
%% ============================================

test10 :-
    format('~n=== Test 10: Output as SELECT ===~n'),
    compile_predicate_to_sql(rolling_3_sum/3, [format(select)], SQL),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% RUN ALL TESTS
%% ============================================

test_all :-
    format('~n========================================~n'),
    format('  SQL Window Frame Tests (Phase 5d)~n'),
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
