<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 6: Window Functions

Window functions perform calculations across sets of rows related to the current row, without collapsing them into groups like aggregations do.

## Setup

```prolog
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [
    id-integer,
    name-text,
    dept-text,
    salary-integer,
    hire_date-text
]).

:- sql_table(sales, [
    id-integer,
    salesperson-text,
    region-text,
    amount-real,
    sale_date-text
]).

:- sql_table(orders, [
    id-integer,
    customer_id-integer,
    product-text,
    amount-real,
    order_date-text
]).
```

## Window Function Syntax

```prolog
sql_window(Function, Args, PartitionBy, OrderBy)
sql_window(Function, Args, PartitionBy, OrderBy, FrameSpec)
```

Parameters:
- `Function`: The window function name (rank, row_number, sum, etc.)
- `Args`: Arguments to the function (empty list `[]` for ranking functions)
- `PartitionBy`: List of columns to partition by (like GROUP BY but without collapsing)
- `OrderBy`: List of `(Column, Direction)` tuples for ordering within partition

## Ranking Functions

### ROW_NUMBER

Assigns a unique sequential number to each row within a partition:

```prolog
employees_row_num(Name, Dept, Salary, RowNum) :-
    employees(_, Name, Dept, Salary, _),
    RowNum = sql_window(row_number, [], [Dept], [(Salary, desc)]).

?- compile_predicate_to_sql(employees_row_num/4, [], SQL).
```

Generated SQL:
```sql
SELECT name, dept, salary,
       ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)
FROM employees;
```

### RANK

Assigns rank with gaps for ties:

```prolog
employees_ranked(Name, Dept, Salary, Rank) :-
    employees(_, Name, Dept, Salary, _),
    Rank = sql_window(rank, [], [Dept], [(Salary, desc)]).

% Two employees with same salary get same rank (e.g., 1, 1, 3, 4...)
```

### DENSE_RANK

Assigns rank without gaps for ties:

```prolog
employees_dense_ranked(Name, Dept, Salary, DenseRank) :-
    employees(_, Name, Dept, Salary, _),
    DenseRank = sql_window(dense_rank, [], [Dept], [(Salary, desc)]).

% Two employees with same salary get same rank (e.g., 1, 1, 2, 3...)
```

### Comparing Ranking Functions

| Salary | ROW_NUMBER | RANK | DENSE_RANK |
|--------|------------|------|------------|
| 100000 | 1 | 1 | 1 |
| 100000 | 2 | 1 | 1 |
| 90000 | 3 | 3 | 2 |
| 80000 | 4 | 4 | 3 |

## Value Functions

### LAG

Access data from a previous row:

```prolog
sales_with_previous(Salesperson, Amount, SaleDate, PrevAmount) :-
    sales(_, Salesperson, _, Amount, SaleDate),
    PrevAmount = sql_window(lag, [Amount, 1], [Salesperson], [(SaleDate, asc)]).

% LAG(amount, 1) OVER (PARTITION BY salesperson ORDER BY sale_date)
```

The second argument `1` is the offset (how many rows back).

### LEAD

Access data from a following row:

```prolog
sales_with_next(Salesperson, Amount, SaleDate, NextAmount) :-
    sales(_, Salesperson, _, Amount, SaleDate),
    NextAmount = sql_window(lead, [Amount, 1], [Salesperson], [(SaleDate, asc)]).

% LEAD(amount, 1) OVER (PARTITION BY salesperson ORDER BY sale_date)
```

### FIRST_VALUE and LAST_VALUE

```prolog
% First salary in department (highest if ordered desc)
emp_with_top_salary(Name, Dept, Salary, TopSalary) :-
    employees(_, Name, Dept, Salary, _),
    TopSalary = sql_window(first_value, [Salary], [Dept], [(Salary, desc)]).

% Last salary in department (lowest if ordered desc)
emp_with_bottom_salary(Name, Dept, Salary, BottomSalary) :-
    employees(_, Name, Dept, Salary, _),
    BottomSalary = sql_window(last_value, [Salary], [Dept], [(Salary, desc)]).
```

## Aggregate Window Functions

Regular aggregates can be used as window functions:

### Running Total

```prolog
running_sales_total(Salesperson, SaleDate, Amount, RunningTotal) :-
    sales(_, Salesperson, _, Amount, SaleDate),
    RunningTotal = sql_window(sum, [Amount], [Salesperson], [(SaleDate, asc)]).

% SUM(amount) OVER (PARTITION BY salesperson ORDER BY sale_date)
```

### Running Average

```prolog
moving_avg_sales(Salesperson, SaleDate, Amount, AvgSoFar) :-
    sales(_, Salesperson, _, Amount, SaleDate),
    AvgSoFar = sql_window(avg, [Amount], [Salesperson], [(SaleDate, asc)]).
```

### Running Count

```prolog
sales_count_by_date(Salesperson, SaleDate, SaleNum) :-
    sales(_, Salesperson, _, _, SaleDate),
    SaleNum = sql_window(count, [*], [Salesperson], [(SaleDate, asc)]).
```

## Window Frame Specifications

Control exactly which rows are included in the calculation:

```prolog
sql_window(Function, Args, PartitionBy, OrderBy, FrameSpec)
```

### Frame Types

| Frame Spec | Description |
|------------|-------------|
| `rows_between(Start, End)` | Physical row-based |
| `range_between(Start, End)` | Logical value-based |

### Frame Boundaries

| Boundary | Meaning |
|----------|---------|
| `unbounded_preceding` | From first row of partition |
| `current_row` | The current row |
| `unbounded_following` | To last row of partition |
| `N preceding` | N rows before current |
| `N following` | N rows after current |

### Running Total (Explicit Frame)

```prolog
running_total(Salesperson, SaleDate, Amount, Total) :-
    sales(_, Salesperson, _, Amount, SaleDate),
    Total = sql_window(sum, [Amount], [Salesperson], [(SaleDate, asc)],
                       rows_between(unbounded_preceding, current_row)).

% SUM(amount) OVER (
%     PARTITION BY salesperson
%     ORDER BY sale_date
%     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
% )
```

### 3-Day Moving Average

```prolog
moving_avg_3day(Salesperson, SaleDate, Amount, MovingAvg) :-
    sales(_, Salesperson, _, Amount, SaleDate),
    MovingAvg = sql_window(avg, [Amount], [Salesperson], [(SaleDate, asc)],
                           rows_between(2 preceding, current_row)).

% AVG(amount) OVER (
%     PARTITION BY salesperson
%     ORDER BY sale_date
%     ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
% )
```

### Full Partition Aggregate

```prolog
salary_vs_dept_total(Name, Dept, Salary, DeptTotal) :-
    employees(_, Name, Dept, Salary, _),
    DeptTotal = sql_window(sum, [Salary], [Dept], [],
                           rows_between(unbounded_preceding, unbounded_following)).

% SUM(salary) OVER (PARTITION BY dept)
```

## Partitioning Strategies

### No Partition (Whole Table)

```prolog
global_rank(Name, Salary, GlobalRank) :-
    employees(_, Name, _, Salary, _),
    GlobalRank = sql_window(rank, [], [], [(Salary, desc)]).

% RANK() OVER (ORDER BY salary DESC)
```

### Single Partition Column

```prolog
dept_rank(Name, Dept, Salary, DeptRank) :-
    employees(_, Name, Dept, Salary, _),
    DeptRank = sql_window(rank, [], [Dept], [(Salary, desc)]).

% RANK() OVER (PARTITION BY dept ORDER BY salary DESC)
```

### Multiple Partition Columns

```prolog
region_year_rank(Salesperson, Region, SaleDate, Amount, RegionRank) :-
    sales(_, Salesperson, Region, Amount, SaleDate),
    Year = sql_extract(year, SaleDate),
    RegionRank = sql_window(rank, [], [Region, Year], [(Amount, desc)]).

% RANK() OVER (PARTITION BY region, STRFTIME('%Y', sale_date) ORDER BY amount DESC)
```

## Multiple Window Functions

Use multiple window functions in the same query:

```prolog
comprehensive_ranking(Name, Dept, Salary, RowNum, Rank, DenseRank) :-
    employees(_, Name, Dept, Salary, _),
    RowNum = sql_window(row_number, [], [Dept], [(Salary, desc)]),
    Rank = sql_window(rank, [], [Dept], [(Salary, desc)]),
    DenseRank = sql_window(dense_rank, [], [Dept], [(Salary, desc)]).
```

## Filtering Window Results

Use a CTE or subquery to filter on window function results (Chapter 7 covers CTEs):

```prolog
% Top 3 earners per department
% This requires a CTE - simplified here
top_earners_per_dept(Name, Dept, Salary, Rank) :-
    employees(_, Name, Dept, Salary, _),
    Rank = sql_window(rank, [], [Dept], [(Salary, desc)]),
    Rank =< 3.  % Note: This needs CTE to work properly
```

## Complete Example

```prolog
% file: window_functions_example.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, hire_date-text]).
:- sql_table(sales, [id-integer, salesperson-text, region-text, amount-real, sale_date-text]).

% Ranking functions
emp_row_number(Name, Dept, Salary, RowNum) :-
    employees(_, Name, Dept, Salary, _),
    RowNum = sql_window(row_number, [], [Dept], [(Salary, desc)]).

emp_rank(Name, Dept, Salary, Rank) :-
    employees(_, Name, Dept, Salary, _),
    Rank = sql_window(rank, [], [Dept], [(Salary, desc)]).

emp_dense_rank(Name, Dept, Salary, DenseRank) :-
    employees(_, Name, Dept, Salary, _),
    DenseRank = sql_window(dense_rank, [], [Dept], [(Salary, desc)]).

% Value functions
sales_lag(Person, Amount, Date, PrevAmt) :-
    sales(_, Person, _, Amount, Date),
    PrevAmt = sql_window(lag, [Amount, 1], [Person], [(Date, asc)]).

sales_lead(Person, Amount, Date, NextAmt) :-
    sales(_, Person, _, Amount, Date),
    NextAmt = sql_window(lead, [Amount, 1], [Person], [(Date, asc)]).

% Aggregate windows
running_total(Person, Date, Amount, Total) :-
    sales(_, Person, _, Amount, Date),
    Total = sql_window(sum, [Amount], [Person], [(Date, asc)]).

% With frame specification
moving_avg(Person, Date, Amount, Avg3Day) :-
    sales(_, Person, _, Amount, Date),
    Avg3Day = sql_window(avg, [Amount], [Person], [(Date, asc)],
                         rows_between(2 preceding, current_row)).

% Global rank (no partition)
global_salary_rank(Name, Salary, GlobalRank) :-
    employees(_, Name, _, Salary, _),
    GlobalRank = sql_window(rank, [], [], [(Salary, desc)]).

% Multiple windows
all_rankings(Name, Dept, Salary, Row, Rnk, Dense) :-
    employees(_, Name, Dept, Salary, _),
    Row = sql_window(row_number, [], [Dept], [(Salary, desc)]),
    Rnk = sql_window(rank, [], [Dept], [(Salary, desc)]),
    Dense = sql_window(dense_rank, [], [Dept], [(Salary, desc)]).

test :-
    format('~n=== Window Function Examples ===~n~n'),

    Tests = [
        emp_row_number/4 - 'ROW_NUMBER by dept',
        emp_rank/4 - 'RANK by dept',
        emp_dense_rank/4 - 'DENSE_RANK by dept',
        sales_lag/4 - 'LAG (previous amount)',
        sales_lead/4 - 'LEAD (next amount)',
        running_total/4 - 'Running total (SUM)',
        moving_avg/4 - '3-day moving average',
        global_salary_rank/3 - 'Global rank (no partition)',
        all_rankings/6 - 'Multiple window functions'
    ],

    forall(
        member(Pred - Desc, Tests),
        (
            format('~w:~n', [Desc]),
            compile_predicate_to_sql(Pred, [], SQL),
            format('  ~w~n~n', [SQL])
        )
    ).

:- initialization(test, main).
```

## Exercises

1. **Basic Ranking**: Create a query that ranks all employees by salary globally (not partitioned).

2. **Departmental Ranking**: Rank employees within each department by hire date (earliest = 1).

3. **LAG/LEAD**: For each order, show the order amount and the difference from the previous order by the same customer.

4. **Running Calculations**: Calculate a running total and running average of sales by salesperson.

5. **Complex**: Create a query showing:
   - Employee name, department, salary
   - Rank within department by salary
   - Percentage of department total salary
   - Difference from department average

## Summary

In this chapter, you learned:

- Ranking functions: `row_number`, `rank`, `dense_rank`
- Value functions: `lag`, `lead`, `first_value`, `last_value`
- Aggregate window functions: `sum`, `avg`, `count`, `min`, `max`
- Window frame specifications: `rows_between`, `range_between`
- Partitioning strategies
- Using multiple window functions in one query

## Next Chapter

In Chapter 7, we'll explore Common Table Expressions (CTEs):
- WITH clause for query organization
- Simplifying complex queries
- Using CTEs with window functions
