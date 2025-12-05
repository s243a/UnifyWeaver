# SQL Target: Window Functions (Phase 5)

**Date:** 2025-12-05
**Status:** Planning
**Branch:** `feature/sql-target-window-functions` (to be created)

## Overview

Add support for SQL window functions, enabling ranking, running totals, and row-relative calculations without reducing result set size.

## Motivation

Window functions are essential for:
- **Ranking**: Top N per group, percentiles
- **Running calculations**: Cumulative sums, moving averages
- **Row comparisons**: Compare to previous/next row (LAG/LEAD)
- **Analytics**: Business intelligence queries

## Prolog-to-SQL Mapping

### Design Philosophy

Window functions don't naturally exist in Prolog's set-based semantics, but we can express the **declarative intent**:
- "I want each employee with their rank by salary" → `rank(Rank, [order_by(Salary, desc)])`
- "I want running total of sales" → `window_sum(Amount, RunningTotal, [order_by(Date)])`

The Prolog code declares *what* is wanted; SQL compilation determines *how*.

---

## Phase 5a: Ranking Functions

### Supported Functions

| Prolog Predicate | SQL Function | Description |
|------------------|--------------|-------------|
| `row_number(Result, Options)` | `ROW_NUMBER() OVER(...)` | Unique sequential number |
| `rank(Result, Options)` | `RANK() OVER(...)` | Rank with gaps for ties |
| `dense_rank(Result, Options)` | `DENSE_RANK() OVER(...)` | Rank without gaps |
| `ntile(N, Result, Options)` | `NTILE(N) OVER(...)` | Divide into N buckets |

### Options Syntax

```prolog
Options = [
    partition_by(Column),           % Single column
    partition_by([Col1, Col2]),     % Multiple columns
    order_by(Column),               % ASC (default)
    order_by(Column, asc),          % Explicit ASC
    order_by(Column, desc),         % DESC
    order_by([Col1, Col2-desc])     % Multiple columns with direction
]
```

### Example 1: Simple ROW_NUMBER

```prolog
:- sql_table(employees, [id-integer, name-text, salary-integer]).

employee_numbered(Name, Salary, RowNum) :-
    employees(_, Name, Salary),
    row_number(RowNum, [order_by(Salary, desc)]).
```

**Generated SQL:**
```sql
CREATE VIEW employee_numbered AS
SELECT name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num
FROM employees;
```

### Example 2: RANK with PARTITION BY

```prolog
:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer]).

employee_dept_rank(Name, Dept, Salary, Rank) :-
    employees(_, Name, Dept, Salary),
    rank(Rank, [partition_by(Dept), order_by(Salary, desc)]).
```

**Generated SQL:**
```sql
CREATE VIEW employee_dept_rank AS
SELECT name, dept, salary,
       RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rank
FROM employees;
```

### Example 3: DENSE_RANK

```prolog
employee_dense_rank(Name, Salary, DenseRank) :-
    employees(_, Name, _, Salary),
    dense_rank(DenseRank, [order_by(Salary, desc)]).
```

**Generated SQL:**
```sql
SELECT name, salary,
       DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rank
FROM employees;
```

### Example 4: NTILE (Quartiles)

```prolog
employee_quartile(Name, Salary, Quartile) :-
    employees(_, Name, _, Salary),
    ntile(4, Quartile, [order_by(Salary, desc)]).
```

**Generated SQL:**
```sql
SELECT name, salary,
       NTILE(4) OVER (ORDER BY salary DESC) AS quartile
FROM employees;
```

### Example 5: Multiple Window Functions

```prolog
employee_full_ranking(Name, Salary, RowNum, Rank, Quartile) :-
    employees(_, Name, _, Salary),
    row_number(RowNum, [order_by(Salary, desc)]),
    rank(Rank, [order_by(Salary, desc)]),
    ntile(4, Quartile, [order_by(Salary, desc)]).
```

**Generated SQL:**
```sql
SELECT name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num,
       RANK() OVER (ORDER BY salary DESC) AS rank,
       NTILE(4) OVER (ORDER BY salary DESC) AS quartile
FROM employees;
```

---

## Phase 5b: Window Aggregates

### Supported Functions

| Prolog Predicate | SQL Function | Description |
|------------------|--------------|-------------|
| `window_sum(Field, Result, Options)` | `SUM(field) OVER(...)` | Running/windowed sum |
| `window_avg(Field, Result, Options)` | `AVG(field) OVER(...)` | Running/windowed average |
| `window_count(Result, Options)` | `COUNT(*) OVER(...)` | Running/windowed count |
| `window_min(Field, Result, Options)` | `MIN(field) OVER(...)` | Running/windowed minimum |
| `window_max(Field, Result, Options)` | `MAX(field) OVER(...)` | Running/windowed maximum |

### Example 6: Running Total

```prolog
:- sql_table(sales, [id-integer, date-text, amount-integer]).

sales_running_total(Date, Amount, RunningTotal) :-
    sales(_, Date, Amount),
    window_sum(Amount, RunningTotal, [order_by(Date)]).
```

**Generated SQL:**
```sql
SELECT date, amount,
       SUM(amount) OVER (ORDER BY date) AS running_total
FROM sales;
```

### Example 7: Running Average with PARTITION

```prolog
:- sql_table(sales, [id-integer, region-text, date-text, amount-integer]).

regional_running_avg(Region, Date, Amount, RunningAvg) :-
    sales(_, Region, Date, Amount),
    window_avg(Amount, RunningAvg, [partition_by(Region), order_by(Date)]).
```

**Generated SQL:**
```sql
SELECT region, date, amount,
       AVG(amount) OVER (PARTITION BY region ORDER BY date) AS running_avg
FROM sales;
```

---

## Phase 5c: Value Functions (LAG/LEAD)

### Supported Functions

| Prolog Predicate | SQL Function | Description |
|------------------|--------------|-------------|
| `lag(Field, Offset, Result, Options)` | `LAG(field, offset) OVER(...)` | Previous row value |
| `lead(Field, Offset, Result, Options)` | `LEAD(field, offset) OVER(...)` | Next row value |
| `first_value(Field, Result, Options)` | `FIRST_VALUE(field) OVER(...)` | First value in window |
| `last_value(Field, Result, Options)` | `LAST_VALUE(field) OVER(...)` | Last value in window |

### Example 8: LAG (Previous Value)

```prolog
:- sql_table(stock_prices, [date-text, symbol-text, price-real]).

price_with_previous(Symbol, Date, Price, PrevPrice) :-
    stock_prices(Date, Symbol, Price),
    lag(Price, 1, PrevPrice, [partition_by(Symbol), order_by(Date)]).
```

**Generated SQL:**
```sql
SELECT symbol, date, price,
       LAG(price, 1) OVER (PARTITION BY symbol ORDER BY date) AS prev_price
FROM stock_prices;
```

### Example 9: LEAD (Next Value)

```prolog
price_with_next(Symbol, Date, Price, NextPrice) :-
    stock_prices(Date, Symbol, Price),
    lead(Price, 1, NextPrice, [partition_by(Symbol), order_by(Date)]).
```

**Generated SQL:**
```sql
SELECT symbol, date, price,
       LEAD(price, 1) OVER (PARTITION BY symbol ORDER BY date) AS next_price
FROM stock_prices;
```

---

## Phase 5d: Frame Specification (Future)

Advanced window frame control for partial windows.

### Frame Options

```prolog
% Frame specification
frame(Type, Start, End)

% Types: rows, range, groups
% Bounds: unbounded_preceding, unbounded_following, current_row, N preceding, N following

% Example: Last 3 rows including current
window_sum(Amount, Sum3, [
    order_by(Date),
    frame(rows, 2 preceding, current_row)
]).
```

**Generated SQL:**
```sql
SUM(amount) OVER (
    ORDER BY date
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
)
```

**Status:** Deferred to Phase 5d (complex, less commonly needed)

---

## Implementation Design

### Pattern Detection

Window functions are detected during constraint parsing:

```prolog
%% is_window_function(+Goal, -Type, -Details)
is_window_function(row_number(Result, Opts), ranking, window(row_number, Result, Opts)).
is_window_function(rank(Result, Opts), ranking, window(rank, Result, Opts)).
is_window_function(dense_rank(Result, Opts), ranking, window(dense_rank, Result, Opts)).
is_window_function(ntile(N, Result, Opts), ranking, window(ntile(N), Result, Opts)).
is_window_function(window_sum(Field, Result, Opts), aggregate, window(sum, Field, Result, Opts)).
is_window_function(lag(Field, N, Result, Opts), value, window(lag(N), Field, Result, Opts)).
```

### Separate Window Goals from Constraints

```prolog
%% separate_goals_with_windows(+Goals, -TableGoals, -Constraints, -WindowGoals)
separate_goals_with_windows(Goals, TableGoals, Constraints, WindowGoals) :-
    separate_goals(Goals, TableGoals, AllConstraints),
    partition(is_window_function, AllConstraints, WindowGoals, Constraints).
```

### Generate OVER Clause

```prolog
%% generate_over_clause(+Options, -OverClause)
generate_over_clause(Options, OverClause) :-
    (   member(partition_by(PartCols), Options)
    ->  generate_partition_clause(PartCols, PartClause)
    ;   PartClause = ''
    ),
    (   member(order_by(OrderSpec), Options)
    ->  generate_order_clause(OrderSpec, OrderClause)
    ;   member(order_by(OrderSpec, Dir), Options)
    ->  generate_order_clause(OrderSpec, Dir, OrderClause)
    ;   OrderClause = ''
    ),
    combine_over_parts(PartClause, OrderClause, OverClause).

%% generate_partition_clause(+Cols, -Clause)
generate_partition_clause(Col, Clause) :-
    atom(Col), !,
    format(string(Clause), 'PARTITION BY ~w', [Col]).
generate_partition_clause(Cols, Clause) :-
    is_list(Cols),
    atomic_list_concat(Cols, ', ', ColsStr),
    format(string(Clause), 'PARTITION BY ~w', [ColsStr]).
```

### Modify SELECT Generation

Window functions add columns to SELECT:

```prolog
%% generate_select_with_windows(+Args, +TableGoals, +WindowGoals, -SelectClause)
generate_select_with_windows(Args, TableGoals, WindowGoals, SelectClause) :-
    generate_select_items(Args, TableGoals, RegularItems),
    generate_window_items(WindowGoals, WindowItems),
    append(RegularItems, WindowItems, AllItems),
    atomic_list_concat(AllItems, ', ', ItemsStr),
    format(string(SelectClause), 'SELECT ~w', [ItemsStr]).
```

---

## Test Plan

### Phase 5a Tests (Ranking)

```
Test 1: Simple ROW_NUMBER
Test 2: ROW_NUMBER with ORDER BY DESC
Test 3: RANK with PARTITION BY
Test 4: DENSE_RANK
Test 5: NTILE (quartiles)
Test 6: Multiple window functions in one query
Test 7: Multiple PARTITION BY columns
Test 8: Multiple ORDER BY columns
```

### Phase 5b Tests (Aggregates)

```
Test 9: Running SUM
Test 10: Running AVG with PARTITION
Test 11: Running COUNT
Test 12: Running MIN/MAX
Test 13: Aggregate + Ranking combined
```

### Phase 5c Tests (Value Functions)

```
Test 14: LAG with offset 1
Test 15: LEAD with offset 1
Test 16: LAG with PARTITION BY
Test 17: LAG/LEAD combined
```

---

## Implementation Timeline

### Phase 5a: Ranking Functions (Priority: HIGH)
- **Scope**: row_number, rank, dense_rank, ntile
- **Options**: partition_by, order_by
- **Estimated**: ~100-150 lines
- **Tests**: 8 test cases

### Phase 5b: Window Aggregates (Priority: MEDIUM)
- **Scope**: window_sum, window_avg, window_count, window_min, window_max
- **Estimated**: ~50-80 lines (reuses 5a infrastructure)
- **Tests**: 5 test cases

### Phase 5c: Value Functions (Priority: MEDIUM)
- **Scope**: lag, lead, first_value, last_value
- **Estimated**: ~50-80 lines
- **Tests**: 4 test cases

### Phase 5d: Frame Specification (Priority: LOW)
- **Scope**: ROWS/RANGE BETWEEN ... AND ...
- **Estimated**: ~80-100 lines
- **Tests**: 4 test cases
- **Status**: Deferred (implement if needed)

---

## Compatibility

| Database | ROW_NUMBER | RANK | Window Aggregates | LAG/LEAD |
|----------|------------|------|-------------------|----------|
| SQLite 3.25+ | ✅ | ✅ | ✅ | ✅ |
| PostgreSQL | ✅ | ✅ | ✅ | ✅ |
| MySQL 8.0+ | ✅ | ✅ | ✅ | ✅ |
| SQL Server | ✅ | ✅ | ✅ | ✅ |

**Note:** SQLite requires version 3.25 or later for window functions.

---

## Files to Modify

1. **`sql_target.pl`**
   - Add window function pattern detection
   - Add OVER clause generation
   - Modify SELECT generation to include window columns
   - Add window function compilation

2. **`test_sql_window_functions.pl`** (new)
   - Comprehensive test suite

3. **`proposals/SQL_WINDOW_FUNCTIONS_DESIGN.md`** (this file)
   - Design documentation

---

## Risk Assessment

**Low Risk:**
- Ranking functions are straightforward
- OVER clause syntax is consistent across databases

**Medium Risk:**
- Variable binding for window results needs careful handling
- Multiple window functions in same query need coordination

**Mitigation:**
- Start with single window function per query
- Add multiple window support incrementally
- Comprehensive test coverage

---

## Summary

| Phase | Functions | Priority | Complexity |
|-------|-----------|----------|------------|
| 5a | row_number, rank, dense_rank, ntile | HIGH | Low |
| 5b | window_sum, window_avg, window_count | MEDIUM | Low |
| 5c | lag, lead, first_value, last_value | MEDIUM | Medium |
| 5d | Frame specification | LOW | High |

**Recommended approach:** Implement Phase 5a first, validate with tests, then proceed to 5b and 5c.
