# PR Title

```
Add SQL Window Functions Phase 5b+5c (Aggregates & LAG/LEAD)
```

# PR Description

## Summary

Adds test coverage for **window aggregate functions** (Phase 5b) and **value functions** (Phase 5c). The infrastructure from Phase 5a required no changes - all predicates work out of the box.

## Phase 5b: Window Aggregates

| Prolog Syntax | SQL Output | Description |
|---------------|------------|-------------|
| `window_sum(Field, Result, Options)` | `SUM(field) OVER(...)` | Running/windowed sum |
| `window_avg(Field, Result, Options)` | `AVG(field) OVER(...)` | Running/windowed average |
| `window_count(Result, Options)` | `COUNT(*) OVER(...)` | Running/windowed count |
| `window_min(Field, Result, Options)` | `MIN(field) OVER(...)` | Running/windowed minimum |
| `window_max(Field, Result, Options)` | `MAX(field) OVER(...)` | Running/windowed maximum |

## Phase 5c: Value Functions

| Prolog Syntax | SQL Output | Description |
|---------------|------------|-------------|
| `lag(Field, Offset, Result, Options)` | `LAG(field, N) OVER(...)` | Previous row value |
| `lead(Field, Offset, Result, Options)` | `LEAD(field, N) OVER(...)` | Next row value |

## Examples

### Running Total with PARTITION BY

```prolog
regional_running_avg(Region, Date, Amount, RunningAvg) :-
    sales(_, Region, Date, Amount),
    window_avg(Amount, RunningAvg, [partition_by(Region), order_by(Date)]).
```

**Generated SQL:**
```sql
SELECT region, date, amount,
       AVG(amount) OVER (PARTITION BY region ORDER BY date ASC) AS running_avg
FROM sales;
```

### LAG - Previous Value

```prolog
price_with_previous(Symbol, Date, Price, PrevPrice) :-
    stock_prices(_, Symbol, Date, Price),
    lag(Price, 1, PrevPrice, [partition_by(Symbol), order_by(Date)]).
```

**Generated SQL:**
```sql
SELECT symbol, date, price,
       LAG(price, 1) OVER (PARTITION BY symbol ORDER BY date ASC) AS prev_value
FROM stock_prices;
```

### LAG + LEAD Together

```prolog
price_with_neighbors(Symbol, Date, Price, PrevPrice, NextPrice) :-
    stock_prices(_, Symbol, Date, Price),
    lag(Price, 1, PrevPrice, [partition_by(Symbol), order_by(Date)]),
    lead(Price, 1, NextPrice, [partition_by(Symbol), order_by(Date)]).
```

**Generated SQL:**
```sql
SELECT symbol, date, price,
       LAG(price, 1) OVER (PARTITION BY symbol ORDER BY date ASC) AS prev_value,
       LEAD(price, 1) OVER (PARTITION BY symbol ORDER BY date ASC) AS next_value
FROM stock_prices;
```

## Test Results

```
✅ Test 1: Running Sum (window_sum)
✅ Test 2: Running Average with PARTITION BY (window_avg)
✅ Test 3: Running Count (window_count)
✅ Test 4: Running Min (window_min)
✅ Test 5: Running Max (window_max)
✅ Test 6: LAG - Previous Value
✅ Test 7: LEAD - Next Value
✅ Test 8: LAG with offset 2
✅ Test 9: Multiple Window Aggregates
✅ Test 10: LAG and LEAD together
✅ Test 11: Window Aggregate with WHERE Clause
✅ Test 12: Mix Ranking with Aggregates
```

## Files Changed

- `test_sql_window_5bc.pl` - Test suite (new, 192 lines)

## Implementation Notes

The predicates were already implemented in Phase 5a. This PR validates they work correctly with comprehensive test coverage. No code changes to `sql_target.pl` were needed.

## Breaking Changes

None. Pure test addition.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
