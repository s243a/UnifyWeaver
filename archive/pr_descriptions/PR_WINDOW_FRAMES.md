# PR Title

```
Add Window Frame Specification support (Phase 5d)
```

# PR Description

## Summary

Implements **ROWS/RANGE BETWEEN** frame clauses for window functions, completing the window function feature set.

## Frame Syntax

```prolog
frame(Type, Start, End)
```

| Parameter | Values |
|-----------|--------|
| Type | `rows`, `range`, `groups` |
| Start/End | `unbounded_preceding`, `unbounded_following`, `current_row`, `preceding(N)`, `following(N)` |

## Examples

### Rolling 3-Day Sum

```prolog
rolling_3_sum(Date, Amount, RollingSum) :-
    sales(_, Date, Amount),
    window_sum(Amount, RollingSum, [
        order_by(Date),
        frame(rows, preceding(2), current_row)
    ]).
```

**Generated SQL:**
```sql
SELECT date, amount,
       SUM(amount) OVER (ORDER BY date ASC
           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS running_sum
FROM sales;
```

### Centered Moving Average

```prolog
window_avg(Amount, CenterAvg, [
    order_by(Date),
    frame(rows, preceding(1), following(1))
])
```

**Generated SQL:**
```sql
AVG(amount) OVER (ORDER BY date ASC
    ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)
```

### Cumulative Sum

```prolog
frame(rows, unbounded_preceding, current_row)
```

**Generated SQL:**
```sql
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
```

### With PARTITION BY

```prolog
window_avg(Price, RollingAvg, [
    partition_by(Symbol),
    order_by(Date),
    frame(rows, preceding(4), current_row)
])
```

**Generated SQL:**
```sql
AVG(price) OVER (
    PARTITION BY symbol
    ORDER BY date ASC
    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
)
```

## Test Results

```
✅ Test 1: ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
✅ Test 2: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
✅ Test 3: ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
✅ Test 4: ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
✅ Test 5: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
✅ Test 6: Frame with PARTITION BY
✅ Test 7: Window function without frame (backward compat)
✅ Test 8: ROWS UNBOUNDED PRECEDING (single bound)
✅ Test 9: Multiple window functions with different frames
✅ Test 10: Output as SELECT
```

All existing tests continue to pass.

## Files Changed

- `src/unifyweaver/targets/sql_target.pl` - Core implementation (+75 lines)
- `test_sql_window_frames.pl` - Test suite (new, 200 lines)

## Breaking Changes

None. Backward compatible - window functions without frame specification work as before.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
