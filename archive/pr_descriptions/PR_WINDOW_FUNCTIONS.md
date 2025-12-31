# PR Title

```
Add SQL Window Functions support (Phase 5a)
```

# PR Description

## Summary

Implements **ranking window functions** for the SQL target, enabling ROW_NUMBER, RANK, DENSE_RANK, and NTILE operations with PARTITION BY and ORDER BY support.

## New Predicates

| Prolog Syntax | SQL Output | Description |
|---------------|------------|-------------|
| `row_number(Result, Options)` | `ROW_NUMBER() OVER(...)` | Unique sequential number |
| `rank(Result, Options)` | `RANK() OVER(...)` | Rank with gaps for ties |
| `dense_rank(Result, Options)` | `DENSE_RANK() OVER(...)` | Rank without gaps |
| `ntile(N, Result, Options)` | `NTILE(N) OVER(...)` | Divide into N buckets |

## Options Syntax

```prolog
Options = [
    partition_by(Column),           % Single column
    partition_by([Col1, Col2]),     % Multiple columns
    order_by(Column),               % ASC (default)
    order_by(Column, desc)          % DESC
]
```

## Examples

### RANK with PARTITION BY

```prolog
employee_dept_rank(Name, Dept, Salary, Rank) :-
    employees(_, Name, Dept, Salary),
    rank(Rank, [partition_by(Dept), order_by(Salary, desc)]).
```

**Generated SQL:**
```sql
SELECT name, dept, salary,
       RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rank
FROM employees;
```

### Multiple Window Functions

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
       NTILE(4) OVER (ORDER BY salary DESC) AS ntile
FROM employees;
```

### Window Function with WHERE Clause

```prolog
high_salary_numbered(Name, Salary, RowNum) :-
    employees(_, Name, _, Salary),
    Salary > 50000,
    row_number(RowNum, [order_by(Salary, desc)]).
```

**Generated SQL:**
```sql
SELECT name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num
FROM employees
WHERE salary > 50000;
```

## Test Results

```
✅ Test 1: Simple ROW_NUMBER
✅ Test 2: RANK with PARTITION BY
✅ Test 3: DENSE_RANK
✅ Test 4: NTILE (Quartiles)
✅ Test 5: Multiple Window Functions
✅ Test 6: ROW_NUMBER with WHERE Clause
✅ Test 7: RANK on different table
✅ Test 8: Output as SELECT (not VIEW)
```

## Files Changed

- `src/unifyweaver/targets/sql_target.pl` - Core implementation (+308 lines)
- `test_sql_window_functions.pl` - Test suite (new, 148 lines)

## Key Implementation Details

- **Window Function Detection**: `is_window_function/1` identifies window function goals
- **Separation**: `separate_window_functions/3` extracts window functions from WHERE constraints
- **OVER Clause Generation**: `generate_over_clause/3` handles PARTITION BY and ORDER BY
- **Smart Aliasing**: Column aliases based on function type (row_num, rank, dense_rank, ntile)
- **Integration**: Modified `compile_clause_to_select/5` and `compile_single_clause/6`

## Infrastructure for Future Phases

The implementation includes stubs for Phase 5b and 5c:
- `window_sum`, `window_avg`, `window_count`, `window_min`, `window_max`
- `lag`, `lead` value functions

## Breaking Changes

None. Pure feature addition, backwards compatible with existing SQL target functionality.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
