# PR Title

```
Add ORDER BY and LIMIT/OFFSET support to SQL Target
```

# PR Description

## Summary

Implements **query-level sorting** and **pagination** for the SQL target, enabling sorted results and row limiting.

## New Predicates

| Prolog Syntax | SQL Output | Description |
|---------------|------------|-------------|
| `sql_order_by(Column)` | `ORDER BY column ASC` | Sort ascending |
| `sql_order_by(Column, asc)` | `ORDER BY column ASC` | Sort ascending (explicit) |
| `sql_order_by(Column, desc)` | `ORDER BY column DESC` | Sort descending |
| `sql_limit(N)` | `LIMIT N` | Limit to N rows |
| `sql_offset(N)` | `OFFSET N` | Skip first N rows |

## Examples

### ORDER BY with LIMIT

```prolog
top_5_employees(Name, Salary) :-
    employees(_, Name, _, Salary),
    sql_order_by(Salary, desc),
    sql_limit(5).
```

**Generated SQL:**
```sql
SELECT name, salary
FROM employees
ORDER BY salary DESC
LIMIT 5;
```

### Pagination (LIMIT + OFFSET)

```prolog
employees_page_2(Name, Salary) :-
    employees(_, Name, _, Salary),
    sql_order_by(Name),
    sql_limit(10),
    sql_offset(10).
```

**Generated SQL:**
```sql
SELECT name, salary
FROM employees
ORDER BY name ASC
LIMIT 10 OFFSET 10;
```

### Multiple ORDER BY Columns

```prolog
employees_by_dept_salary(Name, Dept, Salary) :-
    employees(_, Name, Dept, Salary),
    sql_order_by(Dept, asc),
    sql_order_by(Salary, desc).
```

**Generated SQL:**
```sql
SELECT name, dept, salary
FROM employees
ORDER BY dept ASC, salary DESC;
```

### Combined with WHERE and Window Functions

```prolog
ranked_employees_sorted(Name, Salary, Rank) :-
    employees(_, Name, _, Salary),
    rank(Rank, [order_by(Salary, desc)]),
    sql_order_by(Salary, desc),
    sql_limit(10).
```

**Generated SQL:**
```sql
SELECT name, salary, RANK() OVER (ORDER BY salary DESC) AS rank
FROM employees
ORDER BY salary DESC
LIMIT 10;
```

## Test Results

```
✅ Test 1: Simple ORDER BY (ascending)
✅ Test 2: ORDER BY descending
✅ Test 3: Multiple ORDER BY columns
✅ Test 4: Simple LIMIT
✅ Test 5: LIMIT with OFFSET (pagination)
✅ Test 6: ORDER BY with WHERE clause
✅ Test 7: ORDER BY + LIMIT + WHERE
✅ Test 8: OFFSET only
✅ Test 9: ORDER BY with window function
✅ Test 10: Output as SELECT (not VIEW)
```

## Files Changed

- `src/unifyweaver/targets/sql_target.pl` - Core implementation (+125 lines)
- `test_sql_orderby_limit.pl` - Test suite (new, 200 lines)

## Key Implementation Details

- **Query Modifiers**: `is_query_modifier/1` identifies ORDER BY, LIMIT, OFFSET
- **Separation**: `separate_query_modifiers/4` extracts modifiers from WHERE constraints
- **Order Specs**: Supports multiple ORDER BY columns with individual directions
- **format_sql_extended**: New SQL formatter handling all clause types

## Breaking Changes

None. Pure feature addition, backwards compatible.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
