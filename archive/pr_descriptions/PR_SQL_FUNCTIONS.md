# PR Title

```
Add SQL functions: NULL handling, string, date, BETWEEN, LIKE
```

# PR Description

## Summary

Adds comprehensive **SQL function support** including NULL handling, string manipulation, date functions, and WHERE clause predicates (BETWEEN, LIKE, IN).

## New Features

### NULL Handling Functions

| Prolog | SQL |
|--------|-----|
| `sql_coalesce([Col, Default])` | `COALESCE(col, default)` |
| `sql_nullif(Col, Value)` | `NULLIF(col, value)` |
| `sql_ifnull(Col, Default)` | `IFNULL(col, default)` |

```prolog
employee_with_default_dept(Name, Dept) :-
    employees(_, Name, D, _, _, _),
    Dept = sql_coalesce([D, 'Unknown']).
```

**Generated SQL:**
```sql
SELECT name, COALESCE(dept, 'Unknown') FROM employees;
```

### String Functions

| Prolog | SQL |
|--------|-----|
| `sql_concat([A, B, C])` | `a \|\| b \|\| c` |
| `sql_upper(Col)` | `UPPER(col)` |
| `sql_lower(Col)` | `LOWER(col)` |
| `sql_substring(Col, Start, Len)` | `SUBSTR(col, start, len)` |
| `sql_trim(Col)` | `TRIM(col)` |
| `sql_ltrim(Col)` / `sql_rtrim(Col)` | `LTRIM(col)` / `RTRIM(col)` |
| `sql_length(Col)` | `LENGTH(col)` |
| `sql_replace(Col, From, To)` | `REPLACE(col, from, to)` |

```prolog
employee_display(DisplayName) :-
    employees(_, Name, Dept, _, _, _),
    DisplayName = sql_as(sql_concat([sql_upper(Name), ' (', Dept, ')']), display_name).
```

**Generated SQL:**
```sql
SELECT UPPER(name) || ' (' || dept || ')' AS display_name FROM employees;
```

### Date Functions

| Prolog | SQL |
|--------|-----|
| `sql_date(Col)` | `DATE(col)` |
| `sql_datetime(Col)` | `DATETIME(col)` |
| `sql_date_add(Col, N, Unit)` | `DATE(col, '+N unit')` |
| `sql_date_diff(Col1, Col2)` | `JULIANDAY(col1) - JULIANDAY(col2)` |
| `sql_extract(Part, Col)` | `STRFTIME('%Y', col)` (for year) |
| `sql_strftime(Format, Col)` | `STRFTIME(format, col)` |

```prolog
order_due_date(Id, DueDate) :-
    orders(Id, _, OrderDate, _, _),
    DueDate = sql_date_add(OrderDate, 7, days).
```

**Generated SQL:**
```sql
SELECT id, DATE(order_date, '+7 days') FROM orders;
```

### WHERE Clause Predicates

#### BETWEEN

```prolog
mid_salary_employees(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    sql_between(Salary, 50000, 100000).
```

**Generated SQL:**
```sql
SELECT name, salary FROM employees WHERE salary BETWEEN 50000 AND 100000;
```

#### LIKE / NOT LIKE

```prolog
employees_starting_with_j(Name) :-
    employees(_, Name, _, _, _, _),
    sql_like(Name, 'J%').
```

**Generated SQL:**
```sql
SELECT name FROM employees WHERE name LIKE 'J%';
```

#### IN / NOT IN (list values)

```prolog
engineering_depts(Name, Dept) :-
    employees(_, Name, Dept, _, _, _),
    sql_in(Dept, [engineering, 'r&d', development]).
```

**Generated SQL:**
```sql
SELECT name, dept FROM employees WHERE dept IN ('engineering', 'r&d', 'development');
```

#### IS NULL / IS NOT NULL

```prolog
unshipped_orders(Id) :-
    orders(Id, _, _, ShipDate, _),
    sql_is_null(ShipDate).
```

**Generated SQL:**
```sql
SELECT id FROM orders WHERE ship_date IS NULL;
```

### Nested Functions

Functions can be nested and combined:

```prolog
upper_trimmed_name(Name, Result) :-
    employees(_, Name, _, _, _, _),
    Result = sql_upper(sql_trim(Name)).
```

**Generated SQL:**
```sql
SELECT name, UPPER(TRIM(name)) FROM employees;
```

## Test Results

```
✅ Test 1: COALESCE
✅ Test 2: NULLIF
✅ Test 3: IFNULL
✅ Test 4: CONCAT (string concatenation)
✅ Test 5: UPPER/LOWER
✅ Test 6: SUBSTRING
✅ Test 7: TRIM
✅ Test 8: LENGTH
✅ Test 9: REPLACE
✅ Test 10: DATE
✅ Test 11: DATE_ADD
✅ Test 12: DATE_DIFF
✅ Test 13: EXTRACT (date parts)
✅ Test 14: STRFTIME
✅ Test 15: BETWEEN (numeric)
✅ Test 16: BETWEEN (dates)
✅ Test 17: NOT BETWEEN
✅ Test 18: LIKE (starts with)
✅ Test 19: LIKE (contains)
✅ Test 20: NOT LIKE
✅ Test 21: GLOB (SQLite)
✅ Test 22: IN (list)
✅ Test 23: NOT IN (list)
✅ Test 24: IS NULL
✅ Test 25: IS NOT NULL
✅ Test 26: Nested functions
✅ Test 27: Combined functions with alias
✅ Test 28: Output as SELECT
```

All existing tests continue to pass.

## Files Changed

- `src/unifyweaver/targets/sql_target.pl` - Core implementation (+379 lines)
- `test_sql_functions.pl` - Test suite (new, 425 lines)

## Breaking Changes

None. Pure feature addition, backwards compatible.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
