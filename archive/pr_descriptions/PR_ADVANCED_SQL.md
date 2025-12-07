# PR Title

```
Add advanced SQL features: DISTINCT, CTEs, and expression infrastructure
```

# PR Description

## Summary

Adds **DISTINCT support**, **Common Table Expressions (CTEs)**, and infrastructure for **CASE WHEN expressions**, **scalar subqueries**, **column aliases**, and **derived tables**.

## New Features

### DISTINCT

```prolog
unique_departments(Dept) :-
    employees(_, _, Dept, _, _),
    sql_distinct.
```

**Generated SQL:**
```sql
SELECT DISTINCT dept FROM employees;
```

### CTEs (WITH clause)

```prolog
% Define a predicate for the CTE
high_earners(Name, Salary) :-
    employees(_, Name, _, Salary, _),
    Salary > 50000.

% Use compile_with_cte/4
compile_with_cte(
    [cte(high_earners, high_earners/2)],
    main_query/2,
    [],
    SQL
).
```

**Generated SQL:**
```sql
WITH high_earners AS (
    SELECT name, salary FROM employees WHERE salary > 50000
)
SELECT ...;
```

### DISTINCT with ORDER BY and Window Functions

```prolog
ranked_unique_depts(Dept, Rank) :-
    employees(_, _, Dept, Salary, _),
    sql_distinct,
    rank(Rank, [order_by(Salary, desc)]).
```

**Generated SQL:**
```sql
SELECT DISTINCT dept, RANK() OVER (ORDER BY salary DESC) AS rank
FROM employees;
```

## Infrastructure Added (for future use)

| Feature | Predicate | Description |
|---------|-----------|-------------|
| CASE WHEN | `sql_case/2`, `sql_case/3` | Conditional expressions |
| Scalar subqueries | `sql_scalar/2`, `sql_scalar/3` | Subquery in SELECT |
| Column aliases | `sql_as/2` | Column AS alias |
| Derived tables | `sql_from/2` | Subquery in FROM |

## New Exported Predicate

```prolog
compile_with_cte(+CTEs, +MainPred, +Options, -SQLCode)
```

## Test Results

```
✅ Test 1: DISTINCT
✅ Test 2: DISTINCT with ORDER BY
✅ Test 3: CTE (WITH clause)
✅ Test 4: Multiple CTEs
✅ Test 5: DISTINCT with simple query
✅ Test 6: DISTINCT with Window Function
✅ Test 7: CTE with View Name
✅ Test 8: DISTINCT output as SELECT
```

All existing tests continue to pass.

## Files Changed

- `src/unifyweaver/targets/sql_target.pl` - Core implementation (+260 lines)
- `test_sql_advanced.pl` - Test suite (new, 180 lines)

## Breaking Changes

None. Pure feature addition, backwards compatible.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
