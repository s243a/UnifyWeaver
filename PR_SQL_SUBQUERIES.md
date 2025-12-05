# PR Title

```
Add SQL subquery support (Phase 4)
```

# PR Description

## Summary

Implements **IN**, **NOT IN**, **EXISTS**, and **NOT EXISTS** subqueries for the SQL target, enabling nested queries within WHERE clauses.

## New Predicates

| Predicate | SQL Output |
|-----------|------------|
| `in_query(Var, Pred/Arity)` | `column IN (SELECT ...)` |
| `not_in_query(Var, Pred/Arity)` | `column NOT IN (SELECT ...)` |
| `exists(Goal)` | `EXISTS (SELECT 1 FROM ... WHERE correlation)` |
| `not_exists(Goal)` | `NOT EXISTS (SELECT 1 FROM ... WHERE correlation)` |

## Examples

### IN Subquery

```prolog
high_budget_depts(Id) :-
    departments(Id, _, Budget, _),
    Budget > 100000.

high_budget_employees(Name) :-
    employees(_, Name, DeptId, _),
    in_query(DeptId, high_budget_depts/1).
```

**Generated SQL:**
```sql
SELECT name FROM employees
WHERE dept_id IN (
    SELECT id FROM departments WHERE budget > 100000
);
```

### EXISTS (Correlated)

```prolog
customers_with_orders(Name) :-
    customers(CustId, Name, _),
    exists(orders(_, CustId, _, _)).
```

**Generated SQL:**
```sql
SELECT name FROM customers
WHERE EXISTS (
    SELECT 1 FROM orders WHERE orders.customer_id = customers.id
);
```

## Test Results

```
✅ Test 1: Simple IN subquery
✅ Test 2: NOT IN subquery
✅ Test 3: EXISTS (correlated)
✅ Test 4: NOT EXISTS
✅ Test 5: Multiple subqueries (IN + EXISTS combined)
✅ Test 6: IN with multiple conditions in subquery
```

## Files Changed

- `src/unifyweaver/targets/sql_target.pl` - Core implementation (+130 lines)
- `test_sql_subqueries.pl` - Test suite (new, 150 lines)
- `proposals/SQL_SUBQUERIES_DESIGN.md` - Design document (new)

## Key Implementation Details

- **Correlation Detection**: EXISTS subqueries automatically detect correlated variables between outer and inner queries
- **Subquery Compilation**: Reuses existing clause compilation infrastructure
- **Multiple Subqueries**: Can combine IN and EXISTS in same WHERE clause

## Breaking Changes

None. Pure feature addition.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
