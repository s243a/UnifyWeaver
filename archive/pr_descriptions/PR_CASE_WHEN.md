# PR Title

```
Integrate CASE WHEN expressions into SELECT generation
```

# PR Description

## Summary

Integrates **CASE WHEN expressions** into SQL SELECT generation, completing the CASE/alias infrastructure added in the advanced SQL features PR.

## New Features

### Simple CASE (Value Mapping)

```prolog
employee_status_label(Name, Status, StatusLabel) :-
    employees(_, Name, _, _, Status),
    StatusLabel = sql_case(Status, [
        active-'Active Employee',
        inactive-'Inactive',
        terminated-'Terminated'
    ], 'Unknown').
```

**Generated SQL:**
```sql
SELECT name, status, CASE status
    WHEN 'active' THEN 'Active Employee'
    WHEN 'inactive' THEN 'Inactive'
    WHEN 'terminated' THEN 'Terminated'
    ELSE 'Unknown' END
FROM employees;
```

### Searched CASE (Condition-Based)

```prolog
salary_tier(Name, Salary, Tier) :-
    employees(_, Name, _, Salary, _),
    Tier = sql_case([
        when(Salary > 100000, 'Executive'),
        when(Salary > 50000, 'Senior'),
        when(Salary > 30000, 'Mid-Level')
    ], 'Entry-Level').
```

**Generated SQL:**
```sql
SELECT name, salary, CASE
    WHEN salary > 100000 THEN 'Executive'
    WHEN salary > 50000 THEN 'Senior'
    WHEN salary > 30000 THEN 'Mid-Level'
    ELSE 'Entry-Level' END
FROM employees;
```

### Column Aliases

```prolog
employee_with_alias(EmployeeName, Dept) :-
    employees(_, Name, Dept, _, _),
    EmployeeName = sql_as(Name, employee_name).
```

**Generated SQL:**
```sql
SELECT name AS employee_name, dept FROM employees;
```

### CASE with Alias

```prolog
priority_order(Id, Amount, PriorityLabel) :-
    orders(Id, _, Amount, Priority),
    PriorityLabel = sql_as(
        sql_case(Priority, [
            high-'Urgent',
            medium-'Normal',
            low-'Low Priority'
        ], 'Unclassified'),
        priority_label
    ).
```

**Generated SQL:**
```sql
SELECT id, amount, CASE priority
    WHEN 'high' THEN 'Urgent'
    WHEN 'medium' THEN 'Normal'
    WHEN 'low' THEN 'Low Priority'
    ELSE 'Unclassified' END AS priority_label
FROM orders;
```

## Implementation Details

Added expression binding extraction to capture `Var = sql_case(...)` patterns from clause body constraints and apply them to head arguments before SELECT generation.

Key predicates added/modified:
- `extract_expression_bindings/3` - Extract CASE/alias bindings from constraints
- `apply_expression_bindings/3` - Apply bindings to head arguments
- Modified `compile_single_clause` to use expression bindings

## Test Results

```
✅ Test 1: Simple CASE (value mapping)
✅ Test 2: Searched CASE (condition-based)
✅ Test 3: Column with alias
✅ Test 4: CASE with alias
✅ Test 5: Multiple CASE expressions
✅ Test 6: CASE with less-than condition
✅ Test 7: CASE with equality condition
✅ Test 8: Simple CASE with ELSE NULL
✅ Test 9: Output as SELECT
✅ Test 10: CASE with no ELSE
```

All existing tests continue to pass.

## Files Changed

- `src/unifyweaver/targets/sql_target.pl` - Core implementation (+92 lines modified)
- `test_sql_case_when.pl` - Test suite (new, 205 lines)

## Breaking Changes

None. Pure feature integration, backwards compatible.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
