# SQL Target: Subquery Support (Phase 4)

**Date:** 2025-12-05
**Status:** Design Phase
**Branch:** `feature/sql-target-subqueries`

## Overview

Add support for SQL subqueries, enabling nested queries within WHERE clauses and SELECT expressions.

## Subquery Types

### 1. WHERE IN Subquery (Priority: HIGH)

Most common subquery pattern - filter rows where a value exists in another query's results.

**SQL:**
```sql
SELECT name FROM employees
WHERE department_id IN (SELECT id FROM departments WHERE budget > 100000);
```

**Proposed Prolog Syntax:**

```prolog
% Option A: Using in_query/2
high_budget_employees(Name) :-
    employees(_, Name, DeptId),
    in_query(DeptId, high_budget_dept_ids/1).

high_budget_dept_ids(Id) :-
    departments(Id, _, Budget),
    Budget > 100000.

% Option B: Using member-style with findall reference
high_budget_employees(Name) :-
    employees(_, Name, DeptId),
    subquery_member(DeptId, departments(Id, _, Budget), Id, Budget > 100000).

% Option C: Inline subquery (most SQL-like)
high_budget_employees(Name) :-
    employees(_, Name, DeptId),
    DeptId in (departments(Id, _, Budget), Budget > 100000, select(Id)).
```

**Recommendation:** Option A - cleanest separation, reuses existing predicate compilation.

---

### 2. WHERE NOT IN Subquery (Priority: HIGH)

Negated membership check.

**SQL:**
```sql
SELECT name FROM employees
WHERE department_id NOT IN (SELECT id FROM departments WHERE active = false);
```

**Proposed Prolog Syntax:**

```prolog
active_employees(Name) :-
    employees(_, Name, DeptId),
    not_in_query(DeptId, inactive_dept_ids/1).

inactive_dept_ids(Id) :-
    departments(Id, _, _, Active),
    Active = false.
```

---

### 3. WHERE EXISTS Subquery (Priority: MEDIUM)

Check if a correlated subquery returns any rows.

**SQL:**
```sql
SELECT name FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
```

**Proposed Prolog Syntax:**

```prolog
customers_with_orders(Name) :-
    customers(CustId, Name, _),
    exists_query(orders(_, CustId, _, _)).

% Or with explicit correlation
customers_with_orders(Name) :-
    customers(CustId, Name, _),
    exists(orders(_, CustId, _, _)).
```

---

### 4. Scalar Subquery (Priority: LOW)

Single-value subquery in SELECT clause.

**SQL:**
```sql
SELECT name,
       (SELECT COUNT(*) FROM orders WHERE customer_id = c.id) as order_count
FROM customers c;
```

**Proposed Prolog Syntax:**

```prolog
customer_order_count(Name, Count) :-
    customers(CustId, Name, _),
    scalar_query(Count, count, orders(_, CustId, _, _)).
```

---

## Implementation Design

### Pattern Detection

```prolog
%% is_subquery_pattern(+Goal, -Type, -Details)
%  Detect subquery patterns in clause body
%
is_subquery_pattern(in_query(Var, Pred/Arity), in, subquery(Var, Pred, Arity)).
is_subquery_pattern(not_in_query(Var, Pred/Arity), not_in, subquery(Var, Pred, Arity)).
is_subquery_pattern(exists_query(Goal), exists, subquery(Goal)).
is_subquery_pattern(exists(Goal), exists, subquery(Goal)).
```

### Code Generation

```prolog
%% compile_subquery(+Type, +Details, +Context, -SQLFragment)
%
compile_subquery(in, subquery(Var, Pred, Arity), Context, SQL) :-
    % 1. Compile the referenced predicate to a SELECT
    compile_predicate_to_sql(Pred/Arity, [format(select)], SubquerySQL),

    % 2. Get the variable's column name from context
    var_to_column(Var, Context, ColumnName),

    % 3. Generate IN clause
    format(string(SQL), '~w IN (~w)', [ColumnName, SubquerySQL]).

compile_subquery(not_in, subquery(Var, Pred, Arity), Context, SQL) :-
    compile_predicate_to_sql(Pred/Arity, [format(select)], SubquerySQL),
    var_to_column(Var, Context, ColumnName),
    format(string(SQL), '~w NOT IN (~w)', [ColumnName, SubquerySQL]).

compile_subquery(exists, subquery(Goal), Context, SQL) :-
    % Extract predicate from goal
    functor(Goal, Pred, Arity),

    % Compile with correlated variables
    extract_correlated_vars(Goal, Context, CorrelatedVars),
    compile_correlated_subquery(Pred/Arity, CorrelatedVars, SubquerySQL),

    format(string(SQL), 'EXISTS (~w)', [SubquerySQL]).
```

### Correlated Subqueries

For EXISTS and correlated IN, we need to handle variables from the outer query:

```prolog
%% extract_correlated_vars(+Goal, +OuterContext, -CorrelatedVars)
%  Find variables in Goal that are bound in OuterContext
%
extract_correlated_vars(Goal, OuterContext, CorrelatedVars) :-
    term_variables(Goal, GoalVars),
    findall(Var-Column,
            (member(Var, GoalVars),
             var_bound_in_context(Var, OuterContext, Column)),
            CorrelatedVars).

%% compile_correlated_subquery(+Pred, +CorrelatedVars, -SQL)
%  Generate subquery with correlation conditions
%
compile_correlated_subquery(Pred/Arity, CorrelatedVars, SQL) :-
    % Compile base query
    compile_predicate_to_sql(Pred/Arity, [format(select)], BaseSQL),

    % Add correlation conditions
    generate_correlation_conditions(CorrelatedVars, CondSQL),

    (   CondSQL = ''
    ->  SQL = BaseSQL
    ;   format(string(SQL), '~w WHERE ~w', [BaseSQL, CondSQL])
    ).
```

---

## Test Cases

### Test 1: Simple IN Subquery

```prolog
:- sql_table(employees, [id-integer, name-text, dept_id-integer]).
:- sql_table(departments, [id-integer, name-text, budget-integer]).

% Employees in high-budget departments
high_budget_employees(Name) :-
    employees(_, Name, DeptId),
    in_query(DeptId, high_budget_depts/1).

high_budget_depts(Id) :-
    departments(Id, _, Budget),
    Budget > 100000.
```

**Expected SQL:**
```sql
CREATE VIEW high_budget_employees AS
SELECT employees.name
FROM employees
WHERE employees.dept_id IN (
    SELECT departments.id
    FROM departments
    WHERE departments.budget > 100000
);
```

### Test 2: NOT IN Subquery

```prolog
% Employees NOT in inactive departments
active_employees(Name) :-
    employees(_, Name, DeptId),
    not_in_query(DeptId, inactive_depts/1).

inactive_depts(Id) :-
    departments(Id, _, _, Active),
    Active = 0.
```

**Expected SQL:**
```sql
CREATE VIEW active_employees AS
SELECT employees.name
FROM employees
WHERE employees.dept_id NOT IN (
    SELECT departments.id
    FROM departments
    WHERE departments.active = 0
);
```

### Test 3: EXISTS Subquery

```prolog
:- sql_table(customers, [id-integer, name-text]).
:- sql_table(orders, [id-integer, customer_id-integer, total-integer]).

% Customers who have placed orders
customers_with_orders(Name) :-
    customers(CustId, Name),
    exists(orders(_, CustId, _)).
```

**Expected SQL:**
```sql
CREATE VIEW customers_with_orders AS
SELECT customers.name
FROM customers
WHERE EXISTS (
    SELECT 1
    FROM orders
    WHERE orders.customer_id = customers.id
);
```

### Test 4: Multiple Subqueries

```prolog
% Employees in high-budget depts who have active projects
elite_employees(Name) :-
    employees(EmpId, Name, DeptId),
    in_query(DeptId, high_budget_depts/1),
    exists(project_assignments(_, EmpId, _)).
```

**Expected SQL:**
```sql
CREATE VIEW elite_employees AS
SELECT employees.name
FROM employees
WHERE employees.dept_id IN (
    SELECT departments.id FROM departments WHERE departments.budget > 100000
)
AND EXISTS (
    SELECT 1 FROM project_assignments WHERE project_assignments.emp_id = employees.id
);
```

---

## Implementation Plan

### Phase 4a: WHERE IN Subquery

1. Add `in_query/2` pattern detection
2. Implement subquery compilation
3. Generate nested SELECT
4. Add WHERE IN clause generation
5. Tests: 3 basic IN subquery tests

### Phase 4b: WHERE NOT IN Subquery

1. Add `not_in_query/2` pattern
2. Reuse IN compilation with negation
3. Tests: 2 NOT IN tests

### Phase 4c: EXISTS Subquery

1. Add `exists/1` pattern detection
2. Implement correlated variable extraction
3. Generate EXISTS with correlation
4. Tests: 3 EXISTS tests (simple, correlated, combined)

### Phase 4d: Advanced (Future)

1. Scalar subqueries in SELECT
2. Subqueries in FROM (derived tables)
3. Multiple levels of nesting

---

## API Changes

### New Predicates

```prolog
%% in_query(+Value, +PredicateIndicator)
%  True if Value is in the result set of Predicate
%  Compiles to: WHERE Value IN (SELECT ... FROM Predicate)
%
in_query(Value, Pred/Arity).

%% not_in_query(+Value, +PredicateIndicator)
%  True if Value is NOT in the result set of Predicate
%  Compiles to: WHERE Value NOT IN (SELECT ... FROM Predicate)
%
not_in_query(Value, Pred/Arity).

%% exists(+Goal)
%  True if Goal has at least one solution
%  Compiles to: WHERE EXISTS (SELECT 1 FROM ... WHERE correlation)
%
exists(Goal).
```

### Options

```prolog
compile_predicate_to_sql(Pred/Arity, Options, SQL).

% New options:
%   - subquery_style(in|exists|scalar) - hint for subquery detection
%   - correlation_mode(auto|manual) - how to detect correlated vars
```

---

## Compatibility

- **SQLite**: Full support for IN, NOT IN, EXISTS
- **PostgreSQL**: Full support + additional optimizations
- **MySQL**: Full support

---

## Files to Modify

1. **`sql_target.pl`**
   - Add `in_query/2`, `not_in_query/2`, `exists/1` pattern detection
   - Add `compile_subquery/4` predicate
   - Modify `compile_where_clause/3` to handle subqueries

2. **New test file**: `test_sql_subqueries.pl`

3. **Documentation**: Update README with subquery examples

---

## Risk Assessment

**Low Risk:**
- IN/NOT IN subqueries are straightforward
- Reuses existing compilation infrastructure

**Medium Risk:**
- EXISTS correlation requires careful variable tracking
- Need to handle nested scopes correctly

**Mitigation:**
- Start with non-correlated subqueries
- Add correlation in Phase 4c
- Comprehensive test coverage
