<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 5: Subqueries

Subqueries are queries nested inside other queries. They enable complex filtering based on results from another query.

## Setup

```prolog
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [
    id-integer,
    name-text,
    dept-text,
    salary-integer,
    manager_id-integer
]).

:- sql_table(departments, [
    id-integer,
    name-text,
    budget-real,
    location-text
]).

:- sql_table(orders, [
    id-integer,
    customer_id-integer,
    product-text,
    amount-real,
    order_date-text
]).

:- sql_table(customers, [
    id-integer,
    name-text,
    email-text,
    country-text
]).

:- sql_table(products, [
    id-integer,
    name-text,
    category-text,
    price-real
]).
```

## IN Subquery

Find rows where a column value exists in another query's results.

### Basic IN Subquery

```prolog
sql_in_subquery(Column, SubqueryPredicate/Arity)
```

Example: Find employees in departments with budget over 1 million:

```prolog
% First, define the subquery predicate
high_budget_dept(DeptName) :-
    departments(_, DeptName, Budget, _),
    Budget > 1000000.

% Main query using IN subquery
employees_in_rich_depts(EmpName, Dept) :-
    employees(_, EmpName, Dept, _, _),
    sql_in_subquery(Dept, high_budget_dept/1).

?- compile_predicate_to_sql(employees_in_rich_depts/2, [], SQL).
```

Generated SQL:
```sql
SELECT name, dept FROM employees
WHERE dept IN (
    SELECT name FROM departments WHERE budget > 1000000
);
```

### IN with Aggregation Subquery

Find customers who have ordered more than average:

```prolog
% Subquery: customer IDs with high order totals
high_spenders(CustomerId) :-
    orders(_, CustomerId, _, Amount, _),
    sql_group_by([CustomerId]),
    TotalSpent = sql_sum(Amount),
    sql_having(TotalSpent > 5000).

% Main query
valuable_customers(Name, Email) :-
    customers(Id, Name, Email, _),
    sql_in_subquery(Id, high_spenders/1).

% SELECT name, email FROM customers
% WHERE id IN (
%     SELECT customer_id FROM orders
%     GROUP BY customer_id HAVING SUM(amount) > 5000
% );
```

## NOT IN Subquery

Find rows where a column value does NOT exist in the subquery results.

```prolog
sql_not_in_subquery(Column, SubqueryPredicate/Arity)
```

Example: Find employees NOT in the Engineering department:

```prolog
engineering_dept(DeptName) :-
    departments(_, DeptName, _, _),
    DeptName = 'Engineering'.

non_engineering_employees(EmpName) :-
    employees(_, EmpName, Dept, _, _),
    sql_not_in_subquery(Dept, engineering_dept/1).

% SELECT name FROM employees
% WHERE dept NOT IN (SELECT name FROM departments WHERE name = 'Engineering');
```

### Finding Unmatched Rows

Find customers who have never ordered:

```prolog
customers_who_ordered(CustomerId) :-
    orders(_, CustomerId, _, _, _).

customers_never_ordered(Name) :-
    customers(Id, Name, _, _),
    sql_not_in_subquery(Id, customers_who_ordered/1).

% SELECT name FROM customers
% WHERE id NOT IN (SELECT customer_id FROM orders);
```

## EXISTS Subquery

Check if ANY rows exist in the subquery. More efficient than IN for large datasets.

```prolog
sql_exists(SubqueryPredicate)
```

### Basic EXISTS

Find departments that have at least one employee:

```prolog
dept_has_employees(DeptName) :-
    employees(_, _, DeptName, _, _).

active_departments(DeptName, Budget) :-
    departments(_, DeptName, Budget, _),
    sql_exists(dept_has_employees(DeptName)).

% SELECT name, budget FROM departments d
% WHERE EXISTS (SELECT 1 FROM employees WHERE dept = d.name);
```

### EXISTS with Conditions

Find customers who have ordered a specific product:

```prolog
ordered_product(CustomerId, Product) :-
    orders(_, CustomerId, Product, _, _).

customers_who_bought_laptop(Name) :-
    customers(Id, Name, _, _),
    sql_exists(ordered_product(Id, 'Laptop')).

% SELECT name FROM customers c
% WHERE EXISTS (SELECT 1 FROM orders WHERE customer_id = c.id AND product = 'Laptop');
```

## NOT EXISTS Subquery

Find rows where NO matching rows exist in the subquery.

```prolog
sql_not_exists(SubqueryPredicate)
```

### Finding Missing Relationships

Find products that have never been ordered:

```prolog
product_ordered(ProductName) :-
    orders(_, _, ProductName, _, _).

products_never_ordered(Name, Price) :-
    products(_, Name, _, Price),
    sql_not_exists(product_ordered(Name)).

% SELECT name, price FROM products p
% WHERE NOT EXISTS (SELECT 1 FROM orders WHERE product = p.name);
```

### Finding Employees Without Subordinates

```prolog
has_subordinate(ManagerId) :-
    employees(_, _, _, _, ManagerId).

non_managers(Name) :-
    employees(Id, Name, _, _, _),
    sql_not_exists(has_subordinate(Id)).

% SELECT name FROM employees e
% WHERE NOT EXISTS (SELECT 1 FROM employees WHERE manager_id = e.id);
```

## Correlated vs Non-Correlated Subqueries

### Non-Correlated Subquery

The subquery runs once, independently of the outer query:

```prolog
% Subquery doesn't reference outer query
avg_salary_dept(Dept) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    AvgSal = sql_avg(Salary),
    sql_having(AvgSal > 60000).

employees_in_high_avg_depts(Name) :-
    employees(_, Name, Dept, _, _),
    sql_in_subquery(Dept, avg_salary_dept/1).
```

### Correlated Subquery

The subquery references a value from the outer query (runs once per outer row):

```prolog
% EXISTS subqueries are typically correlated
dept_employee_exists(DeptName) :-
    employees(_, _, DeptName, _, _).  % DeptName comes from outer query

depts_with_staff(Name) :-
    departments(_, Name, _, _),
    sql_exists(dept_employee_exists(Name)).  % Name is correlated
```

## Combining Subquery Types

Complex queries can combine multiple subquery patterns:

```prolog
% Find employees who:
% 1. Are in departments with budget > 500000
% 2. Have at least one subordinate

high_budget(DeptName) :-
    departments(_, DeptName, Budget, _),
    Budget > 500000.

has_report(ManagerId) :-
    employees(_, _, _, _, ManagerId).

senior_managers_rich_depts(Name, Dept) :-
    employees(Id, Name, Dept, _, _),
    sql_in_subquery(Dept, high_budget/1),
    sql_exists(has_report(Id)).

% SELECT name, dept FROM employees e
% WHERE dept IN (SELECT name FROM departments WHERE budget > 500000)
% AND EXISTS (SELECT 1 FROM employees WHERE manager_id = e.id);
```

## Subqueries vs JOINs

Many subqueries can be rewritten as JOINs. Choose based on:

| Use Subquery When | Use JOIN When |
|-------------------|---------------|
| Checking existence only | Need data from both tables |
| NOT IN / NOT EXISTS patterns | Simple relationship lookups |
| Complex filtering logic | Performance is critical |
| Readability matters | Multiple columns needed |

### Example: Same Result, Different Approach

Using subquery:
```prolog
has_order(CustomerId) :-
    orders(_, CustomerId, _, _, _).

customers_with_orders_sub(Name) :-
    customers(Id, Name, _, _),
    sql_exists(has_order(Id)).
```

Using JOIN (more efficient):
```prolog
customers_with_orders_join(Name) :-
    customers(Id, Name, _, _),
    orders(_, Id, _, _, _),
    sql_distinct.
```

## Complete Example

```prolog
% file: subqueries_example.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, manager_id-integer]).
:- sql_table(departments, [id-integer, name-text, budget-real, location-text]).
:- sql_table(orders, [id-integer, customer_id-integer, product-text, amount-real, order_date-text]).
:- sql_table(customers, [id-integer, name-text, email-text, country-text]).

% Subquery predicates
high_budget_dept(DeptName) :-
    departments(_, DeptName, Budget, _),
    Budget > 1000000.

customer_with_orders(CustomerId) :-
    orders(_, CustomerId, _, _, _).

has_subordinate(ManagerId) :-
    employees(_, _, _, _, ManagerId).

dept_has_employee(DeptName) :-
    employees(_, _, DeptName, _, _).

% IN subquery
rich_dept_employees(Name, Dept) :-
    employees(_, Name, Dept, _, _),
    sql_in_subquery(Dept, high_budget_dept/1).

% NOT IN subquery
customers_no_orders(Name) :-
    customers(Id, Name, _, _),
    sql_not_in_subquery(Id, customer_with_orders/1).

% EXISTS subquery
active_depts(Name, Budget) :-
    departments(_, Name, Budget, _),
    sql_exists(dept_has_employee(Name)).

% NOT EXISTS subquery
non_managers(Name) :-
    employees(Id, Name, _, _, _),
    sql_not_exists(has_subordinate(Id)).

% Combined subqueries
senior_in_rich_dept(Name, Dept) :-
    employees(Id, Name, Dept, Salary, _),
    Salary > 80000,
    sql_in_subquery(Dept, high_budget_dept/1),
    sql_exists(has_subordinate(Id)).

test :-
    format('~n=== Subquery Examples ===~n~n'),

    Tests = [
        rich_dept_employees/2 - 'IN subquery (high budget depts)',
        customers_no_orders/1 - 'NOT IN subquery (no orders)',
        active_depts/2 - 'EXISTS (depts with employees)',
        non_managers/1 - 'NOT EXISTS (no subordinates)',
        senior_in_rich_dept/2 - 'Combined: IN + EXISTS + WHERE'
    ],

    forall(
        member(Pred - Desc, Tests),
        (
            format('~w:~n', [Desc]),
            compile_predicate_to_sql(Pred, [], SQL),
            format('  ~w~n~n', [SQL])
        )
    ).

:- initialization(test, main).
```

## Exercises

1. **Basic IN**: Write a query to find all orders for products in the 'Electronics' category.

2. **NOT IN**: Find all departments that have no employees.

3. **EXISTS**: Write a query to find customers who have placed at least one order over $1000.

4. **NOT EXISTS**: Find products that have never been ordered.

5. **Combined**: Write a query to find employees who:
   - Earn above average salary for their department
   - Work in a department with budget over $500,000
   - Have no subordinates

## Performance Tips

1. **EXISTS vs IN**: Use EXISTS for large subquery results (it stops at first match)
2. **NOT EXISTS vs NOT IN**: NOT EXISTS handles NULLs better and is often faster
3. **Correlated subqueries**: Can be slow; consider rewriting as JOINs
4. **Index columns**: Ensure columns used in subquery conditions are indexed

## Summary

In this chapter, you learned:

- IN subqueries with `sql_in_subquery/2`
- NOT IN subqueries with `sql_not_in_subquery/2`
- EXISTS subqueries with `sql_exists/1`
- NOT EXISTS subqueries with `sql_not_exists/1`
- Difference between correlated and non-correlated subqueries
- When to use subqueries vs JOINs

## Next Chapter

In Chapter 6, we'll explore window functions:
- RANK, ROW_NUMBER, DENSE_RANK
- LAG and LEAD for row comparisons
- Window frame specifications

---

## Navigation

**←** [Previous: Chapter 4: Aggregations](04_aggregations) | [📖 Book 10: SQL Target](./) | [Next: Chapter 6: Window Functions →](06_window_functions)
