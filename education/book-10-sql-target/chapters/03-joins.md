<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 3: JOINs

JOINs combine data from multiple tables. This chapter covers all JOIN types supported by the SQL target.

## Setup

```prolog
:- use_module('src/unifyweaver/targets/sql_target').

% Customers
:- sql_table(customers, [
    id-integer,
    name-text,
    email-text,
    country-text
]).

% Orders
:- sql_table(orders, [
    id-integer,
    customer_id-integer,
    product-text,
    amount-real,
    order_date-text
]).

% Shipments
:- sql_table(shipments, [
    id-integer,
    order_id-integer,
    tracking-text,
    ship_date-text
]).

% Employees (for self-join examples)
:- sql_table(employees, [
    id-integer,
    name-text,
    manager_id-integer,
    dept-text
]).

% Departments
:- sql_table(departments, [
    id-integer,
    name-text,
    budget-real
]).
```

## INNER JOIN (Implicit)

The most common JOIN type. In UnifyWeaver, INNER JOINs happen implicitly when you reference multiple tables with shared variables.

### Basic INNER JOIN

```prolog
customer_orders(CustomerName, Product, Amount) :-
    customers(CustId, CustomerName, _, _),
    orders(_, CustId, Product, Amount, _).

?- compile_predicate_to_sql(customer_orders/3, [], SQL).
```

Generated SQL:
```sql
SELECT customers.name, orders.product, orders.amount
FROM customers
INNER JOIN orders ON orders.customer_id = customers.id;
```

The shared variable `CustId` creates the JOIN condition.

### Multiple Join Conditions

```prolog
% Join on multiple columns (less common but supported)
matched_records(A, B) :-
    table1(Id, Code, A),
    table2(Id, Code, B).

% Generates: ... ON table2.id = table1.id AND table2.code = table1.code
```

### Three-Table JOIN

```prolog
full_order_info(CustomerName, Product, Tracking) :-
    customers(CustId, CustomerName, _, _),
    orders(OrderId, CustId, Product, _, _),
    shipments(_, OrderId, Tracking, _).

?- compile_predicate_to_sql(full_order_info/3, [], SQL).
```

Generated SQL:
```sql
SELECT customers.name, orders.product, shipments.tracking
FROM customers
INNER JOIN orders ON orders.customer_id = customers.id
INNER JOIN shipments ON shipments.order_id = orders.id;
```

## LEFT JOIN

LEFT JOIN returns all rows from the left table and matching rows from the right table. Non-matching rows have NULL values.

### Basic LEFT JOIN

```prolog
customers_with_orders(CustomerName, Product) :-
    customers(CustId, CustomerName, _, _),
    sql_left_join(orders(_, CustId, Product, _, _)).

?- compile_predicate_to_sql(customers_with_orders/2, [], SQL).
```

Generated SQL:
```sql
SELECT customers.name, orders.product
FROM customers
LEFT JOIN orders ON orders.customer_id = customers.id;
```

This returns ALL customers, even those without orders (Product will be NULL).

### Chained LEFT JOINs

```prolog
customer_shipments(CustomerName, Product, Tracking) :-
    customers(CustId, CustomerName, _, _),
    sql_left_join(orders(OrderId, CustId, Product, _, _)),
    sql_left_join(shipments(_, OrderId, Tracking, _)).

?- compile_predicate_to_sql(customer_shipments/3, [], SQL).
```

Generated SQL:
```sql
SELECT customers.name, orders.product, shipments.tracking
FROM customers
LEFT JOIN orders ON orders.customer_id = customers.id
LEFT JOIN shipments ON shipments.order_id = orders.id;
```

### Finding Non-Matching Rows

Combine LEFT JOIN with IS NULL to find rows without matches:

```prolog
customers_without_orders(CustomerName) :-
    customers(CustId, CustomerName, _, _),
    sql_left_join(orders(OrderId, CustId, _, _, _)),
    sql_is_null(OrderId).

?- compile_predicate_to_sql(customers_without_orders/1, [], SQL).
```

Generated SQL:
```sql
SELECT customers.name
FROM customers
LEFT JOIN orders ON orders.customer_id = customers.id
WHERE orders.id IS NULL;
```

## RIGHT JOIN

RIGHT JOIN returns all rows from the right table and matching rows from the left table.

```prolog
orders_with_customers(OrderId, CustomerName) :-
    orders(OrderId, CustId, _, _, _),
    sql_right_join(customers(CustId, CustomerName, _, _)).

?- compile_predicate_to_sql(orders_with_customers/2, [], SQL).
```

Generated SQL:
```sql
SELECT orders.id, customers.name
FROM orders
RIGHT JOIN customers ON customers.id = orders.customer_id;
```

This returns ALL customers, even those without orders.

## FULL OUTER JOIN

FULL OUTER JOIN returns all rows from both tables, with NULLs where there's no match.

```prolog
all_customer_order_combos(CustomerName, Product) :-
    customers(CustId, CustomerName, _, _),
    sql_full_outer_join(orders(_, CustId, Product, _, _)).

?- compile_predicate_to_sql(all_customer_order_combos/2, [], SQL).
```

Generated SQL:
```sql
SELECT customers.name, orders.product
FROM customers
FULL OUTER JOIN orders ON orders.customer_id = customers.id;
```

Note: SQLite doesn't support FULL OUTER JOIN directly. For SQLite, you'd need to use UNION of LEFT and RIGHT JOINs.

## Self-Joins

Join a table to itself. Useful for hierarchical data.

### Employee-Manager Relationship

```prolog
employee_with_manager(EmpName, ManagerName) :-
    employees(_, EmpName, ManagerId, _),
    employees(ManagerId, ManagerName, _, _).

?- compile_predicate_to_sql(employee_with_manager/2, [], SQL).
```

Generated SQL:
```sql
SELECT e1.name, e2.name
FROM employees e1
INNER JOIN employees e2 ON e2.id = e1.manager_id;
```

### With LEFT JOIN (Include Employees Without Managers)

```prolog
all_employees_with_manager(EmpName, ManagerName) :-
    employees(_, EmpName, ManagerId, _),
    sql_left_join(employees(ManagerId, ManagerName, _, _)).
```

## Mixed JOIN Types

Combine different JOIN types in one query:

```prolog
% INNER JOIN customers to orders, LEFT JOIN to shipments
orders_maybe_shipped(CustomerName, Product, Tracking) :-
    customers(CustId, CustomerName, _, _),
    orders(OrderId, CustId, Product, _, _),           % INNER JOIN
    sql_left_join(shipments(_, OrderId, Tracking, _)). % LEFT JOIN
```

Generated SQL:
```sql
SELECT customers.name, orders.product, shipments.tracking
FROM customers
INNER JOIN orders ON orders.customer_id = customers.id
LEFT JOIN shipments ON shipments.order_id = orders.id;
```

## JOIN with Filters

Add WHERE conditions to JOINed queries:

```prolog
high_value_orders(CustomerName, Product, Amount) :-
    customers(CustId, CustomerName, _, Country),
    orders(_, CustId, Product, Amount, _),
    Country = 'USA',
    Amount > 1000.

?- compile_predicate_to_sql(high_value_orders/3, [], SQL).
```

Generated SQL:
```sql
SELECT customers.name, orders.product, orders.amount
FROM customers
INNER JOIN orders ON orders.customer_id = customers.id
WHERE customers.country = 'USA' AND orders.amount > 1000;
```

## JOIN with ORDER BY and LIMIT

```prolog
top_orders_by_customer(CustomerName, Amount) :-
    customers(CustId, CustomerName, _, _),
    orders(_, CustId, _, Amount, _),
    sql_order_by(Amount, desc),
    sql_limit(10).
```

Generated SQL:
```sql
SELECT customers.name, orders.amount
FROM customers
INNER JOIN orders ON orders.customer_id = customers.id
ORDER BY orders.amount DESC
LIMIT 10;
```

## Understanding JOIN Logic

### When to Use Each JOIN Type

| JOIN Type | Use When |
|-----------|----------|
| INNER | You only want rows that match in both tables |
| LEFT | You want all rows from left table, even without matches |
| RIGHT | You want all rows from right table, even without matches |
| FULL OUTER | You want all rows from both tables |

### Visual Representation

```
INNER JOIN:        LEFT JOIN:         RIGHT JOIN:        FULL OUTER:
   ┌───┐              ┌───┐              ┌───┐              ┌───┐
   │███│              │███│              │   │              │███│
┌──┼───┼──┐        ┌──┼───┼──┐        ┌──┼───┼──┐        ┌──┼───┼──┐
│  │███│  │        │██│███│  │        │  │███│██│        │██│███│██│
└──┼───┼──┘        └──┼───┼──┘        └──┼───┼──┘        └──┼───┼──┘
   │███│              │   │              │███│              │███│
   └───┘              └───┘              └───┘              └───┘
```

## Complete Example

```prolog
% file: joins_example.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(customers, [id-integer, name-text, email-text, country-text]).
:- sql_table(orders, [id-integer, customer_id-integer, product-text, amount-real, order_date-text]).
:- sql_table(shipments, [id-integer, order_id-integer, tracking-text, ship_date-text]).
:- sql_table(employees, [id-integer, name-text, manager_id-integer, dept-text]).

% INNER JOIN
customer_orders(CustName, Product) :-
    customers(CustId, CustName, _, _),
    orders(_, CustId, Product, _, _).

% LEFT JOIN
customers_maybe_ordered(CustName, Product) :-
    customers(CustId, CustName, _, _),
    sql_left_join(orders(_, CustId, Product, _, _)).

% Chained LEFT JOINs
full_chain(CustName, Product, Tracking) :-
    customers(CustId, CustName, _, _),
    sql_left_join(orders(OrderId, CustId, Product, _, _)),
    sql_left_join(shipments(_, OrderId, Tracking, _)).

% Find customers without orders
no_orders(CustName) :-
    customers(CustId, CustName, _, _),
    sql_left_join(orders(OrderId, CustId, _, _, _)),
    sql_is_null(OrderId).

% Self-join
emp_manager(EmpName, MgrName) :-
    employees(_, EmpName, MgrId, _),
    sql_left_join(employees(MgrId, MgrName, _, _)).

% Mixed with filters
usa_high_value(CustName, Amount) :-
    customers(CustId, CustName, _, Country),
    orders(_, CustId, _, Amount, _),
    Country = 'USA',
    Amount > 500,
    sql_order_by(Amount, desc).

test :-
    format('~n=== JOIN Examples ===~n~n'),

    Tests = [
        customer_orders/2 - 'INNER JOIN (implicit)',
        customers_maybe_ordered/2 - 'LEFT JOIN',
        full_chain/3 - 'Chained LEFT JOINs',
        no_orders/1 - 'Find non-matching (LEFT JOIN + IS NULL)',
        emp_manager/2 - 'Self-join (employee-manager)',
        usa_high_value/2 - 'JOIN with filters'
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

1. **Basic JOIN**: Write a predicate that joins `departments` and `employees` to show department name and employee name.

2. **LEFT JOIN**: Modify exercise 1 to show ALL departments, even those without employees.

3. **Three Tables**: Create a query joining customers, orders, and shipments to show customer name, order date, and ship date.

4. **Self-Join**: Write a query to find employees who are in the same department as their manager.

5. **Complex**: Find all customers from 'Canada' who have orders over $100, showing customer name, product, and amount, sorted by amount descending.

## Summary

In this chapter, you learned:

- INNER JOINs via shared variables (implicit)
- LEFT JOIN with `sql_left_join/1`
- RIGHT JOIN with `sql_right_join/1`
- FULL OUTER JOIN with `sql_full_outer_join/1`
- Chaining multiple JOINs
- Self-joins for hierarchical data
- Finding non-matching rows with LEFT JOIN + IS NULL
- Combining JOINs with filters, ORDER BY, and LIMIT

## Next Chapter

In Chapter 4, we'll explore aggregations:
- COUNT, SUM, AVG, MIN, MAX
- GROUP BY clauses
- HAVING for filtering aggregated results
