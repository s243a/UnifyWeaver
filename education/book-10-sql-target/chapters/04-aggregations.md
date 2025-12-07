<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 4: Aggregations

Aggregations summarize data across multiple rows. This chapter covers aggregate functions, GROUP BY, and HAVING clauses.

## Setup

```prolog
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [
    id-integer,
    name-text,
    dept-text,
    salary-integer,
    hire_date-text
]).

:- sql_table(orders, [
    id-integer,
    customer_id-integer,
    product-text,
    quantity-integer,
    amount-real,
    order_date-text
]).

:- sql_table(products, [
    id-integer,
    name-text,
    category-text,
    price-real,
    stock-integer
]).
```

## Aggregate Functions

The SQL target supports these aggregate functions:

| Function | Prolog Syntax | SQL Output |
|----------|---------------|------------|
| Count | `sql_count(*)` or `sql_count(Col)` | `COUNT(*)` or `COUNT(col)` |
| Sum | `sql_sum(Col)` | `SUM(col)` |
| Average | `sql_avg(Col)` | `AVG(col)` |
| Minimum | `sql_min(Col)` | `MIN(col)` |
| Maximum | `sql_max(Col)` | `MAX(col)` |

### COUNT

Count all rows:

```prolog
total_employees(Count) :-
    employees(_, _, _, _, _),
    Count = sql_count(*).

?- compile_predicate_to_sql(total_employees/1, [], SQL).
% SELECT COUNT(*) FROM employees;
```

Count non-null values in a column:

```prolog
employees_with_dept(Count) :-
    employees(_, _, Dept, _, _),
    Count = sql_count(Dept).

% SELECT COUNT(dept) FROM employees;
```

### SUM

```prolog
total_salary(Total) :-
    employees(_, _, _, Salary, _),
    Total = sql_sum(Salary).

?- compile_predicate_to_sql(total_salary/1, [], SQL).
% SELECT SUM(salary) FROM employees;
```

### AVG

```prolog
average_salary(Avg) :-
    employees(_, _, _, Salary, _),
    Avg = sql_avg(Salary).

?- compile_predicate_to_sql(average_salary/1, [], SQL).
% SELECT AVG(salary) FROM employees;
```

### MIN and MAX

```prolog
salary_range(MinSal, MaxSal) :-
    employees(_, _, _, Salary, _),
    MinSal = sql_min(Salary),
    MaxSal = sql_max(Salary).

?- compile_predicate_to_sql(salary_range/2, [], SQL).
% SELECT MIN(salary), MAX(salary) FROM employees;
```

## GROUP BY

Group rows by one or more columns using `sql_group_by/1`:

```prolog
sql_group_by([Column1, Column2, ...])
```

### Single Column Grouping

```prolog
employees_per_dept(Dept, Count) :-
    employees(_, _, Dept, _, _),
    sql_group_by([Dept]),
    Count = sql_count(*).

?- compile_predicate_to_sql(employees_per_dept/2, [], SQL).
% SELECT dept, COUNT(*) FROM employees GROUP BY dept;
```

### Multiple Aggregates per Group

```prolog
dept_stats(Dept, Count, AvgSalary, MaxSalary) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    Count = sql_count(*),
    AvgSalary = sql_avg(Salary),
    MaxSalary = sql_max(Salary).

?- compile_predicate_to_sql(dept_stats/4, [], SQL).
% SELECT dept, COUNT(*), AVG(salary), MAX(salary) FROM employees GROUP BY dept;
```

### Multiple Column Grouping

```prolog
orders_by_customer_product(CustomerId, Product, TotalQty, TotalAmount) :-
    orders(_, CustomerId, Product, Quantity, Amount, _),
    sql_group_by([CustomerId, Product]),
    TotalQty = sql_sum(Quantity),
    TotalAmount = sql_sum(Amount).

?- compile_predicate_to_sql(orders_by_customer_product/4, [], SQL).
% SELECT customer_id, product, SUM(quantity), SUM(amount)
% FROM orders GROUP BY customer_id, product;
```

## HAVING

Filter groups (not individual rows) using `sql_having/1`:

```prolog
sql_having(Condition)
```

### Basic HAVING

Find departments with more than 5 employees:

```prolog
large_depts(Dept, Count) :-
    employees(_, _, Dept, _, _),
    sql_group_by([Dept]),
    Count = sql_count(*),
    sql_having(Count > 5).

?- compile_predicate_to_sql(large_depts/2, [], SQL).
% SELECT dept, COUNT(*) FROM employees GROUP BY dept HAVING COUNT(*) > 5;
```

### HAVING with Aggregate Functions

```prolog
high_avg_salary_depts(Dept, AvgSalary) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    AvgSalary = sql_avg(Salary),
    sql_having(AvgSalary > 75000).

% SELECT dept, AVG(salary) FROM employees GROUP BY dept HAVING AVG(salary) > 75000;
```

### Multiple HAVING Conditions

```prolog
qualified_depts(Dept, Count, AvgSalary) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    Count = sql_count(*),
    AvgSalary = sql_avg(Salary),
    sql_having(Count >= 3),
    sql_having(AvgSalary > 50000).

% SELECT dept, COUNT(*), AVG(salary) FROM employees
% GROUP BY dept HAVING COUNT(*) >= 3 AND AVG(salary) > 50000;
```

## WHERE vs HAVING

- **WHERE**: Filters rows BEFORE grouping
- **HAVING**: Filters groups AFTER aggregation

```prolog
% Filter rows first, then group
engineering_stats(Dept, Count) :-
    employees(_, _, Dept, Salary, _),
    Salary > 50000,              % WHERE: filter rows first
    sql_group_by([Dept]),
    Count = sql_count(*),
    sql_having(Count > 2).       % HAVING: filter groups after

% SELECT dept, COUNT(*) FROM employees
% WHERE salary > 50000
% GROUP BY dept
% HAVING COUNT(*) > 2;
```

## Aggregations with JOINs

Combine aggregations with table joins:

```prolog
customer_order_totals(CustomerId, OrderCount, TotalSpent) :-
    customers(CustomerId, _, _, _),
    orders(_, CustomerId, _, _, Amount, _),
    sql_group_by([CustomerId]),
    OrderCount = sql_count(*),
    TotalSpent = sql_sum(Amount).

% SELECT customers.id, COUNT(*), SUM(orders.amount)
% FROM customers
% INNER JOIN orders ON orders.customer_id = customers.id
% GROUP BY customers.id;
```

## Aggregations with ORDER BY

Sort aggregated results:

```prolog
depts_by_avg_salary(Dept, AvgSalary) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    AvgSalary = sql_avg(Salary),
    sql_order_by(AvgSalary, desc).

% SELECT dept, AVG(salary) FROM employees
% GROUP BY dept ORDER BY AVG(salary) DESC;
```

### Top N Groups

```prolog
top_3_depts(Dept, TotalSalary) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    TotalSalary = sql_sum(Salary),
    sql_order_by(TotalSalary, desc),
    sql_limit(3).

% SELECT dept, SUM(salary) FROM employees
% GROUP BY dept ORDER BY SUM(salary) DESC LIMIT 3;
```

## Column Aliases with Aggregates

Use `sql_as/2` to name aggregate columns:

```prolog
dept_summary(Dept, EmpCount, AvgPay) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    EmpCount = sql_as(sql_count(*), employee_count),
    AvgPay = sql_as(sql_avg(Salary), average_salary).

% SELECT dept, COUNT(*) AS employee_count, AVG(salary) AS average_salary
% FROM employees GROUP BY dept;
```

## DISTINCT with Aggregates

Count distinct values:

```prolog
unique_products_ordered(Count) :-
    orders(_, _, Product, _, _, _),
    Count = sql_count_distinct(Product).

% SELECT COUNT(DISTINCT product) FROM orders;
```

## Complete Example

```prolog
% file: aggregations_example.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, hire_date-text]).
:- sql_table(orders, [id-integer, customer_id-integer, product-text, quantity-integer, amount-real, order_date-text]).

% Simple aggregates
total_employees(Count) :-
    employees(_, _, _, _, _),
    Count = sql_count(*).

salary_stats(MinSal, MaxSal, AvgSal, TotalSal) :-
    employees(_, _, _, Salary, _),
    MinSal = sql_min(Salary),
    MaxSal = sql_max(Salary),
    AvgSal = sql_avg(Salary),
    TotalSal = sql_sum(Salary).

% GROUP BY
employees_per_dept(Dept, Count) :-
    employees(_, _, Dept, _, _),
    sql_group_by([Dept]),
    Count = sql_count(*).

dept_salary_stats(Dept, Count, AvgSalary, MaxSalary) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    Count = sql_count(*),
    AvgSalary = sql_avg(Salary),
    MaxSalary = sql_max(Salary).

% HAVING
large_depts(Dept, Count) :-
    employees(_, _, Dept, _, _),
    sql_group_by([Dept]),
    Count = sql_count(*),
    sql_having(Count > 5).

high_paying_depts(Dept, AvgSalary) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    AvgSalary = sql_avg(Salary),
    sql_having(AvgSalary > 70000).

% WHERE + HAVING
senior_large_depts(Dept, Count) :-
    employees(_, _, Dept, Salary, _),
    Salary > 60000,               % WHERE
    sql_group_by([Dept]),
    Count = sql_count(*),
    sql_having(Count >= 3).       % HAVING

% With ORDER BY and LIMIT
top_depts_by_payroll(Dept, TotalPay) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    TotalPay = sql_sum(Salary),
    sql_order_by(TotalPay, desc),
    sql_limit(5).

% With aliases
summary_report(Dept, EmpCount, AvgPay, TotalPay) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    EmpCount = sql_as(sql_count(*), num_employees),
    AvgPay = sql_as(sql_avg(Salary), avg_salary),
    TotalPay = sql_as(sql_sum(Salary), total_payroll).

test :-
    format('~n=== Aggregation Examples ===~n~n'),

    Tests = [
        total_employees/1 - 'Total count',
        salary_stats/4 - 'Multiple aggregates',
        employees_per_dept/2 - 'GROUP BY single column',
        dept_salary_stats/4 - 'GROUP BY with multiple aggregates',
        large_depts/2 - 'HAVING (count > 5)',
        high_paying_depts/2 - 'HAVING (avg > 70000)',
        senior_large_depts/2 - 'WHERE + HAVING combined',
        top_depts_by_payroll/2 - 'GROUP BY + ORDER BY + LIMIT',
        summary_report/4 - 'With column aliases'
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

1. **Basic Aggregates**: Write predicates to find:
   - Total number of orders
   - Average order amount
   - Highest and lowest order amounts

2. **GROUP BY**: Create queries to show:
   - Number of orders per customer
   - Total quantity and amount per product

3. **HAVING**: Write queries for:
   - Products with total sales over $10,000
   - Customers with more than 5 orders

4. **Combined**: Create a query showing the top 3 products by total revenue, with columns: product name, order count, total revenue (aliased appropriately).

5. **Complex**: Write a query that:
   - Joins employees and departments
   - Groups by department name
   - Shows department name, employee count, and average salary
   - Only includes departments with 3+ employees
   - Orders by average salary descending

## Summary

In this chapter, you learned:

- Aggregate functions: `sql_count`, `sql_sum`, `sql_avg`, `sql_min`, `sql_max`
- Grouping with `sql_group_by/1`
- Filtering groups with `sql_having/1`
- Difference between WHERE (row filter) and HAVING (group filter)
- Combining aggregations with JOINs, ORDER BY, LIMIT
- Using `sql_as/2` to alias aggregate columns

## Next Chapter

In Chapter 5, we'll explore subqueries:
- IN and NOT IN subqueries
- EXISTS and NOT EXISTS
- Correlated subqueries
