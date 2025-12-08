<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 7: Common Table Expressions (CTEs)

CTEs (WITH clauses) let you define temporary named result sets that can be referenced in the main query. They improve readability and enable complex query patterns.

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
```

## Basic CTE Syntax

```prolog
compile_with_cte(CTEName, CTEPredicate/Arity, MainPredicate/Arity, SQL)
```

Parameters:
- `CTEName`: Name for the temporary result set
- `CTEPredicate/Arity`: Predicate that defines the CTE's content
- `MainPredicate/Arity`: Main query that references the CTE
- `SQL`: Output variable for generated SQL

## Your First CTE

Let's create a CTE that identifies high earners, then query it:

```prolog
% Step 1: Define the CTE content
high_earner_data(Id, Name, Salary) :-
    employees(Id, Name, _, Salary, _),
    Salary > 100000.

% Step 2: Define the main query (references CTE by name)
high_earner_result(Name, Salary) :-
    high_earners(_, Name, Salary).  % References CTE name

% Step 3: Compile
?- compile_with_cte(high_earners, high_earner_data/3, high_earner_result/2, SQL).
```

Generated SQL:
```sql
WITH high_earners AS (
    SELECT id, name, salary
    FROM employees
    WHERE salary > 100000
)
SELECT name, salary
FROM high_earners;
```

## Why Use CTEs?

### 1. Improved Readability

Break complex queries into logical steps:

```prolog
% Without CTE: nested subquery
complex_query(Name) :-
    employees(_, Name, Dept, _, _),
    sql_in_subquery(Dept, (
        departments(_, Dept, Budget, _),
        Budget > 1000000
    )).

% With CTE: cleaner separation
rich_depts_cte(DeptName) :-
    departments(_, DeptName, Budget, _),
    Budget > 1000000.

employees_in_rich_result(Name) :-
    employees(_, Name, Dept, _, _),
    rich_depts(Dept).  % Clear reference
```

### 2. Reuse Within Query

Reference the same CTE multiple times:

```prolog
% CTE for department averages
dept_avg_cte(Dept, AvgSalary) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    AvgSalary = sql_avg(Salary).

% Main query uses CTE twice
above_dept_avg(Name, Salary, DeptAvg) :-
    employees(_, Name, Dept, Salary, _),
    dept_averages(Dept, DeptAvg),      % First reference
    Salary > DeptAvg.
```

### 3. Filter Window Function Results

CTEs enable filtering on window function results:

```prolog
% CTE: Add rank to all employees
ranked_employees_cte(Name, Dept, Salary, Rank) :-
    employees(_, Name, Dept, Salary, _),
    Rank = sql_window(rank, [], [Dept], [(Salary, desc)]).

% Main query: Filter to top 3 per department
top_3_per_dept(Name, Dept, Salary, Rank) :-
    ranked_employees(Name, Dept, Salary, Rank),
    Rank =< 3.

?- compile_with_cte(ranked_employees, ranked_employees_cte/4, top_3_per_dept/4, SQL).
```

Generated SQL:
```sql
WITH ranked_employees AS (
    SELECT name, dept, salary,
           RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rank
    FROM employees
)
SELECT name, dept, salary, rank
FROM ranked_employees
WHERE rank <= 3;
```

## CTEs with Aggregations

```prolog
% CTE: Calculate order statistics per customer
customer_stats_cte(CustomerId, OrderCount, TotalSpent) :-
    orders(_, CustomerId, _, Amount, _),
    sql_group_by([CustomerId]),
    OrderCount = sql_count(*),
    TotalSpent = sql_sum(Amount).

% Main query: Find high-value customers
vip_customers(CustName, Email, TotalSpent) :-
    customer_stats(CustId, _, TotalSpent),
    TotalSpent > 10000,
    customers(CustId, CustName, Email, _).

?- compile_with_cte(customer_stats, customer_stats_cte/3, vip_customers/3, SQL).
```

Generated SQL:
```sql
WITH customer_stats AS (
    SELECT customer_id, COUNT(*) AS order_count, SUM(amount) AS total_spent
    FROM orders
    GROUP BY customer_id
)
SELECT customers.name, customers.email, customer_stats.total_spent
FROM customer_stats
INNER JOIN customers ON customers.id = customer_stats.customer_id
WHERE customer_stats.total_spent > 10000;
```

## CTEs with JOINs

The main query can JOIN with the CTE:

```prolog
% CTE: Manager info
manager_info_cte(Id, Name) :-
    employees(Id, Name, _, _, _),
    employees(_, _, _, _, Id).  % Has at least one subordinate

% Main query: Employees with their manager names
emp_with_manager(EmpName, ManagerName) :-
    employees(_, EmpName, _, _, MgrId),
    managers(MgrId, ManagerName).

?- compile_with_cte(managers, manager_info_cte/2, emp_with_manager/2, SQL).
```

## CTEs for Data Transformation

Use CTEs to transform data before the main query:

```prolog
% CTE: Categorize salaries
salary_categories_cte(Id, Name, Category) :-
    employees(Id, Name, _, Salary, _),
    Category = sql_case([
        when(Salary > 100000, 'Senior'),
        when(Salary > 60000, 'Mid')
    ], 'Junior').

% Main query: Count by category
category_counts(Category, Count) :-
    salary_categories(_, _, Category),
    sql_group_by([Category]),
    Count = sql_count(*).

?- compile_with_cte(salary_categories, salary_categories_cte/3, category_counts/2, SQL).
```

## CTE Best Practices

### When to Use CTEs

| Use CTE When | Don't Use When |
|--------------|----------------|
| Query has multiple logical steps | Simple single-table query |
| Need to filter window results | Performance is critical (test both) |
| Same subquery used multiple times | CTE would only be referenced once |
| Improve readability | Query is already simple |

### Performance Considerations

1. **Materialization**: Some databases materialize CTEs (compute once, store), others inline them
2. **Optimization**: CTEs can prevent some query optimizer optimizations
3. **Testing**: Always test CTE vs subquery performance for critical queries

## CTE vs Subquery Comparison

### Same Query, Different Approaches

**Subquery approach:**
```prolog
high_earner_depts(Dept) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    AvgSal = sql_avg(Salary),
    sql_having(AvgSal > 80000).

employees_in_high_depts(Name) :-
    employees(_, Name, Dept, _, _),
    sql_in_subquery(Dept, high_earner_depts/1).
```

**CTE approach:**
```prolog
high_depts_cte(Dept) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    AvgSal = sql_avg(Salary),
    sql_having(AvgSal > 80000).

employees_result(Name) :-
    employees(_, Name, Dept, _, _),
    high_depts(Dept).

?- compile_with_cte(high_depts, high_depts_cte/1, employees_result/1, SQL).
```

Both produce similar SQL, but the CTE version is more readable for complex queries.

## Complete Example

```prolog
% file: cte_example.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, manager_id-integer]).
:- sql_table(orders, [id-integer, customer_id-integer, product-text, amount-real, order_date-text]).
:- sql_table(customers, [id-integer, name-text, email-text, country-text]).

% Example 1: Simple CTE
high_earner_cte(Id, Name, Salary) :-
    employees(Id, Name, _, Salary, _),
    Salary > 100000.

high_earner_main(Name, Salary) :-
    high_earners(_, Name, Salary).

% Example 2: CTE with window function filtering
ranked_cte(Name, Dept, Salary, Rank) :-
    employees(_, Name, Dept, Salary, _),
    Rank = sql_window(rank, [], [Dept], [(Salary, desc)]).

top_3_main(Name, Dept, Salary) :-
    ranked_employees(Name, Dept, Salary, Rank),
    Rank =< 3.

% Example 3: CTE with aggregation
customer_totals_cte(CustId, TotalSpent) :-
    orders(_, CustId, _, Amount, _),
    sql_group_by([CustId]),
    TotalSpent = sql_sum(Amount).

vip_main(CustName, TotalSpent) :-
    customer_totals(CustId, TotalSpent),
    TotalSpent > 5000,
    customers(CustId, CustName, _, _).

% Example 4: CTE for department stats
dept_stats_cte(Dept, EmpCount, AvgSalary) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    EmpCount = sql_count(*),
    AvgSalary = sql_avg(Salary).

large_depts_main(Dept, EmpCount, AvgSalary) :-
    dept_stats(Dept, EmpCount, AvgSalary),
    EmpCount >= 5.

test :-
    format('~n=== CTE Examples ===~n~n'),

    format('1. Simple CTE:~n'),
    compile_with_cte(high_earners, high_earner_cte/3, high_earner_main/2, SQL1),
    format('   ~w~n~n', [SQL1]),

    format('2. CTE with window function (top 3 per dept):~n'),
    compile_with_cte(ranked_employees, ranked_cte/4, top_3_main/3, SQL2),
    format('   ~w~n~n', [SQL2]),

    format('3. CTE with aggregation (VIP customers):~n'),
    compile_with_cte(customer_totals, customer_totals_cte/2, vip_main/2, SQL3),
    format('   ~w~n~n', [SQL3]),

    format('4. CTE for filtering aggregated results:~n'),
    compile_with_cte(dept_stats, dept_stats_cte/3, large_depts_main/3, SQL4),
    format('   ~w~n~n', [SQL4]).

:- initialization(test, main).
```

## Exercises

1. **Basic CTE**: Create a CTE that calculates total orders per product, then find products with more than 100 orders.

2. **Window + CTE**: Use a CTE with ROW_NUMBER to find the most recent order for each customer.

3. **Aggregation CTE**: Create a CTE that calculates department averages, then find employees earning above their department average.

4. **Multiple Steps**: Design a query that:
   - CTE 1: Calculate customer order totals
   - Main query: Join with customers, filter to country = 'USA', order by total descending

5. **Complex**: Create a CTE that ranks salespeople by monthly sales, then find the top performer for each month.

## Summary

In this chapter, you learned:

- Creating CTEs with `compile_with_cte/4`
- Using CTEs for improved readability
- Filtering window function results with CTEs
- CTEs with aggregations and JOINs
- When to use CTEs vs subqueries
- Performance considerations

## Next Chapter

In Chapter 8, we'll explore Recursive CTEs:
- WITH RECURSIVE for hierarchical data
- Org charts and tree structures
- Graph traversal queries

---

## Navigation

**←** [Previous: Chapter 6: Window Functions](06_window_functions) | [📖 Book 10: SQL Target](./) | [Next: Chapter 8: Recursive CTEs →](08_recursive_ctes)
