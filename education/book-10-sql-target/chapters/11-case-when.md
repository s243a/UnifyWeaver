<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 11: CASE WHEN Expressions

CASE WHEN provides conditional logic in SQL queries, similar to if-else statements in programming. Use it to transform values, categorize data, or create computed columns.

## Setup

```prolog
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [
    id-integer,
    name-text,
    dept-text,
    salary-integer,
    hire_date-text,
    performance_score-integer
]).

:- sql_table(products, [
    id-integer,
    name-text,
    price-real,
    stock-integer,
    category-text
]).

:- sql_table(orders, [
    id-integer,
    customer_id-integer,
    amount-real,
    status-text,
    order_date-text
]).
```

## CASE WHEN Syntax

```prolog
sql_case([when(Condition1, Result1), when(Condition2, Result2), ...], Default)
```

- Each `when(Condition, Result)` pair is evaluated in order
- First matching condition's result is returned
- `Default` is returned if no conditions match

## Basic CASE WHEN

### Simple Categorization

```prolog
salary_category(Name, Category) :-
    employees(_, Name, _, Salary, _, _),
    Category = sql_case([
        when(Salary > 100000, 'High'),
        when(Salary > 60000, 'Medium')
    ], 'Low').

?- compile_predicate_to_sql(salary_category/2, [], SQL).
```

Generated SQL:
```sql
SELECT name,
       CASE
           WHEN salary > 100000 THEN 'High'
           WHEN salary > 60000 THEN 'Medium'
           ELSE 'Low'
       END
FROM employees;
```

### Multiple Conditions

```prolog
performance_rating(Name, Rating) :-
    employees(_, Name, _, Salary, _, Score),
    Rating = sql_case([
        when((Score >= 90, Salary > 80000), 'Star Performer'),
        when(Score >= 90, 'High Performer'),
        when(Score >= 70, 'Good'),
        when(Score >= 50, 'Needs Improvement')
    ], 'Below Expectations').
```

## CASE WHEN with Column Values

### Mapping Values

```prolog
order_status_display(Id, StatusDisplay) :-
    orders(Id, _, _, Status, _),
    StatusDisplay = sql_case([
        when(Status = 'P', 'Pending'),
        when(Status = 'S', 'Shipped'),
        when(Status = 'D', 'Delivered'),
        when(Status = 'C', 'Cancelled')
    ], 'Unknown').

% SELECT id, CASE WHEN status = 'P' THEN 'Pending'
%                 WHEN status = 'S' THEN 'Shipped'
%                 WHEN status = 'D' THEN 'Delivered'
%                 WHEN status = 'C' THEN 'Cancelled'
%                 ELSE 'Unknown' END FROM orders;
```

### Boolean Conditions

```prolog
stock_status(Name, Status) :-
    products(_, Name, _, Stock, _),
    Status = sql_case([
        when(Stock = 0, 'Out of Stock'),
        when(Stock < 10, 'Low Stock'),
        when(Stock < 50, 'In Stock')
    ], 'Well Stocked').
```

## CASE WHEN with Numeric Results

### Computing Values

```prolog
discount_rate(Name, Price, Discount) :-
    products(_, Name, Price, _, Category),
    Discount = sql_case([
        when(Category = 'Electronics', Price * 0.15),
        when(Category = 'Clothing', Price * 0.20),
        when(Category = 'Books', Price * 0.10)
    ], Price * 0.05).

% SELECT name, price, CASE WHEN category = 'Electronics' THEN price * 0.15
%                          WHEN category = 'Clothing' THEN price * 0.20
%                          WHEN category = 'Books' THEN price * 0.10
%                          ELSE price * 0.05 END FROM products;
```

### Tax Calculations

```prolog
order_with_tax(Id, Amount, Tax) :-
    orders(Id, _, Amount, _, _),
    Tax = sql_case([
        when(Amount > 1000, Amount * 0.10),
        when(Amount > 500, Amount * 0.08),
        when(Amount > 100, Amount * 0.05)
    ], 0).
```

## CASE WHEN with Aliasing

Use `sql_as/2` to name CASE WHEN columns:

```prolog
employee_tier(Name, Salary, Tier) :-
    employees(_, Name, _, Salary, _, _),
    Tier = sql_as(
        sql_case([
            when(Salary > 100000, 'Executive'),
            when(Salary > 75000, 'Senior'),
            when(Salary > 50000, 'Mid-Level')
        ], 'Entry-Level'),
        salary_tier
    ).

% SELECT name, salary, CASE ... END AS salary_tier FROM employees;
```

## CASE WHEN in WHERE Clauses

Filter based on conditional logic:

```prolog
priority_orders(Id, Amount) :-
    orders(Id, _, Amount, Status, _),
    1 = sql_case([
        when((Amount > 1000, Status = 'P'), 1),
        when((Amount > 500, Status = 'P'), 1)
    ], 0).

% WHERE CASE WHEN amount > 1000 AND status = 'P' THEN 1
%            WHEN amount > 500 AND status = 'P' THEN 1
%            ELSE 0 END = 1
```

## CASE WHEN with Aggregations

### Conditional Counting

```prolog
dept_salary_breakdown(Dept, HighCount, MidCount, LowCount) :-
    employees(_, _, Dept, Salary, _, _),
    sql_group_by([Dept]),
    HighCount = sql_sum(sql_case([when(Salary > 100000, 1)], 0)),
    MidCount = sql_sum(sql_case([when((Salary > 60000, Salary =< 100000), 1)], 0)),
    LowCount = sql_sum(sql_case([when(Salary =< 60000, 1)], 0)).

% SELECT dept,
%        SUM(CASE WHEN salary > 100000 THEN 1 ELSE 0 END) AS high,
%        SUM(CASE WHEN salary > 60000 AND salary <= 100000 THEN 1 ELSE 0 END) AS mid,
%        SUM(CASE WHEN salary <= 60000 THEN 1 ELSE 0 END) AS low
% FROM employees GROUP BY dept;
```

### Conditional Sums

```prolog
category_revenue(Category, OnlineRev, StoreRev) :-
    orders(_, _, Amount, _, _),
    products(_, _, _, _, Category),
    sql_group_by([Category]),
    OnlineRev = sql_sum(sql_case([when(channel = 'online', Amount)], 0)),
    StoreRev = sql_sum(sql_case([when(channel = 'store', Amount)], 0)).
```

## Nested CASE WHEN

CASE expressions can be nested:

```prolog
complex_category(Name, Salary, Score, Category) :-
    employees(_, Name, _, Salary, _, Score),
    Category = sql_case([
        when(Salary > 100000,
             sql_case([
                 when(Score >= 90, 'Executive-Star'),
                 when(Score >= 70, 'Executive-Good')
             ], 'Executive-Needs Work')),
        when(Salary > 60000,
             sql_case([
                 when(Score >= 90, 'Senior-Star'),
                 when(Score >= 70, 'Senior-Good')
             ], 'Senior-Needs Work'))
    ], 'Standard').
```

## CASE WHEN with NULL Handling

```prolog
safe_division(Id, Numerator, Denominator, Result) :-
    some_table(Id, Numerator, Denominator),
    Result = sql_case([
        when(Denominator = 0, 0),
        when(sql_is_null(Denominator), 0)
    ], Numerator / Denominator).
```

## CASE WHEN for Data Cleansing

### Standardizing Values

```prolog
standardized_status(Id, CleanStatus) :-
    orders(Id, _, _, Status, _),
    CleanStatus = sql_case([
        when(sql_in(Status, ['P', 'p', 'pending', 'PENDING']), 'Pending'),
        when(sql_in(Status, ['S', 's', 'shipped', 'SHIPPED']), 'Shipped'),
        when(sql_in(Status, ['D', 'd', 'delivered', 'DELIVERED']), 'Delivered')
    ], 'Unknown').
```

### Handling Edge Cases

```prolog
safe_price(Name, SafePrice) :-
    products(_, Name, Price, _, _),
    SafePrice = sql_case([
        when(sql_is_null(Price), 0),
        when(Price < 0, 0),
        when(Price > 1000000, 1000000)  % Cap at max
    ], Price).
```

## Complete Example

```prolog
% file: case_when_example.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, hire_date-text, score-integer]).
:- sql_table(products, [id-integer, name-text, price-real, stock-integer, category-text]).
:- sql_table(orders, [id-integer, customer_id-integer, amount-real, status-text, order_date-text]).

% Basic categorization
salary_tier(Name, Tier) :-
    employees(_, Name, _, Salary, _, _),
    Tier = sql_case([
        when(Salary > 100000, 'High'),
        when(Salary > 60000, 'Medium')
    ], 'Entry').

% Value mapping
status_display(Id, Display) :-
    orders(Id, _, _, Status, _),
    Display = sql_case([
        when(Status = 'P', 'Pending'),
        when(Status = 'S', 'Shipped'),
        when(Status = 'D', 'Delivered')
    ], 'Unknown').

% Numeric computation
discount(Name, Price, DiscountAmt) :-
    products(_, Name, Price, _, Cat),
    DiscountAmt = sql_case([
        when(Cat = 'Electronics', Price * 0.15),
        when(Cat = 'Clothing', Price * 0.20)
    ], Price * 0.05).

% With alias
stock_alert(Name, Alert) :-
    products(_, Name, _, Stock, _),
    Alert = sql_as(
        sql_case([
            when(Stock = 0, 'OUT OF STOCK'),
            when(Stock < 10, 'LOW'),
            when(Stock < 50, 'OK')
        ], 'GOOD'),
        stock_status
    ).

% Conditional aggregation
dept_breakdown(Dept, High, Low) :-
    employees(_, _, Dept, Salary, _, _),
    sql_group_by([Dept]),
    High = sql_sum(sql_case([when(Salary > 80000, 1)], 0)),
    Low = sql_sum(sql_case([when(Salary =< 80000, 1)], 0)).

% Complex with multiple conditions
rating(Name, Rating) :-
    employees(_, Name, _, Salary, _, Score),
    Rating = sql_case([
        when((Salary > 100000, Score >= 90), 'Star Executive'),
        when((Salary > 100000, Score >= 70), 'Executive'),
        when((Salary > 60000, Score >= 90), 'Rising Star'),
        when(Score >= 90, 'High Performer')
    ], 'Standard').

test :-
    format('~n=== CASE WHEN Examples ===~n~n'),

    Tests = [
        salary_tier/2 - 'Basic salary categorization',
        status_display/2 - 'Value mapping',
        discount/3 - 'Numeric computation',
        stock_alert/2 - 'With alias',
        dept_breakdown/3 - 'Conditional aggregation',
        rating/2 - 'Multiple conditions'
    ],

    forall(
        member(Pred - Desc, Tests),
        (
            format('~w:~n', [Desc]),
            compile_predicate_to_sql(Pred, [], SQL),
            format('   ~w~n~n', [SQL])
        )
    ).

:- initialization(test, main).
```

## Exercises

1. **Basic CASE**: Create a query that categorizes products by price: 'Budget' (< $25), 'Mid-Range' ($25-$100), 'Premium' (> $100).

2. **Conditional Counting**: Write a query showing each department with counts of employees in each salary tier.

3. **Value Mapping**: Create a query that maps single-letter order statuses to full words.

4. **Computed Values**: Write a query that calculates shipping cost based on order amount (different rates for different amounts).

5. **Complex**: Create a customer loyalty tier based on total orders and average order value:
   - Platinum: > 10 orders AND avg > $500
   - Gold: > 10 orders OR avg > $500
   - Silver: > 5 orders
   - Bronze: Otherwise

## Summary

In this chapter, you learned:

- Basic CASE WHEN syntax: `sql_case([when(Cond, Result), ...], Default)`
- Categorizing data with conditional logic
- Computing values based on conditions
- Using CASE WHEN with aggregations (conditional counting/summing)
- Aliasing CASE WHEN columns with `sql_as/2`
- Nesting CASE expressions
- Data cleansing patterns

## Next Chapter

In Chapter 12, we'll bring it all together with practical, real-world applications and complete projects using the SQL target.
