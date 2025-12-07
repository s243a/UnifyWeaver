<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 9: Set Operations

Set operations combine results from multiple queries. They're useful for finding common elements, differences, or merging result sets.

## Setup

```prolog
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees_us, [id-integer, name-text, dept-text, salary-integer]).
:- sql_table(employees_uk, [id-integer, name-text, dept-text, salary-integer]).
:- sql_table(contractors, [id-integer, name-text, dept-text, rate-integer]).
:- sql_table(customers_2023, [id-integer, name-text, email-text]).
:- sql_table(customers_2024, [id-integer, name-text, email-text]).
:- sql_table(products_active, [id-integer, name-text, price-real]).
:- sql_table(products_discontinued, [id-integer, name-text, price-real]).
```

## Set Operation Syntax

```prolog
compile_set_operation(Operation, [Pred1/Arity, Pred2/Arity, ...], Options, SQL)
```

Parameters:
- `Operation`: `union`, `union_all`, `intersect`, `intersect_all`, `except`, `except_all`
- `Predicates`: List of predicates (must have same column structure)
- `Options`: Compilation options
- `SQL`: Output SQL string

## UNION

Combines results from multiple queries, removing duplicates.

### Basic UNION

```prolog
% All employees from US
us_employees(Name, Dept) :-
    employees_us(_, Name, Dept, _).

% All employees from UK
uk_employees(Name, Dept) :-
    employees_uk(_, Name, Dept, _).

% Combined (duplicates removed)
?- compile_set_operation(union, [us_employees/2, uk_employees/2], [], SQL).
```

Generated SQL:
```sql
SELECT name, dept FROM employees_us
UNION
SELECT name, dept FROM employees_uk;
```

### Multi-Way UNION

Combine more than two queries:

```prolog
all_workers(Name, Dept) :-
    employees_us(_, Name, Dept, _).
all_workers(Name, Dept) :-
    employees_uk(_, Name, Dept, _).
contractor_workers(Name, Dept) :-
    contractors(_, Name, Dept, _).

?- compile_set_operation(union,
    [us_employees/2, uk_employees/2, contractor_workers/2],
    [], SQL).
```

Generated SQL:
```sql
SELECT name, dept FROM employees_us
UNION
SELECT name, dept FROM employees_uk
UNION
SELECT name, dept FROM contractors;
```

## UNION ALL

Combines results keeping all rows (including duplicates). More efficient than UNION.

```prolog
% Include duplicates
?- compile_set_operation(union_all, [us_employees/2, uk_employees/2], [], SQL).
```

Generated SQL:
```sql
SELECT name, dept FROM employees_us
UNION ALL
SELECT name, dept FROM employees_uk;
```

### When to Use UNION ALL

| Use UNION When | Use UNION ALL When |
|----------------|-------------------|
| Need unique results | Duplicates are acceptable |
| Combining overlapping data | Combining non-overlapping data |
| Result size is small | Performance is critical |
| Business requires dedup | Counting requires all rows |

## INTERSECT

Returns only rows that appear in ALL queries.

### Find Common Elements

```prolog
% Customers in 2023
customers_23(Name, Email) :-
    customers_2023(_, Name, Email).

% Customers in 2024
customers_24(Name, Email) :-
    customers_2024(_, Name, Email).

% Customers who were active in both years
?- compile_set_operation(intersect, [customers_23/2, customers_24/2], [], SQL).
```

Generated SQL:
```sql
SELECT name, email FROM customers_2023
INTERSECT
SELECT name, email FROM customers_2024;
```

### INTERSECT ALL

Keeps duplicate rows from the intersection:

```prolog
?- compile_set_operation(intersect_all, [customers_23/2, customers_24/2], [], SQL).
```

Generated SQL:
```sql
SELECT name, email FROM customers_2023
INTERSECT ALL
SELECT name, email FROM customers_2024;
```

## EXCEPT

Returns rows from the first query that don't appear in subsequent queries.

### Find Differences

```prolog
% Products currently active
active(Name, Price) :-
    products_active(_, Name, Price).

% Products discontinued
discontinued(Name, Price) :-
    products_discontinued(_, Name, Price).

% Products that are active but not discontinued
?- compile_set_operation(except, [active/2, discontinued/2], [], SQL).
```

Generated SQL:
```sql
SELECT name, price FROM products_active
EXCEPT
SELECT name, price FROM products_discontinued;
```

### Order Matters for EXCEPT

```prolog
% A EXCEPT B: Items in A but not in B
?- compile_set_operation(except, [active/2, discontinued/2], [], SQL1).
% SELECT ... FROM products_active EXCEPT SELECT ... FROM products_discontinued;

% B EXCEPT A: Items in B but not in A
?- compile_set_operation(except, [discontinued/2, active/2], [], SQL2).
% SELECT ... FROM products_discontinued EXCEPT SELECT ... FROM products_active;
```

### EXCEPT ALL

Keeps duplicates in the difference:

```prolog
?- compile_set_operation(except_all, [active/2, discontinued/2], [], SQL).
```

## Set Operations with Filters

Each predicate can include WHERE conditions:

```prolog
% US engineering employees
us_engineering(Name) :-
    employees_us(_, Name, Dept, _),
    Dept = 'Engineering'.

% UK engineering employees
uk_engineering(Name) :-
    employees_uk(_, Name, Dept, _),
    Dept = 'Engineering'.

% All engineering employees from both locations
?- compile_set_operation(union, [us_engineering/1, uk_engineering/1], [], SQL).
```

Generated SQL:
```sql
SELECT name FROM employees_us WHERE dept = 'Engineering'
UNION
SELECT name FROM employees_uk WHERE dept = 'Engineering';
```

## Set Operations with ORDER BY

Add ORDER BY to the final result:

```prolog
us_emps(Name, Salary) :-
    employees_us(_, Name, _, Salary).

uk_emps(Name, Salary) :-
    employees_uk(_, Name, _, Salary).

% Combine and sort by salary descending
?- compile_set_operation(union, [us_emps/2, uk_emps/2],
    [order_by([(Salary, desc)])], SQL).
```

Generated SQL:
```sql
SELECT name, salary FROM employees_us
UNION
SELECT name, salary FROM employees_uk
ORDER BY salary DESC;
```

## Column Compatibility Requirements

All predicates in a set operation must have:
1. Same number of columns
2. Compatible column types

```prolog
% VALID: Same structure
pred1(Name, Age) :- table1(_, Name, Age).
pred2(Name, Age) :- table2(_, Name, Age).

% INVALID: Different column count
pred1(Name) :- table1(_, Name, _).
pred2(Name, Age) :- table2(_, Name, Age).
```

## Practical Use Cases

### 1. Finding New Customers

```prolog
customers_last_year(Email) :-
    customers_2023(_, _, Email).

customers_this_year(Email) :-
    customers_2024(_, _, Email).

% New customers (in 2024 but not 2023)
?- compile_set_operation(except, [customers_this_year/1, customers_last_year/1], [], SQL).
```

### 2. Finding Churned Customers

```prolog
% Churned customers (in 2023 but not 2024)
?- compile_set_operation(except, [customers_last_year/1, customers_this_year/1], [], SQL).
```

### 3. Finding Retained Customers

```prolog
% Retained customers (in both years)
?- compile_set_operation(intersect, [customers_last_year/1, customers_this_year/1], [], SQL).
```

### 4. Merging Data Sources

```prolog
all_employee_names(Name) :- employees_us(_, Name, _, _).
all_employee_names(Name) :- employees_uk(_, Name, _, _).
all_contractor_names(Name) :- contractors(_, Name, _, _).

% All people working for the company
?- compile_set_operation(union,
    [all_employee_names/1, all_contractor_names/1], [], SQL).
```

## Complete Example

```prolog
% file: set_operations_example.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees_us, [id-integer, name-text, dept-text, salary-integer]).
:- sql_table(employees_uk, [id-integer, name-text, dept-text, salary-integer]).
:- sql_table(customers_2023, [id-integer, name-text, email-text]).
:- sql_table(customers_2024, [id-integer, name-text, email-text]).

% Employee predicates
us_emps(Name, Dept) :- employees_us(_, Name, Dept, _).
uk_emps(Name, Dept) :- employees_uk(_, Name, Dept, _).

us_eng(Name) :- employees_us(_, Name, Dept, _), Dept = 'Engineering'.
uk_eng(Name) :- employees_uk(_, Name, Dept, _), Dept = 'Engineering'.

% Customer predicates
cust_23(Email) :- customers_2023(_, _, Email).
cust_24(Email) :- customers_2024(_, _, Email).

test :-
    format('~n=== Set Operation Examples ===~n~n'),

    format('1. UNION (all employees):~n'),
    compile_set_operation(union, [us_emps/2, uk_emps/2], [], SQL1),
    format('   ~w~n~n', [SQL1]),

    format('2. UNION ALL (with duplicates):~n'),
    compile_set_operation(union_all, [us_emps/2, uk_emps/2], [], SQL2),
    format('   ~w~n~n', [SQL2]),

    format('3. INTERSECT (retained customers):~n'),
    compile_set_operation(intersect, [cust_23/1, cust_24/1], [], SQL3),
    format('   ~w~n~n', [SQL3]),

    format('4. EXCEPT (new customers in 2024):~n'),
    compile_set_operation(except, [cust_24/1, cust_23/1], [], SQL4),
    format('   ~w~n~n', [SQL4]),

    format('5. EXCEPT (churned customers):~n'),
    compile_set_operation(except, [cust_23/1, cust_24/1], [], SQL5),
    format('   ~w~n~n', [SQL5]),

    format('6. UNION with filters:~n'),
    compile_set_operation(union, [us_eng/1, uk_eng/1], [], SQL6),
    format('   ~w~n~n', [SQL6]).

:- initialization(test, main).
```

## Exercises

1. **Basic UNION**: Combine employees from US and UK offices, showing name and salary.

2. **INTERSECT**: Find products that appear in both the active and discontinued lists (maybe re-released).

3. **EXCEPT**: Find departments that exist in the US office but not the UK office.

4. **Multi-way**: Combine employees from 3 different region tables using UNION ALL.

5. **Customer Analysis**: Create queries to find:
   - All customers (across all years)
   - Customers who ordered every year (2022, 2023, 2024)
   - Customers who only ordered in 2024 (new this year)

## Summary

In this chapter, you learned:

- `UNION` and `UNION ALL` for combining results
- `INTERSECT` and `INTERSECT ALL` for finding common elements
- `EXCEPT` and `EXCEPT ALL` for finding differences
- Column compatibility requirements
- Adding ORDER BY to set operations
- Practical use cases: new/churned/retained customer analysis

## Next Chapter

In Chapter 10, we'll explore SQL Functions:
- NULL handling (COALESCE, NULLIF)
- String functions (CONCAT, UPPER, SUBSTRING)
- Date functions (DATE, DATE_ADD, EXTRACT)
- Pattern matching (LIKE, BETWEEN, IN)
