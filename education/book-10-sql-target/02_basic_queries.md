<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 2: Basic Queries

This chapter covers fundamental SQL query patterns: comparison operators, sorting, limiting results, and column aliasing.

## Setup

For all examples in this chapter, we'll use these table declarations:

```prolog
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [
    id-integer,
    name-text,
    dept-text,
    salary-integer,
    hire_date-text,
    email-text
]).

:- sql_table(products, [
    id-integer,
    name-text,
    price-real,
    category-text,
    stock-integer
]).
```

## Comparison Operators

Prolog comparison operators translate directly to SQL:

| Prolog | SQL | Description |
|--------|-----|-------------|
| `=` | `=` | Equal |
| `\=` | `<>` or `!=` | Not equal |
| `<` | `<` | Less than |
| `>` | `>` | Greater than |
| `=<` | `<=` | Less than or equal |
| `>=` | `>=` | Greater than or equal |

### Equality

```prolog
engineering_employees(Name) :-
    employees(_, Name, Dept, _, _, _),
    Dept = 'Engineering'.

?- compile_predicate_to_sql(engineering_employees/1, [], SQL).
% SELECT name FROM employees WHERE dept = 'Engineering';
```

### Inequality

```prolog
non_engineering(Name, Dept) :-
    employees(_, Name, Dept, _, _, _),
    Dept \= 'Engineering'.

?- compile_predicate_to_sql(non_engineering/2, [], SQL).
% SELECT name, dept FROM employees WHERE dept <> 'Engineering';
```

### Numeric Comparisons

```prolog
% Greater than
high_salary(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    Salary > 75000.

% Less than
low_salary(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    Salary < 50000.

% Greater than or equal
minimum_wage(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    Salary >= 30000.

% Less than or equal
budget_friendly(Name, Price) :-
    products(_, Name, Price, _, _),
    Price =< 100.0.
```

### Combining Conditions

Multiple conditions in a clause body become AND:

```prolog
mid_range_salary(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    Salary >= 50000,
    Salary =< 80000.

?- compile_predicate_to_sql(mid_range_salary/2, [], SQL).
% SELECT name, salary FROM employees WHERE salary >= 50000 AND salary <= 80000;
```

## ORDER BY

Use `sql_order_by/2` to sort results:

```prolog
sql_order_by(Column, Direction)
% Direction: asc or desc
```

### Single Column Sort

```prolog
employees_by_salary(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    sql_order_by(Salary, desc).

?- compile_predicate_to_sql(employees_by_salary/2, [], SQL).
% SELECT name, salary FROM employees ORDER BY salary DESC;
```

### Ascending Order

```prolog
employees_alphabetical(Name) :-
    employees(_, Name, _, _, _, _),
    sql_order_by(Name, asc).

?- compile_predicate_to_sql(employees_alphabetical/1, [], SQL).
% SELECT name FROM employees ORDER BY name ASC;
```

### Multiple Column Sort

Call `sql_order_by/2` multiple times for multi-column sorting:

```prolog
employees_by_dept_salary(Name, Dept, Salary) :-
    employees(_, Name, Dept, Salary, _, _),
    sql_order_by(Dept, asc),
    sql_order_by(Salary, desc).

?- compile_predicate_to_sql(employees_by_dept_salary/3, [], SQL).
% SELECT name, dept, salary FROM employees ORDER BY dept ASC, salary DESC;
```

## LIMIT and OFFSET

Control the number of returned rows with `sql_limit/1` and `sql_offset/1`.

### LIMIT Only

```prolog
top_5_earners(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    sql_order_by(Salary, desc),
    sql_limit(5).

?- compile_predicate_to_sql(top_5_earners/2, [], SQL).
% SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5;
```

### LIMIT with OFFSET (Pagination)

```prolog
% Get rows 11-20 (page 2 with 10 per page)
page_2_employees(Name) :-
    employees(_, Name, _, _, _, _),
    sql_order_by(Name, asc),
    sql_limit(10),
    sql_offset(10).

?- compile_predicate_to_sql(page_2_employees/1, [], SQL).
% SELECT name FROM employees ORDER BY name ASC LIMIT 10 OFFSET 10;
```

### Combining with Filters

```prolog
top_3_engineering(Name, Salary) :-
    employees(_, Name, Dept, Salary, _, _),
    Dept = 'Engineering',
    sql_order_by(Salary, desc),
    sql_limit(3).

?- compile_predicate_to_sql(top_3_engineering/2, [], SQL).
% SELECT name, salary FROM employees WHERE dept = 'Engineering' ORDER BY salary DESC LIMIT 3;
```

## DISTINCT

Use `sql_distinct/0` to remove duplicate rows:

```prolog
unique_departments(Dept) :-
    employees(_, _, Dept, _, _, _),
    sql_distinct.

?- compile_predicate_to_sql(unique_departments/1, [], SQL).
% SELECT DISTINCT dept FROM employees;
```

### DISTINCT with ORDER BY

```prolog
unique_depts_sorted(Dept) :-
    employees(_, _, Dept, _, _, _),
    sql_distinct,
    sql_order_by(Dept, asc).

?- compile_predicate_to_sql(unique_depts_sorted/1, [], SQL).
% SELECT DISTINCT dept FROM employees ORDER BY dept ASC;
```

### DISTINCT with Multiple Columns

```prolog
unique_dept_salary_combos(Dept, Salary) :-
    employees(_, _, Dept, Salary, _, _),
    sql_distinct.

?- compile_predicate_to_sql(unique_dept_salary_combos/2, [], SQL).
% SELECT DISTINCT dept, salary FROM employees;
```

## Column Aliasing with sql_as/2

Rename columns in output using `sql_as/2`:

```prolog
sql_as(Column, AliasName)
```

### Simple Alias

```prolog
employee_info(EmpName, Department) :-
    employees(_, Name, Dept, _, _, _),
    EmpName = sql_as(Name, employee_name),
    Department = sql_as(Dept, department).

?- compile_predicate_to_sql(employee_info/2, [], SQL).
% SELECT name AS employee_name, dept AS department FROM employees;
```

### Alias with Expressions

Aliases are especially useful with computed values:

```prolog
salary_in_thousands(Name, SalaryK) :-
    employees(_, Name, _, Salary, _, _),
    SalaryK = sql_as(Salary / 1000, salary_k).

?- compile_predicate_to_sql(salary_in_thousands/2, [], SQL).
% SELECT name, salary / 1000 AS salary_k FROM employees;
```

## Arithmetic Operations

Basic arithmetic works in SELECT clauses:

```prolog
% Addition
with_bonus(Name, Total) :-
    employees(_, Name, _, Salary, _, _),
    Total = Salary + 5000.

% Multiplication
annual_salary(Name, Annual) :-
    employees(_, Name, _, Monthly, _, _),
    Annual = Monthly * 12.

% Division
daily_rate(Name, Daily) :-
    employees(_, Name, _, Salary, _, _),
    Daily = Salary / 365.

% Combined
tax_estimate(Name, Tax) :-
    employees(_, Name, _, Salary, _, _),
    Tax = (Salary * 0.25) - 1000.
```

## String Matching Preview

While detailed in Chapter 10, here's a preview of pattern matching:

```prolog
% Names starting with 'A'
names_starting_a(Name) :-
    employees(_, Name, _, _, _, _),
    sql_like(Name, 'A%').

% Email from specific domain
company_emails(Name, Email) :-
    employees(_, Name, _, _, _, Email),
    sql_like(Email, '%@company.com').
```

## Complete Example

```prolog
% file: basic_queries.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, hire_date-text, email-text]).

% Various query patterns
all_employees(Name, Dept, Salary) :-
    employees(_, Name, Dept, Salary, _, _).

high_earners(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    Salary > 80000.

mid_range(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    Salary >= 50000,
    Salary =< 80000.

top_earners(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    sql_order_by(Salary, desc),
    sql_limit(5).

paginated(Name, Page) :-
    employees(_, Name, _, _, _, _),
    sql_order_by(Name, asc),
    sql_limit(10),
    sql_offset(Page * 10).

unique_depts(Dept) :-
    employees(_, _, Dept, _, _, _),
    sql_distinct,
    sql_order_by(Dept, asc).

with_aliases(EmpName, Dept, SalaryK) :-
    employees(_, Name, D, Salary, _, _),
    EmpName = sql_as(Name, employee),
    Dept = sql_as(D, department),
    SalaryK = sql_as(Salary / 1000, salary_thousands).

% Test runner
test :-
    format('~n=== Basic Query Patterns ===~n~n'),

    Queries = [
        all_employees/3 - 'All employees',
        high_earners/2 - 'High earners (> 80000)',
        mid_range/2 - 'Mid-range (50000-80000)',
        top_earners/2 - 'Top 5 earners',
        unique_depts/1 - 'Unique departments',
        with_aliases/3 - 'With column aliases'
    ],

    forall(
        member(Pred - Desc, Queries),
        (
            format('~w:~n', [Desc]),
            compile_predicate_to_sql(Pred, [], SQL),
            format('  ~w~n~n', [SQL])
        )
    ).

:- initialization(test, main).
```

Run:

```bash
swipl basic_queries.pl
```

## Exercises

1. **Comparison Practice**: Write predicates for:
   - Products with price between $10 and $50
   - Employees not in 'Sales' department
   - Products with stock less than 10

2. **Sorting**: Create a predicate that returns products sorted by category (ascending), then by price (descending).

3. **Pagination**: Write a predicate `product_page(Name, Price, PageNum)` that returns 20 products per page.

4. **Aliases**: Create a query that returns employee name as "full_name" and salary * 1.1 as "salary_with_raise".

5. **Combined**: Write a predicate for the top 10 highest-priced products in the 'Electronics' category, sorted by price descending, with an alias for the price column.

## Summary

In this chapter, you learned:

- All comparison operators (`=`, `\=`, `<`, `>`, `=<`, `>=`)
- Sorting with `sql_order_by/2` (single and multiple columns)
- Limiting results with `sql_limit/1` and `sql_offset/1`
- Removing duplicates with `sql_distinct/0`
- Aliasing columns with `sql_as/2`
- Basic arithmetic in SELECT clauses

## Next Chapter

In Chapter 3, we'll explore JOINs:
- Implicit INNER JOINs via shared variables
- Explicit LEFT, RIGHT, and FULL OUTER JOINs
- Multi-table joins
- Self-joins

---

## Navigation

**←** [Previous: Chapter 1: Getting Started with SQL Target](01_getting_started) | [📖 Book 10: SQL Target](./) | [Next: Chapter 3: JOINs →](03_joins)
