<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 10: SQL Functions

This chapter covers built-in SQL functions for NULL handling, string manipulation, date operations, and pattern matching.

## Setup

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
    description-text,
    price-real,
    category-text,
    created_date-text
]).

:- sql_table(orders, [
    id-integer,
    customer_id-integer,
    order_date-text,
    ship_date-text,
    amount-real
]).
```

## NULL Handling Functions

### COALESCE

Returns the first non-NULL value from a list:

```prolog
sql_coalesce([Value1, Value2, ...])
```

Example:

```prolog
employee_with_dept(Name, Dept) :-
    employees(_, Name, D, _, _, _),
    Dept = sql_coalesce([D, 'Unknown']).

?- compile_predicate_to_sql(employee_with_dept/2, [], SQL).
% SELECT name, COALESCE(dept, 'Unknown') FROM employees;
```

Multiple fallbacks:

```prolog
contact_info(Id, Contact) :-
    employees(Id, _, _, _, _, Email),
    Contact = sql_coalesce([Email, 'no-email@company.com', 'N/A']).
```

### NULLIF

Returns NULL if two values are equal, otherwise returns the first value:

```prolog
sql_nullif(Value, CompareValue)
```

Example (treat zero as NULL):

```prolog
product_real_price(Name, Price) :-
    products(_, Name, _, P, _, _),
    Price = sql_nullif(P, 0).

% SELECT name, NULLIF(price, 0) FROM products;
```

### IFNULL

SQLite-specific: Returns second value if first is NULL:

```prolog
sql_ifnull(Value, Default)
```

Example:

```prolog
order_amount(Id, Amount) :-
    orders(Id, _, _, _, A),
    Amount = sql_ifnull(A, 0).

% SELECT id, IFNULL(amount, 0) FROM orders;
```

### IS NULL / IS NOT NULL

Test for NULL values in WHERE clauses:

```prolog
% Find unshipped orders
unshipped_orders(Id) :-
    orders(Id, _, _, ShipDate, _),
    sql_is_null(ShipDate).

% SELECT id FROM orders WHERE ship_date IS NULL;

% Find shipped orders
shipped_orders(Id, ShipDate) :-
    orders(Id, _, _, ShipDate, _),
    sql_is_not_null(ShipDate).

% SELECT id, ship_date FROM orders WHERE ship_date IS NOT NULL;
```

## String Functions

### CONCAT

Concatenate multiple values:

```prolog
sql_concat([Value1, Value2, ...])
```

Example:

```prolog
full_info(Info) :-
    employees(_, Name, Dept, _, _, _),
    Info = sql_concat([Name, ' - ', Dept]).

% SELECT name || ' - ' || dept FROM employees;
```

### UPPER / LOWER

Change case:

```prolog
name_cases(Name, Upper, Lower) :-
    employees(_, Name, _, _, _, _),
    Upper = sql_upper(Name),
    Lower = sql_lower(Name).

% SELECT name, UPPER(name), LOWER(name) FROM employees;
```

### SUBSTRING

Extract part of a string:

```prolog
sql_substring(Value, Start, Length)
```

Example:

```prolog
name_initials(Name, Initials) :-
    employees(_, Name, _, _, _, _),
    Initials = sql_substring(Name, 1, 1).

% SELECT name, SUBSTR(name, 1, 1) FROM employees;
```

### TRIM / LTRIM / RTRIM

Remove whitespace:

```prolog
trimmed_name(Name, Trimmed) :-
    products(_, Name, _, _, _, _),
    Trimmed = sql_trim(Name).

% SELECT name, TRIM(name) FROM products;

% Left trim only
left_trimmed(Name, Trimmed) :-
    products(_, Name, _, _, _, _),
    Trimmed = sql_ltrim(Name).

% Right trim only
right_trimmed(Name, Trimmed) :-
    products(_, Name, _, _, _, _),
    Trimmed = sql_rtrim(Name).
```

### LENGTH

Get string length:

```prolog
name_length(Name, Len) :-
    employees(_, Name, _, _, _, _),
    Len = sql_length(Name).

% SELECT name, LENGTH(name) FROM employees;
```

### REPLACE

Replace occurrences in a string:

```prolog
sql_replace(Value, Search, Replace)
```

Example:

```prolog
sanitized_email(Name, Email) :-
    employees(_, Name, _, _, _, E),
    Email = sql_replace(E, '@', ' at ').

% SELECT name, REPLACE(email, '@', ' at ') FROM employees;
```

## Date Functions

### DATE

Extract date from datetime:

```prolog
order_date_only(Id, DateOnly) :-
    orders(Id, _, OrderDate, _, _),
    DateOnly = sql_date(OrderDate).

% SELECT id, DATE(order_date) FROM orders;
```

### DATETIME

Convert to datetime:

```prolog
as_datetime(Id, DT) :-
    orders(Id, _, OrderDate, _, _),
    DT = sql_datetime(OrderDate).

% SELECT id, DATETIME(order_date) FROM orders;
```

### DATE_ADD

Add interval to date:

```prolog
sql_date_add(DateValue, Amount, Unit)
% Units: days, months, years
```

Example:

```prolog
due_date(Id, DueDate) :-
    orders(Id, _, OrderDate, _, _),
    DueDate = sql_date_add(OrderDate, 7, days).

% SELECT id, DATE(order_date, '+7 days') FROM orders;
```

### DATE_DIFF

Calculate days between dates:

```prolog
sql_date_diff(Date1, Date2)
```

Example:

```prolog
shipping_days(Id, Days) :-
    orders(Id, _, OrderDate, ShipDate, _),
    Days = sql_date_diff(ShipDate, OrderDate).

% SELECT id, JULIANDAY(ship_date) - JULIANDAY(order_date) FROM orders;
```

### EXTRACT

Extract date parts:

```prolog
sql_extract(Part, DateValue)
% Parts: year, month, day, hour, minute, second
```

Example:

```prolog
hire_year_month(Name, Year, Month) :-
    employees(_, Name, _, _, HireDate, _),
    Year = sql_extract(year, HireDate),
    Month = sql_extract(month, HireDate).

% SELECT name, STRFTIME('%Y', hire_date), STRFTIME('%m', hire_date) FROM employees;
```

### STRFTIME

Format dates with custom patterns:

```prolog
sql_strftime(Format, DateValue)
```

Example:

```prolog
formatted_date(Id, Formatted) :-
    orders(Id, _, OrderDate, _, _),
    Formatted = sql_strftime('%Y-%m-%d', OrderDate).

% SELECT id, STRFTIME('%Y-%m-%d', order_date) FROM orders;
```

Common format codes:
| Code | Meaning |
|------|---------|
| `%Y` | 4-digit year |
| `%m` | Month (01-12) |
| `%d` | Day (01-31) |
| `%H` | Hour (00-23) |
| `%M` | Minute (00-59) |
| `%S` | Second (00-59) |

## Pattern Matching

### BETWEEN

Test if value is within a range:

```prolog
sql_between(Value, Low, High)
sql_not_between(Value, Low, High)
```

Examples:

```prolog
mid_salary(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    sql_between(Salary, 50000, 100000).

% SELECT name, salary FROM employees WHERE salary BETWEEN 50000 AND 100000;

extreme_salary(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    sql_not_between(Salary, 40000, 80000).

% SELECT name, salary FROM employees WHERE salary NOT BETWEEN 40000 AND 80000;
```

Date ranges:

```prolog
q1_orders(Id, OrderDate) :-
    orders(Id, _, OrderDate, _, _),
    sql_between(OrderDate, '2024-01-01', '2024-03-31').
```

### LIKE

Pattern matching with wildcards:

```prolog
sql_like(Value, Pattern)
sql_not_like(Value, Pattern)
```

Wildcards:
- `%` matches any sequence of characters
- `_` matches any single character

Examples:

```prolog
% Names starting with 'J'
j_names(Name) :-
    employees(_, Name, _, _, _, _),
    sql_like(Name, 'J%').

% SELECT name FROM employees WHERE name LIKE 'J%';

% Names containing 'son'
son_names(Name) :-
    employees(_, Name, _, _, _, _),
    sql_like(Name, '%son%').

% Names ending with 'a'
a_ending(Name) :-
    employees(_, Name, _, _, _, _),
    sql_like(Name, '%a').

% NOT LIKE
not_test_products(Name) :-
    products(_, Name, _, _, _, _),
    sql_not_like(Name, 'Test%').
```

### GLOB

SQLite-specific pattern matching (case-sensitive):

```prolog
sql_glob(Value, Pattern)
```

```prolog
glob_products(Name) :-
    products(_, Name, _, _, _, _),
    sql_glob(Name, '*Pro*').

% SELECT name FROM products WHERE name GLOB '*Pro*';
```

### IN (List)

Test if value is in a list:

```prolog
sql_in(Value, [Item1, Item2, ...])
sql_not_in(Value, [Item1, Item2, ...])
```

Examples:

```prolog
engineering_depts(Name, Dept) :-
    employees(_, Name, Dept, _, _, _),
    sql_in(Dept, [engineering, 'r&d', development]).

% SELECT name, dept FROM employees WHERE dept IN ('engineering', 'r&d', 'development');

non_admin(Name, Dept) :-
    employees(_, Name, Dept, _, _, _),
    sql_not_in(Dept, [admin, hr, management]).

% SELECT name, dept FROM employees WHERE dept NOT IN ('admin', 'hr', 'management');
```

## Nested Functions

Functions can be nested:

```prolog
upper_trimmed(Name, Result) :-
    employees(_, Name, _, _, _, _),
    Result = sql_upper(sql_trim(Name)).

% SELECT name, UPPER(TRIM(name)) FROM employees;
```

Complex nesting:

```prolog
display_name(DisplayName) :-
    employees(_, Name, Dept, _, _, _),
    DisplayName = sql_as(
        sql_concat([sql_upper(Name), ' (', Dept, ')']),
        display_name
    ).

% SELECT UPPER(name) || ' (' || dept || ')' AS display_name FROM employees;
```

## Complete Example

```prolog
% file: sql_functions_example.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/sql_target').

:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, hire_date-text, email-text]).
:- sql_table(orders, [id-integer, customer_id-integer, order_date-text, ship_date-text, amount-real]).

% NULL handling
with_default_dept(Name, Dept) :-
    employees(_, Name, D, _, _, _),
    Dept = sql_coalesce([D, 'Unknown']).

unshipped(Id) :-
    orders(Id, _, _, ShipDate, _),
    sql_is_null(ShipDate).

% String functions
full_info(Info) :-
    employees(_, Name, Dept, _, _, _),
    Info = sql_concat([Name, ' - ', Dept]).

name_upper(Name, Upper) :-
    employees(_, Name, _, _, _, _),
    Upper = sql_upper(Name).

% Date functions
hire_year(Name, Year) :-
    employees(_, Name, _, _, HireDate, _),
    Year = sql_extract(year, HireDate).

shipping_time(Id, Days) :-
    orders(Id, _, OrderDate, ShipDate, _),
    Days = sql_date_diff(ShipDate, OrderDate).

% Pattern matching
mid_range_salary(Name, Salary) :-
    employees(_, Name, _, Salary, _, _),
    sql_between(Salary, 50000, 100000).

j_employees(Name) :-
    employees(_, Name, _, _, _, _),
    sql_like(Name, 'J%').

tech_depts(Name, Dept) :-
    employees(_, Name, Dept, _, _, _),
    sql_in(Dept, [engineering, development, 'IT']).

% Nested
display(DisplayName) :-
    employees(_, Name, Dept, _, _, _),
    DisplayName = sql_concat([sql_upper(Name), ' (', sql_lower(Dept), ')']).

test :-
    format('~n=== SQL Functions Examples ===~n~n'),

    Tests = [
        with_default_dept/2 - 'COALESCE',
        unshipped/1 - 'IS NULL',
        full_info/1 - 'CONCAT',
        name_upper/2 - 'UPPER',
        hire_year/2 - 'EXTRACT year',
        shipping_time/2 - 'DATE_DIFF',
        mid_range_salary/2 - 'BETWEEN',
        j_employees/1 - 'LIKE',
        tech_depts/2 - 'IN list',
        display/1 - 'Nested functions'
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

1. **NULL Handling**: Write a query that shows order ID and amount, replacing NULL amounts with 0.

2. **String Functions**: Create a query that shows employee names in uppercase with their email domain (part after @).

3. **Date Functions**: Find all orders placed in the last 30 days (using date arithmetic).

4. **Pattern Matching**: Write queries to find:
   - Products with 'Pro' anywhere in the name
   - Employees with salary between $60,000 and $90,000
   - Employees NOT in admin or HR departments

5. **Combined**: Create a display column that shows: "NAME (Dept) - Hired: YYYY" using string, date, and nested functions.

## Summary

In this chapter, you learned:

- NULL handling: `sql_coalesce`, `sql_nullif`, `sql_ifnull`, `sql_is_null`, `sql_is_not_null`
- String functions: `sql_concat`, `sql_upper`, `sql_lower`, `sql_substring`, `sql_trim`, `sql_length`, `sql_replace`
- Date functions: `sql_date`, `sql_datetime`, `sql_date_add`, `sql_date_diff`, `sql_extract`, `sql_strftime`
- Pattern matching: `sql_between`, `sql_like`, `sql_glob`, `sql_in`
- Nesting functions for complex expressions

## Next Chapter

In Chapter 11, we'll explore CASE WHEN expressions for conditional logic in queries.

---

## Navigation

**←** [Previous: Chapter 9: Set Operations](09_set_operations) | [📖 Book 10: SQL Target](./) | [Next: Chapter 11: CASE WHEN Expressions →](11_case_when)
