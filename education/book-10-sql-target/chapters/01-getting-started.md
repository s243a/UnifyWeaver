<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 1: Getting Started with SQL Target

In this chapter, you'll learn how to set up and use UnifyWeaver's SQL target to compile Prolog predicates into SQL queries.

## What is the SQL Target?

The SQL target is unique among UnifyWeaver's compilation targets. While other targets (Bash, C#, Go, Rust) produce executable programs, the SQL target generates declarative SQL queries meant for execution on relational databases.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Prolog         │     │  UnifyWeaver    │     │  SQL Query      │
│  Predicate      │ ──> │  SQL Target     │ ──> │  (SELECT...)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        v
                                                ┌─────────────────┐
                                                │  Database       │
                                                │  (SQLite, etc.) │
                                                └─────────────────┘
```

This approach lets you:
- Write queries in declarative Prolog
- Generate optimized SQL for any database
- Validate query logic before deployment
- Create reusable query templates

## Loading the SQL Target Module

To use the SQL target, load the module in your Prolog file:

```prolog
:- use_module('src/unifyweaver/targets/sql_target').
```

Or from the SWI-Prolog REPL:

```prolog
?- use_module('src/unifyweaver/targets/sql_target').
true.
```

## Declaring Table Schemas

Before writing queries, you must declare your database tables using `sql_table/2`. This tells UnifyWeaver the structure of your data:

```prolog
:- sql_table(TableName, [Column1-Type1, Column2-Type2, ...]).
```

### Supported Types

| Type | Description | SQL Equivalent |
|------|-------------|----------------|
| `integer` | Whole numbers | INTEGER |
| `text` | Strings | TEXT/VARCHAR |
| `real` | Floating point | REAL/FLOAT |

### Example: Employee Database

```prolog
% Employees table
:- sql_table(employees, [
    id-integer,
    name-text,
    dept-text,
    salary-integer,
    hire_date-text
]).

% Departments table
:- sql_table(departments, [
    id-integer,
    name-text,
    budget-real,
    manager_id-integer
]).

% Orders table
:- sql_table(orders, [
    id-integer,
    customer_id-integer,
    product-text,
    amount-real,
    order_date-text
]).
```

## Your First Predicate-to-SQL Compilation

Let's write a simple predicate and compile it to SQL.

### Step 1: Define a Predicate

```prolog
% Get all employee names
employee_names(Name) :-
    employees(_, Name, _, _, _).
```

This predicate extracts the `name` column from the `employees` table. The underscores (`_`) indicate columns we don't need.

### Step 2: Compile to SQL

```prolog
?- compile_predicate_to_sql(employee_names/1, [], SQL).
SQL = 'SELECT name FROM employees;'.
```

The `compile_predicate_to_sql/3` predicate takes:
1. `Predicate/Arity` - The predicate to compile
2. `Options` - A list of compilation options (empty for defaults)
3. `SQL` - The generated SQL string (output)

### Step 3: Run on Database

Save the SQL and run it on SQLite:

```bash
# Create a test database
sqlite3 test.db "CREATE TABLE employees (
    id INTEGER, name TEXT, dept TEXT, salary INTEGER, hire_date TEXT
);"

# Insert test data
sqlite3 test.db "INSERT INTO employees VALUES
    (1, 'Alice', 'Engineering', 85000, '2020-01-15'),
    (2, 'Bob', 'Sales', 65000, '2019-06-01'),
    (3, 'Carol', 'Engineering', 95000, '2018-03-20');"

# Run the generated query
sqlite3 test.db "SELECT name FROM employees;"
```

Output:
```
Alice
Bob
Carol
```

## Adding WHERE Clauses

Prolog constraints become SQL WHERE clauses:

```prolog
% Employees earning over 80000
high_earners(Name, Salary) :-
    employees(_, Name, _, Salary, _),
    Salary > 80000.

?- compile_predicate_to_sql(high_earners/2, [], SQL).
SQL = 'SELECT name, salary FROM employees WHERE salary > 80000;'.
```

Multiple constraints are combined with AND:

```prolog
% Engineering employees earning over 80000
senior_engineers(Name, Salary) :-
    employees(_, Name, Dept, Salary, _),
    Dept = 'Engineering',
    Salary > 80000.

?- compile_predicate_to_sql(senior_engineers/2, [], SQL).
SQL = 'SELECT name, salary FROM employees WHERE dept = \'Engineering\' AND salary > 80000;'.
```

## Compilation Options

The second argument to `compile_predicate_to_sql/3` accepts options:

### format(select) - Default

Generates a SELECT statement:

```prolog
?- compile_predicate_to_sql(employee_names/1, [format(select)], SQL).
SQL = 'SELECT name FROM employees;'.
```

### format(view) with view_name(Name)

Generates a CREATE VIEW statement:

```prolog
?- compile_predicate_to_sql(high_earners/2, [format(view), view_name(top_earners)], SQL).
SQL = 'CREATE VIEW IF NOT EXISTS top_earners AS SELECT name, salary FROM employees WHERE salary > 80000;'.
```

## Writing SQL to Files

Use `write_sql_file/2` to save generated SQL:

```prolog
?- compile_predicate_to_sql(high_earners/2, [], SQL),
   write_sql_file('output/high_earners.sql', SQL).
```

This creates a file with proper headers:

```sql
-- Generated by UnifyWeaver SQL Target
-- Timestamp: 2025-01-15T10:30:00

SELECT name, salary FROM employees WHERE salary > 80000;
```

## Complete Example

Here's a complete working example you can save and run:

```prolog
% file: sql_intro.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/sql_target').

% Table declarations
:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, hire_date-text]).

% Predicates
all_employees(Id, Name, Dept, Salary) :-
    employees(Id, Name, Dept, Salary, _).

engineering_team(Name, Salary) :-
    employees(_, Name, Dept, Salary, _),
    Dept = 'Engineering'.

high_earners(Name, Salary) :-
    employees(_, Name, _, Salary, _),
    Salary > 80000.

% Test runner
test :-
    format('~n=== SQL Target Introduction ===~n~n'),

    format('1. All employees:~n'),
    compile_predicate_to_sql(all_employees/4, [], SQL1),
    format('   ~w~n~n', [SQL1]),

    format('2. Engineering team:~n'),
    compile_predicate_to_sql(engineering_team/2, [], SQL2),
    format('   ~w~n~n', [SQL2]),

    format('3. High earners:~n'),
    compile_predicate_to_sql(high_earners/2, [], SQL3),
    format('   ~w~n~n', [SQL3]),

    format('4. As CREATE VIEW:~n'),
    compile_predicate_to_sql(high_earners/2, [format(view), view_name(top_earners)], SQL4),
    format('   ~w~n~n', [SQL4]).

:- initialization(test, main).
```

Run it:

```bash
swipl sql_intro.pl
```

Output:

```
=== SQL Target Introduction ===

1. All employees:
   SELECT id, name, dept, salary FROM employees;

2. Engineering team:
   SELECT name, salary FROM employees WHERE dept = 'Engineering';

3. High earners:
   SELECT name, salary FROM employees WHERE salary > 80000;

4. As CREATE VIEW:
   CREATE VIEW IF NOT EXISTS top_earners AS SELECT name, salary FROM employees WHERE salary > 80000;
```

## Exercises

1. **Basic Query**: Create a predicate `employee_depts(Dept)` that returns all unique departments. Compile it to SQL.

2. **Filtered Query**: Create a predicate `recent_hires(Name, HireDate)` that returns employees hired after '2020-01-01'.

3. **Multiple Columns**: Create a predicate `employee_details(Id, Name, Dept)` that returns id, name, and department for all employees.

4. **CREATE VIEW**: Generate a CREATE VIEW statement for employees in the 'Sales' department.

## Summary

In this chapter, you learned:

- The SQL target generates declarative SQL queries (not executable code)
- Use `sql_table/2` to declare table schemas
- Use `compile_predicate_to_sql/3` to compile predicates to SQL
- Prolog constraints become WHERE clauses
- Options control output format (SELECT vs CREATE VIEW)
- Use `write_sql_file/2` to save generated SQL

## Next Chapter

In Chapter 2, we'll explore more complex queries including:
- Multiple comparison operators
- ORDER BY, LIMIT, and OFFSET
- DISTINCT queries
- Column aliasing with `sql_as/2`
