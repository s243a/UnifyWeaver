# Skill: SQL Target

Generate SQL queries, views, and CTEs from Prolog predicates. Supports SQLite, PostgreSQL, and MySQL.

## When to Use

- User asks "how do I generate SQL from Prolog?"
- User wants to create database views from predicates
- User asks about GROUP BY, HAVING, or aggregation in SQL
- User needs recursive CTEs for transitive closure
- User wants to export Prolog facts to a database
- User asks about window functions in SQL

## Quick Start

### Basic Compilation

```prolog
:- use_module('src/unifyweaver/targets/sql_target').

% Define table schema
:- sql_table(employee, [name-text, dept-text, salary-integer]).

% Define predicate
high_earner(Name, Dept) :-
    employee(Name, Dept, Salary),
    Salary > 100000.

% Compile to SQL
?- compile_predicate_to_sql(high_earner/2, [dialect(sqlite)], SQL).
```

**Output:**
```sql
CREATE VIEW IF NOT EXISTS high_earner AS
SELECT employee.name, employee.dept
FROM employee
WHERE employee.salary > 100000;
```

## Database Dialects

| Dialect | Option | Notes |
|---------|--------|-------|
| SQLite | `dialect(sqlite)` | Default, widely compatible |
| PostgreSQL | `dialect(postgres)` | Full CTE support, window functions |
| MySQL | `dialect(mysql)` | Some CTE limitations in older versions |

```prolog
% PostgreSQL output
compile_predicate_to_sql(my_pred/2, [dialect(postgres)], SQL).

% MySQL output
compile_predicate_to_sql(my_pred/2, [dialect(mysql)], SQL).
```

## Output Formats

| Format | Option | Description |
|--------|--------|-------------|
| View | `format(view)` | `CREATE VIEW` statement (default) |
| CTE | `format(cte)` | Common Table Expression |
| Select | `format(select)` | Standalone SELECT query |

```prolog
% Generate just a SELECT statement
compile_predicate_to_sql(my_pred/2, [format(select)], SQL).
```

## Aggregation: GROUP BY

```prolog
% Count employees per department
dept_count(Dept, Count) :-
    group_by(Dept, employee(_, Dept, _), count, Count).

% Compiles to:
% SELECT dept, COUNT(*) AS count
% FROM employee
% GROUP BY dept
```

### Aggregation Operators

| Prolog | SQL |
|--------|-----|
| `count` | `COUNT(*)` |
| `sum` | `SUM(column)` |
| `avg` | `AVG(column)` |
| `min` | `MIN(column)` |
| `max` | `MAX(column)` |

### HAVING Clause

```prolog
% Departments with more than 10 employees
large_depts(Dept, Count) :-
    group_by(Dept, employee(_, Dept, _), count, Count),
    Count >= 10.

% Compiles to:
% SELECT dept, COUNT(*) AS count
% FROM employee
% GROUP BY dept
% HAVING COUNT(*) >= 10
```

## Window Functions

Window functions compute values across rows related to the current row.

```prolog
% Rank employees by salary within department
ranked_employees(Name, Dept, Salary, Rank) :-
    employee(Name, Dept, Salary),
    rank(Rank, [partition_by(Dept), order_by(Salary, desc)]).

% Compiles to:
% SELECT name, dept, salary,
%        RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rank
% FROM employee
```

### Available Window Functions

| Prolog | SQL | Description |
|--------|-----|-------------|
| `row_number(R, Opts)` | `ROW_NUMBER()` | Sequential row number |
| `rank(R, Opts)` | `RANK()` | Rank with gaps |
| `dense_rank(R, Opts)` | `DENSE_RANK()` | Rank without gaps |
| `ntile(N, R, Opts)` | `NTILE(N)` | Divide into N buckets |
| `window_sum(F, R, Opts)` | `SUM() OVER` | Running sum |
| `window_avg(F, R, Opts)` | `AVG() OVER` | Running average |
| `lag(F, Offset, R, Opts)` | `LAG()` | Previous row value |
| `lead(F, Offset, R, Opts)` | `LEAD()` | Next row value |

### Window Options

```prolog
[
    partition_by(Column),           % PARTITION BY
    order_by(Column, asc|desc),     % ORDER BY
    frame(rows, Start, End)         % Window frame
]
```

## Recursive CTEs

For transitive closure (ancestor, reachability, etc.):

```prolog
% Ancestor relationship
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Compile with recursive CTE
compile_recursive_cte(
    ancestor,
    [x, y],
    ancestor/2,
    ancestor/2,
    SQL
).
```

**Output:**
```sql
WITH RECURSIVE ancestor(x, y) AS (
    SELECT parent.x, parent.y FROM parent
    UNION
    SELECT parent.x, ancestor.y
    FROM parent
    JOIN ancestor ON parent.y = ancestor.x
)
SELECT * FROM ancestor;
```

## Set Operations

Combine multiple predicates with UNION, INTERSECT, EXCEPT:

```prolog
% Union of two predicates
compile_set_operation(
    union,
    [active_users/1, premium_users/1],
    [dialect(postgres)],
    SQL
).
```

| Operation | SQL |
|-----------|-----|
| `union` | `UNION` (distinct) |
| `union_all` | `UNION ALL` |
| `intersect` | `INTERSECT` |
| `except` | `EXCEPT` |

## Exporting Facts

Export Prolog facts as SQL INSERT statements:

```prolog
% Define facts
parent(alice, bob).
parent(bob, carol).
parent(carol, dave).

% Export to SQL
compile_facts_to_sql(parent, 2, SQL).
```

**Output:**
```sql
CREATE TABLE IF NOT EXISTS parent (arg1 TEXT, arg2 TEXT);
INSERT INTO parent VALUES ('alice', 'bob');
INSERT INTO parent VALUES ('bob', 'carol');
INSERT INTO parent VALUES ('carol', 'dave');
```

## Writing to File

```prolog
% Compile and write to file
compile_predicate_to_sql(my_view/2, [dialect(postgres)], SQL),
write_sql_file(SQL, 'output/my_view.sql').
```

## Schema Declarations

Define table schemas for proper column resolution:

```prolog
% Declare table schema
:- sql_table(orders, [
    id-integer,
    customer_id-integer,
    product-text,
    quantity-integer,
    price-real
]).

% Now columns are properly resolved
order_totals(CustomerId, Total) :-
    group_by(CustomerId, orders(_, CustomerId, _, Qty, Price), sum, Total),
    % Aggregates Qty * Price
```

## Commands

### Generate View
```bash
swipl -g "use_module('src/unifyweaver/targets/sql_target'), \
          compile_predicate_to_sql(my_pred/2, [dialect(sqlite)], SQL), \
          write(SQL)" -t halt my_predicates.pl
```

### Export Facts to SQL
```bash
swipl -g "use_module('src/unifyweaver/targets/sql_target'), \
          compile_facts_to_sql(my_facts, 3, SQL), \
          write_sql_file(SQL, 'output.sql')" -t halt facts.pl
```

## Related

**Parent Skill:**
- `skill_query_patterns.md` - Query patterns sub-master

**Sibling Skills:**
- `skill_stream_aggregation.md` - Runtime aggregation in Go/Python
- `skill_aggregation_patterns.md` - Aggregation overview
- `skill_fuzzy_search.md` - Fuzzy score blending

**Other Skills:**
- `skill_unifyweaver_compile.md` - Basic compilation

**Documentation:**
- `education/book-10-sql-target/` - SQL target tutorial

**Code:**
- `src/unifyweaver/targets/sql_target.pl` - SQL compilation implementation
- `src/unifyweaver/sources/sqlite_source.pl` - SQLite data source
