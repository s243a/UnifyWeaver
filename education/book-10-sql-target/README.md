<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 10: SQL Target

**From Prolog to SQL: Declarative Query Generation**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers UnifyWeaver's SQL target, enabling you to compile Prolog predicates directly to SQL queries for execution on relational databases.

## Prerequisites

**Required:**
- [Book 1: Foundations](../book-01-foundations/README.md)

**Recommended:**
- [Book 9: Rust Target](../book-09-rust-target/README.md) - first book in the Specialized Targets section

**Technical:**
- Basic SQL knowledge (SELECT, JOIN, WHERE, GROUP BY)
- Access to SQLite or another SQL database (for testing generated queries)

## What You'll Learn

- How to declare SQL table schemas in Prolog
- Compiling predicates to SELECT statements
- All types of JOINs (INNER, LEFT, RIGHT, FULL OUTER)
- Aggregations with GROUP BY and HAVING
- Subqueries (IN, EXISTS, NOT IN, NOT EXISTS)
- Window functions (RANK, ROW_NUMBER, LAG, LEAD)
- Common Table Expressions (CTEs)
- Recursive CTEs for hierarchical data
- Set operations (UNION, INTERSECT, EXCEPT)
- SQL functions (string, date, NULL handling)

## SQL Target Overview

The SQL target is unique among UnifyWeaver's compilation targets:

| Target | Output | Execution |
|--------|--------|-----------|
| Bash | Shell scripts | Run directly |
| C# | .NET source | Compile & run |
| Go | Go source | Compile & run |
| **SQL** | SQL queries | Run on database |

Unlike other targets that produce executable programs, the SQL target generates declarative queries meant for external database execution.

## Book Structure

### Chapter 1: Getting Started with SQL Target
- Loading the SQL target module
- Declaring table schemas with `sql_table/2`
- Your first predicate-to-SQL compilation
- Running generated SQL on SQLite

### Chapter 2: Basic Queries
- SELECT with column projection
- WHERE clauses from Prolog constraints
- Comparison operators (=, <, >, <=, >=, \=)
- ORDER BY, LIMIT, OFFSET
- DISTINCT queries

### Chapter 3: JOINs
- Implicit INNER JOINs via shared variables
- Explicit `sql_left_join/1`
- RIGHT and FULL OUTER JOINs
- Multi-table nested JOINs
- Self-joins

### Chapter 4: Aggregations
- Aggregate functions (COUNT, SUM, AVG, MIN, MAX)
- GROUP BY with `sql_group_by/1`
- HAVING clauses with `sql_having/1`
- Combining aggregations with other features

### Chapter 5: Subqueries
- IN subqueries with `sql_in_subquery/2`
- NOT IN subqueries
- EXISTS with `sql_exists/1`
- NOT EXISTS
- Correlated subqueries

### Chapter 6: Window Functions
- ROW_NUMBER, RANK, DENSE_RANK
- LAG and LEAD for row comparisons
- FIRST_VALUE and LAST_VALUE
- PARTITION BY and ORDER BY in windows
- Window frame specifications (ROWS/RANGE BETWEEN)

### Chapter 7: Common Table Expressions
- WITH clause using `compile_with_cte/4`
- Simplifying complex queries
- Multiple CTEs
- CTE vs subquery trade-offs

### Chapter 8: Recursive CTEs
- Understanding hierarchical data
- `sql_recursive_table/2` declarations
- Base case and recursive case predicates
- `compile_recursive_cte/5` usage
- Use cases: org charts, category trees, graph traversal
- UNION vs UNION ALL in recursion

### Chapter 9: Set Operations
- UNION and UNION ALL
- INTERSECT and INTERSECT ALL
- EXCEPT and EXCEPT ALL
- Combining multiple queries

### Chapter 10: SQL Functions
- NULL handling (COALESCE, NULLIF, IFNULL, IS NULL)
- String functions (CONCAT, UPPER, LOWER, SUBSTRING, TRIM)
- Date functions (DATE, DATE_ADD, DATE_DIFF, EXTRACT)
- WHERE predicates (BETWEEN, LIKE, GLOB, IN list)

### Chapter 11: CASE WHEN Expressions
- Conditional logic in SELECT
- Simple CASE expressions
- Searched CASE expressions
- Combining with other features

### Chapter 12: Practical Applications
- Building a reporting system
- Data warehouse queries
- ETL query generation
- Database migration scripts

## Quick Start Example

```prolog
% Load the SQL target
:- use_module('src/unifyweaver/targets/sql_target').

% Declare table schema
:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer]).

% Define a predicate
high_earners(Name, Salary) :-
    employees(_, Name, _, Salary),
    Salary > 100000.

% Compile to SQL
?- compile_predicate_to_sql(high_earners/2, [], SQL).
% SQL = 'SELECT name, salary FROM employees WHERE salary > 100000;'
```

Run the generated SQL on SQLite:
```bash
sqlite3 mydb.db "SELECT name, salary FROM employees WHERE salary > 100000;"
```

## Feature Summary

| Feature | Prolog Syntax | SQL Output |
|---------|---------------|------------|
| Basic SELECT | `table(A, B, C)` | `SELECT a, b, c FROM table` |
| WHERE | `Salary > 50000` | `WHERE salary > 50000` |
| INNER JOIN | Shared variables | `INNER JOIN ... ON ...` |
| LEFT JOIN | `sql_left_join(...)` | `LEFT JOIN ... ON ...` |
| GROUP BY | `sql_group_by([Dept])` | `GROUP BY dept` |
| Aggregate | `sql_count(*)` | `COUNT(*)` |
| Window | `sql_window(rank, ...)` | `RANK() OVER (...)` |
| CTE | `compile_with_cte(...)` | `WITH ... AS (...)` |
| Recursive | `compile_recursive_cte(...)` | `WITH RECURSIVE ...` |
| UNION | `compile_set_operation(union, ...)` | `... UNION ...` |
| CASE | `sql_case([when(...), ...], Default)` | `CASE WHEN ... END` |

## Database Compatibility

Generated SQL is primarily SQLite-compatible but works with:
- SQLite (full support)
- PostgreSQL (minor syntax adjustments)
- MySQL/MariaDB (minor syntax adjustments)
- SQL Server (ANSI mode)

## Example Projects

Throughout this book, you'll build:
- Employee directory with department lookups
- Sales report with aggregations and rankings
- Org chart with recursive hierarchy
- Product catalog with category trees
- Order tracking with complex joins

## When to Use SQL Target

**Use SQL Target when:**
- Integrating with existing relational databases
- Building reporting and analytics queries
- Need database-level query optimization
- Working with large datasets that benefit from database indexes
- Creating database views for BI tools

**Use other targets when:**
- Need standalone executable programs
- Working with non-relational data
- Building real-time streaming applications
- No database infrastructure available

## Testing Generated SQL

```bash
# Create test database
sqlite3 test.db < schema.sql

# Run generated query
sqlite3 test.db "$(cat generated.sql)"

# Or use the write_sql_file/2 predicate
?- compile_predicate_to_sql(my_query/2, [], SQL),
   write_sql_file('output/my_query.sql', SQL).
```

## Learning Path

1. **Start with Chapters 1-2** - Basic query generation
2. **Work through Chapters 3-4** - JOINs and aggregations
3. **Advance to Chapters 5-6** - Subqueries and window functions
4. **Master Chapters 7-8** - CTEs and recursive queries
5. **Complete Chapters 9-11** - Set operations, functions, CASE WHEN
6. **Apply in Chapter 12** - Real-world applications

## Additional Resources

- Main documentation: `docs/targets/sql.md`
- Test suites: `test_sql_*.pl` files in project root
- SQL target source: `src/unifyweaver/targets/sql_target.pl`
- PR documentation: `PR_SQL_*.md` files

## Going Further

After completing this book, you'll be able to:
- Compile any Prolog predicate to SQL
- Generate complex analytical queries
- Build hierarchical data queries with recursive CTEs
- Integrate UnifyWeaver with database workflows
- Create reusable SQL views from Prolog specifications

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.

## Feedback

Found an issue or have suggestions?
- GitHub Issues: https://github.com/s243a/UnifyWeaver/issues
- Discussions: https://github.com/s243a/UnifyWeaver/discussions
