# SQL Target Design Document

## Overview

The SQL target compiles Prolog predicates to SQL queries, enabling UnifyWeaver to work with any SQL database (PostgreSQL, MySQL, SQLite, SQL Server, etc.). Unlike embedded database targets (bbolt, redb), the SQL target generates declarative queries rather than imperative code.

## Motivation

### Why SQL Target?

1. **Leverage Existing Infrastructure**: Every organization has SQL databases with massive datasets
2. **Natural Translation**: Prolog and SQL are both declarative languages
3. **Database Portability**: Single Prolog source → works with any SQL database
4. **Immediate Value**: Reuses Phase 9 aggregation work (GROUP BY, HAVING, etc.)
5. **Enterprise Use Cases**: Analytics, reporting, ETL, data migration

### Current vs SQL Approach

**Embedded Targets (Go/Rust + bbolt/redb)**:
- Generate application code with database API calls
- Embedded, single-process, file-based
- Procedural iteration over records

**SQL Target**:
- Generate SQL queries
- Client-server or embedded (SQLite)
- Declarative set operations
- Database engine handles optimization

## Design Philosophy

### Core Principle: Direct SQL Generation

Unlike other targets that generate a full program, the SQL target generates **pure SQL queries** that can be:
1. Executed directly via `sqlite3`, `psql`, `mysql` CLI
2. Embedded in applications
3. Used in BI tools, notebooks, reporting systems
4. Composed with existing SQL workflows

### Compilation Output

```prolog
% Input: user.pl
adult(Name) :- person(Name, Age), Age >= 18.
city_counts(City, Count) :-
    group_by(City, person(Name, Age, City), count, Count).
```

```sql
-- Output: user.sql

-- adult/1
CREATE VIEW adult AS
SELECT name
FROM person
WHERE age >= 18;

-- city_counts/2
CREATE VIEW city_counts AS
SELECT city, COUNT(*) as count
FROM person
GROUP BY city;
```

## Architecture

### Three-Layer Compilation

```
┌─────────────────┐
│  Prolog Source  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SQL Target    │
│   Compiler      │
│                 │
│ • Schema map    │
│ • Query gen     │
│ • Optimization  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SQL Queries   │
│  (Views/CTEs)   │
└─────────────────┘
```

### Key Components

1. **Schema Mapper**: Maps Prolog predicates to SQL tables/columns
2. **Query Generator**: Translates Prolog rules to SQL SELECT
3. **Aggregation Translator**: Converts `group_by/3` to GROUP BY
4. **Constraint Compiler**: Translates Prolog constraints to WHERE/HAVING
5. **Optimizer**: Applies SQL optimization patterns

## Phase 1: Basic SELECT Queries

### Scope

- Simple facts → table lookups
- Simple rules → SELECT with WHERE
- Conjunctions → JOINs or WHERE clauses
- Basic constraints (comparison, arithmetic)

### Schema Declaration

Users declare how Prolog predicates map to SQL:

```prolog
% Schema declarations
:- sql_table(person, [name-text, age-integer, city-text]).
:- sql_table(department, [id-integer, name-text]).
:- sql_table(employee, [name-text, dept_id-integer, salary-real]).
```

**Alternative (auto-inference)**:
```prolog
% Compiler infers schema from predicate structure
person("Alice", 30, "NYC").
person("Bob", 25, "LA").

% → Creates table: person(arg1 text, arg2 integer, arg3 text)
% Better: use named fields
person(name: "Alice", age: 30, city: "NYC").
% → Creates table: person(name text, age integer, city text)
```

### Translation Examples

#### Example 1: Simple Filter

**Prolog:**
```prolog
adult(Name) :- person(Name, Age, _), Age >= 18.
```

**Generated SQL:**
```sql
CREATE VIEW adult AS
SELECT name
FROM person
WHERE age >= 18;
```

#### Example 2: Join

**Prolog:**
```prolog
employee_dept(EmpName, DeptName) :-
    employee(EmpName, DeptId, _),
    department(DeptId, DeptName).
```

**Generated SQL:**
```sql
CREATE VIEW employee_dept AS
SELECT e.name AS emp_name, d.name AS dept_name
FROM employee e
JOIN department d ON e.dept_id = d.id;
```

#### Example 3: Multiple Constraints

**Prolog:**
```prolog
senior_employee(Name, Salary) :-
    employee(Name, _, Salary),
    Salary > 100000,
    Name \= "Admin".
```

**Generated SQL:**
```sql
CREATE VIEW senior_employee AS
SELECT name, salary
FROM employee
WHERE salary > 100000
  AND name != 'Admin';
```

## Phase 2: Aggregations (Leverage Phase 9 Work!)

### Direct Mapping

Phase 9c aggregations map directly to SQL:

#### Multiple Aggregations (Phase 9c-1)

**Prolog:**
```prolog
city_stats(City, Count, AvgAge, MaxAge) :-
    group_by(City,
             person(_, Age, City),
             [count(Count), avg(Age, AvgAge), max(Age, MaxAge)]).
```

**Generated SQL:**
```sql
CREATE VIEW city_stats AS
SELECT
    city,
    COUNT(*) as count,
    AVG(age) as avg_age,
    MAX(age) as max_age
FROM person
GROUP BY city;
```

#### HAVING Clause (Phase 9c-2)

**Prolog:**
```prolog
large_cities(City, Count) :-
    group_by(City, person(_, _, City), count, Count),
    Count > 100.
```

**Generated SQL:**
```sql
CREATE VIEW large_cities AS
SELECT city, COUNT(*) as count
FROM person
GROUP BY city
HAVING COUNT(*) > 100;
```

#### Nested Grouping (Phase 9c-3)

**Prolog:**
```prolog
state_city_counts(State, City, Count) :-
    group_by([State, City],
             location(_, State, City),
             count, Count).
```

**Generated SQL:**
```sql
CREATE VIEW state_city_counts AS
SELECT state, city, COUNT(*) as count
FROM location
GROUP BY state, city;
```

### Aggregation Translation Table

| Prolog Operation | SQL Equivalent |
|-----------------|----------------|
| `count` | `COUNT(*)` |
| `count(Var)` | `COUNT(column)` |
| `sum(Var, Result)` | `SUM(column)` |
| `avg(Var, Result)` | `AVG(column)` |
| `max(Var, Result)` | `MAX(column)` |
| `min(Var, Result)` | `MIN(column)` |

## Phase 3: Advanced Features

### Subqueries and CTEs

**Prolog:**
```prolog
above_avg_salary(Name, Salary) :-
    employee(Name, _, Salary),
    avg_salary(AvgSal),
    Salary > AvgSal.

avg_salary(Avg) :-
    group_by(all, employee(_, _, Salary), avg(Salary, Avg)).
```

**Generated SQL (WITH CTE):**
```sql
WITH avg_salary AS (
    SELECT AVG(salary) as avg_sal
    FROM employee
)
SELECT e.name, e.salary
FROM employee e
CROSS JOIN avg_salary a
WHERE e.salary > a.avg_sal;
```

### Window Functions

**Prolog (new syntax):**
```prolog
employee_rank(Name, Dept, Salary, Rank) :-
    employee(Name, Dept, Salary),
    rank_over([partition(Dept), order_by(Salary, desc)], Rank).
```

**Generated SQL:**
```sql
CREATE VIEW employee_rank AS
SELECT
    name,
    dept_id,
    salary,
    RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) as rank
FROM employee;
```

### String Operations

**Prolog:**
```prolog
gmail_users(Name) :-
    person(Name, Email),
    string_contains(Email, "@gmail.com").
```

**Generated SQL:**
```sql
CREATE VIEW gmail_users AS
SELECT name
FROM person
WHERE email LIKE '%@gmail.com%';
```

## Implementation Plan

### Phase 1: Foundation (Week 1-2)

**Tasks:**
1. Create `src/unifyweaver/targets/sql_target.pl`
2. Implement schema declaration parsing
3. Basic SELECT query generation
4. Simple WHERE clause translation
5. Test with SQLite

**Deliverables:**
- Compiler generates basic SQL views
- Test suite with 10+ basic queries
- SQLite integration working

### Phase 2: Aggregations (Week 3)

**Tasks:**
1. Integrate with Phase 9 `group_by` infrastructure
2. Generate GROUP BY clauses
3. Generate HAVING clauses
4. Support nested grouping
5. Test with all Phase 9 test cases

**Deliverables:**
- All Phase 9 Prolog examples work in SQL
- Test suite: 13 aggregation tests from Phase 9c
- Performance comparison vs Go target

### Phase 3: JOINs (Week 4)

**Tasks:**
1. Detect join patterns in Prolog rules
2. Generate INNER/LEFT/RIGHT JOIN
3. Optimize join order
4. Handle multiple joins
5. Test with multi-table schemas

**Deliverables:**
- Multi-table query support
- JOIN optimization
- Test suite with 15+ join scenarios

### Phase 4: Advanced (Week 5+)

**Tasks:**
1. Subqueries and CTEs
2. UNION/INTERSECT/EXCEPT
3. Window functions (if time)
4. String/date functions
5. Database-specific optimizations

## Testing Strategy

### Test Databases

1. **SQLite** - Primary testing (embedded, easy setup)
2. **PostgreSQL** - Full-featured reference
3. **MySQL** - Compatibility testing

### Test Categories

**1. Correctness Tests**
- Compare SQL results with Go target results
- Use same test data for both targets
- Verify identical output

**2. Schema Tests**
- Auto-inference from facts
- Explicit schema declarations
- Type mapping correctness

**3. Aggregation Tests**
- Reuse all 13 Phase 9c tests
- Verify GROUP BY, HAVING, nested grouping
- Compare performance with Go target

**4. JOIN Tests**
- 2-table joins
- 3+ table joins
- Self-joins
- Outer joins

**5. Edge Cases**
- Empty results
- NULL handling
- Division by zero
- String escaping

### Test Harness

```bash
# Generate SQL from Prolog
swipl -q -t "compile_to_sql('test.pl', 'test.sql'), halt"

# Load schema and data
sqlite3 test.db < schema.sql
sqlite3 test.db < data.sql

# Load generated views
sqlite3 test.db < test.sql

# Execute queries and compare
sqlite3 test.db "SELECT * FROM adult;" > sql_output.txt
./adult_go > go_output.txt
diff sql_output.txt go_output.txt
```

## Schema Mapping Options

### Option 1: Explicit Declarations (Recommended for Phase 1)

```prolog
:- sql_table(person, [
    name-text,
    age-integer,
    city-text
]).

person(Name, Age, City) :- sql_row(person, [Name, Age, City]).
```

**Pros:** Clear, explicit, no ambiguity
**Cons:** Requires boilerplate

### Option 2: Convention-Based

```prolog
% Compiler assumes predicate name = table name
% Argument positions = column positions
person(Name, Age, City).

% → SELECT * FROM person
```

**Pros:** Minimal syntax
**Cons:** No type information, positional coupling

### Option 3: Inline Metadata

```prolog
person(name:text(Name), age:int(Age), city:text(City)).
```

**Pros:** Self-documenting
**Cons:** Verbose

**Decision:** Start with Option 1, add Option 2 later for convenience.

## Optimization Opportunities

### 1. Predicate Pushdown

Push constraints as deep as possible:

```sql
-- Bad
SELECT * FROM (SELECT * FROM person) WHERE age > 18;

-- Good
SELECT * FROM person WHERE age > 18;
```

### 2. Join Ordering

Optimize join order based on selectivity:

```sql
-- If dept has 10 rows, employee has 1M rows
-- Bad: Large table first
SELECT * FROM employee e JOIN department d ...

-- Good: Small table first
SELECT * FROM department d JOIN employee e ...
```

### 3. Index Hints

Allow user to provide index hints:

```prolog
:- sql_index(person, [city]).
```

Generates:
```sql
CREATE INDEX idx_person_city ON person(city);
```

## SQL Dialect Support

### Standard SQL (Phase 1)

Target SQL-92/SQL-99 for maximum compatibility.

### Dialect-Specific (Phase 2+)

Support database-specific features:

**PostgreSQL:**
- Array types
- JSON operators
- Window functions
- CTEs with RECURSIVE

**MySQL:**
- LIMIT syntax
- String functions

**SQLite:**
- No RIGHT JOIN (use LEFT JOIN with reversed tables)
- Limited ALTER TABLE

**Approach:** Generate standard SQL by default, add `sql_dialect(postgres)` option for extensions.

## File Organization

```
src/unifyweaver/targets/
├── sql_target.pl           # Main entry point
├── sql/
│   ├── schema_mapper.pl    # Schema declarations
│   ├── query_gen.pl        # SELECT generation
│   ├── aggregation.pl      # GROUP BY/HAVING
│   ├── join_optimizer.pl   # JOIN ordering
│   └── dialect.pl          # Database-specific code
```

## Success Criteria

**Phase 1 Complete:**
- ✅ Generate SQL views for 20+ Prolog predicates
- ✅ SQLite integration working
- ✅ Test suite with 30+ tests passing

**Phase 2 Complete:**
- ✅ All Phase 9c aggregation tests working in SQL
- ✅ Performance within 2x of Go target
- ✅ PostgreSQL and MySQL compatibility

**Phase 3 Complete:**
- ✅ Multi-table joins working
- ✅ 50+ test cases passing
- ✅ Documentation with examples

## Example End-to-End Workflow

**1. Write Prolog:**
```prolog
% analytics.pl
:- sql_table(sales, [product-text, amount-real, date-text]).

high_value_products(Product, TotalSales) :-
    group_by(Product,
             sales(Product, Amount, _),
             sum(Amount, TotalSales)),
    TotalSales > 10000.
```

**2. Compile:**
```bash
swipl -q -t "compile_to_sql('analytics.pl', 'analytics.sql'), halt"
```

**3. Generated SQL:**
```sql
-- analytics.sql
CREATE VIEW high_value_products AS
SELECT product, SUM(amount) as total_sales
FROM sales
GROUP BY product
HAVING SUM(amount) > 10000;
```

**4. Execute:**
```bash
# PostgreSQL
psql mydb -f analytics.sql
psql mydb -c "SELECT * FROM high_value_products;"

# SQLite
sqlite3 mydb.db < analytics.sql
sqlite3 mydb.db "SELECT * FROM high_value_products;"
```

## Future Extensions

### View Materialization

```prolog
:- sql_materialized_view(expensive_query).
```

Generates:
```sql
CREATE MATERIALIZED VIEW expensive_query AS ...
```

### Query Parameterization

```prolog
adults_in_city(City, Name) :-
    person(Name, Age, City),
    Age >= 18.
```

Generates prepared statement:
```sql
PREPARE adults_in_city (text) AS
SELECT name FROM person
WHERE city = $1 AND age >= 18;
```

### Incremental Updates

Generate INSERT/UPDATE/DELETE from Prolog assertions:

```prolog
:- assert(person("Charlie", 35, "Boston")).
```

Generates:
```sql
INSERT INTO person (name, age, city)
VALUES ('Charlie', 35, 'Boston');
```

## Open Questions

1. **Transaction Support**: How to handle Prolog backtracking in SQL context?
2. **Recursive Queries**: Use WITH RECURSIVE or limit to non-recursive?
3. **NULL Handling**: Map to Prolog `_` or special `null` value?
4. **Output Format**: Views, CTEs, or standalone SELECT statements?

## References

- SQL-92 Standard: https://www.contrib.andrew.cmu.edu/~shadow/sql/sql1992.txt
- PostgreSQL Docs: https://www.postgresql.org/docs/
- SQLite Docs: https://www.sqlite.org/docs.html
- Datalog-to-SQL: https://github.com/frankmcsherry/blog/blob/master/posts/2016-06-21.md
