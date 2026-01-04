# SQL Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: SQL code generation target testing

## Overview

This test plan covers the SQL target for UnifyWeaver, which generates SQL queries and recursive CTEs from Prolog predicates, supporting multiple database backends.

## Prerequisites

### System Requirements

- SQLite 3.35+ (for recursive CTE support)
- PostgreSQL 14+ (optional, for advanced features)
- MySQL 8.0+ (optional, for MySQL dialect)
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify SQLite (primary test database)
sqlite3 --version

# Verify PostgreSQL (optional)
psql --version

# Verify MySQL (optional)
mysql --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

These tests verify SQL code generation without executing queries.

#### 1.1 Basic Generator Tests

```bash
# Run SQL generator tests
swipl -g "use_module('tests/core/test_sql_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `select_generation` | Basic SELECT | Correct column selection |
| `where_clause` | WHERE predicates | Proper condition syntax |
| `join_generation` | Table joins | JOIN syntax correct |
| `aggregation` | GROUP BY/HAVING | Aggregate functions |
| `ordering` | ORDER BY/LIMIT | Sort and pagination |

#### 1.2 Recursive CTE Tests

```bash
swipl -g "use_module('tests/core/test_sql_recursive_cte'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `transitive_closure` | Ancestor query | WITH RECURSIVE syntax |
| `base_case` | CTE base | UNION ALL structure |
| `recursive_case` | CTE recursion | Self-referencing query |
| `termination` | Cycle prevention | Proper termination |

#### 1.3 Advanced SQL Features

```bash
swipl -g "use_module('tests/core/test_sql_advanced'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `window_functions` | OVER clauses | ROW_NUMBER, RANK, etc. |
| `subqueries` | Nested queries | Correlated subqueries |
| `case_when` | Conditional | CASE WHEN THEN ELSE |
| `left_join` | Outer joins | LEFT/RIGHT/FULL JOIN |
| `union_queries` | Set operations | UNION/INTERSECT/EXCEPT |

### 2. Integration Tests (Generation + Execution)

#### 2.1 SQLite Tests

```bash
./tests/integration/test_sql_sqlite.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `create_schema` | Table creation | DDL executes |
| `insert_facts` | Fact insertion | Data loaded |
| `query_execution` | Run generated SQL | Correct results |
| `recursive_query` | CTE execution | Transitive closure works |

#### 2.2 PostgreSQL Tests

```bash
# Requires PostgreSQL connection
PGHOST=localhost PGUSER=test ./tests/integration/test_sql_postgres.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `pg_connect` | Connection | Successfully connects |
| `pg_recursive` | Recursive CTE | PostgreSQL syntax |
| `pg_window` | Window functions | Full window support |
| `pg_json` | JSON operations | JSONB queries |

#### 2.3 MySQL Tests

```bash
# Requires MySQL connection
MYSQL_HOST=localhost MYSQL_USER=test ./tests/integration/test_sql_mysql.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `mysql_connect` | Connection | Successfully connects |
| `mysql_recursive` | Recursive CTE | MySQL 8.0+ syntax |
| `mysql_dialect` | Dialect differences | Backtick quoting |

### 3. Dialect Compatibility Tests

#### 3.1 Cross-Database Queries

```bash
./tests/integration/test_sql_dialects.sh
```

**Dialect Matrix**:
| Feature | SQLite | PostgreSQL | MySQL |
|---------|--------|------------|-------|
| Recursive CTE | 3.35+ | Yes | 8.0+ |
| Window functions | 3.25+ | Yes | 8.0+ |
| BOOLEAN type | INTEGER | Yes | TINYINT |
| String concat | \|\| | \|\| | CONCAT() |
| LIMIT syntax | LIMIT n | LIMIT n | LIMIT n |
| Identifier quote | " | " | ` |

### 4. Generated SQL Structure

#### 4.1 Fact Tables

```sql
-- Generated schema for facts
CREATE TABLE IF NOT EXISTS parent (
    arg1 TEXT NOT NULL,
    arg2 TEXT NOT NULL,
    PRIMARY KEY (arg1, arg2)
);

INSERT INTO parent (arg1, arg2) VALUES
    ('john', 'mary'),
    ('mary', 'susan');
```

#### 4.2 Recursive Queries

```sql
-- Generated recursive CTE for transitive closure
WITH RECURSIVE ancestor(x, y) AS (
    -- Base case: direct parent
    SELECT arg1, arg2 FROM parent
    UNION
    -- Recursive case: ancestor of ancestor
    SELECT a.x, p.arg2
    FROM ancestor a
    JOIN parent p ON a.y = p.arg1
)
SELECT DISTINCT x, y FROM ancestor;
```

### 5. Performance Tests

#### 5.1 Query Performance

```bash
./tests/perf/test_sql_performance.sh
```

**Benchmarks**:
| Test | Rows | Expected Time |
|------|------|---------------|
| Simple SELECT | 1K | < 10ms |
| JOIN query | 10K | < 100ms |
| Recursive CTE (depth 10) | 1K | < 500ms |
| Aggregation | 100K | < 1s |

#### 5.2 Index Optimization

```bash
# Test with EXPLAIN QUERY PLAN
sqlite3 :memory: < /tmp/generated.sql
```

**Optimization Checks**:
| Check | Expected |
|-------|----------|
| Index usage | Uses PRIMARY KEY |
| Full scan avoided | No SCAN TABLE |
| Join order | Optimal order |

### 6. SQL Injection Prevention

#### 6.1 Safety Tests

```bash
swipl -g "use_module('tests/core/test_sql_safety'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `quoted_strings` | String literals | Properly escaped |
| `identifier_quoting` | Table/column names | Quoted identifiers |
| `parameterized` | Value binding | Uses ? placeholders |

## Test Commands Reference

### Quick Smoke Test

```bash
# Generate SQL code
swipl -g "
    use_module('src/unifyweaver/targets/sql_target'),
    compile_to_sql(test_query, SQL),
    format('~w~n', [SQL])
" -t halt
```

### Execute Generated SQL

```bash
# Generate and execute in SQLite
swipl -g "compile_to_sql(ancestor_query, SQL), format('~w', [SQL])" -t halt | sqlite3 :memory:
```

### Full Test Suite

```bash
# Run all SQL tests
./tests/run_sql_tests.sh

# Or individually:
swipl -g "use_module('tests/core/test_sql_generator'), run_tests" -t halt
swipl -g "use_module('tests/core/test_sql_recursive_cte'), run_tests" -t halt
./tests/integration/test_sql_sqlite.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SQL_DIALECT` | Target database | `sqlite` |
| `SQLITE_PATH` | SQLite binary | `sqlite3` |
| `PGHOST` | PostgreSQL host | (none) |
| `PGUSER` | PostgreSQL user | (none) |
| `MYSQL_HOST` | MySQL host | (none) |
| `SKIP_SQL_EXECUTION` | Skip runtime tests | `0` |
| `KEEP_SQL_ARTIFACTS` | Preserve generated SQL | `0` |

## Known Issues

1. **SQLite CTE depth**: Default recursion limit is 1000
2. **MySQL < 8.0**: No recursive CTE support
3. **NULL handling**: Varies by dialect
4. **Date/time types**: Dialect-specific formatting

## Related Documentation

- [SQL Target Implementation](../../architecture/targets/sql_target.md)
- [Recursive CTE Generation](../../architecture/recursive_cte.md)
- [Database Dialects](../../architecture/sql_dialects.md)
- [Query Optimization](../../architecture/query_optimization.md)
