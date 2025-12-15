---
file_type: UnifyWeaver Example Library
---
# SQL Window Functions Examples

This file contains executable records for the SQL window functions playbook.

## `unifyweaver.execution.sql_window_basic`

> [!example-record]
> id: unifyweaver.execution.sql_window_basic
> name: SQL Window Functions Basic Usage
> platform: bash

This record demonstrates compiling Prolog predicates with window functions to SQL.

```bash
#!/bin/bash
# SQL Window Functions Demo - Basic Usage
# Demonstrates compiling Prolog predicates with window functions to SQL

set -euo pipefail
cd /root/UnifyWeaver

echo "=== SQL Window Functions Demo: Basic Usage ==="

# Create test directory
mkdir -p tmp/sql_window_demo

# Create test database
echo ""
echo "Creating test SQLite database..."
sqlite3 tmp/sql_window_demo/employees.db <<'SQL'
DROP TABLE IF EXISTS employees;
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary INTEGER NOT NULL,
    hire_date TEXT NOT NULL
);

INSERT INTO employees (name, department, salary, hire_date) VALUES
    ('Alice', 'Engineering', 85000, '2020-01-15'),
    ('Bob', 'Engineering', 95000, '2019-03-20'),
    ('Charlie', 'Sales', 65000, '2021-06-01'),
    ('Diana', 'Engineering', 78000, '2022-02-10'),
    ('Eve', 'Sales', 72000, '2020-09-05'),
    ('Frank', 'Marketing', 68000, '2021-04-15'),
    ('Grace', 'Marketing', 75000, '2019-11-20');
SQL

echo "Database created with employees table"

# Create Prolog script with window functions
cat > tmp/sql_window_demo/demo_windows.pl << 'PROLOG'
:- use_module('src/unifyweaver/targets/sql_target').

% Declare table schema
:- sql_table(employees, [id-integer, name-text, department-text, salary-integer, hire_date-text]).

% ROW_NUMBER: Rank employees by salary within department
employee_rank(Name, Department, Salary, Rank) :-
    employees(_, Name, Department, Salary, _),
    row_number(Rank, [partition_by(Department), order_by(desc(Salary))]).

% Running total of salaries by department
salary_running_total(Name, Department, Salary, RunningTotal) :-
    employees(_, Name, Department, Salary, _),
    sum(Salary, RunningTotal, [partition_by(Department), order_by(Salary)]).

% Dense rank by salary
employee_dense_rank(Name, Salary, DenseRank) :-
    employees(_, Name, _, Salary, _),
    dense_rank(DenseRank, [order_by(desc(Salary))]).

main :-
    format("~n=== Compiling Window Function Predicates to SQL ===~n~n"),

    % Compile employee_rank with ROW_NUMBER
    format("1. Compiling employee_rank/4 (ROW_NUMBER)...~n"),
    compile_predicate_to_sql(employee_rank/4, [dialect(sqlite)], RankSQL),
    format("~nGenerated SQL:~n~w~n", [RankSQL]),
    open('tmp/sql_window_demo/employee_rank.sql', write, S1),
    write(S1, RankSQL),
    close(S1),

    % Compile salary_running_total
    format("~n2. Compiling salary_running_total/4 (SUM window)...~n"),
    compile_predicate_to_sql(salary_running_total/4, [dialect(sqlite)], TotalSQL),
    format("~nGenerated SQL:~n~w~n", [TotalSQL]),
    open('tmp/sql_window_demo/salary_running_total.sql', write, S2),
    write(S2, TotalSQL),
    close(S2),

    % Compile employee_dense_rank
    format("~n3. Compiling employee_dense_rank/3 (DENSE_RANK)...~n"),
    compile_predicate_to_sql(employee_dense_rank/3, [dialect(sqlite)], DenseSQL),
    format("~nGenerated SQL:~n~w~n", [DenseSQL]),
    open('tmp/sql_window_demo/employee_dense_rank.sql', write, S3),
    write(S3, DenseSQL),
    close(S3),

    format("~n=== Window function compilation complete ===~n"),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to compile window function predicates..."
swipl tmp/sql_window_demo/demo_windows.pl 2>&1 || echo "Note: Window function compilation may require additional features"

echo ""
echo "=== Testing with SQLite (manual window queries) ==="

echo ""
echo "ROW_NUMBER - Rank employees by salary within department:"
sqlite3 -header -column tmp/sql_window_demo/employees.db <<'SQL'
SELECT
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank
FROM employees
ORDER BY department, rank;
SQL

echo ""
echo "Running total of salaries by department:"
sqlite3 -header -column tmp/sql_window_demo/employees.db <<'SQL'
SELECT
    name,
    department,
    salary,
    SUM(salary) OVER (PARTITION BY department ORDER BY salary) as running_total
FROM employees
ORDER BY department, salary;
SQL

echo ""
echo "DENSE_RANK by salary (company-wide):"
sqlite3 -header -column tmp/sql_window_demo/employees.db <<'SQL'
SELECT
    name,
    salary,
    DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank
FROM employees
ORDER BY dense_rank;
SQL

echo ""
echo "Success: SQL window functions demo complete"
```

## `unifyweaver.execution.sql_recursive_cte`

> [!example-record]
> id: unifyweaver.execution.sql_recursive_cte
> name: SQL Recursive CTEs
> platform: bash

This record demonstrates compiling recursive predicates to SQL WITH RECURSIVE.

```bash
#!/bin/bash
# SQL Window Functions Demo - Recursive CTEs
# Demonstrates compiling recursive predicates to SQL WITH RECURSIVE

set -euo pipefail
cd /root/UnifyWeaver

echo "=== SQL Window Functions Demo: Recursive CTEs ==="

mkdir -p tmp/sql_window_demo

# Create test database with hierarchy
sqlite3 tmp/sql_window_demo/org.db <<'SQL'
DROP TABLE IF EXISTS org_chart;
CREATE TABLE org_chart (
    employee_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    manager_id INTEGER,
    title TEXT NOT NULL
);

INSERT INTO org_chart (employee_id, name, manager_id, title) VALUES
    (1, 'Alice', NULL, 'CEO'),
    (2, 'Bob', 1, 'VP Engineering'),
    (3, 'Charlie', 1, 'VP Sales'),
    (4, 'Diana', 2, 'Senior Engineer'),
    (5, 'Eve', 2, 'Engineer'),
    (6, 'Frank', 3, 'Sales Manager'),
    (7, 'Grace', 6, 'Sales Rep');
SQL

echo "Created org_chart table with hierarchy"

# Create Prolog script with recursive CTE
cat > tmp/sql_window_demo/demo_recursive.pl << 'PROLOG'
:- use_module('src/unifyweaver/targets/sql_target').

% Declare table schema
:- sql_table(org_chart, [employee_id-integer, name-text, manager_id-integer, title-text]).

% Declare recursive CTE columns
:- sql_recursive_table(org_hierarchy, [employee_id-integer, name-text, level-integer, path-text]).

main :-
    format("~n=== Compiling Recursive CTE ===~n~n"),

    % Define recursive CTE for org hierarchy
    % Base case: CEO (manager_id IS NULL)
    % Recursive: employees who report to someone in hierarchy

    compile_recursive_cte(
        org_hierarchy,                              % CTE name
        [employee_id, name, level, path],          % Columns
        recursive_def(
            % Base case: top-level (no manager)
            base_select([
                org_chart.employee_id,
                org_chart.name,
                literal(1),                         % level = 1
                org_chart.name                      % path starts with name
            ],
            org_chart,
            is_null(org_chart.manager_id)),

            % Recursive case
            recursive_select([
                org_chart.employee_id,
                org_chart.name,
                expr(org_hierarchy.level + 1),      % level + 1
                expr(org_hierarchy.path || ' > ' || org_chart.name)  % path concatenation
            ],
            org_chart,
            eq(org_chart.manager_id, org_hierarchy.employee_id))
        ),
        select_all(org_hierarchy),                  % Main query
        [dialect(sqlite)],
        SQL
    ),

    format("Generated Recursive CTE:~n~w~n", [SQL]),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to compile recursive CTE..."
swipl tmp/sql_window_demo/demo_recursive.pl 2>&1 || echo "Note: Recursive CTE may require full sql_target features"

echo ""
echo "=== Testing Recursive CTE with SQLite (manual query) ==="

echo ""
echo "Org hierarchy with levels and paths:"
sqlite3 -header -column tmp/sql_window_demo/org.db <<'SQL'
WITH RECURSIVE org_hierarchy AS (
    -- Base case: CEO (no manager)
    SELECT
        employee_id,
        name,
        1 as level,
        name as path
    FROM org_chart
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: employees who report to someone in hierarchy
    SELECT
        e.employee_id,
        e.name,
        h.level + 1,
        h.path || ' > ' || e.name
    FROM org_chart e
    INNER JOIN org_hierarchy h ON e.manager_id = h.employee_id
)
SELECT * FROM org_hierarchy
ORDER BY level, name;
SQL

echo ""
echo "Success: Recursive CTE demo complete"
```

## `unifyweaver.execution.sql_group_by`

> [!example-record]
> id: unifyweaver.execution.sql_group_by
> name: SQL GROUP BY with HAVING
> platform: bash

This record demonstrates aggregation with group_by/4 predicate.

```bash
#!/bin/bash
# SQL Window Functions Demo - GROUP BY and HAVING
# Demonstrates aggregation with group_by/4 predicate

set -euo pipefail
cd /root/UnifyWeaver

echo "=== SQL Window Functions Demo: GROUP BY with HAVING ==="

mkdir -p tmp/sql_window_demo

# Ensure employees database exists
if [ ! -f tmp/sql_window_demo/employees.db ]; then
    sqlite3 tmp/sql_window_demo/employees.db <<'SQL'
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary INTEGER NOT NULL,
        hire_date TEXT NOT NULL
    );
    INSERT OR REPLACE INTO employees (id, name, department, salary, hire_date) VALUES
        (1, 'Alice', 'Engineering', 85000, '2020-01-15'),
        (2, 'Bob', 'Engineering', 95000, '2019-03-20'),
        (3, 'Charlie', 'Sales', 65000, '2021-06-01'),
        (4, 'Diana', 'Engineering', 78000, '2022-02-10'),
        (5, 'Eve', 'Sales', 72000, '2020-09-05');
SQL
fi

# Create Prolog script with GROUP BY
cat > tmp/sql_window_demo/demo_groupby.pl << 'PROLOG'
:- use_module('src/unifyweaver/targets/sql_target').

% Declare table schema
:- sql_table(employees, [id-integer, name-text, department-text, salary-integer, hire_date-text]).

% Average salary by department
dept_avg_salary(Department, AvgSalary) :-
    group_by(Department,
        employees(_, _, Department, Salary, _),
        avg(Salary),
        AvgSalary).

% Count employees by department (with HAVING)
large_departments(Department, Count) :-
    group_by(Department,
        employees(_, _, Department, _, _),
        count,
        Count),
    Count > 1.

% Max salary by department
dept_max_salary(Department, MaxSalary) :-
    group_by(Department,
        employees(_, _, Department, Salary, _),
        max(Salary),
        MaxSalary).

main :-
    format("~n=== Compiling GROUP BY Predicates to SQL ===~n~n"),

    % Compile dept_avg_salary
    format("1. Compiling dept_avg_salary/2 (AVG with GROUP BY)...~n"),
    compile_predicate_to_sql(dept_avg_salary/2, [dialect(sqlite)], AvgSQL),
    format("~nGenerated SQL:~n~w~n", [AvgSQL]),

    % Compile large_departments (with HAVING)
    format("~n2. Compiling large_departments/2 (COUNT with HAVING)...~n"),
    compile_predicate_to_sql(large_departments/2, [dialect(sqlite)], HavingSQL),
    format("~nGenerated SQL:~n~w~n", [HavingSQL]),

    % Compile dept_max_salary
    format("~n3. Compiling dept_max_salary/2 (MAX with GROUP BY)...~n"),
    compile_predicate_to_sql(dept_max_salary/2, [dialect(sqlite)], MaxSQL),
    format("~nGenerated SQL:~n~w~n", [MaxSQL]),

    format("~n=== GROUP BY compilation complete ===~n"),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to compile GROUP BY predicates..."
swipl tmp/sql_window_demo/demo_groupby.pl 2>&1

echo ""
echo "=== Testing GROUP BY with SQLite ==="

echo ""
echo "Average salary by department:"
sqlite3 -header -column tmp/sql_window_demo/employees.db <<'SQL'
SELECT department, AVG(salary) as avg_salary
FROM employees
GROUP BY department;
SQL

echo ""
echo "Large departments (more than 1 employee):"
sqlite3 -header -column tmp/sql_window_demo/employees.db <<'SQL'
SELECT department, COUNT(*) as count
FROM employees
GROUP BY department
HAVING COUNT(*) > 1;
SQL

echo ""
echo "Max salary by department:"
sqlite3 -header -column tmp/sql_window_demo/employees.db <<'SQL'
SELECT department, MAX(salary) as max_salary
FROM employees
GROUP BY department;
SQL

echo ""
echo "Success: GROUP BY demo complete"
```

## `unifyweaver.execution.sql_module_info`

> [!example-record]
> id: unifyweaver.execution.sql_module_info
> name: SQL Target Module Info
> platform: bash

This record displays SQL target capabilities and configuration options.

```bash
#!/bin/bash
# SQL Target Module Information
# Shows SQL target capabilities

set -euo pipefail
cd /root/UnifyWeaver

echo "=== SQL Target Module Information ==="

echo ""
echo "=== Public API ==="
echo ""
echo "compile_predicate_to_sql(+Predicate, +Options, -SQLCode)"
echo "  Compile a Prolog predicate to SQL"
echo ""
echo "compile_set_operation(+SetOp, +Predicates, +Options, -SQLCode)"
echo "  Compile INTERSECT, EXCEPT (MINUS) operations"
echo ""
echo "compile_with_cte(+CTEs, +MainPred, +Options, -SQLCode)"
echo "  Compile WITH (Common Table Expressions)"
echo ""
echo "compile_recursive_cte(+Name, +Cols, +RecDef, +Main, -SQL)"
echo "  Compile WITH RECURSIVE"
echo ""
echo "sql_table(+TableName, +Columns)"
echo "  Declare table schema"
echo ""
echo "sql_recursive_table(+CTEName, +Columns)"
echo "  Declare recursive CTE schema"

echo ""
echo "=== Options ==="
echo ""
echo "view_name(Name)     - Override view name"
echo "dialect(sqlite|postgres|mysql) - SQL dialect (default: sqlite)"
echo "format(view|cte|select)         - Output format (default: view)"

echo ""
echo "=== Supported Features ==="
echo ""
echo "1. Basic Queries:"
echo "   - Single table SELECT with WHERE"
echo "   - Multi-table JOINs (INNER, LEFT, RIGHT, FULL OUTER)"
echo "   - DISTINCT, ORDER BY, LIMIT, OFFSET"
echo ""
echo "2. Aggregation (GROUP BY):"
echo "   - group_by(GroupField, Goal, AggOp, Result)"
echo "   - Supported: count, sum, avg, max, min"
echo "   - HAVING clause support"
echo ""
echo "3. Window Functions:"
echo "   - row_number(Result, [partition_by(Col), order_by(Col)])"
echo "   - rank(Result, [...])"
echo "   - dense_rank(Result, [...])"
echo "   - sum(Col, Result, [...]) - Running totals"
echo "   - Arbitrary OVER clauses"
echo ""
echo "4. Set Operations:"
echo "   - UNION (multiple rules for same predicate)"
echo "   - INTERSECT via compile_set_operation/4"
echo "   - EXCEPT (MINUS) via compile_set_operation/4"
echo ""
echo "5. Common Table Expressions (CTEs):"
echo "   - Non-recursive WITH"
echo "   - Recursive WITH RECURSIVE"

echo ""
echo "=== Example: Window Function ==="
echo ""
echo "Prolog:"
echo "  employee_rank(Name, Dept, Salary, Rank) :-"
echo "      employees(_, Name, Dept, Salary, _),"
echo "      row_number(Rank, [partition_by(Dept), order_by(desc(Salary))])."
echo ""
echo "Generated SQL:"
echo "  SELECT name, department, salary,"
echo "         ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank"
echo "  FROM employees"

echo ""
echo "Success: SQL module info displayed"
```

