# Playbook: SQL Window Functions and Advanced Queries

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents orchestrate UnifyWeaver to compile Prolog predicates into SQL with window functions, CTEs, and aggregations.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "sql_window" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use sql window"


## Workflow Overview
Use UnifyWeaver's sql_target module to:
1. Compile predicates with window functions (ROW_NUMBER, RANK, etc.)
2. Generate GROUP BY with HAVING clauses
3. Create recursive CTEs for hierarchical data
4. Produce set operations (UNION, INTERSECT, EXCEPT)

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** - `playbooks/examples_library/sql_window_examples.md`
2. **Environment Setup Skill** - `skills/skill_unifyweaver_environment.md`
3. **Extraction Skill** - `skills/skill_extract_records.md`

## Execution Guidance

### Step 1: Navigate to project root
```bash
cd /root/UnifyWeaver
```

### Step 2: Extract the basic window functions demo
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.sql_window_basic" \
  playbooks/examples_library/sql_window_examples.md \
  > tmp/run_sql_window.sh
```

### Step 3: Make it executable and run
```bash
chmod +x tmp/run_sql_window.sh
bash tmp/run_sql_window.sh
```

**Expected Output**:
```
=== SQL Window Functions Demo: Basic Usage ===

Creating test SQLite database...
Database created with employees table

Running Prolog to compile window function predicates...

=== Compiling Window Function Predicates to SQL ===

1. Compiling employee_rank/4 (ROW_NUMBER)...

Generated SQL:
-- View: employee_rank
CREATE VIEW IF NOT EXISTS employee_rank AS
SELECT name, department, salary,
       ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank
FROM employees;

...

ROW_NUMBER - Rank employees by salary within department:
name        department   salary      rank
----------  -----------  ----------  ----
Bob         Engineering  95000       1
Alice       Engineering  85000       2
Diana       Engineering  78000       3
...

Success: SQL window functions demo complete
```

### Step 4: Test GROUP BY compilation (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.sql_group_by" \
  playbooks/examples_library/sql_window_examples.md \
  > tmp/run_sql_groupby.sh
chmod +x tmp/run_sql_groupby.sh
bash tmp/run_sql_groupby.sh
```

### Step 5: Test recursive CTEs (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.sql_recursive_cte" \
  playbooks/examples_library/sql_window_examples.md \
  > tmp/run_sql_recursive.sh
chmod +x tmp/run_sql_recursive.sh
bash tmp/run_sql_recursive.sh
```

### Step 6: View module info (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.sql_module_info" \
  playbooks/examples_library/sql_window_examples.md \
  > tmp/run_sql_info.sh
chmod +x tmp/run_sql_info.sh
bash tmp/run_sql_info.sh
```

## What This Playbook Demonstrates

1. **sql_target module** (`src/unifyweaver/targets/sql_target.pl`):
   - `compile_predicate_to_sql/3` - Main compilation function
   - `compile_set_operation/4` - INTERSECT, EXCEPT operations
   - `compile_with_cte/4` - Common Table Expressions
   - `compile_recursive_cte/5` - Recursive CTEs
   - `sql_table/2` - Table schema declarations

2. **Window functions**:
   - `row_number(Result, [partition_by(Col), order_by(Col)])`
   - `rank(Result, [...])`
   - `dense_rank(Result, [...])`
   - `sum(Col, Result, [...])` - Running totals
   - Custom OVER clauses

3. **Aggregation (GROUP BY)**:
   - `group_by(GroupField, Goal, AggOp, Result)`
   - Supported: count, sum, avg, max, min
   - HAVING clause for filtering groups

4. **Configuration options**:
   - `dialect(sqlite|postgres|mysql)` - SQL dialect
   - `format(view|cte|select)` - Output format
   - `view_name(Name)` - Override view name

## Example: Window Function

### Prolog:
```prolog
employee_rank(Name, Department, Salary, Rank) :-
    employees(_, Name, Department, Salary, _),
    row_number(Rank, [partition_by(Department), order_by(desc(Salary))]).
```

### Generated SQL:
```sql
CREATE VIEW IF NOT EXISTS employee_rank AS
SELECT name, department, salary,
       ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank
FROM employees;
```

## Example: GROUP BY with HAVING

### Prolog:
```prolog
large_departments(Department, Count) :-
    group_by(Department,
        employees(_, _, Department, _, _),
        count,
        Count),
    Count > 1.
```

### Generated SQL:
```sql
SELECT department, COUNT(*) as count
FROM employees
GROUP BY department
HAVING COUNT(*) > 1;
```

## Example: Recursive CTE

### Prolog:
```prolog
:- sql_recursive_table(org_hierarchy, [employee_id-integer, name-text, level-integer]).

% Defined via compile_recursive_cte/5
```

### Generated SQL:
```sql
WITH RECURSIVE org_hierarchy AS (
    -- Base case
    SELECT employee_id, name, 1 as level
    FROM org_chart
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case
    SELECT e.employee_id, e.name, h.level + 1
    FROM org_chart e
    INNER JOIN org_hierarchy h ON e.manager_id = h.employee_id
)
SELECT * FROM org_hierarchy;
```

## Common Mistakes to Avoid

- **DO NOT** run extracted scripts with `swipl` - they are bash scripts
- **DO** declare table schemas with `sql_table/2` before compiling
- **DO** use appropriate dialect option for your database
- **DO** ensure sqlite3 is installed for testing

## Expected Outcome
- SQL queries with window functions
- Aggregation views with GROUP BY/HAVING
- Recursive CTEs for hierarchical data
- Compatible with SQLite, PostgreSQL, MySQL

## Citations
[1] playbooks/examples_library/sql_window_examples.md
[2] src/unifyweaver/targets/sql_target.pl
[3] skills/skill_unifyweaver_environment.md

