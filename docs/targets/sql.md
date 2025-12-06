# SQL Target (`target(sql)`)

`sql_target` compiles Prolog predicates to SQL queries. Unlike other targets that emit executable code, the SQL target generates declarative SQL statements (SELECT, CREATE VIEW) that can be executed directly on relational databases.

## Current Scope

The SQL target is feature-complete for standard relational queries:

- **Basic queries**: SELECT, FROM, WHERE, ORDER BY, LIMIT, OFFSET, DISTINCT
- **JOINs**: INNER, LEFT, RIGHT, FULL OUTER, and nested multi-table joins
- **Aggregations**: GROUP BY, HAVING with COUNT, SUM, AVG, MIN, MAX
- **Subqueries**: IN, NOT IN, EXISTS, NOT EXISTS
- **Window functions**: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, FIRST_VALUE, LAST_VALUE
- **Window frames**: ROWS/RANGE BETWEEN (UNBOUNDED PRECEDING, N PRECEDING, etc.)
- **CTEs**: WITH clause for common table expressions
- **Recursive CTEs**: WITH RECURSIVE for hierarchical queries
- **Set operations**: UNION, INTERSECT, EXCEPT (with ALL variants)
- **CASE WHEN**: Conditional expressions in SELECT
- **NULL handling**: COALESCE, NULLIF, IFNULL, IS NULL, IS NOT NULL
- **String functions**: CONCAT, UPPER, LOWER, SUBSTRING, TRIM, LENGTH, REPLACE
- **Date functions**: DATE, DATETIME, DATE_ADD, DATE_DIFF, EXTRACT, STRFTIME
- **WHERE predicates**: BETWEEN, LIKE, GLOB, IN (list values)

## Module Interface

```prolog
:- use_module('src/unifyweaver/targets/sql_target').

% Core exports
compile_predicate_to_sql(Predicate/Arity, Options, SQL)
compile_set_operation(Op, Predicates, Options, SQL)
compile_with_cte(CTEName, CTEPredicate, MainPredicate, SQL)
compile_recursive_cte(CTEName, Columns, recursive_cte(Base, Recursive), Main, SQL)
write_sql_file(FilePath, SQL)
sql_table(TableName, Schema)
sql_recursive_table(CTEName, Schema)
```

## Table Declaration

Tables are declared with column names and types:

```prolog
:- sql_table(employees, [id-integer, name-text, dept-text, salary-integer, hire_date-text]).
:- sql_table(departments, [id-integer, name-text, budget-real]).
:- sql_table(orders, [id-integer, customer_id-integer, product-text, amount-real, order_date-text]).
```

## Basic Queries

### Simple SELECT

```prolog
employee_name(Name) :-
    employees(_, Name, _, _, _).

?- compile_predicate_to_sql(employee_name/1, [], SQL).
% SELECT name FROM employees;
```

### With WHERE clause

```prolog
high_earner(Name, Salary) :-
    employees(_, Name, _, Salary, _),
    Salary > 100000.

?- compile_predicate_to_sql(high_earner/2, [], SQL).
% SELECT name, salary FROM employees WHERE salary > 100000;
```

### ORDER BY, LIMIT, OFFSET

```prolog
top_earners(Name, Salary) :-
    employees(_, Name, _, Salary, _),
    sql_order_by(Salary, desc),
    sql_limit(10).

?- compile_predicate_to_sql(top_earners/2, [], SQL).
% SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 10;
```

## JOINs

### INNER JOIN

```prolog
employee_dept(EmpName, DeptName) :-
    employees(_, EmpName, DeptId, _, _),
    departments(DeptId, DeptName, _).

?- compile_predicate_to_sql(employee_dept/2, [], SQL).
% SELECT employees.name, departments.name FROM employees
% INNER JOIN departments ON employees.dept = departments.id;
```

### LEFT JOIN

```prolog
employee_with_dept(EmpName, DeptName) :-
    employees(_, EmpName, DeptId, _, _),
    sql_left_join(departments(DeptId, DeptName, _)).

?- compile_predicate_to_sql(employee_with_dept/2, [], SQL).
% SELECT employees.name, departments.name FROM employees
% LEFT JOIN departments ON employees.dept = departments.id;
```

### Nested LEFT JOINs

```prolog
customer_shipments(CustName, Product, Tracking) :-
    customers(CustId, CustName),
    sql_left_join(orders(_, CustId, Product, _, _)),
    sql_left_join(shipments(_, OrderId, Tracking)).

% Generates chained LEFT JOINs
```

### RIGHT and FULL OUTER JOINs

```prolog
% RIGHT JOIN
all_depts(EmpName, DeptName) :-
    departments(DeptId, DeptName, _),
    sql_right_join(employees(_, EmpName, DeptId, _, _)).

% FULL OUTER JOIN
full_mapping(EmpName, DeptName) :-
    employees(_, EmpName, DeptId, _, _),
    sql_full_outer_join(departments(DeptId, DeptName, _)).
```

## Aggregations

```prolog
dept_avg_salary(Dept, AvgSalary) :-
    employees(_, _, Dept, Salary, _),
    sql_group_by([Dept]),
    AvgSalary = sql_avg(Salary).

?- compile_predicate_to_sql(dept_avg_salary/2, [], SQL).
% SELECT dept, AVG(salary) FROM employees GROUP BY dept;
```

### With HAVING

```prolog
large_depts(Dept, Count) :-
    employees(_, _, Dept, _, _),
    sql_group_by([Dept]),
    Count = sql_count(*),
    sql_having(Count > 5).

% SELECT dept, COUNT(*) FROM employees GROUP BY dept HAVING COUNT(*) > 5;
```

## Subqueries

### IN Subquery

```prolog
employees_in_large_depts(Name) :-
    employees(_, Name, Dept, _, _),
    sql_in_subquery(Dept, large_dept/1).

large_dept(Dept) :-
    employees(_, _, Dept, _, _),
    sql_group_by([Dept]),
    Count = sql_count(*),
    Count > 10.

% SELECT name FROM employees WHERE dept IN (SELECT dept FROM employees GROUP BY dept HAVING COUNT(*) > 10);
```

### EXISTS

```prolog
managers(Name) :-
    employees(Id, Name, _, _, _),
    sql_exists(subordinate(Id)).

subordinate(ManagerId) :-
    employees(_, _, _, _, _),
    employees:manager_id = ManagerId.

% SELECT name FROM employees e WHERE EXISTS (SELECT 1 FROM employees WHERE manager_id = e.id);
```

## Window Functions

```prolog
ranked_employees(Name, Salary, Rank) :-
    employees(_, Name, Dept, Salary, _),
    Rank = sql_window(rank, [], [Dept], [(Salary, desc)]).

% SELECT name, salary, RANK() OVER (PARTITION BY dept ORDER BY salary DESC) FROM employees;
```

### Window Frame Specifications

```prolog
running_total(Name, Salary, RunningTotal) :-
    employees(_, Name, Dept, Salary, _),
    RunningTotal = sql_window(sum, [Salary], [Dept], [(Salary, asc)],
                              rows_between(unbounded_preceding, current_row)).

% SELECT name, salary, SUM(salary) OVER (PARTITION BY dept ORDER BY salary
%   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM employees;
```

## CTEs (WITH Clause)

```prolog
high_earners_cte(Id, Name, Salary) :-
    employees(Id, Name, _, Salary, _),
    Salary > 100000.

result(Name, Salary) :-
    high_earners(_, Name, Salary).

?- compile_with_cte(high_earners, high_earners_cte/3, result/2, SQL).
% WITH high_earners AS (
%     SELECT id, name, salary FROM employees WHERE salary > 100000
% )
% SELECT name, salary FROM high_earners;
```

## Recursive CTEs

For hierarchical queries like org charts, category trees, and graph traversal:

```prolog
% Declare CTE schema
:- sql_recursive_table(org_tree, [id-integer, name-text, manager_id-integer]).

% Base case: root nodes
org_base(Id, Name, ManagerId) :-
    employees(Id, Name, ManagerId, _),
    sql_is_null(ManagerId).

% Recursive case: children
org_recursive(Id, Name, ManagerId) :-
    employees(Id, Name, ManagerId, _),
    org_tree(ManagerId, _, _).

% Main query
org_result(Id, Name, ManagerId) :-
    org_tree(Id, Name, ManagerId).

?- compile_recursive_cte(
    org_tree,
    [id, name, manager_id],
    recursive_cte(org_base/3, org_recursive/3),
    org_result/3,
    SQL
).
% WITH RECURSIVE org_tree(id, name, manager_id) AS (
%     SELECT id, name, manager_id FROM employees WHERE manager_id IS NULL
%     UNION ALL
%     SELECT employees.id, employees.name, employees.manager_id
%     FROM employees INNER JOIN org_tree ON employees.manager_id = org_tree.id
% )
% SELECT id, name, manager_id FROM org_tree;
```

### Recursive CTE Options

```prolog
% Use UNION (distinct) instead of UNION ALL
compile_recursive_cte(cte_name, Cols, Spec, Main, [union_type(distinct)], SQL).

% Wrap result in CREATE VIEW
compile_recursive_cte(cte_name, Cols, Spec, Main, [view_name(my_view)], SQL).
```

## Set Operations

```prolog
% UNION
?- compile_set_operation(union, [pred1/2, pred2/2], [], SQL).

% UNION ALL
?- compile_set_operation(union_all, [pred1/2, pred2/2], [], SQL).

% INTERSECT
?- compile_set_operation(intersect, [pred1/2, pred2/2], [], SQL).

% EXCEPT
?- compile_set_operation(except, [pred1/2, pred2/2], [], SQL).
```

## CASE WHEN Expressions

```prolog
salary_category(Name, Category) :-
    employees(_, Name, _, Salary, _),
    Category = sql_case([
        when(Salary > 100000, 'High'),
        when(Salary > 50000, 'Medium')
    ], 'Low').

% SELECT name, CASE WHEN salary > 100000 THEN 'High'
%                   WHEN salary > 50000 THEN 'Medium'
%                   ELSE 'Low' END FROM employees;
```

## SQL Functions

### NULL Handling

```prolog
% COALESCE - return first non-null
Dept = sql_coalesce([D, 'Unknown'])

% NULLIF - return null if equal
Price = sql_nullif(P, 0)

% IFNULL - SQLite-specific
Amount = sql_ifnull(A, 0)

% IS NULL / IS NOT NULL
sql_is_null(ShipDate)
sql_is_not_null(ShipDate)
```

### String Functions

```prolog
sql_concat([FirstName, ' ', LastName])    % firstName || ' ' || lastName
sql_upper(Name)                           % UPPER(name)
sql_lower(Name)                           % LOWER(name)
sql_substring(Name, 1, 3)                 % SUBSTR(name, 1, 3)
sql_trim(Name)                            % TRIM(name)
sql_length(Name)                          % LENGTH(name)
sql_replace(Email, '@', ' at ')           % REPLACE(email, '@', ' at ')
```

### Date Functions

```prolog
sql_date(OrderDate)                       % DATE(order_date)
sql_datetime(Timestamp)                   % DATETIME(timestamp)
sql_date_add(OrderDate, 7, days)          % DATE(order_date, '+7 days')
sql_date_diff(ShipDate, OrderDate)        % JULIANDAY(ship_date) - JULIANDAY(order_date)
sql_extract(year, HireDate)               % STRFTIME('%Y', hire_date)
sql_strftime('%Y-%m-%d', OrderDate)       % STRFTIME('%Y-%m-%d', order_date)
```

### WHERE Predicates

```prolog
sql_between(Salary, 50000, 100000)        % salary BETWEEN 50000 AND 100000
sql_not_between(Salary, 40000, 80000)     % salary NOT BETWEEN 40000 AND 80000
sql_like(Name, 'J%')                      % name LIKE 'J%'
sql_not_like(Name, 'Test%')               % name NOT LIKE 'Test%'
sql_glob(Name, '*Pro*')                   % name GLOB '*Pro*' (SQLite)
sql_in(Dept, [eng, sales, hr])            % dept IN ('eng', 'sales', 'hr')
sql_not_in(Dept, [admin, temp])           % dept NOT IN ('admin', 'temp')
```

## Output Options

```prolog
% Generate SELECT statement (default)
compile_predicate_to_sql(pred/2, [format(select)], SQL).

% Generate CREATE VIEW
compile_predicate_to_sql(pred/2, [format(view), view_name(my_view)], SQL).

% Write to file
write_sql_file('output.sql', SQL).
```

## Use Cases

- **Data analysis**: Complex queries with aggregations, window functions, CTEs
- **Report generation**: SQL views for business intelligence tools
- **Schema migration**: Generate DDL from Prolog specifications
- **Query validation**: Test query logic in Prolog before deploying to database
- **Hierarchical data**: Org charts, category trees, bill of materials with recursive CTEs
- **Graph queries**: Reachability, path finding with recursive CTEs

## Database Compatibility

The SQL target generates standard SQL with SQLite-specific functions (DATE, SUBSTR, etc.). Most output is compatible with:

- SQLite
- PostgreSQL (with minor adjustments)
- MySQL/MariaDB (with minor adjustments)
- SQL Server (ANSI mode)

## Performance Considerations

- Generated SQL relies on database query optimizers
- Window functions and CTEs may be slower on older database versions
- Recursive CTEs have depth limits on some databases
- Consider adding indexes based on generated JOIN conditions

## Limitations

- No INSERT/UPDATE/DELETE generation (SELECT only)
- No stored procedure generation
- Schema inference from Prolog types is basic (integer, text, real)
- No automatic index recommendations

## Relationship to Other Targets

The SQL target is unique among UnifyWeaver targets:

- **Bash/C#/Go targets**: Emit executable programs
- **SQL target**: Emits declarative queries for external database execution

This makes SQL ideal for integration with existing database infrastructure rather than standalone execution.
