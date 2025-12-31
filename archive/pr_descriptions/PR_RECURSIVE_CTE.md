# PR Title

```
Add recursive CTE support (WITH RECURSIVE)
```

# PR Description

## Summary

Adds **recursive CTE support** (`WITH RECURSIVE`) for hierarchical and graph queries, completing the SQL target feature set.

## New Predicates

| Predicate | Description |
|-----------|-------------|
| `compile_recursive_cte/5` | Compile recursive CTE with base and recursive cases |
| `compile_recursive_cte/6` | With additional options (union_type, view_name) |
| `sql_recursive_table/2` | Declare CTE schema for self-referencing queries |

## Syntax

```prolog
compile_recursive_cte(
    CTEName,                                % Name of the recursive CTE
    [col1, col2, ...],                      % Column names
    recursive_cte(BasePred, RecursivePred), % Base and recursive predicates
    MainPred,                               % Main query using the CTE
    SQL                                     % Generated SQL
)
```

## Example: Org Chart

```prolog
% Declare the CTE schema
:- sql_recursive_table(org_tree, [id-integer, name-text, manager_id-integer]).

% Base case: top-level employees (no manager)
org_base(Id, Name, ManagerId) :-
    employees(Id, Name, ManagerId, _),
    sql_is_null(ManagerId).

% Recursive case: employees with managers in the tree
org_recursive(Id, Name, ManagerId) :-
    employees(Id, Name, ManagerId, _),
    org_tree(ManagerId, _, _).

% Main query
org_result(Id, Name, ManagerId) :-
    org_tree(Id, Name, ManagerId).

% Compile
compile_recursive_cte(
    org_tree,
    [id, name, manager_id],
    recursive_cte(org_base/3, org_recursive/3),
    org_result/3,
    SQL
)
```

**Generated SQL:**
```sql
WITH RECURSIVE org_tree(id, name, manager_id) AS (
    SELECT id, name, manager_id
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT employees.id, employees.name, employees.manager_id
    FROM employees
    INNER JOIN org_tree ON employees.manager_id = org_tree.id
)
SELECT id, name, manager_id
FROM org_tree;
```

## Options

| Option | Description |
|--------|-------------|
| `union_type(all)` | Use UNION ALL (default) |
| `union_type(distinct)` | Use UNION (removes duplicates) |
| `view_name(Name)` | Wrap in CREATE VIEW |

## Use Cases

- **Org Charts**: Employee hierarchies with manager relationships
- **Category Trees**: Nested categories with parent-child relationships
- **Graph Reachability**: Find all nodes reachable from a starting point
- **Path Finding**: Find paths between nodes in a graph
- **Ancestors/Descendants**: Traverse up or down hierarchies
- **Bill of Materials**: Parts explosion for manufacturing

## Test Results

```
✅ Test 1: Simple Org Chart
✅ Test 2: Category Tree
✅ Test 3: Graph Reachability
✅ Test 4: Path Finding
✅ Test 5: With UNION (distinct) option
✅ Test 6: With view_name option
✅ Test 7: Ancestors (upward traversal)
✅ Test 8: Bill of Materials
```

All existing tests continue to pass.

## Files Changed

- `src/unifyweaver/targets/sql_target.pl` - Core implementation (+110 lines)
- `test_sql_recursive_cte.pl` - Test suite (new, 280 lines)

## SQL Target Feature Complete

With this PR, the SQL target now supports:

- ✅ Basic SELECT/FROM/WHERE
- ✅ JOINs (INNER, LEFT, RIGHT, FULL OUTER, nested)
- ✅ Aggregations (GROUP BY, HAVING)
- ✅ Subqueries (IN, NOT IN, EXISTS, NOT EXISTS)
- ✅ Window functions (ROW_NUMBER, RANK, LAG, LEAD, etc.)
- ✅ Window frames (ROWS/RANGE BETWEEN)
- ✅ ORDER BY, LIMIT, OFFSET
- ✅ DISTINCT
- ✅ CASE WHEN expressions
- ✅ CTEs (WITH clause)
- ✅ **Recursive CTEs (WITH RECURSIVE)** ← NEW
- ✅ Set operations (UNION, INTERSECT, EXCEPT)
- ✅ NULL handling (COALESCE, NULLIF, IFNULL)
- ✅ String functions (CONCAT, UPPER, LOWER, etc.)
- ✅ Date functions (DATE, DATE_ADD, DATE_DIFF, etc.)
- ✅ BETWEEN, LIKE, GLOB, IN (list)

## Breaking Changes

None. Pure feature addition, backwards compatible.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
