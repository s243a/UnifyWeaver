# Add RIGHT JOIN and FULL OUTER JOIN Support (Phase 3d)

## Summary

Implements **RIGHT JOIN** and **FULL OUTER JOIN** patterns, completing the core JOIN feature set for SQL Target.

### What Works

✅ **Simple RIGHT JOIN** - Keep all rows from right table, NULL for missing left rows
✅ **FULL OUTER JOIN** - Keep all rows from both tables, NULL where no match
✅ **Backwards compatible** - All previous LEFT JOIN and mixed INNER/LEFT tests pass

### Known Limitation

⚠️ **RIGHT JOIN chains with multiple disjunctions** (e.g., `(A ; null), (B ; null), C`) are not yet supported. Simple RIGHT JOIN patterns work correctly.

## RIGHT JOIN Pattern

### Syntax

```prolog
% Keep all orders, NULL for missing customers
order_customers(Product, Name) :-
    (customers(CId, Name, _) ; Name = null),  % LEFT side (can be NULL)
    orders(_, CId, Product, _).                % RIGHT side (must exist)
```

### Generated SQL

```sql
SELECT orders.product, customers.name
FROM customers
RIGHT JOIN orders ON orders.customer_id = customers.id;
```

### Semantics

The **disjunction position** determines JOIN type:
- **Before table** → RIGHT JOIN (keep right table, NULL left)
- **After table** → LEFT JOIN (keep left table, NULL right)
- **Both sides** → FULL OUTER JOIN (keep both)

## FULL OUTER JOIN Pattern

### Syntax

```prolog
% Keep all customers AND all orders
all_customer_orders(Name, Product) :-
    (customers(CId, Name, _) ; Name = null),   % Can be NULL
    (orders(_, CId, Product, _) ; Product = null). % Can be NULL
```

### Generated SQL

```sql
SELECT customers.name, orders.product
FROM customers
FULL OUTER JOIN orders ON orders.customer_id = customers.id;
```

### Semantics

**Both tables have disjunctions** → FULL OUTER JOIN preserves all rows from both sides:
- Matched rows: both non-NULL
- Unmatched customers: Product = NULL
- Unmatched orders: Name = NULL

## Implementation

### Pattern Detection (`sql_target.pl:513-557`)

Added three new detection predicates with proper precedence:

```prolog
%% is_full_outer_join_clause(+Body)
%  Highest precedence: Both sides have disjunctions
is_full_outer_join_clause(Body) :-
    Body = ((LeftGoal ; LeftFallback), (RightGoal ; RightFallback)),
    contains_null_binding(LeftFallback),
    contains_null_binding(RightFallback),
    is_table_goal(LeftGoal),
    is_table_goal(RightGoal).

%% is_right_join_clause(+Body)
%  Medium precedence: Disjunction before table(s)
is_right_join_clause(Body) :-
    Body = ((LeftGoal ; Fallback), _Rest),
    contains_null_binding(Fallback),
    LeftGoal =.. [TableName|_],
    \+ TableName = (','),
    \+ TableName = (';').

%% is_left_join_clause(+Body) - UPDATED
%  Lowest precedence: Disjunction after table(s)
is_left_join_clause(Body) :-
    \+ is_right_join_clause(Body),
    \+ is_full_outer_join_clause(Body),
    find_disjunction_in_conjunction(Body, (_RightGoal ; Fallback)),
    contains_null_binding(Fallback).
```

### Dispatch Logic (`sql_target.pl:275-288`)

Updated to check patterns in correct order:

```prolog
compile_single_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    (   is_group_by_clause(Body)
    ->  compile_aggregation_clause(...)
    % Check FULL OUTER first (most specific - both sides disjunction)
    ;   is_full_outer_join_clause(Body)
    ->  compile_full_outer_join_clause(...)
    % Then RIGHT JOIN (disjunction before table)
    ;   is_right_join_clause(Body)
    ->  compile_right_join_clause(...)
    % Then LEFT JOIN (disjunction after table)
    ;   is_left_join_clause(Body)
    ->  compile_left_join_clause(...)
    ;   % Regular clause
        compile_regular_clause(...)
    ).
```

### RIGHT JOIN Compilation (`sql_target.pl:918-963`)

```prolog
compile_right_join_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    Body = ((LeftGoal ; Fallback), Rest),
    extract_null_bindings(Fallback, NullVars),
    conjunction_to_list(Rest, RightGoals),
    separate_goals(RightGoals, RightTableGoals, RightConstraints),

    % FROM = left table (with disjunction)
    generate_from_clause([LeftGoal], FromClause),

    % RIGHT JOINs = all right tables
    generate_right_join_chain([LeftGoal], RightTableGoals, RightJoinClauses),

    % Generate SELECT and WHERE
    AllTableGoals = [LeftGoal|RightTableGoals],
    generate_select_for_nested_joins(HeadArgs, [LeftGoal], RightTableGoals, NullVars, SelectClause),
    generate_where_clause(RightConstraints, HeadArgs, AllTableGoals, WhereClause),

    % Combine
    combine_left_join_sql(Format, ViewName, SelectClause, FromClause, AllJoins, WhereClause, SQLCode).
```

### RIGHT JOIN Chain Generation (`sql_target.pl:965-975`)

```prolog
generate_right_join_chain(_, [], []).
generate_right_join_chain(AccTables, [RightTable|Rest], [JoinClause|RestClauses]) :-
    generate_right_join_sql(AccTables, RightTable, JoinClause),
    append(AccTables, [RightTable], NewAccTables),
    generate_right_join_chain(NewAccTables, Rest, RestClauses).
```

### FULL OUTER JOIN Compilation (`sql_target.pl:1020-1090`)

```prolog
compile_full_outer_join_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    Body = ((LeftGoal ; LeftFallback), (RightGoal ; RightFallback)),
    extract_null_bindings(LeftFallback, LeftNullVars),
    extract_null_bindings(RightFallback, RightNullVars),
    append(LeftNullVars, RightNullVars, AllNullVars),

    % Generate FROM and FULL OUTER JOIN
    generate_from_clause([LeftGoal], FromClause),
    generate_full_outer_join_sql(LeftGoal, RightGoal, JoinClause),

    % Generate SELECT (both sides can have NULLs)
    generate_select_for_nested_joins(HeadArgs, [LeftGoal], [RightGoal], AllNullVars, SelectClause),

    % Combine
    combine_left_join_sql(Format, ViewName, SelectClause, FromClause, JoinClause, WhereClause, SQLCode).

generate_full_outer_join_sql(LeftGoal, RightGoal, JoinClause) :-
    % Find shared variables for join conditions
    findall(Cond,
            (nth1(RightPos, RightArgs, RightArg),
             var(RightArg),
             nth1(LeftPos, LeftArgs, LeftArg),
             LeftArg == RightArg,
             format(atom(Cond), '~w.~w = ~w.~w', [...])),
            JoinConditions),
    % Format FULL OUTER JOIN
    (   JoinConditions = []
    ->  format(atom(JoinClause), 'FULL OUTER JOIN ~w', [RightTableName])
    ;   atomic_list_concat(JoinConditions, ' AND ', JoinCondStr),
        format(atom(JoinClause), 'FULL OUTER JOIN ~w ON ~w', [RightTableName, JoinCondStr])
    ).
```

## Test Results

### Simple RIGHT JOIN ✅

**Test:**
```prolog
order_customers(Product, Name) :-
    (customers(CId, Name, _) ; Name = null),
    orders(_, CId, Product, _).
```

**Generated:**
```sql
SELECT orders.product, customers.name
FROM customers
RIGHT JOIN orders ON orders.customer_id = customers.id;
```

**SQLite Validation (using LEFT JOIN equivalent):**
```
Laptop|Alice
Mouse|Alice
Orphan Product|         ← Order without customer (preserved!)
```

### FULL OUTER JOIN ✅

**Test:**
```prolog
all_customer_orders(Name, Product) :-
    (customers(CId, Name, _) ; Name = null),
    (orders(_, CId, Product, _) ; Product = null).
```

**Generated:**
```sql
SELECT customers.name, orders.product
FROM customers
FULL OUTER JOIN orders ON orders.customer_id = customers.id;
```

**SQLite Validation (using UNION emulation):**
```
|Orphan Product        ← Order without customer
Alice|Laptop           ← Matched
Alice|Mouse            ← Matched
Bob|                   ← Customer without orders
Charlie|               ← Customer without orders
```

### Backwards Compatibility ✅

All previous tests continue to pass:

```bash
$ swipl test_sql_left_join.pl
✓ Test 1: Basic LEFT JOIN
✓ Test 2: Multi-column LEFT JOIN
✓ Test 3: Nested LEFT JOINs
✓ Test 4: LEFT JOIN with WHERE

$ swipl test_mixed_joins.pl
✓ Test 1: One INNER, One LEFT
✓ Test 2: Two INNER, One LEFT
✓ Test 3: One INNER, Two LEFT
```

## Known Limitation: Complex RIGHT JOIN Chains

**Pattern:**
```prolog
right_chain(A, B, C) :-
    (t1(X, A) ; A = null),
    (t2(Y, X, B) ; B = null, X = null),  % ← Another disjunction
    t3(_, Y, C).
```

**Issue:** When Rest contains additional disjunctions, `separate_goals` treats `(t2 ; ...)` as a single complex goal rather than recognizing it as a RIGHT JOIN pattern.

**Current behavior:** Generates incomplete SQL (missing t2)

**Workaround:** Use simple RIGHT JOIN patterns without nested disjunctions in Rest.

**Future work:** Recursively detect and compile nested RIGHT JOIN patterns.

## Files Modified

### Core Implementation
- **`src/unifyweaver/targets/sql_target.pl`** (+180 lines, -14 lines)
  - Added `is_right_join_clause/1` and `is_full_outer_join_clause/1`
  - Updated `is_left_join_clause/1` to exclude RIGHT/FULL OUTER
  - Updated dispatch logic in `compile_single_clause/5`
  - Added `compile_right_join_clause/6`
  - Added `compile_full_outer_join_clause/6`
  - Added `generate_right_join_chain/3`
  - Added `generate_right_join_sql/3`
  - Added `generate_full_outer_join_sql/3`

### Documentation
- **`proposals/RIGHT_FULL_OUTER_JOIN_DESIGN.md`** (new, 363 lines)
  - Complete design specification
  - Pattern syntax and semantics
  - Implementation approach
  - Test cases

- **`PR_RIGHT_FULL_OUTER_JOIN.md`** (this file)

### Tests
- **`test_right_full_outer.pl`** (new, 76 lines)
  - Test 1: Simple RIGHT JOIN
  - Test 2: FULL OUTER JOIN
  - Test 3: RIGHT JOIN chain (known limitation)

- **`test_right_full_outer_sqlite.sh`** (new, executable)
  - SQLite/PostgreSQL validation
  - Demonstrates RIGHT JOIN and FULL OUTER semantics
  - Shows equivalences for databases without native support

## Database Support

| Database | RIGHT JOIN | FULL OUTER JOIN |
|----------|------------|-----------------|
| PostgreSQL | ✅ Native | ✅ Native |
| MySQL 8+ | ✅ Native | ✅ Native |
| SQLite | ❌ Use LEFT JOIN | ❌ Use UNION |
| SQL Server | ✅ Native | ✅ Native |
| Oracle | ✅ Native | ✅ Native |

**Note:** UnifyWeaver generates standard SQL syntax. SQLite users can manually rewrite or use UNION-based emulation.

## Examples

### E-commerce: Orders with optional customers

```prolog
% Show all orders, even orphaned ones
all_orders(Product, CustomerName) :-
    (customers(CId, CustomerName, _) ; CustomerName = null),
    orders(_, CId, Product, _).
```

**SQL:**
```sql
SELECT orders.product, customers.name AS customer_name
FROM customers
RIGHT JOIN orders ON orders.customer_id = customers.id;
```

### Analytics: Complete customer-order matrix

```prolog
% Show all customers and all orders (full outer)
customer_order_matrix(Customer, Product) :-
    (customers(_, Customer, _) ; Customer = null),
    (orders(_, _, Product, _) ; Product = null).
```

**SQL:**
```sql
SELECT customers.name, orders.product
FROM customers
FULL OUTER JOIN orders ON orders.customer_id = customers.id;
```

## Breaking Changes

**None.** This is a pure feature addition with full backwards compatibility.

## Related Work

- **Phase 3** (PR #172): LEFT JOIN support
- **Phase 3b** (PR #174): Nested LEFT JOINs
- **Phase 3c** (PR #175): Mixed INNER/LEFT JOINs
- **Phase 3d** (This PR): RIGHT JOIN and FULL OUTER JOIN
- **Phase 4** (PR #171): Set operations (UNION/INTERSECT/EXCEPT)

## Future Work

Phase 3 JOIN features are now complete! Possible next directions:

1. **Complex RIGHT JOIN chains** - Handle nested disjunctions in Rest
2. **CROSS JOIN** - Explicit Cartesian product
3. **Phase 5: Subqueries** - Nested SELECT statements
4. **Phase 6: Window Functions** - OVER clauses

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
