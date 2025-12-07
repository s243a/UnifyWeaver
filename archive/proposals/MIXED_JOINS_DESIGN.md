# Mixed INNER/LEFT JOIN Design (Phase 3c)

**Date:** 2025-12-04
**Status:** Design Phase
**Previous:** Phase 3b (Nested LEFT JOINs)

## Goal

Support queries where some tables use **INNER JOIN** (no disjunction) and others use **LEFT JOIN** (with disjunction).

## Problem

Current implementation treats all table goals before disjunctions as a single FROM clause. This doesn't distinguish between:
- Tables that should be INNER JOINed to previous tables
- Tables in the initial FROM clause

### Example Pattern

```prolog
result(Name, Category, Product) :-
    customers(CustomerId, Name, _),
    categories(CategoryId, CustomerId, Category),   % INNER JOIN (no disjunction)
    ( orders(_, CustomerId, CategoryId, Product)    % LEFT JOIN (disjunction)
    ; Product = null
    ).
```

**Current behavior:** Likely treats `categories` as part of FROM:
```sql
-- Incorrect or suboptimal:
FROM customers, categories
LEFT JOIN orders ON ...
```

**Desired behavior:**
```sql
SELECT customers.name, categories.category, orders.product
FROM customers
JOIN categories ON categories.customer_id = customers.id
LEFT JOIN orders ON orders.customer_id = customers.id
              AND orders.category_id = categories.id;
```

## Design Approach

### Key Insight

Tables that appear **between** the initial FROM table and disjunctions should be INNER JOINed if they share variables with previous tables.

### Classification

1. **FROM table** - First table in the clause (no JOIN needed)
2. **INNER JOIN tables** - Tables with shared variables, no disjunction
3. **LEFT JOIN tables** - Tables in disjunction with NULL fallback

### Algorithm

```
1. Extract all disjunctions (already done in Phase 3b)
2. Split left goals into:
   a. FROM table (first table goal)
   b. INNER JOIN tables (remaining table goals before disjunctions)
   c. Constraints (non-table goals)
3. Generate SQL:
   - FROM: first table
   - JOIN: for each INNER JOIN table, generate ON condition
   - LEFT JOIN: for each disjunction (already done)
   - WHERE: constraints
```

### Example Walkthrough

```prolog
customers(CId, Name, _),
categories(CatId, CId, Category),
( orders(_, CId, CatId, Product) ; Product = null )
```

**Step 1: Extract**
- Disjunctions: `[(orders(...) ; Product = null)]`
- Left goals: `[customers(...), categories(...)]`

**Step 2: Classify**
- FROM table: `customers`
- INNER JOIN tables: `[categories]`
- LEFT JOIN tables: `[orders]`

**Step 3: Generate**
```sql
FROM customers
JOIN categories ON categories.customer_id = customers.id
LEFT JOIN orders ON orders.customer_id = customers.id
              AND orders.category_id = categories.id
```

## Implementation Plan

### 1. Update `separate_goals/3`

Currently separates into `TableGoals` and `Constraints`. Need to further split `TableGoals`:

```prolog
%% separate_table_goals(+TableGoals, -FromTable, -InnerJoinTables)
%  Split table goals into FROM and INNER JOIN tables
%
separate_table_goals([FirstTable|Rest], FirstTable, Rest).
separate_table_goals([], error, []).  % Error case
```

### 2. Generate INNER JOIN clauses

```prolog
%% generate_inner_joins(+InnerJoinTables, +FromTable, -JoinClauses)
%  Generate INNER JOIN clauses with ON conditions
%
generate_inner_joins([], _, []).
generate_inner_joins([Table|Rest], AccTables, [JoinClause|RestJoins]) :-
    % Find shared variables with accumulated tables
    generate_join_sql(AccTables, Table, 'JOIN', JoinClause),
    % Add this table to accumulated for next iteration
    append(AccTables, [Table], NewAccTables),
    generate_inner_joins(Rest, NewAccTables, RestJoins).
```

### 3. Update `compile_left_join_clause/6`

```prolog
compile_left_join_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    % ... (existing extraction code) ...

    % Separate left goals into FROM, INNER JOIN, and constraints
    separate_goals(LeftGoals, LeftTableGoals, LeftConstraints),
    separate_table_goals(LeftTableGoals, FromTable, InnerJoinTables),

    % Generate FROM clause (just the first table)
    generate_from_clause([FromTable], FromClause),

    % Generate INNER JOIN clauses
    generate_inner_joins(InnerJoinTables, [FromTable], InnerJoinClauses),
    atomic_list_concat(InnerJoinClauses, '\n', InnerJoins),

    % Generate LEFT JOIN clauses (existing code)
    append([FromTable|InnerJoinTables], AllLeftTables),  % All tables accumulated
    process_left_joins(Disjunctions, AllLeftTables, LeftJoinClauses, AllRightGoals, AllNullVars),
    atomic_list_concat(LeftJoinClauses, '\n', LeftJoins),

    % Combine all JOINs
    atomic_list_concat([InnerJoins, LeftJoins], '\n', AllJoins),

    % ... (rest of code) ...
```

### 4. Generalize JOIN generation

Extract common logic from LEFT JOIN generation:

```prolog
%% generate_join_sql(+LeftTables, +RightTable, +JoinType, -JoinClause)
%  Generate JOIN clause with ON conditions
%  JoinType: 'JOIN' or 'LEFT JOIN'
%
generate_join_sql(LeftTables, RightTable, JoinType, JoinClause) :-
    RightTable =.. [RightTableName|RightArgs],

    % Find shared variables (join keys)
    findall(Cond,
            (nth1(RightPos, RightArgs, RightArg),
             var(RightArg),
             % Find this variable in left tables
             member(LeftTable, LeftTables),
             LeftTable =.. [LeftTableName|LeftArgs],
             nth1(LeftPos, LeftArgs, LeftArg),
             LeftArg == RightArg,  % Same variable
             % Get column names
             get_column_name_from_schema(LeftTableName, LeftPos, LeftCol),
             get_column_name_from_schema(RightTableName, RightPos, RightCol),
             % Format condition
             format(atom(Cond), '~w.~w = ~w.~w',
                    [RightTableName, RightCol, LeftTableName, LeftCol])
            ),
            JoinConditions),

    % Build JOIN clause
    (   JoinConditions = []
    ->  format(atom(JoinClause), '~w ~w', [JoinType, RightTableName])
    ;   atomic_list_concat(JoinConditions, ' AND ', JoinCondStr),
        format(atom(JoinClause), '~w ~w ON ~w', [JoinType, RightTableName, JoinCondStr])
    ).
```

## Test Cases

### Test 1: One INNER, One LEFT

```prolog
customer_category_orders(Name, Category, Product) :-
    customers(CId, Name, _),
    categories(CatId, CId, Category),
    (orders(_, CId, CatId, Product) ; Product = null).
```

**Expected:**
```sql
SELECT customers.name, categories.category, orders.product
FROM customers
JOIN categories ON categories.customer_id = customers.id
LEFT JOIN orders ON orders.customer_id = customers.id
              AND orders.category_id = categories.id;
```

### Test 2: Two INNER, One LEFT

```prolog
chain(A, B, C, D) :-
    t1(X, A),
    t2(Y, X, B),
    t3(Z, Y, C),
    (t4(_, Z, D) ; D = null).
```

**Expected:**
```sql
SELECT t1.a, t2.b, t3.c, t4.d
FROM t1
JOIN t2 ON t2.x = t1.x
JOIN t3 ON t3.y = t2.y
LEFT JOIN t4 ON t4.z = t3.z;
```

### Test 3: One INNER, Two LEFT (nested)

```prolog
result(A, B, C, D) :-
    t1(X, A),
    t2(Y, X, B),
    (t3(Z, Y, C) ; C = null, Z = null),
    (t4(_, Z, D) ; D = null).
```

**Expected:**
```sql
SELECT t1.a, t2.b, t3.c, t4.d
FROM t1
JOIN t2 ON t2.x = t1.x
LEFT JOIN t3 ON t3.y = t2.y
LEFT JOIN t4 ON t4.z = t3.z;
```

### Test 4: Multiple INNER, Multiple LEFT

```prolog
complex(A, B, C, D, E) :-
    t1(W, A),
    t2(X, W, B),
    t3(Y, X, C),
    (t4(Z, Y, D) ; D = null, Z = null),
    (t5(_, Z, E) ; E = null).
```

**Expected:**
```sql
SELECT t1.a, t2.b, t3.c, t4.d, t5.e
FROM t1
JOIN t2 ON t2.w = t1.w
JOIN t3 ON t3.x = t2.x
LEFT JOIN t4 ON t4.y = t3.y
LEFT JOIN t5 ON t5.z = t4.z;
```

## Edge Cases

### 1. No INNER JOINs (already supported)

```prolog
result(A, B) :-
    t1(X, A),
    (t2(X, B) ; B = null).
```

**Expected:**
```sql
SELECT t1.a, t2.b
FROM t1
LEFT JOIN t2 ON t2.x = t1.x;
```

This should continue to work (backwards compatible).

### 2. No LEFT JOINs (regular query)

```prolog
result(A, B) :-
    t1(X, A),
    t2(X, B).
```

**Expected:** Should NOT be detected as LEFT JOIN clause.
Falls back to regular query compilation.

### 3. Independent INNER JOINs

```prolog
result(A, B, C) :-
    t1(X, A),
    t2(Y, B),  % No shared variable with t1!
    (t3(X, C) ; C = null).
```

**Expected:**
```sql
FROM t1, t2  -- Cross join (no shared variable)
LEFT JOIN t3 ON t3.x = t1.x
```

**Note:** This might need special handling or could be a user error.

## Backwards Compatibility

✅ **Phase 3 (single LEFT JOIN)** - No INNER JOINs before disjunction
✅ **Phase 3b (nested LEFT JOINs)** - No INNER JOINs before disjunctions

Both should continue to work:
- If `InnerJoinTables = []`, generate no INNER JOIN clauses
- LEFT JOIN logic remains unchanged

## Success Criteria

- ✅ Test 1-4 generate correct mixed INNER/LEFT JOIN SQL
- ✅ All Phase 3 and 3b tests continue to pass
- ✅ SQLite validates generated SQL executes correctly
- ✅ JOIN conditions correctly reference all previous tables

## Implementation Order

1. [ ] Add `separate_table_goals/3`
2. [ ] Add `generate_join_sql/4` (generalized)
3. [ ] Add `generate_inner_joins/3`
4. [ ] Update `compile_left_join_clause/6`
5. [ ] Test with mixed join cases
6. [ ] Validate backwards compatibility
7. [ ] SQLite integration tests

## Files to Modify

- `src/unifyweaver/targets/sql_target.pl`
  - Add `separate_table_goals/3`
  - Add `generate_join_sql/4`
  - Add `generate_inner_joins/3`
  - Modify `compile_left_join_clause/6`
  - Refactor `generate_left_join_sql/3` to use `generate_join_sql/4`
