# RIGHT JOIN and FULL OUTER JOIN Design (Phase 3d)

**Date:** 2025-12-04
**Status:** Design Phase
**Previous:** Phase 3c (Mixed INNER/LEFT JOINs)

## Goal

Implement **RIGHT JOIN** and **FULL OUTER JOIN** to complete the JOIN feature set.

## Background: JOIN Types Summary

| JOIN Type | Keeps rows from | NULL where |
|-----------|----------------|------------|
| INNER | Both tables (only matches) | Never (rows filtered) |
| LEFT OUTER | Left table (all rows) | Right table (no match) |
| RIGHT OUTER | Right table (all rows) | Left table (no match) |
| FULL OUTER | Both tables (all rows) | Either side (no match) |

## LEFT JOIN Pattern (Review)

```prolog
% LEFT JOIN: Keep all customers, NULL for orders
customers(CId, Name, _),
(orders(_, CId, Product, _) ; Product = null).
```

**SQL:**
```sql
FROM customers
LEFT JOIN orders ON orders.customer_id = customers.id
```

**Semantics:** All customers appear, even without orders (Product = NULL).

## RIGHT JOIN Pattern

### Proposed Syntax

```prolog
% RIGHT JOIN: Keep all orders, NULL for customers
(customers(CId, Name, _) ; Name = null),
orders(_, CId, Product, _).
```

**SQL:**
```sql
FROM customers
RIGHT JOIN orders ON orders.customer_id = customers.id
```

**Semantics:** All orders appear, even without customers (Name = NULL).

### Key Insight

The **position** of the disjunction determines the JOIN type:
- **Disjunction AFTER table** → LEFT JOIN (keep left, NULL right)
- **Disjunction BEFORE table** → RIGHT JOIN (keep right, NULL left)

### Example

```prolog
% All orders with customer names (or NULL if no customer)
order_customers(Product, Name) :-
    (customers(CId, Name, _) ; Name = null),  % RIGHT JOIN pattern
    orders(_, CId, Product, _).
```

**Generated:**
```sql
SELECT orders.product, customers.name
FROM customers
RIGHT JOIN orders ON orders.customer_id = customers.id;
```

## FULL OUTER JOIN Pattern

### Proposed Syntax

```prolog
% FULL OUTER JOIN: Keep all customers AND all orders
(customers(CId, Name, _) ; Name = null),
(orders(_, CId, Product, _) ; Product = null).
```

**SQL:**
```sql
FROM customers
FULL OUTER JOIN orders ON orders.customer_id = customers.id
```

**Semantics:** All customers appear (even without orders) AND all orders appear (even without customers).

### Example

```prolog
% All customers and all orders, matched where possible
all_customer_orders(Name, Product) :-
    (customers(CId, Name, _) ; Name = null),
    (orders(_, CId, Product, _) ; Product = null).
```

**Result includes:**
- Customers with orders (both non-NULL)
- Customers without orders (Product = NULL)
- Orders without customers (Name = NULL)

## Pattern Detection

### Classification Rules

```
1. No disjunctions anywhere:
   → Regular query (no JOINs needed, or all INNER)

2. Disjunction(s) only AFTER tables:
   → LEFT JOIN(s)
   Example: A, (B ; ...) or A, (B ; ...), (C ; ...)

3. Disjunction(s) only BEFORE tables:
   → RIGHT JOIN(s)
   Example: (A ; ...), B or (A ; ...), (B ; ...), C

4. Disjunctions BOTH before AND after:
   → FULL OUTER JOIN
   Example: (A ; ...), (B ; ...)

5. Mixed positions (complex):
   → Combination of LEFT, RIGHT, and INNER
   Example: A, (B ; ...), (C ; ...), D
   = A INNER JOIN B (LEFT) INNER JOIN C (LEFT) INNER JOIN D
```

### Detection Algorithm

```prolog
detect_join_types(Body, FromTable, InnerJoins, LeftJoins, RightJoins, FullOuterJoins) :-
    % Parse body into sequence of (possibly disjunctive) goals
    parse_body_sequence(Body, Goals),

    % Classify each goal
    classify_goals(Goals, FromTable, InnerJoins, LeftJoins, RightJoins, FullOuterJoins).

classify_goals([First|Rest], FromTable, Inners, Lefts, Rights, Fulls) :-
    classify_goal(First, Type),
    % First non-disjunctive table is FROM
    ...
```

## Implementation Plan

### Phase 1: RIGHT JOIN Only

Start with simpler case - just RIGHT JOIN (disjunction before table).

#### 1. Pattern Detection

```prolog
%% is_right_join_pattern(+Goal)
%  Check if goal is RIGHT JOIN pattern: (TableGoal ; Fallback), NextTable
%
is_right_join_pattern(((_ ; Fallback), TableGoal)) :-
    % Fallback must contain null bindings
    contains_null_binding(Fallback),
    % TableGoal must be a table (not a disjunction)
    TableGoal =.. [TableName|_],
    \+ TableName = (','),
    \+ TableName = (';').
```

#### 2. SQL Generation

```prolog
%% generate_right_join_sql(+LeftGoal, +Fallback, +RightTable, -JoinClause)
%  Generate RIGHT JOIN clause
%
generate_right_join_sql(LeftGoal, Fallback, RightTable, JoinClause) :-
    % Extract table names and find shared variables
    LeftGoal =.. [LeftTableName|LeftArgs],
    RightTable =.. [RightTableName|RightArgs],

    % Find join conditions
    find_join_conditions([LeftGoal, RightTable], JoinSpecs),

    % Generate RIGHT JOIN
    (   JoinSpecs = []
    ->  format(atom(JoinClause), 'RIGHT JOIN ~w', [RightTableName])
    ;   generate_join_conditions(JoinSpecs, CondStr),
        format(atom(JoinClause), 'RIGHT JOIN ~w ON ~w', [RightTableName, CondStr])
    ).
```

### Phase 2: FULL OUTER JOIN

After RIGHT JOIN works, add FULL OUTER.

#### Detection

```prolog
%% is_full_outer_join_pattern(+Body)
%  Check if body contains pattern: (TableA ; ...), (TableB ; ...)
%
is_full_outer_join_pattern(Body) :-
    Body = ((LeftTable ; LeftFallback), (RightTable ; RightFallback)),
    % Both must be table goals with null fallbacks
    contains_null_binding(LeftFallback),
    contains_null_binding(RightFallback),
    is_table_goal(LeftTable),
    is_table_goal(RightTable).
```

#### SQL Generation

```sql
-- FULL OUTER JOIN
FROM table1
FULL OUTER JOIN table2 ON table1.id = table2.id
```

## Test Cases

### Test 1: Simple RIGHT JOIN

```prolog
order_customers(Product, Name) :-
    (customers(CId, Name, _) ; Name = null),
    orders(_, CId, Product, _).
```

**Expected:**
```sql
SELECT orders.product, customers.name
FROM customers
RIGHT JOIN orders ON orders.customer_id = customers.id;
```

### Test 2: RIGHT JOIN Chain

```prolog
chain_right(A, B, C) :-
    (t1(X, A) ; A = null),
    (t2(Y, X, B) ; B = null, X = null),
    t3(_, Y, C).
```

**Expected:**
```sql
FROM t1
RIGHT JOIN t2 ON t2.x = t1.x
RIGHT JOIN t3 ON t3.y = t2.y;
```

### Test 3: FULL OUTER JOIN

```prolog
all_customer_orders(Name, Product) :-
    (customers(CId, Name, _) ; Name = null),
    (orders(_, CId, Product, _) ; Product = null).
```

**Expected:**
```sql
FROM customers
FULL OUTER JOIN orders ON orders.customer_id = customers.id;
```

### Test 4: Mixed LEFT and RIGHT

```prolog
mixed(A, B, C) :-
    t1(X, A),                     % FROM
    (t2(Y, X, B) ; B = null),     % LEFT JOIN
    (t3(_, Y, C) ; C = null, Y = null).  % Could be LEFT or error
```

**Question:** What should this be?
- Option 1: Second LEFT JOIN (both after tables)
- Option 2: FULL OUTER between t2 and t3?

**Decision:** Both disjunctions are AFTER tables, so both are LEFT JOINs.

## Challenges

### 1. Semantic Ambiguity

**Problem:** Some patterns are ambiguous:

```prolog
(A ; ...), B, (C ; ...)
```

Is this:
- A (RIGHT JOIN) B (INNER) C (LEFT)?
- Something else?

**Solution:** Require explicit positioning. If disjunction immediately before B, it's RIGHT JOIN to B. If immediately after B, it's LEFT JOIN from B.

### 2. SQLite Limitation

SQLite doesn't support FULL OUTER JOIN directly. Need to emulate:

```sql
-- FULL OUTER via UNION
SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id
UNION
SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id;
```

**Decision:** Generate FULL OUTER JOIN syntax anyway (works in PostgreSQL, MySQL 8+). SQLite users can use Phase 4 (UNION) workaround manually.

### 3. Backwards Compatibility

Current code assumes disjunction = LEFT JOIN. Need to check position.

**Solution:**
- Check if disjunction is immediately before a table (RIGHT)
- Or immediately after a table (LEFT)
- Update pattern detection logic

## Success Criteria

- ✅ RIGHT JOIN generates correct SQL
- ✅ FULL OUTER JOIN generates correct SQL
- ✅ All previous tests (Phase 3, 3b, 3c) still pass
- ✅ PostgreSQL/MySQL validates generated SQL
- ✅ Clear error message for unsupported patterns

## Implementation Order

1. [ ] Design complete (this document)
2. [ ] Implement RIGHT JOIN pattern detection
3. [ ] Implement RIGHT JOIN SQL generation
4. [ ] Test RIGHT JOIN
5. [ ] Implement FULL OUTER JOIN pattern detection
6. [ ] Implement FULL OUTER JOIN SQL generation
7. [ ] Test FULL OUTER JOIN
8. [ ] Backwards compatibility verification
9. [ ] Database validation (PostgreSQL/MySQL)

## Files to Modify

- `src/unifyweaver/targets/sql_target.pl`
  - Update `is_left_join_clause/1` to check position
  - Add `is_right_join_clause/1`
  - Add `is_full_outer_join_clause/1`
  - Add RIGHT/FULL OUTER compilation predicates

## Open Questions

1. **Should we support mixed LEFT/RIGHT in same query?**
   - Example: `A, (B ; ...), (C ; ...), D` where first is LEFT, second is RIGHT?
   - **Decision:** Start simple - support pure RIGHT and pure FULL OUTER first

2. **How to represent FULL OUTER in Prolog clearly?**
   - Current: Both tables in disjunctions
   - Alternative: Special syntax?
   - **Decision:** Stick with double disjunction - it's explicit

3. **What about RIGHT JOIN chains?**
   - `(A ; ...), (B ; ...), C`
   - Is C the final table, or another RIGHT JOIN?
   - **Decision:** If there's a disjunction immediately before C, it's RIGHT JOIN. Otherwise, INNER.
