# Nested LEFT JOIN Design (Phase 3b)

**Date:** 2025-12-04
**Status:** Design Phase
**Previous:** Phase 3 (Single LEFT JOIN)

## Goal

Support multiple sequential LEFT JOINs by processing multiple disjunctions in order.

## Problem

Current implementation only processes the first disjunction:

```prolog
customer_shipments(Name, Product, Tracking) :-
    customers(CustomerId, Name, _),
    ( orders(OrderId, CustomerId, Product, _)    ← First disjunction
    ; Product = null, OrderId = null
    ),
    ( shipments(_, OrderId, Tracking)             ← Second disjunction (MISSED!)
    ; Tracking = null
    ).
```

**Current Output:**
```sql
SELECT customers.name, unknown, shipments.tracking
FROM customers
LEFT JOIN shipments;  -- Missing: LEFT JOIN orders, wrong join condition
```

**Desired Output:**
```sql
SELECT customers.name, orders.product, shipments.tracking
FROM customers
LEFT JOIN orders ON orders.customer_id = customers.id
LEFT JOIN shipments ON shipments.order_id = orders.id;
```

## Body Structure Analysis

```
Body = customers(...), (orders(...) ; Product=null, OrderId=null), (shipments(...) ; Tracking=null)
```

Parsed as:
```
CONJUNCTION:
  customers(...)
  CONJUNCTION:
    DISJUNCTION_1: (orders(...) ; ...)
    DISJUNCTION_2: (shipments(...) ; ...)
```

## Design Approach

### Key Insight

Process disjunctions **iteratively** from left to right, accumulating:
1. **LEFT goals** - tables and constraints encountered so far
2. **JOIN clauses** - one for each disjunction
3. **NULL-able variables** - tracked across all disjunctions

### Algorithm

```
1. Extract all disjunctions from body in order
2. Initialize LeftGoals with non-disjunction goals before first disjunction
3. For each disjunction:
   a. Extract RightGoal and Fallback
   b. Validate pattern (RightGoal is table, Fallback has null bindings)
   c. Generate LEFT JOIN clause
   d. Add RightGoal to LeftGoals for next iteration
4. Generate final SQL with multiple LEFT JOIN clauses
```

### Example Walkthrough

**Input:**
```prolog
customers(CId, Name, _),
( orders(OId, CId, Product, _) ; Product = null, OId = null ),
( shipments(_, OId, Tracking) ; Tracking = null )
```

**Iteration 1:**
- LeftGoals: `[customers(CId, Name, _)]`
- Disjunction: `(orders(OId, CId, Product, _) ; Product = null, OId = null)`
- Generate: `LEFT JOIN orders ON orders.customer_id = customers.id`
- Update LeftGoals: `[customers(...), orders(...)]`

**Iteration 2:**
- LeftGoals: `[customers(...), orders(...)]`
- Disjunction: `(shipments(_, OId, Tracking) ; Tracking = null)`
- Generate: `LEFT JOIN shipments ON shipments.order_id = orders.id`
- Update LeftGoals: `[customers(...), orders(...), shipments(...)]`

**Result:**
```sql
SELECT customers.name, orders.product, shipments.tracking
FROM customers
LEFT JOIN orders ON orders.customer_id = customers.id
LEFT JOIN shipments ON shipments.order_id = orders.id;
```

## Implementation Plan

### 1. Extract All Disjunctions

```prolog
%% extract_all_disjunctions(+Body, -LeftGoals, -Disjunctions)
%  Extract all disjunctions in order, with left goals before first disjunction
%
extract_all_disjunctions(Body, LeftGoals, Disjunctions) :-
    extract_disjunctions_iter(Body, [], LeftGoals, Disjunctions).

extract_disjunctions_iter((A, B), Acc, LeftGoals, Disjs) :-
    (   A = (_ ; _)
    ->  % Found first disjunction
        reverse(Acc, LeftGoals),
        collect_remaining_disjunctions((A, B), Disjs)
    ;   % Not a disjunction, accumulate as left goal
        extract_disjunctions_iter(B, [A|Acc], LeftGoals, Disjs)
    ).

collect_remaining_disjunctions((A, B), [A|Rest]) :-
    A = (_ ; _), !,
    collect_remaining_disjunctions(B, Rest).
collect_remaining_disjunctions((A ; B), [(A ; B)]) :- !.
collect_remaining_disjunctions(_, []).
```

### 2. Process Disjunctions Iteratively

```prolog
%% process_left_joins(+Disjunctions, +InitialLeftGoals, -JoinClauses, -AllNullVars)
%  Process each disjunction to generate JOIN clauses
%
process_left_joins([], _, [], []).
process_left_joins([Disj|Rest], LeftGoals, [JoinClause|RestJoins], AllNullVars) :-
    % Extract pattern
    Disj = (RightGoal ; Fallback),

    % Extract NULL bindings
    extract_null_bindings(Fallback, NullVars),

    % Generate LEFT JOIN clause
    generate_left_join_sql(LeftGoals, RightGoal, JoinClause),

    % Add RightGoal to LeftGoals for next iteration
    append(LeftGoals, [RightGoal], NewLeftGoals),

    % Recurse
    process_left_joins(Rest, NewLeftGoals, RestJoins, RestNullVars),

    % Accumulate NULL vars
    append(NullVars, RestNullVars, AllNullVars).
```

### 3. Update Main Compilation

```prolog
compile_left_join_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    % Extract all disjunctions
    extract_all_disjunctions(Body, LeftGoals, Disjunctions),

    % Verify we have at least one disjunction
    Disjunctions \= [],

    % Separate left goals into tables and constraints
    separate_goals(LeftGoals, LeftTableGoals, LeftConstraints),

    % Process all disjunctions iteratively
    process_left_joins(Disjunctions, LeftTableGoals, JoinClauses, AllNullVars),

    % Parse head arguments
    Head =.. [Name|HeadArgs],

    % Generate FROM clause
    generate_from_clause(LeftTableGoals, FromClause),

    % Combine all JOIN clauses
    atomic_list_concat(JoinClauses, '\n', AllJoinsCombined),

    % Generate SELECT clause
    generate_select_for_nested_joins(HeadArgs, LeftTableGoals, Disjunctions, AllNullVars, SelectClause),

    % Generate WHERE clause
    generate_where_clause(LeftConstraints, HeadArgs, LeftTableGoals, WhereClause),

    % Combine into final SQL
    combine_nested_left_join_sql(Format, ViewName, SelectClause, FromClause, AllJoinsCombined, WhereClause, SQLCode).
```

### 4. Update SELECT Generation

The SELECT clause needs to find columns across multiple right tables:

```prolog
generate_select_for_nested_joins(HeadArgs, LeftGoals, Disjunctions, AllNullVars, SelectClause) :-
    % Extract all right tables from disjunctions
    findall(RightGoal,
            (member((RightGoal ; _), Disjunctions)),
            RightGoals),

    % Combine left and right goals
    append(LeftGoals, RightGoals, AllGoals),

    % Generate column list
    findall(ColExpr,
            (nth1(Idx, HeadArgs, Arg),
             var(Arg),
             find_column_in_all_goals(Arg, AllGoals, ColExpr)),
            Columns),

    atomic_list_concat(Columns, ', ', ColStr),
    format(atom(SelectClause), 'SELECT ~w', [ColStr]).

find_column_in_all_goals(Var, Goals, ColExpr) :-
    (   find_column_in_goals(Var, Goals, ColExpr)
    ->  true
    ;   ColExpr = 'unknown'
    ).
```

## Test Cases

### Test 1: Two LEFT JOINs (customers → orders → shipments)

```prolog
customer_shipments(Name, Product, Tracking) :-
    customers(CId, Name, _),
    (orders(OId, CId, Product, _) ; Product = null, OId = null),
    (shipments(_, OId, Tracking) ; Tracking = null).
```

**Expected:**
```sql
SELECT customers.name, orders.product, shipments.tracking
FROM customers
LEFT JOIN orders ON orders.customer_id = customers.id
LEFT JOIN shipments ON shipments.order_id = orders.id;
```

### Test 2: Three LEFT JOINs (a → b → c → d)

```prolog
chain(A, B, C, D) :-
    t1(X, A),
    (t2(Y, X, B) ; B = null, Y = null),
    (t3(Z, Y, C) ; C = null, Z = null),
    (t4(_, Z, D) ; D = null).
```

**Expected:**
```sql
SELECT t1.a, t2.b, t3.c, t4.d
FROM t1
LEFT JOIN t2 ON t2.x = t1.x
LEFT JOIN t3 ON t3.y = t2.y
LEFT JOIN t4 ON t4.z = t3.z;
```

### Test 3: With WHERE Clause

```prolog
eu_customer_shipments(Name, Product, Tracking) :-
    customers(CId, Name, Region),
    Region = 'EU',
    (orders(OId, CId, Product, _) ; Product = null, OId = null),
    (shipments(_, OId, Tracking) ; Tracking = null).
```

**Expected:**
```sql
SELECT customers.name, orders.product, shipments.tracking
FROM customers
LEFT JOIN orders ON orders.customer_id = customers.id
LEFT JOIN shipments ON shipments.order_id = orders.id
WHERE region = 'EU';
```

## Edge Cases

### 1. Mixed INNER and LEFT JOINs

```prolog
result(A, B, C) :-
    t1(X, A),
    t2(X, Y, B),              % INNER JOIN (no disjunction)
    (t3(Y, C) ; C = null).    % LEFT JOIN
```

**Expected:**
```sql
SELECT t1.a, t2.b, t3.c
FROM t1
JOIN t2 ON t2.x = t1.x        -- INNER
LEFT JOIN t3 ON t3.y = t2.y;  -- LEFT
```

**Note:** This requires distinguishing between tables added via INNER JOIN vs LEFT JOIN.

### 2. Independent LEFT JOINs (No Dependency)

```prolog
result(A, B, C) :-
    t1(X, A),
    (t2(X, B) ; B = null),
    (t3(X, C) ; C = null).    % Both join to t1, not to each other
```

**Expected:**
```sql
SELECT t1.a, t2.b, t3.c
FROM t1
LEFT JOIN t2 ON t2.x = t1.x
LEFT JOIN t3 ON t3.x = t1.x;  -- Joins to t1, not t2
```

## Implementation Order

1. ✅ **Phase 3**: Single LEFT JOIN (completed)
2. 🔄 **Phase 3b**: Multiple sequential LEFT JOINs (current)
   - [ ] Implement `extract_all_disjunctions/3`
   - [ ] Implement `process_left_joins/4`
   - [ ] Update `compile_left_join_clause/6`
   - [ ] Update `generate_select_for_nested_joins/5`
   - [ ] Test with 2-3 JOIN cases
   - [ ] Validate with SQLite
3. 📋 **Phase 3c**: Mixed INNER/LEFT JOINs (future)
4. 📋 **Phase 3d**: RIGHT JOIN, FULL OUTER JOIN (future)

## Success Criteria

- ✅ Test 3 generates correct SQL with 2 LEFT JOINs
- ✅ 3-table chain generates 3 LEFT JOINs
- ✅ JOIN conditions correctly reference previous tables
- ✅ SELECT clause resolves columns from all tables
- ✅ SQLite validates generated SQL executes correctly
- ✅ WHERE clauses work with multiple JOINs

## Files to Modify

- `src/unifyweaver/targets/sql_target.pl`
  - Add `extract_all_disjunctions/3`
  - Add `process_left_joins/4`
  - Modify `compile_left_join_clause/6`
  - Add `generate_select_for_nested_joins/5`
  - Update `combine_nested_left_join_sql/7`

## Backwards Compatibility

All existing tests must continue to pass:
- ✅ Test 1: Basic LEFT JOIN (single disjunction)
- ✅ Test 2: Multi-column LEFT JOIN (single disjunction)
- ✅ Test 4: LEFT JOIN with WHERE (single disjunction)

The changes should be additive - if there's only one disjunction, behavior is identical to Phase 3.
