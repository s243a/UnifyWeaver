# Fix Complex RIGHT JOIN Chains (Phase 3d Fix)

**Date:** 2025-12-04
**Status:** Implementation
**Issue:** RIGHT JOIN chains with multiple disjunctions don't compile correctly

## Problem

**Current behavior** for pattern `(t1 ; null), (t2 ; null), t3`:
```sql
FROM t1
RIGHT JOIN t3    ← Missing t2!
```

**Expected behavior:**
```sql
FROM t1
RIGHT JOIN t2 ON t2.x = t1.x
RIGHT JOIN t3 ON t3.y = t2.y
```

## Root Cause

Current `compile_right_join_clause/6` only handles the FIRST disjunction:

```prolog
Body = ((LeftGoal ; Fallback), Rest),  % Only extracts first disjunction
conjunction_to_list(Rest, RightGoals),  % [(t2 ; null), t3]
separate_goals(RightGoals, ...), % Can't handle (t2 ; null) as a goal!
```

When `Rest` contains another disjunction `(t2 ; null)`, `conjunction_to_list` returns it as a single element `[(t2 ; null), t3]`. Then `separate_goals` doesn't know how to handle the complex goal `(t2 ; null)` - it's neither a simple table nor a constraint.

## Solution: Extract ALL Leading Disjunctions

Use the same approach as LEFT JOIN, but adapted for RIGHT JOIN semantics:

### LEFT JOIN (for reference)
```prolog
% Pattern: t1, (t2 ; null), (t3 ; null)
extract_all_disjunctions(Body, LeftPart, Disjunctions)
% → LeftPart = t1
% → Disjunctions = [(t2 ; null), (t3 ; null)]
```

### RIGHT JOIN (new)
```prolog
% Pattern: (t1 ; null), (t2 ; null), t3
extract_leading_disjunctions(Body, LeadingDisjunctions, RemainingGoals)
% → LeadingDisjunctions = [(t1 ; null), (t2 ; null)]
% → RemainingGoals = [t3]
```

## Implementation Plan

### 1. Add `extract_leading_disjunctions/3`

Extract all disjunctions that appear BEFORE any non-disjunction goal:

```prolog
%% extract_leading_disjunctions(+Body, -LeadingDisjunctions, -RemainingGoals)
%  Extract all leading disjunctions (RIGHT JOIN pattern)
%
extract_leading_disjunctions(Body, LeadingDisjunctions, RemainingGoals) :-
    extract_leading_disjs_iter(Body, [], LeadingDisjunctions, RemainingGoals).

extract_leading_disjs_iter((Disj, Rest), Acc, AllDisjs, Remaining) :-
    Disj = (_ ; _),  % Is a disjunction
    !,
    extract_leading_disjs_iter(Rest, [Disj|Acc], AllDisjs, Remaining).
extract_leading_disjs_iter((Disj ; _), Acc, AllDisjs, []) :-
    % Final goal is a disjunction
    !,
    reverse([Disj|Acc], AllDisjs).
extract_leading_disjs_iter(Rest, Acc, AllDisjs, Remaining) :-
    % Hit a non-disjunction
    Acc \= [],
    reverse(Acc, AllDisjs),
    conjunction_to_list(Rest, Remaining).
```

### 2. Add `process_right_joins/5`

Similar to `process_left_joins` but generates RIGHT JOINs:

```prolog
%% process_right_joins(+Goals, +AccTables, -JoinClauses, -AllTables, -AllNullVars)
%  Process goals iteratively to generate RIGHT JOIN clauses
%  Goals can be disjunctions or regular table goals
%
process_right_joins([], _, [], [], []).
process_right_joins([Goal|Rest], AccTables, [JoinClause|RestJoins], [Table|RestTables], AllNullVars) :-
    % Check if Goal is a disjunction
    (   Goal = (TableGoal ; Fallback)
    ->  % Disjunction - extract table and NULL bindings
        extract_null_bindings(Fallback, NullVars),
        Table = TableGoal
    ;   % Regular table goal
        NullVars = [],
        Table = Goal
    ),

    % Generate RIGHT JOIN for this table
    generate_right_join_sql(AccTables, Table, JoinClause),

    % Add this table to accumulated
    append(AccTables, [Table], NewAccTables),

    % Recurse
    process_right_joins(Rest, NewAccTables, RestJoins, RestTables, RestNullVars),

    % Accumulate NULL vars
    append(NullVars, RestNullVars, AllNullVars).
```

### 3. Update `compile_right_join_clause/6`

Replace the current implementation:

```prolog
compile_right_join_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    % Extract ALL leading disjunctions
    extract_leading_disjunctions(Body, LeadingDisjunctions, RemainingGoals),

    % Verify we have at least one leading disjunction
    LeadingDisjunctions \= [],

    % First disjunction is FROM
    LeadingDisjunctions = [FirstDisj|RestDisjs],
    FirstDisj = (FirstTable ; FirstFallback),
    extract_null_bindings(FirstFallback, FirstNullVars),

    % Generate FROM clause
    generate_from_clause([FirstTable], FromClause),

    % Combine remaining disjunctions and regular goals
    append(RestDisjs, RemainingGoals, AllRightGoals),

    % Process all right goals (disjunctions + regular) to generate RIGHT JOINs
    process_right_joins(AllRightGoals, [FirstTable], JoinClauses, AllRightTables, RestNullVars),

    % Combine NULL vars
    append(FirstNullVars, RestNullVars, AllNullVars),

    % Combine all JOINs
    atomic_list_concat(JoinClauses, '\n', AllJoins),

    % Generate SELECT
    Head =.. [Name|HeadArgs],
    AllTableGoals = [FirstTable|AllRightTables],
    generate_select_for_nested_joins(HeadArgs, [FirstTable], AllRightTables, AllNullVars, SelectClause),

    % Generate WHERE (no constraints in simple RIGHT JOIN)
    WhereClause = '',

    % Combine
    (member(format(Format), Options) -> true ; Format = view),
    (member(view_name(ViewName), Options) -> true ; ViewName = Name),
    combine_left_join_sql(Format, ViewName, SelectClause, FromClause, AllJoins, WhereClause, SQLCode).
```

## Test Cases

### Test 1: Simple RIGHT JOIN (existing, should still work)
```prolog
order_customers(Product, Name) :-
    (customers(CId, Name, _) ; Name = null),
    orders(_, CId, Product, _).
```

**Expected:**
```sql
FROM customers
RIGHT JOIN orders ON orders.customer_id = customers.id
```

### Test 2: RIGHT JOIN Chain (currently broken)
```prolog
right_chain(A, B, C) :-
    (t1(X, A) ; A = null),
    (t2(Y, X, B) ; B = null, X = null),
    t3(_, Y, C).
```

**Expected:**
```sql
FROM t1
RIGHT JOIN t2 ON t2.x = t1.x
RIGHT JOIN t3 ON t3.y = t2.y
```

### Test 3: Mixed disjunctions (new)
```prolog
mixed(A, B, C, D) :-
    (t1(X, A) ; A = null),
    t2(Y, X, B),  % No disjunction
    (t3(Z, Y, C) ; C = null, Z = null),
    t4(_, Z, D).
```

**Issue:** This is ambiguous! First disjunction suggests RIGHT JOIN, but t2 has no disjunction (INNER JOIN?), then another disjunction (LEFT JOIN?).

**Decision:** For now, this should FAIL pattern detection. Only support pure RIGHT JOIN chains where:
- All leading goals are disjunctions OR
- All leading goals are disjunctions, followed by regular goals (all treated as RIGHT JOINs)

## Success Criteria

- ✅ Test 1 (simple RIGHT JOIN) still works
- ✅ Test 2 (RIGHT JOIN chain) generates correct SQL
- ✅ Test 3 (FULL OUTER) still works
- ✅ All backwards compatibility tests pass
- ✅ SQLite validation for chain pattern

## Files to Modify

- `src/unifyweaver/targets/sql_target.pl`
  - Add `extract_leading_disjunctions/3`
  - Add `process_right_joins/5`
  - Update `compile_right_join_clause/6`

---

**Status:** Ready to implement
