# CROSS JOIN Support (Phase 3e)

**Date:** 2025-12-04
**Status:** Implementation
**Issue:** Pure CROSS JOIN patterns fail to compile due to bug in `generate_from_clause_multi`

## Goal

Support explicit CROSS JOIN syntax for Cartesian products (tables with no shared variables).

## Current Behavior

### Test Results

**Test 1: Pure CROSS JOIN (2 tables)**
```prolog
colors(_, Color), sizes(_, Size)
```
**Status:** ✗ FAILS to compile

**Test 2: Pure CROSS JOIN (3 tables)**
```prolog
colors(_, Color), sizes(_, Size), products(_, Product)
```
**Status:** ✗ FAILS to compile

**Test 3: Mixed INNER + CROSS**
```prolog
categories(CatId, CatName),
products_cat(_, CatId, ProdName),  % INNER JOIN
tags(_, TagName).                   % CROSS JOIN
```
**Status:** ✅ Works! Generates:
```sql
FROM categories
INNER JOIN products_cat ON categories.id = products_cat.cat_id
CROSS JOIN tags;
```

## Root Cause

In `generate_from_clause_multi` (line 1217-1221), there's a bug when handling tables with no shared variables:

```prolog
(   JoinSpecs = []
->  % No shared variables - use CROSS JOIN
    findall(TN, member(G, [FirstGoal|RestGoals]), (functor(G, TN, _)), Tables),
    %       ^^^ WRONG SYNTAX! ^^^
    atomic_list_concat(Tables, ', ', TablesStr),
    format(string(FromClause), 'FROM ~w', [TablesStr])
```

**Problems:**
1. **Syntax error:** `findall` parentheses are wrong - `functor` is outside the goal
2. **Wrong output:** Even if fixed, generates `FROM t1, t2, t3` instead of explicit CROSS JOIN

## Solution

### Fix 1: Correct the `findall` syntax

```prolog
findall(TN, (member(G, [FirstGoal|RestGoals]), functor(G, TN, _)), Tables),
%           ^^^ Add parentheses around the conjunction ^^^
```

### Fix 2: Generate explicit CROSS JOIN syntax

Instead of:
```sql
FROM t1, t2, t3  ← Implicit CROSS JOIN (valid but unclear)
```

Generate:
```sql
FROM t1
CROSS JOIN t2
CROSS JOIN t3    ← Explicit (clearer intent)
```

## Implementation

### Updated `generate_from_clause_multi/2`

```prolog
generate_from_clause_multi([FirstGoal|RestGoals], FromClause) :-
    functor(FirstGoal, FirstTable, _),
    find_join_conditions([FirstGoal|RestGoals], JoinSpecs),

    (   JoinSpecs = []
    ->  % No shared variables - generate explicit CROSS JOINs
        findall(TN, (member(G, RestGoals), functor(G, TN, _)), RestTables),
        generate_cross_joins(RestTables, CrossJoinClauses),
        atomic_list_concat(CrossJoinClauses, '\n', CrossJoinsStr),
        format(string(FromClause), 'FROM ~w\n~w', [FirstTable, CrossJoinsStr])
    ;   % Shared variables - generate INNER JOINs
        generate_join_clause(FirstTable, RestGoals, JoinSpecs, FromClause)
    ).

%% generate_cross_joins(+Tables, -JoinClauses)
%  Generate CROSS JOIN clauses for list of tables
%
generate_cross_joins([], []).
generate_cross_joins([Table|Rest], [JoinClause|RestClauses]) :-
    format(atom(JoinClause), 'CROSS JOIN ~w', [Table]),
    generate_cross_joins(Rest, RestClauses).
```

## Test Cases

### Test 1: Simple CROSS JOIN
```prolog
color_size_combos(Color, Size) :-
    colors(_, Color),
    sizes(_, Size).
```

**Expected:**
```sql
SELECT colors.name, sizes.size
FROM colors
CROSS JOIN sizes;
```

### Test 2: Triple CROSS JOIN
```prolog
all_combinations(Color, Size, Product) :-
    colors(_, Color),
    sizes(_, Size),
    products(_, Product).
```

**Expected:**
```sql
SELECT colors.name, sizes.size, products.product
FROM colors
CROSS JOIN sizes
CROSS JOIN products;
```

### Test 3: Mixed INNER + CROSS (already works)
```prolog
products_with_tags(ProdName, CatName, TagName) :-
    categories(CatId, CatName),
    products_cat(_, CatId, ProdName),
    tags(_, TagName).
```

**Expected:** (already correct)
```sql
FROM categories
INNER JOIN products_cat ON categories.id = products_cat.cat_id
CROSS JOIN tags;
```

## Files to Modify

- `src/unifyweaver/targets/sql_target.pl`
  - Fix `generate_from_clause_multi/2` (line 1212)
  - Add `generate_cross_joins/2` helper

## Success Criteria

- ✅ Test 1: Simple CROSS JOIN compiles and generates correct SQL
- ✅ Test 2: Triple CROSS JOIN compiles and generates correct SQL
- ✅ Test 3: Mixed INNER + CROSS still works
- ✅ All backwards compatibility tests pass
- ✅ Explicit `CROSS JOIN` syntax (not implicit comma-separated)

---

**Status:** Ready to implement
