# Add CROSS JOIN Support (Phase 3e)

## Summary

Fixes a bug in `generate_from_clause_multi` and adds explicit **CROSS JOIN** syntax support for Cartesian products (tables with no shared variables).

### What's Fixed

✅ **Pure CROSS JOIN patterns** - Tables with no shared variables now compile correctly
✅ **Explicit CROSS JOIN syntax** - Generates `CROSS JOIN` instead of implicit comma-separated tables
✅ **Mixed INNER + CROSS** - Already worked, continues to work
✅ **Backwards compatible** - All previous JOIN tests pass

## Problem

**Before this fix**, patterns with no shared variables failed to compile:

```prolog
% This failed!
color_size_combos(Color, Size) :-
    colors(_, Color),
    sizes(_, Size).
```

**Root cause:** Bug in `generate_from_clause_multi` (line 1219):
```prolog
findall(TN, member(G, [FirstGoal|RestGoals]), (functor(G, TN, _)), Tables),
%       ^^^ Wrong syntax - functor outside the goal ^^^
```

Additionally, even if the syntax were correct, it would generate:
```sql
FROM colors, sizes  ← Implicit CROSS JOIN (unclear)
```

Instead of explicit:
```sql
FROM colors
CROSS JOIN sizes    ← Explicit (clear intent)
```

## Solution

### 1. Fixed `findall` Syntax

```prolog
% Before (broken):
findall(TN, member(G, RestGoals), (functor(G, TN, _)), Tables)

% After (fixed):
findall(TN, (member(G, RestGoals), functor(G, TN, _)), RestTables)
%           ^^^ Parentheses around the conjunction ^^^
```

### 2. Added Explicit CROSS JOIN Generation

New helper predicate:
```prolog
%% generate_cross_joins(+Tables, -JoinClauses)
generate_cross_joins([], []).
generate_cross_joins([Table|Rest], [JoinClause|RestClauses]) :-
    format(atom(JoinClause), 'CROSS JOIN ~w', [Table]),
    generate_cross_joins(Rest, RestClauses).
```

### 3. Updated `generate_from_clause_multi/2`

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
```

## Test Results

### Test 1: Simple CROSS JOIN ✅ (NOW WORKS!)

**Pattern:**
```prolog
color_size_combos(Color, Size) :-
    colors(_, Color),
    sizes(_, Size).
```

**Before:** ✗ Failed to compile

**After:**
```sql
CREATE VIEW color_size_combos AS
SELECT colors.name, sizes.size
FROM colors
CROSS JOIN sizes;
```

### Test 2: Triple CROSS JOIN ✅ (NOW WORKS!)

**Pattern:**
```prolog
all_combinations(Color, Size, Product) :-
    colors(_, Color),
    sizes(_, Size),
    products(_, Product).
```

**Before:** ✗ Failed to compile

**After:**
```sql
SELECT colors.name, sizes.size, products.product
FROM colors
CROSS JOIN sizes
CROSS JOIN products;
```

### Test 3: Mixed INNER + CROSS ✅ (Still Works)

**Pattern:**
```prolog
products_with_tags(ProdName, CatName, TagName) :-
    categories(CatId, CatName),
    products_cat(_, CatId, ProdName),  % INNER JOIN
    tags(_, TagName).                   % CROSS JOIN
```

**Generated:**
```sql
SELECT products_cat.name, categories.name, tags.tag_name
FROM categories
INNER JOIN products_cat ON categories.id = products_cat.cat_id
CROSS JOIN tags;
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

$ swipl test_right_full_outer.pl
✓ Test 1: Simple RIGHT JOIN
✓ Test 2: FULL OUTER JOIN
✓ Test 3: RIGHT JOIN Chain
```

## Implementation Details

### Bug Fix

The original bug was a **parenthesis placement error** in `findall/3`:

```prolog
% Incorrect (Prolog interprets this as passing functor/2 as a separate argument):
findall(TN, member(G, List), (functor(G, TN, _)), Result)

% Correct (functor/2 is part of the goal conjunction):
findall(TN, (member(G, List), functor(G, TN, _)), Result)
```

This caused the predicate to fail with a type error.

### CROSS JOIN Detection

CROSS JOIN is detected when:
1. Multiple table goals exist in the body
2. `find_join_conditions` returns empty list (no shared variables)

### SQL Generation

Instead of generating:
```sql
FROM t1, t2, t3  ← Valid SQL but implicit
```

We now generate:
```sql
FROM t1
CROSS JOIN t2
CROSS JOIN t3    ← Explicit and clearer
```

Both are semantically equivalent, but explicit CROSS JOIN makes the Cartesian product intent clear.

## Use Cases

### Product Catalog Combinations

```prolog
% Generate all color/size combinations for a product line
product_variants(Color, Size) :-
    available_colors(_, Color),
    available_sizes(_, Size).
```

### Test Data Generation

```prolog
% Generate test combinations for integration testing
test_scenarios(Browser, OS, Resolution) :-
    browsers(_, Browser),
    operating_systems(_, OS),
    screen_resolutions(_, Resolution).
```

### Report Templates

```prolog
% All employees × all departments for org chart template
org_chart_template(Employee, Department) :-
    employees(_, Employee),
    departments(_, Department).
```

## Files Modified

### Core Implementation
- **`src/unifyweaver/targets/sql_target.pl`** (+11 lines, -3 lines)
  - Added `generate_cross_joins/2` helper (5 lines)
  - Fixed `generate_from_clause_multi/2` (6 lines changed)
  - Updated documentation

### Tests
- **`test_sql_cross_join.pl`** (new, 76 lines)
  - Test 1: Simple CROSS JOIN (2 tables)
  - Test 2: Triple CROSS JOIN (3 tables)
  - Test 3: Mixed INNER + CROSS

### Documentation
- **`proposals/CROSS_JOIN_DESIGN.md`** (new, design document)
- **`PR_CROSS_JOIN.md`** (this file)

### Debug Files
- **`test_cross_join_current.pl`** (debug script, can be removed)

## Breaking Changes

**None.** This is a pure bug fix that enables a previously broken feature.

## Related Work

- **Phase 3** (PR #172): LEFT JOIN support
- **Phase 3b** (PR #174): Nested LEFT JOINs
- **Phase 3c** (PR #175): Mixed INNER/LEFT JOINs
- **Phase 3d** (PR #176): RIGHT JOIN and FULL OUTER JOIN
- **Phase 3d Fix** (PR #177): Complex RIGHT JOIN chains
- **Phase 3e** (This PR): CROSS JOIN support

## Impact

This completes the JOIN feature set for SQL Target! All standard SQL JOIN types are now supported:

| JOIN Type | Status | Pattern |
|-----------|--------|---------|
| INNER JOIN | ✅ | Tables with shared variables |
| LEFT OUTER JOIN | ✅ | `A, (B ; null)` |
| RIGHT OUTER JOIN | ✅ | `(A ; null), B` |
| FULL OUTER JOIN | ✅ | `(A ; null), (B ; null)` |
| CROSS JOIN | ✅ | Tables with NO shared variables |

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
