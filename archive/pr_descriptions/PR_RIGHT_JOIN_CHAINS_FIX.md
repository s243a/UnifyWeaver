# Fix Complex RIGHT JOIN Chains (Phase 3d Fix)

## Summary

Fixes the **known limitation** from Phase 3d where RIGHT JOIN chains with multiple disjunctions didn't compile correctly.

### What's Fixed

✅ **RIGHT JOIN chains** - Pattern `(A ; null), (B ; null), C` now generates correct SQL
✅ **Multiple disjunctions** - Recursively handles all leading disjunctions
✅ **Backwards compatible** - All previous tests continue to pass

## Problem

**Before this fix**, the pattern `(t1 ; null), (t2 ; null), t3` generated incomplete SQL:

```sql
SELECT t1.a, unknown, t3.c    ← "unknown" instead of t2.b
FROM t1
RIGHT JOIN t3;                 ← Missing t2!
```

**Root cause:** `compile_right_join_clause/6` only extracted the FIRST disjunction, treating Rest as a list of simple goals. When Rest contained another disjunction `(t2 ; null)`, it couldn't handle it.

## Solution

Adopted the same approach used by LEFT JOIN: **extract ALL leading disjunctions** before processing.

### New Predicates

#### 1. `extract_leading_disjunctions/3`

Extracts all disjunctions that appear before any non-disjunction goal:

```prolog
%% Pattern: (t1 ; null), (t2 ; null), t3
extract_leading_disjunctions(Body, LeadingDisjunctions, RemainingGoals)
% → LeadingDisjunctions = [(t1 ; null), (t2 ; null)]
% → RemainingGoals = [t3]
```

#### 2. `process_right_joins/5`

Iteratively processes goals (both disjunctions and regular tables) to generate RIGHT JOIN clauses:

```prolog
process_right_joins([Goal|Rest], AccTables, [JoinClause|RestJoins], [Table|RestTables], AllNullVars) :-
    % Handle both (table ; null) and regular table goals
    (   Goal = (TableGoal ; Fallback)
    ->  extract_null_bindings(Fallback, NullVars),
        Table = TableGoal
    ;   NullVars = [],
        Table = Goal
    ),
    generate_right_join_sql(AccTables, Table, JoinClause),
    append(AccTables, [Table], NewAccTables),
    process_right_joins(Rest, NewAccTables, RestJoins, RestTables, RestNullVars),
    append(NullVars, RestNullVars, AllNullVars).
```

### Updated Predicate

#### `compile_right_join_clause/6`

Now uses the new approach:

```prolog
compile_right_join_clause(Name, Arity, Body, Head, Options, SQLCode) :-
    % Extract ALL leading disjunctions
    extract_leading_disjunctions(Body, LeadingDisjunctions, RemainingGoals),
    LeadingDisjunctions \= [],

    % First disjunction = FROM table
    LeadingDisjunctions = [FirstDisj|RestDisjs],
    FirstDisj = (FirstTable ; FirstFallback),
    extract_null_bindings(FirstFallback, FirstNullVars),
    generate_from_clause([FirstTable], FromClause),

    % Combine remaining disjunctions + regular goals
    append(RestDisjs, RemainingGoals, AllRightGoals),

    % Process all to generate RIGHT JOINs
    process_right_joins(AllRightGoals, [FirstTable], JoinClauses, AllRightTables, RestNullVars),

    % ... generate SELECT and combine ...
```

## Test Results

### Test 1: Simple RIGHT JOIN ✅ (Still Works)

**Pattern:**
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

### Test 2: FULL OUTER JOIN ✅ (Still Works)

**Pattern:**
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

### Test 3: RIGHT JOIN Chain ✅ (FIXED!)

**Pattern:**
```prolog
right_chain(A, B, C) :-
    (t1(X, A) ; A = null),
    (t2(Y, X, B) ; B = null, X = null),
    t3(_, Y, C).
```

**Before (broken):**
```sql
SELECT t1.a, unknown, t3.c    ← Wrong!
FROM t1
RIGHT JOIN t3;                 ← Missing t2!
```

**After (fixed):**
```sql
SELECT t1.a, t2.b, t3.c        ← Correct!
FROM t1
RIGHT JOIN t2 ON t2.x = t1.x   ← t2 included!
RIGHT JOIN t3 ON t3.y = t2.y;
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

## Implementation Details

### Algorithm Flow

1. **Extract leading disjunctions:**
   - `(t1 ; null), (t2 ; null), t3`
   - → `[(t1 ; null), (t2 ; null)]` + `[t3]`

2. **First disjunction = FROM:**
   - `FROM t1`

3. **Process remaining (disjunctions + regular tables):**
   - `[(t2 ; null), t3]` → Generate RIGHT JOIN for each
   - `RIGHT JOIN t2 ON ...`
   - `RIGHT JOIN t3 ON ...`

4. **Generate SELECT:**
   - Include all tables: `t1.a, t2.b, t3.c`
   - Mark NULL-able columns from disjunction fallbacks

### Pattern Matching

`extract_leading_disjs_iter/4` recursively extracts disjunctions:

```prolog
extract_leading_disjs_iter((Disj, Rest), Acc, AllDisjs, Remaining) :-
    Disj = (_ ; _),  % Is a disjunction
    !,
    extract_leading_disjs_iter(Rest, [Disj|Acc], AllDisjs, Remaining).

extract_leading_disjs_iter(Rest, Acc, AllDisjs, Remaining) :-
    % Hit a non-disjunction
    Acc \= [],
    reverse(Acc, AllDisjs),
    conjunction_to_list(Rest, Remaining).
```

## Files Modified

### Core Implementation
- **`src/unifyweaver/targets/sql_target.pl`** (+61 lines, -35 lines)
  - Added `extract_leading_disjunctions/3`
  - Added `extract_leading_disjs_iter/4`
  - Added `process_right_joins/5`
  - Updated `compile_right_join_clause/6` to use new approach
  - Removed obsolete `generate_right_join_chain/3` (replaced by `process_right_joins/5`)

### Documentation
- **`proposals/RIGHT_JOIN_CHAINS_FIX.md`** (new, design document)
- **`PR_RIGHT_JOIN_CHAINS_FIX.md`** (this file)

### Debug Tools
- **`debug_right_chain.pl`** (new, debug script used during development)

## Breaking Changes

**None.** This is a pure bug fix that completes the Phase 3d implementation.

## Related Work

- **Phase 3** (PR #172): LEFT JOIN support
- **Phase 3b** (PR #174): Nested LEFT JOINs
- **Phase 3c** (PR #175): Mixed INNER/LEFT JOINs
- **Phase 3d** (PR #176): RIGHT JOIN and FULL OUTER JOIN
- **Phase 3d Fix** (This PR): Complete RIGHT JOIN chain support

## Impact

This fix completes the RIGHT JOIN feature set, removing the documented limitation. Users can now write complex RIGHT JOIN chains with multiple disjunctions, enabling more expressive queries.

### Example Use Case

```prolog
% Analytics: Show all transactions, with optional customer and account info
all_transactions(TxId, CustomerName, AccountType) :-
    (customers(CustId, CustomerName, _) ; CustomerName = null),
    (accounts(AcctId, CustId, AccountType) ; AccountType = null),
    transactions(TxId, AcctId, _, _).  % All transactions preserved
```

**Generates:**
```sql
SELECT transactions.id, customers.name, accounts.type
FROM customers
RIGHT JOIN accounts ON accounts.customer_id = customers.id
RIGHT JOIN transactions ON transactions.account_id = accounts.id;
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
