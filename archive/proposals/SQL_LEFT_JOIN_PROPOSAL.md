# SQL Target: LEFT JOIN via Prolog Disjunction

## Proposal Summary

Implement LEFT JOIN support in the SQL target by detecting and compiling Prolog disjunction patterns that naturally express outer join semantics.

**Status:** Proposal (Phase 3)
**Author:** Claude (via Claude Code)
**Date:** 2025-12-04

## Motivation

Advanced JOINs (LEFT, RIGHT, FULL OUTER) are essential SQL features but challenging to express in Prolog due to NULL semantics. This proposal uses **native Prolog disjunction** (`;`) to express "try this, or fallback" semantics that map directly to LEFT JOIN.

## Proposed Syntax

### Basic Pattern

```prolog
% LEFT JOIN using disjunction
customer_orders(Name, Product) :-
    customers(CustomerId, Name, _),
    ( orders(_, CustomerId, Product, _)    % Try to find matching orders
    ; Product = null                        % Fallback: bind Product to null
    ).
```

**SQL Output:**
```sql
CREATE VIEW customer_orders AS
SELECT customers.name, orders.product
FROM customers
LEFT JOIN orders ON customers.customer_id = orders.customer_id;
```

### Multiple NULL Bindings

```prolog
% Multiple columns from right table
customer_order_details(Name, Product, Amount) :-
    customers(CustomerId, Name, _),
    ( orders(_, CustomerId, Product, Amount)
    ; Product = null, Amount = null         % Bind all right-table variables
    ).
```

### Nested LEFT JOINs

```prolog
% Chain multiple LEFT JOINs
full_customer_data(Name, Product, Discount) :-
    customers(CustomerId, Name, _),
    ( orders(OrderId, CustomerId, Product, _)
    ; Product = null, OrderId = null
    ),
    ( discounts(OrderId, Discount)
    ; Discount = null
    ).
```

**SQL Output:**
```sql
SELECT customers.name, orders.product, discounts.discount
FROM customers
LEFT JOIN orders ON customers.customer_id = orders.customer_id
LEFT JOIN discounts ON orders.order_id = discounts.order_id;
```

## Pattern Detection Algorithm

### 1. Identify Disjunction Pattern

```
Pattern: LeftGoals, (RightGoal ; Fallback)

Where:
- LeftGoals: Goals that bind the join key(s)
- RightGoal: Goal that uses the join key(s) from a different table
- Fallback: Binds right-table variables to null
```

### 2. Validate Join Relationship

```prolog
is_left_join_pattern(LeftGoals, RightGoal, Fallback) :-
    % 1. Extract join keys from LeftGoals
    extract_bound_vars(LeftGoals, BoundVars),

    % 2. Verify RightGoal uses those keys
    goal_uses_vars(RightGoal, BoundVars),

    % 3. Verify RightGoal is from different table
    \+ same_table(LeftGoals, RightGoal),

    % 4. Verify Fallback binds right-table variables to null
    extracts_null_bindings(Fallback, NullVars),
    vars_in_goal(RightGoal, RightVars),
    subset(NullVars, RightVars).
```

### 3. Generate LEFT JOIN SQL

```prolog
compile_left_join(LeftGoals, RightGoal, SQL) :-
    % Generate FROM clause from left table
    generate_from_clause(LeftGoals, FromClause),

    % Generate LEFT JOIN clause
    extract_table_and_vars(RightGoal, RightTable, RightVars),
    infer_join_condition(LeftGoals, RightGoal, JoinCondition),
    format(string(JoinClause), 'LEFT JOIN ~w ON ~w',
           [RightTable, JoinCondition]),

    % Combine
    format(string(SQL), '~w~n~w', [FromClause, JoinClause]).
```

## Implementation Plan

### Phase 3a: Basic LEFT JOIN (Week 1)

**Scope:**
- Detect single LEFT JOIN pattern
- Compile to SQL LEFT JOIN
- Handle single fallback (`Y = null`)

**Deliverables:**
- Pattern detection in `parse_clause_body/2`
- LEFT JOIN generation in `generate_join_clauses/3`
- Test suite with 5 basic LEFT JOIN tests

### Phase 3b: Multiple Columns (Week 2)

**Scope:**
- Support multiple NULL bindings
- Handle complex fallback patterns
- Validate all right-table variables are bound

**Deliverables:**
- Enhanced fallback parsing
- Multi-column NULL binding tests
- Edge case handling (missing bindings, wrong order)

### Phase 3c: Nested LEFT JOINs (Week 3)

**Scope:**
- Chain multiple LEFT JOINs
- Preserve join order
- Handle dependencies between joins

**Deliverables:**
- Nested pattern detection
- Join chain generation
- Complex multi-table test cases

## Examples

### Example 1: Basic LEFT JOIN

**Input:**
```prolog
:- sql_table(customers, [id-integer, name-text]).
:- sql_table(orders, [id-integer, customer_id-integer, product-text]).

customer_orders(Name, Product) :-
    customers(CustomerId, Name),
    ( orders(_, CustomerId, Product)
    ; Product = null
    ).
```

**Output:**
```sql
CREATE VIEW customer_orders AS
SELECT customers.name, orders.product
FROM customers
LEFT JOIN orders ON customers.id = orders.customer_id;
```

**Test Data:**
```
Customers: Alice(1), Bob(2), Charlie(3)
Orders: Alice→Widget, Alice→Gadget, Bob→Gizmo

Expected Results:
Alice   | Widget
Alice   | Gadget
Bob     | Gizmo
Charlie | NULL     ← LEFT JOIN preserves Charlie
```

### Example 2: Multi-Column LEFT JOIN

**Input:**
```prolog
customer_totals(Name, Product, Amount) :-
    customers(CustomerId, Name),
    ( orders(_, CustomerId, Product, Amount)
    ; Product = null, Amount = null
    ).
```

**Output:**
```sql
CREATE VIEW customer_totals AS
SELECT customers.name, orders.product, orders.amount
FROM customers
LEFT JOIN orders ON customers.id = orders.customer_id;
```

### Example 3: Chained LEFT JOINs

**Input:**
```prolog
customer_shipments(Name, Product, Tracking) :-
    customers(CustomerId, Name),
    ( orders(OrderId, CustomerId, Product)
    ; Product = null, OrderId = null
    ),
    ( shipments(OrderId, Tracking)
    ; Tracking = null
    ).
```

**Output:**
```sql
CREATE VIEW customer_shipments AS
SELECT customers.name, orders.product, shipments.tracking
FROM customers
LEFT JOIN orders ON customers.id = orders.customer_id
LEFT JOIN shipments ON orders.id = shipments.order_id;
```

## Advantages

### 1. Pure Prolog Syntax
- No new language constructs
- No annotations or magic predicates
- Standard Prolog disjunction

### 2. Natural Semantics
- Disjunction (`X ; Y`) naturally expresses "try X, or do Y"
- Explicit NULL binding makes intent clear
- Fallback handling is first-class

### 3. Compile-Time Detection
- Pattern is structurally detectable
- No runtime overhead in Prolog
- Clean compilation to SQL

### 4. Composable
- Works with existing INNER JOIN detection
- Chains naturally for multiple LEFT JOINs
- Mixes with WHERE clauses and aggregations

## Challenges and Solutions

### Challenge 1: Disjunction Ambiguity

**Problem:**
```prolog
% Is this a LEFT JOIN or just alternative logic?
result(X, Y) :- table1(X), (table2(X, Y) ; Y = 0).
```

**Solution:**
Only treat as LEFT JOIN when:
1. Right goal accesses a table
2. Fallback binds variables to `null` (not other values)
3. Join relationship can be inferred

### Challenge 2: Variable Binding Validation

**Problem:**
```prolog
% Forgot to bind Amount
customer_orders(Name, Product, Amount) :-
    customers(CustomerId, Name),
    ( orders(_, CustomerId, Product, Amount)
    ; Product = null                          % Missing: Amount = null
    ).
```

**Solution:**
Compile-time validation:
- Extract all variables from right goal
- Verify all are bound in fallback
- Error if any are missing

### Challenge 3: Complex Fallback Patterns

**Problem:**
```prolog
% Which bindings matter?
result(X, Y, Z) :-
    table1(X),
    ( table2(X, Y, Z)
    ; Y = null, Z = null, write('No match'), true
    ).
```

**Solution:**
- Parse fallback as conjunction
- Extract only variable bindings (ignore side effects)
- Warn about non-binding goals in fallback

## Testing Strategy

### Unit Tests
1. Pattern detection tests
2. Join condition inference tests
3. NULL binding extraction tests

### Integration Tests
1. Single LEFT JOIN with 1 column
2. Multi-column LEFT JOIN
3. Chained LEFT JOINs (2-3 tables)
4. Mixed INNER + LEFT JOIN
5. LEFT JOIN + WHERE clause
6. LEFT JOIN + GROUP BY

### SQL Validation
- Generate SQL and execute in SQLite
- Verify NULL values appear correctly
- Compare row counts with expected results

## Future Extensions

### RIGHT JOIN
```prolog
% RIGHT JOIN: All orders, even without customers
order_customers(Product, Name) :-
    orders(_, CustomerId, Product),
    ( customers(CustomerId, Name)
    ; Name = null
    ).
```

### FULL OUTER JOIN
```prolog
% FULL OUTER: All customers and all orders
full_data(Name, Product) :-
    ( customers(CustomerId, Name)
    ; Name = null, CustomerId = null
    ),
    ( orders(_, CustomerId, Product)
    ; Product = null
    ).
```

### Conditional Fallbacks
```prolog
% Use different defaults based on conditions
customer_orders(Name, Product) :-
    customers(CustomerId, Name, Region),
    ( orders(_, CustomerId, Product)
    ; Region = 'EU', Product = 'N/A'
    ; Product = null
    ).
```

## Success Criteria

**Phase 3a Complete:**
- ✅ Detect single LEFT JOIN pattern
- ✅ Generate correct SQL LEFT JOIN
- ✅ 5/5 basic tests passing
- ✅ SQLite integration validates NULLs

**Phase 3b Complete:**
- ✅ Multi-column NULL bindings work
- ✅ 10/10 tests passing (including edge cases)
- ✅ Validation errors for incomplete bindings

**Phase 3c Complete:**
- ✅ Nested LEFT JOINs work
- ✅ 15/15 tests passing (complex scenarios)
- ✅ Documentation with examples

## Open Questions

1. **Error handling**: Should incomplete fallbacks error or warn?
2. **Performance**: In pure Prolog (non-compiled), should we optimize disjunction?
3. **Syntax sugar**: Should we add a `left_join/3` helper for convenience?
4. **Mixed semantics**: How to handle LEFT JOIN + aggregation in same query?

## References

- Current SQL Target: `src/unifyweaver/targets/sql_target.pl`
- SQL-92 LEFT JOIN spec
- Prolog disjunction semantics (ISO Prolog standard)
