<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 UnifyWeaver Contributors
-->
# Goal Optimizer (The "Codd" Phase)

## Overview

The Goal Optimizer is a compilation phase that reorders Prolog rule bodies to improve execution efficiency. It mimics some behaviors of relational database query optimizers by pushing filters down and prioritizing selective unifications.

The phase is named the **"Codd Phase"** in honor of Edgar F. Codd, the father of the relational model.

## Why Optimize?

In standard Prolog, the order of goals in a body defines the execution order. A poorly ordered rule can lead to massive intermediate result sets:

```prolog
% Inefficient: Iterates all users, then filters by ID
find_user(ID, Name) :- 
    json_record([id-ID, name-Name]), 
    ID = "u123".
```

The optimizer transforms this into:

```prolog
% Efficient: Constrains ID first, then looks up the record
find_user(ID, Name) :- 
    ID = "u123", 
    json_record([id-ID, name-Name]).
```

## Optimization Heuristics

The optimizer uses a greedy greedy algorithm to select the next "best" executable goal. A goal is **executable** (ready) if all its input variable dependencies are met.

### Priority Ranking (High to Low)

1.  **Ground Unification (`X = literal`)**: These are the strongest filters and provide initial bindings for other goals.
2.  **Comparisons (`X > 100`, `A \= B`)**: These act as filters. They are prioritized as soon as their variables are bound to reduce the data set size.
3.  **Assignments (`X is Expr`)**: Necessary for computing values used in later steps.
4.  **Generators with Bindings**: Goals like `parent(X, Y)` where `X` is already bound. These are highly selective.
5.  **Pure Generators**: Goals where no variables are yet bound. These are the last resort as they produce the largest initial sets.

## Constraints and Safety

### Ordering Independence (`unordered`)

Optimization only occurs if the predicate is declared as `unordered(true)`.

*   **Default:** `unordered(true)` (Optimizer active).
*   **Ordered Data:** If order matters (e.g., temporal logs), use `:- constraint(my_pred/2, [ordered]).` to disable reordering.

### Variable Identity (==)

The optimizer tracks "Bound Variables" using identity (`==`) rather than unification (`=`). This prevents the optimizer from accidentally unifying variables during its analysis phase.

## Implementation Details

*   **Module:** `src/unifyweaver/core/optimizer.pl`
*   **Logic:**
    *   Extracts goals into a list.
    *   Identifies variables provided by the `Head` (inputs).
    *   Iteratively selects the highest-priority "ready" goal.
    *   Maintains a set of "currently bound variables" to satisfy dependencies of remaining goals.

## Target Integration

The optimizer is integrated at the core level and affects the following targets:

*   **Bash Target:** Affects pipeline order and procedural script blocks.
*   **Go Target:** Optimizes `json_input` mode and recursive steps.
*   **Rust Target:** Optimizes `json_input` mode.
*   **Recursive Compiler:** Optimizes bodies before pattern classification.

## Future Work

*   **Selectivity Estimation:** Use constant values to estimate which generator is more restrictive.
*   **Index Hints:** Use `:- index/2` directives to prioritize goals that can use indexed lookups (e.g. in Bbolt).
*   **Mode Analysis:** Better detection of input vs. output variables in generic predicates.
