# Proposal: Cost-Based Query Optimization (Go/Bbolt)

**Status:** In Progress
**Author:** John William Creighton (@s243a)
**Date:** 2025-12-23

## Executive Summary
This proposal introduces cost-based optimization to the UnifyWeaver compiler. By collecting and utilizing statistics about data distribution (cardinality and selectivity), the compiler can make intelligent decisions about join ordering and index selection, significantly outperforming simple heuristic-based approaches.

## Motivation
UnifyWeaver currently uses a "count bound variables" heuristic to order goals. This is insufficient for real-world data where field selectivity varies wildly. For example, a filter `Gender = 'Male'` might match 50% of records, while `SSN = '...'` matches 1 record. Treating them equally leads to suboptimal query plans.

## Goals
1.  **Statistics Collection**: Provide a standard way to analyze Bbolt databases and store metadata (cardinality, unique value counts).
2.  **Cost Estimation**: Implement a cost model `Cost = Cardinality * Selectivity` to estimate the result size of any goal.
3.  **Dynamic Reordering**: Update the optimizer to sort goals by estimated cost.

## Architecture

### 1. The Statistics Module (`core/statistics.pl`)
A central registry for predicate statistics.
- `declare_stats(Pred, Stats)`: Registers stats for a predicate.
- `estimate_cost(Goal, BoundVars, Cost)`: Calculates cost.

### 2. The Analyzer Tool (`db_mode(analyze)`)
A new compilation mode for the Go target.
- **Input**: A Prolog predicate mapped to a Bbolt bucket.
- **Output**: A standalone Go binary.
- **Behavior**:
    1.  Scans the entire bucket.
    2.  Counts total records ($N$).
    3.  Uses HyperLogLog (or exact sets for small data) to count unique values ($U$) for each field.
    4.  Calculates Selectivity $S = 1/U$.
    5.  Saves `{ "cardinality": N, "fields": { "field": { "selectivity": S } } }` to `_stats` bucket.

### 3. Optimizer Integration (`core/optimizer.pl`)
The "Codd Phase" optimizer is updated to use `statistics:estimate_cost/3`.

**Cost Formula:**
$$ Cost(G) = N_{rows} \times \prod_{f \in BoundFields} S_f $$

Where:
- $N_{rows}$ is the table cardinality.
- $S_f$ is the selectivity of field $f$.
- If stats are missing, fallback to $10000 / (1 + NumBoundVars)$.

## Implementation Plan

### Phase 1: Foundation (Completed)
- [x] Create `statistics.pl`.
- [x] Update `optimizer.pl` to use cost model.
- [x] Implement `db_mode(analyze)` in Go target.

### Phase 2: Integration (Next Steps)
- [ ] **Stats Loading**: Add a mechanism for `go_target.pl` to read the `_stats` bucket (or a JSON export of it) *during compilation* to populate the `statistics` module.
    - *Approach*: Since the compiler runs in Prolog (host) and the DB is Bbolt (target/binary), we likely need the user to export stats to a JSON file first, then load that file during compilation.
    - *Directive*: `:- load_stats('stats.json').`

- [ ] **Refinement**: Improve the cost formula for range queries (`>`,`<`) using histograms if possible (or default selectivity like 0.33).

## Usage Example

1.  **Generate Analyzer**:
    ```prolog
    compile_predicate_to_go(user/2, [db_backend(bbolt), db_mode(analyze)], Code).
    ```
2.  **Run Analysis**:
    ```bash
    ./user_analyzer
    # Output: Statistics saved to _stats bucket.
    ```
3.  **Export Stats** (Manual step for now):
    ```bash
    # Tool to dump _stats bucket to stats.json
    ```
4.  **Compile with Stats**:
    ```prolog
    :- load_stats('stats.json').
    query(X) :- ...
    ```
