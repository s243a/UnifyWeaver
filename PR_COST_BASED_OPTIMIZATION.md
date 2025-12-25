# PR Description: Cost-Based Optimization (Go/Bbolt)

**Title:** `feat(core): implement cost-based optimization and statistics collection`

## Overview
This PR introduces the foundation for cost-based query optimization in UnifyWeaver. It shifts the optimizer from simple heuristics (variable counting) to a cost model based on cardinality and selectivity. It also adds tooling to collect these statistics from Bbolt databases.

## Key Features

### 1. Statistics Module (`core/statistics.pl`)
- **Cost Model:** Estimates execution cost based on goal type (unification < comparison < generator).
- **Storage:** Dynamic predicates to store table cardinality and field selectivity.
- **JSON Loading:** `load_stats(Path)` allows importing pre-calculated statistics from a JSON file.
- **API:** `estimate_cost/4` calculates cost for a goal given bound variables and the parent predicate context.

### 2. Cost-Based Reordering (`core/optimizer.pl`)
- **Enhanced Comparator:** Replaced the simple "count bound variables" heuristic with `statistics:estimate_cost/4`.
- **Context Awareness:** The optimizer passes the current predicate indicator down to the cost model for accurate table-specific lookups.
- **Benefit:** Goals are now ordered by estimated cost, preferring cheap filters and highly selective lookups based on real data distribution.

### 3. Statistics Collection (`db_mode(analyze)`)
- **New Mode:** `compile_predicate_to_go(..., [db_mode(analyze)], Code)`.
- **Function:** Generates a standalone Go program that scans a Bbolt bucket.
- **Metrics:** Counts total records and unique values for each field.
- **Output:** Saves a JSON statistics object to a reserved `_stats` bucket in the database.

## Verification
- Verified with `tests/test_go_analyze.pl` (created and deleted).
- Confirmed correct Go code generation for statistics gathering.
