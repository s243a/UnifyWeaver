# PR Description: Phase 9d - Statistical Aggregations (Go/Bbolt)

**Title:** `feat(go): implement Phase 9d - Statistical Aggregations (stddev, median, percentile)`

## Overview
This PR implements **Phase 9d** of the Go/Bbolt target, adding advanced statistical aggregation functions. These features enable powerful data analysis directly within UnifyWeaver pipelines and database queries, matching capabilities found in advanced SQL dialects and statistical packages.

## Key Features

### 1. New Aggregation Functions
- **`stddev(Field, Result)`**: Calculates the sample standard deviation.
  - Implementation: **Welford's Online Algorithm** for numerically stable, single-pass computation.
  - Memory: O(1) constant space per group.
- **`median(Field, Result)`**: Calculates the median value (50th percentile).
  - Implementation: Collects values into a slice and uses `sort.Float64s`.
  - Memory: O(N) space per group.
- **`percentile(Field, P, Result)`**: Calculates the P-th percentile (0-100).
  - Implementation: Linear interpolation between closest ranks after sorting.
  - Memory: O(N) space per group.

### 2. Deep Integration
- **Simple Aggregations:** Works with `aggregate/3`.
  ```prolog
  aggregate(stddev(Age), json_record([age-Age]), Dev).
  ```
- **Grouped Aggregations:** Works with `group_by/4` and multiple-aggregation lists.
  ```prolog
  group_by(City, Goal, [count(N), median(Age, Med), stddev(Age, Dev)]).
  ```
- **HAVING Clause:** Full support for filtering groups by statistical results.
  ```prolog
  group_by(..., stddev(Price, Volatility)), Volatility > 100.0.
  ```

### 3. Optimized Code Generation
- **Smart Imports:** Automatically detects if `math` or `sort` packages are needed and adds them to the Go imports.
- **Helper Injection:** Dynamically appends `calculateMedian` and `calculatePercentile` helper functions to the Go output only when required.

## Verification
- Verified with `tests/test_go_stats.pl`.
- Validated correctness of Welford's algorithm and sorting logic.
- Checked integration with existing `count`, `sum`, `avg` operations.
