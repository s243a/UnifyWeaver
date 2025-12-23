# Phase 9f: Window Functions - Implementation Plan

## Overview
Implement Window Functions (`row_number`, `rank`, `dense_rank`) within the aggregation framework. Unlike standard aggregations which collapse a group into a single row, window functions operate on a group of rows and return *all* of them (enriched with rank information), or a subset (Top-N).

## Goals
1.  **Row Numbering**: Assign a unique sequential integer to rows within a partition.
2.  **Ranking**: Assign ranks based on a value (with gaps for ties).
3.  **Top-N**: Efficiently filter for the top N records per group.
4.  **Integration**: Fit within the existing `group_by/4` syntax if possible, or introduce `window/4`.

## Proposed Syntax

### Option A: Overloading `group_by` (Preferred)
If a window function is detected in the aggregation list, the compiler switches to "Window Mode" (emitting multiple rows per group).

```prolog
% Rank users by age within each city
city_age_rank(City, Name, Age, Rank) :-
    group_by(City,
             json_record([city-City, name-Name, age-Age]),
             rank(Age, Rank)).
```

### Option B: Dedicated Predicate
```prolog
city_age_rank(City, Name, Age, Rank) :-
    window(City,
           json_record([city-City, name-Name, age-Age]),
           [sort_by(Age), rank(Rank)]).
```

**Decision**: Option A is more consistent with UnifyWeaver's design. We effectively treat `rank` as a special aggregation that implies "do not collapse rows".

## Supported Operations

1.  **`row_number(OrderField, Result)`**: 1, 2, 3... (arbitrary order for ties)
2.  **`rank(OrderField, Result)`**: 1, 2, 2, 4... (gaps for ties)
3.  **`dense_rank(OrderField, Result)`**: 1, 2, 2, 3... (no gaps)

*Note: `OrderField` implies ascending order. We might need `desc(Field)` later.*

## Go Implementation

### State Structure
Unlike `sum` or `count`, window functions need access to individual records.
We must accumulate the *entire record* (or at least the needed fields) in the group state.

```go
type Record struct {
    data map[string]interface{}
    // or specific typed fields if we optimize
}

type GroupStats struct {
    rows []Record
}
```

### Accumulation
```go
// Inside scanner loop
if _, exists := stats[groupKey]; !exists {
    stats[groupKey] = &GroupStats{}
}
// Store full data map for later output
stats[groupKey].rows = append(stats[groupKey].rows, Record{data: data})
```

### Output Generation
Iterate over groups, then sort and iterate over rows.

```go
for groupKey, s := range stats {
    // 1. Sort rows
    sort.Slice(s.rows, func(i, j int) bool {
        // dynamic comparison based on OrderField
        valI := s.rows[i].data["age"].(float64)
        valJ := s.rows[j].data["age"].(float64)
        return valI < valJ
    })

    // 2. Assign ranks and emit
    rank := 1
    for i, row := range s.rows {
        // Calculate rank (handle ties for rank/dense_rank)
        // ...

        // Inject rank into output
        row.data["rank"] = rank
        
        // Output
        jsonBytes, _ := json.Marshal(row.data)
        fmt.Println(string(jsonBytes))
        
        rank++
    }
}
```

## Optimization: Top-N
If the user adds a `HAVING` clause like `Rank =< 3`, we can optimize.
Instead of storing *all* rows, we can maintain a **Min-Heap** of size N.
This reduces memory from O(TotalRecords) to O(Groups * N).

## Implementation Tasks

1.  **Detection**: `is_window_function/1` predicate.
2.  **Mode Switch**: `compile_group_by_mode` must detect window functions and use a different code generator (`generate_window_group_code`).
3.  **Accumulation**: Generate code to collect `map[string]interface{}`.
4.  **Sorting**: Generate custom `sort.Slice` comparators based on the `OrderField`.
5.  **Ranking Logic**: Generate the loop to assign ranks.
6.  **Output**: Generate the loop to emit multiple JSON lines per group.

## Timeline
-   Plan & Design: 45 mins
-   Implementation: 3 hours
-   Testing: 1 hour

## Success Criteria
-   ✅ `row_number`, `rank`, `dense_rank` work.
-   ✅ Sorting works for numbers and strings.
-   ✅ Output format is correct (multiple rows per group).
