# Phase 9: Aggregations - Implementation Plan

## Overview

Add support for aggregation operations (`sum`, `count`, `avg`, `max`, `min`) to the Go target, working with both database queries and stream processing.

## Goals

1. **Simple Aggregations**: Count, sum, average, max, min over entire dataset
2. **Grouped Aggregations**: Aggregate by grouping fields (like SQL GROUP BY)
3. **Database Integration**: Work with bbolt database queries
4. **Stream Processing**: Work with JSONL input streams
5. **Performance**: Efficient single-pass aggregation

## Prolog Syntax Design

### Option 1: Aggregate Predicate (SQL-like)

```prolog
% Simple count
user_count(Count) :-
    aggregate(count, json_record([name-_Name]), Count).

% Sum with field
total_age(Sum) :-
    aggregate(sum(Age), json_record([age-Age]), Sum).

% Average
avg_age(Avg) :-
    aggregate(avg(Age), json_record([age-Age]), Avg).

% Max/Min
oldest(MaxAge) :-
    aggregate(max(Age), json_record([age-Age]), MaxAge).
```

### Option 2: Dedicated Predicates (Simpler)

```prolog
% Count all records
user_count(Count) :-
    count_records(json_record([name-_]), Count).

% Sum field values
total_age(Sum) :-
    sum_field(age, json_record([age-Age]), Sum).

% Average
avg_age(Avg) :-
    avg_field(age, json_record([age-Age]), Avg).
```

### Option 3: Group By Support (Most Powerful)

```prolog
% Count users by city
city_counts(City, Count) :-
    group_by(City, json_record([city-City, name-_]), count, Count).

% Average age by city
city_avg_age(City, AvgAge) :-
    group_by(City, json_record([city-City, age-Age]), avg(Age), AvgAge).

% Multiple aggregations
city_stats(City, Count, AvgAge, MaxAge) :-
    group_by(City,
             json_record([city-City, age-Age]),
             [count(Count), avg(Age, AvgAge), max(Age, MaxAge)]).
```

**Recommendation**: Start with **Option 1** (aggregate predicate) as it's most SQL-like and extensible. Add group_by in Phase 9b.

## Generated Go Code Patterns

### Count Aggregation

**Prolog**:
```prolog
user_count(Count) :-
    aggregate(count, json_record([name-_Name]), Count).
```

**Generated Go**:
```go
package main

import (
    "encoding/json"
    "fmt"
    "os"
    bolt "go.etcd.io/bbolt"
)

func main() {
    db, err := bolt.Open("users.db", 0600, &bolt.Options{ReadOnly: true})
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
    defer db.Close()

    count := 0
    err = db.View(func(tx *bolt.Tx) error {
        bucket := tx.Bucket([]byte("users"))
        if bucket == nil {
            return fmt.Errorf("bucket not found")
        }

        return bucket.ForEach(func(k, v []byte) error {
            var data map[string]interface{}
            if err := json.Unmarshal(v, &data); err != nil {
                return nil // Skip invalid records
            }

            // Check if name field exists
            if _, ok := data["name"]; ok {
                count++
            }
            return nil
        })
    })

    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }

    // Output result
    fmt.Println(count)
}
```

### Sum Aggregation

**Prolog**:
```prolog
total_age(Sum) :-
    aggregate(sum(Age), json_record([age-Age]), Sum).
```

**Generated Go**:
```go
sum := 0.0
count := 0

err = db.View(func(tx *bolt.Tx) error {
    bucket := tx.Bucket([]byte("users"))
    return bucket.ForEach(func(k, v []byte) error {
        var data map[string]interface{}
        json.Unmarshal(v, &data)

        // Extract age field
        if ageRaw, ok := data["age"]; ok {
            if ageFloat, ok := ageRaw.(float64); ok {
                sum += ageFloat
                count++
            }
        }
        return nil
    })
})

fmt.Println(sum)
```

### Average Aggregation

**Prolog**:
```prolog
avg_age(Avg) :-
    aggregate(avg(Age), json_record([age-Age]), Avg).
```

**Generated Go**:
```go
sum := 0.0
count := 0

err = db.View(func(tx *bolt.Tx) error {
    bucket := tx.Bucket([]byte("users"))
    return bucket.ForEach(func(k, v []byte) error {
        var data map[string]interface{}
        json.Unmarshal(v, &data)

        if ageRaw, ok := data["age"]; ok {
            if ageFloat, ok := ageRaw.(float64); ok {
                sum += ageFloat
                count++
            }
        }
        return nil
    })
})

avg := 0.0
if count > 0 {
    avg = sum / float64(count)
}

fmt.Println(avg)
```

### Max/Min Aggregation

**Prolog**:
```prolog
max_age(Max) :-
    aggregate(max(Age), json_record([age-Age]), Max).
```

**Generated Go**:
```go
maxAge := 0.0
first := true

err = db.View(func(tx *bolt.Tx) error {
    bucket := tx.Bucket([]byte("users"))
    return bucket.ForEach(func(k, v []byte) error {
        var data map[string]interface{}
        json.Unmarshal(v, &data)

        if ageRaw, ok := data["age"]; ok {
            if ageFloat, ok := ageRaw.(float64); ok {
                if first || ageFloat > maxAge {
                    maxAge = ageFloat
                    first = false
                }
            }
        }
        return nil
    })
})

fmt.Println(maxAge)
```

## Implementation Plan

### Phase 9a: Simple Aggregations

**Step 1**: Detection
- Add `is_aggregation_predicate/1` to detect aggregation patterns
- Recognize `aggregate(Op, Goal, Result)` syntax
- Extract aggregation operation, field, and result variable

**Step 2**: Code Generation
- Add `generate_aggregation_code/4` for each operation:
  - `generate_count_aggregation/3`
  - `generate_sum_aggregation/4`
  - `generate_avg_aggregation/4`
  - `generate_max_aggregation/4`
  - `generate_min_aggregation/4`

**Step 3**: Integration
- Modify `compile_database_read_mode/4` to detect aggregations
- Route to aggregation code generator when detected
- Support both database and JSONL input modes

**Step 4**: Testing
- Test each aggregation type with sample data
- Verify correct results
- Test edge cases (empty dataset, null values, type mismatches)

### Phase 9b: Grouped Aggregations (Future)

**Features**:
- `group_by/4` predicate for grouping
- Multiple aggregations per group
- Output one result per group

**Example**:
```prolog
city_stats(City, Count, Avg) :-
    group_by(City,
             json_record([city-City, age-Age]),
             [count(Count), avg(Age, Avg)]).
```

**Generated Go**:
```go
type GroupStats struct {
    sum   float64
    count int
}

groups := make(map[string]*GroupStats)

// Accumulate
bucket.ForEach(func(k, v []byte) error {
    city := data["city"].(string)
    age := data["age"].(float64)

    if _, ok := groups[city]; !ok {
        groups[city] = &GroupStats{}
    }
    groups[city].sum += age
    groups[city].count++
    return nil
})

// Output
for city, stats := range groups {
    avg := stats.sum / float64(stats.count)
    output := map[string]interface{}{
        "city": city,
        "count": stats.count,
        "avg": avg,
    }
    json.Marshal(output)
    fmt.Println(string(output))
}
```

## File Locations

**Core Implementation**: `src/unifyweaver/targets/go_target.pl`

New predicates to add:
- `is_aggregation_predicate/1` (~line 1700)
- `extract_aggregation_spec/3` (~line 1720)
- `generate_aggregation_code/4` (~line 2500)
- `generate_count_aggregation/3` (~line 2520)
- `generate_sum_aggregation/4` (~line 2550)
- `generate_avg_aggregation/4` (~line 2580)
- `generate_max_aggregation/4` (~line 2610)
- `generate_min_aggregation/4` (~line 2640)

**Tests**: `test_phase_9.pl`

Test cases:
- Count all records
- Sum numeric field
- Average numeric field
- Max value
- Min value
- Empty dataset
- Null values
- Type mismatches

## Example Use Cases

### Use Case 1: User Statistics

```prolog
% Count total users
total_users(Count) :-
    aggregate(count, json_record([name-_]), Count).

% Total age of all users
total_age(Sum) :-
    aggregate(sum(Age), json_record([age-Age]), Sum).

% Average user age
average_age(Avg) :-
    aggregate(avg(Age), json_record([age-Age]), Avg).

% Oldest user
oldest_age(Max) :-
    aggregate(max(Age), json_record([age-Age]), Max).

% Youngest user
youngest_age(Min) :-
    aggregate(min(Age), json_record([age-Age]), Min).
```

### Use Case 2: Sales Analytics

```prolog
% Total revenue
total_revenue(Sum) :-
    aggregate(sum(Amount), json_record([amount-Amount]), Sum).

% Average transaction value
avg_transaction(Avg) :-
    aggregate(avg(Amount), json_record([amount-Amount]), Avg).

% Largest sale
max_sale(Max) :-
    aggregate(max(Amount), json_record([amount-Amount]), Max).
```

### Use Case 3: Log Analysis

```prolog
% Total errors
error_count(Count) :-
    aggregate(count,
              (json_record([level-Level]), Level = "ERROR"),
              Count).

% Average response time
avg_response_time(Avg) :-
    aggregate(avg(ResponseTime),
              json_record([response_time-ResponseTime]),
              Avg).
```

## Performance Considerations

**Single-Pass Aggregation**:
- All aggregations computed in one database scan
- O(n) complexity where n = number of records
- Minimal memory usage (only accumulator variables)

**Optimization Opportunities**:
- For count-only queries, could use bucket.Stats() for instant results
- For indexed fields, could use key optimization from Phase 8c
- Group-by could use maps for efficient grouping

## Error Handling

**Type Mismatches**:
- Skip records with wrong field types
- Don't fail on individual records
- Log warnings for debugging

**Empty Datasets**:
- count returns 0
- sum returns 0
- avg returns 0 (or null/error?)
- max/min return null/error (no valid value)

**Null Values**:
- Skip null values in aggregations
- Don't count nulls in averages
- Treat as missing data

## Testing Strategy

1. **Unit Tests**: Test each aggregation type individually
2. **Integration Tests**: Test with real database
3. **Edge Cases**: Empty data, nulls, type mismatches
4. **Performance Tests**: Large datasets (100K+ records)

## Timeline

- **Phase 9a** (Simple Aggregations): 2-3 hours
  - Detection logic: 30min
  - Code generation: 1.5 hours
  - Testing: 1 hour

- **Phase 9b** (Group By): 2-3 hours (future)
  - Group detection: 30min
  - Map-based grouping: 1 hour
  - Multiple aggregations: 1 hour
  - Testing: 30min

## Success Criteria

✅ Count aggregation works correctly
✅ Sum aggregation works correctly
✅ Average aggregation works correctly
✅ Max aggregation works correctly
✅ Min aggregation works correctly
✅ Works with database queries
✅ Works with JSONL streams
✅ Handles edge cases gracefully
✅ Well-tested and documented

## References

- SQL Aggregation Functions: Standard reference for expected behavior
- Phase 8a/8b/8c: Existing database query infrastructure
- bbolt documentation: Database iteration patterns
