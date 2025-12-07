# Phase 9c: Advanced Aggregation Features

## Overview

Extend Phase 9 (Simple & GROUP BY aggregations) with advanced features for more powerful analytics.

## Goals

1. **Multiple Aggregations**: Compute multiple aggregations in one query
2. **HAVING Clause**: Filter groups by aggregation results
3. **Nested Grouping**: Group by multiple fields
4. **Statistical Functions**: stddev, median, percentile
5. **Performance**: Maintain single-pass efficiency

## Priority Features (Phase 9c)

### Feature 1: Multiple Aggregations per Query (HIGHEST PRIORITY)

**Problem**: Currently must write separate queries for each aggregation.

**Current (Phase 9b)**:
```prolog
% Need 3 separate predicates
city_count(City, Count) :-
    group_by(City, json_record([city-City, age-Age]), count, Count).

city_avg(City, Avg) :-
    group_by(City, json_record([city-City, age-Age]), avg(Age), Avg).

city_max(City, Max) :-
    group_by(City, json_record([city-City, age-Age]), max(Age), Max).
```

**Desired (Phase 9c)**:
```prolog
% Single predicate with multiple aggregations
city_stats(City, Count, AvgAge, MaxAge) :-
    group_by(City,
             json_record([city-City, age-Age]),
             [count(Count), avg(Age, AvgAge), max(Age, MaxAge)]).
```

**Benefits**:
- Single database scan for multiple metrics
- Cleaner, more readable code
- Matches SQL SELECT capabilities
- More efficient (one pass vs multiple passes)

**Generated Go Code**:
```go
type GroupStats struct {
    count  int
    sum    float64  // for avg
    maxAge float64
    maxFirst bool
}
stats := make(map[string]*GroupStats)

// Single pass accumulation
err = db.View(func(tx *bolt.Tx) error {
    bucket := tx.Bucket([]byte("users"))
    return bucket.ForEach(func(k, v []byte) error {
        // ... extract fields ...
        if _, exists := stats[groupStr]; !exists {
            stats[groupStr] = &GroupStats{maxFirst: true}
        }
        stats[groupStr].count++
        stats[groupStr].sum += ageFloat
        if stats[groupStr].maxFirst || ageFloat > stats[groupStr].maxAge {
            stats[groupStr].maxAge = ageFloat
            stats[groupStr].maxFirst = false
        }
        return nil
    })
})

// Output with all metrics
for group, s := range stats {
    avg := 0.0
    if s.count > 0 {
        avg = s.sum / float64(s.count)
    }
    result := map[string]interface{}{
        "city": group,
        "count": s.count,
        "avg": avg,
        "max": s.maxAge,
    }
    output, _ := json.Marshal(result)
    fmt.Println(string(output))
}
```

**Output**:
```json
{"city":"NYC","count":5,"avg":35.2,"max":62}
{"city":"LA","count":3,"avg":42.0,"max":55}
```

**Implementation Strategy**:
1. Modify `extract_group_by_spec/4` to handle list of operations
2. Create `generate_multi_agg_struct/2` to build struct with all needed fields
3. Modify `generate_group_by_code/5` to handle multiple operations
4. Generate accumulation code for each operation
5. Generate output with all metrics

### Feature 2: HAVING Clause Filtering (HIGH PRIORITY)

**Problem**: Can't filter groups by aggregation results.

**Desired Syntax**:
```prolog
% Cities with more than 100 users
large_cities(City, Count) :-
    group_by(City, json_record([city-City, name-_]), count, Count),
    Count > 100.

% Cities with average age over 40
mature_cities(City, Avg) :-
    group_by(City, json_record([city-City, age-Age]), avg(Age), Avg),
    Avg > 40.0.

% Complex filtering with multiple aggregations
active_cities(City, Count, Avg) :-
    group_by(City, json_record([city-City, age-Age]),
             [count(Count), avg(Age, Avg)]),
    Count >= 10,
    Avg > 30.0.
```

**Generated Go Code**:
```go
// Compute aggregations
for group, s := range stats {
    avg := s.sum / float64(s.count)

    // Apply HAVING filters
    if !(s.count >= 10) {
        continue  // Skip this group
    }
    if !(avg > 30.0) {
        continue  // Skip this group
    }

    // Output passing groups
    result := map[string]interface{}{
        "city": group,
        "count": s.count,
        "avg": avg,
    }
    output, _ := json.Marshal(result)
    fmt.Println(string(output))
}
```

**Implementation Strategy**:
1. Detect constraints on aggregation result variables
2. Separate aggregation constraints from record constraints
3. Generate filtering code in output loop (not accumulation loop)
4. Support all comparison operators: `>`, `<`, `>=`, `<=`, `=`, `=\=`

### Feature 3: Nested Grouping (MEDIUM PRIORITY)

**Problem**: Can only group by one field.

**Desired Syntax**:
```prolog
% Group by state AND city
state_city_counts(State, City, Count) :-
    group_by([State, City],
             json_record([state-State, city-City, name-_]),
             count, Count).

% Group by multiple fields with aggregations
region_stats(Region, Category, Count, AvgPrice) :-
    group_by([Region, Category],
             json_record([region-Region, category-Category, price-Price]),
             [count(Count), avg(Price, AvgPrice)]).
```

**Generated Go Code**:
```go
// Composite key type
type GroupKey struct {
    state string
    city  string
}

counts := make(map[GroupKey]int)

// Accumulate with composite key
err = db.View(func(tx *bolt.Tx) error {
    bucket := tx.Bucket([]byte("users"))
    return bucket.ForEach(func(k, v []byte) error {
        // ... extract fields ...
        key := GroupKey{state: stateStr, city: cityStr}
        counts[key]++
        return nil
    })
})

// Output
for key, count := range counts {
    result := map[string]interface{}{
        "state": key.state,
        "city": key.city,
        "count": count,
    }
    output, _ := json.Marshal(result)
    fmt.Println(string(output))
}
```

**Alternative**: Use string concatenation with delimiter (simpler, but less type-safe)
```go
// String-based composite key
counts := make(map[string]int)

key := stateStr + "|" + cityStr
counts[key]++

// Parse back when outputting
for key, count := range counts {
    parts := strings.Split(key, "|")
    result := map[string]interface{}{
        "state": parts[0],
        "city": parts[1],
        "count": count,
    }
    // ...
}
```

**Implementation Strategy**:
1. Start with string concatenation approach (simpler)
2. Detect list of group fields
3. Generate composite key construction code
4. Generate composite key parsing for output
5. Later: Consider struct-based keys for type safety

## Implementation Phases

### Phase 9c-1: Multiple Aggregations ⭐ START HERE

**Timeline**: 2-3 hours

**Tasks**:
1. Modify `extract_group_by_spec/4` to handle operation list
2. Create struct field generator for multiple operations
3. Update accumulation code generator
4. Update output code generator
5. Add tests for multiple aggregations

**Success Criteria**:
- ✅ Support `[count(C), avg(A, Avg)]` syntax
- ✅ Generate single-pass code
- ✅ All operations in same struct
- ✅ JSON output with all metrics
- ✅ Tests passing

### Phase 9c-2: HAVING Clause

**Timeline**: 1-2 hours

**Tasks**:
1. Detect constraints on result variables
2. Separate HAVING constraints from WHERE constraints
3. Generate filtering in output loop
4. Add tests for HAVING

**Success Criteria**:
- ✅ Support `Count > 100` after `group_by`
- ✅ Support multiple HAVING conditions
- ✅ Correct filtering in output
- ✅ Tests passing

### Phase 9c-3: Nested Grouping

**Timeline**: 2-3 hours

**Tasks**:
1. Support `group_by([Field1, Field2], ...)`
2. Generate composite key code
3. Handle key parsing in output
4. Add tests

**Success Criteria**:
- ✅ Support list of group fields
- ✅ Correct composite key generation
- ✅ Proper output formatting
- ✅ Tests passing

## Future Enhancements (Phase 9d+)

### Statistical Functions
```prolog
city_stats(City, StdDev, Median) :-
    group_by(City, json_record([city-City, age-Age]),
             [stddev(Age, StdDev), median(Age, Median)]).
```

**Challenges**:
- stddev requires two passes (mean, then variance)
- median requires sorting or quickselect
- May need to collect all values per group

### Array Aggregations
```prolog
city_ages(City, Ages) :-
    group_by(City, json_record([city-City, age-Age]),
             collect_list(Age, Ages)).

city_unique_ages(City, UniqueAges) :-
    group_by(City, json_record([city-City, age-Age]),
             collect_set(Age, UniqueAges)).
```

**Challenges**:
- Requires storing arrays per group
- Memory overhead for large groups
- Output as JSON arrays

### Window Functions
```prolog
% Rank within group
user_rank(City, Name, Rank) :-
    group_by(City, json_record([city-City, name-Name, age-Age]),
             rank(Age, Rank)).
```

**Challenges**:
- Requires sorting within groups
- More complex output (one row per record, not per group)
- May need separate compilation mode

## Testing Strategy

### Phase 9c-1 Tests
1. Two aggregations: `[count(C), avg(A, Avg)]`
2. Three aggregations: `[count(C), sum(S, Sum), max(M, Max)]`
3. All five operations: `[count, sum, avg, max, min]`
4. Mixed simple and complex: `[count(C), avg(Age, A)]`

### Phase 9c-2 Tests
1. Simple HAVING: `Count > 10`
2. Multiple HAVING: `Count > 10, Avg < 50`
3. HAVING with multiple aggregations
4. Complex conditions: `Count >= 10, Count <= 100`

### Phase 9c-3 Tests
1. Two-field grouping: `[State, City]`
2. Three-field grouping: `[Region, State, City]`
3. Nested grouping with multiple aggregations
4. Nested grouping with HAVING

## Example Use Cases

### Business Analytics
```prolog
% Regional sales summary
region_sales(Region, TotalRevenue, AvgSale, MaxSale, SaleCount) :-
    group_by(Region,
             json_record([region-Region, amount-Amount]),
             [sum(Amount, TotalRevenue),
              avg(Amount, AvgSale),
              max(Amount, MaxSale),
              count(SaleCount)]).

% High-performing regions only
top_regions(Region, Revenue, Count) :-
    group_by(Region, json_record([region-Region, amount-Amount]),
             [sum(Amount, Revenue), count(Count)]),
    Revenue > 100000,
    Count > 50.
```

### User Analytics
```prolog
% Detailed city statistics
city_user_stats(City, Users, AvgAge, MinAge, MaxAge) :-
    group_by(City, json_record([city-City, age-Age]),
             [count(Users),
              avg(Age, AvgAge),
              min(Age, MinAge),
              max(Age, MaxAge)]).

% Cities with significant user base
active_cities(City, Users, AvgAge) :-
    city_user_stats(City, Users, AvgAge, _, _),
    Users >= 100,
    AvgAge >= 25.0,
    AvgAge =< 55.0.
```

### Multi-dimensional Analysis
```prolog
% Sales by region and category
detailed_sales(Region, Category, Revenue, Orders, AvgOrder) :-
    group_by([Region, Category],
             json_record([region-Region, category-Category, amount-Amount]),
             [sum(Amount, Revenue),
              count(Orders),
              avg(Amount, AvgOrder)]),
    Orders >= 10.  % HAVING clause
```

## Performance Considerations

**Multiple Aggregations**:
- Still single-pass O(n)
- Slightly more memory per group (struct vs scalar)
- Worth it: 3 aggregations in 1 query vs 3 separate queries

**HAVING Clause**:
- No performance impact on accumulation
- Filtering happens during output (already fast)
- Reduces output size (can be beneficial)

**Nested Grouping**:
- Still O(n) for accumulation
- More unique keys → more memory
- String concatenation adds small overhead
- Parsing adds small overhead to output

**Overall**: All features maintain single-pass efficiency with minimal overhead.

## Success Metrics

**Phase 9c Complete When**:
- ✅ Multiple aggregations work in single query
- ✅ HAVING clause filters groups correctly
- ✅ Nested grouping (2+ fields) works
- ✅ All combinations tested
- ✅ Documentation updated
- ✅ Performance remains O(n) single-pass

## Implementation Order

1. **Phase 9c-1** (Multiple Aggregations) - Most valuable, moderate complexity
2. **Phase 9c-2** (HAVING Clause) - High value, low complexity, builds on 9c-1
3. **Phase 9c-3** (Nested Grouping) - Good value, moderate complexity

Then consider Phase 9d (statistical functions, array aggregations).
