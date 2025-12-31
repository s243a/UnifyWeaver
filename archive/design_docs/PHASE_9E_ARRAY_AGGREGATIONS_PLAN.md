# Phase 9e: Array Aggregations - Implementation Plan

## Overview
Extend Phase 9 with array aggregation capabilities, allowing values to be collected into lists or sets during grouping. This is essential for denormalizing data (e.g., getting a list of tags for each user).

## Goals
1.  **`collect_list(Field, Result)`**: Collect all values into a list (allows duplicates).
2.  **`collect_set(Field, Result)`**: Collect unique values into a list (deduplicated).
3.  **Integration**: Support simple and grouped aggregations.
4.  **Performance**: Efficient slice appending and map-based deduplication.

## Proposed Syntax

### 1. Collect List
```prolog
% Get list of tags per user
user_tags(User, Tags) :-
    group_by(User, json_record([user-User, tag-Tag]), collect_list(Tag, Tags)).
```

### 2. Collect Set
```prolog
% Get unique cities per country
country_cities(Country, Cities) :-
    group_by(Country, json_record([country-Country, city-City]), collect_set(City, Cities)).
```

### 3. Mixed with Statistics
```prolog
% User stats: count and list of roles
user_summary(User, RoleCount, Roles) :-
    group_by(User,
             json_record([user-User, role-Role]),
             [count(RoleCount), collect_set(Role, Roles)]).
```

## Go Implementation

### Data Structures
For `collect_list`, we need a slice in the state struct.
For `collect_set`, we can use a `map[type]bool` for accumulation and convert to slice for output, OR check for existence before appending (O(N) vs O(1) lookup). A map is better for large sets.

**Struct Fields**:
```go
type GroupStats struct {
    // ... existing ...
    list_Tag []string          // for collect_list
    set_City map[string]bool   // for collect_set (intermediate)
}
```

### Accumulation Logic

**collect_list**:
```go
if val, ok := data["tag"].(string); ok {
    state.list_Tag = append(state.list_Tag, val)
}
```

**collect_set**:
```go
if val, ok := data["city"].(string); ok {
    if state.set_City == nil {
        state.set_City = make(map[string]bool)
    }
    state.set_City[val] = true
}
```

### Output Logic

**collect_list**: Output directly.

**collect_set**: Convert map keys to slice.
```go
cities := make([]string, 0, len(state.set_City))
for k := range state.set_City {
    cities = append(cities, k)
}
// Sort for deterministic output? Optional but recommended.
sort.Strings(cities)
```

## Implementation Tasks

1.  **Update `parse_single_agg_op`**: Recognize `collect_list` and `collect_set`.
2.  **Update `needed_struct_field`**: Map to slice/map fields.
3.  **Update `go_struct_field_line`**: Generate slice/map definitions.
4.  **Update `jsonl_op_update`**: Add appending logic.
5.  **Update `jsonl_op_print_arg`**: Handle map-to-slice conversion.
6.  **Add `sort` dependency**: If deterministic set output is desired.

## Timeline
-   Plan & Design: 30 mins
-   Implementation: 2 hours
-   Testing: 1 hour

## Success Criteria
-   ✅ `collect_list` works with strings and numbers.
-   ✅ `collect_set` dedups values correctly.
-   ✅ Works in `group_by`.
-   ✅ Clean JSON output (arrays).
