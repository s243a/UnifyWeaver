# Phase 8b: Enhanced Filtering - Implementation Plan

## Status: Planning

Building on Phase 8a's foundation, this phase adds advanced filtering capabilities including string operations, list membership, and key optimization.

## Goals

### 1. String Operations

Add support for string-specific comparison and matching operations.

**Operators to Implement:**
- `=@=` - Case-insensitive equality
- `contains(String, Substring)` - Substring matching
- `substring(String, Start, Length, Result)` - Substring extraction

**Example Usage:**
```prolog
% Case-insensitive city search
user_by_city(Name, City) :-
    json_record([name-Name, city-City]),
    City =@= "nyc".  % Matches "NYC", "nyc", "Nyc", etc.

% Search by name substring
users_with_substring(Name) :-
    json_record([name-Name]),
    contains(Name, "ali").  % Matches "Alice", "Natalie", etc.
```

**Go Implementation:**
```go
// Case-insensitive comparison
if !(strings.EqualFold(field2, "nyc")) {
    return nil
}

// Substring search
if !(strings.Contains(field1, "ali")) {
    return nil
}
```

### 2. List Membership

Support checking if field values match a list of options.

**Syntax:**
```prolog
% Match multiple cities
major_city_users(Name, City) :-
    json_record([name-Name, city-City]),
    member(City, ["NYC", "SF", "LA", "Chicago"]).

% Age ranges
age_group(Name, Age) :-
    json_record([name-Name, age-Age]),
    member(Age, [25, 30, 35, 40]).
```

**Go Implementation:**
```go
// City membership check
cityOptions := []string{"NYC", "SF", "LA", "Chicago"}
found := false
for _, opt := range cityOptions {
    if field2 == opt {
        found = true
        break
    }
}
if !found {
    return nil
}
```

### 3. Key Optimization Detection

Automatically detect when constraints can use direct key lookup instead of full scan.

**Optimization Opportunities:**

**Case 1: Single Field Key + Equality**
```prolog
% Key strategy: field(name)
% Constraint: Name = "Alice"
% Optimization: Use db.Get([]byte("Alice")) instead of ForEach
user_by_name(Name, Age) :-
    json_record([name-Name, age-Age]),
    Name = "Alice".
```

**Case 2: Composite Key + Partial Match**
```prolog
% Key strategy: composite([field(city), field(name)])
% Constraint: City = "NYC"
% Optimization: Use prefix scan with Cursor.Seek([]byte("NYC:"))
nyc_users(City, Name, Age) :-
    json_record([city-City, name-Name, age-Age]),
    City = "NYC".
```

**Implementation Strategy:**
1. Extract constraints during compilation
2. Compare constraint fields with key strategy fields
3. If all key fields are constrained with `=`, use direct lookup
4. If prefix of key fields constrained, use prefix scan
5. Otherwise, fall back to full scan with filters

**Detection Logic:**
```prolog
% Detect if constraints can use key optimization
analyze_key_optimization(KeyStrategy, Constraints, OptimizationType) :-
    KeyStrategy = field(KeyField),
    member(Constraint, Constraints),
    Constraint = (Var = Value),
    get_field_name(Var, KeyField),
    !,
    OptimizationType = direct_lookup(Value).

analyze_key_optimization(KeyStrategy, Constraints, OptimizationType) :-
    KeyStrategy = composite(KeyFields),
    check_prefix_match(KeyFields, Constraints, MatchedPrefix),
    length(MatchedPrefix, PrefixLen),
    length(KeyFields, TotalLen),
    (   PrefixLen = TotalLen
    ->  OptimizationType = direct_lookup(MatchedPrefix)
    ;   PrefixLen > 0
    ->  OptimizationType = prefix_scan(MatchedPrefix)
    ;   OptimizationType = full_scan
    ).
```

**Generated Go Code for Direct Lookup:**
```go
// Direct key lookup optimization
keyStr := fmt.Sprintf("%v", "Alice")
key := []byte(keyStr)

err = db.View(func(tx *bolt.Tx) error {
    bucket := tx.Bucket([]byte("users"))
    if bucket == nil {
        return fmt.Errorf("bucket 'users' not found")
    }

    value := bucket.Get(key)
    if value == nil {
        return nil // Key not found
    }

    // Deserialize and output
    var data map[string]interface{}
    if err := json.Unmarshal(value, &data); err != nil {
        return nil
    }

    // Apply remaining filters (if any)
    // ... filter code ...

    // Output result
    // ... output code ...

    return nil
})
```

**Generated Go Code for Prefix Scan:**
```go
// Prefix scan optimization
prefix := []byte("NYC:")

err = db.View(func(tx *bolt.Tx) error {
    bucket := tx.Bucket([]byte("users"))
    if bucket == nil {
        return fmt.Errorf("bucket 'users' not found")
    }

    cursor := bucket.Cursor()

    for k, v := cursor.Seek(prefix); k != nil && bytes.HasPrefix(k, prefix); k, v = cursor.Next() {
        // Deserialize record
        var data map[string]interface{}
        if err := json.Unmarshal(v, &data); err != nil {
            continue
        }

        // Apply remaining filters
        // ... filter code ...

        // Output result
        // ... output code ...
    }

    return nil
})
```

## Implementation Tasks

### Task 1: String Operations

**Files to Modify:**
- `src/unifyweaver/targets/go_target.pl`

**Functions to Add:**
- `is_string_operation/1` - Detect `=@=`, `contains/2`, `substring/4`
- `generate_string_operation_check/3` - Generate Go code for string ops
- Update `field_term_to_go_expr/3` to handle function calls

**Go Imports Needed:**
- Add `strings` package when string operations detected

### Task 2: List Membership

**Files to Modify:**
- `src/unifyweaver/targets/go_target.pl`

**Functions to Add:**
- `is_member_constraint/1` - Detect `member(Var, List)` patterns
- `generate_member_check/3` - Generate Go membership check code
- Handle list literals in constraints

### Task 3: Key Optimization

**Files to Modify:**
- `src/unifyweaver/targets/go_target.pl`

**Functions to Add:**
- `analyze_key_optimization/3` - Determine optimization strategy
- `generate_optimized_lookup/4` - Generate direct lookup code
- `generate_prefix_scan/4` - Generate prefix scan code
- `extract_constraint_values/3` - Extract constant values from constraints

**New Compilation Option:**
- `optimize_keys(true/false)` - Enable/disable optimization (default: true)

### Task 4: Testing

**Test Cases:**

**String Operations:**
- Case-insensitive equality (`=@=`)
- Substring matching (`contains`)
- Mixed string and numeric filters

**List Membership:**
- String list membership
- Numeric list membership
- Empty list handling

**Key Optimization:**
- Direct lookup (single field key)
- Direct lookup (composite key)
- Prefix scan (partial composite key)
- Fallback to full scan (non-key fields constrained)
- Performance comparison (optimized vs full scan)

**Test File:** `test_phase_8b.pl`
**Runner:** `run_phase_8b_tests.sh`

## Design Decisions

### Why These String Operations?

- **`=@=`**: Standard Prolog operator, familiar to developers
- **`contains/2`**: Common use case, maps directly to Go `strings.Contains`
- **`substring/4`**: Powerful for text processing, standard in many Prologs

### Why Automatic Key Optimization?

- **Zero configuration**: No special syntax needed
- **Performance**: Can be 10-100x faster for specific lookups
- **Backward compatible**: Existing code automatically benefits
- **Opt-out available**: Can disable if needed

### Member/2 List Syntax

Using standard Prolog list syntax `[Value1, Value2, ...]` keeps the DSL minimal and familiar.

## Success Criteria

1. ✅ String operations compile to correct Go code
2. ✅ List membership generates efficient Go loops
3. ✅ Key optimization correctly detects opportunities
4. ✅ Optimized code is faster than full scan (benchmark)
5. ✅ All tests pass
6. ✅ Documentation complete
7. ✅ Backward compatible with Phase 8a

## Timeline Estimate

- **Task 1 (Strings)**: ~2-3 hours
- **Task 2 (Member)**: ~1-2 hours
- **Task 3 (Optimization)**: ~3-4 hours (most complex)
- **Task 4 (Testing)**: ~2-3 hours
- **Documentation**: ~1 hour

**Total**: ~9-13 hours

## References

- Phase 8a implementation: `go_target.pl:1695-2060`
- Key strategies: `go_target.pl:1440-1690`
- Standard Prolog string operations: ISO/IEC 13211-1
- Go strings package: https://pkg.go.dev/strings
- bbolt cursor API: https://pkg.go.dev/go.etcd.io/bbolt#Cursor
