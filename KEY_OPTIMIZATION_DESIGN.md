# Key Optimization Detection Design

## Overview

Automatically detect when database filtering constraints can use efficient key-based lookups instead of full bucket scans, improving query performance by 10-100x for specific queries.

## Problem Statement

Currently, all Phase 8 queries use `bucket.ForEach()` to scan every record in the database:

```go
return bucket.ForEach(func(k, v []byte) error {
    // Deserialize JSON
    // Extract fields
    // Apply filters
    // Output if matches
})
```

**Performance Issue**: For a database with 1M records, finding a single user by exact name requires scanning all 1M records, even though we know the exact key.

## Optimization Opportunities

### 1. Direct Key Lookup (10-100x faster)

When a constraint exactly matches the key strategy, use `db.Get()` for O(1) lookup:

**Example**: Key strategy = `name`, Constraint = `Name = "Alice"`

```prolog
% Current (full scan)
user_by_name(Name) :-
    json_record([name-Name, age-_Age]),
    Name = "Alice".

% Optimized (direct lookup)
% Should generate: db.Get([]byte("Alice"))
```

**Performance**: 1M records → 1 record fetch

### 2. Prefix Scan (10-50x faster)

When a constraint matches the first component of a composite key, use `Cursor.Seek()` for range scan:

**Example**: Key strategy = `[city, name]`, Constraint = `City = "NYC"`

```prolog
% Current (full scan)
nyc_users(Name, City) :-
    json_record([name-Name, city-City]),
    City = "NYC".

% Optimized (prefix scan)
% Should generate: cursor.Seek([]byte("NYC:"))
```

**Performance**: 1M records → ~1000 NYC records scanned

### 3. Full Scan Fallback

When constraints don't match key strategy, maintain current `ForEach()` behavior:

```prolog
% No optimization possible
old_users(Name, Age) :-
    json_record([name-Name, age-Age]),
    Age > 50.  % Age not part of key
```

## Detection Algorithm

### Input Analysis

For each predicate with `json_record` + constraints:

1. **Extract Key Strategy** from `json_store` predicate
2. **Extract Constraints** from predicate body
3. **Match Constraints to Key Components**

### Key Strategy Patterns

```prolog
% Pattern 1: Single field key
json_store([name-Name], users, [name]).

% Pattern 2: Composite key (2 fields)
json_store([city-City, name-Name], users, [city, name]).

% Pattern 3: Composite key (3+ fields)
json_store([state-State, city-City, name-Name], users, [state, city, name]).
```

### Constraint Matching Rules

#### Rule 1: Direct Lookup

**Condition**: Single equality constraint on exact key field

```prolog
% Key: [name]
Name = "Alice"  ✅ Direct lookup

% Key: [city, name]
City = "NYC", Name = "Alice"  ✅ Direct lookup (both components)
```

**Detection**:
```prolog
can_use_direct_lookup(KeyStrategy, Constraints) :-
    KeyStrategy = [SingleKey],
    member(Var = Value, Constraints),
    Var == SingleKey,
    ground(Value).  % Value must be concrete

can_use_direct_lookup(KeyStrategy, Constraints) :-
    % All key components have exact equality constraints
    forall(member(KeyField, KeyStrategy),
           member(_ = _, Constraints)).  % Simplified check
```

#### Rule 2: Prefix Scan

**Condition**: Equality constraint on first N components of composite key

```prolog
% Key: [city, name]
City = "NYC"  ✅ Prefix scan (first component)

% Key: [state, city, name]
State = "NY", City = "NYC"  ✅ Prefix scan (first two components)
State = "NY"  ✅ Prefix scan (first component)
City = "NYC"  ❌ Not first component
```

**Detection**:
```prolog
can_use_prefix_scan(KeyStrategy, Constraints, PrefixLength) :-
    KeyStrategy = [First|_Rest],
    length(KeyStrategy, KeyLen),
    KeyLen > 1,  % Must be composite key
    % Find longest matching prefix
    find_prefix_length(KeyStrategy, Constraints, PrefixLength),
    PrefixLength > 0,
    PrefixLength < KeyLen.

find_prefix_length([Field|Rest], Constraints, Length) :-
    member(Var = Value, Constraints),
    Var == Field,
    ground(Value),
    !,
    find_prefix_length(Rest, Constraints, RestLen),
    Length is RestLen + 1.
find_prefix_length(_, _, 0).
```

#### Rule 3: Full Scan Fallback

**Condition**: None of the above

```prolog
% Key: [name]
Age > 50  ❌ Different field
Name =@= "alice"  ❌ Case-insensitive (not exact match)
contains(Name, "ali")  ❌ Substring (not exact match)

% Key: [city, name]
Name = "Alice"  ❌ Not first component
Age > 30, City = "NYC"  ❌ Non-key constraint present
```

## Generated Go Code Patterns

### Pattern 1: Direct Lookup (Single Key)

**Input**:
```prolog
json_store([name-Name], users, [name]).

user_by_name(Name, Age) :-
    json_record([name-Name, age-Age]),
    Name = "Alice".
```

**Generated Go**:
```go
func main() {
    db, err := bolt.Open("users.db", 0600, &bolt.Options{ReadOnly: true})
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
    defer db.Close()

    err = db.View(func(tx *bolt.Tx) error {
        bucket := tx.Bucket([]byte("users"))
        if bucket == nil {
            return fmt.Errorf("bucket not found")
        }

        // DIRECT LOOKUP: Use Get() for exact key match
        key := []byte("Alice")
        value := bucket.Get(key)
        if value == nil {
            return nil // Key not found
        }

        // Deserialize record
        var data map[string]interface{}
        if err := json.Unmarshal(value, &data); err != nil {
            return err
        }

        // Extract fields
        name, nameOk := data["name"]
        age, ageOk := data["age"]
        if !nameOk || !ageOk {
            return nil
        }

        // Output (no filtering needed - key match guarantees Name="Alice")
        output, _ := json.Marshal(map[string]interface{}{"name": name, "age": age})
        fmt.Println(string(output))
        return nil
    })

    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
}
```

### Pattern 2: Direct Lookup (Composite Key)

**Input**:
```prolog
json_store([city-City, name-Name], users, [city, name]).

user_exact(Name, City, Age) :-
    json_record([name-Name, city-City, age-Age]),
    City = "NYC",
    Name = "Alice".
```

**Generated Go**:
```go
// DIRECT LOOKUP: Composite key
key := []byte("NYC:Alice")  // Note: separator from key strategy
value := bucket.Get(key)
if value == nil {
    return nil
}

// Deserialize and output (no filtering needed)
var data map[string]interface{}
json.Unmarshal(value, &data)
// ... extract and output fields
```

### Pattern 3: Prefix Scan (Composite Key)

**Input**:
```prolog
json_store([city-City, name-Name], users, [city, name]).

nyc_users(Name, City, Age) :-
    json_record([name-Name, city-City, age-Age]),
    City = "NYC".
```

**Generated Go**:
```go
func main() {
    db, err := bolt.Open("users.db", 0600, &bolt.Options{ReadOnly: true})
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
    defer db.Close()

    err = db.View(func(tx *bolt.Tx) error {
        bucket := tx.Bucket([]byte("users"))
        if bucket == nil {
            return fmt.Errorf("bucket not found")
        }

        // PREFIX SCAN: Seek to first key starting with "NYC:"
        cursor := bucket.Cursor()
        prefix := []byte("NYC:")

        for k, v := cursor.Seek(prefix); k != nil && bytes.HasPrefix(k, prefix); k, v = cursor.Next() {
            // Deserialize record
            var data map[string]interface{}
            if err := json.Unmarshal(v, &data); err != nil {
                continue
            }

            // Extract fields
            name, nameOk := data["name"]
            city, cityOk := data["city"]
            age, ageOk := data["age"]
            if !nameOk || !cityOk || !ageOk {
                continue
            }

            // No city filter needed - prefix guarantees City="NYC"
            // Apply any additional constraints here (e.g., Age > 30)

            // Output
            output, _ := json.Marshal(map[string]interface{}{
                "name": name, "city": city, "age": age,
            })
            fmt.Println(string(output))
        }
        return nil
    })

    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
}
```

### Pattern 4: Prefix Scan with Additional Filters

**Input**:
```prolog
json_store([city-City, name-Name], users, [city, name]).

nyc_young_users(Name, City, Age) :-
    json_record([name-Name, city-City, age-Age]),
    City = "NYC",
    Age < 40.
```

**Generated Go**:
```go
cursor := bucket.Cursor()
prefix := []byte("NYC:")

for k, v := cursor.Seek(prefix); k != nil && bytes.HasPrefix(k, prefix); k, v = cursor.Next() {
    var data map[string]interface{}
    json.Unmarshal(v, &data)

    name, _ := data["name"]
    city, _ := data["city"]
    age, _ := data["age"]

    // OPTIMIZATION: City check omitted (guaranteed by prefix)
    // Apply remaining constraints
    ageFloat, _ := age.(float64)
    if ageFloat >= 40 {
        continue  // Skip record
    }

    // Output matching record
    output, _ := json.Marshal(map[string]interface{}{
        "name": name, "city": city, "age": age,
    })
    fmt.Println(string(output))
}
```

## Implementation Plan

### Phase 1: Detection Logic (`src/unifyweaver/targets/go_target.pl`)

**New Predicates**:

```prolog
%% analyze_key_optimization(+Predicate, +KeyStrategy, -OptType, -OptDetails)
%  Analyze if predicate can use optimized key lookup
%  OptType: direct_lookup | prefix_scan | full_scan
%  OptDetails: Details needed for code generation
%
analyze_key_optimization(Predicate, KeyStrategy, OptType, OptDetails) :-
    extract_constraints(Predicate, Constraints),
    (   can_use_direct_lookup(KeyStrategy, Constraints, KeyValue)
    ->  OptType = direct_lookup,
        OptDetails = key_value(KeyValue)
    ;   can_use_prefix_scan(KeyStrategy, Constraints, PrefixFields, PrefixValues)
    ->  OptType = prefix_scan,
        OptDetails = prefix(PrefixFields, PrefixValues)
    ;   OptType = full_scan,
        OptDetails = none
    ).

%% can_use_direct_lookup(+KeyStrategy, +Constraints, -KeyValue)
%  Check if all key fields have exact equality constraints
%
can_use_direct_lookup(KeyStrategy, Constraints, KeyValue) :-
    maplist(has_exact_constraint(Constraints), KeyStrategy, Values),
    build_composite_key(Values, KeyValue).

has_exact_constraint(Constraints, Field, Value) :-
    member(Var = Value, Constraints),
    Var == Field,
    ground(Value),
    \+ is_case_insensitive_op(Value).  % Must be exact =, not =@=

%% can_use_prefix_scan(+KeyStrategy, +Constraints, -PrefixFields, -PrefixValues)
%  Check if first N fields have exact equality constraints
%
can_use_prefix_scan(KeyStrategy, Constraints, PrefixFields, PrefixValues) :-
    KeyStrategy = [_,_|_],  % Must be composite key
    find_matching_prefix(KeyStrategy, Constraints, PrefixFields, PrefixValues),
    PrefixFields \= [],
    PrefixFields \= KeyStrategy.  % Not all fields (that's direct lookup)

find_matching_prefix([Field|Rest], Constraints, [Field|RestFields], [Value|RestValues]) :-
    member(Var = Value, Constraints),
    Var == Field,
    ground(Value),
    \+ is_case_insensitive_op(Value),
    !,
    find_matching_prefix(Rest, Constraints, RestFields, RestValues).
find_matching_prefix(_, _, [], []).
```

**Modified Predicates**:

```prolog
%% generate_query_function(+Predicate, +KeyStrategy, -GoCode)
%  Modified to detect and use key optimizations
%
generate_query_function(Predicate, KeyStrategy, GoCode) :-
    % Analyze optimization opportunity
    analyze_key_optimization(Predicate, KeyStrategy, OptType, OptDetails),

    % Generate code based on optimization type
    (   OptType = direct_lookup
    ->  generate_direct_lookup_code(Predicate, OptDetails, GoCode)
    ;   OptType = prefix_scan
    ->  generate_prefix_scan_code(Predicate, OptDetails, GoCode)
    ;   % OptType = full_scan
        generate_full_scan_code(Predicate, GoCode)  % Existing behavior
    ).
```

### Phase 2: Code Generation

**File**: `src/unifyweaver/targets/go_target.pl`

**New Predicates** (approx. lines 2300-2500):

```prolog
%% generate_direct_lookup_code(+Predicate, +KeyValue, -GoCode)
generate_direct_lookup_code(Predicate, key_value(KeyValue), GoCode) :-
    % Extract components
    extract_bucket_name(Predicate, BucketName),
    extract_output_fields(Predicate, OutputFields),

    % Build key bytes
    format_key_value(KeyValue, KeyBytes),

    % Generate optimized code
    format(string(GoCode), '~s', [
        % Database open
        % View transaction
        % bucket.Get(KeyBytes)
        % Deserialize
        % Output (no filtering)
    ]).

%% generate_prefix_scan_code(+Predicate, +PrefixDetails, -GoCode)
generate_prefix_scan_code(Predicate, prefix(PrefixFields, PrefixValues), GoCode) :-
    % Build prefix bytes
    build_composite_key(PrefixValues, PrefixKey),

    % Extract remaining constraints (not in prefix)
    extract_non_prefix_constraints(Predicate, PrefixFields, RemainingConstraints),

    % Generate code with cursor.Seek()
    format(string(GoCode), '~s', [
        % Database open
        % View transaction
        % cursor.Seek(prefixBytes)
        % Loop with bytes.HasPrefix check
        % Deserialize
        % Apply remaining constraints
        % Output
    ]).
```

### Phase 3: Composite Key Building

**File**: `src/unifyweaver/targets/go_target.pl`

```prolog
%% build_composite_key(+Values, -CompositeKey)
%  Build composite key with separator (default ":")
%
build_composite_key([Single], Single) :- !.
build_composite_key(Values, CompositeKey) :-
    maplist(value_to_string, Values, Strings),
    atomic_list_concat(Strings, ':', CompositeKey).

value_to_string(Value, String) :-
    (   atom(Value) -> atom_string(Value, String)
    ;   string(Value) -> String = Value
    ;   number(Value) -> format(string(String), '~w', [Value])
    ;   format(string(String), '~w', [Value])
    ).
```

### Phase 4: Constraint Filtering

**File**: `src/unifyweaver/targets/go_target.pl`

```prolog
%% extract_non_prefix_constraints(+Predicate, +PrefixFields, -RemainingConstraints)
%  Remove constraints that are satisfied by prefix scan
%
extract_non_prefix_constraints(Predicate, PrefixFields, RemainingConstraints) :-
    extract_constraints(Predicate, AllConstraints),
    exclude(is_prefix_constraint(PrefixFields), AllConstraints, RemainingConstraints).

is_prefix_constraint(PrefixFields, Var = _Value) :-
    member(Field, PrefixFields),
    Var == Field.
```

## Test Cases

### Test 1: Direct Lookup (Single Key)

**File**: `test_key_opt_direct.pl`

```prolog
:- use_module('src/unifyweaver/targets/go_target').

json_store([name-Name], users, [name]).

% Write mode: Populate database
populate_users(Name, Age, City) :-
    json_record([name-Name, age-Age, city-City]).

% Read mode: Direct lookup by exact name
user_alice(Name, Age, City) :-
    json_record([name-Name, age-Age, city-City]),
    Name = "Alice".

:- compile_goal(populate_users(_, _, _), output_key_opt_populate, [write_mode]).
:- compile_goal(user_alice(_, _, _), output_key_opt_alice, []).
```

**Expected**: `bucket.Get([]byte("Alice"))` instead of `ForEach()`

### Test 2: Direct Lookup (Composite Key)

```prolog
json_store([city-City, name-Name], users, [city, name]).

user_nyc_alice(Name, Age) :-
    json_record([name-Name, city-City, age-Age]),
    City = "NYC",
    Name = "Alice".
```

**Expected**: `bucket.Get([]byte("NYC:Alice"))`

### Test 3: Prefix Scan

```prolog
json_store([city-City, name-Name], users, [city, name]).

nyc_users(Name, Age) :-
    json_record([name-Name, city-City, age-Age]),
    City = "NYC".
```

**Expected**: `cursor.Seek([]byte("NYC:"))` with `bytes.HasPrefix()` loop

### Test 4: Prefix Scan + Additional Filters

```prolog
json_store([city-City, name-Name], users, [city, name]).

nyc_young(Name, Age) :-
    json_record([name-Name, city-City, age-Age]),
    City = "NYC",
    Age < 40.
```

**Expected**: Prefix scan + age filter in loop (no city filter)

### Test 5: Full Scan Fallback (Non-Key Field)

```prolog
json_store([name-Name], users, [name]).

old_users(Name, Age) :-
    json_record([name-Name, age-Age]),
    Age > 50.
```

**Expected**: `ForEach()` (no optimization)

### Test 6: Full Scan Fallback (Case-Insensitive)

```prolog
json_store([name-Name], users, [name]).

user_insensitive(Name, Age) :-
    json_record([name-Name, age-Age]),
    Name =@= "alice".
```

**Expected**: `ForEach()` with `strings.EqualFold()` (can't use Get)

### Test 7: Full Scan Fallback (Substring)

```prolog
json_store([name-Name], users, [name]).

users_with_ali(Name) :-
    json_record([name-Name]),
    contains(Name, "ali").
```

**Expected**: `ForEach()` with `strings.Contains()`

### Test 8: Prefix Scan (Multi-Level)

```prolog
json_store([state-State, city-City, name-Name], users, [state, city, name]).

ny_nyc_users(Name) :-
    json_record([state-State, city-City, name-Name]),
    State = "NY",
    City = "NYC".
```

**Expected**: `cursor.Seek([]byte("NY:NYC:"))` with 2-level prefix

## Performance Expectations

### Benchmark Setup
- Database: 1,000,000 records
- Hardware: Standard laptop (4 cores, 16GB RAM)
- bbolt defaults (4KB page size)

### Expected Results

| Query Type | Method | Records Scanned | Time (ms) | Speedup |
|------------|--------|-----------------|-----------|---------|
| Full Scan | ForEach() | 1,000,000 | 5,000 | 1x |
| Direct Lookup | Get() | 1 | 5 | 1000x |
| Prefix (0.1%) | Seek() | 1,000 | 50 | 100x |
| Prefix (1%) | Seek() | 10,000 | 250 | 20x |
| Prefix (10%) | Seek() | 100,000 | 1,500 | 3x |

### Test Data Distribution

**Single Key Test** (`name`):
- Total records: 100,000 unique names
- Target: "Alice" (1 record)

**Composite Key Test** (`[city, name]`):
- Cities: 100 cities (uniform distribution)
- Names: 1,000 unique names per city
- Target: City="NYC" (~10,000 records), City+Name="NYC:Alice" (1 record)

## Edge Cases

### 1. Empty Results

**Query**: Direct lookup for non-existent key
```prolog
Name = "NonExistent"
```

**Expected**: `bucket.Get()` returns `nil`, no output, no error

### 2. Partial Composite Key Match

**Query**: Second component of composite key
```prolog
json_store([city-City, name-Name], users, [city, name]).
Name = "Alice".  % Missing City constraint
```

**Expected**: Fall back to full scan (can't use prefix on non-first component)

### 3. Multiple Exact Constraints

**Query**: Extra constraints beyond key
```prolog
json_store([name-Name], users, [name]).
Name = "Alice", Age > 30.
```

**Expected**: Direct lookup `Get("Alice")` + age filter on result

### 4. OR Constraints

**Query**: Disjunction in constraints
```prolog
json_store([city-City], users, [city]).
(City = "NYC" ; City = "SF").
```

**Expected**: Fall back to full scan (can't optimize OR without query planning)

Future enhancement: Generate two separate Get() calls

### 5. Variable Key Components

**Query**: Key field is unbound
```prolog
json_store([name-Name], users, [name]).
Age > 50.  % Name is unbound
```

**Expected**: Full scan (can't build key from unbound variable)

## Implementation Checklist

### Detection Logic
- [ ] Implement `analyze_key_optimization/4`
- [ ] Implement `can_use_direct_lookup/3`
- [ ] Implement `can_use_prefix_scan/4`
- [ ] Implement `find_matching_prefix/4`
- [ ] Implement `has_exact_constraint/3`
- [ ] Add case-insensitive operator detection

### Code Generation
- [ ] Implement `generate_direct_lookup_code/3`
- [ ] Implement `generate_prefix_scan_code/3`
- [ ] Implement `build_composite_key/2`
- [ ] Implement `format_key_value/2`
- [ ] Add `bytes` package import detection
- [ ] Implement constraint filtering for optimized queries

### Integration
- [ ] Modify `generate_query_function/3` to use optimization analysis
- [ ] Update `compile_goal/3` to pass key strategy to code generator
- [ ] Ensure backward compatibility with existing full-scan queries

### Testing
- [ ] Create `test_key_opt.pl` with 8 test cases
- [ ] Create `run_key_opt_tests.sh` test runner
- [ ] Benchmark performance improvements
- [ ] Test edge cases (empty results, partial matches, etc.)

### Documentation
- [ ] Update `GO_JSON_FEATURES.md` with Phase 8c section
- [ ] Add optimization examples and generated code
- [ ] Document when optimizations apply vs fall back
- [ ] Add performance benchmarks

## Migration Path

**Zero Breaking Changes**: All existing queries continue working.

```prolog
% Before optimization (still works identically)
user_by_name(Name, Age) :-
    json_record([name-Name, age-Age]),
    Name = "Alice".
% Output: Full scan (slow but correct)

% After optimization (automatic, no code changes)
user_by_name(Name, Age) :-
    json_record([name-Name, age-Age]),
    Name = "Alice".
% Output: Direct lookup (1000x faster, same results)
```

## Future Enhancements

### Phase 8d: Multi-Key Queries
Handle OR constraints with multiple lookups:
```prolog
(Name = "Alice" ; Name = "Bob").
% Generate: Get("Alice") + Get("Bob")
```

### Phase 8e: Range Scans
Optimize range queries on ordered keys:
```prolog
Age > 30, Age < 50.
% If key=[age], use: cursor.Seek(31) until cursor reaches 50
```

### Phase 8f: Index Hints
Allow manual optimization hints:
```prolog
user_by_name(Name) :-
    json_record([name-Name]),
    optimize(direct_lookup, Name = "Alice").
```

## References

- **bbolt Get()**: https://pkg.go.dev/go.etcd.io/bbolt#Bucket.Get
- **bbolt Cursor**: https://pkg.go.dev/go.etcd.io/bbolt#Cursor
- **bytes.HasPrefix()**: https://pkg.go.dev/bytes#HasPrefix
- **Phase 8a**: PR #155 (numeric comparisons)
- **Phase 8b**: Current PR (string operations)
