# Nested JSON Field Access - Design

Implement support for accessing nested JSON structures in the Go target.

## Goal

Enable extraction of values from nested JSON objects:

```json
{
  "user": {
    "name": "Alice",
    "address": {
      "city": "NYC",
      "zip": "10001"
    }
  }
}
```

## Prolog Syntax

### Option 1: json_get/2 with path list (CHOSEN)

```prolog
city(City) :- json_get([user, address, city], City).
```

**Pros:**
- Clear path representation
- Easy to parse
- Matches JSONPath-like syntax

### Option 2: Dot notation

```prolog
city(City) :- json_get('user.address.city', City).
```

**Pros:**
- Familiar syntax
- Compact

**Cons:**
- String parsing required
- Less Prolog-like

## Implementation Approach

### 1. Extend JSON Input Mode

Modify `compile_json_input_mode/4` to:
1. Detect `json_get/2` calls in the body
2. Extract paths and variables
3. Generate nested access code

### 2. Go Helper Function

Generate a reusable helper:

```go
func getNestedField(data map[string]interface{}, path []string) (interface{}, bool) {
    current := interface{}(data)

    for i, key := range path {
        // Check if current is a map
        currentMap, ok := current.(map[string]interface{})
        if !ok {
            return nil, false
        }

        // Get the value at this key
        value, exists := currentMap[key]
        if !exists {
            return nil, false
        }

        current = value
    }

    return current, true
}
```

### 3. Generated Code Pattern

```go
// For: city(City) :- json_get([user, address, city], City).

field1, field1Ok := getNestedField(data, []string{"user", "address", "city"})
if !field1Ok {
    continue
}

result := fmt.Sprintf("%v", field1)
fmt.Println(result)
```

## Example Usage

### Example 1: Simple Nested Access

**Prolog:**
```prolog
city(City) :- json_get([user, address, city], City).

compile_predicate_to_go(city/1, [json_input(true)], Code).
```

**Input:**
```json
{"user": {"address": {"city": "NYC", "zip": "10001"}}}
```

**Output:**
```
NYC
```

### Example 2: Multiple Nested Fields

**Prolog:**
```prolog
user_info(Name, City) :-
    json_get([user, name], Name),
    json_get([user, address, city], City).
```

**Input:**
```json
{"user": {"name": "Alice", "address": {"city": "NYC"}}}
```

**Output:**
```
Alice:NYC
```

### Example 3: Mixed Flat and Nested

**Prolog:**
```prolog
data(Id, City) :-
    json_record([id-Id]),
    json_get([location, city], City).
```

**Input:**
```json
{"id": 123, "location": {"city": "Boston", "state": "MA"}}
```

**Output:**
```
123:Boston
```

## Implementation Steps

### Step 1: Detect json_get in Body

```prolog
%% extract_json_operations(+Body, -Ops)
%  Extract both json_record and json_get operations
extract_json_operations((A, B), Ops) :- !,
    extract_json_operations(A, OpsA),
    extract_json_operations(B, OpsB),
    append(OpsA, OpsB, Ops).
extract_json_operations(json_record(Fields), [record(Fields)]) :- !.
extract_json_operations(json_get(Path, Var), [get(Path, Var)]) :- !.
extract_json_operations(_, []).
```

### Step 2: Generate Helper Function

```prolog
generate_nested_helper(HelperCode) :-
    HelperCode = '
func getNestedField(data map[string]interface{}, path []string) (interface{}, bool) {
\tcurrent := interface{}(data)
\t
\tfor _, key := range path {
\t\tcurrentMap, ok := current.(map[string]interface{})
\t\tif !ok {
\t\t\treturn nil, false
\t\t}
\t\t
\t\tvalue, exists := currentMap[key]
\t\tif !exists {
\t\t\treturn nil, false
\t\t}
\t\t
\t\tcurrent = value
\t}
\t
\treturn current, true
}
'.
```

### Step 3: Generate Field Extractions

```prolog
generate_nested_extraction(get(Path, Var), Pos, Code) :-
    % Convert path atoms to Go string slice
    maplist(atom_string, Path, PathStrs),
    atomic_list_concat(PathStrs, '", "', PathStr),
    format(atom(Code),
        '\t\tfield~w, field~wOk := getNestedField(data, []string{"~w"})~n\t\tif !field~wOk {~n\t\t\tcontinue~n\t\t}',
        [Pos, Pos, PathStr, Pos]).
```

## Testing Strategy

### Test 1: Simple Nested (2 levels)

```json
{"user": {"name": "Alice"}}
```

### Test 2: Deep Nested (3+ levels)

```json
{"company": {"department": {"team": {"lead": "Bob"}}}}
```

### Test 3: Mixed Operations

```json
{"id": 1, "user": {"address": {"city": "NYC"}}}
```

### Test 4: Array in Path (Future)

```json
{"users": [{"name": "Alice"}, {"name": "Bob"}]}
```

## Limitations (v1)

1. **No array indexing** - Arrays require separate support
2. **Type assertions** - Values extracted as `interface{}`
3. **Error handling** - Failed lookups skip the record
4. **No wildcards** - Exact path matching only

## Future Enhancements (v2+)

1. **Array support**: `json_get([users, 0, name], Name)`
2. **Wildcards**: `json_get([users, *, name], Names)`
3. **Type hints**: `json_get([age], Age:int)`
4. **Default values**: `json_get([city], City, "Unknown")`

## Backward Compatibility

✅ Fully backward compatible:
- `json_record/1` continues to work for flat fields
- Can mix `json_record` and `json_get` in same predicate
- Helper function only generated when needed
- No changes to existing code

## Success Criteria

- ✅ Can access 2-level nested fields
- ✅ Can access 3+ level nested fields
- ✅ Can mix flat and nested access
- ✅ Helper function generated only when needed
- ✅ Type assertions work correctly
- ✅ All tests pass
