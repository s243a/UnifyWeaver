# Go Target: JSON I/O Features

Complete JSON input/output support for the Go target.

## Features Implemented

### ✅ Phase 1: JSON Input (JSONL)

Parse JSON Lines (JSONL) input streams.

**Prolog Syntax:**
```prolog
user(Name, Age) :- json_record([name-Name, age-Age]).

compile_predicate_to_go(user/2, [json_input(true)], Code).
```

**Input:**
```json
{"name": "Alice", "age": 25}
{"name": "Bob", "age": 30}
```

**Output:**
```
Alice:25
Bob:30
```

**Key Features:**
- Dynamic typing with `map[string]interface{}`
- Supports all JSON types (string, number, boolean, null)
- JSONL streaming format (one JSON per line)
- Works with all delimiter options (colon, tab, comma, etc.)
- Automatic type handling via `%v` formatting

### ✅ Phase 2: JSON Output

Generate JSON from delimiter-based input.

**Prolog Syntax:**
```prolog
person(Name, Age).

compile_predicate_to_go(person/2, [
    json_output(true),
    json_fields([name, age])
], Code).
```

**Input:**
```
Alice:25
Bob:30
```

**Output:**
```json
{"name":"Alice","age":25}
{"name":"Bob","age":30}
```

**Key Features:**
- Struct-based output with JSON tags
- Smart type conversion (int, float, bool, string)
- Custom field names via `json_fields([...])`
- Auto-generated default names (Field1, Field2, ...)
- Reads any delimiter format (colon, tab, comma, etc.)

## Usage Examples

### Example 1: JSON Transformation Pipeline

```bash
# Input: JSONL
cat users.jsonl | ./parse_json | ./filter_adults | ./to_json
```

### Example 2: Mixed Format Pipeline

```bash
# CSV → JSON
cat data.csv | ./csv_parser | ./to_json > output.jsonl

# JSON → TSV
cat data.jsonl | ./json_parser | ./format_tsv > output.tsv
```

### Example 3: Round-Trip

```bash
# JSON → Colon → JSON (preserves types)
cat input.jsonl | ./json_in | ./json_out
```

## Type Handling

### JSON Input → Go Types

| JSON Type | Go Type | Example |
|-----------|---------|---------|
| String | `string` | `"Alice"` → `Alice` |
| Number | `float64` | `25` → `25` |
| Boolean | `bool` | `true` → `true` |
| Null | `nil` | `null` → `nil` |
| Object | `map[string]interface{}` | `{"a":1}` |
| Array | `[]interface{}` | `[1,2,3]` |

### Go → JSON Output Types

Automatic type detection:
- `"25"` → `25` (integer)
- `"3.14"` → `3.14` (float)
- `"true"` → `true` (boolean)
- `"Alice"` → `"Alice"` (string)

## Options Reference

### JSON Input Options

```prolog
compile_predicate_to_go(user/2, [
    json_input(true),           % Enable JSON input mode
    field_delimiter(colon),     % Output delimiter (default: colon)
    unique(true)                % Filter duplicates (default: true)
], Code).
```

### JSON Output Options

```prolog
compile_predicate_to_go(person/2, [
    json_output(true),          % Enable JSON output mode
    json_fields([name, age]),   % Custom field names
    field_delimiter(tab)        % Input delimiter (default: colon)
], Code).
```

## Testing

### Run All Tests

```bash
# JSON Input tests
./run_json_tests.sh

# JSON Output tests
./run_json_output_tests.sh
```

### Test Results

**JSON Input:**
- ✅ String and numeric fields
- ✅ Mixed types (string, number, boolean)
- ✅ Variable field counts (1-4 fields)
- ✅ Unique filtering
- ✅ Custom delimiters

**JSON Output:**
- ✅ Custom field names
- ✅ Default field names
- ✅ Three+ fields with mixed types
- ✅ Tab-delimited input
- ✅ Round-trip conversion

## Implementation Details

### Generated Code Structure

**JSON Input:**
```go
scanner := bufio.NewScanner(os.Stdin)
for scanner.Scan() {
    var data map[string]interface{}
    json.Unmarshal(scanner.Bytes(), &data)

    field1, field1Ok := data["name"]
    field2, field2Ok := data["age"]

    result := fmt.Sprintf("%v:%v", field1, field2)
    fmt.Println(result)
}
```

**JSON Output:**
```go
type PERSONRecord struct {
    NAME interface{} `json:"name"`
    AGE  interface{} `json:"age"`
}

// Parse input, auto-convert types
jsonBytes, _ := json.Marshal(record)
fmt.Println(string(jsonBytes))
```

## Future Enhancements (Phase 3+)

### Nested Field Access (Planned)

```prolog
city(City) :- json_get([user, address, city], City).
```

### Array Support (Planned)

```prolog
user_name(Name) :-
    json_get([users], UserList),
    json_array_member(UserList, User),
    json_get(User, [name], Name).
```

### Schema Support (Planned)

```prolog
:- json_schema(person, [
    field(name, string),
    field(age, int)
]).
```

## Comparison with Other Targets

| Feature | Python | C# | Go |
|---------|--------|----|----|
| **JSON Input** | ✅ JSONL | ❌ Manual | ✅ JSONL |
| **JSON Output** | ✅ Native | ❌ Manual | ✅ Native |
| **Type System** | Dynamic | Static (tuples) | Hybrid |
| **Nested Access** | ✅ Yes | ❌ Flat only | ⏳ Planned |
| **Arrays** | ✅ Yes | ❌ No | ⏳ Planned |
| **Performance** | Medium | Fast | Fast |

## References

- Design: `GO_JSON_DESIGN.md`
- Implementation Plan: `GO_JSON_IMPL_PLAN.md`
- Comparison: `GO_JSON_COMPARISON.md`
- Tests: `test_json_input.pl`, `test_json_output.pl`
