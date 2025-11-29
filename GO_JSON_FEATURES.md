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

### ✅ Phase 3: Nested Field Access (NEW!)

Access nested JSON structures with path-based extraction.

**Prolog Syntax:**
```prolog
% Simple nested (2 levels)
city(City) :- json_get([user, city], City).

% Deep nested (3+ levels)
location(City) :- json_get([user, address, city], City).

% Multiple nested fields
user_info(Name, City) :-
    json_get([user, name], Name),
    json_get([user, address, city], City).

% Mixed flat and nested
data(Id, City) :-
    json_record([id-Id]),
    json_get([location, city], City).
```

**Input:**
```json
{"user": {"address": {"city": "NYC", "zip": "10001"}}}
```

**Output:**
```
NYC
```

**Key Features:**
- Path-based access with `json_get([path, to, field], Var)`
- Supports arbitrary nesting depth (2, 3, 4+ levels)
- Mix flat (`json_record`) and nested (`json_get`) in same predicate
- Automatic helper function generation
- Type-safe traversal with existence checking

**Generated Helper:**
```go
func getNestedField(data map[string]interface{}, path []string) (interface{}, bool) {
    current := interface{}(data)
    for _, key := range path {
        currentMap, ok := current.(map[string]interface{})
        if !ok {
            return nil, false
        }
        value, exists := currentMap[key]
        if !exists {
            return nil, false
        }
        current = value
    }
    return current, true
}
```

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

### ✅ Phase 4: Schema Support (NEW!)

Define JSON schemas for type-safe field extraction and validation.

**Prolog Syntax:**
```prolog
% Define schema with typed fields
:- json_schema(user, [
    field(name, string),
    field(age, integer)
]).

% Use schema in predicates
user(Name, Age) :- json_record([name-Name, age-Age]).

% Compile with schema
compile_predicate_to_go(user/2, [
    json_input(true),
    json_schema(user)
], Code).
```

**Generated Go (Type-Safe):**
```go
// String field - type validated
field1Raw, field1RawOk := data["name"]
if !field1RawOk {
    continue
}
field1, field1IsString := field1Raw.(string)
if !field1IsString {
    fmt.Fprintf(os.Stderr, "Error: field 'name' is not a string\n")
    continue
}

// Integer field - type validated with conversion
field2Raw, field2RawOk := data["age"]
if !field2RawOk {
    continue
}
field2Float, field2FloatOk := field2Raw.(float64)
if !field2FloatOk {
    fmt.Fprintf(os.Stderr, "Error: field 'age' is not a number\n")
    continue
}
field2 := int(field2Float)
```

**Input:**
```json
{"name": "Alice", "age": 25}
{"name": "Bob", "age": "thirty"}
```

**Output:**
```
Alice:25
Error: field 'age' is not a number
```

**Supported Types:**
- `string` - String values
- `integer` - Integer numbers (auto-converts from JSON float64)
- `float` - Float64 numbers
- `boolean` - Boolean values
- `any` - Untyped (interface{}, fallback)

**Key Features:**
- Type-safe field extraction at compile time
- Runtime type validation with clear error messages
- Works with both flat (`json_record`) and nested (`json_get`) fields
- Optional - existing code without schemas continues to work
- Error messages to stderr, invalid records skipped
- Zero performance overhead for valid data

## Future Enhancements (Phase 5+)

### Array Support (Planned)

```prolog
user_name(Name) :-
    json_get([users], UserList),
    json_array_member(UserList, User),
    json_get(User, [name], Name).
```

### Advanced Schema Features (Planned)

```prolog
:- json_schema(user, [
    field(age, integer, [min(0), max(150)]),
    field(email, string, [format(email)]),
    field(tags, array(string))
]).
```

### ✅ Phase 5: Database Integration (bbolt) (NEW!)

Store validated JSON records in embedded bbolt database.

**Prolog Syntax:**
```prolog
:- json_schema(user, [
    field(name, string),
    field(age, integer),
    field(email, string)
]).

user(Name, Age, Email) :-
    json_record([name-Name, age-Age, email-Email]).

compile_predicate_to_go(user/3, [
    json_input(true),
    json_schema(user),
    db_backend(bbolt),
    db_file('users.db'),
    db_bucket(users),
    db_key_field(name)
], Code).
```

**Input:**
```json
{"name": "Alice", "age": 30, "email": "alice@example.com"}
{"name": "Bob", "age": "invalid", "email": "bob@example.com"}
{"name": "Charlie", "age": 35, "email": "charlie@example.com"}
```

**Output (stderr):**
```
Error: field 'age' is not a number
Stored 2 records, 0 errors
```

**Database Contents:**
```
Key: Alice   → Value: {"name":"Alice","age":30,"email":"alice@example.com"}
Key: Charlie → Value: {"name":"Charlie","age":35,"email":"charlie@example.com"}
```

**Key Features:**
- Embedded database (bbolt) - no external dependencies
- ACID transactions with B+tree storage
- Schema validation before storage
- Configurable key field selection
- Full JSON record storage as values
- Error tracking and summary reporting
- Continues processing on validation errors

**Database Options:**
- `db_backend(bbolt)` - Database backend (only bbolt currently supported)
- `db_file(Path)` - Database file path (default: `'data.db'`)
- `db_bucket(Name)` - Bucket name like table/collection (default: predicate name)
- `db_key_field(Field)` - Which field to use as key (default: first field)
- `db_mode(read|write)` - Operation mode (default: write with json_input)

**Use Cases:**
1. **JSON Data Ingestion**: `cat users.jsonl | ./ingest_users`
2. **Data Validation + Storage**: Invalid records rejected, valid stored
3. **ETL Pipelines**: JSON → Validate → Transform → Store
4. **Persistent Storage**: Alternative to file-based output

**Complete Pipeline:**
```prolog
% JSON Input → Schema Validation → Database Storage
:- json_schema(user, [
    field(name, string),
    field(age, integer)
]).

ingest_user(Name, Age) :-
    json_record([name-Name, age-Age]).

:- compile_predicate_to_go(ingest_user/2, [
    json_input(true),      % Parse JSONL
    json_schema(user),     % Validate types
    db_backend(bbolt),     % Store in bbolt
    db_file('users.db'),
    db_bucket(users),
    db_key_field(name)
], Code).
```

**Generated Code Pattern:**
```go
import bolt "go.etcd.io/bbolt"

db, _ := bolt.Open("users.db", 0600, nil)
defer db.Close()

// Create bucket
db.Update(func(tx *bolt.Tx) error {
    _, err := tx.CreateBucketIfNotExists([]byte("users"))
    return err
})

// Process JSON input
for scanner.Scan() {
    var data map[string]interface{}
    json.Unmarshal(scanner.Bytes(), &data)

    // Type-safe extraction with schema
    name, nameOk := data["name"].(string)
    ageFloat, ageOk := data["age"].(float64)
    age := int(ageFloat)

    // Store in database
    db.Update(func(tx *bolt.Tx) error {
        bucket := tx.Bucket([]byte("users"))
        key := []byte(name)
        value, _ := json.Marshal(data)
        return bucket.Put(key, value)
    })
}
```

**Integration with Existing Features:**
- ✅ Works with JSON input (Phase 1)
- ✅ Works with schema validation (Phase 4)
- ✅ Works with nested field extraction (Phase 3)
- ✅ No stdout output in database mode (only stderr for errors/summary)

## Comparison with Other Targets

| Feature | Python | C# | Go |
|---------|--------|----|----|
| **JSON Input** | ✅ JSONL | ❌ Manual | ✅ JSONL |
| **JSON Output** | ✅ Native | ❌ Manual | ✅ Native |
| **Type System** | Dynamic | Static (tuples) | Hybrid |
| **Nested Access** | ✅ Yes | ❌ Flat only | ✅ Yes |
| **Schema Validation** | ⏳ Planned | ❌ No | ✅ Yes |
| **Database Storage** | ✅ SQLite | ✅ LiteDB | ✅ bbolt |
| **Arrays** | ✅ Yes | ❌ No | ⏳ Planned |
| **Performance** | Medium | Fast | Fast |

## References

- Design: `GO_JSON_DESIGN.md`
- Implementation Plan: `GO_JSON_IMPL_PLAN.md`
- Comparison: `GO_JSON_COMPARISON.md`
- Schema Design: `JSON_SCHEMA_DESIGN.md`
- Database Design: `BBOLT_INTEGRATION_DESIGN.md`
- Tests: `test_json_input.pl`, `test_json_output.pl`, `test_json_nested.pl`, `test_schema_*.pl`, `test_bbolt_*.pl`
- Test Runners: `run_json_tests.sh`, `run_json_output_tests.sh`, `run_nested_tests.sh`, `run_schema_tests.sh`, `run_bbolt_tests.sh`
