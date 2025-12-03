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

### ✅ Phase 6: Database Read Mode (NEW!)

Read all records from bbolt database and output as JSON.

**Prolog Syntax:**
```prolog
% Simple read predicate - outputs all records from database
read_users :-
    true.  % No body needed for read mode

compile_predicate_to_go(read_users/0, [
    db_backend(bbolt),
    db_file('test_users.db'),
    db_bucket(users),
    db_mode(read),
    package(main)
], Code).
```

**Database Contents:**
```
Key: Alice   → Value: {"name":"Alice","age":30}
Key: Bob     → Value: {"name":"Bob","age":25}
Key: Charlie → Value: {"name":"Charlie","age":35}
```

**Output (JSONL):**
```json
{"age":30,"name":"Alice"}
{"age":25,"name":"Bob"}
{"age":35,"name":"Charlie"}
```

**Key Features:**
- Read-only database access (`&bolt.Options{ReadOnly: true}`)
- Iterates all records in bucket with `bucket.ForEach()`
- Deserializes stored JSON values
- Outputs each record as JSONL to stdout
- Error recovery (continues on individual record failures)
- No modifications to database (safe concurrent reads)

**Generated Code Pattern:**
```go
import bolt "go.etcd.io/bbolt"

// Open database in read-only mode
db, err := bolt.Open("test_users.db", 0600, &bolt.Options{ReadOnly: true})
if err != nil {
    fmt.Fprintf(os.Stderr, "Error opening database: %v\n", err)
    os.Exit(1)
}
defer db.Close()

// Read all records from bucket
err = db.View(func(tx *bolt.Tx) error {
    bucket := tx.Bucket([]byte("users"))
    if bucket == nil {
        return fmt.Errorf("bucket 'users' not found")
    }

    return bucket.ForEach(func(k, v []byte) error {
        // Deserialize JSON record
        var data map[string]interface{}
        if err := json.Unmarshal(v, &data); err != nil {
            fmt.Fprintf(os.Stderr, "Error unmarshaling record: %v\n", err)
            return nil // Continue with next record
        }

        // Output as JSON
        output, err := json.Marshal(data)
        if err != nil {
            fmt.Fprintf(os.Stderr, "Error marshaling output: %v\n", err)
            return nil // Continue with next record
        }

        fmt.Println(string(output))
        return nil
    })
})
```

**Complete Pipeline (Write + Read):**
```bash
# Write: JSON → Schema Validation → Database
echo '{"name": "Alice", "age": 30}' | ./user_store
echo '{"name": "Bob", "age": 25}' | ./user_store

# Read: Database → JSON
./read_users > users.jsonl
cat users.jsonl
# {"age":30,"name":"Alice"}
# {"age":25,"name":"Bob"}
```

**Use Cases:**
1. **Data Export**: Extract all records from database as JSONL
2. **Database Backup**: Export to JSON for archival or migration
3. **Data Transformation**: Read from DB, pipe to another tool
4. **Debugging**: Inspect database contents in human-readable format

**Read vs Write Mode:**

| Feature | Write Mode (`db_mode(write)`) | Read Mode (`db_mode(read)`) |
|---------|------------------------------|----------------------------|
| **Input** | stdin (JSONL) | Database file |
| **Output** | Database file | stdout (JSONL) |
| **Transaction** | `db.Update()` | `db.View()` |
| **Database Access** | Read-write | Read-only |
| **Bucket Creation** | Creates if missing | Fails if missing |
| **Error Handling** | Skip invalid, continue | Skip invalid, continue |
| **Summary** | Reports stored/errors | None (silent on success) |

### ✅ Phase 7: Flexible Key Strategies (NEW!)

Compositional key generation system supporting simple fields, composite keys, hashes, and complex combinations.

**Legacy Syntax (Still Supported):**
```prolog
% Single field key (backward compatible)
db_key_field(name)
```

**New Key Strategy Syntax:**
```prolog
% Single field
db_key_strategy(field(name))

% Composite key (multiple fields)
db_key_strategy(composite([field(name), field(city)]))

% Hash of field
db_key_strategy(hash(field(content)))

% Complex composite (field + hash)
db_key_strategy(composite([field(name), hash(field(content))]))

% Custom delimiter (default: ':')
db_key_delimiter(':')
```

**Example 1: Composite Keys (name + city)**
```prolog
:- json_schema(user_location, [
    field(name, string),
    field(age, integer),
    field(city, string)
]).

user_location(Name, Age, City) :-
    json_record([name-Name, age-Age, city-City]).

compile_predicate_to_go(user_location/3, [
    json_input(true),
    json_schema(user_location),
    db_backend(bbolt),
    db_file('users.db'),
    db_bucket(users),
    db_key_strategy(composite([field(name), field(city)])),
    db_key_delimiter(':'),
    package(main)
], Code).
```

**Input:**
```json
{"name": "Alice", "age": 30, "city": "NYC"}
{"name": "Bob", "age": 25, "city": "SF"}
{"name": "Alice", "age": 28, "city": "LA"}
```

**Database Keys Generated:**
```
Alice:NYC → {"name":"Alice","age":30,"city":"NYC"}
Bob:SF    → {"name":"Bob","age":25,"city":"SF"}
Alice:LA  → {"name":"Alice","age":28,"city":"LA"}
```

**Example 2: Hash Keys (Content-Based)**
```prolog
document(DocName, Content) :-
    json_record([name-DocName, content-Content]).

compile_predicate_to_go(document/2, [
    json_input(true),
    db_backend(bbolt),
    db_file('documents.db'),
    db_bucket(documents),
    db_key_strategy(hash(field(content))),  % SHA-256 hash
    package(main)
], Code).
```

**Input:**
```json
{"name": "doc1", "content": "Hello World"}
{"name": "doc2", "content": "Goodbye World"}
```

**Database Keys Generated:**
```
a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e → {"name":"doc1","content":"Hello World"}
f3c2f1c6e1e0e3b6d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6 → {"name":"doc2","content":"Goodbye World"}
```

**Example 3: Complex Composite (Name + Hash)**
```prolog
compile_predicate_to_go(document/2, [
    json_input(true),
    db_backend(bbolt),
    db_file('docs.db'),
    db_bucket(documents),
    db_key_strategy(composite([
        field(name),
        hash(field(content))
    ])),
    db_key_delimiter(':'),
    package(main)
], Code).
```

**Database Keys Generated:**
```
mydoc:a591a6d40bf420404a011733cfb7b190... → {"name":"mydoc","content":"Hello World"}
otherdoc:f3c2f1c6e1e0e3b6d4e5f6g7h8i9j0... → {"name":"otherdoc","content":"Goodbye World"}
```

**Key Strategy Expression Types:**

| Expression | Description | Example Output |
|-----------|-------------|----------------|
| `field(Name)` | Single field value | `"Alice"` |
| `composite([Expr1, Expr2, ...])` | Concatenate with delimiter | `"Alice:NYC"` |
| `hash(Expr)` | SHA-256 hash (default) | `"a591a6d4..."` |
| `hash(Expr, sha256)` | SHA-256 hash (explicit) | `"a591a6d4..."` |
| `hash(Expr, md5)` | MD5 hash | `"5d41402a..."` |
| `literal(Value)` | Static string | `"prefix"` |

**Key Features:**
- **Compositional**: Nest and combine expressions arbitrarily
- **Type-Safe**: Field validation with schema support
- **Hash Support**: SHA-256 and MD5 cryptographic hashing
- **Automatic Imports**: Hash expressions add `crypto/sha256` and `encoding/hex` automatically
- **Unused Field Handling**: Fields validated by schema but not used in key are marked with `_ = fieldN`
- **Backward Compatible**: Legacy `db_key_field(name)` still works, converted to `db_key_strategy(field(name))`

**Generated Code Pattern (Composite):**
```go
// Schema validation extracts all fields
field1, field1IsString := field1Raw.(string)  // name
field2 := int(field2Float)                     // age
field3, field3IsString := field3Raw.(string)  // city

_ = field2  // Unused in key

// Generate key using strategy
keyStr := fmt.Sprintf("%s:%s", fmt.Sprintf("%v", field1), fmt.Sprintf("%v", field3))
key := []byte(keyStr)

// Store full JSON record
value, err := json.Marshal(data)
bucket.Put(key, value)
```

**Generated Code Pattern (Hash):**
```go
import (
    "crypto/sha256"
    "encoding/hex"
)

// Generate hash-based key
keyStr := func() string {
    valStr := fmt.Sprintf("%v", field2)
    hash := sha256.Sum256([]byte(valStr))
    return hex.EncodeToString(hash[:])
}()
key := []byte(keyStr)
```

**Use Cases:**
1. **Multi-Tenant Keys**: `composite([field(org_id), field(user_id)])` → `"acme:alice"`
2. **Content Deduplication**: `hash(field(content))` → Same content = same key
3. **Versioned Documents**: `composite([field(doc_id), field(version)])` → `"doc1:v2"`
4. **Namespace + Hash**: `composite([literal("data"), hash(field(payload))])` → `"data:a591a6d4..."`
5. **Geographic Partitioning**: `composite([field(country), field(user_id)])` → `"US:12345"`

**Database Options Reference:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `db_key_field(Field)` | atom | First field | Legacy: Single field key (auto-converted) |
| `db_key_strategy(Expr)` | expression | `field(FirstField)` | Flexible key expression |
| `db_key_delimiter(Delim)` | atom | `':'` | Delimiter for composite keys |

**Testing:**
```bash
# Run all key strategy tests
./run_key_strategy_tests.sh

# Tests include:
# 1. Composite keys (name:city)
# 2. Backward compatibility (single field)
# 3. Hash keys (SHA-256)
# 4. Complex composite (name:hash)
```

**Implementation Details:**
- Key expression compiler: `compile_key_expression/5` in `go_target.pl`
- Expression AST with recursive evaluation
- Import tracking propagates hash dependencies
- Unused field detection via `extract_used_field_positions/3`
- Backward compatibility via `normalize_key_strategy/2`

### ✅ Phase 8a: Database Query/Filter Predicates (NEW!)

Filter database records using native Prolog comparison operators in predicate bodies.

**Supported Operators:**
- Numeric: `>`, `<`, `>=`, `=<` (requires numeric fields)
- Equality: `=`, `\=` (works with any type)

**Prolog Syntax:**
```prolog
:- json_schema(user_data, [
    field(name, string),
    field(age, integer),
    field(city, string),
    field(salary, integer)
]).

% Simple age filter
adults(Name, Age) :-
    json_record([name-Name, age-Age, city-_City, salary-_Salary]),
    Age >= 30.

% Multi-field filter with AND
nyc_young_adults(Name, Age) :-
    json_record([name-Name, age-Age, city-City, salary-_Salary]),
    Age > 25,
    City = "NYC".

% Salary range filter
middle_income(Name, Salary) :-
    json_record([name-Name, age-_Age, city-_City, salary-Salary]),
    30000 =< Salary,
    Salary =< 80000.

compile_predicate_to_go(adults/2, [
    db_backend(bbolt),
    db_file('users.db'),
    db_bucket(users),
    db_mode(read),
    package(main)
], Code).
```

**Input (Database):**
```
Key: Alice → {"name": "Alice", "age": 35, "city": "NYC", "salary": 75000}
Key: Bob   → {"name": "Bob", "age": 28, "city": "SF", "salary": 90000}
Key: Eve   → {"name": "Eve", "age": 31, "city": "NYC", "salary": 55000}
```

**Output (adults/2 - Age >= 30):**
```json
{"name":"Alice","age":35}
{"name":"Eve","age":31}
```

**Output (nyc_young_adults/2 - NYC AND Age > 25):**
```json
{"name":"Alice","age":35}
{"name":"Eve","age":31}
```

**Key Features:**
- Native Prolog constraint syntax (no custom DSL)
- Smart type conversion for numeric comparisons
- Automatic field selection (only outputs fields in head)
- String comparison support
- Unused field handling (avoids Go compiler warnings)
- Multiple constraints combined with implicit AND

**Generated Go Code:**
```go
// Extract field with type conversion for constraint
field2Raw, field2Ok := data["age"]
if !field2Ok {
    return nil
}
field2Float, field2FloatOk := field2Raw.(float64)
if !field2FloatOk {
    return nil
}
field2 := field2Float

// Apply filter
if !(field2 >= 30) {
    return nil // Skip record
}

// Output selected fields only
output, err := json.Marshal(map[string]interface{}{"name": field1, "age": field2})
```

**Implementation Details:**
- Constraint extraction: `extract_db_constraints/3` in `go_target.pl`
- Numeric vs equality detection: `is_numeric_constraint/1`
- Smart field extraction: `generate_field_extractions_for_read/4`
- Filter code generation: `generate_filter_checks/3`
- Position-based variable mapping for correct type conversions

**SQL Compatibility:**
The same constraint syntax will map directly to SQL WHERE clauses when SQL target is implemented:
```prolog
Age >= 30  →  WHERE age >= 30
City = "NYC"  →  WHERE city = 'NYC'
```

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
- Tests: `test_json_input.pl`, `test_json_output.pl`, `test_json_nested.pl`, `test_schema_*.pl`, `test_bbolt_*.pl`, `test_composite_keys.pl`
- Test Runners: `run_json_tests.sh`, `run_json_output_tests.sh`, `run_nested_tests.sh`, `run_schema_tests.sh`, `run_bbolt_tests.sh`, `run_key_strategy_tests.sh`
