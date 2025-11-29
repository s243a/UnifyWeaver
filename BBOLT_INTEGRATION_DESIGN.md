# bbolt Database Integration Design

## Overview

Add embedded database support to the Go target using bbolt, enabling complete data pipelines: **JSON Input → Schema Validation → Database Storage**.

## Goals

1. **Persistent Storage** - Store validated JSON records in bbolt database
2. **ACID Transactions** - Guarantee data consistency
3. **Simple API** - Intuitive Prolog syntax for database operations
4. **Integration** - Seamless integration with JSON I/O and schemas
5. **Performance** - Efficient batch operations and transactions
6. **Zero Dependencies** - Pure Go (bbolt has no external dependencies)

## Use Cases

### Use Case 1: JSON Data Ingestion
```bash
# Ingest JSONL data into database
cat users.jsonl | ./ingest_users
```

### Use Case 2: Data Validation + Storage
```bash
# Validate with schema, store valid records, report errors
cat data.jsonl | ./validate_and_store 2>errors.log
```

### Use Case 3: Query Database
```bash
# Read from database and output as JSON
./query_users > users.jsonl
```

## Prolog Syntax Design

### Option 1: Implicit Database (RECOMMENDED)

Simple syntax where database operations are inferred from compilation options:

```prolog
% Define schema
:- json_schema(user, [
    field(name, string),
    field(age, integer),
    field(email, string)
]).

% Simple predicate - database storage inferred from options
store_user(Name, Age, Email) :-
    json_record([name-Name, age-Age, email-Email]).

% Compile with database backend
compile_predicate_to_go(store_user/3, [
    json_input(true),
    json_schema(user),
    db_backend(bbolt),
    db_file('users.db'),
    db_bucket(users),
    db_key_field(name)  % Use 'name' field as key
], Code).
```

**Pros:**
- Clean Prolog code
- Database is deployment concern, not logic concern
- Easy to switch between file output and database storage

**Cons:**
- Database operations not visible in predicate
- Less explicit about side effects

### Option 2: Explicit Database Operations

Explicit `db_store/3` predicate in the body:

```prolog
% Define schema
:- json_schema(user, [
    field(name, string),
    field(age, integer),
    field(email, string)
]).

% Explicit database store operation
store_user(Name, Age, Email) :-
    json_record([name-Name, age-Age, email-Email]),
    db_store(users, Name, [name-Name, age-Age, email-Email]).

% Compile with database backend
compile_predicate_to_go(store_user/3, [
    json_input(true),
    json_schema(user),
    db_backend(bbolt),
    db_file('users.db')
], Code).
```

**Pros:**
- Explicit side effects
- Clear what's being stored
- Flexible - can store subset of fields

**Cons:**
- More verbose
- Mixes logic with storage concerns

### Decision: Use Option 1 (Implicit)

**Rationale:** Keep Prolog logic clean and declarative. Database storage is a compilation/deployment concern, not a logical concern. This matches the philosophy of separating "what" (logic) from "how" (implementation).

## Compilation Options

### Database Options

```prolog
compile_predicate_to_go(Predicate, [
    % Existing options
    json_input(true),
    json_schema(SchemaName),

    % New database options
    db_backend(bbolt),              % Database backend (only bbolt for now)
    db_file('data.db'),             % Database file path
    db_bucket(BucketName),          % Bucket name (like table/collection)
    db_key_field(FieldName),        % Which field to use as key
    db_mode(Mode)                   % insert | update | upsert (default: upsert)
], Code).
```

### Option Details

**`db_backend(bbolt)`**
- Currently only supports `bbolt`
- Future: `badger`, `pebble`, etc.

**`db_file(Path)`**
- Path to database file
- Created if doesn't exist
- Default: `'data.db'`

**`db_bucket(Name)`**
- Bucket name (bbolt's equivalent of table/collection)
- Created if doesn't exist
- Default: predicate name

**`db_key_field(Field)`**
- Which field to use as the key
- Must be a field in the schema
- For composite keys, use `db_key_fields([Field1, Field2])`
- Default: first field

**`db_mode(Mode)`**
- `insert` - Fail if key exists
- `update` - Fail if key doesn't exist
- `upsert` - Insert or update (default)

## Generated Go Code Pattern

### Basic Structure

```go
package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "os"

    bolt "go.etcd.io/bbolt"
)

func main() {
    // Open database
    db, err := bolt.Open("users.db", 0600, nil)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error opening database: %v\n", err)
        os.Exit(1)
    }
    defer db.Close()

    // Create bucket if needed
    err = db.Update(func(tx *bolt.Tx) error {
        _, err := tx.CreateBucketIfNotExists([]byte("users"))
        return err
    })
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error creating bucket: %v\n", err)
        os.Exit(1)
    }

    // Process input
    scanner := bufio.NewScanner(os.Stdin)
    recordCount := 0
    errorCount := 0

    for scanner.Scan() {
        var data map[string]interface{}
        if err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
            errorCount++
            fmt.Fprintf(os.Stderr, "JSON parse error: %v\n", err)
            continue
        }

        // Type-safe extraction (with schema)
        nameRaw, nameOk := data["name"]
        if !nameOk {
            errorCount++
            continue
        }
        name, nameIsString := nameRaw.(string)
        if !nameIsString {
            errorCount++
            fmt.Fprintf(os.Stderr, "Error: field 'name' is not a string\n")
            continue
        }

        ageRaw, ageOk := data["age"]
        if !ageOk {
            errorCount++
            continue
        }
        ageFloat, ageIsNum := ageRaw.(float64)
        if !ageIsNum {
            errorCount++
            fmt.Fprintf(os.Stderr, "Error: field 'age' is not a number\n")
            continue
        }
        age := int(ageFloat)

        // Store in database
        err = db.Update(func(tx *bolt.Tx) error {
            bucket := tx.Bucket([]byte("users"))

            // Use name as key
            key := []byte(name)

            // Serialize entire record as value
            value, err := json.Marshal(data)
            if err != nil {
                return err
            }

            return bucket.Put(key, value)
        })

        if err != nil {
            errorCount++
            fmt.Fprintf(os.Stderr, "Database error: %v\n", err)
            continue
        }

        recordCount++
    }

    // Summary
    fmt.Fprintf(os.Stderr, "Stored %d records, %d errors\n", recordCount, errorCount)
}
```

### Batch Optimization

For better performance, batch multiple records per transaction:

```go
const batchSize = 1000
batch := make([]Record, 0, batchSize)

for scanner.Scan() {
    // ... parse and validate ...

    batch = append(batch, record)

    if len(batch) >= batchSize {
        // Store batch in single transaction
        err := db.Update(func(tx *bolt.Tx) error {
            bucket := tx.Bucket([]byte("users"))
            for _, rec := range batch {
                key := []byte(rec.Name)
                value, _ := json.Marshal(rec)
                if err := bucket.Put(key, value); err != nil {
                    return err
                }
            }
            return nil
        })
        if err != nil {
            fmt.Fprintf(os.Stderr, "Batch error: %v\n", err)
        } else {
            recordCount += len(batch)
        }
        batch = batch[:0] // Reset batch
    }
}

// Store remaining records
if len(batch) > 0 {
    // ... store final batch ...
}
```

## Implementation Plan

### Phase 1: Basic Storage (Minimum Viable Product)

**Goal:** Store JSON records in bbolt with schema validation

**Features:**
- Open/close database
- Create bucket
- Store validated records
- Use specified field as key
- Error reporting

**Implementation Steps:**
1. Add database option parsing to `compile_predicate_to_go/3`
2. Create `compile_database_mode/4` predicate
3. Generate database initialization code
4. Generate storage code within input loop
5. Add error tracking and summary

### Phase 2: Batch Optimization

**Goal:** Improve performance with batch transactions

**Features:**
- Configurable batch size (`db_batch_size(N)`)
- Single transaction per batch
- Progress reporting option

### Phase 3: Read Operations

**Goal:** Query data from database

**Features:**
- `db_backend(bbolt)` with `db_mode(read)`
- Iterate over bucket contents
- Output as JSON or delimited format
- Optional filtering

```prolog
% Read from database and output as JSON
read_users(Name, Age, Email) :-
    db_read(users, Name, [name-Name, age-Age, email-Email]).

compile_predicate_to_go(read_users/3, [
    json_output(true),
    db_backend(bbolt),
    db_file('users.db'),
    db_bucket(users),
    db_mode(read)
], Code).
```

### Phase 4: Advanced Features

- Composite keys
- Secondary indexes (via separate buckets)
- Range queries (iterate with prefix)
- Conditional updates (check before update)

## Error Handling

### Database Errors

```go
err := db.Update(func(tx *bolt.Tx) error {
    // ... operations ...
})
if err != nil {
    fmt.Fprintf(os.Stderr, "Database error: %v\n", err)
    errorCount++
    continue // Continue processing next record
}
```

### Validation Errors

Schema validation errors should:
1. Log to stderr with descriptive message
2. Increment error counter
3. Continue processing (don't abort on single error)
4. Report summary at end

### Exit Codes

- `0` - Success (all records processed)
- `1` - Fatal error (database open failed, bucket creation failed)
- `2` - Partial failure (some records failed validation/storage)

## Testing Strategy

### Test Cases

1. **Basic Storage** - Store simple records, verify with bbolt CLI
2. **Schema Validation** - Invalid records rejected, valid stored
3. **Duplicate Keys** - Upsert behavior (overwrite)
4. **Batch Processing** - Large dataset (10k+ records)
5. **Error Recovery** - Continue after validation errors
6. **Empty Input** - Handle gracefully
7. **Database Permissions** - Handle file permission errors

### Test Files

- `test_bbolt_basic.pl` - Basic store operation
- `test_bbolt_schema.pl` - With schema validation
- `test_bbolt_batch.pl` - Batch processing
- `run_bbolt_tests.sh` - Automated test runner

### Verification

Use bbolt CLI to verify stored data:

```bash
# Install bbolt CLI
go install go.etcd.io/bbolt/cmd/bbolt@latest

# Inspect database
bbolt info users.db
bbolt keys users.db users
bbolt get users.db users "Alice"
```

## Integration with Existing Features

### JSON Input + Schema + Database

Perfect pipeline integration:

```prolog
:- json_schema(user, [
    field(name, string),
    field(age, integer),
    field(email, string)
]).

ingest_user(Name, Age, Email) :-
    json_record([name-Name, age-Age, email-Email]).

:- compile_predicate_to_go(ingest_user/3, [
    json_input(true),           % Parse JSONL
    json_schema(user),          % Validate types
    db_backend(bbolt),          % Store in bbolt
    db_file('users.db'),
    db_bucket(users),
    db_key_field(name)
], Code),
   write_go_program(Code, 'ingest_users.go').
```

**Result:** Type-safe JSON ingestion with persistent storage!

### Nested Fields

Works with nested field extraction:

```prolog
:- json_schema(profile, [
    field(id, string),
    field(name, string),
    field(city, string)
]).

store_profile(Id, Name, City) :-
    json_get([user, id], Id),
    json_get([user, name], Name),
    json_get([user, address, city], City).

:- compile_predicate_to_go(store_profile/3, [
    json_input(true),
    json_schema(profile),
    db_backend(bbolt),
    db_key_field(id)
], Code).
```

## Performance Considerations

### Write Performance

- **Batch Size:** Default 1000 records per transaction
- **Write Amplification:** bbolt uses B+tree, minimal WAL overhead
- **Sync Policy:** By default, bbolt syncs on every commit (safe but slower)
- **NoSync Option:** Can enable `NoSync` for testing (faster but unsafe)

### Database Size

- **Storage:** ~1.5x JSON size (due to B+tree overhead)
- **Compaction:** bbolt doesn't auto-compact, may need periodic `db.Update(func(tx *Tx) error { return tx.Compact() })`

### Concurrency

- **Single Writer:** bbolt allows one writer at a time (ACID guarantee)
- **Multiple Readers:** Concurrent reads are supported
- **Pipeline:** For this use case (stdin ingestion), single writer is fine

## Documentation Updates

### GO_JSON_FEATURES.md

Add new section: "Phase 5: Database Integration"

### README.md

Update Go target to v0.5, add "Database Storage" bullet

### New Document

Create `GO_BBOLT_GUIDE.md` with:
- Installation instructions
- Usage examples
- CLI tools reference
- Performance tuning
- Troubleshooting

## Migration Path

### From File Output to Database

**Before (file output):**
```prolog
compile_predicate_to_go(user/2, [json_input(true)], Code).
```

**After (database):**
```prolog
compile_predicate_to_go(user/2, [
    json_input(true),
    db_backend(bbolt),
    db_file('users.db')
], Code).
```

**No change to predicate definition!** Just add database options.

## Future Enhancements

### Phase 6: Multiple Backends

Support additional databases:
- `db_backend(badger)` - BadgerDB
- `db_backend(pebble)` - Pebble
- `db_backend(sqlite)` - SQLite (via CGo)

### Phase 7: Queries

Support reading from database with filtering:

```prolog
% Query by key prefix
users_starting_with_a(Name, Age) :-
    db_query(users, Name, [prefix("A"), limit(100)]),
    db_get(users, Name, [name-Name, age-Age]).
```

### Phase 8: Indexes

Secondary indexes for efficient lookups:

```prolog
:- db_index(users, email, unique).
:- db_index(users, [city, age], composite).
```

## Open Questions

1. **Transaction Scope** - Should each record be a transaction, or batch multiple records?
   - **Decision:** Batch by default (better performance)

2. **Key Serialization** - How to handle composite keys?
   - **Decision:** Start with single field, add composite later

3. **Value Format** - Store full JSON or just specified fields?
   - **Decision:** Store full JSON record (flexible, matches input)

4. **Error Reporting** - Fail fast or collect all errors?
   - **Decision:** Collect errors, report summary (continue processing)

5. **Database Lifecycle** - When to close database?
   - **Decision:** Use `defer db.Close()` (guaranteed cleanup)

## Summary

This design provides:
- ✅ Clean Prolog syntax (implicit database operations)
- ✅ Seamless integration with JSON I/O and schemas
- ✅ Efficient batch processing with ACID transactions
- ✅ Comprehensive error handling and reporting
- ✅ Pure Go with zero external dependencies
- ✅ Clear migration path from file-based to database storage

**Next Step:** Implement Phase 1 (Basic Storage) with tests.
