# Add bbolt Database Read Mode (Phase 6)

## Summary

Implements Phase 6 of the Go target bbolt integration: **database read mode** for extracting records from bbolt databases as JSONL.

This completes the bidirectional data flow:
- **Phase 5 (Write Mode)**: JSON → Schema Validation → Database Storage
- **Phase 6 (Read Mode)**: Database → JSON Output

## Changes

### Core Implementation (`go_target.pl`)

Added `compile_database_read_mode/4` predicate (lines 1425-1497):
- Opens database in read-only mode with `&bolt.Options{ReadOnly: true}`
- Uses `db.View()` transaction (read-only, not write)
- Iterates all records with `bucket.ForEach()`
- Deserializes stored JSON values
- Outputs each record as JSONL to stdout
- Error recovery pattern (continues on individual record failures)

### Generated Code

**Read-Only Database Access:**
```go
db, err := bolt.Open("test_users.db", 0600, &bolt.Options{ReadOnly: true})
if err != nil {
    fmt.Fprintf(os.Stderr, "Error opening database: %v\n", err)
    os.Exit(1)
}
defer db.Close()
```

**Bucket Iteration:**
```go
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

### Testing

Created comprehensive test suite:

**`test_bbolt_read.pl`** - Test file for read mode compilation
```prolog
read_users :-
    true.  % No body needed for read mode

test_read_bbolt :-
    compile_predicate_to_go(read_users/0, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),
    % ... write generated code ...
```

**Updated `run_bbolt_tests.sh`** - Added Test 3 for read mode with verification:
- Compiles and builds the read mode program
- Copies database from Test 1 (write mode)
- Runs the program and captures output
- Verifies all expected records are present (Alice, Bob, Charlie)
- Validates record count (expects exactly 3 records)

### Test Results

All tests pass with 100% success rate:

```bash
Test 3: Reading from bbolt database
------------------------------------
Generated: output_bbolt_read/read_users.go
Building Go program...
Reading all records from database...
Verifying output matches expected records...
✓ All expected records found
✓ Correct number of records (3)
✓ Test 3 passed
```

**Output:**
```json
{"age":30,"name":"Alice"}
{"age":25,"name":"Bob"}
{"age":35,"name":"Charlie"}
```

### Documentation

Updated documentation files:

**`GO_JSON_FEATURES.md`** - Added Phase 6 section (lines 479-590):
- Usage examples for read mode
- Generated code patterns
- Complete pipeline demonstration (write + read)
- Read vs Write mode comparison table
- Use cases (data export, backup, transformation, debugging)

**`BBOLT_INTEGRATION_DESIGN.md`** - Updated Phase 3 (lines 333-435):
- Marked Phase 3 as ✅ COMPLETE
- Added implementation details and generated code
- Included test results and complete pipeline examples

## Files Changed

- `src/unifyweaver/targets/go_target.pl` - Core implementation (73 lines added)
- `test_bbolt_read.pl` - Test file (29 lines, new)
- `output_bbolt_read/read_users.go` - Generated code (52 lines, new)
- `run_bbolt_tests.sh` - Test runner (41 lines added for Test 3)
- `GO_JSON_FEATURES.md` - Documentation (112 lines added)
- `BBOLT_INTEGRATION_DESIGN.md` - Design documentation (107 lines added)

**Total: 414 insertions(+)**

## Complete Pipeline

The implementation enables complete bidirectional data flow:

```bash
# Phase 5: Write Mode - JSON → Database
cat users.jsonl | ./user_store
# Stored 3 records, 0 errors

# Phase 6: Read Mode - Database → JSON
./read_users > output.jsonl
cat output.jsonl
# {"age":30,"name":"Alice"}
# {"age":25,"name":"Bob"}
# {"age":35,"name":"Charlie"}
```

## Key Features

✅ **Read-Only Access** - Safe concurrent reads with `&bolt.Options{ReadOnly: true}`
✅ **Complete Iteration** - `bucket.ForEach()` visits all records
✅ **JSONL Output** - One JSON object per line to stdout
✅ **Error Recovery** - Continues processing on individual record failures
✅ **Simple Syntax** - Minimal Prolog code, database operations inferred from options
✅ **Zero Dependencies** - Pure Go with bbolt (no external dependencies)

## Use Cases

1. **Data Export**: Extract all records from database as JSONL
2. **Database Backup**: Export to JSON for archival or migration
3. **Data Transformation**: Read from DB, pipe to another processing tool
4. **Debugging**: Inspect database contents in human-readable format

## Breaking Changes

None. This is a pure feature addition that's backwards compatible with existing bbolt write mode functionality.

## Test Plan

```bash
# Run all bbolt tests including read mode
./run_bbolt_tests.sh
```

All 3 test cases pass:
- ✅ Test 1: Basic bbolt storage (write mode)
- ✅ Test 2: Schema validation with storage
- ✅ Test 3: Database read mode (NEW!)

## Next Steps

Phase 6 completes the core bidirectional database I/O. Future enhancements (Phase 4+) could include:
- Batch optimization for write mode (Phase 2)
- Filtering/queries for read mode (key prefix, range queries)
- Composite keys and secondary indexes
- Additional database backends (BadgerDB, Pebble)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
