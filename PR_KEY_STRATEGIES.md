# Add Flexible Database Key Strategies for bbolt (Phase 7)

## Summary

Implements a compositional key generation system for bbolt database integration, supporting:
- **Simple field keys**: `field(name)` → `"Alice"`
- **Composite keys**: `composite([field(name), field(city)])` → `"Alice:NYC"`
- **Hash-based keys**: `hash(field(content))` → `"a591a6d40bf420404a011733cfb7b190..."`
- **Complex combinations**: `composite([field(name), hash(field(content))])` → `"mydoc:a591a6d4..."`

This provides the flexibility to handle diverse key generation requirements including multi-tenant systems, content deduplication, document versioning, and geographic partitioning.

## Technical Implementation

### Key Expression Compiler (`go_target.pl`)

Added ~300 lines of code implementing a compositional AST-based key expression compiler:

```prolog
% Expression types:
compile_key_expr(field(Name), ...)          % Single field access
compile_key_expr(composite([...]), ...)     % Concatenation with delimiter
compile_key_expr(hash(Expr), ...)           % SHA-256 hash (default)
compile_key_expr(hash(Expr, Algorithm), ...)% Explicit hash algorithm
compile_key_expr(literal(String), ...)      % Static string constant
```

**Key Features:**
- **Recursive Evaluation**: Expressions can be nested arbitrarily (e.g., `hash(composite([field(a), field(b)]))`)
- **Import Tracking**: Hash expressions automatically add `crypto/sha256` and `encoding/hex` imports
- **Unused Field Detection**: Fields validated by schema but not used in key are marked with `_ = fieldN` to avoid Go "unused variable" errors
- **Backward Compatibility**: Legacy `db_key_field(name)` auto-converts to `db_key_strategy(field(name))`

### Generated Code Examples

**Composite Key (name:city):**
```go
// Schema validation extracts all fields
field1, field1IsString := field1Raw.(string)  // name
field2 := int(field2Float)                     // age
field3, field3IsString := field3Raw.(string)  // city

_ = field2  // Unused in key - avoids Go error

// Generate composite key
keyStr := fmt.Sprintf("%s:%s",
    fmt.Sprintf("%v", field1),
    fmt.Sprintf("%v", field3))
key := []byte(keyStr)
```

**Hash Key (SHA-256 of content):**
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

## Use Cases

1. **Multi-Tenant Keys**: `composite([field(org_id), field(user_id)])` → `"acme:alice"`
   - Partitions data by organization and user
   - Natural hierarchical structure

2. **Content Deduplication**: `hash(field(content))` → `"a591a6d4..."`
   - Same content produces same key
   - Automatic deduplication at storage layer

3. **Versioned Documents**: `composite([field(doc_id), field(version)])` → `"doc1:v2"`
   - Multiple versions of same document
   - Easy range queries by document ID

4. **Namespace + Hash**: `composite([literal("data"), hash(field(payload))])` → `"data:a591a6d4..."`
   - Prefix-based organization
   - Content-addressed with namespace

5. **Geographic Partitioning**: `composite([field(country), field(user_id)])` → `"US:12345"`
   - Data locality for compliance
   - Regional access patterns

## Testing

Created comprehensive test suite with 4 test cases:

```bash
./run_key_strategy_tests.sh
```

**Test Coverage:**
- ✅ **Test 1**: Composite keys (name + city) → 3 records stored
- ✅ **Test 2**: Backward compatibility (single field) → 2 records stored
- ✅ **Test 3**: Hash keys (SHA-256 of content) → 2 records stored
- ✅ **Test 4**: Complex composite (name + hash) → 2 records stored

**All tests pass with 100% success rate.**

## API Reference

### Database Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `db_key_field(Field)` | atom | First field | Legacy: Single field key (auto-converted) |
| `db_key_strategy(Expr)` | expression | `field(FirstField)` | Flexible key expression |
| `db_key_delimiter(Delim)` | atom | `':'` | Delimiter for composite keys |

### Key Expression Types

| Expression | Description | Example Output |
|-----------|-------------|----------------|
| `field(Name)` | Single field value | `"Alice"` |
| `composite([Expr1, Expr2, ...])` | Concatenate with delimiter | `"Alice:NYC"` |
| `hash(Expr)` | SHA-256 hash (default) | `"a591a6d4..."` |
| `hash(Expr, sha256)` | SHA-256 hash (explicit) | `"a591a6d4..."` |
| `hash(Expr, md5)` | MD5 hash | `"5d41402a..."` |
| `literal(Value)` | Static string | `"prefix"` |

### Example Usage

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

**Database Keys:**
```
Alice:NYC → {"name":"Alice","age":30,"city":"NYC"}
Bob:SF    → {"name":"Bob","age":25,"city":"SF"}
Alice:LA  → {"name":"Alice","age":28,"city":"LA"}
```

## Files Changed

- `src/unifyweaver/targets/go_target.pl` - Core implementation (~300 lines added)
  - `compile_key_expression/5` - Main entry point
  - `compile_key_expr/5` - Expression compiler with pattern matching
  - `normalize_key_strategy/2` - Backward compatibility
  - `extract_used_field_positions/3` - Unused field detection
  - Modified `wrap_with_database/6` to integrate key compiler

- `GO_JSON_FEATURES.md` - Documentation (~200 lines added)
  - Added Phase 7 section with complete API reference
  - Examples for all key strategies
  - Use cases and generated code patterns
  - Updated references section

- `test_composite_keys.pl` - Test suite (129 lines)
  - 4 comprehensive test cases
  - Covers all key strategy types
  - Includes backward compatibility test

- `run_key_strategy_tests.sh` - Test runner (125 lines)
  - Automated build and test execution
  - Database verification instructions
  - Summary reporting

**Total: ~724 insertions(+), ~27 deletions(-)**

## Breaking Changes

None. This is a pure feature addition that's fully backward compatible:
- Legacy `db_key_field(name)` still works (auto-converts internally)
- Existing code without key strategies continues to use first field as key
- No changes to generated code structure for default behavior

## Future Enhancements

The compositional system is designed for extensibility:
- `uuid()` - Generate UUID v4 keys
- `auto_increment()` - Sequential counter
- `substring(Expr, Start, Len)` - Extract substring
- `timestamp()` - Current timestamp
- `lowercase(Expr)` - Normalize to lowercase

## Documentation

Complete documentation added to `GO_JSON_FEATURES.md`:
- Phase 7 section with syntax reference
- 3 detailed examples (composite, hash, complex)
- API reference table
- Use case descriptions
- Generated code patterns
- Testing instructions

🤖 Generated with [Claude Code](https://claude.com/claude-code)
