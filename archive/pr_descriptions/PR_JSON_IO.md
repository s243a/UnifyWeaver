# Add JSON I/O Support to Go Target

Implements native JSON input/output for the Go target, enabling seamless processing of JSON data streams without external tools.

## Summary

This PR adds complete JSON I/O capability to the Go target through two main features:

1. **JSON Input (Phase 1)**: Parse JSONL (JSON Lines) streams with dynamic typing
2. **JSON Output (Phase 2)**: Generate JSON from delimiter-based input with automatic type conversion

## Features

### ✅ JSON Input (JSONL Parsing)

Parse JSON Lines format with full type support:

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
- Supports all JSON types (string, number, boolean, null, objects, arrays)
- JSONL streaming format (one JSON per line)
- Works with all delimiter options
- Backward compatible with existing features

### ✅ JSON Output (JSON Generation)

Generate JSON from delimiter-based input:

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
- Auto-generated default field names
- Reads any delimiter format

## Implementation

### Commits

1. **4bea8a1** - Phase 1: JSON Input (152 lines)
   - Add `json_input(true)` option detection
   - Implement JSONL parsing with `encoding/json`
   - Dynamic typing via `map[string]interface{}`
   - Support for all JSON types

2. **5055fd0** - Phase 2: JSON Output (276 lines)
   - Add `json_output(true)` option detection
   - Generate Go structs with JSON tags
   - Smart type conversion for output
   - Custom and default field naming

3. **2255e8d** - Documentation (243 lines)
   - Update README with JSON features
   - Create GO_JSON_FEATURES.md with comprehensive docs
   - Usage examples and type reference tables

**Total Changes:**
- **671 additions, 3 deletions**
- Files modified: 1 (`src/unifyweaver/targets/go_target.pl`)
- Files added: 7 (docs, tests, test runners)

### Testing

**Comprehensive test suite with 100% pass rate:**

**JSON Input Tests (`run_json_tests.sh`):**
- ✅ Mixed types (string, number, boolean)
- ✅ Variable field counts (1-4 fields)
- ✅ Unique filtering (unique=true/false)
- ✅ Custom delimiters (colon, tab)
- ✅ Duplicate handling

**JSON Output Tests (`run_json_output_tests.sh`):**
- ✅ Custom field names
- ✅ Default field names
- ✅ Three+ fields with mixed types
- ✅ Tab-delimited input
- ✅ Round-trip: JSON → Delimiter → JSON

## Usage Examples

### Example 1: JSON Transformation Pipeline

```bash
cat users.jsonl | ./parse_json | ./filter_adults | ./to_json
```

### Example 2: Format Conversion

```bash
# CSV → JSON
cat data.csv | ./csv_parser | ./to_json > output.jsonl

# JSON → TSV
cat data.jsonl | ./json_parser | ./format_tsv > output.tsv
```

### Example 3: Round-Trip

```bash
# Preserves types through conversion
cat input.jsonl | ./json_in | ./json_out
```

## Type Handling

### Automatic Type Detection

**JSON Input → Go:**
- JSON strings → Go `string`
- JSON numbers → Go `float64`
- JSON booleans → Go `bool`
- JSON null → Go `nil`
- JSON objects → Go `map[string]interface{}`
- JSON arrays → Go `[]interface{}`

**Go → JSON Output:**
- `"25"` → `25` (integer)
- `"3.14"` → `3.14` (float)
- `"true"` → `true` (boolean)
- `"Alice"` → `"Alice"` (string)

## Design Decisions

### Why JSONL?

- **Streaming**: Process one record at a time
- **Standard**: Widely used format (ndjson)
- **Simple**: One JSON object per line
- **Compatible**: Works with existing Unix tools

### Why Dynamic Typing?

- **Flexibility**: Handles any JSON structure
- **Simplicity**: No schema required
- **Python-like**: Similar to Python target approach
- **Future-proof**: Can add typed structs later (Phase 3+)

## Comparison with Other Targets

| Feature | Python | C# | Go (This PR) |
|---------|--------|----|----|
| **JSON Input** | ✅ JSONL | ❌ Manual | ✅ JSONL |
| **JSON Output** | ✅ Native | ❌ Manual | ✅ Native |
| **Type System** | Dynamic | Static (tuples) | Hybrid |
| **Type Inference** | ✅ Yes | ❌ All strings | ✅ Yes |
| **Performance** | Medium | Fast | Fast |

**Go target advantages:**
- Native JSON types (not all-strings like C#)
- Automatic type conversion for output
- Struct-based marshaling for performance

## Documentation

- **GO_JSON_FEATURES.md**: Complete feature documentation
- **GO_JSON_DESIGN.md**: Design rationale and options
- **GO_JSON_COMPARISON.md**: Comparison with Python/C# approaches
- **GO_JSON_IMPL_PLAN.md**: Implementation roadmap
- **README.md**: Updated with JSON features (v0.1 → v0.2)

## Future Enhancements

Phase 3 and beyond (separate PRs):
- **Nested field access**: `json_get([user, address, city], City)`
- **Array support**: Iterate over JSON arrays
- **Schema support**: Optional typed structs for performance
- **JSONPath queries**: Advanced path expressions

## Backward Compatibility

✅ **Fully backward compatible**
- All existing Go target features work unchanged
- JSON modes are opt-in via options
- No breaking changes to existing code
- Delimiter-based mode remains default

## Testing Instructions

```bash
# Run all JSON input tests
./run_json_tests.sh

# Run all JSON output tests
./run_json_output_tests.sh

# Test round-trip conversion
cat data.jsonl | ./user_json | ./output_custom
```

## Checklist

- ✅ Code compiles and passes all existing tests
- ✅ New features have comprehensive test coverage
- ✅ Documentation updated (README, feature docs)
- ✅ Design documents included
- ✅ Backward compatible
- ✅ No breaking changes
- ✅ Follows existing code style
- ✅ Commit messages are descriptive

## Related Issues

Addresses the need for native JSON processing in the Go target, eliminating the need for external tools like `jq`.

---

**Generated with [Claude Code](https://claude.com/claude-code)**

Co-Authored-By: Claude <noreply@anthropic.com>
