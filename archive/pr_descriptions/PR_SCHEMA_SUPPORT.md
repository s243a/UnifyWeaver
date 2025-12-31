# Add JSON schema support (Phase 4)

## Summary

Implements Phase 4 of the Go target JSON I/O feature set: **JSON Schema Support** for type-safe field extraction with runtime validation.

This adds the ability to define typed JSON schemas and automatically generate type-safe Go code with validation:

```prolog
% Define schema with typed fields
:- json_schema(user, [
    field(name, string),
    field(age, integer)
]).

% Use schema in predicates
user(Name, Age) :- json_record([name-Name, age-Age]).

% Compile with type safety
compile_predicate_to_go(user/2, [
    json_input(true),
    json_schema(user)
], Code).
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

## Changes

### Core Implementation (`go_target.pl:23-78, 1438-1441, 1714-1824`)

**Schema Infrastructure:**
- Added `json_schema/2` directive to define schemas with typed fields
- Implemented `validate_schema_fields/1` for schema validation
- Added `get_json_schema/2` and `get_field_type/3` for schema lookup
- Dynamic storage via `json_schema_def/2` facts
- Exported schema predicates from module interface

**Typed Compilation:**
- Implemented `compile_json_to_go_typed/6` for type-safe JSON input
- Added `generate_typed_field_extractions/4` to generate typed extraction code
- Implemented `generate_typed_flat_field_extraction/4` for flat field validation
- Implemented `generate_typed_nested_field_extraction/4` for nested field validation
- Modified `compile_json_input_mode/4` to detect and use schemas when provided

**Supported Types:**
- `string` - String values
- `integer` - Integer numbers (auto-converts from JSON float64)
- `float` - Float64 numbers
- `boolean` - Boolean values
- `any` - Untyped (interface{}, fallback to current behavior)

### Generated Code Examples

**String Type Validation:**
```go
field1Raw, field1RawOk := data["name"]
if !field1RawOk {
    continue
}
field1, field1IsString := field1Raw.(string)
if !field1IsString {
    fmt.Fprintf(os.Stderr, "Error: field 'name' is not a string\n")
    continue
}
```

**Integer Type Validation:**
```go
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

**Nested Field with Type:**
```go
field1Raw, field1RawOk := getNestedField(data, []string{"user", "name"})
if !field1RawOk {
    continue
}
field1, field1IsString := field1Raw.(string)
if !field1IsString {
    fmt.Fprintf(os.Stderr, "Error: nested field 'name' is not a string\n")
    continue
}
```

### Testing

Created comprehensive test suite with 4 test cases:
- ✅ `test_schema_basic.pl` - Basic types (string + integer)
- ✅ `test_schema_all_types.pl` - All primitive types (string + float + integer + boolean)
- ✅ `test_schema_nested.pl` - Nested fields with type validation
- ✅ `test_schema_mixed.pl` - Mixed flat and nested with types

All tests pass with 100% success rate.

**Test runner:**
```bash
./run_schema_tests.sh
```

### Documentation

- Updated `GO_JSON_FEATURES.md` with Phase 4 section and examples
- Updated `README.md` to Go Target v0.4 with schema support
- Created `JSON_SCHEMA_DESIGN.md` with comprehensive design documentation
- Updated comparison table to show Go has schema validation

## Files Changed

- `src/unifyweaver/targets/go_target.pl` - Core implementation (184 lines added)
- `JSON_SCHEMA_DESIGN.md` - Design documentation (514 lines, new)
- `test_schema_basic.pl` - Basic types test (29 lines, new)
- `test_schema_all_types.pl` - All types test (32 lines, new)
- `test_schema_nested.pl` - Nested fields test (32 lines, new)
- `test_schema_mixed.pl` - Mixed fields test (31 lines, new)
- `run_schema_tests.sh` - Test runner (77 lines, new)
- `GO_JSON_FEATURES.md` - Updated with Phase 4 (93 lines added)
- `README.md` - Version bump v0.3 → v0.4 (3 lines changed)

**Total: 984 insertions(+), 11 deletions(-)**

## Test Plan

```bash
# Run schema validation tests
./run_schema_tests.sh
```

All 4 test cases pass successfully with correct type validation and error reporting.

## Key Features

- **Type Safety** - Compile-time type definitions with runtime validation
- **Clear Errors** - Descriptive error messages to stderr for type mismatches
- **Nested Support** - Type validation works with both flat and nested fields
- **Backward Compatible** - Schemas are optional, existing code unchanged
- **Zero Overhead** - No performance penalty for valid data
- **Flexible** - Fallback to dynamic typing with 'any' type

## Breaking Changes

None. This is a pure feature addition that's completely opt-in and backward compatible with all existing JSON I/O functionality.

## Next Steps

Phase 4 completes the type-safe JSON processing foundation. Future enhancements (Phase 5+) could include:
- Array type support with iteration
- Advanced validation rules (min/max, format patterns)
- Required vs optional fields
- Schema composition and inheritance
- Custom type validators

🤖 Generated with [Claude Code](https://claude.com/claude-code)
