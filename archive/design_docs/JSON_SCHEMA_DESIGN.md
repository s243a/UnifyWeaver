# JSON Schema Support Design

## Overview

Phase 4 of the Go target JSON I/O features adds **schema support** for type-safe JSON processing. This enables compile-time type definitions, runtime validation, and better error messages.

## Goals

1. **Type Safety** - Define expected types for JSON fields at compile time
2. **Validation** - Validate JSON data against schemas at runtime
3. **Better Error Messages** - Clear errors when data doesn't match schema
4. **Performance** - Type-specific extraction eliminates runtime type assertions
5. **Optional** - Schemas are opt-in; existing code without schemas continues to work

## Schema Syntax

### Option 1: Directive-Based (RECOMMENDED)

Define schemas using Prolog directives before predicate definitions:

```prolog
:- json_schema(person, [
    field(name, string),
    field(age, integer),
    field(active, boolean),
    field(salary, float)
]).

% Use the schema in predicates
user(Name, Age) :- json_record([name-Name, age-Age]).

% Compile with schema
compile_predicate_to_go(user/2, [
    json_input(true),
    json_schema(person)
], Code).
```

### Option 2: Inline in Options

Pass schema definition directly in compilation options:

```prolog
compile_predicate_to_go(user/2, [
    json_input(true),
    json_schema([
        field(name, string),
        field(age, integer)
    ])
], Code).
```

**Decision: Use Option 1** (directive-based) for better separation of concerns and reusability.

## Supported Types

### Primitive Types

| Schema Type | Go Type | JSON Type | Example |
|-------------|---------|-----------|---------|
| `string` | `string` | string | `"Alice"` |
| `integer` | `int` | number | `25` |
| `float` | `float64` | number | `3.14` |
| `boolean` | `bool` | boolean | `true` |

### Special Types (Future)

| Schema Type | Go Type | JSON Type | Notes |
|-------------|---------|-----------|-------|
| `number` | `float64` | number | Auto-convert int/float |
| `any` | `interface{}` | any | Current behavior |
| `array(T)` | `[]T` | array | Phase 5 |
| `object(Schema)` | struct | object | Phase 6 |

## Implementation Plan

### 1. Schema Storage

Store schemas as Prolog facts:

```prolog
:- dynamic json_schema_def/2.

% Store schema definition
json_schema(Name, Fields) :-
    assertz(json_schema_def(Name, Fields)).
```

### 2. Schema Lookup

Retrieve schema during compilation:

```prolog
get_json_schema(SchemaName, Fields) :-
    json_schema_def(SchemaName, Fields).
```

### 3. Field Type Mapping

Map field names to types from schema:

```prolog
% Extract type for a specific field
get_field_type(SchemaName, FieldName, Type) :-
    json_schema_def(SchemaName, Fields),
    member(field(FieldName, Type), Fields).
```

### 4. Type-Safe Code Generation

#### Current (Untyped) Extraction:
```go
field1, field1Ok := data["name"]
if !field1Ok {
    continue
}
```

#### New (Typed) Extraction:
```go
// String field
field1Raw, field1Ok := data["name"]
if !field1Ok {
    continue
}
field1, field1IsString := field1Raw.(string)
if !field1IsString {
    fmt.Fprintf(os.Stderr, "Error: field 'name' is not a string\n")
    continue
}

// Integer field
field2Raw, field2Ok := data["age"]
if !field2Ok {
    continue
}
// JSON numbers are float64, need conversion
field2Float, field2IsNum := field2Raw.(float64)
if !field2IsNum {
    fmt.Fprintf(os.Stderr, "Error: field 'age' is not a number\n")
    continue
}
field2 := int(field2Float)
```

### 5. Integration Points

#### Modify `compile_json_input_mode/4`:

```prolog
compile_json_input_mode(Pred, Arity, Options, GoCode) :-
    % Get schema if specified
    (   option(json_schema(SchemaName), Options)
    ->  get_json_schema(SchemaName, Schema),
        format('  Using schema: ~w~n', [SchemaName])
    ;   Schema = none
    ),

    % Extract field mappings
    extract_json_field_mappings(SingleBody, FieldMappings),

    % Generate typed or untyped extraction
    (   Schema \= none
    ->  compile_json_to_go_typed(HeadArgs, FieldMappings, Schema, FieldDelim, Unique, ScriptBody)
    ;   compile_json_to_go(HeadArgs, FieldMappings, FieldDelim, Unique, ScriptBody)
    ).
```

#### New Predicate: `compile_json_to_go_typed/6`

```prolog
compile_json_to_go_typed(HeadArgs, FieldMappings, Schema, FieldDelim, Unique, GoCode) :-
    map_field_delimiter(FieldDelim, DelimChar),

    % Generate typed field extractions
    generate_typed_field_extractions(FieldMappings, Schema, HeadArgs, ExtractCode),

    % Generate output expression (same as untyped)
    generate_json_output_expr(HeadArgs, DelimChar, OutputExpr),

    % Build main loop
    (   Unique = true ->
        SeenDecl = '\tseen := make(map[string]bool)\n\t',
        UniqueCheck = '\t\tif !seen[result] {\n\t\t\tseen[result] = true\n\t\t\tfmt.Println(result)\n\t\t}\n'
    ;   SeenDecl = '\t',
        UniqueCheck = '\t\tfmt.Println(result)\n'
    ),

    format(string(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
~w
\tfor scanner.Scan() {
\t\tvar data map[string]interface{}
\t\tif err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
\t\t\tcontinue
\t\t}
\t\t
~s
\t\t
\t\tresult := ~s
~s\t}
', [SeenDecl, ExtractCode, OutputExpr, UniqueCheck]).
```

#### New Predicate: `generate_typed_field_extractions/4`

```prolog
generate_typed_field_extractions(FieldMappings, Schema, HeadArgs, ExtractCode) :-
    Schema = json_schema(SchemaName, _),
    findall(ExtractLine,
        (   nth1(Pos, FieldMappings, Mapping),
            format(atom(VarName), 'field~w', [Pos]),
            % Dispatch based on mapping type and get field type from schema
            (   Mapping = Field-_Var
            ->  get_field_type(SchemaName, Field, Type),
                generate_typed_flat_field_extraction(Field, VarName, Type, ExtractLine)
            ;   Mapping = nested(Path, _Var)
            ->  % For nested, get type from last element of path
                last(Path, LastField),
                get_field_type(SchemaName, LastField, Type),
                generate_typed_nested_field_extraction(Path, VarName, Type, ExtractLine)
            )
        ),
        ExtractLines),
    atomic_list_concat(ExtractLines, '\n', ExtractCode).
```

#### New Predicate: `generate_typed_flat_field_extraction/4`

```prolog
generate_typed_flat_field_extraction(FieldName, VarName, Type, ExtractCode) :-
    atom_string(FieldName, FieldStr),
    format(atom(RawVar), '~wRaw', [VarName]),

    % Generate extraction based on type
    (   Type = string ->
        format(atom(ExtractCode), '\t\t~w, ~wOk := data["~s"]\n\t\tif !~wOk {\n\t\t\tcontinue\n\t\t}\n\t\t~w, ~wIsString := ~w.(string)\n\t\tif !~wIsString {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a string\\n")\n\t\t\tcontinue\n\t\t}',
            [RawVar, RawVar, FieldStr, RawVar, VarName, VarName, RawVar, VarName, FieldStr])
    ;   Type = integer ->
        format(atom(ExtractCode), '\t\t~wRaw, ~wOk := data["~s"]\n\t\tif !~wOk {\n\t\t\tcontinue\n\t\t}\n\t\t~wFloat, ~wIsNum := ~wRaw.(float64)\n\t\tif !~wIsNum {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a number\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w := int(~wFloat)',
            [VarName, VarName, FieldStr, VarName, VarName, VarName, VarName, FieldStr, VarName, VarName])
    ;   Type = float ->
        format(atom(ExtractCode), '\t\t~wRaw, ~wOk := data["~s"]\n\t\tif !~wOk {\n\t\t\tcontinue\n\t\t}\n\t\t~w, ~wIsNum := ~wRaw.(float64)\n\t\tif !~wIsNum {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a number\\n")\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, FieldStr, VarName, VarName, VarName, VarName, FieldStr])
    ;   Type = boolean ->
        format(atom(ExtractCode), '\t\t~wRaw, ~wOk := data["~s"]\n\t\tif !~wOk {\n\t\t\tcontinue\n\t\t}\n\t\t~w, ~wIsBool := ~wRaw.(bool)\n\t\tif !~wIsBool {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a boolean\\n")\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, FieldStr, VarName, VarName, VarName, VarName, FieldStr])
    ;   % Fallback to untyped (for 'any' type)
        generate_flat_field_extraction(FieldStr, VarName, ExtractCode)
    ).
```

## Examples

### Example 1: Basic Typed Input

**Prolog:**
```prolog
:- json_schema(user, [
    field(name, string),
    field(age, integer)
]).

user(Name, Age) :- json_record([name-Name, age-Age]).

:- compile_predicate_to_go(user/2, [
    json_input(true),
    json_schema(user)
], Code),
   write_go_program(Code, 'typed_user.go').
```

**Generated Go:**
```go
package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "os"
)

func main() {
    scanner := bufio.NewScanner(os.Stdin)
    seen := make(map[string]bool)

    for scanner.Scan() {
        var data map[string]interface{}
        if err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
            continue
        }

        field1Raw, field1Ok := data["name"]
        if !field1Ok {
            continue
        }
        field1, field1IsString := field1Raw.(string)
        if !field1IsString {
            fmt.Fprintf(os.Stderr, "Error: field 'name' is not a string\n")
            continue
        }

        field2Raw, field2Ok := data["age"]
        if !field2Ok {
            continue
        }
        field2Float, field2IsNum := field2Raw.(float64)
        if !field2IsNum {
            fmt.Fprintf(os.Stderr, "Error: field 'age' is not a number\n")
            continue
        }
        field2 := int(field2Float)

        result := fmt.Sprintf("%v:%v", field1, field2)
        if !seen[result] {
            seen[result] = true
            fmt.Println(result)
        }
    }
}
```

**Test Input:**
```json
{"name": "Alice", "age": 25}
{"name": "Bob", "age": "thirty"}
{"name": "Charlie", "age": 35}
```

**Output:**
```
Alice:25
Error: field 'age' is not a number
Charlie:35
```

### Example 2: Mixed Types with Validation

**Prolog:**
```prolog
:- json_schema(employee, [
    field(name, string),
    field(salary, float),
    field(active, boolean)
]).

employee(Name, Salary, Active) :-
    json_record([name-Name, salary-Salary, active-Active]).
```

**Input:**
```json
{"name": "Alice", "salary": 75000.50, "active": true}
{"name": "Bob", "salary": "high", "active": true}
{"name": "Charlie", "salary": 65000, "active": "yes"}
```

**Output:**
```
Alice:75000.5:true
Error: field 'salary' is not a number
Error: field 'active' is not a boolean
```

### Example 3: Nested Fields with Schema

**Prolog:**
```prolog
:- json_schema(profile, [
    field(name, string),
    field(city, string),
    field(age, integer)
]).

user_info(Name, City) :-
    json_get([user, name], Name),
    json_get([user, address, city], City).

:- compile_predicate_to_go(user_info/2, [
    json_input(true),
    json_schema(profile)
], Code).
```

## Testing Strategy

### Test Cases

1. **Valid Data** - All fields match schema types
2. **Type Mismatch** - String provided for integer field
3. **Missing Fields** - Required field not present
4. **Extra Fields** - JSON has fields not in schema (should ignore)
5. **Mixed Valid/Invalid** - Some records pass, some fail
6. **All Primitive Types** - Test string, integer, float, boolean
7. **Nested Fields with Types** - Type validation in nested structures

### Test Files

- `test_schema_basic.pl` - Basic typed field extraction
- `test_schema_validation.pl` - Type validation errors
- `test_schema_nested.pl` - Typed nested fields
- `run_schema_tests.sh` - Automated test runner

## Backward Compatibility

**Zero Breaking Changes:**

1. Schemas are optional - existing code without schemas works unchanged
2. Untyped mode (current behavior) is preserved when no schema specified
3. New predicates (`compile_json_to_go_typed`, etc.) don't affect existing ones
4. All existing tests continue to pass

## Performance Impact

**Positive:**
- Type-specific extraction eliminates runtime type switches
- Compile-time type knowledge enables optimizations
- Validation happens once during extraction (not later)

**Negative:**
- Slightly more generated code for type assertions
- Error messages add minimal overhead (stderr writes)

## Future Enhancements (Phase 5+)

### Array Types
```prolog
:- json_schema(team, [
    field(name, string),
    field(members, array(string))
]).
```

### Nested Object Types
```prolog
:- json_schema(person, [
    field(name, string),
    field(address, object([
        field(city, string),
        field(zip, integer)
    ]))
]).
```

### Optional Fields
```prolog
:- json_schema(user, [
    field(name, string, required),
    field(email, string, optional)
]).
```

### Custom Validators
```prolog
:- json_schema(user, [
    field(age, integer, [min(0), max(150)]),
    field(email, string, [format(email)])
]).
```

## Implementation Checklist

- [ ] 1. Define `json_schema/2` directive handling
- [ ] 2. Implement schema storage (`json_schema_def/2`)
- [ ] 3. Add schema lookup (`get_json_schema/2`, `get_field_type/3`)
- [ ] 4. Modify `compile_json_input_mode/4` to detect schema option
- [ ] 5. Implement `compile_json_to_go_typed/6`
- [ ] 6. Implement `generate_typed_field_extractions/4`
- [ ] 7. Implement `generate_typed_flat_field_extraction/4`
- [ ] 8. Implement `generate_typed_nested_field_extraction/4` (nested support)
- [ ] 9. Create comprehensive test suite
- [ ] 10. Update documentation (GO_JSON_FEATURES.md, README.md)
- [ ] 11. Create examples demonstrating schema usage
- [ ] 12. Verify all existing tests still pass (backward compatibility)

## Questions & Decisions

### Q1: How to handle type coercion?

**Decision:** Strict typing - no automatic coercion except JSON number (float64) → int

Rationale: Clear errors are better than silent coercion mistakes

### Q2: Should schemas be required or optional?

**Decision:** Optional - schemas are opt-in

Rationale: Backward compatibility, gradual adoption

### Q3: Where to store schema definitions?

**Decision:** Prolog dynamic facts (`json_schema_def/2`)

Rationale: Simple, accessible during compilation, standard Prolog mechanism

### Q4: Support schema inheritance/composition?

**Decision:** Not in Phase 4 - consider for Phase 5+

Rationale: Keep initial implementation simple

### Q5: Error handling strategy?

**Decision:** Print to stderr and skip invalid records

Rationale: Matches existing pipeline behavior, allows filtering

## References

- Current JSON implementation: `go_target.pl:1349-1700`
- JSON features doc: `GO_JSON_FEATURES.md`
- Nested fields design: `NESTED_FIELDS_DESIGN.md`
- Test examples: `test_json_input.pl`, `test_json_nested.pl`
