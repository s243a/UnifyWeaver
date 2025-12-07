# Go JSON Implementation Plan

Based on analysis of Python target's JSONL approach.

## Phase 1: Basic JSON Input (JSONL)

### Goal
Parse JSON Lines (JSONL) input and extract flat fields.

### Example
```prolog
% Input (JSONL):
% {"name": "Alice", "age": 25}
% {"name": "Bob", "age": 30}

user(Name, Age) :- json_record([name-Name, age-Age]).

% Compile with:
compile_predicate_to_go(user/2, [json_input(true)], Code)
```

### Generated Go Code
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
            continue  // Skip invalid JSON
        }

        // Extract fields with type assertions
        name, nameOk := data["name"].(string)
        ageFloat, ageOk := data["age"].(float64)  // JSON numbers are float64
        if !nameOk || !ageOk {
            continue
        }
        age := int(ageFloat)

        // Output in configured format (colon-delimited by default)
        result := fmt.Sprintf("%s:%d", name, age)
        if !seen[result] {
            seen[result] = true
            fmt.Println(result)
        }
    }
}
```

### Implementation Tasks
1. Add `json_input(true)` option detection
2. Modify `compile_predicate_to_go_normal` to detect JSON mode
3. Generate JSON unmarshaling code
4. Handle type assertions (string, float64→int, bool)
5. Map JSON fields to Prolog variables

## Phase 2: JSON Output

### Example
```prolog
% Transform colon-delimited to JSON
user_json(Name, Age) :-
    user_data(Name, Age).

% Compile with:
compile_predicate_to_go(user_json/2, [json_output(true)], Code)
```

### Generated Go Code
```go
type UserRecord struct {
    Name string `json:"name"`
    Age  int    `json:"age"`
}

func main() {
    scanner := bufio.NewScanner(os.Stdin)

    for scanner.Scan() {
        parts := strings.Split(scanner.Text(), ":")
        if len(parts) != 2 {
            continue
        }

        age, _ := strconv.Atoi(parts[1])
        record := UserRecord{
            Name: parts[0],
            Age:  age,
        }

        jsonBytes, _ := json.Marshal(record)
        fmt.Println(string(jsonBytes))
    }
}
```

### Implementation Tasks
1. Add `json_output(true)` option
2. Generate struct definition from predicate signature
3. Generate JSON marshaling code
4. Handle field name mapping (predicate args → JSON keys)

## Phase 3: Nested Field Access

### Example
```prolog
% Input: {"user": {"name": "Alice", "address": {"city": "NYC"}}}

city(City) :-
    json_get([user, address, city], City).
```

### Generated Go Code
```go
// Helper function
func getNestedField(data map[string]interface{}, path []string) (interface{}, bool) {
    current := data
    for i, key := range path {
        if i == len(path)-1 {
            val, ok := current[key]
            return val, ok
        }
        next, ok := current[key].(map[string]interface{})
        if !ok {
            return nil, false
        }
        current = next
    }
    return nil, false
}

// In main:
city, ok := getNestedField(data, []string{"user", "address", "city"})
if !ok {
    continue
}
cityStr := city.(string)
```

### Implementation Tasks
1. Implement `json_get/2` predicate recognition
2. Generate nested field access helper
3. Handle type assertions at each level
4. Support both dot notation and path lists

## Phase 4: JSON Arrays (Future)

### Example
```prolog
% Input: {"users": [{"name": "Alice"}, {"name": "Bob"}]}

user_name(Name) :-
    json_get([users], UserList),
    json_array_member(UserList, User),
    json_get(User, [name], Name).
```

## Implementation Strategy

### Step 1: Modify go_target.pl (Lines to Add)

1. **Option Detection** (~line 90)
```prolog
% Check for JSON mode
(   member(json_input(true), Options)
->  JsonInput = true
;   JsonInput = false
),
(   member(json_output(true), Options)
->  JsonOutput = true
;   JsonOutput = false
),
```

2. **JSON Input Mode** (~line 400)
```prolog
%% compile_json_input_mode(+Head, +Body, +Options, -GoCode)
compile_json_input_mode(Head, Body, Options, GoCode) :-
    Head =.. [_PredName|Args],
    length(Args, Arity),

    % Extract field names from body
    extract_json_fields(Body, FieldMappings),

    % Generate field extraction code
    generate_json_field_extractions(FieldMappings, Args, ExtractCode),

    % Generate output format
    option(field_delimiter(Delim), Options, colon),
    map_field_delimiter(Delim, DelimChar),
    generate_output_expr(Args, DelimChar, OutputExpr),

    % Build main function
    format(string(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tvar data map[string]interface{}
\t\tif err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
\t\t\tcontinue
\t\t}
\t\t
~s
\t\t
\t\tresult := ~s
\t\tif !seen[result] {
\t\t\tseen[result] = true
\t\t\tfmt.Println(result)
\t\t}
\t}
', [ExtractCode, OutputExpr]).
```

3. **JSON Field Extraction**
```prolog
%% extract_json_fields(+Body, -FieldMappings)
extract_json_fields(json_record(Fields), FieldMappings) :-
    extract_field_mappings(Fields, FieldMappings).

extract_field_mappings([], []).
extract_field_mappings([Field-Var|Rest], [Field-Var|Mappings]) :-
    extract_field_mappings(Rest, Mappings).
```

### Step 2: Test Cases

Create `test_json_input.pl`:
```prolog
:- use_module('src/unifyweaver/targets/go_target').

% Test 1: Simple JSON fields
test_simple :-
    compile_predicate_to_go(user/2, [json_input(true)], Code),
    write_go_program(Code, 'user_json.go'),
    format('Generated user_json.go~n').

% Test 2: With field names
user(Name, Age) :- json_record([name-Name, age-Age]).
```

### Step 3: Incremental Development

1. **Week 1**: JSON input (flat structures)
   - Add option parsing
   - Generate unmarshaling code
   - Test with simple examples

2. **Week 2**: JSON output
   - Generate struct definitions
   - Add marshaling code
   - Test round-trip (JSON→JSON)

3. **Week 3**: Nested structures
   - Implement json_get/2
   - Add nested access helpers
   - Test with real-world data

4. **Week 4**: Arrays and advanced features
   - Array iteration
   - Schema support (optional)
   - Performance optimization

## Success Criteria

- [ ] Can parse JSONL input
- [ ] Can extract flat JSON fields
- [ ] Can generate JSON output
- [ ] Can access nested fields (dot notation)
- [ ] Can handle arrays
- [ ] Backward compatible with existing delimiter-based mode
- [ ] Comprehensive test coverage
- [ ] Documentation with examples

## Next Immediate Steps

1. Create `test_json_input.pl` with simple test case
2. Add JSON option detection to `compile_predicate_to_go_normal`
3. Implement `compile_json_input_mode/4`
4. Generate and test first JSON→colon transformation
5. Iterate and refine

