# Go Target: JSON I/O and Record Structures - Design Document

## Overview

Add native JSON input/output and nested record structure support to the Go target, enabling direct JSON processing without external tools.

## Goals

1. **JSON Input**: Parse JSON from stdin into Prolog-compatible records
2. **JSON Output**: Generate JSON output from Prolog predicates
3. **Nested Structures**: Support nested field access (e.g., `user.address.city`)
4. **Type Safety**: Leverage Go's type system where possible
5. **Flexibility**: Support both typed (struct-based) and dynamic (map-based) approaches

## Use Cases

### Use Case 1: JSON Transformation
```prolog
% Input: {"name": "Alice", "age": 25}
% Transform to: {"user": "Alice", "years": 25}
transformed(User, Years) :-
    person(Name, Age),
    User = Name,
    Years = Age.
```

### Use Case 2: Nested Field Extraction
```prolog
% Input: {"user": {"name": "Bob", "address": {"city": "NYC"}}}
% Extract city from nested structure
city(City) :-
    data(User),
    json_get(User, [address, city], City).
```

### Use Case 3: JSON Array Processing
```prolog
% Input: [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
% Extract names
user_name(Name) :-
    users(UserList),
    json_array_member(UserList, User),
    json_get(User, [name], Name).
```

## Design Options

### Option A: Schema-Based (Typed Structs)

**Pros:**
- Type safety at compile time
- Better performance
- Clear structure definition

**Cons:**
- Requires schema definition upfront
- Less flexible for varying JSON shapes

**Example:**
```prolog
% Define schema
:- json_schema(person, [
    field(name, string),
    field(age, int),
    field(address, [
        field(city, string),
        field(zip, string)
    ])
]).

% Use typed access
city(City) :- person(P), P.address.city = City.
```

**Generated Go:**
```go
type Person struct {
    Name    string  `json:"name"`
    Age     int     `json:"age"`
    Address Address `json:"address"`
}

type Address struct {
    City string `json:"city"`
    Zip  string `json:"zip"`
}
```

### Option B: Dynamic (Map-Based)

**Pros:**
- No schema required
- Handles varying JSON shapes
- Simpler implementation

**Cons:**
- Type assertions at runtime
- Less efficient
- Potential runtime errors

**Example:**
```prolog
% No schema needed
city(City) :-
    json_input(Data),
    json_get(Data, "address.city", City).
```

**Generated Go:**
```go
var data map[string]interface{}
json.Unmarshal(input, &data)
city := data["address"].(map[string]interface{})["city"].(string)
```

### Option C: Hybrid Approach (RECOMMENDED)

Use dynamic parsing with optional schema hints:
- Default to `map[string]interface{}` for flexibility
- Allow schema annotations for performance-critical paths
- Provide helper predicates for common patterns

## Proposed Syntax

### 1. JSON Input Mode

```prolog
% Option 1: Inline JSON mode
:- json_input(true).

user(Name, Age) :-
    json_record([name-Name, age-Age]).

% Option 2: With field path
city(City) :-
    json_get("address.city", City).

% Option 3: Nested access
city(City) :-
    json_field(address, Addr),
    json_get(Addr, city, City).
```

### 2. JSON Output Mode

```prolog
% Option 1: Inline JSON output
:- json_output(true).

result(Name, Age) :-
    user(Name, Age).
% Outputs: {"Name": "...", "Age": ...}

% Option 2: With mapping
:- json_output([
    field(user, name),
    field(years, age)
]).

result(Name, Age) :- user(Name, Age).
% Outputs: {"user": "...", "years": ...}
```

### 3. Nested Field Access

```prolog
% Dot notation (compile-time)
City :- Data.address.city

% Path notation (runtime)
json_get(Data, [address, city], City)

% String path (most flexible)
json_get(Data, "address.city", City)
```

## Implementation Plan

### Phase 1: JSON Input (Basic)
1. Add `json_input(true)` option
2. Parse JSON to `map[string]interface{}`
3. Support flat structures only
4. Generate Go code with `encoding/json`

### Phase 2: JSON Output (Basic)
1. Add `json_output(true)` option
2. Generate JSON from flat records
3. Use Go structs with json tags

### Phase 3: Nested Structures
1. Implement `json_get/3` predicate
2. Support dot notation for field access
3. Handle nested maps in Go

### Phase 4: Arrays
1. Support JSON arrays
2. Add `json_array_member/2` predicate
3. Generate array iteration code

### Phase 5: Schema Support (Optional)
1. Add `json_schema/2` directive
2. Generate Go structs from schema
3. Optimize with type assertions

## Go Code Generation Strategy

### Input Processing
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

        // Extract fields
        name := data["name"].(string)
        age := int(data["age"].(float64))

        // Process and output
        result := fmt.Sprintf("%s:%d", name, age)
        if !seen[result] {
            seen[result] = true
            fmt.Println(result)
        }
    }
}
```

### Output Generation
```go
type Result struct {
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
        result := Result{
            Name: parts[0],
            Age:  age,
        }

        jsonBytes, _ := json.Marshal(result)
        fmt.Println(string(jsonBytes))
    }
}
```

## Options and Configuration

```prolog
compile_predicate_to_go(user/2, [
    json_input(true),           % Parse JSON input
    json_output(true),          % Generate JSON output
    json_schema(user, [...]),   % Optional schema
    field_delimiter(json)       % Use JSON mode
], Code)
```

## Compatibility

- Maintain backward compatibility with colon/tab/comma delimiters
- JSON mode is opt-in via options
- Can mix JSON I/O with other features (match, constraints, etc.)

## Testing Strategy

1. **Basic JSON parsing**: Flat objects
2. **Nested structures**: Objects within objects
3. **Arrays**: JSON arrays and member extraction
4. **Mixed types**: Strings, numbers, booleans, null
5. **Error handling**: Invalid JSON, type mismatches
6. **Integration**: JSON I/O with match predicates and constraints

## Questions to Resolve

1. How to handle JSON arrays in Prolog? (Lists, or special predicate?)
2. How to represent null values?
3. Should we support JSON paths (JSONPath-like syntax)?
4. How to handle type conversions (string to int, etc.)?
5. Support for JSON streaming (large files)?

## Next Steps

1. Start with Phase 1: Basic JSON input with flat structures
2. Create test cases
3. Implement `json_input` option parsing
4. Generate Go code with `encoding/json`
5. Test and iterate

## References

- Go `encoding/json`: https://pkg.go.dev/encoding/json
- SWI-Prolog JSON: https://www.swi-prolog.org/pldoc/man?section=jsonjson
- JSONPath: https://goessner.net/articles/JsonPath/
