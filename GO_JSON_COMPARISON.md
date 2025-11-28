# Go JSON Approach - Comparison with Python and C# Targets

## Summary of Approaches

### Python Target (Dynamic, JSONL-based)
**Data Representation:**
- Python dicts: `Dict[str, Any]`
- JSONL format (one JSON per line)
- NUL-delimited JSON option

**Code Example:**
```python
import json

def read_jsonl(stream):
    for line in stream:
        if line.strip():
            yield json.loads(line)  # Parse to dict

def write_jsonl(records, stream):
    for record in records:
        stream.write(json.dumps(record) + '\n')  # Dict to JSON
```

**Pros:**
- ✅ Flexible - handles any JSON shape
- ✅ Simple - no schema required
- ✅ Streaming - line-by-line processing

**Cons:**
- ❌ Runtime type errors
- ❌ No compile-time validation
- ❌ Performance overhead from dynamic typing

### C# Target (Static, Tuple-based)
**Data Representation:**
- ValueTuples for fixed-arity records
- Arity 1: `string`
- Arity 2: `(string, string)`
- Arity 3: `(string, string, string)`

**Code Example:**
```csharp
// Tuple literal creation
var record = ("Alice", "25");  // (string, string)

// Field access
Console.WriteLine($"{record.Item1}:{record.Item2}");

// LINQ pipeline
var results = data
    .Select(x => (x.Name, x.Age))
    .Where(x => x.Age > 18);
```

**Pros:**
- ✅ Type-safe at compile time
- ✅ Clean syntax with ValueTuples
- ✅ Good performance

**Cons:**
- ❌ Fixed schema only
- ❌ Limited to supported arities (1-3)
- ❌ All fields are strings (no native JSON types)

## Proposed Go Approach (Hybrid)

### Phase 1: Dynamic (like Python)
Start with flexible dynamic typing using `map[string]interface{}`.

**Code Example:**
```go
import "encoding/json"

// Parse JSONL
scanner := bufio.NewScanner(os.Stdin)
for scanner.Scan() {
    var data map[string]interface{}
    if err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
        continue
    }

    // Extract fields with type assertions
    name := data["name"].(string)
    ageFloat := data["age"].(float64)  // JSON numbers are float64
    age := int(ageFloat)
}
```

**Pros:**
- ✅ Handles any JSON shape (like Python)
- ✅ Native Go types (string, float64, bool, etc.)
- ✅ No schema required
- ✅ Streaming support

**Cons:**
- ❌ Runtime type assertions needed
- ❌ Potential panics if types don't match

### Phase 2: Typed Structs (like C#)
Add optional struct generation for known schemas.

**Code Example:**
```go
type User struct {
    Name string `json:"name"`
    Age  int    `json:"age"`
}

// With schema
var user User
if err := json.Unmarshal(data, &user); err != nil {
    continue
}

// Type-safe field access
fmt.Printf("%s:%d", user.Name, user.Age)
```

**Pros:**
- ✅ Type-safe (like C#)
- ✅ Better performance
- ✅ JSON tags for field mapping
- ✅ Compile-time validation

**Cons:**
- ❌ Requires schema definition
- ❌ Less flexible

## Comparison Table

| Feature | Python | C# | Go (Phase 1) | Go (Phase 2) |
|---------|--------|-----|--------------|--------------|
| **Type System** | Dynamic | Static (tuples) | Dynamic (maps) | Static (structs) |
| **Schema Required** | No | No | No | Yes (optional) |
| **Field Access** | `dict["key"]` | `tuple.Item1` | `data["key"].(type)` | `struct.Field` |
| **Type Safety** | Runtime | Compile | Runtime | Compile |
| **Flexibility** | ✅ High | ❌ Low | ✅ High | ⚠️ Medium |
| **Performance** | ⚠️ Slower | ✅ Fast | ⚠️ Medium | ✅ Fast |
| **JSON Support** | ✅ Native | ❌ Manual | ✅ Native | ✅ Native |
| **Nested Data** | ✅ Yes | ❌ Flat only | ✅ Yes | ✅ Yes |
| **Arrays** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Error Handling** | Try/catch | N/A | Type assertion | Unmarshal error |

## Key Design Decisions

### 1. JSON Format: JSONL (like Python)
**Rationale:** Standard, streamable, widely supported
```
{"name": "Alice", "age": 25}
{"name": "Bob", "age": 30}
```

### 2. Dynamic Typing First (like Python)
**Rationale:** Maximum flexibility, handles varying shapes
```go
var data map[string]interface{}
```

### 3. Optional Schemas Later (like C#)
**Rationale:** Performance and type safety when schema is known
```go
type Record struct { ... }
```

### 4. Native JSON Types (better than C#)
**Rationale:** Preserve type information from JSON
- Strings: `string`
- Numbers: `float64` (can convert to `int`)
- Booleans: `bool`
- Null: `nil`
- Objects: `map[string]interface{}`
- Arrays: `[]interface{}`

### 5. Nested Access Helpers
**Rationale:** Neither Python nor C# provide this, but it's useful
```prolog
city(City) :- json_get([user, address, city], City).
```

```go
func getNestedField(data map[string]interface{}, path []string) (interface{}, bool) {
    // Helper for nested access
}
```

## What We Learn from Each

### From Python:
- ✅ JSONL is a good streaming format
- ✅ Dynamic maps work well
- ✅ Simple read/write helpers
- ⚠️ Need better error handling than Python

### From C#:
- ✅ Type-safe approaches are valuable
- ✅ Structured output (tuples) is clean
- ✅ Schema-based optimization matters
- ⚠️ All-strings limitation is too restrictive

### Our Hybrid Approach:
- ✅ **Best of both worlds**
- ✅ Start flexible (maps)
- ✅ Add optimization later (structs)
- ✅ Native JSON types (Go advantage)
- ✅ Nested access (our innovation)

## Implementation Strategy

### Immediate (Phase 1):
1. JSONL input parsing → maps
2. Type assertions for field extraction
3. Colon-delimited output (backward compatible)

### Short-term (Phase 2):
4. JSON output generation
5. Struct-based output (optional)

### Medium-term (Phase 3):
6. Nested field access helpers
7. Schema annotations (optional)

### Long-term (Phase 4):
8. Array support
9. Performance optimizations
10. Advanced type conversions

## Conclusion

The Go approach combines:
- **Python's flexibility** (dynamic maps, JSONL)
- **C#'s type safety** (optional structs)
- **Go's native JSON** (better types than C#'s all-strings)
- **Our innovation** (nested access, hybrid approach)

This gives us maximum flexibility initially while allowing optimization for performance-critical use cases later.
