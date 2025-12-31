# C# Target: Handling High-Arity Prolog Predicates & LINQ Limitations

## Status
- **Phase 1 (Extensions):** Implemented
- **Phase 2 (Provider):** Proposed / Conceptual

## Problem Statement
The UnifyWeaver C# target maps Prolog predicates to a fixed `PtEntity` class. Prolog predicates often have high arity (many arguments) or dynamic fields that do not map 1-to-1 with the rigid properties of `PtEntity` (Title, About, etc.).

These extra fields are stored in a `BsonDocument Raw` property.
**Current Limitation:** Accessing these `Raw` fields inside standard LINQ queries (e.g., `.Where(x => x.Raw["age"].AsInt32 > 21)`) is difficult because:
1.  The standard LiteDB LINQ provider may not translate dictionary/BsonDocument lookups into SQL-like queries efficiently.
2.  The syntax is verbose and error-prone.
3.  There is no type safety.

## Phase 1: Developer Ergonomics (Implemented)

### Solution
We introduced `PtEntityExtensions` to provide a clean, typed API for accessing raw data.

**Key Features:**
- Typed accessors: `GetRawString`, `GetRawInt`, `GetRawBool`, etc.
- Safe retrieval: Returns `default(T)` or `null` if keys are missing or types mismatch.
- `JsonNode` support: Allows complex dynamic manipulation if needed.

### Example Usage
```csharp
// Before
var age = entity.Raw.TryGetValue("age", out var val) ? val.AsInt32 : (int?)null;

// After
var age = entity.GetRawInt("age");
```

## Phase 2: LiteDB LINQ Provider Integration (Proposed)

### Goal
Ensure that calls to `PtEntityExtensions` are translated into efficient server-side LiteDB `BsonExpression` queries, rather than fetching all data and filtering in memory (client-side).

### Conceptual Approach
Extend the LiteDB `ExpressionVisitor` or wrap the `ILiteQueryable`.

1.  **Custom Query Visitor:**
    - Intercept calls to methods like `PtEntityExtensions.GetRawInt(x, "age")`.
    - Translate them to LiteDB BsonExpression: `$.Raw.age`.

2.  **Implementation Strategy:**
    - This likely requires a custom `IQueryProvider` wrapper around LiteDB's native provider.
    - We must ensure that `x.GetRawInt("age") > 10` becomes valid LiteDB query syntax (e.g., `Raw.age > 10`).

### Testing Strategy (Phase 2)
- **Performance Benchmarks:** Compare query times of In-Memory LINQ vs. Provider-Optimized LINQ on datasets > 10k records.
- **Translation Verification:** Inspect generated BsonExpressions to ensure `$.Raw` paths are generated correctly.
