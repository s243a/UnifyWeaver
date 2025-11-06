<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 2 C# Examples

This directory contains working examples of UnifyWeaver-generated C# code.

## test_link Example

**Location:** `test_link/`

Demonstrates a simple transitive closure query (grandparent-like relation).

### Prolog Source (conceptual):
```prolog
test_fact(alice, bob).
test_fact(bob, charlie).

test_link(X, Z) :- test_fact(X, Y), test_fact(Y, Z).
```

### Generated Files:

**`Program.cs`** - Entry point
- Builds the query plan
- Creates a QueryExecutor
- Executes the query
- Prints results

**`QueryRuntime.cs`** - Runtime library
- `PredicateId` - Identifies predicates (name/arity)
- `IRelationProvider` - Supplies base facts
- `PlanNode` hierarchy - IR nodes (RelationScan, Selection, Projection, Join, Union, etc.)
- `QueryExecutor` - Evaluates query plans with fixpoint iteration
- Semi-naive evaluation for recursive queries

**`TestLinkQueryModule.cs`** - Generated query
- Populates base facts (test_fact/2)
- Builds query plan using IR nodes
- Returns provider and plan for execution

### Running the Example:

```bash
cd test_link
dotnet new console -n TestLink
cp *.cs TestLink/
cd TestLink

# Fix namespace/using if needed
dotnet build
dotnet run
```

**Expected Output:**
```
alice,bob,bob,charlie
```

This represents the join result: `test_link(alice, charlie)` - Alice is linked to Charlie via Bob.

### Understanding the Generated Code:

The key line in `TestLinkQueryModule.cs`:
```csharp
new ProjectionNode(
    new JoinNode(
        new RelationScanNode(new PredicateId("test_fact", 2)),  // Left: test_fact(X, Y)
        new RelationScanNode(new PredicateId("test_fact", 2)),  // Right: test_fact(Y, Z)
        (left, right) => Equals(left[1], right[0]),             // Join: Y matches
        (left, right) => new object[]{ left[0], left[1], right[0], right[1] }  // (X, Y, Y, Z)
    ),
    tuple => new object[]{ tuple[0], tuple[3] }  // Project to (X, Z)
)
```

**Breaking this down:**
1. **RelationScanNode** (left) - Scan test_fact for (X, Y)
2. **RelationScanNode** (right) - Scan test_fact for (Y, Z)
3. **JoinNode predicate** - Join where left[1] (Y) equals right[0] (Y)
4. **JoinNode project** - Combine to (X, Y, Y, Z)
5. **ProjectionNode** - Extract only (X, Z)

**Execution trace:**
```
Scan test_fact: (alice, bob), (bob, charlie)

Join:
  (alice, bob) × (alice, bob) → Y mismatch (bob ≠ alice)
  (alice, bob) × (bob, charlie) → Y match! (bob = bob) → (alice, bob, bob, charlie)
  (bob, charlie) × (alice, bob) → Y mismatch (charlie ≠ alice)
  (bob, charlie) × (bob, charlie) → Y mismatch (charlie ≠ bob)

Project:
  (alice, bob, bob, charlie) → (alice, charlie)

Result: test_link(alice, charlie)
```

## More Examples (To Be Added)

Future examples will demonstrate:
- Recursive predicates (ancestor/2)
- Multiple clauses with Union
- Fixpoint iteration
- Database-backed providers
- ASP.NET integration

## References

- Chapter 2: C# Stream Target
- Chapter 3: Query Engine Deep Dive
- Chapter 4: Runtime Libraries and Deployment

## License

These examples are dual-licensed under MIT OR Apache-2.0.
