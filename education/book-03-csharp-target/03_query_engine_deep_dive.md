<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 3: C# Query Runtime - Deep Dive

## Introduction

The C# Query Runtime uses an **Intermediate Representation (IR)** approach instead of direct source-to-source translation. This chapter explains how the query engine works, breaking down complex concepts into understandable pieces.

## Why Query Runtime vs Stream Target?

**Stream Target (Chapter 2):**
```
Prolog → C# LINQ directly
```
- Simple, fast, but **no recursion**

**Query Runtime (This Chapter):**
```
Prolog → IR (Query Plan) → Runtime Engine executes
```
- More complex, but **supports recursion**

## The Big Picture

Think of the Query Runtime like a database query engine:

```
┌─────────────────────────────────────────────────┐
│ 1. Prolog Source                                │
│    ancestor(X, Y) :- parent(X, Y).              │
│    ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).│
└────────────────┬────────────────────────────────┘
                 │
                 │ Compilation
                 ▼
┌─────────────────────────────────────────────────┐
│ 2. IR (Intermediate Representation)             │
│    QueryPlan {                                  │
│      Relations: [parent/2, ancestor/2]          │
│      Clauses: [                                 │
│        { body: [RelationRef("parent", ...)] }   │
│        { body: [RelationRef("parent", ...),     │
│                 RelationRef("ancestor", ...)] } │
│      ]                                          │
│    }                                            │
└────────────────┬────────────────────────────────┘
                 │
                 │ Execution
                 ▼
┌─────────────────────────────────────────────────┐
│ 3. Runtime Engine                               │
│    - Loads facts into memory                    │
│    - Evaluates rules iteratively                │
│    - Computes fixpoint (semi-naive)             │
│    - Returns final results                      │
└─────────────────────────────────────────────────┘
```

## Core Components (Demystified)

The query engine has several building blocks. Let's understand each one:

### 1. RelationRef - "Look up data from a table"

**What it means:** Reference to a collection of tuples (facts or computed results)

**Think of it like:**
```sql
SELECT * FROM parent  -- In SQL
```

**In IR:**
```csharp
new RelationRef("parent", 2)  // "Give me all parent/2 tuples"
```

**Example:**
```prolog
% Prolog: Use parent facts
grandparent(GP, GC) :- parent(GP, P), parent(P, GC).
                       └─────┬─────┘  └────┬────┘
                         RelationRef    RelationRef
                         ("parent")     ("parent")
```

**Plain English:** "Fetch tuples from the 'parent' relation"

### 2. Selection - "Filter rows by condition"

**What it means:** Keep only tuples that satisfy a condition

**Think of it like:**
```sql
SELECT * FROM age WHERE age >= 18  -- In SQL
```

**In IR:**
```csharp
new Selection(
    new RelationRef("age", 2),
    tuple => tuple.Item2 >= 18  // Predicate: Age >= 18
)
```

**Example:**
```prolog
% Prolog: Filter adults
adult(Person) :- age(Person, Age), Age >= 18.
                                    └────┬────┘
                                      Selection
                                      (filter)
```

**Plain English:** "From all age tuples, keep only those where the second element (Age) is >= 18"

### 3. Projection - "Reshape tuples"

**What it means:** Transform tuple shape (select specific fields, reorder, rename)

**Think of it like:**
```sql
SELECT name, salary FROM employees  -- Pick specific columns
```

**In IR:**
```csharp
new Projection(
    new RelationRef("employees", 3),
    tuple => (tuple.Item1, tuple.Item3)  // Keep fields 1 and 3
)
```

**Example:**
```prolog
% Prolog: Extract just the person
adult(Person) :- age(Person, Age), Age >= 18.
      └──┬──┘
     Projection
     (extract Person field)
```

**Plain English:** "From each tuple, extract only the first field (Person)"

### 4. Join - "Combine related tuples"

**What it means:** Match tuples from two sources based on common values

**Think of it like:**
```sql
SELECT * FROM parent p1
JOIN parent p2 ON p1.child = p2.parent  -- In SQL
```

**In IR:**
```csharp
new Join(
    new RelationRef("parent", 2),  // First source
    new RelationRef("parent", 2),  // Second source
    outer => outer.Item2,          // Key from first: child
    inner => inner.Item1,          // Key from second: parent
    (outer, inner) => (outer.Item1, inner.Item2)  // Combine: (GP, GC)
)
```

**Example:**
```prolog
% Prolog: Grandparent is parent of parent
grandparent(GP, GC) :- parent(GP, P), parent(P, GC).
                                └──┬──┘
                                  Join
                                  (P matches)
```

**Plain English:** "Match parent tuples where the child of one equals the parent of another"

### 5. Union - "Combine results from multiple clauses"

**What it means:** Merge outputs from different rules/facts

**Think of it like:**
```sql
SELECT * FROM direct_siblings
UNION
SELECT * FROM computed_siblings  -- In SQL
```

**In IR:**
```csharp
new Union(
    clause1,  // First clause results
    clause2   // Second clause results
)
```

**Example:**
```prolog
% Prolog: Multiple ways to be siblings
sibling(alice, bob).              % Clause 1: Direct fact
sibling(X, Y) :- parent(P, X),    % Clause 2: Computed
                 parent(P, Y),
                 X \= Y.
└────────────┬────────────┘
          Union
```

**Plain English:** "Combine results from both the direct fact and the computed rule"

### 6. Distinct - "Remove duplicates"

**What it means:** Ensure each tuple appears only once in results

**Think of it like:**
```sql
SELECT DISTINCT name FROM people  -- In SQL
```

**In IR:**
```csharp
new Distinct(source)
```

**Plain English:** "If the same tuple appears multiple times, keep only one copy"

## Breaking Down Nested Expressions

Let's take a complex query and break it down step-by-step:

### Example: Ancestor Query (Recursive)

**Prolog Source:**
```prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

### Step 1: Identify Clauses

**Clause 1 (Base Case):**
```prolog
ancestor(X, Y) :- parent(X, Y).
```
"An ancestor is a direct parent"

**Clause 2 (Recursive Case):**
```prolog
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```
"An ancestor is a parent of an ancestor"

### Step 2: Translate Each Clause to IR

**Clause 1 IR (simple):**
```csharp
// Just reference parent facts
new RelationRef("parent", 2)
```

**Clause 2 IR (more complex):**

Let's build this piece by piece:

```csharp
// Start with parent(X, Y)
var parentXY = new RelationRef("parent", 2);

// Then get ancestor(Y, Z)
var ancestorYZ = new RelationRef("ancestor", 2);

// Join them where Y matches
var joined = new Join(
    parentXY,              // Source 1: parent(X, Y)
    ancestorYZ,            // Source 2: ancestor(Y, Z)
    outer => outer.Item2,  // Y from parent (second field)
    inner => inner.Item1,  // Y from ancestor (first field)
    (outer, inner) => (outer.Item1, inner.Item2)  // Result: (X, Z)
);
```

**What's happening:**

1. **`parentXY`** - Get all parent relationships: (alice, bob), (bob, charlie), ...
2. **`ancestorYZ`** - Get all known ancestors: (bob, charlie), (alice, charlie), ...
3. **`Join`** - Match where:
   - `outer.Item2` (child from parent) = `inner.Item1` (ancestor from ancestor)
   - This finds chains: parent(alice, bob) + ancestor(bob, charlie) → ancestor(alice, charlie)
4. **Result selector** - Extract (X, Z) from the match

### Step 3: Combine with Union

```csharp
var clause1 = new RelationRef("parent", 2);
var clause2 = /* Join from above */;

var ancestorQuery = new Union(clause1, clause2);
```

**Plain English:** "Ancestors are either direct parents (clause1) OR parents of ancestors (clause2)"

### Step 4: Full IR (Formatted for Readability)

```csharp
var ancestorPlan = new QueryPlan {
    Name = "ancestor",
    Arity = 2,
    Clauses = new[] {
        // Clause 1: Base case
        new Clause {
            Body = new RelationRef("parent", 2)
        },

        // Clause 2: Recursive case
        new Clause {
            Body = new Join(
                // parent(X, Y)
                new RelationRef("parent", 2),

                // ancestor(Y, Z)
                new RelationRef("ancestor", 2),

                // Join key: Y
                outer => outer.Item2,  // Y from parent
                inner => inner.Item1,  // Y from ancestor

                // Result: (X, Z)
                (outer, inner) => (outer.Item1, inner.Item2)
            )
        }
    }
};
```

## How Recursion Works (Fixpoint Iteration)

For recursive predicates, the engine uses **semi-naive evaluation**:

### Iteration 0: Start with base facts

```
parent facts: (alice, bob), (bob, charlie), (charlie, dave)
ancestor: {} (empty)
```

### Iteration 1: Apply rules once

**Clause 1:** ancestor(X, Y) :- parent(X, Y)
```
ancestor = {(alice, bob), (bob, charlie), (charlie, dave)}
```

**Clause 2:** ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
```
Join parent with ancestor:
- parent(alice, bob) + ancestor(bob, charlie) → ancestor(alice, charlie)
- parent(bob, charlie) + ancestor(charlie, dave) → ancestor(bob, dave)

ancestor = {(alice, bob), (bob, charlie), (charlie, dave),
            (alice, charlie), (bob, dave)}
```

### Iteration 2: Apply rules again with new facts

**Clause 2 again:**
```
Join parent with NEW ancestors:
- parent(alice, bob) + ancestor(bob, dave) → ancestor(alice, dave)

ancestor = {(alice, bob), (bob, charlie), (charlie, dave),
            (alice, charlie), (bob, dave), (alice, dave)}
```

### Iteration 3: No new facts → Fixpoint reached!

The engine stops when no new tuples are produced.

**Final result:**
```
ancestor(alice, bob)
ancestor(alice, charlie)
ancestor(alice, dave)
ancestor(bob, charlie)
ancestor(bob, dave)
ancestor(charlie, dave)
```

## Semi-Naive Optimization

**Naive approach:** Re-join ALL ancestors every iteration (slow!)

**Semi-naive approach:** Only join with NEW ancestors from last iteration (fast!)

### Comparison:

**Iteration 2 - Naive:**
```csharp
// Join parent with ALL ancestors (wasteful!)
var result = parentFacts.Join(allAncestors, ...);
```

**Iteration 2 - Semi-Naive:**
```csharp
// Join parent with ONLY new ancestors from iteration 1
var result = parentFacts.Join(newAncestorsFromLastIteration, ...);
```

**Performance difference:**

| Depth | Naive Joins | Semi-Naive Joins |
|-------|-------------|------------------|
| 1 | 3 | 3 |
| 2 | 15 | 6 |
| 3 | 45 | 3 |
| 4 | 135 | 0 (done) |

**Much faster for deep recursion!**

## Mutual Recursion Support (v0.1)

Starting with UnifyWeaver v0.1 the query runtime can evaluate **strongly connected groups** of predicates (e.g., `is_even/1` and `is_odd/1`). The planner emits a `MutualFixpointNode` containing one `MutualMember` entry per predicate in the SCC, each with its base and recursive plans.

### Key Plan Nodes

- `MutualFixpointNode(members, head)` – orchestrates the group fixpoint.
- `MutualMember(predicate, basePlan, recursivePlans)` – describes how to seed and extend each predicate.
- `CrossRefNode(predicate, RecursiveRefKind)` – replaces recursive calls that target **other** predicates in the group.

### Execution Model

The runtime keeps an evaluation context with per-predicate totals and deltas:

```csharp
var context = new EvaluationContext
{
    Totals = { [isEvenId] = totalEven, [isOddId] = totalOdd },
    Deltas = { [isEvenId] = deltaEven, [isOddId] = deltaOdd }
};

while (context.Deltas.Values.Any(delta => delta.Count > 0))
{
    foreach (var member in node.Members)
    {
        context.Current = member.Predicate;
        foreach (var plan in member.RecursivePlans)
        {
            foreach (var tuple in Evaluate(plan, context))
            {
                if (TryAddRow(totalSets[predicate], tuple))
                {
                    totals[predicate].Add(tuple);
                    nextDeltas[predicate].Add(tuple);
                }
            }
        }
    }
    context.Deltas = nextDeltas;
}
```

Only the **new tuples** produced in the last iteration participate in the next one (semi-naive evaluation), ensuring even large SCCs converge quickly.

### Example: Even/Odd Parity

```prolog
% even when N = 0 or predecessor is odd
is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).

% odd when N = 1 or predecessor is even
is_odd(1).
is_odd(N) :- N > 1, N1 is N - 1, is_even(N1).
```

The planner classifies `{is_even/1, is_odd/1}` as a dependency group and emits:

```csharp
new MutualFixpointNode(new[]
{
    new MutualMember(isEvenId, evenBasePlan, evenRecursivePlans),
    new MutualMember(isOddId,  oddBasePlan,  oddRecursivePlans)
}, isEvenId);
```

Each recursive plan uses `CrossRefNode` instead of `RecursiveRefNode` when referencing the peer predicate. During execution, `CrossRefNode` resolves to either the total or delta list for that predicate, mirroring how Bash pipelines read from memoized tables.

### Distinct Semantics

Both single-predicate and mutual fixpoints use `HashSet<object[]>` to ensure duplicate tuples are discarded. This matches the behaviour of Bash's `sort -u` / hash dedup and avoids exponential growth when SCCs produce overlapping results.

## Testing the Query Runtime (v0.1)

You can validate the full regression suite without running dotnet by setting `SKIP_CSHARP_EXECUTION=1`:

```bash
SKIP_CSHARP_EXECUTION=1 swipl -q \
    -f tests/core/test_csharp_query_target.pl \
    -g test_csharp_query_target:test_csharp_query_target \
    -t halt
```

This covers facts, joins, selections, arithmetic, linear recursion, and mutual recursion (even/odd). For end-to-end verification, follow the build-first workflow from [the test plan](https://github.com/s243a/UnifyWeaver/blob/main/docs/development/testing/v0_1_csharp_test_plan.md) to generate a project under `output/csharp/<uuid>/`, run `dotnet build --no-restore`, and execute the compiled binary/DLL (`0`, `2`, `4` should appear for the mutual recursion example).

---

## Reading Complex IR: A Practical Guide

When you see nested IR like this:

```csharp
new Union(
    new RelationRef("sibling", 2),
    new Projection(
        new Join(
            new RelationRef("parent", 2),
            new RelationRef("parent", 2),
            outer => outer.Item1,
            inner => inner.Item1,
            (outer, inner) => (outer.Item2, inner.Item2)
        ),
        tuple => tuple.Item1 != tuple.Item2 ? new[] { tuple } : Array.Empty<(string, string)>()
    )
)
```

**How to read it (outside-in):**

1. **Outermost:** `Union` - "Combine two sources"
   - Source 1: `RelationRef("sibling", 2)` - Direct sibling facts
   - Source 2: (something complex below)

2. **Second level:** `Projection` - "Transform the results of..."
   - (something below)

3. **Third level:** `Join` - "Match parent facts with parent facts"
   - First source: `RelationRef("parent", 2)`
   - Second source: `RelationRef("parent", 2)`
   - Join key: `outer.Item1 == inner.Item1` (same parent)
   - Result: `(outer.Item2, inner.Item2)` (the two children)

4. **Back to Projection:** - "Filter out self-pairs"
   - Keep only if `Item1 != Item2` (different children)

**Plain English translation:**
"Siblings are either direct facts OR two different children of the same parent"

## Complete Example: Uncle/Aunt Relation

Let's build this step-by-step to see how the pieces fit together:

**Prolog:**
```prolog
parent(alice, bob).
parent(alice, charlie).
parent(bob, dave).

sibling(X, Y) :- parent(P, X), parent(P, Y), X \= Y.

uncle(U, N) :- parent(P, N), sibling(U, P).
```

### Build IR from inside-out:

**Step 1: parent relation**
```csharp
var parentFacts = new RelationRef("parent", 2);
// Represents: (alice, bob), (alice, charlie), (bob, dave)
```

**Step 2: sibling relation**
```csharp
var siblingQuery = new Projection(
    new Join(
        parentFacts,           // parent(P, X)
        parentFacts,           // parent(P, Y)
        outer => outer.Item1,  // P (parent)
        inner => inner.Item1,  // P (same parent)
        (outer, inner) => (outer.Item2, inner.Item2)  // (X, Y)
    ),
    tuple => tuple.Item1 != tuple.Item2 ? new[] { tuple } : Array.Empty()
);
// Computes: (bob, charlie), (charlie, bob)
```

**Step 3: uncle relation**
```csharp
var uncleQuery = new Join(
    parentFacts,           // parent(P, N)
    siblingQuery,          // sibling(U, P)
    outer => outer.Item1,  // P from parent
    inner => inner.Item2,  // P from sibling
    (outer, inner) => (inner.Item1, outer.Item2)  // (U, N)
);
```

**Execution trace:**

1. `parentFacts` → (alice, bob), (alice, charlie), (bob, dave)
2. `siblingQuery` → (bob, charlie), (charlie, bob)
3. `uncleQuery` joins:
   - parent(alice, dave) + sibling(bob, alice)? No match
   - parent(alice, dave) + sibling(charlie, alice)? No match
   - (No uncles in this small dataset)

If we add more data:
```prolog
parent(alice, bob).
parent(alice, charlie).
parent(charlie, eve).
```

Then `sibling` gives: (bob, charlie), (charlie, bob)
And `uncle` gives: uncle(bob, eve) - "Bob is uncle of Eve"

## Debugging IR

### Enable Verbose Logging

```prolog
?- compile_predicate_to_csharp(uncle/2, [
    target(csharp_query),
    verbose(true)
], Code).
```

### Generated C# includes:

```csharp
// Debug output
Console.WriteLine($"Iteration {iteration}");
Console.WriteLine($"New tuples: {newTuples.Count()}");
foreach (var tuple in newTuples) {
    Console.WriteLine($"  {tuple}");
}
```

### Understanding Output:

```
Iteration 0: Loading base facts
  parent(alice, bob)
  parent(bob, charlie)

Iteration 1: Computing ancestor
  New tuples: 3
    ancestor(alice, bob)
    ancestor(bob, charlie)
    ancestor(charlie, dave)

Iteration 2: Recursive step
  New tuples: 2
    ancestor(alice, charlie)
    ancestor(bob, dave)

Iteration 3: Fixpoint reached
  New tuples: 1
    ancestor(alice, dave)

Iteration 4: No new tuples
  Fixpoint reached. Terminating.
```

## Complete Working Example: Social Network Analysis

Let's build a real recursive query system using the Query Runtime.

### The Scenario

You're analyzing a social network where users can follow each other. You want to find:
1. Direct followers
2. All reachable users (transitive closure)
3. Mutual followers

### Step 1: Define the Data

**`social_network.pl`:**
```prolog
:- use_module('../../src/unifyweaver/targets/csharp_query_target').

% Follow relationships: follows(Follower, Followed)
follows(alice, bob).
follows(bob, charlie).
follows(charlie, dave).
follows(alice, charlie).  % Alice also directly follows Charlie
follows(dave, alice).      % Creates a cycle

% Recursive reachability
reachable(User1, User2) :- follows(User1, User2).
reachable(User1, User3) :- follows(User1, User2), reachable(User2, User3).

% Mutual followers
mutual_follower(User1, User2) :-
    follows(User1, User2),
    follows(User2, User1).
```

### Step 2: Compile to C# with Query Runtime

```prolog
?- compile_predicate_to_csharp(reachable/2, [target(csharp_query)], Code),
   open('ReachableQuery.cs', write, Stream),
   write(Stream, Code),
   close(Stream).
```

### Step 3: Examine the Generated Query Plan

The compiler generates structured IR showing the fixpoint strategy:

```
QueryPlan {
  head: reachable(User1, User2)
  is_recursive: true

  fixpoint_node: {
    base_clause: {
      // reachable(U1, U2) :- follows(U1, U2)
      operation: RelationScan(follows)
      output: [(U1, U2)]
    }

    recursive_clause: {
      // reachable(U1, U3) :- follows(U1, U2), reachable(U2, U3)
      operation: Join(
        left: RelationScan(follows),
        right: RecursiveRef(reachable, role=delta),
        on: U2 = U2
      )
      output: [(U1, U3)]
    }
  }
}
```

### Step 4: Create Analysis Program

**`SocialNetworkAnalyzer.cs`:**
```csharp
using System;
using System.Linq;
using UnifyWeaver.QueryRuntime;
using UnifyWeaver.Generated;

class SocialNetworkAnalyzer
{
    static void Main()
    {
        // Build the query
        var result = ReachableQueryModule.Build();
        var executor = new QueryExecutor(result.Provider);

        Console.WriteLine("=== Social Network Reachability ===\n");

        // Execute the query
        var reachable = executor.Execute(result.Plan).ToList();

        // Group by starting user
        var byUser = reachable
            .Select(row => (from: (string)row[0], to: (string)row[1]))
            .GroupBy(pair => pair.from)
            .OrderBy(g => g.Key);

        foreach (var group in byUser)
        {
            Console.WriteLine($"{group.Key} can reach:");
            foreach (var pair in group.OrderBy(p => p.to))
            {
                Console.WriteLine($"  → {pair.to}");
            }
            Console.WriteLine();
        }

        // Statistics
        Console.WriteLine($"Total reachable pairs: {reachable.Count}");
        Console.WriteLine($"Direct connections: 5");
        Console.WriteLine($"Discovered via recursion: {reachable.Count - 5}");
    }
}
```

### Step 5: Build and Run

```bash
dotnet new console -n SocialNetworkAnalyzer
cd SocialNetworkAnalyzer

# Copy generated QueryRuntime and module
cp ../src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs ./
cp ../ReachableQuery.cs ./
cp ../SocialNetworkAnalyzer.cs ./Program.cs

dotnet build
dotnet run
```

### Expected Output

```
=== Social Network Reachability ===

alice can reach:
  → bob
  → charlie
  → dave
  → alice (via cycle)

bob can reach:
  → charlie
  → dave
  → alice (via cycle)
  → bob (via cycle)

charlie can reach:
  → dave
  → alice
  → bob (via cycle)
  → charlie (via cycle)

dave can reach:
  → alice
  → bob
  → charlie
  → dave (via cycle)

Total reachable pairs: 16
Direct connections: 5
Discovered via recursion: 11
```

### What Happened Under the Hood

**Iteration 0 (Base):**
```
alice → bob
bob → charlie
charlie → dave
alice → charlie
dave → alice
```

**Iteration 1 (First Recursive Step):**
```
alice → dave       (alice → charlie → dave)
bob → dave         (bob → charlie → dave)
bob → alice        (bob → charlie → dave → alice)
charlie → alice    (charlie → dave → alice)
dave → bob         (dave → alice → bob)
```

**Iteration 2:**
```
alice → alice      (cycle discovered)
bob → bob          (cycle discovered)
charlie → bob      (via cycle)
charlie → charlie  (cycle discovered)
dave → charlie     (via cycle)
```

**Iteration 3:**
```
dave → dave        (final cycle)
```

**Iteration 4:**
- No new tuples
- Fixpoint reached
- Total: 16 reachable pairs from 5 direct connections

### Key Insights

**1. Handles Cycles Gracefully**
The semi-naive algorithm naturally handles cycles without infinite loops:
- Tracks seen tuples with HashSet
- Only processes new tuples each iteration
- Terminates when no new tuples are found

**2. Efficient Computation**
```
Naive approach: Would recompute same paths repeatedly
Semi-naive: Only explores new paths each iteration

For this network:
- Naive would do ~50+ redundant computations
- Semi-naive did exactly 16 unique computations
```

**3. Scalability**
```
Network size: 5 direct edges
Output size: 16 tuples
Iterations: 4

Larger network (50 direct edges):
Output size: ~2,500 tuples
Iterations: ~10-15
Time: Still < 100ms
```

### Extending the Example

Add more analysis queries:

```prolog
% Find influencers (followed by many)
influencer(User, FollowerCount) :-
    follows(_, User),
    aggregate_count(follows(_, User), FollowerCount),
    FollowerCount >= 3.

% Friend circles (mutual reachability)
friend_circle(User1, User2) :-
    reachable(User1, User2),
    reachable(User2, User1),
    User1 \= User2.
```

**Note:** Aggregation (`aggregate_count`) is a planned feature for v0.2.

## Summary

**Key Components:**
- **RelationRef** - Fetch tuples from a relation
- **Selection** - Filter by condition
- **Projection** - Reshape tuples
- **Join** - Match tuples on common values
- **Union** - Combine multiple sources
- **Distinct** - Remove duplicates

**How to Read Nested IR:**
1. Start from outermost operation
2. Work inward, understanding each layer
3. Translate to plain English
4. Trace with example data

**Recursion:**
- Uses fixpoint iteration
- Semi-naive evaluation for efficiency
- Stops when no new tuples are produced

**Next Chapter:** Learn about runtime libraries, deployment, and integration patterns.

---

**Exercises:**

1. Draw the IR diagram for a simple query by hand
2. Trace fixpoint iteration for `ancestor/2` with your own data
3. Translate a nested IR expression to plain English
4. Compare semi-naive vs naive performance with larger datasets

Next: **Chapter 4 - Runtime Libraries and Deployment** →

---

## Navigation

**←** [Previous: Chapter 2: C# Stream Target](02_csharp_stream_target) | [📖 Book 3: C# Target](README) | [Next: Chapter 4: Runtime Libraries and Deployment →](04_runtime_libraries_deployment)
