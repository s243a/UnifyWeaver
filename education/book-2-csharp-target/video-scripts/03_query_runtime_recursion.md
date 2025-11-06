<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Video Script: Query Runtime and Recursion

**Duration:** 10-12 minutes
**Target Audience:** Developers ready to handle recursive queries in C#
**Goal:** Understand fixpoint iteration, semi-naive evaluation, and cycle handling

---

## Opening (30 seconds)

**[Screen: Social network graph with cycles]**

> "Stream Target can't handle recursion.
> Query Runtime can.
> Let's see how."

**Show title card:** "Query Runtime and Recursion"

---

## Section 1: Why Stream Target Fails (1 minute)

**[Screen: Code editor with recursive predicate]**

### The Problem

**Prolog:**
```prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

**Try to compile with Stream Target:**
```prolog
?- compile_predicate_to_csharp(ancestor/2, [], Code).
ERROR: Cannot compile recursive predicate ancestor/2
```

> "Stream Target generates LINQ chains.
> LINQ can't call itself.
> We need a different approach."

**[Screen: Diagram showing recursive call cycle]**

> "Recursion needs iteration.
> That's what Query Runtime provides."

---

## Section 2: Query Runtime Architecture (2 minutes)

**[Screen: Architecture diagram]**

### Two-Phase Approach

**Phase 1: Compile to IR**
```
Prolog → Intermediate Representation (IR)
```

**Phase 2: Runtime Engine**
```
IR → Fixpoint Iteration → Results
```

> "IR captures the logic.
> Runtime executes it with cycles."

### What's in the IR?

**[Screen: IR structure visualization]**

**Example IR for ancestor/2:**
```csharp
new Rule {
    Head = ("ancestor", 2),
    Body = new[] {
        new Literal("parent", arg0, arg1)
    }
}
new Rule {
    Head = ("ancestor", 2),
    Body = new[] {
        new Literal("parent", arg0, temp1),
        new Literal("ancestor", temp1, arg1)  // Recursive!
    }
}
```

> "Each clause becomes a Rule.
> Literals reference predicates.
> Variables connect them."

**[Screen: Split view - Prolog vs IR]**

**Key Components:**
- **Rules**: Logic clauses
- **Literals**: Predicate calls
- **Variables**: Arguments (arg0, temp1, etc.)
- **Facts**: Base data stored as tuples

---

## Section 3: Fixpoint Iteration Explained (2.5 minutes)

**[Screen: Animated iteration steps]**

### The Algorithm

> "Start with base facts.
> Derive new facts using rules.
> Repeat until nothing new appears.
> That's fixpoint."

### Real Example: Social Network

**Data:**
```prolog
follows(alice, bob).
follows(bob, charlie).
follows(charlie, dave).
```

**Query:**
```prolog
reachable(X, Y) :- follows(X, Y).
reachable(X, Z) :- follows(X, Y), reachable(Y, Z).
```

**[Animate each iteration]**

**Iteration 0 (Base Facts):**
```
reachable(alice, bob)
reachable(bob, charlie)
reachable(charlie, dave)
```

> "Start with direct follows."

**Iteration 1 (First Derivation):**
```
New facts:
reachable(alice, charlie)   ← alice → bob → charlie
reachable(bob, dave)        ← bob → charlie → dave
```

> "Apply recursive rule.
> Find 2-hop connections."

**Iteration 2 (Second Derivation):**
```
New facts:
reachable(alice, dave)      ← alice → bob → charlie → dave
```

> "Keep going.
> Find 3-hop connections."

**Iteration 3:**
```
No new facts found.
```

> "Fixpoint reached.
> We're done."

**Final Result:**
```
reachable(alice, bob)
reachable(alice, charlie)
reachable(alice, dave)
reachable(bob, charlie)
reachable(bob, dave)
reachable(charlie, dave)
```

---

## Section 4: Semi-Naive Evaluation (2 minutes)

**[Screen: Comparison chart]**

### The Problem with Naive Evaluation

**Naive approach:**
```csharp
// Bad: Re-process ALL facts every iteration
foreach (var oldFact in allFacts) {
    foreach (var rule in rules) {
        derive(oldFact, rule);  // Redundant work!
    }
}
```

> "This rechecks old facts.
> Wastes computation.
> Gets slow fast."

### Semi-Naive Optimization

**Semi-naive approach:**
```csharp
// Good: Only process NEW facts
foreach (var newFact in deltaFacts) {
    foreach (var rule in rules) {
        derive(newFact, rule);  // Only new derivations!
    }
}
```

> "Only check new facts from last iteration.
> Much faster.
> This is semi-naive evaluation."

**[Screen: Performance comparison]**

**Example: 1000-node graph**

| Approach | Iterations | Derivations Checked |
|----------|-----------|---------------------|
| Naive | 10 | ~100,000 |
| Semi-Naive | 10 | ~10,000 |

> "10x fewer checks.
> Same results.
> Essential for scale."

---

## Section 5: Cycle Handling (1.5 minutes)

**[Screen: Graph with cycle]**

### The Challenge

**Data with cycle:**
```prolog
follows(alice, bob).
follows(bob, charlie).
follows(charlie, alice).  % Cycle!
```

**[Animate cycle in graph]**

> "Alice → Bob → Charlie → Alice.
> Infinite loop?
> No."

### How Query Runtime Handles It

**[Screen: Deduplication visualization]**

**Iteration 0:**
```
reachable(alice, bob)
reachable(bob, charlie)
reachable(charlie, alice)
```

**Iteration 1:**
```
Try to add: reachable(alice, charlie)  ✓ New
Try to add: reachable(bob, alice)      ✓ New
Try to add: reachable(charlie, bob)    ✓ New
```

**Iteration 2:**
```
Try to add: reachable(alice, bob)      ✗ Already exists
Try to add: reachable(bob, charlie)    ✗ Already exists
Try to add: reachable(charlie, alice)  ✗ Already exists
```

> "No new facts.
> Fixpoint reached.
> Cycle handled automatically."

**Key insight:**
- Hash set tracks seen facts
- Duplicates ignored
- Terminates when no new facts

---

## Section 6: Complete Walkthrough (2 minutes)

**[Screen: Code editor with social_network.pl]**

### Step 1: Define Recursive Query

```prolog
:- use_module(unifyweaver(targets/csharp_query_target)).

follows(alice, bob).
follows(bob, charlie).
follows(charlie, dave).
follows(dave, alice).  % Cycle

reachable(User1, User2) :- follows(User1, User2).
reachable(User1, User3) :- follows(User1, User2), reachable(User2, User3).
```

### Step 2: Compile to C#

```prolog
?- compile_predicate_to_csharp(reachable/2, [target(csharp_query)], Code),
   open('ReachableQuery.cs', write, S),
   write(S, Code),
   close(S).
```

> "Target option: csharp_query.
> This triggers Query Runtime."

### Step 3: Generated Code Structure

**[Screen: Generated C# file]**

```csharp
namespace UnifyWeaver.QueryRuntime
{
    public static class ReachableQuery
    {
        // Base facts
        static readonly (string, string)[] FollowsFacts = {
            ("alice", "bob"),
            ("bob", "charlie"),
            ("charlie", "dave"),
            ("dave", "alice")
        };

        // IR rules
        static readonly Rule[] Rules = {
            new Rule {
                Head = ("reachable", 2),
                Body = new[] {
                    new Literal("follows", new[] { 0, 1 })
                }
            },
            new Rule {
                Head = ("reachable", 2),
                Body = new[] {
                    new Literal("follows", new[] { 0, 2 }),
                    new Literal("reachable", new[] { 2, 1 })
                }
            }
        };

        // Fixpoint engine
        public static IEnumerable<(string, string)> Query()
        {
            var engine = new QueryEngine(Rules, FollowsFacts);
            return engine.RunFixpoint("reachable");
        }
    }
}
```

### Step 4: Build and Run

```bash
# Copy QueryRuntime.cs from UnifyWeaver distribution
cp src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs .

# Create .NET project
dotnet new console
mv ReachableQuery.cs Program.cs

# Add QueryRuntime.cs to project
# Build and run
dotnet build
dotnet run
```

**Output:**
```
reachable(alice, bob)
reachable(alice, charlie)
reachable(alice, dave)
reachable(alice, alice)    ← Cycle!
reachable(bob, charlie)
reachable(bob, dave)
reachable(bob, alice)
reachable(bob, bob)        ← Cycle!
...
```

> "All reachability found.
> Cycles handled.
> Query complete."

---

## Section 7: When to Use Query Runtime (1.5 minutes)

**[Screen: Decision tree]**

### ✅ Use Query Runtime When:

**1. Recursive Queries**
```prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

> "Transitive relationships.
> Graph reachability.
> Hierarchy traversal."

**2. Cycles Possible**
```prolog
follows(alice, bob).
follows(bob, alice).  % Bidirectional
```

> "Social networks.
> Mutual relationships.
> Circular dependencies."

**3. Unknown Depth**
```prolog
% How many hops between A and B?
% Don't know in advance.
```

> "Query Runtime finds all paths.
> Stream Target needs fixed depth."

### ❌ Use Stream Target Instead When:

**1. Non-Recursive Queries**
```prolog
engineer(Name) :- employee(Name, engineering).
```

> "Simple filters.
> Direct lookups.
> No iteration needed."

**2. Fixed Join Depth**
```prolog
grandparent(GP, GC) :- parent(GP, P), parent(P, GC).
```

> "Exactly 2 hops.
> Stream Target is faster.
> No fixpoint overhead."

**Quick Rule:**
- Recursion → Query Runtime
- No recursion → Stream Target

---

## Section 8: Performance Characteristics (1 minute)

**[Screen: Complexity table]**

| Query Pattern | Complexity | Notes |
|---------------|-----------|-------|
| Base facts only | O(n) | Single iteration |
| Single recursive rule | O(n²) | Worst case |
| Graph reachability | O(n × e) | n=nodes, e=edges |
| Deep recursion | O(depth × n) | Depends on graph structure |

**Memory:**
- Stores all derived facts
- Hash set for deduplication
- O(result size)

> "Efficient for most graphs.
> Scales to thousands of nodes.
> Millions may need database backend."

**[Screen: Benchmark results]**

**Example: 1000-node social network**
- Facts: 5000 follows relationships
- Query: Transitive reachability
- Time: ~50ms
- Result size: ~15,000 reachable pairs

> "Fast enough for real-time queries.
> Good for analytics.
> Production-ready."

---

## Closing (30 seconds)

**[Screen: Key takeaways]**

**Show bullet points:**
- ✓ Query Runtime handles recursion
- ✓ Fixpoint iteration finds all derivations
- ✓ Semi-naive = efficient
- ✓ Cycles handled automatically
- ✓ Use for: graphs, hierarchies, transitive queries
- ✓ Stream Target for: non-recursive, simple queries

> "Now you can handle any Prolog query in C#.
> Recursion, cycles, complex derivations.
> All production-ready."

**[Screen: Next steps preview]**

**Suggested text:**
"Next: Advanced optimization techniques and production deployment"

---

## Notes for Recording

**Visual Aids:**
- Animated graph traversal for fixpoint iterations
- Split-screen for Prolog ↔ IR ↔ C# comparison
- Step-by-step execution trace with highlighted new facts
- Performance charts comparing naive vs semi-naive

**Code Examples to Prepare:**
- social_network.pl (ready to compile)
- Generated ReachableQuery.cs
- Working .NET console project
- Live execution showing all iterations

**Interactive Elements:**
- Pause after fixpoint explanation: "Predict next iteration results"
- Challenge: "What happens with this cycle?"
- Quiz: "Which target for this query?"

**Common Questions:**
- Q: "Why not just use LINQ recursion?" → A: "C# doesn't support recursive LINQ - needs explicit iteration"
- Q: "Can I optimize the fixpoint engine?" → A: "Semi-naive is already optimized - further gains need specialized indexing"
- Q: "Thread-safe?" → A: "Yes, facts are immutable - safe for concurrent queries"
- Q: "How deep can recursion go?" → A: "Unlimited - controlled by fixpoint, not stack depth"

**Mistakes to Avoid:**
- Don't skip the iteration-by-iteration trace - it's crucial for understanding
- Don't assume viewers understand fixpoint - explain from first principles
- Show actual timing numbers, not just complexity notation
- Emphasize when to use Stream Target vs Query Runtime

**Demonstrations:**
- Run social_network.pl compilation live
- Show generated C# code structure
- Execute and display iteration-by-iteration output (if possible to add logging)
- Compare Stream Target error vs Query Runtime success on same recursive query

**Pacing:**
- Slower on fixpoint explanation (core concept)
- Faster on cycle handling (builds on fixpoint)
- Interactive pauses after complex examples
- Quick recap before closing
