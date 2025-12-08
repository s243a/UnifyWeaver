<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 3: Generator Mode - Fixpoint Evaluation

Generator mode implements **semi-naive fixpoint evaluation**, a technique from Datalog that efficiently computes recursive queries. Unlike procedural mode which processes records individually, generator mode materializes sets of facts and iterates until no new facts can be derived.

## When to Use Generator Mode

Generator mode is essential for:

- **Transitive closure** - Finding all paths in a graph
- **Reachability queries** - Can node A reach node B?
- **Recursive joins** - Joining relations with recursive dependencies
- **Deep recursion** - Avoiding Python's recursion limit

### The Problem with Procedural Recursion

Consider computing ancestors in a deep family tree:

```prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

In procedural mode, this could hit Python's recursion limit (default ~1000). Generator mode has no such limit because it uses iteration instead of recursion.

## How Semi-Naive Evaluation Works

The algorithm maintains two sets:

- **`total`** - All facts discovered so far
- **`delta`** - Facts discovered in the current iteration (new facts)

### Algorithm Steps

1. Initialize `total` and `delta` with input facts
2. For each rule, apply it to facts in `delta`
3. Add newly derived facts to `new_delta`
4. If `new_delta` is empty, we've reached fixpoint - done!
5. Otherwise, `delta = new_delta` and repeat from step 2

### Why "Semi-Naive"?

A naive approach would apply rules to ALL facts in `total` each iteration. Semi-naive optimization only considers facts in `delta` (the new ones), dramatically reducing redundant computation.

## Basic Example: Transitive Closure

**Prolog:**
```prolog
edge(a, b).
edge(b, c).
edge(c, d).

path(X, Y) :- edge(X, Y).
path(X, Z) :- edge(X, Y), path(Y, Z).
```

**Compilation:**
```prolog
?- compile_predicate_to_python(path/2, [mode(generator)], Code).
```

**Generated Python (simplified):**
```python
from typing import Dict, Set, Iterator, Tuple, Any
import json
import sys

class FrozenDict:
    """Immutable, hashable dictionary for use in sets."""
    def __init__(self, items: Tuple[Tuple[str, Any], ...]):
        self._items = items
        self._hash = hash(items)

    @staticmethod
    def from_dict(d: Dict) -> 'FrozenDict':
        return FrozenDict(tuple(sorted(d.items())))

    def to_dict(self) -> Dict:
        return dict(self._items)

    def get(self, key: str, default=None):
        for k, v in self._items:
            if k == key:
                return v
        return default

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self._items == other._items

def process_stream_generator(records: Iterator[Dict]) -> Iterator[Dict]:
    total: Set[FrozenDict] = set()
    delta: Set[FrozenDict] = set()

    # Initialize with input facts
    for record in records:
        frozen = FrozenDict.from_dict(record)
        if frozen not in total:
            total.add(frozen)
            delta.add(frozen)
            yield record

    # Fixpoint iteration
    while delta:
        new_delta: Set[FrozenDict] = set()

        for fact in delta:
            # Apply Rule 1: path(X, Y) :- edge(X, Y)
            # (Copy rule - already in total as edge facts)

            # Apply Rule 2: path(X, Z) :- edge(X, Y), path(Y, Z)
            for new_fact in _apply_rule_2(fact, total):
                if new_fact not in total:
                    total.add(new_fact)
                    new_delta.add(new_fact)
                    yield new_fact.to_dict()

        delta = new_delta

def _apply_rule_2(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    """path(X, Z) :- edge(X, Y), path(Y, Z)."""
    # If fact is an edge(X, Y), look for path(Y, Z) in total
    x = fact.get('arg0')
    y = fact.get('arg1')
    if x is not None and y is not None:
        for other in total:
            if other.get('arg0') == y:  # Join on Y
                z = other.get('arg1')
                new_fact = FrozenDict.from_dict({'arg0': x, 'arg1': z})
                yield new_fact
```

## FrozenDict: The Key Data Structure

Python's `dict` is mutable and unhashable, so it can't be stored in a `set`. The `FrozenDict` class provides:

- **Immutability** - Cannot be modified after creation
- **Hashability** - Can be used as set elements
- **Dictionary interface** - `get()` method for accessing values

```python
# Creating a FrozenDict
fd = FrozenDict.from_dict({'name': 'Alice', 'age': 30})

# Using in a set
facts = set()
facts.add(fd)

# Checking membership (O(1) average)
fd2 = FrozenDict.from_dict({'name': 'Alice', 'age': 30})
print(fd2 in facts)  # True

# Converting back to dict
d = fd.to_dict()  # {'name': 'Alice', 'age': 30}
```

## Running Generator Mode

**Input (JSONL):**
```json
{"arg0": "a", "arg1": "b"}
{"arg0": "b", "arg1": "c"}
{"arg0": "c", "arg1": "d"}
```

**Command:**
```bash
cat edges.jsonl | python3 path.py
```

**Output:**
```json
{"arg0": "a", "arg1": "b"}
{"arg0": "b", "arg1": "c"}
{"arg0": "c", "arg1": "d"}
{"arg0": "a", "arg1": "c"}
{"arg0": "b", "arg1": "d"}
{"arg0": "a", "arg1": "d"}
```

The last three lines are derived facts:
- `a → c` (via `a → b → c`)
- `b → d` (via `b → c → d`)
- `a → d` (via `a → b → c → d`)

## Supported Rule Patterns

### Copy Rules (Fact Projection)

```prolog
path(X, Y) :- edge(X, Y).
```

Facts matching `edge(X, Y)` are copied to `path(X, Y)`.

### Binary Joins

```prolog
path(X, Z) :- edge(X, Y), path(Y, Z).
```

Joins `edge` and `path` on the shared variable `Y`.

### N-Way Joins

```prolog
triple_path(X, W) :- edge(X, Y), edge(Y, Z), edge(Z, W).
```

Multiple goals in the body are joined sequentially.

### Disjunction

```prolog
connected(X, Y) :- edge(X, Y).
connected(X, Y) :- edge(Y, X).  % Symmetric
```

Multiple clauses are all applied during fixpoint iteration.

## Negation (Stratified)

Generator mode supports negation-as-failure with stratification:

```prolog
blocked(b, c).

safe_path(X, Y) :- edge(X, Y), \+ blocked(X, Y).
safe_path(X, Z) :- edge(X, Y), \+ blocked(X, Y), safe_path(Y, Z).
```

**Generated Python:**
```python
def _apply_rule(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    x = fact.get('arg0')
    y = fact.get('arg1')

    # Check negation: \+ blocked(X, Y)
    blocked_fact = FrozenDict.from_dict({'relation': 'blocked', 'arg0': x, 'arg1': y})
    if blocked_fact in total:
        return  # Negation fails

    # Negation succeeds, continue with rule
    yield FrozenDict.from_dict({'arg0': x, 'arg1': y})
```

**Stratification Requirement:** Negated predicates must not depend (directly or indirectly) on the predicate being defined. The compiler validates this at compile time.

## Comparison with C# Query Runtime

Both Python Generator Mode and C# Query Runtime use semi-naive fixpoint evaluation:

| Feature | Python Generator | C# Query Runtime |
|---------|------------------|------------------|
| Algorithm | Semi-naive | Semi-naive |
| Data structure | FrozenDict in sets | Immutable records |
| Memory | Higher (Python objects) | Lower (value types) |
| Speed | Good | Excellent |
| Dependencies | None | .NET Runtime |
| Debugging | Easy (readable code) | Requires .NET tools |

**Choose Python Generator when:**
- Quick prototyping
- Integration with Python ecosystem
- Readable generated code for debugging
- No .NET dependency desired

**Choose C# Query when:**
- Maximum performance needed
- Large datasets
- Production deployment

## Performance Tips

### 1. Minimize Fact Size

Smaller facts = faster hashing and comparison:

```python
# Prefer
{"id": 123}

# Over
{"id": 123, "metadata": {"created": "...", "updated": "..."}}
```

### 2. Use Appropriate Input Format

For large inputs, NUL-delimited JSON can be faster:

```prolog
compile_predicate_to_python(path/2, [mode(generator), record_format(nul_json)], Code).
```

### 3. Filter Early

If possible, filter input facts before feeding to the fixpoint:

```bash
# Filter edges first, then compute closure
jq 'select(.weight > 0)' edges.jsonl | python3 path.py
```

### 4. Understand Complexity

Fixpoint iteration is O(n²) in the worst case for binary joins. For very large graphs, consider:
- Partitioning the input
- Using indexed data structures (like Go generator mode)
- Switching to C# for better performance

## Debugging Generator Mode

### Print Iteration Progress

Modify the generated code to track iterations:

```python
iteration = 0
while delta:
    iteration += 1
    print(f"Iteration {iteration}: {len(delta)} new facts", file=sys.stderr)
    # ... rest of loop
```

### Inspect Total Set

```python
# At end of fixpoint
print(f"Total facts: {len(total)}", file=sys.stderr)
for fact in total:
    print(f"  {fact.to_dict()}", file=sys.stderr)
```

### Trace Rule Application

```python
def _apply_rule_2(fact, total):
    print(f"Applying rule 2 to {fact.to_dict()}", file=sys.stderr)
    # ... rule logic
```

## Complete Example: Social Network Analysis

Let's compute "friends of friends" in a social network:

**Prolog:**
```prolog
% Direct friendship
friend(alice, bob).
friend(bob, carol).
friend(carol, dave).
friend(dave, eve).

% Friends of friends (2 hops)
fof(X, Z) :- friend(X, Y), friend(Y, Z), X \= Z.

% Extended network (transitive)
connected(X, Y) :- friend(X, Y).
connected(X, Z) :- friend(X, Y), connected(Y, Z).
```

**Compilation:**
```prolog
?- compile_predicate_to_python(connected/2, [mode(generator)], Code).
```

**Input:**
```json
{"arg0": "alice", "arg1": "bob"}
{"arg0": "bob", "arg1": "carol"}
{"arg0": "carol", "arg1": "dave"}
{"arg0": "dave", "arg1": "eve"}
```

**Output includes derived connections:**
```json
{"arg0": "alice", "arg1": "carol"}
{"arg0": "alice", "arg1": "dave"}
{"arg0": "alice", "arg1": "eve"}
{"arg0": "bob", "arg1": "dave"}
{"arg0": "bob", "arg1": "eve"}
{"arg0": "carol", "arg1": "eve"}
```

## Summary

Generator mode provides:

- **Fixpoint evaluation** - Guaranteed termination for Datalog queries
- **No recursion limit** - Iteration instead of call stack
- **Semi-naive optimization** - Only process new facts each iteration
- **Set semantics** - Automatic deduplication
- **Negation support** - Stratified negation-as-failure

Use generator mode whenever you need transitive closure, graph algorithms, or recursive queries that might exceed Python's recursion depth.

---

## Navigation

**←** [Previous: Chapter 2: Procedural Mode](02_procedural_mode) | [📖 Book 5: Python Target](./) | [Next: Chapter 4: Recursion Patterns →](04_recursion_patterns)
