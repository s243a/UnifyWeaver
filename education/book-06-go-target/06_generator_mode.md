<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 6: Generator Mode - Fixpoint Datalog Evaluation

Generator mode (`mode(generator)`) enables fixpoint-based Datalog evaluation in Go. This powerful feature supports recursive closure, joins, stratified negation, aggregation, parallel execution, and database persistence.

## Overview

While the basic Go target generates streaming record processors, generator mode creates a complete Datalog engine that:

- Computes transitive closures (recursive rules)
- Performs joins with O(1) indexed lookups
- Supports stratified negation
- Provides aggregation with HAVING clauses
- Enables parallel execution with goroutines
- Persists state to databases for incremental computation

## Status

Generator mode is **production-ready** with all planned features implemented:

| Feature | Description |
|---------|-------------|
| Core fixpoint | Iterative evaluation until no new facts |
| Indexed joins | O(1) hash-based lookups |
| Stratified negation | `\+ goal` with compile-time validation |
| Aggregation | count, sum, min, max, avg |
| HAVING clause | Post-aggregation filtering |
| JSON input | Load facts from stdin |
| Parallel execution | Multi-goroutine processing |
| Database persistence | BoltDB incremental computation |

## Basic Usage

### Compiling a Recursive Predicate

```prolog
% Load UnifyWeaver
:- use_module('src/unifyweaver/targets/go_target').

% Define facts
parent(john, mary).
parent(mary, sue).
parent(sue, alice).

% Define recursive rules
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Compile to Go with generator mode
?- compile_predicate_to_go(ancestor/2, [mode(generator)], Code),
   open('ancestor.go', write, S),
   write(S, Code),
   close(S).
```

### Running the Generated Code

```bash
go run ancestor.go
```

### Output (JSONL format)

```json
{"relation":"parent","args":{"arg0":"john","arg1":"mary"}}
{"relation":"parent","args":{"arg0":"mary","arg1":"sue"}}
{"relation":"parent","args":{"arg0":"sue","arg1":"alice"}}
{"relation":"ancestor","args":{"arg0":"john","arg1":"mary"}}
{"relation":"ancestor","args":{"arg0":"john","arg1":"sue"}}
{"relation":"ancestor","args":{"arg0":"john","arg1":"alice"}}
{"relation":"ancestor","args":{"arg0":"mary","arg1":"sue"}}
{"relation":"ancestor","args":{"arg0":"mary","arg1":"alice"}}
{"relation":"ancestor","args":{"arg0":"sue","arg1":"alice"}}
```

## Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `mode(generator)` | - | **Required.** Enable generator mode |
| `json_input(true)` | false | Load additional facts from stdin JSONL |
| `workers(N)` | 1 | Parallel execution with N goroutines |
| `db_backend(bbolt)` | - | Enable database persistence |
| `db_file(Path)` | `'facts.db'` | Database file path |
| `db_bucket(Name)` | `'facts'` | Database bucket name |

### Option Examples

```prolog
% Basic generator mode
compile_predicate_to_go(ancestor/2, [mode(generator)], Code).

% With JSON input from stdin
compile_predicate_to_go(ancestor/2, [mode(generator), json_input(true)], Code).

% With 4 parallel workers
compile_predicate_to_go(ancestor/2, [mode(generator), workers(4)], Code).

% With database persistence
compile_predicate_to_go(ancestor/2, [mode(generator), db_backend(bbolt),
                                      db_file('my.db')], Code).

% Combined: parallel + database + json input
compile_predicate_to_go(ancestor/2, [mode(generator), workers(4),
                                      db_backend(bbolt), json_input(true)], Code).
```

## Stratified Negation

Generator mode supports negation-as-failure with stratification validation:

```prolog
edge(a, b).
edge(b, c).
edge(c, d).
blocked(b, c).

% Find paths that aren't blocked
path(X, Y) :- edge(X, Y), \+ blocked(X, Y).
path(X, Z) :- edge(X, Y), \+ blocked(X, Y), path(Y, Z).
```

The compiler validates that negation doesn't create cycles (stratification check).

## Aggregation

Generator mode supports `aggregate_all/3` (ungrouped) and `aggregate_all/4` (grouped):

### Ungrouped Aggregation

```prolog
item(widget, 10).
item(gadget, 20).
item(thing, 15).

% Count all items
item_count(N) :- aggregate_all(count, item(_, _), N).

% Sum all values
total_value(T) :- aggregate_all(sum(V), item(_, V), T).
```

### Grouped Aggregation

```prolog
salary(eng, 1000).
salary(eng, 1500).
salary(sales, 800).
salary(sales, 1700).
salary(hr, 600).

% Sum salaries by department
dept_total(Dept, Total) :-
    aggregate_all(sum(S), salary(Dept, S), Dept, Total).
```

### HAVING Clause (Post-Aggregation Filtering)

Add constraints after aggregation to filter results:

```prolog
% Only departments with total > 1000
dept_high(Dept, Total) :-
    aggregate_all(sum(S), salary(Dept, S), Dept, Total),
    Total > 1000.
```

**Supported aggregation operations:** `count`, `sum`, `min`, `max`, `avg`

## JSON Input

Load facts dynamically from stdin instead of hardcoding:

```prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Enable JSON input
:- compile_predicate_to_go(ancestor/2, [mode(generator), json_input(true)], Code), ...
```

### Running with Piped Data

```bash
echo '{"relation":"parent","args":{"arg0":"john","arg1":"mary"}}
{"relation":"parent","args":{"arg0":"mary","arg1":"sue"}}' | go run ancestor.go
```

## Parallel Execution

Speed up computation with multiple goroutines:

```prolog
:- compile_predicate_to_go(ancestor/2, [mode(generator), workers(4)], Code), ...
```

The fixpoint loop distributes facts across workers, collecting results via channels.

**Tip:** Use `workers(N)` where N matches your CPU core count for best performance.

## Database Persistence

Enable incremental Datalog with BoltDB:

```prolog
:- compile_predicate_to_go(ancestor/2, [mode(generator), db_backend(bbolt),
                                         db_file('ancestry.db')], Code), ...
```

### Workflow

```bash
# First run - computes and saves to ancestry.db
./ancestor

# Later: add more facts via stdin, continue from saved state
echo '{"relation":"parent","args":{"arg0":"alice","arg1":"bob"}}' | ./ancestor
```

The database stores all derived facts, so subsequent runs start from the existing state.

## How It Works: Fixpoint Evaluation

The generated Go code implements semi-naive fixpoint evaluation:

```go
func Solve() map[string]Fact {
    total := make(map[string]Fact)

    // Initialize with base facts
    for _, fact := range GetInitialFacts() {
        total[fact.Key()] = fact
    }

    // Build index for O(1) joins
    idx := BuildIndex(total)

    // Iterate until fixpoint
    changed := true
    for changed {
        changed = false
        var newFacts []Fact

        // Apply all rules to all facts
        for _, fact := range total {
            newFacts = append(newFacts, ApplyRule_1(fact, total, idx)...)
        }

        // Add new facts to total
        for _, nf := range newFacts {
            if _, exists := total[nf.Key()]; !exists {
                total[nf.Key()] = nf
                idx.Add(&nf)
                changed = true
            }
        }
    }

    return total
}
```

### Indexed Joins

Joins use O(1) hash lookups instead of O(n) linear scans:

```go
// Without indexing (O(n)):
for _, j1 := range total {
    if j1.Relation == "ancestor" && j1.Args["arg0"] == fact.Args["arg1"] { ... }
}

// With indexing (O(1)):
for _, j1Ptr := range idx.Lookup("ancestor", "arg0", fact.Args["arg1"]) {
    j1 := *j1Ptr
    ...
}
```

## Tips and Tricks

### Filter Output by Relation

```bash
go run ancestor.go | jq 'select(.relation == "ancestor")'
```

### Pretty Print JSON

```bash
go run ancestor.go | jq .
```

### Count Results

```bash
go run ancestor.go | wc -l
```

### Build Standalone Binary

```bash
go build -o ancestor ancestor.go
./ancestor
```

## Comparison with Other Modes

| Feature | Streaming Mode | Generator Mode |
|---------|---------------|----------------|
| Recursion | Limited | Full (fixpoint) |
| Joins | Stream-based | Indexed O(1) |
| Negation | No | Stratified |
| Aggregation | No | Yes |
| Parallelism | No | Yes |
| Persistence | No | Yes (BoltDB) |
| Use case | Simple transforms | Complex Datalog |

## Summary

Generator mode transforms UnifyWeaver's Go target from a simple record processor into a complete Datalog engine. Use it when you need:

- Recursive queries (transitive closure)
- Complex joins across relations
- Negation-as-failure
- Aggregation with grouping
- Parallel execution for large datasets
- Incremental computation with persistence

---

## Navigation

**←** [Previous: Chapter 5: Semantic Crawling & RDF Processing](05_semantic_crawling) | [📖 Book 6: Go Target](./) | [Next: Book 7: Cross-Target Glue →](../book-07-cross-target-glue/)
