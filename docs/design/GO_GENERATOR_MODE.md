# Go Generator Mode Design Document

## Overview

This document describes the Go generator mode (`mode(generator)`) in UnifyWeaver. Generator mode enables fixpoint-based Datalog evaluation with recursive closure, joins, stratified negation, aggregation, and more.

## Status: ✅ Complete

All planned features have been implemented:

| Feature | PR | Description |
|---------|-----|-------------|
| Core (joins, negation, fixpoint) | #233 | Base generator mode with fixpoint loop |
| Aggregation | #234 | `aggregate_all/3,4` with count, sum, min, max, avg |
| Indexing | #235 | O(1) joins via hash-based index |
| JSON Input | #236 | Load initial facts from stdin JSONL |
| Parallel Execution | #237 | `workers(N)` for concurrent fixpoint |
| Database Persistence | #238 | `db_backend(bbolt)` for incremental computation |
| HAVING Clause | #239 | Post-aggregation filtering |

---

## Quick Start

### Basic Usage

```prolog
% Define rules
parent(john, mary).
parent(mary, sue).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Compile to Go
?- compile_predicate_to_go(ancestor/2, [mode(generator)], Code).
```

### Running the Generated Code

```bash
go run ancestor.go
# Output (JSONL):
# {"relation":"parent","args":{"arg0":"john","arg1":"mary"}}
# {"relation":"ancestor","args":{"arg0":"john","arg1":"sue"}}
# ...
```

---

## Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `mode(generator)` | - | **Required.** Enable generator mode |
| `json_input(true)` | false | Load additional facts from stdin JSONL |
| `workers(N)` | 1 | Parallel execution with N goroutines |
| `db_backend(bbolt)` | - | Enable database persistence |
| `db_file(Path)` | `'facts.db'` | Database file path |
| `db_bucket(Name)` | `'facts'` | Database bucket name |

### Examples

```prolog
% Basic
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

---

## Features

### Fixpoint Evaluation

The core loop iterates until no new facts are derived:

```go
func Solve() map[string]Fact {
    total := make(map[string]Fact)
    for _, fact := range GetInitialFacts() {
        total[fact.Key()] = fact
    }
    
    idx := BuildIndex(total)
    
    changed := true
    for changed {
        changed = false
        var newFacts []Fact
        for _, fact := range total {
            newFacts = append(newFacts, ApplyRule_1(fact, total, idx)...)
        }
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
// Before (linear scan):
for _, j1 := range total {
    if j1.Relation == "ancestor" && j1.Args["arg0"] == fact.Args["arg1"] { ... }
}

// After (indexed):
for _, j1Ptr := range idx.Lookup("ancestor", "arg0", fact.Args["arg1"]) {
    j1 := *j1Ptr
    ...
}
```

### Stratified Negation

Negation is validated at compile time and checked against the total set:

```prolog
path(X, Y) :- edge(X, Y), \+ blocked(X, Y).
```

```go
negFact := Fact{Relation: "blocked", Args: map[string]interface{}{"arg0": x, "arg1": y}}
if _, exists := total[negFact.Key()]; exists {
    continue  // Skip if negated fact exists
}
```

### Aggregation

Supports `aggregate_all/3` (ungrouped) and `aggregate_all/4` (grouped):

```prolog
% Ungrouped: count all items
item_count(N) :- aggregate_all(count, item(_, _), N).

% Grouped: sum by department
dept_total(Dept, Total) :- aggregate_all(sum(S), salary(Dept, S), Dept, Total).
```

**Supported operations:** `count`, `sum`, `min`, `max`, `avg`

### HAVING Clause

Builtins after aggregation filter results:

```prolog
% Only departments with total > 5000
dept_high(Dept, Total) :- 
    aggregate_all(sum(S), salary(Dept, S), Dept, Total),
    Total > 5000.
```

```go
if len(values) > 0 {
    agg := 0.0; for _, v := range values { agg += v }
    // HAVING clause filter
    if !(agg > 5000) {
        continue
    }
    results = append(results, ...)
}
```

### JSON Input

Load facts from stdin in JSONL format:

```bash
echo '{"relation":"parent","args":{"arg0":"alice","arg1":"bob"}}' | ./ancestor
```

### Parallel Execution

Split fact processing across goroutines:

```prolog
compile_predicate_to_go(ancestor/2, [mode(generator), workers(4)], Code).
```

Workers process chunks of facts concurrently, with results collected via channel.

### Database Persistence

Enable incremental Datalog with bbolt:

```prolog
compile_predicate_to_go(ancestor/2, [mode(generator), db_backend(bbolt)], Code).
```

1. **Load:** Read existing facts from database at startup
2. **Compute:** Run fixpoint including loaded facts
3. **Save:** Persist all facts after fixpoint completes

---

## Fact Representation

```go
type Fact struct {
    Relation string                 `json:"relation"`
    Args     map[string]interface{} `json:"args"`
}

func (f Fact) Key() string {
    b, _ := json.Marshal(f)
    return string(b)
}
```

---

## Testing

```bash
# Run generator tests
swipl -g "run_tests" -t halt tests/core/test_go_generator.pl

# Run aggregation tests  
swipl -g "run_tests" -t halt tests/core/test_go_generator_aggregates.pl
```

---

## References

- [proposals/generator_mode_cross_target.md](file:///home/s243a/Projects/UnifyWeaver/proposals/generator_mode_cross_target.md) — Original cross-target proposal
- [docs/HANDOFF_CSHARP_GENERATOR.md](file:///home/s243a/Projects/UnifyWeaver/docs/HANDOFF_CSHARP_GENERATOR.md) — C# generator mode status
- [src/unifyweaver/targets/common_generator.pl](file:///home/s243a/Projects/UnifyWeaver/src/unifyweaver/targets/common_generator.pl) — Shared generator infrastructure
