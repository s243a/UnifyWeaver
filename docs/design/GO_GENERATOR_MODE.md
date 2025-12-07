# Go Generator Mode Design Document

## Overview

This document describes the design for adding `mode(generator)` to the Go target in UnifyWeaver. Generator mode enables fixpoint-based Datalog evaluation, bringing Go to feature parity with C# and Python generator modes for recursive closure, joins, and stratified negation.

## Motivation

The current Go target excels at streaming operations (JSONL processing, aggregation, GROUP BY) but lacks support for recursive Datalog evaluation. Many use cases require computing transitive closures or derived facts that depend on previously derived facts—this requires a fixpoint iteration loop.

### Example Use Case

```prolog
% Transitive closure - requires fixpoint
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Compile with generator mode
?- compile_predicate_to_go(ancestor/2, [mode(generator)], Code).
```

Without generator mode, Go cannot evaluate this. With generator mode, it produces a self-contained Go program with a fixpoint loop.

---

## Architecture

### Execution Model Comparison

| Target | Streaming Mode | Generator Mode |
|--------|----------------|----------------|
| **C#** | LINQ pipeline | ✅ Fixpoint with `HashSet<Fact>` |
| **Python** | Iterator pipeline | ✅ Fixpoint with `Set[FrozenDict]` |
| **Go** | stdin→stdout | 🆕 Fixpoint with `map[string]Fact` |

### High-Level Flow

```
compile_predicate_to_go(Pred/Arity, [mode(generator)], Code)
    │
    ├─ compute_generator_dependency_closure/2  (find all related predicates)
    ├─ guard_stratified_negation_go/3          (validate negation is stratified)
    ├─ compile_go_generator_facts/3            (emit GetInitialFacts())
    ├─ compile_go_generator_rules/5            (emit ApplyRule_N functions)
    └─ compile_go_generator_execution/4        (emit Solve() fixpoint loop)
```

### Integration with Existing Go Target

Generator mode is **additive**—existing streaming functionality is preserved:

```prolog
compile_predicate_to_go(PredIndicator, Options, GoCode) :-
    ...
    % NEW: Generator mode (check first)
    (   option(mode(generator), Options)
    ->  compile_generator_mode_go(Pred, Arity, Options, GoCode)
    % Existing modes unchanged...
    ;   is_aggregation_predicate(Body)
    ->  compile_aggregation_mode(...)
    ;   is_group_by_predicate(Body)
    ->  compile_group_by_mode(...)
    ...
```

---

## Design Details

### Shared Infrastructure via `common_generator.pl`

Generator mode uses shared predicates for language-agnostic logic:

| Predicate | Purpose |
|-----------|---------|
| `build_variable_map/2` | Map Prolog variables → accessor tokens |
| `translate_expr_common/4` | Translate expressions using target config |
| `translate_builtin_common/4` | Translate builtins (comparisons, arithmetic) |
| `prepare_negation_data/4` | Build key-value pairs for negation checks |

Each target provides a config block:

```prolog
go_generator_config(Config) :-
    Config = [
        access_fmt-"fact[\"arg~w\"]",      % fact["arg0"]
        atom_fmt-"\"~w\"",                  % "value"
        null_val-"nil",
        ops-[
            + - "+", - - "-", * - "*", / - "/",
            > - ">", < - "<", >= - ">=", =< - "<=", 
            =:= - "==", =\= - "!="
        ]
    ].
```

### Fact Representation

```go
// Fact represents a relation tuple
type Fact struct {
    Relation string
    Args     map[string]interface{}
}

// Key returns a canonical string for set membership
func (f Fact) Key() string {
    b, _ := json.Marshal(f)
    return string(b)
}
```

Using JSON marshaling for the key ensures structural equality matching regardless of map iteration order.

### Fixpoint Loop

```go
func Solve() map[string]Fact {
    total := make(map[string]Fact)
    
    // Initialize with base facts
    for _, fact := range GetInitialFacts() {
        total[fact.Key()] = fact
    }
    
    // Iterate until no new facts
    changed := true
    for changed {
        changed = false
        var newFacts []Fact
        
        for _, fact := range total {
            newFacts = append(newFacts, ApplyRule_1(fact, total)...)
            newFacts = append(newFacts, ApplyRule_2(fact, total)...)
            // ... more rules
        }
        
        for _, nf := range newFacts {
            key := nf.Key()
            if _, exists := total[key]; !exists {
                total[key] = nf
                changed = true
            }
        }
    }
    
    return total
}
```

### Negation

Stratified negation is validated at compile time (non-stratified negation is rejected). Generated code checks `total` set:

```go
// For rule: path(X,Y) :- edge(X,Y), \+ blocked(X,Y)
negFact := Fact{Relation: "blocked", Args: map[string]interface{}{"arg0": x, "arg1": y}}
if _, exists := total[negFact.Key()]; exists {
    return results  // Skip if negated fact exists
}
```

### Joins

Multi-relation joins iterate over `total`:

```go
// For rule: grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
func ApplyRule_1(fact Fact, total map[string]Fact) []Fact {
    var results []Fact
    if fact.Relation != "parent" {
        return results
    }
    x := fact.Args["arg0"]
    y := fact.Args["arg1"]
    
    // Join with second parent relation
    for _, f2 := range total {
        if f2.Relation == "parent" && f2.Args["arg0"] == y {
            z := f2.Args["arg1"]
            results = append(results, Fact{
                Relation: "grandparent",
                Args: map[string]interface{}{"arg0": x, "arg1": z},
            })
        }
    }
    return results
}
```

---

## Aggregation Support

### Syntax Normalization

Both syntaxes are supported via aliasing:

```prolog
% Normalize aggregate/3 → aggregate_all/3 in generator mode
normalize_aggregate_goal(aggregate(Op, Goal, R), aggregate_all(Op, Goal, R)) :- !.
normalize_aggregate_goal(G, G).
```

This ensures backward compatibility with existing code using `aggregate/3`.

### Aggregation Over Derived Facts

Unlike streaming mode (which aggregates input), generator mode aggregates over the computed `total` set:

```prolog
% Example: Sum salaries of reachable employees (derived via transitive closure)
reachable_salary(Total) :- 
    aggregate_all(sum(S), reachable_employee(_, _, S), Total).
```

### Supported Operations

| Operation | Go Code |
|-----------|---------|
| `count` | `agg := float64(len(values))` |
| `sum` | `for _, v := range values { agg += v }` |
| `min` | `for _, v := range values { if v < agg { agg = v } }` |
| `max` | `for _, v := range values { if v > agg { agg = v } }` |
| `avg` | `agg = sum / float64(len(values))` |

### Grouped Aggregation

`aggregate_all/4` supports grouping:

```prolog
dept_total(Dept, Total) :- aggregate_all(sum(S), salary(Dept, S), Dept, Total).
```

Generates:

```go
groups := make(map[interface{}][]float64)
for _, f := range total {
    if f.Relation == "salary" {
        key := f.Args["arg0"]  // Dept
        val := f.Args["arg1"]  // S
        groups[key] = append(groups[key], val)
    }
}
for key, values := range groups {
    agg := 0.0
    for _, v := range values { agg += v }
    results = append(results, Fact{...})
}
```

---

## Implementation Phases

| Phase | Description | Effort |
|-------|-------------|--------|
| **1** | Infrastructure: `common_generator` import, config block, mode dispatch | 1-2 days |
| **2** | Fixpoint engine: `Solve()` function, `Fact` type, iteration loop | 2-3 days |
| **3** | Rule compilation: joins, negation, builtins via shared abstractions | 2-3 days |
| **4** | Testing: unit tests, cross-target comparison with C#/Python | 1-2 days |
| **5** | Aggregation: `aggregate_all/3,4` with syntax aliasing | 2-3 days |

**Total: ~2-3 weeks**

---

## Testing Strategy

### Unit Tests

```prolog
% tests/core/test_go_generator.pl

test(transitive_closure) :-
    % parent facts + ancestor rules → compile and run
    compile_predicate_to_go(ancestor/2, [mode(generator)], Code),
    % Write, compile, execute, verify output
    ...

test(stratified_negation) :-
    % edge + blocked facts, path rule with negation
    ...

test(cross_target_comparison) :-
    % Same program compiled to Go, C#, Python
    % All outputs must match
    ...
```

### Manual Verification

1. Compile same Datalog program to all three targets
2. Run on identical input
3. Diff outputs (sorted, canonicalized JSON)

---

## Future Work

- **Indexing**: arg0/arg1 buckets (like C#) for faster joins
- **Parallel evaluation**: Go's goroutines are well-suited for this
- **I/O integration**: Use `json_input(true)` to load initial facts from stdin
- **Database persistence**: Combine with `db_backend(bbolt)` for incremental computation
- **HAVING clause**: Post-aggregation filtering

---

## References

- [proposals/generator_mode_cross_target.md](file:///home/s243a/Projects/UnifyWeaver/proposals/generator_mode_cross_target.md) — Original cross-target proposal
- [docs/HANDOFF_CSHARP_GENERATOR.md](file:///home/s243a/Projects/UnifyWeaver/docs/HANDOFF_CSHARP_GENERATOR.md) — C# generator mode status
- [src/unifyweaver/targets/common_generator.pl](file:///home/s243a/Projects/UnifyWeaver/src/unifyweaver/targets/common_generator.pl) — Shared generator infrastructure
