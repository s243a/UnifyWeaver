# Computed Recursive Increments: Specification

## Pattern Definition

A **computed recursive increment** predicate has this general form:

```prolog
% Base case
pred(X, Y, BaseVal) :-
    edge(X, Y),
    aux(X, AuxVal),                    % optional auxiliary join
    BaseVal is f(AuxVal).              % computed base value

% Recursive case
pred(X, Z, Acc) :-
    edge(X, Y),
    aux(X, AuxVal),                    % auxiliary join
    pred(Y, Z, Acc1),                  % recursive call
    Acc is Acc1 + g(AuxVal).           % computed increment
```

Where:
- `edge/2` is the traversal relation
- `aux/N` is an auxiliary relation providing per-node or per-edge data
- `f()` and `g()` are arithmetic expressions over auxiliary values
- The accumulator `Acc` is linearly composed (addition, multiplication,
  min, max — not arbitrary recursion)

## Pattern Recognition

The compiler must detect this pattern by analyzing the recursive clause
body. The key components to identify:

### 1. Edge goal (already detected)

The goal that provides the traversal variable binding:
```prolog
edge(X, Y)        % X bound from head, Y is the step variable
```

### 2. Auxiliary goals (new)

Goals that bind values used in the accumulator arithmetic but do NOT
participate in the recursive call:
```prolog
aux(X, AuxVal)    % X from edge, AuxVal used in arithmetic only
```

Detection criteria:
- Not the recursive call
- Not the edge relation
- Binds a variable that appears in the `is/2` expression
- Does not bind the step variable (Y) or accumulator variable

### 3. Computed accumulator (new)

The `is/2` goal whose left side is the head accumulator and whose
right side references both the recursive accumulator and auxiliary
values:
```prolog
Acc is Acc1 + g(AuxVal)
```

Detection criteria:
- Left side matches the accumulator position in the head
- Right side contains the accumulator from the recursive call
- Right side contains variables bound by auxiliary goals
- The operator is associative (for correctness of bottom-up evaluation)

## Classification

Extend `classify_goal_sequence` (or the target-specific pattern
matchers) to recognize these sub-patterns:

```prolog
% New classification result
goal_classification(
    edge_goal(EdgePred, SourceVar, StepVar),
    auxiliary_goals([aux(AuxPred, BindVars, ResultVars)]),
    recursive_call(RecPred, RecAccVar),
    computed_increment(Expression, AccVar, RecAccVar, AuxVars)
).
```

## Supported Expressions

The initial implementation should support:

| Expression | Example | Use case |
|-----------|---------|----------|
| `Acc is Acc1 + AuxVal` | Weighted path sum | Edge-weighted graphs |
| `Acc is Acc1 + f(AuxVal)` | `log(Deg)/log(N)` | Semantic distance |
| `Acc is Acc1 * AuxVal` | Quantity multiplication | Supply chains |
| `Acc is min(Acc1, AuxVal)` | Bottleneck path | Network bandwidth |
| `Acc is max(Acc1, AuxVal)` | Worst-case path | Latency analysis |

Expressions that should NOT be supported (out of scope):
- Non-linear accumulation: `Acc is Acc1 * Acc1 + AuxVal`
- Multiple accumulators: `pred(X, Z, Acc1, Acc2) :- ...`
- Accumulators that depend on the recursive depth itself

## Target-Specific Code Generation

### Go

```go
// Auxiliary lookup (precomputed)
degreeMap := buildDegreeMap(adj)

// In DFS loop
for _, neighbor := range adj[current] {
    if visited[neighbor] { continue }
    auxVal := degreeMap[current]
    step := computeStep(auxVal)  // generated from expression
    newAcc := acc + step
    results = append(results, Result{seed, neighbor, newAcc})
    // ... recurse
}
```

### Rust

```rust
let degree_map: HashMap<String, usize> = build_degree_map(&adj);

for neighbor in adj.get(current).unwrap_or(&vec![]) {
    if visited.contains(neighbor) { continue; }
    let aux_val = *degree_map.get(current).unwrap_or(&1) as f64;
    let step = aux_val.ln() / (n as f64).ln();
    let new_acc = acc + step;
    results.push((seed.clone(), neighbor.clone(), new_acc));
    // ... recurse
}
```

### Python

```python
degree_map = {node: len(neighbors) for node, neighbors in adj.items()}

for neighbor in adj.get(current, []):
    if neighbor in visited: continue
    aux_val = degree_map.get(current, 1)
    step = math.log(aux_val) / math.log(n)
    new_acc = acc + step
    results.append((seed, neighbor, new_acc))
    # ... recurse
```

### C# Query Engine

The C# query engine should lower this pattern to a dedicated
**path-aware linear accumulation** plan node rather than to the generic
`FixpointNode`.

Rationale:
- `FixpointNode` deduplicates by tuples, not by derivation path
- visited-list semantics require branch-local state
- computed increments must be evaluated at each hop with the current
  auxiliary bindings

Conceptually, the emitted node carries:

```text
PathAwareLinearAccumulationNode
  edge relation
  auxiliary lookup specs
  base accumulator expression
  recursive increment expression
  optional group/invariant columns
  optional max-depth bound
```

At runtime, evaluation proceeds by DFS from each seed:

1. Bind the outgoing edge from the current node
2. Resolve auxiliary values needed by the increment expression
3. Evaluate the base or recursive accumulator expression
4. Emit the new result row
5. Recurse with a copied visited set for that branch

The existing `PathAwareTransitiveClosureNode` is the degenerate case of
this more general mechanism where:
- there are no auxiliary relations
- the base expression is a constant
- the recursive expression is `PrevAcc + Constant`

## Auxiliary Relation Sources

The auxiliary relation can come from:

1. **Explicit facts**: `node_degree(physics, 12).` — user-provided
2. **Derived from edge relation**: degree = count of edges from a node.
   The compiler could auto-derive this when it recognizes a degree
   pattern (e.g., `aggregate_all(count, edge(X, _), Deg)`)
3. **External data**: loaded from TSV/CSV at runtime

For the evaluation problem (semantic distance), option 2 is most
natural — the degree is derivable from `category_parent/2`. The
compiler should recognize this and generate the degree map as a
preprocessing step or derived lookup relation for the path-aware
accumulation runtime.

## Correctness Criteria

For the evaluation (degree-corrected semantic distance with n=5):

1. **Path correctness**: Same paths found as existing hop-count version
2. **Distance correctness**: Each path's semantic distance =
   `Σ log_5(degree(node_i))` for nodes along the path
3. **Aggregation correctness**: d_eff_semantic matches reference
   implementation (Prolog or hand-crafted Python)
4. **Cross-target consistency**: All targets produce the same output
   (within floating-point tolerance)

## Test Datasets

Reuse existing benchmark datasets:
- Dev (198 edges) — correctness validation
- 300 (6008 edges) — cross-target comparison
- 1K+ — scaling analysis

The degree distribution provides natural variation:
- Dev: degrees range from 1 to ~10
- 10K: degrees range from 1 to ~200+
- High-degree nodes exercise the log correction significantly
