# Computed Recursive Increments: Philosophy

## Theoretical Motivation

### Spectral Dimensionality and Graph Geometry

The effective distance formula `d_eff = (Σ dᵢ^(-n))^(-1/n)` uses a
dimensionality parameter `n` that characterizes the intrinsic geometry
of the graph. In a d-dimensional lattice, the number of nodes reachable
within radius r grows as r^d. The parameter `n` encodes this growth
rate, controlling how aggressively short paths dominate the distance
measure.

### Semantic Distance vs Routing Distance

Not all hops in a graph carry equal meaning. This distinction gives
rise to two fundamentally different distance measures:

**Routing distance** counts raw hops. Dense hub nodes with high degree
are beneficial — they reduce path lengths (the scale-free advantage).
In routing, you WANT hubs because they provide shortcuts. Kleinberg's
small-world routing result shows that efficient O(log²n) routing
requires link probability proportional to 1/r^α where α equals the
network dimension.

**Semantic distance** measures how much meaning is traversed. A hop
through a category with 100 children carries less semantic specificity
than a hop through a category with 3 children. The high-degree node
acts as a "collapsed tree" — it compresses what would be a deeper,
more specific hierarchy into a single level.

### The Collapsed Tree Model

A node with out-degree k in a hierarchy can be modeled as having
collapsed a balanced tree of depth log_b(k) into one level, where b
is the natural branching factor of the graph (its spectral dimension).

For a graph with dimension n:
- A node with degree n contributes exactly 1 semantic hop (natural)
- A node with degree n² contributes 2 semantic hops (collapsed 2 levels)
- A node with degree √n contributes 0.5 semantic hops (sub-natural)

The corrected semantic distance through a path becomes:

    d_semantic = Σ_i log_n(degree(node_i))

The dimensionality parameter `n` does triple duty:
1. **Aggregation exponent** in the d_eff power mean
2. **Log base** for the degree correction
3. **Spectral dimension** characterizing the graph geometry

### When to Penalize and When Not To

| Application | Distance type | Dense hubs | Degree correction |
|------------|--------------|-----------|------------------|
| Network routing | Routing | Help (shortcuts) | No — reward hubs |
| Semantic classification | Semantic | Dilute meaning | Yes — penalize |
| Category influence | Configurable | Application-specific | Optional |
| Random walk analysis | Probabilistic | Attract walkers | Depends on goal |
| Supply chain costing | Weighted | Neutral (cost is data) | Use actual costs |

The key insight is that the SAME recursive computation framework must
support both — the difference is only in what the increment expression
computes. A constant increment of 1 gives routing distance. A computed
increment of `log_n(degree)` gives semantic distance. The compilation
machinery should handle both without special-casing either.

## The Generalization Gap

UnifyWeaver's native lowering currently handles recursive predicates
with **constant increments**:

```prolog
path(X, Y, H) :- edge(X, Y), H is 1.
path(X, Z, H) :- edge(X, Y), path(Y, Z, H1), H is H1 + 1.
```

The increment `1` is a literal. Every target (Go, Rust, Python, AWK, C#)
compiles this by hard-coding the increment into the DFS loop or fixpoint
iteration.

But many real recursive computations have **increments that depend on
data**:

```prolog
path(X, Z, H) :- edge(X, Y), cost(X, Cost), path(Y, Z, H1), H is H1 + Cost.
```

Here `Cost` comes from a joined relation. The increment varies per hop.
This pattern appears in:

- **Weighted shortest path** — edge weights from a cost relation
- **Semantic distance** — node degree from the adjacency structure
- **Network latency** — measured delay per link
- **Bill of materials** — quantity multipliers in supply chains
- **Accumulated probability** — transition probabilities in Markov chains

None of these compile today. The native lowering sees `H is H1 + Cost`
and either fails (Cost is unbound) or falls back to a generic fixpoint
that doesn't handle per-path visited semantics.

## Design Principle: Eval, Not Goal

The degree-corrected semantic distance is the **evaluation criterion**
for this work, not the goal. The goal is to make UnifyWeaver's
compilation handle the general pattern:

```prolog
recursive_pred(X, Z, Acc) :-
    edge_relation(X, Y),
    auxiliary_relation(X, AuxVal),          % joined lookup
    recursive_pred(Y, Z, Acc1),
    Acc is Acc1 + f(AuxVal).               % computed increment
```

This pattern has three parts the compiler must recognize:

1. **Edge traversal** — `edge_relation(X, Y)` (already handled)
2. **Auxiliary join** — `auxiliary_relation(X, AuxVal)` (new)
3. **Computed accumulator** — `Acc is Acc1 + f(AuxVal)` (new)

If UnifyWeaver handles this generally, it handles weighted graphs,
degree-corrected distance, supply chains, and any other recursive
accumulation over joined data.

## Why This Matters for UnifyWeaver

The current constant-increment pattern was sufficient for the effective
distance benchmark because hop counting is the simplest case. But it
limits UnifyWeaver to a narrow class of graph problems.

Most real-world graph computations involve **edge or node properties**
that participate in the recursion:

| Domain | Edge/node property | Accumulation |
|--------|-------------------|-------------|
| Knowledge graphs | Semantic type weight | Weighted path length |
| Supply chains | Quantity multiplier | Total units required |
| Network routing | Latency/bandwidth | Total cost |
| Social networks | Trust score | Propagated trust |
| Ontologies | Subsumption depth | Inheritance distance |

Supporting computed increments moves UnifyWeaver from "can compile
basic graph reachability" to "can compile real graph analytics."

## Evaluation: Degree-Corrected Semantic Distance

The specific evaluation problem:

```prolog
% Each hop contributes log_n(degree) instead of 1
semantic_distance(Cat, Anc, D, Visited) :-
    category_parent(Cat, Mid),
    \+ member(Mid, Visited),
    node_degree(Cat, Deg),
    Step is log(Deg) / log(N),
    semantic_distance(Mid, Anc, D1, [Mid|Visited]),
    D is D1 + Step.
```

This exercises the full pattern:
- Edge traversal (`category_parent`)
- Auxiliary join (`node_degree` — derived from the adjacency structure)
- Computed increment (`log(Deg) / log(N)` — arithmetic over joined data)
- Per-path visited semantics (same as existing benchmark)

Success means: the same Prolog source compiles to all targets and
produces correct semantic distances matching a reference implementation.

## Two Compilation Strategies

### Strategy 1: Extended native lowering (Go, Rust, Python, AWK)

Extend the DFS template generators to:
1. Recognize auxiliary joins in the recursive clause body
2. Generate lookup code for auxiliary relations
3. Compute the increment inline in the DFS loop

The generated code would look like (Go example):

```go
for _, neighbor := range adj[current] {
    if visited[neighbor] { continue }
    deg := float64(len(adj[current]))
    step := math.Log(deg) / math.Log(float64(n))
    results = append(results, Result{seed, neighbor, dist + step})
    // ... recurse with updated visited
}
```

### Strategy 2: Richer fixpoint plan composition (C# query engine)

The plan compiler builds a recursive plan that joins the edge relation
with auxiliary relations during fixpoint iteration. The IR composes:
- `JoinNode` to bind auxiliary values
- `ArithmeticNode` to compute the increment
- `FixpointNode` to drive the iteration

No new node type is needed — just richer plan composition within the
existing framework.

## Non-Goals

- Implementing degree-corrected distance as a built-in node type
- Optimizing for this specific problem (general solution preferred)
- Supporting arbitrary recursive functions (focus on linear accumulation
  with joined lookups)
