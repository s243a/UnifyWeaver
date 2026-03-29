# Cross-Target Effective Distance Benchmark: Philosophy, Theory & Future Work

## Motivation

UnifyWeaver's core promise is **write once in Prolog, compile to many targets**.
Each target brings different strengths — AWK's streaming speed, Go's compiled
performance, C#'s query engine with native fixpoint iteration, Python's ecosystem
richness. But until now, we have not had a single non-trivial workload that
exercises all targets on the same task, making it impossible to compare them
meaningfully.

The **effective distance computation over a Wikipedia category hierarchy** is an
ideal benchmark because it combines three capabilities that stress-test each
target differently:

1. **Recursive graph traversal** (finding all paths through a DAG)
2. **Aggregation over recursive results** (summing a derived quantity)
3. **Post-aggregation arithmetic** (applying a power-mean formula)

## Theoretical Foundation

### The Effective Distance Formula

Given an article connected to a target folder through multiple paths of lengths
d₁, d₂, ..., dₖ through the Wikipedia category hierarchy:

    d_eff = (Σ dᵢ^(-n))^(-1/n)

where **n is a dimensionality parameter** (default n=5).

This is a **generalized power mean** with negative exponent. The parameter n
controls the relative contribution of short versus long paths:

| n value | Behavior | Interpretation |
|---------|----------|----------------|
| n → ∞   | min(dᵢ) | Only shortest path matters |
| n = 5   | Short paths dominate, long paths contribute | Default balance |
| n = 2   | Euclidean-like mean | Moderate blending |
| n = 1   | Harmonic mean | All paths contribute significantly |
| n → 0   | Geometric mean | Equal log-weighting |

### Why n=5?

The choice of n=5 reflects the **intrinsic dimensionality** of the combined
Wikipedia + Pearltrees graph structure:

- Wikipedia's category graph has effective dimension ~6 (consistent with
  Kleinberg's small-world routing theory where link probability ∝ 1/r^α and
  α must equal the network dimension for efficient O(log²n) routing)
- Pearltrees' organizational hierarchy has effective dimension ~4 (shallower,
  more curated tree)
- n=5 balances these two structures

This connects to the project's broader work on **spectral dimensionality**:
the dimension parameter determines how the graph's metric structure weights
locality versus global connectivity. Items closer to root (shorter paths)
should sort closer to the root node, and the dimensionality controls how
aggressively this proximity sorting behaves.

**Prior work:**
- `docs/proposals/SMALL_WORLD_ROUTING.md` — Kleinberg's critical result:
  α must equal network dimension d for efficient routing
- `docs/proposals/custom_distance_blending.md` — L^n power mean distance
  blending framework
- `docs/proposals/KERNEL_SMOOTHING_THEORY.md` — Graph Laplacian spectral
  decomposition, Green's functions as low-pass filters on eigenvalues
- `docs/CLUSTER_COUNT_THEORY.md` — Effective rank (Σσ)²/Σσ² as intrinsic
  dimensionality estimator

### Why This is a Good Benchmark

The effective distance computation is interesting precisely because it is
**not a flat scan** — it requires the interaction of three computational
patterns:

**Pattern 1: Recursive Path Finding (Datalog-style)**

Finding all paths from an article through the category hierarchy to a target
folder requires transitive closure with path collection. This is natural in
Prolog/Datalog and maps directly to fixpoint iteration, but challenges
streaming and imperative targets.

**Pattern 2: Aggregation Over Recursive Results**

Once paths are found, we must aggregate per-(article, folder) pair. This
requires grouping and summing d^(-n) values — a relational aggregation over
the output of a recursive computation. This is where the C# query engine's
`AggregateNode` + `FixpointNode` composition should shine.

**Pattern 3: Post-Aggregation Arithmetic**

The final d_eff = WeightSum^(-1/n) is straightforward arithmetic but must
compose cleanly with the aggregation output. This tests each target's ability
to chain computation stages.

### Why Multiple Targets Matter

Each target makes different trade-offs:

| Target | Recursion Strategy | Aggregation | Expected Strength |
|--------|-------------------|-------------|-------------------|
| **C# Query** | Native FixpointNode, TransitiveClosureNode, semi-naive iteration | AggregateNode with typed delegates | Complex recursive queries; caching |
| **Go** | Compiled fixpoint loops, map-based lookups | Built-in sum/count/avg + stddev/median/percentile/collect | Raw compiled speed; rich aggregation |
| **AWK** | Transitive closure (recently added) | Built-in sum/count/avg/min/max | Streaming speed on large flat data |
| **Python** | Tail recursion → while loops, memoized recursion | Via native Python (needs transpilation support) | Ecosystem; reference comparison to hand-crafted code |

The benchmark will expose where each target's compilation strategy succeeds
and where it hits limitations — providing concrete evidence for prioritizing
feature work.

### Bottom-Up vs. Top-Down Evaluation and Memory

Wikipedia's category graph contains cycles. This forces all targets to
implement cycle detection, but the strategies differ fundamentally:

**Top-down with memoization** (Go, Python, AWK): The transpiled code
traverses the graph recursively, carrying a visited set through each call.
Each path exploration allocates stack frames and visited-set copies. For
deep graphs with high branching factor (category paths 10-20 hops deep),
this means deep call stacks and per-frame memory overhead.

**Bottom-up / semi-naive** (C# Query Engine): The `FixpointNode` evaluates
iteratively — each round discovers new (Cat, Ancestor, Hops) tuples (the
"delta set"), merges them into the accumulated result `HashSet<T>`, and
repeats until no new tuples are found. This approach:

- **No recursion stack**: Iteration is flat, so deep graphs don't risk
  stack overflow
- **Automatic deduplication**: The `HashSet` ensures each tuple is
  discovered exactly once, regardless of how many paths lead to it
- **Bounded working memory**: Only the delta set (newly discovered tuples
  per iteration) is in active working memory; the full result set grows
  monotonically but is never re-traversed

This is a genuine architectural advantage of the query plan representation
— not just a speed difference, but a fundamentally more memory-efficient
evaluation strategy for recursive graph queries over large datasets. At
50K+ articles with dense category hierarchies, the top-down approaches
may struggle with memory while the bottom-up approach scales predictably.

### Connection to Existing Hand-Crafted Python

The `wikipedia_categories.py` module (`src/unifyweaver/data/`) already
implements this exact computation in hand-crafted Python:

- `walk_hierarchy_all_paths()` — BFS collecting all paths through the DAG
- `find_all_folder_connections()` — Groups paths by matched folder
- `compute_effective_distance()` — Applies the d_eff formula

This serves as the **reference implementation** and ground truth. The
benchmark will compare transpiled solutions (from a single Prolog source)
against this hand-crafted baseline, demonstrating that UnifyWeaver can
generate correct, performant code across targets from a declarative
specification.

**Prior work:**
- `src/unifyweaver/data/wikipedia_categories.py` — Hand-crafted reference
- `scripts/fetch_wikipedia_physics.py` — Cohere Wikipedia dataset fetcher
  with semantic filtering
- `scripts/fetch_wikipedia_categories.py` — Category hierarchy parser
  (enwiki categorylinks SQL dump → SQLite)
- `docs/proposals/wikipedia_hierarchy_bridge.md` — Architecture for bridging
  Wikipedia categories to Pearltrees organizational structure
- `examples/physics-mindmap/generate_data.py` — Physics mindmap data
  generation from Supabase/wikipedia-en-embeddings (Cohere)

### Decomposition Principle

The key insight is that d_eff decomposes into operations that existing targets
already support:

```prolog
% d_eff = (Σ dᵢ^(-n))^(-1/n)  where n=5
effective_distance(Article, Folder, Deff) :-
    aggregate_all(sum(W),
        (path_to_folder(Article, Folder, Hops),
         W is Hops ^ (-5)),
        WeightSum),
    Deff is WeightSum ^ (-1/5).
```

- `path_to_folder/3` — recursive, all targets have some recursion support
- `W is Hops ^ (-5)` — arithmetic, all targets support `is/2`
- `aggregate_all(sum(W), ...)` — sum aggregation, supported in all targets
- `Deff is WeightSum ^ (-1/5)` — post-aggregation arithmetic

No single step requires a new primitive. The challenge is composing them
correctly in each target language.

## Future Work: Tree Construction from Effective Distance

The effective distance computation has a natural extension to **tree/mindmap
construction**. In a clean tree, node depths are integers. In a DAG with
multiple paths, d_eff produces fractional values that reflect how "between
layers" a node sits:

- d_eff ≈ 1.0 → first layer (direct category match)
- d_eff ≈ 2.0 → second layer
- d_eff ≈ 1.4 → between layers (multiple paths of different lengths)

Subsequent layers of a mindmap should have roughly integer distance to root.
The dimensionality parameter n controls how sharply layers separate: high n
gives crisper integer-like layers (shortest path dominates), low n gives
fuzzier layering where alternative paths blur boundaries.

Since the underlying data is graph-based (a DAG, not a tree), the mapping
to tree layers is inherently approximate. But the effective distance provides
a principled continuous depth that can guide tree construction algorithms:

- **Layer assignment**: Round d_eff to nearest integer for layer placement
- **Ambiguous nodes**: Fractional d_eff identifies nodes that could belong
  to multiple layers — candidates for cross-links or "see also" sections
- **Subtree coherence**: Nodes with similar d_eff to the same folder form
  natural sibling groups

This connects to the project's MST circle folder grouping work
(`docs/proposals/mst_circle_folder_grouping.md`) where tree structure
emerges from graph partitioning, and the hierarchy objective function
(`scripts/mindmap/hierarchy_objective.py`) that evaluates tree quality
via semantic distance and entropy metrics.

Once the cross-target benchmark establishes correctness and performance of
the effective distance computation, extending it to tree construction would
be a natural next step — with the same Prolog program compiled to all
targets, now also generating organizational hierarchies.
