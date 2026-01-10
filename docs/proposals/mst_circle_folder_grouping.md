# Proposal: MST Circle-Based Folder Grouping

## Problem Statement

When organizing large collections of mindmaps into folders, we want:
1. Semantically coherent folders (related items together)
2. Controlled folder sizes (4-8 items per folder ideally)
3. Hierarchical structure that reflects natural topic relationships

Current approaches (K-means clustering) partition based on embedding centroids, which may not respect the natural semantic topology.

## Proposed Algorithm: MST Circle Partitioning

### Core Concepts

1. **MST (Minimum Spanning Tree)**: Built from all items using embedding distances as edge weights. Represents minimal semantic connections between items.

2. **Cuts**: Edges removed from the MST to partition it into disconnected regions.

3. **Circles**: The connected regions that remain after all cuts are made. Each circle becomes a folder.

4. **Boundary Nodes**: Nodes adjacent to cut edges - the "interface" of each circle.

### Distance Metrics

- **Within a circle**: Distance between two nodes = sum of MST edge weights along the path connecting them (using original MST structure before cuts).

- **Between circles**: Must traverse through branches, accumulating edge weights.

- **Direct transitions**: Within a circle, you can transition directly between any nodes on the cuts (boundary nodes), but distance is still calculated via MST path.

### Optimization Objective

**Minimize the sum of pairwise distances between boundary nodes of each circle.**

For a circle with boundary nodes {A, B, C, D}:
```
Circle cost = d(A,B) + d(A,C) + d(A,D) + d(B,C) + d(B,D) + d(C,D)
```

Where d(X,Y) = sum of MST edge weights along path from X to Y.

**Total cost = sum of all circle costs**

### Constraints

- Target folder size: N nodes per circle (or range [min_size, max_size])
- Branching factor: 4-8 boundary nodes per circle (manageable pairwise combinations)
- **Max depth**: Maximum folder hierarchy depth - forces larger groups to fit all items within depth limit. This prevents over-fragmentation and encourages meaningful groupings. **Scales logarithmically with collection size**: `max_depth ≈ log_F(N)` where F = target folder size. User specifies "I want ~20 items per folder" and depth constraint follows: `log(N) / log(F)`. For N=10000, F=20 → ~3 levels.

### Emergent Properties

This metric naturally favors:
1. **Tight clusters**: Small distances between boundary nodes
2. **Lower branching**: Fewer boundary nodes = fewer pairwise combinations

With branching of 4-8:
- 4 boundary nodes → C(4,2) = 6 pairs
- 6 boundary nodes → C(6,2) = 15 pairs
- 8 boundary nodes → C(8,2) = 28 pairs

### Algorithm Sketch

1. **Build MST** from all items (embedding cosine distance as edge weight)

2. **Initialize**: Each node is its own circle

3. **Bottom-up merge**:
   - Consider merging adjacent circles
   - Compute new circle cost (pairwise boundary distances)
   - Merge if cost is below threshold AND size constraint satisfied
   - Continue until no more beneficial merges

4. **Result**: Each circle becomes a folder

### Alternative: Top-Down Approach

1. Start with full MST as one circle
2. Find cut that minimizes total cost of resulting circles
3. Recursively split circles that exceed max_size
4. Stop when all circles are within size constraints

### Folder Hierarchy

The cuts naturally define a hierarchy:
- Circles at the same "level" of cuts become sibling folders
- Parent-child relationships follow the MST branch structure
- Folder depth emerges from the number of cuts traversed

### Comparison to Current Approach (K-means)

| Aspect | K-means | MST Circles |
|--------|---------|-------------|
| Basis | Embedding centroids | Tree topology |
| Respects natural structure | Partially | Yes |
| Size control | Via K | Via merge/split constraints |
| Folder naming | Representative title | Boundary node representative |
| Hierarchy | Imposed | Emergent from MST |

## Implementation Notes

- Pre-compute all pairwise MST path distances (or compute on demand)
- Use Union-Find for efficient circle merging
- Cache boundary node sets for each circle
- Consider parallel computation for large collections

## Computational Complexity

The optimization problem is:
- **Minimize**: Sum of pairwise boundary distances across all circles
- **Subject to**: Folder size constraints, max depth

General graph partitioning is NP-hard, but since we operate on a **tree** (MST), efficient solutions may exist:
- Dynamic programming on trees often yields polynomial solutions
- Greedy heuristics may work well in practice
- Local search can refine initial solutions

### Greedy Local Search Strategy

1. **Identify** large cuts (high cost due to quadratic pairwise growth)
2. **Find** semantically close neighboring cuts
3. **Move** a boundary node from large cut → smaller nearby cut
4. **Accept** if total cost decreases (n² → (n-1)² on large cut offsets small increase on destination)
5. **Repeat** until no improving moves

### Randomized Variant (Stochastic Hill Climbing)

1. **Randomly pick a node**
2. **Flip coin** (50/50): use hop-based or semantic-distance neighborhood
3. **Select top-k sibling branches** from chosen criteria
4. **Evaluate moves** among this sample
5. **Accept** improving moves
6. **Repeat** until convergence or iteration limit

This adds stochasticity to escape local optima and reduce computation via sampling.

### Initialization Strategies

**Option A: Start with no cuts, add cuts**
- Begin with one circle containing all nodes
- Gradually add cuts to split
- Cuts near roots/leaves only useful at start of optimization

**Option B: Start with all cuts, merge**
- Every node starts as its own circle
- Merge circles together
- Problem: initially violates depth constraints

**Option C: Incremental growth (recommended - try first)**
- Start with one seed node
- Add neighboring nodes from MST one at a time
- At each step: decide same circle or new cut?
- Optimize locally as you grow
- Never violates constraints - builds organically
- Assumes 2D tree (branches don't cross) for well-defined contiguous regions

All three options are valid; recommend starting with Option C as it avoids constraint violations and builds the solution organically. **Grow from the root** of the tree for natural top-down traversal.

### Dynamic Level Constraint

As nodes are added, adjust the depth constraint dynamically:
- Start loose (few elements, plenty of depth room)
- Get stricter as you approach total elements
- Prevents "running out of depth" at the end
- Example: `allowed_depth = max_depth × (1 - elements_added/total) + min_depth`

### Two Algorithm Versions

| Aspect | MST Version | Curated Hierarchy Version |
|--------|-------------|---------------------------|
| Tree source | Built from embeddings | Pearltrees actual hierarchy |
| Constraint | Soft (emergent from similarity) | Hard (fixed structure) |
| Flexibility | Can reorganize freely | Must respect existing parent-child |
| Use case | Fresh organization | Enhance existing curation |

Same cut optimization problem applied to different underlying trees. The curated version respects human curation; the MST version is purely algorithmic.

**MST advantage**: More flexible for items without a clear hierarchical home:
- Orphans with no parent
- Items from other accounts
- New items that don't fit existing categories
- Cross-cutting topics spanning multiple branches

**Hybrid approach** (future work): Use MST on projected embeddings that encode both semantic content and hierarchical position. Implement MST and curated versions separately first.

This naturally balances cut sizes while respecting semantic proximity. The quadratic penalty on large cuts drives items toward smaller, nearby cuts.

**Loop detection required**: Use tabu list (forbid reversing recent moves), track visited states, or only accept strictly improving moves to prevent oscillation.

**Efficiency optimizations**:
- Start at leaves (bottom-up) - smaller subproblems first
- Only consider local neighborhoods - define by:
  - **Hop count**: 2, 4, or 8 branch traversals (powers of 2)
  - **Semantic distance**: threshold on accumulated MST edge weights
  - Or both: "within 4 hops AND distance < threshold"
  - **Alternating**: Switch between hop-based and distance-based rounds to escape local optima
- Don't compare distant branches - prune to nearby candidates
- This reduces from O(n²) candidate moves to O(n × branching_factor)

Further analysis needed to determine if exact polynomial solution exists.

## Open Questions

1. Optimal merge/split strategy (greedy vs. dynamic programming)?
2. How to handle very dense vs. very sparse regions?
3. Should circle cost be normalized by size?
4. How to name folders based on circle content?
5. Is there a polynomial-time exact algorithm for tree partitioning with these constraints?

## Implementation Order

Implement the four versions in this order:

1. **MST Version** - Pure algorithmic approach using embeddings
2. **Curated Hierarchy Version** - Respects existing Pearltrees parent-child relationships
3. **Hybrid Version** - Combines MST flexibility with curated structure (MST on projected embeddings)
4. **Tangent Deviation Version** - Hybrid with manifold-based deviation penalty

### Version 4: Tangent Deviation Hybrid

This version uses differential geometry to measure deviation from curated structure:

**Cost function**: `Total_Cost = MST_circle_cost + λ × tangent_deviation_penalty`

**Tangent deviation at each node**:
```python
def tangent_deviation(node, embeddings, mst_neighbors, curated_neighbors):
    node_vec = embeddings[node]

    # MST-induced tangent (avg direction to MST neighbors)
    t_mst = mean([embeddings[n] - node_vec for n in mst_neighbors])

    # Curated-induced tangent (avg direction to curated neighbors)
    t_curated = mean([embeddings[n] - node_vec for n in curated_neighbors])

    # Deviation = 1 - cosine_similarity
    return 1 - cos_sim(t_mst, t_curated)

# Total penalty = sum over all nodes
tangent_deviation_penalty = sum(tangent_deviation(n, ...) for n in nodes)
```

**Intuition**: The graph Laplacian is the discrete Laplace-Beltrami operator. Graph connectivity induces a manifold approximation. If MST and curated graphs induce the same local differential structure (tangent directions), they are equivalent. Deviation = difference in induced geometry.

**Properties**:
- Cheap to compute (dot products)
- Intuitive: "do the two graphs point you in the same direction at each point?"
- Grounded in differential geometry
- λ controls strength of curated "attraction force"

### Alternative Hybrid Approaches (Future Work)

The hybrid approach described above (MST on projected embeddings) is one option. Other hybrid approaches to explore:

- **Secondary Attraction Force**: Add a term that pulls nodes toward their curated positions, balancing MST-derived semantic grouping with curated placement. The "force" is the gradient of a cost penalty term: `Total_Cost = MST_circle_cost + λ × curated_deviation_penalty`. Candidates for `curated_deviation_penalty`:
  - **Graph Laplacian deviation**: The graph Laplacian is the discrete analog of the Laplace-Beltrami operator on manifolds. Key principle: **graph proximity should reflect manifold locality** - connected nodes should have similar differential properties (tangent spaces, curvature, gradient behavior).

    **Concrete test**: Use MST connectivity to define "nearby" and compute induced differential properties. Do the same with curated hierarchy. If the derivatives match, structures are equivalent. If not, MST has deviated from the curated-induced manifold. The curated structure encodes human knowledge about semantic relationships; deviation means losing that structure.

    **Practical measure**: Cosine similarity between tangent vectors. At each node, estimate tangent from directions to neighbors. Compare MST-induced tangent vs curated-induced tangent: `cos(θ) = t_mst · t_curated`. If cos ≈ 1, structures agree locally. Cheap to compute (dot products), intuitive (do the graphs point you in the same direction?).

    Refinement needed: want invariance to "harmless" rearrangements (sibling reordering). Candidates: spectral distance, resistance distance, heat kernel trace
  - Distance from curated parent in embedding space
  - Hierarchy depth deviation
  - Penalty for separating curated siblings
- **Weighted Edges**: Use curated hierarchy to weight MST edges (shorter distance = same curated parent)
- **Constrained Cuts**: Only allow cuts that don't split curated parent-child pairs

These can be explored after the core MST and Curated versions are working.

## Next Steps

1. Implement MST version prototype on existing mindmap collection
2. Compare folder coherence vs. K-means
3. Implement Curated hierarchy version
4. Implement Hybrid version (projected embeddings)
5. Implement Tangent Deviation version
6. Tune size constraints and λ parameter
7. Evaluate folder naming strategies
8. Compare all four versions on test collection
