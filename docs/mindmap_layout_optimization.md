# Mind Map Layout Optimization Proposal

## Current State

The `generate_simplemind_map.py` script produces an initial radial layout with:
- **Micro-clustering**: K-means on embeddings creates 4-8 children per node
- **Level-based radii**: Each level's radius grows to maintain circumferential spacing
- **Semantic hierarchy**: Similar items clustered together based on embedding similarity

This provides a good starting point, but nodes can overlap when angular sectors become narrow at deeper levels.

## Proposed Optimization Approaches

### 1. Force-Directed Refinement

Apply physics simulation to push overlapping nodes apart while preserving hierarchy.

```
Algorithm:
1. Initialize with current positions
2. For each iteration:
   a. Calculate repulsion forces between overlapping/nearby nodes
   b. Calculate attraction forces to maintain parent-child relationships
   c. Add radial constraint force (soft preference for original level)
   d. Update positions with damping
3. Converge when movement falls below threshold
```

**Implemented:** Via `--optimize` flag with configurable iterations.

**Parameters (tuned for zero overlaps):**
- `repulsion_strength`: How hard nodes push apart (default: 100000)
- `attraction_strength`: Weak pull to parent only when >500px away (default: 0.001)
- `radial_weight`: Disabled (default: 0.0) - radial layout provides initialization
- `min_distance`: Target spacing between nodes (default: 120px)
- `iterations`: Max simulation steps (default: 300, configurable via `--optimize-iterations`)

**Results:** Tested on 225-node cluster:
- Before optimization: 120 overlapping node pairs
- After optimization: 0 overlapping node pairs
- Layout expands as needed to fit all nodes

**Pros:** Handles arbitrary overlap patterns, guarantees no overlaps
**Cons:** O(n²) per iteration; layout may expand significantly

### 2. Radial Jitter with Collision Detection

Simpler approach: offset nodes radially to avoid collisions.

```
Algorithm:
1. For each level (outer to inner):
   a. Detect overlapping node pairs
   b. Move one node outward by node_radius + margin
   c. Repeat until no overlaps
```

**Pros:** Fast, preserves angular positions
**Cons:** Can create irregular level boundaries

### 3. Adaptive Node Sizing

Make nodes with more children larger, giving visual hierarchy cues.

```
node_radius = base_radius * (1 + log(1 + n_children) * scale_factor)
```

Then adjust spacing calculations to account for variable node sizes.

**Implemented:** Basic version using `log2(1 + descendants) * 0.4` for font scaling.

**Enhancement: Density-Aware Sizing**

In crowded areas, reduce node size to prevent overlap:

```
local_density = count_nodes_within_radius(node, radius=200) / area
density_factor = 1 / (1 + local_density * k)
final_scale = base_scale * density_factor
```

This means hub nodes in sparse areas stay large, but hubs in crowded areas shrink to fit.

**Pros:** Visual indication of importance, adapts to local density
**Cons:** Requires second pass after positioning

### 4. Angular Rebalancing

Redistribute angular space based on subtree size rather than equal division.

```
Algorithm:
1. Calculate subtree_size(node) recursively
2. Divide parent's angular span proportionally:
   child_span = parent_span * (child_subtree_size / total_subtree_size)
```

**Pros:** Gives more space to larger subtrees
**Cons:** Can create very narrow sectors for small subtrees

### 5. Simulated Annealing

Global optimization to minimize an energy function.

```
Energy function:
E = w1 * overlap_penalty
  + w2 * hierarchy_violation_penalty
  + w3 * angular_deviation_penalty
  + w4 * radial_deviation_penalty

Algorithm:
1. Start with current layout, high temperature T
2. For each step:
   a. Propose random node movement
   b. Calculate delta_E
   c. Accept if delta_E < 0, or with probability exp(-delta_E/T)
   d. Decrease T
3. Return best layout found
```

**Pros:** Can escape local minima, highly configurable
**Cons:** Slow, requires tuning

### 6. Constraint-Based Layout (Linear Programming)

Formulate as optimization problem with constraints.

```
Minimize: sum of squared deviations from initial positions
Subject to:
  - No overlaps: distance(i,j) >= radius_i + radius_j + margin
  - Hierarchy: child radius > parent radius (radial ordering)
  - Sector bounds: child angles within parent's sector
```

**Pros:** Guarantees no overlaps if feasible
**Cons:** May be infeasible for dense graphs, computationally expensive

## Implementation Status

### Completed
- **Force-Directed Refinement** - `--optimize` flag
- **Node Sizing by Descendants** - Enabled by default, disable with `--no-scaling`
- **Mass-Based Repulsion** - Hubs push apart more strongly (mass = 1 + sqrt(descendants))
- **Tethered Leaves** - Leaf nodes stay close to parents (attraction ∝ 1/mass)
- **Connection-Aware Repulsion** - Parent-child pairs repel 0.3x, non-connected pairs repel 1.5x
- **Edge Crossing Minimization** - `--minimize-crossings` flag with `--crossing-passes` control
- **Sibling Edge Detection** - Heuristic for curved line crossings from same parent
- **Larger Leaf Fonts** - Minimum scale 1.2x for leaf node readability

### Results

| Cluster | Nodes | Initial Crossings | Final Crossings |
|---------|-------|-------------------|-----------------|
| people | 47 | 5 | 0 |
| graphlpane | 43 | 6 | 1 |
| differential_geometry | 225 | 147 | 29 |

### Future Enhancements

1. **Time Budget Control** (`--time-limit`)
   - Allow users to cap optimization time
   - Progressive refinement for large clusters

2. **Spatial Indexing** (R-trees)
   - Reduce O(n²) crossing detection to O(n log n)
   - Critical for clusters with 500+ nodes

3. **Angular Rebalancing**
   - Proportional angular allocation based on subtree size
   - Can be added to initial layout pass

4. **LLM-Guided Refinement**
   - Generate image, send to multimodal LLM for suggestions
   - Optional for users wanting further polish

## Edge Crossing Minimization

Reducing edge crossings improves readability. This is an NP-hard problem with many local minima, making gradient-based optimization difficult.

**Status: Implemented** via `--minimize-crossings` flag.

### Hierarchical Priority Approach

Process edges in order of ancestor depth (shallowest first):

```
Algorithm:
1. Sort edges by min(depth(parent), depth(child)) ascending
2. For each edge in order:
   a. Check for crossings with already-placed edges
   b. If crossing detected, try angular adjustments:
      - Rotate subtree within parent's sector
      - Swap sibling subtree positions
   c. Accept configuration that minimizes crossings
3. Deeper edges adapt around already-fixed backbone edges
```

**Rationale:** Edges closer to the root form the "backbone" of the layout. By fixing these first, deeper edges (which have more flexibility) can route around them.

### Detection

Edge crossing detection requires line segment intersection tests:

```python
def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 crosses segment p3-p4."""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return (ccw(p1,p3,p4) != ccw(p2,p3,p4)) and (ccw(p1,p2,p3) != ccw(p1,p2,p4))
```

For n edges, naive detection is O(n²). Spatial indexing (R-trees) can reduce average case.

### Two-Phase Optimization

Combine force-directed layout with crossing minimization:

```
Algorithm:
1. Phase 1: Force-directed until convergence (zero overlaps)
2. Phase 2: For each node (sorted by depth, shallowest first):
   a. Try angular/radial adjustments within bounds
   b. Count edge crossings for each candidate position
   c. Accept position that minimizes crossings
   d. Run brief force-directed pass to restore overlap-free state
   e. Repeat until no improvement for this node
3. Continue until full pass yields no improvements
```

**Key insight:** By moving one node at a time and re-running force-directed, we maintain the overlap-free invariant while optimizing for crossings. Shallow nodes are adjusted first so deeper nodes can adapt.

### Stochastic Methods

Simulated annealing or genetic algorithms could escape local minima:
- Randomly swap subtree positions
- Accept worse configurations probabilistically
- Gradually reduce temperature/mutation rate

**Challenge:** Defining moves that preserve hierarchy while exploring layout space.

## Integration with SimpleMind

The `.smmx` format supports manual editing. Users can:
1. Generate initial layout with this tool
2. Open in SimpleMind
3. Manually adjust problem areas
4. Save refined version

This human-in-the-loop approach may be preferable to fully automated optimization for many use cases.

## Future: LLM-Guided Layout

An LLM could potentially:
- Suggest which nodes to group more tightly
- Identify semantically important nodes to emphasize
- Propose alternative hierarchies based on content understanding

This would complement the embedding-based clustering with higher-level semantic reasoning.
