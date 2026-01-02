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

**Parameters:**
- `repulsion_strength`: How hard nodes push apart
- `attraction_strength`: How strongly children stay near parents
- `radial_weight`: How much to preserve level assignment
- `iterations`: Max simulation steps

**Pros:** Handles arbitrary overlap patterns
**Cons:** Can distort semantic clustering if over-applied

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

**Pros:** Visual indication of importance
**Cons:** Large nodes may increase overlap

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

## Recommended Implementation Order

1. **Angular Rebalancing** (Low effort, high impact)
   - Proportional angular allocation based on subtree size
   - Can be added to current layout pass

2. **Radial Jitter** (Medium effort, fixes remaining overlaps)
   - Post-processing pass after initial layout
   - Simple collision detection and resolution

3. **Force-Directed Refinement** (Higher effort, polish)
   - For cases where jitter isn't sufficient
   - Configurable via CLI flags

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
