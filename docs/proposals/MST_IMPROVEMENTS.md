# MST Improvements for Hierarchy Construction

**Status:** Proposed
**Date:** 2026-01-13
**Prerequisites:** J-guided tree construction (implemented)

## Problem Statement

Greedy MST creates uneven hierarchies with long chains in dense regions:

| Method | Branching Factor | Max Depth |
|--------|-----------------|-----------|
| Greedy MST | 1.94 | 21 |
| J-Guided | 2.25 | 9 |

The average branching factor (~2) is reasonable, but some subtrees become long chains while others branch properly. This happens because MST optimizes *total edge weight*, not hierarchical quality.

## Proposed Improvements

### 1. Depth-Penalized MST

Modify edge weights during construction to discourage deep attachments:

```python
def depth_penalized_weight(dist, depth, lambda_depth=0.1):
    """
    Effective edge weight that penalizes depth.

    Args:
        dist: Original cosine distance
        depth: Depth of attachment point
        lambda_depth: Penalty coefficient

    Returns:
        Modified weight favoring shallower attachments
    """
    return dist * (1 + lambda_depth * depth)
```

**Rationale:** Making deep attachments more expensive encourages branching over chaining.

**Algorithm:**
1. Start with root (most central node)
2. For each unattached node, compute penalized weight to all attached nodes
3. Attach to minimum penalized weight (not just minimum distance)
4. Update depths

**Expected behavior:**
- Low λ (0.0): Equivalent to standard MST
- High λ (1.0): Strong preference for shallow trees
- Sweet spot: λ ~ 0.1-0.3 balances depth vs semantic coherence

### 2. Chain-Breaking Post-Process

After standard MST, detect and break long chains:

```python
def detect_chains(tree, threshold_factor=2.0):
    """
    Find paths longer than expected.

    Expected depth ≈ log_b(N) where b is branching factor.
    Flag paths > threshold_factor * expected_depth.
    """
    n_nodes = len(tree.nodes)
    expected_depth = math.log(n_nodes) / math.log(2)  # Binary assumption
    max_allowed = threshold_factor * expected_depth

    chains = []
    for leaf in tree.leaves:
        path = tree.path_to_root(leaf)
        if len(path) > max_allowed:
            chains.append(path)
    return chains

def break_chain(tree, chain, embeddings):
    """
    Reattach chain nodes to reduce depth.

    For each node in chain below threshold, find better parent
    among nodes at shallower depths.
    """
    threshold = len(chain) // 2
    for node in chain[threshold:]:
        candidates = [n for n in tree.nodes if tree.depth(n) < tree.depth(node) - 1]
        best = min(candidates, key=lambda c: cosine_dist(embeddings[node], embeddings[c]))
        tree.reattach(node, best)
```

**Rationale:** Fix MST's mistakes without changing the core algorithm.

### 3. Max-Depth Constraint

Hard constraint during MST construction:

```python
def constrained_mst(embeddings, max_depth=10):
    """
    Build MST with depth constraint.

    Only consider attachment points at depth < max_depth.
    """
    # ... standard Prim's algorithm, but filter candidates:
    valid_candidates = [n for n in attached if depth[n] < max_depth]
    if not valid_candidates:
        # All candidates too deep - attach to shallowest
        best = min(attached, key=lambda n: depth[n])
    else:
        best = min(valid_candidates, key=lambda n: cosine_dist(node, n))
```

**Trade-off:** May sacrifice some semantic coherence for depth control.

### 4. Hybrid: MST + J-Guided Refinement

Use MST for initial structure, then refine with J:

```python
def hybrid_construction(embeddings, texts):
    """
    Two-phase construction:
    1. MST for initial structure (fast)
    2. J-guided refinement (precise)
    """
    # Phase 1: Quick MST
    tree = build_mst(embeddings)

    # Phase 2: Refine attachments using J
    for node in tree.nodes_by_depth(reverse=True):  # Deepest first
        if tree.depth(node) > expected_depth(node):
            # Try reattaching to reduce J
            best_parent = min(
                tree.nodes,
                key=lambda p: compute_j(node, p, tree, embeddings, texts)
            )
            if compute_j(node, best_parent, ...) < compute_j(node, tree.parent(node), ...):
                tree.reattach(node, best_parent)
```

**Rationale:** MST is fast; J-guided refinement fixes problem areas.

## Comparison Matrix

| Method | Compute Cost | Preserves MST? | Depth Control | Semantic Quality |
|--------|--------------|----------------|---------------|------------------|
| Depth-Penalized | O(N²) | Partial | Soft | Good |
| Chain-Breaking | O(N) post | Yes | Hard | May degrade |
| Max-Depth | O(N²) | Partial | Hard | May degrade |
| MST + J Refine | O(N²) | Mostly | Soft | Best |

## Recommendation

**Short-term:** Implement depth-penalized MST (simplest, good results expected).

**Long-term:** Hybrid MST + J-guided refinement for best quality.

## Implementation Plan

### Phase 1: Depth-Penalized MST
- [ ] Add `depth_penalty` parameter to `mst_folder_grouping.py`
- [ ] Implement penalized weight computation
- [ ] Test on Wikipedia physics dataset
- [ ] Compare with J-guided baseline

### Phase 2: Chain-Breaking (if needed)
- [ ] Implement chain detection
- [ ] Implement reattachment heuristic
- [ ] Evaluate semantic quality impact

### Phase 3: Hybrid (if Phase 1 insufficient)
- [ ] Add J-guided refinement pass
- [ ] Profile performance impact
- [ ] Tune refinement criteria

## Success Criteria

1. Max depth reduced to ~log₂(N) (e.g., 9 for N=300)
2. Depth-surprisal correlation > 0.25
3. Branching factor > 2.0
4. Semantic coherence (D) not significantly degraded

## References

- [hierarchy_objective.py](../../scripts/mindmap/hierarchy_objective.py) - J-guided implementation
- [test_j_guided_tree.py](../../scripts/mindmap/test_j_guided_tree.py) - Benchmark comparison
- [ADAPTIVE_NODE_SUBDIVISION.md](ADAPTIVE_NODE_SUBDIVISION.md) - Related entropy-guided work
