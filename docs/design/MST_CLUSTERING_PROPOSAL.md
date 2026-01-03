# Proposal: MST-Based Clustering for Federated Projection Training

## Summary

Replace K-means clustering with Minimum Spanning Tree (MST) based clustering in the federated projection training pipeline. This approach better preserves the natural neighborhood structure of the embedding space, which is more appropriate for hierarchical folder systems like Pearltrees.

## Motivation

### Current Approach: K-means

The current `cluster_by_embedding()` function uses K-means clustering:
- Creates spherical clusters around centroids
- Assigns points to nearest centroid
- Does not preserve local neighborhood relationships
- Can split natural clusters that span non-spherical regions

### Proposed Approach: MST-based Clustering

MST-based clustering:
- Builds a minimum spanning tree connecting all points by their nearest neighbors
- Cuts the longest edges to form clusters
- Preserves local neighborhood relationships
- Creates clusters that follow the natural structure of the embedding space
- More appropriate for hierarchical data like folder trees

### Why MST is Better for Pearltrees

1. **Hierarchical Nature**: Pearltrees folders form a tree structure; MST clustering respects tree-like relationships in embedding space

2. **Semantic Neighborhoods**: Related folders (siblings, parent-child) tend to be close in embedding space; MST keeps them together

3. **Non-spherical Clusters**: Folder categories may form elongated or irregular shapes; MST handles these better than K-means

4. **Consistent with Existing Code**: MST is already used in `generate_simplemind_map.py` for organizing output folders

## Scope

### In Scope
- Add `cluster_by_mst()` function to `train_pearltrees_federated.py`
- Add `mst` option to `--cluster-method` argument
- Keep K-means (`embedding`) as default initially
- Document the new option
- Benchmark comparison with K-means

### Out of Scope
- Changes to inference pipeline
- Changes to other clustering methods (path_depth, per-tree)
- Automatic selection between methods

## Success Criteria

1. MST clustering produces comparable or better recall@k metrics
2. Training time remains reasonable (< 2x K-means)
3. Cluster sizes are balanced (configurable min/max)
4. No increase in model size or inference time

## Risks

1. **Computational Cost**: MST construction is O(n² log n) vs O(n k t) for K-means
   - Mitigation: Use efficient scipy implementation, sample for large datasets

2. **Unbalanced Clusters**: MST cuts may produce very small or large clusters
   - Mitigation: Post-process to merge small clusters, split large ones

3. **Edge Cases**: Single long chain instead of clusters
   - Mitigation: Use multiple edge cuts, not just longest

## Timeline

Not specifying timeline - implementation steps only.

## References

- scipy.sparse.csgraph.minimum_spanning_tree
- Existing usage in `scripts/generate_simplemind_map.py`
- Current K-means implementation in `train_pearltrees_federated.py`
