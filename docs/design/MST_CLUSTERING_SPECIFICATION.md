# Specification: MST-Based Clustering for Federated Training

## Overview

This document specifies the technical design for MST-based clustering in the federated projection training pipeline.

## Algorithm

### Input
- `A_emb`: Target embeddings matrix (N x D), where N = number of items, D = embedding dimension
- `max_clusters`: Maximum number of clusters to create
- `min_cluster_size`: Minimum items per cluster (optional, default 10)
- `max_cluster_size`: Maximum items per cluster (optional, default 1000)

### Output
- `clusters`: Dict[str, List[int]] mapping cluster_id to list of item indices

### Steps

```
1. COMPUTE pairwise cosine distances between all embeddings
   distances = pdist(A_emb, metric='cosine')
   distance_matrix = squareform(distances)

2. BUILD minimum spanning tree
   mst = minimum_spanning_tree(distance_matrix)

3. IDENTIFY edges to cut
   - Sort MST edges by weight (distance) descending
   - Select top (max_clusters - 1) edges to cut

4. FORM clusters from connected components
   - Remove selected edges from MST
   - Find connected components
   - Each component becomes a cluster

5. POST-PROCESS for size constraints
   - Merge clusters smaller than min_cluster_size with nearest neighbor
   - Split clusters larger than max_cluster_size using sub-MST

6. RETURN cluster assignments
```

### Pseudocode

```python
def cluster_by_mst(
    A_emb: np.ndarray,
    max_clusters: int = 50,
    min_cluster_size: int = 10,
    max_cluster_size: int = 1000
) -> Dict[str, List[int]]:
    """
    Cluster targets by MST edge cutting.

    Builds MST on cosine distances, cuts longest edges to form clusters.
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
    from scipy.sparse import csr_matrix

    n = len(A_emb)

    # Step 1: Compute pairwise cosine distances
    distances = squareform(pdist(A_emb, metric='cosine'))

    # Step 2: Build MST
    mst = minimum_spanning_tree(csr_matrix(distances))
    mst_dense = mst.toarray()

    # Make symmetric for edge extraction
    mst_symmetric = mst_dense + mst_dense.T

    # Step 3: Find edges and sort by weight
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if mst_symmetric[i, j] > 0:
                edges.append((i, j, mst_symmetric[i, j]))
    edges.sort(key=lambda x: x[2], reverse=True)  # Longest first

    # Step 4: Cut top (k-1) edges to form k clusters
    num_cuts = min(max_clusters - 1, len(edges))
    edges_to_remove = edges[:num_cuts]

    # Build adjacency matrix without cut edges
    adj = mst_symmetric.copy()
    for i, j, _ in edges_to_remove:
        adj[i, j] = 0
        adj[j, i] = 0

    # Find connected components
    n_components, labels = connected_components(csr_matrix(adj), directed=False)

    # Step 5: Group indices by component
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[f"mst_cluster_{label}"].append(idx)

    # Step 6: Post-process for size constraints
    clusters = _enforce_size_constraints(clusters, A_emb, min_cluster_size, max_cluster_size)

    return dict(clusters)
```

## Size Constraint Enforcement

### Merging Small Clusters

```python
def _merge_small_clusters(clusters, A_emb, min_size):
    """Merge clusters smaller than min_size with nearest neighbor."""
    centroids = {cid: A_emb[indices].mean(axis=0) for cid, indices in clusters.items()}

    while True:
        small = [(cid, len(idx)) for cid, idx in clusters.items() if len(idx) < min_size]
        if not small:
            break

        # Find smallest cluster
        smallest_cid = min(small, key=lambda x: x[1])[0]
        smallest_centroid = centroids[smallest_cid]

        # Find nearest neighbor cluster
        best_dist = float('inf')
        best_neighbor = None
        for cid, centroid in centroids.items():
            if cid != smallest_cid:
                dist = 1 - np.dot(smallest_centroid, centroid) / (
                    np.linalg.norm(smallest_centroid) * np.linalg.norm(centroid))
                if dist < best_dist:
                    best_dist = dist
                    best_neighbor = cid

        # Merge
        clusters[best_neighbor].extend(clusters[smallest_cid])
        del clusters[smallest_cid]
        centroids[best_neighbor] = A_emb[clusters[best_neighbor]].mean(axis=0)
        del centroids[smallest_cid]

    return clusters
```

### Splitting Large Clusters

```python
def _split_large_clusters(clusters, A_emb, max_size):
    """Split clusters larger than max_size using recursive MST."""
    result = {}
    split_idx = 0

    for cid, indices in clusters.items():
        if len(indices) <= max_size:
            result[cid] = indices
        else:
            # Recursively apply MST clustering to this cluster
            sub_emb = A_emb[indices]
            num_subclusters = (len(indices) // max_size) + 1
            sub_clusters = cluster_by_mst(sub_emb, max_clusters=num_subclusters,
                                          min_cluster_size=1, max_cluster_size=max_size)
            for sub_cid, sub_indices in sub_clusters.items():
                # Map back to global indices
                global_indices = [indices[i] for i in sub_indices]
                result[f"{cid}_split_{split_idx}"] = global_indices
                split_idx += 1

    return result
```

## Integration Points

### train_pearltrees_federated.py

1. Add import statements:
```python
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix
```

2. Add `cluster_by_mst()` function after existing clustering functions

3. Update `--cluster-method` argument:
```python
parser.add_argument("--cluster-method",
    choices=["path_depth", "embedding", "per-tree", "mst"],
    default="embedding",  # Keep existing default initially
    help="Clustering method: 'embedding' (K-means), 'mst' (MST-based), ...")
```

4. Update cluster selection in `train_federated()`:
```python
if cluster_method == "mst":
    clusters = cluster_by_mst(A_emb, max_clusters, min_cluster_size=10, max_cluster_size=max_items)
```

## Memory Considerations

- Distance matrix: N x N x 8 bytes (float64)
  - For 13,000 items: ~1.3 GB
  - May need chunked computation for larger datasets

- MST: N-1 edges, sparse representation
  - Negligible memory

- Mitigation for large datasets:
  - Sample embeddings for initial clustering
  - Assign remaining items to nearest cluster

## Testing Strategy

1. **Unit Tests**
   - Test MST construction correctness
   - Test edge cutting produces expected number of clusters
   - Test size constraint enforcement

2. **Integration Tests**
   - Train model with MST clustering
   - Verify model loads and infers correctly

3. **Benchmark Comparison**
   - Compare recall@k metrics: MST vs K-means
   - Compare training time
   - Compare cluster size distribution

## Rollout Plan

1. Implement as optional `--cluster-method mst`
2. Test on current dataset
3. Compare metrics with K-means
4. If better, make MST the default
