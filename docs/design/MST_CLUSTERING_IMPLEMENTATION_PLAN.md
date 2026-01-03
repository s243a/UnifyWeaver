# Implementation Plan: MST-Based Clustering

## Overview

Step-by-step implementation plan for adding MST-based clustering to the federated projection training pipeline.

## Prerequisites

- scipy (already a dependency)
- numpy (already a dependency)

## Implementation Steps

### Step 1: Add Helper Functions

**File:** `scripts/train_pearltrees_federated.py`

Add after existing imports:
```python
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix
```

### Step 2: Implement `cluster_by_mst()`

Add after `cluster_by_embedding()` function:

```python
def cluster_by_mst(
    A_emb: np.ndarray,
    max_clusters: int = 50,
    min_cluster_size: int = 10,
    max_cluster_size: int = 1000
) -> Dict[str, List[int]]:
    """
    Cluster targets by MST edge cutting.

    Builds minimum spanning tree on cosine distances between embeddings,
    then cuts the longest edges to form clusters. This preserves local
    neighborhood relationships better than K-means.

    Args:
        A_emb: Target embeddings (N x D)
        max_clusters: Maximum number of clusters to create
        min_cluster_size: Merge clusters smaller than this
        max_cluster_size: Split clusters larger than this

    Returns:
        Dict mapping cluster_id to list of item indices
    """
    n = len(A_emb)

    if n <= max_clusters:
        # Each item is its own cluster
        return {f"mst_{i}": [i] for i in range(n)}

    logger.info(f"Building MST for {n} items...")

    # Compute pairwise cosine distances
    # For large n, this could be memory-intensive
    if n > 10000:
        logger.warning(f"Large dataset ({n} items), distance matrix will use ~{n*n*8/1e9:.1f} GB")

    distances = squareform(pdist(A_emb, metric='cosine'))

    # Build MST
    mst = minimum_spanning_tree(csr_matrix(distances))

    # Extract edges with weights
    mst_coo = mst.tocoo()
    edges = list(zip(mst_coo.row, mst_coo.col, mst_coo.data))
    edges.sort(key=lambda x: x[2], reverse=True)  # Sort by weight descending

    logger.info(f"MST has {len(edges)} edges, cutting {min(max_clusters-1, len(edges))} longest...")

    # Cut top (k-1) edges to form k clusters
    num_cuts = min(max_clusters - 1, len(edges))

    # Build adjacency without cut edges
    adj = mst.toarray()
    adj = adj + adj.T  # Make symmetric

    for i in range(num_cuts):
        r, c, _ = edges[i]
        adj[r, c] = 0
        adj[c, r] = 0

    # Find connected components
    n_components, labels = connected_components(csr_matrix(adj), directed=False)

    logger.info(f"Created {n_components} initial clusters")

    # Group indices by component
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[f"mst_{label}"].append(idx)

    # Log size distribution
    sizes = [len(v) for v in clusters.values()]
    logger.info(f"  Size range: {min(sizes)}-{max(sizes)}, Avg: {np.mean(sizes):.1f}")

    # Enforce size constraints
    clusters = _mst_merge_small(clusters, A_emb, min_cluster_size)
    clusters = _mst_split_large(clusters, A_emb, max_cluster_size)

    return dict(clusters)


def _mst_merge_small(
    clusters: Dict[str, List[int]],
    A_emb: np.ndarray,
    min_size: int
) -> Dict[str, List[int]]:
    """Merge clusters smaller than min_size with nearest neighbor."""
    if min_size <= 1:
        return clusters

    clusters = dict(clusters)  # Copy

    # Compute centroids
    centroids = {cid: A_emb[indices].mean(axis=0) for cid, indices in clusters.items()}

    merged_count = 0
    while True:
        # Find smallest cluster below threshold
        small = [(cid, len(idx)) for cid, idx in clusters.items() if len(idx) < min_size]
        if not small:
            break

        smallest_cid = min(small, key=lambda x: x[1])[0]
        smallest_centroid = centroids[smallest_cid]

        # Find nearest neighbor cluster (by cosine similarity)
        best_sim = -1
        best_neighbor = None
        for cid, centroid in centroids.items():
            if cid != smallest_cid:
                sim = np.dot(smallest_centroid, centroid) / (
                    np.linalg.norm(smallest_centroid) * np.linalg.norm(centroid) + 1e-10)
                if sim > best_sim:
                    best_sim = sim
                    best_neighbor = cid

        if best_neighbor is None:
            break

        # Merge
        clusters[best_neighbor].extend(clusters[smallest_cid])
        del clusters[smallest_cid]
        centroids[best_neighbor] = A_emb[clusters[best_neighbor]].mean(axis=0)
        del centroids[smallest_cid]
        merged_count += 1

    if merged_count > 0:
        logger.info(f"  Merged {merged_count} small clusters")

    return clusters


def _mst_split_large(
    clusters: Dict[str, List[int]],
    A_emb: np.ndarray,
    max_size: int
) -> Dict[str, List[int]]:
    """Split clusters larger than max_size."""
    result = {}
    split_count = 0

    for cid, indices in clusters.items():
        if len(indices) <= max_size:
            result[cid] = indices
        else:
            # Split into chunks (simple approach)
            # Could use recursive MST for better splits
            n_splits = (len(indices) + max_size - 1) // max_size
            chunk_size = len(indices) // n_splits

            for i in range(n_splits):
                start = i * chunk_size
                end = start + chunk_size if i < n_splits - 1 else len(indices)
                result[f"{cid}_s{i}"] = indices[start:end]
                split_count += 1

    if split_count > len(clusters):
        logger.info(f"  Split into {len(result)} clusters (was {len(clusters)})")

    return result
```

### Step 3: Update Argument Parser

In `main()`:

```python
parser.add_argument("--cluster-method",
    choices=["path_depth", "embedding", "per-tree", "mst"],
    default="embedding",
    help="Clustering method: 'embedding' (K-means), 'mst' (MST-based), "
         "'path_depth', or 'per-tree'")
```

### Step 4: Update Cluster Selection

In `train_federated()`:

```python
# Cluster the data
if cluster_method == "path_depth":
    clusters = cluster_by_path_depth(data, max_clusters)
elif cluster_method == "per-tree":
    clusters = cluster_by_tree(data)
elif cluster_method == "mst":
    clusters = cluster_by_mst(A_emb, max_clusters, min_cluster_size=10, max_cluster_size=max_items)
else:  # "embedding" (default)
    clusters = cluster_by_embedding(A_emb, max_clusters, max_items)
```

### Step 5: Add CLI Arguments for Size Constraints

```python
parser.add_argument("--min-cluster-size", type=int, default=10,
    help="Minimum cluster size (for MST method)")
parser.add_argument("--max-cluster-size", type=int, default=1000,
    help="Maximum cluster size")
```

## Testing Commands

### Train with MST clustering:
```bash
python3 scripts/train_pearltrees_federated.py \
  --cluster-method mst \
  --max-clusters 100 \
  reports/pearltrees_targets_full_multi_account.jsonl \
  models/pearltrees_federated_mst_100.pkl
```

### Compare with K-means:
```bash
# K-means baseline
python3 scripts/train_pearltrees_federated.py \
  --cluster-method embedding \
  --max-clusters 100 \
  reports/pearltrees_targets_full_multi_account.jsonl \
  models/pearltrees_federated_kmeans_100.pkl
```

### Benchmark both:
```bash
# Test inference with each model and compare recall@k
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated_mst_100.pkl \
  --query "test query" --top-k 10

python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated_kmeans_100.pkl \
  --query "test query" --top-k 10
```

## Verification Checklist

- [ ] `cluster_by_mst()` function implemented
- [ ] Size constraint helpers implemented
- [ ] CLI argument added
- [ ] Cluster selection updated
- [ ] Model trains successfully
- [ ] Model infers correctly
- [ ] Metrics comparable to K-means
- [ ] Documentation updated

## Files Modified

1. `scripts/train_pearltrees_federated.py` - Main implementation
2. `docs/design/MST_CLUSTERING_*.md` - Design documents

## Future Enhancements

1. **Hierarchical MST**: Use MST to build multi-level hierarchy
2. **Adaptive Cuts**: Cut based on edge weight threshold, not count
3. **Hybrid Approach**: Use MST for initial structure, K-means for refinement
4. **Memory Optimization**: Chunked distance computation for large datasets
