# Hierarchical Smoothing Vision

**Status:** Proposed (Not Yet Tested)
**Date:** 2025-12-24
**Related:**
- [KERNEL_SMOOTHING_THEORY.md](KERNEL_SMOOTHING_THEORY.md)
- [CROSS_CLUSTER_SMOOTHING.md](CROSS_CLUSTER_SMOOTHING.md)
- [COUPLED_OPTIMIZATION_IMPROVEMENTS.md](COUPLED_OPTIMIZATION_IMPROVEMENTS.md)

## Executive Summary

This document describes an architectural vision for combining:
1. **Federation-based hierarchical clustering**
2. **Materialized path distance** for pruning
3. **Kernel smoothing** only where needed

The key insight: high-quality embeddings (e.g., ModernBERT) produce naturally orthogonal clusters for distinct topics, making cross-cluster smoothing unnecessary in most cases. Smoothing becomes relevant only for semantically overlapping clusters, which can be identified efficiently using the hierarchical structure.

**Status:** This is a theoretical framework based on initial experiments. It has not been validated at scale.

## Empirical Observations

### Experiment: ModernBERT on 20 Book-Based Clusters

```
Kernel condition number: ~1.0
Effective rank: 20/20 (all clusters independent)
cos(centroid_i, centroid_j) ≈ 0 for i ≠ j
```

**Result:** Cluster centroids are nearly orthogonal. Kernel smoothing degenerates to identity (no effect).

### Why This Happened

Our clusters are organized by book, each covering a distinct topic:
- Book on Django → distinct embedding region
- Book on Flask → different region
- Book on Asyncio → yet another region

ModernBERT's 768-dimensional space easily separates these topics.

### When This Would Break Down

Generic questions that span domains:
- "What is an array?" → applies to all programming languages
- "How do I define a function?" → universal concept
- "What is a decorator?" → Python-general, not book-specific

These would create either:
1. Large diffuse clusters (high variance, unstable projections)
2. Multiple nearby clusters (high inter-cluster similarity)

In case 2, kernel smoothing becomes valuable.

## Architectural Vision

### The Hierarchy

```
                        [Root]
                       /      \
              [Programming]    [Cooking]
              /     |    \          \
        [Python] [Java] [Rust]    [Italian]
        /   |   \
  [Django][Flask][NumPy]
```

Each node has a **materialized path**:
- `/programming/python/django`
- `/programming/python/flask`
- `/programming/java/spring`

### Materialized Path Distance

```python
def path_distance(path_a: str, path_b: str) -> int:
    """Count edges to common ancestor and back."""
    parts_a = path_a.strip('/').split('/')
    parts_b = path_b.strip('/').split('/')

    # Find common prefix length
    common = 0
    for a, b in zip(parts_a, parts_b):
        if a == b:
            common += 1
        else:
            break

    # Distance = steps up from A + steps down to B
    return (len(parts_a) - common) + (len(parts_b) - common)
```

Examples:
- `django ↔ flask`: distance = 2 (up to python, down to flask)
- `django ↔ spring`: distance = 4 (up to programming, down to java/spring)
- `django ↔ italian`: distance = 6 (up to root, down to cooking/italian)

### Smoothing Strategy

```python
def should_smooth(cluster_a, cluster_b, path_distance, kernel_value, threshold=0.1):
    """Decide if two clusters should be smoothed together."""

    # Fast rejection: too far in hierarchy
    if path_distance > 3:
        return False

    # Compute kernel only for nearby clusters
    if kernel_value < threshold:
        return False  # Orthogonal even though nearby

    return True  # Worth smoothing
```

### Computational Complexity

**Tree construction (one-time):**
- Agglomerative clustering: O(N² log N) or O(N log N) with optimizations
- If hierarchy is given (e.g., book/chapter structure): O(N)

**Smoothing pass:**

| Approach | Complexity | When to Use |
|----------|------------|-------------|
| Full kernel | O(N² × d) | Small N, unclear structure |
| Path-pruned | O(N log N + N × k) | Hierarchical data, k = avg siblings |
| No smoothing | O(N) | Well-separated clusters (ModernBERT + distinct topics) |

Where:
- N = number of clusters
- k = average nearby clusters in path (typically 5-10)
- d = embedding dimension
- O(N log N) = tree traversal to find neighbors

### Processing Pipeline

```
Query arrives
    │
    ▼
Route to cluster (standard centroid matching)
    │
    ▼
Check: Is cluster well-separated?
    │
    ├─ YES → Apply local W_i directly (fast path)
    │
    └─ NO → Find nearby clusters via materialized path
            │
            ▼
         Compute kernel only for neighbors
            │
            ▼
         Apply weighted smoothing
```

## What Needs Testing

### 1. Scale Validation
- Does orthogonality hold at 100, 1000, 10000 clusters?
- At what N do we start seeing significant cluster overlap?

### 2. Data Diversity
- Test with cross-cutting topics (generic programming questions)
- Test with overlapping domains (Python web frameworks)
- Test with hierarchical concepts (OOP → inheritance → Python inheritance)

### 3. Path Distance Correlation
- Do nearby clusters in the hierarchy actually have higher kernel values?
- What's the right path distance threshold for pruning?

### 4. Dynamic Thresholds
- Should the smoothing threshold adapt to local density?
- Can we learn the threshold from data?

### 5. Incremental Updates
- When a new cluster is added, which existing clusters need re-smoothing?
- Can we use path distance to limit the update scope?

## Proposed Experiments

### Experiment 1: Cluster Separation at Scale

```python
def measure_separation_by_scale(max_clusters_list=[20, 50, 100, 200]):
    for n in max_clusters_list:
        clusters = load_clusters(max_n=n)
        centroids = compute_centroids(clusters)
        K = compute_kernel_matrix(centroids)

        # Metrics
        off_diag = K[~np.eye(n, dtype=bool)]
        print(f"N={n}: mean_sim={off_diag.mean():.4f}, max_sim={off_diag.max():.4f}")
```

### Experiment 2: Path Distance vs Kernel Correlation

```python
def correlate_path_and_kernel(clusters_with_paths):
    path_distances = []
    kernel_values = []

    for (path_a, c_a), (path_b, c_b) in pairs(clusters_with_paths):
        path_distances.append(path_distance(path_a, path_b))
        kernel_values.append(kernel(c_a.centroid, c_b.centroid))

    correlation = np.corrcoef(path_distances, kernel_values)[0, 1]
    print(f"Correlation: {correlation:.3f}")
    # Expect: negative correlation (closer path → higher kernel)
```

### Experiment 3: Smoothing Value by Cluster Type

```python
def measure_smoothing_benefit(clusters):
    # Group by: well-separated vs overlapping
    for cluster in clusters:
        neighbors = find_path_neighbors(cluster, max_dist=2)
        neighbor_sims = [kernel(cluster, n) for n in neighbors]

        if max(neighbor_sims) < 0.1:
            group = "well_separated"
        else:
            group = "overlapping"

        # Compare MRR with/without smoothing for each group
```

## Summary

| Scenario | Smoothing Needed? | Complexity |
|----------|-------------------|------------|
| Distinct topics (books) | No | O(N) |
| Related subtopics (Flask/Django) | Maybe | O(N log N + N × k) |
| Generic concepts (arrays) | Yes | O(N log N + N × k) to O(N²) |

Note: Tree construction is a one-time O(N log N) to O(N²) cost, amortized over many queries.

The federation hierarchy and materialized paths give us a principled way to:
1. **Predict** where smoothing might help (path distance)
2. **Prune** unnecessary kernel computations
3. **Scale** to large N without O(N²) overhead

**Key insight:** The structure we built for federation naturally serves as a spatial index for smoothing decisions.

## References

- Federation implementation: `src/unifyweaver/targets/go_runtime/federation/`
- Kernel smoothing: `src/unifyweaver/targets/python_runtime/smoothing_basis.py`
- Materialized paths: Used in HNSW routing and cluster organization
