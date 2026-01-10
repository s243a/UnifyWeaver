# Proposal: Memory-Reduced Embedding Projections

## Problem Statement

Current embedding-based routing loads all embeddings into memory:
- O(N × dim) memory for N items
- Causes OOM on memory-constrained systems
- Unnecessary when only a subset is needed for routing decisions

## Proposed Solution: Hierarchical Sparse Routing

### Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Query Embedding                                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Cluster Selection                                 │
│  Route over centroids → select top-K clusters               │
│  Memory: O(K_clusters × dim)                                │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Load W Matrices                                   │
│  Load projection matrices for selected clusters only        │
│  Memory: O(K × dim × dim)                                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Within-Cluster Routing                            │
│  Route over cluster representatives (configurable)          │
│  Memory: O(representatives × dim)                           │
└─────────────────────────────────────────────────────────────┘
```

### Configurable Routing Granularity

Stage 3 routing can operate at different granularity levels:

| Level | Items per Cluster | Memory | Accuracy |
|-------|-------------------|--------|----------|
| `centroid` | 1 | Minimal | Coarse |
| `svd` | k singular vectors | Small, fixed | Good |
| `trees` | N_trees | Medium | Good |
| `dispersed` | k sampled items | Bounded | Good |
| `trees+pearls` | All items | High | Best |

**Recommendation**: `svd` or `trees` as default, with fallback to `centroid` for empty clusters.

### SVD-Based Cluster Representation

Instead of storing individual embeddings, represent each cluster by its principal components:

```python
def compute_cluster_svd(cluster_embeddings, k=10):
    """Compress cluster to top-k singular vectors."""
    U, S, Vt = np.linalg.svd(cluster_embeddings, full_matrices=False)
    return Vt[:k], S[:k]  # (k, dim), (k,)
```

**Storage per cluster**:
- Vt: k × dim floats
- S: k floats
- Total: k × (dim + 1) ≈ 4KB for k=10, dim=384

**Benefits**:
- Fixed size regardless of cluster membership
- Captures maximum variance directions
- Mathematically principled compression

### Random Superposition Sampling

For routing, generate random representatives from the SVD:

```python
def sample_cluster_direction(Vt, S):
    """Generate random vector in cluster's principal subspace."""
    # Sample mixing coefficients proportional to singular values
    # Dirichlet concentrates mass on larger singular values
    alpha = S + 1e-6  # Avoid zero
    weights = np.random.dirichlet(alpha)

    # Superposition: weighted sum of singular vectors
    return weights @ Vt  # (dim,)
```

**Why not softmax?**
- Softmax's exponential over-weights the largest singular value
- Dirichlet sampling preserves diversity while respecting importance
- Each sample explores a different region of the cluster subspace

**Routing with multiple samples**:

```python
def cluster_affinity(query_emb, Vt, S, n_samples=5):
    """Monte Carlo estimate of query-cluster affinity."""
    scores = []
    for _ in range(n_samples):
        rep = sample_cluster_direction(Vt, S)
        scores.append(cosine_similarity(query_emb, rep))
    return np.mean(scores)
```

## Memory Comparison

For a collection with N=100,000 items, 50 clusters, dim=384:

| Approach | Memory |
|----------|--------|
| Load all embeddings | 100K × 384 × 4 = **150 MB** |
| Centroids only | 50 × 384 × 4 = **75 KB** |
| SVD (k=10) | 50 × 10 × 385 × 4 = **770 KB** |
| Trees only (~5K) | 5K × 384 × 4 = **7.5 MB** |

SVD approach uses **0.5%** of full embedding memory while capturing cluster structure.

## Application to Transformer Compression Training

The random superposition sampling is ideal for training compression transformers:

### Training Data Generation

```python
def generate_training_batch(clusters_svd, batch_size):
    """Generate diverse training samples from cluster subspaces."""
    samples = []
    labels = []

    for cluster_id, (Vt, S) in clusters_svd.items():
        n_samples = batch_size // len(clusters_svd)
        for _ in range(n_samples):
            # Random superposition weighted by singular values
            weights = np.random.dirichlet(S)
            sample = weights @ Vt
            samples.append(sample)
            labels.append(cluster_id)

    return np.array(samples), np.array(labels)
```

### Benefits for Training

1. **Infinite data augmentation**: Generate unlimited novel samples from finite clusters
2. **Subspace coverage**: Training samples span the full cluster geometry
3. **Importance curriculum**: Larger singular values → more training weight
4. **No overfitting to centroids**: Model learns subspace structure, not memorized points
5. **Cheap generation**: Just Dirichlet sampling + matrix multiply

### Training Loop

```python
# Precompute SVD for all clusters (one-time cost)
clusters_svd = {
    cluster_id: compute_cluster_svd(embeddings[cluster_mask])
    for cluster_id, cluster_mask in cluster_assignments.items()
}

# Training
for epoch in range(n_epochs):
    # Generate fresh samples each epoch
    batch_emb, batch_labels = generate_training_batch(clusters_svd, 256)

    # Train transformer to predict cluster / compress embedding
    compressed = transformer(batch_emb)
    loss = compression_loss(compressed, batch_labels)
    loss.backward()
```

## Implementation Plan

### Phase 1: SVD Cluster Storage

1. Compute SVD for each cluster during indexing
2. Store in SQLite: `clusters(id, Vt BLOB, S BLOB)`
3. Load on-demand during routing

### Phase 2: Configurable Routing

Add CLI options to batch_repair.py:

```bash
python batch_repair.py \
  --route-level svd \        # centroid|svd|trees|dispersed|all
  --svd-k 10 \               # Number of singular vectors
  --svd-samples 5            # Monte Carlo samples for affinity
```

### Phase 3: Transformer Training Integration

1. Use SVD sampling for training data generation
2. Compare model quality vs. training on actual embeddings
3. Measure generalization to unseen items

## Configuration Options

```python
@dataclass
class RoutingConfig:
    # Cluster selection
    top_k_clusters: int = 5

    # Within-cluster routing
    route_level: str = 'svd'  # centroid|svd|trees|dispersed|all

    # SVD options
    svd_k: int = 10           # Singular vectors to keep
    svd_samples: int = 5      # Monte Carlo samples

    # Fallback
    fallback_to_centroid: bool = True  # If cluster has no trees
```

## Precomputed Representatives (Optional Optimization)

For maximum query speed, precompute and store random superpositions during indexing:

```python
def count_dominant_singular_values(S, max_condition=10):
    """Count singular values within condition number threshold.

    Keep only σ_i where σ_max/σ_i ≤ max_condition.

    We use max_condition=10 (not 100) for two reasons:
    1. Memory efficiency: Tighter threshold → smaller subspace → fewer reps needed
    2. Excellent conditioning: κ ≤ 10 is well-conditioned numerically

    While κ ≤ 100 is acceptable from a pure numerical stability standpoint,
    it can result in a much larger subspace (5-10 dimensions vs 2-4),
    significantly increasing storage requirements.

    Example with S = [1.0, 0.5, 0.2, 0.08, 0.01]:
    - max_condition=10:  keep σ ≥ 0.1  → 3 dominant (1.0, 0.5, 0.2)
    - max_condition=100: keep σ ≥ 0.01 → 5 dominant (all)

    Note: Theoretically, if all singular values were tightly clustered
    (e.g., [1.0, 0.9, 0.8, ...]), many would pass the κ ≤ 10 threshold.
    However, this is rare because a cluster by definition has structure -
    items share common directions, which means dominant singular values.
    If all singular values were equal (no dominant directions), you'd have
    a spherical blob, not a cluster. Singular value decay is definitional
    to clustering, not just empirical.
    """
    threshold = S[0] / max_condition
    return np.sum(S >= threshold)


def compute_num_representatives(S, max_condition=10):
    """Determine how many reps needed for good conditioning.

    Need at least as many representatives as dominant singular vectors
    to adequately span the cluster's principal subspace.
    """
    k_dominant = count_dominant_singular_values(S, max_condition)

    # Need at least k_dominant reps to span the subspace
    # Use 2x for better coverage
    return max(k_dominant, k_dominant * 2)


def precompute_cluster_representatives(cluster_embeddings, svd_k=10, max_condition=10):
    """Precompute random superpositions for fast loading.

    Number of representatives is adaptive based on cluster complexity:
    - Simple cluster (1-2 dominant directions) -> fewer reps
    - Complex cluster (many dominant directions) -> more reps
    """
    U, S, Vt = np.linalg.svd(cluster_embeddings, full_matrices=False)
    Vt, S = Vt[:svd_k], S[:svd_k]

    # Adaptive: need enough reps for good conditioning (κ ≤ 10)
    n_reps = compute_num_representatives(S, max_condition)

    # Generate random superpositions
    representatives = []
    for _ in range(n_reps):
        weights = np.random.dirichlet(S)
        rep = weights @ Vt
        representatives.append(rep)

    return np.array(representatives), Vt, S  # (n_reps, dim), (k, dim), (k,)
```

**Conditioning requirement**: The number of precomputed superposition vectors should be at least equal to the number of dominant singular vectors. If there are k dominant directions (capturing 90% of variance), you need ≥k representatives to adequately span that subspace. Using 2x provides better coverage.

**Storage format** (SQLite):
```sql
CREATE TABLE cluster_reps (
    cluster_id INTEGER,
    rep_index INTEGER,
    embedding BLOB,  -- (dim,) float32
    PRIMARY KEY (cluster_id, rep_index)
);
```

**Trade-offs**:

| Approach | Storage | Query Time | Flexibility |
|----------|---------|------------|-------------|
| Store SVD (Vt, S) | ~4 KB/cluster | Medium (sample + matmul) | Can generate unlimited samples |
| Store precomputed reps | ~6-30 KB/cluster (adaptive) | Fast (just load) | Fixed set, well-conditioned |
| Store both | ~10-34 KB/cluster | Fast | Best of both |

**Adaptive storage**: Simple clusters (1-2 dominant directions) need fewer reps (~6 KB), complex clusters (5+ dominant directions) need more (~30 KB). This naturally allocates storage where it's needed.

Recommendation: Store both SVD and precomputed reps. Use precomputed for fast queries, SVD for training data generation.

## Open Questions

1. Optimal k for SVD compression (10? 20? adaptive per cluster?)
2. Best Dirichlet concentration parameter (use S directly, or S^α?)
3. How many Monte Carlo samples needed for stable routing?
4. Does SVD-trained transformer generalize better than centroid-trained?

## Related Work

- Mixture of Experts (sparse routing)
- Product Quantization (embedding compression)
- Random Projections (Johnson-Lindenstrauss)
- Variational Autoencoders (learned subspace sampling)

## Next Steps

1. Implement SVD computation during cluster indexing
2. Add routing level configuration to batch_repair.py
3. Benchmark memory usage across routing levels
4. Integrate with transformer compression training pipeline
