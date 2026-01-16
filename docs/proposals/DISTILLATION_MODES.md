# Proposal: Distillation Mode Improvements

## Status
Proposal

## Context

The current `distill_federated_to_transformer.py` script assumes a single mode: distilling a federated model with K-means clustered W matrices using softmax routing.

This works for the general case but doesn't handle:
1. Per-tree W matrices (where each tree has its own projection)
2. Global W matrices (single projection for all data)
3. Different input data types (trees vs pearls)

### Federated Routing Flow

The correct query-based routing uses cluster centroids to select W matrices:

```
Query → sim(query, centroids) → softmax(sims / T) → Σ weight_i × (query @ W_i)
```

1. Compare query embedding to K cluster centroids
2. Apply temperature-scaled softmax to get cluster weights
3. Select top-k clusters by weight
4. Compute weighted sum of projections through each cluster's W matrix

This is more efficient than comparing to all N training queries (O(K) vs O(N) where K << N).

### Cluster Granularity Analysis

Testing on the pearltrees_federated_minilm model (700 clusters, T=0.1):

| Coverage | Required k |
|----------|-----------|
| 50% | 152 |
| 80% | 388 |
| 95% | 588 |

The centroid similarity range (0.18-0.64) indicates the 700 clusters may be over-segmented. Options:

1. **Meta-clustering**: Cluster the 700 centroids into 50-100 meta-clusters
2. **Lower K**: Retrain with fewer clusters (K=100-200)
3. **Lower temperature**: Use T=0.01 for more peaked routing
4. **Accept approximation**: Use fixed top-k=10 (captures ~7% of weight)

The current default (k=10) trades accuracy for speed. For better accuracy with reasonable speed, consider reducing cluster count or using hierarchical routing.

### Post-hoc Meta-Clustering

If centroids are too similar after initial training, meta-cluster them:

```bash
python3 scripts/metacluster_federated.py \
    models/federated_700.pkl \
    models/federated_100.pkl \
    --target-clusters 100 \
    --auto-suggest
```

**Algorithm:**

1. **Analyze**: Compute pairwise centroid similarity matrix
2. **Suggest K'**: Find optimal meta-cluster count where:
   - Target: top-1 softmax weight > 50% on average
   - Or: k ≤ 10 captures 80% of weight
3. **Meta-cluster**: Cluster the K centroids into K' groups
   - Methods: K-means, hierarchical, or circle method
4. **Merge W matrices**: For each meta-cluster, compute merged W:
   - Option A: Weighted average by cluster size
   - Option B: SVD-based low-rank combination
   - Option C: Train new W on combined data

**Suggestion heuristic:**

```python
def suggest_meta_clusters(centroids, temperature, target_top1_weight=0.5):
    """Suggest K' such that average top-1 softmax weight >= target."""
    # Binary search for K' where softmax is peaked enough
    for k_prime in [10, 25, 50, 100, 200]:
        meta_centroids = kmeans(centroids, k_prime)
        avg_top1 = mean([softmax_top1(q, meta_centroids, T) for q in test_queries])
        if avg_top1 >= target_top1_weight:
            return k_prime
    return len(centroids)  # No reduction possible
```

This allows iterative refinement: train with many clusters for accuracy, then consolidate for efficiency.

### Theoretical Basis for Cluster Count

**Question**: How many clusters K are needed for effective routing?

**Intrinsic dimensionality analysis** of the 700 centroids (384-dim embeddings):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Effective rank | 61 | Participation ratio of singular values |
| 80% variance | r=8 | Dimensions for 80% of variance |
| Stable rank | 1.5 | Data dominated by first singular value |
| √N heuristic | 27 | Common K-means rule of thumb |

**Theoretical connections:**

1. **Covering numbers**: For r-dimensional manifold, need O(ε^(-r)) balls. With r=8 (80% variance) and b=2 bins per dimension: K = 2^8 = 256.

2. **Effective rank**: The participation ratio Σs_i² / (Σs_i)² ≈ 61 suggests ~61 clusters capture the spectral structure.

3. **Product quantization** (FAISS): Splits space into m subspaces with k centroids each → k^m codes. For m=8, k=2: 256 codes.

4. **Rate-distortion**: Optimal quantization needs K ∝ 2^(r·R) for rate R. Lower R (coarser) → fewer clusters.

**Recommendation**:
- K = effective_rank ≈ 61-100 for this dataset
- Or K = 2^r where r captures 80% variance (K ≈ 256)
- Current K=700 is over-segmented → spread softmax weights

**Suggested heuristic:**
```python
def suggest_cluster_count(embeddings, target_variance=0.80):
    """Suggest K based on intrinsic dimensionality."""
    U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)

    # Effective rank
    eff_rank = (np.sum(S)**2) / np.sum(S**2)

    # Dimensions for target variance
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    r = np.searchsorted(cumvar, target_variance) + 1

    # Return range of suggestions
    return {
        'effective_rank': int(eff_rank),
        'power_of_2': 2 ** r,
        'sqrt_n': int(np.ceil(np.sqrt(len(embeddings)))),
        'recommended': max(int(eff_rank), 2 ** min(r, 8))
    }
```

## Current Architecture

```
Federated Model (K=700 clusters):
  Query → Softmax(query @ centroids / T) → Weighted blend of W_cluster projections

Distilled Transformer (H=4, L=5):
  Query → Transformer layers → Approximated projection
```

**Compression**: 103M params → 6.2M params (16.6x smaller)

## Proposed Modes

### 1. `--routing-mode` Options

| Mode | Description | W Count | Routing |
|------|-------------|---------|---------|
| `federated` | Current: K-means clusters | K | Softmax over centroids |
| `global` | Single W for all | 1 | None |
| `per-tree` | Each tree_id → W | N_trees | tree_id lookup |
| `hierarchical` | Tree path determines W | Variable | Path-based |

### 2. `--source-model-type` Options

Since different model formats exist in the codebase:

```python
# Global model format
{
    'type': 'global_procrustes',
    'W': np.array([D, D]),  # Single W matrix
    'scale': float,
    'target_embeddings': ...,
}

# Federated model format
{
    'cluster_ids': [...],
    'cluster_centroids': np.array([K, D]),
    'temperature': float,
    'cluster_dir': str,  # Points to {cluster_id}.npz files
}

# Per-tree model format (proposed)
{
    'type': 'per_tree',
    'tree_ids': [...],
    'W_dict': {tree_id: W_matrix},  # Or file references
}
```

### 3. `--distillation-strategy` Options

| Strategy | Description | Best For |
|----------|-------------|----------|
| `projection` | Learn input → projected output | All modes |
| `routing-only` | Learn input → cluster selection | Large federated |
| `two-stage` | First route, then project | Complex routing |

## Implementation Changes

### distill_federated_to_transformer.py

```python
parser.add_argument("--routing-mode",
    choices=["federated", "global", "per-tree", "hierarchical"],
    default="federated",
    help="How the source model routes queries to W matrices")

parser.add_argument("--source-type",
    choices=["federated", "global", "per-tree", "auto"],
    default="auto",
    help="Source model format (auto-detect if not specified)")

parser.add_argument("--tree-embed-mode",
    choices=["title", "path", "id-hash"],
    default="title",
    help="How to embed tree identity for per-tree routing")
```

### New Wrapper Classes

```python
class GlobalProjectionWrapper:
    """Wrapper for single-W global models."""
    def project(self, q_emb):
        return q_emb @ self.W

class PerTreeProjectionWrapper:
    """Wrapper for per-tree W models."""
    def project(self, q_emb, tree_id):
        W = self.W_dict[tree_id]
        return q_emb @ W

    def project_batch_with_trees(self, q_embs, tree_ids):
        """Batch project with tree_id routing."""
        return np.array([
            self.project(q, tid)
            for q, tid in zip(q_embs, tree_ids)
        ])

class HierarchicalProjectionWrapper:
    """Wrapper using tree path for routing."""
    def project(self, q_emb, tree_path):
        # Use path to determine which W(s) to use
        W = self._select_W_from_path(tree_path)
        return q_emb @ W
```

### Modified Training Loop

For per-tree mode, the transformer needs to take both query embedding and tree identity:

```python
if args.routing_mode == "per-tree":
    # Concatenate query embedding with tree embedding
    tree_emb = embed_tree_identity(tree_id, mode=args.tree_embed_mode)
    combined_input = np.concatenate([q_emb, tree_emb])
    # Or use tree_emb as conditioning signal
```

## Memory Considerations

| Mode | W Storage | Browser Viable |
|------|-----------|----------------|
| Global | D² = 147KB | Yes |
| Federated K=700 | K×D² = 103MB | Via transformer only |
| Per-tree N=38K | N×D² = 5.6GB | Via transformer only |
| Per-pearl | Huge | Via transformer only |

The transformer distillation is essential for browser deployment of per-tree and per-pearl models.

## Recommended Default Workflow

1. **Small data (<1000 items)**: Global W, no distillation needed
2. **Medium data (1K-10K trees)**: Per-tree W or clustered, distill to transformer
3. **Large data (>10K items)**: Federated clustering (K=100-1000), distill to transformer
4. **With pearls**: Always cluster, federated approach essential

## Testing Strategy

```bash
# Test global distillation
python3 scripts/distill_federated_to_transformer.py \
    models/pearltrees_proj_physics_global.pkl \
    models/transformer_global.pt \
    --source-type global

# Test per-tree distillation
python3 scripts/distill_federated_to_transformer.py \
    models/per_tree_model.pkl \
    models/transformer_pertree.pt \
    --routing-mode per-tree \
    --tree-embed-mode title

# Compare results
python3 scripts/compare_distillation_modes.py \
    models/transformer_*.pt \
    --test-data datasets/test_queries.jsonl
```

## Success Metrics

- **Cosine similarity**: Distilled vs original projection ≥ 0.90
- **MSE**: Per-sample error < 0.01
- **Compression ratio**: ≥ 10x for federated, ≥ 100x for per-tree
- **Inference speed**: Transformer faster than original routing

## Related Work

- `scripts/train_pearltrees_global.py` - Already has `--mode` for global/per-cluster/per-query
- `scripts/train_pearltrees_federated.py` - Federated training with K-means
- `src/.../projection_transformer.py` - Transformer architecture

## Next Steps

1. Add `--source-type auto` detection based on model keys
2. Implement `GlobalProjectionWrapper` for simpler distillation
3. Add `--routing-mode per-tree` with tree identity embedding
4. Create comparison benchmark script
