# Proposal: Custom Multi-Space Distance Blending

## Context

The density explorer's Blend tab supports input↔output embedding space blending
(modes: None, Visualization, Tree Distance, Both). A fifth mode, "Custom," was
deferred because it involves blending across distance spaces with different
dimensionalities and semantics.

### Observation: Information-Preserving Models

The Bivector Paired model is designed to preserve input embedding geometry (good
hit@k with minimal training). Testing shows this produces nearly identical
distance orderings in input vs output space — only 8/299 MST edges differ. The
input↔output blend is therefore a near-identity operation for tree distances.

This means the existing input↔output blend can be collapsed into a single
"embedding distance" term in a larger weighted sum, rather than occupying two
separate slots.

## Design

### Distance-Level Blending

All available distance spaces can be reduced to N×N distance matrices regardless
of their native dimensionality. Blending at the distance level sidesteps the
dimensionality mismatch problem entirely:

```
d_custom = w_emb * d_emb(α) + w_weights * d_weights + w_learned * d_learned + w_wiki * d_wiki
```

Where:
- `d_emb(α)` = blended input↔output embedding distance (using existing α slider)
- `d_weights` = cosine distance in weight space (64D)
- `d_learned` = Euclidean distance in learned organizational metric space (64D)
- `d_wiki` = predicted distance from Wikipedia physics distance model

Weights `w_*` are normalized to sum to 1.

### Available Distance Spaces

| Space | Dim | Distance | Semantics |
|-------|-----|----------|-----------|
| Embedding (input↔output blend) | 768D | Cosine | Semantic similarity |
| Weight Space | 64D | Cosine | Transformation recipe similarity |
| Learned Metric | 64D | Euclidean | Organizational proximity |
| Wikipedia Physics | N/A (pairwise) | Predicted | Category hierarchy distance |

### Why Linear Distance Blending Works

Linear interpolation of distance matrices preserves the metric properties needed
by tree algorithms:
- **Non-negativity**: Sum of non-negative distances is non-negative
- **Zero diagonal**: Sum of zero-diagonal matrices has zero diagonal
- **Symmetry**: Sum of symmetric matrices is symmetric
- Triangle inequality is not guaranteed but MST/tree algorithms don't require it

### Collapsing Input↔Output Into One Term

Since the Bivector model preserves input geometry, `d_emb(α)` for any α produces
nearly the same distance matrix. Rather than wasting two weight slots on input
and output separately:

1. The existing α slider controls the internal input↔output mix
2. The result gets one weight `w_emb` in the custom sum
3. Users who want to explore the (small) input↔output difference can still use α
4. The significant blending happens between `d_emb` and the other spaces

## UI Design

### Custom Mode Controls

When "Custom" is selected in the Blend tab:

```
Embedding Distance     [====|====] 0.40    α: [==|=] 0.50
Weight Space Distance  [==|======] 0.20
Learned Metric         [===|=====] 0.30
Wikipedia Physics      [=|=======] 0.10
                                   ----
                            Total: 1.00    [Normalize] [Apply]
```

- Each space gets a weight slider (0.0–1.0) and numeric input
- "Normalize" button rescales weights to sum to 1.0
- Embedding Distance includes a nested α sub-slider for input↔output mix
- Spaces that require a model (Weights, Learned) are disabled when no model is
  selected, similar to existing 2D Projection Mode behavior
- Wikipedia Physics is disabled when that distance model isn't available

### Scope: Tree Distance Only

Custom blending applies to **tree distance computation only**, not 2D layout.
Rationale:
- 2D layout blending requires projecting each space to 2D independently, then
  blending coordinates. With 4+ spaces this creates too many SVD projections
  and the blended 2D result loses interpretability.
- Tree distances are NxN matrices that blend cleanly with weighted sums.
- The existing Visualization blend (input↔output) remains available for layout.

### Presets

Common configurations as one-click presets:
- **Semantic Only**: w_emb=1.0, rest=0 (baseline)
- **Hierarchical**: w_emb=0.3, w_weights=0.3, w_learned=0.4
- **Wikipedia-Guided**: w_emb=0.5, w_wiki=0.5
- **Equal Mix**: all weights equal

## Implementation

### Backend (`density_core.py`)

Add `compute_custom_distance_matrix()`:
```python
def compute_custom_distance_matrix(
    embeddings: np.ndarray,
    input_embeddings: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    metric_model=None,
    blend_weights: dict,  # {embedding: 0.4, weights: 0.2, learned: 0.3, wiki: 0.1}
    embedding_alpha: float = 0.5,  # input↔output mix within embedding term
) -> np.ndarray:
    """Compute weighted sum of distance matrices from multiple spaces."""
    dist_matrices = {}

    # Embedding distance (with optional input↔output blend)
    if blend_weights.get('embedding', 0) > 0:
        if embedding_alpha is not None and input_embeddings is not None:
            dist_matrices['embedding'] = blend_distance_matrices(
                input_embeddings, embeddings, embedding_alpha
            )
        else:
            dist_matrices['embedding'] = cosine_distance_matrix(embeddings)

    # Weight space distance
    if blend_weights.get('weights', 0) > 0 and weights is not None:
        dist_matrices['weights'] = cosine_distance_matrix(weights)

    # Learned metric distance
    if blend_weights.get('learned', 0) > 0 and weights is not None:
        metric_emb = compute_metric_embeddings(
            input_embeddings, embeddings, weights, metric_model
        )
        dist_matrices['learned'] = euclidean_distance_matrix(metric_emb)

    # Wikipedia physics predicted distance
    if blend_weights.get('wiki', 0) > 0:
        dist_matrices['wiki'] = compute_wikipedia_physics_distances(
            input_embeddings or embeddings
        )

    # Normalize scale: each distance matrix to [0, 1] range
    for key in dist_matrices:
        d = dist_matrices[key]
        d_max = d.max()
        if d_max > 0:
            dist_matrices[key] = d / d_max

    # Weighted sum
    result = np.zeros_like(next(iter(dist_matrices.values())))
    total_weight = 0
    for key, d in dist_matrices.items():
        w = blend_weights.get(key, 0)
        result += w * d
        total_weight += w

    if total_weight > 0:
        result /= total_weight

    np.fill_diagonal(result, 0)
    return result
```

### Scale Normalization

Different distance spaces have different scales:
- Cosine distance: [0, 2] but typically [0, 1] for normalized embeddings
- Euclidean in learned metric: unbounded, depends on model
- Wikipedia physics predicted: model-specific range

Before blending, normalize each distance matrix to [0, 1] by dividing by its
max value. This ensures each space contributes proportionally to its weight
regardless of native scale.

### Backend (`flask_api.py`)

Accept `custom_blend_weights` dict and `custom_embedding_alpha` in the
`/api/compute` request body. When present, compute the custom distance matrix
and pass it as `dist_matrix_override` to the tree builder.

### Frontend (`index.html`)

- Add Custom mode UI with weight sliders, normalization button, and presets
- Send `custom_blend_weights` and `custom_embedding_alpha` in API request
- Disable unavailable spaces based on model/data availability

## Open Questions

1. **Scale normalization**: Dividing by max is simple but sensitive to outliers.
   Alternatives: percentile-based, z-score, or rank-based normalization. Rank
   normalization (`d_rank[i,j] = rank(d[i,j]) / N^2`) would make all spaces
   strictly comparable but loses magnitude information.

2. **Visualization blending**: Should Custom mode also support layout blending?
   This would require MDS on the blended distance matrix (feasible but slower
   than SVD on individual embedding spaces).

3. **Dynamic availability**: Some distance spaces require specific models or data.
   Should unavailable spaces be hidden entirely or shown as disabled with
   explanation?

4. **Persistence**: Should custom weight presets be saveable/loadable?

5. **Interaction with Tree Distance Metric selector**: The existing selector
   (Data tab) picks a single distance metric. Custom blend mode supersedes
   this. Should the selector be disabled/hidden when Custom blend is active?
