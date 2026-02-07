# Proposal: Custom Multi-Space Distance Blending

## Context

The density explorer's Blend tab supports input↔output embedding space blending
(modes: None, Visualization, Tree Distance, Both). These controls handle
blending between input and output embedding spaces via an α slider.

Below these controls, a **Customization** section (separated by a horizontal
line) combines the blended embedding distance with other distance spaces for
tree construction. This is a second layer: Blend produces `d_emb(α)`, then
Customization mixes `d_emb(α)` with other distance metrics.

The Visualization blend (2D layout) remains independent and is not affected
by the Customization section.

### Observation: Information-Preserving Models

The Bivector Paired model is designed to preserve input embedding geometry (good
hit@k with minimal training). Testing shows this produces nearly identical
distance orderings in input vs output space — only 8/299 MST edges differ. The
input↔output blend is therefore a near-identity operation for tree distances.

This means the existing input↔output blend can be collapsed into a single
"embedding distance" term in the customization layer, rather than occupying two
separate slots. The significant differences come from mixing embedding distances
with other distance spaces (weights, learned metric, Wikipedia physics).

## Architecture: Inline Customization

Each blend slider has its own customization dropdown placed directly beneath it.
The customization is contextually paired with the slider it modifies. The number
of customization dropdowns depends on which blend mode is selected:

```
Mode: Visualization                Mode: Tree Distance
┌──────────────────────────┐      ┌──────────────────────────┐
│ Viz α: [===|===] 0.50    │      │ Tree α: [===|===] 0.50   │
│ Customize: [None      ▾] │      │ Customize: [None      ▾] │
│ [L^n controls if active] │      │ [L^n controls if active] │
└──────────────────────────┘      └──────────────────────────┘

Mode: Both
┌──────────────────────────┐
│ Viz α: [===|===] 0.50    │
│ Customize: [None      ▾] │
│ [L^n controls if active] │
├──────────────────────────┤  ← <hr>
│ Tree α: [===|===] 0.50   │
│ Customize: [None      ▾] │
│ [L^n controls if active] │
└──────────────────────────┘
```

- Each slider produces `d_emb(α)` from its input↔output blend
- Each customization dropdown independently combines `d_emb(α)` with other
  distance spaces for its purpose (layout or tree)
- When customization is "None", only `d_emb(α)` is used (current behavior)
- In "Both" mode, the two customization dropdowns are independent — you can
  have L^n for tree distances while visualization uses no customization

### Visualization Customization

When a custom distance matrix is computed for visualization, 2D coordinates
are obtained via **MDS** (multidimensional scaling) on the blended distance
matrix, since SVD on a single embedding space no longer applies when mixing
multiple spaces. This uses the existing `classical_mds_2d()` function.

### Tree Customization

L^n power mean of distance matrices, passed as `dist_matrix_override` to the
tree builder. This is the primary use case.

## Available Distance Spaces

| Space | Dim | Distance | Semantics | Requires |
|-------|-----|----------|-----------|----------|
| Embedding `d_emb(α)` | 768D | Cosine | Semantic similarity | Always available |
| Weight Space `d_w` | 64D | Cosine | Transformation recipe similarity | Model |
| Learned Metric `d_l` | 64D | Euclidean | Organizational proximity | Model |
| Wikipedia Physics `d_wiki` | N/A | Predicted | Category hierarchy distance | Wiki model |

All spaces are reduced to N×N distance matrices and normalized to [0,1] before
combination, sidestepping the dimensionality mismatch.

## Customization Methods

### Method 1: None (default)

No customization. Tree uses `d_emb(α)` from the blend controls (or raw
embedding distance if blend is also None). This is the current behavior.

### Method 2: L^n (Generalized Power Mean)

```
d_custom = (w₁·d₁ⁿ + w₂·d₂ⁿ + ... + wₖ·dₖⁿ)^(1/n)
```

Where `dᵢ` are normalized distance matrices and `wᵢ` are weights summing to 1.

The exponent `n` controls the blending behavior:

| n | Name | Behavior |
|---|------|----------|
| -1 | Harmonic | Strongly favors spaces where points are close |
| 0 | Geometric | Geometric mean (balanced, multiplicative) |
| 0.5 | Lenient | Being close in ANY space helps |
| **1** | **Linear** | **Weighted average (default)** |
| 2 | Euclidean | Points must be close in ALL spaces |
| ∞ | Max | Dominated by largest distance |

#### UI Controls

```
Method: [L^n          ▾]

  n: [1.0  ]   Presets: [Linear] [Euclidean] [Harmonic]

  Space                    Weight    Effective
  ─────────────────────────────────────────────
  Embedding (blended)      [3    ]   = 0.43
  Weight Space             [2    ]   = 0.29      (disabled if no model)
  Learned Metric           [2    ]   = 0.29      (disabled if no model)
  Wikipedia Physics        [0    ]   = 0.00      (disabled if no wiki model)
                           ─────
                    Total: 7       [Normalize]

  [Apply Customization]
```

- **Weight text boxes**: Users enter relative weights (any positive number).
  Effective (normalized) weights shown beside each input.
- **Normalize button**: Rescales raw weights to sum to 1.0 in-place.
- **n text box**: Defaults to 1.0. Named preset buttons set common values.
- **Disabled spaces**: Grayed out when prerequisites aren't met, with tooltip
  explaining why (e.g., "Requires a Transformation Model").

#### Special Cases for n

- **n=0** (geometric mean): Use `exp(w₁·log(d₁) + w₂·log(d₂) + ...)` to avoid
  numerical issues. Requires all distances > 0; replace zeros with ε=1e-10.
- **n→±∞**: Not implemented directly. Cap at reasonable range (e.g., [-10, 10]).

#### Presets

Quick-set buttons for common configurations:

| Preset | n | Weights | Use case |
|--------|---|---------|----------|
| Semantic Only | 1 | emb=1, rest=0 | Baseline |
| Hierarchical | 1 | emb=0.3, w=0.3, l=0.4 | Organizational tree |
| Wikipedia-Guided | 1 | emb=0.5, wiki=0.5 | Category-informed |
| Equal Mix | 1 | all equal | Exploratory |
| Strict Consensus | 2 | all equal | Must be close in all spaces |

### Method 3: Formulaic (future)

A formula editor where each distance matrix is a named variable:

```
Variables: E = d_emb(α), W = d_weights, L = d_learned, P = d_wiki

Formula: [sqrt(0.5 * E^2 + 0.3 * W^2 + 0.2 * L^2)          ]
```

This subsumes L^n (the formula `(w1*E^n + w2*W^n)^(1/n)` is just one case)
and enables arbitrary combinations:
- `min(E, W)` — take the closer distance in either space
- `E * (1 - 0.3 * W)` — semantic distance discounted by weight similarity
- `E + 0.5 * max(L - 0.3, 0)` — semantic with learned penalty for distant items

#### Implementation Considerations

- **Expression parser**: Use a safe math expression evaluator (e.g., math.js in
  the browser, or `numexpr`/`ast.literal_eval` in Python). Must NOT use `eval()`.
- **Matrix operations**: Variables are N×N matrices. The parser must support
  element-wise operations, not matrix multiplication.
- **Validation**: Check that the formula produces a valid distance matrix
  (symmetric, non-negative, zero diagonal) and warn if violated.
- **Error handling**: Show parse errors inline, don't submit invalid formulas.
- **Presets as formulas**: The L^n presets can also be shown as their formula
  equivalents for learning.

**Status: Deferred.** Implement L^n first; Formulaic is a later enhancement.

## Scale Normalization

Before combining, each distance matrix is normalized to [0, 1]:

```python
d_normalized = d / d.max()
```

This ensures each space contributes proportionally to its weight regardless of
native scale. Alternatives considered:

| Method | Pros | Cons |
|--------|------|------|
| **Max normalization** | Simple, preserves ordering | Sensitive to outliers |
| Percentile (95th) | Robust to outliers | Clips extreme distances |
| Z-score | Statistical normalization | Can produce negatives |
| Rank normalization | Perfectly comparable | Loses magnitude info |

**Recommendation:** Start with max normalization. If outlier sensitivity is a
problem in practice, switch to 95th percentile.

## Implementation Plan

### Backend (`density_core.py`)

Add `compute_custom_distance_matrix()`:

```python
def compute_custom_distance_matrix(
    embeddings: np.ndarray,
    input_embeddings: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    metric_model=None,
    space_weights: dict,       # {embedding: 3, weights: 2, learned: 2, wiki: 0}
    embedding_alpha: float,    # from existing blend slider
    power_n: float = 1.0,     # L^n exponent
) -> np.ndarray:
    """
    Compute L^n power mean of distance matrices from multiple spaces.

    Normalizes each distance matrix to [0,1], raises to power n,
    computes weighted sum, then takes the (1/n)-th root.
    """
    dist_matrices = {}

    # 1. Compute each distance matrix
    if space_weights.get('embedding', 0) > 0:
        if embedding_alpha is not None and input_embeddings is not None:
            dist_matrices['embedding'] = blend_distance_matrices(
                input_embeddings, embeddings, embedding_alpha
            )
        else:
            dist_matrices['embedding'] = cosine_distance_matrix(embeddings)

    if space_weights.get('weights', 0) > 0 and weights is not None:
        dist_matrices['weights'] = cosine_distance_matrix(weights)

    if space_weights.get('learned', 0) > 0 and weights is not None:
        metric_emb = compute_metric_embeddings(
            input_embeddings, embeddings, weights, metric_model
        )
        dist_matrices['learned'] = euclidean_distance_matrix(metric_emb)

    if space_weights.get('wiki', 0) > 0:
        dist_matrices['wiki'] = compute_wikipedia_physics_distances(
            input_embeddings or embeddings
        )

    # 2. Normalize each to [0, 1]
    for key in dist_matrices:
        d = dist_matrices[key]
        d_max = d.max()
        if d_max > 0:
            dist_matrices[key] = d / d_max

    # 3. Normalize weights to sum to 1
    active_weights = {k: space_weights[k] for k in dist_matrices}
    total = sum(active_weights.values())
    if total > 0:
        active_weights = {k: v/total for k, v in active_weights.items()}

    # 4. Compute L^n power mean
    n = power_n
    if abs(n) < 1e-10:
        # Geometric mean: exp(Σ wᵢ·log(dᵢ))
        log_sum = np.zeros_like(next(iter(dist_matrices.values())))
        for key, d in dist_matrices.items():
            w = active_weights[key]
            log_sum += w * np.log(d + 1e-10)
        result = np.exp(log_sum)
    else:
        # General case: (Σ wᵢ·dᵢⁿ)^(1/n)
        powered_sum = np.zeros_like(next(iter(dist_matrices.values())))
        for key, d in dist_matrices.items():
            w = active_weights[key]
            powered_sum += w * np.power(d + 1e-10, n)
        result = np.power(powered_sum, 1.0 / n)

    np.fill_diagonal(result, 0)
    return result
```

### Backend (`flask_api.py`)

Accept in `/api/compute` request body:
```json
{
  "custom_space_weights": {"embedding": 3, "weights": 2, "learned": 2, "wiki": 0},
  "custom_power_n": 1.0
}
```

When present, compute the custom distance matrix and pass as
`dist_matrix_override` to the tree builder. The existing `blend_tree_alpha`
feeds into the embedding distance term.

### Frontend (`index.html`)

Each blend slider group (Visualization, Tree) gets a customization dropdown
placed directly beneath its slider + text input:

```html
<!-- Visualization group -->
<div id="blendVizGroup">
  <label>Layout Blend: 0.50</label>
  <input type="range" ...>
  <select id="vizCustomize">  <!-- NEW -->
    <option value="none">Customize: None</option>
    <option value="ln">Customize: L^n</option>
    <option value="formula" disabled>Customize: Formulaic (coming soon)</option>
  </select>
  <div id="vizCustomizeControls" style="display:none">
    <!-- n input, weight text boxes, normalize button -->
  </div>
</div>

<!-- Horizontal rule (only in Both mode) -->
<hr id="blendSeparator">

<!-- Tree group -->
<div id="blendTreeGroup">
  <label>Tree Blend: 0.50</label>
  <input type="range" ...>
  <select id="treeCustomize">  <!-- NEW -->
    ...same options...
  </select>
  <div id="treeCustomizeControls" style="display:none">
    <!-- n input, weight text boxes, normalize button -->
  </div>
</div>
```

The existing "Apply Blend" button triggers the full computation including
any active customization. No separate button needed — customization params
are sent alongside blend params in the same API request.

## Interaction with Existing Controls

- **Tree Distance Metric selector** (Data tab): When tree customization is
  active (not None), it supersedes the Tree Distance Metric selector. The
  selector could show "(overridden by Customization)" or be visually dimmed.
- **Blend α sliders**: Still control the input↔output mix within the embedding
  distance term. Active even when customization is enabled — the blended
  embedding distance feeds into the customization as one of the distance spaces.
- **Each customization is independent**: Viz customization and tree
  customization can use different methods, different weights, and different n
  values. The only shared element is the blend mode selector that determines
  which slider groups are visible.

## Open Questions

1. **Scale normalization**: Start with max normalization or use percentile?

2. **Dynamic availability**: Show unavailable spaces as disabled (with tooltip)
   or hide entirely?

3. **Persistence**: Should custom weight configurations be saveable/loadable?

4. **Formulaic safety**: Which expression parser to use? math.js (browser) is
   well-tested but adds a dependency. A minimal custom parser for basic
   arithmetic + element-wise functions might suffice.

5. **MDS performance**: Visualization customization uses MDS on the custom
   distance matrix. For 200-300 points this is fast, but for larger datasets
   it may be slower than SVD. Should there be a point count threshold that
   disables viz customization?

6. **Shared customization**: Should "Both" mode offer a "Link customization"
   toggle that applies the same method/weights to both viz and tree, reducing
   the number of controls for the common case?
