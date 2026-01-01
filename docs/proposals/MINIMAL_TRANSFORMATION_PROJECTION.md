# Minimal Transformation Projection

**Status:** Proposed
**Date:** 2025-12-24
**Related:** [LDA_PHILOSOPHY.md](LDA_PHILOSOPHY.md)

## Motivation

Current approaches solve regularized least squares:
```
min ||Q @ W - A||² + λ||W||²
```

This requires tuning λ and doesn't directly encode what we want: the **minimal transformation** that maps centroids to answers.

A minimal transformation:
- Uses only the degrees of freedom necessary (rotations, scaling, translation)
- Automatically creates null space for deviations
- Has no hyperparameter to tune

## The Procrustes Formulation

Given:
- C = matrix of centroid embeddings (k × d)
- A = matrix of answer embeddings (k × d)

Find the transformation W that:
1. Maps centroids to answers: C @ W ≈ A
2. Is minimal: uses only rotation + scaling

### Orthogonal Procrustes (Rotation Only)

```
min ||C @ R - A||²
subject to: R^T @ R = I  (R is orthogonal)
```

**Closed-form solution:**
```python
def orthogonal_procrustes(C, A):
    """Find optimal rotation from C to A."""
    M = C.T @ A           # Cross-covariance
    U, S, Vt = svd(M)     # SVD
    R = U @ Vt            # Optimal rotation
    return R
```

### Procrustes with Uniform Scaling

```
min ||s * C @ R - A||²
subject to: R^T @ R = I
```

**Closed-form solution:**
```python
def scaled_procrustes(C, A):
    """Find optimal rotation + uniform scaling."""
    M = C.T @ A
    U, S, Vt = svd(M)
    R = U @ Vt

    # Optimal scale
    s = np.trace(S) / np.trace(C.T @ C)

    return s * R
```

### Procrustes with Per-Dimension Scaling

```
min ||C @ R @ S - A||²
subject to: R^T @ R = I, S diagonal
```

This allows different scaling along different directions (stretching/compression).

## Why This Works

### Automatic Null Space

The rotation R is orthogonal, meaning:
- It preserves lengths and angles
- Directions orthogonal to the centroid-answer plane are unchanged
- Deviations in these directions → 0 (mapped to orthogonal directions in output)

### Minimal Degrees of Freedom

| Approach | Parameters |
|----------|------------|
| Full W matrix | d² |
| Procrustes (rotation) | d(d-1)/2 |
| Procrustes + uniform scale | d(d-1)/2 + 1 |
| Procrustes + diagonal scale | d(d-1)/2 + d |
| Regularized least squares | d² (but λ constrains) |

For d=384, k=20:
- Full W: 147,456 parameters
- Procrustes: ~73,500 rotation + 384 scaling = still large but structured

But the key insight: **most of these rotation parameters are identity** because we only need to rotate in the k-dimensional subspace spanned by centroids.

### Effective Dimensionality

If centroids span a k-dimensional subspace (k << d):
- Only rotations within this subspace matter
- Orthogonal directions: R = I (no rotation needed)
- Effective parameters: O(k²) not O(d²)

## Algorithm: Subspace Procrustes

```python
def subspace_procrustes(C, A, k=None):
    """
    Procrustes in the subspace spanned by centroids.

    Args:
        C: Centroid matrix (n_clusters × d)
        A: Answer matrix (n_clusters × d)
        k: Subspace dimension (default: n_clusters)

    Returns:
        W: Minimal transformation matrix (d × d)
    """
    n, d = C.shape
    k = k or n

    # Step 1: Find principal subspace of centroids
    U_c, S_c, Vt_c = svd(C, full_matrices=False)
    V_k = Vt_c[:k].T  # Top k right singular vectors (d × k)

    # Step 2: Project to subspace
    C_k = C @ V_k     # (n × k)
    A_k = A @ V_k     # (n × k)

    # Step 3: Procrustes in subspace
    M = C_k.T @ A_k   # (k × k)
    U, S, Vt = svd(M)
    R_k = U @ Vt      # Rotation in k-space

    # Step 4: Optimal scaling
    s = np.trace(S) / np.trace(C_k.T @ C_k)

    # Step 5: Lift back to full space
    # W acts as: project to subspace, rotate, scale, project back
    W = V_k @ (s * R_k) @ V_k.T

    # Step 6: Add identity on orthogonal complement
    # (optional: could also zero out orthogonal directions)
    W = W + (np.eye(d) - V_k @ V_k.T)

    return W
```

### Complexity

- SVD of C: O(n × d × min(n,d))
- SVD of M: O(k³)
- Matrix products: O(d × k²)

For k << d, this is efficient.

## Handling Multiple Clusters Smoothly

The above gives one global W. For per-cluster projections with smoothness:

### Option 1: Shared Rotation, Per-Cluster Scaling

```python
def smooth_procrustes(clusters):
    """
    Shared rotation, smooth per-cluster scaling.
    """
    # Stack all centroids and answers
    C_all = np.vstack([c for c, a in clusters])
    A_all = np.vstack([a for c, a in clusters])

    # Global rotation
    R = orthogonal_procrustes(C_all, A_all)

    # Per-cluster scaling (can be smoothed)
    scales = []
    for C_i, A_i in clusters:
        C_rot = C_i @ R
        s_i = optimal_scale(C_rot, A_i)
        scales.append(s_i)

    # Smooth the scales across similar clusters
    scales_smooth = kernel_smooth(scales, cluster_similarities)

    return R, scales_smooth
```

### Option 2: Interpolated Rotations

Rotations live on a manifold (SO(d)). Interpolation requires geodesics:
```
R(t) = R_1 @ expm(t * logm(R_1.T @ R_2))
```

This is more complex but principled for blending rotations between similar clusters.

## Connection to Current Approaches

| Current | Minimal Transformation |
|---------|----------------------|
| λ regularization | Not needed (structure is built-in) |
| FFT smoothing | Could smooth scaling factors |
| Kernel smoothing | Could smooth rotations on manifold |
| Null space | Automatic from orthogonality |

## What This Doesn't Handle

1. **Non-linear relationships**: Procrustes is linear/affine
2. **Multiple answers per cluster**: Need to aggregate first
3. **Very high k**: If k approaches d, savings diminish

## Future Exploration

### Clifford Algebra / Geometric Algebra

Rotations can be expressed elegantly using bivectors (wedge products):
- In 3D: rotation axis via cross product a × b
- In dD: rotation plane via wedge product a ∧ b

```
Rotation in plane (u, v):
R = exp(θ/2 * (u ∧ v))      # Rotor in geometric algebra
x' = R x R†                  # Apply rotation
```

This provides:
- Natural parameterization of rotations in any dimension
- Composition via geometric product
- Potential for more efficient computation

Worth exploring whether geometric algebra provides better algorithms for:
- Interpolating rotations (geodesics on SO(d))
- Composing per-cluster transformations
- Expressing the "minimal transformation" constraint directly

### Lie Groups and Exponential Maps

SO(d) is a Lie group with Lie algebra so(d) (skew-symmetric matrices).
- Rotations near identity: R ≈ I + A where A is skew-symmetric
- Exponential map: R = expm(A)
- Logarithm: A = logm(R)

This might enable:
- Smooth interpolation of rotations
- Gradient-based optimization on the rotation manifold
- Better understanding of the "minimal" structure

## Proposed Experiments

### Experiment 1: Procrustes vs Regularized LS

```python
def compare_approaches(clusters, test_queries):
    # Regularized least squares (current)
    W_reg = train_regularized(clusters, lambda_=0.01)
    mrr_reg = evaluate(W_reg, test_queries)

    # Procrustes
    W_proc = subspace_procrustes(centroids, answers)
    mrr_proc = evaluate(W_proc, test_queries)

    print(f"Regularized LS: MRR={mrr_reg:.4f}")
    print(f"Procrustes:     MRR={mrr_proc:.4f}")
```

### Experiment 2: Deviation Suppression

```python
def measure_deviation_response(W, clusters):
    """How much do within-cluster deviations get amplified?"""
    for C_i, A_i in clusters:
        centroid = C_i.mean(axis=0)
        deviations = C_i - centroid

        # Response to deviations
        dev_response = deviations @ W
        dev_norm = np.linalg.norm(dev_response) / np.linalg.norm(deviations)

        print(f"Deviation amplification: {dev_norm:.4f}")
```

### Experiment 3: Effective Rank

```python
def analyze_structure(W):
    """Is W actually low-rank as predicted?"""
    U, S, Vt = svd(W)

    # How many significant singular values?
    cumsum = np.cumsum(S) / np.sum(S)
    effective_rank = np.searchsorted(cumsum, 0.99) + 1

    print(f"Effective rank: {effective_rank} / {len(S)}")
```

## Empirical Results

**Benchmark on all-MiniLM (384-dim, 20 clusters):**

| Method | MRR | R@1 | Train Time |
|--------|-----|-----|------------|
| **MinTrans_none (Procrustes)** | **0.8688** | 79% | 32ms |
| MinTrans_fft_f7 | 0.8621 | 78% | 154ms |
| FFT_c0.5 | 0.4418 | 22% | 153ms |
| AdaptiveCond_K8 | 0.4079 | 18% | 5,195ms |
| All other methods | 0.26-0.44 | 10-22% | 1-8 sec |

**Key finding:** Pure minimal transformation (no smoothing) achieves nearly **double the MRR** of the next best method.

**Why smoothing doesn't help:**
1. Good embeddings produce well-separated clusters (like ModernBERT case)
2. The exact centroid→answer mapping is already correct
3. Smoothing corrupts the exact minimal transform

**Where smoothing should happen:**
- Not on W matrices (keep them as exact Procrustes transforms)
- On the **routing layer** (softmax temperature, density weighting)
- The routing already interpolates between clusters for "in-between" queries

## Summary

The minimal transformation framing suggests:
1. **Procrustes** as the direct solution (rotation + scaling)
2. **Automatic null space** from orthogonality structure
3. **No λ hyperparameter** - the constraint is structural
4. **O(k²) effective parameters** instead of O(d²)
5. **Dramatically better performance** (2x MRR improvement)

This is more principled than regularized least squares with an arbitrary λ. The structure we want (map centroids, ignore deviations) is built into the formulation, not imposed via a penalty.

## Handling Overdetermined Systems

When clusters have many Q-A pairs (n >> d), the system is **overdetermined** rather than underdetermined.

### The Problem

For large clusters:
```
Q ∈ ℝⁿˣᵈ  (n queries, d dimensions)
A ∈ ℝⁿˣᵈ  (n answers)
```

If n > d, naive least squares finds the unique W minimizing ||Q @ W - A||².
But this can overfit to per-query variations.

### Our Solution: Centroid Reduction

Instead of fitting n Q-A pairs, we reduce to **centroids**:

```python
def centroid_procrustes(Q, A):
    """
    Reduce overdetermined system via centroid.
    
    One W per cluster, computed from centroid → centroid.
    """
    Q_centroid = Q.mean(axis=0, keepdims=True)  # (1, d)
    A_centroid = A.mean(axis=0, keepdims=True)  # (1, d)
    
    # Procrustes on centroids (1×d → 1×d)
    W, scale, info = compute_minimal_transform(Q_centroid, A_centroid)
    return W
```

### Why This Works

1. **Denoising**: Centroid averages out per-query noise
2. **Minimal transform**: Single Q→A pair gives unique rotation
3. **Cluster-level semantics**: We care about cluster direction, not individual variations
4. **O(d²) per cluster**: Constant complexity regardless of cluster size

### Mathematical Justification

For n queries in a cluster, the centroid is the **maximum likelihood estimate** of the underlying query direction under Gaussian noise:

```
Q_centroid = (1/n) Σᵢ qᵢ ≈ μ_Q (true query direction)
```

The Procrustes transform W finds the minimal rotation from μ_Q to μ_A.
Individual queries qᵢ that deviate from μ_Q will be projected correctly if they're **close enough** to the centroid (soft routing handles edge cases).

### Alternative: SVD Truncation

For very large clusters where centroid is too aggressive:

```python
def svd_reduced_procrustes(Q, A, k=None):
    """
    Reduce via top-k principal components of query space.
    """
    k = k or min(Q.shape[0], Q.shape[1]) // 2
    
    U, S, Vt = svd(Q, full_matrices=False)
    Q_reduced = U[:, :k] @ np.diag(S[:k])  # Top-k components
    
    # Procrustes on reduced space
    W = scaled_procrustes(Q_reduced, A[:, :k])
    return W
```

This preserves within-cluster variation while regularizing.

## References

- Procrustes analysis: Schönemann, P.H. (1966). "A generalized solution of the orthogonal Procrustes problem"
- Geometric algebra: Hestenes, D. (1984). "Clifford Algebra to Geometric Calculus"
- Rotation interpolation: Shoemake, K. (1985). "Animating rotation with quaternion curves" (SLERP)

