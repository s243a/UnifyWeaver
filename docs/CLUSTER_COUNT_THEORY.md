# Theoretical Basis for Cluster Count Selection

This document explains the theoretical foundations for selecting the optimal number of clusters K in federated models.

## Overview

When training a federated model with K clusters, choosing K involves a trade-off:
- **Too few clusters**: Poor approximation, loss of fine-grained structure
- **Too many clusters**: Over-segmented, spread softmax weights, inefficient routing

We provide three theoretically-grounded methods for suggesting K.

---

## 1. Effective Rank (Recommended)

### Definition

The **effective rank** (also called **participation ratio**) measures how many dimensions meaningfully contribute to the data:

```
effective_rank = (Σ σᵢ)² / Σ σᵢ²
```

where σᵢ are the singular values of the data matrix.

### Intuition

- If all singular values are equal (σᵢ = c): effective_rank = n (full rank)
- If one dominates (σ₁ >> σᵢ for i>1): effective_rank ≈ 1
- In between: effective_rank captures the "number of significant directions"

### Theoretical Basis

The effective rank relates to:

1. **Spectral entropy**: It's the exponential of the spectral entropy H = -Σ pᵢ log pᵢ where pᵢ = σᵢ²/Σσⱼ²

2. **Matrix approximation**: For low-rank approximation, keeping r = effective_rank singular values captures most of the Frobenius norm

3. **Statistical degrees of freedom**: In ridge regression, effective_rank equals the effective degrees of freedom

### References

- Roy, O., & Vetterli, M. (2007). "The effective rank: A measure of effective dimensionality"
- Vershynin, R. (2012). "Introduction to the non-asymptotic analysis of random matrices"

### When to Use

Best for: General-purpose clustering when you want to match the intrinsic spectral structure of the data.

---

## 2. Covering Number (2^r)

### Definition

The **covering number** N(ε) is the minimum number of balls of radius ε needed to cover a set.

For a d-dimensional manifold, the covering number scales as:

```
N(ε) ~ (1/ε)^d
```

If we discretize each dimension into b bins:

```
K = b^d ≈ b^r
```

where r is the intrinsic dimensionality (dimensions for target variance).

### Intuition

- If data lies on a 2D surface in 100D space, you only need O(1/ε²) balls to cover it
- The exponent d (or r) is the intrinsic dimension, not the ambient dimension

### Theoretical Basis

1. **Metric entropy**: The log of the covering number is the metric entropy, fundamental in:
   - Kolmogorov complexity
   - Rate-distortion theory
   - Statistical learning theory (VC dimension bounds)

2. **Manifold hypothesis**: Real data often lies on low-dimensional manifolds embedded in high-dimensional space

3. **Johnson-Lindenstrauss**: Random projections preserve distances with O(log n) dimensions, suggesting intrinsic dimension matters more than ambient dimension

### Calculation

We find r such that the top r singular values capture 80% of variance:

```python
cumvar = cumsum(σ²) / sum(σ²)
r = argmin(cumvar >= 0.80) + 1
K = 2^r  # Binary discretization per dimension
```

### References

- Kolmogorov, A.N., & Tikhomirov, V.M. (1961). "ε-entropy and ε-capacity of sets"
- Tenenbaum, J.B., et al. (2000). "A global geometric framework for nonlinear dimensionality reduction" (Isomap)

### When to Use

Best for: Data with clear low-dimensional structure where you want geometric coverage guarantees.

---

## 3. √N Heuristic

### Definition

A common rule of thumb for K-means:

```
K ≈ √(N/2) ≈ √N
```

where N is the number of data points (or in our case, the current number of clusters).

### Intuition

- Balances cluster size with cluster count
- If you have N points and want each cluster to have ~√N points, you need ~√N clusters
- For meta-clustering K=700 clusters: √700 ≈ 27

### Theoretical Basis

1. **Optimal quantization**: For uniform distribution on [0,1]^d, optimal K-means distortion is:
   ```
   D(K) ~ K^(-2/d)
   ```
   Minimizing total complexity (K + distortion cost) often yields K ~ √N

2. **Lloyd's algorithm convergence**: Analysis shows optimal K grows sub-linearly with N

3. **Akaike/BIC criteria**: Model selection criteria often suggest K ~ √N to balance fit and complexity

### References

- Mardia, K.V., et al. (1979). "Multivariate Analysis" (cluster analysis chapter)
- Sugar, C.A., & James, G.M. (2003). "Finding the number of clusters in a dataset"

### When to Use

Best for: Quick estimation when you don't need precise intrinsic dimensionality analysis.

---

## Comparison

| Method | Formula | Theoretical Grounding | Computational Cost |
|--------|---------|----------------------|-------------------|
| Effective rank | (Σσ)²/Σσ² | Spectral theory, information theory | O(K³) for SVD |
| Covering (2^r) | 2^r where r = dim(80% var) | Metric geometry, manifold theory | O(K³) for SVD |
| √N | √K | Heuristic, model selection | O(1) |

## Recommendation

1. **Default**: Use effective rank - balances theory and practicality
2. **Low-dimensional data**: Use covering number if you know data has clear low-dim structure
3. **Quick estimate**: Use √N when computational resources are limited

## Implementation

See `scripts/analyze_federated_clusters.py` for implementation:

```bash
# Analyze and get suggestions
python3 scripts/analyze_federated_clusters.py models/federated.pkl

# Refine using effective rank (recommended)
python3 scripts/analyze_federated_clusters.py models/federated.pkl \
    --refine --method effective_rank --output models/federated_refined.pkl

# Refine with custom target
python3 scripts/analyze_federated_clusters.py models/federated.pkl \
    --refine --target 100 --output models/federated_100.pkl
```

## Softmax Peakedness Criterion

Beyond cluster count, we also analyze **routing quality** via softmax peakedness:

- **top-1 weight**: Average weight on the highest-scoring cluster
- **top-10 weight**: Average cumulative weight on top 10 clusters

Target: top-1 weight ≥ 50% indicates well-separated clusters.

If top-1 weight < 50%, the model is over-segmented and meta-clustering is recommended.
