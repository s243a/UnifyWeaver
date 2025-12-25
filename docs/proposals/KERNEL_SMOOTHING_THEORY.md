# Kernel Smoothing Theory for Cross-Cluster Projections

**Status:** Proposed
**Version:** 0.1
**Date:** 2025-12-24
**Extends:** [CROSS_CLUSTER_SMOOTHING.md](CROSS_CLUSTER_SMOOTHING.md), [SMOOTHING_BASIS_PROJECTION.md](SMOOTHING_BASIS_PROJECTION.md)

## Executive Summary

This document presents the theoretical foundation for smoothing projection matrices across semantically similar clusters. We show that:

1. **1D FFT smoothing** has fundamental limitations for high-dimensional embeddings
2. **Graph Laplacian smoothing** is the principled approach
3. **Green's functions** provide the solution kernel
4. **Kernel methods** offer numerically stable implementations
5. **Matérn kernels** balance expressiveness with numerical stability

## Motivation: Why 1D FFT Falls Short

### The 1D Ordering Problem

Our current FFT smoothing:
1. Orders clusters into a 1D sequence (by similarity to first cluster)
2. Applies low-pass filter along this line
3. Smooths "adjacent" clusters in this arbitrary ordering

But semantic similarity in d-dimensional embedding space is **inherently multi-dimensional**:

```
Cluster similarity graph (reality):       Our 1D ordering (what FFT sees):

    A ──── B                               A → B → C → D → E → F
   /│\     │                               (loses structure)
  C │ D    E
   \│/
    F
```

Clusters A, C, D, F might all be mutually similar, but the 1D ordering forces them into a line. The FFT smooths along this line, missing the true graph structure.

### Embedding Space Dimensionality

| Embedding Model | Dimensions | 1D FFT Captures |
|-----------------|------------|-----------------|
| all-MiniLM-L6-v2 | 384 | ~0.26% of structure |
| ModernBERT | 768 | ~0.13% of structure |

For ModernBERT, our benchmarks showed standalone basis learning (K=4) outperformed FFT-based methods, suggesting the 1D approximation was actively harmful.

## Graph Laplacian Smoothing

### The Principled Formulation

Instead of 1D ordering, use the full cluster similarity structure:

**Loss function:**
```
L = Σ_i ||Q_i W_i - A_i||² + λ Σ_{i,j} S_ij ||W_i - W_j||²
    ├────────────────────┘   └────────────────────────────┘
    Data fidelity term        Smoothness regularization
```

Where:
- S_ij = semantic similarity between clusters i and j
- λ = regularization strength

**Matrix form:**
```
L = ||QW - A||²_F + λ tr(W^T L W)
```

Where L is the **graph Laplacian**:
```
L = D - S
D_ii = Σ_j S_ij  (degree matrix)
```

### The Lipschitz Smoothness Interpretation

The regularization term enforces a **Lipschitz constraint**:

```
||W_i - W_j||_F ≤ (1/λ) · d_semantic(cluster_i, cluster_j)
```

This says: "The rate of change of W with respect to semantic distance should be bounded."

- Large λ → strict smoothness, W varies slowly
- Small λ → more freedom, W can vary rapidly

### Optimal Solution

The optimal W satisfies:

```
(I + λL) W = W_target
```

Where W_target is the unconstrained per-cluster solution. The solution is:

```
W = (I + λL)^{-1} W_target
```

## Green's Functions

### Definition

The **Green's function** G is the inverse of the differential operator:

```
(I + λL) G = I
G = (I + λL)^{-1}
```

The smoothed solution is:
```
W_smoothed = G · W_target
W_smoothed_i = Σ_j G(i,j) W_target_j
```

Each cluster's W is a **weighted average** of all clusters, weighted by G(i,j).

### Physical Interpretation

- G(i,j) = "influence of cluster j on cluster i"
- Nearby clusters (small graph distance) → strong influence
- Distant clusters → weak influence
- λ controls the "influence radius"

### Spectral Decomposition

In the eigenbasis of L with eigenvalues μ_k and eigenvectors v_k:

```
G = Σ_k v_k v_k^T / (1 + λ μ_k)
```

This is a **low-pass filter**:
- Low eigenvalues (smooth modes): preserved
- High eigenvalues (rough modes): attenuated

## Connection to Spherical Harmonics

### When Clusters Lie on a Sphere

If cluster centroids are normalized (unit norm), they lie on S^{d-1}.

**Key insight:** Spherical harmonics Y_l^m are eigenfunctions of the Laplacian on the sphere:
```
∇²_{sphere} Y_l^m = -l(l+1) Y_l^m
```

The Green's function on a sphere has expansion:
```
G(x, x') = Σ_{l,m} Y_l^m(x) Y_l^m*(x') / (1 + λ l(l+1))
```

### Spherical FFT

For data on a sphere, spherical FFT provides O(N log²N) computation:

1. Expand W in spherical harmonics: W = Σ_{l,m} w_lm Y_l^m
2. Apply filter: ŵ_lm → ŵ_lm / (1 + λ l(l+1))
3. Inverse transform

**Limitation:** Spherical harmonics capture **angular** structure. Our similarity is **distance-based**, not angular.

## Radial Basis Functions

### Why Radial?

The influence of cluster j on cluster i depends on their **distance**, not angular position:

```
G(i,j) = f(d(centroid_i, centroid_j))
```

This suggests radial basis functions are more natural than spherical harmonics.

### Radial Green's Function

For the screened Laplacian in d dimensions:

```
G(r) ∝ r^{-(d-2)/2} K_{(d-2)/2}(r/ξ)
```

Where K is the modified Bessel function (exponential decay at large r).

### With Complex Frequency (Damped Oscillations)

```
G(r) = e^{-r/ξ} · cos(ωr + φ) / r^{(d-2)/2}
```

Components:
- **Decay**: e^{-r/ξ} — distant clusters have weak influence
- **Oscillation**: cos(ωr) — possible alternating positive/negative influence

### Integral Transforms

For radial functions, the natural transform is the **Hankel transform**:

```
F̂(k) = ∫₀^∞ f(r) J_n(kr) r dr
```

Where J_n are Bessel functions (radial eigenfunctions of the Laplacian).

| Transform | Basis | Use Case |
|-----------|-------|----------|
| Fourier | e^{iωx} | Translation-invariant |
| Laplace | e^{-sx} | Decay/causal systems |
| **Hankel** | J_n(kr) | Radial functions |
| Spherical | Y_l^m | Angular on sphere |

## Kernel Methods

### The Unified View

```
Differential Operator  ←→  Green's Function  ←→  Kernel
      (I + λL)                   G(r)              K(x,x')
         ↓                        ↓                  ↓
   Spectral Decomp         Integral Transform    RKHS Theory
    (eigenvalues)          (Hankel/Fourier)    (regularization)
```

The Green's function IS a kernel. This gives us:
- **RKHS (Reproducing Kernel Hilbert Space)** theory
- **Representer theorem**: solution is sum of kernel evaluations
- **Regularization guarantees**: well-posed inverse problem
- **Efficient computation**: kernel matrix operations

### Kernel Smoothing Formula

```python
W_smoothed_i = Σ_j K(centroid_i, centroid_j) · W_target_j / Σ_j K(...)
```

This is weighted averaging with kernel-defined weights.

## Numerical Stability

### The Laplacian Kernel Problem

The exponential (Laplacian) kernel K(r) = exp(-r/ξ) has numerical issues:

1. **Ill-conditioned kernel matrix** — small ξ → peaked → near-singular
2. **Inverse Laplace transform instability** — small errors amplify
3. **Dynamic range** — exp(-100) ≈ 0 causes underflow
4. **Singularity at r=0** — Green's function has 1/r factor

### The Matérn Family

The **Matérn kernel** interpolates between exponential and Gaussian:

```
K_ν(r) = (2^{1-ν}/Γ(ν)) (√(2ν)r/ℓ)^ν K_ν(√(2ν)r/ℓ)
```

Where ν controls smoothness:

| ν | Kernel | Smoothness | Numerical Stability |
|---|--------|------------|---------------------|
| 0.5 | Exponential | C⁰ (rough) | Poor |
| 1.5 | Matérn-3/2 | C² | Good |
| **2.5** | **Matérn-5/2** | **C⁴** | **Excellent** |
| ∞ | Gaussian RBF | C^∞ | Excellent |

### Closed-Form Matérn Kernels

```python
def matern_kernel(r, length_scale, nu=2.5):
    """
    Matérn kernel - standard choice in Gaussian Processes.

    nu=0.5: Exponential (Laplacian)
    nu=1.5: Once differentiable
    nu=2.5: Twice differentiable (recommended)
    nu→∞:   Gaussian RBF
    """
    scaled = r / length_scale

    if nu == 0.5:
        return np.exp(-scaled)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3) * scaled
        return (1 + sqrt3) * np.exp(-sqrt3)
    elif nu == 2.5:
        sqrt5 = np.sqrt(5) * scaled
        return (1 + sqrt5 + sqrt5**2/3) * np.exp(-sqrt5)
    else:  # Gaussian limit
        return np.exp(-scaled**2 / 2)
```

### Why Matérn-5/2?

1. **C⁴ smoothness** — twice differentiable, physically realistic
2. **Closed form** — no special functions needed
3. **Well-conditioned** — eigenvalues decay gracefully
4. **Industry standard** — default in scikit-learn, GPy, GPflow

## Proposed Implementation

### Configurable Smoothing Methods

```python
class SmoothingMethod(Enum):
    FFT_1D = "fft_1d"           # Current approach
    GAUSSIAN_RBF = "gaussian"   # exp(-r²/2σ²)
    MATERN_32 = "matern_32"     # (1+√3r/ℓ)exp(-√3r/ℓ)
    MATERN_52 = "matern_52"     # (1+√5r/ℓ+5r²/3ℓ²)exp(-√5r/ℓ)
    LAPLACIAN = "laplacian"     # exp(-r/ξ) - use with caution
    GRAPH_LAPLACIAN = "graph"   # Full spectral decomposition

class KernelSmoothingProjection:
    def __init__(
        self,
        method: SmoothingMethod = SmoothingMethod.MATERN_52,
        length_scale: float = 1.0,
        regularization: float = 1e-6,
    ):
        self.method = method
        self.length_scale = length_scale
        self.regularization = regularization

    def compute_kernel_matrix(self, centroids):
        """Compute kernel matrix from cluster centroids."""
        distances = pairwise_distances(centroids)

        if self.method == SmoothingMethod.GAUSSIAN_RBF:
            K = np.exp(-distances**2 / (2 * self.length_scale**2))
        elif self.method == SmoothingMethod.MATERN_52:
            scaled = np.sqrt(5) * distances / self.length_scale
            K = (1 + scaled + scaled**2/3) * np.exp(-scaled)
        elif self.method == SmoothingMethod.MATERN_32:
            scaled = np.sqrt(3) * distances / self.length_scale
            K = (1 + scaled) * np.exp(-scaled)
        # ... other methods

        # Add regularization for numerical stability
        K += self.regularization * np.eye(len(K))

        return K

    def smooth(self, W_matrices, centroids):
        """Smooth W matrices using kernel-weighted averaging."""
        K = self.compute_kernel_matrix(centroids)
        K_normalized = K / K.sum(axis=1, keepdims=True)
        return K_normalized @ W_matrices
```

### Comparison to FFT

| Aspect | 1D FFT | Kernel Smoothing |
|--------|--------|------------------|
| Captures structure | 1D only | Full d-dimensional |
| Complexity | O(N log N) | O(N²) for dense kernel |
| Numerical stability | Good | Depends on kernel |
| Hyperparameters | cutoff frequency | length_scale, kernel type |
| Theory | Signal processing | RKHS, Green's functions |

## Future Directions

### 1. Sparse Kernel Approximations

For large N, O(N²) kernel matrix is expensive. Options:
- **Nyström approximation**: Low-rank approximation using subset
- **Random Fourier features**: Approximate kernel with random projections
- **Inducing points**: Sparse GP methods

**Small Length Scale Optimization:**

When the length scale is small, the kernel decays rapidly with distance:

```
K(r) ≈ 0  for r >> length_scale
```

This means only **nearby clusters** contribute significantly to smoothing. We can exploit this:

1. **Sparse kernel matrix**: Set K(i,j) = 0 for distant cluster pairs
2. **k-NN approximation**: Only consider k nearest neighbors in the spanning tree
3. **Threshold pruning**: Zero out entries below ε (e.g., K < 0.01)

Computational benefit:
```
Dense kernel:  O(N²) storage, O(N²) multiply
Sparse (k-NN): O(Nk) storage, O(Nk) multiply
```

For our benchmark results showing optimal length_scale ≈ 0.5, with typical centroid
distances of 1-2, this means ~3-5 effective neighbors per cluster - highly sparse!

**Iterative Solvers for Coupled Constraints:**

When solving the full system (I + λL)W = W_target with inter-dependent constraints:

1. **Sparse Laplacian**: L has O(Nk) non-zeros when using k-NN graph
2. **Conjugate Gradient**: Converges in O(√κ) iterations where κ = condition number
3. **Preconditioning**: Jacobi or incomplete Cholesky reduces iterations further

For well-conditioned graph Laplacians with moderate λ, typically 10-20 iterations suffice.
Each iteration is O(Nk) for sparse matrix-vector product, giving total complexity O(Nk × iters).

This can outperform FFT's O(N log N) when k is small and the system is well-conditioned.

### 2. Learned Length Scales

Instead of fixed length_scale:
- **Per-dimension**: Different scales for different embedding directions
- **Automatic relevance determination (ARD)**: Learn from data
- **Deep kernels**: Neural network + kernel

### 3. Hierarchical Smoothing

Combine multiple scales:
```
W = W_local + W_regional + W_global
```

Using kernels at different length scales.

## Summary

| Concept | Role in Smoothing |
|---------|-------------------|
| Graph Laplacian | Principled regularization operator |
| Green's function | Solution kernel, inverse of Laplacian |
| Spherical harmonics | Eigenfunctions on sphere (angular) |
| Bessel/Hankel | Eigenfunctions for radial problems |
| Matérn kernel | Numerically stable kernel family |
| RKHS | Theoretical foundation for kernel methods |

**Recommendation:** Start with **Matérn-5/2 kernel smoothing** as a drop-in replacement for 1D FFT. It's theoretically principled, numerically stable, and captures multi-dimensional similarity structure.

## References

1. **Spectral Graph Theory**: Chung, F. (1997). "Spectral Graph Theory."
2. **Green's Functions**: Stakgold, I. (1998). "Green's Functions and Boundary Value Problems."
3. **Gaussian Processes**: Rasmussen & Williams (2006). "Gaussian Processes for Machine Learning."
4. **Matérn Kernels**: Stein, M. (1999). "Interpolation of Spatial Data: Some Theory for Kriging."
5. **Kernel Methods**: Schölkopf & Smola (2002). "Learning with Kernels."
6. **Spherical Harmonics**: Driscoll & Healy (1994). "Computing Fourier Transforms on the 2-Sphere."
