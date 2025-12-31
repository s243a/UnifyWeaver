# Proposals for Improving Coupled Kernel-Basis Optimization

**Status:** Proposed
**Date:** 2025-12-24
**Related:** [KERNEL_SMOOTHING_THEORY.md](KERNEL_SMOOTHING_THEORY.md)

## Problem Statement

The `UnifiedKernelBasisProjection` uses coupled optimization:

```
L = Σ_i ||Q_i W_i - A_i||² + λ Σ_{i,j} K(i,j) ||W_i - W_j||²
    └─── Data fidelity ───┘   └──── Laplacian smoothness ────┘
```

**Current results (hash embeddings, 20 clusters):**
| Approach | MRR |
|----------|-----|
| Sequential Hybrid | 0.5124 |
| Coupled CG | 0.3519 |

The coupled approach underperforms because the two loss terms fight during optimization.

## Root Cause Analysis

1. **Gradient conflict**: Data fidelity gradients point toward per-cluster optima; Laplacian gradients point toward the weighted average.

2. **Bad local minima**: When λ is set at the start, the Laplacian dominates early training, preventing the basis from learning discriminative features.

3. **Uniform regularization**: All clusters get the same λ, but sparse clusters need more regularization than dense ones.

## Proposal 1: Curriculum Learning (λ-Annealing)

### Concept
Start with λ=0 (pure data fidelity), gradually increase to target λ.

### Implementation

```python
def train_with_curriculum(
    self,
    clusters,
    target_lambda: float = 0.1,
    warmup_iters: int = 20,
    schedule: str = "linear"  # or "exponential", "cosine"
):
    for iteration in range(num_iterations):
        progress = min(1.0, iteration / warmup_iters)

        if schedule == "linear":
            current_lambda = target_lambda * progress
        elif schedule == "exponential":
            current_lambda = target_lambda * (1 - np.exp(-3 * progress))
        elif schedule == "cosine":
            current_lambda = target_lambda * (1 - np.cos(np.pi * progress / 2))

        self.smoothing_strength = current_lambda
        # ... CG step with current_lambda
```

### Expected Benefit
- Basis learns discriminative features first (λ=0 phase)
- Smoothing refines without destroying structure (λ>0 phase)
- Gradual transition avoids sudden gradient conflicts

### Hyperparameters
- `warmup_iters`: 20-50% of total iterations
- `schedule`: Linear is simple; exponential/cosine may be smoother

## Proposal 2: Warm-Start from Sequential Solution

### Concept
Initialize coupled optimization from the already-trained sequential hybrid.

### Implementation

```python
def train_warm_start(self, clusters, fine_tune_iters: int = 10):
    # Phase 1: Train sequential hybrid (known to work well)
    hybrid = KernelSmoothedBasisProjection(
        num_basis=self.num_basis,
        length_scale=self.length_scale,
        kernel_blend=0.5,
    )
    hybrid.train(clusters)

    # Phase 2: Initialize from hybrid's solution
    self.basis = [G.copy() for G in hybrid.basis_proj.basis]
    self.alpha = hybrid.basis_proj.alpha.copy()

    # Phase 3: Fine-tune with coupled CG
    self.smoothing_strength = 0.01  # Much smaller than default
    for _ in range(fine_tune_iters):
        self._coupled_cg_step(clusters)
```

### Expected Benefit
- Starts from a high-quality solution (MRR 0.51)
- Coupling only makes small adjustments
- Avoids bad local minima

### Risk
- May not improve if sequential solution is already optimal
- Extra training time (sequential + fine-tune)

## Proposal 3: Adaptive Per-Cluster λ

### Concept
Sparse clusters need more regularization; dense clusters need less.

### Implementation

```python
def compute_adaptive_lambda(self, clusters) -> np.ndarray:
    lambdas = []
    for Q, A in clusters:
        n_samples = Q.shape[0]
        # More samples → less regularization needed
        # Inverse sqrt scaling (common in regularization)
        cluster_lambda = self.base_lambda / np.sqrt(max(1, n_samples))
        lambdas.append(cluster_lambda)
    return np.array(lambdas)

# In the Laplacian term:
# Instead of: λ Σ_{i,j} K(i,j) ||W_i - W_j||²
# Use:        Σ_{i,j} λ_i K(i,j) ||W_i - W_j||²
```

### Expected Benefit
- Dense clusters (many questions) fit their data without over-smoothing
- Sparse clusters (1-2 questions) borrow heavily from neighbors
- Adapts automatically to data distribution

### Variants
- `λ_i = λ_0 / sqrt(n_i)` — inverse sqrt scaling
- `λ_i = λ_0 / n_i` — inverse linear (stronger effect)
- `λ_i = λ_0 * exp(-n_i / τ)` — exponential decay

## Proposal 4: Residual Coupling

### Concept
Instead of smoothing W directly, smooth the *deviation* from a baseline.

### Formulation

```
W_i = W_baseline_i + ΔW_i

Loss = Σ_i ||Q_i(W_baseline_i + ΔW_i) - A_i||²
     + λ Σ_{i,j} K(i,j) ||ΔW_i - ΔW_j||²
```

### Baseline Options
1. **Per-cluster pseudoinverse**: W_baseline = pinv(Q) @ A
2. **Global average**: W_baseline = mean of all per-cluster W
3. **FFT-smoothed**: W_baseline from FFTSmoothingProjection
4. **Kernel-smoothed**: W_baseline from KernelSmoothingProjection

### Implementation

```python
def train_residual_coupled(self, clusters):
    # Step 1: Compute baseline
    W_baseline = []
    for Q, A in clusters:
        W, _, _, _ = np.linalg.lstsq(Q, A, rcond=None)
        W_baseline.append(W)

    # Step 2: Optimize residual ΔW with Laplacian coupling
    # ΔW_i = Σ_k α_ik G_k
    # Regularize: λ Σ_{i,j} K(i,j) ||ΔW_i - ΔW_j||²

    # The basis G and coefficients α are for the RESIDUAL
```

### Expected Benefit
- Baseline captures most of the signal
- Laplacian only affects the deviation, preserving per-cluster structure
- Similar to ResidualBasisProjection, but with kernel coupling

## Proposal 5: Alternating Projection

### Concept
Instead of jointly optimizing, alternate between:
1. Per-cluster optimal α (ignoring Laplacian)
2. Project α onto Laplacian-smooth manifold

### Implementation

```python
def train_alternating_projection(self, clusters, blend: float = 0.5):
    for iteration in range(num_iterations):
        # Step 1: Per-cluster optimal (closed-form, no Laplacian)
        alpha_optimal = np.zeros_like(self.alpha)
        for i, (Q, A) in enumerate(clusters):
            alpha_optimal[i] = solve_for_alpha(Q, A, self.basis)

        # Step 2: Kernel-smooth the coefficients
        K_normalized = self.K_matrix / self.K_matrix.sum(axis=1, keepdims=True)
        alpha_smooth = np.zeros_like(self.alpha)
        for k in range(self.num_basis):
            alpha_smooth[:, k] = K_normalized @ alpha_optimal[:, k]

        # Step 3: Blend
        self.alpha = (1 - blend) * alpha_optimal + blend * alpha_smooth

        # Step 4: Update basis via gradient descent
        self._update_basis(clusters)
```

### Expected Benefit
- Each step is simple and interpretable
- No complex CG solver needed
- `blend` parameter controls smoothness explicitly
- Similar in spirit to proximal gradient methods

## Proposal 6: Multi-Scale Coupling

### Concept
Apply different λ at different scales (like hierarchical smoothing).

### Implementation

```python
def train_multiscale(self, clusters, lambdas=[0.01, 0.05, 0.1], length_scales=[0.5, 1.0, 2.0]):
    # Build kernels at different scales
    kernels = [self._compute_kernel(ls) for ls in length_scales]

    # Combined Laplacian: λ_1 L_1 + λ_2 L_2 + λ_3 L_3
    L_combined = sum(lam * self._graph_laplacian(K)
                     for lam, K in zip(lambdas, kernels))

    # Use L_combined in CG solver
```

### Expected Benefit
- Small length scale: smooth nearby clusters strongly
- Large length scale: weak global coherence
- Captures both local and global structure

## Proposal 7: Adaptive Conditioning-Based λ (Bisection)

### Concept

**This is an Adaptive Hybrid approach, not unified optimization.**

Instead of choosing λ as a hyperparameter, derive it from the conditioning of the least squares problem. For k basis vectors, we need at least k significant singular values. Use bisection to find the minimum λ that achieves a target condition number.

### Key Insight

For k basis vectors, Q^T Q is only k×k. Computing its condition number is O(k³) — essentially free for small k.

```python
def effective_condition(Q, lam):
    """Condition of regularized normal equations. O(m×k² + k³)."""
    QTQ = Q.T @ Q  # k×k matrix
    regularized = QTQ + lam * np.eye(QTQ.shape[0])
    s = np.linalg.svd(regularized, compute_uv=False)
    return s[0] / s[-1]
```

### Implementation

```python
def find_min_lambda(Q, target_cond=50, tol=1e-3, max_iter=20):
    """Bisect to find minimum λ achieving target condition number."""

    # Check if already well-conditioned
    if effective_condition(Q, 0) <= target_cond:
        return 0.0  # No regularization needed

    # Find upper bound where condition is satisfied
    lam_high = 1.0
    while effective_condition(Q, lam_high) > target_cond:
        lam_high *= 2
        if lam_high > 1e6:
            return lam_high  # Extremely ill-conditioned

    # Bisect to find minimum sufficient λ
    lam_low = 0.0
    for _ in range(max_iter):
        lam_mid = (lam_low + lam_high) / 2
        if effective_condition(Q, lam_mid) > target_cond:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

        if lam_high - lam_low < tol:
            break

    return lam_high
```

### Per-Cluster Adaptive Regularization

```python
def train_adaptive(self, clusters, target_cond=50):
    for i, (Q, A) in enumerate(clusters):
        # Find minimum λ for this cluster
        lam_i = find_min_lambda(Q, target_cond)

        # Solve regularized least squares
        QTQ = Q.T @ Q
        regularized = QTQ + lam_i * np.eye(QTQ.shape[0])
        W_i = np.linalg.solve(regularized, Q.T @ A)
```

### For Laplacian Smoothing (Cross-Cluster)

Same principle, but regularize with Laplacian instead of identity:

```python
regularized = QTQ + lam * L  # L = graph Laplacian
```

### Computational Cost

| Step | Cost |
|------|------|
| Form Q^T Q | O(m × k²) |
| SVD per bisection step | O(k³) |
| Total per cluster | O(m × k² + k³ × log(1/tol)) |

For k=4 and 20 iterations: essentially O(m × 16) per cluster.

### Expected Benefit

- **No hyperparameter tuning**: λ derived from data
- **Minimum regularization**: Only as much smoothing as needed
- **Per-cluster adaptation**: Ill-conditioned clusters get more, well-conditioned get less
- **Principled**: Based on numerical stability, not heuristics

### Comparison to Proposal 3

Proposal 3 uses sample count as a proxy for conditioning. This proposal uses conditioning directly — sample count can be misleading (100 nearly-identical questions are still ill-conditioned).

## Recommended Experiments

### Priority 1: Adaptive Conditioning (Proposal 7)
- Most principled approach
- No hyperparameter to tune (just target condition number)
- Computationally cheap for small k
- Can be combined with other proposals

### Priority 2: Curriculum Learning
- Simple to implement
- Low risk, likely to help
- Test with linear schedule first

### Priority 3: Warm-Start
- Uses existing working code
- Guaranteed to start from good solution
- May achieve best of both worlds

### Priority 4: Alternating Projection
- Interpretable, no CG needed
- Good for understanding the optimization landscape
- `blend` is intuitive hyperparameter

## Implementation Plan

1. **Implement Proposal 7** as `AdaptiveConditioningProjection`
2. Add `curriculum_warmup` parameter to `UnifiedKernelBasisProjection.__init__`
3. Add `warm_start` flag that initializes from sequential hybrid
4. Benchmark on hash embeddings first (fast iteration)
5. Test winner on ModernBERT

## References

1. **Curriculum Learning**: Bengio et al. (2009). "Curriculum Learning."
2. **Warm-Starting**: Fine-tuning from pretrained models, transfer learning
3. **Adaptive Regularization**: Bayesian approaches, ARD
4. **Proximal Methods**: Parikh & Boyd (2014). "Proximal Algorithms."
