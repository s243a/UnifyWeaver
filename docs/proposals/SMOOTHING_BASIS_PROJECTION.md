# Proposal: Smoothing Basis for Multi-Cluster Projection

**Status:** Proposal
**Version:** 0.2
**Date:** 2025-12-15
**Extends:** [SMOOTHNESS_REGULARIZATION.md](SMOOTHNESS_REGULARIZATION.md), [MULTI_HEAD_PROJECTION_THEORY.md](MULTI_HEAD_PROJECTION_THEORY.md)

## Executive Summary

When question data is sparse (few examples per cluster), we can leverage **answer similarity** to regularize projections. However, naive smoothness regularization is computationally expensive - coupling N cluster matrices creates an effectively giant optimization problem.

This proposal describes a **gradient-based smoothing basis** approach: use gradient directions as the basis, with simple dot-product projections. This reduces parameters from O(N × d²) to O(N × K + K × d²) while enabling differential optimization with richer loss functions.

The approach is simple enough to implement and test.

## Motivation

### The Sparse Question Problem

Multi-head LDA works best with sufficient questions per cluster:
- 5+ questions: stable centroid estimation
- 10+ questions: reliable projection learning
- 2-3 questions: noisy, potentially overfitting

But collecting many question variants is expensive.

### Answer-Driven Regularization

Key insight: **answers are often easier to obtain than questions**.

- Documentation exists (answers)
- User queries are sparse (questions)
- Related answers share structure

If cluster A (about JWT) and cluster B (about OAuth) have similar answers, their projections should be similar - even if each has few training questions.

### Computational Complexity

**Naive smoothness regularization:**

```
Loss = Σ_i ||Q_i W_i - A_i||² + λ Σ_{i,j} S_ij ||W_i - W_j||²
```

Where S_ij is answer similarity. This couples all N matrices:
- N clusters × d × d parameters
- For d=384, N=100: 14.7M parameters
- Optimization sees one giant coupled system

## The Smoothing Basis Approach

### Core Idea

Instead of independent W_i matrices, express each as a linear combination of shared basis matrices:

```
W_i = Σ_{k=1}^{K} α_ik × B_k
```

Where:
- B_k: shared basis matrices (K of them, K << N)
- α_ik: per-cluster coefficients
- W_i: cluster i's effective projection

### Parameter Reduction

| Approach | Parameters | Example (d=384, N=100, K=8) |
|----------|------------|----------------------------|
| Independent | N × d² | 14.7M |
| Smoothing basis | N×K + K×d² | 1.18M |
| Reduction | | 12.5x |

### Implicit Smoothness

The basis provides smoothness automatically:
- Clusters share the same basis matrices
- Only coefficients α_ik vary per cluster
- Similar clusters → similar coefficients → similar projections

No explicit regularization term needed (or smaller λ).

### Enables Differential Optimization

The parameter reduction makes gradient-based optimization practical:

**Current approach (pseudoinverse):**
- Closed-form: W = pinv(Q) @ A
- Fast, but limited to MSE loss
- Hard to incorporate other objectives

**With basis approach:**
- 12x fewer parameters → gradient descent tractable
- Can optimize richer objectives:

```python
loss = (
    mse_weight * mse_loss(pred, target) +
    cosine_weight * (1 - cosine_sim(pred, target)) +  # Essential per transformer work
    sparsity_weight * L1(alpha) +                      # Sparse coefficient selection
    smoothness_weight * answer_similarity_penalty      # Explicit smoothness if needed
)
loss.backward()  # Now tractable with reduced parameters
```

This opens up:
- **Cosine loss**: Critical for directional alignment (learned from transformer distillation)
- **Coefficient sparsity**: Automatic basis selection
- **Multi-objective optimization**: Balance multiple criteria
- **Online/incremental updates**: SGD-style training on streaming data

## Gradient-Based Basis Formulation

### Core Idea: Gradients as Basis

Instead of learning arbitrary basis matrices, use **gradient directions** from the loss landscape. Gradients point toward useful directions - where the loss decreases.

```
G_k = gradient direction k (d × d matrix)
W_i = Σ_k α_ik G_k
```

### Projection via Dot Product

The coefficient α_ik is simply the Frobenius inner product:

```
α_ik = ⟨W_i, G_k⟩_F = trace(W_i^T G_k) = Σ_mn W_i[m,n] G_k[m,n]
```

The projection onto basis element G_k:
```
proj_{G_k}(W_i) = α_ik × G_k = ⟨W_i, G_k⟩_F × G_k
```

This is a rank-1 operation - O(d²) not O(d³).

### Constrained Optimization with Lagrangian

**Objective:**
```
minimize   Σ_i L_i(W_i)
where      L_i = (1-λ)||Q_i W_i - A_i||²_F + λ(1 - cos_sim(Q_i W_i, A_i))
```

**Constraints:**
```
W_i = Σ_k α_ik G_k         (basis constraint)
||G_k||_F = 1              (normalized basis)
⟨G_j, G_k⟩_F = 0 for j≠k   (orthogonal basis, optional)
```

**Lagrangian:**
```
𝓛 = Σ_i L_i(W_i)
  + Σ_i μ_i · (W_i - Σ_k α_ik G_k)     # basis constraint
  + Σ_k ν_k · (||G_k||²_F - 1)          # normalization
  + Σ_{j<k} ρ_jk · ⟨G_j, G_k⟩_F         # orthogonality (optional)
```

### Algorithm: Alternating Optimization

Key insight from optimization theory: **at the optimum, all derivatives are zero**.

For our constrained problem, this means the gradient projected onto the tangent plane equals zero:
```
⟨∇_W L, G_k⟩_F = 0    for all k
```

But this only gives directions, not magnitudes. The magnitudes come from **least squares**:

Given basis G_k fixed, the optimal coefficients α_i for cluster i satisfy:
```
Σ_k α_ik (Q_i @ G_k) ≈ A_i
```

This is a least squares problem with closed-form solution!

**Alternating optimization:**
1. Fix basis → solve for coefficients (closed-form least squares)
2. Fix coefficients → update basis (gradient descent)

```python
def train_smoothing_basis(clusters, K, num_iterations, lr):
    """
    Train projection matrices using gradient-based smoothing basis.

    Uses alternating optimization:
    - Coefficients: closed-form least squares (given basis)
    - Basis: gradient-based updates

    Args:
        clusters: List of (Q_i, A_i) question/answer embedding pairs
        K: Number of basis directions
        num_iterations: Training iterations
        lr: Learning rate for basis updates
    """
    d = clusters[0][0].shape[1]  # embedding dimension
    N = len(clusters)

    # Initialize: compute per-cluster solutions, extract basis from gradients
    W_init = [compute_initial_W(Q, A) for Q, A in clusters]
    all_grads = [compute_gradient(W_init[i], clusters[i]) for i in range(N)]
    G = extract_orthogonal_basis(all_grads, K)

    # Initialize coefficients via least squares
    alpha = np.zeros((N, K))
    for i in range(N):
        alpha[i] = solve_for_alpha(clusters[i][0], clusters[i][1], G)

    # Alternating optimization
    for iteration in range(num_iterations):

        # Step 1: Fix basis, solve for coefficients (closed form)
        for i, (Q_i, A_i) in enumerate(clusters):
            alpha[i] = solve_for_alpha(Q_i, A_i, G)

        # Step 2: Fix coefficients, update basis via gradient
        total_grad = [np.zeros_like(G[k]) for k in range(K)]
        for i, (Q_i, A_i) in enumerate(clusters):
            W_i = sum(alpha[i, k] * G[k] for k in range(K))
            grad_W = compute_gradient(W_i, Q_i, A_i)

            # Accumulate gradient for each basis element
            for k in range(K):
                total_grad[k] += alpha[i, k] * grad_W

        # Update basis
        for k in range(K):
            G[k] -= lr * total_grad[k]

        # Re-orthogonalize basis periodically
        if iteration % 10 == 0:
            G = reorthogonalize(G)

        # Check convergence: gradient projected onto tangent plane ≈ 0
        if iteration % log_interval == 0:
            max_tangent_grad = max(
                abs(frobenius_inner(compute_gradient(
                    sum(alpha[i, k] * G[k] for k in range(K)),
                    clusters[i][0], clusters[i][1]
                ), G[k]))
                for i in range(N) for k in range(K)
            )
            total_loss = sum(compute_loss(alpha[i], G, clusters[i]) for i in range(N))
            print(f"Iter {iteration}: loss={total_loss:.6f}, max_tangent_grad={max_tangent_grad:.6f}")

    return alpha, G


def solve_for_alpha(Q_i, A_i, basis):
    """
    Given basis, solve for optimal coefficients via least squares.

    We want: Σ_k α_k (Q_i @ G_k) ≈ A_i

    This is linear in α, so we can solve directly.
    """
    K = len(basis)

    # Project each basis through Q_i
    # P[:, k] = vec(Q_i @ G_k)
    P = np.column_stack([(Q_i @ G_k).ravel() for G_k in basis])

    # Least squares: P @ α ≈ vec(A_i)
    alpha, residuals, rank, s = np.linalg.lstsq(P, A_i.ravel(), rcond=None)

    return alpha


def frobenius_inner(A, B):
    """Frobenius inner product: ⟨A, B⟩_F = trace(A^T B)"""
    return np.sum(A * B)


def extract_orthogonal_basis(gradients, K):
    """Extract K orthogonal basis directions from gradients via Gram-Schmidt."""
    basis = []
    for g in gradients[:K]:
        # Orthogonalize against existing basis
        g_orth = g.copy()
        for b in basis:
            g_orth -= frobenius_inner(g_orth, b) * b
        # Normalize
        norm = np.sqrt(frobenius_inner(g_orth, g_orth))
        if norm > 1e-8:
            basis.append(g_orth / norm)
    return basis


def compute_gradient(W, Q, A, cosine_weight=0.5):
    """Compute gradient of combined MSE + cosine loss."""
    pred = Q @ W

    # MSE gradient
    grad_mse = 2 * Q.T @ (pred - A) / len(Q)

    # Cosine gradient (simplified)
    pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    grad_cos = -Q.T @ A_norm / len(Q)  # Simplified; full version more complex

    return (1 - cosine_weight) * grad_mse + cosine_weight * grad_cos
```

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Frobenius inner product | O(d²) | Just element-wise multiply + sum |
| Reconstruct W_i | O(K × d²) | K scalar-matrix multiplies |
| Gradient computation | O(n_i × d²) | Standard backprop |
| Basis update | O(K² × d²) | Gram-Schmidt orthogonalization |
| **Total per iteration** | O(N × K × d²) | Linear in clusters and basis size |

Compare to naive: O(N × d³) for coupled matrix optimization.

## Theoretical Connections

### Connection to Mixture of Experts

The basis approach resembles MoE:
```
MoE:   output = Σ_k g_k(x) × expert_k(x)
Basis: W_i = Σ_k α_ik × B_k
```

Difference: MoE gates are input-dependent, basis coefficients are cluster-fixed.

### Connection to Matrix Factorization

If we factorize each basis B_k = U_k V_k^T:
```
W_i = Σ_k α_ik U_k V_k^T
```

This resembles collaborative filtering / matrix completion.

### Connection to Transfer Learning

The shared basis captures "common projection knowledge" while coefficients capture "cluster-specific adaptation". Similar to:
- Fine-tuning (shared base + task head)
- Adapters (shared transformer + small per-task adapters)

## Validation Plan

### 1. Synthetic Experiments

Create synthetic data with known structure:
```python
# Generate K "true" basis matrices
true_basis = [random_orthogonal_matrix(d) for _ in range(K)]

# Generate N clusters as combinations
for i in range(N):
    true_alpha = random_sparse_coefficients(K)
    W_true[i] = sum(true_alpha[k] * true_basis[k] for k in range(K))
    Q_i = random_queries(n_i, d)
    A_i = Q_i @ W_true[i] + noise

# Test: can algorithm recover true_basis and true_alpha?
```

### 2. Comparison to Baselines

| Method | Parameters | Expected Quality |
|--------|------------|------------------|
| Independent W_i (pseudoinverse) | N × d² | Best fit, may overfit |
| Independent W_i (gradient) | N × d² | Similar, slower |
| Smoothing basis (this proposal) | N×K + K×d² | Slightly worse fit, better generalization |
| Single shared W | d² | Underfits |

### 3. Real Data Test

Use existing LDA database:
```bash
python scripts/train_smoothing_basis.py \
    --db playbooks/lda-training-data/lda.db \
    --num-basis 8 \
    --iterations 1000 \
    --cosine-weight 0.7
```

Compare Recall@1 on held-out queries vs multi-head baseline.

## Relationship to Existing Work

### vs. SMOOTHNESS_REGULARIZATION.md

| Aspect | Graph Laplacian | Smoothing Basis |
|--------|-----------------|-----------------|
| Approach | Explicit penalty term | Structural constraint |
| Parameters | Still N × d² | Reduced to N×K + K×d² |
| Smoothness source | Regularization loss | Shared basis |
| Complexity | High (coupled optimization) | Lower (factored) |

The basis approach may subsume explicit smoothness regularization.

### vs. TRANSFORMER_DISTILLATION.md

Transformer distillation compresses N heads into one network.
Smoothing basis keeps N clusters but shares structure.

Could combine: basis projection → transformer distillation.

## Implementation Roadmap

### Phase 1: Core Implementation
- [ ] Implement `frobenius_inner()` and `extract_orthogonal_basis()`
- [ ] Implement `compute_gradient()` with MSE + cosine loss
- [ ] Implement main `train_smoothing_basis()` loop
- [ ] Add to `src/unifyweaver/targets/python_runtime/smoothing_basis.py`

### Phase 2: Synthetic Validation
- [ ] Create synthetic test with known basis structure
- [ ] Verify algorithm recovers true basis and coefficients
- [ ] Test sensitivity to K (number of basis matrices)
- [ ] Test sensitivity to noise levels

### Phase 3: Real Data Testing
- [ ] Create `scripts/train_smoothing_basis.py`
- [ ] Test on LDA database with sparse clusters
- [ ] Compare Recall@1 vs multi-head baseline
- [ ] Measure computational performance

### Phase 4: Integration
- [ ] Update database schema for basis storage
- [ ] Add basis loading to Rust/Go runtimes
- [ ] Documentation and examples

## Open Questions

While the formulation is concrete, some questions remain:

1. **Optimal K**: How to choose number of basis matrices?
   - Start with K = sqrt(N) as heuristic
   - Cross-validation for tuning

2. **Basis update frequency**: How often to re-orthogonalize?
   - Every iteration? Every 10? Once at start?

3. **Initialization**: Which gradients to use initially?
   - Random subset of clusters?
   - Clusters with most questions?
   - All clusters, then select K most different?

4. **Convergence**: What are the convergence guarantees?
   - Lagrangian methods have known convergence properties
   - Need to verify empirically

## Summary

The gradient-based smoothing basis approach offers:

- **Parameter efficiency**: 12x+ reduction (1.18M vs 14.7M for d=384, N=100, K=8)
- **Enables differential optimization**: Gradient descent with MSE + cosine loss
- **Simple formulation**: Dot product projections, Lagrangian constraints
- **Implicit regularization**: Shared basis provides smoothness structurally
- **Answer-driven learning**: Leverage answer similarity when questions are sparse
- **Computational efficiency**: O(N × K × d²) per iteration vs O(N × d³) naive

The algorithm is concrete enough to implement and test. Key hyperparameters:
- K: Number of basis matrices (start with sqrt(N))
- cosine_weight: Balance MSE vs directional alignment (0.5-0.7)
- lr: Learning rate for coefficient updates
- basis_update_interval: How often to refresh basis from gradients

## References

1. **Matrix Factorization**: Koren et al. (2009). "Matrix Factorization Techniques for Recommender Systems."
2. **Mixture of Experts**: Shazeer et al. (2017). "Outrageously Large Neural Networks."
3. **Adapters**: Houlsby et al. (2019). "Parameter-Efficient Transfer Learning for NLP."
4. **Low-Rank Adaptation**: Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models."
