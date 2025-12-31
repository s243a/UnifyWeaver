# Smoothness Regularization for Multi-Head Projection

**Status:** Proposal
**Version:** 0.1
**Date:** 2025-12-14
**Extends:** [MULTI_HEAD_PROJECTION_THEORY.md](MULTI_HEAD_PROJECTION_THEORY.md)

## Executive Summary

As the Q-A database grows, semantically similar answers (e.g., API variants, version differences) may cluster together with sparse question coverage per answer. This proposal introduces **smoothness regularization** that constrains projection parameters to vary continuously across semantic space—answers that are similar should have similar projections.

The key insight: when individual answer clusters are underdetermined (few questions), we can **borrow strength from semantically similar answers** by enforcing that projection parameters change smoothly as a function of answer similarity.

## Motivation

### The Sparse Cluster Problem

Consider a database with many similar answers:

```
Answer A: "Use sqlite3.connect('db.sqlite')"     → 2 training questions
Answer B: "Use psycopg2.connect('postgresql://')" → 1 training question
Answer C: "Use mysql.connector.connect()"        → 3 training questions
Answer D: "Use pymongo.MongoClient()"            → 2 training questions
```

Each answer has few training questions, making individual head training unreliable. But they're all "database connection" answers—semantically close in embedding space.

### Current Multi-Head Behavior

Without regularization, each head is trained independently:

```
Head_A: centroid_A, answer_A  (from 2 questions)
Head_B: centroid_B, answer_B  (from 1 question)
Head_C: centroid_C, answer_C  (from 3 questions)
Head_D: centroid_D, answer_D  (from 2 questions)
```

**Problems:**
1. **High variance**: Heads with few questions have unreliable centroids
2. **Inconsistent routing**: Similar queries may route differently to similar answers
3. **No information sharing**: Answer B's head learns nothing from A, C, D

### The Smoothness Intuition

If answers A and B are semantically similar (high cosine similarity), their projection behavior should also be similar. A query about "database connection" should be projected similarly whether it ends up matching A or B.

Mathematically: **the projection function should be Lipschitz continuous in answer space**.

### Underdetermined Projections

With few training questions per answer, the projection parameters are **underdetermined**—there isn't enough data to reliably estimate them. This leads to:

1. **High variance**: Small changes in training data cause large changes in projection
2. **Overfitting**: Projection memorizes the few training questions rather than generalizing
3. **Noise sensitivity**: Outlier questions disproportionately influence the centroid

Smoothness regularization addresses this by adding constraints: instead of freely fitting each head independently, we require that the solution varies smoothly across similar answers. This effectively **borrows statistical strength** from neighboring answers:

```
Few questions for answer A?
→ Borrow information from similar answers B, C, D
→ Projection for A is regularized toward their projections
→ Reduces variance, prevents overfitting
```

When you have abundant training data, the regularization has minimal effect (data dominates). When data is sparse, it prevents the projection from fitting noise.

## Mathematical Formulation

### Definitions

Let:
- A = {a₁, a₂, ..., aₙ} be answer embeddings (d-dimensional)
- Cₖ = centroid for answer k
- sim(aᵢ, aⱼ) = cosine similarity between answers i and j
- θₖ = projection parameters for head k (centroid + optional W matrix)

### Answer Similarity Graph

Construct a weighted graph G = (V, E, W) where:
- Vertices V = {1, 2, ..., n} (one per answer)
- Edge weight Wᵢⱼ = sim(aᵢ, aⱼ) if sim(aᵢ, aⱼ) > threshold, else 0

This creates a **semantic neighborhood structure** over answers.

### Graph Laplacian

The graph Laplacian L captures how parameters vary across the graph:

```
L = D - W

where:
  W = similarity matrix (Wᵢⱼ = sim(aᵢ, aⱼ))
  D = degree matrix (Dᵢᵢ = Σⱼ Wᵢⱼ)
```

For projection parameters θ (stacked as a matrix):

```
Smoothness penalty = trace(θᵀ L θ) = Σᵢⱼ Wᵢⱼ ||θᵢ - θⱼ||²
```

This penalizes **squared differences between parameters, weighted by answer similarity**.

### Regularized Training Objective

The original per-head training objective for head k:

```
L_fit(k) = Σᵢ ||Wₖ · qᵢ - aₖ||²
```

With smoothness regularization:

```
L_total = Σₖ L_fit(k) + λ_smooth · Σᵢⱼ sim(aᵢ, aⱼ) · ||θᵢ - θⱼ||²
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         smoothness penalty over answer graph
```

Where λ_smooth controls the regularization strength.

### Semantic Distance Metric

We use cosine similarity directly for the smoothness weight:

```
sim(aᵢ, aⱼ) = (aᵢ · aⱼ) / (||aᵢ|| · ||aⱼ||)
```

Alternative metrics:
- **Thresholded**: sim(aᵢ, aⱼ) if > τ, else 0 (sparse graph)
- **Exponential**: exp(-||aᵢ - aⱼ||² / σ²) (Gaussian kernel)
- **k-NN**: 1 if j ∈ k-nearest-neighbors(i), else 0 (sparse, symmetric after symmetrization)

### Interpretation

| Component | Meaning |
|-----------|---------|
| sim(aᵢ, aⱼ) | How similar are answers i and j? |
| \|\|θᵢ - θⱼ\|\|² | How different are their projections? |
| λ_smooth | Trade-off: fit training data vs. smooth across answers |

**Effect**: Heads with few training questions are "pulled toward" similar heads with more data.

## Differential Geometry Perspective

### Projection as a Vector Field

View the projection as a vector field over answer space:

```
P: ℝᵈ → ℝᵈ
P(a) = projection parameters for answer a
```

The smoothness constraint is a **regularity condition** on P:

```
||∇P|| ≤ κ   (gradient bounded)
```

In discrete form (our answer set):

```
||P(aᵢ) - P(aⱼ)|| ≤ κ · ||aᵢ - aⱼ||
```

This is **Lipschitz continuity**—the projection can't change faster than a constant times the semantic distance.

### Manifold Assumption

Answers likely lie on a lower-dimensional manifold in embedding space. The smoothness constraint respects this structure:

```
                    P(a₁)
                   ↗
          a₁ ----→ a₂ ----→ a₃    (answer manifold)
                   ↓
                    P(a₂) ≈ P(a₁)  (smooth projection)
```

## When to Apply Smoothness

### Criteria for Activation

Smoothness regularization is most valuable when:

1. **Sparse clusters**: Few questions per answer (< 5)
2. **Dense answer neighborhoods**: Many similar answers (sim > 0.8)
3. **Growing database**: New answers added incrementally

### Automatic Detection

```python
def should_use_smoothness(answers, questions_per_answer, similarity_threshold=0.8):
    """
    Determine if smoothness regularization would help.

    Returns:
        bool: True if smoothness regularization is recommended
        float: Suggested λ_smooth value
    """
    # Count sparse clusters
    sparse_count = sum(1 for q in questions_per_answer if q < 5)
    sparse_ratio = sparse_count / len(answers)

    # Count dense neighborhoods
    dense_pairs = 0
    for i in range(len(answers)):
        for j in range(i+1, len(answers)):
            if cosine_sim(answers[i], answers[j]) > similarity_threshold:
                dense_pairs += 1

    neighborhood_density = dense_pairs / (len(answers) * (len(answers)-1) / 2)

    # Recommend smoothness if both conditions met
    if sparse_ratio > 0.3 and neighborhood_density > 0.1:
        # Scale λ by sparsity
        lambda_smooth = 0.1 * sparse_ratio
        return True, lambda_smooth

    return False, 0.0
```

### Example Scenarios

| Scenario | Sparse Clusters | Dense Neighborhoods | Use Smoothness? |
|----------|-----------------|---------------------|-----------------|
| Well-covered topics | Low | Any | No |
| Diverse answers | Any | Low | No |
| API variants (sqlite/psycopg2/mysql) | High | High | **Yes** |
| Version documentation (v1/v2/v3) | High | High | **Yes** |
| Incremental database growth | High (new) | Variable | **Yes** for new |

## Implementation Approaches

### Approach 1: Joint Training

Train all heads jointly with smoothness penalty:

```python
def train_with_smoothness(clusters, answer_embeddings, lambda_smooth=0.1):
    """
    Train multi-head projection with smoothness regularization.
    """
    n_heads = len(clusters)

    # Compute answer similarity matrix
    W = np.zeros((n_heads, n_heads))
    for i in range(n_heads):
        for j in range(n_heads):
            W[i,j] = cosine_sim(answer_embeddings[i], answer_embeddings[j])

    # Graph Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Initialize centroids from individual clusters
    centroids = [compute_centroid(cluster) for cluster in clusters]
    centroids = np.stack(centroids)  # (n_heads, d)

    # Optimize with smoothness penalty
    # L_total = Σₖ L_fit(k) + λ · trace(Cᵀ L C)

    for iteration in range(max_iters):
        # Gradient of fit term (per-cluster)
        grad_fit = compute_fit_gradient(centroids, clusters)

        # Gradient of smoothness term
        # d/dC [trace(Cᵀ L C)] = 2 L C
        grad_smooth = 2 * L @ centroids

        # Combined gradient
        grad = grad_fit + lambda_smooth * grad_smooth

        # Update
        centroids = centroids - learning_rate * grad

    return centroids
```

### Approach 2: Post-Hoc Smoothing

Train heads independently, then smooth:

```python
def smooth_centroids(centroids, answer_embeddings, lambda_smooth=0.1, iterations=10):
    """
    Apply Laplacian smoothing to pre-trained centroids.
    """
    # Compute similarity-weighted Laplacian
    W = compute_similarity_matrix(answer_embeddings)
    D = np.diag(W.sum(axis=1))
    L_normalized = np.eye(len(W)) - np.linalg.inv(D) @ W

    # Iterative smoothing: C_new = (1-α)C + α(W/D)C
    alpha = lambda_smooth / (1 + lambda_smooth)

    smoothed = centroids.copy()
    for _ in range(iterations):
        smoothed = (1 - alpha) * centroids + alpha * (W @ smoothed) / D.diagonal()[:, None]

    return smoothed
```

### Approach 3: Hierarchical Smoothing

Group similar answers first, then smooth within groups:

```python
def hierarchical_smooth(centroids, answer_embeddings, n_groups=5):
    """
    Cluster answers, smooth within clusters.
    """
    from sklearn.cluster import KMeans

    # Group similar answers
    kmeans = KMeans(n_clusters=n_groups)
    groups = kmeans.fit_predict(answer_embeddings)

    smoothed = centroids.copy()

    # Smooth within each group
    for g in range(n_groups):
        mask = groups == g
        if mask.sum() > 1:
            # Average centroids within group (weighted by question count)
            group_centroids = centroids[mask]
            group_mean = group_centroids.mean(axis=0)

            # Pull toward group mean
            smoothed[mask] = 0.7 * centroids[mask] + 0.3 * group_mean

    return smoothed
```

### Approach 4: Shared + Delta Decomposition

Instead of independent heads, decompose projection into shared (domain) and specific (answer) components:

```
W_answer = W_domain + ΔW_answer
```

Where:
- **W_domain**: Shared projection for semantically similar answers (e.g., all database connections)
- **ΔW_answer**: Answer-specific refinement (e.g., sqlite-specific details)

```python
def train_shared_delta(clusters, answer_embeddings, domain_threshold=0.8):
    """
    Train with shared domain components + answer-specific deltas.
    """
    # Identify domains via clustering
    domains = cluster_by_similarity(answer_embeddings, threshold=domain_threshold)

    # Train shared W per domain
    W_domains = {}
    for domain_id, answer_indices in domains.items():
        # Pool all questions from domain
        domain_questions = concat([clusters[i] for i in answer_indices])
        domain_answers = mean([answer_embeddings[i] for i in answer_indices])

        W_domains[domain_id] = train_W(domain_questions, domain_answers)

    # Train per-answer deltas (residual from domain projection)
    deltas = {}
    for answer_idx, (questions, answer_emb) in enumerate(clusters):
        domain_id = get_domain(answer_idx, domains)
        W_domain = W_domains[domain_id]

        # Delta corrects what domain projection gets wrong
        residual = answer_emb - W_domain @ centroid(questions)
        deltas[answer_idx] = residual  # Simplified: just store residual vector

    return W_domains, deltas
```

**Advantages:**
- Domain component is well-determined (pooled data from multiple answers)
- Delta can be small/zero for sparse clusters (falls back to domain)
- Captures hierarchical structure: "database connection" → "sqlite specifically"

## Hyperparameter Selection

### λ_smooth Selection

| λ_smooth | Effect |
|----------|--------|
| 0 | No smoothing (independent heads) |
| 0.01 | Light smoothing (slight regularization) |
| 0.1 | Moderate smoothing (recommended starting point) |
| 1.0 | Strong smoothing (heavy regularization) |
| → ∞ | All heads collapse to global average |

### Cross-Validation Strategy

```python
def select_lambda(clusters, answer_embeddings, lambdas=[0, 0.01, 0.1, 0.5]):
    """
    Select λ_smooth via leave-one-out cross-validation.
    """
    best_lambda = 0
    best_recall = 0

    for lam in lambdas:
        recalls = []

        for held_out in range(len(clusters)):
            # Train on all but held_out
            train_clusters = [c for i, c in enumerate(clusters) if i != held_out]
            train_answers = [a for i, a in enumerate(answer_embeddings) if i != held_out]

            centroids = train_with_smoothness(train_clusters, train_answers, lam)

            # Test: can held_out questions find their answer?
            test_questions = clusters[held_out]
            test_answer = answer_embeddings[held_out]

            recall = evaluate_recall(test_questions, test_answer, centroids, train_answers)
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        if mean_recall > best_recall:
            best_recall = mean_recall
            best_lambda = lam

    return best_lambda
```

### Similarity Threshold

For sparse graphs, threshold the similarity matrix:

```python
def sparsify_similarity(W, threshold=0.7):
    """
    Zero out similarities below threshold.
    """
    W_sparse = W.copy()
    W_sparse[W_sparse < threshold] = 0
    return W_sparse
```

Lower threshold = denser graph = more smoothing across distant answers.

## Relationship to Softmax Routing

The softmax temperature τ in multi-head projection already provides **implicit smoothing at inference time**:

| Mechanism | When Applied | Effect |
|-----------|--------------|--------|
| Softmax (low τ) | Inference | Blend nearby heads by query similarity |
| Smoothness reg. | Training | Blend head parameters by answer similarity |

**Complementary effects:**
- Softmax handles query-side smoothing (query routes to similar centroids)
- Smoothness handles answer-side smoothing (similar answers have similar projections)

For well-covered clusters, softmax is sufficient. For sparse clusters, smoothness regularization provides additional stability.

## Computational Considerations

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Similarity matrix | O(n² · d) |
| Graph Laplacian | O(n²) |
| Smoothness gradient | O(n² · d) per iteration |
| Total training | O(n² · d · iterations) |

For n=1000 answers, d=384: manageable. For n=100,000: consider sparse approximations.

### Sparse Approximations

For large n, use k-NN graph instead of full similarity:

```python
def sparse_laplacian(answer_embeddings, k=10):
    """
    Compute k-NN sparse Laplacian.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(answer_embeddings)

    distances, indices = nn.kneighbors(answer_embeddings)

    # Build sparse similarity matrix
    W_sparse = lil_matrix((n, n))
    for i in range(n):
        for j, dist in zip(indices[i], distances[i]):
            W_sparse[i, j] = 1 - dist  # cosine similarity

    # Symmetrize
    W_sparse = (W_sparse + W_sparse.T) / 2

    return W_sparse.tocsr()
```

## Future Extensions

### Adaptive λ per Cluster

Instead of global λ, adapt based on cluster size:

```
λₖ = λ_base · (N_typical / Nₖ)

where Nₖ = number of questions in cluster k
```

Sparse clusters get more regularization.

### Temporal Smoothness

As the database evolves, enforce continuity over time:

```
L_temporal = λ_time · ||θ(t) - θ(t-1)||²
```

Prevents sudden jumps when retraining.

### Multi-Scale Smoothness

Apply different smoothness at different similarity scales:

```
L_smooth = λ_local · Σ (sim > 0.9) ||θᵢ - θⱼ||²
         + λ_global · Σ (sim > 0.7) ||θᵢ - θⱼ||²
```

Strong smoothing for very similar answers, weaker for moderately similar.

## Summary

Smoothness regularization addresses the sparse cluster problem in multi-head projection by constraining projection parameters to vary continuously in semantic space. The key ideas:

1. **Semantic distance matters**: Similar answers should have similar projections
2. **Graph Laplacian**: Encodes smoothness as a quadratic penalty
3. **Automatic detection**: Apply when clusters are sparse and neighborhoods are dense
4. **Complementary to softmax**: Softmax smooths at inference, regularization smooths at training

This becomes increasingly relevant as the Q-A database grows and accumulates similar answers with sparse individual coverage.

## References

1. **Belkin & Niyogi** (2003). "Laplacian Eigenmaps for Dimensionality Reduction and Data Representation." *Neural Computation*.
   - Graph Laplacian for manifold learning

2. **Zhu, Ghahramani & Lafferty** (2003). "Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions." *ICML*.
   - Label propagation via graph Laplacian

3. **Zhou et al.** (2004). "Learning with Local and Global Consistency." *NeurIPS*.
   - Smoothness on graphs for semi-supervised learning

4. **MULTI_HEAD_PROJECTION_THEORY.md** (this project)
   - Multi-head architecture this proposal extends

5. **SEMANTIC_PROJECTION_LDA.md** (this project)
   - Foundation of LDA projection
