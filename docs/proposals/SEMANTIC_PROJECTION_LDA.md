# Proposal: LDA-Based Semantic Projection for RAG

## Summary

This proposal describes a method for learning an affine transformation matrix W that maps query embeddings to answer embeddings. The approach is grounded in Linear Discriminant Analysis (LDA) and uses playbook-generated Q-A pairs to derive optimal projections.

## Relationship to RAG_MAPPING_SYSTEM

This proposal complements [RAG_MAPPING_SYSTEM.md](RAG_MAPPING_SYSTEM.md):

| RAG_MAPPING_SYSTEM | This Proposal |
|--------------------|---------------|
| How to generate Q-A pairs | How to learn W from Q-A pairs |
| Text-based mappings | Geometric/algebraic approach |
| Multiple mapping types | Single learned transformation |
| Cosine similarity search | Projected similarity search |

The mapping system generates the training data; this proposal describes how to use it.

## Motivation

Given clusters of questions that all map to the same answer (document), we want to learn a transformation W such that:

1. **Cluster centroids map to answers**: W · q̄ₖ ≈ aₖ
2. **Within-cluster variation is suppressed**: W · (qᵢ - q̄) ≈ 0

This is exactly the objective of Linear Discriminant Analysis: maximize between-class separation while minimizing within-class variance.

## Mathematical Formulation

### Setup

For each answer aₖ (e.g., a document embedding), we have a cluster of questions:
- Questions: q₁, q₂, ..., qₙ that should all retrieve aₖ
- Centroid: q̄ₖ = weighted mean of questions
- Residuals: δqᵢ = qᵢ - q̄ₖ (within-cluster variation)

### The Optimization Problem

We want W that:

```
Minimizes:  ||W · Q̄ - A||² + λ · Σₖᵢ wₖᵢ · ||W · δqₖᵢ||²
            ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            centroid mapping   residual suppression (weighted)
```

Where:
- Q̄ = matrix of cluster centroids (d × m)
- A = matrix of answer embeddings (d × m)
- δqₖᵢ = qᵢ - q̄ₖ (residual from centroid)
- wₖᵢ = qᵢ · q̄ₖ (similarity weight: trust representative questions more)
- λ = regularization strength

### Closed-Form Solution

```
W = A · Q̄ᵀ · (Q̄ · Q̄ᵀ + λ · Δw · Δwᵀ)⁻¹
```

Where Δw has columns √wₖᵢ · δqₖᵢ (weighted residuals).

### Numerical Stability

If `(Q̄ · Q̄ᵀ + λ · Δw · Δwᵀ)` is ill-conditioned (e.g., few clusters, high dimensionality), options include:

1. **Pseudo-inverse**: Use `pinv()` instead of `inv()` to handle rank deficiency
2. **Additional ridge term**: Add μI for numerical stability:
   ```
   W = A · Q̄ᵀ · (Q̄ · Q̄ᵀ + λ · Δw · Δwᵀ + μI)⁻¹
   ```
3. **Truncated SVD**: Compute SVD of the covariance, discard small singular values, then invert

The structured regularization (λ · Δw · Δwᵀ) helps condition the matrix, but when m << d (few clusters relative to embedding dimension), additional regularization may be needed.

### Interpretation

| Component | Meaning |
|-----------|---------|
| Q̄ · Q̄ᵀ | Between-class scatter (centroid covariance) |
| Δw · Δwᵀ | Within-class scatter (residual covariance) |
| λ | Trade-off: accuracy vs. noise suppression |

This is structurally identical to LDA's objective of maximizing S_b / S_w.

## Connection to Attention Mechanisms

Each cluster (answer + questions) acts like a **single attention head**:

```
One Q-A pair:   W = a ⊗ q         (rank-1, outer product)
Multiple pairs: W = Σₖ aₖ ⊗ q̄ₖ   (sum of rank-1 heads)
```

The transformation W encodes:
- **Key directions** (q̄ₖ): What to attend to in the query
- **Value outputs** (aₖ): What to retrieve for each key

With SVD decomposition (W = UΣVᵀ), we get orthogonal attention heads where each singular vector captures an independent semantic direction.

## Algorithm

### Bootstrapping Problem: Computing Weighted Centroids

To compute the weighted centroid q̄, we need weights based on similarity to the centroid - but we need the centroid to compute the weights. This chicken-and-egg problem is solved iteratively:

1. **Initialize**: Start with uniform weights (simple average)
2. **Iterate**: Compute centroid → recompute weights from similarity to centroid → repeat
3. **Converge**: Typically stabilizes in 2-3 iterations

This is an EM-like algorithm where representative questions (close to the emerging centroid) gain weight, while outliers lose influence.

```python
import numpy as np

def compute_weighted_centroid(questions, max_iter=3):
    """Iteratively compute weighted centroid (solves bootstrapping problem)."""
    n = len(questions)
    weights = np.ones(n) / n

    for _ in range(max_iter):
        q_bar = sum(w * q for w, q in zip(weights, questions))
        q_bar = q_bar / np.linalg.norm(q_bar)
        sims = [q @ q_bar for q in questions]
        weights = np.exp(sims) / sum(np.exp(sims))  # softmax

    return q_bar, weights

def compute_W(clusters, lambda_reg=1.0, ridge=1e-6):
    """
    Compute transformation matrix W from Q-A clusters.

    Args:
        clusters: List of (answer_embedding, [question_embeddings])
        lambda_reg: Regularization strength for residual suppression
        ridge: Additional ridge regularization for numerical stability

    Returns:
        W: d × d transformation matrix
    """
    centroids = []
    answers = []
    residuals_weighted = []

    for answer, questions in clusters:
        q_bar, weights = compute_weighted_centroid(questions)
        centroids.append(q_bar)
        answers.append(answer)

        for q, w in zip(questions, weights):
            delta = q - q_bar
            residuals_weighted.append(np.sqrt(w) * delta)

    Q_bar = np.column_stack(centroids)      # d × m
    A = np.column_stack(answers)            # d × m
    Delta_w = np.column_stack(residuals_weighted)  # d × n_total
    d = Q_bar.shape[0]

    # Compute regularized solution with ridge for numerical stability
    cov = Q_bar @ Q_bar.T + lambda_reg * Delta_w @ Delta_w.T + ridge * np.eye(d)
    W = A @ Q_bar.T @ np.linalg.pinv(cov)

    return W

def query(W, q, documents, top_k=5):
    """
    Query using learned transformation.

    Args:
        W: Transformation matrix
        q: Query embedding
        documents: List of (doc_id, embedding)
        top_k: Number of results

    Returns:
        Top-k document IDs
    """
    q_projected = W @ q

    scores = []
    for doc_id, doc_emb in documents:
        score = q_projected @ doc_emb / (
            np.linalg.norm(q_projected) * np.linalg.norm(doc_emb)
        )
        scores.append((score, doc_id))

    scores.sort(reverse=True)
    return [doc_id for _, doc_id in scores[:top_k]]
```

## Choosing λ (Regularization)

**Cross-validation:**
```python
for lambda_val in [0.01, 0.1, 1.0, 10.0]:
    W = compute_W(train_clusters, lambda_val)
    accuracy = evaluate(W, test_pairs)
    # Pick λ with best accuracy
```

**Heuristics:**
- Well-separated clusters → small λ (little noise suppression needed)
- Overlapping clusters → large λ (need to suppress confusion)
- Few Q-A pairs → small λ (avoid over-regularization)

## Implementation Plan

### Phase 1: Data Collection (via Playbooks)
- [ ] Add "Generate Search Queries" step to playbooks
- [ ] Collect Q-A pairs from LLM playbook runs
- [ ] Store as (answer_doc_id, [question_texts])

### Phase 2: Embedding
- [ ] Embed all collected questions using Go embedder
- [ ] Embed answer documents (or use existing embeddings)
- [ ] Store as (answer_embedding, [question_embeddings])

### Phase 3: Learn W
- [ ] Implement `compute_W` in Go or Python
- [ ] Cross-validate λ on held-out queries
- [ ] Store W matrix for inference

### Phase 4: Integration
- [ ] Add projected search mode to embedder
- [ ] Compare: direct cosine vs. projected cosine
- [ ] A/B test on real queries

### Phase 5: Iteration
- [ ] Collect failed queries
- [ ] Add as new Q-A pairs
- [ ] Recompute W periodically

## Future Extensions

### Multiple Attention Heads

Currently: Single W maps all queries to answer space.

Future: Separate input and output heads:
```
Input heads:  V_in (project query onto semantic directions)
Output heads: V_out (project answer onto semantic directions)

Matching in shared k-dimensional space:
  query_proj = V_in.T @ query
  doc_proj = V_out.T @ doc
  score = query_proj @ doc_proj
```

This allows different semantic structures in query vs. document space.

### Recursive Refinement

Use retrieved documents to refine the query projection:
```
h_0 = V_in.T @ query           # Initial projection
docs_0 = retrieve(h_0)         # First retrieval
h_1 = α·h_0 + β·V_out.T @ embed(docs_0)  # Refined
docs_1 = retrieve(h_1)         # Better retrieval
```

### Per-Domain Transforms

If a single W is insufficient (different domains need different mappings):
1. Cluster queries by domain
2. Learn W_domain for each cluster
3. Route queries to appropriate W at inference

## Theoretical Foundations

This approach is grounded in well-established statistical methods:

### Linear Discriminant Analysis (LDA)

Fisher's original formulation (1936) maximizes the ratio of between-class to within-class scatter. Our formulation is the regression analog.

### Connection to Ridge Regression

Standard ridge: `W = A · Qᵀ · (QQᵀ + λI)⁻¹`
Our form: `W = A · Q̄ᵀ · (Q̄Q̄ᵀ + λ Δw Δwᵀ)⁻¹`

The difference: we use **structured regularization** (penalize residual directions specifically) rather than uniform regularization.

### Neural Tangent Kernel

The theoretical justification for why this linear approach works: for semantically similar inputs, neural embeddings behave approximately linearly (local linearization around the data manifold).

## References

1. **Fisher, R.A.** (1936). "The use of multiple measurements in taxonomic problems." *Annals of Eugenics*.
   - Foundation of Linear Discriminant Analysis

2. **Guo, Y., Xue, L., & Weimer, M.** (2024). "[Linear Discriminant Regularized Regression](https://arxiv.org/abs/2402.14260)." *arXiv:2402.14260*.
   - Connects LDA to regression with structured regularization

3. **Jacot, A., Gabriel, F., & Hongler, C.** (2018). "Neural Tangent Kernel: Convergence and Generalization in Neural Networks." *NeurIPS*.
   - Theoretical basis for local linearization of neural networks

4. **Friedman, J.H.** (1989). "Regularized Discriminant Analysis." *JASA*.
   - Regularized variants of LDA for high-dimensional settings

5. **IBM Think Topics**: "[Linear Discriminant Analysis](https://www.ibm.com/think/topics/linear-discriminant-analysis)"
   - Overview of LDA concepts

6. **scikit-learn**: "[Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html)"
   - Practical implementation reference

## Success Metrics

- **Retrieval accuracy**: % of queries where correct doc is in top-k
- **Mean Reciprocal Rank (MRR)**: Average 1/rank of correct doc
- **Improvement over baseline**: Projected vs. direct cosine similarity
- **Effective rank of W**: How many dimensions matter (via SVD)

## Open Questions

1. How much training data (Q-A pairs) is needed for stable W?
2. Should W be full-rank or explicitly low-rank?
3. How often should W be recomputed as new pairs are collected?
4. Can we interpret the singular vectors of W semantically?

## Appendix: Geometric Interpretation

The input space decomposes as:
```
Input space = span(centroids) ⊕ span(residuals)
              ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
              "signal space"    "noise space"
```

W should:
- **Act on signal space**: Map centroids to correct answers
- **Annihilate noise space**: Ignore within-cluster variation

The λ parameter controls how strictly we enforce this decomposition.

When λ → 0: W only fits centroids, noise can leak through
When λ → ∞: W actively suppresses noise, possibly at cost to accuracy

The optimal λ balances accurate centroid mapping with robustness to query variation.
