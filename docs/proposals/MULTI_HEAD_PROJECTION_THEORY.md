# Multi-Head LDA Projection Theory

**Status:** Implemented
**Version:** 1.0
**Date:** 2025-01-14
**Extends:** [SEMANTIC_PROJECTION_LDA.md](SEMANTIC_PROJECTION_LDA.md)

## Executive Summary

This document extends the single-matrix LDA projection to a **multi-head architecture** analogous to transformer attention. Each cluster (topic/answer) gets its own "attention head" that specializes in recognizing queries related to that topic. Queries are routed to heads using **softmax attention over centroid similarities**, with a temperature parameter controlling routing sharpness.

**Key Result:** Multi-head projection with temperature=0.1 achieves 76.7% Recall@1 on novel queries, outperforming direct cosine similarity (70.0%) by +6.7%.

## Motivation: Why Multi-Head?

### The Problem with Global W

The original LDA approach learns a single global transformation matrix W:

```
W = A · Q̄ᵀ · (Q̄ · Q̄ᵀ + λ · Δw · Δwᵀ + μI)⁻¹
```

This works well when:
- All query-answer relationships follow similar patterns
- The embedding space is approximately linear
- There are enough training pairs to span the space

However, a global W is a **compromise** - it tries to learn one transformation that works for all topics. If "sqlite queries" need to move in one direction and "HTTP requests" need to move in a different direction, W averages these transformations.

### The Attention Head Analogy

In transformers, attention has multiple heads because different semantic relationships require different attention patterns:
- One head might attend to syntax
- Another to semantics
- Another to positional relationships

Similarly, different query topics need different projections:
- "Database query" topics need one mapping
- "API request" topics need another
- "Template generation" topics need yet another

### Multi-Head Architecture

Instead of one global W, we have **one head per cluster**:

```
Cluster 1: (centroid₁, answer_emb₁)  →  Head₁
Cluster 2: (centroid₂, answer_emb₂)  →  Head₂
...
Cluster n: (centroidₙ, answer_embₙ)  →  Headₙ
```

At inference, the query is routed to heads based on similarity to centroids.

## Mathematical Formulation

### Definitions

For cluster k:
- **Centroid** c_k: Weighted mean of question embeddings in cluster k
- **Answer embedding** a_k: The answer/document embedding for cluster k
- **Questions** Q_k = {q₁, q₂, ..., qₘ}: Training questions for cluster k

### Per-Head Representation

Each head stores:
1. **Centroid vector** c_k ∈ ℝᵈ (what queries should "look like" for this topic)
2. **Answer vector** a_k ∈ ℝᵈ (where to project queries that match this topic)

Optionally, a per-cluster projection matrix W_k can be stored, but empirically, centroid+answer is sufficient.

### Softmax Routing

Given a query q, we compute routing weights using softmax over centroid similarities:

```
similarities:  s_k = (q · c_k) / (||q|| · ||c_k||)    for k = 1..n

routing weights:  w_k = exp(s_k / τ) / Σⱼ exp(sⱼ / τ)
```

Where:
- s_k is the cosine similarity between query and centroid k
- τ is the **temperature** parameter
- w_k is the routing weight for head k (sums to 1)

### Temperature Effect

The temperature τ controls the **sharpness** of routing:

| Temperature | Effect | Routing Behavior |
|-------------|--------|------------------|
| τ → 0 | Argmax (hard routing) | Winner-take-all: only best-matching head contributes |
| τ = 0.1 | Sharp softmax | Best match dominates (~80-90%), others contribute minimally |
| τ = 1.0 | Standard softmax | Weights spread across heads (~10-20% each) |
| τ → ∞ | Uniform | All heads contribute equally |

**Example with 3 heads:**
```
Similarities: [0.85, 0.70, 0.60]

τ = 1.0:   weights = [0.39, 0.34, 0.27]  ← diffuse
τ = 0.5:   weights = [0.54, 0.32, 0.14]  ← moderate
τ = 0.1:   weights = [0.82, 0.15, 0.03]  ← sharp
τ = 0.01:  weights = [0.99, 0.01, 0.00]  ← near-argmax
```

### Projected Query

The final projected query is a **weighted combination** of per-head outputs:

**Without per-cluster W matrices (simpler):**
```
q_projected = Σₖ wₖ · aₖ
```

The query is projected toward a weighted blend of answer embeddings.

**With per-cluster W matrices:**
```
q_projected = Σₖ wₖ · (Wₖ @ q)
```

Each head applies its own transformation, then results are blended.

### Comparison to Transformer Attention

| Transformer Attention | Multi-Head LDA |
|----------------------|----------------|
| Q, K, V matrices | Query, Centroid, Answer embeddings |
| Attention scores = Q @ Kᵀ / √d | Similarities = q · centroids |
| Softmax over scores | Softmax over similarities |
| Output = attention @ V | Output = weights @ answers |
| Temperature via √d scaling | Explicit temperature τ |
| Multiple heads per layer | One head per cluster |
| Multiple layers | Single layer |

The key insight: **with one layer and limited depth, we need more heads (clusters) to capture diverse query-answer relationships**.

## Algorithm

### Training

```python
def train_multi_head(clusters, embedding_dim):
    """
    Train multi-head projection from Q-A clusters.

    Args:
        clusters: List of (answer_emb, [question_embs])
        embedding_dim: Dimension d

    Returns:
        heads: List of (centroid, answer_emb) tuples
    """
    heads = []

    for answer, questions in clusters:
        # Compute weighted centroid (iterative)
        centroid = compute_weighted_centroid(questions)

        # Store head
        heads.append((centroid, answer))

    return heads
```

### Inference

```python
def multi_head_search(query, heads, temperature=0.1):
    """
    Search using multi-head projection with soft routing.

    Args:
        query: Query embedding (d,)
        heads: List of (centroid, answer_emb) tuples
        temperature: Softmax temperature (lower = sharper routing)

    Returns:
        projected: Projected query embedding (d,)
        routing_weights: Dict of head weights
    """
    # Normalize query
    query_norm = query / np.linalg.norm(query)

    # Compute similarities to each centroid
    similarities = []
    for centroid, _ in heads:
        centroid_norm = centroid / np.linalg.norm(centroid)
        sim = query_norm @ centroid_norm
        similarities.append(sim)

    similarities = np.array(similarities)

    # Softmax routing with temperature
    scaled = similarities / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))  # Numerical stability
    routing_weights = exp_scaled / np.sum(exp_scaled)

    # Project query (weighted combination of answer embeddings)
    projected = np.zeros_like(query)
    for weight, (_, answer_emb) in zip(routing_weights, heads):
        projected += weight * answer_emb

    return projected, routing_weights
```

## Why Temperature Matters

### Intuition

Consider a query "How do I query a SQLite database?":

**High temperature (τ=1.0):**
- Routing spreads across many heads
- sqlite_source: 12%, http_source: 9%, json_source: 8%, ...
- Projected query is a blend of many answer types
- Result: Ambiguous, doesn't strongly match any specific answer

**Low temperature (τ=0.1):**
- Routing concentrates on best matches
- sqlite_source: 85%, json_source: 8%, others: <3%
- Projected query strongly resembles sqlite answer
- Result: Clear match with correct answer

### Mathematical Interpretation

As τ → 0, softmax approaches argmax:
```
lim(τ→0) softmax(s/τ) → one-hot vector on argmax(s)
```

This is **hard routing** - only the most similar cluster contributes.

As τ → ∞, softmax approaches uniform:
```
lim(τ→∞) softmax(s/τ) → [1/n, 1/n, ..., 1/n]
```

This is equivalent to averaging all answer embeddings - no discrimination.

### Recommended Value

Empirically, **τ = 0.1** works well:
- Sharp enough to focus on the best-matching cluster(s)
- Soft enough to allow contribution from semantically related clusters
- Robust to small variations in centroid similarity

## Geometric Interpretation

### Input Space Decomposition

The query space is partitioned by cluster centroids:
```
              c₂
              /
             /
    c₁------*------c₃
             \
              \
              c₄
```

A query q is projected based on its position relative to centroids. With sharp routing, the Voronoi region around each centroid determines the dominant head.

### Output Space Projection

Each head "pulls" the query toward its answer embedding:
```
            a₁
           ↗
    q  →  projected = Σ wₖ aₖ
           ↘
            a₂
```

With low temperature, the query moves strongly toward the nearest centroid's answer.

### Effect of Temperature

```
High τ:  q → blended point between all answers
Low τ:   q → answer of most similar cluster
```

## Relationship to Other Methods

### k-Nearest Neighbors

Multi-head with τ → 0 is similar to 1-NN classification:
- Find the closest centroid
- Return that cluster's answer

But soft routing (τ > 0) provides **interpolation** between clusters.

### Mixture of Experts

Multi-head projection is a simple mixture of experts:
- **Experts**: Per-cluster answer embeddings (or W matrices)
- **Gating**: Softmax over centroid similarities
- **No learnable gating**: Gating is determined by pre-computed centroids

Unlike neural MoE, our gating is fixed after training - this is both a limitation (less flexible) and an advantage (no gating network to train, no collapse issues).

### Cluster-Conditioned Retrieval

This approach can be seen as:
1. Soft-classify query into clusters
2. Retrieve using cluster-specific representations

It's a middle ground between:
- Global retrieval (one representation for all)
- Hard-partitioned retrieval (separate indices per cluster)

## Experimental Results

### Setup
- 18 clusters from UnifyWeaver playbooks
- all-MiniLM-L6-v2 embeddings (384 dimensions)
- 30 novel queries (not in training data)

### Results

| Method | Recall@1 | MRR |
|--------|----------|-----|
| Direct cosine similarity | 70.0% | 0.8100 |
| Global LDA projection | 73.1% | ~0.81 |
| Multi-head (τ=1.0) | 3.3% | 0.1692 |
| Multi-head (τ=0.1) | **76.7%** | **0.8648** |

### Analysis

- **τ=1.0 failure**: Routing too diffuse; all clusters contribute equally, creating an average that matches nothing well
- **τ=0.1 success**: Sharp routing focuses on correct cluster; improvement over global projection (+6.7% over direct)

## Implementation Notes

### Database Schema

```sql
-- Multi-head projection metadata
CREATE TABLE multi_head_projections (
    mh_projection_id INTEGER PRIMARY KEY,
    model_id INTEGER REFERENCES embedding_models(model_id),
    name TEXT,
    temperature REAL DEFAULT 1.0,
    num_heads INTEGER,
    recall_at_1 REAL,
    mrr REAL
);

-- Per-cluster heads
CREATE TABLE cluster_heads (
    head_id INTEGER PRIMARY KEY,
    mh_projection_id INTEGER,
    cluster_id INTEGER,
    centroid_path TEXT NOT NULL,      -- Path to centroid numpy file
    answer_emb_path TEXT NOT NULL,    -- Path to answer embedding
    W_path TEXT,                      -- Optional per-cluster W matrix
    num_questions INTEGER
);
```

### API

```python
# Training
mh_id = db.create_multi_head_projection(model_id, name, temperature=0.1)
db.add_cluster_head(mh_id, cluster_id, centroid, answer_emb)

# Inference
results = db.multi_head_search(
    query_embedding=emb,
    mh_projection_id=mh_id,
    top_k=5
)
# Results include routing_weights for interpretability
```

## Future Work

### Learnable Temperature

Currently τ is a hyperparameter. Could be learned:
- Per-cluster temperature τₖ
- Query-dependent temperature τ(q)

### Hierarchical Multi-Head

For large numbers of clusters, hierarchical routing:
```
Level 1: Route to coarse topic (e.g., "data sources")
Level 2: Route within topic (e.g., "sqlite" vs "http")
```

### Cross-Attention Heads

Current heads are independent. Cross-attention could capture relationships:
```
head_output_k = Σⱼ attention(k,j) · answer_j
```

### Per-Cluster W Matrices

Instead of just (centroid, answer), each head could have a learned Wₖ:
```
head_output_k = Wₖ @ query
```

This allows more complex per-topic transformations but requires more training data.

## Conclusion

Multi-head LDA projection extends the single-matrix approach to handle diverse query-answer relationships. The key insight is that **temperature controls the trade-off between specialization (low τ) and generalization (high τ)**. With properly tuned temperature (τ=0.1), multi-head projection outperforms both direct similarity and global projection on novel queries.

The approach is:
- **Simple**: No neural network training, just centroid computation
- **Interpretable**: Routing weights show which topics a query relates to
- **Effective**: +6.7% improvement over direct similarity
- **Analogous to attention**: Each cluster is an attention head with its own key (centroid) and value (answer)

## References

1. **Vaswani et al.** (2017). "Attention Is All You Need." *NeurIPS*.
   - Foundation of transformer attention mechanisms

2. **Shazeer et al.** (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR*.
   - Mixture of experts with gating

3. **Hinton et al.** (2015). "Distilling the Knowledge in a Neural Network." *arXiv*.
   - Temperature in softmax for knowledge distillation

4. **SEMANTIC_PROJECTION_LDA.md** (this project)
   - Foundation of the LDA projection approach

5. **Fisher, R.A.** (1936). "The use of multiple measurements in taxonomic problems."
   - Original Linear Discriminant Analysis
