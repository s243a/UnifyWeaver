# Proposal: Q/A Transformer Distillation with Enhanced Routing

## Overview

This document describes the federated Q/A model training and transformer distillation pipeline, reports experimental results, and proposes enhancements using per-question routing with probability-weighted softmax.

## What We Did

### Data Characteristics

| Attribute | Value |
|-----------|-------|
| Q/A Pairs | 1,888 |
| Embedding Model | nomic-ai/nomic-embed-text-v1.5 |
| Embedding Dimension | 768 |
| Clusters | 124 (auto-selected via effective rank) |
| Cluster Method | K-means on answer embeddings |

The data consists of skills-related Q/A pairs where:
- **Questions**: Natural language queries about UnifyWeaver capabilities
- **Answers**: Detailed explanations with code examples and references

### Training Pipeline

1. **Embedding Generation**: Both questions and answers embedded with nomic model
2. **Clustering**: K-means on answer embeddings, cluster count determined by effective rank criterion
3. **W Matrix Training**: Per-cluster Procrustes (minimal transform) projecting question space → answer space
4. **Transformer Distillation**: Compress federated model (124 W matrices) into compact transformer

### Federated Model

The federated model uses:
- **Routing**: Query similarity to cluster centroids → softmax → weighted W combination
- **Projection**: `projected = Σ weight_i × (query @ W_i)` for top-K clusters
- **Temperature**: 0.1 for softmax sharpness

## Experimental Results

### Transformer Architecture Comparison

| Architecture | Capacity | Parameters | Compression | Train Loss | Test Cosine Sim |
|--------------|----------|------------|-------------|------------|-----------------|
| H=12, L=2 | 144 | 10.6M | 6.9x | 0.030 | 0.673 |
| H=6, L=3 | 216 | 15.4M | 4.8x | 0.024 | **0.705** |
| H=4, L=4 | 256 | 20.1M | 3.6x | 0.021 | 0.693 |

**Observations**:
- **6³ architecture achieved best generalization** (0.705 cosine sim) despite lower capacity than 4⁴
- **4⁴ had lowest training loss** but slightly worse test performance, suggesting mild overfitting
- **12² was most efficient** but insufficient depth for complex projection patterns
- **3 layers appears optimal** - enough depth for moderate complexity without overfitting

### Interpretation

The ~0.70 cosine similarity ceiling suggests:
1. The projection task has inherent difficulty that more capacity doesn't solve
2. The nomic embedding model handles most semantic understanding
3. The transformer learns a relatively simple mapping function

## Proposed Enhancement: Per-Question Routing

### W Matrix Options

Two approaches for the projection matrices:

1. **Per-Q/A pair W**: One W matrix per question-answer pair
   - Most expressive
   - Impractical: 1,888 × 768 × 768 × 4 bytes ≈ 4.4 GB storage

2. **Per-cluster W** (what we use): Single W matrix per cluster via minimal transform
   - Practical storage: 124 × 768 × 768 × 4 bytes ≈ 290 MB
   - Better results than attempting to smooth/blend W matrices
   - Each cluster's W is computed via Procrustes on all Q/A pairs in that cluster

### Current Approach (Centroid-Based Routing)

```
query → sim(query, 124 centroids) → softmax → select cluster W → query @ W
```

- Routes based on 124 cluster centroids
- Coarse-grained: all questions in a cluster share the same centroid
- Training signal limited to centroid-level patterns

### Proposed Approach (Question-Based Routing)

```
query → sim(query, 1888 questions) → softmax → aggregate to cluster weights → select cluster W
```

**Key insight**: Each training question acts like an **attention head**. The same cluster W is selected, but the question similarities weight the softmax routing in **non-linear ways**.

- The W matrices remain fixed (one per cluster)
- Questions provide fine-grained routing signal
- Non-linearity comes from softmax over question similarities, then aggregating to cluster selection

**Benefits**:
1. **Richer routing signal**: 1,888 question similarities vs 124 centroid similarities
2. **Non-linear attention**: Questions near cluster boundaries create nuanced soft-routing
3. **Better generalization**: Learns question-level routing patterns

**Challenges**:
1. **Similarity computation**: O(N²) for question similarities vs O(N×K) for centroids
2. **Aggregation**: Need to map question-level softmax weights to cluster-level selection

**Computational Optimization**:

Since W is the same for all questions in a cluster, aggregation is simple:

```python
# 1. Compute question similarities (dot products)
sims = query @ questions.T  # [1, N] @ [N, D].T = [1, N]

# 2. Softmax to get fractional probabilities
probs = softmax(sims / temperature)  # [N]

# 3. Aggregate to cluster weights (sum of fractional probabilities per cluster)
cluster_weights = zeros(K)
for i, q in enumerate(questions):
    cluster_weights[cluster[i]] += probs[i]

# 4. Single matrix multiply per cluster
result = sum(cluster_weights[k] * (query @ W[k]) for k in range(K))
```

The cluster weight is just the sum of fractional probabilities for questions in that cluster. This reduces to K matrix multiplies regardless of N.

**Final softmax routing over clusters**:

```python
# Cluster weights now have complex weighting from question-level attention
# Apply final softmax routing over clusters
final_weights = softmax(cluster_weights / temperature)
result = sum(final_weights[k] * (query @ W[k]) for k in range(K))
```

### Analogy to Statistical Mechanics

This two-level routing is analogous to **degeneracy (state multiplicity)** in statistical mechanics:

- **Cluster** = energy level
- **Questions in cluster** = degenerate microstates at that energy
- **Question similarity** = Boltzmann factor for each microstate
- **Cluster weight** = partition function contribution (sum over microstates)

In stat mech: `P(energy E) ∝ g(E) × exp(-E/kT)`

Where `g(E)` is the degeneracy (number of microstates at energy E).

In our routing: `weight(cluster k) ∝ Σ_q∈k sim(query, q)`

Clusters with more similar questions have higher "degeneracy" and thus higher routing weight - not just because of centroid similarity, but because of the multiplicity of matching questions.

## Probability-Weighted Softmax

### Motivation

Not all training examples are equally informative. We can modify softmax weights based on **prediction confidence/uncertainty** to:
- Upweight uncertain/boundary cases (more informative)
- Downweight highly confident predictions (less informative)

### Standard Softmax

```
weights_i = exp(sim_i / τ) / Σ exp(sim_j / τ)
```

### Probability-Weighted Softmax

```
weights_i = p_i × exp(sim_i / τ) / Σ p_j × exp(sim_j / τ)
```

Where `p_i` is a probability/importance weight derived from model confidence.

## Deriving Probability from Entropy

### Approach 1: Fisher Information

Fisher information measures how much information an observation provides about model parameters:

```
I(θ) = E[(∂/∂θ log p(x|θ))²]
```

For our routing context:
- **High Fisher information**: Sample is informative (near decision boundary)
- **Low Fisher information**: Sample is uninformative (clearly belongs to one cluster)

**Practical computation**:
```python
# Compute gradient of log-likelihood w.r.t. routing weights
grad = ∂L/∂weights
fisher_score = ||grad||²

# Weight by Fisher information (upweight informative samples)
importance = fisher_score / mean(fisher_scores)
```

**Entropy-based approximation**:
```python
# Softmax distribution entropy
H = -Σ weights_i × log(weights_i)

# High entropy = uncertain = informative
# Low entropy = confident = less informative
importance = H / log(num_clusters)  # Normalized entropy
```

### Approach 2: BERT Logit Values

The nomic embedding model (based on BERT architecture) produces logits before the final projection. These logits contain uncertainty information:

**Token-level uncertainty**:
```python
# Get logits from BERT's masked language model head
logits = bert.cls(hidden_states)  # [batch, seq, vocab]

# Convert to probabilities
probs = softmax(logits, dim=-1)

# Compute per-token entropy
token_entropy = -Σ probs × log(probs)

# Aggregate to sequence-level
sequence_uncertainty = mean(token_entropy)
```

**Embedding-level uncertainty** (if using contrastive models):
```python
# Some models expose confidence scores
# For nomic, we can use the embedding norm as a proxy
embedding_confidence = ||embedding|| / expected_norm

# Or use dropout-based uncertainty
with_dropout = [model(x, dropout=True) for _ in range(N)]
uncertainty = std(with_dropout)
```

### Combining Approaches

```python
def compute_importance_weight(query_emb, cluster_weights, bert_logits=None):
    # Routing entropy (from softmax weights)
    routing_entropy = -sum(w * log(w) for w in cluster_weights if w > 0)

    # Optional: BERT-based uncertainty
    if bert_logits is not None:
        bert_entropy = compute_token_entropy(bert_logits)
        combined = α * routing_entropy + (1-α) * bert_entropy
    else:
        combined = routing_entropy

    # Normalize to [0, 1] importance weight
    # High entropy = high importance (upweight uncertain samples)
    importance = combined / max_entropy

    return importance
```

## Implementation Plan

### Phase 1: Per-Question Routing
1. Modify `FederatedProjectionWrapper.project()` to use question-level routing
2. Add cluster aggregation: question weights → cluster weights
3. Benchmark against centroid-based routing

### Phase 2: Importance Weighting
1. Compute routing entropy for each training sample
2. Apply importance weights during distillation loss computation
3. Compare: uniform weights vs entropy-weighted vs Fisher-weighted

### Phase 3: BERT Uncertainty Integration
1. Extract logits/hidden states from nomic model during embedding
2. Compute token-level entropy
3. Combine with routing entropy for final importance weight

## Expected Outcomes

| Enhancement | Expected Impact |
|-------------|-----------------|
| Per-question routing | +5-10% cosine sim (richer training signal) |
| Entropy weighting | Better generalization on boundary cases |
| BERT uncertainty | More principled importance weights |

## Related Work

- **Knowledge Distillation**: Hinton et al. (2015) - temperature-scaled softmax
- **Importance Sampling**: Prioritizing informative training examples
- **Uncertainty Quantification**: Gal & Ghahramani (2016) - dropout-based uncertainty
- **Fisher Information in Deep Learning**: Martens (2014) - natural gradient methods

## Files

| File | Description |
|------|-------------|
| `scripts/train_pearltrees_federated.py` | Federated model training with cluster criterion |
| `scripts/distill_federated_to_transformer.py` | Transformer distillation |
| `src/unifyweaver/targets/python_runtime/projection_transformer.py` | Transformer architecture |
| `models/skills_qa_federated_optimized.pkl` | Trained federated model (124 clusters) |
| `models/skills_qa_transformer_6x3.pt` | Best distilled transformer (H=6, L=3) |
