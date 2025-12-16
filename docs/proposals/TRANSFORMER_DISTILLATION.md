# Proposal: Transformer Distillation from LDA Projection

**Status:** Proposal
**Version:** 0.1
**Date:** 2025-12-15
**Extends:** [MULTI_HEAD_PROJECTION_THEORY.md](MULTI_HEAD_PROJECTION_THEORY.md)

## Executive Summary

As the number of Q-A clusters grows large (hundreds to thousands), the flat softmax routing over all LDA heads becomes computationally expensive. This proposal describes a **knowledge distillation** approach where a compact transformer learns to approximate the LDA projection function using fewer heads.

Key insight: Train the transformer with **MSE on output embeddings** rather than cross-entropy on routing weights. This provides a continuous optimization target and allows the transformer to learn compressed representations of the routing function.

## Motivation

### The Scaling Problem

Current LDA multi-head projection:
```
query → softmax(query @ centroids.T / τ) → weights → weights @ answers → projected
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        O(n) computation where n = num_clusters
```

For n = 1000 clusters with d = 384 dimensions:
- Centroid matrix: 1000 × 384 = 384K floats
- Full matrix multiply per query
- Softmax over 1000 values

### The Compression Opportunity

Many clusters are semantically related. A transformer with k << n heads can learn to:
1. Group similar clusters implicitly
2. Route queries using learned attention patterns
3. Output embeddings that approximate the full LDA projection

```
LDA (teacher):     n=1000 heads, exact routing
Transformer (student): k=64 heads, approximate routing, ~16x compression
```

## Mathematical Formulation

### LDA Projection (Teacher)

Given query q, the LDA projection computes:

```
s_i = (q · c_i) / (||q|| · ||c_i||)     # similarity to centroid i
w_i = softmax(s / τ)_i                   # routing weight
p_lda = Σ_i w_i · a_i                    # projected embedding
```

Where:
- c_i = centroid of cluster i
- a_i = answer embedding of cluster i
- τ = temperature

### Transformer Projection (Student)

The transformer learns to directly predict the projected embedding:

```
p_transformer = Transformer(q)
```

Architecture options:
1. **Single-layer attention**: q → MultiHeadAttention(Q=q, K=learned, V=learned) → p
2. **MLP + attention**: q → MLP → MultiHeadAttention → p
3. **Full transformer block**: q → [Attention + FFN] × L → p

### Training Objective

Instead of cross-entropy on routing weights, use MSE on output embeddings:

```
L = ||p_transformer - p_lda||²
```

**Why MSE works better here:**
- Continuous loss landscape (no discrete argmax)
- Gradients flow smoothly through all parameters
- Doesn't require the transformer to exactly match routing weights
- Allows transformer to find alternative routes to same output

### Optional: Auxiliary Routing Loss

To encourage interpretable routing, add soft constraint:

```
L_total = L_mse + λ · KL(w_transformer || w_lda)
```

Where w_transformer is derived from attention weights. Keep λ small to not over-constrain.

## Hierarchical Transformer Ensemble

For very large cluster counts, use multiple transformers each approximating a subset:

```
                    ┌─────────────────┐
                    │  Meta-Router    │
                    │  (small MLP)    │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Transformer  │  │ Transformer  │  │ Transformer  │
    │   Domain A   │  │   Domain B   │  │   Domain C   │
    │ (clusters    │  │ (clusters    │  │ (clusters    │
    │  1-100)      │  │  101-200)    │  │  201-300)    │
    └──────────────┘  └──────────────┘  └──────────────┘
```

### Domain Assignment

Clusters can be assigned to domains via:
1. **K-means on centroids**: Automatic grouping by similarity
2. **Manual domain labels**: "data sources", "generators", "templates"
3. **Hierarchical clustering**: Build tree, cut at desired level

### Meta-Router

Small MLP that predicts which domain transformer(s) to invoke:

```python
domain_weights = softmax(meta_router(query))  # [0.8, 0.15, 0.05]
p = Σ_d domain_weights[d] · transformer_d(query)
```

For sparse routing, use top-k:
```python
top_domains = topk(domain_weights, k=2)
p = Σ_d∈top_domains normalized_weight[d] · transformer_d(query)
```

### Training the Ensemble

1. **Pre-train domain transformers**: Each on its subset of LDA heads
2. **Train meta-router**: On full query set, minimize MSE to full LDA output
3. **Fine-tune end-to-end**: Optional joint optimization

## Architecture Details

### Recommended Architecture

For moderate scale (100-1000 clusters):

```python
class ProjectionTransformer(nn.Module):
    def __init__(self, embed_dim=384, num_heads=8, ff_dim=512):
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # Learned keys and values (compressed cluster representations)
        self.keys = nn.Parameter(torch.randn(num_heads * 8, embed_dim))
        self.values = nn.Parameter(torch.randn(num_heads * 8, embed_dim))

        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query):
        # query: (batch, embed_dim)
        q = self.input_proj(query).unsqueeze(0)  # (1, batch, embed_dim)

        k = self.keys.unsqueeze(1).expand(-1, query.size(0), -1)  # (num_keys, batch, embed_dim)
        v = self.values.unsqueeze(1).expand(-1, query.size(0), -1)

        # Cross-attention: query attends to learned keys/values
        attn_out, _ = self.attention(q, k, v)
        attn_out = attn_out.squeeze(0)  # (batch, embed_dim)

        # Residual + FFN
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))

        return x
```

### Hyperparameter Guidelines

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| num_heads | 8-16 | Attention heads, not cluster heads |
| num_keys | 64-128 | Learned "compressed clusters" |
| ff_dim | 2× embed_dim | Standard transformer ratio |
| layers | 1-2 | More layers for larger cluster counts |

### Compression Ratio

```
LDA:         n clusters × d dimensions = n × d parameters (centroids + answers)
Transformer: k keys × d + attention params ≈ k × d + overhead

Example: n=1000, k=64, d=384
  LDA:         1000 × 384 × 2 = 768K params
  Transformer: 64 × 384 × 2 + attn ≈ 50K + 150K = 200K params
  Compression: ~4x
```

## Training Procedure

### Phase 1: Train LDA (Teacher)

Standard multi-head LDA training:
```bash
python scripts/train_multi_head_projection.py \
    --db playbooks/lda-training-data/lda.db \
    --temperature 0.1
```

### Phase 2: Generate Distillation Dataset

For each training query, compute LDA projection:
```python
distillation_data = []
for query in training_queries:
    query_emb = embedder.encode(query)
    lda_output = lda_projection.project(query_emb)
    distillation_data.append((query_emb, lda_output))
```

### Phase 3: Train Transformer (Student)

```python
transformer = ProjectionTransformer(embed_dim=384, num_heads=8)
optimizer = AdamW(transformer.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for query_emb, lda_target in dataloader:
        pred = transformer(query_emb)
        loss = F.mse_loss(pred, lda_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Phase 4: Validate

Compare retrieval quality:
```python
# LDA retrieval
lda_results = search_with_projection(query, lda_projection, top_k=10)

# Transformer retrieval
transformer_results = search_with_projection(query, transformer, top_k=10)

# Compute recall overlap
recall_match = len(set(lda_results) & set(transformer_results)) / top_k
```

Target: >95% recall match indicates successful distillation.

## When to Use

| Scale | Recommendation |
|-------|----------------|
| Moderate | Use LDA directly (flat softmax is very efficient) |
| Large | Consider single transformer if latency becomes an issue |
| Very Large | Hierarchical ensemble (2-4 domain transformers) |
| Massive | Full hierarchical with ANN for meta-routing |

*Note: Specific thresholds depend on hardware and latency requirements. The flat softmax approach is surprisingly efficient even at large scales due to optimized matrix operations. Profile before switching strategies.*

### Decision Criteria

Use transformer distillation when:
1. Query latency becomes a bottleneck (profile first!)
2. Cluster count grows large enough that softmax dominates latency
3. Memory is constrained (mobile/edge deployment)
4. Clusters have natural domain structure for hierarchical routing

Stay with LDA when:
1. Latency is acceptable (it usually is - flat softmax is fast)
2. Interpretability is important (LDA routing weights are meaningful)
3. Clusters change frequently (retraining transformer is expensive)
4. Training data is limited (transformer needs more data)

## Implementation Roadmap

### Phase 1: Single Transformer
- [ ] Implement ProjectionTransformer in Python
- [ ] Training script with MSE loss
- [ ] Validation script comparing to LDA
- [ ] Benchmark latency improvement

### Phase 2: Hierarchical Ensemble
- [ ] Domain clustering script
- [ ] Meta-router implementation
- [ ] Per-domain transformer training
- [ ] End-to-end fine-tuning

### Phase 3: Rust Integration
- [ ] Port transformer to Rust (candle)
- [ ] Load trained weights from Python
- [ ] Integrate with PtSearcher

## Theoretical Connection

### Relationship to Attention

The LDA multi-head projection is structurally similar to attention:

| LDA Multi-Head | Transformer Attention |
|----------------|----------------------|
| Centroids | Keys |
| Answers | Values |
| Query | Query |
| Softmax routing | Attention weights |

The transformer distillation learns to **compress** the key-value pairs while preserving the input-output mapping.

### Relationship to Autoencoders

The transformer can be viewed as learning a compressed representation:
```
LDA: query → [n centroids] → routing → [n answers] → output
Transformer: query → [k keys] → attention → [k values] → output
```

Where k << n, the transformer acts like a bottleneck autoencoder on the routing function.

## References

1. **Hinton et al.** (2015). "Distilling the Knowledge in a Neural Network."
   - Knowledge distillation framework

2. **Vaswani et al.** (2017). "Attention Is All You Need."
   - Transformer architecture

3. **Fedus et al.** (2022). "Switch Transformers: Scaling to Trillion Parameter Models."
   - Sparse mixture of experts routing

4. **MULTI_HEAD_PROJECTION_THEORY.md** (this project)
   - LDA multi-head projection (teacher model)

5. **SMOOTHNESS_REGULARIZATION.md** (this project)
   - Related work on regularization for sparse clusters
