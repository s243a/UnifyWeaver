# Proposal: Transformer Distillation from LDA Projection

**Status:** Implemented
**Version:** 1.0
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

Use **combined MSE + cosine similarity loss** on output embeddings:

```
L = (1 - λ) × ||p_transformer - p_lda||² + λ × (1 - cosine_sim(p_transformer, p_lda))
```

Recommended: λ = 0.5 to 0.7 (cosine-weighted)

**Why combined loss works:**
- **MSE** ensures magnitude matching
- **Cosine loss** ensures directional alignment (essential!)
- Continuous loss landscape (no discrete argmax)
- Gradients flow smoothly through all parameters
- Doesn't require the transformer to exactly match routing weights

**Important:** MSE alone can achieve low loss with wrong direction. Always include cosine loss.

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
from projection_transformer import ProjectionTransformer, train_distillation

# Choose architecture: H^L ≈ N_lda_heads
# For N=64 heads: H=4, L=3 (4³=64)
# For N=18 heads: H=4, L=2 (4²=16, close enough)
transformer = ProjectionTransformer(
    embed_dim=384,
    num_heads=4,
    num_layers=2
)

# Train with combined MSE + cosine loss
train_distillation(
    transformer=transformer,
    lda_projection=lda,
    query_embeddings=train_queries,
    num_epochs=200,
    cosine_weight=0.7  # Important: use cosine loss!
)
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
- [x] Implement ProjectionTransformer in Python (`src/unifyweaver/targets/python_runtime/projection_transformer.py`)
- [x] Training script with combined MSE + cosine loss
- [x] Validation script comparing to LDA (`scripts/test_transformer_distillation.py`)
- [x] Benchmark latency (LDA faster at small scale, as expected)

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

### Capacity Equivalence Conjecture: H^L = N

**Conjecture:** A transformer with H heads per layer and L layers has routing capacity equivalent to H^L flat LDA heads.

**Intuition:**
- Flat LDA: N independent heads, N distinct routing patterns
- L-layer transformer: Each layer routes through H attention patterns
- Layers compose sequentially, multiplying possibilities: H × H × ... × H = H^L

**Equivalence formula:**
```
H^L = N_flat

Solving for layers needed to match N flat heads:
L = log(N) / log(H)
```

**Examples:**

| Flat LDA Heads (N) | Heads/Layer (H) | Layers Needed (L) |
|--------------------|-----------------|-------------------|
| 64 | 8 | 2 |
| 512 | 8 | 3 |
| 4096 | 8 | 4 |
| 1000 | 10 | 3 |

**Practical implications:**
- 2-layer, 8-head transformer ≈ 64 flat heads
- 3-layer, 8-head transformer ≈ 512 flat heads
- Exponential compression: adding one layer squares the capacity

**Optimal H and L for minimum total heads:**

Given constraint H^L = N, minimize total heads H × L:
```
Total = H × L = H × log(N) / log(H)

Taking derivative and setting to zero:
d/dH [H / log(H)] = 0
→ H_optimal = e ≈ 2.718
```

So theoretically, H ≈ 3 minimizes total heads. However, practical constraints favor powers of 2 for GPU efficiency.

| Target N | H=2, L=? | H=4, L=? | H=8, L=? | Minimum Total |
|----------|----------|----------|----------|---------------|
| 64 | L=6, T=12 | L=3, T=12 | L=2, T=16 | H=2 or H=4 |
| 512 | L=9, T=18 | L=4.5, T=18 | L=3, T=24 | H=2 or H=4 |
| 1000 | L=10, T=20 | L=5, T=20 | L=3.3, T=26 | H=2 or H=4 |

*T = total heads (H × L)*

**Recommendation:** Use H=4 with appropriate L for a good balance between total heads and layer count.

**Caveats:**
1. Upper bound - not all H^L combinations may be reachable
2. Heads within a layer share output projection (some redundancy)
3. Actual capacity depends on attention pattern expressivity
4. Training dynamics may not fully exploit theoretical capacity

**Validation Results (2025-12-15):**

Initial test with N=18 LDA heads, transformer H=4, L=2 (equivalent=16):

| Metric | Result |
|--------|--------|
| Mean Cosine Similarity | 0.9928 ± 0.0027 |
| Min Cosine Similarity | 0.9827 |
| Max Cosine Similarity | 0.9954 |
| Training Epochs | 200 |
| Cosine Loss Weight | 0.7 |

**Key findings:**
1. **Cosine loss is essential** - MSE-only training achieved ~0.58 cosine similarity; adding cosine loss improved to 0.99+
2. **Architecture matching matters** - Using floor instead of ceiling for L calculation gives better approximation
3. **High fidelity** - 99.28% average cosine similarity demonstrates strong equivalence

This validates that the H^L = N conjecture provides a principled way to choose transformer architecture based on the number of LDA clusters being replaced.

**Latency Benchmark (N=18 heads, 1000 queries):**

| Method | Single Query | Batch (32) |
|--------|-------------|------------|
| LDA (NumPy CPU) | 0.046 ms | 0.046 ms |
| Transformer (CUDA) | 1.110 ms | 0.069 ms |

**Crossover Analysis:**

| Scenario | Slowdown at N=18 | Estimated Crossover |
|----------|------------------|---------------------|
| Single query | 24x | >400 heads (lower bound) |
| Batched (32) | 1.5x | ~27 heads |

The transformer's advantage is logarithmic scaling (L = log(N)/log(H)) vs LDA's linear O(N). However, GPU kernel launch overhead dominates at small scales. For single-query inference, the transformer would need significantly more than 400 heads to outperform LDA - the exact crossover point is unknown without benchmarking at larger scales.

**Note on embedding models:** These benchmarks used all-MiniLM-L6-v2 (384 dim, 256 token context), which is very fast. Larger models like nomic-embed-text-v1.5 (768 dim, 8192 token context) would shift these numbers but likely preserve the relative scaling behavior.

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

## Per-Tree Clustering Benchmarks

When using `--cluster-method per-tree` for federated training, the clustering is based on user organization (Pearltrees folder structure) rather than semantic similarity.

### Hit Rate Analysis

Benchmark on 500 items across 255 trees:

| k | Federated W | Transformer | Gap |
|---|-------------|-------------|-----|
| 1 | 63.4% | 53.2% | -10.2% |
| 3 | 73.8% | 67.6% | -6.2% |
| 5 | 78.4% | 73.2% | -5.2% |
| 10 | 85.4% | 79.2% | -6.2% |
| 20 | 90.8% | 87.0% | -3.8% |
| 50 | 96.2% | 93.8% | -2.4% |

- Mean rank: Federated 9.0, Transformer 15.1
- Median rank: Both 1.0
- Potentially misfiled (rank > 50): 3.8%

### Misfiling Detection

Per-tree clustering has a secondary use: **detecting misfiled bookmarks**.

If an item's actual folder doesn't appear in the top-k semantic search results, it may indicate the item was filed in a semantically mismatched folder.

```
Query: "quantum physics paper"
Actual folder: "Recipes"
Top 10 results: Physics, Science, Papers...

Actual folder NOT in top 10 → Potential misfiling
```

Interpretation:
- 63.4% at rank 1: Perfect semantic match with folder
- 85.4% at rank 10: Good filing (semantically nearby)
- 3.8% at rank > 50: Potentially misfiled items

**Note:** These results are user-specific. Hit rates depend on organization style:

| Organization Style | Expected Recall@10 |
|-------------------|-------------------|
| Semantic (Physics → Quantum → Papers) | High (>90%) |
| Project-based (Work, Personal, Archive) | Mixed (50-80%) |
| Chaotic (Misc, Stuff, TODO) | Low (<50%) |

The recall@k metric can serve as a "semantic organization score" measuring how well a user's folder structure aligns with content semantics.

### Data Quality Dependency

The entire pipeline depends on training data quality:

```
User's Organization → Structured Lists → W Matrices → Search Quality
```

The W matrices learn to transform query embeddings to match **structured path embeddings**:

```
/2492215/2496226
- account
  - Physics
    - Quantum Mechanics
```

This assumes:
- Folder paths are meaningful labels
- Hierarchical structure reflects semantic relationships
- Siblings in a folder are related

Per-tree recall@k tests these assumptions. Low recall indicates the structured lists don't encode semantic meaning well - and more sophisticated approaches (MST, transformer distillation) won't fix fundamental data quality issues.

### Per-Tree vs MST/Embedding Clustering

| Aspect | Per-Tree | MST/Embedding |
|--------|----------|---------------|
| Cluster count | Fixed (= folder count) | Configurable (`--max-clusters`) |
| Boundaries | User-defined (arbitrary) | Semantic (coherent) |
| Distillation quality | ~40% cosine sim | Expected >95% |
| W matrix generalization | Limited | Better |
| Use case | Preserve user's mental model | Better transforms |

**Per-tree clustering:**
- Preserves user's folder structure as clusters
- Useful for misfiling detection
- Distillation quality depends on user's organization style

**MST/Embedding clustering:**
- Groups semantically similar items regardless of user's folders
- Creates internally coherent clusters
- Better for distillation since boundaries are meaningful
- Allows tuning cluster granularity

Both approaches depend on underlying data quality (structured lists), but MST ensures each cluster is internally coherent even if the user's overall organization is messy.

### Key Insight: N = num_clusters

For per-tree with `--transform-mode single`, architecture sizing uses N = number of clusters (trees), not number of training queries:

```
Per-tree (255 clusters): H=4, L=4 → H^L = 256 ≈ 255 ✓
```

The transformer learns 255 distinct W transformations, achieving 7.5x compression (37.6M → 5M parameters).
