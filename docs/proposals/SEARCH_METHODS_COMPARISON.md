# Search Methods Comparison

**Status:** Reference
**Date:** 2025-12-17
**Related:** [MULTI_HEAD_PROJECTION_THEORY.md](MULTI_HEAD_PROJECTION_THEORY.md), [ROADMAP_KG_TOPOLOGY.md](ROADMAP_KG_TOPOLOGY.md)

## Overview

UnifyWeaver provides two semantic search methods with different characteristics. This document explains when to use each and their performance trade-offs.

## Method 1: Multi-Head Search (Recommended)

**Implementation:** `lda_database.py::multi_head_search()`

### How It Works

```
1. Compute centroid similarities:  sim_k = query · centroid_k
2. Softmax routing:                weights = softmax(sim / temperature)
3. Project query:                  projected = weights @ answer_embeddings
4. Search all answers:             scores = all_answers @ projected
5. Return top-k
```

### Key Characteristics

| Aspect | Value |
|--------|-------|
| **Routing** | Softmax over cluster centroids |
| **Temperature** | τ=0.1 (sharp routing, best match dominates ~80-90%) |
| **Projection** | Query projected toward weighted blend of answer embeddings |
| **Accuracy** | +6.7% Recall@1 over direct similarity |

### When to Use

- **Production search** - Best retrieval accuracy
- **Cluster-based training data** - When you have "many questions → one answer" mappings
- **Diverse topics** - Different query types need different projections

### Advantages

1. **Learned projection** - Adapts to your specific Q-A relationships
2. **Topic routing** - Query is "pulled" toward relevant answer space
3. **Proven accuracy** - 76.7% Recall@1 on novel queries (vs 70.0% baseline)

### Disadvantages

1. **Requires clusters** - Need cluster centroids and answer embeddings stored
2. **Training overhead** - Must train multi-head projection first
3. **Cluster dependency** - Performance tied to cluster quality

---

## Method 2: Direct Search (Baseline)

**Implementation:** `kg_topology_api.py::_direct_search()`

### How It Works

```
1. Embed query:       query_emb = embed(query_text)
2. Search all:        scores = all_answer_embeddings @ query_emb
3. Return top-k
```

### Key Characteristics

| Aspect | Value |
|--------|-------|
| **Routing** | None - direct comparison to all answers |
| **Projection** | None - raw embedding similarity |
| **Accuracy** | Baseline (70.0% Recall@1 in experiments) |

### When to Use

- **No projection defined** - Fallback when multi-head projection unavailable
- **Baseline comparison** - Measure improvement from learned projection
- **1:1 Q-A mappings** - After answer smoothing dissolves clusters
- **Simple deployments** - When training overhead isn't justified

### Advantages

1. **No training required** - Works immediately with embeddings
2. **Simpler architecture** - Just matrix multiplication
3. **Cluster-independent** - Works with any Q-A structure

### Disadvantages

1. **Lower accuracy** - No learned adaptation to your data
2. **No topic routing** - Same comparison for all query types
3. **Embedding quality dependent** - Only as good as base embeddings

---

## Performance Comparison

### Accuracy (from MULTI_HEAD_PROJECTION_THEORY.md experiments)

| Method | Recall@1 | MRR | Notes |
|--------|----------|-----|-------|
| Direct Search | 70.0% | 0.8100 | Baseline |
| Global LDA Projection | 73.1% | ~0.81 | Single W matrix |
| Multi-Head (τ=1.0) | 3.3% | 0.1692 | Too diffuse |
| **Multi-Head (τ=0.1)** | **76.7%** | **0.8648** | Sharp routing (recommended) |

### Computational Complexity

| Method | Complexity | Operations |
|--------|------------|------------|
| Direct Search | O(N × d) | One matrix multiply |
| Multi-Head | O(C × d) + O(N × d) | Centroid routing + answer search |

Where:
- N = number of answers
- C = number of clusters (typically C << N)
- d = embedding dimension

Both methods are fast for typical scales due to efficient matrix operations.

### Memory Requirements

| Method | Storage |
|--------|---------|
| Direct Search | Answer embeddings only |
| Multi-Head | Answer embeddings + cluster centroids + projection metadata |

---

## Temperature Effect (Multi-Head Only)

The temperature parameter τ controls routing sharpness:

```
Similarities: [0.85, 0.70, 0.60]

τ = 1.0:   weights = [0.39, 0.34, 0.27]  ← diffuse (BAD)
τ = 0.5:   weights = [0.54, 0.32, 0.14]  ← moderate
τ = 0.1:   weights = [0.82, 0.15, 0.03]  ← sharp (RECOMMENDED)
τ = 0.01:  weights = [0.99, 0.01, 0.00]  ← near-argmax
```

**Recommendation:** Use τ=0.1 for sharp routing that focuses on best-matching cluster while allowing minor contribution from related clusters.

---

## Decision Guide

```
                    ┌─────────────────────────────┐
                    │  Do you have trained        │
                    │  multi-head projection?     │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────┴───────────────┐
                    │                             │
                   YES                           NO
                    │                             │
                    ▼                             ▼
        ┌───────────────────┐         ┌───────────────────┐
        │ Use multi_head_   │         │ Use _direct_      │
        │ search()          │         │ search()          │
        │                   │         │                   │
        │ +6.7% accuracy    │         │ Baseline accuracy │
        │ Sharp routing     │         │ No training needed│
        └───────────────────┘         └───────────────────┘
```

### Use Multi-Head When:
- You have cluster-based training data
- Accuracy is important
- You've trained a projection

### Use Direct Search When:
- No projection is available
- You need a quick baseline
- You're transitioning to 1:1 Q-A mappings
- Training overhead isn't justified for your use case

---

## API Usage

### Multi-Head Search (via search_with_context)

```python
from kg_topology_api import KGTopologyAPI

api = KGTopologyAPI("lda.db")

# Uses multi_head_search internally (recommended)
results = api.search_with_context(
    query_text="How do I read CSV files?",
    model_name="all-MiniLM-L6-v2",
    mh_projection_id=1,  # Trained projection
    top_k=5
)
```

### Direct Search (baseline)

```python
# Force direct search (baseline comparison)
results = api.search_with_context(
    query_text="How do I read CSV files?",
    model_name="all-MiniLM-L6-v2",
    mh_projection_id=None,  # No projection
    top_k=5,
    use_direct_search=True
)
```

### Prolog Interface

```prolog
% Multi-head search (default when mh_projection_id is configured)
search_with_context(Config, "How do I read CSV?", 5, [], Results).

% Direct search (baseline)
search_with_context(Config, "How do I read CSV?", 5,
    [use_direct_search(true)], Results).
```

---

## Future: Answer Smoothing Transition

As answer smoothing creates 1:1 Q-A mappings (see [SEED_QUESTION_TOPOLOGY.md](SEED_QUESTION_TOPOLOGY.md)):

1. **Current state:** Clusters group "many questions → one answer"
2. **After smoothing:** Each question gets its own tailored answer
3. **Impact:** Cluster centroids become less meaningful
4. **Recommendation:** Direct search may become primary approach for 1:1 mappings

This transition is tracked in [ROADMAP_KG_TOPOLOGY.md](ROADMAP_KG_TOPOLOGY.md).

---

## References

- [MULTI_HEAD_PROJECTION_THEORY.md](MULTI_HEAD_PROJECTION_THEORY.md) - Full theory and experimental results
- [ROADMAP_KG_TOPOLOGY.md](ROADMAP_KG_TOPOLOGY.md) - Implementation roadmap
- [SEED_QUESTION_TOPOLOGY.md](SEED_QUESTION_TOPOLOGY.md) - Answer smoothing proposal
