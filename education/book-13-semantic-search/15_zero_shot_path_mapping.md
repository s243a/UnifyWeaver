# Chapter 15: Zero-Shot Path Mapping

This chapter covers the minimal transform approach for mapping semantic embeddings to hierarchical paths without per-query training.

## The Problem

Given a new bookmark title, find the best folder in a large hierarchy (38k+ items across multiple accounts). Challenges:

1. **Cold start**: New queries have no training examples
2. **Hierarchy depth**: Paths can be 5-10 levels deep
3. **Multi-account**: Items span personal and group accounts
4. **Scale**: Must handle thousands of folders efficiently

## The Minimal Transform

### Core Insight

Instead of learning a complex model, learn a **minimal linear transform** (W matrix) that projects query embeddings to path embeddings:

```
path_score = W @ query_embedding
```

The W matrix is small (768x768 or 384x384) and captures the systematic difference between "what the query says" and "where it should go."

### Why Linear Works

For semantic search, the relationship between query and target is approximately linear:
- Similar queries → similar folders
- The embedding space already captures semantic similarity
- W just rotates/scales to align query space with path space

This is a **Procrustes transformation** - finding the optimal rotation/scaling between two point clouds.

## Federated Architecture

### Per-Tree Clustering

The hierarchy is partitioned into **clusters** of semantically similar subtrees:

```
Cluster 1: Programming languages, frameworks
Cluster 2: Science, physics, math
Cluster 3: Art, design, media
...
Cluster 51: Miscellaneous
```

Each cluster gets its own W matrix, trained on items within that subtree.

### Why Federate?

1. **Specialization**: Each W learns domain-specific mappings
2. **Smaller matrices**: 51 small W's vs one huge W
3. **Parallel training**: Clusters train independently
4. **Interpretability**: Can inspect per-domain behavior

### Routing

At query time, route to the right cluster(s):

```python
# Softmax routing over cluster centroids
cluster_scores = softmax(query @ cluster_centroids.T)

# Top-k clusters
top_clusters = argsort(cluster_scores)[-k:]

# Query each cluster's W matrix
candidates = []
for c in top_clusters:
    scores = W[c] @ query
    candidates.extend(top_items(scores, n=10))
```

## Materialized Paths

### Path Format

Each item stores its full path as text:

```
- s243a
- STEM
- Programming
- Languages
- Python
/s243a/STEM/Programming/Languages/Python
```

The path is embedded alongside the item title, creating a joint representation of "what" and "where."

### Path Embedding

Two embedding strategies:

**Option A: Concatenated**
```python
path_emb = embed(title + "\n" + path_text)
```

**Option B: Dual objective**
```python
title_emb = embed_semantic(title)      # What it is
path_emb = embed_structural(path)       # Where it goes
combined = alpha * path_emb + (1-alpha) * title_emb
```

The dual objective (Option B) allows tuning semantic vs structural weight.

## Training the W Matrix

### Procrustes Solution

Given paired (query, target) embeddings, find W minimizing:

```
||W @ Q - T||²
```

Closed-form solution (orthogonal Procrustes):

```python
U, S, Vt = svd(T @ Q.T)
W = U @ Vt
```

This finds the optimal rotation. For scaling, use:

```python
W = (T @ Q.T) @ inv(Q @ Q.T)  # Least squares
```

### Training Data

Use existing folder placements as supervision:

```python
for bookmark, folder in existing_placements:
    query = embed(bookmark.title)
    target = embed(folder.path)
    pairs.append((query, target))

Q = stack([p[0] for p in pairs])  # [n_samples, embed_dim]
T = stack([p[1] for p in pairs])  # [n_samples, embed_dim]
W = solve_procrustes(Q, T)
```

### Per-Cluster Training

```python
for cluster_id, items in clusters.items():
    Q_cluster = queries_in_cluster(items)
    T_cluster = targets_in_cluster(items)
    W[cluster_id] = solve_procrustes(Q_cluster, T_cluster)
```

## Multi-Account Support

### Account Boundaries

The hierarchy spans multiple accounts:

```
s243a (personal)
├── STEM/
├── Hobbies/
└── ...

s243a_groups (shared)
├── Group Projects/
├── Shared Resources/
└── ...
```

### Cross-Account Routing

The router considers account boundaries:

```python
def route_query(query, preferred_account=None):
    scores = {}
    for cluster in clusters:
        base_score = query @ cluster.centroid
        if preferred_account and cluster.account != preferred_account:
            base_score *= 0.5  # Penalize cross-account
        scores[cluster] = base_score
    return top_k(scores, k=3)
```

## Inference Pipeline

### Full Pipeline

```python
def infer(query_text, top_k=10):
    # 1. Embed query
    query = embed(query_text)

    # 2. Route to clusters
    cluster_scores = softmax(query @ centroids.T / temperature)
    active_clusters = where(cluster_scores > threshold)

    # 3. Score within each cluster
    all_candidates = []
    for c in active_clusters:
        projected = W[c] @ query
        scores = projected @ items[c].T
        all_candidates.extend(zip(items[c], scores))

    # 4. Merge and rank
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    return all_candidates[:top_k]
```

### Cached Embeddings

For speed, pre-compute and cache:

```python
# Offline (once)
item_embeddings = {item: embed(item.path) for item in all_items}
cluster_centroids = compute_centroids(clusters)
save_cache(item_embeddings, cluster_centroids)

# Online (per query)
query_emb = embed(query)  # Only embedding computation
scores = cached_W @ query_emb @ cached_items.T  # Matrix ops only
```

Result: ~30ms inference for 38k items.

## Performance

### Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Recall@1 | 93% | Correct folder in top result |
| Recall@5 | 99% | Correct folder in top 5 |
| Recall@10 | 99.5% | Correct folder in top 10 |
| Inference time | ~30ms | Cached embeddings |
| Model size | 160MB | 51 W matrices + centroids |

### Why So Good?

1. **Semantic embeddings** already encode meaning well
2. **Hierarchical paths** provide strong signal
3. **Federated routing** focuses search
4. **Linear transform** is surprisingly expressive for this task

## Comparison with Alternatives

| Approach | Recall@1 | Speed | Model Size |
|----------|----------|-------|------------|
| Raw cosine similarity | 75% | 10ms | 0 |
| Single W matrix | 85% | 20ms | 2MB |
| **Federated W (ours)** | **93%** | **30ms** | **160MB** |
| Fine-tuned transformer | 95% | 500ms | 500MB+ |
| Full reranker | 97% | 2s | 1GB+ |

The federated approach hits a sweet spot: near-best accuracy with fast inference.

## Code Example

```python
from unifyweaver.semantic import FederatedPathMapper

# Load pre-trained model
mapper = FederatedPathMapper.load("models/pearltrees_federated.pkl")

# Single query
results = mapper.search("Introduction to neural networks", top_k=10)
for item, score in results:
    print(f"{score:.3f} {item.path}")

# Output:
# 0.328 s243a/STEM/AI & ML/Deep Learning
# 0.315 s243a/STEM/AI & ML/Neural Networks
# 0.298 s243a/STEM/AI & ML/Machine Learning
# ...
```

## Connection to Fuzzy Logic

The W transform can be combined with fuzzy boosting (Chapter 16):

```python
# Base semantic scores
semantic_scores = W @ query

# Fuzzy boost for specific terms
boost_scores = compute_boost("bash:0.9,linux:0.5", items)

# Combined (fuzzy AND)
final_scores = semantic_scores * boost_scores
```

This allows domain experts to inject knowledge without retraining W.

## Summary

The minimal transform approach:

1. **Simple**: Just a linear W matrix per cluster
2. **Fast**: 30ms inference with caching
3. **Accurate**: 93% Recall@1 on real data
4. **Interpretable**: Can inspect W and cluster assignments
5. **Extensible**: Combines with fuzzy logic and LLM reasoning

The next chapter covers how to combine these semantic scores with LLM reasoning for the full bookmark filing assistant.
