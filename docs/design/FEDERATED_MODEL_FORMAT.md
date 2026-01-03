# Federated Procrustes Model Format

This document describes the file format for federated Procrustes projection models used in the Pearltrees semantic search system.

## Overview

A federated model consists of:
1. **Main metadata file** (`.pkl`) - Python pickle containing model configuration
2. **Cluster directory** - Contains per-cluster transformation matrices and embeddings

## File Structure

```
models/
├── pearltrees_federated_single.pkl      # Main metadata
└── pearltrees_federated_single/         # Cluster directory
    ├── routing_data.npz                 # Query routing data
    ├── cluster_0.npz                    # Per-cluster data
    ├── cluster_1.npz
    └── ...
```

## Main Metadata File (`.pkl`)

A Python pickle file containing a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `cluster_ids` | `List[str]` | Cluster names, e.g., `["cluster_0", "cluster_1", ...]` |
| `cluster_centroids` | `np.ndarray` | Shape `(num_clusters, embedding_dim)`. Centroid embeddings for routing. |
| `temperature` | `float` | Softmax temperature for soft routing (default: 0.1) |
| `global_target_ids` | `List[str]` | Tree IDs in dataset order |
| `global_target_titles` | `List[str]` | Titles in dataset order |
| `num_clusters` | `int` | Number of clusters |
| `cluster_dir` | `str` | Path to cluster `.npz` files |
| `data_path` | `str` | Path to training JSONL (for account lookup at inference) |
| `metrics` | `dict` | Training metrics (currently empty) |

### Example: Loading Metadata

```python
import pickle

with open("models/pearltrees_federated_single.pkl", "rb") as f:
    meta = pickle.load(f)

print(f"Clusters: {meta['num_clusters']}")
print(f"Targets: {len(meta['global_target_ids'])}")
print(f"Embedding dim: {meta['cluster_centroids'].shape[1]}")
```

## Routing Data (`routing_data.npz`)

NumPy compressed archive for query-to-cluster routing:

| Array | Shape | Description |
|-------|-------|-------------|
| `query_embeddings` | `(n_items, embedding_dim)` | Embedded query texts (titles) |
| `target_embeddings` | `(n_items, embedding_dim)` | Embedded target texts (paths) |
| `idx_to_cluster_keys` | `(n_items,)` | Dataset indices |
| `idx_to_cluster_values` | `(n_items,)` | Corresponding cluster IDs |

## Cluster Files (`cluster_*.npz`)

Each cluster has a NumPy compressed archive containing:

| Array | Shape | Description |
|-------|-------|-------------|
| `W_stack` | `(1, embedding_dim, embedding_dim)` | Procrustes transformation matrix |
| `target_embeddings` | `(n_items_in_cluster, embedding_dim)` | Target embeddings for this cluster |
| `indices` | `(n_items_in_cluster,)` | Original dataset indices |

### The W Matrix

The W matrix is an orthogonal transformation (rotation ± reflection) that maps query embeddings to target embeddings:

```
projected_query = query_embedding @ W
```

Properties:
- `W.T @ W ≈ I` (orthogonal)
- Preserves embedding norms and angles
- Computed via SVD (Schönemann, 1966)

## Training Data Format (JSONL)

The training JSONL contains one record per tree:

```json
{
  "type": "Tree",
  "target_text": "/path/ids\n- account\n  - folder\n    - subfolder\n      - Title",
  "raw_title": "Title",
  "query": "locate_tree('Title')",
  "cluster_id": "https://www.pearltrees.com/account/folder/id123",
  "tree_id": "123456",
  "account": "s243a",
  "uri": "https://www.pearltrees.com/account/title/id123456"
}
```

Key fields:
- `raw_title`: Used as query text for embedding
- `target_text`: Used as target text for embedding (includes path hierarchy)
- `tree_id`: Unique identifier, used for account lookup
- `account`: Account name (s243a, s243a_groups, etc.)

## Account-Specific Models

For cross-account operations, separate models can be trained per account:

| Model | Data | Clusters | Use Case |
|-------|------|----------|----------|
| `pearltrees_federated_single.pkl` | All accounts | 51 | General search |
| `pearltrees_federated_s243a.pkl` | s243a only | 275 | s243a-specific search |
| `pearltrees_federated_s243a_groups.pkl` | s243a_groups only | 48 | Cross-account migration |

The `data_path` field enables automatic account lookup at inference time.

## Inference Pipeline

```
1. Embed query text
2. Compute similarity to cluster centroids
3. Soft-route: weight clusters by softmax(similarity / temperature)
4. Project query through weighted sum of cluster W matrices
5. Compare projected query to all target embeddings
6. Return top-k by cosine similarity
```

## Memory Considerations

| Component | Size Formula | Example (768-dim, 275 clusters) |
|-----------|--------------|--------------------------------|
| W matrix | `768 × 768 × 4 bytes` | 2.4 MB per cluster |
| Cluster centroids | `n_clusters × 768 × 4` | 0.8 MB |
| Target embeddings | `n_items × 768 × 4` | 27 MB (8800 items) |
| Total model | ~2.4 MB × n_clusters + overhead | ~665 MB |

## Related Documentation

- [Theory: Procrustes Analysis](../../sandbox/paper-minimum-projection/theory_book.md)
- [Book 13: Semantic Search](https://github.com/s243a/UnifyWeaver/blob/main/education/book-13-semantic-search/README.md)
- [Bookmark Filing Guide](https://github.com/s243a/UnifyWeaver/blob/main/education/book-13-semantic-search/16_bookmark_filing.md)

## Clustering Methods

The training script supports four clustering methods via `--cluster-method`:

| Method | Description | Use Case |
|--------|-------------|----------|
| `mst` | MST edge-cutting on embeddings | Best semantic coherence, recommended |
| `embedding` | K-means on target embeddings | Fast, good baseline |
| `path_depth` | Group by materialized path depth | Simple hierarchy-based grouping |
| `per-tree` | One cluster per parent tree | Preserves Pearltrees folder structure |

### MST Clustering

The `mst` method builds a Minimum Spanning Tree on cosine distances between embeddings, then cuts the longest edges to form clusters. This preserves local neighborhood structure better than K-means.

**Advantages:**
- Items in a cluster are similar to their neighbors (not just to a centroid)
- Better projection quality due to more coherent training signal
- Fewer duplicate results in search output

**Parameters:**
- `--max-clusters`: Maximum clusters to create (cuts N-1 longest edges)
- `--min-cluster-size`: Merge clusters smaller than this (default: 10)

**Example:**
```bash
python3 scripts/train_pearltrees_federated.py data.jsonl model.pkl \
  --cluster-method mst \
  --max-clusters 50 \
  --min-cluster-size 10
```

See `docs/design/MST_CLUSTERING_*.md` for detailed design documentation.

### Per-Tree Clustering

The `per-tree` method creates one cluster per Pearltrees folder, grouping items with their siblings based on the `cluster_id` field (which contains the parent tree URI).

**When to use:**
- You want to preserve the existing tree structure
- Each folder should have its own W matrix for projection
- You're using smaller models (MiniLM) since there will be many clusters

**Example:**
```bash
python3 scripts/train_pearltrees_federated.py data.jsonl model.pkl \
  --cluster-method per-tree \
  --model sentence-transformers/all-MiniLM-L6-v2
```

**Memory considerations for per-tree:**
- Many small clusters (~4 items avg per tree)
- With MiniLM (384-dim): ~590 KB per cluster
- With Nomic (768-dim): ~2.4 MB per cluster
- Recommended: Use MiniLM for per-tree to manage memory

### Cluster File Naming

| Method | File Pattern | Example |
|--------|--------------|---------|
| `embedding` | `cluster_N.npz` | `cluster_0.npz`, `cluster_1.npz` |
| `path_depth` | `depth_N.npz` or `cluster_N.npz` | `depth_0.npz`, `cluster_0.npz` |
| `per-tree` | `tree_ID.npz` | `tree_12345.npz`, `tree_67890.npz` |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_pearltrees_federated.py` | Train a new model |
| `scripts/infer_pearltrees_federated.py` | Run inference |
| `scripts/generate_account_training_data.py` | Filter JSONL by account |
| `scripts/pearltrees_multi_account_generator.py` | Generate training data from RDF |

### Training Script Options

```bash
python3 scripts/train_pearltrees_federated.py INPUT_JSONL OUTPUT_MODEL \
  --cluster-method {mst,embedding,path_depth,per-tree} \
  --transform-mode {single,per-query} \
  --max-clusters N \
  --min-cluster-size N \
  --max-memory-mb N \
  --model MODEL_NAME
```

| Option | Default | Description |
|--------|---------|-------------|
| `--cluster-method` | `embedding` | Clustering algorithm (`mst` recommended) |
| `--transform-mode` | `single` | `single` (1 W/cluster) or `per-query` (N W/cluster) |
| `--max-clusters` | `50` | Max clusters for embedding/mst/path_depth methods |
| `--min-cluster-size` | `10` | Min cluster size (MST method only) |
| `--max-memory-mb` | `800` | Max memory per cluster (per-query mode) |
| `--model` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |
