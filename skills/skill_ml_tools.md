# Skill: ML Tools (Sub-Master)

Machine learning tools for embeddings, semantic search, model training, inference, and hierarchy optimization.

## When to Use

- User asks "which embedding model should I use?"
- User wants to train a semantic search model
- User needs to run inference or folder suggestion
- User asks about visualizing embeddings or clusters
- User wants to evaluate hierarchy quality with J = D/(1+H)
- User asks about Procrustes projection or federation

## Skill Hierarchy

```
skill_data_tools.md (parent)
└── skill_ml_tools.md (this file)
    ├── skill_embedding_models.md - Model selection (nomic, MiniLM, BERT, ModernBERT)
    ├── skill_density_explorer.md - Interactive embedding visualization
    ├── skill_train_model.md - Federated model training with Procrustes
    ├── skill_semantic_inference.md - Running inference for folder suggestion
    └── skill_hierarchy_objective.md - J = D/(1+H) optimization
```

## Quick Start

### Choose an Embedding Model

| Use Case | Model | Why |
|----------|-------|-----|
| Q/A Semantic Search | nomic-embed-text-v1.5 | Best asymmetric search quality |
| Fast/Lightweight | all-MiniLM-L6-v2 | Smallest, fastest |
| Entropy Computation | BERT or ModernBERT | Calibrated logits |
| Go/Rust/C# Deployment | all-MiniLM-L6-v2 | Widest ONNX support |

### Train a Model

```bash
python3 scripts/train_pearltrees_federated.py \
  reports/pearltrees_targets.jsonl \
  models/federated.pkl \
  --model nomic-ai/nomic-embed-text-v1.5 \
  --cluster-method mst
```

### Run Inference

```bash
python3 scripts/infer_pearltrees_federated.py \
  --model models/federated.pkl \
  --query "quantum computing basics" \
  --top-k 5
```

### Evaluate Hierarchy

```bash
python3 scripts/mindmap/hierarchy_objective.py \
  --tree hierarchy.json \
  --embeddings embeddings.npy
```

### Visualize Embeddings

```bash
# Start density explorer
python tools/density_explorer/flask_api.py --port 5000
cd tools/density_explorer/vue && npm run dev
# Open http://localhost:5173
```

## The ML Pipeline

```
1. Data Preparation
   └── Extract text, generate embeddings

2. Model Training
   └── Procrustes projection, MST clustering

3. Inference
   └── Query → embedding → projection → ranked results

4. Evaluation
   └── J = D/(1+H) objective, depth-surprisal correlation

5. Visualization
   └── Density explorer, tree overlay, re-rooting
```

## Embedding Models

### Target Runtime Support

| Model | Python | Go | Rust | C# |
|-------|--------|----|----|---|
| nomic-embed-text-v1.5 | Full | - | - | - |
| all-MiniLM-L6-v2 | Full | Full | Full | Full |
| BERT-base | Full | Full | Full | - |
| ModernBERT | Full | - | Full | - |

### Model Properties

| Model | Dimensions | Context | Speed | Notes |
|-------|------------|---------|-------|-------|
| nomic-embed-text-v1.5 | 768 | 8192 | Moderate | Asymmetric prefixes |
| all-MiniLM-L6-v2 | 384 | 256 | Very fast | Resource-constrained |
| BERT-base | 768 | 512 | Moderate | Fine-tunable |
| ModernBERT | 768 | 8192 | Fast | Needs transformers >= 4.48 |

## Training Concepts

### Procrustes Projection

Learns orthogonal transformation from query space to document space:

```
Q × W ≈ D
```

Where:
- Q = query embeddings
- W = learned projection matrix
- D = document embeddings

### MST Clustering

Minimum Spanning Tree clustering preserves hierarchy:

```bash
python3 scripts/train_pearltrees_federated.py \
  data.jsonl output.pkl \
  --cluster-method mst \
  --max-clusters 50
```

### Federation

Separate projection matrices per cluster for better local accuracy:

```python
# Federated model structure
{
    "clusters": {
        "science": {"W": [...], "centroid": [...], "items": [...]},
        "technology": {"W": [...], "centroid": [...], "items": [...]}
    },
    "global_W": [...],
    "embedding_cache": {...}
}
```

## Hierarchy Objective

### The J = D/(1+H) Formula

Good hierarchies have:
- **Low D** - nodes close to their parent (tight clusters)
- **High H** - each level adds meaning (informative splits)

```
J = D / (1 + H)
```

**Lower J = better hierarchy.**

### Entropy Sources

| Source | Speed | Accuracy | Requirements |
|--------|-------|----------|--------------|
| Fisher | Fast | Good | Embeddings only (default) |
| Logits | Slow | Best | Text + transformer model |

```bash
# Fisher entropy (fast)
--entropy-source fisher

# BERT logits entropy (accurate)
--entropy-source logits --entropy-model bert-base-uncased
```

### Quality Metrics

```
Hierarchy Statistics:
  Objective J: 0.0955      # Lower is better (< 0.2 good)
  Semantic Distance D: 0.2341
  Entropy Gain H: 1.4532
  Depth-Surprisal Corr: 0.65  # Higher is better (> 0.5 good)
```

## Visualization with Density Explorer

### Features

- **Density Heatmap** - KDE visualization of embedding space
- **Tree Overlay** - MST or J-guided tree structure
- **Node Selection** - Click to inspect, re-root tree
- **Depth Filtering** - Show only first N levels
- **Export** - JSON with depth filtering

### Tree Types

| Type | Description | Use Case |
|------|-------------|----------|
| MST | Minimum Spanning Tree | Fast, globally optimal |
| J-Guided | Optimizes J at each step | Better semantic hierarchy |

### Re-rooting

Re-rooting creates a focal point for exploration:
1. Click to select a node
2. Re-root action rebuilds tree
3. Related concepts radiate outward

## Common Workflows

### Semantic Search Setup

```bash
# 1. Generate embeddings
python3 scripts/generate_embeddings.py \
  --input data.jsonl \
  --output embeddings.npz \
  --model nomic-ai/nomic-embed-text-v1.5

# 2. Train projection
python3 scripts/train_pearltrees_federated.py \
  data.jsonl models/search.pkl \
  --cluster-method mst

# 3. Test inference
python3 scripts/infer_pearltrees_federated.py \
  --model models/search.pkl \
  --query "test query"
```

### Hierarchy Analysis

```bash
# 1. Build tree
python3 scripts/mindmap/build_mst_index.py \
  --embeddings embeddings.npy \
  --output tree.json

# 2. Evaluate quality
python3 scripts/mindmap/hierarchy_objective.py \
  --tree tree.json \
  --embeddings embeddings.npy

# 3. Visualize
python tools/density_explorer/flask_api.py
```

### Bookmark Filing

```bash
# File a new bookmark
python3 scripts/bookmark_filing_assistant.py \
  --model models/federated.pkl \
  --bookmark "https://example.com/article" \
  --title "Article Title" \
  --suggest-folders 5
```

## Child Skills

- `skill_embedding_models.md` - Model selection and configuration
- `skill_density_explorer.md` - Interactive visualization
- `skill_train_model.md` - Training workflows
- `skill_semantic_inference.md` - Running inference
- `skill_hierarchy_objective.md` - J = D/(1+H) theory and implementation

## Related

**Parent Skill:**
- `skill_data_tools.md` - Data tools master

**Sibling Sub-Masters:**
- `skill_query_patterns.md` - Query and aggregation
- `skill_data_sources.md` - Data source handling

**Documentation:**
- `docs/design/FEDERATED_MODEL_FORMAT.md` - Model format spec
- `docs/design/MST_IMPROVEMENTS_PROPOSAL.md` - J-guided improvements
- `docs/QUICKSTART_MINDMAP_LINKING.md` - End-to-end workflow

**Education:**
- `education/book-13-semantic-search/` - Semantic search concepts
- `education/book-14-ai-training/` - Training and embeddings

**Code:**
- `scripts/train_pearltrees_federated.py` - Training script
- `scripts/infer_pearltrees_federated.py` - Inference script
- `scripts/mindmap/hierarchy_objective.py` - J objective
- `tools/density_explorer/` - Visualization tool
