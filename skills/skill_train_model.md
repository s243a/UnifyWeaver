# Skill: Train Federated Model

Train semantic search models using Procrustes projection for folder suggestion and bookmark filing.

## When to Use

- User wants to train a model for their data
- User needs to create embeddings for Pearltrees/mindmaps
- User asks "how do I train a model?"
- Before using folder suggestion or bookmark filing

## Quick Start

```bash
python3 scripts/train_pearltrees_federated.py \
  reports/pearltrees_targets.jsonl \
  models/pearltrees_federated.pkl \
  --cluster-method mst \
  --model nomic-ai/nomic-embed-text-v1.5
```

## Commands

### Train with Default Settings
```bash
python3 scripts/train_pearltrees_federated.py \
  "INPUT_JSONL" \
  "OUTPUT_MODEL.pkl"
```

### Train with MST Clustering
```bash
python3 scripts/train_pearltrees_federated.py \
  "INPUT_JSONL" \
  "OUTPUT_MODEL.pkl" \
  --cluster-method mst \
  --max-clusters 50
```

### Train with Specific Embedding Model
```bash
python3 scripts/train_pearltrees_federated.py \
  "INPUT_JSONL" \
  "OUTPUT_MODEL.pkl" \
  --model nomic-ai/nomic-embed-text-v1.5
```

### Get All Options
```bash
python3 scripts/train_pearltrees_federated.py --help
```

## Input Format

JSONL with `target_text` field (materialized paths work best):

```jsonl
{"id": "123", "target_text": "Science/Physics/Quantum Mechanics", "title": "Quantum Computing"}
{"id": "456", "target_text": "Science/Biology/Genetics", "title": "CRISPR Guide"}
```

## Embedding Models

| Model | Dimensions | Notes |
|-------|------------|-------|
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Default, recommended |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Faster, smaller |
| `BAAI/bge-small-en-v1.5` | 384 | Good quality |

## Clustering Methods

| Method | Use Case |
|--------|----------|
| `mst` | Recommended - preserves hierarchy |
| `kmeans` | Faster for large datasets |
| `none` | Single global projection |

## Output

Creates a `.pkl` file containing:
- Federated W matrices (one per cluster)
- Cluster assignments
- Embedding cache

## Related

**Parent Skill:**
- `skill_ml_tools.md` - ML tools sub-master

**Sibling Skills:**
- `skill_semantic_inference.md` - Run inference with trained model
- `skill_embedding_models.md` - Model selection
- `skill_hierarchy_objective.md` - Hierarchy optimization
- `skill_density_explorer.md` - Visualization

**Other Skills:**
- `skill_folder_suggestion.md` - Suggest folders for mindmaps
- `skill_bookmark_filing.md` - File bookmarks automatically

**Documentation:**
- `docs/design/FEDERATED_MODEL_FORMAT.md` - Model format specification
- `docs/QUICKSTART_MINDMAP_LINKING.md` - End-to-end workflow

**Education (in `education/` subfolder):**
- `book-14-ai-training/05_training_pipeline.md` - Training concepts
- `book-14-ai-training/03_lda_projection.md` - Projection methods (note: Procrustes preferred over LDA)
- `book-14-ai-training/02_embedding_providers.md` - Embedding model options
- `book-13-semantic-search/08_advanced_federation.md` - Federation architecture

**Code:**
- `scripts/train_pearltrees_federated.py` - Main training script
- `scripts/mindmap/hierarchy_objective.py` - Clustering objective function
