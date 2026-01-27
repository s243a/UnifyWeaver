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
  --cluster-method embedding \
  --cluster-criterion effective_rank \
  --model nomic-ai/nomic-embed-text-v1.5
```

## Commands

### Train with Default Settings
```bash
python3 scripts/train_pearltrees_federated.py \
  "INPUT_JSONL" \
  "OUTPUT_MODEL.pkl"
```

### Train with Optimal Cluster Count (Recommended)
```bash
python3 scripts/train_pearltrees_federated.py \
  "INPUT_JSONL" \
  "OUTPUT_MODEL.pkl" \
  --cluster-method embedding \
  --cluster-criterion effective_rank
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
| `embedding` | K-means on answer embeddings - uniform cluster sizes, good for RAG |
| `mst` | MST edge-cutting - preserves local topology, for hierarchical data |
| `per-tree` | One cluster per source tree/folder |
| `path_depth` | Cluster by materialized path depth |

**When to use which:**
- **`embedding` (K-means)**: Recommended for RAG and semantic search. Produces more uniform cluster sizes, faster training.
- **`mst`**: Better for hierarchical applications where preserving parent-child relationships matters. However, the J-guided hierarchy objective (`skill_hierarchy_objective.md`) is even better for hierarchical optimization.

## Cluster Count Optimization

Use `--cluster-criterion` to auto-select optimal cluster count:

| Criterion | Description |
|-----------|-------------|
| `effective_rank` | Recommended - spectral participation ratio of answer embeddings |
| `covering` | 2^r where r dimensions capture target variance (default 80%) |
| `sqrt_n` | √N heuristic (common K-means rule of thumb) |
| `auto` | Same as effective_rank |

**Example with auto cluster count:**
```bash
python3 scripts/train_pearltrees_federated.py \
  input.jsonl output.pkl \
  --cluster-method embedding \
  --cluster-criterion effective_rank
```

**Override with fixed cluster count:**
```bash
python3 scripts/train_pearltrees_federated.py \
  input.jsonl output.pkl \
  --max-clusters 50
```

For covering method, adjust target variance:
```bash
python3 scripts/train_pearltrees_federated.py \
  input.jsonl output.pkl \
  --cluster-criterion covering \
  --target-variance 0.90
```

## Output

Creates a `.pkl` file containing:
- Federated W matrices (one per cluster)
- Cluster assignments
- Embedding cache

## Fast Inference: Orthogonal Codebook (Recommended for Mobile)

For mobile/edge deployment, train an orthogonal codebook transformer:

```bash
python3 scripts/train_orthogonal_codebook.py \
  --train-multisource \
  --federated-models models/skills_qa_federated.pkl models/books_qa_federated.pkl \
  --codebook-method canonical \
  --n-components 64 \
  --epochs 50 \
  --save-transformer models/orthogonal_transformer.pt
```

**Key benefits:**
- **39× faster** than weighted baseline (27K/s vs 700/s)
- **Matches raw embedding quality** (-0.2% Hit@1)
- Uses Rodrigues formula: O(K×d) vs O(d³) matrix exp

**Why it works:**
- Canonical planes are perfectly commutative (independent learning)
- Nomic's Matryoshka structure means 64 planes (dims 0-127) covers semantic core

**When to use:**
| Approach | Use Case |
|----------|----------|
| Orthogonal codebook | Mobile/edge, fast inference |
| Weighted baseline | Server-side, many clusters |
| Full rotation | Maximum accuracy |

**Note:** 64 planes works well with Matryoshka embeddings (Nomic, OpenAI text-embedding-3). For non-Matryoshka models, you may need more planes (128+).

See `docs/design/ORTHOGONAL_CODEBOOK_DESIGN.md` for theory and details.

## Model Registry

Trained models are tracked in the Model Registry. Query training commands:

```bash
# Show training command for a model
python3 -m unifyweaver.config.model_registry --training pearltrees_federated_nomic

# Check if model is available
python3 -m unifyweaver.config.model_registry --check pearltrees_federated_nomic

# List missing models for a task
python3 -m unifyweaver.config.model_registry --missing bookmark_filing
```

See `skill_model_registry.md` for full registry documentation.

## Related

**Parent Skill:**
- `skill_ml_tools.md` - ML tools sub-master

**Sibling Skills:**
- `skill_model_registry.md` - Model discovery and selection
- `skill_semantic_inference.md` - Run inference with trained model
- `skill_embedding_models.md` - Model selection
- `skill_hierarchy_objective.md` - Hierarchy optimization
- `skill_density_explorer.md` - Visualization

**Other Skills:**
- `skill_folder_suggestion.md` - Suggest folders for mindmaps
- `skill_bookmark_filing.md` - File bookmarks automatically

**Documentation:**
- `docs/design/FEDERATED_MODEL_FORMAT.md` - Model format specification
- `docs/design/ORTHOGONAL_CODEBOOK_DESIGN.md` - Fast inference for mobile
- `docs/QUICKSTART_MINDMAP_LINKING.md` - End-to-end workflow

**Education (in `education/` subfolder):**
- `book-14-ai-training/05_training_pipeline.md` - Training concepts
- `book-14-ai-training/03_lda_projection.md` - Projection methods (note: Procrustes preferred over LDA)
- `book-14-ai-training/02_embedding_providers.md` - Embedding model options
- `book-13-semantic-search/08_advanced_federation.md` - Federation architecture

**Code:**
- `scripts/train_pearltrees_federated.py` - Main training script
- `scripts/mindmap/hierarchy_objective.py` - Clustering objective function
