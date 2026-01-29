# Skill: Semantic Inference

Run inference using trained federated models for folder suggestion, bookmark filing, and semantic search.

## When to Use

- User wants to find similar folders or documents
- User asks "where should I file this bookmark?"
- User wants to run semantic search on their data
- User has a trained model and wants predictions

## Quick Start

```bash
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated.pkl \
  --query "quantum computing basics" \
  --top-k 5
```

## Commands

### Basic Inference
```bash
python3 scripts/infer_pearltrees_federated.py \
  --model "MODEL.pkl" \
  --query "QUERY_TEXT"
```

### Batch Inference from File
```bash
python3 scripts/infer_pearltrees_federated.py \
  --model "MODEL.pkl" \
  --input queries.txt \
  --output results.jsonl
```

### Inference with Specific Cluster
```bash
python3 scripts/infer_pearltrees_federated.py \
  --model "MODEL.pkl" \
  --query "QUERY_TEXT" \
  --cluster science
```

### Bookmark Filing Assistant
```bash
python3 scripts/bookmark_filing_assistant.py \
  --model models/pearltrees_federated.pkl \
  --bookmark "https://example.com/article" \
  --title "Interesting Article" \
  --suggest-folders 5
```

### Get All Options
```bash
python3 scripts/infer_pearltrees_federated.py --help
```

## Output Format

```json
{
  "query": "quantum computing basics",
  "results": [
    {
      "path": "Science/Physics/Quantum Mechanics",
      "score": 0.89,
      "cluster": "science"
    },
    {
      "path": "Technology/Computing/Quantum",
      "score": 0.82,
      "cluster": "technology"
    }
  ]
}
```

## Inference Modes

| Mode | Use Case |
|------|----------|
| `single` | One query at a time (default) |
| `batch` | Process file of queries |
| `interactive` | REPL for testing |
| `server` | HTTP API endpoint |

### Interactive Mode

Enter a REPL for testing queries interactively:

```bash
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated.pkl \
  --interactive
```

**REPL Commands:**

| Command | Description |
|---------|-------------|
| `<query text>` | Search for matching folders |
| `:top N` | Set number of results (default: 10) |
| `:cluster NAME` | Restrict to specific cluster |
| `:cluster all` | Search all clusters |
| `:json` | Toggle JSON output format |
| `:tree` | Toggle tree output format |
| `:help` | Show available commands |
| `:quit` or `Ctrl+D` | Exit the REPL |

**Example Session:**

```
$ python3 scripts/infer_pearltrees_federated.py --model model.pkl --interactive
Loaded model with 275 clusters
> quantum computing

Results:
  [0.89] Science/Physics/Quantum Mechanics
  [0.82] Technology/Computing/Quantum
  [0.75] Science/Physics/Theory

> :top 3
Set top_k to 3

> :cluster science
Filtering to cluster: science

> machine learning
Results:
  [0.91] Science/AI/Machine Learning
  [0.84] Science/AI/Deep Learning
  [0.78] Science/Statistics

> :quit
Goodbye!
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Path to trained .pkl model | From registry |
| `--infer MODEL` | Model name from registry | - |
| `--query` | Query text for inference | - |
| `--top-k` | Number of results to return | 5 |
| `--interactive` | Enter REPL mode | - |
| `--json` | Output as JSON | - |
| `--tree` | Show results as merged hierarchical tree | - |
| `--data` | JSONL data for account lookup and tree mode | - |
| `--tree-data` | Fallback JSONL files for tree display | - |
| `--rdf` | RDF export for parent relationship enrichment | - |
| `--api-trees` | API trees directory for parent info | - |
| `--queue-missing` | Queue trees missing parent info for harvesting | - |
| `--harvest-queue` | Path to harvest queue JSON file | auto |
| `--account` | Filter to single account | - |
| `--accounts` | Filter to multiple accounts (comma-separated) | - |
| `--accounts-tree` | Filter + tree display (shorthand) | - |

## Hierarchical Tree Display

Show results as a merged tree with full folder paths:

```bash
python3 scripts/infer_pearltrees_federated.py \
  --query "quantum physics" \
  --tree \
  --data reports/pearltrees_targets_full_multi_account.jsonl
```

**With RDF enrichment** (fills in missing parent relationships):

```bash
python3 scripts/infer_pearltrees_federated.py \
  --query "quantum physics" \
  --tree \
  --data reports/pearltrees_targets_full_multi_account.jsonl \
  --rdf context/PT/pearltrees_export_s243a.rdf \
  --api-trees .local/data/pearltrees_api/trees
```

**Queue missing data for harvesting:**

```bash
python3 scripts/infer_pearltrees_federated.py \
  --query "quantum physics" \
  --tree \
  --rdf context/PT/pearltrees_export_s243a.rdf \
  --api-trees .local/data/pearltrees_api/trees \
  --queue-missing
```

Trees with incomplete parent info are queued to `{api-trees}/../harvest_queue.json`.

## Account Filtering

Filter results to specific Pearltrees accounts:

```bash
# Single account
python3 scripts/infer_pearltrees_federated.py \
  --query "machine learning" \
  --account s243a

# Multiple accounts with tree display
python3 scripts/infer_pearltrees_federated.py \
  --query "machine learning" \
  --accounts-tree s243a,s243a_groups
```

## Fast Inference with Orthogonal Codebook

For mobile/edge deployment, use the orthogonal codebook transformer:

```bash
# Evaluate Hit@K on orthogonal codebook
python3 scripts/train_orthogonal_codebook.py \
  --hit-at-k \
  --federated-model models/federated.pkl \
  --orthogonal-codebook models/orthogonal_codebook.npz
```

**Performance comparison:**

| Approach | Hit@1 | Speed |
|----------|-------|-------|
| Raw embeddings | 57.5% | baseline |
| **Orthogonal** | **57.3%** | 27,713/s |
| Weighted baseline | 14.1% | 706/s |

The orthogonal codebook is **39× faster** than the weighted baseline while matching raw embedding quality.

See `docs/design/ORTHOGONAL_CODEBOOK_DESIGN.md` for theory.

## Related

**Parent Skill:**
- `skill_ml_tools.md` - ML tools sub-master

**Sibling Skills:**
- `skill_train_model.md` - Train the model first
- `skill_embedding_models.md` - Model selection
- `skill_hierarchy_objective.md` - Hierarchy optimization
- `skill_density_explorer.md` - Visualization

**Other Skills:**
- `skill_bookmark_filing.md` - Specialized bookmark filing
- `skill_folder_suggestion.md` - Suggest folders for mindmaps

**Documentation:**
- `docs/design/FEDERATED_MODEL_FORMAT.md` - Model format specification
- `docs/QUICKSTART_MINDMAP_LINKING.md` - End-to-end workflow

**Education (in `education/` subfolder):**
- `book-13-semantic-search/01_introduction.md` - Semantic search overview
- `book-13-semantic-search/02_graph_rag.md` - Graph-based retrieval
- `book-13-semantic-search/03_semantic_data_pipeline.md` - Data preparation
- `book-13-semantic-search/05_semantic_playbook.md` - Best practices
- `book-13-semantic-search/06_distributed_search.md` - Scaling inference
- `book-13-semantic-search/07_density_scoring.md` - Scoring methods
- `book-13-semantic-search/08_advanced_federation.md` - Federation architecture
- `book-13-semantic-search/13_advanced_routing.md` - Query routing
- `book-13-semantic-search/15_zero_shot_path_mapping.md` - Zero-shot inference
- `book-13-semantic-search/16_bookmark_filing.md` - Bookmark filing workflow

**Code:**
- `scripts/infer_pearltrees_federated.py` - Main inference script
- `scripts/infer_pearltrees.py` - Basic inference (non-federated)
- `scripts/bookmark_filing_assistant.py` - Bookmark filing tool
