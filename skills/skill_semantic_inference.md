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

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Path to trained .pkl model | Required |
| `--query` | Query text for inference | - |
| `--input` | Input file for batch mode | - |
| `--output` | Output file for results | stdout |
| `--top-k` | Number of results to return | 10 |
| `--threshold` | Minimum similarity score | 0.0 |
| `--cluster` | Restrict to specific cluster | all |

## Related

**Skills:**
- `skill_train_model.md` - Train the model first
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
