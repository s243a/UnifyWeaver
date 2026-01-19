# Skill: Mindmap Linking

Enrich SimpleMind mindmaps with Pearltrees links using hierarchical semantic matching.

## When to Use

- User has exported a mindmap and wants to connect it to Pearltrees
- User asks to "link", "enrich", or "connect" a mindmap to Pearltrees

## Quick Start

```bash
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap "MINDMAP_PATH" \
  --trees reports/pearltrees_targets_s243a.jsonl \
  --projection-model models/pearltrees_federated_nomic.pkl \
  --output "OUTPUT_PATH"
```

## Commands

### Basic Linking (Title Matching Only)
```bash
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap "MINDMAP_PATH" \
  --trees reports/pearltrees_targets_s243a.jsonl \
  --output "OUTPUT_PATH"
```

### With URL Matching
```bash
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap "MINDMAP_PATH" \
  --trees reports/pearltrees_targets_s243a.jsonl \
  --url-db data/databases/children_index.db \
  --output "OUTPUT_PATH"
```

### With Hierarchical Projection (Recommended)
```bash
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap "MINDMAP_PATH" \
  --trees reports/pearltrees_targets_s243a.jsonl \
  --embeddings datasets/pearltrees_combined_embeddings.npz \
  --projection-model models/pearltrees_federated_nomic.pkl \
  --output "OUTPUT_PATH"
```

### Full Options
```bash
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap "MINDMAP_PATH" \
  --trees reports/pearltrees_targets_s243a.jsonl \
  --pearls reports/pearltrees_targets_full_pearls.jsonl \
  --url-db data/databases/children_index.db \
  --embeddings datasets/pearltrees_combined_embeddings.npz \
  --projection-model models/pearltrees_federated_nomic.pkl \
  --threshold 0.7 \
  --output "OUTPUT_PATH" \
  --verbose
```

### Preview Changes (Dry Run)
```bash
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap "MINDMAP_PATH" \
  --trees reports/pearltrees_targets_s243a.jsonl \
  --dry-run --verbose
```

### Get All Options
```bash
python3 scripts/mindmap/link_pearltrees.py --help
```

## Output Labels

Child nodes are added with these labels:

| Label | Meaning | Trigger |
|-------|---------|---------|
| **PP** | PearlPage | Node URL matches PagePearl's external URL |
| **PT** | Pearltree | Exact title match (including disambiguated) |
| **PT?** | Pearltree (fuzzy) | Semantic similarity above threshold |

## Model Information

Uses the same models as bookmark filing:

| Model | Embedder | Use Case |
|-------|----------|----------|
| `pearltrees_federated_nomic.pkl` | Nomic (768D) | General matching |
| `pearltrees_federated_single.pkl` | MiniLM (384D) | Faster inference |

Model auto-selects embedder based on dimension.

## Data Requirements

| Data | Flag | Purpose |
|------|------|---------|
| Trees JSONL | `--trees` | Title matching, semantic search |
| Pearls JSONL | `--pearls` | Additional title matching |
| URL Database | `--url-db` | URL matching (for PP labels) |
| Embeddings | `--embeddings` | Semantic similarity search |
| Projection Model | `--projection-model` | Hierarchical disambiguation |

Minimum: `--trees` for basic title matching.
Recommended: All options for best matching quality.

## Integration with Bookmark Filing

This skill complements bookmark filing:

1. **Bookmark Filing**: Finds where to save a new bookmark in Pearltrees
2. **Mindmap Linking**: Enriches existing mindmaps with links to Pearltrees content

Both use the same projection model and hierarchical matching approach.

## Related

**Skills:**
- `skill_bookmark_filing.md` - Filing new bookmarks (same projection model)
- `skill_train_model.md` - Train the projection model
- `skill_semantic_inference.md` - General inference concepts
- `skill_mst_folder_grouping.md` - Group folders with MST

**Documentation:**
- `docs/QUICKSTART_MINDMAP_LINKING.md` - Detailed guide with explanations
- `docs/design/FEDERATED_MODEL_FORMAT.md` - Model format specification
- `scripts/mindmap/README.md` - Full mindmap tools documentation

**Education (in `education/` subfolder):**
- `book-13-semantic-search/01_introduction.md` - Semantic search overview
- `book-13-semantic-search/05_semantic_playbook.md` - Best practices
- `book-13-semantic-search/07_density_scoring.md` - Scoring methods
- `book-13-semantic-search/08_advanced_federation.md` - Federation architecture
- `book-13-semantic-search/15_zero_shot_path_mapping.md` - Zero-shot inference
- `book-14-ai-training/05_training_pipeline.md` - Model training

**Code:**
- `scripts/mindmap/link_pearltrees.py` - Main linking script
- `scripts/infer_pearltrees_federated.py` - Projection model inference (same approach)
