# Quickstart: Mindmap Linking with Pearltrees

This guide shows how to enrich exported SimpleMind mindmaps with Pearltrees links using hierarchical semantic matching.

## Prerequisites

1. **Python 3.8+** with dependencies:
   ```bash
   pip install sentence-transformers numpy
   ```

2. **Data files** (one or more):
   - JSONL targets file (e.g., `reports/pearltrees_targets_s243a.jsonl`)
   - Embeddings file (e.g., `datasets/pearltrees_combined_embeddings.npz`)
   - URL database (e.g., `data/databases/children_index.db`)

3. **Projection model** (for hierarchical matching):
   - Federated Procrustes: `models/pearltrees_federated_nomic.pkl`
   - Or distilled transformer: `models/pearltrees_transformer.pt`

## Quick Start

### Basic Linking (URL + Title Matching)

```bash
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap your_mindmap.smmx \
  --trees reports/pearltrees_targets_s243a.jsonl \
  --output your_mindmap_linked.smmx
```

### With Hierarchical Matching (Recommended)

```bash
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap your_mindmap.smmx \
  --trees reports/pearltrees_targets_s243a.jsonl \
  --embeddings datasets/pearltrees_combined_embeddings.npz \
  --projection-model models/pearltrees_federated_nomic.pkl \
  --output your_mindmap_linked.smmx
```

### Preview Without Writing

```bash
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap your_mindmap.smmx \
  --trees reports/pearltrees_targets_s243a.jsonl \
  --dry-run --verbose
```

## What Gets Linked

The tool adds child nodes to matching topics:

| Child Label | Meaning | Match Type |
|-------------|---------|------------|
| **PP** | PearlPage | URL matches a bookmarked page (requires `--url-db` or `--api-cache`) |
| **PT** | Pearltree | Exact title match (including disambiguated multiple matches) |
| **PT?** | Pearltree (fuzzy) | Semantic similarity match (no exact title) |

## Understanding Projection Models

### Why Projection?

When the same URL exists in multiple Pearltrees folders, the projection model selects the best location by comparing against **hierarchical context**, not just semantic similarity.

The model encodes:
- **Materialized path**: `/account/folder/subfolder/tree_id`
- **Structure list**: Indented hierarchy showing folder nesting

This means "Machine Learning" bookmarked in `STEM/AI/Deep Learning` will score higher when the mindmap context is about neural networks than the same bookmark in `Misc/Unsorted`.

### Model Types

| Type | File | Description |
|------|------|-------------|
| **Federated Procrustes** | `.pkl` | Cluster-based projection with routing. Faster inference, interpretable. |
| **Distilled Transformer** | `.pt` | Neural approximation. Single model, slightly slower. |

Both produce equivalent results. Use `.pkl` for production, `.pt` for experimentation.

### Model Format Details

See [`docs/design/FEDERATED_MODEL_FORMAT.md`](design/FEDERATED_MODEL_FORMAT.md) for the complete specification:
- Cluster directory structure
- W matrix storage
- Routing data format

## Generating Models

*Note: Pre-trained models and training documentation are being prepared for HuggingFace distribution.*

### Current Training Pipeline

1. **Prepare JSONL targets** with `target_text` field (materialized paths):
   ```bash
   python3 scripts/prepare_pearltrees_targets.py \
     --input data/pearltrees_export.rdf \
     --output reports/pearltrees_targets.jsonl
   ```

2. **Generate embeddings**:
   ```bash
   python3 scripts/embed_pearltrees_targets.py \
     --input reports/pearltrees_targets.jsonl \
     --model nomic-ai/nomic-embed-text-v1.5 \
     --output datasets/pearltrees_embeddings.npz
   ```

3. **Train federated model**:
   ```bash
   python3 scripts/train_pearltrees_federated.py \
     --embeddings datasets/pearltrees_embeddings.npz \
     --output models/pearltrees_federated.pkl
   ```

## Related Tools

| Tool | Purpose |
|------|---------|
| `scripts/infer_pearltrees_federated.py` | Bookmark filing assistant |
| `scripts/mindmap/add_relative_links.py` | Add `cloudmapref` cross-links |
| `scripts/mindmap/mst_folder_grouping.py` | MST-based folder organization |

## Troubleshooting

### "No module named 'numpy._core'"

NumPy version mismatch. The model was saved with NumPy 2.0+. The tool includes a compatibility shim, but if issues persist:
```bash
pip install --upgrade numpy
```

### Low match rate

- Check `--threshold` (default 0.7, lower = more matches)
- Ensure embeddings contain your Pearltrees data
- Use `--verbose` to see scoring details

### Wrong locations selected

The projection model may need retraining with your specific hierarchy. Check that your JSONL includes `target_text` with proper materialized paths.

## Further Reading

- [`scripts/mindmap/README.md`](../scripts/mindmap/README.md) - Full mindmap tools documentation
- [`docs/design/FEDERATED_MODEL_FORMAT.md`](design/FEDERATED_MODEL_FORMAT.md) - Model format specification
- [`skills/skill_bookmark_filing.md`](../skills/skill_bookmark_filing.md) - AI skill for bookmark filing
