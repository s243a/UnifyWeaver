# Mindmap Repair Workflow

This document describes the process for identifying and repairing incomplete mindmaps (trees with only a root node and no children).

## Overview

Pearltrees RDF exports may contain trees where children weren't properly fetched. This workflow:
1. Scans for incomplete mindmaps
2. Groups children by parent tree from RDF
3. Repairs mindmaps by loading children from the index
4. Prioritizes remaining repairs using semantic filtering

## Files and Tools

| File | Description |
|------|-------------|
| `scripts/scan_incomplete_mindmaps.py` | Scan for mindmaps with ≤N nodes |
| `scripts/build_children_index.py` | Build parent→children SQLite index from RDF |
| `scripts/repair_incomplete_mindmaps.py` | Batch repair using children index |
| `scripts/organize_incomplete_trees.py` | Organize remaining trees hierarchically |

## Step 1: Scan for Incomplete Mindmaps

```bash
python3 scripts/scan_incomplete_mindmaps.py output/mindmaps_curated/ \
  --threshold 1 \
  --output .local/data/scans/incomplete_mindmaps.json
```

Options:
- `--threshold N` - Max topic count to consider incomplete (default: 1)
- `--format urls` - Output Pearltrees URLs for fetching
- `--format tree` - Show hierarchical summary
- `--exclude-private` - Skip private trees

## Step 2: Build Children Index from RDF

```bash
python3 scripts/build_children_index.py \
  --rdf data/rdf/pearltrees_export.rdf \
  --output .local/data/children_index.db
```

This parses the RDF and groups children by `pt:parentTree`, storing:
- `parent_tree_id` - Tree ID
- `pearl_type` - PagePearl, RefPearl, AliasPearl, SectionPearl, etc.
- `title`, `pos_order`, `external_url`, `see_also_uri`

## Step 3: Repair Incomplete Mindmaps

```bash
# Dry run first
python3 scripts/repair_incomplete_mindmaps.py \
  --scan .local/data/scans/incomplete_mindmaps.json \
  --children-index .local/data/children_index.db \
  --output-dir output/mindmaps_curated \
  --dry-run

# Then actual repair
python3 scripts/repair_incomplete_mindmaps.py \
  --scan .local/data/scans/incomplete_mindmaps.json \
  --children-index .local/data/children_index.db \
  --output-dir output/mindmaps_curated
```

## Step 4: Organize Remaining Incomplete Trees

After repair, some trees may still be empty (no children in RDF). Organize them hierarchically for review:

```bash
python3 scripts/organize_incomplete_trees.py \
  --scan .local/data/scans/incomplete_mindmaps.json \
  --data reports/pearltrees_targets_full_multi_account.jsonl \
  --format tree --show-urls \
  --output .local/data/scans/incomplete_tree_full.txt
```

## Step 5: Semantic Filtering for Priority Fetching

Filter incomplete trees by semantic similarity to a topic:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Load or create embeddings from hierarchical paths
nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Embed each tree's hierarchical path (from JSONL target_text)
embeddings = nomic.encode(path_texts)

# Query by topic
query_emb = nomic.encode(["science"])[0]

# Or use existing output embedding from a related tree
# This captures structural relationships, not just lexical
science_tree_emb = output_nomic[science_tree_idx]

# Compute cosine similarities
similarities = [cosine_sim(query_emb, emb) for emb in embeddings]
```

### Two Filtering Approaches

1. **Lexical query** (e.g., embed "science"):
   - Finds trees with similar words in title/path
   - Good for topic-based filtering

2. **Structural query** (use existing tree's output embedding):
   - Finds trees in similar hierarchical positions
   - Captures organizational relationships

### Generate Filtered Tree View

```bash
python3 scripts/organize_incomplete_trees.py \
  --scan .local/data/scans/incomplete_mindmaps.json \
  --filter-embeddings .local/data/scans/incomplete_embeddings.npz \
  --query "science" \
  --top-k 10 \
  --format tree --show-urls
```

## Repair Statistics Example

After running the full workflow:

| Status | Count |
|--------|-------|
| Original incomplete | 3,720 |
| Repaired from RDF | 1,074 |
| Remaining (no children in RDF) | 2,646 |

The remaining 2,646 trees need API fetching to repair.

## API Fetching (Future Work)

For trees without children in RDF, use Playwright to fetch from Pearltrees:

```bash
# Get prioritized URLs
python3 scripts/organize_incomplete_trees.py \
  --scan .local/data/scans/incomplete_mindmaps.json \
  --format urls \
  --filter-query "science" \
  --top-k 50 > urls_to_fetch.txt

# Fetch with rate limiting (example)
# scripts/fetch_pearltrees_batch.py --urls urls_to_fetch.txt --delay 5
```

## Semantic Filtering Methods

The `filter_incomplete_trees.py` script supports multiple filtering approaches:

### 1. Lexical Filtering
Embed the query text directly. Good for finding trees with similar words:

```bash
python3 scripts/filter_incomplete_trees.py --query "science" --top-k 10
```

### 2. Structural Filtering
Use an existing tree's output embedding. Finds trees in similar hierarchy positions:

```bash
python3 scripts/filter_incomplete_trees.py --tree-query "science" --top-k 10
```

### 3. Blended Filtering
Combine both methods with alpha weighting:

```bash
python3 scripts/filter_incomplete_trees.py \
  --query "science" \
  --tree-query "science" \
  --alpha 0.5 \
  --top-k 10 --format tree
```

Output shows both scores: `[L:0.775 S:0.662 = 0.718]`

## Integration with Bookmark Filing Assistant

This mindmap repair workflow complements the bookmark filing assistant:

1. **Repair Phase**: Fix incomplete mindmaps to improve folder coverage
2. **Filing Phase**: Use repaired hierarchy for better bookmark filing suggestions
3. **Feedback Loop**: Missing trees identified during filing can trigger repair

### Future Integration Points

- `scripts/bookmark_filing_assistant.py` can use repair status to deprioritize incomplete folders
- Dual-objective scoring can leverage both input (semantic) and output (structural) embeddings
- Users without API access can still benefit from RDF-based repairs

### Embedding Reuse

The embeddings created for filtering (`incomplete_embeddings.npz`) can be:
- Cached for fast repeated queries
- Used for clustering to identify related incomplete trees
- Merged with the main federated model for unified search

## Related Documentation

- `docs/proposals/pearltrees_unifyweaver_native.md` - Future UnifyWeaver-native approach using aggregates
- `docs/ai-skills/bookmark-filing-agent.md` - Semantic search for bookmark filing
- `scripts/filter_incomplete_trees.py` - Semantic filtering tool
