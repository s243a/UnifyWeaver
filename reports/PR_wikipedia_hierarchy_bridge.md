# Proposal: Wikipedia Hierarchy Bridge for Organizational Metric

## Current Training Pipeline

This section documents the current process for building models used in the density explorer.

### Step 1: Build Federated Model

```bash
python scripts/train_pearltrees_federated.py \
    reports/pearltrees_targets_full_multi_account.jsonl \
    models/pearltrees_federated_nomic.pkl \
    --cluster-method embedding \
    --cluster-criterion effective_rank \
    --model nomic-ai/nomic-embed-text-v1.5
```

**Key options:**
| Option | Values | Current | Description |
|--------|--------|---------|-------------|
| `--cluster-method` | `embedding`, `path_depth`, `per-tree`, `mst` | `embedding` | How to cluster items |
| `--cluster-criterion` | `effective_rank`, `target_variance`, `max_clusters` | `effective_rank` | When to stop clustering |
| `--model` | HuggingFace model ID | `nomic-ai/nomic-embed-text-v1.5` | Embedding model (768D) |

**Output:** `models/pearltrees_federated_nomic.pkl` - federated model with per-cluster rotations

### Step 2: Distill into Transformer

```bash
python scripts/train_orthogonal_codebook.py \
    --train \
    --federated-model models/pearltrees_federated_nomic.pkl \
    --build-canonical \
    --n-components 64 \
    --layers 3 \
    --teacher rotational \
    --transformer-type composed \
    --epochs 50 \
    --output models/bivector_paired.pt
```

**Key options:**
| Option | Values | Current | Description |
|--------|--------|---------|-------------|
| `--teacher` | `rotational`, `jax`, `orthogonal` | `rotational` | Teacher model (rotational = logm/expm distillation) |
| `--transformer-type` | `composed`, `additive`, `bivector-blend` | `composed` | Transformer architecture |
| `--n-components` | int | `64` | Number of basis vectors/planes |
| `--layers` | int | `3` | Transformer layers |
| `--build-canonical` | flag | yes | Build canonical basis from federated clusters |

**Output:** `models/bivector_paired.pt` - fast transformer that approximates federated rotations

### Step 3: Train Organizational Metric

```bash
python scripts/train_organizational_metric.py \
    --jsonl reports/pearltrees_targets_full_multi_account.jsonl \
    --model-dir models/pearltrees_federated_nomic \
    --transformer models/bivector_paired.pt \
    --account s243a \
    --epochs 50 \
    --num-pairs 100000 \
    --output models/organizational_metric.pt
```

**Key options:**
| Option | Values | Current | Description |
|--------|--------|---------|-------------|
| `--jsonl` | path | `reports/pearltrees_targets_full_multi_account.jsonl` | Hierarchy data |
| `--account` | string | `s243a` | Filter to items traceable to this root |
| `--num-pairs` | int | `100000` | Training pairs to generate |

**Output:** `models/organizational_metric.pt` - learned metric where distance = organizational proximity

### Step 4: Visualize in Density Explorer

```bash
# Start Flask API
python tools/density_explorer/flask_api.py --port 5000

# Open web UI
python -m http.server 8080 -d tools/density_explorer/web
```

**API options for `/api/compute`:**
| Option | Values | Description |
|--------|--------|-------------|
| `projection_mode` | `embedding`, `weights`, `learned` | 2D projection space |
| `tree_distance_metric` | `embedding`, `weights`, `learned` | Tree distance computation |
| `tree_type` | `mst`, `j-guided` | Tree algorithm |
| `model` | `bivector_paired` | Projection model |

**Projection modes:**
- `embedding` - Semantic similarity (output embeddings)
- `weights` - Transformation recipes (which clusters are blended)
- `learned` - Organizational structure (from trained metric)

---

## Problem Statement

The learned organizational metric performs well for items represented in Pearltrees training data, but struggles with edge cases like "David Lee (physicist)" appearing as a central node in physics concept trees. The model lacks training data to learn that:
- Physicist articles belong near a "Physicists" folder
- Most Wikipedia physicist articles don't have "(physicist)" in the title (only disambiguation cases do)

## Solution Overview

Use Wikipedia's category hierarchy as a bridge to connect Wikipedia articles to the Pearltrees organizational structure.

## Primary Approach: Wikipedia Categorylinks Bridge

### Data Source

Download from `dumps.wikimedia.org/enwiki/latest/`:
- `enwiki-latest-categorylinks.sql.gz` (~2.4 GB compressed)
- Contains: `page_id → category_name` mappings

### Algorithm

```
Wikipedia article: "David Lee (physicist)"
    ↓ categorylinks lookup
Categories: ["American physicists", "Cornell University faculty", "Nobel laureates"]
    ↓ navigate up Wikipedia category hierarchy
"American physicists" → "Physicists by nationality" → "Physicists"
    ↓ match against Pearltrees folders
Pearltrees folder: "Physicists" (exists at Society → people → Scientists → Physicists)
    ↓ compute organizational distance
Distance = (Wikipedia category hops) + (Pearltrees path distance)
```

### Implementation Steps

1. **Parse categorylinks**
   ```python
   def load_categorylinks(sql_path: str) -> Dict[str, List[str]]:
       """Parse SQL dump to get page_title → [category1, category2, ...]"""
       pass
   ```

2. **Build category hierarchy**
   - Categories are also pages in Wikipedia
   - Extract parent-child category relationships
   ```python
   def build_category_tree(categorylinks: Dict) -> Dict[str, str]:
       """Return category → parent_category mapping"""
       pass
   ```

3. **Create Pearltrees category matcher**
   ```python
   def find_connection_point(
       article_categories: List[str],
       category_tree: Dict[str, str],
       pearltrees_folders: Set[str]
   ) -> Tuple[str, int]:
       """
       Walk up category hierarchy until matching Pearltrees folder.
       Returns: (matching_folder, hops_to_reach)
       """
       for category in article_categories:
           current = category
           hops = 0
           while current:
               if current in pearltrees_folders:
                   return current, hops
               current = category_tree.get(current)
               hops += 1
       return None, float('inf')
   ```

4. **Store in SQLite database**
   - ~10-15 GB uncompressed fits SQLite's sweet spot (good up to ~100 GB)
   - Index on `page_title` and `category` for fast lookups
   - Single file, no server needed
   - Can cache hot data (category hierarchy) in memory

   ```python
   import sqlite3
   conn = sqlite3.connect('wikipedia_categories.db')
   conn.execute('''CREATE TABLE categorylinks
                   (page_id INT, page_title TEXT, category TEXT)''')
   conn.execute('CREATE INDEX idx_title ON categorylinks(page_title)')
   conn.execute('CREATE INDEX idx_category ON categorylinks(category)')
   ```

### Primary Usage: Incremental Training

The primary approach is **training** on Wikipedia categories, not inference-time fallback. Training happens incrementally as you explore data in the density explorer.

**Workflow:**
```
User loads 200 articles in density explorer
    ↓
Check training metadata: which are stale?
    ↓
Maybe 50 haven't been trained recently
    ↓
Look up their Wikipedia categories
    ↓
Generate training pairs via category bridge
    ↓
Quick training pass (~50 pairs, fast)
    ↓
Update iteration counters
    ↓
Model improves on data you're actively exploring
```

**Benefits:**
- Fast updates (only stale subset, not full retrain)
- Model learns what you care about (data you're exploring)
- Naturally balances coverage over time
- Could run in background while browsing

---

## Training Metadata Tracking

To avoid overtraining on any one dataset, track when each datapoint was last trained:

```python
training_metadata = {
    "datapoint_id": {
        "source": "wikipedia" | "pearltrees" | "synthetic",
        "last_trained_iter": 1542,
        "times_sampled": 3,
        "created_iter": 1200,
    }
}
```

### Staleness Check

```python
def needs_training(datapoint_id, current_iter, staleness_threshold=100):
    last = metadata.get(datapoint_id, {}).get("last_trained_iter", 0)
    return (current_iter - last) > staleness_threshold
```

### Balanced Sampling

```python
def sample_training_batch(metadata, current_iter):
    # Prioritize datapoints not seen recently
    staleness = current_iter - item["last_trained_iter"]

    # Weight by staleness and inverse of times_sampled
    weight = staleness / (1 + item["times_sampled"])

    # Also balance across sources
    source_weight = 1.0 / source_counts[item["source"]]

    return weighted_sample(weights=weight * source_weight)
```

### Update After Training

```python
for datapoint_id in trained_batch:
    metadata[datapoint_id]["last_trained_iter"] = current_iter
    metadata[datapoint_id]["times_sampled"] += 1
```

This ensures the model doesn't overfit to frequently-explored data while still learning from new explorations.

---

## Additional Enhancements

### 1. Training Data Refinements

**Weight by item type:**
- Trees (folders): weight 1.0 - intentional organizational structure
- Pearls in main sections: weight 0.8 - content filing
- Pearls in "see also" / "related": weight 0.3 - less certain

**Balanced sampling:**
- If 100 pearls per tree, pearls dominate even with lower weights
- Options: equal sampling, per-tree normalization, or 10-100x tree weights

```python
def compute_pair_weight(item_a, item_b, path_a, path_b):
    # Base weight from path length
    path_weight = 1.0 / (1.0 + path_length)

    # Section-based adjustment
    section_weight = 0.3 if 'see also' in path_a.lower() else 1.0

    # Item type adjustment
    type_weight = 1.0 if is_tree(item_a) else 0.8

    return path_weight * section_weight * type_weight
```

### 2. Synthetic Training Data from Wikipedia

**From category memberships:**
```python
# Article belongs to category → synthetic training pair
(newton_embedding, physicists_folder_embedding, distance=1)
```

**From Wikipedia lists:**
- Parse "List of physicists" → extract names
- Create synthetic hierarchy: Science → Physics → Physicists → [names]
- Generate training pairs with appropriate distances

**From Wikidata:**
- Structured data: `instance_of: human`, `occupation: physicist`
- More reliable than text pattern matching

### 3. Pattern-based Augmentation

For rare "(profession)" disambiguation titles:
```python
if "(physicist)" in title:
    # Generate synthetic pair to Physicists folder
    pairs.append((embedding, physicists_folder_emb, distance=1))
```

Limited utility since most articles don't use this pattern.

---

## Future Enhancement: Pagelinks Graph

**Not in initial scope** - `pagelinks.sql.gz` is ~6.8 GB and noisier.

**Potential future uses:**
- Build full knowledge graph of article relationships
- Find related articles by link proximity (PageRank-style)
- Detect topical clusters from link structure
- Cross-validate category-based distances

**Challenges:**
- Links include "See also", references, tangential mentions
- Less clean hierarchical signal than categories
- Much larger to process and store

---

## Implementation Priority

| Phase | Task | Effort |
|-------|------|--------|
| 1 | Download and parse categorylinks to SQLite | Medium |
| 1 | Build category hierarchy tree | Medium |
| 1 | Implement connection point finder | Low |
| 2 | Create training metadata tracker | Low |
| 2 | Implement incremental training from density explorer | Medium |
| 2 | Add training weight refinements | Low |
| 3 | Add staleness-based sampling | Low |
| 3 | Balance across data sources | Low |
| Future | Pagelinks graph integration | High |

## Files to Create/Modify

| File | Action |
|------|--------|
| `scripts/fetch_wikipedia_categories.py` | CREATE - Download and parse categorylinks to SQLite |
| `src/unifyweaver/data/wikipedia_categories.py` | CREATE - Category tree and lookup functions |
| `src/unifyweaver/training/metadata_tracker.py` | CREATE - Training iteration tracking |
| `scripts/train_organizational_metric.py` | MODIFY - Add incremental training, weight refinements |
| `tools/density_explorer/flask_api.py` | MODIFY - Trigger incremental training on data load

## Success Criteria

1. "David Lee (physicist)" correctly routes near Physicists folder, not as central physics node
2. Other edge cases (musicians, places, etc.) properly categorized via Wikipedia bridge
3. Minimal latency impact on tree building (category lookup should be fast)

## Data Requirements

- `enwiki-latest-categorylinks.sql.gz` (~2.4 GB download)
- SQLite database for parsed categories (~10-15 GB)
- Training metadata JSON/SQLite (~1 MB, grows with explored data)
- One-time parsing, then fast indexed lookups

## Storage Architecture

```
data/
├── wikipedia_categories.db    # SQLite: categorylinks + hierarchy
├── training_metadata.json     # Iteration tracking per datapoint
└── category_cache.pkl         # In-memory cache of hot categories
```
