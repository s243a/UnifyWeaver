# Skill: Density Explorer

Interactive visualization tool for exploring embedding spaces, tree structures, and density manifolds.

## When to Use

- User asks "how do I visualize my embeddings?"
- User wants to explore clustering quality
- User asks about the density explorer tool
- User needs to debug semantic search results
- User wants to export tree structures
- User asks about MST vs J-guided visualization

## Quick Start

### Build the Vue Frontend

```bash
cd tools/density_explorer/vue
npm install
npm run build
```

### Start the Flask API

```bash
# From project root
python tools/density_explorer/flask_api.py --port 5000
```

### Start the Vue Dev Server

```bash
cd tools/density_explorer/vue
npm run dev
```

Then open http://localhost:5173 (Vue dev server proxies to Flask API).

### Alternative: Streamlit App

For quick exploration without building the Vue frontend:

```bash
streamlit run tools/density_explorer/streamlit_app.py
```

## Loading Data

### Prepare Embeddings

The explorer expects `.npz` files with embeddings:

```python
import numpy as np

np.savez("datasets/my_data.npz",
         embeddings=embeddings_array,  # (N, D) float array
         titles=titles_list,           # List of N strings
         texts=texts_list)             # Optional: original texts
```

### Using Nomic Embeddings (Recommended)

Nomic embeddings produce better clustering and separation in the 2D projection:

```bash
python tools/density_explorer/scripts/generate_nomic_embeddings.py
```

Or generate your own:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

# Use document prefix for corpus items
texts_prefixed = [f"search_document: {t}" for t in texts]
embeddings = model.encode(texts_prefixed, show_progress_bar=True)

np.savez("datasets/my_nomic_data.npz",
         embeddings=embeddings,
         titles=titles,
         texts=texts)
```

**Why nomic?** Nomic's 768-dimensional embeddings with asymmetric prefixes produce more distinct clusters in the 2D SVD projection, making the visualization clearer and the tree structure more meaningful.

## UI Controls

### Tree Overlay

| Control | Description |
|---------|-------------|
| **Show Tree** | Toggle tree edge visibility |
| **Tree Type** | MST (Minimum Spanning Tree) or J-Guided |
| **Max Depth** | Filter tree to show only nodes up to this depth |

### Switching Tree Type

- **MST**: Traditional minimum spanning tree based on cosine distance
- **J-Guided**: Tree optimized using J = D/(1+H) objective function

```
Tree Type dropdown → Select "J-Guided" → Click "Compute Manifold"
```

### Density Controls

| Control | Description |
|---------|-------------|
| **Bandwidth** | KDE smoothing (lower = sharper peaks, higher = smoother) |
| **Grid Size** | Resolution of density heatmap (50-200) |
| **Show Contours** | Toggle contour lines on density field |

### Display Options

| Control | Description |
|---------|-------------|
| **Show Points** | Toggle individual point markers |
| **Show Peaks** | Toggle density peak markers (yellow stars) |
| **Number of Peaks** | How many local maxima to highlight |

## Node Selection

Click any point in the visualization to select it:

- Selected node shows tooltip with title and coordinates
- Node depth is displayed if tree overlay is active
- Click emits event for downstream processing

### Re-rooting the Tree

To re-root the tree on a selected node:

1. Click to select a node
2. Use the re-root action (rebuilds tree with selected node as root)
3. Tree structure updates to show hierarchy from new perspective

**Why re-root?** Beyond just changing perspective, re-rooting creates a central focal point for studying related concepts. When you re-root on a topic of interest, all semantically related items radiate outward, making it easier to explore connections and build understanding around that topic.

## Tree Construction Modes

### MST (Minimum Spanning Tree)

Default mode. Connects all nodes with minimum total edge weight:

```
API: POST /api/compute
Body: { "tree_type": "mst", ... }
```

Properties:
- Fast computation
- Globally optimal for total distance
- May produce long chains

### J-Guided Tree

Optimizes J = D/(1+H) at each attachment decision:

```
API: POST /api/compute
Body: { "tree_type": "j-guided", ... }
```

Properties:
- Balances distance (D) and entropy (H)
- Produces more balanced hierarchies
- Better for semantic categorization

### Entropy Source (J-Guided Mode)

When using J-guided trees, you can switch entropy computation:

| Source | Speed | Description |
|--------|-------|-------------|
| **Fisher** | Fast | Geometric proxy using between/within cluster variance |
| **BERT** | Slow | True Shannon entropy from transformer logits |

To use BERT entropy:

```python
# In API call
{
    "tree_type": "j-guided",
    "entropy_source": "bert",
    "entropy_model": "bert-base-uncased"
}
```

For short text, Fisher and BERT produce similar results. BERT may be more accurate for longer documents.

## Exports

### Basic JSON Export

The Streamlit app provides direct download:

```
Export Data → Download JSON
```

This exports the full `DensityManifoldData` structure.

### Depth-Filtered Export

Exports respect the current **Max Depth** setting. Set depth before exporting to get only the portion of the tree you want:

1. Set **Max Depth** to desired level (e.g., 3 for top 3 levels)
2. Click **Download JSON**
3. Export contains only nodes/edges up to that depth

### Enriched Exports

Use `link_pearltrees.py` to enrich exports with Pearltrees links and metadata:

```bash
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap output/physics_export.smmx \
  --trees reports/pearltrees_targets_s243a.jsonl \
  --url-db .local/data/children_index.db \
  --embeddings datasets/pearltrees_embeddings.npz \
  --projection-model models/federated_model.pkl \
  --output output/physics_enriched.smmx \
  --threshold 0.7 \
  --report enrichment_report.json
```

The tool enriches via:
1. **URL matching** - Matches node URLs to PagePearl external URLs
2. **Title matching** - Exact normalized title match
3. **Semantic matching** - Embedding similarity with projection model

When multiple matches exist for a URL, it uses the projection model to select the best hierarchical fit.

### Programmatic Export via API

```python
import requests

response = requests.post("http://localhost:5000/api/compute", json={
    "dataset": "wikipedia_physics",
    "top_k": 200,
    "tree_type": "j-guided",
    "include_tree": True,
    "include_peaks": True
})

data = response.json()

# Filter to specific depth
max_depth = 3
filtered_nodes = [n for n in data['tree']['nodes'] if n['depth'] <= max_depth]
filtered_edges = [
    e for e in data['tree']['edges']
    if any(n['id'] == e['target_id'] and n['depth'] <= max_depth for n in data['tree']['nodes'])
]
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/datasets` | GET | List available .npz files in datasets/ |
| `/api/compute` | POST | Compute full density manifold |
| `/api/compute/from-embeddings` | POST | Compute from raw embedding array |
| `/api/tree/rebuild` | POST | Rebuild just tree (faster, no density recompute) |

### Example: Rebuild Tree Only

Quickly switch tree type without recomputing density:

```python
response = requests.post("http://localhost:5000/api/tree/rebuild", json={
    "dataset": "wikipedia_physics",
    "top_k": 200,
    "tree_type": "j-guided"
})

new_tree = response.json()
```

## Interpreting the Visualization

### Density Heatmap

- **Yellow/bright areas**: High density (many similar embeddings)
- **Purple/dark areas**: Low density (sparse regions)
- **Peaks (stars)**: Local maxima - potential cluster centers

### Tree Overlay

- **Cyan lines**: Tree edges connecting parent → child
- **Cyan square**: Root node
- **Edge length**: Proportional to semantic distance

### Good Signs

- Distinct colored regions with clear peaks
- Tree edges connecting semantically similar items
- Balanced tree depth (not all chains)

### Warning Signs

- Uniform density (embeddings not well-separated)
- Very long tree chains (poor hierarchy)
- Disconnected clusters (may need different embedding model)

## Troubleshooting

### "Dataset not found"

Ensure `.npz` file is in `datasets/` directory:

```bash
ls datasets/*.npz
```

### Clustering looks poor

Try nomic embeddings instead of MiniLM:

```bash
python tools/density_explorer/scripts/generate_nomic_embeddings.py
```

### Vue build fails

```bash
cd tools/density_explorer/vue
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Flask API won't start

Check port availability:

```bash
lsof -i :5000
# Kill existing process if needed
```

## Related

**Parent Skill:**
- `skill_mindmap_bookmark_tools.md` - Master for mindmaps and bookmarks

**Sibling Skills:**
- `skill_mindmap_tools.md` - SimpleMind file management
- `skill_bookmark_tools.md` - Bookmark filing and organization

**ML Foundation:**
- `skill_ml_tools.md` - ML tools (embeddings, training, inference)
- `skill_embedding_models.md` - Choosing embedding models
- `skill_hierarchy_objective.md` - J = D/(1+H) theory

**Other Skills:**
- `skill_mst_folder_grouping.md` - MST for folder organization

**Documentation:**
- `docs/design/MST_IMPROVEMENTS_PROPOSAL.md` - J-guided improvements

**Code:**
- `tools/density_explorer/flask_api.py` - REST API
- `tools/density_explorer/streamlit_app.py` - Streamlit alternative
- `tools/density_explorer/vue/` - Vue.js frontend
- `tools/density_explorer/shared/density_core.py` - Core computation
- `scripts/mindmap/link_pearltrees.py` - Export enrichment tool
