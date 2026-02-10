# Physics Mindmap Builder

Interactive mindmap builder for Wikipedia physics articles. Uses embedding-based distances to build hierarchical trees with density-guided algorithms.

## Features

- **Two tree algorithms**: J-guided (density-ordered greedy insertion) and MST (minimum spanning tree)
- **Re-rootable**: Click any node to make it the new root
- **Hierarchy-aware**: Breadth penalty suppresses parent categories when viewing subtopics
- **Multiple export formats**: FreeMind, SimpleMind, OPML, VUE, Mermaid, JSON
- **2D density visualization** with Plotly

## Quick Start

### 1. Generate data

```bash
pip install datasets sentence-transformers numpy scipy
python generate_data.py
```

This downloads ~100K Wikipedia articles from HuggingFace, filters ~300 physics articles, embeds them with [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), and computes the distance matrix, MST, density grid, and hierarchy scores.

### 2. Serve

```bash
python -m http.server 8080
```

Open http://localhost:8080

## Options

```
python generate_data.py --top-k 200      # fewer articles (faster)
python generate_data.py --max-rows 50000  # download fewer Wikipedia rows
python generate_data.py --output data/physics_bundle.json  # custom output path
```

## How It Works

The app pre-computes a pairwise distance matrix from article embeddings, then builds trees on-the-fly in the browser:

- **J-guided tree**: Places nodes in density order (densest first). Each node attaches to its nearest already-placed neighbor. Dense hubs become natural tree centers.
- **MST tree**: Uses a pre-computed minimum spanning tree, re-rooted at the selected node.
- **Breadth penalty**: Nodes broader than the root (measured by k-NN density) are demoted in processing order and penalized in distance, preventing parent categories from dominating subtopic trees.

## Distance Metrics

The default metric is cosine distance. The Settings tab also offers:
- **Chord**: L2 distance on the unit sphere. Compresses large angles.
- **Angular**: Arc length on the unit sphere. Linear with angle.

These are recomputed from embeddings in the browser when you switch metrics.
