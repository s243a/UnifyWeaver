# Density Explorer - Data Files

This folder contains large generated data files that are not tracked in git.
Use the scripts below to regenerate them.

## Required Files

| File | Size | Description |
|------|------|-------------|
| `physics.json` | ~1.5 MB | Pre-computed density manifold (points, grid, tree, peaks) |
| `physics_nomic_embeddings.json` | ~1.3 MB | 200 Wikipedia physics articles, 768D Nomic embeddings |
| `projection.onnx` | ~61 MB | Bivector projection model (ONNX export) |
| `orthogonal_projection.onnx` | ~39 MB | Orthogonal projection model (ONNX export) |

## Generation Scripts

### `physics.json`

```bash
cd tools/density_explorer
python scripts/generate_physics_json.py
```

Source: [`scripts/generate_physics_json.py`](../../scripts/generate_physics_json.py)
— Loads embeddings from `datasets/wikipedia_physics.npz`, computes the density
manifold (SVD projection, density grid, MST/J-guided trees, peaks), and writes
the result to `web/data/physics.json`.

Requires: `numpy`, `scipy`

### `physics_nomic_embeddings.json`

```bash
cd tools/density_explorer
python scripts/generate_nomic_embeddings.py
```

Source: [`scripts/generate_nomic_embeddings.py`](../../scripts/generate_nomic_embeddings.py)
— Loads the first 200 articles from `datasets/wikipedia_physics.npz`, computes
768D Nomic embeddings via `sentence-transformers`, and saves as JSON.

Requires: `numpy`, `sentence-transformers`, `torch`

### ONNX Models

The ONNX model files are exported from trained PyTorch models. See the training
scripts in the project root for details.
