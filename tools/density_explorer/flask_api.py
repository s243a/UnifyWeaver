#!/usr/bin/env python3
"""
Density Manifold Explorer - Flask API

REST API for density manifold computation with optional projection.

Usage:
    python tools/density_explorer/flask_api.py

Endpoints:
    POST /api/compute - Compute density manifold from embeddings
    POST /api/project - Project embeddings through orthogonal transformer
    GET /api/datasets - List available embedding datasets
    GET /api/models - List available projection models
    GET /api/health - Health check
"""

import sys
import hashlib
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools/density_explorer"))

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch

from shared import (
    load_embeddings,
    compute_density_manifold,
    load_and_compute,
    DensityManifoldData
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Default directories
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache" / "projections"

# Lazy-loaded projection models
_projection_models = {}


def _get_projection_model(model_path: str):
    """Load and cache projection model."""
    if model_path not in _projection_models:
        from scripts.train_orthogonal_codebook import load_orthogonal_transformer
        transformer, codebook = load_orthogonal_transformer(model_path)
        transformer.eval_mode()
        _projection_models[model_path] = (transformer, codebook)
    return _projection_models[model_path]


def _get_cache_path(dataset_path: str, model_path: str) -> Path:
    """Generate cache path for projected embeddings."""
    # Hash the paths for a unique cache key
    key = f"{dataset_path}:{model_path}"
    hash_key = hashlib.md5(key.encode()).hexdigest()[:12]
    dataset_name = Path(dataset_path).stem
    model_name = Path(model_path).stem
    return CACHE_DIR / f"{dataset_name}_{model_name}_{hash_key}.npz"


def _project_embeddings(embeddings: np.ndarray, model_path: str, batch_size: int = 256) -> np.ndarray:
    """Project embeddings through the orthogonal transformer."""
    transformer, _ = _get_projection_model(model_path)

    n_samples = len(embeddings)
    projected = np.zeros_like(embeddings)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = embeddings[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch.astype(np.float32)).to(transformer.device)
            proj_batch, _, _, _ = transformer.forward(batch_tensor)
            projected[i:i+batch_size] = proj_batch.cpu().numpy()

    return projected


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'density-manifold-api'})


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List available embedding datasets."""
    datasets = []

    if DATASETS_DIR.exists():
        for f in DATASETS_DIR.glob('*.npz'):
            try:
                data = np.load(f, allow_pickle=True)
                n_points = len(data['embeddings'])
                has_titles = 'titles' in data
                datasets.append({
                    'name': f.stem,
                    'path': str(f),
                    'n_points': n_points,
                    'has_titles': has_titles
                })
            except Exception as e:
                datasets.append({
                    'name': f.stem,
                    'path': str(f),
                    'error': str(e)
                })

    return jsonify({'datasets': datasets})


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available projection models."""
    models = []

    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob('*.pt'):
            try:
                # Quick check - just load metadata
                data = torch.load(f, map_location='cpu', weights_only=False)
                metadata = data.get('metadata', {})
                models.append({
                    'name': f.stem,
                    'path': str(f),
                    'n_planes': data.get('embed_dim', 768) // 12,  # Rough estimate
                    'metadata': metadata
                })
            except Exception as e:
                models.append({
                    'name': f.stem,
                    'path': str(f),
                    'error': str(e)
                })

    return jsonify({'models': models})


@app.route('/api/project', methods=['POST'])
def project_embeddings():
    """
    Project embeddings through orthogonal transformer.

    Request JSON:
    {
        "dataset": "wikipedia_physics" or path to .npz file,
        "model": "orthogonal_transformer_rotational" or path to .pt file,
        "use_cache": true,      // optional, cache results (default: true)
        "top_k": 1000           // optional, limit number of points
    }

    Response:
    {
        "n_points": 1000,
        "from_cache": false,
        "cache_path": "cache/projections/...",
        "projection_time_ms": 150
    }
    """
    import time

    try:
        data = request.get_json()

        # Resolve dataset path
        dataset = data.get('dataset')
        if not dataset:
            return jsonify({'error': 'dataset required'}), 400

        if not dataset.endswith('.npz'):
            dataset_path = DATASETS_DIR / f"{dataset}.npz"
        else:
            dataset_path = Path(dataset)

        if not dataset_path.exists():
            return jsonify({'error': f'Dataset not found: {dataset_path}'}), 404

        # Resolve model path
        model = data.get('model', 'orthogonal_transformer_rotational')
        if not model.endswith('.pt'):
            model_path = MODELS_DIR / f"{model}.pt"
        else:
            model_path = Path(model)

        if not model_path.exists():
            return jsonify({'error': f'Model not found: {model_path}'}), 404

        use_cache = data.get('use_cache', True)
        top_k = data.get('top_k')

        # Check cache
        cache_path = _get_cache_path(str(dataset_path), str(model_path))

        if use_cache and cache_path.exists():
            return jsonify({
                'n_points': top_k or 'all',
                'from_cache': True,
                'cache_path': str(cache_path),
                'projection_time_ms': 0
            })

        # Load embeddings
        embeddings, titles = load_embeddings(str(dataset_path))
        if top_k:
            embeddings = embeddings[:top_k]
            titles = titles[:top_k] if titles is not None else None

        # Project
        start = time.time()
        projected = _project_embeddings(embeddings, str(model_path))
        projection_time = (time.time() - start) * 1000

        # Cache results
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                embeddings=projected,
                titles=titles if titles is not None else [],
                source_dataset=str(dataset_path),
                model=str(model_path)
            )

        return jsonify({
            'n_points': len(projected),
            'from_cache': False,
            'cache_path': str(cache_path) if use_cache else None,
            'projection_time_ms': round(projection_time, 1)
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/compute', methods=['POST'])
def compute_manifold():
    """
    Compute density manifold from embeddings.

    Request JSON:
    {
        "dataset": "wikipedia_physics" or path to .npz file,
        "model": null,          // optional, projection model (.pt file)
        "top_k": 300,           // optional, limit number of points
        "bandwidth": 0.1,       // optional, KDE bandwidth
        "grid_size": 100,       // optional, density grid resolution
        "include_tree": true,   // optional, compute tree overlay
        "tree_type": "mst",     // optional, "mst" or "j-guided"
        "include_peaks": true,  // optional, find density peaks
        "n_peaks": 5            // optional, number of peaks
    }

    Response:
        Full DensityManifoldData as JSON
    """
    try:
        data = request.get_json()

        # Resolve dataset path
        dataset = data.get('dataset', 'wikipedia_physics')
        if not dataset.endswith('.npz'):
            dataset_path = DATASETS_DIR / f"{dataset}.npz"
        else:
            dataset_path = Path(dataset)

        if not dataset_path.exists():
            return jsonify({'error': f'Dataset not found: {dataset_path}'}), 404

        # Check for projection model
        model = data.get('model')
        use_projected = False

        if model:
            if not model.endswith('.pt'):
                model_path = MODELS_DIR / f"{model}.pt"
            else:
                model_path = Path(model)

            if model_path.exists():
                # Check for cached projection
                cache_path = _get_cache_path(str(dataset_path), str(model_path))
                if cache_path.exists():
                    # Use cached projected embeddings
                    dataset_path = cache_path
                    use_projected = True
                else:
                    # Project on-the-fly and use projected embeddings
                    embeddings, titles = load_embeddings(str(dataset_path))
                    top_k = data.get('top_k')
                    if top_k:
                        embeddings = embeddings[:top_k]
                        titles = titles[:top_k] if titles is not None else None

                    projected = _project_embeddings(embeddings, str(model_path))

                    # Cache for future use
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        cache_path,
                        embeddings=projected,
                        titles=titles if titles is not None else [],
                        source_dataset=str(dataset_path),
                        model=str(model_path)
                    )
                    dataset_path = cache_path
                    use_projected = True

        # Extract parameters
        top_k = data.get('top_k')
        bandwidth = data.get('bandwidth')
        grid_size = data.get('grid_size', 100)
        include_tree = data.get('include_tree', True)
        tree_type = data.get('tree_type', 'mst')
        include_peaks = data.get('include_peaks', True)
        n_peaks = data.get('n_peaks', 5)

        # Compute manifold
        result = load_and_compute(
            str(dataset_path),
            top_k=top_k,
            bandwidth=bandwidth,
            grid_size=grid_size,
            include_tree=include_tree,
            tree_type=tree_type,
            include_peaks=include_peaks,
            n_peaks=n_peaks
        )

        # Return as JSON
        return result.to_json(), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compute/from-embeddings', methods=['POST'])
def compute_from_embeddings():
    """
    Compute density manifold from raw embeddings (not file).

    Request JSON:
    {
        "embeddings": [[0.1, 0.2, ...], ...],  // N x D array
        "titles": ["title1", "title2", ...],   // optional
        "bandwidth": 0.1,
        "grid_size": 100,
        "include_tree": true,
        "tree_type": "mst",
        "include_peaks": true,
        "n_peaks": 5
    }
    """
    try:
        data = request.get_json()

        embeddings = np.array(data['embeddings'])
        titles = data.get('titles')

        result = compute_density_manifold(
            embeddings=embeddings,
            titles=titles,
            bandwidth=data.get('bandwidth'),
            grid_size=data.get('grid_size', 100),
            include_tree=data.get('include_tree', True),
            tree_type=data.get('tree_type', 'mst'),
            include_peaks=data.get('include_peaks', True),
            n_peaks=data.get('n_peaks', 5)
        )

        return result.to_json(), 200, {'Content-Type': 'application/json'}

    except KeyError as e:
        return jsonify({'error': f'Missing required field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tree/rebuild', methods=['POST'])
def rebuild_tree():
    """
    Rebuild just the tree overlay with different settings.

    Useful for quickly switching tree types without recomputing density.

    Request JSON:
    {
        "dataset": "wikipedia_physics",
        "top_k": 300,
        "tree_type": "j-guided"
    }
    """
    try:
        data = request.get_json()

        dataset = data.get('dataset', 'wikipedia_physics')
        if not dataset.endswith('.npz'):
            dataset_path = DATASETS_DIR / f"{dataset}.npz"
        else:
            dataset_path = Path(dataset)

        if not dataset_path.exists():
            return jsonify({'error': f'Dataset not found: {dataset_path}'}), 404

        # Load embeddings
        embeddings, titles = load_embeddings(str(dataset_path))

        top_k = data.get('top_k')
        if top_k:
            embeddings = embeddings[:top_k]
            titles = titles[:top_k] if titles else None

        tree_type = data.get('tree_type', 'mst')

        # Import tree builders
        from shared.density_core import build_mst_tree, build_j_guided_tree, project_to_2d

        # Project to 2D
        points_2d, _, _ = project_to_2d(embeddings)

        # Build tree
        if tree_type == 'mst':
            tree = build_mst_tree(embeddings, points_2d, titles)
        elif tree_type == 'j-guided':
            tree = build_j_guided_tree(embeddings, points_2d, titles)
        else:
            return jsonify({'error': f'Unknown tree type: {tree_type}'}), 400

        # Return tree structure
        from dataclasses import asdict
        return jsonify(asdict(tree)), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Density Manifold API')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print(f"Starting Density Manifold API on http://{args.host}:{args.port}")
    print("Endpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/datasets - List available datasets")
    print("  GET  /api/models - List available projection models")
    print("  POST /api/project - Project embeddings through model (cached)")
    print("  POST /api/compute - Compute density manifold (with optional projection)")
    print("  POST /api/compute/from-embeddings - Compute from raw embeddings")
    print("  POST /api/tree/rebuild - Rebuild tree with different type")

    app.run(host=args.host, port=args.port, debug=args.debug)
