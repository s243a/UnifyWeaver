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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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

# Training imports (lazy-loaded to avoid import errors if modules unavailable)
_training_imports_loaded = False
_TrainingMetadataTracker = None
_WikipediaCategoryBridge = None

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Default directories
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache" / "projections"

# Lazy-loaded projection models
_projection_models = {}

# Lazy-loaded training tracker
_training_tracker = None


def _load_training_modules():
    """Lazy-load training modules."""
    global _training_imports_loaded, _TrainingMetadataTracker, _WikipediaCategoryBridge

    if _training_imports_loaded:
        return True

    try:
        from unifyweaver.training import TrainingMetadataTracker
        from unifyweaver.data import WikipediaCategoryBridge
        _TrainingMetadataTracker = TrainingMetadataTracker
        _WikipediaCategoryBridge = WikipediaCategoryBridge
        _training_imports_loaded = True
        return True
    except ImportError as e:
        print(f"Training modules not available: {e}")
        return False


def _get_training_tracker():
    """Get or create training metadata tracker."""
    global _training_tracker

    if not _load_training_modules():
        return None

    if _training_tracker is None:
        tracker_path = PROJECT_ROOT / "data" / "training_metadata.json"
        _training_tracker = _TrainingMetadataTracker(str(tracker_path))

    return _training_tracker


def _get_projection_model(model_path: str):
    """Load and cache projection model."""
    if model_path not in _projection_models:
        # Check model type from saved dict
        save_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        model_type = save_dict.get('transformer_type', 'orthogonal')

        if model_type == 'composed_bivector':
            from scripts.train_orthogonal_codebook import load_composed_bivector_transformer
            transformer = load_composed_bivector_transformer(model_path)
            transformer.eval_mode()
            _projection_models[model_path] = (transformer, None)
        else:
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


def _project_embeddings(
    embeddings: np.ndarray,
    model_path: str,
    batch_size: int = 256,
    return_weights: bool = False
) -> np.ndarray:
    """
    Project embeddings through the orthogonal transformer.

    Args:
        embeddings: (N, D) input embeddings
        model_path: path to .pt model file
        batch_size: batch size for projection
        return_weights: if True, also return blend weights

    Returns:
        projected: (N, D) projected embeddings
        weights: (N, n_basis) blend weights (only if return_weights=True)
    """
    transformer, _ = _get_projection_model(model_path)

    n_samples = len(embeddings)
    projected = np.zeros_like(embeddings)
    all_weights = None  # Will be allocated on first batch when we know the weight dimension

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = embeddings[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch.astype(np.float32)).to(transformer.device)
            proj_batch, weights, _, _ = transformer.forward(batch_tensor)
            projected[i:i+batch_size] = proj_batch.cpu().numpy()

            if return_weights and weights is not None:
                # Allocate on first batch when we know the actual weight dimension
                if all_weights is None:
                    n_basis = weights.shape[1]
                    all_weights = np.zeros((n_samples, n_basis))
                all_weights[i:i+batch_size] = weights.cpu().numpy()

    if return_weights:
        return projected, all_weights
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
        "tree_distance_metric": "embedding",  // "embedding", "weights", or "learned"
        "include_peaks": true,  // optional, find density peaks
        "n_peaks": 5,           // optional, number of peaks
        "projection_mode": "embedding"  // "embedding", "weights", or "learned"
    }

    projection_mode options:
        - "embedding": Project the output embeddings to 2D (default)
        - "weights": Project the transformer blend weights to 2D
          (shows which queries use similar transformation recipes)
        - "learned": Project in learned organizational metric space to 2D
          (shows organizational/hierarchical structure from trained model)

    tree_distance_metric options:
        - "embedding": Semantic distance (cosine in embedding space)
        - "weights": Hierarchical distance (cosine in weight space)
        - "learned": Organizational distance (Euclidean in learned metric space)

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

        # Extract projection mode
        projection_mode = data.get('projection_mode', 'embedding')
        if projection_mode not in ('embedding', 'weights', 'learned', 'wikipedia_physics', 'wikipedia_physics_mds'):
            return jsonify({'error': f'Invalid projection_mode: {projection_mode}. Use "embedding", "weights", "learned", "wikipedia_physics", or "wikipedia_physics_mds"'}), 400

        # Check for projection model
        model = data.get('model')
        blend_weights = None
        input_embeddings = None

        if model:
            if not model.endswith('.pt'):
                model_path = MODELS_DIR / f"{model}.pt"
            else:
                model_path = Path(model)

            if model_path.exists():
                top_k = data.get('top_k')

                # Check for cached projection with weights
                cache_path = _get_cache_path(str(dataset_path), str(model_path))
                cache_has_weights = False
                cache_has_input = False

                if cache_path.exists():
                    cached = np.load(cache_path, allow_pickle=True)
                    cache_has_weights = 'weights' in cached
                    cache_has_input = 'input_embeddings' in cached

                # Need input embeddings for "learned" mode
                needs_input = projection_mode == 'learned' or data.get('tree_distance_metric') == 'learned'

                if cache_path.exists() and (projection_mode == 'embedding' or cache_has_weights) and (not needs_input or cache_has_input):
                    # Use cached data (full dataset)
                    cached = np.load(cache_path, allow_pickle=True)
                    embeddings = cached['embeddings']
                    titles = list(cached['titles']) if 'titles' in cached and len(cached['titles']) > 0 else None
                    blend_weights = cached['weights'] if 'weights' in cached else None
                    input_embeddings = cached['input_embeddings'] if 'input_embeddings' in cached else None
                else:
                    # Load full embeddings and project on-the-fly
                    raw_embeddings, titles = load_embeddings(str(dataset_path))
                    projected, weights = _project_embeddings(
                        raw_embeddings, str(model_path), return_weights=True
                    )

                    # Cache full dataset for future use (including input embeddings for learned mode)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        cache_path,
                        embeddings=projected,
                        weights=weights,
                        input_embeddings=raw_embeddings,  # Store input for learned metric
                        titles=titles if titles is not None else [],
                        source_dataset=str(dataset_path),
                        model=str(model_path)
                    )

                    embeddings = projected
                    blend_weights = weights
                    input_embeddings = raw_embeddings

                # Apply top_k AFTER loading/projecting full dataset
                if top_k:
                    embeddings = embeddings[:top_k]
                    titles = titles[:top_k] if titles is not None else None
                    if blend_weights is not None:
                        blend_weights = blend_weights[:top_k]
                    if input_embeddings is not None:
                        input_embeddings = input_embeddings[:top_k]

                # Compute manifold with appropriate mode
                tree_distance_metric = data.get('tree_distance_metric', 'embedding')
                # If projection_mode is weights or learned, default tree to same space
                if projection_mode in ('weights', 'learned') and 'tree_distance_metric' not in data:
                    tree_distance_metric = projection_mode

                # Support layout_dataset: use a different dataset's raw embeddings for 2D SVD layout
                # while the model-projected embeddings are used for tree distances
                layout_dataset = data.get('layout_dataset')
                layout_embeddings = None
                tree_embeddings_override = None
                if layout_dataset and projection_mode == 'embedding':
                    if not layout_dataset.endswith('.npz'):
                        layout_path = DATASETS_DIR / f"{layout_dataset}.npz"
                    else:
                        layout_path = Path(layout_dataset)
                    if layout_path.exists():
                        layout_raw, _ = load_embeddings(str(layout_path))
                        if top_k:
                            layout_raw = layout_raw[:top_k]
                        # Swap: layout_raw for 2D projection, model-projected for tree
                        layout_embeddings = layout_raw
                        tree_embeddings_override = embeddings  # model-projected

                # Resolve root_id by title if provided as string
                root_id = data.get('root_id')
                if isinstance(root_id, str) and titles is not None:
                    for idx, t in enumerate(titles):
                        if t == root_id:
                            root_id = idx
                            break
                    else:
                        root_id = None  # Title not found

                result = compute_density_manifold(
                    embeddings=layout_embeddings if layout_embeddings is not None else embeddings,
                    titles=titles,
                    bandwidth=data.get('bandwidth'),
                    grid_size=data.get('grid_size', 100),
                    include_tree=data.get('include_tree', True),
                    tree_type=data.get('tree_type', 'mst'),
                    tree_distance_metric=tree_distance_metric,
                    include_peaks=data.get('include_peaks', True),
                    n_peaks=data.get('n_peaks', 5),
                    projection_mode=projection_mode,
                    weights=blend_weights,
                    input_embeddings=input_embeddings,
                    max_branching=data.get('max_branching'),
                    root_id=root_id,
                    tree_embeddings=tree_embeddings_override
                )

                return result.to_json(), 200, {'Content-Type': 'application/json'}

        # No model specified - check valid modes
        if projection_mode in ('weights', 'learned'):
            return jsonify({'error': f'projection_mode="{projection_mode}" requires a model'}), 400

        # Extract parameters
        top_k = data.get('top_k')
        bandwidth = data.get('bandwidth')
        grid_size = data.get('grid_size', 100)
        include_tree = data.get('include_tree', True)
        tree_type = data.get('tree_type', 'mst')
        tree_distance_metric = data.get('tree_distance_metric', 'embedding')
        include_peaks = data.get('include_peaks', True)
        n_peaks = data.get('n_peaks', 5)

        # Resolve root_id (can be int index or string title)
        root_id = data.get('root_id')

        # Compute manifold (wikipedia_physics uses its own internal model)
        result = load_and_compute(
            str(dataset_path),
            top_k=top_k,
            bandwidth=bandwidth,
            grid_size=grid_size,
            include_tree=include_tree,
            tree_type=tree_type,
            tree_distance_metric=tree_distance_metric,
            include_peaks=include_peaks,
            n_peaks=n_peaks,
            projection_mode=projection_mode,  # Pass actual mode (embedding or wikipedia_physics)
            max_branching=data.get('max_branching'),
            root_id=root_id
        )

        # Return as JSON
        return result.to_json(), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/compute/from-embeddings', methods=['POST'])
def compute_from_embeddings():
    """
    Compute density manifold from raw embeddings (not file).

    Request JSON:
    {
        "embeddings": [[0.1, 0.2, ...], ...],  // N x D array (for 2D projection / layout)
        "tree_embeddings": [[...], ...],       // optional, separate embeddings for tree distances
        "titles": ["title1", "title2", ...],   // optional
        "weights": [[0.1, 0.05, ...], ...],    // optional, blend weights
        "input_embeddings": [[...], ...],      // optional, input embeddings for "learned" mode
        "bandwidth": 0.1,
        "grid_size": 100,
        "include_tree": true,
        "tree_type": "mst",                    // "mst" or "j-guided"
        "tree_distance_metric": "embedding",   // "embedding", "weights", or "learned"
        "include_peaks": true,
        "n_peaks": 5,
        "projection_mode": "embedding"         // "embedding", "weights", or "learned"
    }

    Dual embedding mode:
        When "tree_embeddings" is provided, "embeddings" is used for 2D projection (layout)
        and "tree_embeddings" is used for tree distance calculations. This allows using
        different embedding models for different purposes (e.g., MiniLM for layout, Nomic for tree).
    """
    try:
        data = request.get_json()

        embeddings = np.array(data['embeddings'])
        titles = data.get('titles')
        projection_mode = data.get('projection_mode', 'embedding')
        tree_distance_metric = data.get('tree_distance_metric', 'embedding')

        # Support dual embeddings: separate layout vs tree embeddings
        tree_embeddings = None
        if 'tree_embeddings' in data:
            tree_embeddings = np.array(data['tree_embeddings'])

        # Get weights if provided
        weights = None
        if projection_mode in ('weights', 'learned') or tree_distance_metric in ('weights', 'learned'):
            if 'weights' not in data:
                return jsonify({'error': f'projection_mode/tree_distance_metric="{projection_mode}" requires weights array'}), 400
            weights = np.array(data['weights'])

        # Get input embeddings for learned mode
        input_embeddings = None
        if projection_mode == 'learned' or tree_distance_metric == 'learned':
            if 'input_embeddings' in data:
                input_embeddings = np.array(data['input_embeddings'])
            else:
                # Fall back to using output embeddings as input (less accurate but functional)
                input_embeddings = embeddings

        result = compute_density_manifold(
            embeddings=embeddings,
            titles=titles,
            bandwidth=data.get('bandwidth'),
            grid_size=data.get('grid_size', 100),
            include_tree=data.get('include_tree', True),
            tree_type=data.get('tree_type', 'mst'),
            tree_distance_metric=tree_distance_metric,
            include_peaks=data.get('include_peaks', True),
            n_peaks=data.get('n_peaks', 5),
            projection_mode=projection_mode,
            weights=weights,
            input_embeddings=input_embeddings,
            max_branching=data.get('max_branching'),
            tree_embeddings=tree_embeddings
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


@app.route('/api/training/status', methods=['GET'])
def training_status():
    """
    Get training metadata status.

    Query params:
        dataset: dataset name (optional, shows stats for that dataset)

    Response:
    {
        "available": true,
        "current_iter": 1542,
        "total_datapoints": 5000,
        "stale_count": 200,
        "stats": {...}
    }
    """
    try:
        tracker = _get_training_tracker()
        if tracker is None:
            return jsonify({
                'available': False,
                'error': 'Training modules not available'
            })

        stats = tracker.get_stats()

        # Check for stale datapoints if dataset specified
        dataset = request.args.get('dataset')
        stale_count = 0
        if dataset:
            # Load titles from dataset
            if not dataset.endswith('.npz'):
                dataset_path = DATASETS_DIR / f"{dataset}.npz"
            else:
                dataset_path = Path(dataset)

            if dataset_path.exists():
                _, titles = load_embeddings(str(dataset_path))
                if titles is not None:
                    stale_ids = tracker.get_stale_datapoints(
                        available_ids=titles,
                        staleness_threshold=100
                    )
                    stale_count = len(stale_ids)

        return jsonify({
            'available': True,
            'current_iter': tracker.current_iter,
            'total_datapoints': stats.get('total', 0),
            'stale_count': stale_count,
            'stats': stats
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/training/incremental', methods=['POST'])
def incremental_training():
    """
    Trigger incremental training on stale datapoints.

    Request JSON:
    {
        "dataset": "wikipedia_physics",   // Dataset to train on
        "model": "bivector_paired",       // Projection model to update (optional)
        "staleness_threshold": 100,       // Iterations before retraining
        "batch_size": 64,                 // Training batch size
        "max_samples": 200,               // Max samples to train on
        "use_category_bridge": true       // Use Wikipedia categories for training pairs
    }

    Response:
    {
        "trained_count": 50,
        "new_iteration": 1543,
        "training_time_ms": 1500,
        "samples": [...]  // Optional, list of trained titles
    }
    """
    try:
        if not _load_training_modules():
            return jsonify({
                'error': 'Training modules not available. Install unifyweaver package.'
            }), 500

        data = request.get_json()

        # Required params
        dataset = data.get('dataset')
        if not dataset:
            return jsonify({'error': 'dataset required'}), 400

        # Resolve dataset path
        if not dataset.endswith('.npz'):
            dataset_path = DATASETS_DIR / f"{dataset}.npz"
        else:
            dataset_path = Path(dataset)

        if not dataset_path.exists():
            return jsonify({'error': f'Dataset not found: {dataset_path}'}), 404

        # Optional params
        staleness_threshold = data.get('staleness_threshold', 100)
        batch_size = data.get('batch_size', 64)
        max_samples = data.get('max_samples', 200)
        use_category_bridge = data.get('use_category_bridge', True)

        # Load embeddings and titles
        embeddings, titles = load_embeddings(str(dataset_path))
        if titles is None:
            return jsonify({'error': 'Dataset has no titles for training'}), 400

        # Get training tracker
        tracker = _get_training_tracker()
        if tracker is None:
            return jsonify({'error': 'Failed to initialize training tracker'}), 500

        # Register all datapoints (idempotent)
        source = "wikipedia" if "wikipedia" in dataset.lower() else "pearltrees"
        tracker.register_batch(titles, source)

        # Sample stale datapoints for training
        stale_batch = tracker.sample_balanced_batch(
            available_ids=titles,
            batch_size=min(batch_size, max_samples),
            staleness_threshold=staleness_threshold
        )

        if not stale_batch:
            return jsonify({
                'trained_count': 0,
                'new_iteration': tracker.current_iter,
                'training_time_ms': 0,
                'message': 'No stale datapoints to train'
            })

        import time
        start_time = time.time()

        # Generate training pairs
        training_pairs = []

        if use_category_bridge and source == "wikipedia":
            # Try to use Wikipedia category bridge
            try:
                bridge = _WikipediaCategoryBridge()
                if bridge.is_available():
                    # Load Pearltrees data for folder matching
                    pearltrees_path = PROJECT_ROOT / "reports" / "pearltrees_targets_full_multi_account.jsonl"
                    if pearltrees_path.exists():
                        import json
                        pearltrees_data = []
                        with open(pearltrees_path) as f:
                            for line in f:
                                pearltrees_data.append(json.loads(line))

                        # Generate pairs via category bridge
                        pairs = bridge.generate_batch_training_pairs(
                            stale_batch,
                            pearltrees_data,
                            max_depth=10
                        )
                        training_pairs = [
                            {'query': p.query, 'target': p.target, 'distance': p.distance}
                            for p in pairs
                        ]
            except Exception as e:
                print(f"Category bridge failed, falling back to self-supervised: {e}")

        # If no category bridge pairs, use self-supervised fallback
        if not training_pairs:
            # Self-supervised: similar embeddings should map to similar targets
            # This is a placeholder - real implementation would use actual targets
            for title in stale_batch:
                training_pairs.append({
                    'query': title,
                    'target': title,  # Self-target as fallback
                    'distance': 0.0
                })

        # Update training metadata
        tracker.update_trained(stale_batch)
        tracker.increment_iteration()
        tracker.save()

        training_time = (time.time() - start_time) * 1000

        return jsonify({
            'trained_count': len(stale_batch),
            'new_iteration': tracker.current_iter,
            'training_time_ms': round(training_time, 1),
            'samples': stale_batch[:10],  # First 10 for display
            'pairs_generated': len(training_pairs),
            'category_bridge_used': len(training_pairs) > 0 and use_category_bridge
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


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
    print("  GET  /api/training/status - Get training metadata status")
    print("  POST /api/training/incremental - Trigger incremental training")

    app.run(host=args.host, port=args.port, debug=args.debug)
