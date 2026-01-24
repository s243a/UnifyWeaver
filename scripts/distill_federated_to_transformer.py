#!/usr/bin/env python3
"""
Distill Federated Procrustes Model to Transformer.

Takes a trained federated model (with per-cluster W matrices) and distills
it into a compact transformer that approximates the projection behavior.

The transformer learns to replace:
    Route to cluster → Load W → Q_emb @ W

With a single forward pass:
    Q_emb → Transformer → Projected_emb

Key insight: For --transform-mode single, N = num_clusters (not num_queries)
since each cluster shares one W matrix.

Usage:
    python3 scripts/distill_federated_to_transformer.py \
        models/federated.pkl \
        models/transformer.pt \
        --num-heads 4 \
        --num-layers 6 \
        --epochs 200

See docs/proposals/TRANSFORMER_DISTILLATION.md for theory.
"""

import sys
import json
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from types import ModuleType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NumPy 2.0 compatibility shim for pickle files saved with numpy._core
if not hasattr(np, '_core'):
    np_core = ModuleType('numpy._core')
    np_core.multiarray = np.core.multiarray
    np_core.umath = np.core.umath
    sys.modules['numpy._core'] = np_core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from projection_transformer import (
    ProjectionTransformer,
    train_distillation,
    evaluate_equivalence,
    optimal_architecture,
    generate_candidate_architectures,
    compute_aic_student_t,
    compute_aic_cosine
)


class FederatedProjectionWrapper:
    """
    Wrapper to provide .project() interface for federated model.

    Used as the teacher model for transformer distillation.
    """

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)

        # Load model metadata
        logger.info(f"Loading federated model from {model_path}...")
        with open(model_path, "rb") as f:
            self.meta = pickle.load(f)

        self.cluster_ids = self.meta["cluster_ids"]
        self.temperature = self.meta.get("temperature", 0.1)
        self.num_clusters = len(self.cluster_ids)

        # Determine cluster directory
        if "cluster_dir" in self.meta:
            self.cluster_dir = Path(self.meta["cluster_dir"])
        else:
            self.cluster_dir = self.model_path.with_suffix('')

        logger.info(f"Model has {self.num_clusters} clusters")

        # Load cluster centroids for routing
        self.cluster_centroids = self.meta.get("cluster_centroids")
        if self.cluster_centroids is not None:
            logger.info(f"Loaded {len(self.cluster_centroids)} cluster centroids for routing")

        # Load routing data
        self._load_routing_data()

        # Load cluster W matrices
        self._load_clusters()

        # Determine embedding dimension
        if self.clusters:
            first_W = next(iter(self.clusters.values()))["W"]
            self.embed_dim = first_W.shape[0]
        else:
            self.embed_dim = 384  # Default

        logger.info(f"Embedding dimension: {self.embed_dim}")

    def _load_routing_data(self):
        """Load query embeddings and index-to-cluster mapping for routing."""
        routing_path = self.cluster_dir / "routing_data.npz"

        if routing_path.exists():
            logger.info(f"Loading routing data from {routing_path}...")
            data = np.load(routing_path)
            self.query_embeddings = data["query_embeddings"]
            self.target_embeddings = data["target_embeddings"]

            # Reconstruct index-to-cluster mapping
            keys = data["idx_to_cluster_keys"]
            values = data["idx_to_cluster_values"]
            self.idx_to_cluster = {int(k): str(v) for k, v in zip(keys, values)}

            logger.info(f"Loaded {len(self.query_embeddings)} query embeddings for routing")
        else:
            raise FileNotFoundError(f"Routing data not found at {routing_path}")

    def _load_clusters(self):
        """Load W matrices from each cluster."""
        self.clusters = {}

        for cid in self.cluster_ids:
            cluster_path = self.cluster_dir / f"{cid}.npz"
            if cluster_path.exists():
                data = np.load(cluster_path)
                self.clusters[cid] = {
                    "W": data["W_stack"][0],  # Single W per cluster
                    "indices": data["indices"]
                }

        logger.info(f"Loaded {len(self.clusters)} cluster W matrices")

    def project(self, q_emb: np.ndarray, top_k_routing: int = 10, k_coverage: float = None) -> np.ndarray:
        """
        Project query using federated softmax routing over cluster centroids.

        Flow: query → sim(query, centroids) → softmax → Σ weight_i × (query @ W_i)

        Args:
            q_emb: Query embedding (embed_dim,)
            top_k_routing: Fixed number of top clusters (default: 10)
            k_coverage: If set, override top_k with coverage-based k (e.g., 0.5 for 50%)

        Returns:
            Projected embedding (embed_dim,)
        """
        # Normalize query
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        if self.cluster_centroids is not None:
            # Query-based routing: compare query to cluster centroids
            sims = q_norm @ self.cluster_centroids.T

            # Softmax weights over centroids (temperature-scaled)
            sims_shifted = (sims - np.max(sims)) / self.temperature
            weights = np.exp(sims_shifted)
            weights /= weights.sum()

            # Determine k: coverage-based if specified, else fixed
            if k_coverage is not None:
                # Auto-k: find minimum k to capture k_coverage of weight
                sorted_indices = np.argsort(weights)[::-1]
                cumsum = 0.0
                k = 1
                for i, idx in enumerate(sorted_indices):
                    cumsum += weights[idx]
                    if cumsum >= k_coverage:
                        k = i + 1
                        break
                k = max(1, min(k, len(weights)))
            else:
                k = top_k_routing

            # Get top-k clusters by weight
            top_indices = np.argsort(weights)[-k:]

            # Weighted projection: Σ weight_i × (query @ W_i)
            proj = np.zeros_like(q_emb)
            total_weight = 0.0

            for idx in top_indices:
                cid = self.cluster_ids[idx]
                if cid in self.clusters:
                    W = self.clusters[cid]["W"]
                    w = weights[idx]
                    proj += w * (q_emb @ W)
                    total_weight += w

            if total_weight > 0:
                proj /= total_weight

            return proj
        else:
            # Fallback if no centroids: use training query similarity
            sims = q_norm @ self.query_embeddings.T

            sims_shifted = (sims - np.max(sims)) / self.temperature
            weights = np.exp(sims_shifted)
            weights /= weights.sum()

            # Determine k: coverage-based if specified, else fixed
            if k_coverage is not None:
                sorted_indices = np.argsort(weights)[::-1]
                cumsum = 0.0
                k = 1
                for i, idx in enumerate(sorted_indices):
                    cumsum += weights[idx]
                    if cumsum >= k_coverage:
                        k = i + 1
                        break
                k = max(1, min(k, len(weights)))
            else:
                k = top_k_routing

            top_indices = np.argsort(weights)[-k:]

            proj = np.zeros_like(q_emb)
            total_weight = 0.0

            for idx in top_indices:
                cid = self.idx_to_cluster.get(int(idx))
                if cid and cid in self.clusters:
                    W = self.clusters[cid]["W"]
                    w = weights[idx]
                    proj += w * (q_emb @ W)
                    total_weight += w

            if total_weight > 0:
                proj /= total_weight

            return proj

    def project_batch(self, q_embs: np.ndarray, top_k_routing: int = 10, k_coverage: float = None) -> np.ndarray:
        """Project a batch of query embeddings."""
        return np.array([self.project(q, top_k_routing, k_coverage) for q in q_embs])


def main():
    parser = argparse.ArgumentParser(
        description="Distill federated Procrustes model to compact transformer"
    )
    parser.add_argument("input_model", type=Path,
                       help="Input federated model (.pkl)")
    parser.add_argument("output_model", type=Path,
                       help="Output transformer model (.pt)")
    parser.add_argument("--architecture", type=str, default="auto",
                       choices=["auto", "suggest", "compare"],
                       help="Architecture selection mode: auto (pick one), suggest (show options), compare (train and select)")
    parser.add_argument("--num-suggestions", type=int, default=3,
                       help="Number of architecture suggestions for suggest/compare mode (default: 3)")
    parser.add_argument("--selection-criterion", type=str, default="aic_t",
                       choices=["best_cosine", "aic_t", "aic_gaussian", "bic_t"],
                       help="Selection criterion for compare mode (default: aic_t)")
    parser.add_argument("--compare-epochs", type=int, default=50,
                       help="Training epochs for compare mode (default: 50)")
    parser.add_argument("--num-heads", type=int, default=None,
                       help="Attention heads per layer (default: auto from H^L=N)")
    parser.add_argument("--num-layers", type=int, default=None,
                       help="Number of transformer layers (default: auto from H^L=N)")
    parser.add_argument("--ff-dim", type=int, default=None,
                       help="Feed-forward dimension (default: 2x embed_dim)")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--cosine-weight", type=float, default=0.7,
                       help="Weight for cosine loss vs MSE (default: 0.7)")
    parser.add_argument("--test-split", type=float, default=0.1,
                       help="Fraction for test set (default: 0.1)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto, cuda, cpu (default: auto)")
    parser.add_argument("--top-k-routing", type=int, default=10,
                       help="Top-k clusters for routing (default: 10)")
    parser.add_argument("--k-coverage", type=float, default=None,
                       help="Use coverage-based k instead of fixed (e.g., 0.5 for 50%%)")
    parser.add_argument("--export-onnx", type=Path, default=None,
                       help="Export trained model to ONNX format for browser deployment")
    parser.add_argument("--onnx-opset", type=int, default=14,
                       help="ONNX opset version (default: 14)")
    parser.add_argument("--verify-onnx", action="store_true",
                       help="Verify ONNX export matches PyTorch output")

    args = parser.parse_args()

    if not args.input_model.exists():
        logger.error(f"Input model not found: {args.input_model}")
        return 1

    print("=" * 60)
    print("Federated → Transformer Distillation")
    print("=" * 60)

    # Load federated model (teacher)
    federated = FederatedProjectionWrapper(args.input_model)

    n_clusters = federated.num_clusters
    embed_dim = federated.embed_dim

    print(f"\nTeacher model:")
    print(f"  Clusters (N): {n_clusters}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Temperature: {federated.temperature}")

    # Handle architecture modes
    if args.architecture == "suggest":
        # Just show suggestions and exit
        candidates = generate_candidate_architectures(n_clusters, embed_dim, args.num_suggestions)
        print(f"\n{'=' * 60}")
        print(f"Architecture Suggestions for {n_clusters} clusters")
        print(f"{'=' * 60}")
        print(f"\n{'H':>4} {'L':>4} {'H^L':>6} {'Overhead':>10} {'~Params':>12} {'Notes'}")
        print(f"{'-'*4} {'-'*4} {'-'*6} {'-'*10} {'-'*12} {'-'*20}")
        for i, c in enumerate(candidates):
            notes = []
            if c['num_layers'] == 2:
                notes.append("shallow, fast")
            elif c['num_layers'] == 3:
                notes.append("balanced")
            elif c['num_layers'] >= 4:
                notes.append("deep, expressive")
            if c['overhead_pct'] < 20:
                notes.append("low overhead")
            notes_str = ", ".join(notes) if notes else ""
            print(f"{c['num_heads']:>4} {c['num_layers']:>4} {c['capacity']:>6} "
                  f"{c['overhead_pct']:>9.1f}% {c['params_estimate']:>11,} {notes_str}")
        print(f"\nTo train a specific architecture:")
        print(f"  --num-heads H --num-layers L")
        print(f"\nTo compare and auto-select:")
        print(f"  --architecture compare --selection-criterion aic_t")
        return 0

    if args.architecture == "compare":
        # Compare mode: train multiple, select by criterion
        candidates = generate_candidate_architectures(n_clusters, embed_dim, args.num_suggestions)
        print(f"\n{'=' * 60}")
        print(f"Comparing {len(candidates)} architectures ({args.compare_epochs} epochs each)")
        print(f"Selection criterion: {args.selection_criterion}")
        print(f"{'=' * 60}")

        # Get training data early for comparison
        query_embeddings = federated.query_embeddings.astype(np.float32)
        n_samples = len(query_embeddings)
        n_test = int(n_samples * args.test_split)
        n_train = n_samples - n_test

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_queries = query_embeddings[train_idx]
        test_queries = query_embeddings[test_idx]

        ff_dim = args.ff_dim or embed_dim * 2

        comparison_results = []

        for i, c in enumerate(candidates):
            print(f"\n--- Candidate {i+1}/{len(candidates)}: H={c['num_heads']}, L={c['num_layers']} (capacity={c['capacity']}) ---")

            # Create and train
            transformer = ProjectionTransformer(
                embed_dim=embed_dim,
                num_heads=c['num_heads'],
                num_layers=c['num_layers'],
                ff_dim=ff_dim,
                device=args.device
            )
            n_params = transformer.get_info()['total_parameters']

            losses = train_distillation(
                transformer=transformer,
                lda_projection=federated,
                query_embeddings=train_queries,
                num_epochs=args.compare_epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                log_interval=args.compare_epochs,  # Only log at end
                cosine_weight=args.cosine_weight
            )

            # Evaluate
            results = evaluate_equivalence(transformer, federated, test_queries)

            # Compute AIC
            cosine_sims = np.array([results['mean_cosine_sim']] * len(test_queries))
            # Recompute per-sample cosine sims for proper AIC
            cosine_sims_per_sample = []
            for q in test_queries:
                lda_out = federated.project(q)
                trans_out = transformer.project(q)
                cos = np.dot(lda_out, trans_out) / (np.linalg.norm(lda_out) * np.linalg.norm(trans_out) + 1e-8)
                cosine_sims_per_sample.append(cos)
            cosine_sims_per_sample = np.array(cosine_sims_per_sample)

            aic_result = compute_aic_cosine(cosine_sims_per_sample, n_params)

            comparison_results.append({
                'candidate': c,
                'transformer': transformer,
                'n_params': n_params,
                'mean_cosine_sim': results['mean_cosine_sim'],
                'std_cosine_sim': results['std_cosine_sim'],
                'final_loss': losses[-1],
                'aic_t': aic_result['aic_t'],
                'aic_gaussian': aic_result['aic_gaussian'],
                'bic_t': aic_result['bic_t'],
                'df_estimated': aic_result['df_estimated'],
            })

            print(f"  Cosine: {results['mean_cosine_sim']:.4f} ± {results['std_cosine_sim']:.4f}")
            print(f"  AIC(t): {aic_result['aic_t']:.2f}, AIC(gauss): {aic_result['aic_gaussian']:.2f}")
            print(f"  Estimated df: {aic_result['df_estimated']:.1f}")

        # Select winner
        print(f"\n{'=' * 60}")
        print("Comparison Results")
        print(f"{'=' * 60}")

        if args.selection_criterion == "best_cosine":
            comparison_results.sort(key=lambda r: -r['mean_cosine_sim'])
            criterion_name = "Cosine Similarity"
        elif args.selection_criterion == "aic_t":
            comparison_results.sort(key=lambda r: r['aic_t'])
            criterion_name = "AIC (Student's t)"
        elif args.selection_criterion == "aic_gaussian":
            comparison_results.sort(key=lambda r: r['aic_gaussian'])
            criterion_name = "AIC (Gaussian)"
        elif args.selection_criterion == "bic_t":
            comparison_results.sort(key=lambda r: r['bic_t'])
            criterion_name = "BIC (Student's t)"

        print(f"\nRanked by {criterion_name}:")
        print(f"\n{'Rank':>4} {'H':>4} {'L':>4} {'Params':>12} {'Cosine':>10} {'AIC(t)':>12} {'AIC(g)':>12}")
        print(f"{'-'*4} {'-'*4} {'-'*4} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")

        for rank, r in enumerate(comparison_results):
            c = r['candidate']
            marker = "***" if rank == 0 else ""
            print(f"{rank+1:>4} {c['num_heads']:>4} {c['num_layers']:>4} {r['n_params']:>12,} "
                  f"{r['mean_cosine_sim']:>10.4f} {r['aic_t']:>12.2f} {r['aic_gaussian']:>12.2f} {marker}")

        # Winner
        winner = comparison_results[0]
        winner_c = winner['candidate']
        print(f"\n*** Selected: H={winner_c['num_heads']}, L={winner_c['num_layers']} ***")
        print(f"    Cosine: {winner['mean_cosine_sim']:.4f}, AIC(t): {winner['aic_t']:.2f}")

        # Retrain winner with full epochs if compare_epochs < epochs
        if args.compare_epochs < args.epochs:
            print(f"\nRetraining winner with full {args.epochs} epochs...")
            transformer = ProjectionTransformer(
                embed_dim=embed_dim,
                num_heads=winner_c['num_heads'],
                num_layers=winner_c['num_layers'],
                ff_dim=ff_dim,
                device=args.device
            )
            losses = train_distillation(
                transformer=transformer,
                lda_projection=federated,
                query_embeddings=train_queries,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                log_interval=20,
                cosine_weight=args.cosine_weight
            )
            results = evaluate_equivalence(transformer, federated, test_queries)
            print(f"Final cosine: {results['mean_cosine_sim']:.4f}")
        else:
            transformer = winner['transformer']

        # Save and continue to export logic
        args.num_heads = winner_c['num_heads']
        args.num_layers = winner_c['num_layers']

        # Save winner
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        transformer.save(str(args.output_model))
        print(f"\nSaved transformer to {args.output_model}")

        # Export ONNX if requested (same as normal flow)
        if args.export_onnx:
            print(f"\nExporting to ONNX...")
            args.export_onnx.parent.mkdir(parents=True, exist_ok=True)
            onnx_info = transformer.export_onnx(str(args.export_onnx), opset_version=args.onnx_opset)
            print(f"  Output: {onnx_info['path']}")
            print(f"  Size: {onnx_info['size_mb']:.2f} MB")

        print("\nDone!")
        return 0

    # Calculate optimal architecture if not specified (auto mode)
    if args.num_heads is None or args.num_layers is None:
        h, l = optimal_architecture(n_clusters, prefer_h=4, embed_dim=embed_dim)
        if args.num_heads is None:
            args.num_heads = h
        if args.num_layers is None:
            args.num_layers = l

    equivalent = args.num_heads ** args.num_layers
    print(f"\nTransformer architecture:")
    print(f"  Heads (H): {args.num_heads}")
    print(f"  Layers (L): {args.num_layers}")
    print(f"  Equivalent capacity: H^L = {equivalent} (target: {n_clusters})")

    # FF dimension
    ff_dim = args.ff_dim or embed_dim * 2

    # Get training data (query embeddings from federated model)
    query_embeddings = federated.query_embeddings.astype(np.float32)
    n_samples = len(query_embeddings)

    print(f"\nTraining data:")
    print(f"  Query embeddings: {n_samples}")

    # Split train/test
    n_test = int(n_samples * args.test_split)
    n_train = n_samples - n_test

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_queries = query_embeddings[train_idx]
    test_queries = query_embeddings[test_idx]

    print(f"  Train: {n_train}, Test: {n_test}")

    # Create transformer (student)
    print(f"\nCreating transformer...")
    transformer = ProjectionTransformer(
        embed_dim=embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=ff_dim,
        device=args.device
    )

    info = transformer.get_info()
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Device: {info['device']}")

    # Calculate compression ratio
    federated_params = n_clusters * embed_dim * embed_dim
    transformer_params = info['total_parameters']
    compression = federated_params / transformer_params

    print(f"\nCompression:")
    print(f"  Federated W params: {federated_params:,}")
    print(f"  Transformer params: {transformer_params:,}")
    print(f"  Ratio: {compression:.1f}x smaller")

    # Train via distillation
    print(f"\nTraining for {args.epochs} epochs (cosine_weight={args.cosine_weight})...")
    losses = train_distillation(
        transformer=transformer,
        lda_projection=federated,
        query_embeddings=train_queries,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_interval=20,
        cosine_weight=args.cosine_weight
    )

    # Evaluate on test set
    if n_test > 0:
        print("\nEvaluating on test set...")
        results = evaluate_equivalence(transformer, federated, test_queries)

        print(f"\n{'=' * 40}")
        print("Results:")
        print(f"  Mean MSE: {results['mean_mse']:.6f} ± {results['std_mse']:.6f}")
        print(f"  Mean Cosine Sim: {results['mean_cosine_sim']:.4f} ± {results['std_cosine_sim']:.4f}")
        print(f"  Min Cosine Sim: {results['min_cosine_sim']:.4f}")
        print(f"  Max Cosine Sim: {results['max_cosine_sim']:.4f}")
        print(f"{'=' * 40}")

        # Interpretation
        if results['mean_cosine_sim'] > 0.95:
            print("\n✓ Excellent: Transformer closely approximates federated model")
        elif results['mean_cosine_sim'] > 0.90:
            print("\n✓ Good: Transformer reasonably approximates federated model")
        elif results['mean_cosine_sim'] > 0.80:
            print("\n~ Fair: Transformer partially approximates federated model")
        else:
            print("\n✗ Poor: Consider more epochs or larger architecture")

    # Save transformer
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    transformer.save(str(args.output_model))

    print(f"\nSaved transformer to {args.output_model}")
    print(f"Size: {args.output_model.stat().st_size / (1024*1024):.1f} MB")

    # Export to ONNX if requested
    if args.export_onnx:
        print(f"\nExporting to ONNX...")
        args.export_onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx_info = transformer.export_onnx(str(args.export_onnx), opset_version=args.onnx_opset)

        print(f"  Output: {onnx_info['path']}")
        print(f"  Size: {onnx_info['size_mb']:.2f} MB")
        print(f"  Opset: {onnx_info['opset_version']}")
        print(f"  Input: {onnx_info['input_shape']}")
        print(f"  Output: {onnx_info['output_shape']}")

        # Verify if requested
        if args.verify_onnx:
            print("\nVerifying ONNX export...")
            verify_result = transformer.verify_onnx(str(args.export_onnx))
            if verify_result.get('verified'):
                print(f"  ✓ Verified (max_diff={verify_result['max_diff']:.6f})")
            elif 'error' in verify_result:
                print(f"  ⚠ Skipped: {verify_result['error']}")
            else:
                print(f"  ✗ Mismatch (max_diff={verify_result['max_diff']:.6f})")

    print("\nDone!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
