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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from projection_transformer import (
    ProjectionTransformer,
    train_distillation,
    evaluate_equivalence,
    optimal_architecture
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

    def project(self, q_emb: np.ndarray, top_k_routing: int = 10) -> np.ndarray:
        """
        Project query using federated routing.

        This is what the transformer will learn to approximate.

        Args:
            q_emb: Query embedding (embed_dim,)
            top_k_routing: Number of training queries for soft routing

        Returns:
            Projected embedding (embed_dim,)
        """
        # Normalize query
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        # Compute similarities to all training queries
        sims = q_norm @ self.query_embeddings.T

        # Softmax weights
        sims_shifted = (sims - np.max(sims)) / self.temperature
        weights = np.exp(sims_shifted)
        weights /= weights.sum()

        # Get top-k training queries
        top_indices = np.argsort(weights)[-top_k_routing:]

        # Weighted projection using their clusters' W
        proj = np.zeros_like(q_emb)
        total_weight = 0.0

        for idx in top_indices:
            cid = self.idx_to_cluster.get(int(idx))
            if cid and cid in self.clusters:
                W = self.clusters[cid]["W"]
                w = weights[idx]
                proj += w * (q_emb @ W)
                total_weight += w

        # Normalize by total weight used
        if total_weight > 0:
            proj /= total_weight

        return proj

    def project_batch(self, q_embs: np.ndarray, top_k_routing: int = 10) -> np.ndarray:
        """Project a batch of query embeddings."""
        return np.array([self.project(q, top_k_routing) for q in q_embs])


def main():
    parser = argparse.ArgumentParser(
        description="Distill federated Procrustes model to compact transformer"
    )
    parser.add_argument("input_model", type=Path,
                       help="Input federated model (.pkl)")
    parser.add_argument("output_model", type=Path,
                       help="Output transformer model (.pt)")
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
                       help="Top-k for federated routing (default: 10)")

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

    # Calculate optimal architecture if not specified
    if args.num_heads is None or args.num_layers is None:
        h, l = optimal_architecture(n_clusters, prefer_h=4)
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
    print("\nDone!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
