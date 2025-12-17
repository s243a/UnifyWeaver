#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Test transformer distillation from LDA multi-head projection.

Validates the H^L = N equivalence conjecture by:
1. Loading LDA projection from database
2. Creating transformer with equivalent capacity (H^L ≈ N)
3. Training via MSE distillation
4. Comparing outputs on held-out queries

Usage:
    python scripts/test_transformer_distillation.py --db playbooks/lda-training-data/lda.db
"""

import argparse
import sys
import logging
from pathlib import Path

import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from lda_database import LDAProjectionDB
from projection_transformer import (
    ProjectionTransformer,
    train_distillation,
    evaluate_equivalence,
    optimal_architecture
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class LDAProjectionWrapper:
    """Wrapper to provide .project() interface for LDA multi-head."""

    def __init__(self, db: LDAProjectionDB, mh_projection_id: int):
        self.db = db
        self.mh_projection_id = mh_projection_id

        # Load projection info
        mh_proj = db.get_multi_head_projection(mh_projection_id)
        self.num_heads = mh_proj['num_heads']
        self.temperature = mh_proj['temperature']

        # Load all heads
        self.heads = []
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT cluster_id, centroid_path, answer_emb_path
            FROM cluster_heads
            WHERE mh_projection_id = ?
        """, (mh_projection_id,))

        for row in cursor.fetchall():
            centroid = np.load(row['centroid_path'])
            answer_emb = np.load(row['answer_emb_path'])
            self.heads.append({
                'cluster_id': row['cluster_id'],
                'centroid': centroid,
                'answer_emb': answer_emb
            })

        logger.info(f"Loaded LDA projection: {len(self.heads)} heads, temp={self.temperature}")

    def project(self, query_emb: np.ndarray) -> np.ndarray:
        """Project query using multi-head routing."""
        # Normalize query
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Compute similarities to centroids
        similarities = []
        for head in self.heads:
            centroid_norm = head['centroid'] / (np.linalg.norm(head['centroid']) + 1e-8)
            sim = np.dot(query_norm, centroid_norm)
            similarities.append(sim)

        similarities = np.array(similarities)

        # Softmax with temperature
        scaled = similarities / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        weights = exp_scaled / np.sum(exp_scaled)

        # Weighted combination of answer embeddings
        projected = np.zeros_like(query_emb)
        for i, head in enumerate(self.heads):
            projected += weights[i] * head['answer_emb']

        return projected


def main():
    parser = argparse.ArgumentParser(description="Test transformer distillation from LDA")
    parser.add_argument("--db", required=True, help="Path to LDA database")
    parser.add_argument("--mh-id", type=int, default=1, help="Multi-head projection ID")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--cosine-weight", type=float, default=0.5, help="Weight for cosine loss (0=MSE only)")
    parser.add_argument("--save", help="Path to save trained transformer")

    args = parser.parse_args()

    if not Path(args.db).exists():
        logger.error(f"Database not found: {args.db}")
        return 1

    print("=" * 60)
    print("Transformer Distillation Test")
    print("=" * 60)

    # Load LDA projection
    db = LDAProjectionDB(args.db)
    lda = LDAProjectionWrapper(db, args.mh_id)

    n_flat_heads = lda.num_heads
    print(f"\nLDA projection: {n_flat_heads} flat heads")

    # Calculate optimal transformer architecture
    h, l = optimal_architecture(n_flat_heads, prefer_h=4)
    equivalent = h ** l
    print(f"Transformer architecture: H={h}, L={l} (equivalent={equivalent} heads)")

    # Get training data from database
    print("\nCollecting training queries from database...")

    # Get all question embeddings
    model = db.get_model('all-MiniLM-L6-v2')
    if not model:
        logger.error("Model 'all-MiniLM-L6-v2' not found in database")
        return 1

    model_id = model['model_id']
    cursor = db.conn.cursor()

    # Get question embeddings
    cursor.execute("""
        SELECT e.vector_path
        FROM embeddings e
        WHERE e.model_id = ? AND e.entity_type = 'question'
    """, (model_id,))

    query_embeddings = []
    for row in cursor.fetchall():
        emb = np.load(row['vector_path'])
        query_embeddings.append(emb)

    if not query_embeddings:
        logger.error("No question embeddings found in database")
        return 1

    query_embeddings = np.stack(query_embeddings).astype(np.float32)
    print(f"Loaded {len(query_embeddings)} query embeddings")

    # Split train/test
    n_test = int(len(query_embeddings) * args.test_split)
    n_train = len(query_embeddings) - n_test

    indices = np.random.permutation(len(query_embeddings))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_queries = query_embeddings[train_idx]
    test_queries = query_embeddings[test_idx]

    print(f"Train: {len(train_queries)}, Test: {len(test_queries)}")

    # Create transformer
    print(f"\nCreating transformer (H={h}, L={l})...")
    transformer = ProjectionTransformer(
        embed_dim=384,
        num_heads=h,
        num_layers=l,
        ff_dim=512,
        device="auto"
    )
    info = transformer.get_info()
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Device: {info['device']}")

    # Train
    print(f"\nTraining for {args.epochs} epochs (cosine_weight={args.cosine_weight})...")
    losses = train_distillation(
        transformer=transformer,
        lda_projection=lda,
        query_embeddings=train_queries,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        log_interval=20,
        cosine_weight=args.cosine_weight
    )

    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluate_equivalence(transformer, lda, test_queries)

    print(f"\n{'=' * 40}")
    print("Results:")
    print(f"  Mean MSE: {results['mean_mse']:.6f} ± {results['std_mse']:.6f}")
    print(f"  Mean Cosine Sim: {results['mean_cosine_sim']:.4f} ± {results['std_cosine_sim']:.4f}")
    print(f"  Min Cosine Sim: {results['min_cosine_sim']:.4f}")
    print(f"  Max Cosine Sim: {results['max_cosine_sim']:.4f}")
    print(f"{'=' * 40}")

    # Interpretation
    if results['mean_cosine_sim'] > 0.95:
        print("\n✓ Excellent: Transformer closely approximates LDA projection")
    elif results['mean_cosine_sim'] > 0.90:
        print("\n✓ Good: Transformer reasonably approximates LDA projection")
    elif results['mean_cosine_sim'] > 0.80:
        print("\n~ Fair: Transformer partially approximates LDA projection")
    else:
        print("\n✗ Poor: Transformer does not well approximate LDA projection")

    # Compression ratio
    lda_params = n_flat_heads * 384 * 2  # centroids + answers
    transformer_params = info['total_parameters']
    if transformer_params < lda_params:
        ratio = lda_params / transformer_params
        print(f"\nCompression: {ratio:.1f}x fewer parameters than LDA")
    else:
        ratio = transformer_params / lda_params
        print(f"\nNote: Transformer has {ratio:.1f}x more parameters than LDA")

    # Save if requested
    if args.save:
        transformer.save(args.save)
        print(f"\nSaved transformer to {args.save}")

    db.close()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
