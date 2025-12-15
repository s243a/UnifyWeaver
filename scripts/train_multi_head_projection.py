#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Train multi-head LDA projection from database clusters

"""
Train a multi-head projection where each cluster gets its own "attention head".

This approach is analogous to multi-head attention in transformers:
- Each head specializes in a different type of query-answer relationship
- At inference, queries are routed to heads based on centroid similarity
- The final projection is a weighted combination of per-head outputs

Usage:
    python scripts/train_multi_head_projection.py --db playbooks/lda-training-data/lda.db
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from lda_database import LDAProjectionDB
from projection import compute_weighted_centroid


def train_multi_head(
    db: LDAProjectionDB,
    model_name: str,
    temperature: float = 1.0,
    name: str = None
) -> int:
    """Train a multi-head projection from database clusters.

    Args:
        db: Database connection
        model_name: Embedding model name
        temperature: Softmax temperature for routing
        name: Optional projection name

    Returns:
        mh_projection_id
    """
    # Get model
    model = db.get_model(model_name)
    if not model:
        raise ValueError(f"Model '{model_name}' not found in database")
    model_id = model['model_id']

    # Get all clusters
    clusters = db.list_clusters()
    if not clusters:
        raise ValueError("No clusters found in database")

    print(f"Training multi-head projection from {len(clusters)} clusters")
    print(f"Model: {model_name} (dim={model['dimension']})")
    print(f"Temperature: {temperature}")

    # Create multi-head projection
    mh_id = db.create_multi_head_projection(
        model_id=model_id,
        name=name or f"multi_head_{model_name}",
        temperature=temperature
    )
    print(f"\nCreated multi-head projection: mh_id={mh_id}")

    # Process each cluster
    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        cluster_name = cluster.get('name', f'cluster_{cluster_id}')

        # Get embeddings
        answer_embs, question_embs = db.get_cluster_embeddings(cluster_id, model_id)

        if not answer_embs or not question_embs:
            print(f"  Skipping {cluster_name}: no embeddings")
            continue

        # Stack question embeddings
        Q = np.stack(question_embs)
        num_questions = len(Q)

        # Compute weighted centroid
        centroid, weights = compute_weighted_centroid(Q)

        # Use first answer embedding (or could average if multiple)
        answer_emb = answer_embs[0]
        if len(answer_embs) > 1:
            # Average multiple answer embeddings
            answer_emb = np.mean(answer_embs, axis=0)
            answer_emb = answer_emb / np.linalg.norm(answer_emb)

        # Add cluster head (without per-cluster W for now - routing + centroids is simpler)
        head_id = db.add_cluster_head(
            mh_projection_id=mh_id,
            cluster_id=cluster_id,
            centroid=centroid,
            answer_emb=answer_emb,
            W=None,  # No per-cluster W, rely on routing
            num_questions=num_questions
        )

        print(f"  Added head for '{cluster_name}': {num_questions} questions")

    # Summary
    mh_proj = db.get_multi_head_projection(mh_id)
    print(f"\nMulti-head projection complete:")
    print(f"  ID: {mh_id}")
    print(f"  Heads: {mh_proj['num_heads']}")
    print(f"  Temperature: {mh_proj['temperature']}")

    return mh_id


def validate_multi_head(
    db: LDAProjectionDB,
    mh_id: int,
    embedder
) -> dict:
    """Validate multi-head projection on training data.

    Args:
        db: Database connection
        mh_id: Multi-head projection ID
        embedder: SentenceTransformer or similar

    Returns:
        Dict with recall_at_1, mrr
    """
    mh_proj = db.get_multi_head_projection(mh_id)
    model_id = mh_proj['model_id']

    # Get all clusters for validation
    clusters = db.list_clusters()

    correct = 0
    total = 0
    mrr_scores = []

    print("\nValidating on training data:")

    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        cluster_name = cluster.get('name', f'cluster_{cluster_id}')
        full_cluster = db.get_cluster(cluster_id)

        if not full_cluster:
            continue

        # Get expected answer IDs for this cluster
        expected_answer_ids = {a['answer_id'] for a in full_cluster['answers']}

        # Test each question
        for q in full_cluster['questions']:
            total += 1
            query_text = q['text']

            # Embed and search
            query_emb = embedder.encode(query_text, convert_to_numpy=True)
            results = db.multi_head_search(
                query_embedding=query_emb,
                mh_projection_id=mh_id,
                top_k=10,
                log=False
            )

            if not results:
                mrr_scores.append(0)
                continue

            # Find rank of correct answer
            for rank, r in enumerate(results, 1):
                if r['answer_id'] in expected_answer_ids:
                    if rank == 1:
                        correct += 1
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0)

    recall_at_1 = correct / total if total > 0 else 0
    mrr = np.mean(mrr_scores) if mrr_scores else 0

    print(f"  Total questions: {total}")
    print(f"  Recall@1: {recall_at_1:.1%} ({correct}/{total})")
    print(f"  MRR: {mrr:.4f}")

    # Update metrics in database
    db.update_multi_head_metrics(mh_id, recall_at_1=recall_at_1, mrr=mrr)

    return {'recall_at_1': recall_at_1, 'mrr': mrr, 'total': total, 'correct': correct}


def main():
    parser = argparse.ArgumentParser(
        description='Train multi-head LDA projection from database clusters'
    )
    parser.add_argument(
        '--db', required=True,
        help='Path to LDA database'
    )
    parser.add_argument(
        '--model', default='all-MiniLM-L6-v2',
        help='Embedding model name (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Softmax temperature for routing (default: 1.0)'
    )
    parser.add_argument(
        '--name',
        help='Name for the multi-head projection'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Validate on training data after training'
    )

    args = parser.parse_args()

    # Check database exists
    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}")
        return 1

    print("=" * 60)
    print("Multi-Head LDA Projection Training")
    print("=" * 60)

    # Open database
    db = LDAProjectionDB(args.db)

    try:
        # Train
        mh_id = train_multi_head(
            db=db,
            model_name=args.model,
            temperature=args.temperature,
            name=args.name
        )

        # Optionally validate
        if args.validate:
            try:
                from sentence_transformers import SentenceTransformer
                embedder = SentenceTransformer(args.model)
                validate_multi_head(db, mh_id, embedder)
            except ImportError:
                print("\nWarning: sentence-transformers not installed, skipping validation")

        print(f"\nDone! Multi-head projection ID: {mh_id}")
        return 0

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
