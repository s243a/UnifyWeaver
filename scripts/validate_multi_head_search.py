#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Validate multi-head LDA projection using held-out test set

"""
Compare multi-head projection against direct similarity and global projection.

Tests novel queries from a dedicated test set file (qa_pairs_test.json) 
to measure generalization performance (Recall@1, MRR).

Usage:
    python scripts/validate_multi_head_search.py \
        --db playbooks/lda-training-data/lda.db \
        --test-set playbooks/lda-training-data/raw/qa_pairs_test.json \
        --mh-id 1
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from lda_database import LDAProjectionDB

def load_test_queries(file_path):
    """Load novel queries from JSON test set."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_queries = {}
    for cluster in data.get('clusters', []):
        cluster_id = cluster.get('id')
        queries = cluster.get('queries', {}).get('novel', [])
        if cluster_id and queries:
            test_queries[cluster_id] = queries
    
    return test_queries

def validate_novel_queries(
    db: LDAProjectionDB,
    mh_id: int,
    test_queries: dict,
    embedder,
    model_name: str
):
    """Test novel queries comparing multi-head vs direct similarity.

    Args:
        db: Database connection
        mh_id: Multi-head projection ID
        test_queries: Dictionary mapping cluster_id to list of query strings
        embedder: SentenceTransformer or similar
        model_name: Model name for getting answer embeddings
    """
    model = db.get_model(model_name)
    if not model:
        raise ValueError(f"Model '{model_name}' not found")
    model_id = model['model_id']

    # Get all clusters to find expected answer IDs
    clusters = db.list_clusters()
    # Map both ID and Name to cluster_id for flexibility
    cluster_map = {c['name']: c['cluster_id'] for c in clusters if c.get('name')}
    # Also add ID -> ID mapping if name isn't the ID
    for c in clusters:
        # Some clusters might store the ID in a different field or simply as the ID itself
        # This logic depends on how cluster IDs are stored in lda.db vs the JSON file
        # Assuming we can find them by name which matches the JSON 'id'
        pass

    # Get all answer embeddings for direct comparison
    answer_ids, answer_matrix = db.get_all_answer_embeddings(model_id)

    if len(answer_ids) == 0:
        print("No answer embeddings found!")
        return

    # Normalize answer embeddings
    answer_norms = np.linalg.norm(answer_matrix, axis=1, keepdims=True)
    answer_norms = np.where(answer_norms > 0, answer_norms, 1)
    answer_matrix_normed = answer_matrix / answer_norms

    print("\n" + "=" * 70)
    print("Held-Out Test Set Validation: Multi-Head vs Direct Similarity")
    print("=" * 70)

    total = 0
    correct_mh = 0
    correct_direct = 0
    mrr_mh = []
    mrr_direct = []

    for topic, queries in test_queries.items():
        # Find expected cluster
        expected_cluster_id = cluster_map.get(topic)
        if expected_cluster_id is None:
            # Try finding by looking up cluster IDs directly if stored differently
            # For now, assume topic matches cluster name
            print(f"\nSkipping topic '{topic}': no matching cluster in DB")
            continue

        # Get expected answer IDs for this cluster
        full_cluster = db.get_cluster(expected_cluster_id)
        if not full_cluster:
            continue
        expected_answer_ids = {a['answer_id'] for a in full_cluster['answers']}
        
        if not expected_answer_ids:
             print(f"\nSkipping topic '{topic}': no answers in cluster")
             continue

        print(f"\nTopic: {topic}")

        for query in queries:
            total += 1

            # Embed query
            query_emb = embedder.encode(query, convert_to_numpy=True)

            # --- Multi-head search ---
            mh_results = db.multi_head_search(
                query_embedding=query_emb,
                mh_projection_id=mh_id,
                top_k=10,
                log=False
            )

            mh_rank = None
            for rank, r in enumerate(mh_results, 1):
                if r['answer_id'] in expected_answer_ids:
                    mh_rank = rank
                    break

            if mh_rank == 1:
                correct_mh += 1
            mrr_mh.append(1.0 / mh_rank if mh_rank else 0)

            # --- Direct similarity ---
            query_norm = np.linalg.norm(query_emb)
            query_normed = query_emb / query_norm if query_norm > 0 else query_emb

            direct_sims = answer_matrix_normed @ query_normed
            direct_ranking = np.argsort(-direct_sims)

            direct_rank = None
            for rank, idx in enumerate(direct_ranking[:10], 1):
                if answer_ids[idx] in expected_answer_ids:
                    direct_rank = rank
                    break

            if direct_rank == 1:
                correct_direct += 1
            mrr_direct.append(1.0 / direct_rank if direct_rank else 0)

            # Display result
            mh_status = "OK" if mh_rank == 1 else f"rank={mh_rank}" if mh_rank else "miss"
            direct_status = "OK" if direct_rank == 1 else f"rank={direct_rank}" if direct_rank else "miss"

            # Show routing weights for insight
            if mh_results:
                top_routes = sorted(
                    mh_results[0].get('routing_weights', {}).items(),
                    key=lambda x: -x[1]
                )[:3]
                route_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_routes)
            else:
                route_str = "n/a"

            print(f"  [{mh_status:>6}|{direct_status:>6}] \"{query[:40]}\" routes:[{route_str}]")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    recall_mh = correct_mh / total if total > 0 else 0
    recall_direct = correct_direct / total if total > 0 else 0
    avg_mrr_mh = np.mean(mrr_mh) if mrr_mh else 0
    avg_mrr_direct = np.mean(mrr_direct) if mrr_direct else 0

    print(f"\nNovel queries tested: {total}")
    print(f"\n{'Metric':<20} {'Multi-Head':>12} {'Direct':>12} {'Improvement':>12}")
    print("-" * 56)
    print(f"{'Recall@1':<20} {recall_mh:>12.1%} {recall_direct:>12.1%} {(recall_mh - recall_direct):>+12.1%}")
    print(f"{'MRR':<20} {avg_mrr_mh:>12.4f} {avg_mrr_direct:>12.4f} {(avg_mrr_mh - avg_mrr_direct):>+12.4f}")

    if recall_mh > recall_direct:
        print("\n[+] Multi-head improves over direct similarity!")
    elif recall_mh == recall_direct:
        print("\n[=] Multi-head matches direct similarity")
    else:
        print("\n[-] Direct similarity outperforms multi-head")

    # Update metrics in DB if requested (optional logic)
    # db.update_multi_head_metrics(mh_id, recall_at_1=recall_mh, mrr=avg_mrr_mh)

    return {
        'recall_mh': recall_mh,
        'recall_direct': recall_direct,
        'mrr_mh': avg_mrr_mh,
        'mrr_direct': avg_mrr_direct,
        'total': total,
        'correct_mh': correct_mh,
        'correct_direct': correct_direct
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate multi-head LDA projection with held-out test set'
    )
    parser.add_argument(
        '--db', required=True,
        help='Path to LDA database'
    )
    parser.add_argument(
        '--test-set', required=True,
        help='Path to JSON test set file'
    )
    parser.add_argument(
        '--mh-id', type=int, required=True,
        help='Multi-head projection ID to validate'
    )
    parser.add_argument(
        '--model', default='all-MiniLM-L6-v2',
        help='Embedding model name (default: all-MiniLM-L6-v2)'
    )

    args = parser.parse_args()

    # Check database exists
    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}")
        return 1
        
    # Check test set exists
    if not Path(args.test_set).exists():
        print(f"Error: Test set file not found: {args.test_set}")
        return 1

    # Load embedder
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return 1

    print(f"Loading embedding model: {args.model}")
    embedder = SentenceTransformer(args.model)
    
    # Load test queries
    print(f"Loading test queries from: {args.test_set}")
    test_queries = load_test_queries(args.test_set)
    print(f"Loaded {sum(len(q) for q in test_queries.values())} queries across {len(test_queries)} clusters")

    # Open database
    db = LDAProjectionDB(args.db)

    try:
        # Validate
        validate_novel_queries(
            db=db,
            mh_id=args.mh_id,
            test_queries=test_queries,
            embedder=embedder,
            model_name=args.model
        )
        return 0

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
