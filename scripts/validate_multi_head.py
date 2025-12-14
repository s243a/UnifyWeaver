#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Validate multi-head LDA projection

"""
Compare multi-head projection against direct similarity and global projection.

Tests novel queries not in training data to measure generalization.

Usage:
    python scripts/validate_multi_head.py --db playbooks/lda-training-data/lda.db --mh-id 1
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

# Novel test queries by cluster topic
# These are NOT in the training data
NOVEL_QUERIES = {
    "csv_data_source": [
        "load tabular data from file",
        "process comma-separated values",
        "import data from spreadsheet",
    ],
    "mutual_recursion": [
        "two functions calling each other",
        "alternating recursive calls",
        "interdependent predicates",
    ],
    "xml_python_source": [
        "process structured markup data",
        "extract data from xml tags",
        "use python for xml transformation",
    ],
    "component_registry": [
        "manage system modules",
        "plugin registration system",
        "lazy loading components",
    ],
    "lda_projection": [
        "transform search queries",
        "improve vector search",
        "embedding space transformation",
    ],
    "sqlite_source": [
        "run SQL against local database",
        "fetch records from db file",
        "database query integration",
    ],
    "http_source": [
        "get data from web api",
        "fetch remote endpoint",
        "REST service integration",
    ],
    "json_source": [
        "filter nested objects",
        "extract array elements",
        "parse config files",
    ],
    "template_system": [
        "generate code with placeholders",
        "variable substitution in templates",
        "parameterized code generation",
    ],
    "cross_target_glue": [
        "connect different languages",
        "pass data between scripts",
        "polyglot pipeline",
    ],
}


def validate_novel_queries(
    db: LDAProjectionDB,
    mh_id: int,
    embedder,
    model_name: str
):
    """Test novel queries comparing multi-head vs direct similarity.

    Args:
        db: Database connection
        mh_id: Multi-head projection ID
        embedder: SentenceTransformer or similar
        model_name: Model name for getting answer embeddings
    """
    model = db.get_model(model_name)
    if not model:
        raise ValueError(f"Model '{model_name}' not found")
    model_id = model['model_id']

    # Get all clusters to find expected answer IDs
    clusters = db.list_clusters()
    cluster_by_name = {c['name']: c['cluster_id'] for c in clusters if c.get('name')}

    # Get all answer embeddings for direct comparison
    answer_ids, answer_matrix = db.get_all_answer_embeddings(model_id)

    if len(answer_ids) == 0:
        print("No answer embeddings found!")
        return

    # Normalize answer embeddings
    answer_norms = np.linalg.norm(answer_matrix, axis=1, keepdims=True)
    answer_norms = np.where(answer_norms > 0, answer_norms, 1)
    answer_matrix_normed = answer_matrix / answer_norms

    # Build answer_id -> cluster mapping
    answer_to_cluster = {}
    for cluster in clusters:
        full_cluster = db.get_cluster(cluster['cluster_id'])
        if full_cluster:
            for a in full_cluster['answers']:
                answer_to_cluster[a['answer_id']] = cluster['cluster_id']

    print("\n" + "=" * 70)
    print("Novel Query Validation: Multi-Head vs Direct Similarity")
    print("=" * 70)

    total = 0
    correct_mh = 0
    correct_direct = 0
    mrr_mh = []
    mrr_direct = []

    for topic, queries in NOVEL_QUERIES.items():
        # Find expected cluster
        expected_cluster_id = cluster_by_name.get(topic)
        if expected_cluster_id is None:
            print(f"\nSkipping topic '{topic}': no matching cluster")
            continue

        # Get expected answer IDs for this cluster
        full_cluster = db.get_cluster(expected_cluster_id)
        if not full_cluster:
            continue
        expected_answer_ids = {a['answer_id'] for a in full_cluster['answers']}

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

    # Update metrics
    db.update_multi_head_metrics(mh_id, recall_at_1=recall_mh, mrr=avg_mrr_mh)

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
        description='Validate multi-head LDA projection with novel queries'
    )
    parser.add_argument(
        '--db', required=True,
        help='Path to LDA database'
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

    # Load embedder
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return 1

    print(f"Loading embedding model: {args.model}")
    embedder = SentenceTransformer(args.model)

    # Open database
    db = LDAProjectionDB(args.db)

    try:
        # Validate
        validate_novel_queries(
            db=db,
            mh_id=args.mh_id,
            embedder=embedder,
            model_name=args.model
        )
        return 0

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
