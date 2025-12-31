#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Compare Multi-Head LDA vs Smoothing Basis Projection.

Tests the hypothesis that smoothing basis performs better when questions are sparse
by leveraging answer similarity structure.

Usage:
    python scripts/compare_multihead_vs_smoothing.py \
        --data playbooks/lda-training-data/raw/smoothing_test_data.json \
        --max-questions 2 \
        --num-basis 4
"""

import argparse
import json
import sys
import logging
import time
from pathlib import Path

import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from smoothing_basis import SmoothingBasisProjection, MultiHeadLDABaseline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_embedder():
    """Load sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)


def evaluate_retrieval(projector, query_emb: np.ndarray, answer_embs: dict,
                       expected_answer_id: str, temperature: float = 0.1) -> dict:
    """
    Evaluate retrieval for a single query.

    Returns dict with:
        - correct: bool
        - rank: int (1-indexed rank of correct answer)
        - top_answer_id: str
    """
    # Project query
    projected = projector.project(query_emb, temperature=temperature)

    # Compute similarity to all answers
    scores = {}
    for answer_id, answer_emb in answer_embs.items():
        proj_norm = projected / (np.linalg.norm(projected) + 1e-8)
        ans_norm = answer_emb / (np.linalg.norm(answer_emb) + 1e-8)
        scores[answer_id] = np.dot(proj_norm, ans_norm)

    # Rank answers
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_answer_id = ranked[0][0]

    # Find rank of expected answer
    rank = 1
    for aid, _ in ranked:
        if aid == expected_answer_id:
            break
        rank += 1

    return {
        'correct': top_answer_id == expected_answer_id,
        'rank': rank,
        'top_answer_id': top_answer_id,
        'expected_answer_id': expected_answer_id
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Multi-Head LDA vs Smoothing Basis Projection"
    )
    parser.add_argument("--data", required=True, help="Path to test data JSON")
    parser.add_argument("--max-questions", type=int, default=1,
                        help="Maximum questions per cluster (sparse training)")
    parser.add_argument("--num-basis", type=int, default=4,
                        help="Number of basis matrices for smoothing")
    parser.add_argument("--cosine-weight", type=float, default=0.5,
                        help="Weight for cosine loss in smoothing basis")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Training iterations for smoothing basis")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Softmax temperature for routing")
    parser.add_argument("--difficulty", type=str, default="all",
                        choices=["easy", "medium", "hard", "all"],
                        help="Filter test queries by difficulty")

    args = parser.parse_args()

    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        return 1

    print("=" * 60)
    print("Multi-Head LDA vs Smoothing Basis Comparison")
    print("=" * 60)

    # Load data
    with open(args.data) as f:
        data = json.load(f)

    clusters = data['clusters']
    test_queries = data.get('test_queries', [])

    # Filter by difficulty if specified
    if args.difficulty != "all":
        test_queries = [q for q in test_queries if q.get('difficulty', 'easy') == args.difficulty]

    print(f"\nDataset: {len(clusters)} clusters, {len(test_queries)} test queries")
    if args.difficulty != "all":
        print(f"Difficulty filter: {args.difficulty}")
    print(f"Max questions per cluster: {args.max_questions}")
    print(f"Num basis matrices: {args.num_basis}")

    # Load embedder
    print("\nLoading embedding model...")
    embedder = load_embedder()

    # Embed all data
    print("Embedding clusters...")
    embedded_clusters = []
    answer_embs = {}

    for cluster in clusters:
        # Limit questions to max_questions (sparse training)
        questions = cluster['questions'][:args.max_questions]
        answer = cluster['answer']
        answer_id = cluster['answer_id']

        # Embed
        Q = embedder.encode(questions)
        A = embedder.encode([answer])[0]

        embedded_clusters.append((Q, A.reshape(1, -1)))
        answer_embs[answer_id] = A

    print(f"Embedded {len(embedded_clusters)} clusters")

    # Embed test queries
    print("Embedding test queries...")
    test_data = []
    for tq in test_queries:
        query_emb = embedder.encode(tq['query'])
        test_data.append({
            'query': tq['query'],
            'query_emb': query_emb,
            'expected_answer_id': tq['expected_answer_id'],
            'expected_topic': tq['expected_topic']
        })

    # Train Multi-Head LDA Baseline
    print("\n" + "-" * 40)
    print("Training Multi-Head LDA Baseline...")
    print("-" * 40)
    baseline = MultiHeadLDABaseline()
    t0 = time.perf_counter()
    baseline.train(embedded_clusters)
    baseline_train_time = time.perf_counter() - t0

    # Train Smoothing Basis
    print("\n" + "-" * 40)
    print("Training Smoothing Basis Projection...")
    print("-" * 40)
    smoothing = SmoothingBasisProjection(
        num_basis=args.num_basis,
        cosine_weight=args.cosine_weight
    )
    t0 = time.perf_counter()
    smoothing.train(
        embedded_clusters,
        num_iterations=args.iterations,
        lr=0.01,
        log_interval=25
    )
    smoothing_train_time = time.perf_counter() - t0

    # Evaluate both methods
    print("\n" + "-" * 40)
    print("Evaluating on test queries...")
    print("-" * 40)

    baseline_results = []
    smoothing_results = []

    for td in test_data:
        # Baseline
        br = evaluate_retrieval(
            baseline, td['query_emb'], answer_embs,
            td['expected_answer_id'], args.temperature
        )
        baseline_results.append(br)

        # Smoothing
        sr = evaluate_retrieval(
            smoothing, td['query_emb'], answer_embs,
            td['expected_answer_id'], args.temperature
        )
        smoothing_results.append(sr)

    # Compute metrics
    baseline_correct = sum(1 for r in baseline_results if r['correct'])
    smoothing_correct = sum(1 for r in smoothing_results if r['correct'])

    baseline_mrr = np.mean([1.0 / r['rank'] for r in baseline_results])
    smoothing_mrr = np.mean([1.0 / r['rank'] for r in smoothing_results])

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nTest queries: {len(test_data)}")
    print(f"Max questions per cluster: {args.max_questions}")
    print()
    print(f"{'Method':<25} {'Recall@1':>10} {'MRR':>10} {'Train Time':>12}")
    print("-" * 59)
    print(f"{'Multi-Head LDA':<25} {baseline_correct/len(test_data)*100:>9.1f}% {baseline_mrr:>10.3f} {baseline_train_time*1000:>10.2f}ms")
    print(f"{'Smoothing Basis':<25} {smoothing_correct/len(test_data)*100:>9.1f}% {smoothing_mrr:>10.3f} {smoothing_train_time*1000:>10.2f}ms")
    print()

    # Improvement
    if smoothing_correct > baseline_correct:
        improvement = (smoothing_correct - baseline_correct) / len(test_data) * 100
        print(f"Smoothing Basis improves Recall@1 by +{improvement:.1f}%")
    elif smoothing_correct < baseline_correct:
        decline = (baseline_correct - smoothing_correct) / len(test_data) * 100
        print(f"Multi-Head LDA better by +{decline:.1f}%")
    else:
        print("Both methods perform equally")

    # Show some examples
    print("\n" + "-" * 40)
    print("Sample Results:")
    print("-" * 40)
    for i, (td, br, sr) in enumerate(zip(test_data[:5], baseline_results[:5], smoothing_results[:5])):
        print(f"\nQuery: {td['query'][:50]}...")
        print(f"  Expected: {td['expected_answer_id']}")
        print(f"  Baseline: {br['top_answer_id']} ({'✓' if br['correct'] else '✗'})")
        print(f"  Smoothing: {sr['top_answer_id']} ({'✓' if sr['correct'] else '✗'})")

    print("\n" + "=" * 60)
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
