#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Training script for LDA semantic projection

"""
Train LDA projection matrix W from Q-A pairs.

Usage:
    python scripts/train_lda_projection.py \
        --input datasets/lda_training/raw/qa_pairs_v1.json \
        --model all-MiniLM-L6-v2 \
        --output datasets/lda_training/all-MiniLM-L6-v2/W_matrix.npy

Requirements:
    pip install sentence-transformers numpy
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from projection import compute_W, save_W, compute_weighted_centroid


def load_qa_pairs(filepath: str) -> Dict[str, Any]:
    """Load Q-A pairs from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def get_answer_text(cluster: Dict, model_name: str) -> str:
    """Get the appropriate answer text for the given model."""
    answers = cluster.get("answers", {})

    # Try model-specific answer first
    if model_name in answers:
        return answers[model_name]

    # Fall back to default
    if "default" in answers:
        return answers["default"]

    # Legacy format: single answer_text field
    if "answer_text" in cluster:
        return cluster["answer_text"]

    raise ValueError(f"No answer text found for cluster {cluster.get('id')}")


def get_all_queries(cluster: Dict) -> List[str]:
    """Flatten all query variants into a single list."""
    queries = cluster.get("queries", {})
    all_queries = []
    for length_type in ["short", "medium", "long"]:
        if length_type in queries:
            all_queries.extend(queries[length_type])
    return all_queries


class SentenceTransformersEmbedder:
    """Embedder using sentence-transformers library."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.dim}")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        return self.model.encode(texts, convert_to_numpy=True)


class OnnxEmbedder:
    """Embedder using ONNX runtime (for local models)."""

    def __init__(self, model_path: str, vocab_path: str):
        from onnx_embedding import OnnxEmbeddingProvider
        self.provider = OnnxEmbeddingProvider(model_path, vocab_path)
        self.dim = 384  # Default for MiniLM

    def embed(self, text: str) -> np.ndarray:
        return np.array(self.provider.get_embedding(text))

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


def create_embedder(model_name: str, model_path: Optional[str] = None):
    """Create an embedder based on model name or path."""
    if model_path and os.path.exists(model_path):
        # Local ONNX model
        vocab_path = os.path.join(os.path.dirname(model_path), "vocab.txt")
        return OnnxEmbedder(model_path, vocab_path)
    else:
        # HuggingFace model via sentence-transformers
        return SentenceTransformersEmbedder(model_name)


def build_clusters(
    qa_data: Dict,
    embedder: Any,
    model_name: str
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build embedding clusters from Q-A pairs."""
    clusters = []

    for cluster_data in qa_data["clusters"]:
        cluster_id = cluster_data.get("id", "unknown")
        print(f"  Processing cluster: {cluster_id}")

        # Get answer text for this model
        answer_text = get_answer_text(cluster_data, model_name)

        # Get all queries
        queries = get_all_queries(cluster_data)

        if not queries:
            print(f"    Warning: No queries found, skipping")
            continue

        # Embed
        answer_emb = embedder.embed(answer_text)
        query_embs = embedder.embed_batch(queries)

        print(f"    Answer: {len(answer_text)} chars")
        print(f"    Queries: {len(queries)}")

        clusters.append((answer_emb, query_embs))

    return clusters


def evaluate_projection(
    W: np.ndarray,
    clusters: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, float]:
    """Evaluate projection quality."""

    total_queries = 0
    correct_at_1 = 0
    reciprocal_ranks = []

    # For each cluster, check if projected queries are closest to their answer
    all_answers = np.array([c[0] for c in clusters])

    for i, (answer, queries) in enumerate(clusters):
        # Project queries
        projected = queries @ W.T

        # Normalize
        proj_norms = np.linalg.norm(projected, axis=1, keepdims=True)
        proj_norms = np.where(proj_norms == 0, 1, proj_norms)
        projected_normed = projected / proj_norms

        ans_norms = np.linalg.norm(all_answers, axis=1, keepdims=True)
        ans_norms = np.where(ans_norms == 0, 1, ans_norms)
        answers_normed = all_answers / ans_norms

        # Compute similarities to all answers
        for q_idx in range(len(queries)):
            sims = projected_normed[q_idx] @ answers_normed.T
            ranking = np.argsort(-sims)

            rank = np.where(ranking == i)[0][0] + 1  # 1-indexed

            if rank == 1:
                correct_at_1 += 1
            reciprocal_ranks.append(1.0 / rank)
            total_queries += 1

    return {
        "recall_at_1": correct_at_1 / total_queries if total_queries > 0 else 0,
        "mrr": np.mean(reciprocal_ranks) if reciprocal_ranks else 0,
        "total_queries": total_queries,
        "num_clusters": len(clusters)
    }


def cross_validate_lambda(
    clusters: List[Tuple[np.ndarray, np.ndarray]],
    lambda_values: List[float] = [0.01, 0.1, 1.0, 10.0],
    ridge: float = 1e-6
) -> Tuple[float, np.ndarray]:
    """Find best lambda via simple evaluation (no held-out set for small data)."""

    best_lambda = 1.0
    best_score = -1
    best_W = None

    print("\nCross-validating lambda:")
    for lam in lambda_values:
        W = compute_W(clusters, lambda_reg=lam, ridge=ridge)
        metrics = evaluate_projection(W, clusters)
        score = metrics["mrr"]

        print(f"  lambda={lam}: MRR={score:.4f}, R@1={metrics['recall_at_1']:.4f}")

        if score > best_score:
            best_score = score
            best_lambda = lam
            best_W = W

    print(f"\nBest lambda: {best_lambda} (MRR={best_score:.4f})")
    return best_lambda, best_W


def main():
    parser = argparse.ArgumentParser(description="Train LDA projection matrix")
    parser.add_argument("--input", required=True, help="Path to Q-A pairs JSON")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                       help="Model name (HuggingFace) or path to ONNX model")
    parser.add_argument("--model-path", help="Path to local ONNX model (optional)")
    parser.add_argument("--output", required=True, help="Output path for W matrix")
    parser.add_argument("--lambda-reg", type=float, default=None,
                       help="Lambda regularization (if not set, cross-validates)")
    parser.add_argument("--ridge", type=float, default=1e-6,
                       help="Ridge regularization for numerical stability")
    parser.add_argument("--save-clusters", action="store_true",
                       help="Also save embedded clusters as .npz")

    args = parser.parse_args()

    # Load data
    print(f"Loading Q-A pairs from: {args.input}")
    qa_data = load_qa_pairs(args.input)
    print(f"  Clusters: {len(qa_data['clusters'])}")

    # Create embedder
    embedder = create_embedder(args.model, args.model_path)

    # Build clusters
    print(f"\nEmbedding with model: {args.model}")
    clusters = build_clusters(qa_data, embedder, args.model)
    print(f"\nBuilt {len(clusters)} clusters")

    if len(clusters) < 2:
        print("Warning: Very few clusters. Results may be unstable.")

    # Optionally save embedded clusters
    if args.save_clusters:
        cluster_path = args.output.replace(".npy", "_clusters.npz")
        np.savez(cluster_path,
                 answers=[c[0] for c in clusters],
                 questions=[c[1] for c in clusters])
        print(f"Saved clusters to: {cluster_path}")

    # Compute W
    if args.lambda_reg is not None:
        print(f"\nComputing W with lambda={args.lambda_reg}")
        W = compute_W(clusters, lambda_reg=args.lambda_reg, ridge=args.ridge)
        metrics = evaluate_projection(W, clusters)
        print(f"  MRR={metrics['mrr']:.4f}, R@1={metrics['recall_at_1']:.4f}")
    else:
        best_lambda, W = cross_validate_lambda(clusters, ridge=args.ridge)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_W(W, args.output)
    print(f"\nSaved W matrix to: {args.output}")
    print(f"  Shape: {W.shape}")

    # Final evaluation
    print("\nFinal evaluation:")
    metrics = evaluate_projection(W, clusters)
    print(f"  Recall@1: {metrics['recall_at_1']:.4f}")
    print(f"  MRR: {metrics['mrr']:.4f}")
    print(f"  Total queries: {metrics['total_queries']}")


if __name__ == "__main__":
    main()
