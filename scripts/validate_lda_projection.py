#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Validation script for LDA semantic projection

"""
Validate LDA projection with novel queries.

Tests:
1. Training queries still work (sanity check)
2. Novel queries not seen in training (generalization)
3. Comparison: projected vs. direct cosine similarity
"""

import sys
from pathlib import Path
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from projection import LDAProjection

# Novel test queries (not in training data)
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
}

# Answer texts (matching training data)
ANSWER_TEXTS = {
    "csv_data_source": "CSV source: source(csv, users, [csv_file(Path), has_header(true)]). Compile with compile_dynamic_source/3.",
    "mutual_recursion": "Mutual recursion: is_even/is_odd predicates compiled together. UnifyWeaver generates scripts that call each other.",
    "xml_python_source": "Python XML source: source(python, pred, [python_inline(Code)]). Use xml.etree.ElementTree for parsing.",
    "component_registry": "Component registry: declare_component(Category, Name, Type, Config) to register, invoke_component/4 to call.",
    "lda_projection": "LDA projection: W = A * Q_bar^T * inv(covariance). Maps query embeddings to answer space for better retrieval.",
}


def main():
    print("=" * 60)
    print("LDA Projection Validation")
    print("=" * 60)

    # Load model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed")
        return 1

    model_name = "all-MiniLM-L6-v2"
    W_path = "playbooks/lda-training-data/trained/all-MiniLM-L6-v2/W_matrix.npy"

    print(f"\nLoading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    print(f"Loading W matrix: {W_path}")
    projection = LDAProjection(W_path, embedding_dim=384)

    # Embed all answers
    print("\nEmbedding answer documents...")
    answer_ids = list(ANSWER_TEXTS.keys())
    answer_texts = [ANSWER_TEXTS[aid] for aid in answer_ids]
    answer_embs = embedder.encode(answer_texts, convert_to_numpy=True)

    # Normalize answers
    answer_norms = np.linalg.norm(answer_embs, axis=1, keepdims=True)
    answer_embs_normed = answer_embs / answer_norms

    print(f"  {len(answer_ids)} answer documents embedded")

    # Test novel queries
    print("\n" + "-" * 60)
    print("Testing Novel Queries (not in training data)")
    print("-" * 60)

    total = 0
    correct_projected = 0
    correct_direct = 0
    mrr_projected = []
    mrr_direct = []

    for target_id, queries in NOVEL_QUERIES.items():
        target_idx = answer_ids.index(target_id)
        print(f"\nTarget: {target_id}")

        for query in queries:
            total += 1

            # Embed query
            query_emb = embedder.encode(query, convert_to_numpy=True)

            # --- Projected similarity ---
            projected = projection.project(query_emb)
            proj_norm = np.linalg.norm(projected)
            if proj_norm > 0:
                projected_normed = projected / proj_norm
            else:
                projected_normed = projected

            proj_sims = projected_normed @ answer_embs_normed.T
            proj_ranking = np.argsort(-proj_sims)
            proj_rank = np.where(proj_ranking == target_idx)[0][0] + 1

            if proj_rank == 1:
                correct_projected += 1
            mrr_projected.append(1.0 / proj_rank)

            # --- Direct similarity ---
            query_norm = np.linalg.norm(query_emb)
            if query_norm > 0:
                query_normed = query_emb / query_norm
            else:
                query_normed = query_emb

            direct_sims = query_normed @ answer_embs_normed.T
            direct_ranking = np.argsort(-direct_sims)
            direct_rank = np.where(direct_ranking == target_idx)[0][0] + 1

            if direct_rank == 1:
                correct_direct += 1
            mrr_direct.append(1.0 / direct_rank)

            # Display
            status_proj = "✓" if proj_rank == 1 else f"rank={proj_rank}"
            status_dir = "✓" if direct_rank == 1 else f"rank={direct_rank}"
            print(f"  [{status_proj}|{status_dir}] \"{query[:50]}...\"" if len(query) > 50 else f"  [{status_proj}|{status_dir}] \"{query}\"")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    recall_proj = correct_projected / total
    recall_dir = correct_direct / total
    avg_mrr_proj = np.mean(mrr_projected)
    avg_mrr_dir = np.mean(mrr_direct)

    print(f"\nNovel queries tested: {total}")
    print(f"\n{'Metric':<20} {'Projected':>12} {'Direct':>12} {'Improvement':>12}")
    print("-" * 56)
    print(f"{'Recall@1':<20} {recall_proj:>12.2%} {recall_dir:>12.2%} {(recall_proj - recall_dir):>+12.2%}")
    print(f"{'MRR':<20} {avg_mrr_proj:>12.4f} {avg_mrr_dir:>12.4f} {(avg_mrr_proj - avg_mrr_dir):>+12.4f}")

    if recall_proj > recall_dir:
        print("\n✓ Projection improves retrieval on novel queries!")
    elif recall_proj == recall_dir:
        print("\n~ Projection has same performance as direct similarity")
    else:
        print("\n✗ Direct similarity outperforms projection")

    return 0


if __name__ == "__main__":
    sys.exit(main())
