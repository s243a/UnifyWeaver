#!/usr/bin/env python3
"""
Precompute semantic edge weights for the min_semantic_distance benchmark.

Reads category_parent.tsv and computes:
    weight(From, To) = 1 - cosine_similarity(embedding(From), embedding(To))

Outputs a Prolog facts file with edge_weight/3 clauses, plus a TSV file
for non-Prolog targets.

Usage:
    python precompute_edge_weights.py <category_parent.tsv> <output_dir> [--model MODEL]

Requirements:
    pip install sentence-transformers numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edges_tsv", type=Path, help="category_parent.tsv file")
    parser.add_argument("output_dir", type=Path, help="Output directory for weight files")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding batch size (default: 256)",
    )
    return parser.parse_args()


def load_edges(path: Path) -> list[tuple[str, str]]:
    edges = []
    with open(path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
    return edges


def compute_weights(
    edges: list[tuple[str, str]], model_name: str, batch_size: int
) -> list[tuple[str, str, float]]:
    from sentence_transformers import SentenceTransformer

    # Collect unique nodes
    nodes = sorted(set(n for edge in edges for n in edge))
    print(f"  {len(nodes)} unique nodes, {len(edges)} edges", file=sys.stderr)

    # Compute embeddings
    print(f"  Encoding with {model_name}...", file=sys.stderr)
    model = SentenceTransformer(model_name)
    # Replace underscores with spaces for better semantic representation
    node_texts = [n.replace("_", " ") for n in nodes]
    embeddings = model.encode(node_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    emb_map = {node: emb for node, emb in zip(nodes, embeddings)}

    # Compute weights
    print("  Computing cosine distances...", file=sys.stderr)
    weighted_edges = []
    for child, parent in edges:
        emb_a = emb_map[child]
        emb_b = emb_map[parent]
        dot = np.dot(emb_a, emb_b)
        norm = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        sim = float(dot / norm) if norm > 0 else 0.0
        weight = max(0.0, 1.0 - sim)  # clamp to [0, ∞)
        weighted_edges.append((child, parent, round(weight, 6)))

    return weighted_edges


def write_prolog(weighted_edges: list[tuple[str, str, float]], path: Path) -> None:
    with open(path, "w") as f:
        f.write("%% Precomputed semantic edge weights\n")
        f.write("%% weight = 1 - cosine_similarity(embedding(from), embedding(to))\n\n")
        for child, parent, weight in weighted_edges:
            # Escape single quotes in Prolog atoms
            c = child.replace("'", "\\'")
            p = parent.replace("'", "\\'")
            f.write(f"edge_weight('{c}', '{p}', {weight}).\n")
    print(f"  Wrote {len(weighted_edges)} edge_weight/3 facts to {path}", file=sys.stderr)


def write_tsv(weighted_edges: list[tuple[str, str, float]], path: Path) -> None:
    with open(path, "w") as f:
        f.write("child\tparent\tweight\n")
        for child, parent, weight in weighted_edges:
            f.write(f"{child}\t{parent}\t{weight}\n")
    print(f"  Wrote {len(weighted_edges)} rows to {path}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    edges = load_edges(args.edges_tsv)
    if not edges:
        print("No edges found", file=sys.stderr)
        return 1

    print(f"Precomputing semantic edge weights...", file=sys.stderr)
    weighted_edges = compute_weights(edges, args.model, args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_prolog(weighted_edges, args.output_dir / "edge_weights.pl")
    write_tsv(weighted_edges, args.output_dir / "edge_weights.tsv")

    # Stats
    weights = [w for _, _, w in weighted_edges]
    print(f"\n  Weight stats: min={min(weights):.4f} max={max(weights):.4f} "
          f"mean={sum(weights)/len(weights):.4f} median={sorted(weights)[len(weights)//2]:.4f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
