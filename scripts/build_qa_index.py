#!/usr/bin/env python3
"""
Build Q/A retrieval index with nomic embeddings.

Creates embeddings for question-answer pairs to enable semantic search.

Usage:
    python3 scripts/build_qa_index.py --input training-data/by-topic/skills-qa/tailored/skills_qa.jsonl
    python3 scripts/build_qa_index.py --input datasets/skills_qa/tailored_scored/skills_qa.jsonl
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

def load_qa_pairs(input_path: Path) -> List[Dict[str, Any]]:
    """Load Q/A pairs from JSONL file."""
    pairs = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def generate_nomic_embeddings(texts: List[str], prefix: str = "search_document: ") -> np.ndarray:
    """Generate embeddings using nomic-embed-text-v1.5."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    # Add prefix for nomic model
    prefixed_texts = [f"{prefix}{t}" for t in texts]

    embeddings = model.encode(prefixed_texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def build_index(input_path: Path, output_dir: Path):
    """Build Q/A retrieval index."""

    print(f"Loading Q/A pairs from {input_path}...")
    pairs = load_qa_pairs(input_path)
    print(f"Loaded {len(pairs)} pairs")

    # Extract questions and answers
    questions = [p.get('question', '') for p in pairs]
    answers = [p.get('answer', '') for p in pairs]

    print(f"\nGenerating question embeddings (nomic-embed-text-v1.5)...")
    q_embeddings = generate_nomic_embeddings(questions, prefix="search_query: ")

    print(f"\nGenerating answer embeddings...")
    a_embeddings = generate_nomic_embeddings(answers, prefix="search_document: ")

    # Save embeddings
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "qa_embeddings_nomic.npz"
    np.savez_compressed(
        embeddings_path,
        q_embeddings=q_embeddings,
        a_embeddings=a_embeddings
    )
    print(f"\nSaved embeddings: {embeddings_path}")

    # Save metadata
    metadata = []
    for i, p in enumerate(pairs):
        metadata.append({
            "idx": i,
            "pair_id": p.get("pair_id", ""),
            "cluster_id": p.get("cluster_id", ""),
            "question": p.get("question", ""),
            "answer": p.get("answer", "")[:500],  # Truncate for metadata
            "confidence": p.get("confidence", 0.0),
            "related_skills": p.get("related_skills", []),
            "topics": p.get("topics", [])
        })

    metadata_path = output_dir / "qa_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "count": len(pairs),
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "pairs": metadata
        }, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    # Also save full Q/A for retrieval
    qa_path = output_dir / "qa_full.jsonl"
    with open(qa_path, 'w') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')
    print(f"Saved full Q/A: {qa_path}")

    print(f"\n=== Index Complete ===")
    print(f"Q/A pairs: {len(pairs)}")
    print(f"Embedding dim: {q_embeddings.shape[1]}")
    print(f"Output dir: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build Q/A retrieval index with nomic embeddings")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with Q/A pairs")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")

    args = parser.parse_args()

    output_dir = args.output or args.input.parent

    build_index(args.input, output_dir)


if __name__ == "__main__":
    main()
