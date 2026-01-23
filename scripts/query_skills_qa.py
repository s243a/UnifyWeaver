#!/usr/bin/env python3
"""
Query the Skills Q/A Retrieval Model

Interactive Q/A interface that finds the most relevant answers based on
semantic similarity to your question.

Usage:
    python3 scripts/query_skills_qa.py "How do I compile Prolog to bash?"
    python3 scripts/query_skills_qa.py --interactive
    python3 scripts/query_skills_qa.py --top-k 5 "deployment options"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

project_root = Path(__file__).parent.parent


def load_qa_data(dataset_dir: Path) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Load Q/A pairs and embeddings."""
    # Load JSONL data
    qa_pairs = []
    qa_file = dataset_dir / "skills_qa.jsonl"
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))

    # Load embeddings
    embeddings_file = dataset_dir / "skills_embeddings_all-minilm.npz"
    data = np.load(embeddings_file)
    q_embeddings = data['q_embeddings']
    a_embeddings = data['a_embeddings']

    return qa_pairs, q_embeddings, a_embeddings


def embed_query(query: str, model) -> np.ndarray:
    """Embed a query using the model."""
    return model.encode([query], convert_to_numpy=True)[0]


def find_similar(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """Find most similar embeddings using cosine similarity."""
    # Normalize for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute similarities
    similarities = np.dot(embeddings_norm, query_norm)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(int(idx), float(similarities[idx])) for idx in top_indices]

    return results


def deduplicate_by_cluster(
    results: List[Tuple[int, float]],
    qa_pairs: List[Dict]
) -> List[Tuple[int, float]]:
    """Remove duplicate answers (same cluster_id)."""
    seen_clusters = set()
    unique_results = []

    for idx, score in results:
        cluster_id = qa_pairs[idx]['cluster_id']
        if cluster_id not in seen_clusters:
            seen_clusters.add(cluster_id)
            unique_results.append((idx, score))

    return unique_results


def format_answer(qa_pair: Dict, score: float, show_metadata: bool = True) -> str:
    """Format an answer for display."""
    lines = []
    lines.append(f"Score: {score:.3f}")

    if show_metadata:
        lines.append(f"Skill: {qa_pair.get('cluster_id', 'unknown')}")
        topics = qa_pair.get('topics', [])
        if topics:
            lines.append(f"Topics: {' > '.join(topics)}")

        related = qa_pair.get('related_skills', [])
        if related:
            lines.append(f"Related Skills: {', '.join(related)}")

    lines.append("")
    lines.append(f"Q: {qa_pair['question']}")
    lines.append("")
    lines.append(qa_pair['answer'])

    return '\n'.join(lines)


def query_qa(
    query: str,
    qa_pairs: List[Dict],
    q_embeddings: np.ndarray,
    model,
    top_k: int = 3,
    dedupe: bool = True
) -> List[Dict]:
    """Query the Q/A model and return results."""
    # Embed query
    query_emb = embed_query(query, model)

    # Find similar questions
    results = find_similar(query_emb, q_embeddings, top_k=top_k * 3 if dedupe else top_k)

    # Deduplicate by cluster
    if dedupe:
        results = deduplicate_by_cluster(results, qa_pairs)[:top_k]

    # Format results
    formatted = []
    for idx, score in results:
        formatted.append({
            'qa_pair': qa_pairs[idx],
            'score': score,
            'index': idx
        })

    return formatted


def interactive_mode(qa_pairs, q_embeddings, model, top_k: int = 3):
    """Interactive Q/A loop."""
    print("Skills Q/A Retrieval - Interactive Mode")
    print("=" * 50)
    print(f"Loaded {len(qa_pairs)} Q/A pairs")
    print("Type your question, or 'quit' to exit.\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        results = query_qa(query, qa_pairs, q_embeddings, model, top_k=top_k)

        print("\n" + "=" * 50)
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(format_answer(result['qa_pair'], result['score']))
            print()
        print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Query the Skills Q/A retrieval model"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="The question to ask (omit for interactive mode)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Number of results to return"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "datasets" / "skills_qa",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Don't deduplicate by answer cluster"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    # Load model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers required. Install with: pip install sentence-transformers")
        return 1

    print("Loading model...", file=sys.stderr)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load data
    print("Loading Q/A data...", file=sys.stderr)
    qa_pairs, q_embeddings, a_embeddings = load_qa_data(args.dataset)
    print(f"Loaded {len(qa_pairs)} Q/A pairs", file=sys.stderr)

    if args.interactive or not args.query:
        interactive_mode(qa_pairs, q_embeddings, model, args.top_k)
    else:
        results = query_qa(
            args.query,
            qa_pairs,
            q_embeddings,
            model,
            top_k=args.top_k,
            dedupe=not args.no_dedupe
        )

        if args.json:
            output = []
            for r in results:
                output.append({
                    'score': r['score'],
                    'cluster_id': r['qa_pair']['cluster_id'],
                    'question': r['qa_pair']['question'],
                    'answer': r['qa_pair']['answer'],
                    'topics': r['qa_pair'].get('topics', []),
                    'related_skills': r['qa_pair'].get('related_skills', [])
                })
            print(json.dumps(output, indent=2))
        else:
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(format_answer(result['qa_pair'], result['score']))
                print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
