#!/usr/bin/env python3
"""
Query the federated Q/A retrieval system.

Given a natural language question, finds the most relevant Q/A pairs
from the skills knowledge base using semantic search with federated projection.

Usage:
    # Single query
    python3 scripts/infer_qa_federated.py \
      --model models/skills_qa_federated.pkl \
      --query "how do I create an HTTP server?"

    # Interactive mode
    python3 scripts/infer_qa_federated.py \
      --model models/skills_qa_federated.pkl \
      --interactive

    # Batch mode
    python3 scripts/infer_qa_federated.py \
      --model models/skills_qa_federated.pkl \
      --input queries.txt \
      --output results.jsonl
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


def load_model(model_path: Path) -> Dict:
    """Load the federated Q/A model."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data


def load_qa_data(qa_path: Path) -> List[Dict]:
    """Load the full Q/A data for returning answers."""
    qa_data = []
    with open(qa_path) as f:
        for line in f:
            if line.strip():
                qa_data.append(json.loads(line))
    return qa_data


def encode_query(query: str, embedder) -> np.ndarray:
    """Encode a query using the embedding model."""
    # Use search_query prefix for nomic model
    text = f"search_query: {query}"
    embedding = embedder.encode([text], convert_to_numpy=True)
    return embedding[0]


def project_query(q: np.ndarray, model_data: Dict, top_k_clusters: int = 3) -> np.ndarray:
    """Project query using federated model."""
    q = q.flatten()

    cluster_centroids = model_data["cluster_centroids"]
    temperature = model_data.get("temperature", 0.1)
    clusters = model_data["clusters"]

    # Compute similarity to cluster centroids
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    centroid_norms = cluster_centroids / (np.linalg.norm(cluster_centroids, axis=1, keepdims=True) + 1e-8)

    similarities = q_norm @ centroid_norms.T

    # Get top-k clusters
    top_indices = np.argsort(similarities)[::-1][:top_k_clusters]
    top_sims = similarities[top_indices]

    # Softmax over top-k
    top_sims_shifted = (top_sims - np.max(top_sims)) / temperature
    weights = np.exp(top_sims_shifted)
    weights /= weights.sum()

    # Weighted projection
    dim = model_data["embedding_dim"]
    projected = np.zeros(dim)

    for idx, weight in zip(top_indices, weights):
        cluster = clusters[idx]
        W = cluster["W"]
        projected += weight * (q @ W)

    return projected


def search(
    query_emb: np.ndarray,
    model_data: Dict,
    embeddings_path: Path,
    top_k: int = 5,
) -> List[Dict]:
    """
    Search for similar Q/A pairs.

    Returns list of (index, score) tuples.
    """
    # Project the query
    projected = project_query(query_emb, model_data)

    # Load answer embeddings
    emb_data = np.load(embeddings_path)
    A_emb = emb_data["a_embeddings"]

    # Normalize
    proj_norm = projected / (np.linalg.norm(projected) + 1e-8)
    A_norm = A_emb / (np.linalg.norm(A_emb, axis=1, keepdims=True) + 1e-8)

    # Compute similarities
    scores = A_norm @ proj_norm

    # Get top-k
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "index": int(idx),
            "score": float(scores[idx])
        })

    return results


def format_result(result: Dict, qa_data: List[Dict]) -> str:
    """Format a search result for display."""
    idx = result["index"]
    score = result["score"]
    pair = qa_data[idx]

    question = pair.get("question", "")
    answer = pair.get("answer", "")
    skill = pair.get("skill", pair.get("source_skill", "unknown"))
    confidence = pair.get("confidence", 0.0)

    # Truncate long answers
    if len(answer) > 500:
        answer = answer[:500] + "..."

    return f"""
[Score: {score:.4f}] {skill}
Q: {question}
A: {answer}
(confidence: {confidence:.2f})
"""


def interactive_mode(model_data: Dict, embeddings_path: Path, qa_data: List[Dict], embedder):
    """Run interactive REPL for querying."""
    print(f"\nLoaded model with {len(model_data['clusters'])} clusters")
    print("Enter a question to search, or :quit to exit\n")

    top_k = 5

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query == ":quit" or query == ":q":
            print("Goodbye!")
            break

        if query.startswith(":top "):
            try:
                top_k = int(query.split()[1])
                print(f"Set top_k to {top_k}")
            except (IndexError, ValueError):
                print("Usage: :top N")
            continue

        if query == ":help":
            print("""
Commands:
  <query>      Search for Q/A pairs
  :top N       Set number of results (default: 5)
  :quit        Exit
  :help        Show this help
""")
            continue

        # Search
        query_emb = encode_query(query, embedder)
        results = search(query_emb, model_data, embeddings_path, top_k=top_k)

        print(f"\n--- Top {len(results)} Results ---")
        for result in results:
            print(format_result(result, qa_data))


def main():
    parser = argparse.ArgumentParser(description="Query federated Q/A retrieval system")
    parser.add_argument("--model", type=Path, required=True,
                       help="Path to federated model (.pkl)")
    parser.add_argument("--embeddings", type=Path, default=None,
                       help="Path to embeddings (.npz). Auto-detected if not specified.")
    parser.add_argument("--qa-data", type=Path, default=None,
                       help="Path to Q/A JSONL. Auto-detected if not specified.")
    parser.add_argument("--query", type=str, default=None,
                       help="Query text for single search")
    parser.add_argument("--input", type=Path, default=None,
                       help="Input file with queries (one per line)")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output file for results (JSONL)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive REPL")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of results to return")
    parser.add_argument("--embedding-model", type=str,
                       default="nomic-ai/nomic-embed-text-v1.5",
                       help="Embedding model to use")

    args = parser.parse_args()

    # Auto-detect paths
    if args.embeddings is None:
        # Look for embeddings in common locations
        for candidate in [
            Path("datasets/skills_qa/qa_embeddings_nomic.npz"),
            args.model.parent / "qa_embeddings_nomic.npz",
        ]:
            if candidate.exists():
                args.embeddings = candidate
                break
        if args.embeddings is None:
            print("Error: Could not find embeddings file. Specify with --embeddings")
            return

    if args.qa_data is None:
        for candidate in [
            Path("datasets/skills_qa/tailored_scored/skills_qa.jsonl"),
            Path("datasets/skills_qa/tailored/skills_qa.jsonl"),
        ]:
            if candidate.exists():
                args.qa_data = candidate
                break
        if args.qa_data is None:
            print("Error: Could not find Q/A data file. Specify with --qa-data")
            return

    print(f"Loading model from {args.model}...")
    model_data = load_model(args.model)

    print(f"Loading Q/A data from {args.qa_data}...")
    qa_data = load_qa_data(args.qa_data)

    print(f"Loading embedding model: {args.embedding_model}...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(args.embedding_model, trust_remote_code=True)

    if args.interactive:
        interactive_mode(model_data, args.embeddings, qa_data, embedder)
    elif args.query:
        # Single query
        query_emb = encode_query(args.query, embedder)
        results = search(query_emb, model_data, args.embeddings, top_k=args.top_k)

        print(f"\n--- Query: {args.query} ---")
        print(f"--- Top {len(results)} Results ---")
        for result in results:
            print(format_result(result, qa_data))
    elif args.input:
        # Batch mode
        with open(args.input) as f:
            queries = [line.strip() for line in f if line.strip()]

        all_results = []
        for query in queries:
            query_emb = encode_query(query, embedder)
            results = search(query_emb, model_data, args.embeddings, top_k=args.top_k)

            result_entry = {
                "query": query,
                "results": []
            }
            for r in results:
                idx = r["index"]
                pair = qa_data[idx]
                result_entry["results"].append({
                    "score": r["score"],
                    "question": pair.get("question", ""),
                    "answer": pair.get("answer", ""),
                    "skill": pair.get("skill", pair.get("source_skill", "")),
                })
            all_results.append(result_entry)

        if args.output:
            with open(args.output, "w") as f:
                for entry in all_results:
                    f.write(json.dumps(entry) + "\n")
            print(f"Saved {len(all_results)} results to {args.output}")
        else:
            for entry in all_results:
                print(json.dumps(entry, indent=2))
    else:
        print("Specify --query, --input, or --interactive")
        return


if __name__ == "__main__":
    main()
