#!/usr/bin/env python3
"""
Suggest best folder for a mindmap based on semantic similarity.

Uses a two-stage approach:
1. Top-k centroid filtering (cheap): Find k nearest folder centroids
2. Procrustes fit scoring (precise): Evaluate fit using folder's W matrix

The final suggestion uses softmax over Procrustes fit scores.

Usage:
    # Suggest folder for a mindmap by tree ID
    python3 scripts/mindmap/suggest_folder.py --tree-id 12345678

    # Suggest folder for a mindmap by title
    python3 scripts/mindmap/suggest_folder.py --title "Machine Learning Tutorial"

    # Suggest folder with more candidates
    python3 scripts/mindmap/suggest_folder.py --tree-id 12345678 --top-k 10

    # Show detailed scores
    python3 scripts/mindmap/suggest_folder.py --tree-id 12345678 --verbose

    # Batch mode: check all mindmaps in a folder
    python3 scripts/mindmap/suggest_folder.py --check-folder output/mindmaps_curated/Economics/
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_centroids(conn: sqlite3.Connection) -> Tuple[List[str], np.ndarray]:
    """Load all folder centroids for top-k filtering."""
    cursor = conn.execute(
        "SELECT folder_path, centroid FROM folder_projections ORDER BY folder_path"
    )
    rows = cursor.fetchall()
    if not rows:
        return [], np.array([])
    folders = [r[0] for r in rows]
    centroids = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return folders, centroids


def load_w_matrix(conn: sqlite3.Connection, folder: str) -> Optional[np.ndarray]:
    """Load W matrix for a specific folder."""
    cursor = conn.execute(
        "SELECT w_matrix FROM folder_projections WHERE folder_path = ?",
        (folder,)
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32).reshape(768, 768)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and each row in matrix."""
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norms @ query_norm


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax with temperature."""
    x_scaled = x / temperature
    x_shifted = x_scaled - x_scaled.max()  # Numerical stability
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum()


def get_embedding_for_tree(
    tree_id: str,
    embeddings_data: dict
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Get input and output embeddings for a tree ID."""
    tree_ids = embeddings_data['tree_ids']
    idx = None
    for i, tid in enumerate(tree_ids):
        if str(tid) == str(tree_id):
            idx = i
            break
    if idx is None:
        return None
    return (
        embeddings_data['input_nomic'][idx],
        embeddings_data['output_nomic'][idx]
    )


def embed_title(title: str) -> Tuple[np.ndarray, np.ndarray]:
    """Embed a title using Nomic model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    embedding = model.encode(title, convert_to_numpy=True)

    # For a new title, input and output embeddings are the same
    # (no hierarchy information yet)
    return embedding, embedding


def suggest_folder(
    input_emb: np.ndarray,
    output_emb: np.ndarray,
    conn: sqlite3.Connection,
    top_k: int = 5,
    temperature: float = 0.1,
    verbose: bool = False
) -> List[Tuple[str, float, float]]:
    """
    Suggest best folder for a mindmap.

    Args:
        input_emb: Input embedding (768-dim)
        output_emb: Output embedding (768-dim)
        conn: Database connection
        top_k: Number of candidates from centroid filtering
        temperature: Softmax temperature (lower = sharper)
        verbose: Print debug info

    Returns:
        List of (folder_path, probability, fit_score) tuples, sorted by probability
    """
    # Stage 1: Top-k centroid filtering
    folders, centroids = load_centroids(conn)
    if len(folders) == 0:
        return []

    centroid_sims = cosine_similarity_batch(input_emb, centroids)
    top_k_indices = np.argsort(centroid_sims)[-top_k:][::-1]

    if verbose:
        print(f"Stage 1: Top-{top_k} by centroid similarity:")
        for idx in top_k_indices:
            print(f"  {folders[idx]}: {centroid_sims[idx]:.4f}")
        print()

    # Stage 2: Procrustes fit scoring
    fit_scores = []
    for idx in top_k_indices:
        folder = folders[idx]
        W = load_w_matrix(conn, folder)
        if W is None:
            fit_scores.append(-1.0)
            continue

        # Project input through folder's W matrix
        projected = input_emb @ W

        # Fit score is cosine similarity between projected and output
        fit = cosine_similarity(projected, output_emb)
        fit_scores.append(fit)

    fit_scores = np.array(fit_scores)

    if verbose:
        print(f"Stage 2: Procrustes fit scores:")
        for i, idx in enumerate(top_k_indices):
            print(f"  {folders[idx]}: {fit_scores[i]:.4f}")
        print()

    # Softmax over fit scores
    # Handle any negative scores by shifting
    valid_mask = fit_scores >= 0
    if not valid_mask.any():
        return []

    valid_scores = fit_scores[valid_mask]
    valid_indices = top_k_indices[valid_mask]

    probs = softmax(valid_scores, temperature)

    if verbose:
        print(f"Softmax probabilities (temperature={temperature}):")
        for i, idx in enumerate(valid_indices):
            print(f"  {folders[idx]}: {probs[i]:.4f}")
        print()

    # Build results
    results = []
    for i, idx in enumerate(valid_indices):
        results.append((folders[idx], float(probs[i]), float(valid_scores[i])))

    # Sort by probability (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def get_current_folder(tree_id: str, index: Dict[str, str]) -> Optional[str]:
    """Get the current folder for a tree from the index."""
    if tree_id not in index:
        return None
    rel_path = index[tree_id]
    parts = rel_path.rsplit('/', 1)
    return parts[0] if len(parts) > 1 else ""


def main():
    parser = argparse.ArgumentParser(
        description="Suggest best folder for a mindmap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--tree-id", "-t",
        help="Tree ID to suggest folder for"
    )
    parser.add_argument(
        "--title",
        help="Title to suggest folder for (embeds on the fly)"
    )
    parser.add_argument(
        "--projections-db", "-p",
        type=Path,
        default=Path("output/mindmaps_curated/folder_projections.db"),
        help="Path to folder projections database"
    )
    parser.add_argument(
        "--embeddings", "-e",
        type=Path,
        default=Path("models/dual_embeddings_combined_2026-01-02_trees_only.npz"),
        help="Path to dual embeddings NPZ file"
    )
    parser.add_argument(
        "--index", "-i",
        type=Path,
        default=Path("output/mindmaps_curated/index.json"),
        help="Path to mindmap index JSON file"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of candidates from centroid filtering (default: 5)"
    )
    parser.add_argument(
        "--temperature", "-T",
        type=float,
        default=0.1,
        help="Softmax temperature (default: 0.1, lower = sharper)"
    )
    parser.add_argument(
        "--check-folder",
        type=Path,
        help="Check all mindmaps in a folder for misplacements"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for misplacement warning (default: 0.5)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.tree_id and not args.title and not args.check_folder:
        parser.error("Must specify --tree-id, --title, or --check-folder")

    if not args.projections_db.exists():
        print(f"Error: Projections database not found: {args.projections_db}", file=sys.stderr)
        print("Run build_folder_projections.py first.", file=sys.stderr)
        sys.exit(1)

    # Open database
    conn = sqlite3.connect(args.projections_db)

    # Load embeddings if needed
    embeddings_data = None
    if args.tree_id or args.check_folder:
        if not args.embeddings.exists():
            print(f"Error: Embeddings file not found: {args.embeddings}", file=sys.stderr)
            sys.exit(1)
        embeddings_data = np.load(args.embeddings, allow_pickle=True)

    # Load index if needed
    index = None
    if args.tree_id or args.check_folder:
        if args.index.exists():
            with open(args.index) as f:
                index_data = json.load(f)
            index = index_data['index']

    # Single tree-id mode
    if args.tree_id:
        embs = get_embedding_for_tree(args.tree_id, embeddings_data)
        if embs is None:
            print(f"Error: Tree ID {args.tree_id} not found in embeddings", file=sys.stderr)
            sys.exit(1)

        input_emb, output_emb = embs
        results = suggest_folder(
            input_emb, output_emb, conn,
            top_k=args.top_k,
            temperature=args.temperature,
            verbose=args.verbose
        )

        current_folder = get_current_folder(args.tree_id, index) if index else None

        if args.json:
            output = {
                "tree_id": args.tree_id,
                "current_folder": current_folder,
                "suggestions": [
                    {"folder": f, "probability": p, "fit_score": s}
                    for f, p, s in results
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            if current_folder is not None:
                print(f"Tree ID: {args.tree_id}")
                print(f"Current folder: {current_folder or '(root)'}")
                print()

            print("Suggested folders:")
            for i, (folder, prob, score) in enumerate(results):
                marker = " <-- current" if folder == current_folder else ""
                display = folder if folder else "(root)"
                print(f"  {i+1}. {display}: {prob:.1%} (fit={score:.4f}){marker}")

    # Title mode
    elif args.title:
        print(f"Embedding title: {args.title}")
        input_emb, output_emb = embed_title(args.title)

        results = suggest_folder(
            input_emb, output_emb, conn,
            top_k=args.top_k,
            temperature=args.temperature,
            verbose=args.verbose
        )

        if args.json:
            output = {
                "title": args.title,
                "suggestions": [
                    {"folder": f, "probability": p, "fit_score": s}
                    for f, p, s in results
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            print()
            print("Suggested folders:")
            for i, (folder, prob, score) in enumerate(results):
                display = folder if folder else "(root)"
                print(f"  {i+1}. {display}: {prob:.1%} (fit={score:.4f})")

    # Check folder mode
    elif args.check_folder:
        import re

        folder_path = args.check_folder
        if not folder_path.exists():
            print(f"Error: Folder not found: {folder_path}", file=sys.stderr)
            sys.exit(1)

        # Get relative folder path
        base_dir = Path(index_data.get('base_dir', '')) if index else folder_path.parent
        try:
            rel_folder = str(folder_path.relative_to(base_dir))
        except ValueError:
            rel_folder = folder_path.name

        # Find all mindmaps in folder
        smmx_files = list(folder_path.glob("*.smmx"))
        print(f"Checking {len(smmx_files)} mindmaps in {rel_folder}...")
        print()

        misplaced = []
        for smmx in sorted(smmx_files):
            # Extract tree ID from filename
            match = re.search(r'id(\d+)\.smmx$', smmx.name)
            if not match:
                match = re.search(r'_(\d+)\.smmx$', smmx.name)
            if not match:
                continue

            tree_id = match.group(1)
            embs = get_embedding_for_tree(tree_id, embeddings_data)
            if embs is None:
                continue

            input_emb, output_emb = embs
            results = suggest_folder(
                input_emb, output_emb, conn,
                top_k=args.top_k,
                temperature=args.temperature,
                verbose=False
            )

            if not results:
                continue

            best_folder, best_prob, best_score = results[0]

            # Check if current folder matches best suggestion
            if best_folder != rel_folder and best_prob > args.threshold:
                title = embeddings_data['titles'][
                    list(embeddings_data['tree_ids']).index(tree_id)
                ] if tree_id in list(embeddings_data['tree_ids']) else smmx.name

                misplaced.append({
                    "file": smmx.name,
                    "tree_id": tree_id,
                    "title": str(title),
                    "suggested_folder": best_folder,
                    "probability": best_prob,
                    "fit_score": best_score
                })

        if args.json:
            print(json.dumps({
                "folder": rel_folder,
                "total_checked": len(smmx_files),
                "misplaced": misplaced
            }, indent=2))
        else:
            if misplaced:
                print(f"Found {len(misplaced)} potentially misplaced mindmaps:")
                print()
                for item in misplaced:
                    print(f"  {item['title'][:50]}...")
                    print(f"    File: {item['file']}")
                    print(f"    Suggested: {item['suggested_folder'] or '(root)'} ({item['probability']:.1%})")
                    print()
            else:
                print(f"No misplacements found (threshold: {args.threshold:.0%})")

    conn.close()


if __name__ == "__main__":
    main()
