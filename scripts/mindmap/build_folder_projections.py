#!/usr/bin/env python3
"""
Build folder projections for mindmap organization suggestions.

For each folder containing mindmaps, computes:
1. Centroid: average of tree embeddings (for fast top-k filtering)
2. W matrix: Procrustes projection (for precise fit scoring)

The W matrix maps input embeddings to output embeddings via orthogonal
transformation, capturing the folder's "semantic projection style".

Usage:
    # Full rebuild
    python3 scripts/mindmap/build_folder_projections.py \
        --embeddings models/dual_embeddings_combined_2026-01-02_trees_only.npz \
        --index output/mindmaps_curated/index.json \
        --output output/mindmaps_curated/folder_projections.db

    # Incremental (only changed folders)
    python3 scripts/mindmap/build_folder_projections.py \
        --embeddings models/dual_embeddings_combined_2026-01-02_trees_only.npz \
        --index output/mindmaps_curated/index.json \
        --output output/mindmaps_curated/folder_projections.db \
        --incremental

    # Specific folders only
    python3 scripts/mindmap/build_folder_projections.py \
        --embeddings models/dual_embeddings_combined_2026-01-02_trees_only.npz \
        --index output/mindmaps_curated/index.json \
        --output output/mindmaps_curated/folder_projections.db \
        --folders "Media_Reviews" "Economics"
"""

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.linalg import orthogonal_procrustes
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def create_db(db_path: Path) -> sqlite3.Connection:
    """Create or open the folder projections database."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS folder_projections (
            folder_path TEXT PRIMARY KEY,
            centroid BLOB NOT NULL,
            w_matrix BLOB NOT NULL,
            n_trees INTEGER NOT NULL,
            procrustes_scale REAL NOT NULL,
            last_updated REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    return conn


def get_folder_timestamps(conn: sqlite3.Connection) -> Dict[str, float]:
    """Get last update timestamps for all folders."""
    cursor = conn.execute("SELECT folder_path, last_updated FROM folder_projections")
    return dict(cursor.fetchall())


def save_projection(
    conn: sqlite3.Connection,
    folder_path: str,
    centroid: np.ndarray,
    w_matrix: np.ndarray,
    n_trees: int,
    scale: float
):
    """Save a folder projection to the database."""
    conn.execute("""
        INSERT OR REPLACE INTO folder_projections
        (folder_path, centroid, w_matrix, n_trees, procrustes_scale, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        folder_path,
        centroid.astype(np.float32).tobytes(),
        w_matrix.astype(np.float32).tobytes(),
        n_trees,
        scale,
        time.time()
    ))
    conn.commit()


def load_projection(conn: sqlite3.Connection, folder_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, int, float]]:
    """Load a folder projection from the database."""
    cursor = conn.execute("""
        SELECT centroid, w_matrix, n_trees, procrustes_scale
        FROM folder_projections WHERE folder_path = ?
    """, (folder_path,))
    row = cursor.fetchone()
    if row is None:
        return None
    centroid = np.frombuffer(row[0], dtype=np.float32)
    w_matrix = np.frombuffer(row[1], dtype=np.float32).reshape(768, 768)
    return centroid, w_matrix, row[2], row[3]


def load_all_centroids(conn: sqlite3.Connection) -> Tuple[List[str], np.ndarray]:
    """Load all folder centroids for top-k filtering."""
    cursor = conn.execute("SELECT folder_path, centroid FROM folder_projections ORDER BY folder_path")
    rows = cursor.fetchall()
    if not rows:
        return [], np.array([])
    folders = [r[0] for r in rows]
    centroids = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return folders, centroids


def group_trees_by_folder(
    index: Dict[str, str],
    tree_ids: np.ndarray
) -> Dict[str, List[int]]:
    """Group tree indices by their folder path."""
    # Build tree_id to index mapping
    id_to_idx = {str(tid): i for i, tid in enumerate(tree_ids)}

    # Group by folder
    folder_trees = defaultdict(list)
    for tree_id, rel_path in index.items():
        if tree_id not in id_to_idx:
            continue
        # Extract folder from path (everything before the filename)
        parts = rel_path.rsplit('/', 1)
        folder = parts[0] if len(parts) > 1 else ""
        folder_trees[folder].append(id_to_idx[tree_id])

    return dict(folder_trees)


def compute_procrustes(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute orthogonal Procrustes: find W that minimizes ||X @ W - Y||.

    Returns:
        W: 768x768 orthogonal matrix
        scale: scaling factor
    """
    if not HAS_SCIPY:
        # Fallback: simple SVD-based solution
        M = X.T @ Y
        U, _, Vt = np.linalg.svd(M)
        W = U @ Vt
        scale = 1.0
    else:
        W, scale = orthogonal_procrustes(X, Y)
    return W, scale


def build_folder_projection(
    input_embs: np.ndarray,
    output_embs: np.ndarray,
    indices: List[int],
    min_trees: int = 2
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Build projection for a single folder.

    Returns:
        centroid: average input embedding
        W: Procrustes transformation matrix
        scale: Procrustes scale factor
    """
    if len(indices) < min_trees:
        return None

    X = input_embs[indices]  # n x 768
    Y = output_embs[indices]  # n x 768

    # Centroid is average of input embeddings
    centroid = X.mean(axis=0)

    # Procrustes: find W minimizing ||X @ W - Y||
    W, scale = compute_procrustes(X, Y)

    return centroid, W, scale


def get_folder_mtime(base_dir: Path, folder: str) -> float:
    """Get the newest modification time of any .smmx file in the folder."""
    folder_path = base_dir / folder if folder else base_dir
    if not folder_path.exists():
        return 0.0

    mtimes = [f.stat().st_mtime for f in folder_path.glob("*.smmx")]
    return max(mtimes) if mtimes else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Build folder projections for mindmap organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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
        "--output", "-o",
        type=Path,
        default=Path("output/mindmaps_curated/folder_projections.db"),
        help="Output SQLite database path"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only update folders with changed files"
    )
    parser.add_argument(
        "--folders", "-f",
        nargs="+",
        help="Only update specific folders (by name, not full path)"
    )
    parser.add_argument(
        "--min-trees",
        type=int,
        default=2,
        help="Minimum trees required to compute projection (default: 2)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Check scipy
    if not HAS_SCIPY:
        print("Warning: scipy not available, using fallback Procrustes", file=sys.stderr)

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    data = np.load(args.embeddings, allow_pickle=True)
    input_embs = data['input_nomic']  # n x 768
    output_embs = data['output_nomic']  # n x 768
    tree_ids = data['tree_ids']
    print(f"  Loaded {len(tree_ids)} trees with {input_embs.shape[1]}-dim embeddings")

    # Load index
    print(f"Loading index from {args.index}...")
    with open(args.index) as f:
        index_data = json.load(f)
    index = index_data['index']
    base_dir = Path(index_data.get('base_dir', args.index.parent))
    print(f"  Index contains {len(index)} entries")

    # Group by folder
    folder_trees = group_trees_by_folder(index, tree_ids)
    print(f"  Found {len(folder_trees)} folders with trees")

    # Open/create database
    args.output.parent.mkdir(parents=True, exist_ok=True)
    conn = create_db(args.output)

    # Get existing timestamps for incremental mode
    existing_timestamps = get_folder_timestamps(conn) if args.incremental else {}

    # Filter folders if specified
    folders_to_process = list(folder_trees.keys())
    if args.folders:
        # Match by folder name (last component)
        folder_names = set(args.folders)
        folders_to_process = [
            f for f in folders_to_process
            if f.split('/')[-1] in folder_names or f in folder_names
        ]
        print(f"  Filtered to {len(folders_to_process)} specified folders")

    # Process folders
    updated = 0
    skipped = 0
    too_small = 0

    start_time = time.time()

    for folder in sorted(folders_to_process):
        indices = folder_trees[folder]
        display_name = folder if folder else "(root)"

        # Check if update needed (incremental mode)
        if args.incremental and folder in existing_timestamps:
            folder_mtime = get_folder_mtime(base_dir, folder)
            if folder_mtime <= existing_timestamps[folder]:
                if args.verbose:
                    print(f"  Skipping {display_name} (unchanged)")
                skipped += 1
                continue

        # Compute projection
        result = build_folder_projection(input_embs, output_embs, indices, args.min_trees)

        if result is None:
            if args.verbose:
                print(f"  Skipping {display_name} (only {len(indices)} trees, need {args.min_trees})")
            too_small += 1
            continue

        centroid, W, scale = result
        save_projection(conn, folder, centroid, W, len(indices), scale)
        updated += 1

        if args.verbose:
            print(f"  {display_name}: {len(indices)} trees, scale={scale:.4f}")

    elapsed = time.time() - start_time

    # Save metadata
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("embedding_dim", str(input_embs.shape[1]))
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("embeddings_file", str(args.embeddings))
    )
    conn.commit()
    conn.close()

    # Summary
    print()
    print(f"Completed in {elapsed:.2f}s")
    print(f"  Updated: {updated} folders")
    print(f"  Skipped (unchanged): {skipped} folders")
    print(f"  Skipped (too small): {too_small} folders")
    print(f"  Output: {args.output}")

    # Show storage stats
    db_size = args.output.stat().st_size
    print(f"  Database size: {db_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
