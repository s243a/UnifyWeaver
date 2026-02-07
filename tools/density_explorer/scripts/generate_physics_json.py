#!/usr/bin/env python3
"""Generate physics.json with embeddings for density explorer.

Uses the deduplicated wikipedia_physics_nomic.npz (300 unique articles)
rather than wikipedia_physics.npz (2198 categories with duplicates).
"""

import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.density_core import load_and_compute


def main():
    # Paths - use the clean Nomic 300 dataset (0 duplicate titles)
    embeddings_path = Path(__file__).parent.parent.parent.parent / "datasets" / "wikipedia_physics_nomic.npz"
    output_path = Path(__file__).parent.parent / "web" / "data" / "physics.json"

    if not embeddings_path.exists():
        print(f"ERROR: Dataset not found: {embeddings_path}")
        print("The wikipedia_physics_nomic.npz dataset contains 300 unique Wikipedia")
        print("physics articles with 768D Nomic embeddings.")
        return 1

    print(f"Loading embeddings from: {embeddings_path}")

    # Compute manifold
    data = load_and_compute(
        str(embeddings_path),
        top_k=200,  # Use first 200 articles
        grid_size=100,
        include_tree=True,
        tree_type='mst',
        include_peaks=True,
        n_peaks=5
    )

    print(f"Computed manifold: {data.n_points} points")

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(data.to_json())

    # Report file size
    size_kb = output_path.stat().st_size / 1024
    print(f"Written to: {output_path}")
    print(f"File size: {size_kb:.1f} KB")

    return 0

if __name__ == "__main__":
    exit(main())
