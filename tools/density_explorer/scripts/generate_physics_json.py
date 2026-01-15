#!/usr/bin/env python3
"""Generate physics.json with embeddings for density explorer."""

import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.density_core import load_and_compute

def main():
    # Paths
    embeddings_path = Path(__file__).parent.parent.parent.parent / "datasets" / "wikipedia_physics.npz"
    output_path = Path(__file__).parent.parent / "web" / "data" / "physics.json"

    print(f"Loading embeddings from: {embeddings_path}")

    # Compute manifold with both tree types and embeddings
    data = load_and_compute(
        str(embeddings_path),
        top_k=200,  # Use first 200 articles
        tree_types=['mst', 'j-guided'],
        include_embeddings=True,
        embedding_precision=4,  # 4 decimal places for reasonable file size
        grid_size=100,
        n_peaks=5
    )

    print(f"Computed manifold: {data.n_points} points")
    print(f"Trees: {list(data.trees.keys()) if data.trees else 'none'}")
    print(f"Embeddings included: {len(data.embeddings) if data.embeddings else 0} vectors")

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(data.to_json())

    # Report file size
    size_kb = output_path.stat().st_size / 1024
    print(f"Written to: {output_path}")
    print(f"File size: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()
