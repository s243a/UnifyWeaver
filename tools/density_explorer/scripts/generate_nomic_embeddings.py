#!/usr/bin/env python3
"""Generate Nomic embeddings for physics dataset."""

import json
import numpy as np
from pathlib import Path

def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    physics_npz = project_root / "datasets" / "wikipedia_physics.npz"
    output_path = Path(__file__).parent.parent / "web" / "data" / "physics_nomic_embeddings.json"

    print(f"Loading physics data from: {physics_npz}")

    data = np.load(physics_npz, allow_pickle=True)
    texts = list(data['texts'])
    titles = list(data['titles'])

    # Use first 200 to match physics.json
    texts = texts[:200]
    titles = titles[:200]

    print(f"Computing Nomic embeddings for {len(texts)} texts...")

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        # Use CPU to avoid CUDA memory issues, or smaller batch size
        device = 'cpu'  # Force CPU for reliability
        print(f"Using device: {device}")

        # Load Nomic model
        model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True, device=device)

        # Nomic requires "search_document: " prefix for documents
        prefixed_texts = [f"search_document: {t}" for t in texts]

        # Compute embeddings with small batch size
        embeddings = model.encode(prefixed_texts, show_progress_bar=True, batch_size=8)

        print(f"Embeddings shape: {embeddings.shape}")

        # Round to 4 decimal places for reasonable file size
        embeddings_rounded = [[round(float(v), 4) for v in row] for row in embeddings]

        # Save as JSON
        output_data = {
            "embeddings": embeddings_rounded,
            "model": "nomic-embed-text-v1",
            "dimension": embeddings.shape[1]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f)

        size_kb = output_path.stat().st_size / 1024
        print(f"Written to: {output_path}")
        print(f"File size: {size_kb:.1f} KB")

    except ImportError:
        print("sentence-transformers not installed. Install with:")
        print("  pip install sentence-transformers")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
