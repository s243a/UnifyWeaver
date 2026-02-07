#!/usr/bin/env python3
"""Generate reproducible Wikipedia physics embedding datasets.

Phase 1 of the reproducible embedding datasets proposal:
- wikipedia_physics_nomic_titles.npz: Embed article titles only
- wikipedia_physics_nomic_text.npz: Embed article text_preview content

Both use nomic-embed-text-v1.5 with "search_document: " prefix.
Source: reports/wikipedia_physics_articles.jsonl (300 articles)
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def compute_file_hash(path: Path) -> str:
    """SHA-256 hash of source file for provenance tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_articles(jsonl_path: Path):
    """Load titles and text_previews from the JSONL source."""
    titles = []
    text_previews = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            titles.append(entry["title"])
            text_previews.append(entry["text_preview"])
    return titles, text_previews


def embed(model, texts: list[str], prefix: str = "search_document: ", batch_size: int = 8):
    """Embed texts with the given prefix."""
    prefixed = [f"{prefix}{t}" for t in texts]
    embeddings = model.encode(prefixed, show_progress_bar=True, batch_size=batch_size)
    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    return embeddings.astype(np.float32)


def save_dataset(
    path: Path,
    embeddings: np.ndarray,
    titles: list[str],
    input_texts: list[str],
    model_name: str,
    prefix: str,
    source_hash: str,
    source_file: str,
    embed_field: str,
):
    """Save embeddings with full provenance metadata."""
    metadata = {
        "model": model_name,
        "prefix": prefix,
        "embed_field": embed_field,
        "source_file": source_file,
        "source_hash": source_hash,
        "num_items": len(titles),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/generate_reproducible_embeddings.py",
    }

    np.savez_compressed(
        path,
        embeddings=embeddings,
        titles=np.array(titles),
        input_texts=np.array(input_texts),
        metadata=np.array(json.dumps(metadata)),
    )
    size_kb = path.stat().st_size / 1024
    print(f"  Saved: {path} ({size_kb:.0f} KB, {embeddings.shape})")
    return metadata


def main():
    project_root = Path(__file__).parent.parent
    jsonl_path = project_root / "reports" / "wikipedia_physics_articles.jsonl"
    output_dir = project_root / "datasets"

    if not jsonl_path.exists():
        print(f"ERROR: Source file not found: {jsonl_path}")
        return 1

    print(f"Source: {jsonl_path}")
    source_hash = compute_file_hash(jsonl_path)
    print(f"SHA-256: {source_hash}")

    titles, text_previews = load_articles(jsonl_path)
    print(f"Loaded {len(titles)} articles")

    # Load model
    print("\nLoading nomic-embed-text-v1.5...")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu"
        )
        model_name = "nomic-embed-text-v1.5"
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return 1

    prefix = "search_document: "

    # --- Titles dataset ---
    print(f"\n--- Embedding titles ({len(titles)} items) ---")
    title_embeddings = embed(model, titles, prefix=prefix)
    title_meta = save_dataset(
        path=output_dir / "wikipedia_physics_nomic_titles.npz",
        embeddings=title_embeddings,
        titles=titles,
        input_texts=titles,
        model_name=model_name,
        prefix=prefix,
        source_hash=source_hash,
        source_file="reports/wikipedia_physics_articles.jsonl",
        embed_field="title",
    )

    # --- Text dataset ---
    print(f"\n--- Embedding text_previews ({len(text_previews)} items) ---")
    text_embeddings = embed(model, text_previews, prefix=prefix)
    text_meta = save_dataset(
        path=output_dir / "wikipedia_physics_nomic_text.npz",
        embeddings=text_embeddings,
        titles=titles,
        input_texts=text_previews,
        model_name=model_name,
        prefix=prefix,
        source_hash=source_hash,
        source_file="reports/wikipedia_physics_articles.jsonl",
        embed_field="text_preview",
    )

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"Model: {model_name}")
    print(f"Prefix: '{prefix}'")
    print(f"Source: {jsonl_path.name} (SHA-256: {source_hash[:16]}...)")
    print(f"Titles:  {title_embeddings.shape} → wikipedia_physics_nomic_titles.npz")
    print(f"Text:    {text_embeddings.shape} → wikipedia_physics_nomic_text.npz")

    # Quick sanity: cosine similarity between title and text embeddings
    cos_sim = np.sum(title_embeddings * text_embeddings, axis=1)
    print(f"\nTitle-vs-text cosine similarity: mean={cos_sim.mean():.3f}, min={cos_sim.min():.3f}, max={cos_sim.max():.3f}")

    return 0


if __name__ == "__main__":
    exit(main())
