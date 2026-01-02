#!/usr/bin/env python3
"""
Generate dual embeddings for Physics subset.

Creates:
- Input embeddings: raw_title with Nomic AND ModernBERT
- Output embeddings: functional query (locate_node/locate_url) with Nomic
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import argparse


def load_data(jsonl_path: Path) -> list:
    """Load JSONL data."""
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


def get_output_query(item: dict) -> str:
    """Generate functional overlay query based on item type."""
    item_type = item.get("type", "Tree")
    title = item.get("raw_title", "")
    
    if item_type == "PagePearl":
        return f"locate_url('{title}')"
    else:  # Tree or other
        return f"locate_node('{title}')"


def main():
    parser = argparse.ArgumentParser(description="Generate dual embeddings")
    parser.add_argument("--data", type=Path, required=True,
                       help="Path to JSONL data file")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output NPZ file path")
    parser.add_argument("--nomic-model", default="nomic-ai/nomic-embed-text-v1.5",
                       help="Nomic model name")
    parser.add_argument("--alt-model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Alternative model for Input Objective (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    print(f"Loaded {len(data)} items")
    
    # Extract texts and metadata
    titles = [item.get("raw_title", "") for item in data]
    output_queries = [get_output_query(item) for item in data]
    item_types = [item.get("type", "Tree") for item in data]
    tree_ids = [item.get("tree_id", "") for item in data]
    # URIs for unique identification (uri for Trees, pearl_uri for PagePearls)
    uris = [item.get("uri") or item.get("pearl_uri", "") for item in data]
    
    # Load models
    print(f"Loading Nomic model: {args.nomic_model}...")
    nomic = SentenceTransformer(args.nomic_model, trust_remote_code=True)
    
    print(f"Loading alternative model: {args.alt_model}...")
    alt_model = SentenceTransformer(args.alt_model, trust_remote_code=True)
    
    # Generate Input embeddings
    print("Generating Input embeddings (Nomic - raw titles)...")
    input_nomic = nomic.encode(titles, show_progress_bar=True, convert_to_numpy=True)
    
    print("Generating Input embeddings (Alt Model - raw titles)...")
    input_alt = alt_model.encode(titles, show_progress_bar=True, convert_to_numpy=True)
    
    # Generate Output embeddings (functional queries with Nomic)
    print("Generating Output embeddings (Nomic - functional queries)...")
    output_nomic = nomic.encode(output_queries, show_progress_bar=True, convert_to_numpy=True)
    
    # Save
    print(f"Saving to {args.output}...")
    np.savez_compressed(
        args.output,
        input_nomic=input_nomic.astype(np.float32),
        input_alt=input_alt.astype(np.float32),
        output_nomic=output_nomic.astype(np.float32),
        titles=np.array(titles, dtype=object),
        item_types=np.array(item_types, dtype=object),
        tree_ids=np.array(tree_ids, dtype=object),
        uris=np.array(uris, dtype=object),  # Unique identifiers
        output_queries=np.array(output_queries, dtype=object)
    )
    
    print("Done!")
    print(f"  Input Nomic shape: {input_nomic.shape}")
    print(f"  Input Alt shape: {input_alt.shape}")
    print(f"  Output Nomic shape: {output_nomic.shape}")


if __name__ == "__main__":
    main()
