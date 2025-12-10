#!/usr/bin/env python3
"""
Direct test of semantic search without ONNX model.
Uses dummy query embeddings to test the search pipeline.
"""

import sys
import json
import numpy as np
import random
import math

def normalize_vector(vec):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec

def generate_query_embedding(query_text, dimensions=384):
    """Generate a deterministic query embedding."""
    seed = hash(query_text) % 10000
    np.random.seed(seed)
    embedding = np.random.randn(dimensions)
    return normalize_vector(embedding)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_vectors(query_embedding, vector_store_path, top_k=5, threshold=0.0):
    """Search vector store for similar documents."""
    with open(vector_store_path, "r") as f:
        vector_store = json.load(f)

    results = []
    for doc_id, doc_data in vector_store.items():
        doc_embedding = np.array(doc_data["embedding"])
        similarity = cosine_similarity(query_embedding, doc_embedding)

        if similarity >= threshold:
            results.append((doc_id, float(similarity), doc_data.get("text", "")))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def main():
    if len(sys.argv) < 2:
        print("Usage: test_search.py <query>")
        sys.exit(1)

    query = sys.argv[1]
    print(f"\nQuery: '{query}'")
    print("="*60)

    # Generate query embedding
    query_embedding = generate_query_embedding(query)
    print(f"Query embedding: {len(query_embedding)} dimensions")

    # Search vector store
    results = search_vectors(
        query_embedding,
        "papers_vectors.json",
        top_k=5,
        threshold=0.0
    )

    print(f"\nTop {len(results)} results:")
    print("-"*60)
    for rank, (doc_id, score, text) in enumerate(results, 1):
        print(f"\n{rank}. {doc_id} (similarity: {score:.4f})")
        print(f"   {text[:80]}...")

    print("\n" + "="*60)
    print(f"✓ Search completed successfully ({len(results)} results)")

if __name__ == "__main__":
    main()
