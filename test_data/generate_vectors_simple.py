#!/usr/bin/env python3
"""
Generate test vector store with dummy but deterministic embeddings.
Simple version without external dependencies.
"""

import json
import math
import random

# Sample documents about programming topics
DOCUMENTS = {
    "doc1": "Python is a high-level programming language known for its simplicity and readability.",
    "doc2": "Machine learning algorithms can learn patterns from data without explicit programming.",
    "doc3": "Rust is a systems programming language focused on safety, speed, and concurrency.",
    "doc4": "Vector databases enable efficient similarity search over high-dimensional embeddings.",
    "doc5": "Natural language processing helps computers understand and generate human language.",
    "doc6": "Prolog is a logic programming language based on formal logic and automated reasoning.",
    "doc7": "Neural networks are computational models inspired by biological neural networks.",
    "doc8": "Semantic search finds results based on meaning rather than exact keyword matches.",
}

def normalize_vector(vec):
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(x*x for x in vec))
    if norm > 0:
        return [x / norm for x in vec]
    return vec

def generate_deterministic_embedding(text, dimensions=384):
    """Generate a deterministic embedding based on text hash."""
    # Use text hash as seed for reproducibility
    seed = hash(text) % 10000
    random.seed(seed)

    # Generate random vector
    embedding = [random.gauss(0, 1) for _ in range(dimensions)]

    # Normalize to unit vector
    return normalize_vector(embedding)

def main():
    vector_store = {}

    print(f"Generating embeddings for {len(DOCUMENTS)} documents...")

    for doc_id, text in DOCUMENTS.items():
        print(f"  Processing {doc_id}...", end=" ")
        embedding = generate_deterministic_embedding(text)
        vector_store[doc_id] = {
            "embedding": embedding,
            "text": text
        }
        print(f"✓ ({len(embedding)} dims)")

    # Save vector store
    output_path = "papers_vectors.json"
    with open(output_path, "w") as f:
        json.dump(vector_store, f, indent=2)

    print(f"\n✓ Vector store saved: {output_path}")
    print(f"  Documents: {len(vector_store)}")
    print(f"  Embedding dimensions: {len(vector_store['doc1']['embedding'])}")
    print("\nSample documents:")
    for doc_id in list(DOCUMENTS.keys())[:3]:
        print(f"  {doc_id}: {DOCUMENTS[doc_id][:60]}...")

if __name__ == "__main__":
    main()
