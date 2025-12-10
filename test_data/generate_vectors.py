#!/usr/bin/env python3
"""
Generate test vector store with real ONNX embeddings.
Uses all-MiniLM-L6-v2 model for embedding generation.
"""

import json
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

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

def get_embedding_onnx(text, session, tokenizer):
    """Generate embedding using ONNX model."""
    # Tokenize
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )

    # Run inference
    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
    )

    # Mean pooling over sequence
    embedding = outputs[0][0]
    attention_mask = inputs["attention_mask"][0]
    mask_expanded = np.expand_dims(attention_mask, -1)
    sum_embeddings = np.sum(embedding * mask_expanded, axis=0)
    sum_mask = np.clip(np.sum(attention_mask), a_min=1e-9, a_max=None)
    pooled = sum_embeddings / sum_mask

    # Normalize
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm

    return pooled.tolist()

def main():
    # Initialize model
    model_path = "models/all-MiniLM-L6-v2.onnx"
    vocab_path = "models/vocab.txt"

    print(f"Loading model from {model_path}...")
    try:
        session = ort.InferenceSession(model_path)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nFalling back to dummy embeddings for testing...")
        # Generate dummy embeddings (384 dimensions, normalized)
        vector_store = {}
        for doc_id, text in DOCUMENTS.items():
            # Create deterministic but varied embeddings based on text hash
            seed = hash(text) % 10000
            np.random.seed(seed)
            embedding = np.random.randn(384)
            embedding = embedding / np.linalg.norm(embedding)
            vector_store[doc_id] = {
                "embedding": embedding.tolist(),
                "text": text
            }

        output_path = "test_data/papers_vectors.json"
        with open(output_path, "w") as f:
            json.dump(vector_store, f, indent=2)
        print(f"\n✓ Generated dummy vector store: {output_path}")
        print(f"  Documents: {len(vector_store)}")
        print(f"  Embedding dimensions: 384")
        return

    # Generate embeddings for all documents
    vector_store = {}
    print(f"\nGenerating embeddings for {len(DOCUMENTS)} documents...")

    for doc_id, text in DOCUMENTS.items():
        print(f"  Processing {doc_id}...", end=" ")
        embedding = get_embedding_onnx(text, session, tokenizer)
        vector_store[doc_id] = {
            "embedding": embedding,
            "text": text
        }
        print(f"✓ ({len(embedding)} dims)")

    # Save vector store
    output_path = "test_data/papers_vectors.json"
    with open(output_path, "w") as f:
        json.dump(vector_store, f, indent=2)

    print(f"\n✓ Vector store saved: {output_path}")
    print(f"  Documents: {len(vector_store)}")
    print(f"  Embedding dimensions: {len(vector_store['doc1']['embedding'])}")

if __name__ == "__main__":
    main()
