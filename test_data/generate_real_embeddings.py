#!/usr/bin/env python3
"""
Generate real embeddings using the ONNX all-MiniLM-L6-v2 model.
"""

import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Sample documents
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
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])).astype(np.int64)
        }
    )

    # Mean pooling over sequence
    embedding = outputs[0][0]
    attention_mask = inputs["attention_mask"][0]
    mask_expanded = np.expand_dims(attention_mask, -1)
    sum_embeddings = np.sum(embedding * mask_expanded, axis=0)
    sum_mask = np.clip(np.sum(attention_mask), a_min=1e-9, a_max=None)
    pooled = sum_embeddings / sum_mask

    # L2 normalization
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm

    return pooled.tolist()

def main():
    # Initialize model and tokenizer
    model_path = "../models/all-MiniLM-L6-v2/onnx/model.onnx"
    tokenizer_path = "../models/all-MiniLM-L6-v2"

    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path)

    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"\nGenerating embeddings for {len(DOCUMENTS)} documents...")
    print("="*60)

    vector_store = {}
    for doc_id, text in DOCUMENTS.items():
        print(f"Processing {doc_id}...", end=" ")
        embedding = get_embedding_onnx(text, session, tokenizer)
        vector_store[doc_id] = {
            "embedding": embedding,
            "text": text
        }
        print(f"✓ ({len(embedding)} dims)")

    # Save vector store
    output_path = "papers_vectors.json"
    with open(output_path, "w") as f:
        json.dump(vector_store, f, indent=2)

    print("\n" + "="*60)
    print(f"✓ Vector store saved: {output_path}")
    print(f"  Documents: {len(vector_store)}")
    print(f"  Embedding dimensions: {len(vector_store['doc1']['embedding'])}")
    print("\nSample embedding (first 5 values):")
    print(f"  {vector_store['doc1']['embedding'][:5]}")

if __name__ == "__main__":
    main()
