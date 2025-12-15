#!/bin/bash
# test_papers - Semantic search (bash + Python ONNX backend)

test_papers() {
    local query="$1"

    if [ -z "$query" ]; then
        echo "Error: Query argument required" >&2
        return 1
    fi

    python3 - "$query" <<'PYTHON_EOF'
import sys
import json
import numpy as np

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    """Compute negative euclidean distance (so higher is better)."""
    return -np.linalg.norm(a - b)

def dot_product(a, b):
    """Compute dot product similarity."""
    return np.dot(a, b)

def get_embedding_onnx(text):
    """Generate embedding using ONNX model."""
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Load model and tokenizer
        session = ort.InferenceSession("models/all-MiniLM-L6-v2.onnx")
        tokenizer = AutoTokenizer.from_pretrained("models/vocab.txt")

        # Tokenize input
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)

        # Run inference
        outputs = session.run(None, {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        })

        # Extract embedding (mean pooling over sequence)
        embedding = outputs[0][0]
        attention_mask = inputs["attention_mask"][0]
        mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(embedding * mask_expanded, axis=0)
        sum_mask = np.clip(np.sum(attention_mask), a_min=1e-9, a_max=None)
        embedding = sum_embeddings / sum_mask

    # L2 normalization
    embedding = embedding / np.linalg.norm(embedding)

        return embedding
    except Exception as e:
        print(f"Error generating embedding: {{e}}", file=sys.stderr)
        sys.exit(1)

def search_vectors(query_embedding, vector_store_path, top_k, threshold, similarity_func):
    """Search vector store for similar documents."""
    try:
        with open(vector_store_path, "r") as f:
            vector_store = json.load(f)

        results = []
        for doc_id, doc_vector in vector_store.items():
            doc_embedding = np.array(doc_vector)
            similarity = similarity_func(query_embedding, doc_embedding)

            if similarity >= threshold:
                results.append((doc_id, float(similarity)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    except FileNotFoundError:
        print(f"Error: Vector store not found: {{vector_store_path}}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in vector store: {{e}}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error searching vectors: {{e}}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: <script> <query>", file=sys.stderr)
        sys.exit(1)

    query = sys.argv[1]
    query_embedding = get_embedding_onnx(query)

    results = search_vectors(
        query_embedding,
        "test_data/papers_vectors.json",  # vector_store_path
        5,    # top_k
        0.6,    # threshold
        cosine_similarity     # similarity_func
    )

    for doc_id, score in results:
        print(f"{{doc_id}}:{{score:.4f}}")

if __name__ == "__main__":
    main()

PYTHON_EOF
}

test_papers_stream() {
    test_papers "$@"
}

test_papers_batch() {
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            test_papers "$line"
        fi
    done
}

export -f test_papers
export -f test_papers_stream
export -f test_papers_batch
