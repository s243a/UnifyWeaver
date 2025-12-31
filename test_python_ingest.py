#!/usr/bin/env python3
"""
Python Pearltrees Ingestion Test

Tests the Python runtime's AWK fragment processing with embeddings,
comparing against C# and Rust implementations.

Usage:
    awk -f scripts/utils/select_xml_elements.awk \
        -v tag="pt:Tree|pt:RefPearl|pt:Pearl" \
        context/PT/pearltrees_export.rdf | \
        python3 test_python_ingest.py
"""

import sys
import os

# Add python_runtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'unifyweaver', 'targets'))

from python_runtime.importer import PtImporter
from python_runtime.crawler import PtCrawler

# Try to use sentence-transformers, fall back to ONNX or no embeddings
embedder = None
try:
    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers model...", file=sys.stderr)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    class SentenceTransformerProvider:
        def __init__(self, model):
            self.model = model

        def get_embedding(self, text):
            return self.model.encode(text, convert_to_numpy=True).tolist()

    embedder = SentenceTransformerProvider(model)
    print("✓ Sentence-transformers model loaded", file=sys.stderr)
except ImportError:
    print("Warning: sentence-transformers not available, trying ONNX...", file=sys.stderr)
    try:
        from python_runtime.onnx_embedding import OnnxEmbeddingProvider
        model_dir = "models/all-MiniLM-L6-v2-onnx"
        model_path = os.path.join(model_dir, "model.onnx")
        vocab_path = os.path.join(model_dir, "vocab.txt")

        if os.path.exists(model_path) and os.path.exists(vocab_path):
            embedder = OnnxEmbeddingProvider(model_path, vocab_path)
            print("✓ ONNX model loaded", file=sys.stderr)
        else:
            print(f"Warning: ONNX model not found at {model_path}", file=sys.stderr)
            print("Continuing without embeddings...", file=sys.stderr)
    except ImportError as e:
        print(f"Warning: ONNX not available ({e}), continuing without embeddings...", file=sys.stderr)

def main():
    print("=== Python Pearltrees Ingestion ===", file=sys.stderr)

    db_path = "pt_ingest_test.db"
    print(f"Creating database: {db_path}", file=sys.stderr)

    # Initialize components
    importer = PtImporter(db_path)
    crawler = PtCrawler(importer, embedder)

    # Process fragments from stdin
    print("Reading XML fragments from stdin...", file=sys.stderr)
    print("(Expecting null-delimited fragments from AWK)", file=sys.stderr)

    try:
        crawler.process_fragments_from_stdin()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n✓ Ingestion complete!", file=sys.stderr)
    print(f"Database created: {db_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
