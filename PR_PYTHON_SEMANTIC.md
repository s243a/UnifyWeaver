# feat(python): Semantic Crawling & Search Runtime

## Summary
This PR introduces a Python Runtime Library (`src/unifyweaver/targets/python_runtime/`) that ports the C# target's advanced capabilities to Python. It provides the building blocks for **semantic crawling** and **vector similarity search** using standard Python libraries (`sqlite3`, `numpy`, `onnxruntime`).

## Components
- **`importer.py`**: Manages a SQLite database with tables for objects (`json`), embeddings (`blob`), and links.
- **`crawler.py`**: Implements a breadth-first crawler that fetches XML streams, flattens them, generates embeddings, and upserts to SQLite.
- **`onnx_embedding.py`**: Provides an ONNX-based embedding generator compatible with sentence-transformers models (e.g., `all-MiniLM-L6-v2`).
- **`searcher.py`**: Performs in-memory cosine similarity search over vectors stored in SQLite.

## Documentation
- Added `docs/PYTHON_RUNTIME.md` detailing the runtime components.
- Updated `README.md` to link to the new documentation.

## Tests
- Added `tests/core/test_python_runtime_manual.py` to verify the components (requires `numpy`).

## Dependencies
- `lxml`
- `numpy`
- `onnxruntime` (optional, for embeddings)