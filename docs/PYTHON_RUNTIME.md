# Python Semantic Runtime Library

The `unifyweaver.targets.python_runtime` package provides the core components for semantic crawling and search in Python-based agents.

## Components

### `PtImporter` (`importer.py`)
Manages the SQLite database for storing objects and embeddings.

```python
importer = PtImporter("data.db")
importer.upsert_object("id1", "type", {"key": "value"})
importer.upsert_embedding("id1", vector_list)
```

**Schema:**
- `objects`: `id` (TEXT PK), `type` (TEXT), `data` (JSON)
- `embeddings`: `id` (TEXT PK), `vector` (BLOB - float32 array)
- `links`: `source_id`, `target_id`

### `PtCrawler` (`crawler.py`)
Orchestrates the crawling process.

```python
crawler = PtCrawler(importer, embedder)
crawler.crawl(seed_ids, fetch_func)
```

**Features:**
- Breadth-first traversal
- XML stream processing (via `lxml`)
- Automatic embedding generation

### `PtSearcher` (`searcher.py`)
Performs semantic search.

```python
searcher = PtSearcher("data.db", embedder)
results = searcher.search("query", top_k=10)
# results: [(score, id), ...]
```

### `OnnxEmbeddingProvider` (`onnx_embedding.py`)
Generates embeddings using an ONNX model.

```python
embedder = OnnxEmbeddingProvider("model.onnx", "vocab.txt")
vector = embedder.get_embedding("text")
```

## Requirements
- `sqlite3` (Built-in)
- `numpy`
- `lxml`
- `onnxruntime` (Optional, for `OnnxEmbeddingProvider`)
