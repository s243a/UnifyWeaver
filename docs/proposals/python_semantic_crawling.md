# Proposal: Python Semantic Crawling & Search Runtime

## Overview
This proposal outlines porting the C# target's advanced capabilities (XML streaming, database population, embedding generation, crawling, and semantic search) to the Python target. This will enable UnifyWeaver to generate Python agents capable of semantic ETL and focused crawling.

## Architecture

We will introduce a Python Runtime Library (embedded or importable) mirroring the C# architecture but leveraging Python's ecosystem.

### 1. Storage (PtImporter Equivalent)
**Technology:** `sqlite3` (Standard Library)
**Tables:**
- `objects`: Stores flattened XML objects (JSON column or flattened columns).
- `embeddings`: Stores vector embeddings (BLOB column for packed float array).
- `links`: Stores graph relationships (parent-child).

### 2. Embedding Provider (OnnxEmbeddingProvider Equivalent)
**Technology:** `onnxruntime` + `numpy`
**Reasoning:** Lighter weight than `sentence-transformers` / `pytorch`, easier to deploy in restricted environments. Matches C# implementation exactly.
**Features:**
- Load `.onnx` model (e.g., all-MiniLM-L6-v2).
- Simple tokenization (WordPiece/BERT tokenizer implemented in Python).
- Mean pooling and normalization.

### 3. Crawler (PtCrawler Equivalent)
**Technology:** `requests` + `lxml` (via `read_xml_lxml`)
**Logic:**
- Maintains a `frontier` (Queue) and `seen` (Set).
- Fetches XML content.
- Uses `read_xml_lxml` to stream and flatten.
- Upserts to SQLite.
- Generates embeddings for title/about/text.
- Discovers children links and adds to frontier.

### 4. Searcher (PtSearcher Equivalent)
**Technology:** `numpy` (for vector ops) or `sqlite-vss` (if available)
**Logic:**
- Embeds query string.
- Performs similarity search:
    - **Small Scale:** Load all vectors from SQLite to numpy, compute cosine similarity.
    - **Large Scale:** Use `sqlite-vss` extension or `faiss` / `chromadb`.
- Returns ranked object IDs.

## Integration with UnifyWeaver

We will expose these capabilities via new Prolog predicates or compilation options in `python_target`.

### Example: Semantic Search Predicate
```prolog
% Search for trees about physics
find_physics_trees(Id, Title, Score) :-
    semantic_search('physics quantum mechanics', 10, Results),
    member(Result, Results),
    Id = Result.id,
    Title = Result.title,
    Score = Result.score.
```

Compiles to:
```python
def find_physics_trees():
    searcher = PtSearcher("data.db", embedding_provider)
    results = searcher.search("physics quantum mechanics", top_k=10)
    for r in results:
        yield {"id": r.id, "title": r.title, "score": r.score}
```

### Example: Focused Crawler
```prolog
crawl_physics(MaxDepth) :-
    % Get seeds semantically
    semantic_search('physics', 5, Seeds),
    % Crawl starting from seeds
    crawler_run(Seeds, MaxDepth).
```

## Implementation Phases

1.  **Phase 1: Runtime Library**: Implement `crawler.py`, `importer.py`, `embedding.py` (ONNX), `searcher.py`.
2.  **Phase 2: Embedder Integration**: Update `python_target.pl` to embed or reference these modules.
3.  **Phase 3: Predicate Compilation**: Add support for compiling `semantic_search/3` and `crawler_run/2`.

## Comparison with C# Target

| Feature | C# Target | Python Target (Proposed) |
| :--- | :--- | :--- |
| **Storage** | LiteDB (NoSQL) | SQLite (Relational/JSON) |
| **Vectors** | BsonArray / Linear Scan | BLOB / Numpy Scan |
| **Model** | ONNX Runtime | ONNX Runtime |
| **XML** | XmlStreamReader | read_xml_lxml |
| **Parsing** | Manual Tokenizer | Manual/HuggingFace Tokenizer |

This alignment ensures consistent behavior across targets while utilizing platform strengths.
