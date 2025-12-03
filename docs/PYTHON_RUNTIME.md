# Python Semantic Runtime

The **Python Semantic Runtime** is a standalone library embedded within generated UnifyWeaver Python scripts. It provides the "heavy lifting" for Semantic AI features, including Crawling, Storage, Embedding, and Search.

## Core Components

### 1. `PtImporter` (Storage)
*   **File**: `src/unifyweaver/targets/python_runtime/importer.py`
*   **Role**: Manages the SQLite database (`data.db`).
*   **Tables**:
    *   `objects`: Stores raw JSON data (`id`, `type`, `data`).
    *   `embeddings`: Stores float32 vectors (`id`, `vector`).
    *   `links`: Stores graph relationships (`source_id`, `target_id`).

### 2. `PtCrawler` (Ingestion)
*   **File**: `src/unifyweaver/targets/python_runtime/crawler.py`
*   **Role**: Fetches and parses XML/RDF data streams.
*   **Features**:
    *   **Streaming**: Uses `etree.iterparse` for low memory usage.
    *   **Link Extraction**: Automatically identifies `rdf:resource` and `pt:parentTree` attributes to populate the `links` table.
    *   **Embedding**: Automatically calls the embedder for textual content (`title`, `text`).

### 3. `PtSearcher` (Retrieval)
*   **File**: `src/unifyweaver/targets/python_runtime/searcher.py`
*   **Role**: Performs retrieval operations.
*   **Methods**:
    *   `search(query, top_k)`: Standard Cosine Similarity search.
    *   `graph_search(query, top_k, hops)`: **Graph RAG**.
        1.  Finds "Anchor" nodes via Vector Search.
        2.  Expands anchors to find Parents (outgoing links) and Children (incoming links).
        3.  Returns a structured context object.

### 4. `LLMProvider` (Synthesis)
*   **File**: `src/unifyweaver/targets/python_runtime/llm.py`
*   **Role**: Interface to Large Language Models.
*   **Implementation**: Wraps the `gemini` CLI tool.
*   **Usage**: `llm_ask(Prompt, Context, Response)`.

### 5. `OnnxEmbeddingProvider` (Vectors)
*   **File**: `src/unifyweaver/targets/python_runtime/onnx_embedding.py`
*   **Role**: Generates text embeddings locally.
*   **Model**: Supports ONNX-exported Transformer models (e.g., `all-MiniLM-L6-v2`).
*   **Execution**: Prefers CPU execution for broad compatibility (e.g., Android/Termux).

## Integration with Prolog

The `python_target` compiler translates Prolog predicates into calls to this runtime:

| Prolog Predicate | Python Runtime Call |
| :--- | :--- |
| `upsert_object(Id, Type, Data)` | `importer.upsert_object(...)` |
| `crawler_run(Seeds, Depth)` | `crawler.crawl(...)` |
| `semantic_search(Q, K, R)` | `searcher.search(...)` |
| `graph_search(Q, K, H, R)` | `searcher.graph_search(...)` |
| `llm_ask(P, C, R)` | `llm.ask(...)` |

## Configuration

The runtime looks for models in the `./models/` directory:
*   `models/model.onnx`: The ONNX model file.
*   `models/vocab.txt`: The BERT vocabulary file.

If these are missing, embedding capabilities are gracefully disabled.