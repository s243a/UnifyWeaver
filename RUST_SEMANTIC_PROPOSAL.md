# Rust Semantic Target Proposal

## Objective
Achieve **Semantic Capability Parity** with the Python and Go targets in the Rust backend. This includes:
1.  **Streaming XML Ingestion** (`source(xml)`).
2.  **Vector Embeddings** (via ONNX Runtime).
3.  **Vector Search** (Cosine Similarity).
4.  **Local Storage** (SQLite via `rusqlite`).
5.  **Graph RAG** (Vector + Link Traversal).

## Architecture

### 1. Rust Semantic Runtime (`src/unifyweaver/targets/rust_runtime/`)
We will create a reusable Rust library that the generated code can depend on or inline.

*   **`importer.rs`**: Manages SQLite connection.
    *   *Crate*: `rusqlite`
    *   *Tables*: `objects`, `embeddings`, `links`.
*   **`crawler.rs`**: High-performance XML streaming.
    *   *Crate*: `quick-xml`
    *   *Features*: Link extraction (`rdf:resource`), flattening.
*   **`embedding.rs`**: Vector generation.
    *   *Crate*: `ort` (ONNX Runtime bindings) or `candle-core`.
*   **`searcher.rs`**: Search logic.
    *   *Features*: Vector Search, Text Search (SQL LIKE), Graph Traversal (1-hop).
*   **`llm.rs`**: LLM integration.
    *   *Implementation*: Wrapper around `gemini` CLI via `std::process::Command`.

### 2. Key Construction (`generate_key/2`)
Port the key generation logic from Go/Python.
*   **Strategies**: `composite`, `hash` (via `sha2`), `uuid` (via `uuid` crate).
*   **Implementation**: Modify `rust_target.pl` to generate Rust expressions.

### 3. Compilation Model
The `rust_target.pl` will be updated to:
*   Detect `semantic` mode or specific predicates (`crawler_run`, `graph_search`).
*   Generate a `Cargo.toml` with necessary dependencies (`rusqlite`, `quick-xml`, `tokio`, `anyhow`).
*   Generate `main.rs` that orchestrates the runtime components.

## Implementation Plan

### Phase 1: Key Construction & Basic Logic
*   Implement `generate_key/2` support in `rust_target.pl`.
*   Add `sha2` and `uuid` to dependency detection.

### Phase 2: Runtime Foundation (SQLite + Crawler)
*   Create `rust_runtime` directory.
*   Implement `importer.rs` (SQLite schema).
*   Implement `crawler.rs` (XML parsing).

### Phase 3: Search & RAG
*   Implement `searcher.rs` (Vector/Text search).
*   Implement `graph_search` predicate translation.

## Dependencies
```toml
[dependencies]
rusqlite = { version = "0.29", features = ["bundled"] }
quick-xml = "0.30"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
sha2 = "0.10"
uuid = { version = "1.4", features = ["v4"] }
# ort = "..." (Optional for embeddings)
```
