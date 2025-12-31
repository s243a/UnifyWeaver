# Rust Semantic Target Proposal

## Objective
Achieve **Semantic Capability Parity** with the Python and Go targets in the Rust backend.

## Storage Strategy: Key-Value vs Relational

While Python uses SQLite (relational) and C# uses LiteDB (document), the Go target successfully uses `bbolt` (pure Go KV) by manually handling indexing.

For Rust, to maintain the "self-contained" philosophy without external C dependencies (like `sqlite3`), we will prioritize **Pure Rust Embedded** databases.

### Candidates
1.  **Redb** (Primary): Embedded ACID Key-Value store. Simple, fast, pure Rust. Ideal for object storage and manual indexing.
2.  **LanceDB** (Secondary): Columnar vector database. Ideal if vector search performance becomes a bottleneck.
3.  **IndraDB** (Tertiary): Graph database. Useful if graph traversal complexity increases significantly.

### Decision
We will start with **Redb**. It matches the architecture we successfully used in Go (`bbolt`), allowing us to reuse the design patterns for manual indexing (links table) and storage (objects bucket). Vector search will be handled in-memory (or via a lightweight linear scan) initially, as `redb` stores vectors as blobs.

## Architecture

#### 1. Rust Semantic Runtime (`src/unifyweaver/targets/rust_runtime/`)

*   **`importer.rs`**: Manages Redb storage.
    *   *Crate*: `redb`
    *   *Tables*:
        *   `objects`: `Table<String, String>` (ID -> JSON)
        *   `embeddings`: `Table<String, Vec<u8>>` (ID -> Vector Blob)
        *   `links`: `Table<String, String>` (Source -> Target) - *Note: May need composite key or multimap for links*
*   **`crawler.rs`**: XML streaming.
    *   *Crate*: `quick-xml`
*   **`searcher.rs`**:
    *   *Vector Search*: Manual Cosine Similarity (load vectors from Redb).
    *   *Graph Traversal*: Key lookups in Redb.

#### 2. Key Construction (`generate_key/2`)
*   **Status**: Implemented (Phase 1).

#### 3. Compilation Model
*   Generate `Cargo.toml` with `redb`, `quick-xml`, `serde`, `serde_json`.

## Implementation Plan

### Phase 1: Key Construction (Completed)
*   Implemented `generate_key/2`.

### Phase 2: Runtime Foundation (Redb + Crawler)
*   Create `rust_runtime` directory.
*   Implement `importer.rs` using **Redb**.
*   Implement `crawler.rs`.

### Phase 3: Search & RAG
*   Implement `searcher.rs`.

## Dependencies
```toml
[dependencies]
redb = "1.4.0"  # Pure Rust embedded DB
quick-xml = "0.30"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
sha2 = "0.10"
uuid = { version = "1.4", features = ["v4"] }
```
