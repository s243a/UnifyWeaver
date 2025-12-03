# Proposal: Go Semantic Runtime with Hugot

## Overview
To achieve parity with the C# and Python targets, we will implement a **Semantic Runtime for Go**. This runtime will enable Go-based agents to perform semantic crawling, embedding generation, and vector similarity search.

## Core Decisions

### 1. Embeddings: `knights-analytics/hugot`
We will use [hugot](https://github.com/knights-analytics/hugot) for embedding generation.
*   **Why:** It provides a high-level, "Python-like" pipeline API for Transformers in Go. It handles tokenization and ONNX Runtime interaction, abstracting away the raw tensor manipulation.
*   **Model:** `all-MiniLM-L6-v2` (ONNX format).
*   **Constraint:** Requires CGO and ONNX Runtime shared libraries on the host.

### 2. Storage: `bbolt`
We will stick with [bbolt](https://github.com/etcd-io/bbolt) for storage.
*   **Why:** Pure Go, already integrated into `go_target`, efficient Key-Value store.
*   **Schema:**
    *   Bucket `objects`: JSON documents.
    *   Bucket `embeddings`: Binary blobs (float32 slices).
    *   Bucket `links`: Graph edges.

### 3. Vector Search: Linear Scan
We will implement an in-memory linear scan or a stream-based linear scan.
*   **Why:** For small-to-medium datasets (<100k), scanning `bbolt` keys and computing cosine similarity is fast enough and requires no extra complex dependencies (like `faiss`).
*   **Future:** Could look into `kelindar/search` for HNSW indexing if needed.

### 4. Crawling: Native Go
We will use standard `net/http` and `encoding/xml`.
*   **Why:** Go's standard library is excellent for this. We reuse the flattening logic designed for the XML Input mode.

## Architecture

We will create a Go module structure in `src/unifyweaver/targets/go_runtime/`.

```
go_runtime/
├── embedder/
│   └── hugot_client.go    # Wraps hugot pipeline
├── storage/
│   └── bbolt_store.go     # Wraps bbolt for objects/vectors
├── search/
│   └── vector_search.go   # Cosine similarity logic
└── crawler/
    └── crawler.go         # Fetch loop + XML flattening
```

## Integration Strategy

The `go_target.pl` compiler will be updated to:
1.  **Detect Semantic Predicates**: `semantic_search/3`, `crawl/2`.
2.  **Inject Runtime**: Unlike Python (where we can inline easily), for Go we might generate a `go.mod` file or expect the user to have the runtime in their path.
    *   *Decision:* We will generate a **standalone main file** that *includes* the necessary logic inline (simplified versions) OR imports a shared library if we decide to publish one.
    *   *Initial approach:* Generate a `main.go` that imports the necessary 3rd party libs (`hugot`, `bbolt`) and contains our helper structs/funcs inline (or in a generated `runtime.go` sidecar).

## Example Generated Code

```go
package main

import (
    "github.com/knights-analytics/hugot"
    "github.com/knights-analytics/hugot/pipelines"
    bolt "go.etcd.io/bbolt"
)

func main() {
    // Initialize pipelines
    session, _ := hugot.NewSession()
    defer session.Destroy()
    
    embPipeline, _ := session.NewFeatureExtractionPipeline(...)
    
    // Run search
    results := SemanticSearch(embPipeline, "query", 10)
}
```
