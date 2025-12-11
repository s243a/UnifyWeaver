# Go Semantic Runtime with Hugot

## Status: Implemented

The Go semantic runtime is fully implemented with real embeddings support.

## Overview
The Go runtime provides semantic crawling, embedding generation, and vector similarity search with feature parity to C#, Python, and Rust targets.

## Core Decisions

### 1. Embeddings: `knights-analytics/hugot`
We use [hugot](https://github.com/knights-analytics/hugot) for embedding generation.
*   **Why:** High-level, "Python-like" pipeline API for Transformers in Go.
*   **Model:** `all-MiniLM-L6-v2` (ONNX format, 384 dimensions).

#### Backend Options

| Backend | Function | CGO Required | Performance | Use Case |
|---------|----------|--------------|-------------|----------|
| **Pure Go** | `hugot.NewGoSession()` | No | Slower | Simple deployment, no dependencies |
| **ONNX Runtime** | `hugot.NewORTSession()` | Yes | Faster | Production, requires libtokenizers |

**Current default:** Pure Go backend (`NewGoSession()`) - no C dependencies required.

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

## Example Code

```go
package main

import (
    "github.com/knights-analytics/hugot"
    bolt "go.etcd.io/bbolt"

    "unifyweaver/targets/go_runtime/embedder"
    "unifyweaver/targets/go_runtime/search"
    "unifyweaver/targets/go_runtime/storage"
)

func main() {
    // Initialize store
    store, _ := storage.NewStore("pearltrees.bolt")
    defer store.Close()

    // Initialize embedder (pure Go - no C dependencies)
    emb, _ := embedder.NewHugotEmbedder("models/all-MiniLM-L6-v2-onnx", "all-MiniLM-L6-v2")
    defer emb.Close()

    // Bookmark filing suggestions
    filer := search.NewBookmarkFiler(store, store, emb)
    result, _ := filer.SuggestBookmarks("quantum physics", 3)
    fmt.Println(result)
}
```

## Installation

### Option 1: Pure Go (Recommended - No Dependencies)

This is the simplest option with no C dependencies:

```bash
# 1. Get the model
mkdir -p models
# Download all-MiniLM-L6-v2 ONNX model from Hugging Face

# 2. Build and run
cd examples/pearltrees_go
go build -o pearltrees_go ./...
./pearltrees_go --search "quantum physics"
```

**Pros:** Simple deployment, cross-compiles easily, no CGO
**Cons:** Slower inference (~100 embeddings/sec vs 1000+/sec with ORT)

### Option 2: ONNX Runtime Backend (Faster, Requires C)

For production workloads requiring higher throughput:

```bash
# 1. Install ONNX Runtime
# Ubuntu/Debian:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
sudo ldconfig

# 2. Install libtokenizers (from daulet/tokenizers, not knights-analytics fork)
git clone https://github.com/daulet/tokenizers
cd tokenizers
cargo build --release
sudo cp target/release/libtokenizers.a /usr/local/lib/

# 3. Build with ORT tag
go build -tags ORT -o pearltrees_go ./...
```

**Note:** The knights-analytics/tokenizers fork exports different symbols and won't link correctly. Use daulet/tokenizers.

### Option 3: Alternative Embedding Libraries

| Library | Language | Notes |
|---------|----------|-------|
| **hugot (pure Go)** | Go | Current default, no deps |
| **hugot (ORT)** | Go+C | Faster, needs libtokenizers |
| **candle** | Rust | Used by Rust target |
| **ONNX Runtime** | C# | Used by C# target |

## Tested Results

With pure Go backend on all-MiniLM-L6-v2:

```
Query: "quantum physics"
→ "Quantum Mechanics" (similarity: 0.902)

Query: "python programming"
→ "python" (similarity: 0.907)

Query: "machine learning tutorials"
→ "Machine Learning" (similarity: 0.631)
```
