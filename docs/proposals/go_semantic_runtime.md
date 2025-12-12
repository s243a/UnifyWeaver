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

| Backend | Build Tag | Dependencies | GPU Support | Performance | Portability |
|---------|-----------|--------------|-------------|-------------|-------------|
| **Pure Go** | (default) | None | No | ~100/sec | Most portable |
| **Candle** | `-tags candle` | Rust | Yes (CUDA) | ~500/sec | Very portable |
| **ORT** | `-tags ort` | C (ONNX Runtime) | Yes (CUDA) | ~1000/sec | Moderate |
| **XLA** | `-tags xla` | C++ (PJRT) | Yes (CUDA/Metal/TPU) | ~1000/sec | Least portable |

**Current default:** Pure Go backend - no external dependencies required.

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

### Prerequisites (All Backends)

```bash
# 1. Download embedding model
mkdir -p models
cd models
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 all-MiniLM-L6-v2-onnx
cd ..

# 2. Verify Go version (1.21+ required)
go version
```

---

### Option 1: Pure Go (Most Portable - No Dependencies)

**Best for:** Simple deployment, cross-compilation, environments without C/Rust toolchains

```bash
# Build (no special tags needed)
cd examples/pearltrees_go
go build -o pearltrees_go ./...

# Run
./pearltrees_go --search "quantum physics"
```

**Pros:**
- No external dependencies
- Cross-compiles to any GOOS/GOARCH
- Simple CI/CD deployment
- Works in minimal containers

**Cons:**
- CPU only (no GPU acceleration)
- Slower inference (~100 embeddings/sec)
- Best for small batches (<32 inputs)

---

### Option 2: Candle Backend (Go + Rust)

**Best for:** GPU acceleration without C dependencies, good balance of portability and performance

**Prerequisites:**
```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify Rust version (1.70+ required)
rustc --version
```

**Build the Candle library:**
```bash
# Clone and build candle-semantic-router
git clone https://github.com/vllm-project/semantic-router
cd semantic-router/candle-binding
cargo build --release

# Copy library to system path
sudo cp target/release/libcandle_semantic_router.so /usr/local/lib/
sudo ldconfig

# Or set LD_LIBRARY_PATH for local use
export LD_LIBRARY_PATH=$(pwd)/target/release:$LD_LIBRARY_PATH
```

**Build Go with Candle:**
```bash
cd examples/pearltrees_go
go build -tags candle -o pearltrees_go ./...
./pearltrees_go --search "quantum physics"
```

**For GPU support (CUDA):**
```bash
# Build with CUDA features
cd semantic-router/candle-binding
cargo build --release --features cuda

# Ensure CUDA is installed
nvidia-smi  # Should show your GPU
```

**Pros:**
- GPU acceleration (CUDA)
- No C dependencies (Rust only)
- Good performance (~500 embeddings/sec CPU, faster on GPU)
- Portable Rust binary

**Cons:**
- Requires Rust toolchain
- Larger binary size
- CUDA requires Nvidia drivers

---

### Option 3: ONNX Runtime Backend (Go + C)

**Best for:** Maximum CPU performance, production workloads

**Prerequisites:**
```bash
# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
sudo ldconfig

# Install Rust (for building tokenizers)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Build libtokenizers:**
```bash
# IMPORTANT: Use daulet/tokenizers, NOT knights-analytics/tokenizers
# The knights-analytics fork exports different symbols that won't link correctly
git clone https://github.com/daulet/tokenizers
cd tokenizers
cargo build --release
sudo cp target/release/libtokenizers.a /usr/local/lib/
```

**Build Go with ORT:**
```bash
cd examples/pearltrees_go
go build -tags ort -o pearltrees_go ./...
./pearltrees_go --search "quantum physics"
```

**For GPU support (CUDA):**
```bash
# Download CUDA version of ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-gpu-1.16.0.tgz
tar xzf onnxruntime-linux-x64-gpu-1.16.0.tgz
sudo cp onnxruntime-linux-x64-gpu-1.16.0/lib/* /usr/local/lib/
sudo ldconfig
```

**Pros:**
- Fastest CPU inference (~1000+ embeddings/sec)
- GPU acceleration (CUDA)
- Battle-tested ONNX Runtime

**Cons:**
- Requires C dependencies
- Complex setup (ONNX Runtime + libtokenizers)
- Symbol mismatch issues with wrong tokenizers fork

---

### Option 4: XLA Backend (Go + C++)

**Best for:** Advanced hardware support (TPU, Metal), training/fine-tuning

**Prerequisites:**
```bash
# Install PJRT libraries from GoMLX
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_linux_amd64.sh | bash

# For CUDA GPU support
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_cuda_linux_amd64.sh | bash
```

**Build Go with XLA:**
```bash
cd examples/pearltrees_go
go build -tags xla -o pearltrees_go ./...
./pearltrees_go --search "quantum physics"
```

**Pros:**
- Broadest hardware support (CPU, CUDA, Metal, TPU)
- JIT compilation for optimized inference
- Supports fine-tuning

**Cons:**
- Most complex setup
- Largest dependencies
- Requires C++ libraries

---

## Backend Selection Guide

| Use Case | Recommended Backend |
|----------|---------------------|
| Development/Testing | Pure Go |
| Simple deployment | Pure Go |
| Cross-compilation needed | Pure Go |
| GPU acceleration (simple) | Candle |
| Maximum CPU performance | ORT |
| GPU + production scale | ORT or XLA |
| TPU or Apple Metal | XLA |
| Fine-tuning models | XLA |

---

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

---

## Troubleshooting

### Pure Go: "model not found"
Ensure the model path points to a directory containing `model.onnx` and tokenizer files.

### Candle: "libcandle_semantic_router.so not found"
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# Or add to ~/.bashrc
```

### ORT: "undefined reference to tokenizers_encode"
You're using the wrong tokenizers library. Use `daulet/tokenizers`, not `knights-analytics/tokenizers`.

### ORT: "onnxruntime.so not found"
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
sudo ldconfig
```

### XLA: "PJRT plugin not found"
Re-run the GoMLX installation script for your platform.
