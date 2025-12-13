# Go Embedder Backends - Semantic Search with Multiple Runtime Options

The Go embedder package provides text embedding for semantic search with multiple backend options, ranked by ease of installation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Backend Comparison](#backend-comparison)
4. [Backend 1: Pure Go (Easiest)](#backend-1-pure-go-easiest)
5. [Backend 2: Candle/Rust (Recommended)](#backend-2-candlerust-recommended)
6. [Backend 3: ONNX Runtime (ORT)](#backend-3-onnx-runtime-ort)
7. [Backend 4: XLA/PJRT (Most Complex)](#backend-4-xlapjrt-most-complex)
8. [Model Setup](#model-setup)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

**Just want it working? Use Pure Go (no dependencies):**

```bash
# Download the model
pip install huggingface_hub
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2

# Build and run
go build -o myapp ./examples/pearltrees_go
MODEL_PATH="models/all-MiniLM-L6-v2" ./myapp --search "your query"
```

**Want best performance? Use Candle:**

```bash
# Install Rust and build the library
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
git clone https://github.com/vllm-project/semantic-router.git /tmp/semantic-router
cd /tmp/semantic-router/candle-binding && cargo build --release
sudo cp target/release/libcandle_semantic_router.so /usr/local/lib/ && sudo ldconfig

# Build and run
go build -tags candle -o myapp ./examples/pearltrees_go
MODEL_PATH="sentence-transformers/all-MiniLM-L6-v2" LD_LIBRARY_PATH=/usr/local/lib ./myapp --search "your query"
```

---

## Overview

The embedder package uses Go build tags to select backends at compile time:

```bash
# Pure Go (default - no build tag needed)
go build ./...

# Candle (Rust FFI)
go build -tags candle ./...

# ONNX Runtime
go build -tags ORT ./...

# XLA/PJRT
go build -tags XLA ./...
```

All backends implement the same `Embedder` interface:

```go
type Embedder interface {
    Embed(text string) ([]float32, error)
    Close()
}
```

---

## Backend Comparison

| Backend | Difficulty | Performance | GPU Support | Dependencies |
|---------|------------|-------------|-------------|--------------|
| Pure Go | Easy | Moderate | No | None (pure Go) |
| Candle | Easy-Medium | **Fastest** | CUDA | Rust toolchain, libcandle_semantic_router.so |
| ORT | Medium | Moderate | CUDA | ONNX Runtime 1.22+, libtokenizers.a |
| XLA | Hard | Slow (CPU) | CUDA/TPU/Metal | PJRT libraries, C++ toolchain |

**Recommendation:** Start with Pure Go for development, use Candle for production.

### Performance Benchmarks

Tested on GTX 1660 Ti, all-MiniLM-L6-v2 model, 5 queries x 3 runs averaged:

| Backend | CPU Time | GPU Time | Notes |
|---------|----------|----------|-------|
| **Candle** | **0.21s** | 0.36s | Fastest on CPU |
| Pure Go | 0.34s | N/A | No GPU support |
| ORT | 0.45s | 0.44s | Marginal GPU gain |
| XLA | 0.70s | ~0.72s | WSL2 may fall back to CPU |

**GPU Note:** For single-query workloads, CPU is often faster due to GPU transfer overhead. The embedding model is small enough that CPU inference completes before GPU data transfer finishes. GPU acceleration benefits appear with batch processing (multiple embeddings at once) or larger models.

---

## Backend 1: Pure Go (Easiest)

The pure Go backend uses the `hugot` library with its built-in Go ONNX runtime. No external dependencies required.

### Installation

```bash
# No additional installation needed - just build
go build ./...
```

### Pros
- Zero external dependencies
- Cross-platform (works everywhere Go works)
- Simplest setup

### Cons
- CPU only (no GPU acceleration)
- Slower than native backends for large batches

### Usage

```bash
# Build (default, no tags)
go build -o myapp ./examples/pearltrees_go

# Run
./myapp --search "quantum physics"
```

---

## Backend 2: Candle/Rust (Recommended)

The Candle backend uses the Rust `candle` ML library via FFI. It provides excellent performance with optional GPU support.

### Prerequisites

1. **Rust toolchain**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Clone and build semantic-router library**
   ```bash
   git clone https://github.com/vllm-project/semantic-router.git /tmp/semantic-router
   cd /tmp/semantic-router/candle-binding
   cargo build --release
   ```

3. **Install the shared library**
   ```bash
   sudo cp target/release/libcandle_semantic_router.so /usr/local/lib/
   sudo ldconfig
   ```

### Build

```bash
go build -tags candle -o myapp ./examples/pearltrees_go
```

### GPU Support (CUDA)

For GPU acceleration, build semantic-router with CUDA:

```bash
cd /tmp/semantic-router/candle-binding
cargo build --release --features cuda
sudo cp target/release/libcandle_semantic_router.so /usr/local/lib/
sudo ldconfig
```

Then run with:
```bash
USE_GPU=1 ./myapp --search "machine learning"
```

### Model Path

Candle can use HuggingFace model IDs directly:
```bash
# Uses HuggingFace Hub (downloads on first use)
MODEL_PATH="sentence-transformers/all-MiniLM-L6-v2" ./myapp --search "test"

# Or local safetensors path
MODEL_PATH="/path/to/model" ./myapp --search "test"
```

### Pros
- Fast inference (native Rust performance)
- GPU support via CUDA
- Can download models from HuggingFace automatically
- Single library dependency

### Cons
- Requires Rust toolchain to build
- Linux/macOS only (no Windows support yet)

---

## Backend 3: ONNX Runtime (ORT)

The ORT backend uses Microsoft's ONNX Runtime for inference. Requires C dependencies but provides excellent performance.

### Prerequisites

#### 1. Install ONNX Runtime 1.22.0+

**Important:** ONNX Runtime version must be 1.22.0 or newer (API version 22 required by hugot).

```bash
# Download ONNX Runtime
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xzf onnxruntime-linux-x64-1.22.0.tgz

# Install libraries
sudo cp onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0 /usr/local/lib/

# Create required symlinks (order matters!)
cd /usr/local/lib
sudo ln -sf libonnxruntime.so.1.22.0 libonnxruntime.so.1
sudo ln -sf libonnxruntime.so.1 libonnxruntime.so
sudo ln -sf libonnxruntime.so onnxruntime.so  # hugot looks for this name
sudo ldconfig

# Install headers (for building)
sudo cp -r /tmp/onnxruntime-linux-x64-1.22.0/include/* /usr/local/include/
```

#### 2. Build libtokenizers.a

The tokenizers library must be built from the daulet/tokenizers fork (not knights-analytics/tokenizers):

```bash
# Clone the correct repository
git clone https://github.com/daulet/tokenizers.git /tmp/daulet-tokenizers
cd /tmp/daulet-tokenizers

# Build the Rust library
cargo build --release

# Install the static library
sudo cp target/release/libtokenizers.a /usr/local/lib/
```

### Build

```bash
# Note: Use uppercase ORT (required by hugot library)
go build -tags ORT -o myapp ./examples/pearltrees_go
```

### Usage

```bash
# Must set LD_LIBRARY_PATH for runtime
LD_LIBRARY_PATH=/usr/local/lib ./myapp --search "quantum physics"
```

### Model Path

ORT requires a local directory containing both `model.onnx` and `tokenizer.json`:

```bash
# Point to directory with both model.onnx and tokenizer.json
MODEL_PATH="../../models/all-MiniLM-L6-v2" LD_LIBRARY_PATH=/usr/local/lib ./myapp --search "test"
```

### Common Issues

1. **API version mismatch**: "requested ORT API version 22 is not available"
   - Solution: Upgrade to ONNX Runtime 1.22.0+

2. **Library not found**: "cannot open shared object file"
   - Solution: Set `LD_LIBRARY_PATH=/usr/local/lib` or run `sudo ldconfig`

3. **Multiple ONNX files**: "multiple .onnx file detected"
   - Solution: The embedder now defaults to `model.onnx`. Ensure your model directory structure is correct.

4. **Tokenizer not found**: "feature extraction pipeline requires a tokenizer"
   - Solution: Ensure `tokenizer.json` is in the same directory as `model.onnx`

### GPU Support (CUDA)

For GPU acceleration with ORT:

```bash
# Download GPU version of ONNX Runtime
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz
tar xzf onnxruntime-linux-x64-gpu-1.22.0.tgz

# Install GPU libraries
sudo cp onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime_providers_cuda.so /usr/local/lib/
sudo cp onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime_providers_shared.so /usr/local/lib/

# Install cuDNN 9 (required for CUDA provider)
sudo apt install libcudnn9-cuda-12

sudo ldconfig
```

Then run with:
```bash
USE_GPU=1 LD_LIBRARY_PATH=/usr/local/lib ./myapp --search "machine learning"
```

### Pros
- Fast inference
- GPU support (with CUDA build)
- Well-maintained by Microsoft

### Cons
- Complex installation (multiple dependencies)
- Version sensitivity (API v22 requirement)
- Requires runtime library path configuration
- GPU requires cuDNN 9

---

## Backend 4: XLA/PJRT (Most Complex)

The XLA backend uses Google's XLA compiler via PJRT. Optimized for hardware accelerators (TPU, GPU), but slower on CPU due to compilation overhead.

### Prerequisites

#### CPU-only Install

```bash
# Option 1: System-wide (requires sudo)
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_linux_amd64.sh | bash

# Option 2: Local install (no sudo) - RECOMMENDED
GOPJRT_INSTALL_DIR=$HOME/.local GOPJRT_NOSUDO=1 \
  bash -c 'curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_linux_amd64.sh | bash'
```

The local install places libraries in `~/.local/lib/` (~250MB PJRT plugin).

#### GPU/CUDA Install (Complex)

XLA CUDA support requires Python and JAX. This is significantly more complex than other backends:

**Option 1: Official install script** (requires python3-venv):
```bash
sudo apt install python3-venv python3-dev
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_cuda.sh | bash
```

**Option 2: Manual install with Python 3.9+** (if python3-venv not available):
```bash
# Create venv with newer Python
python3.9 -m venv /tmp/jax_cuda_venv
source /tmp/jax_cuda_venv/bin/activate
pip install --upgrade pip
pip install "jax[cuda12]"

# Copy CUDA PJRT plugin
mkdir -p ~/.local/lib/gomlx/pjrt
cp /tmp/jax_cuda_venv/lib/python3.9/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so \
   ~/.local/lib/gomlx/pjrt/pjrt_c_api_cuda_plugin.so

# Copy nvidia libraries
cp -r /tmp/jax_cuda_venv/lib/python3.9/site-packages/nvidia/* ~/.local/lib/gomlx/nvidia/

# Cleanup
rm -rf /tmp/jax_cuda_venv
```

**Note:** XLA GPU may not work properly in WSL2 due to `/dev/nvidia*` device detection issues. For GPU acceleration in WSL2, **Candle or ORT are better alternatives**.

### Build

```bash
# With system-wide install
go build -tags XLA -o myapp ./examples/pearltrees_go

# With local install
CGO_LDFLAGS="-L$HOME/.local/lib" go build -tags XLA -o myapp ./examples/pearltrees_go
```

### Usage

```bash
# System-wide
./myapp --search "quantum physics"

# Local install
LD_LIBRARY_PATH=$HOME/.local/lib ./myapp --search "quantum physics"
```

### Status

XLA backend is **working** but slower than other backends on CPU (0.67s vs 0.21s for Candle). This is expected - XLA is optimized for:
- TPU inference
- GPU batch processing
- Large model compilation

### Pros
- Supports TPU, Metal (Apple Silicon)
- Optimizing compiler (benefits large models)
- Best for batch processing on accelerators

### Cons
- Slowest on CPU (compilation overhead)
- Most complex installation
- Large dependencies (~250MB PJRT plugin)
- CUDA install requires python3-venv and JAX

---

## Model Setup

### Download all-MiniLM-L6-v2

The recommended model is `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).

#### For Pure Go / ORT / XLA

Download the full model with ONNX files:

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2

# Or using git lfs
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2
```

Required files:
- `model.onnx` (or in `onnx/` subdirectory)
- `tokenizer.json`
- `config.json`

#### For Candle

Candle can download models automatically from HuggingFace:

```bash
# Just set the model ID - downloads on first use
MODEL_PATH="sentence-transformers/all-MiniLM-L6-v2" ./myapp
```

Or use a local safetensors model:
```bash
MODEL_PATH="/path/to/model/with/model.safetensors" ./myapp
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model directory or HuggingFace ID | Backend-specific default |
| `USE_GPU` | Enable GPU (`1` or `true`) | `false` |

### EmbedderConfig

```go
type EmbedderConfig struct {
    ModelPath    string  // Path to model directory
    ModelName    string  // Model identifier
    OnnxFilename string  // Specific ONNX file (default: "model.onnx")
    Dimensions   int     // Output dimensions (default: 384)
    UseGPU       bool    // Enable GPU acceleration
    MaxLength    int     // Max token length (default: 512)
}
```

### Build Tags Reference

| Tag | Backend | Session Type |
|-----|---------|--------------|
| (none) | Pure Go | `hugot.NewGoSession()` |
| `candle` | Candle/Rust | FFI to libcandle_semantic_router |
| `ORT` or `ort` | ONNX Runtime | `hugot.NewORTSession()` |
| `XLA` or `xla` | XLA/PJRT | `hugot.NewXLASession()` |

---

## Troubleshooting

### Check Which Backend is Active

```go
backend := embedder.AvailableBackend()
fmt.Printf("Using backend: %s\n", backend)
```

Or at runtime:
```bash
./myapp --search "test" 2>&1 | grep Backend
# Output: Backend: ort
```

### Fallback to Stub Embedder

If the embedder fails to load, the system automatically falls back to a stub embedder that returns random vectors. This allows the application to run but with degraded search quality.

```
Warning: Could not load embedder (ort): ...
Falling back to stub embedder...
```

### Library Path Issues

```bash
# Check if libraries are found
ldd ./myapp | grep -E "onnx|candle|tokenizer|pjrt"

# Check for missing libraries
ldd ./myapp | grep "not found"

# Add to library path permanently
echo '/usr/local/lib' | sudo tee /etc/ld.so.conf.d/local.conf
sudo ldconfig

# For XLA local install
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
```

### Build Tag Verification

```bash
# Verify build tags are applied
go build -tags ORT -v ./... 2>&1 | grep embedder

# Should show embedder_ort.go being compiled, NOT embedder_purego.go
```

### Common Errors by Backend

#### Candle
- **"libcandle_semantic_router.so: cannot open"**: Library not installed or not in LD_LIBRARY_PATH
  ```bash
  sudo ldconfig
  # or
  LD_LIBRARY_PATH=/usr/local/lib ./myapp
  ```

#### ORT
- **"ORT API version 22 is not available"**: Wrong ONNX Runtime version
  ```bash
  # Must use v1.22.0+
  wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
  ```

- **"libcudnn.so.9: cannot open"**: Missing cuDNN for GPU
  ```bash
  sudo apt install libcudnn9-cuda-12
  sudo ldconfig
  ```

- **"multiple .onnx file detected"**: Model directory has multiple ONNX files
  - The embedder defaults to `model.onnx`. Ensure only one exists or it's named correctly.

#### XLA
- **"libpjrt_c_api_cpu_plugin.so: cannot open"**: PJRT not installed
  ```bash
  # Local install
  GOPJRT_INSTALL_DIR=$HOME/.local GOPJRT_NOSUDO=1 \
    bash -c 'curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_linux_amd64.sh | bash'
  ```

- **"ml-dtypes build failed"**: Missing Python headers for CUDA install
  ```bash
  sudo apt install python3-dev
  ```

### Verify GPU is Being Used

```bash
# Check GPU memory before/after running
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Run with GPU enabled
USE_GPU=1 LD_LIBRARY_PATH=/usr/local/lib ./myapp --search "test"

# Check GPU memory again - should increase if GPU is active
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

**Note:** For small models like all-MiniLM-L6-v2, CPU may be faster than GPU for single queries due to data transfer overhead.

---

## See Also

- [GO_TARGET.md](GO_TARGET.md) - Go target compiler documentation
- [hugot library](https://github.com/knights-analytics/hugot) - Underlying embedding library
- [semantic-router](https://github.com/vllm-project/semantic-router) - Candle-based embedding library
