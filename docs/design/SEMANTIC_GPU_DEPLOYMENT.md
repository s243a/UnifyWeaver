# GPU Deployment Guide — Semantic Search

## Quick Start

Declare a provider with GPU device selection:

```prolog
:- semantic_provider(find_similar/3, [
    targets([
        target(python, [provider(transformers), model('all-MiniLM-L6-v2'), device(gpu)]),
        target(go, [provider(hugot), model('all-MiniLM-L6-v2'), device(gpu)]),
        target(rust, [provider(candle), model('all-MiniLM-L6-v2'), device(gpu)]),
        target(csharp, [provider(onnx), model('all-MiniLM-L6-v2'), device(gpu)]),
        target(elixir, [provider(bumblebee), model('all-MiniLM-L6-v2'), device(gpu)])
    ]),
    fallback([provider(onnx), model('all-MiniLM-L6-v2')])
]).
```

Every target generates code with automatic CPU fallback if the GPU is unavailable at runtime.

## Device Options

| Device | Description |
|--------|-------------|
| `gpu` | Prefer GPU (CUDA on NVIDIA, DirectML on Windows, Metal/MPS on macOS) |
| `mps` | Apple Silicon GPU specifically (Python only) |
| `cpu` | Force CPU execution |
| `auto` | Auto-detect best available device (default) |

## Per-Target Runtime Requirements

### Python (transformers)

```
pip install torch sentence-transformers
```

| Device | Backend | Fallback chain |
|--------|---------|---------------|
| `gpu` | CUDA | CUDA → MPS → CPU |
| `mps` | MPS | MPS → CPU |
| `cpu` | CPU | — |
| `auto` | Best available | CUDA → MPS → CPU |

Generated code checks `torch.cuda.is_available()` and `torch.backends.mps.is_available()` at runtime.

### Go (hugot)

```
go get github.com/knights-analytics/hugot
```

| Device | Backend | Fallback |
|--------|---------|----------|
| `gpu` | `WithGPU()` | Retries with `WithCPU()` on error |
| `cpu` | `WithCPU()` | — |
| `auto` | Default | Library chooses |

Generated code uses Go error handling: if GPU init fails, logs a warning and retries with CPU.

### Rust (candle / onnx)

```toml
# Cargo.toml
candle-core = { version = "0.6", features = ["cuda"] }
# or
ort = { version = "2", features = ["cuda"] }
```

| Device | Backend | Fallback |
|--------|---------|----------|
| `gpu` | `Device::new_cuda(0)` | `unwrap_or_else` → `Device::Cpu` |
| `cpu` | `Device::Cpu` | — |
| `auto` | `cuda_if_available(0)` | Falls back to CPU |

### C# (ONNX Runtime)

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime.DirectML" />
```

| Device | Backend | Fallback |
|--------|---------|----------|
| `gpu` | `AppendExecutionProvider_DML()` | try/catch → `AppendExecutionProvider_CPU()` |
| `cpu` | `AppendExecutionProvider_CPU()` | — |
| `auto` | CPU only | — |

### Elixir (Bumblebee/Nx)

```elixir
# mix.exs deps
{:bumblebee, "~> 0.5"},
{:nx, "~> 0.7"},
{:exla, "~> 0.7"}
```

| Device | Backend | Fallback |
|--------|---------|----------|
| `gpu` | `{EXLA.Backend, client: :cuda}` | Manual — set to `:host` if CUDA unavailable |
| `cpu` | `EXLA.Backend` | — |
| `auto` | Default Nx backend | — |

## Inline Options (semantic_search/4)

Override provider defaults per call:

```prolog
semantic_search(Query, 10, Results, [
    threshold(0.7),           % minimum similarity score
    model('large-model'),     % override model name
    index("custom.db")        % override vector database path
]).
```

Inline options merge with the provider config at compile time. They take precedence over the `semantic_provider` declaration.

## Vector Database Configuration

Use `input_source` to specify where the vector index lives:

```prolog
compile_predicate_to_python(find_similar/3, [
    input(vector_db("embeddings.db", sqlite))
], Code).
```

Supported formats: `auto`, `sqlite`, `redb`, `faiss`. The `auto` format lets the target choose based on file extension.
