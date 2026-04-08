# Semantic Search Interface Specification (v1.0)

## 1. Declarative API

### 1.1 The `semantic_provider/2` Directive
Used to define the compilation strategy for a semantic search predicate.

```prolog
:- semantic_provider(Predicate/Arity, Options).
```

### 1.2 Configuration Options

| Option | Description |
|--------|-------------|
| `targets(List)` | Target-specific configurations (see 1.3). |
| `fallback(List)` | Fallback configuration used if the target is not in the list. |
| `device(Device)` | Preferred execution device: `auto`, `gpu`, `cpu`. Defaults to `auto`. |
| `model(Model)` | Default model name (e.g., `all-MiniLM-L6-v2`). |

### 1.3 Target-Specific Options
Each entry in the `targets` list has the form `target(Name, TargetOptions)`.

| Option | Description |
|--------|-------------|
| `provider(P)` | Backend provider (e.g., `transformers`, `candle`, `onnx`, `hugot`). |
| `model(M)` | Overrides the global model name for this target. |
| `device(D)` | Overrides the global device preference for this target. |
| `index(Path)` | Path to the vector database or index file. |

## 2. Compilation Logic

### 2.1 Predicate Recognition
The compiler identifies calls to `semantic_search/3` (and custom predicates registered via `semantic_provider/2`) and dispatches them to the `semantic_compiler` module.

### 2.2 Device Fallback Logic
The compiler emits initialization code that attempts to use the preferred device. If `device(gpu)` is specified but the target environment lacks GPU support, the emitted code will fall back to `cpu` automatically if the target supports it (e.g., ONNX Runtime, Candle).

### 2.3 `semantic_dispatch/5` Multifile Hooks
Targets implement this hook to generate language-specific initialization and invocation code.

```prolog
semantic_dispatch(+Target, +Goal, +ProviderInfo, +VarMap, -Code)
```

## 3. Supported Providers & Targets

| Target | Default Provider | Alternate Providers | Device Support |
|--------|------------------|---------------------|----------------|
| **Python** | `transformers` | `onnx` | CUDA, MPS, CPU |
| **Go** | `hugot` | `candle` (FFI), `ort` (FFI) | CUDA, CPU |
| **Rust** | `candle` | `onnx` | CUDA, Metal, CPU |
| **C#** | `onnx` | — | DirectML, CUDA, CPU |
| **C/C++** | `onnx` | `libtorch` | CUDA, CPU |
