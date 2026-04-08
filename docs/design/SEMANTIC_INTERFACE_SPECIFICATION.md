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
| **Python** | `transformers` | `onnx` | CUDA, CPU (MPS planned) |
| **Go** | `hugot` | `onnx` | GPU, CPU |
| **Rust** | `candle` | `onnx` | CUDA, CPU |
| **C#** | `onnx` | — | DirectML, CPU |

## 4. Fuzzy Logic Compilation

### 4.1 Overview

The `semantic_compiler` also provides a `fuzzy_dispatch/3` multifile hook for compiling fuzzy logic operations across targets. This enables the Prolog fuzzy DSL (`f_and`, `f_or`, `f_dist_or`, `f_union`, `f_not`) to be compiled to target-specific code.

### 4.2 Fuzzy Dispatch Hook

```prolog
:- multifile semantic_compiler:fuzzy_dispatch/3.
% fuzzy_dispatch(+Target, +Goal, -Code)
```

### 4.3 Supported Operations

| Operation | Arity | Description | Formula |
|-----------|-------|-------------|---------|
| `f_and` | 2 | Product t-norm | w1\*t1 \* w2\*t2 \* ... |
| `f_or` | 2 | Probabilistic sum | 1 - (1-w1\*t1)(1-w2\*t2)... |
| `f_dist_or` | 3 | Distributed OR | 1 - (1-base\*w1\*t1)(1-base\*w2\*t2)... |
| `f_union` | 3 | Non-distributed OR | base \* f\_or result |
| `f_not` | 2 | Complement | 1 - score |
| `blend_scores` | 4 | Weighted interpolation | alpha\*s1 + (1-alpha)\*s2 |
| `top_k` | 3 | Top-K selection | Sort by score descending |

### 4.4 Target Support

| Target | Core ops | Batch ops | Utility ops |
|--------|----------|-----------|-------------|
| **Python** | f\_and, f\_or, f\_dist\_or, f\_union, f\_not | f\_and\_batch, f\_or\_batch, f\_dist\_or\_batch | blend, top\_k, filter, boost |
| **Go** | f\_and, f\_or, f\_dist\_or, f\_union, f\_not | — | blend, top\_k |

### 4.5 Usage Example

```prolog
% Define weighted terms
Terms = [w(bash, 0.9), w(shell, 0.5)],

% Compile fuzzy AND for Go
compile_fuzzy_call(go, f_and(Terms, Result), Code).
% Code generates:
%   result := 1.0
%   result *= 0.9 * termScores["bash"]
%   result *= 0.5 * termScores["shell"]
```
