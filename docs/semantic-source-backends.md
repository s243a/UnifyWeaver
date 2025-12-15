# Semantic Source - Target × Backend Matrix

The semantic source plugin supports multiple target languages and embedding backends, allowing flexible deployment options for semantic search capabilities.

## Supported Combinations

| Target     | Backend        | Status | Description                                    |
|------------|----------------|--------|------------------------------------------------|
| bash       | python_onnx    | ✅     | Bash wrapper with embedded Python ONNX backend |
| bash       | go_service     | ✅     | Bash wrapper calling Go HTTP service           |
| bash       | rust_candle    | ✅     | Bash wrapper calling Rust Candle binary        |
| python     | python_onnx    | ✅     | Native Python with ONNX Runtime                |
| python     | go_service     | ✅     | Python calling Go HTTP service                 |
| python     | rust_candle    | ✅     | Python calling Rust Candle binary              |
| powershell | python_onnx    | ✅     | PowerShell with embedded Python ONNX           |
| powershell | go_service     | ✅     | PowerShell calling Go HTTP service             |
| powershell | rust_candle    | ✅     | PowerShell calling Rust Candle binary          |
| csharp     | csharp_native  | ✅     | Native C# with Microsoft.ML.OnnxRuntime        |
| csharp     | go_service     | ✅     | C# calling Go HTTP service                     |
| csharp     | rust_candle    | ✅     | C# calling Rust Candle binary                  |

## Backends

### 1. python_onnx
**Description**: Uses Python with ONNX Runtime for embedding generation.

**Requirements**:
- Python 3.x
- onnxruntime
- transformers (for tokenization)
- numpy

**Backend Config**:
```prolog
% Simple (uses defaults)
backend_config(default)

% Custom model
backend_config('all-MiniLM-L6-v2')

% Full config
backend_config([
    model_path('models/model.onnx'),
    vocab_path('models/vocab.txt'),
    dimensions(384),
    max_length(512)
])
```

**Target Support**:
- `bash`: Embeds Python code in bash script via heredoc
- `python`: Native Python implementation
- `powershell`: Embeds Python code in PowerShell script

**Example**:
```prolog
:- source(semantic, research_papers, [
    vector_store('papers.json'),
    embedding_backend(python_onnx),
    backend_config(default),
    threshold(0.7),
    top_k(10)
]).
```

### 2. go_service
**Description**: Calls a Go HTTP service for embedding and search.

**Requirements**:
- Running Go service on specified URL
- HTTP client in target language (curl, urllib, Invoke-RestMethod, HttpClient)

**Backend Config**:
```prolog
backend_config([url('http://localhost:8080')])
```

**API Contract**:
```json
POST /search
Content-Type: application/json

{
  "query": "search query",
  "top_k": 5,
  "threshold": 0.6,
  "metric": "cosine",
  "vector_store": "path/to/vectors.json"
}

Response:
{
  "results": [
    {"id": "doc1", "score": 0.95},
    {"id": "doc2", "score": 0.87}
  ]
}
```

**Target Support**:
- `bash`: Uses curl + jq
- `python`: Uses urllib.request
- `powershell`: Uses Invoke-RestMethod
- `csharp`: Uses HttpClient with async/await

**Example**:
```prolog
:- source(semantic, research_papers, [
    vector_store('papers.json'),
    embedding_backend(go_service),
    backend_config([url('http://localhost:8080')]),
    threshold(0.7),
    top_k(10)
]).
```

### 3. rust_candle
**Description**: Calls a Rust binary using the Candle ML framework.

**Requirements**:
- Compiled Rust binary with Candle
- Binary accessible at specified path

**Backend Config**:
```prolog
% Default path
backend_config([binary_path('./rust_semantic_search')])

% Custom path
backend_config([binary_path('/usr/local/bin/semantic-search')])
```

**Binary Interface**:
```bash
./rust_semantic_search search \
  --query "search query" \
  --vector-store "path/to/vectors.json" \
  --top-k 5 \
  --threshold 0.6

Output (one per line):
doc1:0.9500
doc2:0.8700
```

**Target Support**:
- `bash`: Direct binary execution
- `python`: Uses subprocess module
- `powershell`: Uses Start-Process
- `csharp`: Uses System.Diagnostics.Process

**Example**:
```prolog
:- source(semantic, research_papers, [
    vector_store('papers.json'),
    embedding_backend(rust_candle),
    backend_config([binary_path('./semantic_search')]),
    threshold(0.7),
    top_k(10)
]).
```

### 4. csharp_native
**Description**: Native C# implementation using Microsoft.ML.OnnxRuntime.

**Requirements**:
- .NET 6.0+ (or .NET Framework 4.7.2+)
- NuGet packages:
  - Microsoft.ML.OnnxRuntime
  - System.Text.Json

**Backend Config**:
```prolog
backend_config([
    model_path('models/model.onnx'),
    vocab_path('models/vocab.txt'),
    dimensions(384)
])
```

**Target Support**:
- `csharp` only

**Example**:
```prolog
:- source(semantic, research_papers, [
    vector_store('papers.json'),
    embedding_backend(csharp_native),
    backend_config(default),
    threshold(0.7),
    top_k(10)
]).
```

## Usage Examples

### Compile for Different Targets

```prolog
% Compile for Bash with Python ONNX backend
?- compile_dynamic_source(
    research_papers/2,
    [target(bash)],
    BashCode
).

% Compile for Python with Python ONNX backend
?- compile_dynamic_source(
    research_papers/2,
    [target(python)],
    PythonCode
).

% Compile for PowerShell with Go service backend
?- compile_dynamic_source(
    research_papers/2,
    [target(powershell)],
    PowerShellCode
).

% Compile for C# with native ONNX backend
?- compile_dynamic_source(
    research_papers/2,
    [target(csharp)],
    CSharpCode
).
```

### Configuration Options

All semantic sources support these common options:

- `vector_store(Path)`: Path to JSON file with pre-computed embeddings (required)
- `embedding_backend(Backend)`: Backend type (required)
  - `python_onnx`
  - `go_service`
  - `rust_candle`
  - `csharp_native`
- `backend_config(Config)`: Backend-specific configuration (required)
- `threshold(Float)`: Minimum similarity score (0.0-1.0, default: 0.6)
- `top_k(Integer)`: Maximum number of results (default: 5)
- `similarity_metric(Metric)`: Distance metric (default: cosine)
  - `cosine`: Cosine similarity
  - `euclidean`: Negative Euclidean distance
  - `dot`: Dot product
- `normalize(Boolean)`: L2 normalize embeddings (default: true)

## Implementation Details

### Generator Architecture

The semantic source plugin uses a modular generator architecture:

```
generate_wrapper_for_target_and_backend(
    Target,          % bash, python, powershell, csharp
    Backend,         % python_onnx, go_service, rust_candle, csharp_native
    Pred,            % Predicate name
    Arity,           % Predicate arity
    StorePath,       % Vector store path
    BackendConfig,   % Backend configuration
    Threshold,       % Similarity threshold
    TopK,            % Max results
    Metric,          % Similarity metric
    Normalize,       % Normalization flag
    GeneratedCode    % Output code
).
```

Each Target × Backend combination has a dedicated generator clause that:
1. Validates and expands backend configuration
2. Selects appropriate templates
3. Renders target-specific code
4. Injects backend integration

### Template System

Templates are organized by naming convention:
- `semantic_{target}_{backend}`: Main template
- Examples:
  - `semantic_bash_python_onnx`
  - `semantic_python_go_service`
  - `semantic_csharp_rust_candle`

### File Organization

```
src/unifyweaver/sources/semantic_source.pl:
  ├── Validation & Configuration
  ├── BASH WRAPPERS
  │   ├── bash + python_onnx
  │   ├── bash + go_service
  │   └── bash + rust_candle
  ├── PYTHON WRAPPERS
  │   ├── python + python_onnx
  │   ├── python + go_service
  │   └── python + rust_candle
  ├── POWERSHELL WRAPPERS
  │   ├── powershell + python_onnx
  │   ├── powershell + go_service
  │   └── powershell + rust_candle
  ├── C# IMPLEMENTATIONS
  │   ├── csharp + csharp_native
  │   ├── csharp + go_service
  │   └── csharp + rust_candle
  ├── FALLBACK (unsupported combinations)
  └── TEMPLATES
      ├── Bash templates
      ├── Python templates
      ├── PowerShell templates
      └── C# templates
```

## Performance Considerations

### Backend Performance

1. **python_onnx**: Fast inference, Python overhead
   - Best for: Prototyping, Python-heavy environments
   - Latency: ~10-50ms per query

2. **go_service**: HTTP overhead, concurrent requests
   - Best for: Multi-user systems, microservices
   - Latency: ~20-100ms per query (includes network)

3. **rust_candle**: Fastest inference, binary startup cost
   - Best for: Batch processing, performance-critical
   - Latency: ~5-20ms per query

4. **csharp_native**: Good performance, .NET ecosystem
   - Best for: Windows environments, .NET applications
   - Latency: ~10-40ms per query

### Deployment Patterns

1. **Standalone Scripts** (bash, python, powershell):
   - Use for: CLI tools, automation scripts
   - Backends: python_onnx, rust_candle

2. **Microservices** (any target + go_service):
   - Use for: Distributed systems, multi-tenant
   - Backend: go_service

3. **Embedded** (csharp + csharp_native):
   - Use for: Desktop apps, integrated solutions
   - Backend: csharp_native

4. **High Performance** (any target + rust_candle):
   - Use for: Latency-sensitive, high-throughput
   - Backend: rust_candle

## Future Expansions

Potential future Target × Backend combinations:

- **JavaScript/Node.js targets**:
  - `javascript + onnx_js`
  - `javascript + go_service`

- **Go target**:
  - `go + go_native` (direct Go implementation)

- **Rust target**:
  - `rust + rust_candle` (native Rust)

- **Additional backends**:
  - `ollama_service` (Ollama API)
  - `openai_service` (OpenAI Embeddings API)
  - `huggingface_service` (HuggingFace Inference API)

## See Also

- [Semantic Source Plugin Documentation](semantic-source.md)
- [Template System](../src/unifyweaver/core/template_system.pl)
- [Dynamic Source Compiler](../src/unifyweaver/core/dynamic_source_compiler.pl)
