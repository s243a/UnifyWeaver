# Playbook: Semantic/Vector Search Source

## Audience
This playbook demonstrates semantic search capabilities using embedding vectors and similarity search through UnifyWeaver's semantic_source plugin.

## Overview
The `semantic_source` plugin enables semantic retrieval over document collections using:
- Multiple embedding backends: python_onnx, go_service, rust_candle, csharp_native
- Multiple target wrappers: bash, powershell, csharp, python
- Cosine similarity, euclidean distance, or dot product metrics

## When to Use

✅ **Use semantic_source when:**
- Implementing semantic/vector search
- Building RAG (Retrieval-Augmented Generation) systems
- Finding similar documents by meaning (not keywords)
- Need embedding-based similarity matching

## Prerequisites

Requires one of the following embedding backends:
- **python_onnx**: Python 3.8+ with numpy, onnxruntime
- **go_service**: Go HTTP service
- **rust_candle**: Rust with Candle ML framework
- **csharp_native**: C# with LiteDB and ONNX Runtime

## Agent Inputs

1. **Executable Records** – `playbooks/examples_library/semantic_source_examples.md`
2. **Source Module** – `src/unifyweaver/sources/semantic_source.pl`

## Execution Guidance

### Example 1: Basic Semantic Search (Python ONNX)

```bash
cd /path/to/UnifyWeaver

perl scripts/extract_records.pl playbooks/examples_library/semantic_source_examples.md \
    semantic_basic > tmp/semantic_basic.sh
chmod +x tmp/semantic_basic.sh
bash tmp/semantic_basic.sh
```

**Expected Output:**
```
Compiling semantic source: find_papers/3
Generated: tmp/find_papers.sh
Testing semantic search:
Query: machine learning
Results:
doc1:0.85:Deep Learning Fundamentals
doc2:0.78:Neural Networks Overview
```

### Example 2: Semantic Search with Threshold

```bash
perl scripts/extract_records.pl playbooks/examples_library/semantic_source_examples.md \
    semantic_threshold > tmp/semantic_threshold.sh
chmod +x tmp/semantic_threshold.sh
bash tmp/semantic_threshold.sh
```

**Expected Output:**
```
Compiling semantic source: similar_docs/2
High similarity matches (>0.75):
doc1:0.85
doc3:0.82
```

### Example 3: Top-K Results

```bash
perl scripts/extract_records.pl playbooks/examples_library/semantic_source_examples.md \
    semantic_topk > tmp/semantic_topk.sh
chmod +x tmp/semantic_topk.sh
bash tmp/semantic_topk.sh
```

**Expected Output:**
```
Top 5 most similar documents:
1. doc5:0.92
2. doc2:0.88
3. doc7:0.85
4. doc1:0.83
5. doc9:0.81
```

## Configuration Options

**Required:**
- `vector_store(Path)` - Path to vector database
- `embedding_backend(Backend)` - Backend type

**Optional:**
- `similarity_threshold(Float)` - Minimum score 0.0-1.0 (default: 0.5)
- `top_k(Int)` - Max results (default: 10)
- `similarity_metric(Atom)` - cosine, euclidean, dot (default: cosine)
- `normalize_vectors(Bool)` - L2 normalization (default: true)
- `cache_embeddings(Bool)` - Cache queries (default: true)
- `cache_ttl(Seconds)` - Cache lifetime (default: 3600)

## Embedding Backends

### 1. Python ONNX
```prolog
embedding_backend(python_onnx)
```
- Uses Python with onnxruntime
- Good for prototyping
- Requires: numpy, onnxruntime

### 2. Go Service
```prolog
embedding_backend(go_service, [endpoint('http://localhost:8080')])
```
- HTTP-based embedding service
- Good for microservices
- Requires: Running Go service

### 3. Rust Candle
```prolog
embedding_backend(rust_candle)
```
- High-performance Rust backend
- Good for production
- Requires: Rust with Candle

### 4. C# Native
```prolog
embedding_backend(csharp_native)
```
- .NET with LiteDB and ONNX
- Good for Windows environments
- Requires: .NET SDK

## See Also

- `playbooks/json_litedb_playbook.md` - JSON document database
- `playbooks/http_source_playbook.md` - HTTP API calls
- `playbooks/python_source_playbook.md` - Python integration

## Summary

**Key Concepts:**
- ✅ Semantic/vector search over documents
- ✅ Multiple embedding backends
- ✅ Similarity metrics (cosine, euclidean, dot)
- ✅ Configurable thresholds and top-k
- ✅ Perfect for RAG systems and semantic retrieval
