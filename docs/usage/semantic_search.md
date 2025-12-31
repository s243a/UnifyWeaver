<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->

# Semantic Search Usage Guide

This guide covers how to use UnifyWeaver's semantic search capabilities for finding relevant playbook examples and documentation.

## Overview

The semantic search system uses multi-head LDA projection to improve retrieval accuracy. It projects query embeddings into answer space, making it easier to find semantically similar content even when the wording differs.

## Prolog API

### Basic Usage

```prolog
% Load the semantic search module
:- use_module('src/unifyweaver/runtime/semantic_search').

% Find top 5 examples matching a query
?- find_examples("How do I read a CSV file?", 5, Examples).
Examples = [[answer_id(3), score(0.89), text("..."), ...], ...].

% Search with minimum score threshold
?- find_examples("database query", 10, [min_score(0.7)], Examples).
```

### Available Predicates

#### find_examples/3

```prolog
find_examples(+Query, +TopK, -Examples)
```

Find playbook examples matching the query.

- **Query**: Natural language query (atom or string)
- **TopK**: Number of results to return (integer)
- **Examples**: List of result dicts with `answer_id`, `score`, `text`, `record_id`, `source_file`

#### find_examples/4

```prolog
find_examples(+Query, +TopK, +Options, -Examples)
```

Find examples with options.

**Options:**
- `component(Name)` - Use a specific search component instance
- `use_projection(Bool)` - Enable/disable LDA projection (default: true)
- `min_score(Score)` - Filter results below this similarity threshold

#### semantic_search/3, semantic_search/4

Lower-level search predicates with the same signature as `find_examples`.

### Search Modes

The system supports three search modes:

1. **Direct similarity** - Standard cosine similarity without projection
2. **Multi-head projection** - Per-cluster routing with softmax (recommended)
3. **Global projection** - Single W matrix projection

Mode is determined by component configuration:

```prolog
% Multi-head mode (default when mh_projection_id is set)
Config = [db_path('lda.db'), mh_projection_id(2)].

% Global projection mode
Config = [db_path('lda.db'), projection_id(1)].

% Direct mode (no projection)
Config = [db_path('lda.db')].
% Or override with option:
find_examples("query", 5, [use_projection(false)], Results).
```

### Component Initialization

```prolog
% Initialize a named search component
semantic_search:init_component(my_search, [
    db_path('playbooks/lda-training-data/lda.db'),
    model_name('all-MiniLM-L6-v2'),
    mh_projection_id(2)
]).

% Use the named component
find_examples("query", 5, [component(my_search)], Results).

% Shutdown when done
semantic_search:shutdown_component(my_search).
```

## Go API

The Go projection module provides native multi-head LDA projection without Python overhead.

### Loading a Projection

```go
import "unifyweaver/targets/go_runtime/projection"

// Load from directory with centroid_*.npy and answer_emb_*.npy files
mh, err := projection.LoadMultiHead(projection.Config{
    DataDir:     "playbooks/lda-training-data/trained/mh_projection_2/",
    Temperature: 0.1,
})
if err != nil {
    log.Fatal(err)
}

// Or load specific files
mh, err := projection.LoadMultiHead(projection.Config{
    Temperature: 0.1,
    HeadFiles: map[int]projection.HeadFilePair{
        1: {CentroidPath: "centroid_1.npy", AnswerEmbPath: "answer_emb_1.npy"},
        2: {CentroidPath: "centroid_2.npy", AnswerEmbPath: "answer_emb_2.npy"},
    },
})
```

### Projecting Queries

```go
// Project a query embedding
queryEmb := []float32{...} // 384-dimensional embedding
projected, err := mh.Project(queryEmb)
if err != nil {
    log.Fatal(err)
}

// Project with routing weights (for debugging/analysis)
projected, weights, err := mh.ProjectWithWeights(queryEmb)
// weights maps cluster ID to softmax weight
fmt.Printf("Routing: %v\n", weights) // e.g., {1: 0.85, 2: 0.10, 3: 0.05}
```

### Search with Projection

```go
import (
    "unifyweaver/targets/go_runtime/projection"
    "unifyweaver/targets/go_runtime/search"
)

// Load projection
mh, _ := projection.LoadMultiHead(projection.Config{
    DataDir:     "path/to/heads/",
    Temperature: 0.1,
})

// Search with projection
results, err := search.SearchWithOptions(store, queryVec, 10, search.SearchOptions{
    Projection:            mh,
    UseProjection:         true,
    IncludeRoutingWeights: true,
})

for _, r := range results {
    fmt.Printf("%s: %.3f (routing: %v)\n", r.ID, r.Score, r.RoutingWeights)
}
```

## Rust API

The Rust projection module provides native multi-head LDA projection with candle-transformers for ModernBERT support and GPU acceleration.

### Loading a Projection

```rust
use projection::{MultiHeadProjection, Config};
use std::collections::HashMap;

// Load with explicit file paths
let mut head_files: HashMap<i32, (String, String)> = HashMap::new();
head_files.insert(1, (
    "embeddings/mh_2_cluster_1_centroid.npy".to_string(),
    "embeddings/mh_2_cluster_1_answer.npy".to_string(),
));
head_files.insert(2, (
    "embeddings/mh_2_cluster_2_centroid.npy".to_string(),
    "embeddings/mh_2_cluster_2_answer.npy".to_string(),
));

let config = Config {
    data_dir: None,
    temperature: 0.1,
    head_files: Some(head_files),
};

let mh = MultiHeadProjection::load(config)?;
println!("Loaded {} heads, dimension={}", mh.num_heads(), mh.dimension);
```

### Projecting Queries

```rust
// Project a query embedding
let query_emb: Vec<f32> = embedder.get_embedding("How to query SQLite?")?;
let projected = mh.project(&query_emb)?;

// Project with routing weights (for debugging/analysis)
let (projected, weights) = mh.project_with_weights(&query_emb)?;
// weights: HashMap<i32, f32> mapping cluster ID to softmax weight
for (cluster_id, weight) in weights.iter() {
    println!("Cluster {}: {:.4}", cluster_id, weight);
}
```

### Search with Projection

```rust
use searcher::{PtSearcher, SearchOptions};
use embedding::EmbeddingProvider;

// Initialize embedder and searcher with projection
let embedder = EmbeddingProvider::new(model_path, tokenizer_path)?;
let mh = MultiHeadProjection::load(config)?;
let searcher = PtSearcher::with_projection(db_path, embedder, mh)?;

// Search with projection enabled
let options = SearchOptions {
    use_projection: true,
    include_routing_weights: true,
};

let results = searcher.vector_search_with_options("database query", 10, options)?;

for result in results {
    println!("{}: {:.3}", result.id, result.score);
    if let Some(weights) = result.routing_weights {
        println!("  Routing: {:?}", weights);
    }
}
```

### NPY File Format Support

The Rust loader supports both `float32` and `float64` numpy arrays, automatically converting to `float32` for consistency:

```rust
// Both formats work automatically:
// - '<f4' (float32) - loaded directly
// - '<f8' (float64) - converted to float32
```

## Python API

The Python embedding module provides GPU-accelerated embeddings with flash attention support.

### Installation

```bash
# Core dependencies
pip install sentence-transformers torch numpy

# Optional: Flash Attention 2 for faster inference (CUDA only)
pip install flash-attn --no-build-isolation

# Optional: For original ModernBERT (requires Python 3.10+)
# pip install transformers>=4.48.0
```

**Python Version Notes:**
- Python 3.8+: Use `nomic-ai/nomic-embed-text-v1.5` (default) - 8192 context, 768 dim
- Python 3.10+: Can also use `answerdotai/ModernBERT-base` with transformers >= 4.48.0

**Virtual Environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install sentence-transformers torch flash-attn --no-build-isolation
```

### Available Providers

| Provider | Model | Dimension | Context | GPU |
|----------|-------|-----------|---------|-----|
| `ModernBertEmbeddingProvider` | nomic-ai/nomic-embed-text-v1.5 | 768 | 8192 | ✓ |
| `ModernBertEmbeddingProvider` | intfloat/e5-large-v2 | 1024 | 512 | ✓ |
| `SentenceTransformerProvider` | all-MiniLM-L6-v2 | 384 | 256 | ✓ |

### Basic Usage

```python
from unifyweaver.targets.python_runtime.modernbert_embedding import (
    ModernBertEmbeddingProvider,
    create_embedding_provider,
)

# Auto-detect GPU, use default model (nomic-embed-text-v1.5)
embedder = ModernBertEmbeddingProvider()

# Use specific model with explicit device
embedder = ModernBertEmbeddingProvider(
    model_name="intfloat/e5-large-v2",
    device="cuda",  # or "cpu", "mps", "auto"
)

# Get single embedding (query mode - adds "search_query: " prefix)
embedding = embedder.get_embedding("How to query a database?")

# Get document embedding (adds "search_document: " prefix)
doc_embedding = embedder.get_embedding("Database documentation...", is_query=False)

# Batch processing
texts = ["query 1", "query 2", "query 3"]
embeddings = embedder.get_embeddings(texts)

# Encode documents in batch
documents = ["doc 1", "doc 2"]
doc_embeddings = embedder.encode_documents(documents)
```

### Query/Document Prefixes

Different models use different prefixes for optimal performance:

| Model | Query Prefix | Document Prefix |
|-------|--------------|-----------------|
| nomic-* | `search_query: ` | `search_document: ` |
| e5-* | `query: ` | `passage: ` |
| Others | (none) | (none) |

Prefixes are applied automatically based on model name.

### GPU Performance

With CUDA GPU acceleration:
- Single embedding: ~15ms (first call), ~1ms (cached)
- Batch (100 texts): ~1.7ms per text
- Memory: ~530MB for nomic-embed-text-v1.5

### Flash Attention

Flash Attention 2 is used automatically when available:

```python
from unifyweaver.targets.python_runtime.modernbert_embedding import (
    check_flash_attention_available,
    get_best_device,
)

print(f"Flash attention: {check_flash_attention_available()}")
print(f"Best device: {get_best_device()}")  # cuda, mps, or cpu
```

Install flash-attn for optimal performance:
```bash
pip install flash-attn --no-build-isolation
```

## Temperature Tuning

The softmax temperature controls routing sharpness:

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 0.1 | Sharp routing (winner-take-all) | Production retrieval |
| 0.5 | Moderate blending | Exploratory search |
| 1.0 | Diffuse routing | Diverse results |

Lower temperatures (0.1) typically provide better recall for precise queries.

## Example: Playbook Assistant

```prolog
% Find relevant examples for a user task
suggest_examples(UserTask, Suggestions) :-
    find_examples(UserTask, 3, [min_score(0.6)], Results),
    maplist(format_suggestion, Results, Suggestions).

format_suggestion(Result, Suggestion) :-
    member(text(Text), Result),
    member(score(Score), Result),
    member(source_file(File), Result),
    format(string(Suggestion), "~w (score: ~2f, from: ~w)", [Text, Score, File]).

% Usage:
?- suggest_examples("How to parse JSON in Python?", S).
```

## Testing

```bash
# Run Prolog tests
swipl tests/core/test_semantic_search.pl

# Run Python embedding provider test
python3 src/unifyweaver/targets/python_runtime/modernbert_embedding.py

# Run Go tests
cd src/unifyweaver/targets/go_runtime && go test ./projection/... -v

# Run Rust unit tests
cd examples/pearltrees && cargo test --bin demo_bookmark_filing

# Run Rust integration test (requires NPY files)
cd examples/pearltrees && cargo run --bin test_projection_integration
```

## See Also

- [SEMANTIC_PROJECTION_LDA.md](../proposals/SEMANTIC_PROJECTION_LDA.md) - Theory
- [MULTI_HEAD_PROJECTION_THEORY.md](../proposals/MULTI_HEAD_PROJECTION_THEORY.md) - Multi-head design
- [TODO_LDA_PROJECTION.md](../TODO_LDA_PROJECTION.md) - Implementation status
