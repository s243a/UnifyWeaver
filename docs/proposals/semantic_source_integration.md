# Semantic Source Integration: Design Proposal

## Executive Summary

This proposal outlines the integration of semantic/vector search capabilities into UnifyWeaver's dynamic source system. By adding a new `semantic_source.pl` plugin, users will be able to declare semantic search sources using the same familiar syntax as existing CSV, SQLite, and HTTP sources.

**Key Benefits:**
- **Unified Interface**: Semantic search uses the same `:- source(Type, Name, Options).` declaration as other sources
- **No Core Changes**: Leverages existing plugin architecture - zero modifications to core system
- **Multi-Target**: Compiles to bash, C#, or Python depending on target platform
- **Leverage Existing Work**: Builds on `PtSearcher.cs`, `OnnxEmbeddingProvider.cs`, and new Rust embedding infrastructure

**Timeline:** 5 weeks for full implementation (basic functionality in 2 weeks)

---

## 1. Motivation

### Current State

UnifyWeaver supports rich data sources:
- **Structured data**: SQLite, CSV, JSON, YAML
- **APIs**: HTTP sources with caching
- **Text processing**: AWK, Bash pipelines
- **Multi-language**: Python, .NET interop

**Missing capability:** Semantic similarity search over document embeddings.

### Use Cases

1. **Document Retrieval**
   ```prolog
   :- source(semantic, research_papers, [
       vector_store('papers.litedb'),
       embedding_model(onnx, 'all-MiniLM-L6-v2'),
       similarity_threshold(0.7)
   ]).

   find_relevant_papers(Query, PaperId, Score) :-
       research_papers(Query, PaperId, Score),
       Score > 0.8.
   ```

2. **Hybrid Search** (Combine semantic + traditional)
   ```prolog
   :- source(csv, metadata, [csv_file('papers.csv'), has_header(true)]).
   :- source(semantic, embeddings, [vector_store('embeddings.db')]).

   search_with_filters(Query, Author, Year, Title, Score) :-
       embeddings(Query, PaperId, Score),      % Semantic search
       metadata(PaperId, Title, Author, Year),  % Lookup metadata
       Year >= 2020,                            % Traditional filter
       Score > 0.75.
   ```

3. **Multi-Modal Integration**
   ```prolog
   :- source(http, api_posts, [url('https://api.example.com/posts')]).
   :- source(semantic, post_embeddings, [vector_store('posts.db')]).

   find_similar_posts(PostId, SimilarId, Similarity) :-
       api_posts(PostId, Content),
       post_embeddings(Content, SimilarId, Similarity),
       PostId \= SimilarId.
   ```

### Why Now?

1. **Infrastructure Ready**: `PtSearcher.cs` and `OnnxEmbeddingProvider.cs` already implement core functionality
2. **Rust Embeddings Complete**: ModernBERT support (PR #248) provides high-quality embeddings
3. **Plugin Architecture**: Source system designed for exactly this type of extension

---

## 2. Architecture Overview

### 2.1 System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    UnifyWeaver Source System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  User Code:                                                      │
│  :- source(semantic, docs, [vector_store('db.litedb'), ...])    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         sources.pl (Public API)                          │  │
│  │  - source/3: Declaration                                 │  │
│  │  - augment_source_options/3: Add defaults               │  │
│  │  - validate_source_options/3: Type checking              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │    dynamic_source_compiler.pl (Plugin Registry)          │  │
│  │  - register_dynamic_source/3: Store metadata            │  │
│  │  - compile_dynamic_source/3: Route to plugin            │  │
│  │  - normalize_metadata/2: I/O format normalization       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Source Plugins (Type Handlers)                 │  │
│  │  ┌────────────┬────────────┬────────────┬─────────────┐ │  │
│  │  │ CSV        │ SQLite     │ HTTP       │ Semantic    │ │  │
│  │  │ Source     │ Source     │ Source     │ Source      │ │  │
│  │  │            │            │            │ (NEW)       │ │  │
│  │  └────────────┴────────────┴────────────┴─────────────┘ │  │
│  │  Each plugin implements:                                 │  │
│  │    - source_info/1: Metadata                            │  │
│  │    - validate_config/1: Validation                      │  │
│  │    - compile_source/4: Code generation                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         template_system.pl (Code Rendering)              │  │
│  │  - render_named_template/3: Bash/C#/Python output       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Generated Code (Output)                      │  │
│  │  - Bash: Shell scripts with Python subprocess            │  │
│  │  - C#: QueryRuntime with PtSearcher integration         │  │
│  │  - Python: Direct ONNX/service API calls                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction

```
User Declaration
     ↓
sources.pl (Augment + Validate)
     ↓
dynamic_source_compiler.pl (Register)
     ↓
compiler_driver.pl (Detect dynamic source)
     ↓
semantic_source.pl (Plugin: compile_source/4)
     ↓
template_system.pl (Render template)
     ↓
Generated Code (Bash/C#/Python)
     ↓
Runtime Execution
     ↓
Vector Search Results
```

---

## 3. API Design

### 3.1 User-Facing API

#### Basic Semantic Source
```prolog
:- source(semantic, document_search, [
    % Vector database location
    vector_store('data/documents.litedb'),

    % Embedding model configuration
    embedding_model(onnx, [
        model_path('models/all-MiniLM-L6-v2.onnx'),
        vocab_path('models/vocab.txt'),
        dimensions(384)
    ]),

    % Search parameters
    similarity_threshold(0.6),
    top_k(10),

    % Output format
    return_format('{id}:{score}')
]).

% Usage in queries
find_documents(Query, DocId, Score) :-
    document_search(Query, DocId, Score),
    Score > 0.7.
```

#### With External Service
```prolog
:- source(semantic, api_embeddings, [
    vector_store('embeddings.json'),

    % Use external embedding service
    embedding_model(service, [
        provider('ollama'),
        endpoint('http://localhost:11434/api/embeddings'),
        model('nomic-embed-text'),
        dimensions(768)
    ]),

    % Caching configuration
    cache_embeddings(true),
    cache_ttl(3600),

    similarity_threshold(0.5),
    top_k(20)
]).
```

#### With Rust Runtime
```prolog
:- source(semantic, high_performance_search, [
    vector_store('vectors.redb'),

    % Use Rust Candle embeddings
    embedding_model(rust_candle, [
        model_type('modernbert'),
        model_path('models/modernbert-base-safetensors/model.safetensors'),
        tokenizer_path('models/modernbert-base-safetensors/tokenizer.json'),
        device('cuda'),
        dimensions(768)
    ]),

    similarity_metric('cosine'),  % or 'euclidean', 'dot'
    normalize_vectors(true),
    top_k(10)
]).
```

#### Hybrid Search Pattern
```prolog
:- source(csv, papers_metadata, [
    csv_file('papers.csv'),
    has_header(true)
]).

:- source(semantic, papers_embeddings, [
    vector_store('papers.litedb'),
    embedding_model(onnx, 'all-MiniLM-L6-v2'),
    similarity_threshold(0.6)
]).

% Combine semantic similarity with metadata filters
advanced_search(Query, Author, Year, Title, Abstract, Score) :-
    % Step 1: Semantic search for candidates
    papers_embeddings(Query, PaperId, Score),

    % Step 2: Lookup full metadata
    papers_metadata(PaperId, Title, Author, Year, Abstract),

    % Step 3: Apply traditional filters
    Year >= 2020,
    Score > 0.75,

    % Step 4: Optional constraint on author
    (Author = 'Smith' ; Author = 'Jones').
```

### 3.2 Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `vector_store(Path)` | atom | required | Path to vector database (LiteDB, JSON, redb) |
| `embedding_model(Type, Config)` | compound | required | Embedding provider configuration |
| `similarity_threshold(Threshold)` | float | 0.5 | Minimum similarity score (0.0-1.0) |
| `top_k(K)` | integer | 10 | Maximum results to return |
| `similarity_metric(Metric)` | atom | cosine | Similarity function: cosine, euclidean, dot |
| `normalize_vectors(Bool)` | boolean | true | L2 normalization of embeddings |
| `cache_embeddings(Bool)` | boolean | true | Cache computed query embeddings |
| `cache_ttl(Seconds)` | integer | 3600 | Cache time-to-live |
| `filter_field(Field)` | atom | none | Optional metadata filter field |
| `filter_value(Value)` | atom | none | Optional metadata filter value |
| `return_format(Format)` | atom | '{id}:{score}' | Output format template |

### 3.3 Embedding Model Types

#### Type 1: ONNX (Local)
```prolog
embedding_model(onnx, [
    model_path('models/all-MiniLM-L6-v2.onnx'),
    vocab_path('models/vocab.txt'),
    dimensions(384),
    max_length(512)
])
```

#### Type 2: External Service
```prolog
embedding_model(service, [
    provider('ollama' | 'together_ai' | 'openai' | 'custom'),
    endpoint('http://localhost:11434/api/embeddings'),
    model('nomic-embed-text'),
    api_key('sk-...'),  % Optional
    dimensions(768)
])
```

#### Type 3: Rust Candle
```prolog
embedding_model(rust_candle, [
    model_type('bert' | 'modernbert'),
    model_path('models/modernbert-base-safetensors/model.safetensors'),
    tokenizer_path('models/modernbert-base-safetensors/tokenizer.json'),
    device('cuda' | 'cpu' | 'metal'),
    dimensions(768)
])
```

#### Type 4: Simple (Model Name Only)
```prolog
% Shorthand notation - system resolves to full config
embedding_model(onnx, 'all-MiniLM-L6-v2')
embedding_model(service, 'ollama:nomic-embed-text')
embedding_model(rust_candle, 'modernbert-base')
```

---

## 4. Implementation Plan

### Phase 1: Core Plugin Infrastructure (Week 1-2)

**Goal:** Basic semantic source plugin with ONNX embedding support

**Tasks:**
1. Create `/src/unifyweaver/sources/semantic_source.pl`
2. Implement plugin interface:
   - `source_info/1`: Plugin metadata
   - `validate_config/1`: Configuration validation
   - `compile_source/4`: Code generation for bash target
3. Add bash template using Python subprocess with ONNX Runtime
4. Test with existing `OnnxEmbeddingProvider.cs` infrastructure

**Deliverables:**
- Working `semantic_source.pl` plugin (200-300 lines)
- Bash template with Python ONNX code
- Basic test suite

**Success Criteria:**
```prolog
:- source(semantic, test_docs, [
    vector_store('test.litedb'),
    embedding_model(onnx, 'all-MiniLM-L6-v2'),
    top_k(5)
]).

?- compile(test_docs/2, [output_dir('output/')], Scripts).
% Should generate: output/test_docs.sh

% Test execution
$ output/test_docs.sh "quantum physics"
doc123:0.872
doc456:0.834
doc789:0.801
```

---

### Phase 2: Multi-Target Support (Week 2-3)

**Goal:** Support C# and Python code generation targets

**Tasks:**
1. Add C# template using existing `PtSearcher` class
2. Add Python template using sentence-transformers or ONNX Runtime
3. Extend `template_system.pl` for embedding-specific rendering
4. Add target selection logic in `compile_source/4`

**Deliverables:**
- C# template (integrates with QueryRuntime)
- Python template (standalone script)
- Target-aware compilation logic

**Success Criteria:**
```prolog
% Bash target
?- compile(test_docs/2, [target(bash)], BashScript).

% C# target (for UnifyWeaver C# runtime)
?- compile(test_docs/2, [target(csharp)], CSharpCode).

% Python target (for data science workflows)
?- compile(test_docs/2, [target(python)], PythonCode).
```

**Example C# Output:**
```csharp
public class SemanticSource_TestDocs {
    private readonly PtSearcher _searcher;
    private readonly IEmbeddingProvider _embeddingProvider;

    public SemanticSource_TestDocs(string dbPath, IEmbeddingProvider provider) {
        _embeddingProvider = provider;
        _searcher = new PtSearcher(dbPath, provider);
    }

    public IEnumerable<SearchResult> Search(string query) {
        return _searcher.SearchSimilar(
            query,
            topK: 5,
            minScore: 0.5,
            typeFilter: null
        );
    }
}
```

---

### Phase 3: Configuration & Metadata (Week 3)

**Goal:** Integrate with UnifyWeaver's configuration and metadata systems

**Tasks:**
1. Add option augmentation in `sources.pl:augment_source_options/3`
2. Extend metadata extraction in `dynamic_source_compiler.pl:normalize_metadata/2`
3. Add firewall policy support for embedding models/services
4. Implement configuration validation and error reporting

**Deliverables:**
- Augmentation logic for semantic sources
- Extended metadata schema for embeddings
- Firewall policy rules

**Success Criteria:**
```prolog
% Auto-augmentation of defaults
:- source(semantic, docs, [
    vector_store('docs.db')
    % Automatically adds:
    % - embedding_model(onnx, 'all-MiniLM-L6-v2')
    % - similarity_threshold(0.5)
    % - top_k(10)
    % - similarity_metric(cosine)
    % - normalize_vectors(true)
]).

% Firewall enforcement
:- firewall_policy(docs/2, [
    allowed_embedding_models(['all-MiniLM-L6-v2', 'bge-small-en-v1.5']),
    allowed_vector_stores(['data/embeddings/*.litedb']),
    deny_external_services(true)
]).
```

---

### Phase 4: External Service Support (Week 4)

**Goal:** Support remote embedding services (Ollama, Together AI, OpenAI)

**Tasks:**
1. Add HTTP client code generation for embedding services
2. Implement caching layer for expensive API calls
3. Add batch embedding optimization
4. Support multiple service providers with unified interface

**Deliverables:**
- HTTP service client templates
- Embedding cache implementation
- Multi-provider support

**Success Criteria:**
```prolog
% Ollama local service
:- source(semantic, ollama_docs, [
    vector_store('docs.db'),
    embedding_model(service, [
        provider('ollama'),
        endpoint('http://localhost:11434/api/embeddings'),
        model('nomic-embed-text')
    ]),
    cache_embeddings(true),
    cache_ttl(3600)
]).

% Together AI cloud service
:- source(semantic, together_docs, [
    vector_store('docs.db'),
    embedding_model(service, [
        provider('together_ai'),
        model('togethercomputer/m2-bert-80M-8k-retrieval'),
        api_key_env('TOGETHER_API_KEY')
    ])
]).
```

**Generated Code (HTTP Service):**
```python
import requests
import json
import hashlib
import os

CACHE_DIR = ".unifyweaver/embedding_cache"
CACHE_TTL = 3600

def get_embedding(text, endpoint, model):
    # Check cache first
    cache_key = hashlib.md5(f"{model}:{text}".encode()).hexdigest()
    cache_path = f"{CACHE_DIR}/{cache_key}.json"

    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime < CACHE_TTL:
            with open(cache_path) as f:
                return json.load(f)['embedding']

    # Call service
    response = requests.post(endpoint, json={
        'model': model,
        'prompt': text
    })

    embedding = response.json()['embedding']

    # Cache result
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump({'embedding': embedding}, f)

    return embedding
```

---

### Phase 5: Advanced Features (Week 5+)

**Goal:** Production-ready features and optimizations

**Tasks:**
1. Add metadata filtering (filter by document type, date, author, etc.)
2. Implement hybrid search (semantic + traditional constraints)
3. Add re-ranking with cross-encoders
4. Support incremental index updates
5. Add batch processing for initial indexing

**Deliverables:**
- Metadata filtering support
- Hybrid search patterns
- Re-ranking module
- Incremental update strategy

**Success Criteria:**
```prolog
% Metadata filtering
:- source(semantic, filtered_docs, [
    vector_store('docs.db'),
    embedding_model(onnx, 'all-MiniLM-L6-v2'),
    filter_field('document_type'),
    filter_value('research_paper')
]).

% Hybrid search with re-ranking
:- source(semantic, stage1_search, [top_k(100)]).
:- source(semantic, stage2_rerank, [
    embedding_model(cross_encoder, 'ms-marco-MiniLM-L-12-v2'),
    top_k(10)
]).

hybrid_search(Query, Doc, FinalScore) :-
    % Stage 1: Fast semantic search (100 candidates)
    stage1_search(Query, Doc, _Score1),

    % Stage 2: Re-rank with cross-encoder (top 10)
    stage2_rerank(Query, Doc, FinalScore),

    FinalScore > 0.8.
```

---

## 5. Technical Design

### 5.1 Plugin Module Structure

**File:** `/src/unifyweaver/sources/semantic_source.pl`

```prolog
:- module(semantic_source, []).

:- use_module(library(option)).
:- use_module('../../core/template_system').
:- use_module('../../core/dynamic_source_compiler').

% Register this plugin at module load time
:- initialization(
    register_source_type(semantic, semantic_source),
    now
).

%% source_info(-Info) is det.
%
% Plugin metadata.
source_info(info(
    name('Semantic/Vector Search Source'),
    version('0.1.0'),
    description('Query documents via embedding vectors and cosine similarity'),
    supported_arities([2, 3, 4]),  % (Query, Id), (Query, Id, Score), (Query, Id, Score, Metadata)
    author('UnifyWeaver Team'),
    requires([
        'Python 3.8+ with numpy, onnxruntime',
        'or C# with LiteDB and ONNX Runtime'
    ])
)).

%% validate_config(+Config) is det.
%
% Validate semantic source configuration.
validate_config(Config) :-
    % Required options
    must_have_option(vector_store(_), Config, 'vector_store(Path) is required'),

    % Validate vector store exists or is creatable
    option(vector_store(StorePath), Config),
    validate_vector_store(StorePath),

    % Validate embedding model configuration
    (   option(embedding_model(Type, ModelConfig), Config)
    ->  validate_embedding_model(Type, ModelConfig)
    ;   true  % Will use default in augmentation phase
    ),

    % Validate numeric parameters
    (   option(similarity_threshold(Threshold), Config)
    ->  must_be(between(0.0, 1.0), Threshold)
    ;   true
    ),

    (   option(top_k(K), Config)
    ->  must_be(positive_integer, K)
    ;   true
    ).

%% compile_source(+Pred/Arity, +Config, +Options, -BashCode) is det.
%
% Generate code for semantic source.
compile_source(Pred/Arity, Config, Options, GeneratedCode) :-
    format('  Compiling Semantic source: ~w/~w~n', [Pred, Arity]),

    % Extract configuration
    option(vector_store(StorePath), Config),
    option(embedding_model(EmbedType, EmbedConfig), Config),
    option(similarity_threshold(Threshold), Config, 0.5),
    option(top_k(TopK), Config, 10),
    option(similarity_metric(Metric), Config, cosine),
    option(normalize_vectors(Normalize), Config, true),

    % Determine target language
    option(target(Target), Options, bash),

    % Generate code based on target and embedding type
    generate_code(Target, EmbedType, Pred, Arity,
                  StorePath, EmbedConfig, Threshold, TopK, Metric, Normalize,
                  GeneratedCode).

%% generate_code(+Target, +EmbedType, +Pred, +Arity, ..., -Code) is det.
%
% Target-specific code generation.

% Bash target with ONNX embeddings
generate_code(bash, onnx, Pred, Arity,
              StorePath, EmbedConfig, Threshold, TopK, Metric, Normalize,
              BashCode) :-
    option(model_path(ModelPath), EmbedConfig),
    option(vocab_path(VocabPath), EmbedConfig),
    option(dimensions(Dims), EmbedConfig),

    % Generate Python code for ONNX inference
    generate_onnx_python_code(
        ModelPath, VocabPath, Dims,
        StorePath, Threshold, TopK, Metric, Normalize,
        PythonCode
    ),

    % Render bash template with embedded Python
    render_named_template(semantic_bash_onnx, [
        pred=Pred,
        arity=Arity,
        python_code=PythonCode,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK
    ], BashCode).

% C# target using PtSearcher
generate_code(csharp, onnx, Pred, Arity,
              StorePath, EmbedConfig, Threshold, TopK, _Metric, _Normalize,
              CSharpCode) :-
    option(model_path(ModelPath), EmbedConfig),
    option(vocab_path(VocabPath), EmbedConfig),

    % Render C# template using PtSearcher
    render_named_template(semantic_csharp_ptsearcher, [
        pred=Pred,
        arity=Arity,
        vector_store=StorePath,
        model_path=ModelPath,
        vocab_path=VocabPath,
        threshold=Threshold,
        top_k=TopK
    ], CSharpCode).

% Bash target with external service
generate_code(bash, service, Pred, Arity,
              StorePath, ServiceConfig, Threshold, TopK, Metric, Normalize,
              BashCode) :-
    option(endpoint(Endpoint), ServiceConfig),
    option(model(Model), ServiceConfig),
    option(provider(Provider), ServiceConfig),

    % Generate HTTP client code
    generate_service_client_code(
        Provider, Endpoint, Model,
        StorePath, Threshold, TopK, Metric, Normalize,
        PythonCode
    ),

    render_named_template(semantic_bash_service, [
        pred=Pred,
        arity=Arity,
        python_code=PythonCode,
        endpoint=Endpoint,
        model=Model
    ], BashCode).

% Helper: Generate ONNX Python code
generate_onnx_python_code(ModelPath, VocabPath, Dims,
                          StorePath, Threshold, TopK, Metric, Normalize,
                          PythonCode) :-
    format(atom(PythonCode), '
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import json
import sys

# Load ONNX model
session = ort.InferenceSession("~w")
tokenizer = AutoTokenizer.from_pretrained("~w")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    outputs = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })
    embedding = outputs[0][0]  # Take first token (CLS)

    ~w  # Normalization

    return embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, top_k=~w, threshold=~w):
    query_embedding = get_embedding(query)

    # Load vector store
    with open("~w") as f:
        vectors = json.load(f)

    results = []
    for doc_id, doc_vector in vectors.items():
        doc_embedding = np.array(doc_vector)
        similarity = ~w(query_embedding, doc_embedding)

        if similarity >= threshold:
            results.append((doc_id, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    query = sys.argv[1]
    results = search(query)

    for doc_id, score in results:
        print(f"{doc_id}:{score:.4f}")
', [ModelPath, VocabPath,
    (Normalize = true -> 'embedding = embedding / np.linalg.norm(embedding)' ; ''),
    TopK, Threshold, StorePath, Metric]).
```

### 5.2 Template Definitions

Templates are defined using the multifile `template_system:template/2` predicate.

#### Template 1: Bash with ONNX
```prolog
:- multifile template_system:template/2.

template_system:template(semantic_bash_onnx, '
#!/bin/bash
# {{pred}} - Semantic search source (ONNX embeddings)
# Vector store: {{vector_store}}
# Similarity threshold: {{threshold}}
# Top K: {{top_k}}

{{pred}}() {
    local query="$1"

    python3 - "$query" <<\'PYTHON_EOF\'
{{python_code}}
PYTHON_EOF
}

{{pred}}_stream() {
    {{pred}} "$@"
}

# For batch queries
{{pred}}_batch() {
    while IFS= read -r line; do
        {{pred}} "$line"
    done
}
').
```

#### Template 2: C# with PtSearcher
```prolog
template_system:template(semantic_csharp_ptsearcher, '
using UnifyWeaver.Runtime;
using UnifyWeaver.Embeddings;
using LiteDB;
using System.Collections.Generic;

namespace UnifyWeaver.Sources {
    public class {{pred}}_Source {
        private readonly PtSearcher _searcher;
        private readonly IEmbeddingProvider _embeddingProvider;

        public {{pred}}_Source() {
            _embeddingProvider = new OnnxEmbeddingProvider(
                modelPath: "{{model_path}}",
                vocabPath: "{{vocab_path}}"
            );

            _searcher = new PtSearcher(
                dbPath: "{{vector_store}}",
                embeddingProvider: _embeddingProvider
            );
        }

        public IEnumerable<SearchResult> {{pred}}(string query) {
            return _searcher.SearchSimilar(
                queryText: query,
                topK: {{top_k}},
                minScore: {{threshold}},
                typeFilter: null
            );
        }

        // For pipeline integration
        public IEnumerable<object[]> {{pred}}_tuples(string query) {
            foreach (var result in {{pred}}(query)) {
                yield return new object[] {
                    result.Id,
                    result.Score
                };
            }
        }
    }
}
').
```

#### Template 3: Bash with External Service
```prolog
template_system:template(semantic_bash_service, '
#!/bin/bash
# {{pred}} - Semantic search source (External service)
# Endpoint: {{endpoint}}
# Model: {{model}}

{{pred}}() {
    local query="$1"

    python3 - "$query" <<\'PYTHON_EOF\'
{{python_code}}
PYTHON_EOF
}
').
```

### 5.3 Metadata Schema Extension

**Location:** `dynamic_source_compiler.pl:normalize_metadata/2`

```prolog
% Extend metadata extraction for semantic sources
normalize_semantic_metadata(RawOptions, NormalizedMeta) :-
    option(embedding_model(Type, Config), RawOptions),
    option(vector_store(StorePath), RawOptions),
    option(similarity_threshold(Threshold), RawOptions, 0.5),
    option(top_k(TopK), RawOptions, 10),
    option(similarity_metric(Metric), RawOptions, cosine),

    NormalizedMeta = semantic_metadata{
        embedding_provider: Type,
        embedding_config: Config,
        vector_store_path: StorePath,
        vector_store_type: StoreType,  % litedb, json, redb
        similarity_metric: Metric,
        similarity_threshold: Threshold,
        top_k: TopK,
        normalization: true,
        cache_enabled: true,

        % I/O format (for pipeline integration)
        record_separator: '\n',
        field_separator: ':',
        record_format: text_line,
        columns: [id, score],
        output_fields: [id, score]
    }.
```

### 5.4 Firewall Policy Support

**Location:** `firewall.pl` (extend existing firewall rules)

```prolog
% Semantic source firewall policy
firewall_policy(semantic_source, Policy) :-
    Policy = [
        % Allowed embedding models (prevent arbitrary model loading)
        allowed_embedding_models([
            'all-MiniLM-L6-v2',
            'bge-small-en-v1.5',
            'e5-small-v2',
            'modernbert-base'
        ]),

        % Allowed vector store paths
        allowed_vector_stores([
            'data/embeddings/*.litedb',
            'data/embeddings/*.json',
            'data/embeddings/*.redb'
        ]),

        % External service restrictions
        allowed_embedding_services([
            'http://localhost:11434/*',  % Ollama local
            'https://api.together.xyz/*',  % Together AI
            'https://api.openai.com/*'     % OpenAI
        ]),

        % Deny unauthenticated external services
        require_api_key_for_external(true),

        % Resource limits
        max_embedding_dimension(2048),
        max_top_k(1000),
        max_cache_size_mb(500)
    ].

% Validate semantic source against firewall
validate_semantic_firewall(PredIndicator, Options, Firewall) :-
    option(embedding_model(Type, Config), Options),

    % Check embedding model whitelist
    (   Type = onnx
    ->  option(model_path(ModelPath), Config),
        member(allowed_embedding_models(AllowedModels), Firewall),
        file_base_name(ModelPath, ModelName),
        (   member(ModelName, AllowedModels)
        ->  true
        ;   throw(firewall_error(PredIndicator,
                'Embedding model not allowed by firewall', ModelName))
        )
    ;   true
    ),

    % Check vector store path
    option(vector_store(StorePath), Options),
    member(allowed_vector_stores(AllowedPaths), Firewall),
    (   matches_any_pattern(StorePath, AllowedPaths)
    ->  true
    ;   throw(firewall_error(PredIndicator,
            'Vector store path not allowed by firewall', StorePath))
    ).
```

---

## 6. Integration with Existing Systems

### 6.1 Leverage C# Infrastructure

**Existing components to reuse:**

1. **`PtSearcher.cs`** (src/unifyweaver/targets/csharp_query_runtime/PtSearcher.cs:1-180)
   - Already implements vector search over LiteDB
   - Methods: `SearchSimilar()`, `GetSeedIds()`
   - Cosine similarity calculation

2. **`OnnxEmbeddingProvider.cs`** (200+ lines)
   - BERT tokenization
   - ONNX model inference
   - Vector normalization

3. **`IEmbeddingProvider.cs`** (28 lines)
   - Plugin interface for embeddings
   - Easy to add new providers (Ollama, Together AI)

**Integration approach:**

```csharp
// Generated C# code will use existing infrastructure
public class SemanticSource_MyDocs {
    private readonly PtSearcher _searcher;

    public SemanticSource_MyDocs(string dbPath) {
        var embeddingProvider = new OnnxEmbeddingProvider(
            modelPath: "models/all-MiniLM-L6-v2.onnx",
            vocabPath: "models/vocab.txt"
        );

        _searcher = new PtSearcher(dbPath, embeddingProvider);
    }

    public IEnumerable<SearchResult> Search(string query) {
        return _searcher.SearchSimilar(query, topK: 10, minScore: 0.5);
    }
}
```

### 6.2 Leverage Rust Infrastructure

**New components (from PR #248):**

1. **`embedding.rs`** (src/unifyweaver/targets/rust_runtime/embedding.rs:1-274)
   - Multi-model support (BERT, ModernBERT)
   - GPU acceleration via Candle
   - 8192 token context with ModernBERT

2. **Vector database** (redb)
   - High-performance key-value store
   - Memory-mapped for efficiency

**Integration approach:**

```prolog
% Use Rust runtime for high-performance embedding
:- source(semantic, rust_search, [
    vector_store('vectors.redb'),
    embedding_model(rust_candle, [
        model_type('modernbert'),
        model_path('models/modernbert-base-safetensors/model.safetensors'),
        tokenizer_path('models/modernbert-base-safetensors/tokenizer.json'),
        device('cuda')
    ]),
    top_k(10)
]).
```

**Generated code will spawn Rust subprocess:**
```bash
#!/bin/bash
semantic_rust_search() {
    local query="$1"

    # Call Rust binary (compiled from output/rust_vector_test)
    MODEL_TYPE=modernbert \
    CANDLE_DEVICE=cuda \
    ./rust_semantic_search "$query" vectors.redb
}
```

### 6.3 Metadata Pipeline Integration

Semantic sources produce TSV output like other sources:

```
doc123:0.872
doc456:0.834
doc789:0.801
```

This integrates seamlessly with:
- **Stream compiler** (constraint filtering, deduplication)
- **Pipeline chaining** (semantic | parse | analyze)
- **Hybrid queries** (combine with CSV, SQLite, HTTP)

---

## 7. Testing Strategy

### 7.1 Unit Tests

**File:** `tests/sources/semantic_source_tests.pl`

```prolog
:- begin_tests(semantic_source).

test(validate_config_minimal) :-
    Config = [
        vector_store('test.litedb'),
        embedding_model(onnx, [
            model_path('models/test.onnx'),
            vocab_path('models/vocab.txt'),
            dimensions(384)
        ])
    ],
    semantic_source:validate_config(Config).

test(validate_config_missing_store, [fail]) :-
    Config = [
        embedding_model(onnx, 'all-MiniLM-L6-v2')
    ],
    semantic_source:validate_config(Config).

test(compile_bash_target) :-
    Config = [
        vector_store('test.litedb'),
        embedding_model(onnx, [
            model_path('models/all-MiniLM-L6-v2.onnx'),
            vocab_path('models/vocab.txt'),
            dimensions(384)
        ]),
        similarity_threshold(0.7),
        top_k(5)
    ],
    Options = [target(bash), output_dir('test_output/')],
    semantic_source:compile_source(test_search/2, Config, Options, BashCode),
    atom_string(BashCode, Code),
    sub_string(Code, _, _, _, "test_search()").

test(compile_csharp_target) :-
    Config = [vector_store('test.litedb')],
    Options = [target(csharp)],
    semantic_source:compile_source(test_search/2, Config, Options, CSharpCode),
    atom_string(CSharpCode, Code),
    sub_string(Code, _, _, _, "class test_search_Source").

:- end_tests(semantic_source).
```

### 7.2 Integration Tests

**Test scenario 1: End-to-end compilation**
```prolog
test_e2e_compilation :-
    % Declare source
    assert(source(semantic, papers, [
        vector_store('data/papers.litedb'),
        embedding_model(onnx, 'all-MiniLM-L6-v2'),
        top_k(10)
    ])),

    % Compile
    compile(papers/2, [output_dir('test_output/')], Scripts),

    % Verify generated file exists
    exists_file('test_output/papers.sh'),

    % Test execution
    process_create(path('test_output/papers.sh'),
                   ['quantum physics'],
                   [stdout(pipe(Stream))]),
    read_string(Stream, _, Output),
    close(Stream),

    % Verify output format
    split_string(Output, "\n", "", Lines),
    Lines = [FirstLine|_],
    split_string(FirstLine, ":", "", [_DocId, ScoreStr]),
    number_string(Score, ScoreStr),
    Score >= 0.0, Score =< 1.0.
```

**Test scenario 2: Hybrid search**
```prolog
test_hybrid_search :-
    % CSV metadata
    assert(source(csv, metadata, [
        csv_file('test_data/papers.csv'),
        has_header(true)
    ])),

    % Semantic search
    assert(source(semantic, embeddings, [
        vector_store('test_data/embeddings.litedb'),
        embedding_model(onnx, 'all-MiniLM-L6-v2')
    ])),

    % Hybrid query
    findall((Id, Title, Score),
        (
            embeddings('machine learning', Id, Score),
            metadata(Id, Title, Author, Year),
            Year >= 2020,
            Score > 0.7
        ),
        Results
    ),

    length(Results, Count),
    Count > 0.
```

### 7.3 Performance Tests

**Benchmark 1: Query latency**
```prolog
benchmark_query_latency :-
    % Test with different embedding providers
    Providers = [
        onnx('all-MiniLM-L6-v2'),
        service('ollama:nomic-embed-text'),
        rust_candle('modernbert-base')
    ],

    forall(member(Provider, Providers), (
        setup_source(Provider),
        time(run_100_queries(Provider)),
        teardown_source(Provider)
    )).
```

**Benchmark 2: Throughput (queries/second)**
```prolog
benchmark_throughput :-
    % Measure concurrent query throughput
    NumThreads = [1, 2, 4, 8],

    forall(member(N, NumThreads), (
        format('Testing with ~w threads~n', [N]),
        time(run_concurrent_queries(N, 1000)),
        measure_qps(N, QPS),
        format('  QPS: ~w~n', [QPS])
    )).
```

---

## 8. Documentation Plan

### 8.1 User Guide

**File:** `docs/semantic_sources_guide.md`

Topics:
1. Introduction to semantic search
2. Quick start guide
3. Configuration options reference
4. Embedding model selection guide
5. Hybrid search patterns
6. Performance tuning
7. Troubleshooting

### 8.2 API Reference

**File:** `docs/api/semantic_source_api.md`

Topics:
1. Configuration schema
2. Embedding model types
3. Vector store formats
4. Output formats
5. Error codes

### 8.3 Examples

**File:** `examples/semantic_search_demo.pl`

Examples:
1. Basic semantic search
2. Hybrid search (semantic + metadata)
3. Multi-stage re-ranking
4. External service usage (Ollama, Together AI)
5. Rust runtime integration
6. Batch embedding and indexing

---

## 9. Migration Path

### 9.1 For Existing Users

**No breaking changes** - semantic sources are additive:
- Existing sources (CSV, SQLite, HTTP) unchanged
- No modifications to core predicates
- Backward compatible

**Opt-in adoption:**
```prolog
% Before: Traditional text search
:- source(csv, documents, [csv_file('docs.csv')]).
search(Keyword, Doc) :-
    documents(Doc, Content),
    sub_string(Content, _, _, _, Keyword).

% After: Add semantic search alongside
:- source(semantic, doc_embeddings, [vector_store('docs.litedb')]).
semantic_search(Query, Doc, Score) :-
    doc_embeddings(Query, Doc, Score).

% Or hybrid
hybrid_search(Query, Keyword, Doc, Score) :-
    doc_embeddings(Query, Doc, Score),  % Semantic
    documents(Doc, Content),             % Lookup
    sub_string(Content, _, _, _, Keyword).  % Filter
```

### 9.2 For New Users

**Recommended workflow:**
1. Start with existing sources (CSV, SQLite)
2. Add semantic search when text-based retrieval insufficient
3. Use hybrid patterns for best results

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ONNX model compatibility issues | Medium | High | Test with multiple models, provide clear error messages |
| External service API changes | Medium | Medium | Abstract service interface, version API calls |
| Performance degradation with large databases | High | Medium | Implement caching, batch processing, index optimization |
| Firewall bypass attempts | Low | High | Comprehensive validation, sandboxed execution |

### 10.2 Non-Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| User confusion about when to use semantic search | Medium | Low | Clear documentation, examples, decision tree |
| Embedding model licensing issues | Low | High | Only use open-source models, document licenses |
| Storage growth (vector databases) | High | Medium | Document storage requirements, provide cleanup tools |

---

## 11. Future Enhancements

### 11.1 Phase 6+ (Future)

1. **Multi-modal embeddings**
   - Image embeddings (CLIP)
   - Code embeddings (CodeBERT)
   - Graph embeddings

2. **Advanced retrieval**
   - Approximate nearest neighbors (FAISS, Annoy)
   - Quantization for reduced storage
   - GPU-accelerated search

3. **Fine-tuning support**
   - Custom domain embeddings
   - In-context learning
   - Few-shot adaptation

4. **Monitoring and observability**
   - Query latency tracking
   - Cache hit rates
   - Embedding quality metrics

---

## 12. Success Metrics

### 12.1 Technical Metrics

- **Correctness**: 100% of unit tests passing
- **Performance**: < 100ms query latency (with caching)
- **Compatibility**: Works with all existing source types
- **Coverage**: Supports bash, C#, Python targets

### 12.2 User Metrics

- **Adoption**: 5+ real-world use cases within first month
- **Satisfaction**: Positive feedback on ease of use
- **Documentation**: < 30 minutes to first working semantic search

---

## 13. Conclusion

This design integrates semantic search into UnifyWeaver's source system with:

✅ **Minimal invasiveness** - Plugin architecture, no core changes
✅ **Familiar API** - Same `:- source(Type, Name, Options).` pattern
✅ **Multi-target support** - Bash, C#, Python code generation
✅ **Leverage existing work** - PtSearcher, OnnxEmbeddingProvider, Rust embeddings
✅ **Production-ready** - Caching, firewall, error handling
✅ **Extensible** - Easy to add new embedding providers, vector stores

**Timeline:** 5 weeks for full implementation, with basic functionality available in 2 weeks.

**Next steps:** Review this design, get feedback, begin Phase 1 implementation.
