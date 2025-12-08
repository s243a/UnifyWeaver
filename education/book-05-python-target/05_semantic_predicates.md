<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 5: Semantic Predicates

The Python target includes a powerful semantic runtime library that enables AI-powered features: vector search, graph-based retrieval, web crawling, and LLM integration. This chapter introduces these semantic predicates and how they compile to Python.

## Overview

Semantic predicates bridge declarative Prolog logic with modern AI capabilities:

| Predicate | Purpose | Runtime Component |
|-----------|---------|-------------------|
| `semantic_search/3` | Vector similarity search | PtSearcher |
| `graph_search/4` | Graph RAG (multi-hop) | PtSearcher |
| `crawler_run/2` | Web/XML crawling | PtCrawler |
| `upsert_object/3` | Database storage | PtImporter |
| `llm_ask/3` | LLM question answering | LLMProvider |

## The Python Semantic Runtime

The runtime consists of several components that work together:

```
┌─────────────────────────────────────────────────────────────┐
│                    Generated Python Script                   │
├─────────────────────────────────────────────────────────────┤
│  PtSearcher        PtCrawler        PtImporter              │
│  (Vector Search)   (Web Crawling)   (SQLite Storage)        │
├─────────────────────────────────────────────────────────────┤
│  OnnxEmbeddingProvider              LLMProvider             │
│  (Local Embeddings)                 (gemini CLI)            │
├─────────────────────────────────────────────────────────────┤
│                     SQLite Database                          │
│  ┌─────────┐  ┌────────────┐  ┌─────────┐                   │
│  │ objects │  │ embeddings │  │  links  │                   │
│  └─────────┘  └────────────┘  └─────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Semantic Search

### `semantic_search(Query, TopK, Results)`

Performs vector similarity search against stored embeddings.

**Prolog:**
```prolog
find_physics_content(Results) :-
    semantic_search('quantum physics experiments', 10, Results).
```

**Generated Python:**
```python
from python_runtime.searcher import PtSearcher

def find_physics_content():
    searcher = PtSearcher('data.db', 'models/')
    results = searcher.search('quantum physics experiments', top_k=10)
    return results
```

### How It Works

1. **Query embedding** - The query text is converted to a vector using ONNX embeddings
2. **Cosine similarity** - Compare query vector against all stored embeddings
3. **Top-K selection** - Return the K most similar items
4. **Result enrichment** - Fetch full object data for returned IDs

### Result Format

```python
[
    {
        'id': 'doc_123',
        'score': 0.89,
        'type': 'article',
        'data': {'title': 'Quantum Entanglement', 'content': '...'}
    },
    {
        'id': 'doc_456',
        'score': 0.82,
        'type': 'article',
        'data': {'title': 'Wave-Particle Duality', 'content': '...'}
    }
]
```

## Graph Search (Graph RAG)

### `graph_search(Query, TopK, Hops, Results)`

Performs multi-hop graph-based retrieval. This is the core of Graph RAG (Retrieval-Augmented Generation).

**Prolog:**
```prolog
find_context(Query, Context) :-
    graph_search(Query, 5, 2, Context).
```

**Generated Python:**
```python
def find_context(query):
    searcher = PtSearcher('data.db', 'models/')
    context = searcher.graph_search(query, top_k=5, hops=2)
    return context
```

### How Graph Search Works

1. **Anchor search** - Find top-K nodes via vector similarity (same as `semantic_search`)
2. **Expansion** - For each anchor, find connected nodes via the `links` table
3. **Hop iteration** - Repeat expansion for the specified number of hops
4. **Context assembly** - Combine anchors and expanded nodes into context

```
Query: "quantum physics"
    │
    ▼
┌─────────────────┐
│  Vector Search  │ → Anchor nodes (most similar)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Hop 1: Find   │ → Parents & children of anchors
│   linked nodes  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Hop 2: Find   │ → Parents & children of hop-1 nodes
│   linked nodes  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Assemble Context│ → Structured result for LLM
└─────────────────┘
```

### Result Format

```python
{
    'anchors': [
        {'id': 'doc_1', 'score': 0.9, 'data': {...}}
    ],
    'parents': [
        {'id': 'category_physics', 'data': {...}}
    ],
    'children': [
        {'id': 'doc_2', 'data': {...}},
        {'id': 'doc_3', 'data': {...}}
    ],
    'hop_2': {
        'parents': [...],
        'children': [...]
    }
}
```

## Crawling

### `crawler_run(SeedIds, MaxDepth)`

Crawls web pages or XML documents, extracts content, and stores embeddings.

**Prolog:**
```prolog
crawl_documentation(Seeds) :-
    crawler_run(['https://docs.example.com/'], 3).
```

**Generated Python:**
```python
from python_runtime.crawler import PtCrawler

def crawl_documentation():
    crawler = PtCrawler('data.db', 'models/')
    crawler.crawl(['https://docs.example.com/'], max_depth=3)
```

### Crawler Features

- **Streaming XML parsing** - Uses `lxml.etree.iterparse` for memory efficiency
- **Link extraction** - Finds `href`, `rdf:resource`, `pt:parentTree` attributes
- **Automatic embedding** - Embeds `title` and `text` fields
- **Graph construction** - Populates the `links` table

### AWK Pipeline Integration

For high-performance XML processing, the crawler can read from stdin:

```bash
# Extract XML fragments with AWK, pipe to Python crawler
awk -f extract_elements.awk data.xml | python3 -c "
from python_runtime.crawler import PtCrawler
crawler = PtCrawler('data.db', 'models/')
crawler.process_fragments_from_stdin()
"
```

## Database Storage

### `upsert_object(Id, Type, Data)`

Manually stores or updates an object in the database.

**Prolog:**
```prolog
store_article(Id, Title, Content) :-
    upsert_object(Id, 'article', _{title: Title, content: Content}).
```

**Generated Python:**
```python
from python_runtime.importer import PtImporter

def store_article(id, title, content):
    importer = PtImporter('data.db', 'models/')
    importer.upsert_object(id, 'article', {'title': title, 'content': content})
```

### Database Schema

**objects table:**
```sql
CREATE TABLE objects (
    id TEXT PRIMARY KEY,
    type TEXT,
    data TEXT  -- JSON
);
```

**embeddings table:**
```sql
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    vector BLOB  -- float32 array
);
```

**links table:**
```sql
CREATE TABLE links (
    source_id TEXT,
    target_id TEXT,
    PRIMARY KEY (source_id, target_id)
);
```

## LLM Integration

### `llm_ask(Prompt, Context, Response)`

Sends a prompt with context to an LLM and retrieves the response.

**Prolog:**
```prolog
answer_question(Question, Answer) :-
    graph_search(Question, 5, 2, Context),
    llm_ask(Question, Context, Answer).
```

**Generated Python:**
```python
from python_runtime.llm import LLMProvider

def answer_question(question):
    # Get relevant context
    searcher = PtSearcher('data.db', 'models/')
    context = searcher.graph_search(question, top_k=5, hops=2)

    # Ask LLM with context
    llm = LLMProvider()
    answer = llm.ask(question, context)
    return answer
```

### LLM Provider Implementation

The default implementation uses the `gemini` CLI:

```python
class LLMProvider:
    def ask(self, prompt, context):
        # Format context for LLM
        context_str = json.dumps(context, indent=2)
        full_prompt = f"Context:\n{context_str}\n\nQuestion: {prompt}"

        # Call gemini CLI
        result = subprocess.run(
            ['gemini', 'ask', full_prompt],
            capture_output=True,
            text=True
        )
        return result.stdout
```

## ONNX Embeddings

The runtime uses local ONNX models for text embeddings, enabling offline operation.

### Supported Models

- `all-MiniLM-L6-v2` - Fast, good quality (384 dimensions)
- `all-mpnet-base-v2` - Higher quality (768 dimensions)
- Any BERT-compatible ONNX model

### Model Configuration

Place models in the `models/` directory:

```
models/
├── model.onnx       # ONNX model file
└── vocab.txt        # BERT vocabulary
```

### Embedding Generation

```python
from python_runtime.onnx_embedding import OnnxEmbeddingProvider

embedder = OnnxEmbeddingProvider('models/')
vector = embedder.embed("quantum physics")  # Returns float32 array
```

## Complete Example: Semantic Q&A System

Let's build a complete semantic Q&A system:

**Prolog:**
```prolog
:- use_module('src/unifyweaver/targets/python_target').

% Crawl documentation
setup_knowledge_base(URLs) :-
    crawler_run(URLs, 3).

% Answer questions using Graph RAG
answer(Question, Answer) :-
    graph_search(Question, 5, 2, Context),
    llm_ask(Question, Context, Answer).

% Find related content
find_related(Topic, Results) :-
    semantic_search(Topic, 10, Results).
```

**Compile:**
```prolog
?- compile_predicate_to_python(answer/2, [], Code),
   open('qa_system.py', write, S),
   write(S, Code),
   close(S).
```

**Usage:**
```python
# First, populate the knowledge base
from qa_system import setup_knowledge_base
setup_knowledge_base(['https://docs.example.com/'])

# Then, ask questions
from qa_system import answer
response = answer("How does quantum entanglement work?")
print(response)

# Find related content
from qa_system import find_related
related = find_related("quantum physics")
for item in related:
    print(f"{item['score']:.2f}: {item['data']['title']}")
```

## Dependencies

Semantic predicates require additional Python packages:

```bash
# Required
pip install lxml          # XML parsing

# Optional but recommended
pip install onnxruntime   # Local embeddings
pip install numpy         # Vector operations
```

Without `onnxruntime`, embedding features are disabled but crawling and storage still work.

## Configuration

### Database Location

Default: `data.db` in the current directory

```python
# Custom database path
searcher = PtSearcher('/path/to/custom.db', 'models/')
```

### Model Directory

Default: `models/` in the current directory

```python
# Custom model path
embedder = OnnxEmbeddingProvider('/path/to/models/')
```

## Performance Tips

### 1. Batch Embedding

Embed multiple texts at once for efficiency:

```python
texts = ["text1", "text2", "text3"]
vectors = embedder.embed_batch(texts)
```

### 2. Limit Hops for Graph Search

More hops = more results but slower:
- 1 hop: Fast, focused results
- 2 hops: Good balance
- 3+ hops: Slow, may include noise

### 3. Tune Top-K

Larger K means more context but:
- More tokens for LLM (cost/latency)
- Potential dilution of relevant info

Start with K=5-10, adjust based on results.

## Summary

Semantic predicates enable powerful AI capabilities:

- **`semantic_search/3`** - Vector similarity search
- **`graph_search/4`** - Multi-hop Graph RAG
- **`crawler_run/2`** - Web/XML crawling with embedding
- **`upsert_object/3`** - Manual data storage
- **`llm_ask/3`** - LLM integration

These compile to Python using the embedded runtime library, enabling sophisticated AI applications while maintaining Prolog's declarative clarity.

For more on semantic search and Graph RAG, see [Book 13: Semantic Search](../book-13-semantic-search/).

---

## Navigation

**←** [Previous: Chapter 4: Recursion Patterns](04_recursion_patterns) | [📖 Book 5: Python Target](./) | [Next: Book 6: Go Target →](../book-06-go-target/)
