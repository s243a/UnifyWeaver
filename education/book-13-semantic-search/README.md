<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 13: Semantic Search

**Graph RAG, Embeddings, and Vector Databases**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers building intelligent semantic agents using UnifyWeaver's Python semantic runtime. You'll learn to combine declarative logic with modern AI capabilities like vector search, graph RAG, and LLM integration.

## Prerequisites

**Required:**
- [Book 1: Foundations](../book-01-foundations/README.md)

**Recommended:**
- [Book 5: Python Target](../book-05-python-target/README.md) - Python code generation basics

**Technical:**
- Python 3.8+ with pip
- Basic understanding of embeddings and vector similarity (helpful)

## What You'll Learn

- Building semantic agents with the Python runtime
- Graph-based Retrieval Augmented Generation (RAG)
- Vector search with ONNX embeddings
- Web crawling and content extraction
- LLM integration (Gemini, GPT, Claude)
- Semantic data pipelines

## Chapter Overview

### [Chapter 1: Introduction](01_introduction.md)
- Why Python for semantic tasks
- Architecture of the semantic runtime
- Your first semantic script
- Key components: PtImporter, PtCrawler, PtSearcher

### [Chapter 2: Graph RAG](02_graph_rag.md)
- Graph-based knowledge retrieval
- Combining vector search with graph traversal
- Building knowledge graphs from crawled content
- RAG patterns with UnifyWeaver

### [Chapter 3: Semantic Data Pipeline](03_semantic_data_pipeline.md)
- End-to-end semantic pipelines
- Data ingestion and embedding
- Storage with SQLite
- Query and retrieval patterns

### [Chapter 4: Logic and Recursion](04_logic_and_recursion.md)
- Recursive semantic queries
- Transitive closure over knowledge graphs
- Combining Prolog logic with vector similarity
- Advanced query patterns

### [Chapter 5: Semantic Playbook](05_semantic_playbook.md)
- Production deployment patterns
- Performance optimization
- Error handling and fallbacks
- Integration with other targets

## The Semantic Runtime

The Python semantic runtime provides:

| Component | Purpose |
|-----------|---------|
| `PtImporter` | SQLite storage for objects, links, vectors |
| `PtCrawler` | XML/HTML streaming crawler with link extraction |
| `PtSearcher` | Vector search and graph traversal |
| `LLMProvider` | Wraps LLM CLIs for RAG tasks |
| `OnnxEmbeddingProvider` | Local embeddings via ONNX |

## Quick Start

```prolog
:- module(semantic_search, [search/2]).
:- use_module(unifyweaver(targets/python_target)).

% Define a semantic search predicate
search(Query, Result) :-
    embed(Query, Vector),
    vector_search(Vector, 5, Matches),
    member(Result, Matches).

% Compile to Python
?- compile_python_semantic(search/2, [], Code),
   write_file('search.py', Code).
```

## Installation

```bash
# Install Python dependencies
pip install numpy onnxruntime sqlite3

# Optional: Install embedding model
pip install sentence-transformers

# Optional: Install LLM CLI
pip install google-generativeai  # for Gemini
```

## Use Cases

- **Knowledge Base Search**: Build searchable knowledge bases from documents
- **Focused Crawling**: Semantically-guided web crawling
- **Document Q&A**: RAG-based question answering
- **Content Recommendation**: Similarity-based content discovery
- **Research Assistants**: AI-powered research tools

## What's Next?

After completing this book, you'll be able to:
- ✅ Build semantic search applications
- ✅ Implement Graph RAG patterns
- ✅ Create intelligent crawlers
- ✅ Integrate LLMs with declarative logic
- ✅ Deploy production semantic pipelines

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.
