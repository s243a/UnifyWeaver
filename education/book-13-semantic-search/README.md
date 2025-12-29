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
- Density-based scoring for result ranking (flux-softmax)

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

### [Chapter 6: Distributed Search](06_distributed_search.md)
- Multi-interface semantic nodes
- Kleinberg small-world routing
- Path folding and shortcuts
- Federated queries with diversity-weighted aggregation
- Prolog service definitions

### [Chapter 7: Density-Based Confidence Scoring](07_density_scoring.md)
- Kernel Density Estimation (KDE) for semantic clustering
- Flux-softmax: density-weighted probability normalization
- Two-stage pipeline: cluster first, then compute density
- HDBSCAN hierarchical clustering with soft membership
- Adaptive bandwidth selection (cross-validation, balloon estimator)
- Efficiency optimizations: sketching, ANN, O(n log n) scaling
- Integrating density scoring with federated queries

### [Chapter 8: Advanced Federation Features](08_advanced_federation.md)
- Adaptive Federation-K: dynamic node selection based on query entropy
- Query Plan Optimization: SPECIFIC/EXPLORATORY/CONSENSUS strategies
- Hierarchical Federation: regional nodes and topic-based grouping
- Streaming Aggregation: AsyncGenerator-based incremental results
- Server-Sent Events (SSE) for real-time browser updates
- Combining features: hierarchical + adaptive + streaming

### [Chapter 9: Cross-Model Federation](09_cross_model_federation.md)
- Federating across nodes with different embedding models
- Two-phase architecture: density within pools, fusion across pools
- Fusion methods: weighted_sum, RRF, consensus, geometric_mean, max
- Adaptive model weight learning from relevance feedback
- Prolog configuration for multi-model deployments

### [Chapter 10: Go and Rust Code Generation](10_go_rust_codegen.md)
- Polyglot architecture: one Prolog spec, multiple target languages
- Go code generation: goroutines, channels, sync primitives
- Rust code generation: tokio async/await, mpsc channels
- Phase 5 predicates for both targets (5a, 5b, 5c, 5d)
- Deployment patterns: homogeneous, heterogeneous, edge+cloud hybrid

### [Chapter 11: Adversarial Robustness](11_adversarial_robustness.md)
- Soft collisions (outlier detection)
- Hard collisions (KSK-style region locking)
- Consensus voting with quorum
- Trust management (direct and FMS two-dimensional)
- Trust-weighted consensus

### [Chapter 12: Performance Benchmarking](12_performance_benchmarking.md)
- Synthetic network generation
- Query workload patterns (SPECIFIC/EXPLORATORY/CONSENSUS)
- Metrics collection and aggregation
- Benchmark runner and CLI
- Visualizations and HTML reports

### [Chapter 13: Advanced Routing Algorithms](13_advanced_routing.md)
- Greedy forwarding vs Kleinberg routing distinction
- Proper small-world networks (k_local + k_long)
- HNSW hierarchical routing for O(log n) paths
- **Tunable M parameter**: Recall vs memory tradeoff configuration
- Cosine-based angle ordering (not 2D projection)
- Target language support status

### [Chapter 14: Scale-Free Networks](14_scale_free_networks.md)
- Power-law interface distribution (P(k) ∝ k^(-γ))
- Multi-interface nodes with unified binary search
- Capacity-proportional sizing (hubs compress, leaves specialize)
- Internal shortcuts between related interfaces
- Web traffic analogy and design rationale

### [Chapter 15: Zero-Shot Path Mapping](15_zero_shot_path_mapping.md)
- Materialized path format for hierarchical retrieval
- Per-tree Procrustes transformations with softmax routing
- Multi-account paths with cross-account boundaries
- Fast inference (~30ms) with cached embeddings
- Pearltrees use case: 93% Recall@1 with federated model

### [Chapter 16: Bookmark Filing Assistant](16_bookmark_filing.md)
- Semantic search + LLM combining for intelligent filing
- Multi-LLM support: Claude CLI, Gemini CLI, OpenAI, Anthropic, Ollama
- Merged tree output for hierarchical context
- MCP server for tool integration
- Slash command `/file-bookmark` for quick filing


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
- ✅ Distribute search across multiple specialized nodes
- ✅ Use Kleinberg routing for efficient query forwarding
- ✅ Rank results using density-based confidence scoring
- ✅ Adapt federation dynamically based on query characteristics
- ✅ Stream partial results for responsive UIs
- ✅ Federate across heterogeneous embedding models
- ✅ Learn optimal model weights from user feedback
- ✅ Generate Go and Rust federation engines from Prolog specs
- ✅ Deploy polyglot clusters with mixed-language nodes
- ✅ Understand the distinction between greedy routing and Kleinberg routing
- ✅ Build proper small-world networks for O(log²n) path length
- ✅ Design scale-free networks with power-law interface distribution
- ✅ Tune HNSW M parameter for recall vs memory tradeoffs
- ✅ Protect against adversarial nodes (outlier rejection, collision detection, trust)
- ✅ Run comprehensive integration tests for all federation features

## Integration Tests

The semantic search modules have comprehensive integration tests:

```bash
# Run all integration tests
PYTHONPATH="src:src/unifyweaver/targets/python_runtime" \
  python3 -m pytest tests/integration/test_*_integration.py -v

# Individual test suites
python3 -m unittest tests.integration.test_hnsw_integration -v        # HNSW with tunable M
python3 -m unittest tests.integration.test_cross_model_federation -v  # Cross-model fusion
python3 -m unittest tests.integration.test_multi_interface_integration -v  # Scale-free nodes
python3 -m unittest tests.integration.test_adversarial_robustness -v  # Outlier/collision/trust
```

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| HNSW Integration | 18 | Tunable M, layer distribution, P2P routing |
| Cross-Model Federation | 29 | Fusion methods, adaptive weights, serialization |
| Multi-Interface Nodes | 21 | Scale-free distribution, routing, statistics |
| Adversarial Robustness | 27 | Outlier detection, collision, trust management |

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.
