# Roadmap: Knowledge Graph Topology

**Status:** Active
**Date:** 2025-12-17
**Version:** 0.1

## Executive Summary

This roadmap coordinates the development of a knowledge graph topology for Q-A systems, progressing from single-node local implementation to distributed small-world routing.

## Related Proposals

| Proposal | Focus | Status |
|----------|-------|--------|
| [SEED_QUESTION_TOPOLOGY.md](SEED_QUESTION_TOPOLOGY.md) | Provenance tracking, hash-based anchor linking | Proposed |
| [QA_KNOWLEDGE_GRAPH.md](QA_KNOWLEDGE_GRAPH.md) | Relation types (11 types across 3 categories) | Draft |
| [SMALL_WORLD_ROUTING.md](SMALL_WORLD_ROUTING.md) | Distributed routing, Kleinberg/Hyphanet architecture | Proposed |

## Relation Types Summary

From QA_KNOWLEDGE_GRAPH.md, we have 11 relation types:

**Learning Flow (4):**
- `foundational`, `preliminary`, `compositional`, `transitional`

**Scope (2):**
- `refined`, `general`

**Abstraction (5):**
- `generalization`, `implementation`, `axiomatization`, `instance`, `example`

## Phases

### Phase 1: Single Local Model ✅ Complete

**Goal:** Implement knowledge graph on a single node using softmax routing.

**Components:**
1. **Schema Implementation**
   - [x] Extend `answer_relations` table with all 11 relation types
   - [x] Add hash-based anchor linking (from SEED_QUESTION_TOPOLOGY)
   - [x] Implement seed level provenance tracking
   - [x] Folder structure by seed level for training data

2. **Graph Traversal API**
   - [x] `get_foundational()`, `get_prerequisites()`, `get_extensions()`, `get_next_steps()`
   - [x] `get_refined()`, `get_general()`
   - [x] `get_generalizations()`, `get_implementations()`, `get_instances()`, `get_examples()`
   - [x] `search_with_context()` - semantic search with graph context

**Implementation:** See `kg_topology_api.py`, `kg_topology.pl`, `training_data_organizer.py`

3. **Softmax Routing** *(Already Implemented)*

   **Current Implementation (`multi_head_search` in lda_database.py):**
   - [x] Softmax routing over cluster centroids (temperature-controlled)
   - [x] Projects query via weighted combination of answer embeddings
   - [x] Then searches all answers against projected query
   - [x] Achieves +6.7% Recall@1 over direct similarity (see MULTI_HEAD_PROJECTION_THEORY.md)

   **Baseline (`_direct_search` in kg_topology_api.py):**
   - [x] Direct matrix multiplication: query embedding × all Q-A embeddings
   - [x] No learned projection (raw cosine similarity)
   - [x] Use case: comparison baseline, fallback when no projection defined

   **Future (after answer smoothing):**
   - [ ] Transition from cluster-based to 1:1 Q-A mappings
   - [ ] Direct routing may become primary approach once clusters dissolve

**Key clarification:** The current `multi_head_search` uses cluster centroids for routing, but clusters are a training artifact for "many questions → one answer" grouping. As answer smoothing creates 1:1 mappings, the direct search approach may become more appropriate. Both methods are available.

**No networking in Phase 1** - all operations are in-process.
**No Kleinberg routing** - softmax routing is sufficient for local model.

### Scale Optimizations ✅ Implemented

These optimizations are automatically enabled based on data scale:

1. **Transformer Distillation** ✅
   - Full implementation: `projection_transformer.py`
   - API: `check_distillation_recommended()`, `get_distillation_training_embeddings()`
   - Threshold: 100,000 Q-A pairs (configurable)

2. **Interface-First Routing** ✅
   - Pre-filter answers by similarity to interface centroid
   - API: `search_via_interface(use_interface_first_routing=True)`
   - Threshold: 50,000 Q-A pairs (configurable)

3. **Max Distance Filtering** ✅ (convenience option)
   - Post-filter results by maximum distance from interface centroid
   - API: `search_via_interface(max_distance=0.5)`

**Configuration API:**
```python
# Get current config
config = db.get_scale_config()

# Set thresholds
db.set_scale_config(
    interface_first_routing_enabled='auto',  # or 'true', 'false'
    interface_first_routing_threshold=50000,
    transformer_distillation_enabled='auto',
    transformer_distillation_threshold=100000
)

# Check what should be used
status = db.get_optimization_status()
# Returns: {
#   'config': {...},
#   'interface_first_routing': {'use': bool, 'reason': str, ...},
#   'transformer_distillation': {'use': bool, 'reason': str, ...}
# }
```

Default behavior: Auto-detect based on Q-A count.

### Phase 2: Multi-Interface Local Model ✅ Complete

**Goal:** Expose multiple semantic interfaces to the SAME underlying knowledge base and routing.

**Key insight:** Phase 2 does NOT change the routing algorithm. It adds an interface layer that presents focused semantic identities to external clients while using the same softmax routing internally.

**Components:**
1. **Logical Interface Layer**
   - [x] Define interface schema (centroid, topics, exposed clusters)
   - [x] Map incoming queries to appropriate interface (`map_query_to_interface()`)
   - [x] Each interface presents a subset/view of the knowledge base

2. **Interface Management**
   - [x] Auto-generate interfaces from cluster analysis (`auto_generate_interfaces()`)
   - [x] Manual interface curation (`create_interface()`, `update_interface()`, `delete_interface()`)
   - [x] Interface health/coverage metrics (`get_interface_health()`, `compute_interface_coverage()`)

**Implementation:** Extended `kg_topology_api.py` with semantic interfaces

**Same routing as Phase 1 (default)** - interfaces are a presentation layer, not a routing change. However, if "Interface-First Routing" optimization is enabled (see Scale Optimizations above), queries route to the closest interface first, then search only within that interface's Q-A subset.

**Feedback loop:** Multiple interfaces may inform KG expansion priorities (see "Knowledge Graph Expansion" below).

3. **Prerequisites Centroids** ✅
   - [x] Per-answer prerequisites centroid storage (`answer_prerequisites_centroids` table)
   - [x] Compute from metadata relations (`compute_prerequisites_centroid_from_metadata()`)
   - [x] Compute from semantic interface search (`compute_prerequisites_centroid_from_interface()`)
   - [x] Hybrid computation combining both methods
   - [x] Search by prerequisites centroid (`search_by_prerequisites_centroid()`)
   - [x] Batch update all prerequisites centroids (`update_all_prerequisites_centroids()`)

   **Use case:** Each chapter/answer can have a "prerequisites centroid" that represents the semantic space of its prerequisites. This enables:
   - Finding prerequisite-like content for a chapter without explicit relations
   - Searching the prerequisites interface using the chapter's prerequisites centroid
   - Hybrid retrieval combining metadata relations and semantic similarity

   **API:**
   ```python
   # Compute prerequisites centroid from metadata (preliminary/foundational relations)
   db.update_prerequisites_centroid(chapter_id, model_id, method='metadata')

   # Compute from semantic search of prerequisites interface
   db.update_prerequisites_centroid(chapter_id, model_id, method='semantic',
                                     prerequisites_interface_id=prereq_if_id)

   # Search for prerequisite-like content
   results = db.search_by_prerequisites_centroid(chapter_id, model_id, top_k=5)
   ```

4. **Interface Update Methods** ✅
   - [x] Update interface properties (`update_interface()`)
   - [x] Refresh interface centroid (`refresh_interface_centroid()`)

### Phase 3: Distributed Network ✅ Core Implementation Complete

**Goal:** Enable multiple nodes to form a small-world network.

**Prerequisites:**
- Phase 1 & 2 complete ✅
- UnifyWeaver client-server Phases 7-8 (Service Discovery, Tracing) ✅

**Components:**
1. **Node Discovery & Registration** ✅
   - [x] Service registry integration (`discovery_clients.py`)
   - [x] Interface advertisement via discovery metadata (`semantic_centroid`, `interface_topics`)
   - [x] Prolog validation for routing options (`service_validation.pl` lines 1013-1135)
   - [x] Multiple backends: Local (in-memory), Consul, etcd (stub), DNS (stub)

2. **Inter-Node Routing** ✅
   - [x] Kleinberg router (`kleinberg_router.py`)
   - [x] Greedy forwarding to closest interface centroid
   - [x] HTL (Hops-To-Live) limits with configurable max_hops
   - [x] Routing envelope protocol (`__routing` in JSONL requests)

3. **Path Folding** ✅
   - [x] Query shortcuts persisted in `query_shortcuts` table
   - [x] Hit count tracking for shortcut optimization
   - [x] Shortcut pruning API (`prune_shortcuts()`)

4. **Parallel Query Strategies** ✅ (configurable)
   - [x] `parallel_paths` option for concurrent forwarding
   - [x] Softmax routing probability computation
   - [x] Similarity threshold filtering

5. **Code Generation** ✅
   - [x] Python router code generation (`python_target.pl`)
   - [x] Go router code generation (`go_target.pl`)
   - [x] Rust router code generation (`rust_target.pl`)
   - [x] HTTP endpoint generation (`network_glue.pl`)

6. **Privacy Features** (optional, deferred)
   - [ ] Request origin obfuscation
   - [ ] Encrypted inter-node communication
   - [ ] Plausible deniability

**Implementation Files:**
- `src/unifyweaver/core/service_validation.pl` - Kleinberg routing validation
- `src/unifyweaver/targets/python_runtime/discovery_clients.py` - Discovery client implementations
- `src/unifyweaver/targets/python_runtime/kleinberg_router.py` - Core routing logic
- `src/unifyweaver/targets/python_runtime/kg_topology_api.py` - DistributedKGTopologyAPI class
- `src/unifyweaver/glue/network_glue.pl` - KG endpoint generation
- `tests/core/test_kg_distributed.py` - Unit tests
- `tests/integration/test_kg_distributed.sh` - Integration tests

**Query Protocol:**
```json
{
    "__type": "kg_query",
    "__id": "uuid-123",
    "__routing": {
        "origin_node": "node_a",
        "htl": 8,
        "visited": ["node_a"],
        "path_folding_enabled": true
    },
    "__embedding": {
        "model": "all-MiniLM-L6-v2",
        "vector": [0.1, 0.2, ...]
    },
    "payload": {
        "query_text": "How do I parse CSV?",
        "top_k": 5
    }
}
```

**Example Service Definition:**
```prolog
service(csv_expert_node, [
    transport(http('/kg', [host('0.0.0.0'), port(8081)])),
    discovery_enabled(true),
    discovery_backend(consul),
    discovery_tags([kg_node, expert_system]),
    discovery_metadata([
        semantic_centroid("base64_encoded_vector..."),
        embedding_model('all-MiniLM-L6-v2'),
        interface_topics([csv, delimited, tabular])
    ]),
    routing(kleinberg([
        alpha(2.0),
        max_hops(10),
        similarity_threshold(0.5),
        path_folding(true)
    ]))
], [
    receive(Query),
    handle_kg_query(Query, Response),
    respond(Response)
]).
```

### Phase 4: Federated Query Algebra ✅ Core Implementation Complete

**Goal:** Enable multi-node queries with distributed result aggregation.

**Prerequisites:**
- Phase 3 (Distributed Network) complete ✅

**Design Insight:** Deduplication is just aggregation. Federated KG queries are essentially **distributed GROUP BY with pluggable aggregation functions** - the same operations used in SQL, MapReduce, and Datalog.

**Components:**

1. **Core Federation Engine** ✅ (Phase 4a)
   - [x] `FederatedQueryEngine` class with parallel node querying
   - [x] 6 monoid-based aggregators: SUM, MAX, MIN, AVG, COUNT, FIRST
   - [x] Distributed softmax: each node returns `exp_scores[]` + `partition_sum`
   - [x] Protocol messages: `NodeResult`, `NodeResponse`, `AggregatedResult`
   - [x] 9 Prolog validation predicates

2. **Diversity Tracking** ✅ (Phase 4b)
   - [x] `corpus_id` and `data_sources` in discovery metadata
   - [x] Auto-generated corpus_id from database content hash
   - [x] Three-tier diversity-weighted aggregation:
     - Different corpus → full boost (SUM)
     - Same corpus, disjoint sources → partial boost
     - Same corpus, overlapping sources → no boost (MAX)
   - [x] `ResultProvenance` tracking (node_id, corpus_id, data_sources, embedding_model)
   - [x] `diversity_score` in aggregated responses

3. **Prolog Code Generation** ✅ (Phase 4c)
   - [x] `compile_federated_query_python/2` - Python engine factory
   - [x] `compile_federated_service_python/2` - Complete Flask service
   - [x] `compile_federated_query_go/2` - Go engine with full types
   - [x] `generate_federation_endpoint/3` - HTTP endpoints (Python/Go/Rust)

4. **Density-Based Scoring** ✅ (Phase 4d Complete)
   - [x] Kernel Density Estimation (KDE) with Silverman bandwidth
   - [x] Flux-softmax: `P(i) = exp(sᵢ/τ) * (1 + w * dᵢ) / Z`
   - [x] Two-stage pipeline: cluster by similarity, then intra-cluster density
   - [x] `DensityAwareFederatedEngine` class
   - [x] `DENSITY_FLUX` aggregation strategy
   - [x] Prolog validation for density options
   - [x] 56 unit tests, 7 E2E tests

   **Sub-phases (all complete):**
   - [x] Phase 4d-i: Basic KDE + flux-softmax + transaction management
   - [x] Phase 4d-ii: HDBSCAN hierarchical clustering (`cluster_by_hdbscan()`, `get_hdbscan_probabilities()`)
   - [x] Phase 4d-iii: Adaptive bandwidth (`cross_validation_bandwidth()`, `adaptive_local_bandwidth()`)
   - [x] Phase 4d-iv: Efficiency (`DistanceCache`, `sketch_embeddings()`, `approximate_nearest_neighbors()`)

5. **Advanced Features** (Phase 5 Complete)
   - [x] Hierarchical federation (5a - complete)
   - [x] Adaptive federation-k (5b - complete)
   - [x] Query plan optimization (5c - complete)
   - [x] Streaming aggregation (5d - complete)

   **Phase 5d Streaming Aggregation:**
   - `PartialResult`: Data structure for incremental results (confidence, nodes_responded, elapsed_ms, is_final)
   - `StreamingConfig`: Configuration (yield_interval_ms, min_confidence, eager_yield)
   - `StreamingFederatedEngine`: AsyncGenerator-based streaming with as_completed
   - `federated_query_streaming()`: Yields PartialResult as nodes respond
   - `federated_query_sse()`: Server-Sent Events formatter for HTTP/2
   - `create_streaming_engine()`: Factory function
   - Prolog: `is_valid_streaming_option/1` (8 predicates)

   **Phase 5a Hierarchical Federation:**
   - `RegionalNode`: Data structure for regional aggregator nodes
   - `HierarchyConfig`: Configuration (max_levels, min/max_nodes_per_region, thresholds)
   - `NodeHierarchy`: Build hierarchy from topics or centroid similarity
   - `HierarchicalFederatedEngine`: Two-level query routing (regions → children)
   - `create_hierarchical_engine()`: Factory function
   - Prolog: `is_valid_hierarchy_option/1` (7 predicates)

   **Phase 5b Implementation (Adaptive Federation-K):**
   - `QueryMetrics` dataclass: entropy, top_similarity, similarity_variance, historical_consensus, avg_node_latency_ms
   - `AdaptiveKConfig` dataclass: configurable thresholds and weights
   - `AdaptiveKCalculator` class: `compute_k()` with multi-factor adjustment, feedback loop via `record_query_outcome()`
   - `AdaptiveFederatedEngine(FederatedQueryEngine)`: dynamic k selection with latency budget support
   - `create_adaptive_engine()` factory function
   - Prolog validation: `is_valid_adaptive_k_option/1` (11 predicates)
   - Unit tests: 23 tests in `test_adaptive_federation.py`

   **Phase 5c Implementation (Query Plan Optimization):**
   - `QueryType` enum: SPECIFIC, EXPLORATORY, CONSENSUS
   - `QueryClassification`: query analysis with max_similarity, variance, top_nodes
   - `QueryPlanStage`: stage_id, nodes, strategy, parallel, depends_on, cost estimation
   - `QueryPlan`: DAG of stages with `get_execution_order()` for dependency resolution
   - `QueryPlanner`: `classify_query()`, `build_plan()` for SPECIFIC/EXPLORATORY/CONSENSUS
   - `PlanExecutor`: multi-stage execution with parallel level handling
   - `PlannedQueryEngine`: combines planner + executor for automatic optimization
   - `create_planned_engine()` factory function
   - Prolog validation: `is_valid_query_planning_option/1` (9 predicates)
   - Unit tests: 31 tests in `test_query_planner.py`

**Implementation Files:**
- `src/unifyweaver/targets/python_runtime/federated_query.py` - Core engine (~1550 lines, +430 Phase 5b)
- `src/unifyweaver/targets/python_runtime/query_planner.py` - Query plan optimization (~560 lines, Phase 5c)
- `src/unifyweaver/targets/python_runtime/density_scoring.py` - Density scoring (~1200 lines)
- `src/unifyweaver/targets/python_runtime/kg_topology_api.py` - Extended API
- `src/unifyweaver/targets/python_target.pl` - Python code generation
- `src/unifyweaver/targets/go_target.pl` - Go code generation
- `src/unifyweaver/glue/network_glue.pl` - Federation endpoints
- `src/unifyweaver/core/service_validation.pl` - Federation + density + adaptive-k + query planning validation
- `tests/core/test_federated_query.py` - 45 unit tests (federation)
- `tests/core/test_density_scoring.py` - 56 unit tests (density)
- `tests/core/test_adaptive_federation.py` - 23 unit tests (Phase 5b adaptive-k)
- `tests/core/test_query_planner.py` - 31 unit tests (Phase 5c query planning)
- `tests/core/test_hierarchical_federation.py` - 31 unit tests (Phase 5a hierarchical)
- `tests/core/test_streaming_federation.py` - 24 unit tests (Phase 5d streaming)
- `tests/e2e/test_multinode_federation_e2e.py` - 7 E2E tests (multi-node)

**Federated Query Protocol:**
```json
{
    "__type": "kg_federated_query",
    "__id": "uuid-456",
    "__routing": {
        "origin_node": "node_a",
        "federation_k": 3,
        "aggregation": {
            "score_function": "diversity",
            "dedup_key": "answer_hash"
        }
    },
    "payload": {
        "query_text": "How do I parse CSV?",
        "top_k": 5
    }
}
```

**Response Protocol:**
```json
{
    "__type": "kg_federated_response",
    "__id": "uuid-456",
    "source_node": "node_b",
    "results": [
        {
            "answer_id": 42,
            "answer_text": "Use csv.reader()...",
            "answer_hash": "abc123",
            "exp_score": 2.718,
            "metadata": {}
        }
    ],
    "partition_sum": 15.5,
    "node_metadata": {
        "corpus_id": "stackoverflow_2024",
        "data_sources": ["stackoverflow", "github"],
        "embedding_model": "all-MiniLM-L6-v2"
    }
}
```

**Example Federated Service Definition:**
```prolog
service(federated_kg_node, [
    transport(http('/kg', [host('0.0.0.0'), port(8081)])),
    discovery_enabled(true),
    discovery_backend(consul),
    discovery_tags([kg_node, federated]),
    discovery_metadata([
        semantic_centroid("base64_encoded_vector..."),
        embedding_model('all-MiniLM-L6-v2'),
        corpus_id(stackoverflow_2024),
        data_sources([stackoverflow, github])
    ]),
    routing(kleinberg([alpha(2.0), max_hops(10)])),
    federation([
        federation_k(3),
        aggregation(diversity, [dedup_key(answer_hash)]),
        timeout_ms(5000),
        diversity_field(corpus_id)
    ])
], [
    receive(Query),
    handle_federated_query(Query, Response),
    respond(Response)
]).
```

**See Also:** `docs/proposals/FEDERATED_QUERY_ALGEBRA.md` for full design rationale.

## Related Work

### Pearltrees Hierarchical Categorization

The `examples/pearltrees/` work has built a **hierarchical categorization knowledge base** similar to Wikipedia's category system. This is relevant to:

- **Phase 2 (Interfaces)**: Hierarchical categories could inform how to diversify semantic interfaces, making each interface more topically localized
- **Phase 3 (Distributed)**: Category hierarchy could guide node specialization and location assignment

The category structure provides a human-curated taxonomy that complements embedding-based clustering.

### UnifyWeaver Client-Server Architecture

The existing UnifyWeaver client-server architecture (see `docs/design/CLIENT_SERVER_*.md`) provides a comprehensive foundation. The core insight: **client-server is "two opposing pipes"** - extending the pipeline model to bidirectional communication.

**Key design principles:**
- **Transport independence**: Same service definition works in-process → cross-process (Unix sockets) → network (HTTP)
- **Location transparency**: Caller doesn't know if service is local function, separate process, or remote server
- **Protocol consistency**: JSONL request/response with `__type`, `__id`, `__status`, `payload`

**How this maps to KG topology phases:**

| Phase | Transport | Use Case |
|-------|-----------|----------|
| Phase 1 | In-process | Direct softmax routing, no network overhead |
| Phase 2 | In-process or Unix socket | Interfaces as services, same process or split for isolation |
| Phase 3 | HTTP/TCP | Distributed nodes, inter-node Kleinberg routing |

**Service definition for a semantic interface (Phase 2 example):**
```prolog
service(csv_interface, [
    transport(in_process),      % Or unix_socket for isolation
    stateful(false)
], [
    receive(Query),
    route_to_closest_cluster/1, % Softmax over this interface's clusters
    respond(Answer)
]).
```

## Existing Infrastructure

The following existing components can be leveraged:

### Client-Server Infrastructure

**Design Documents (`docs/design/`):**
- `CLIENT_SERVER_PHILOSOPHY.md` - Vision: "two opposing pipes"
- `CLIENT_SERVER_SPECIFICATION.md` - Syntax, protocol, transport specs
- `CLIENT_SERVER_IMPLEMENTATION.md` - Phased implementation plan

**`src/unifyweaver/targets/prolog_service_target.pl`:**
- Prolog-as-Service pattern
- Bash script generation with service functions
- Cross-platform support (Linux/Windows/Darwin)

**`src/unifyweaver/glue/network_glue.pl`:**
```prolog
:- module(network_glue, [
    % Service registry (Phase 2-3: interface/node discovery)
    register_service/3,
    service/2,
    endpoint_url/3,

    % HTTP server generation (Phase 3: node endpoints)
    generate_http_server/4,
    generate_go_http_server/3,
    generate_python_http_server/3,
    generate_rust_http_server/3,

    % HTTP client generation (Phase 3: inter-node routing)
    generate_http_client/4,
    generate_go_http_client/3,
    generate_python_http_client/3,
    generate_bash_http_client/3,

    % Socket communication (Phase 2-3: low-latency option)
    generate_socket_server/4,
    generate_socket_client/4
]).
```

**`src/unifyweaver/sources/semantic_source.pl`:**
- `go_service` backend - HTTP service for embeddings
- Service URL configuration
- JSON API contract for search

### Embedding Infrastructure

**`tools/agentRag/src/agent_rag/core/embedding_service.py`:**
- Embedding generation service
- Can be extended for distributed embeddings

## Knowledge Graph Expansion (Separate Concern)

KG expansion is a **separate concern** from the serving/querying infrastructure above. It may warrant its own roadmap.

### Expansion Strategies

Two main approaches to growing the knowledge base:

1. **Relation-based expansion**: Add related questions using the 11 relation types
   - Discover `refined` variants of existing questions
   - Find `foundational` concepts that are missing
   - Generate `example` instances of patterns

2. **Answer smoothing migration**: Move from cluster-based to per-question answers
   - Phase 1 of SEED_QUESTION_TOPOLOGY: Many questions → one answer
   - Phase 2 of SEED_QUESTION_TOPOLOGY: Each question → tailored answer
   - Apply output smoothing constraints for consistency

### Open Questions for Expansion

- How much to focus on relation-based expansion vs answer smoothing?
- Should expansion priorities be informed by interface coverage gaps?
- How to balance human curation vs automated discovery?

### Dependency on Interfaces

Once multiple interfaces are exposed (Phase 2), they may reveal:
- Which semantic regions need more coverage
- Where relation links are sparse
- Which `refined`/`general` variants are most requested

This feedback should inform expansion priorities.

## Key Parameters (Phase 3 Only)

From Kleinberg's research, the critical parameters for **distributed routing** are:

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| **α** | Link distribution exponent | Must equal effective dimension d |
| **β** | Query path selection exponent | β < α for dispersion |
| **k** | Parallelism factor | Adaptive based on load |
| **max_hops** | HTL limit | O(log n) for n nodes |

## Success Metrics

### Phase 1 ✅
- [x] Softmax routing implemented (`multi_head_search` - cluster-based projection)
- [x] Direct search baseline implemented (`_direct_search` - raw cosine similarity)
- [x] Performance acceptable on modest hardware (matrix ops are fast)
- [x] All 11 relation types implemented and tested (52 unit tests)
- [x] `search_with_context()` returns relevant graph context
- [x] Seed-level folder structure for training data (`training_data_organizer.py`)

### Phase 2 ✅
- [x] Queries route to appropriate interface (`map_query_to_interface()`)
- [x] Interfaces have well-defined centroids (`set_interface_centroid()`, `compute_interface_centroid()`)
- [x] Coverage metrics show no semantic gaps (`compute_interface_coverage()`, `get_interface_health()`)

### Phase 3
- [ ] O(log²n) average routing hops
- [ ] 99% query success rate
- [ ] Cross-node path folding creates shortcuts

## Open Questions

1. **Effective dimension estimation**: How to determine d for high-dimensional embeddings?
2. **Relation discovery**: Can relations be auto-inferred from content?
3. **Interface generation**: Automatic clustering for interface creation?
4. **Adaptive α**: Should α adjust based on network topology changes?

## Timeline

No fixed timeline - progression is need-driven:

- **Phase 1**: Immediate priority (single-node provides value now)
- **Phase 2**: When expert systems cover multiple domains
- **Phase 3**: When actual distribution needs arise

## References

- [Kleinberg, J. (2000). "The Small-World Phenomenon: An Algorithmic Perspective"](https://www.cs.cornell.edu/home/kleinber/icm06-swn.pdf)
- [Hyphanet Routing Wiki](https://github.com/hyphanet/wiki/wiki/Routing)
- [Oskar Sandberg - "Searching in a Small World"](https://www.hyphanet.org/pages/about.html)
