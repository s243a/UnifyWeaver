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

### Phase 1: Single Local Model (Current Focus)

**Goal:** Implement knowledge graph on a single node using softmax routing.

**Components:**
1. **Schema Implementation**
   - [ ] Extend `answer_relations` table with all 11 relation types
   - [ ] Add hash-based anchor linking (from SEED_QUESTION_TOPOLOGY)
   - [ ] Implement seed level provenance tracking
   - [ ] Folder structure by seed level for training data

2. **Graph Traversal API**
   - [ ] `get_foundational()`, `get_prerequisites()`, `get_extensions()`, `get_next_steps()`
   - [ ] `get_refined()`, `get_general()`
   - [ ] `get_generalizations()`, `get_implementations()`, `get_instances()`, `get_examples()`
   - [ ] `search_with_context()` - semantic search with graph context

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

### Scale Optimizations (Optional)

These optimizations are NOT required unless the Q-A knowledge base grows very large:

1. **Transformer Distillation**
   - Distill the embedding + softmax into a smaller, faster model
   - Triggered when: Q-A count exceeds threshold (configurable)

2. **Interface-First Routing (Phase 2 as optimization)**
   - Route query to closest semantic interface first (coarse filter)
   - Then search only within that interface's Q-A subset
   - Effectively a "mega-cluster" approach
   - Triggered when: Multiple interfaces defined AND Q-A count is large

**Configuration:**
```yaml
scale_optimizations:
  transformer_distillation:
    enabled: auto  # or: true, false
    threshold: 100000  # Q-A pairs before considering distillation
  interface_first_routing:
    enabled: auto
    threshold: 50000   # Q-A pairs before routing via interface first
```

Default behavior: No optimizations (direct matrix multiplication scales well).

### Phase 2: Multi-Interface Local Model

**Goal:** Expose multiple semantic interfaces to the SAME underlying knowledge base and routing.

**Key insight:** Phase 2 does NOT change the routing algorithm. It adds an interface layer that presents focused semantic identities to external clients while using the same softmax routing internally.

**Components:**
1. **Logical Interface Layer**
   - [ ] Define interface schema (centroid, topics, exposed clusters)
   - [ ] Map incoming queries to appropriate interface
   - [ ] Each interface presents a subset/view of the knowledge base

2. **Interface Management**
   - [ ] Auto-generate interfaces from cluster analysis
   - [ ] Manual interface curation
   - [ ] Interface health/coverage metrics

**Same routing as Phase 1 (default)** - interfaces are a presentation layer, not a routing change. However, if "Interface-First Routing" optimization is enabled (see Scale Optimizations above), queries route to the closest interface first, then search only within that interface's Q-A subset.

**Feedback loop:** Multiple interfaces may inform KG expansion priorities (see "Knowledge Graph Expansion" below).

### Phase 3: Distributed Network

**Goal:** Enable multiple nodes to form a small-world network.

**Prerequisites:**
- Phase 1 & 2 complete
- Actual need for distribution (scale, geographic, organizational)

**Components:**
1. **Node Discovery & Registration**
   - [ ] Service registry integration
   - [ ] Interface advertisement
   - [ ] Location assignment (Kleinberg-compatible)

2. **Inter-Node Routing**
   - [ ] Greedy forwarding to closest interface
   - [ ] Cross-node backtracking
   - [ ] HTL (Hops-To-Live) limits

3. **Parallel Query Strategies** (optional)
   - [ ] Probability-weighted path selection (β < α)
   - [ ] Stratified sampling across distance bands
   - [ ] Adaptive dispersion

4. **Privacy Features** (optional, deferred)
   - [ ] Request origin obfuscation
   - [ ] Encrypted inter-node communication
   - [ ] Plausible deniability

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

### Phase 1
- [x] Softmax routing implemented (`multi_head_search` - cluster-based projection)
- [x] Direct search baseline implemented (`_direct_search` - raw cosine similarity)
- [x] Performance acceptable on modest hardware (matrix ops are fast)
- [ ] All 11 relation types implemented and tested
- [ ] `search_with_context()` returns relevant graph context
- [ ] Seed-level folder structure for training data

### Phase 2
- [ ] Queries route to appropriate interface
- [ ] Interfaces have well-defined centroids
- [ ] Coverage metrics show no semantic gaps

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
