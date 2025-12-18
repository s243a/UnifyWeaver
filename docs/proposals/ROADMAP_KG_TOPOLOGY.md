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

**Goal:** Implement knowledge graph on a single node without networking.

**Components:**
1. **Schema Implementation**
   - [ ] Extend `answer_relations` table with all 11 relation types
   - [ ] Add hash-based anchor linking (from SEED_QUESTION_TOPOLOGY)
   - [ ] Implement seed level provenance tracking

2. **Graph Traversal API**
   - [ ] `get_foundational()`, `get_prerequisites()`, `get_extensions()`, `get_next_steps()`
   - [ ] `get_refined()`, `get_general()`
   - [ ] `get_generalizations()`, `get_implementations()`, `get_instances()`, `get_examples()`
   - [ ] `search_with_context()` - semantic search with graph context

3. **Small-World Link Distribution**
   - [ ] Implement link distance calculation (embedding space)
   - [ ] Target distribution: 70% short / 20% medium / 10% long-range
   - [ ] Path folding after successful matches

4. **Local Routing**
   - [ ] Greedy routing to cluster centroids
   - [ ] Backtracking on match failure
   - [ ] Maximum hops limit

**No networking in Phase 1** - all operations are in-process.

### Phase 2: Multi-Interface Local Model

**Goal:** Enable one expert system to present multiple semantic identities.

**Components:**
1. **Logical Interface Layer**
   - [ ] Define interface schema (centroid, location, topics, link_topology)
   - [ ] Route queries to specific interfaces
   - [ ] Per-interface path folding

2. **Interface Management**
   - [ ] Auto-generate interfaces from cluster analysis
   - [ ] Manual interface curation
   - [ ] Interface health/coverage metrics

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

## Existing Infrastructure

The following existing components can be leveraged:

### Client-Server Infrastructure

**`src/unifyweaver/targets/prolog_service_target.pl`:**
- Prolog-as-Service pattern
- Bash script generation with service functions
- Cross-platform support (Linux/Windows/Darwin)

**`src/unifyweaver/glue/network_glue.pl`** (documented in book-07):
```prolog
:- module(network_glue, [
    % Service registry
    register_service/3,
    service/2,
    endpoint_url/3,

    % HTTP server generation
    generate_http_server/4,
    generate_go_http_server/3,
    generate_python_http_server/3,
    generate_rust_http_server/3,

    % HTTP client generation
    generate_http_client/4,
    generate_go_http_client/3,
    generate_python_http_client/3,
    generate_bash_http_client/3
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

## Key Parameters

From Kleinberg's research, the critical parameters are:

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| **α** | Link distribution exponent | Must equal effective dimension d |
| **β** | Query path selection exponent | β < α for dispersion |
| **k** | Parallelism factor | Adaptive based on load |
| **max_hops** | HTL limit | O(log n) for n nodes |

## Success Metrics

### Phase 1
- [ ] All 11 relation types implemented and tested
- [ ] `search_with_context()` returns relevant graph context
- [ ] Path folding reduces average lookup time over repeated queries

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
