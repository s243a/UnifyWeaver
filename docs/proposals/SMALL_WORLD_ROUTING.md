# Proposal: Small-World Routing for Q-A Topology

**Status:** Core Implementation Complete (Phase 3)
**Version:** 1.1
**Date:** 2025-12-18
**Context:** Extends [SEED_QUESTION_TOPOLOGY.md](SEED_QUESTION_TOPOLOGY.md) with distributed routing architecture.
**Implementation:**
- `src/unifyweaver/targets/python_runtime/kleinberg_router.py` - Core router
- `src/unifyweaver/targets/python_runtime/discovery_clients.py` - Service discovery
- `src/unifyweaver/targets/python_runtime/kg_topology_api.py` - DistributedKGTopologyAPI

## Executive Summary

This proposal applies **small-world network routing** principles from [Hyphanet](https://en.wikipedia.org/wiki/Freenet) (formerly Freenet) and [Kleinberg's research](https://www.cs.cornell.edu/home/kleinber/icm06-swn.pdf) to the Q-A knowledge graph topology. The goal is efficient, decentralized query routing that scales with O(log²n) path length.

**Key concepts:**
- **Greedy routing**: Forward queries to the semantically closest cluster centroid
- **Small-world topology**: Balance short-range clustering with long-range bridges
- **Path folding**: Dynamically improve topology after successful matches
- **The α parameter**: Critical tuning for link distance distribution

## Motivation: The Diversity Problem

Once we move to 1:1 question→answer mappings (see [SEED_QUESTION_TOPOLOGY.md](SEED_QUESTION_TOPOLOGY.md)), we need a new criterion for answer diversity. The seed(n) approach alone no longer enforces cross-cluster exploration.

Small-world topology provides the solution:
- **Local coherence**: Related questions stay connected (short-range links)
- **Global reachability**: Any question reachable in few hops (long-range links)
- **Serendipitous discovery**: Long-range links surface unexpected connections

## Lessons from Hyphanet

The [Hyphanet project](https://en.wikipedia.org/wiki/Freenet) (formerly Freenet, renamed June 2023) provides a proven architecture for decentralized routing on small-world networks.

### Core Routing Mechanism

From the [Hyphanet Routing documentation](https://github.com/hyphanet/wiki/wiki/Routing):

- Each node and each content key has a **location** (0.0 to 1.0)
- Routing is **greedy**: forward requests to the peer whose location is closest to the target
- The sophistication is in the **topology**, not the algorithm itself

```
Hyphanet Routing Decision:
1. Receive request for key with location 0.73
2. Check eligible peers (exclude failed, overloaded nodes)
3. Forward to peer with location closest to 0.73
4. If rejected, backtrack and try next-closest peer
```

### Three Required Properties

From the [Hyphanet Small-World Topology wiki](https://github.com/hyphanet/wiki/wiki/Small-world-topology):

1. **Full Connectivity**: A route exists between any two nodes
2. **Short Paths**: Short routes exist from any node to any other
3. **Clustering (Triangles)**: If A→B and B→C, increased probability of A→C

### Path Folding

[Oskar Sandberg's research](https://www.hyphanet.org/pages/about.html) on Hyphanet 0.7 shows that **path folding** is critical:

- After successful content retrieval, nodes may establish direct shortcut links
- This dynamically improves the topology over time
- A simple greedy algorithm suffices *if* path folding is active

## Kleinberg's Critical Result

[Jon Kleinberg's research on small-world networks](https://www.cs.cornell.edu/home/kleinber/icm06-swn.pdf) provides the theoretical foundation:

- Long-range links are chosen with probability **proportional to 1/r^α** where r = distance
- **Critical finding**: Only when **α = d** (network dimension) can decentralized algorithms find short paths efficiently
- With correct α: greedy routing achieves **O(log²n)** path length
- With wrong α: path length degrades to **O(n^c)** — exponentially worse

From [Kleinberg's model analysis](https://chih-ling-hsu.github.io/2020/05/15/kleinberg):

> "Kleinberg's most striking result is that no decentralized algorithm can find short paths if α≠d, even when the diameter of the augmented graph is Θ(log n)."

## Applying to Q-A Topology

### Mapping Concepts

| Hyphanet Concept | Q-A Topology Equivalent |
|------------------|------------------------|
| Node location (0.0-1.0) | Cluster centroid in embedding space |
| Content key hash | Question/Answer embedding |
| Greedy routing | Forward to most similar centroid |
| Long-range links (1/r^α) | Cross-domain bridge questions |
| Path folding | Dynamic shortcut creation after successful matches |
| HTL (Hops-To-Live) | Maximum cluster hops before giving up |

### Distance-Weighted Link Distribution

Each answer should maintain links with a target distribution:

```
Link Distribution Target:
├── 70% short-range: links to semantically similar questions (same cluster)
├── 20% medium-range: links to related-but-distinct topics (adjacent clusters)
└── 10% long-range: links to cross-domain connections (bridge questions)
```

### Schema with Small-World Links

```json
{
  "answer_id": "hash(content)",
  "anchor_question": "hash(Q1)",
  "seed_level": 1,
  "link_topology": {
    "short_range": [
      {"hash": "hash(Q2)", "distance": 0.15},
      {"hash": "hash(Q3)", "distance": 0.22}
    ],
    "medium_range": [
      {"hash": "hash(Q_adjacent)", "distance": 0.55}
    ],
    "long_range": [
      {"hash": "hash(Q_distant)", "distance": 0.85, "bridge_type": "analogy"}
    ]
  }
}
```

### The Critical α Parameter

For our embedding space, the **clustering exponent α** must match the effective dimensionality:

```
Link probability ∝ 1/distance^α

If embedding space has effective dimension d:
- α < d → Too many long-range links (chaotic, poor local clustering)
- α > d → Too few long-range links (isolated clusters, slow global routing)
- α = d → Optimal: O(log²n) routing with good clustering
```

For high-dimensional embeddings (e.g., 384-dim), the *effective* dimension after manifold structure is typically much lower (perhaps 10-50). This effective dimension should guide α selection.

### Proposed Routing Algorithm

```python
def route_question(question, max_hops=10):
    """Greedy routing with backtracking and path folding."""
    q_embedding = embed(question)
    visited = set()

    for hop in range(max_hops):
        # Find closest unvisited centroid
        candidates = [c for c in centroids if c.id not in visited]
        if not candidates:
            return None  # Exhausted all options

        closest = min(candidates, key=lambda c: distance(q_embedding, c.embedding))
        visited.add(closest.id)

        # Check for match in this cluster
        match = closest.search(question)
        if match and match.score > threshold:
            # PATH FOLDING: Create shortcut for future queries
            create_shortcut(question, match.answer)
            return match

    return None  # No match within hop limit
```

### Path Folding for Q-A

After a successful question→answer match:

```
Before: Q_new has no direct link to Answer_A
After:  Q_new ──shortcut──→ Answer_A

This creates:
1. Faster future lookups for similar questions
2. Organic growth of the knowledge graph
3. Reinforcement of frequently-used paths
```

## Node Abstraction: Principal vs Logical Nodes

### The Problem: Broad Expert Systems

A **principal node** (unique IP address, process, or expert system) might want to mirror the small-world structure locally — having most Q-A pairs near a semantic centroid with a few long-range relationships.

However, in practice, a principal node often covers a **wide range of topics** without an obvious single cluster center. This creates a problem for routing: the node's location signal is diffuse, making greedy routing less effective.

### Solution: Multi-Interface Node Architecture

One principal expert system can serve **multiple logical node interfaces**, each presenting a focused semantic identity to the network:

```
┌─────────────────────────────────────────────────────┐
│            Principal Node (Expert System)           │
│                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  Interface  │ │  Interface  │ │  Interface  │   │
│  │   "CSV"     │ │   "JSON"    │ │  "Database" │   │
│  │ loc: 0.23   │ │ loc: 0.31   │ │ loc: 0.67   │   │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘   │
│         │               │               │           │
│         └───────────────┼───────────────┘           │
│                         │                           │
│              Shared Knowledge Base                  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
                    To Network
```

### Benefits

1. **Cleaner Routing Signals**: Each interface has a well-defined semantic location (centroid)
2. **Focused Small-World Links**: Each interface maintains its own short/long-range link distribution
3. **Efficient Greedy Routing**: Network peers route to specific interfaces, not the diffuse principal node
4. **Resource Sharing**: One expert system serves multiple logical roles without duplication

### Mapping to Network Concepts

| Concept | Description |
|---------|-------------|
| Principal Node | Physical entity (IP, process, expert system) |
| Logical Interface | Virtual node with focused semantic location |
| Interface Centroid | The semantic center of that interface's Q-A domain |
| Interface Links | Small-world links specific to that interface |

### Schema Extension

```json
{
  "principal_node": {
    "id": "node_abc123",
    "address": "192.168.1.42:8080",
    "interfaces": [
      {
        "interface_id": "csv_specialist",
        "centroid": [0.23, 0.45, ...],  // embedding vector
        "location": 0.23,                // derived scalar location
        "topics": ["csv", "delimited data", "tabular parsing"],
        "link_topology": { ... }
      },
      {
        "interface_id": "json_specialist",
        "centroid": [0.31, 0.52, ...],
        "location": 0.31,
        "topics": ["json", "serialization", "structured data"],
        "link_topology": { ... }
      }
    ]
  }
}
```

### Routing with Multi-Interface Nodes

When routing a query:

1. **Network sees interfaces, not principals**: The query routes to "json_specialist@node_abc123", not just "node_abc123"
2. **Interface responds**: The principal node's shared knowledge base handles the query through the appropriate interface
3. **Path folding per-interface**: Shortcuts are created to specific interfaces, not the principal node

```python
def route_to_interface(question):
    q_embedding = embed(question)

    # Find closest INTERFACE (not principal node)
    best_interface = min(
        all_interfaces,
        key=lambda iface: distance(q_embedding, iface.centroid)
    )

    # Route to that interface
    return best_interface.principal_node.query(
        question,
        via_interface=best_interface.id
    )
```

### Relationship to Hyphanet Darknet Clusters

This is analogous to running multiple Hyphanet nodes in a darknet cluster — each node has its own location, but they share infrastructure. The difference:

- **Hyphanet**: Multiple actual nodes, each with its own state
- **Our model**: One principal node, multiple virtual interfaces sharing state

## Implementation Status

### Phase 3: Core Distributed Routing (Complete)

The following components are now implemented and tested:

#### KleinbergRouter Class (`kleinberg_router.py`)
```python
router = KleinbergRouter(
    discovery_client=LocalDiscoveryClient(),
    alpha=2.0,           # Link distance exponent
    max_hops=10,         # HTL limit
    similarity_threshold=0.5,
    path_folding_enabled=True
)

# Discover nodes via service discovery
nodes = router.discover_nodes(tags=['kg_node'])

# Route query with greedy forwarding
results = router.route_query(query_embedding, query_text, envelope, top_k=5)

# Path folding shortcuts
router.create_shortcut(query_text, node_id, interface_id)
cached = router.check_shortcut(query_text)
```

#### RoutingEnvelope Class
```python
envelope = RoutingEnvelope(
    origin_node="node_a",
    htl=10,                    # Hops-To-Live
    visited={"node_a"},        # Cycle detection
    path_folding_enabled=True
)
```

#### Multi-Interface Nodes (DistributedKGTopologyAPI)
```python
api = DistributedKGTopologyAPI(db_path, embeddings_path)

# Register node with discovery service
api.register_node(
    interface_id="csv_specialist",
    host="0.0.0.0",
    port=8081,
    tags=["kg_node", "expert_system"]
)

# Centroids are published in discovery_metadata
# Other nodes can route to specific interfaces
```

#### Service Discovery Abstraction
```python
# Multiple backends supported
client = create_discovery_client("local")   # In-memory for testing
client = create_discovery_client("consul", host="consul.local")

# Register/discover/heartbeat
client.register(name, id, host, port, tags, metadata)
instances = client.discover(name, tags=["kg_node"])
client.heartbeat(service_id)
```

#### Prolog Validation
```prolog
% Kleinberg routing options
routing(kleinberg([
    alpha(2.0),
    max_hops(10),
    similarity_threshold(0.5),
    path_folding(true)
]))

% Discovery metadata with centroids
discovery_metadata([
    semantic_centroid("base64_encoded_vector..."),
    embedding_model('all-MiniLM-L6-v2'),
    interface_topics([csv, delimited, tabular])
])
```

### Phase 4: Federated Query Aggregation (Complete)

Phase 4 extends routing with SQL-like aggregation semantics:

- **FederatedQueryEngine**: Parallel queries to multiple nodes with pluggable aggregation
- **Diversity-weighted scoring**: SUM for independent corpora, MAX for shared data
- **Prolog code generation**: `compile_federated_query_python/2`, `compile_federated_query_go/2`

See [FEDERATED_QUERY_ALGEBRA.md](FEDERATED_QUERY_ALGEBRA.md) for details.

### Phase 4d: Density-Based Confidence (Proposed)

Extends aggregation with kernel density estimation for consensus detection:

- **Two-stage pipeline**: Cluster results first, then compute density within clusters
- **Distributed cluster aggregators**: Semantic DHT with Freenet-style routing
- **Transaction management**: Aggregator lifecycle with timeouts

See [DENSITY_SCORING_PROPOSAL.md](DENSITY_SCORING_PROPOSAL.md) for details.

## Open Questions & Future Work

### Parallel Request Strategies (Partially Addressed)

**Status:** Core parallelism implemented in Phase 4; advanced strategies remain open.

Hyphanet uses sequential routing (one peer at a time) likely to **prevent network flooding** — a critical concern in a public P2P network where malicious actors could overwhelm nodes with parallel requests.

However, in controlled environments (e.g., a darknet cluster or local Q-A system), parallel requests become viable. Key considerations:

**Probability-Weighted Path Selection:**
```
P(select path) ∝ f(distance, path_type)

Where f() balances:
- Greedy preference: favor short-distance paths (high probability)
- Dispersion requirement: include some long-distance paths (lower but non-zero probability)
```

**Potential Approaches:**

1. **Weighted Sampling**: Select k paths with probability ∝ 1/distance^β (where β < α to ensure some long paths)

2. **Stratified Sampling**: Explicitly sample from distance bands:
   ```
   k_short  = 3 paths from distance < 0.3
   k_medium = 2 paths from distance 0.3-0.6
   k_long   = 1 path  from distance > 0.6
   ```

3. **Adaptive Dispersion**: Start greedy, increase dispersion if initial paths fail:
   ```
   Round 1: Query top-3 closest centroids
   Round 2: If no match, query 3 medium-distance centroids
   Round 3: If still no match, try long-range exploratory paths
   ```

**Darknet Cluster Workaround:**

Even in Hyphanet, one could achieve parallel queries by running multiple nodes in a trusted darknet cluster:
```
┌─────────────────────────────────────┐
│         Darknet Cluster             │
│  ┌──────┐ ┌──────┐ ┌──────┐        │
│  │Node 1│ │Node 2│ │Node 3│        │
│  └──┬───┘ └──┬───┘ └──┬───┘        │
│     │        │        │             │
│     └────────┼────────┘             │
│              ▼                      │
│      Parallel queries to            │
│      different network regions      │
└─────────────────────────────────────┘
```

Each node routes independently, achieving parallelism without violating the single-request-per-node convention.

**Research Questions:**
- What is the optimal β for path selection probability?
- How does dispersion affect latency vs. discovery rate?
- Can we learn the optimal k (parallelism factor) adaptively?

### Long-Range Link Selection Strategies

The key challenge is selecting meaningful long-range links:

1. **Random Sampling**: Simple but may produce noisy/irrelevant connections
2. **Bridge Detection**: Analyze embedding space for natural cross-domain connections
3. **Structural Holes**: Find questions that connect otherwise disconnected clusters
4. **Analogy Mining**: "X is to Y as A is to B" - explicit analogical reasoning

### Anonymity as a Deferred Concern

Hyphanet's architecture includes significant overhead for **anonymity features**:
- Onion-style routing to hide request origins
- Plausible deniability for stored content
- HTL probabilistic decrementing to obscure request sources
- Traffic analysis resistance

These features come at a **substantial performance cost**. For our initial implementation:

```
Phase 1 (Current Focus):
├── Core routing topology (small-world structure)
├── Greedy + parallel query strategies
├── Path folding for dynamic improvement
└── NO anonymity overhead

Phase 2 (Future, Optional):
├── Cross-node privacy features
├── Request origin obfuscation
├── Encrypted inter-node communication
└── Plausible deniability for sensitive Q-A domains
```

**Rationale**: In trusted environments (internal knowledge bases, research clusters), anonymity isn't required. We can optimize for **latency and throughput** first, then layer privacy features for deployments where peer-to-peer traffic hiding is necessary.

## References

1. **Hyphanet (formerly Freenet)**
   - Wikipedia: https://en.wikipedia.org/wiki/Freenet
   - About page: https://www.hyphanet.org/pages/about.html
   - Routing wiki: https://github.com/hyphanet/wiki/wiki/Routing
   - Small-world topology wiki: https://github.com/hyphanet/wiki/wiki/Small-world-topology

2. **Kleinberg's Small-World Research**
   - Original paper: Kleinberg, J. (2000). "The Small-World Phenomenon: An Algorithmic Perspective." ACM STOC.
   - Complex networks survey: https://www.cs.cornell.edu/home/kleinber/icm06-swn.pdf
   - Model explanation: https://chih-ling-hsu.github.io/2020/05/15/kleinberg

3. **Oskar Sandberg's Contributions**
   - "Searching in a Small World" - Licentiate thesis on decentralized small-world construction
   - "Distributed Routing in Small World Networks" - Theoretical basis for Hyphanet 0.7 darknet routing

## Related Proposals

- **[SEED_QUESTION_TOPOLOGY.md](SEED_QUESTION_TOPOLOGY.md)**: Seed levels for provenance tracking and hash-based anchor linking
