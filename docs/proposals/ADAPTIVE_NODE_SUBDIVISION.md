# Adaptive Node Subdivision

**Status:** Proposed
**Date:** 2024-12-20
**Prerequisites:** Phase 5a Hierarchical Federation, Phase 6b Benchmarking

## Problem Statement

Benchmark findings (Phase 6b) revealed that precision varies with cluster density:

| Network | Nodes/Cluster | Precision |
|---------|---------------|-----------|
| 10 nodes | 2 per cluster | 0.883 |
| 50 nodes | 10 per cluster | 0.950 |

Sparse clusters have fewer nodes above similarity thresholds, making it harder to achieve high precision. Dense clusters risk becoming bottlenecks.

**Goal:** Maintain optimal cluster density automatically as the network grows.

## Proposed Solution

Nodes split when they exceed capacity thresholds, similar to B-tree page splits or consistent hashing ring rebalancing.

### Split Triggers

A node splits when ANY of these conditions are met:

| Trigger | Threshold | Rationale |
|---------|-----------|-----------|
| Document count | > 1000 | Prevent query bottlenecks |
| Centroid variance | > 0.5 | Topics too diverse |
| Query latency P99 | > 500ms | Performance degradation |
| Memory usage | > 80% | Resource limits |

### Split Algorithm

```python
def should_split(node: KGNode) -> bool:
    """Check if node should subdivide."""
    return (
        node.document_count > node.config.max_documents or
        node.centroid_variance > node.config.max_variance or
        node.query_latency_p99 > node.config.max_latency_ms or
        node.memory_percent > node.config.max_memory_percent
    )

def split_node(node: KGNode) -> Tuple[KGNode, KGNode]:
    """Split node into two children using k-means."""
    # 1. Cluster documents into 2 groups
    embeddings = node.get_all_embeddings()
    labels = kmeans(embeddings, k=2)

    # 2. Partition documents
    docs_a = [d for d, l in zip(node.documents, labels) if l == 0]
    docs_b = [d for d, l in zip(node.documents, labels) if l == 1]

    # 3. Create child nodes
    child_a = KGNode(
        node_id=f"{node.node_id}_a",
        documents=docs_a,
        centroid=compute_centroid(docs_a),
        parent=node.node_id,
    )
    child_b = KGNode(
        node_id=f"{node.node_id}_b",
        documents=docs_b,
        centroid=compute_centroid(docs_b),
        parent=node.node_id,
    )

    # 4. Register with discovery service
    discovery.register(child_a)
    discovery.register(child_b)

    # 5. Convert parent to region node
    node.become_region_node(children=[child_a.node_id, child_b.node_id])

    return child_a, child_b
```

### Node States

```
┌─────────────┐
│  LEAF       │  ← Stores documents, answers queries
└──────┬──────┘
       │ split()
       ▼
┌─────────────┐
│  REGION     │  ← Routes queries to children, no documents
└─────────────┘
```

**Leaf Node:**
- Stores documents and embeddings
- Answers queries directly
- Monitors split triggers
- Splits when thresholds exceeded

**Region Node:**
- No documents (delegated to children)
- Routes queries to appropriate child based on similarity
- Aggregates results from children
- Can have region children (multi-level hierarchy)

### Visual Example

**Before split:**
```
┌─────────────────────────────────┐
│  Node A (1200 documents)        │
│  centroid: [0.2, 0.5, ...]      │
│  topics: [csv, json, xml, yaml] │
│  variance: 0.6 (HIGH)           │
└─────────────────────────────────┘
```

**After split:**
```
┌─────────────────────────────────┐
│  Node A (REGION)                │
│  children: [A_a, A_b]           │
└───────────────┬─────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌───────────────┐ ┌───────────────┐
│  Node A_a     │ │  Node A_b     │
│  600 docs     │ │  600 docs     │
│  [csv, json]  │ │  [xml, yaml]  │
│  variance:0.3 │ │  variance:0.2 │
└───────────────┘ └───────────────┘
```

## Integration with Existing Systems

### Hierarchical Federation (Phase 5a)

Subdivision creates the hierarchy that `HierarchicalFederatedEngine` expects:

```python
# Before: Manual hierarchy configuration
config = HierarchyConfig(
    max_levels=2,
    nodes_per_region=10,
    build_method="centroid"
)

# After: Dynamic hierarchy from subdivision
# Hierarchy builds itself as nodes split
engine = HierarchicalFederatedEngine(router)
# engine.hierarchy is populated automatically from region nodes
```

### Discovery Service

Region nodes register their children:

```python
# Consul service registration
{
    "ID": "node_a",
    "Name": "kg-node",
    "Tags": ["region"],
    "Meta": {
        "node_type": "region",
        "children": "node_a_a,node_a_b",
        "level": "1"
    }
}
```

### Kleinberg Routing

Router considers both leaf and region nodes:

```python
def route_query(query_embedding: np.ndarray) -> List[str]:
    """Route to best nodes, preferring leaves but using regions."""
    candidates = []

    for node in discovery.get_all_nodes():
        sim = cosine_similarity(query_embedding, node.centroid)

        if node.is_region:
            # Region nodes: slightly lower priority (routing overhead)
            candidates.append((node, sim * 0.95))
        else:
            # Leaf nodes: direct query
            candidates.append((node, sim))

    return sorted(candidates, key=lambda x: x[1], reverse=True)[:k]
```

## Prolog Configuration

```prolog
service(adaptive_kg_node, [
    transport(http('/kg', [port(8080)])),

    % Subdivision configuration
    subdivision([
        enabled(true),
        max_documents(1000),
        max_variance(0.5),
        max_latency_ms(500),
        max_memory_percent(80),
        split_method(kmeans),  % or: medoid, random
        min_child_documents(100)  % Don't split if children too small
    ]),

    federation([
        aggregation_strategy(sum),
        federation_k(5),
        hierarchical(auto)  % Build hierarchy from subdivisions
    ])
], Handler).
```

## Merge Operation (Optional)

Nodes can also merge if underutilized:

```python
def should_merge(node_a: KGNode, node_b: KGNode) -> bool:
    """Check if sibling nodes should merge."""
    return (
        node_a.parent == node_b.parent and
        node_a.document_count + node_b.document_count < 500 and
        node_a.query_rate < 10 and  # queries per minute
        node_b.query_rate < 10
    )
```

This handles:
- Deleted content reducing node size
- Changing query patterns making nodes underutilized
- Cluster drift making siblings semantically similar

## Implementation Plan

### Phase 1: Split Detection
- [ ] Add `SplitConfig` dataclass with thresholds
- [ ] Add metrics tracking (variance, latency P99, memory)
- [ ] Implement `should_split()` check
- [ ] Add Prolog validation for subdivision options

### Phase 2: Split Execution
- [ ] Implement k-means partitioning of documents
- [ ] Create child node registration
- [ ] Convert parent to region node
- [ ] Update discovery service

### Phase 3: Region Routing
- [ ] Extend `KleinbergRouter` to handle region nodes
- [ ] Implement query forwarding to children
- [ ] Aggregate results from child responses

### Phase 4: Integration
- [ ] Connect to `HierarchicalFederatedEngine`
- [ ] Add monitoring/alerting for splits
- [ ] Document operational procedures

### Phase 5: Merge (Optional)
- [ ] Implement merge detection
- [ ] Handle document migration
- [ ] Update parent region node

## Success Criteria

1. **Consistent precision** across network sizes (target: 0.90+ at all scales)
2. **No manual hierarchy configuration** required
3. **Query latency P99 < 500ms** maintained during splits
4. **Zero downtime** during split operations
5. **Automatic rebalancing** as content changes

## Alternatives Considered

### 1. Fixed Sharding

Pre-partition by topic/hash. Rejected because:
- Requires upfront knowledge of topic distribution
- Doesn't adapt to changing content
- Uneven shard sizes over time

### 2. External Rebalancer

Separate service monitors and triggers splits. Rejected because:
- Additional operational complexity
- Latency in rebalancing decisions
- Single point of failure

### 3. Client-Side Splitting

Clients decide when to split. Rejected because:
- Inconsistent policies across clients
- No global view of cluster state
- Harder to coordinate

## References

- [BENCHMARK_FINDINGS.md](../guides/BENCHMARK_FINDINGS.md) - Cluster density analysis
- [ROADMAP_KG_TOPOLOGY.md](ROADMAP_KG_TOPOLOGY.md) - Phase 5a Hierarchical Federation
- [KG_TOPOLOGY_FUTURE_WORK.md](KG_TOPOLOGY_FUTURE_WORK.md) - Integration opportunities
- B-tree splitting: https://en.wikipedia.org/wiki/B-tree
- Consistent hashing: https://en.wikipedia.org/wiki/Consistent_hashing

---

## Appendix: Entropy-Guided Intermediate Categories

*Added: 2026-01-13*

### Observation

When computing branching factors from transformer-based entropy:
- **Information-theoretic branching** can exceed **structural branching**
- This indicates some nodes are "more surprising" than their depth warrants
- Example: branching factor 2.32 with structural branching of 2

### Proposal: Adaptive Subdivision Based on Entropy Residuals

When attaching orphans or reorganizing hierarchies, check if entropy jumps are too large:

```python
def should_add_intermediate(parent, child, slope, intercept, threshold=0.5):
    """Check if an intermediate category is needed."""
    parent_depth = get_depth(parent)
    child_entropy = compute_entropy(child.text)
    
    expected_entropy = intercept + slope * (parent_depth + 1)
    residual = child_entropy - expected_entropy
    
    if residual > threshold:
        # Child is too surprising for depth → needs intermediate
        return True
    return False
```

### Algorithm Sketch

1. Compute entropy slope from existing hierarchy
2. When attaching node N to parent P:
   - Compute entropy(N) and expected entropy at depth(P)+1
   - If residual > threshold:
     - Find or create intermediate category
     - Attach N under intermediate instead
3. Intermediate categories could be:
   - LLM-generated from N's text ("Quantum Entanglement" from "Quantum Entanglement Theory")
   - Clustered siblings with similar high residuals
   - Existing nodes that better match expected entropy

### Expected Benefits

- Smoother entropy gradient across levels
- Branching factor closer to structural branching
- More navigable hierarchies (consistent information gain per click)
- Lower objective J (higher effective H from more contributing levels)

### Open Questions

- How to generate good intermediate category names?
- Threshold selection (fixed vs adaptive)?
- Should this be integrated into MST partitioning or post-processing?
