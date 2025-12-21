# Federation Benchmark Findings

**Date:** 2024-12-20
**Benchmark Version:** Phase 6b
**Network Sizes:** 10, 25, 50 nodes
**Queries per Size:** 50

## What We're Measuring

This benchmark tests **federated aggregation**: querying multiple nodes and combining their results.

### Query Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. ROUTING: Select top-k nodes by centroid similarity          │
│    route_multi_k(query_embedding, k=3)                          │
│    → Returns [(node_a, 0.95), (node_b, 0.82), (node_c, 0.71)]   │
├─────────────────────────────────────────────────────────────────┤
│ 2. QUERY: Send query to each selected node in parallel         │
│    results_a = node_a.query(embedding, top_k=10)                │
│    results_b = node_b.query(embedding, top_k=10)                │
│    results_c = node_c.query(embedding, top_k=10)                │
├─────────────────────────────────────────────────────────────────┤
│ 3. AGGREGATE: Combine results using strategy (SUM, MAX, etc)   │
│    aggregate(results, strategy=SUM)                             │
│    → Documents in multiple nodes get boosted scores             │
└─────────────────────────────────────────────────────────────────┘
```

### Metrics Definitions

| Metric | Definition | Code Location |
|--------|------------|---------------|
| **Routing Precision** | Fraction of queried nodes containing relevant docs | `route_multi_k()` in `adaptive_subdivision.py:771` |
| **Result Precision** | Fraction of returned docs that are relevant | `aggregate_results()` in `federated_query.py:350` |
| **Nodes Queried (Hops)** | Number of leaf nodes sent the query | k parameter |
| **P50/P99 Latency** | Median/99th percentile total query time | End-to-end timing |
| **Regions Traversed** | Region nodes in hierarchy (routing overhead) | `HierarchicalFederatedEngine` |

### Key Code References

**Routing** (`adaptive_subdivision.py:771-809`):
```python
def route_multi_k(self, query_embedding, k=3):
    '''Route to top-k LEAF nodes by centroid similarity.'''
    ranked = [(leaf, cosine_sim(query, leaf.centroid)) for leaf in leaves]
    return sorted(ranked, reverse=True)[:k]
```

**Aggregation** (`federated_query.py:350-420`):
```python
def aggregate_results(results_by_node, strategy="sum"):
    '''Combine results from multiple nodes.'''
    for node_id, results in results_by_node.items():
        for doc_id, score in results:
            if strategy == "sum":
                # Consensus boost: docs in multiple nodes score higher
                doc_scores[doc_id] += score
```

## Executive Summary

| Finding | Recommendation |
|---------|----------------|
| Fixed k=3 outperforms adaptive-k for latency | Use fixed k=3 for latency-sensitive workloads |
| Precision is high (87-95%) across all configs | Default SUM strategy works well |
| Larger networks have better precision | Scale up for better results |
| Streaming adds ~65% latency overhead | Only use when progressive UX needed |

## Detailed Results

### Network Size: 10 Nodes

| Config | P50 Latency | Avg Precision | Nodes Queried |
|--------|-------------|---------------|---------------|
| baseline_sum_k3 | 317ms | 0.883 | 3.0 |
| baseline_max_k3 | 317ms | 0.883 | 3.0 |
| adaptive_default | 979ms | 0.872 | 8.0 |
| hierarchical_2level | 318ms | 0.883 | 3.0 |
| streaming_eager | 519ms | 0.872 | 5.0 |
| planned_auto | 318ms | 0.883 | 3.0 |

### Network Size: 25 Nodes

| Config | P50 Latency | Avg Precision | Nodes Queried |
|--------|-------------|---------------|---------------|
| baseline_sum_k3 | 278ms | 0.883 | 3.0 |
| baseline_max_k3 | 278ms | 0.877 | 3.0 |
| adaptive_default | 730ms | 0.876 | 8.0 |
| hierarchical_2level | 278ms | 0.880 | 3.0 |
| streaming_eager | 460ms | 0.873 | 5.0 |
| planned_auto | 278ms | 0.877 | 3.0 |

### Network Size: 50 Nodes

| Config | P50 Latency | Avg Precision | Nodes Queried |
|--------|-------------|---------------|---------------|
| baseline_sum_k3 | 152ms | 0.950 | 3.0 |
| baseline_max_k3 | 153ms | 0.950 | 3.0 |
| adaptive_default | 594ms | 0.943 | 8.0 |
| hierarchical_2level | 152ms | 0.950 | 3.0 |
| streaming_eager | 368ms | 0.942 | 5.0 |
| planned_auto | 153ms | 0.950 | 3.0 |

## Key Insights

### 1. Fixed k=3 is Optimal for Small Networks

The baseline configurations with k=3 consistently outperform adaptive-k:
- **3x faster** latency (152ms vs 594ms at 50 nodes)
- **Equivalent or better precision** (0.950 vs 0.943)

**Why?** Adaptive-k defaults to higher k (8) for most queries, querying more nodes than necessary. The clustered topic distribution means relevant results are concentrated in fewer nodes.

### 2. Precision Improves with Network Size

| Network Size | Avg Precision |
|--------------|---------------|
| 10 nodes | 0.883 |
| 25 nodes | 0.880 |
| 50 nodes | 0.950 |

**Why?** This is an artifact of **cluster density**, not a fundamental property:

| Network | Nodes/Cluster | Effect |
|---------|---------------|--------|
| 10 nodes | 2 per cluster | Sparse - nodes may not reach similarity threshold |
| 50 nodes | 10 per cluster | Dense - many nodes cluster tightly together |

With sparse clusters, ground truth may only contain 1 node (the target). With dense clusters, ground truth contains 3 nodes that all pass the similarity threshold. This makes it easier to achieve 100% precision.

**Limitation:** Synthetic benchmarks conflate "topologically close" with "semantically relevant." Real-world precision depends on content relevance, not just embedding proximity.

**Future idea:** Adaptive node subdivision - nodes could split when they reach a certain size, maintaining optimal cluster density across the network. This would provide consistent precision regardless of total network size.

### 3. Latency Decreases with Network Size

| Network Size | P50 Latency (baseline) |
|--------------|------------------------|
| 10 nodes | 317ms |
| 25 nodes | 278ms |
| 50 nodes | 152ms |

**Why?** The "mixed" latency profile includes 30% fast nodes (10ms). Larger networks have more fast nodes available for routing.

### 4. SUM vs MAX Shows No Difference

Both aggregation strategies produced identical results in these benchmarks. This is expected because:
- Mock nodes return unique results (no duplicates to aggregate)
- Real-world scenarios with duplicate answers would show SUM boosting consensus

### 5. Streaming Overhead is Significant

Streaming (k=5) adds **65-140% latency overhead** compared to baseline (k=3):
- 10 nodes: 519ms vs 317ms (+64%)
- 25 nodes: 460ms vs 278ms (+65%)
- 50 nodes: 368ms vs 152ms (+142%)

**Recommendation:** Only use streaming when progressive UX is required.

### 6. Hierarchical Shows No Benefit at This Scale

Hierarchical federation performs identically to baseline at 10-50 nodes. Benefits would appear at larger scales (100+ nodes) where region-based routing reduces search space.

## Configuration Recommendations

### Low Latency (< 200ms)
```python
engine = FederatedQueryEngine(router, federation_k=3)
config = AggregationConfig(strategy=AggregationStrategy.SUM)
```

### High Precision (> 0.95)
```python
# Use larger networks (50+ nodes) with topic clustering
network = create_synthetic_network(num_nodes=50, topic_distribution="clustered")
```

### Progressive UX
```python
# Accept latency tradeoff for streaming
engine = create_streaming_engine(router, StreamingConfig(emit_interval_ms=100))
```

### Large Networks (100+ nodes)
```python
# Consider hierarchical routing
engine = create_hierarchical_engine(router, HierarchyConfig(max_levels=2))
```

## Methodology Notes

- **Synthetic nodes** with clustered topic distribution (5 clusters)
- **Mixed latency profile**: 60% normal (50ms), 30% fast (10ms), 10% slow (200ms)
- **Query mix**: 40% SPECIFIC, 30% EXPLORATORY, 30% CONSENSUS
- **Ground truth**: Nodes with cosine similarity > threshold
- **Precision**: Intersection of returned nodes with ground truth

## Raw Data

Full results available in:
- `reports/results.json` - Complete benchmark data
- `reports/federation_benchmark.html` - HTML summary

## Subdivision Integration Benchmarks

**Branch:** `feat/adaptive-node-subdivision`
**Test:** `tests/core/test_subdivision_federation_integration.py`

### Results with Subdivided Nodes

| Scenario | k | Nodes | P50 Latency | Routing Precision | Result Precision | Memory |
|----------|---|-------|-------------|-------------------|------------------|--------|
| Small network, k=1 | 1 | 6 | 10.9ms | 1.000 | 1.000 | 0.01 MB |
| Small network, k=3 | 3 | 6 | 32.1ms | 0.667 | 1.000 | 0.01 MB |
| Medium network, k=3 | 3 | 10 | 32.0ms | 0.667 | 1.000 | 0.01 MB |
| Medium network, k=5 | 5 | 10 | 53.4ms | 0.400 | 1.000 | 0.01 MB |

### Key Observations

1. **Routing Precision decreases with higher k** - When k=1, we always pick the best node (100% precision). With k=3, only ~67% of nodes contain relevant docs.

2. **Result Precision remains 100%** - All returned documents are relevant (within their node's cluster).

3. **Latency scales linearly with k** - Each additional node adds ~10ms (simulated query time).

4. **Memory overhead is minimal** - Subdivision hierarchy adds negligible memory.

### Discrimination Improvement

Subdivision dramatically improves routing discrimination:

| State | Discrimination | Explanation |
|-------|---------------|-------------|
| Before split | 0.0108 | Mixed centroid sits between clusters |
| After split | 1.0039 | Child centroids align with clusters |
| **Improvement** | **9181%** | Geometric separation, not numeric precision |

See: `tests/core/test_subdivision_discrimination.py`

## Routing Algorithm Comparison

**Date:** 2024-12-20
**Branch:** `feat/hnsw-routing`, `feat/small-world-proper`

### Critical Insight: Hops vs Comparisons

When optimizing federated search, **network hops dominate latency**, not intra-node comparisons:

| Operation | Typical Time | Bottleneck |
|-----------|-------------|------------|
| Vector comparison (intra-node) | ~1μs | CPU-bound, very fast |
| Network hop (inter-node) | 10-200ms | Network latency dominates |

**Implication:** Optimize for fewer hops, not fewer comparisons.

### Algorithm Comparison (100 nodes)

| Algorithm | Avg Connections | Avg Comparisons | Avg Hops | Scaling |
|-----------|----------------|-----------------|----------|---------|
| Old Small-World | 3.4 | 86.4 | ~5-10 | O(n) |
| **Proper Small-World** | **19.6** | **48.1** | **~2** | O(√n) |
| HNSW | 32.0 | 120.3 | ~3-4 | O(log n) |

### Key Findings

#### 1. Proper Small-World (k-local + k-long) is Best for Mid-Scale

```python
# Each node has 15-20 connections:
# - k_local=10: nearest semantic neighbors
# - k_long=5: random long-range shortcuts
network = SmallWorldProper(k_local=10, k_long=5)
```

- **Clustering coefficient:** 0.48-0.67 (high local clustering)
- **Avg path length:** 1.98-2.42 hops (very short)
- **k-NN success:** 100% from any starting node

#### 2. HNSW is Best for Large Scale (1000+ nodes)

| Nodes | Comparisons | log₂(n) | Ratio |
|-------|-------------|---------|-------|
| 50 | 137 | 5.6 | 24x |
| 100 | 148 | 6.6 | 22x |
| 500 | 201 | 9.0 | 22x |
| 1000 | 207 | 10.0 | 21x |

Constant ratio confirms O(log n) scaling.

#### 3. Backtracking Enables P2P Routing

| Network State | Greedy Success | Backtrack Success |
|---------------|----------------|-------------------|
| Immature | 54% | 99% |
| Mature | 65% | 100% |

Backtracking allows starting from any node with near-100% success.

### When to Use Each Algorithm

| Scale | Network Type | Algorithm | Why |
|-------|-------------|-----------|-----|
| 10-100 nodes | Centralized | Proper Small-World | Fewest hops, simple |
| 100-1000 nodes | Centralized | HNSW | O(log n) scaling |
| Any size | Distributed/P2P | SW + Backtracking | Start anywhere |
| 1000+ nodes | Distributed | HNSW layers as entry points | 47 entry points at layer 1 |

### Layer Distribution (HNSW)

```
Layer 0: 100 nodes (all)     ← full search here
Layer 1:  47 nodes           ← distributed entry points
Layer 2:  21 nodes           ← fewer, faster
Layer 3:   7 nodes           ← coarse routing
```

For P2P: each peer can be an entry point at layer 1.

## Future Work

1. **Test with real embedding models** - Current benchmarks use random embeddings
2. **Add network latency simulation** - Test with realistic network delays
3. **Benchmark adversarial protection** - Measure overhead of Phase 6f features
4. **Test larger scales** - 100, 500, 1000 node networks
5. ~~**Adaptive node subdivision**~~ ✅ Implemented on `feat/adaptive-node-subdivision` branch
6. ~~**HNSW layered routing**~~ ✅ Implemented on `feat/hnsw-routing` branch
7. ~~**Proper small-world connectivity**~~ ✅ Implemented on `feat/small-world-proper` branch
8. ~~**Angle-based neighbor ordering**~~ ✅ Implemented on `feat/small-world-proper` branch
9. ~~**Multi-interface nodes**~~ ✅ Implemented on `feat/multi-interface-nodes` branch

## Angle-Based Optimized Neighbor Lookup

**Status:** ✅ Implemented in `small_world_proper.py`

### Problem

Even with good connectivity, routing compares against all k neighbors (k=15-20) at each hop. We can reduce comparisons by using angle-sorted neighbor lists.

### Key Insight

Sort neighbors by angular distance from node's centroid. For a query, binary search to find neighbors in that angular region.

```
         n3
        /
       /  query direction
      *----→
     /|\
    / | \
  n1  n2  n4   ← neighbors sorted by angle

Binary search: query angle → [n2, n3] candidates
Only compare 2-3 neighbors instead of all 15
```

### Implementation

The centroid is determined by the node's **data**, not routing connections. Neighbors are pointers to other nodes, so routing changes don't affect the centroid.

```python
# In SWNode class (small_world_proper.py)

def add_neighbor(self, neighbor_id: str, neighbor_vector: np.ndarray = None) -> bool:
    """Add neighbor with insertion sort for angle ordering."""
    if neighbor_vector is not None:
        angle = self.compute_neighbor_angle(neighbor_vector)
        # O(log k) search, O(k) insert
        bisect.insort(self.sorted_neighbors, (angle, neighbor_id))
    return True

def lookup_neighbors_by_angle(self, query_vector: np.ndarray, window_size: int = 5) -> List[str]:
    """Find neighbors near query angle using binary search."""
    query_angle = self.compute_neighbor_angle(query_vector)
    idx = bisect.bisect_left(self.sorted_neighbors, (query_angle,))
    # Return neighbors in window, handling wraparound at -pi/+pi
    return candidates_in_window(idx, window_size)
```

### Benchmark Results

| Metric | Standard Search | Optimized Search |
|--------|-----------------|------------------|
| Avg comparisons | 47.5 | 43.8 |
| Reduction | - | **7.9%** |
| Top-1 match rate | 100% | 100% |

### Trade-offs

| Aspect | Cost | Benefit |
|--------|------|---------|
| Insert | O(k) per neighbor | Maintains sorted order |
| Lookup | O(log k) binary search | vs O(k) linear scan |
| Memory | O(k) angles per node | Minimal overhead |
| Rebuild | O(k log k) | Corrects numeric drift |

### When to Use

- **Incremental builds**: Insertion sort maintains order as network grows
- **High query volume**: Amortize the per-neighbor insert cost
- **Latency-sensitive**: Worth the small memory overhead
- **Periodic rebuild**: Call `rebuild_all_sorted_neighbors()` to correct drift

### Alternatives Considered But Not Implemented

| Approach | Why Not | Complexity vs Benefit |
|----------|---------|----------------------|
| **Angular bins (hash-style)** | With k=15-20 neighbors, binary search is already O(4-5) comparisons. Bins add complexity (granularity choice, wraparound handling) without meaningful speedup. | High complexity, low benefit |
| **Quantile-based bins** | Pre-allocating bins based on expected angular distribution. Same issue as above - small k makes this overkill. | Medium complexity, low benefit |
| **PCA-based angle projection** | Using principal components of neighbors instead of first 2 dimensions. Would improve angle accuracy in high-D but adds computational overhead for each angle calculation. | Medium complexity, marginal benefit |
| **Learned neighbor ordering** | Training a model to predict optimal neighbor order. Training cost isn't justified for evolving networks; only beneficial for mature/stable topologies with very high query volume. | High complexity, situational benefit |
| **Dual occupancy pruning** | Using bin collisions to identify redundant neighbors for pruning. Interesting idea but conflates routing optimization with topology management - better kept separate. | Medium complexity, unclear benefit |

**Design principle:** The simple sorted-list + binary-search approach achieves most of the benefit (7.9% reduction) with minimal complexity. More sophisticated approaches would add implementation/maintenance burden without proportional gains given the small neighbor count (k=15-20).

## Multi-Interface Nodes with Scale-Free Distribution

**Status:** ✅ Implemented in `multi_interface_node.py`
**Branch:** `feat/multi-interface-nodes`

### Problem

With binary search, nodes can efficiently handle many more connections (100-1000 vs 15-20). This enables a new architecture: nodes with multiple logical interfaces, each representing a different semantic region.

### Key Insight

A single physical node can expose multiple interfaces:

```
Physical Node (stores 1000 documents)
├── Interface A: "machine learning" region (centroid A)
├── Interface B: "databases" region (centroid B)
└── Interface C: "networking" region (centroid C)

Query arrives → binary search finds closest interface →
returns that interface as the "entry point" to callers
```

This creates **apparent scale** without proportional physical nodes.

### Scale-Free Interface Distribution

Number of interfaces per node follows power law: P(k) ~ k^(-γ)

| Node Type | Interfaces | Frequency | Role |
|-----------|------------|-----------|------|
| Leaves | 1-2 | ~75% | Store focused data |
| Regional | 3-10 | ~20% | Connect related topics |
| Hubs | 10-100+ | ~5% | Highway on-ramps |

Benchmark results (50 nodes, γ=2.5):
```
Interfaces      Nodes
    1           33 (66%)
    2            9 (18%)
  3-4            3  (6%)
  5-9            4  (8%)
   16            1  (2%)  ← hub node
```

### Internal Binary Search

All interfaces share ONE sorted structure:

```python
class MultiInterfaceNode:
    # Single sorted list for ALL connections across ALL interfaces
    connections: List[ExternalConnection]  # sorted by angle

    def route_query(self, query):
        # 1. Binary search interfaces → find closest
        # 2. Binary search connections → find relevant neighbors
        # Both in O(log n) time
```

Efficiency with 200 connections:
- Window size: 10
- Candidates checked: 21 (10.5%)
- Full scan avoided

### Benefits

| Benefit | Mechanism |
|---------|-----------|
| **Apparent scale** | 50 physical nodes → 75+ logical interfaces |
| **Short paths** | Hub nodes act as highway on-ramps |
| **Robustness** | Scale-free networks survive random failures |
| **Internal shortcuts** | Query routes through closest interface, not node boundary |

### When to Use

- **Large data volumes**: Nodes with many documents benefit from multiple interfaces
- **Topic diversity**: Nodes spanning multiple semantic regions
- **Hub architectures**: Some nodes naturally aggregate connections
- **P2P networks**: Each peer can be a multi-interface node
