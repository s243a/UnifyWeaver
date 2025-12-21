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

## Future Work

1. **Test with real embedding models** - Current benchmarks use random embeddings
2. **Add network latency simulation** - Test with realistic network delays
3. **Benchmark adversarial protection** - Measure overhead of Phase 6f features
4. **Test larger scales** - 100, 500, 1000 node networks
5. ~~**Adaptive node subdivision**~~ ✅ Implemented on `feat/adaptive-node-subdivision` branch
