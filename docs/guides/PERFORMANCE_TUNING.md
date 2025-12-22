# Performance Tuning Guide

**Phase 6b: Federation Performance Optimization**

This guide provides recommendations for optimizing federated semantic search performance based on benchmark findings.

## Quick Reference

| Scenario | Recommended Config | K Value | Strategy |
|----------|-------------------|---------|----------|
| Low latency, few nodes (<20) | Basic | k=3 | MAX |
| High precision, many nodes | Adaptive | 1-10 | SUM |
| Known topic clusters | Hierarchical | k=3 | SUM |
| Real-time results needed | Streaming | k=5 | SUM |
| Optimal routing | Query Planner | auto | auto |

## Federation-K Selection

The `federation_k` parameter controls how many nodes are queried in parallel. This is the most impactful performance tuning option.

### Fixed-K Guidelines

| Network Size | Low Latency | Balanced | High Precision |
|--------------|-------------|----------|----------------|
| 10 nodes | k=2 | k=3 | k=5 |
| 25 nodes | k=3 | k=5 | k=7 |
| 50 nodes | k=3 | k=5 | k=10 |
| 100+ nodes | k=3 | k=7 | k=15 |

**When to use fixed-k:**
- Predictable latency requirements
- Homogeneous node response times
- Simple deployment without adaptive overhead

### Adaptive-K Configuration

Adaptive federation-k adjusts based on query characteristics:

```python
from federated_query import AdaptiveKConfig, create_adaptive_engine

config = AdaptiveKConfig(
    base_k=3,           # Starting point
    min_k=1,            # For high-similarity queries
    max_k=10,           # For exploratory queries
    entropy_weight=0.3, # Weight for similarity entropy
    latency_weight=0.2, # Weight for latency history
    consensus_weight=0.5 # Weight for consensus needs
)

engine = create_adaptive_engine(router, config)
```

**When to use adaptive-k:**
- Mixed query types (specific + exploratory)
- Variable node latencies
- When precision matters more than predictable latency

## Aggregation Strategy Selection

### SUM (Default, Consensus-Boosting)

```python
engine = FederatedQueryEngine(router, aggregation_config=AggregationConfig(
    strategy=AggregationStrategy.SUM
))
```

**Best for:**
- Multi-source verification
- Consensus-seeking queries
- When duplicate results indicate higher confidence

**Tradeoff:** Higher combined scores for duplicates may over-boost popular answers.

### MAX (No Consensus Boost)

```python
config = AggregationConfig(strategy=AggregationStrategy.MAX)
```

**Best for:**
- Single-source authority queries
- When the best single answer is preferred
- Avoiding popularity bias

**Tradeoff:** Ignores cross-node agreement signals.

### DIVERSITY_WEIGHTED (Source Diversity)

```python
config = AggregationConfig(strategy=AggregationStrategy.DIVERSITY_WEIGHTED)
```

**Best for:**
- Cross-corpus research
- When answer diversity is valuable
- Avoiding echo-chamber effects

**Tradeoff:** May prefer worse answers if they come from diverse sources.

## Query Type Optimization

### SPECIFIC Queries (High Similarity to One Node)

Characteristics:
- Query embedding very similar to one node's centroid
- Expected to find answer in 1-2 nodes
- Low similarity variance

**Recommended settings:**
```python
# Use lower k for efficiency
engine = create_adaptive_engine(router, AdaptiveKConfig(
    min_k=1,
    max_k=5,
    similarity_threshold=0.8  # High threshold triggers low k
))
```

### EXPLORATORY Queries (Broad Search)

Characteristics:
- Low/uniform similarity across nodes
- Need to search broadly
- Tolerance for higher latency

**Recommended settings:**
```python
# Use higher k for coverage
engine = create_adaptive_engine(router, AdaptiveKConfig(
    base_k=5,
    max_k=15,
    entropy_threshold=0.5  # Lower threshold keeps k higher
))
```

### CONSENSUS Queries (Agreement-Seeking)

Characteristics:
- Medium similarity to multiple nodes
- Value cross-node agreement
- May benefit from density scoring

**Recommended settings:**
```python
config = AggregationConfig(
    strategy=AggregationStrategy.SUM,  # Boost consensus
    consensus_threshold=2,              # Require 2+ sources
    density_weight=0.3                  # Enable density scoring
)
```

## Backtracking Configuration

Small-world routing supports backtracking to escape local minima. By default, backtracking is **enabled** for reliability.

### Benchmark Results

| Network Size | Greedy Success | Backtrack Success | Greedy Comps | Backtrack Comps |
|--------------|----------------|-------------------|--------------|-----------------|
| 20 nodes | 74% | 100% | 7.4 | 43 (5.8x) |
| 50 nodes | 68% | 100% | 14 | 96 (6.8x) |
| 100 nodes | 66% | 98% | 24 | 163 (6.7x) |
| 200 nodes | 52% | 96% | 36 | 296 (8.3x) |

**Key findings:**
- Backtracking improves success rate from ~50-70% to ~96-100%
- Overhead is ~6-8x more comparisons
- Path lengths are actually **shorter** with backtracking (fewer hops to target)

### When to Disable Backtracking

Use `use_backtrack=False` only when:
- Latency is critical and partial results are acceptable
- Network is mature with many shortcuts
- Starting from known-good positions (e.g., from root in hierarchical network)

```python
from small_world_evolution import SmallWorldNetwork

network = SmallWorldNetwork()
# ... add nodes ...

# High reliability (default)
path, comps = network.route_greedy(query, use_backtrack=True)

# Lower latency, may miss target
path, comps = network.route_greedy(query, use_backtrack=False)
```

### Start-Anywhere Performance

Random-start routing (e.g., federated search from any node) especially benefits from backtracking:

| Condition | Greedy | Backtrack |
|-----------|--------|-----------|
| Immature network | 54% | 99% |
| Mature network | 57% | 100% |

For federated queries that may enter the network at arbitrary nodes, **keep backtracking enabled**.

## Hierarchical Federation

For large networks (50+ nodes), hierarchical routing can reduce query scope:

```python
from federated_query import create_hierarchical_engine, HierarchyConfig

config = HierarchyConfig(
    max_levels=2,
    nodes_per_region=10,
    region_overlap=0.2,  # 20% overlap between regions
    build_method="centroid"  # or "topic"
)

engine = create_hierarchical_engine(router, config)
```

**Benefits:**
- Reduces nodes queried from O(n) to O(log n)
- Exploits topic clustering
- Better for networks with clear specialization

**Tradeoffs:**
- Additional setup complexity
- May miss results if query crosses region boundaries
- Requires periodic hierarchy rebuilding

## Streaming Configuration

For real-time feedback during long queries:

```python
from federated_query import create_streaming_engine, StreamingConfig

config = StreamingConfig(
    emit_interval_ms=100,    # Emit partial results every 100ms
    min_confidence=0.5,       # Only emit when 50%+ confident
    early_termination=True,   # Stop early if confident
    termination_threshold=0.9 # Stop at 90% confidence
)

engine = create_streaming_engine(router, config)

# Async streaming usage
async for partial in engine.stream_query(embedding, top_k=10):
    print(f"Confidence: {partial.confidence}, Results: {len(partial.results)}")
    if partial.is_final:
        break
```

**Benefits:**
- Improved perceived latency
- Progressive refinement UX
- Early termination saves resources

**Tradeoffs:**
- More complex client handling
- Overhead for very fast queries
- May emit unstable intermediate results

## Adversarial Protection Overhead

Phase 6f adds adversarial protection. Here's the performance impact:

| Feature | Overhead | When to Enable |
|---------|----------|----------------|
| Outlier smoothing | ~1-2% | Semi-trusted networks |
| Collision detection | ~5-10% | Public networks |
| Trust weighting | ~2-5% | Networks with history |
| Full protection | ~10-15% | Hostile environments |

```python
config = AggregationConfig(
    adversarial=AdversarialConfig(
        # Enable only what you need
        outlier_rejection=True,  # Low overhead
        collision_detection=False,  # Higher overhead
        trust_enabled=False,
    )
)
```

## Monitoring Recommendations

Key metrics to track in production:

```python
stats = engine.get_stats()

# Watch these metrics
critical_metrics = {
    "p99_latency_ms": stats.get("p99_latency_ms"),
    "avg_nodes_queried": stats.get("avg_nodes_queried"),
    "error_rate": stats.get("error_rate"),
    "consensus_rate": stats.get("avg_consensus_rate"),
}
```

**Alert thresholds:**
- P99 latency > 5x P50 latency
- Error rate > 5%
- Avg nodes queried > 2x federation_k
- Consensus rate < 50% (for SUM strategy)

## Running Benchmarks

Use the benchmark suite to test configurations on your data:

```bash
# Quick test
python -m benchmarks.federation.run_benchmarks --nodes 10 --queries 20 --quick

# Full scalability test
python -m benchmarks.federation.run_benchmarks --nodes 10,25,50 --queries 50 --output reports/

# Test specific configs
python -m benchmarks.federation.run_benchmarks --configs baseline_sum_k3,adaptive_default
```

## Summary

1. **Start simple**: Use `federation_k=3` with SUM aggregation
2. **Measure first**: Run benchmarks before optimizing
3. **Match k to network size**: Larger networks need slightly higher k
4. **Use adaptive-k** for mixed query patterns
5. **Enable hierarchical** for 50+ nodes with clear topic clusters
6. **Add streaming** for real-time applications
7. **Enable adversarial protection** based on trust level

## See Also

- [KG_TOPOLOGY_EXAMPLES.md](KG_TOPOLOGY_EXAMPLES.md) - Usage examples
- [KG_PRODUCTION_DEPLOYMENT.md](KG_PRODUCTION_DEPLOYMENT.md) - Deployment guide
- [ROADMAP_KG_TOPOLOGY.md](../proposals/ROADMAP_KG_TOPOLOGY.md) - Feature roadmap
