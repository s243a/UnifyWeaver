# Federated Query Algebra for Distributed KG Topology

## Status: Implemented (Phase 4a-4c)

## Overview

This proposal extends KG Topology Phase 3 (Kleinberg routing) with a **federated query protocol** that treats distributed KG queries as a form of **distributed query algebra** with pluggable aggregation functions.

Key insight: **Deduplication is just aggregation**. Strategies like "boost duplicates" (SUM), "take best" (MAX), or "count consensus" (COUNT) are the same operations used in SQL GROUP BY. This framing allows us to build a general-purpose distributed query engine rather than a special-case federation system.

## Motivation

Phase 3 implements single-path greedy routing:
```
Query → Find most similar node → Forward → Return results
```

This leaves value on the table:
- **Diverse expertise** - Multiple nodes may have complementary answers
- **Fault tolerance** - Single node failure breaks the query path
- **Consensus detection** - Agreement across independent sources adds confidence

Federated queries address this by querying multiple nodes and aggregating results.

## Core Insight: Distributed Softmax Decomposition

Standard softmax for result ranking:
```
P(result_i) = exp(z_i) / Σ_k exp(z_k)
```

This **decomposes** across nodes:
```
P(result_i) = exp(z_i) / (Σ_node1 exp(z_k) + Σ_node2 exp(z_k) + ...)
            = exp(z_i) / Σ_nodes partition_sum_n
```

Each node returns:
- `exp_scores[]` - Numerators for its local results
- `partition_sum` - Local normalizer: Σ exp(z_k)

Aggregation is then **associative and commutative** - nodes can merge in any order.

## Query Algebra

### SQL-Like Semantics

Federated KG queries map to relational algebra:

```sql
SELECT
    answer_text,
    AGG(exp_score) as combined_score,      -- pluggable aggregation
    SUM(partition_sum) as total_partition,
    COUNT(DISTINCT source_node) as node_count
FROM
    federated_query('How do I parse CSV?', top_k=3)
GROUP BY
    answer_hash                             -- dedup key
HAVING
    node_count >= 2                         -- consensus filter (optional)
ORDER BY
    combined_score / total_partition DESC   -- normalized probability
LIMIT 10
```

### Aggregation Functions

| Function | Local Value | Merge Operation | Use Case |
|----------|-------------|-----------------|----------|
| `SUM` | `exp(z_i)` | `a + b` | Boost consensus (assumes independence) |
| `MAX` | `exp(z_i)` | `max(a, b)` | Dedupe, no boost (conservative) |
| `MIN` | `exp(z_i)` | `min(a, b)` | Pessimistic estimate |
| `AVG` | `(sum, count)` | `(s1+s2, c1+c2)` | Balanced estimate |
| `COUNT` | `1` | `a + b` | Measure consensus strength |
| `DISTINCT` | `{hash}` | `a ∪ b` | Set union for dedup |
| `FIRST` | `(value, ts)` | `min by ts` | Keep earliest response |
| `COLLECT` | `[value]` | `a ++ b` | Keep all (no dedup) |

### Algebraic Properties

For correct distributed aggregation, functions must be:

1. **Associative**: `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)` - enables tree-structured aggregation
2. **Commutative**: `a ⊕ b = b ⊕ a` - order-independent merging
3. **Identity element**: `a ⊕ e = a` - handles missing/empty results

All standard SQL aggregates satisfy these properties.

## Protocol Design

### Query Request

```json
{
    "__type": "kg_federated_query",
    "__id": "uuid-123",
    "__routing": {
        "origin_node": "node_a",
        "htl": 8,
        "visited": ["node_a"],
        "federation_k": 3,
        "aggregation": {
            "score_function": "SUM",
            "dedup_key": "answer_hash",
            "consensus_threshold": null
        }
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

### Node Response

```json
{
    "__type": "kg_federated_response",
    "__id": "uuid-123",
    "source_node": "node_b",
    "results": [
        {
            "answer_id": 42,
            "answer_text": "Use csv.reader() or pandas.read_csv()",
            "answer_hash": "a1b2c3d4",
            "raw_score": 0.85,
            "exp_score": 2.3396,
            "metadata": {
                "source_file": "python_faq.txt",
                "record_id": 123
            }
        }
    ],
    "partition_sum": 7.234,
    "node_metadata": {
        "corpus_id": "python_docs_2024",
        "embedding_model": "all-MiniLM-L6-v2",
        "interface_id": 5
    }
}
```

### Aggregated Response

```json
{
    "__type": "kg_aggregated_response",
    "__id": "uuid-123",
    "results": [
        {
            "answer_text": "Use csv.reader() or pandas.read_csv()",
            "answer_hash": "a1b2c3d4",
            "combined_score": 4.521,
            "normalized_prob": 0.312,
            "source_nodes": ["node_b", "node_c"],
            "node_count": 2,
            "provenance": [
                {"node": "node_b", "exp_score": 2.339, "corpus_id": "python_docs_2024"},
                {"node": "node_c", "exp_score": 2.182, "corpus_id": "stackoverflow_2024"}
            ]
        }
    ],
    "total_partition_sum": 14.502,
    "nodes_queried": 3,
    "nodes_responded": 3
}
```

## Handling Duplicate Bias

### The Independence Problem

Duplicate results have different meanings depending on source independence:

| Scenario | Duplicate Interpretation | Recommended Strategy |
|----------|--------------------------|----------------------|
| Independent corpora | Consensus → high confidence | `SUM` (boost) |
| Shared source data | Echo/bias → no new info | `MAX` (no boost) |
| Unknown provenance | Uncertain | `AVG` (hedge) |

### Source Diversity Tracking

Nodes advertise provenance in discovery metadata:

```prolog
discovery_metadata([
    corpus_id('wikipedia_2024'),
    embedding_model('all-MiniLM-L6-v2'),
    data_sources(['wiki_dump', 'common_crawl']),
    last_updated('2024-01-15')
])
```

### Diversity-Weighted Aggregation

```python
class DiversityWeightedAggregator:
    def merge(self, result_a, result_b):
        # Check if sources are independent
        if result_a.corpus_id != result_b.corpus_id:
            # Independent sources - boost
            return result_a.exp_score + result_b.exp_score
        else:
            # Same source - no boost
            return max(result_a.exp_score, result_b.exp_score)
```

### Configurable Strategy

```python
class AggregationStrategy(Enum):
    SUM = "sum"                    # Always boost duplicates
    MAX = "max"                    # Never boost duplicates
    AVG = "avg"                    # Average scores
    FIRST = "first"               # Keep first seen
    DIVERSITY_WEIGHTED = "diversity"  # Boost only if sources differ
    CUSTOM = "custom"             # User-provided function
```

## Implementation

### FederatedQueryEngine Class

```python
class FederatedQueryEngine:
    def __init__(
        self,
        router: KleinbergRouter,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.SUM,
        federation_k: int = 3,
        timeout_ms: int = 5000
    ):
        self.router = router
        self.strategy = aggregation_strategy
        self.federation_k = federation_k
        self.timeout_ms = timeout_ms

    def federated_query(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> AggregatedResponse:
        # 1. Discover top-k nodes by centroid similarity
        nodes = self.router.discover_nodes()
        ranked = self.router.compute_routing_probability(nodes, query_embedding)
        target_nodes = ranked[:self.federation_k]

        # 2. Query all nodes in parallel
        responses = self._parallel_query(target_nodes, query_text, query_embedding)

        # 3. Aggregate results
        aggregated = self._aggregate(responses)

        # 4. Normalize and rank
        return self._normalize_and_rank(aggregated, top_k)

    def _aggregate(self, responses: List[NodeResponse]) -> Dict[str, AggregatedResult]:
        results = {}  # answer_hash -> AggregatedResult
        total_partition = 0.0

        for resp in responses:
            total_partition += resp.partition_sum

            for r in resp.results:
                key = r.answer_hash
                if key in results:
                    results[key] = self._merge(results[key], r, resp.node_metadata)
                else:
                    results[key] = AggregatedResult.from_single(r, resp.node_metadata)

        return results, total_partition

    def _merge(self, existing, new, node_metadata):
        if self.strategy == AggregationStrategy.SUM:
            existing.combined_score += new.exp_score
        elif self.strategy == AggregationStrategy.MAX:
            existing.combined_score = max(existing.combined_score, new.exp_score)
        elif self.strategy == AggregationStrategy.DIVERSITY_WEIGHTED:
            if self._is_diverse(existing.provenance, node_metadata):
                existing.combined_score += new.exp_score
            else:
                existing.combined_score = max(existing.combined_score, new.exp_score)

        existing.source_nodes.append(node_metadata.node_id)
        existing.provenance.append(new)
        return existing
```

### Prolog Integration

```prolog
%% federated_query(+QueryText, +Options, -Results)
%  Execute federated query across KG network.
%
%  Options:
%    - federation_k(K): Number of nodes to query (default: 3)
%    - aggregation(Strategy): sum, max, avg, diversity (default: sum)
%    - timeout_ms(T): Query timeout (default: 5000)
%    - consensus_threshold(N): Minimum node agreement (optional)

federated_query(QueryText, Options, Results) :-
    get_option(federation_k, Options, 3, K),
    get_option(aggregation, Options, sum, Strategy),
    discover_top_k_nodes(K, Nodes),
    parallel_query_nodes(Nodes, QueryText, Responses),
    aggregate_responses(Responses, Strategy, Aggregated),
    normalize_results(Aggregated, Results).

%% Aggregation is a fold over merge operations
aggregate_responses(Responses, Strategy, Aggregated) :-
    foldl(merge_response(Strategy), Responses, empty_aggregate, Aggregated).

merge_response(sum, Response, Acc, NewAcc) :-
    merge_with_sum(Response, Acc, NewAcc).
merge_response(max, Response, Acc, NewAcc) :-
    merge_with_max(Response, Acc, NewAcc).
merge_response(diversity, Response, Acc, NewAcc) :-
    merge_with_diversity(Response, Acc, NewAcc).
```

## Relation to MapReduce and Datalog

### MapReduce Correspondence

| MapReduce | Federated Query |
|-----------|-----------------|
| Map | Local node query |
| Shuffle | Group by answer_hash |
| Reduce | Aggregation function |

### Datalog Aggregation

Federated queries can be expressed in Datalog with aggregation:

```datalog
% Local results from each node
local_result(Node, AnswerHash, ExpScore, Text) :-
    kg_node(Node),
    query_node(Node, QueryText, AnswerHash, ExpScore, Text).

% Aggregated results
federated_result(AnswerHash, Text, TotalScore, NodeCount) :-
    local_result(_, AnswerHash, _, Text),
    TotalScore = sum(ExpScore : local_result(_, AnswerHash, ExpScore, _)),
    NodeCount = count(Node : local_result(Node, AnswerHash, _, _)).
```

## Phase 5: Advanced Features ✅ Complete

These extensions have been implemented in Phase 5. See `ROADMAP_KG_TOPOLOGY.md` for details.

### 1. Hierarchical Federation ✅ (Phase 5a)

Networks of networks with representative centroids:

```
Global Query
    ├── Region A (centroid_A)
    │   ├── Node A1
    │   └── Node A2
    └── Region B (centroid_B)
        ├── Node B1
        └── Node B2
```

**Implementation:** `NodeHierarchy`, `RegionalNode`, `HierarchicalFederatedEngine`

### 2. Adaptive Federation-K ✅ (Phase 5b)

Dynamically adjust `federation_k` based on:
- Query ambiguity (low top similarity → query more nodes)
- Historical consensus patterns
- Response latency constraints

**Implementation:** `AdaptiveKCalculator`, `AdaptiveFederatedEngine`, `QueryMetrics`

### 3. Streaming Aggregation ✅ (Phase 5d)

For long-running queries, stream partial aggregates:

```python
async def streaming_federated_query(query):
    async for partial in parallel_query_stream(nodes, query):
        yield incremental_aggregate(partial)
```

**Implementation:** `StreamingFederatedEngine`, `PartialResult`, `federated_query_sse()`

### 4. Query Plan Optimization ✅ (Phase 5c)

Optimize federation strategy based on query characteristics:
- High-specificity queries → fewer nodes, greedy routing
- Exploratory queries → more nodes, broader federation

**Implementation:** `QueryPlanner`, `PlanExecutor`, `PlannedQueryEngine`, `QueryType` enum

## Implementation Phases

### Phase 4a: Core Federation ✅ Complete
- [x] `FederatedQueryEngine` class (`federated_query.py`)
- [x] 6 aggregation functions: SUM, MAX, MIN, AVG, COUNT, FIRST (monoid-based)
- [x] Parallel node querying with ThreadPoolExecutor
- [x] Protocol messages: `NodeResult`, `NodeResponse`, `AggregatedResult`, `AggregatedResponse`
- [x] Distributed softmax with `exp_scores[]` + `partition_sum`
- [x] 9 Prolog validation predicates in `service_validation.pl`
- [x] 38 unit tests

### Phase 4b: Diversity Tracking ✅ Complete
- [x] `corpus_id` and `data_sources` in discovery metadata
- [x] Auto-generation of corpus_id from database content hash
- [x] Three-tier diversity-weighted aggregation:
  - Different corpus → full boost (SUM)
  - Same corpus, disjoint data_sources → partial boost (AVG of SUM/MAX)
  - Same corpus, overlapping sources → no boost (MAX)
- [x] `ResultProvenance` with full tracking (node_id, exp_score, corpus_id, data_sources, embedding_model)
- [x] `diversity_score` and `unique_corpora` in aggregated responses
- [x] 7 additional unit tests (45 total)

### Phase 4c: Prolog Integration ✅ Complete
- [x] `compile_federated_query_python/2` - Python FederatedQueryEngine factory
- [x] `compile_federated_service_python/2` - Complete Flask service with federation
- [x] `compile_federated_query_go/2` - Go FederatedQueryEngine with full types
- [x] `generate_federation_endpoint/3` - HTTP endpoints for Python/Go/Rust
- [x] Endpoints: `/kg/federated`, `/kg/federate`, `/kg/federation/stats`
- [x] Integration with Phase 3 Kleinberg routing

**Implementation Files:**
- `src/unifyweaver/targets/python_runtime/federated_query.py` - Core engine (~700 lines)
- `src/unifyweaver/targets/python_runtime/kg_topology_api.py` - Extended with federation
- `src/unifyweaver/targets/python_target.pl` - Python code generation
- `src/unifyweaver/targets/go_target.pl` - Go code generation
- `src/unifyweaver/glue/network_glue.pl` - Federation endpoints
- `src/unifyweaver/core/service_validation.pl` - Federation validation
- `tests/core/test_federated_query.py` - 45 unit tests

### Phase 4d: Advanced Features (Future)
- [ ] Hierarchical federation (tree-structured aggregation)
- [ ] Adaptive federation-k (adjust based on query complexity)
- [ ] Query plan optimization (push-down predicates)
- [ ] Density-based confidence scoring (semantic clustering)

## References

- Phase 3: Distributed Network (ROADMAP_KG_TOPOLOGY.md)
- Kleinberg Small-World Networks
- MapReduce: Simplified Data Processing on Large Clusters
- Datalog and Recursive Query Processing
