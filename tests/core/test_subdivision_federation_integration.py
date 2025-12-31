# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Integration test: Subdivision + Federated Query.

This test demonstrates the FULL FLOW:
1. Create nodes with documents
2. Subdivide nodes when they exceed thresholds
3. Route queries to top-k matching nodes
4. Aggregate results from multiple nodes
5. Measure accuracy, latency, hops, and memory

WHAT WE'RE MEASURING:
- Routing Precision: Do we select the right nodes to query?
- Result Precision: Do aggregated results contain relevant documents?
- Latency: Total time from query to aggregated result
- Hops: Number of nodes queried (routing overhead)
- Memory: Peak memory during query processing
"""

import sys
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

from unifyweaver.targets.python_runtime.adaptive_subdivision import (
    SplitConfig,
    SubdividableNode,
    SubdivisionRegistry,
    NodeType,
    split_node,
)


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str

    # Timing
    routing_time_ms: float = 0.0
    query_time_ms: float = 0.0
    aggregation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Routing
    nodes_considered: int = 0  # Total nodes in registry
    nodes_queried: int = 0     # Nodes actually queried (hops)
    regions_traversed: int = 0 # Region nodes visited during routing

    # Results
    results_per_node: List[int] = field(default_factory=list)
    total_results: int = 0
    unique_results: int = 0  # After dedup

    # Accuracy (requires ground truth)
    routing_precision: float = 0.0  # Did we query the right nodes?
    result_precision: float = 0.0   # Did we get the right documents?
    result_recall: float = 0.0      # Did we miss any relevant documents?

    # Memory
    peak_memory_mb: float = 0.0


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across queries."""
    num_queries: int = 0

    # Timing (P50, P99)
    p50_total_time_ms: float = 0.0
    p99_total_time_ms: float = 0.0
    avg_routing_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0

    # Routing
    avg_nodes_queried: float = 0.0
    avg_regions_traversed: float = 0.0

    # Accuracy
    avg_routing_precision: float = 0.0
    avg_result_precision: float = 0.0
    avg_result_recall: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0


# =============================================================================
# MOCK QUERY INFRASTRUCTURE
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def query_node(
    node: SubdividableNode,
    query_embedding: np.ndarray,
    top_k: int = 5,
    latency_ms: float = 10.0
) -> List[Tuple[str, float]]:
    """
    Simulate querying a node for documents.

    Returns documents ranked by similarity to query.
    """
    # Simulate network latency
    time.sleep(latency_ms / 1000.0)

    if not node.embeddings:
        return []

    # Score all documents
    scored = []
    for doc_id, embedding in zip(node.document_ids, node.embeddings):
        sim = cosine_similarity(query_embedding, embedding)
        scored.append((doc_id, sim))

    # Return top-k
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def aggregate_results(
    results_by_node: Dict[str, List[Tuple[str, float]]],
    strategy: str = "sum"
) -> List[Tuple[str, float]]:
    """
    Aggregate results from multiple nodes.

    Strategies:
    - sum: Add scores for documents appearing in multiple nodes (consensus boost)
    - max: Take maximum score per document
    - first: Keep first occurrence only
    """
    doc_scores: Dict[str, float] = {}
    doc_sources: Dict[str, Set[str]] = {}

    for node_id, results in results_by_node.items():
        for doc_id, score in results:
            if doc_id not in doc_sources:
                doc_sources[doc_id] = set()
            doc_sources[doc_id].add(node_id)

            if strategy == "sum":
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
            elif strategy == "max":
                doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), score)
            elif strategy == "first":
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = score

    # Sort by aggregated score
    ranked = [(doc_id, score) for doc_id, score in doc_scores.items()]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =============================================================================
# FULL INTEGRATION TEST
# =============================================================================

def run_federated_query(
    registry: SubdivisionRegistry,
    query_embedding: np.ndarray,
    query_id: str,
    k: int = 3,
    top_k_per_node: int = 5,
    ground_truth_docs: Set[str] = None,
    ground_truth_nodes: Set[str] = None,
) -> QueryMetrics:
    """
    Execute full federated query with metrics.

    Flow:
    1. Route to top-k nodes (with timing)
    2. Query each node in parallel (simulated)
    3. Aggregate results
    4. Compute accuracy vs ground truth
    """
    metrics = QueryMetrics(query_id=query_id)

    # Start memory tracking
    tracemalloc.start()
    start_total = time.time()

    # --- ROUTING PHASE ---
    start_routing = time.time()

    # Count regions traversed during routing
    regions_traversed = 0

    # Get all nodes for counting
    all_nodes = list(registry.nodes.values())
    metrics.nodes_considered = len(all_nodes)

    # Route to top-k leaf nodes
    candidates = registry.route_multi_k(query_embedding, k=k)

    # Count region nodes we passed through
    for node in all_nodes:
        if node.node_type == NodeType.REGION:
            regions_traversed += 1

    metrics.routing_time_ms = (time.time() - start_routing) * 1000
    metrics.nodes_queried = len(candidates)
    metrics.regions_traversed = regions_traversed

    # --- QUERY PHASE ---
    start_query = time.time()

    results_by_node: Dict[str, List[Tuple[str, float]]] = {}
    for node, similarity in candidates:
        results = query_node(node, query_embedding, top_k=top_k_per_node)
        results_by_node[node.node_id] = results
        metrics.results_per_node.append(len(results))

    metrics.query_time_ms = (time.time() - start_query) * 1000

    # --- AGGREGATION PHASE ---
    start_agg = time.time()

    aggregated = aggregate_results(results_by_node, strategy="sum")

    metrics.aggregation_time_ms = (time.time() - start_agg) * 1000
    metrics.total_results = sum(len(r) for r in results_by_node.values())
    metrics.unique_results = len(aggregated)

    # --- ACCURACY COMPUTATION ---
    if ground_truth_nodes:
        queried_node_ids = {node.node_id for node, _ in candidates}
        correct_nodes = queried_node_ids & ground_truth_nodes
        metrics.routing_precision = len(correct_nodes) / len(queried_node_ids) if queried_node_ids else 0.0

    if ground_truth_docs:
        returned_docs = {doc_id for doc_id, _ in aggregated[:top_k_per_node]}
        correct_docs = returned_docs & ground_truth_docs
        metrics.result_precision = len(correct_docs) / len(returned_docs) if returned_docs else 0.0
        metrics.result_recall = len(correct_docs) / len(ground_truth_docs) if ground_truth_docs else 0.0

    # --- FINALIZE ---
    metrics.total_time_ms = (time.time() - start_total) * 1000

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics.peak_memory_mb = peak / (1024 * 1024)

    return metrics


def compute_aggregate_metrics(query_metrics: List[QueryMetrics]) -> AggregatedMetrics:
    """Compute aggregate statistics across queries."""
    if not query_metrics:
        return AggregatedMetrics()

    agg = AggregatedMetrics(num_queries=len(query_metrics))

    # Timing
    total_times = sorted([m.total_time_ms for m in query_metrics])
    agg.p50_total_time_ms = total_times[len(total_times) // 2]
    agg.p99_total_time_ms = total_times[int(len(total_times) * 0.99)]
    agg.avg_routing_time_ms = np.mean([m.routing_time_ms for m in query_metrics])
    agg.avg_query_time_ms = np.mean([m.query_time_ms for m in query_metrics])

    # Routing
    agg.avg_nodes_queried = np.mean([m.nodes_queried for m in query_metrics])
    agg.avg_regions_traversed = np.mean([m.regions_traversed for m in query_metrics])

    # Accuracy
    agg.avg_routing_precision = np.mean([m.routing_precision for m in query_metrics])
    agg.avg_result_precision = np.mean([m.result_precision for m in query_metrics])
    agg.avg_result_recall = np.mean([m.result_recall for m in query_metrics])

    # Memory
    agg.peak_memory_mb = max(m.peak_memory_mb for m in query_metrics)

    return agg


# =============================================================================
# TEST SCENARIOS
# =============================================================================

def create_test_network(
    num_clusters: int = 5,
    docs_per_cluster: int = 100,
    embedding_dim: int = 64,
    split_threshold: int = 80,
) -> Tuple[SubdivisionRegistry, Dict[str, Set[str]]]:
    """
    Create a test network with clustered documents.

    Returns:
        registry: Populated and subdivided registry
        cluster_docs: Mapping from cluster center to document IDs
    """
    np.random.seed(42)
    registry = SubdivisionRegistry()
    cluster_docs: Dict[str, Set[str]] = {}

    # Create one node per cluster initially
    for c in range(num_clusters):
        # Cluster center (one-hot style for clarity)
        center = np.zeros(embedding_dim)
        center[c * (embedding_dim // num_clusters)] = 1.0
        center_key = f"cluster_{c}"
        cluster_docs[center_key] = set()

        node = SubdividableNode(
            node_id=f"node_{c}",
            config=SplitConfig(
                max_documents=split_threshold,
                min_child_documents=10,
            ),
        )

        # Add documents with noise around cluster center
        for d in range(docs_per_cluster):
            doc_id = f"doc_{c}_{d}"
            noise = np.random.randn(embedding_dim) * 0.1
            embedding = center + noise
            embedding = embedding / np.linalg.norm(embedding)

            node.add_document(doc_id, embedding)
            cluster_docs[center_key].add(doc_id)

        registry.register(node)

    # Trigger automatic subdivision
    split_ids = registry.check_and_split_all(seed=42)

    return registry, cluster_docs


def run_benchmark_scenario(
    scenario_name: str,
    num_clusters: int,
    docs_per_cluster: int,
    num_queries: int,
    k: int,
) -> Dict[str, Any]:
    """Run a benchmark scenario and return results."""

    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    print(f"  Clusters: {num_clusters}")
    print(f"  Docs/cluster: {docs_per_cluster}")
    print(f"  Total docs: {num_clusters * docs_per_cluster}")
    print(f"  Queries: {num_queries}")
    print(f"  k (nodes queried): {k}")

    # Create network
    registry, cluster_docs = create_test_network(
        num_clusters=num_clusters,
        docs_per_cluster=docs_per_cluster,
    )

    print(f"\n  Network structure:")
    print(f"    Total nodes: {len(registry.nodes)}")
    print(f"    Leaf nodes: {len(registry.get_leaf_nodes())}")
    print(f"    Region nodes: {len(registry.get_region_nodes())}")

    # Generate queries targeting each cluster
    np.random.seed(123)
    query_metrics = []

    for q in range(num_queries):
        # Target a specific cluster
        target_cluster = q % num_clusters
        center = np.zeros(64)
        center[target_cluster * (64 // num_clusters)] = 1.0

        # Add slight noise to query
        noise = np.random.randn(64) * 0.05
        query_embedding = center + noise
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Ground truth: documents from target cluster
        ground_truth_docs = cluster_docs[f"cluster_{target_cluster}"]

        # Ground truth nodes: nodes containing target cluster docs
        ground_truth_nodes = set()
        for node in registry.get_leaf_nodes():
            if any(doc_id in ground_truth_docs for doc_id in node.document_ids):
                ground_truth_nodes.add(node.node_id)

        # Run federated query
        metrics = run_federated_query(
            registry=registry,
            query_embedding=query_embedding,
            query_id=f"query_{q}",
            k=k,
            ground_truth_docs=ground_truth_docs,
            ground_truth_nodes=ground_truth_nodes,
        )
        query_metrics.append(metrics)

    # Aggregate results
    agg = compute_aggregate_metrics(query_metrics)

    print(f"\n  RESULTS:")
    print(f"  --------")
    print(f"  Timing:")
    print(f"    P50 total latency: {agg.p50_total_time_ms:.2f} ms")
    print(f"    P99 total latency: {agg.p99_total_time_ms:.2f} ms")
    print(f"    Avg routing time: {agg.avg_routing_time_ms:.3f} ms")
    print(f"    Avg query time: {agg.avg_query_time_ms:.2f} ms")
    print(f"  Routing:")
    print(f"    Avg nodes queried (hops): {agg.avg_nodes_queried:.1f}")
    print(f"    Regions traversed: {agg.avg_regions_traversed:.1f}")
    print(f"  Accuracy:")
    print(f"    Routing precision: {agg.avg_routing_precision:.3f}")
    print(f"    Result precision: {agg.avg_result_precision:.3f}")
    print(f"    Result recall: {agg.avg_result_recall:.3f}")
    print(f"  Memory:")
    print(f"    Peak memory: {agg.peak_memory_mb:.2f} MB")

    return {
        "scenario": scenario_name,
        "config": {
            "num_clusters": num_clusters,
            "docs_per_cluster": docs_per_cluster,
            "num_queries": num_queries,
            "k": k,
        },
        "network": {
            "total_nodes": len(registry.nodes),
            "leaf_nodes": len(registry.get_leaf_nodes()),
            "region_nodes": len(registry.get_region_nodes()),
        },
        "metrics": {
            "p50_latency_ms": agg.p50_total_time_ms,
            "p99_latency_ms": agg.p99_total_time_ms,
            "avg_routing_time_ms": agg.avg_routing_time_ms,
            "avg_nodes_queried": agg.avg_nodes_queried,
            "routing_precision": agg.avg_routing_precision,
            "result_precision": agg.avg_result_precision,
            "result_recall": agg.avg_result_recall,
            "peak_memory_mb": agg.peak_memory_mb,
        },
    }


def print_code_snippets():
    """Print relevant code snippets for documentation."""
    print("\n" + "="*60)
    print("RELEVANT CODE SNIPPETS")
    print("="*60)

    print("""
## Routing (route_multi_k in adaptive_subdivision.py:771-809)

```python
def route_multi_k(
    self,
    query_embedding: np.ndarray,
    k: int = 3,
    start_node_id: Optional[str] = None,
) -> List[Tuple[SubdividableNode, float]]:
    '''Route query to top-k LEAF nodes.'''
    leaves = self.get_leaf_nodes()

    # Rank by cosine similarity to query
    ranked = []
    for leaf in leaves:
        sim = self._similarity(query_embedding, leaf.centroid)
        ranked.append((leaf, sim))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:k]
```

## Aggregation (aggregate_results in this test)

```python
def aggregate_results(results_by_node, strategy="sum"):
    '''Combine results from multiple nodes.'''
    doc_scores = {}

    for node_id, results in results_by_node.items():
        for doc_id, score in results:
            if strategy == "sum":
                # Consensus boost: docs in multiple nodes get higher scores
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
            elif strategy == "max":
                doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), score)

    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
```

## What We're Measuring

| Metric | Definition |
|--------|------------|
| Routing Precision | Fraction of queried nodes that contain relevant docs |
| Result Precision | Fraction of returned docs that are relevant |
| Result Recall | Fraction of relevant docs that were returned |
| Nodes Queried (Hops) | Number of leaf nodes we sent the query to |
| Regions Traversed | Number of region nodes in the hierarchy |
| Peak Memory | Maximum memory used during query processing |
""")


if __name__ == "__main__":
    print("="*60)
    print("SUBDIVISION + FEDERATION INTEGRATION BENCHMARK")
    print("="*60)
    print("""
This benchmark tests the FULL FLOW:
1. Create clustered document network
2. Auto-subdivide nodes exceeding thresholds
3. Route queries to top-k matching nodes
4. Aggregate results from multiple nodes
5. Measure accuracy, latency, hops, memory
""")

    results = []

    # Scenario 1: Small network, k=1 (single node)
    results.append(run_benchmark_scenario(
        scenario_name="Small network, single-node query (k=1)",
        num_clusters=3,
        docs_per_cluster=100,
        num_queries=30,
        k=1,
    ))

    # Scenario 2: Small network, k=3 (federated)
    results.append(run_benchmark_scenario(
        scenario_name="Small network, federated query (k=3)",
        num_clusters=3,
        docs_per_cluster=100,
        num_queries=30,
        k=3,
    ))

    # Scenario 3: Medium network, k=3
    results.append(run_benchmark_scenario(
        scenario_name="Medium network, federated query (k=3)",
        num_clusters=5,
        docs_per_cluster=100,
        num_queries=50,
        k=3,
    ))

    # Scenario 4: Medium network, k=5
    results.append(run_benchmark_scenario(
        scenario_name="Medium network, federated query (k=5)",
        num_clusters=5,
        docs_per_cluster=100,
        num_queries=50,
        k=5,
    ))

    # Print code snippets
    print_code_snippets()

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Scenario':<45} {'k':>3} {'Nodes':>6} {'P50ms':>7} {'Routing':>8} {'Result':>8} {'Mem MB':>7}")
    print("-"*95)
    for r in results:
        print(f"{r['scenario'][:44]:<45} {r['config']['k']:>3} "
              f"{r['network']['leaf_nodes']:>6} "
              f"{r['metrics']['p50_latency_ms']:>7.1f} "
              f"{r['metrics']['routing_precision']:>8.3f} "
              f"{r['metrics']['result_precision']:>8.3f} "
              f"{r['metrics']['peak_memory_mb']:>7.2f}")
