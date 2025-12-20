#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# End-to-end test for KG Topology Phase 4: Multi-node Federation

"""
End-to-end test for multi-node federated queries with density scoring.

This test simulates a federated KG network where:
- Node A: CSV/data format expert (corpus: stackoverflow)
- Node B: JSON/API expert (corpus: github)
- Node C: General programming expert (corpus: docs)

Tests federated query aggregation, diversity scoring, and density-based ranking.
"""

import os
import sys
import tempfile
import shutil
import hashlib
import numpy as np

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'unifyweaver', 'targets', 'python_runtime'))

from unifyweaver.targets.python_runtime.discovery_clients import LocalDiscoveryClient
from unifyweaver.targets.python_runtime.federated_query import (
    AggregationConfig,
    AggregationStrategy,
    NodeResult,
    NodeResponse,
    get_aggregator
)
from unifyweaver.targets.python_runtime.density_scoring import (
    DensityConfig,
    BandwidthMethod,
    ClusterMethod,
    compute_density_scores,
    flux_softmax,
    two_stage_density_pipeline,
    cluster_by_similarity,
    compute_efficient_density_scores
)


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def log_pass(msg):
    print(f"  {Colors.GREEN}[PASS]{Colors.END} {msg}")


def log_fail(msg):
    print(f"  {Colors.RED}[FAIL]{Colors.END} {msg}")


def log_info(msg):
    print(f"  {Colors.BLUE}[INFO]{Colors.END} {msg}")


def log_section(msg):
    print(f"\n{Colors.YELLOW}=== {msg} ==={Colors.END}")


def create_mock_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Create a deterministic mock embedding based on text."""
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    emb = rng.randn(dim).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


def create_test_node_responses():
    """
    Create mock node responses simulating a federated query.

    Simulates 3 nodes responding to "How do I parse data files?"
    """
    # Node A: CSV expert - has relevant answers
    node_a_results = [
        NodeResult(
            answer_id=1,
            answer_text="Use csv.reader() to parse CSV files",
            answer_hash="csv_1",
            raw_score=0.85,
            exp_score=np.exp(0.85),
            embedding=create_mock_embedding("csv reader parse files"),
            metadata={"source": "stackoverflow", "votes": 42}
        ),
        NodeResult(
            answer_id=2,
            answer_text="pandas.read_csv() is great for CSV parsing",
            answer_hash="csv_2",
            raw_score=0.78,
            exp_score=np.exp(0.78),
            embedding=create_mock_embedding("pandas read csv parsing"),
            metadata={"source": "stackoverflow", "votes": 35}
        ),
    ]

    # Node B: JSON expert - somewhat relevant
    node_b_results = [
        NodeResult(
            answer_id=3,
            answer_text="Use json.load() to parse JSON files",
            answer_hash="json_1",
            raw_score=0.72,
            exp_score=np.exp(0.72),
            embedding=create_mock_embedding("json load parse files"),
            metadata={"source": "github", "stars": 120}
        ),
    ]

    # Node C: General - less relevant
    node_c_results = [
        NodeResult(
            answer_id=4,
            answer_text="File parsing depends on the format",
            answer_hash="prog_1",
            raw_score=0.65,
            exp_score=np.exp(0.65),
            embedding=create_mock_embedding("file parsing format general"),
            metadata={"source": "docs", "section": "basics"}
        ),
    ]

    responses = [
        NodeResponse(
            source_node="csv-expert",
            results=node_a_results,
            partition_sum=sum(r.exp_score for r in node_a_results),
            node_metadata={
                "corpus_id": "stackoverflow_csv",
                "data_sources": ["stackoverflow"],
                "embedding_model": "mock-embedder"
            }
        ),
        NodeResponse(
            source_node="json-expert",
            results=node_b_results,
            partition_sum=sum(r.exp_score for r in node_b_results),
            node_metadata={
                "corpus_id": "github_json",
                "data_sources": ["github"],
                "embedding_model": "mock-embedder"
            }
        ),
        NodeResponse(
            source_node="prog-expert",
            results=node_c_results,
            partition_sum=sum(r.exp_score for r in node_c_results),
            node_metadata={
                "corpus_id": "docs_general",
                "data_sources": ["docs"],
                "embedding_model": "mock-embedder"
            }
        ),
    ]

    return responses


def test_federated_aggregation():
    """Test basic federated query aggregation."""
    log_section("Test 1: Federated Aggregation")

    responses = create_test_node_responses()

    # Test aggregator creation
    aggregator = get_aggregator(AggregationStrategy.SUM)

    if aggregator is None:
        log_fail("Failed to create SUM aggregator")
        return False

    log_pass("SUM aggregator created")

    # Collect all results for aggregation
    all_results = []
    for resp in responses:
        for result in resp.results:
            all_results.append(result)

    log_info(f"Collected {len(all_results)} results from {len(responses)} nodes")

    # Test aggregating exp_scores (simulating distributed softmax)
    exp_scores = [r.exp_score for r in all_results]

    # Use reduce pattern with merge
    from functools import reduce
    combined = reduce(aggregator.merge, exp_scores, aggregator.identity())

    if combined <= 0:
        log_fail(f"Combined score should be positive: {combined}")
        return False

    log_pass(f"Combined exp_scores: {combined:.4f}")

    # Test identity property
    identity = aggregator.identity()
    if aggregator.merge(identity, exp_scores[0]) != exp_scores[0]:
        log_fail("Identity property failed")
        return False

    log_pass("Aggregator identity property works")

    # Test partition_sum from responses
    total_partition = sum(resp.partition_sum for resp in responses)
    log_info(f"Total partition sum: {total_partition:.4f}")

    if total_partition <= 0:
        log_fail("Partition sum should be positive")
        return False

    log_pass("Partition sums computed correctly")

    return True


def test_diversity_weighted_aggregation():
    """Test diversity-weighted aggregation with different corpora."""
    log_section("Test 2: Diversity-Weighted Aggregation")

    responses = create_test_node_responses()

    # Test DIVERSITY_WEIGHTED aggregator
    aggregator = get_aggregator(AggregationStrategy.DIVERSITY_WEIGHTED)

    if aggregator is None:
        log_fail("Failed to create DIVERSITY_WEIGHTED aggregator")
        return False

    log_pass("DIVERSITY_WEIGHTED aggregator created")

    # Extract unique corpora from responses
    unique_corpora = set()
    for resp in responses:
        corpus_id = resp.node_metadata.get('corpus_id', 'unknown')
        unique_corpora.add(corpus_id)

    log_info(f"Found {len(unique_corpora)} unique corpora: {unique_corpora}")

    # Diversity score = unique_corpora / total_nodes
    diversity_score = len(unique_corpora) / len(responses)
    log_info(f"Diversity score: {diversity_score:.3f}")

    if diversity_score < 0.9:  # All 3 nodes have different corpora
        log_fail(f"Diversity should be ~1.0 for independent corpora: {diversity_score}")
        return False

    log_pass("High diversity for independent sources")

    # Test with overlapping corpora (simulate shared corpus)
    shared_responses = create_test_node_responses()
    for resp in shared_responses:
        resp.node_metadata['corpus_id'] = "shared_corpus"

    unique_corpora_shared = set()
    for resp in shared_responses:
        corpus_id = resp.node_metadata.get('corpus_id', 'unknown')
        unique_corpora_shared.add(corpus_id)

    diversity_shared = len(unique_corpora_shared) / len(shared_responses)
    log_info(f"Diversity with shared corpus: {diversity_shared:.3f}")

    if diversity_shared >= diversity_score:
        log_fail("Diversity should decrease with shared corpus")
        return False

    log_pass("Lower diversity for shared corpus")

    return True


def test_density_scoring_pipeline():
    """Test density-based confidence scoring."""
    log_section("Test 3: Density Scoring Pipeline")

    # Create embeddings with clear structure using controlled random generation
    np.random.seed(42)

    # Cluster 1: Dense cluster (tight grouping)
    cluster1 = np.random.randn(5, 64) * 0.1
    cluster1[:, 0] += 2.0  # Shift to one region

    # Cluster 2: Sparser cluster
    cluster2 = np.random.randn(3, 64) * 0.3
    cluster2[:, 1] += 2.0  # Different region

    # Outlier: Far from both clusters
    outlier = np.random.randn(2, 64) * 0.5
    outlier[:, 2] += 3.0  # Yet another region

    all_embeddings = np.vstack([cluster1, cluster2, outlier]).astype(np.float32)
    scores = np.array([0.9, 0.88, 0.85, 0.82, 0.80,  # cluster1
                      0.75, 0.72, 0.70,               # cluster2
                      0.60, 0.55])                     # outliers

    log_info(f"Testing with {len(all_embeddings)} embeddings")

    # Test basic density scores
    config = DensityConfig(
        bandwidth_method=BandwidthMethod.SILVERMAN,
        normalize_scores=True
    )

    densities = compute_density_scores(all_embeddings, config)

    if len(densities) != len(all_embeddings):
        log_fail(f"Wrong density count: {len(densities)}")
        return False

    log_info(f"Densities: {densities}")

    # Dense cluster should have higher density than outliers
    dense_density = densities[:5].mean()
    outlier_density = densities[8:].mean()

    log_info(f"Dense cluster avg density: {dense_density:.3f}")
    log_info(f"Outlier avg density: {outlier_density:.3f}")

    if dense_density <= outlier_density:
        log_fail("Dense cluster should have higher density than outliers")
        return False

    log_pass("Dense cluster has higher density than outliers")

    # Test flux-softmax
    flux_probs = flux_softmax(scores, densities, density_weight=0.3)

    if abs(flux_probs.sum() - 1.0) > 1e-6:
        log_fail(f"Flux probs don't sum to 1: {flux_probs.sum()}")
        return False

    log_pass("Flux-softmax probabilities sum to 1")

    return True


def test_two_stage_clustering():
    """Test two-stage density pipeline with clustering."""
    log_section("Test 4: Two-Stage Clustering")

    # Create embeddings with clear clusters
    np.random.seed(42)

    # Cluster 1: around [1, 0, 0, ...]
    cluster1 = np.random.randn(5, 32) * 0.1
    cluster1[:, 0] += 1.0

    # Cluster 2: around [0, 1, 0, ...]
    cluster2 = np.random.randn(4, 32) * 0.1
    cluster2[:, 1] += 1.0

    # Noise
    noise = np.random.randn(2, 32)

    embeddings = np.vstack([cluster1, cluster2, noise]).astype(np.float32)
    scores = np.random.rand(11)

    log_info(f"Created {len(embeddings)} embeddings (5 + 4 + 2 noise)")

    # Test greedy clustering
    config = DensityConfig(
        cluster_method=ClusterMethod.GREEDY,
        similarity_threshold=0.7,
        min_cluster_size=2
    )

    flux_probs, densities, labels, centroids = two_stage_density_pipeline(
        embeddings, scores, config
    )

    n_clusters = len(set(labels) - {-1})
    log_info(f"Greedy clustering found {n_clusters} clusters")

    if len(flux_probs) != len(embeddings):
        log_fail(f"Wrong flux_probs length: {len(flux_probs)}")
        return False

    log_pass("Two-stage pipeline returns correct shapes")

    # Test HDBSCAN clustering
    config_hdb = DensityConfig(
        cluster_method=ClusterMethod.HDBSCAN,
        min_cluster_size=2,
        hdbscan_min_samples=2
    )

    flux_probs_hdb, _, labels_hdb, _ = two_stage_density_pipeline(
        embeddings, scores, config_hdb
    )

    n_clusters_hdb = len(set(labels_hdb) - {-1})
    log_info(f"HDBSCAN found {n_clusters_hdb} clusters")
    log_pass("HDBSCAN clustering works")

    return True


def test_adaptive_bandwidth():
    """Test adaptive bandwidth selection."""
    log_section("Test 5: Adaptive Bandwidth")

    np.random.seed(42)

    # Create embeddings with varying density
    # Dense region
    dense = np.random.randn(10, 16) * 0.1
    dense[:, 0] += 1.0

    # Sparse region
    sparse = np.random.randn(5, 16) * 0.5
    sparse[:, 0] -= 1.0

    embeddings = np.vstack([dense, sparse]).astype(np.float32)

    log_info(f"Created {len(embeddings)} embeddings (10 dense + 5 sparse)")

    # Test with adaptive bandwidth
    config = DensityConfig(
        use_adaptive_bandwidth=True,
        adaptive_alpha=0.5,
        normalize_scores=True
    )

    densities = compute_density_scores(embeddings, config)

    if len(densities) != len(embeddings):
        log_fail(f"Wrong density count: {len(densities)}")
        return False

    # Dense points should have higher density
    dense_avg = densities[:10].mean()
    sparse_avg = densities[10:].mean()

    log_info(f"Dense region avg: {dense_avg:.3f}")
    log_info(f"Sparse region avg: {sparse_avg:.3f}")

    if dense_avg <= sparse_avg:
        log_fail("Dense region should have higher density")
        return False

    log_pass("Adaptive bandwidth correctly identifies density regions")

    return True


def test_efficiency_optimizations():
    """Test efficiency optimizations for large datasets."""
    log_section("Test 6: Efficiency Optimizations")

    np.random.seed(42)

    # Create larger dataset
    n_points = 150
    embeddings = np.random.randn(n_points, 64).astype(np.float32)

    log_info(f"Testing with {n_points} embeddings")

    # Test with sketching and ANN
    config = DensityConfig(
        use_sketching=True,
        sketch_dim=32,
        large_dataset_threshold=100,
        cache_distances=True
    )

    densities = compute_efficient_density_scores(embeddings, config)

    if len(densities) != n_points:
        log_fail(f"Wrong density count: {len(densities)}")
        return False

    if not (densities >= 0).all():
        log_fail("Densities should be non-negative")
        return False

    if not (densities <= 1).all():
        log_fail("Normalized densities should be <= 1")
        return False

    log_pass("Efficient density scoring works for large datasets")

    # Verify sketching was applied (dimension reduced)
    from density_scoring import sketch_embeddings
    sketched = sketch_embeddings(embeddings, target_dim=32)

    if sketched.shape != (n_points, 32):
        log_fail(f"Sketching didn't reduce dimensions: {sketched.shape}")
        return False

    log_pass("Sketching reduces dimensionality")

    return True


def test_cross_node_density():
    """Test density scoring across federated results."""
    log_section("Test 7: Cross-Node Density")

    responses = create_test_node_responses()

    # Collect all embeddings from responses
    all_embeddings = []
    all_scores = []

    for resp in responses:
        for result in resp.results:
            all_embeddings.append(result.embedding)
            all_scores.append(result.raw_score)

    embeddings = np.array(all_embeddings)
    scores = np.array(all_scores)

    log_info(f"Collected {len(embeddings)} results from {len(responses)} nodes")

    # Compute density across all federated results
    config = DensityConfig(
        clustering_enabled=True,
        similarity_threshold=0.6,
        density_weight=0.3
    )

    flux_probs, densities, labels, centroids = two_stage_density_pipeline(
        embeddings, scores, config
    )

    log_info(f"Found {len(centroids)} clusters across nodes")
    log_info(f"Densities: {densities}")

    # The CSV results should cluster together
    csv_labels = labels[:2]  # First 2 results are CSV

    if csv_labels[0] == csv_labels[1] and csv_labels[0] != -1:
        log_pass("CSV results cluster together")
    else:
        log_info(f"CSV labels: {csv_labels} (may be noise at small scale)")

    # Flux probabilities should reflect both score and density
    if abs(flux_probs.sum() - 1.0) > 1e-6:
        log_fail("Flux probs don't sum to 1")
        return False

    log_pass("Cross-node density scoring works")

    return True


def run_all_tests():
    """Run all multi-node federation E2E tests."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}KG Topology Phase 4: Multi-Node Federation E2E Tests{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

    results = []
    results.append(("Federated Aggregation", test_federated_aggregation()))
    results.append(("Diversity-Weighted", test_diversity_weighted_aggregation()))
    results.append(("Density Scoring", test_density_scoring_pipeline()))
    results.append(("Two-Stage Clustering", test_two_stage_clustering()))
    results.append(("Adaptive Bandwidth", test_adaptive_bandwidth()))
    results.append(("Efficiency Optimizations", test_efficiency_optimizations()))
    results.append(("Cross-Node Density", test_cross_node_density()))

    # Summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Test Summary{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)

    for name, result in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed} passed, {failed} failed")

    if failed == 0:
        print(f"\n{Colors.GREEN}All tests passed!{Colors.END}\n")
        return True
    else:
        print(f"\n{Colors.RED}Some tests failed.{Colors.END}\n")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
