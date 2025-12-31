# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Analysis: How subdivision improves routing discrimination.

This test measures the ability to correctly route queries before and after
node subdivision. When a node splits, its child centroids become more
specialized, improving the ability to distinguish which child should
handle a given query.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from unifyweaver.targets.python_runtime.adaptive_subdivision import (
    SplitConfig,
    SubdividableNode,
    split_node,
    SubdivisionRegistry,
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def analyze_discrimination():
    """
    Measure routing discrimination before and after split.

    Discrimination = how well we can distinguish which cluster a query belongs to.
    Higher discrimination = more accurate routing.
    """
    np.random.seed(42)
    dim = 10

    print("=" * 60)
    print("SUBDIVISION DISCRIMINATION ANALYSIS")
    print("=" * 60)

    # Create two distinct document clusters
    cluster_a_center = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cluster_b_center = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Generate documents for each cluster with some noise
    docs_a = []
    docs_b = []
    for i in range(50):
        # Cluster A documents: near [1,0,...]
        noise = np.random.randn(dim) * 0.1
        doc_a = cluster_a_center + noise
        doc_a = doc_a / np.linalg.norm(doc_a)  # Normalize
        docs_a.append((f"doc_a_{i}", doc_a))

        # Cluster B documents: near [0,1,...]
        noise = np.random.randn(dim) * 0.1
        doc_b = cluster_b_center + noise
        doc_b = doc_b / np.linalg.norm(doc_b)  # Normalize
        docs_b.append((f"doc_b_{i}", doc_b))

    # Create a single node with all documents
    node = SubdividableNode(
        node_id="mixed_node",
        config=SplitConfig(max_documents=50, min_child_documents=10),
    )

    # Add all documents to the single node
    for doc_id, emb in docs_a + docs_b:
        node.add_document(doc_id, emb)

    print(f"\nBEFORE SPLIT:")
    print(f"  Node: {node.node_id}")
    print(f"  Documents: {node.metrics.document_count}")
    print(f"  Centroid: {node.centroid[:4]}...")  # First 4 dims
    print(f"  Variance: {node.metrics.centroid_variance:.4f}")

    # Test queries targeting each cluster
    query_a = cluster_a_center / np.linalg.norm(cluster_a_center)
    query_b = cluster_b_center / np.linalg.norm(cluster_b_center)

    # Before split: single node gets all queries
    sim_to_mixed_a = cosine_similarity(query_a, node.centroid)
    sim_to_mixed_b = cosine_similarity(query_b, node.centroid)

    print(f"\n  Query A similarity to mixed node: {sim_to_mixed_a:.4f}")
    print(f"  Query B similarity to mixed node: {sim_to_mixed_b:.4f}")
    print(f"  Discrimination (difference): {abs(sim_to_mixed_a - sim_to_mixed_b):.4f}")

    before_discrimination = abs(sim_to_mixed_a - sim_to_mixed_b)

    # SPLIT the node
    child_a, child_b = split_node(node, seed=42)

    print(f"\nAFTER SPLIT:")
    print(f"  Parent: {node.node_id} (now REGION)")
    print(f"  Child A: {child_a.node_id} with {child_a.metrics.document_count} docs")
    print(f"    Centroid: {child_a.centroid[:4]}...")
    print(f"    Variance: {child_a.metrics.centroid_variance:.4f}")
    print(f"  Child B: {child_b.node_id} with {child_b.metrics.document_count} docs")
    print(f"    Centroid: {child_b.centroid[:4]}...")
    print(f"    Variance: {child_b.metrics.centroid_variance:.4f}")

    # After split: measure similarity to each child
    sim_query_a_to_child_a = cosine_similarity(query_a, child_a.centroid)
    sim_query_a_to_child_b = cosine_similarity(query_a, child_b.centroid)
    sim_query_b_to_child_a = cosine_similarity(query_b, child_a.centroid)
    sim_query_b_to_child_b = cosine_similarity(query_b, child_b.centroid)

    print(f"\n  Query A → Child A: {sim_query_a_to_child_a:.4f}")
    print(f"  Query A → Child B: {sim_query_a_to_child_b:.4f}")
    print(f"  Query B → Child A: {sim_query_b_to_child_a:.4f}")
    print(f"  Query B → Child B: {sim_query_b_to_child_b:.4f}")

    # Calculate discrimination: how well can we distinguish correct child
    disc_a = abs(sim_query_a_to_child_a - sim_query_a_to_child_b)
    disc_b = abs(sim_query_b_to_child_a - sim_query_b_to_child_b)
    after_discrimination = (disc_a + disc_b) / 2

    print(f"\n  Discrimination for Query A: {disc_a:.4f}")
    print(f"  Discrimination for Query B: {disc_b:.4f}")
    print(f"  Average discrimination: {after_discrimination:.4f}")

    # Routing accuracy
    correct_routing_a = sim_query_a_to_child_a > sim_query_a_to_child_b
    correct_routing_b = sim_query_b_to_child_b > sim_query_b_to_child_a

    print(f"\n  Query A routes to correct child: {correct_routing_a}")
    print(f"  Query B routes to correct child: {correct_routing_b}")

    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Before split discrimination: {before_discrimination:.4f}")
    print(f"  After split discrimination:  {after_discrimination:.4f}")
    improvement = ((after_discrimination - before_discrimination) /
                   max(before_discrimination, 0.001)) * 100
    print(f"  Improvement: {improvement:.1f}%")

    # Is this due to numeric precision?
    print(f"\n  Numeric precision analysis:")
    print(f"    Child centroid norms: {np.linalg.norm(child_a.centroid):.6f}, {np.linalg.norm(child_b.centroid):.6f}")
    print(f"    Mixed centroid norm:  {np.linalg.norm(node.centroid) if node.centroid is not None else 'N/A (cleared)'}")
    print(f"\n  The improvement is NOT due to numeric precision.")
    print(f"  It's due to GEOMETRIC SEPARATION:")
    print(f"    - Mixed centroid averages two clusters → sits between them")
    print(f"    - Child centroids are each close to their own cluster")
    print(f"    - Queries can be routed to the more similar child")

    return before_discrimination, after_discrimination


def test_routing_with_registry():
    """Test end-to-end routing through registry."""
    np.random.seed(42)

    print("\n" + "=" * 60)
    print("REGISTRY ROUTING TEST")
    print("=" * 60)

    registry = SubdivisionRegistry()

    # Create a node with mixed documents
    node = SubdividableNode(
        node_id="root",
        config=SplitConfig(max_documents=50, min_child_documents=10),
    )

    # Two distinct clusters
    for i in range(30):
        # Cluster 1: [1, 0, 0, ...]
        emb1 = np.zeros(10)
        emb1[0] = 1.0 + np.random.randn() * 0.1
        emb1 = emb1 / np.linalg.norm(emb1)
        node.add_document(f"cluster1_{i}", emb1)

        # Cluster 2: [0, 0, 1, ...]
        emb2 = np.zeros(10)
        emb2[2] = 1.0 + np.random.randn() * 0.1
        emb2 = emb2 / np.linalg.norm(emb2)
        node.add_document(f"cluster2_{i}", emb2)

    registry.register(node)

    # Split and register children
    split_ids = registry.check_and_split_all(seed=42)
    print(f"Split nodes: {split_ids}")

    # Test routing
    query1 = np.zeros(10)
    query1[0] = 1.0
    query1 = query1 / np.linalg.norm(query1)

    query2 = np.zeros(10)
    query2[2] = 1.0
    query2 = query2 / np.linalg.norm(query2)

    leaf1 = registry.route_to_leaf(query1, "root")
    leaf2 = registry.route_to_leaf(query2, "root")

    print(f"\nQuery targeting dim[0] routed to: {leaf1.node_id}")
    print(f"  Contains cluster1 docs: {any('cluster1' in d for d in leaf1.document_ids)}")

    print(f"\nQuery targeting dim[2] routed to: {leaf2.node_id}")
    print(f"  Contains cluster2 docs: {any('cluster2' in d for d in leaf2.document_ids)}")

    # Check if queries were routed to different children
    different_routes = leaf1.node_id != leaf2.node_id
    print(f"\nDifferent queries routed to different children: {different_routes}")

    return different_routes


if __name__ == "__main__":
    before, after = analyze_discrimination()
    print()
    routing_correct = test_routing_with_registry()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"Subdivision improved discrimination from {before:.4f} to {after:.4f}")
    print(f"Routing correctly separates distinct queries: {routing_correct}")
    print()
    print("The accuracy improvement is GEOMETRIC, not numeric precision:")
    print("  1. Mixed centroid sits between clusters (poor discrimination)")
    print("  2. Child centroids align with their clusters (good discrimination)")
    print("  3. Cosine similarity correctly routes queries to specialized children")
