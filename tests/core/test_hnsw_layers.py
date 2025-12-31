# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Tests for HNSW-style layered graph.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

from unifyweaver.targets.python_runtime.hnsw_layers import (
    HNSWNode,
    HNSWGraph,
    build_hnsw_index,
    cosine_distance,
)


class TestHNSWNode(unittest.TestCase):
    """Tests for HNSWNode."""

    def test_create_node(self):
        """Test node creation."""
        node = HNSWNode(
            node_id="test",
            vector=np.array([1.0, 0.0]),
            max_layer=2,
        )
        self.assertEqual(node.node_id, "test")
        self.assertEqual(node.max_layer, 2)

    def test_add_neighbor(self):
        """Test adding neighbors at different layers."""
        node = HNSWNode(
            node_id="test",
            vector=np.array([1.0, 0.0]),
            max_layer=2,
            max_neighbors=3,
        )

        # Add neighbors at layer 0
        self.assertTrue(node.add_neighbor("n1", layer=0))
        self.assertTrue(node.add_neighbor("n2", layer=0))
        self.assertTrue(node.add_neighbor("n3", layer=0))

        # Layer 0 can have more (max_neighbors_layer0)
        self.assertTrue(node.add_neighbor("n4", layer=0))

        # Layer 1 has stricter limit
        self.assertTrue(node.add_neighbor("n1", layer=1))
        self.assertTrue(node.add_neighbor("n2", layer=1))
        self.assertTrue(node.add_neighbor("n3", layer=1))
        self.assertFalse(node.add_neighbor("n4", layer=1))  # Exceeds max

    def test_get_neighbors_at_layer(self):
        """Test getting neighbors at specific layer."""
        node = HNSWNode(node_id="test", vector=np.array([1.0, 0.0]))
        node.add_neighbor("n1", layer=0)
        node.add_neighbor("n2", layer=0)
        node.add_neighbor("n3", layer=1)

        self.assertEqual(len(node.get_neighbors_at_layer(0)), 2)
        self.assertEqual(len(node.get_neighbors_at_layer(1)), 1)
        self.assertEqual(len(node.get_neighbors_at_layer(2)), 0)


class TestHNSWGraph(unittest.TestCase):
    """Tests for HNSWGraph."""

    def test_create_graph(self):
        """Test graph creation."""
        graph = HNSWGraph()
        self.assertEqual(len(graph.nodes), 0)
        self.assertIsNone(graph.entry_point_id)

    def test_add_single_node(self):
        """Test adding a single node."""
        graph = HNSWGraph()
        node = graph.add_node("n1", np.array([1.0, 0.0]))

        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.entry_point_id, "n1")

    def test_add_multiple_nodes(self):
        """Test adding multiple nodes creates connections."""
        np.random.seed(42)
        graph = HNSWGraph(max_neighbors=4)

        # Add nodes in a cluster
        for i in range(10):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            graph.add_node(f"n{i}", vec)

        self.assertEqual(len(graph.nodes), 10)

        # Check that nodes have connections
        total_edges = sum(
            len(node.neighbors[0]) for node in graph.nodes.values()
        )
        self.assertGreater(total_edges, 0)

    def test_layer_distribution(self):
        """Test that layers follow exponential distribution."""
        np.random.seed(42)
        graph = HNSWGraph()

        # Add many nodes
        for i in range(100):
            vec = np.random.randn(10)
            graph.add_node(f"n{i}", vec)

        stats = graph.get_statistics()
        layer_dist = stats["layer_distribution"]

        print(f"\nLayer distribution: {layer_dist}")

        # Layer 0 should have all nodes
        self.assertEqual(layer_dist.get(0, 0), 100)

        # Higher layers should have fewer nodes (exponential decay)
        if 1 in layer_dist and 2 in layer_dist:
            self.assertGreater(layer_dist[1], layer_dist[2])

    def test_search_finds_nearest(self):
        """Test that search finds the nearest neighbor."""
        np.random.seed(42)
        graph = HNSWGraph()

        # Add nodes
        vectors = []
        for i in range(50):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            vectors.append((f"n{i}", vec))
            graph.add_node(f"n{i}", vec)

        # Query for a specific vector
        query = vectors[25][1]  # Should find n25

        results, _ = graph.search(query, k=5)

        # n25 should be in top results (distance ~0)
        result_ids = [r[0] for r in results]
        self.assertIn("n25", result_ids[:3])

    def test_search_from_any_node(self):
        """Test P2P-style search from any node."""
        np.random.seed(42)
        graph = HNSWGraph()

        # Add nodes
        for i in range(50):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            graph.add_node(f"n{i}", vec)

        # Query
        target = graph.nodes["n30"]
        query = target.vector.copy()

        # Search from different start nodes
        success_count = 0
        for start_id in ["n0", "n10", "n20", "n40", "n49"]:
            results, _ = graph.search_from_any_node(
                query, start_id, k=5, use_backtrack=True
            )
            result_ids = [r[0] for r in results]
            if "n30" in result_ids:
                success_count += 1

        print(f"\nP2P search success: {success_count}/5")
        self.assertGreaterEqual(success_count, 4)

    def test_entry_points_at_layer(self):
        """Test getting entry points at different layers."""
        np.random.seed(42)
        graph = HNSWGraph()

        for i in range(100):
            vec = np.random.randn(10)
            graph.add_node(f"n{i}", vec)

        # Layer 0 should have all nodes
        layer0_entries = graph.get_entry_points_at_layer(0)
        self.assertEqual(len(layer0_entries), 100)

        # Higher layers should have fewer
        layer1_entries = graph.get_entry_points_at_layer(1)
        layer2_entries = graph.get_entry_points_at_layer(2)

        print(f"\nEntry points: L0={len(layer0_entries)}, L1={len(layer1_entries)}, L2={len(layer2_entries)}")

        self.assertLess(len(layer1_entries), len(layer0_entries))


class TestHNSWScaling(unittest.TestCase):
    """Test O(log n) scaling of HNSW."""

    def test_comparisons_scale_logarithmically(self):
        """Test that comparisons grow logarithmically with n."""
        results = []

        for num_nodes in [50, 100, 200, 500, 1000]:
            np.random.seed(42)
            graph = HNSWGraph(max_neighbors=16, ef_construction=50)

            # Build index
            for i in range(num_nodes):
                vec = np.random.randn(32)
                vec = vec / np.linalg.norm(vec)
                graph.add_node(f"n{i}", vec)

            # Run queries
            total_comps = 0
            num_queries = 20

            for q in range(num_queries):
                query = np.random.randn(32)
                query = query / np.linalg.norm(query)
                _, comps = graph.search(query, k=10, ef=50)
                total_comps += comps

            avg_comps = total_comps / num_queries
            log_n = np.log2(num_nodes)

            results.append({
                "n": num_nodes,
                "avg_comps": avg_comps,
                "log_n": log_n,
                "ratio": avg_comps / log_n,
            })

        print("\n" + "=" * 60)
        print("HNSW SCALING TEST")
        print("=" * 60)
        print(f"{'Nodes':>8} {'Comparisons':>12} {'log₂(n)':>10} {'Ratio':>10}")
        print("-" * 42)

        for r in results:
            print(f"{r['n']:>8} {r['avg_comps']:>12.1f} {r['log_n']:>10.1f} {r['ratio']:>10.1f}")

        # Check that ratio stays roughly constant (O(log n) scaling)
        ratios = [r["ratio"] for r in results]
        ratio_variance = np.var(ratios) / np.mean(ratios) ** 2

        print(f"\nRatio variance (normalized): {ratio_variance:.3f}")
        print("(Lower variance = better O(log n) scaling)")

        # For O(log n), ratio should be relatively stable
        # Allow some variance due to graph structure
        self.assertLess(ratio_variance, 1.0)


class TestDistributedEntry(unittest.TestCase):
    """Test distributed/P2P entry point scenarios."""

    def test_multiple_entry_points(self):
        """Test using different entry points for distributed queries."""
        np.random.seed(42)
        graph = HNSWGraph()

        # Build graph
        for i in range(100):
            vec = np.random.randn(16)
            vec = vec / np.linalg.norm(vec)
            graph.add_node(f"n{i}", vec)

        # Get entry points at layer 1 (distributed scenario)
        layer1_entries = graph.get_entry_points_at_layer(1)

        print(f"\nLayer 1 entry points: {len(layer1_entries)}")
        print(f"These could be distributed across {len(layer1_entries)} peers")

        # Query from each entry point
        query = np.random.randn(16)
        query = query / np.linalg.norm(query)

        results_from_entries = []
        for entry_id in layer1_entries[:5]:  # Test first 5
            results, comps = graph.search(
                query, k=5, ef=30,
                start_layer=1,
                start_node_id=entry_id,
            )
            results_from_entries.append((entry_id, results, comps))

        # All should find similar results
        all_result_sets = [set(r[0] for r in res[1]) for res in results_from_entries]

        # Check overlap
        if len(all_result_sets) >= 2:
            overlap = all_result_sets[0].intersection(all_result_sets[1])
            print(f"Result overlap between entries: {len(overlap)}/5")
            self.assertGreater(len(overlap), 0)


def run_scaling_demo():
    """Demonstrate HNSW scaling."""
    print("=" * 70)
    print("HNSW SCALING DEMONSTRATION")
    print("=" * 70)

    np.random.seed(42)

    for num_nodes in [100, 500, 1000, 2000]:
        graph = HNSWGraph(max_neighbors=16, ef_construction=100)

        # Build
        for i in range(num_nodes):
            vec = np.random.randn(64)
            vec = vec / np.linalg.norm(vec)
            graph.add_node(f"n{i}", vec)

        stats = graph.get_statistics()

        # Query
        total_comps = 0
        for _ in range(50):
            query = np.random.randn(64)
            query = query / np.linalg.norm(query)
            _, comps = graph.search(query, k=10, ef=50)
            total_comps += comps

        avg_comps = total_comps / 50

        print(f"\nn={num_nodes}:")
        print(f"  Layers: {stats['max_layer'] + 1}")
        print(f"  Layer distribution: {stats['layer_distribution']}")
        print(f"  Avg comparisons: {avg_comps:.1f}")
        print(f"  log₂(n): {np.log2(num_nodes):.1f}")
        print(f"  Ratio: {avg_comps / np.log2(num_nodes):.1f}x log(n)")


if __name__ == "__main__":
    run_scaling_demo()
    print("\n\nRunning unit tests...")
    unittest.main(verbosity=2)
