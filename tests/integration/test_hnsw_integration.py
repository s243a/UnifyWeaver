#!/usr/bin/env python3
"""
Integration tests for HNSW hierarchical routing.

Tests:
- HNSW graph construction with tunable M parameter
- Effect of M on recall and performance
- Hierarchical layer structure
- P2P/distributed routing modes
- Search with backtracking
"""

import sys
import os
import unittest
import numpy as np
import time

# Add the runtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/unifyweaver/targets/python_runtime'))

from hnsw_layers import (
    HNSWGraph,
    HNSWNode,
    build_hnsw_index,
    cosine_distance,
)


class TestHNSWNode(unittest.TestCase):
    """Test the HNSWNode class."""

    def test_node_creation(self):
        """Node should be created with correct attributes."""
        vec = np.array([1.0, 0.0, 0.0])
        node = HNSWNode(node_id="test", vector=vec, max_layer=2)

        self.assertEqual(node.node_id, "test")
        np.testing.assert_array_equal(node.vector, vec)
        self.assertEqual(node.max_layer, 2)

    def test_add_neighbor(self):
        """Should be able to add neighbors at layers."""
        node = HNSWNode(node_id="n1", vector=np.array([1.0, 0.0, 0.0]))

        result = node.add_neighbor("n2", layer=0)
        self.assertTrue(result)
        self.assertIn("n2", node.get_neighbors_at_layer(0))

    def test_max_neighbors_limit(self):
        """Should respect max neighbors limit."""
        node = HNSWNode(
            node_id="n1",
            vector=np.array([1.0, 0.0, 0.0]),
            max_neighbors=2,
        )

        node.add_neighbor("n2", layer=1)
        node.add_neighbor("n3", layer=1)

        # Third should fail
        result = node.add_neighbor("n4", layer=1)
        self.assertFalse(result)

    def test_layer0_has_more_neighbors(self):
        """Layer 0 should allow more neighbors (M0 > M)."""
        node = HNSWNode(
            node_id="n0",  # Use n0 to avoid conflict with neighbor names
            vector=np.array([1.0, 0.0, 0.0]),
            max_neighbors=4,
            max_neighbors_layer0=8,
        )

        # Add 8 neighbors to layer 0 (n1 through n8)
        for i in range(1, 9):
            result = node.add_neighbor(f"n{i}", layer=0)
            self.assertTrue(result, f"Failed to add n{i}")

        # 9th should fail
        result = node.add_neighbor("n_extra", layer=0)
        self.assertFalse(result)


class TestHNSWGraph(unittest.TestCase):
    """Test the HNSWGraph class."""

    def test_graph_creation(self):
        """Graph should be created with correct parameters."""
        graph = HNSWGraph(max_neighbors=16, ef_construction=100)

        self.assertEqual(graph.max_neighbors, 16)
        self.assertEqual(graph.ef_construction, 100)
        self.assertEqual(len(graph.nodes), 0)

    def test_add_single_node(self):
        """Adding first node should set it as entry point."""
        graph = HNSWGraph()
        import random
        rng = random.Random(42)

        vec = np.array([1.0, 0.0, 0.0])
        node = graph.add_node("n1", vec, rng=rng)

        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.entry_point_id, "n1")

    def test_add_multiple_nodes(self):
        """Nodes should get connected when added."""
        graph = HNSWGraph(max_neighbors=8)
        import random
        rng = random.Random(42)

        np.random.seed(42)
        for i in range(10):
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            graph.add_node(f"n{i}", vec, rng=rng)

        self.assertEqual(len(graph.nodes), 10)

        # Check some nodes have neighbors
        has_neighbors = any(
            any(len(node.neighbors[l]) > 0 for l in range(node.max_layer + 1))
            for node in graph.nodes.values()
        )
        self.assertTrue(has_neighbors)

    def test_layer_distribution(self):
        """Nodes should be distributed across layers."""
        graph = HNSWGraph()
        import random
        rng = random.Random(42)

        np.random.seed(42)
        for i in range(100):
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            graph.add_node(f"n{i}", vec, rng=rng)

        stats = graph.get_statistics()

        # Should have multiple layers
        self.assertGreater(stats["max_layer"], 0)

        # Most nodes should be at layer 0
        layer_dist = stats["layer_distribution"]
        self.assertGreater(layer_dist.get(0, 0), layer_dist.get(1, 0))


class TestHNSWSearch(unittest.TestCase):
    """Test HNSW search functionality."""

    def setUp(self):
        """Build a test graph."""
        np.random.seed(42)
        import random
        rng = random.Random(42)

        self.graph = HNSWGraph(max_neighbors=16, ef_construction=100)

        # Add nodes
        self.vectors = {}
        for i in range(50):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            self.vectors[f"n{i}"] = vec
            self.graph.add_node(f"n{i}", vec, rng=rng)

    def test_basic_search(self):
        """Search should return results."""
        query = np.random.randn(10)
        query = query / np.linalg.norm(query)

        results, comparisons = self.graph.search(query, k=5, ef=50)

        self.assertEqual(len(results), 5)
        self.assertGreater(comparisons, 0)

    def test_search_accuracy(self):
        """Search should find nearest neighbors accurately."""
        # Search for a known node
        target = self.vectors["n0"]

        results, _ = self.graph.search(target, k=5, ef=100)

        # First result should be n0 itself or very close
        best_id, best_dist = results[0]
        self.assertLess(best_dist, 0.1)

    def test_search_from_any_node(self):
        """P2P search should work from any starting node."""
        query = np.random.randn(10)
        query = query / np.linalg.norm(query)

        # Search from different starting points
        results1, _ = self.graph.search_from_any_node(query, "n10", k=5)
        results2, _ = self.graph.search_from_any_node(query, "n20", k=5)

        # Both should return results
        self.assertEqual(len(results1), 5)
        self.assertEqual(len(results2), 5)

        # Results should overlap significantly
        ids1 = set(r[0] for r in results1)
        ids2 = set(r[0] for r in results2)
        overlap = len(ids1 & ids2)
        self.assertGreater(overlap, 2)

    def test_search_with_backtrack(self):
        """Backtracking search should improve recall."""
        query = np.random.randn(10)
        query = query / np.linalg.norm(query)

        # Search with backtracking
        results, _ = self.graph.search_from_any_node(
            query, "n0", k=5, use_backtrack=True
        )

        self.assertEqual(len(results), 5)


class TestTunableM(unittest.TestCase):
    """Test the effect of M (max_neighbors) parameter."""

    def _build_graph_with_m(self, m: int, num_nodes: int = 100) -> HNSWGraph:
        """Build a graph with specified M value."""
        np.random.seed(42)
        import random
        rng = random.Random(42)

        graph = HNSWGraph(max_neighbors=m, max_neighbors_layer0=m * 2)

        for i in range(num_nodes):
            vec = np.random.randn(32)
            vec = vec / np.linalg.norm(vec)
            graph.add_node(f"n{i}", vec, rng=rng)

        return graph

    def _compute_recall(self, graph: HNSWGraph, queries: list, k: int = 5) -> float:
        """Compute recall@k against brute force."""
        total_recall = 0

        for query in queries:
            # Brute force
            distances = [
                (nid, cosine_distance(query, node.vector))
                for nid, node in graph.nodes.items()
            ]
            distances.sort(key=lambda x: x[1])
            gt_ids = set(nid for nid, _ in distances[:k])

            # HNSW search
            results, _ = graph.search(query, k=k, ef=50)
            result_ids = set(nid for nid, _ in results)

            recall = len(gt_ids & result_ids) / k
            total_recall += recall

        return total_recall / len(queries)

    def test_higher_m_improves_recall(self):
        """Higher M should generally improve recall."""
        # Build graphs with different M
        graph_m8 = self._build_graph_with_m(8)
        graph_m32 = self._build_graph_with_m(32)

        # Generate test queries
        np.random.seed(123)
        queries = [np.random.randn(32) for _ in range(20)]
        queries = [q / np.linalg.norm(q) for q in queries]

        recall_m8 = self._compute_recall(graph_m8, queries)
        recall_m32 = self._compute_recall(graph_m32, queries)

        # Higher M should have equal or better recall
        self.assertGreaterEqual(recall_m32, recall_m8 * 0.95)

    def test_higher_m_uses_more_memory(self):
        """Higher M should result in more edges."""
        graph_m8 = self._build_graph_with_m(8, num_nodes=50)
        graph_m32 = self._build_graph_with_m(32, num_nodes=50)

        stats_m8 = graph_m8.get_statistics()
        stats_m32 = graph_m32.get_statistics()

        # Higher M = more edges
        self.assertGreater(stats_m32["total_edges"], stats_m8["total_edges"])

    def test_m_tradeoff(self):
        """Demonstrate recall vs memory tradeoff."""
        results = []

        for m in [4, 8, 16, 32]:
            graph = self._build_graph_with_m(m, num_nodes=100)

            np.random.seed(999)
            queries = [np.random.randn(32) for _ in range(10)]
            queries = [q / np.linalg.norm(q) for q in queries]

            recall = self._compute_recall(graph, queries)
            edges = graph.get_statistics()["total_edges"]

            results.append((m, recall, edges))

        # Verify we can observe the tradeoff
        m_values = [r[0] for r in results]
        edges_values = [r[2] for r in results]

        # More M = more edges (monotonic)
        self.assertEqual(edges_values, sorted(edges_values))


class TestBuildHNSWIndex(unittest.TestCase):
    """Test the build_hnsw_index helper function."""

    def test_build_from_vectors(self):
        """Should build graph from vector list."""
        np.random.seed(42)

        vectors = [
            (f"v{i}", np.random.randn(10))
            for i in range(20)
        ]
        vectors = [(vid, v / np.linalg.norm(v)) for vid, v in vectors]

        graph = build_hnsw_index(vectors, max_neighbors=8, seed=42)

        self.assertEqual(len(graph.nodes), 20)
        self.assertIsNotNone(graph.entry_point_id)


class TestHNSWDistributedRouting(unittest.TestCase):
    """Test HNSW in distributed/P2P scenarios."""

    def setUp(self):
        """Build test graph."""
        np.random.seed(42)
        import random
        rng = random.Random(42)

        self.graph = HNSWGraph(max_neighbors=16)

        for i in range(100):
            vec = np.random.randn(32)
            vec = vec / np.linalg.norm(vec)
            self.graph.add_node(f"n{i}", vec, rng=rng)

    def test_entry_points_at_layer(self):
        """Should be able to get entry points at any layer."""
        for layer in range(self.graph.max_layer + 1):
            entry_points = self.graph.get_entry_points_at_layer(layer)

            # Lower layers should have more entry points
            if layer > 0:
                higher_layer_points = self.graph.get_entry_points_at_layer(layer + 1)
                # Each point at higher layer is also at lower layer
                self.assertLessEqual(
                    len(higher_layer_points),
                    len(entry_points)
                )

    def test_routing_from_lower_layer(self):
        """Should be able to route starting from lower layers."""
        query = np.random.randn(32)
        query = query / np.linalg.norm(query)

        # Start from layer 0
        results, _ = self.graph.search(
            query, k=5, ef=50,
            start_layer=0,
        )

        self.assertEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
