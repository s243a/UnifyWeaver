#!/usr/bin/env python3
"""
Integration tests for Phase 7-8 small-world and multi-interface implementations.

Tests the Python runtime's proper small-world network with:
- AngleOrdering enum (COSINE_BASED vs PROJECTION_2D)
- k_local and k_long connection structure
- Cosine-based angle ordering for binary search
- Greedy routing on the small-world topology
"""

import sys
import os
import unittest
import numpy as np

# Add the runtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/unifyweaver/targets/python_runtime'))

from small_world_proper import (
    SmallWorldProper,
    AngleOrdering,
    SWNode,
    build_small_world,
    cosine_similarity,
    cosine_distance,
    compute_cosine_angle,
    compute_angle_2d,
)


class TestAngleOrdering(unittest.TestCase):
    """Test the AngleOrdering enum."""

    def test_enum_values(self):
        """Verify enum has expected values."""
        self.assertEqual(AngleOrdering.COSINE_BASED.value, "cosine_based")
        self.assertEqual(AngleOrdering.PROJECTION_2D.value, "projection_2d")

    def test_default_ordering(self):
        """SmallWorldProper should default to COSINE_BASED."""
        network = SmallWorldProper()
        self.assertEqual(network.angle_ordering, AngleOrdering.COSINE_BASED)


class TestCosineFunctions(unittest.TestCase):
    """Test cosine similarity and angle functions."""

    def test_cosine_similarity_identical(self):
        """Identical vectors should have similarity 1.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity 0.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=5)

    def test_cosine_similarity_opposite(self):
        """Opposite vectors should have similarity -1.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0, places=5)

    def test_cosine_distance(self):
        """Cosine distance should be 1 - similarity."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.9, 0.1, 0.0])
        sim = cosine_similarity(a, b)
        dist = cosine_distance(a, b)
        self.assertAlmostEqual(sim + dist, 1.0, places=5)

    def test_cosine_angle_identical(self):
        """Identical vectors should have angle 0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        angle = compute_cosine_angle(a, b)
        self.assertAlmostEqual(angle, 0.0, places=2)

    def test_cosine_angle_orthogonal(self):
        """Orthogonal vectors should have angle pi/2."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        angle = compute_cosine_angle(a, b)
        self.assertAlmostEqual(angle, np.pi / 2, places=2)


class TestSWNode(unittest.TestCase):
    """Test the SWNode class."""

    def test_node_creation(self):
        """Node should be created with correct attributes."""
        vec = np.array([1.0, 0.0, 0.0])
        node = SWNode(node_id="test-node", vector=vec)
        self.assertEqual(node.node_id, "test-node")
        np.testing.assert_array_equal(node.vector, vec)
        self.assertEqual(len(node.neighbors), 0)

    def test_add_neighbor(self):
        """Should be able to add neighbors."""
        node = SWNode(node_id="n1", vector=np.array([1.0, 0.0, 0.0]))
        neighbor_vec = np.array([0.9, 0.1, 0.0])

        result = node.add_neighbor("n2", neighbor_vec)
        self.assertTrue(result)
        self.assertIn("n2", node.neighbors)
        self.assertEqual(len(node.sorted_neighbors), 1)

    def test_add_self_neighbor_fails(self):
        """Should not be able to add self as neighbor."""
        node = SWNode(node_id="n1", vector=np.array([1.0, 0.0, 0.0]))
        result = node.add_neighbor("n1", np.array([1.0, 0.0, 0.0]))
        self.assertFalse(result)

    def test_max_neighbors(self):
        """Should respect max_neighbors limit."""
        node = SWNode(node_id="n1", vector=np.array([1.0, 0.0, 0.0]), max_neighbors=2)

        node.add_neighbor("n2", np.array([0.9, 0.1, 0.0]))
        node.add_neighbor("n3", np.array([0.8, 0.2, 0.0]))

        # Third should fail
        result = node.add_neighbor("n4", np.array([0.7, 0.3, 0.0]))
        self.assertFalse(result)
        self.assertEqual(len(node.neighbors), 2)

    def test_lookup_neighbors_by_angle(self):
        """Should find neighbors near query angle."""
        node = SWNode(node_id="n1", vector=np.array([1.0, 0.0, 0.0]))

        # Add several neighbors with different angles
        for i in range(10):
            angle = i * 0.1
            vec = np.array([np.cos(angle), np.sin(angle), 0.0])
            node.add_neighbor(f"n{i+2}", vec)

        # Query for neighbors near specific angle
        query = np.array([0.9, 0.1, 0.0])
        candidates = node.lookup_neighbors_by_angle(query, window_size=3)

        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), 7)  # window_size * 2 + 1


class TestSmallWorldProper(unittest.TestCase):
    """Test the SmallWorldProper network class."""

    def test_network_creation(self):
        """Network should be created with correct parameters."""
        network = SmallWorldProper(k_local=8, k_long=4, alpha=1.5)
        self.assertEqual(network.k_local, 8)
        self.assertEqual(network.k_long, 4)
        self.assertAlmostEqual(network.alpha, 1.5, places=2)

    def test_add_single_node(self):
        """Single node should be added without connections."""
        network = SmallWorldProper()
        node = network.add_node("n1", np.array([1.0, 0.0, 0.0]))
        self.assertIsNotNone(node)
        self.assertEqual(len(network.nodes), 1)

    def test_add_multiple_nodes(self):
        """Multiple nodes should get connected."""
        network = SmallWorldProper(k_local=3, k_long=1)

        for i in range(10):
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        self.assertEqual(len(network.nodes), 10)

        # Each node should have some neighbors
        for node in network.nodes.values():
            if len(network.nodes) > 1:
                self.assertGreater(len(node.neighbors), 0)

    def test_greedy_routing(self):
        """Greedy routing should find path to target."""
        np.random.seed(42)
        network = SmallWorldProper(k_local=5, k_long=2)

        # Add nodes in a cluster
        for i in range(20):
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        # Route to a target
        target = np.array([1.0, 0.0, 0.0])
        path, comparisons = network.route_greedy(target)

        self.assertGreater(len(path), 0)
        self.assertGreater(comparisons, 0)

    def test_knn_search(self):
        """KNN search should find nearest neighbors."""
        np.random.seed(42)
        network = SmallWorldProper(k_local=5, k_long=2)

        # Add nodes
        for i in range(20):
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        # Search for k nearest
        query = np.array([1.0, 0.0, 0.0])
        results, comparisons = network.search_knn(query, k=5)

        self.assertEqual(len(results), 5)
        self.assertGreater(comparisons, 0)

        # Results should be sorted by distance
        distances = [dist for _, dist in results]
        self.assertEqual(distances, sorted(distances))

    def test_optimized_knn_search(self):
        """Optimized KNN search should also work."""
        np.random.seed(42)
        network = SmallWorldProper(k_local=5, k_long=2)

        for i in range(20):
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        query = np.array([1.0, 0.0, 0.0])
        results, comparisons = network.search_knn_optimized(query, k=5, window_size=3)

        self.assertEqual(len(results), 5)
        # Optimized should use fewer comparisons (not always, but trend)

    def test_network_statistics(self):
        """Should compute meaningful statistics."""
        np.random.seed(42)
        network = SmallWorldProper(k_local=5, k_long=2)

        for i in range(20):
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        stats = network.get_statistics()

        self.assertEqual(stats["num_nodes"], 20)
        self.assertGreater(stats["total_edges"], 0)
        self.assertGreater(stats["avg_neighbors"], 0)
        self.assertIn("clustering_coefficient", stats)


class TestBuildSmallWorld(unittest.TestCase):
    """Test the build_small_world helper function."""

    def test_build_from_vectors(self):
        """Should build network from list of vectors."""
        vectors = [
            ("n1", np.array([1.0, 0.0, 0.0])),
            ("n2", np.array([0.9, 0.1, 0.0])),
            ("n3", np.array([0.8, 0.2, 0.0])),
            ("n4", np.array([0.0, 1.0, 0.0])),
            ("n5", np.array([0.1, 0.9, 0.0])),
        ]

        network = build_small_world(vectors, k_local=2, k_long=1, seed=42)

        self.assertEqual(len(network.nodes), 5)
        self.assertEqual(network.k_local, 2)
        self.assertEqual(network.k_long, 1)


class TestSmallWorldProperties(unittest.TestCase):
    """Test that the network exhibits small-world properties."""

    def test_high_clustering(self):
        """Small-world networks should have high clustering coefficient."""
        np.random.seed(42)
        network = SmallWorldProper(k_local=10, k_long=5)

        # Build larger network
        for i in range(50):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        stats = network.get_statistics()

        # Clustering should be > 0 (exact value depends on data)
        self.assertGreater(stats["clustering_coefficient"], 0.0)

    def test_short_paths(self):
        """Small-world networks should have short average path length."""
        np.random.seed(42)
        network = SmallWorldProper(k_local=10, k_long=5)

        for i in range(50):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        stats = network.get_statistics()

        # Path length should be reasonable (less than diameter)
        if stats["estimated_avg_path_length"] != float('inf'):
            self.assertLess(stats["estimated_avg_path_length"], 20)


if __name__ == "__main__":
    unittest.main()
