#!/usr/bin/env python3
"""
Integration tests for Phase 8 multi-interface node implementations.

Tests:
- Scale-free interface distribution (power law)
- Multi-interface node operations
- Unified binary search across interfaces
- Network routing through multi-interface nodes
"""

import sys
import os
import unittest
import numpy as np
from collections import Counter

# Add the runtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/unifyweaver/targets/python_runtime'))

from multi_interface_node import (
    MultiInterfaceNode,
    MultiInterfaceNetwork,
    Interface,
    ExternalConnection,
    generate_scale_free_interface_count,
    create_scale_free_node,
    cosine_similarity,
    cosine_distance,
    compute_cosine_angle,
)


class TestScaleFreeDistribution(unittest.TestCase):
    """Test the power-law interface distribution."""

    def test_basic_generation(self):
        """Should generate values in valid range."""
        import random
        rng = random.Random(42)

        for _ in range(100):
            count = generate_scale_free_interface_count(
                gamma=2.5,
                min_interfaces=1,
                max_interfaces=100,
                rng=rng,
            )
            self.assertGreaterEqual(count, 1)
            self.assertLessEqual(count, 100)

    def test_power_law_shape(self):
        """Distribution should follow power law (most values small)."""
        import random
        rng = random.Random(42)

        counts = [
            generate_scale_free_interface_count(gamma=2.5, rng=rng)
            for _ in range(1000)
        ]

        # Count distribution
        counter = Counter(counts)

        # Most should be 1-2 interfaces
        small_count = sum(counter.get(i, 0) for i in range(1, 3))
        self.assertGreater(small_count / len(counts), 0.5)

        # Few should be 20+
        large_count = sum(counter.get(i, 0) for i in range(20, 101))
        self.assertLess(large_count / len(counts), 0.1)

    def test_gamma_affects_distribution(self):
        """Higher gamma should concentrate more at small values."""
        import random

        def get_mean(gamma):
            rng = random.Random(42)
            counts = [
                generate_scale_free_interface_count(gamma=gamma, rng=rng)
                for _ in range(500)
            ]
            return sum(counts) / len(counts)

        # Higher gamma = lower mean
        mean_2 = get_mean(2.0)
        mean_3 = get_mean(3.0)

        self.assertGreater(mean_2, mean_3)


class TestInterface(unittest.TestCase):
    """Test the Interface class."""

    def test_interface_creation(self):
        """Interface should be created with correct attributes."""
        centroid = np.array([1.0, 0.0, 0.0])
        iface = Interface(interface_id="test_iface", centroid=centroid)

        self.assertEqual(iface.interface_id, "test_iface")
        np.testing.assert_array_equal(iface.centroid, centroid)
        self.assertEqual(iface.document_count, 0)
        self.assertEqual(iface.queries_handled, 0)


class TestMultiInterfaceNode(unittest.TestCase):
    """Test the MultiInterfaceNode class."""

    def test_node_creation(self):
        """Node should be created empty."""
        node = MultiInterfaceNode(node_id="test-node")

        self.assertEqual(node.node_id, "test-node")
        self.assertEqual(node.num_interfaces, 0)
        self.assertEqual(node.total_connections, 0)

    def test_add_interface(self):
        """Should be able to add interfaces."""
        node = MultiInterfaceNode(node_id="n1")

        iface1 = Interface(
            interface_id="iface1",
            centroid=np.array([1.0, 0.0, 0.0])
        )
        node.add_interface(iface1)

        self.assertEqual(node.num_interfaces, 1)

        iface2 = Interface(
            interface_id="iface2",
            centroid=np.array([0.0, 1.0, 0.0])
        )
        node.add_interface(iface2)

        self.assertEqual(node.num_interfaces, 2)

    def test_centroid_computation(self):
        """Node centroid should be average of interface centroids."""
        node = MultiInterfaceNode(node_id="n1")

        node.add_interface(Interface("i1", np.array([1.0, 0.0, 0.0])))
        node.add_interface(Interface("i2", np.array([0.0, 1.0, 0.0])))

        expected = np.array([0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(node.centroid, expected)

    def test_add_connection(self):
        """Should be able to add connections between interfaces."""
        node = MultiInterfaceNode(node_id="n1")
        node.add_interface(Interface("iface1", np.array([1.0, 0.0, 0.0])))

        success = node.add_connection(
            source_interface_id="iface1",
            target_node_id="n2",
            target_interface_id="n2_iface1",
            target_centroid=np.array([0.9, 0.1, 0.0]),
        )

        self.assertTrue(success)
        self.assertEqual(node.total_connections, 1)

    def test_max_connections_per_interface(self):
        """Should respect max connections limit."""
        node = MultiInterfaceNode(node_id="n1", max_connections_per_interface=2)
        node.add_interface(Interface("iface1", np.array([1.0, 0.0, 0.0])))

        # Add max connections
        for i in range(2):
            success = node.add_connection(
                source_interface_id="iface1",
                target_node_id=f"n{i}",
                target_interface_id=f"iface_{i}",
                target_centroid=np.random.randn(3),
            )
            self.assertTrue(success)

        # Third should fail
        success = node.add_connection(
            source_interface_id="iface1",
            target_node_id="n_extra",
            target_interface_id="iface_extra",
            target_centroid=np.random.randn(3),
        )
        self.assertFalse(success)

    def test_find_closest_interfaces(self):
        """Should find interfaces closest to query."""
        node = MultiInterfaceNode(node_id="n1")

        # Add interfaces in different directions
        for i in range(5):
            angle = i * np.pi / 4
            centroid = np.array([np.cos(angle), np.sin(angle), 0.0])
            node.add_interface(Interface(f"iface_{i}", centroid))

        # Query close to first interface
        query = np.array([1.0, 0.1, 0.0])
        closest = node.find_closest_interfaces(query, k=2)

        self.assertEqual(len(closest), 2)
        # First result should be closest
        self.assertEqual(closest[0][0], "iface_0")

    def test_route_query(self):
        """Should route query to appropriate interface."""
        node = MultiInterfaceNode(node_id="n1")

        # Add interface and connections
        node.add_interface(Interface("iface1", np.array([1.0, 0.0, 0.0])))
        node.add_connection(
            "iface1", "n2", "n2_iface1",
            np.array([0.9, 0.1, 0.0])
        )

        query = np.array([1.0, 0.0, 0.0])
        interface_ids, connections = node.route_query(query)

        self.assertIn("iface1", interface_ids)

    def test_get_statistics(self):
        """Should return meaningful statistics."""
        node = MultiInterfaceNode(node_id="n1")
        node.add_interface(Interface("i1", np.array([1.0, 0.0, 0.0])))
        node.add_interface(Interface("i2", np.array([0.0, 1.0, 0.0])))

        stats = node.get_statistics()

        self.assertEqual(stats["node_id"], "n1")
        self.assertEqual(stats["num_interfaces"], 2)


class TestCreateScaleFreeNode(unittest.TestCase):
    """Test the create_scale_free_node function."""

    def test_creates_node_with_interfaces(self):
        """Should create node with power-law distributed interfaces."""
        import random
        rng = random.Random(42)

        node = create_scale_free_node(
            node_id="test",
            base_centroid=np.array([1.0, 0.0, 0.0]),
            gamma=2.5,
            rng=rng,
        )

        self.assertEqual(node.node_id, "test")
        self.assertGreater(node.num_interfaces, 0)

    def test_interface_centroids_near_base(self):
        """Interface centroids should be near base centroid."""
        import random
        rng = random.Random(42)

        base = np.array([1.0, 0.0, 0.0])
        node = create_scale_free_node(
            node_id="test",
            base_centroid=base,
            gamma=2.5,
            min_interfaces=5,
            rng=rng,
        )

        for _, iface in node.interfaces:
            sim = cosine_similarity(base, iface.centroid)
            self.assertGreater(sim, 0.5)  # Should be reasonably close


class TestMultiInterfaceNetwork(unittest.TestCase):
    """Test the MultiInterfaceNetwork class."""

    def test_network_creation(self):
        """Network should be created empty."""
        network = MultiInterfaceNetwork(gamma=2.5)

        self.assertEqual(len(network.nodes), 0)

    def test_add_node(self):
        """Should be able to add nodes."""
        network = MultiInterfaceNetwork(gamma=2.5)

        node = network.add_node("n1", np.array([1.0, 0.0, 0.0]))

        self.assertEqual(len(network.nodes), 1)
        self.assertIn("n1", network.nodes)

    def test_nodes_get_connected(self):
        """Nodes should be connected when added."""
        network = MultiInterfaceNetwork(
            connections_per_interface=5,
            gamma=2.5,
        )

        # Add multiple nodes
        import random
        random.seed(42)
        np.random.seed(42)

        for i in range(5):
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec, num_interfaces=2)

        # Check that nodes have connections
        total_connections = sum(
            n.total_connections for n in network.nodes.values()
        )
        self.assertGreater(total_connections, 0)

    def test_route_query(self):
        """Should be able to route queries through network."""
        network = MultiInterfaceNetwork(gamma=2.5)

        np.random.seed(42)
        import random
        random.seed(42)

        # Build network
        for i in range(10):
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec, num_interfaces=2)

        # Route a query
        query = np.array([1.0, 0.0, 0.0])
        path, comparisons = network.route_query(query, max_hops=5)

        self.assertGreater(len(path), 0)
        self.assertGreater(comparisons, 0)

    def test_get_statistics(self):
        """Should return network statistics."""
        network = MultiInterfaceNetwork(gamma=2.5)

        np.random.seed(42)
        import random
        random.seed(42)

        for i in range(5):
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        stats = network.get_statistics()

        self.assertEqual(stats["num_nodes"], 5)
        self.assertGreater(stats["total_interfaces"], 0)
        self.assertIn("avg_interfaces_per_node", stats)


class TestMultiInterfaceProperties(unittest.TestCase):
    """Test scale-free network properties."""

    def test_scale_free_distribution(self):
        """Network should have scale-free interface distribution."""
        network = MultiInterfaceNetwork(gamma=2.5)

        np.random.seed(42)
        import random
        random.seed(42)

        # Build larger network
        for i in range(50):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        stats = network.get_statistics()
        distribution = stats["interface_distribution"]

        # Should have variety in interface counts
        self.assertGreater(len(distribution), 1)

        # Most nodes should have few interfaces
        small_interface_count = sum(
            count for k, count in distribution.items() if k <= 3
        )
        self.assertGreater(small_interface_count / 50, 0.5)

    def test_hub_and_spoke_structure(self):
        """Some nodes should be hubs with many connections."""
        network = MultiInterfaceNetwork(
            connections_per_interface=10,
            gamma=2.5,
        )

        np.random.seed(42)
        import random
        random.seed(42)

        for i in range(30):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        # Find node with most interfaces
        max_interfaces = max(n.num_interfaces for n in network.nodes.values())

        # Should have some hubs
        self.assertGreater(max_interfaces, 1)


if __name__ == "__main__":
    unittest.main()
