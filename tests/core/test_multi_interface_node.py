# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Tests for multi-interface nodes with scale-free distribution.
"""

import sys
import os
import unittest
import random
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

from unifyweaver.targets.python_runtime.multi_interface_node import (
    Interface,
    MultiInterfaceNode,
    MultiInterfaceNetwork,
    generate_scale_free_interface_count,
    create_scale_free_node,
    cosine_distance,
)


class TestInterface(unittest.TestCase):
    """Tests for Interface."""

    def test_create_interface(self):
        """Test interface creation."""
        centroid = np.array([1.0, 0.0, 0.0])
        iface = Interface(
            interface_id="test_iface",
            centroid=centroid,
        )

        self.assertEqual(iface.interface_id, "test_iface")
        self.assertEqual(iface.queries_handled, 0)


class TestMultiInterfaceNode(unittest.TestCase):
    """Tests for MultiInterfaceNode."""

    def test_create_node(self):
        """Test node creation."""
        node = MultiInterfaceNode(node_id="test_node")
        self.assertEqual(node.node_id, "test_node")
        self.assertEqual(node.num_interfaces, 0)

    def test_add_interfaces(self):
        """Test adding interfaces to a node."""
        node = MultiInterfaceNode(node_id="test_node")

        for i in range(5):
            angle = (i / 5) * 2 * np.pi
            centroid = np.array([np.cos(angle), np.sin(angle), 0.0])
            iface = Interface(
                interface_id=f"iface_{i}",
                centroid=centroid,
            )
            node.add_interface(iface)

        self.assertEqual(node.num_interfaces, 5)

        # Interfaces should be sorted by angle
        angles = [a for a, _ in node.interfaces]
        self.assertEqual(angles, sorted(angles))

    def test_find_closest_interfaces(self):
        """Test finding closest interfaces to a query."""
        np.random.seed(42)
        node = MultiInterfaceNode(node_id="test_node")

        # Add interfaces at known positions
        positions = [
            (1.0, 0.0),   # 0 degrees
            (0.0, 1.0),   # 90 degrees
            (-1.0, 0.0),  # 180 degrees
            (0.0, -1.0),  # 270 degrees
        ]

        for i, (x, y) in enumerate(positions):
            centroid = np.array([x, y, 0.0])
            centroid = centroid / np.linalg.norm(centroid)
            iface = Interface(
                interface_id=f"iface_{i}",
                centroid=centroid,
            )
            node.add_interface(iface)

        # Query near 0 degrees
        query = np.array([0.9, 0.1, 0.0])
        query = query / np.linalg.norm(query)

        closest = node.find_closest_interfaces(query, k=2)

        # iface_0 (0 degrees) should be closest
        self.assertEqual(closest[0][0], "iface_0")

    def test_add_connections(self):
        """Test adding external connections."""
        node = MultiInterfaceNode(node_id="test_node")

        # Add one interface
        centroid = np.array([1.0, 0.0, 0.0])
        iface = Interface(interface_id="iface_0", centroid=centroid)
        node.add_interface(iface)

        # Add connections
        for i in range(10):
            target_centroid = np.random.randn(3)
            target_centroid = target_centroid / np.linalg.norm(target_centroid)

            success = node.add_connection(
                source_interface_id="iface_0",
                target_node_id=f"node_{i}",
                target_interface_id=f"node_{i}_iface_0",
                target_centroid=target_centroid,
            )
            self.assertTrue(success)

        self.assertEqual(node.total_connections, 10)

        # Connections should be sorted by angle
        angles = [c.angle for c in node.connections]
        self.assertEqual(angles, sorted(angles))

    def test_connection_limit(self):
        """Test that connections are limited per interface."""
        node = MultiInterfaceNode(
            node_id="test_node",
            max_connections_per_interface=5,
        )

        centroid = np.array([1.0, 0.0, 0.0])
        iface = Interface(interface_id="iface_0", centroid=centroid)
        node.add_interface(iface)

        # Try to add 10 connections
        added = 0
        for i in range(10):
            target_centroid = np.random.randn(3)
            if node.add_connection(
                source_interface_id="iface_0",
                target_node_id=f"node_{i}",
                target_interface_id=f"node_{i}_iface_0",
                target_centroid=target_centroid,
            ):
                added += 1

        # Should only add 5
        self.assertEqual(added, 5)

    def test_route_query(self):
        """Test query routing through a node."""
        np.random.seed(42)
        node = MultiInterfaceNode(node_id="test_node")

        # Add interfaces
        for i in range(4):
            angle = (i / 4) * 2 * np.pi
            centroid = np.array([np.cos(angle), np.sin(angle), 0.0])
            iface = Interface(interface_id=f"iface_{i}", centroid=centroid)
            node.add_interface(iface)

        # Add connections to each interface
        for i in range(4):
            for j in range(5):
                target_centroid = np.random.randn(3)
                target_centroid = target_centroid / np.linalg.norm(target_centroid)
                node.add_connection(
                    source_interface_id=f"iface_{i}",
                    target_node_id=f"target_{i}_{j}",
                    target_interface_id=f"target_{i}_{j}_iface_0",
                    target_centroid=target_centroid,
                )

        # Route a query
        query = np.array([1.0, 0.1, 0.0])
        query = query / np.linalg.norm(query)

        interface_ids, connections = node.route_query(query)

        self.assertGreater(len(interface_ids), 0)
        self.assertGreater(len(connections), 0)

        print(f"\nRoute query result:")
        print(f"  Primary interface: {interface_ids[0]}")
        print(f"  Connections found: {len(connections)}")


class TestScaleFreeDistribution(unittest.TestCase):
    """Tests for scale-free interface distribution."""

    def test_power_law_generation(self):
        """Test that interface counts follow power law."""
        rng = random.Random(42)
        counts = [
            generate_scale_free_interface_count(gamma=2.5, rng=rng)
            for _ in range(1000)
        ]

        # Distribution should be heavy-tailed
        counter = Counter(counts)

        print("\n" + "=" * 50)
        print("SCALE-FREE INTERFACE DISTRIBUTION")
        print("=" * 50)
        print(f"{'Interfaces':>10} {'Count':>10} {'Fraction':>10}")
        print("-" * 30)

        for k in sorted(counter.keys())[:15]:
            frac = counter[k] / len(counts)
            print(f"{k:>10} {counter[k]:>10} {frac:>10.3f}")

        # Most nodes should have few interfaces
        small_interface_nodes = sum(1 for c in counts if c <= 3)
        self.assertGreater(small_interface_nodes / len(counts), 0.7)

        # Some nodes should have many interfaces
        large_interface_nodes = sum(1 for c in counts if c >= 10)
        self.assertGreater(large_interface_nodes, 0)

    def test_create_scale_free_node(self):
        """Test creating a node with scale-free interfaces."""
        np.random.seed(42)
        rng = random.Random(42)

        base_centroid = np.array([1.0, 0.0, 0.0, 0.0])

        node = create_scale_free_node(
            node_id="test",
            base_centroid=base_centroid,
            gamma=2.5,
            rng=rng,
        )

        self.assertGreater(node.num_interfaces, 0)
        print(f"\nScale-free node created with {node.num_interfaces} interfaces")


class TestMultiInterfaceNetwork(unittest.TestCase):
    """Tests for MultiInterfaceNetwork."""

    def setUp(self):
        """Create test network."""
        np.random.seed(42)
        random.seed(42)

        self.network = MultiInterfaceNetwork(
            connections_per_interface=10,
            gamma=2.5,
        )

    def test_create_network(self):
        """Test network creation."""
        self.assertEqual(len(self.network.nodes), 0)

    def test_add_nodes(self):
        """Test adding nodes to network."""
        for i in range(20):
            centroid = np.random.randn(8)
            centroid = centroid / np.linalg.norm(centroid)
            self.network.add_node(f"node_{i}", centroid)

        stats = self.network.get_statistics()

        print("\n" + "=" * 50)
        print("NETWORK STATISTICS")
        print("=" * 50)
        print(f"Nodes: {stats['num_nodes']}")
        print(f"Total interfaces: {stats['total_interfaces']}")
        print(f"Avg interfaces/node: {stats['avg_interfaces_per_node']:.1f}")
        print(f"Min interfaces: {stats['min_interfaces']}")
        print(f"Max interfaces: {stats['max_interfaces']}")
        print(f"Total connections: {stats['total_connections']}")

        self.assertEqual(stats['num_nodes'], 20)
        self.assertGreater(stats['total_interfaces'], 20)

    def test_network_routing(self):
        """Test routing through the network."""
        # Build network
        for i in range(30):
            centroid = np.random.randn(8)
            centroid = centroid / np.linalg.norm(centroid)
            self.network.add_node(f"node_{i}", centroid)

        # Route queries
        success = 0
        total_hops = 0
        total_comparisons = 0

        for _ in range(20):
            # Pick a target
            target_node = random.choice(list(self.network.nodes.values()))
            if not target_node.interfaces:
                continue

            _, target_iface = target_node.interfaces[0]
            query = target_iface.centroid + np.random.randn(8) * 0.1
            query = query / np.linalg.norm(query)

            path, comparisons = self.network.route_query(query, max_hops=10)

            total_hops += len(path)
            total_comparisons += comparisons

            # Check if we reached target node
            if any(node_id == target_node.node_id for node_id, _ in path):
                success += 1

        print(f"\nRouting test:")
        print(f"  Success: {success}/20")
        print(f"  Avg hops: {total_hops/20:.1f}")
        print(f"  Avg comparisons: {total_comparisons/20:.1f}")

        # Should have some success
        self.assertGreater(success, 5)

    def test_hub_nodes_exist(self):
        """Test that scale-free distribution creates hub nodes."""
        # Build larger network
        for i in range(50):
            centroid = np.random.randn(8)
            centroid = centroid / np.linalg.norm(centroid)
            self.network.add_node(f"node_{i}", centroid)

        stats = self.network.get_statistics()
        distribution = stats['interface_distribution']

        print("\n" + "=" * 50)
        print("INTERFACE DISTRIBUTION (50 nodes)")
        print("=" * 50)
        for k in sorted(distribution.keys()):
            bar = "*" * distribution[k]
            print(f"{k:>3} interfaces: {bar} ({distribution[k]})")

        # Should have some hub nodes (5+ interfaces)
        hub_count = sum(
            count for k, count in distribution.items() if k >= 5
        )
        print(f"\nHub nodes (5+ interfaces): {hub_count}")

        # At least some hubs should exist
        self.assertGreater(hub_count, 0)


class TestBinarySearchEfficiency(unittest.TestCase):
    """Test that binary search provides efficiency gains."""

    def test_large_interface_lookup(self):
        """Test lookup efficiency with many interfaces."""
        np.random.seed(42)

        # Create node with many interfaces (simulating a hub)
        node = MultiInterfaceNode(node_id="hub")

        for i in range(100):
            angle = (i / 100) * 2 * np.pi
            centroid = np.zeros(8)
            centroid[0] = np.cos(angle)
            centroid[1] = np.sin(angle)
            centroid[2:] = np.random.randn(6) * 0.1
            centroid = centroid / np.linalg.norm(centroid)

            iface = Interface(interface_id=f"iface_{i}", centroid=centroid)
            node.add_interface(iface)

        # Query should still be fast via binary search
        query = np.random.randn(8)
        query = query / np.linalg.norm(query)

        closest = node.find_closest_interfaces(query, k=3)

        self.assertEqual(len(closest), 3)

        print(f"\nLarge interface lookup:")
        print(f"  Interfaces: {node.num_interfaces}")
        print(f"  Found {len(closest)} closest")
        print(f"  Top: {closest[0][0]} (dist={closest[0][1]:.4f})")

    def test_large_connection_lookup(self):
        """Test connection lookup with many connections."""
        np.random.seed(42)

        node = MultiInterfaceNode(
            node_id="hub",
            max_connections_per_interface=200,
        )

        # Add interface
        centroid = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        iface = Interface(interface_id="iface_0", centroid=centroid)
        node.add_interface(iface)

        # Add many connections
        for i in range(200):
            target_centroid = np.random.randn(8)
            target_centroid = target_centroid / np.linalg.norm(target_centroid)
            node.add_connection(
                source_interface_id="iface_0",
                target_node_id=f"target_{i}",
                target_interface_id=f"target_{i}_iface_0",
                target_centroid=target_centroid,
            )

        # Query should use binary search
        query = np.random.randn(8)
        query = query / np.linalg.norm(query)

        connections = node.find_connections_for_query(
            query,
            interface_id="iface_0",
            window_size=10,
        )

        print(f"\nLarge connection lookup:")
        print(f"  Total connections: {node.total_connections}")
        print(f"  Found {len(connections)} candidates (window=10)")

        # Should return limited candidates, not all 200
        self.assertLessEqual(len(connections), 21)  # 2*10 + 1


def run_demo():
    """Demonstrate scale-free multi-interface network."""
    print("=" * 70)
    print("SCALE-FREE MULTI-INTERFACE NETWORK DEMO")
    print("=" * 70)

    np.random.seed(42)
    random.seed(42)

    # Create network
    network = MultiInterfaceNetwork(
        connections_per_interface=15,
        gamma=2.5,
    )

    print("\nBuilding network with 50 nodes...")
    for i in range(50):
        centroid = np.random.randn(16)
        centroid = centroid / np.linalg.norm(centroid)
        network.add_node(f"node_{i}", centroid)

    stats = network.get_statistics()

    print(f"\nNetwork statistics:")
    print(f"  Physical nodes: {stats['num_nodes']}")
    print(f"  Logical interfaces: {stats['total_interfaces']}")
    print(f"  Avg interfaces/node: {stats['avg_interfaces_per_node']:.1f}")
    print(f"  Max interfaces (hub): {stats['max_interfaces']}")
    print(f"  Total connections: {stats['total_connections']}")

    # Show distribution
    print(f"\nInterface distribution:")
    for k in sorted(stats['interface_distribution'].keys())[:10]:
        count = stats['interface_distribution'][k]
        bar = "*" * count
        print(f"  {k:>2} interfaces: {bar}")

    # Routing test
    print("\nRouting test (30 queries)...")
    success = 0
    total_hops = 0

    for _ in range(30):
        target_node = random.choice(list(network.nodes.values()))
        if not target_node.interfaces:
            continue

        _, target_iface = target_node.interfaces[0]
        query = target_iface.centroid + np.random.randn(16) * 0.05

        path, _ = network.route_query(query, max_hops=10)
        total_hops += len(path)

        if any(node_id == target_node.node_id for node_id, _ in path):
            success += 1

    print(f"  Success: {success}/30")
    print(f"  Avg hops: {total_hops/30:.1f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_demo()
    print("\n\nRunning unit tests...")
    unittest.main(verbosity=2)
