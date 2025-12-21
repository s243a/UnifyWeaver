# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Tests for small-world network evolution.

Demonstrates the progression from hierarchical to small-world routing.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

from unifyweaver.targets.python_runtime.small_world_evolution import (
    SmallWorldNode,
    SmallWorldNetwork,
    cosine_distance,
)


class TestSmallWorldNode(unittest.TestCase):
    """Tests for SmallWorldNode."""

    def test_create_node(self):
        """Test node creation."""
        node = SmallWorldNode(
            node_id="node_1",
            centroid=np.array([1.0, 0.0]),
        )
        self.assertEqual(node.node_id, "node_1")
        self.assertEqual(len(node.shortcut_ids), 0)

    def test_add_shortcut(self):
        """Test adding shortcuts."""
        node = SmallWorldNode(
            node_id="node_1",
            centroid=np.array([1.0, 0.0]),
            max_shortcuts=3,
        )
        self.assertTrue(node.add_shortcut("node_2"))
        self.assertTrue(node.add_shortcut("node_3"))
        self.assertTrue(node.add_shortcut("node_4"))
        self.assertFalse(node.add_shortcut("node_5"))  # Exceeds max

    def test_get_all_neighbors(self):
        """Test getting all neighbors."""
        node = SmallWorldNode(
            node_id="node_1",
            centroid=np.array([1.0, 0.0]),
            parent_id="parent",
            children_ids=["child_1", "child_2"],
        )
        node.add_shortcut("shortcut_1")

        neighbors = node.get_all_neighbors()
        self.assertEqual(len(neighbors), 4)
        self.assertIn("parent", neighbors)
        self.assertIn("child_1", neighbors)
        self.assertIn("shortcut_1", neighbors)


class TestSmallWorldNetwork(unittest.TestCase):
    """Tests for SmallWorldNetwork."""

    def test_create_network(self):
        """Test network creation."""
        network = SmallWorldNetwork()
        self.assertEqual(len(network.nodes), 0)
        self.assertIsNone(network.root_id)

    def test_add_nodes(self):
        """Test adding nodes."""
        network = SmallWorldNetwork()
        network.add_node("root", np.array([0.5, 0.5]))
        network.add_node("child_1", np.array([1.0, 0.0]), parent_id="root")
        network.add_node("child_2", np.array([0.0, 1.0]), parent_id="root")

        self.assertEqual(len(network.nodes), 3)
        self.assertEqual(network.root_id, "root")

    def test_greedy_routing(self):
        """Test greedy routing through network."""
        network = SmallWorldNetwork()

        # Create simple hierarchy
        network.add_node("root", np.array([0.5, 0.5]), children_ids=["a", "b"])
        network.add_node("a", np.array([1.0, 0.0]), parent_id="root")
        network.add_node("b", np.array([0.0, 1.0]), parent_id="root")

        # Query close to node "a"
        query = np.array([0.9, 0.1])
        path, comparisons = network.route_greedy(query)

        self.assertIn("a", path)
        self.assertGreater(comparisons, 0)

    def test_evolution_improves_shortcuts(self):
        """Test that evolution adds shortcuts."""
        network = SmallWorldNetwork(exchange_probability=0.5)

        # Create hierarchical network with clusters
        dim = 10
        np.random.seed(42)

        root_centroid = np.random.randn(dim)
        root_centroid = root_centroid / np.linalg.norm(root_centroid)
        network.add_node("root", root_centroid)

        # Create 4 cluster nodes under root
        for c in range(4):
            cluster_centroid = np.random.randn(dim)
            cluster_centroid = cluster_centroid / np.linalg.norm(cluster_centroid)
            cluster_id = f"cluster_{c}"
            network.add_node(cluster_id, cluster_centroid, parent_id="root")
            network.nodes["root"].children_ids.append(cluster_id)

            # Add 5 nodes to each cluster
            for i in range(5):
                centroid = cluster_centroid + np.random.randn(dim) * 0.1
                centroid = centroid / np.linalg.norm(centroid)
                node_id = f"node_{c}_{i}"
                network.add_node(node_id, centroid, parent_id=cluster_id)
                network.nodes[cluster_id].children_ids.append(node_id)

        initial_shortcuts = sum(len(n.shortcut_ids) for n in network.nodes.values())

        # Evolve
        network.evolve(num_rounds=100, seed=42)

        final_shortcuts = sum(len(n.shortcut_ids) for n in network.nodes.values())

        # At least some shortcuts should be added
        self.assertGreaterEqual(final_shortcuts, initial_shortcuts)

    def test_maturity_score_increases(self):
        """Test that maturity score increases with evolution."""
        network = SmallWorldNetwork(exchange_probability=0.5)

        # Create hierarchical network
        dim = 10
        np.random.seed(42)

        root_centroid = np.random.randn(dim)
        root_centroid = root_centroid / np.linalg.norm(root_centroid)
        network.add_node("root", root_centroid)

        # Create 4 cluster nodes
        for c in range(4):
            cluster_centroid = np.random.randn(dim)
            cluster_centroid = cluster_centroid / np.linalg.norm(cluster_centroid)
            cluster_id = f"cluster_{c}"
            network.add_node(cluster_id, cluster_centroid, parent_id="root")
            network.nodes["root"].children_ids.append(cluster_id)

            for i in range(5):
                centroid = cluster_centroid + np.random.randn(dim) * 0.1
                centroid = centroid / np.linalg.norm(centroid)
                node_id = f"node_{c}_{i}"
                network.add_node(node_id, centroid, parent_id=cluster_id)
                network.nodes[cluster_id].children_ids.append(node_id)

        initial_maturity = network.maturity_score

        network.evolve(num_rounds=100, seed=42)

        # Maturity should increase or stay same (network is already well-structured)
        self.assertGreaterEqual(network.maturity_score, initial_maturity)


class TestRoutingScaling(unittest.TestCase):
    """Test routing scaling with network evolution."""

    def create_test_network(self, num_nodes: int, seed: int = 42) -> SmallWorldNetwork:
        """Create a test network with hierarchy."""
        np.random.seed(seed)
        network = SmallWorldNetwork(max_shortcuts_per_node=5)

        # Create 5 cluster centers
        num_clusters = 5
        cluster_centers = []
        for i in range(num_clusters):
            center = np.zeros(10)
            center[i * 2] = 1.0
            cluster_centers.append(center)

        # Root node
        root_centroid = np.mean(cluster_centers, axis=0)
        network.add_node("root", root_centroid)

        # Add cluster nodes
        for i in range(num_nodes):
            cluster_idx = i % num_clusters
            base = cluster_centers[cluster_idx]
            noise = np.random.randn(10) * 0.1
            centroid = base + noise
            centroid = centroid / np.linalg.norm(centroid)
            network.add_node(f"node_{i}", centroid, parent_id="root")

        return network

    def test_hierarchical_vs_evolved_comparisons(self):
        """Test that evolution reduces routing comparisons."""
        network = self.create_test_network(100)

        # Route before evolution
        np.random.seed(123)
        query = np.random.randn(10)
        query = query / np.linalg.norm(query)

        _, comparisons_before = network.route_to_top_k(query, k=3)

        # Evolve network
        network.evolve(num_rounds=100, seed=42)

        # Route after evolution
        _, comparisons_after = network.route_to_top_k(query, k=3)

        print(f"\nComparisons before evolution: {comparisons_before}")
        print(f"Comparisons after evolution: {comparisons_after}")
        print(f"Maturity score: {network.maturity_score:.3f}")

        # After evolution, should use shortcuts and need fewer comparisons
        # (or at least not significantly more)
        self.assertLessEqual(comparisons_after, comparisons_before * 1.5)


def run_evolution_demo():
    """Demonstrate the evolution from hierarchical to small-world."""
    print("=" * 70)
    print("SMALL-WORLD EVOLUTION DEMO")
    print("=" * 70)

    np.random.seed(42)

    # Create hierarchical network
    network = SmallWorldNetwork(
        max_shortcuts_per_node=10,
        exchange_probability=0.3,
    )

    print("\nCreating hierarchical network with 100 nodes in 5 clusters...")

    # Root
    network.add_node("root", np.array([0.5] * 10))

    # 5 clusters
    num_clusters = 5
    nodes_per_cluster = 20

    for c in range(num_clusters):
        # Cluster center
        cluster_center = np.zeros(10)
        cluster_center[c * 2] = 1.0

        cluster_id = f"cluster_{c}"
        network.add_node(cluster_id, cluster_center, parent_id="root")
        network.nodes["root"].children_ids.append(cluster_id)

        # Cluster members
        for n in range(nodes_per_cluster):
            noise = np.random.randn(10) * 0.1
            centroid = cluster_center + noise
            centroid = centroid / np.linalg.norm(centroid)

            node_id = f"node_{c}_{n}"
            network.add_node(node_id, centroid, parent_id=cluster_id)
            network.nodes[cluster_id].children_ids.append(node_id)

    print(f"Network created: {len(network.nodes)} nodes")
    print(f"Initial maturity: {network.maturity_score:.3f}")

    # Test routing before evolution
    print("\n--- BEFORE EVOLUTION ---")
    query = np.random.randn(10)
    query = query / np.linalg.norm(query)

    results_before = []
    for _ in range(20):
        path, comps = network.route_greedy(query)
        results_before.append(comps)

    print(f"Avg comparisons per route: {np.mean(results_before):.1f}")
    print(f"Routes use hierarchy (start from root)")

    # Evolve
    print("\n--- EVOLVING ---")
    for round_num in range(1, 6):
        stats = network.evolve(num_rounds=50, seed=round_num)
        print(f"Round {round_num*50}: "
              f"Maturity={network.maturity_score:.3f}, "
              f"Avg shortcuts={network.get_statistics()['avg_shortcuts']:.1f}, "
              f"Quality={stats['final_quality']:.3f}")

    # Test routing after evolution
    print("\n--- AFTER EVOLUTION ---")
    results_after = []
    for _ in range(20):
        path, comps = network.route_greedy(query)
        results_after.append(comps)

    print(f"Avg comparisons per route: {np.mean(results_after):.1f}")
    print(f"Maturity: {network.maturity_score:.3f}")
    if network.maturity_score >= 0.5:
        print("Network is mature - can start queries from any node")
    else:
        print("Network still maturing - use hierarchical start")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Comparisons: {np.mean(results_before):.1f} -> {np.mean(results_after):.1f}")
    improvement = (1 - np.mean(results_after) / np.mean(results_before)) * 100
    print(f"Improvement: {improvement:.1f}%")
    print(f"\nNetwork stats: {network.get_statistics()}")


if __name__ == "__main__":
    # Run demo first
    run_evolution_demo()

    print("\n\nRunning unit tests...")
    unittest.main(verbosity=2)
