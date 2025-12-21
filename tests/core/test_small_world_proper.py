# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Tests for proper small-world network with high connectivity.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import random

from unifyweaver.targets.python_runtime.small_world_proper import (
    SWNode,
    SmallWorldProper,
    build_small_world,
    cosine_similarity,
)


class TestSWNode(unittest.TestCase):
    """Tests for SWNode."""

    def test_create_node(self):
        """Test node creation."""
        node = SWNode(
            node_id="test",
            vector=np.array([1.0, 0.0]),
        )
        self.assertEqual(node.node_id, "test")
        self.assertEqual(len(node.neighbors), 0)

    def test_add_neighbor(self):
        """Test adding neighbors."""
        node = SWNode(
            node_id="test",
            vector=np.array([1.0, 0.0]),
            max_neighbors=3,
        )

        self.assertTrue(node.add_neighbor("n1"))
        self.assertTrue(node.add_neighbor("n2"))
        self.assertTrue(node.add_neighbor("n3"))
        self.assertFalse(node.add_neighbor("n4"))  # Exceeds max

    def test_cannot_add_self(self):
        """Test that node cannot add itself as neighbor."""
        node = SWNode(node_id="test", vector=np.array([1.0, 0.0]))
        self.assertFalse(node.add_neighbor("test"))


class TestSmallWorldProper(unittest.TestCase):
    """Tests for SmallWorldProper network."""

    def test_create_network(self):
        """Test network creation."""
        network = SmallWorldProper()
        self.assertEqual(len(network.nodes), 0)

    def test_add_nodes_with_connections(self):
        """Test that adding nodes creates connections."""
        np.random.seed(42)
        network = SmallWorldProper(k_local=5, k_long=2)

        # Add nodes
        for i in range(20):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        # Check connections
        stats = network.get_statistics()
        self.assertEqual(stats["num_nodes"], 20)
        self.assertGreater(stats["avg_neighbors"], 5)  # Should have k_local + some

    def test_high_connectivity(self):
        """Test that network achieves target connectivity."""
        np.random.seed(42)
        network = SmallWorldProper(k_local=10, k_long=5, max_neighbors=20)

        # Add enough nodes
        for i in range(50):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        stats = network.get_statistics()

        print(f"\nConnectivity test:")
        print(f"  Nodes: {stats['num_nodes']}")
        print(f"  Avg neighbors: {stats['avg_neighbors']:.1f}")
        print(f"  Min neighbors: {stats['min_neighbors']}")

        # All nodes should have substantial connections
        self.assertGreaterEqual(stats["min_neighbors"], 10)
        self.assertGreaterEqual(stats["avg_neighbors"], 15)

    def test_clustering_coefficient(self):
        """Test that network has high clustering."""
        np.random.seed(42)
        network = SmallWorldProper(k_local=10, k_long=3)

        for i in range(50):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        stats = network.get_statistics()

        print(f"\nClustering: {stats['clustering_coefficient']:.3f}")

        # Should have decent clustering (similar nodes connected)
        self.assertGreater(stats["clustering_coefficient"], 0.3)

    def test_short_path_lengths(self):
        """Test that network has short average path length."""
        np.random.seed(42)
        network = SmallWorldProper(k_local=10, k_long=5)

        for i in range(100):
            vec = np.random.randn(10)
            vec = vec / np.linalg.norm(vec)
            network.add_node(f"n{i}", vec)

        stats = network.get_statistics()

        print(f"\nAvg path length: {stats['estimated_avg_path_length']:.2f}")

        # Path length should be small (small-world property)
        self.assertLess(stats["estimated_avg_path_length"], 4.0)


class TestRouting(unittest.TestCase):
    """Tests for routing in proper small-world."""

    def setUp(self):
        """Create test network."""
        np.random.seed(42)
        random.seed(42)

        self.network = SmallWorldProper(k_local=10, k_long=5)

        # Create clustered data
        self.dim = 10
        for cluster in range(5):
            center = np.zeros(self.dim)
            center[cluster * 2] = 1.0
            center = center / np.linalg.norm(center)

            for i in range(10):
                vec = center + np.random.randn(self.dim) * 0.1
                vec = vec / np.linalg.norm(vec)
                self.network.add_node(f"c{cluster}_n{i}", vec)

    def test_greedy_routing_success(self):
        """Test greedy routing finds targets."""
        success = 0
        total = 20

        for _ in range(total):
            # Random target and start
            target_id = random.choice(list(self.network.nodes.keys()))
            start_id = random.choice(list(self.network.nodes.keys()))

            target = self.network.nodes[target_id]
            query = target.vector.copy()

            path, _ = self.network.route_greedy(query, start_node_id=start_id)

            if target_id in path:
                success += 1

        print(f"\nGreedy success: {success}/{total}")
        # Greedy can get stuck - 60% is acceptable, k-NN search is better
        self.assertGreaterEqual(success, total * 0.6)

    def test_knn_search(self):
        """Test k-NN search finds correct neighbors."""
        target_id = "c2_n5"
        target = self.network.nodes[target_id]
        query = target.vector.copy()

        results, comps = self.network.search_knn(query, k=5, ef=30)

        print(f"\nk-NN search:")
        print(f"  Comparisons: {comps}")
        print(f"  Top result: {results[0][0]} (dist={results[0][1]:.4f})")

        # Target should be first (distance ~0)
        self.assertEqual(results[0][0], target_id)
        self.assertLess(results[0][1], 0.01)

    def test_routing_from_any_start(self):
        """Test routing works from any starting node."""
        target_id = "c3_n7"
        target = self.network.nodes[target_id]
        query = target.vector.copy()

        success_count = 0
        for start_id in list(self.network.nodes.keys()):
            results, _ = self.network.search_knn(
                query, k=5, ef=30, start_node_id=start_id
            )
            if any(r[0] == target_id for r in results):
                success_count += 1

        print(f"\nSuccess from all starts: {success_count}/{len(self.network.nodes)}")
        self.assertEqual(success_count, len(self.network.nodes))


class TestRewiring(unittest.TestCase):
    """Tests for network rewiring."""

    def test_rewiring_changes_edges(self):
        """Test that rewiring modifies the network."""
        np.random.seed(42)
        network = SmallWorldProper(k_local=5, k_long=2, rewire_prob=0.3)

        for i in range(30):
            vec = np.random.randn(10)
            network.add_node(f"n{i}", vec)

        # Get initial edges
        initial_edges = set()
        for nid, node in network.nodes.items():
            for neighbor in node.neighbors:
                edge = tuple(sorted([nid, neighbor]))
                initial_edges.add(edge)

        # Rewire
        rewired = network.rewire_random()

        # Get final edges
        final_edges = set()
        for nid, node in network.nodes.items():
            for neighbor in node.neighbors:
                edge = tuple(sorted([nid, neighbor]))
                final_edges.add(edge)

        print(f"\nRewired: {rewired} edges")
        print(f"Edges changed: {len(initial_edges - final_edges)}")

        self.assertGreater(rewired, 0)


class TestScaling(unittest.TestCase):
    """Test scaling behavior."""

    def test_comparisons_grow_slowly(self):
        """Test that comparisons don't grow linearly with n."""
        results = []

        for num_nodes in [50, 100, 200]:
            np.random.seed(42)
            random.seed(42)

            network = SmallWorldProper(k_local=10, k_long=5)

            for i in range(num_nodes):
                vec = np.random.randn(16)
                vec = vec / np.linalg.norm(vec)
                network.add_node(f"n{i}", vec)

            # Run queries
            total_comps = 0
            for _ in range(20):
                query = np.random.randn(16)
                query = query / np.linalg.norm(query)
                _, comps = network.search_knn(query, k=5, ef=30)
                total_comps += comps

            avg_comps = total_comps / 20
            results.append((num_nodes, avg_comps))

        print("\n" + "=" * 50)
        print("SCALING TEST")
        print("=" * 50)
        print(f"{'Nodes':>8} {'Avg Comps':>12} {'Ratio':>10}")
        print("-" * 30)

        for n, comps in results:
            ratio = comps / results[0][1]
            print(f"{n:>8} {comps:>12.1f} {ratio:>10.2f}x")

        # Comparisons should grow sub-linearly
        # 4x nodes should NOT mean 4x comparisons
        growth = results[-1][1] / results[0][1]
        node_growth = results[-1][0] / results[0][0]

        print(f"\nNodes grew {node_growth:.0f}x, comps grew {growth:.1f}x")
        self.assertLess(growth, node_growth)


class TestOptimizedLookup(unittest.TestCase):
    """Tests for angle-based optimized neighbor lookup."""

    def setUp(self):
        """Create test network with sorted neighbors."""
        np.random.seed(42)
        random.seed(42)

        self.network = SmallWorldProper(k_local=10, k_long=5)

        # Create network with clear angular structure
        self.dim = 10
        for i in range(50):
            # Create vectors with varied angular positions
            angle = (i / 50) * 2 * np.pi  # Spread around the unit circle
            vec = np.zeros(self.dim)
            vec[0] = np.cos(angle)
            vec[1] = np.sin(angle)
            # Add some noise in other dimensions
            vec[2:] = np.random.randn(self.dim - 2) * 0.1
            vec = vec / np.linalg.norm(vec)
            self.network.add_node(f"n{i}", vec)

    def test_sorted_neighbors_maintained(self):
        """Test that sorted neighbor list is maintained during insertion."""
        for node_id, node in self.network.nodes.items():
            # Check sorted list has same count as neighbors set
            self.assertEqual(
                len(node.sorted_neighbors),
                len(node.neighbors),
                f"Node {node_id}: sorted list size mismatch"
            )

            # Check sorted order
            angles = [a for a, _ in node.sorted_neighbors]
            self.assertEqual(
                angles,
                sorted(angles),
                f"Node {node_id}: neighbors not sorted by angle"
            )

    def test_angle_lookup_finds_candidates(self):
        """Test that angle-based lookup returns nearby neighbors."""
        node = self.network.nodes["n25"]

        # Query in a specific direction
        query = np.zeros(self.dim)
        query[0] = 1.0  # Direction along x-axis
        query = query / np.linalg.norm(query)

        candidates = node.lookup_neighbors_by_angle(query, window_size=3)

        # Should return some candidates (up to 2*window_size + 1)
        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), 7)  # 2*3 + 1

        print(f"\nAngle lookup returned {len(candidates)} candidates")

    def test_optimized_search_finds_target(self):
        """Test that optimized search finds targets correctly."""
        target_id = "n30"
        target = self.network.nodes[target_id]
        query = target.vector.copy()

        results, comps = self.network.search_knn_optimized(
            query, k=5, ef=30, window_size=5
        )

        # Target should be in results
        result_ids = [r[0] for r in results]
        self.assertIn(target_id, result_ids)

        print(f"\nOptimized search found target with {comps} comparisons")

    def test_optimized_search_comparison_reduction(self):
        """Test that optimized search uses fewer comparisons."""
        np.random.seed(123)

        # Run multiple queries with both methods
        standard_comps = 0
        optimized_comps = 0
        num_queries = 20

        for _ in range(num_queries):
            query = np.random.randn(self.dim)
            query = query / np.linalg.norm(query)

            _, std_c = self.network.search_knn(query, k=5, ef=30)
            _, opt_c = self.network.search_knn_optimized(
                query, k=5, ef=30, window_size=5
            )

            standard_comps += std_c
            optimized_comps += opt_c

        avg_std = standard_comps / num_queries
        avg_opt = optimized_comps / num_queries

        print(f"\n" + "=" * 50)
        print("OPTIMIZED LOOKUP COMPARISON")
        print("=" * 50)
        print(f"Standard search: {avg_std:.1f} avg comparisons")
        print(f"Optimized search: {avg_opt:.1f} avg comparisons")
        print(f"Reduction: {(1 - avg_opt/avg_std)*100:.1f}%")

        # Optimized should use fewer or equal comparisons
        # (may be equal if ef is large relative to neighbors)
        self.assertLessEqual(avg_opt, avg_std * 1.1)  # Allow 10% margin

    def test_optimized_search_quality(self):
        """Test that optimized search maintains result quality."""
        np.random.seed(456)

        matching_results = 0
        total_queries = 20

        for _ in range(total_queries):
            # Query near a specific node
            target = random.choice(list(self.network.nodes.values()))
            query = target.vector + np.random.randn(self.dim) * 0.05
            query = query / np.linalg.norm(query)

            std_results, _ = self.network.search_knn(query, k=5, ef=30)
            opt_results, _ = self.network.search_knn_optimized(
                query, k=5, ef=30, window_size=5
            )

            # Check if top result matches
            if std_results and opt_results:
                if std_results[0][0] == opt_results[0][0]:
                    matching_results += 1

        match_rate = matching_results / total_queries

        print(f"\nTop-1 match rate: {match_rate*100:.1f}%")

        # Should match most of the time (allow some variation due to
        # different exploration paths)
        self.assertGreaterEqual(match_rate, 0.7)

    def test_rebuild_sorted_neighbors(self):
        """Test rebuilding sorted neighbor lists."""
        # Rebuild all
        count = self.network.rebuild_all_sorted_neighbors()

        self.assertEqual(count, len(self.network.nodes))

        # Verify still sorted
        for node in self.network.nodes.values():
            angles = [a for a, _ in node.sorted_neighbors]
            self.assertEqual(angles, sorted(angles))


def run_demo():
    """Demonstrate proper small-world network."""
    print("=" * 70)
    print("PROPER SMALL-WORLD NETWORK DEMO")
    print("=" * 70)

    np.random.seed(42)
    random.seed(42)

    # Build network
    network = SmallWorldProper(k_local=10, k_long=5, max_neighbors=20)

    print("\nBuilding network with 100 nodes...")
    for i in range(100):
        vec = np.random.randn(16)
        vec = vec / np.linalg.norm(vec)
        network.add_node(f"n{i}", vec)

    stats = network.get_statistics()
    print(f"\nNetwork statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['total_edges']}")
    print(f"  Avg neighbors: {stats['avg_neighbors']:.1f}")
    print(f"  Clustering coefficient: {stats['clustering_coefficient']:.3f}")
    print(f"  Avg path length: {stats['estimated_avg_path_length']:.2f}")

    # Routing test
    print("\nRouting test (50 random queries)...")
    total_comps = 0
    success = 0

    for _ in range(50):
        target = random.choice(list(network.nodes.values()))
        query = target.vector + np.random.randn(16) * 0.05

        results, comps = network.search_knn(query, k=5, ef=30)
        total_comps += comps

        if target.node_id in [r[0] for r in results]:
            success += 1

    print(f"  Success rate: {success}/50 ({success*2}%)")
    print(f"  Avg comparisons: {total_comps/50:.1f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_demo()
    print("\n\nRunning unit tests...")
    unittest.main(verbosity=2)
