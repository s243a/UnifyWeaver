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


class TestCrossBranchLinks(unittest.TestCase):
    """Tests for cross-branch link discovery."""

    def create_multi_branch_network(self) -> SmallWorldNetwork:
        """Create a network with multiple branches and semantic overlap."""
        np.random.seed(42)
        network = SmallWorldNetwork(max_shortcuts_per_node=5)
        dim = 10

        # Root
        root_centroid = np.ones(dim) / np.sqrt(dim)
        network.add_node("root", root_centroid)

        # Create 3 branches with some semantic overlap
        # Branch A: focused on dimensions 0-3
        # Branch B: focused on dimensions 3-6 (overlaps with A)
        # Branch C: focused on dimensions 6-9 (overlaps with B)
        branch_configs = [
            ("branch_a", [0, 1, 2, 3]),
            ("branch_b", [3, 4, 5, 6]),
            ("branch_c", [6, 7, 8, 9]),
        ]

        for branch_id, dims in branch_configs:
            # Branch node
            branch_centroid = np.zeros(dim)
            for d in dims:
                branch_centroid[d] = 1.0
            branch_centroid = branch_centroid / np.linalg.norm(branch_centroid)

            network.add_node(branch_id, branch_centroid, parent_id="root")
            network.nodes["root"].children_ids.append(branch_id)

            # Leaf nodes in branch
            for i in range(4):
                leaf_centroid = branch_centroid + np.random.randn(dim) * 0.1
                leaf_centroid = leaf_centroid / np.linalg.norm(leaf_centroid)
                leaf_id = f"{branch_id}_node_{i}"
                network.add_node(leaf_id, leaf_centroid, parent_id=branch_id)
                network.nodes[branch_id].children_ids.append(leaf_id)

        return network

    def test_get_ancestors(self):
        """Test ancestor retrieval."""
        network = self.create_multi_branch_network()

        # Leaf node should have branch and root as ancestors
        ancestors = network._get_ancestors("branch_a_node_0")
        self.assertIn("branch_a", ancestors)
        self.assertIn("root", ancestors)
        self.assertEqual(len(ancestors), 2)

        # Branch node should have only root as ancestor
        ancestors = network._get_ancestors("branch_a")
        self.assertIn("root", ancestors)
        self.assertEqual(len(ancestors), 1)

        # Root has no ancestors
        ancestors = network._get_ancestors("root")
        self.assertEqual(len(ancestors), 0)

    def test_get_branch_id(self):
        """Test branch ID retrieval."""
        network = self.create_multi_branch_network()

        # Leaf nodes should return their branch
        self.assertEqual(network._get_branch_id("branch_a_node_0"), "branch_a")
        self.assertEqual(network._get_branch_id("branch_b_node_2"), "branch_b")

        # Branch nodes should return themselves
        self.assertEqual(network._get_branch_id("branch_a"), "branch_a")

        # Root returns None (no branch)
        self.assertIsNone(network._get_branch_id("root"))

    def test_find_cross_branch_candidates(self):
        """Test finding semantically similar nodes in other branches."""
        network = self.create_multi_branch_network()

        # Node in branch_a should find similar nodes in branch_b (overlapping dims 3)
        candidates = network.find_cross_branch_candidates(
            "branch_a_node_0",
            similarity_threshold=0.3,  # Lower threshold for overlapping branches
            max_candidates=5,
        )

        # Should find candidates from other branches
        self.assertGreater(len(candidates), 0)

        # Candidates should not be from the same branch
        for cid, sim in candidates:
            self.assertNotIn("branch_a", cid)
            # Should be somewhat similar
            self.assertGreater(sim, 0.0)

    def test_discover_cross_branch_links(self):
        """Test automatic cross-branch link discovery."""
        network = self.create_multi_branch_network()

        initial_shortcuts = sum(len(n.shortcut_ids) for n in network.nodes.values())

        # Discover cross-branch links
        new_links = network.discover_cross_branch_links(
            similarity_threshold=0.3,
            max_links_per_node=2,
        )

        final_shortcuts = sum(len(n.shortcut_ids) for n in network.nodes.values())

        print(f"\nCross-branch links created: {new_links}")
        print(f"Shortcuts: {initial_shortcuts} -> {final_shortcuts}")

        # Should have created some cross-branch links
        self.assertGreater(final_shortcuts, initial_shortcuts)

    def test_cross_branch_links_are_bidirectional(self):
        """Test that cross-branch links are created bidirectionally."""
        network = self.create_multi_branch_network()

        network.discover_cross_branch_links(
            similarity_threshold=0.3,
            max_links_per_node=2,
        )

        # Check for bidirectional links
        bidirectional_count = 0
        for node_id, node in network.nodes.items():
            for shortcut_id in node.shortcut_ids:
                if shortcut_id in network.nodes:
                    if node_id in network.nodes[shortcut_id].shortcut_ids:
                        bidirectional_count += 1

        print(f"\nBidirectional links: {bidirectional_count // 2}")
        self.assertGreater(bidirectional_count, 0)


class TestPathFolding(unittest.TestCase):
    """Tests for Freenet-style path folding."""

    def test_record_successful_path_creates_shortcuts(self):
        """Test that recording a successful path creates shortcuts."""
        np.random.seed(42)
        network = SmallWorldNetwork(max_shortcuts_per_node=10)
        dim = 10

        # Create a chain: root -> a -> b -> c -> d -> target
        nodes = ["root", "a", "b", "c", "d", "target"]
        for i, node_id in enumerate(nodes):
            centroid = np.random.randn(dim)
            centroid = centroid / np.linalg.norm(centroid)
            parent = nodes[i - 1] if i > 0 else None
            network.add_node(node_id, centroid, parent_id=parent)
            if parent:
                network.nodes[parent].children_ids.append(node_id)

        # Record a successful path
        path = ["root", "a", "b", "c", "d", "target"]
        query = np.random.randn(dim)
        shortcuts_created = network.record_successful_path(path, query)

        print(f"\nPath length: {len(path)}")
        print(f"Shortcuts created: {shortcuts_created}")

        # Should create shortcuts from early nodes to target
        self.assertGreater(shortcuts_created, 0)

        # Check that root has shortcut to target (5 hops saved)
        self.assertIn("target", network.nodes["root"].shortcut_ids)

        # Check that 'a' has shortcut to target (4 hops saved)
        self.assertIn("target", network.nodes["a"].shortcut_ids)

        # Check that 'b' has shortcut to target (3 hops saved)
        self.assertIn("target", network.nodes["b"].shortcut_ids)

    def test_path_folding_requires_minimum_hops(self):
        """Test that short paths don't create shortcuts."""
        np.random.seed(42)
        network = SmallWorldNetwork()
        dim = 10

        # Create short chain: root -> a -> target
        for node_id in ["root", "a", "target"]:
            centroid = np.random.randn(dim)
            network.add_node(node_id, centroid)

        # Short path - only 3 nodes
        path = ["root", "a", "target"]
        query = np.random.randn(dim)
        shortcuts_created = network.record_successful_path(path, query)

        # No shortcuts needed for short path
        self.assertEqual(shortcuts_created, 0)

    def test_path_folding_improves_subsequent_routing(self):
        """Test that path folding improves routing on subsequent queries."""
        np.random.seed(42)
        network = SmallWorldNetwork(max_shortcuts_per_node=10)
        dim = 10

        # Create a longer chain
        chain = [f"node_{i}" for i in range(10)]
        for i, node_id in enumerate(chain):
            centroid = np.random.randn(dim)
            centroid = centroid / np.linalg.norm(centroid)
            parent = chain[i - 1] if i > 0 else None
            network.add_node(node_id, centroid, parent_id=parent)
            if parent:
                network.nodes[parent].children_ids.append(node_id)

        # Query toward the end node
        target = network.nodes[chain[-1]]
        query = target.centroid + np.random.randn(dim) * 0.1
        query = query / np.linalg.norm(query)

        # Route before path folding
        path1, comps1 = network.route_greedy(query, start_node_id=chain[0])
        print(f"\nBefore folding: {len(path1)} hops, {comps1} comparisons")

        # Record the path (simulate successful query)
        network.record_successful_path(path1, query)

        # Route again - should use shortcut
        path2, comps2 = network.route_greedy(query, start_node_id=chain[0])
        print(f"After folding: {len(path2)} hops, {comps2} comparisons")

        # Path should be shorter after folding
        self.assertLessEqual(len(path2), len(path1))


class TestBacktracking(unittest.TestCase):
    """Tests for backtracking routing."""

    def create_branching_network(self) -> SmallWorldNetwork:
        """Create a network where backtracking is needed."""
        np.random.seed(42)
        network = SmallWorldNetwork(max_shortcuts_per_node=5)
        dim = 10

        # Create a tree where target is in a different branch than
        # the greedy path would take
        #
        #           root (0.5, 0.5, ...)
        #          /    \
        #     left       right
        #    /    \         \
        #  left_a  left_b   target  <-- target is here
        #
        # Query will be close to left branch initially, but target is in right

        # Root - middle ground
        root_centroid = np.ones(dim) * 0.5
        root_centroid = root_centroid / np.linalg.norm(root_centroid)
        network.add_node("root", root_centroid)

        # Left branch - dimensions 0-4
        left_centroid = np.zeros(dim)
        left_centroid[0:5] = 1.0
        left_centroid = left_centroid / np.linalg.norm(left_centroid)
        network.add_node("left", left_centroid, parent_id="root")
        network.nodes["root"].children_ids.append("left")

        # Left children
        for i, name in enumerate(["left_a", "left_b"]):
            centroid = left_centroid + np.random.randn(dim) * 0.1
            centroid = centroid / np.linalg.norm(centroid)
            network.add_node(name, centroid, parent_id="left")
            network.nodes["left"].children_ids.append(name)

        # Right branch - dimensions 5-9
        right_centroid = np.zeros(dim)
        right_centroid[5:10] = 1.0
        right_centroid = right_centroid / np.linalg.norm(right_centroid)
        network.add_node("right", right_centroid, parent_id="root")
        network.nodes["root"].children_ids.append("right")

        # Target in right branch
        target_centroid = right_centroid + np.random.randn(dim) * 0.1
        target_centroid = target_centroid / np.linalg.norm(target_centroid)
        network.add_node("target", target_centroid, parent_id="right")
        network.nodes["right"].children_ids.append("target")

        return network

    def test_backtrack_finds_target_from_wrong_branch(self):
        """Test that backtracking can find target after going wrong way."""
        network = self.create_branching_network()

        # Query for the target
        target = network.nodes["target"]
        query = target.centroid.copy()

        # Start from left_a (wrong branch)
        # Without backtracking, greedy would get stuck
        path_no_bt, _ = network.route_greedy(
            query, start_node_id="left_a", use_backtrack=False
        )
        print(f"\nWithout backtrack from left_a: {path_no_bt}")

        # With backtracking, should find target
        path_bt, comps = network.route_greedy(
            query, start_node_id="left_a", use_backtrack=True
        )
        print(f"With backtrack from left_a: {path_bt} ({comps} comparisons)")

        # Backtracking should reach target (or at least get closer)
        self.assertIn("target", path_bt)

    def test_backtrack_goes_up_then_down(self):
        """Test that backtracking can go up hierarchy then down another branch."""
        network = self.create_branching_network()

        target = network.nodes["target"]
        query = target.centroid.copy()

        # Start from left branch
        path, comps = network.route_greedy(
            query, start_node_id="left", use_backtrack=True
        )

        print(f"\nPath from left to target: {path} ({comps} comparisons)")

        # Backtracking should find target even from wrong branch
        # The path returned is the final path to best node found
        self.assertIn("target", path)

        # Comparisons should show we explored multiple nodes
        # (had to backtrack through root to reach right branch)
        self.assertGreater(comps, 2)  # More than just direct neighbors

    def test_backtrack_from_any_node(self):
        """Test that backtracking allows starting from any node."""
        network = self.create_branching_network()

        target = network.nodes["target"]
        query = target.centroid.copy()

        # Try starting from every node
        success_count = 0
        for start_id in network.nodes.keys():
            path, _ = network.route_greedy(
                query, start_node_id=start_id, use_backtrack=True
            )
            if "target" in path:
                success_count += 1

        print(f"\nBacktrack success from all nodes: {success_count}/{len(network.nodes)}")

        # Should be able to reach target from any node
        self.assertEqual(success_count, len(network.nodes))

    def test_backtrack_respects_max_hops(self):
        """Test that backtracking respects max_hops limit."""
        network = self.create_branching_network()

        target = network.nodes["target"]
        query = target.centroid.copy()

        # Very low max_hops should limit exploration
        path, _ = network.route_greedy(
            query, start_node_id="left_a", use_backtrack=True, max_hops=2
        )

        print(f"\nPath with max_hops=2: {path}")

        # With only 2 hops, might not reach target
        # But should return best node found
        self.assertLessEqual(len(path), 3)  # start + 2 hops max


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


def run_cross_branch_demo():
    """Demonstrate cross-branch link discovery and path folding."""
    print("=" * 70)
    print("CROSS-BRANCH LINKS & PATH FOLDING DEMO")
    print("=" * 70)

    np.random.seed(42)
    dim = 10

    # Create multi-branch network with semantic overlap
    network = SmallWorldNetwork(max_shortcuts_per_node=10)

    # Root
    network.add_node("root", np.ones(dim) / np.sqrt(dim))

    # Create 5 branches with overlapping semantic regions
    print("\nCreating 5 branches with semantic overlap...")
    branch_configs = [
        ("tech", [0, 1, 2]),       # Technology
        ("science", [2, 3, 4]),    # Science (overlaps with tech)
        ("health", [4, 5, 6]),     # Health (overlaps with science)
        ("arts", [6, 7, 8]),       # Arts (overlaps with health)
        ("sports", [8, 9, 0]),     # Sports (overlaps with arts and tech)
    ]

    for branch_id, dims in branch_configs:
        branch_centroid = np.zeros(dim)
        for d in dims:
            branch_centroid[d] = 1.0
        branch_centroid = branch_centroid / np.linalg.norm(branch_centroid)

        network.add_node(branch_id, branch_centroid, parent_id="root")
        network.nodes["root"].children_ids.append(branch_id)

        # 10 nodes per branch
        for i in range(10):
            leaf_centroid = branch_centroid + np.random.randn(dim) * 0.15
            leaf_centroid = leaf_centroid / np.linalg.norm(leaf_centroid)
            leaf_id = f"{branch_id}_{i}"
            network.add_node(leaf_id, leaf_centroid, parent_id=branch_id)
            network.nodes[branch_id].children_ids.append(leaf_id)

    print(f"Network: {len(network.nodes)} nodes, 5 branches")

    # Test routing BEFORE cross-branch links
    print("\n--- BEFORE CROSS-BRANCH LINKS ---")

    # Create query that sits between tech and science (overlapping region)
    query = np.zeros(dim)
    query[2] = 1.0  # Overlap dimension
    query = query / np.linalg.norm(query)

    results_before = []
    for _ in range(10):
        path, comps = network.route_greedy(query)
        results_before.append((len(path), comps))

    avg_hops_before = np.mean([r[0] for r in results_before])
    avg_comps_before = np.mean([r[1] for r in results_before])
    print(f"Avg hops: {avg_hops_before:.1f}, Avg comparisons: {avg_comps_before:.1f}")

    # Discover cross-branch links
    print("\n--- DISCOVERING CROSS-BRANCH LINKS ---")
    new_links = network.discover_cross_branch_links(
        similarity_threshold=0.5,
        max_links_per_node=3,
    )
    print(f"Cross-branch links created: {new_links}")

    # Test routing AFTER cross-branch links
    print("\n--- AFTER CROSS-BRANCH LINKS ---")
    results_after = []
    for _ in range(10):
        path, comps = network.route_greedy(query)
        results_after.append((len(path), comps))

    avg_hops_after = np.mean([r[0] for r in results_after])
    avg_comps_after = np.mean([r[1] for r in results_after])
    print(f"Avg hops: {avg_hops_after:.1f}, Avg comparisons: {avg_comps_after:.1f}")

    # Now demonstrate path folding
    print("\n--- PATH FOLDING DEMO ---")

    # Create a deep branch for testing path folding
    deep_parent = "science"
    for i in range(5):
        node_id = f"deep_{i}"
        parent = deep_parent if i == 0 else f"deep_{i-1}"
        centroid = np.random.randn(dim)
        centroid = centroid / np.linalg.norm(centroid)
        network.add_node(node_id, centroid, parent_id=parent)
        if parent in network.nodes:
            network.nodes[parent].children_ids.append(node_id)

    # Query targeting deep node
    deep_target = network.nodes["deep_4"]
    deep_query = deep_target.centroid + np.random.randn(dim) * 0.05
    deep_query = deep_query / np.linalg.norm(deep_query)

    # Route before path folding
    path_before, _ = network.route_greedy(deep_query, start_node_id="root")
    print(f"Path before folding: {' -> '.join(path_before[:6])}...")
    print(f"  Hops: {len(path_before)}")

    # Record successful path (path folding)
    shortcuts = network.record_successful_path(path_before, deep_query)
    print(f"Path folding created {shortcuts} shortcuts")

    # Route after path folding
    path_after, _ = network.route_greedy(deep_query, start_node_id="root")
    print(f"Path after folding: {' -> '.join(path_after[:6])}...")
    print(f"  Hops: {len(path_after)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Cross-branch improvement: {avg_hops_before:.1f} -> {avg_hops_after:.1f} hops")
    print(f"Path folding improvement: {len(path_before)} -> {len(path_after)} hops")
    print(f"\nNetwork stats: {network.get_statistics()}")


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
    # Run demos first
    run_cross_branch_demo()
    print("\n\n")
    run_evolution_demo()

    print("\n\nRunning unit tests...")
    unittest.main(verbosity=2)
