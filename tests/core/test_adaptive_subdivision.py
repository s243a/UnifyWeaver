# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""Unit tests for adaptive node subdivision."""

import unittest
import sys
import os

# Add source directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

try:
    from adaptive_subdivision import (
        SplitConfig,
        NodeMetrics,
        NodeType,
        SubdividableNode,
        should_split,
        kmeans_split,
        split_node,
        should_merge,
        merge_nodes,
        SubdivisionRegistry,
    )
except ImportError:
    from unifyweaver.targets.python_runtime.adaptive_subdivision import (
        SplitConfig,
        NodeMetrics,
        NodeType,
        SubdividableNode,
        should_split,
        kmeans_split,
        split_node,
        should_merge,
        merge_nodes,
        SubdivisionRegistry,
    )


class TestSplitConfig(unittest.TestCase):
    """Tests for SplitConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SplitConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.max_documents, 1000)
        self.assertEqual(config.max_variance, 0.5)
        self.assertEqual(config.max_latency_p99_ms, 500.0)
        self.assertEqual(config.split_method, "kmeans")

    def test_custom_config(self):
        """Test custom configuration."""
        config = SplitConfig(
            max_documents=500,
            max_variance=0.3,
            split_method="random",
        )
        self.assertEqual(config.max_documents, 500)
        self.assertEqual(config.max_variance, 0.3)
        self.assertEqual(config.split_method, "random")

    def test_to_dict(self):
        """Test serialization."""
        config = SplitConfig()
        d = config.to_dict()
        self.assertIn("enabled", d)
        self.assertIn("max_documents", d)
        self.assertIn("split_method", d)


class TestNodeMetrics(unittest.TestCase):
    """Tests for NodeMetrics class."""

    def test_record_query(self):
        """Test recording queries."""
        metrics = NodeMetrics()
        metrics.record_query(100.0)
        metrics.record_query(150.0)
        metrics.record_query(200.0)

        self.assertEqual(len(metrics.latencies_ms), 3)

    def test_latency_p99(self):
        """Test P99 latency calculation."""
        metrics = NodeMetrics()
        # Add 100 latencies from 1 to 100
        for i in range(1, 101):
            metrics.record_query(float(i))

        p99 = metrics.get_latency_p99()
        self.assertGreaterEqual(p99, 99.0)

    def test_latency_p99_insufficient_data(self):
        """Test P99 with insufficient data."""
        metrics = NodeMetrics()
        metrics.record_query(100.0)
        self.assertEqual(metrics.get_latency_p99(), 0.0)

    def test_to_dict(self):
        """Test serialization."""
        metrics = NodeMetrics(document_count=100, centroid_variance=0.3)
        d = metrics.to_dict()
        self.assertEqual(d["document_count"], 100)
        self.assertEqual(d["centroid_variance"], 0.3)


class TestSubdividableNode(unittest.TestCase):
    """Tests for SubdividableNode class."""

    def test_create_leaf_node(self):
        """Test creating a LEAF node."""
        node = SubdividableNode(
            node_id="node_1",
            node_type=NodeType.LEAF,
        )
        self.assertEqual(node.node_id, "node_1")
        self.assertEqual(node.node_type, NodeType.LEAF)
        self.assertEqual(len(node.document_ids), 0)

    def test_add_document(self):
        """Test adding documents."""
        node = SubdividableNode(node_id="node_1")
        embedding = np.array([0.1, 0.2, 0.3])

        node.add_document("doc_1", embedding)

        self.assertEqual(len(node.document_ids), 1)
        self.assertEqual(node.metrics.document_count, 1)
        self.assertIsNotNone(node.centroid)

    def test_add_document_updates_variance(self):
        """Test variance is computed when adding documents."""
        node = SubdividableNode(node_id="node_1")

        # Add similar embeddings
        node.add_document("doc_1", np.array([0.1, 0.2, 0.3]))
        node.add_document("doc_2", np.array([0.15, 0.25, 0.35]))

        self.assertGreater(node.metrics.centroid_variance, 0)

    def test_cannot_add_to_region(self):
        """Test adding to REGION node fails."""
        node = SubdividableNode(
            node_id="node_1",
            node_type=NodeType.REGION,
        )
        with self.assertRaises(ValueError):
            node.add_document("doc_1", np.array([0.1, 0.2]))

    def test_to_kg_node(self):
        """Test conversion to KGNode."""
        node = SubdividableNode(
            node_id="node_1",
            endpoint="http://localhost:8080",
            centroid=np.array([0.1, 0.2, 0.3]),
            topics=["csv", "json"],
            embedding_model="test-model",
        )
        kg_node = node.to_kg_node()

        self.assertEqual(kg_node.node_id, "node_1")
        self.assertEqual(kg_node.endpoint, "http://localhost:8080")
        self.assertEqual(kg_node.topics, ["csv", "json"])


class TestShouldSplit(unittest.TestCase):
    """Tests for should_split function."""

    def test_split_disabled(self):
        """Test split is skipped when disabled."""
        node = SubdividableNode(
            node_id="node_1",
            config=SplitConfig(enabled=False),
        )
        should, reason = should_split(node)
        self.assertFalse(should)
        self.assertIn("disabled", reason)

    def test_split_region_node(self):
        """Test REGION nodes can't split."""
        node = SubdividableNode(
            node_id="node_1",
            node_type=NodeType.REGION,
        )
        should, reason = should_split(node)
        self.assertFalse(should)
        self.assertIn("LEAF", reason)

    def test_split_too_few_documents(self):
        """Test split is skipped if too few documents."""
        node = SubdividableNode(
            node_id="node_1",
            config=SplitConfig(min_child_documents=100),
        )
        # Add only 50 documents
        for i in range(50):
            node.add_document(f"doc_{i}", np.random.randn(10))

        should, reason = should_split(node)
        self.assertFalse(should)
        self.assertIn("too few", reason)

    def test_split_on_document_count(self):
        """Test split triggers on document count."""
        node = SubdividableNode(
            node_id="node_1",
            config=SplitConfig(max_documents=100, min_child_documents=10),
        )
        # Add 200 documents
        for i in range(200):
            node.add_document(f"doc_{i}", np.random.randn(10))

        should, reason = should_split(node)
        self.assertTrue(should)
        self.assertIn("document count", reason)

    def test_split_on_variance(self):
        """Test split triggers on variance."""
        node = SubdividableNode(
            node_id="node_1",
            config=SplitConfig(max_variance=0.1, min_child_documents=10),
        )
        # Add diverse embeddings
        for i in range(100):
            # High variance embeddings
            emb = np.zeros(10)
            emb[i % 10] = 1.0  # One-hot style
            node.add_document(f"doc_{i}", emb)

        should, reason = should_split(node)
        self.assertTrue(should)
        self.assertIn("variance", reason)


class TestKmeansSplit(unittest.TestCase):
    """Tests for kmeans_split function."""

    def test_split_two_clusters(self):
        """Test k-means with two clear clusters."""
        # Create two clear clusters
        cluster_a = [np.array([1.0, 0.0]) for _ in range(10)]
        cluster_b = [np.array([0.0, 1.0]) for _ in range(10)]
        embeddings = cluster_a + cluster_b

        labels = kmeans_split(embeddings, k=2, seed=42)

        # Check we got two groups
        self.assertEqual(len(labels), 20)
        unique_labels = set(labels)
        self.assertEqual(len(unique_labels), 2)

        # Check clusters are mostly together
        labels_a = labels[:10]
        labels_b = labels[10:]
        # Most of cluster A should have same label
        self.assertGreater(labels_a.count(labels_a[0]), 8)
        # Most of cluster B should have same label
        self.assertGreater(labels_b.count(labels_b[0]), 8)

    def test_split_single_point(self):
        """Test k-means with single point."""
        embeddings = [np.array([1.0, 0.0])]
        labels = kmeans_split(embeddings, k=2)
        self.assertEqual(len(labels), 1)

    def test_split_reproducible(self):
        """Test k-means is reproducible with seed."""
        embeddings = [np.random.randn(10) for _ in range(50)]

        labels1 = kmeans_split(embeddings, k=2, seed=42)
        labels2 = kmeans_split(embeddings, k=2, seed=42)

        self.assertEqual(labels1, labels2)


class TestSplitNode(unittest.TestCase):
    """Tests for split_node function."""

    def test_split_creates_children(self):
        """Test split creates two child nodes."""
        node = SubdividableNode(node_id="parent")
        for i in range(100):
            node.add_document(f"doc_{i}", np.random.randn(10))

        child_a, child_b = split_node(node, seed=42)

        self.assertEqual(child_a.node_id, "parent_a")
        self.assertEqual(child_b.node_id, "parent_b")
        self.assertEqual(child_a.parent_id, "parent")
        self.assertEqual(child_b.parent_id, "parent")

    def test_split_converts_parent_to_region(self):
        """Test parent becomes REGION after split."""
        node = SubdividableNode(node_id="parent")
        for i in range(100):
            node.add_document(f"doc_{i}", np.random.randn(10))

        split_node(node, seed=42)

        self.assertEqual(node.node_type, NodeType.REGION)
        self.assertEqual(len(node.children_ids), 2)
        self.assertEqual(len(node.document_ids), 0)

    def test_split_preserves_documents(self):
        """Test all documents are preserved in children."""
        node = SubdividableNode(node_id="parent")
        for i in range(100):
            node.add_document(f"doc_{i}", np.random.randn(10))

        child_a, child_b = split_node(node, seed=42)

        total_docs = len(child_a.document_ids) + len(child_b.document_ids)
        self.assertEqual(total_docs, 100)

    def test_split_region_fails(self):
        """Test splitting REGION node fails."""
        node = SubdividableNode(
            node_id="region",
            node_type=NodeType.REGION,
        )
        with self.assertRaises(ValueError):
            split_node(node)


class TestSubdivisionRegistry(unittest.TestCase):
    """Tests for SubdivisionRegistry class."""

    def test_register_and_get(self):
        """Test registering and retrieving nodes."""
        registry = SubdivisionRegistry()
        node = SubdividableNode(node_id="node_1")

        registry.register(node)
        retrieved = registry.get("node_1")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.node_id, "node_1")

    def test_get_leaf_nodes(self):
        """Test getting all LEAF nodes."""
        registry = SubdivisionRegistry()
        registry.register(SubdividableNode(node_id="leaf_1", node_type=NodeType.LEAF))
        registry.register(SubdividableNode(node_id="leaf_2", node_type=NodeType.LEAF))
        registry.register(SubdividableNode(node_id="region_1", node_type=NodeType.REGION))

        leaves = registry.get_leaf_nodes()

        self.assertEqual(len(leaves), 2)
        self.assertTrue(all(n.node_type == NodeType.LEAF for n in leaves))

    def test_get_children(self):
        """Test getting children of region node."""
        registry = SubdivisionRegistry()

        parent = SubdividableNode(
            node_id="parent",
            node_type=NodeType.REGION,
            children_ids=["child_a", "child_b"],
        )
        child_a = SubdividableNode(node_id="child_a", parent_id="parent")
        child_b = SubdividableNode(node_id="child_b", parent_id="parent")

        registry.register(parent)
        registry.register(child_a)
        registry.register(child_b)

        children = registry.get_children("parent")

        self.assertEqual(len(children), 2)
        self.assertIn(child_a, children)
        self.assertIn(child_b, children)

    def test_route_to_leaf(self):
        """Test routing query to best leaf."""
        registry = SubdivisionRegistry()

        # Create hierarchy: region -> (leaf_a, leaf_b)
        region = SubdividableNode(
            node_id="region",
            node_type=NodeType.REGION,
            centroid=np.array([0.5, 0.5]),
            children_ids=["leaf_a", "leaf_b"],
        )
        leaf_a = SubdividableNode(
            node_id="leaf_a",
            centroid=np.array([1.0, 0.0]),  # Close to query
            parent_id="region",
        )
        leaf_b = SubdividableNode(
            node_id="leaf_b",
            centroid=np.array([0.0, 1.0]),  # Far from query
            parent_id="region",
        )

        registry.register(region)
        registry.register(leaf_a)
        registry.register(leaf_b)

        # Query similar to leaf_a
        query = np.array([0.9, 0.1])
        result = registry.route_to_leaf(query, "region")

        self.assertIsNotNone(result)
        self.assertEqual(result.node_id, "leaf_a")

    def test_check_and_split_all(self):
        """Test automatic splitting of nodes."""
        registry = SubdivisionRegistry()

        # Create node that exceeds threshold
        node = SubdividableNode(
            node_id="big_node",
            config=SplitConfig(max_documents=50, min_child_documents=10),
        )
        for i in range(100):
            node.add_document(f"doc_{i}", np.random.randn(10))

        registry.register(node)

        split_ids = registry.check_and_split_all(seed=42)

        self.assertEqual(split_ids, ["big_node"])
        self.assertEqual(node.node_type, NodeType.REGION)
        self.assertEqual(len(registry.get_leaf_nodes()), 2)


class TestHierarchyIntegration(unittest.TestCase):
    """Tests for HierarchicalFederatedEngine integration."""

    def test_to_regional_node(self):
        """Test REGION node converts to regional dict."""
        node = SubdividableNode(
            node_id="region_1",
            node_type=NodeType.REGION,
            centroid=np.array([0.5, 0.5]),
            topics=["topic_a", "topic_b"],
            children_ids=["child_1", "child_2"],
        )
        regional = node.to_regional_node()

        self.assertIsNotNone(regional)
        self.assertEqual(regional["region_id"], "region_1")
        self.assertEqual(regional["child_nodes"], ["child_1", "child_2"])

    def test_to_regional_node_leaf_returns_none(self):
        """Test LEAF node returns None for to_regional_node."""
        node = SubdividableNode(node_id="leaf_1")
        self.assertIsNone(node.to_regional_node())

    def test_get_kg_nodes(self):
        """Test extracting KGNodes from registry."""
        registry = SubdivisionRegistry()
        node1 = SubdividableNode(
            node_id="leaf_1",
            endpoint="http://node1:8080",
            centroid=np.array([1.0, 0.0]),
        )
        node2 = SubdividableNode(
            node_id="leaf_2",
            endpoint="http://node2:8080",
            centroid=np.array([0.0, 1.0]),
        )
        registry.register(node1)
        registry.register(node2)

        kg_nodes = registry.get_kg_nodes()

        self.assertEqual(len(kg_nodes), 2)
        self.assertTrue(all(hasattr(n, "node_id") for n in kg_nodes))

    def test_to_node_hierarchy_data(self):
        """Test export for NodeHierarchy integration."""
        registry = SubdivisionRegistry()

        # Create and populate a node
        node = SubdividableNode(
            node_id="parent",
            config=SplitConfig(max_documents=50, min_child_documents=10),
        )
        for i in range(100):
            node.add_document(f"doc_{i}", np.random.randn(10))

        registry.register(node)
        registry.check_and_split_all(seed=42)

        # Export hierarchy data
        data = registry.to_node_hierarchy_data()

        self.assertIn("regions", data)
        self.assertIn("leaf_nodes", data)
        self.assertIn("node_to_region", data)
        self.assertEqual(len(data["regions"]), 1)  # One region (parent)
        self.assertEqual(len(data["leaf_nodes"]), 2)  # Two children

    def test_route_multi_k(self):
        """Test multi-candidate routing."""
        registry = SubdivisionRegistry()

        # Create three distinct nodes
        for i, center in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
            node = SubdividableNode(
                node_id=f"node_{i}",
                centroid=np.array(center, dtype=float),
            )
            registry.register(node)

        # Query near first cluster
        query = np.array([0.9, 0.1, 0.0])
        results = registry.route_multi_k(query, k=2)

        self.assertEqual(len(results), 2)
        # First result should be node_0 (most similar)
        self.assertEqual(results[0][0].node_id, "node_0")
        # Similarity should be highest
        self.assertGreater(results[0][1], results[1][1])


class TestMergeNodes(unittest.TestCase):
    """Tests for merge functionality."""

    def test_should_merge_underutilized(self):
        """Test merge detection for underutilized siblings."""
        config = SplitConfig(merge_enabled=True, min_documents_for_merge=100)

        node_a = SubdividableNode(
            node_id="child_a",
            parent_id="parent",
            config=config,
        )
        node_b = SubdividableNode(
            node_id="child_b",
            parent_id="parent",
            config=config,
        )

        # Add few documents
        for i in range(20):
            node_a.add_document(f"doc_a_{i}", np.random.randn(10))
            node_b.add_document(f"doc_b_{i}", np.random.randn(10))

        should, reason = should_merge(node_a, node_b)
        self.assertTrue(should)

    def test_should_merge_not_siblings(self):
        """Test merge fails for non-siblings."""
        node_a = SubdividableNode(node_id="a", parent_id="parent_1")
        node_b = SubdividableNode(node_id="b", parent_id="parent_2")

        should, reason = should_merge(node_a, node_b)
        self.assertFalse(should)
        self.assertIn("siblings", reason)

    def test_merge_nodes(self):
        """Test merging siblings back into parent."""
        parent = SubdividableNode(
            node_id="parent",
            node_type=NodeType.REGION,
            children_ids=["child_a", "child_b"],
        )
        child_a = SubdividableNode(node_id="child_a", parent_id="parent")
        child_b = SubdividableNode(node_id="child_b", parent_id="parent")

        for i in range(10):
            child_a.add_document(f"doc_a_{i}", np.random.randn(10))
            child_b.add_document(f"doc_b_{i}", np.random.randn(10))

        merged = merge_nodes(child_a, child_b, parent)

        self.assertEqual(merged.node_type, NodeType.LEAF)
        self.assertEqual(len(merged.document_ids), 20)
        self.assertEqual(len(merged.children_ids), 0)


if __name__ == "__main__":
    unittest.main()
