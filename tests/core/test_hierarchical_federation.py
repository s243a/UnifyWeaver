"""
Unit tests for Phase 5a: Hierarchical Federation.

Tests:
- RegionalNode data structure
- HierarchyConfig options
- NodeHierarchy building (topic-based, centroid-based)
- HierarchicalFederatedEngine query routing
- create_hierarchical_engine factory
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from dataclasses import asdict

# Import Phase 5a classes
try:
    from federated_query import (
        RegionalNode, HierarchyConfig, NodeHierarchy,
        HierarchicalFederatedEngine, create_hierarchical_engine,
        AggregationStrategy, AggregationConfig, AggregatedResponse
    )
    from kleinberg_router import KGNode, KleinbergRouter
except ImportError:
    from unifyweaver.targets.python_runtime.federated_query import (
        RegionalNode, HierarchyConfig, NodeHierarchy,
        HierarchicalFederatedEngine, create_hierarchical_engine,
        AggregationStrategy, AggregationConfig, AggregatedResponse
    )
    from unifyweaver.targets.python_runtime.kleinberg_router import KGNode, KleinbergRouter


class TestRegionalNode(unittest.TestCase):
    """Test RegionalNode data structure."""

    def test_creation(self):
        """Test creating a regional node."""
        centroid = np.array([0.1, 0.2, 0.3])
        region = RegionalNode(
            region_id="region_1",
            centroid=centroid,
            topics=["csv", "parsing"],
            child_nodes=["node_1", "node_2"]
        )
        self.assertEqual(region.region_id, "region_1")
        np.testing.assert_array_almost_equal(region.centroid, centroid)
        self.assertEqual(region.topics, ["csv", "parsing"])
        self.assertEqual(region.child_nodes, ["node_1", "node_2"])
        self.assertIsNone(region.parent_region)
        self.assertEqual(region.level, 0)

    def test_with_parent(self):
        """Test regional node with parent."""
        region = RegionalNode(
            region_id="region_child",
            centroid=np.array([0.5, 0.5]),
            topics=["nested"],
            child_nodes=["node_3"],
            parent_region="region_parent",
            level=1
        )
        self.assertEqual(region.parent_region, "region_parent")
        self.assertEqual(region.level, 1)

    def test_to_dict(self):
        """Test serialization."""
        region = RegionalNode(
            region_id="r1",
            centroid=np.array([0.1, 0.2]),
            topics=["t1"],
            child_nodes=["c1", "c2"]
        )
        d = region.to_dict()
        self.assertEqual(d['region_id'], "r1")
        self.assertEqual(d['centroid'], [0.1, 0.2])
        self.assertEqual(d['topics'], ["t1"])
        self.assertEqual(d['child_nodes'], ["c1", "c2"])

    def test_to_dict_none_centroid(self):
        """Test serialization with None centroid."""
        region = RegionalNode(
            region_id="r1",
            centroid=None,
            topics=[],
            child_nodes=[]
        )
        d = region.to_dict()
        self.assertIsNone(d['centroid'])


class TestHierarchyConfig(unittest.TestCase):
    """Test HierarchyConfig defaults and validation."""

    def test_defaults(self):
        """Test default configuration values."""
        config = HierarchyConfig()
        self.assertEqual(config.max_levels, 3)
        self.assertEqual(config.min_nodes_per_region, 2)
        self.assertEqual(config.max_nodes_per_region, 10)
        self.assertEqual(config.topic_similarity_threshold, 0.5)
        self.assertEqual(config.centroid_similarity_threshold, 0.6)

    def test_custom_values(self):
        """Test custom configuration."""
        config = HierarchyConfig(
            max_levels=5,
            min_nodes_per_region=3,
            max_nodes_per_region=20,
            topic_similarity_threshold=0.7,
            centroid_similarity_threshold=0.8
        )
        self.assertEqual(config.max_levels, 5)
        self.assertEqual(config.min_nodes_per_region, 3)
        self.assertEqual(config.max_nodes_per_region, 20)


class TestNodeHierarchy(unittest.TestCase):
    """Test NodeHierarchy building and querying."""

    def _make_node(self, node_id, topics, centroid=None):
        """Helper to create KGNode."""
        return KGNode(
            node_id=node_id,
            endpoint=f"http://{node_id}:8080",
            centroid=centroid if centroid is not None else np.random.randn(384),
            topics=topics,
            embedding_model="test-model"
        )

    def test_empty_build(self):
        """Test building with empty node list."""
        hierarchy = NodeHierarchy()
        hierarchy.build_from_nodes([])
        self.assertEqual(len(hierarchy.regions), 0)
        self.assertEqual(len(hierarchy._leaf_nodes), 0)

    def test_topic_grouping_single_topic(self):
        """Test grouping nodes by single shared topic."""
        nodes = [
            self._make_node("n1", ["csv"]),
            self._make_node("n2", ["csv"]),
            self._make_node("n3", ["csv"]),
        ]
        hierarchy = NodeHierarchy()
        hierarchy.build_from_nodes(nodes)

        # All should be in same region
        self.assertEqual(len(hierarchy.regions), 1)
        region = list(hierarchy.regions.values())[0]
        self.assertEqual(len(region.child_nodes), 3)
        self.assertIn("csv", region.topics)

    def test_topic_grouping_multiple_topics(self):
        """Test grouping nodes with different topics."""
        config = HierarchyConfig(min_nodes_per_region=1)
        nodes = [
            self._make_node("n1", ["csv", "parsing"]),
            self._make_node("n2", ["csv", "validation"]),
            self._make_node("n3", ["json", "parsing"]),
            self._make_node("n4", ["json", "validation"]),
        ]
        hierarchy = NodeHierarchy(config)
        hierarchy.build_from_nodes(nodes)

        # Should create regions based on topic overlap
        self.assertGreaterEqual(len(hierarchy.regions), 1)
        # All nodes should be in some region
        for node in nodes:
            self.assertIn(node.node_id, hierarchy.node_to_region)

    def test_centroid_grouping(self):
        """Test grouping nodes without topics by centroid similarity."""
        np.random.seed(42)
        # Create two clusters
        cluster1_centroid = np.ones(384) * 0.5
        cluster2_centroid = np.ones(384) * -0.5

        config = HierarchyConfig(
            min_nodes_per_region=2,
            centroid_similarity_threshold=0.5
        )

        nodes = [
            self._make_node("n1", [], cluster1_centroid + np.random.randn(384) * 0.1),
            self._make_node("n2", [], cluster1_centroid + np.random.randn(384) * 0.1),
            self._make_node("n3", [], cluster2_centroid + np.random.randn(384) * 0.1),
            self._make_node("n4", [], cluster2_centroid + np.random.randn(384) * 0.1),
        ]

        hierarchy = NodeHierarchy(config)
        hierarchy.build_from_nodes(nodes)

        # Should create at least one region
        self.assertGreaterEqual(len(hierarchy.regions), 1)

    def test_get_regional_nodes(self):
        """Test getting nodes at specific level."""
        nodes = [
            self._make_node("n1", ["t1"]),
            self._make_node("n2", ["t1"]),
        ]
        hierarchy = NodeHierarchy()
        hierarchy.build_from_nodes(nodes)

        level0 = hierarchy.get_regional_nodes(level=0)
        self.assertEqual(len(level0), 1)

        level1 = hierarchy.get_regional_nodes(level=1)
        self.assertEqual(len(level1), 0)  # No level 1 regions

    def test_get_children(self):
        """Test getting child node IDs."""
        nodes = [
            self._make_node("n1", ["t1"]),
            self._make_node("n2", ["t1"]),
        ]
        hierarchy = NodeHierarchy()
        hierarchy.build_from_nodes(nodes)

        region_id = list(hierarchy.regions.keys())[0]
        children = hierarchy.get_children(region_id)
        self.assertIn("n1", children)
        self.assertIn("n2", children)

    def test_get_child_nodes(self):
        """Test getting actual KGNode objects."""
        nodes = [
            self._make_node("n1", ["t1"]),
            self._make_node("n2", ["t1"]),
        ]
        hierarchy = NodeHierarchy()
        hierarchy.build_from_nodes(nodes)

        region_id = list(hierarchy.regions.keys())[0]
        child_nodes = hierarchy.get_child_nodes(region_id)
        self.assertEqual(len(child_nodes), 2)
        self.assertTrue(all(isinstance(n, KGNode) for n in child_nodes))

    def test_get_region_for_node(self):
        """Test looking up node's region."""
        nodes = [
            self._make_node("n1", ["t1"]),
            self._make_node("n2", ["t1"]),
        ]
        hierarchy = NodeHierarchy()
        hierarchy.build_from_nodes(nodes)

        region = hierarchy.get_region_for_node("n1")
        self.assertIsNotNone(region)
        self.assertIn(region, hierarchy.regions)

    def test_get_stats_empty(self):
        """Test stats on empty hierarchy."""
        hierarchy = NodeHierarchy()
        stats = hierarchy.get_stats()
        self.assertEqual(stats['num_regions'], 0)
        self.assertEqual(stats['num_nodes'], 0)
        self.assertEqual(stats['levels'], 0)

    def test_get_stats_populated(self):
        """Test stats on populated hierarchy."""
        nodes = [
            self._make_node("n1", ["t1"]),
            self._make_node("n2", ["t1"]),
            self._make_node("n3", ["t1"]),
        ]
        hierarchy = NodeHierarchy()
        hierarchy.build_from_nodes(nodes)

        stats = hierarchy.get_stats()
        self.assertEqual(stats['num_regions'], 1)
        self.assertEqual(stats['num_nodes'], 3)
        self.assertEqual(stats['avg_nodes_per_region'], 3.0)
        self.assertEqual(stats['levels'], 1)


class TestHierarchicalFederatedEngine(unittest.TestCase):
    """Test HierarchicalFederatedEngine query routing."""

    def _make_node(self, node_id, topics, centroid=None):
        """Helper to create KGNode."""
        return KGNode(
            node_id=node_id,
            endpoint=f"http://{node_id}:8080",
            centroid=centroid if centroid is not None else np.random.randn(384),
            topics=topics,
            embedding_model="test-model"
        )

    def _make_mock_router(self, nodes):
        """Create mock router returning given nodes."""
        router = Mock(spec=KleinbergRouter)
        router.discover_nodes.return_value = nodes
        return router

    def test_engine_creation(self):
        """Test creating hierarchical engine."""
        router = Mock(spec=KleinbergRouter)
        router.discover_nodes.return_value = []

        engine = HierarchicalFederatedEngine(router=router)
        self.assertIsNotNone(engine)
        self.assertEqual(engine.drill_down_k, 2)
        self.assertIsNone(engine.hierarchy)

    def test_engine_with_prebuilt_hierarchy(self):
        """Test engine with pre-built hierarchy."""
        router = Mock(spec=KleinbergRouter)
        hierarchy = NodeHierarchy()

        engine = HierarchicalFederatedEngine(
            router=router,
            hierarchy=hierarchy
        )
        self.assertIs(engine.hierarchy, hierarchy)
        self.assertTrue(engine._hierarchy_built)

    def test_ensure_hierarchy_builds(self):
        """Test hierarchy is built on first query."""
        nodes = [
            self._make_node("n1", ["csv"]),
            self._make_node("n2", ["csv"]),
        ]
        router = self._make_mock_router(nodes)

        engine = HierarchicalFederatedEngine(router=router)
        self.assertFalse(engine._hierarchy_built)

        engine._ensure_hierarchy()
        self.assertTrue(engine._hierarchy_built)
        self.assertIsNotNone(engine.hierarchy)
        self.assertEqual(len(engine.hierarchy.regions), 1)

    def test_rebuild_hierarchy(self):
        """Test forcing hierarchy rebuild."""
        nodes = [self._make_node("n1", ["csv"])] * 2
        router = self._make_mock_router(nodes)

        engine = HierarchicalFederatedEngine(router=router)
        engine._ensure_hierarchy()

        # Rebuild should work
        engine.rebuild_hierarchy()
        self.assertTrue(engine._hierarchy_built)

    def test_rank_regions(self):
        """Test ranking regions by query similarity."""
        router = Mock(spec=KleinbergRouter)
        engine = HierarchicalFederatedEngine(router=router)

        query = np.array([1.0, 0.0, 0.0])
        regions = [
            RegionalNode("r1", np.array([1.0, 0.0, 0.0]), ["t1"], ["n1"]),
            RegionalNode("r2", np.array([0.0, 1.0, 0.0]), ["t2"], ["n2"]),
            RegionalNode("r3", np.array([0.5, 0.5, 0.0]), ["t3"], ["n3"]),
        ]

        ranked = engine._rank_regions(query, regions)
        # Region 1 should be most similar
        self.assertEqual(ranked[0][0].region_id, "r1")
        self.assertAlmostEqual(ranked[0][1], 1.0, places=5)

    def test_get_stats(self):
        """Test combined engine and hierarchy stats."""
        nodes = [
            self._make_node("n1", ["csv"]),
            self._make_node("n2", ["csv"]),
        ]
        router = self._make_mock_router(nodes)

        engine = HierarchicalFederatedEngine(
            router=router,
            drill_down_k=3
        )
        engine._ensure_hierarchy()

        stats = engine.get_stats()
        self.assertIn('hierarchy', stats)
        self.assertEqual(stats['drill_down_k'], 3)
        self.assertEqual(stats['hierarchy']['num_regions'], 1)

    def test_use_hierarchy_false(self):
        """Test bypassing hierarchy with use_hierarchy=False."""
        nodes = [
            self._make_node("n1", ["csv"]),
            self._make_node("n2", ["csv"]),
        ]
        router = self._make_mock_router(nodes)

        engine = HierarchicalFederatedEngine(router=router)

        # Mock the parent class method
        with patch.object(engine.__class__.__bases__[0], 'federated_query') as mock_query:
            mock_query.return_value = AggregatedResponse(
                query_id="test",
                results=[],
                total_partition_sum=0.0,
                nodes_queried=0,
                nodes_responded=0,
                total_time_ms=0.0,
                aggregation_strategy="sum"
            )

            engine.federated_query(
                "test query",
                np.array([0.1, 0.2]),
                use_hierarchy=False
            )
            mock_query.assert_called_once()


class TestCreateHierarchicalEngine(unittest.TestCase):
    """Test create_hierarchical_engine factory."""

    def test_factory_defaults(self):
        """Test factory with default parameters."""
        router = Mock(spec=KleinbergRouter)
        router.discover_nodes.return_value = []

        engine = create_hierarchical_engine(router)

        self.assertIsInstance(engine, HierarchicalFederatedEngine)
        self.assertEqual(engine.drill_down_k, 2)
        self.assertEqual(engine.federation_k, 3)
        self.assertEqual(engine.timeout_ms, 5000)

    def test_factory_custom_params(self):
        """Test factory with custom parameters."""
        router = Mock(spec=KleinbergRouter)
        router.discover_nodes.return_value = []

        engine = create_hierarchical_engine(
            router,
            max_levels=5,
            min_nodes_per_region=3,
            max_nodes_per_region=15,
            drill_down_k=4,
            federation_k=5,
            timeout_ms=10000,
            aggregation_strategy=AggregationStrategy.MAX
        )

        self.assertEqual(engine.drill_down_k, 4)
        self.assertEqual(engine.federation_k, 5)
        self.assertEqual(engine.timeout_ms, 10000)
        self.assertEqual(engine.config.strategy, AggregationStrategy.MAX)
        self.assertEqual(engine.hierarchy_config.max_levels, 5)
        self.assertEqual(engine.hierarchy_config.min_nodes_per_region, 3)
        self.assertEqual(engine.hierarchy_config.max_nodes_per_region, 15)


class TestPrologValidation(unittest.TestCase):
    """Test Prolog validation for Phase 5a options."""

    def test_hierarchical_true(self):
        """Test hierarchical(true) validation."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_federation_option(hierarchical(true)), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)

    def test_hierarchical_false(self):
        """Test hierarchical(false) validation."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_federation_option(hierarchical(false)), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)

    def test_hierarchical_options_list(self):
        """Test hierarchical options list validation."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_federation_option(hierarchical([max_levels(3), drill_down_k(2)])), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)

    def test_hierarchy_option_max_levels(self):
        """Test max_levels validation."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_hierarchy_option(max_levels(5)), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)

    def test_hierarchy_option_thresholds(self):
        """Test threshold validations."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_hierarchy_option(topic_similarity_threshold(0.7)), "
            "is_valid_hierarchy_option(centroid_similarity_threshold(0.8)), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)

    def test_hierarchy_option_regional_aggregation(self):
        """Test regional_aggregation with strategy."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_hierarchy_option(regional_aggregation(sum)), "
            "is_valid_hierarchy_option(regional_aggregation(max)), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)


if __name__ == '__main__':
    unittest.main()
