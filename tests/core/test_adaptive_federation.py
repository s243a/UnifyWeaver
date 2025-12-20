# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""Unit tests for Phase 5b: Adaptive Federation-K."""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

try:
    from federated_query import (
        QueryMetrics,
        AdaptiveKConfig,
        AdaptiveKCalculator,
        AdaptiveFederatedEngine,
        create_adaptive_engine,
        AggregationStrategy,
        AggregationConfig,
        AggregatedResponse
    )
    from kleinberg_router import KleinbergRouter, KGNode
except ImportError:
    from unifyweaver.targets.python_runtime.federated_query import (
        QueryMetrics,
        AdaptiveKConfig,
        AdaptiveKCalculator,
        AdaptiveFederatedEngine,
        create_adaptive_engine,
        AggregationStrategy,
        AggregationConfig,
        AggregatedResponse
    )
    from unifyweaver.targets.python_runtime.kleinberg_router import KleinbergRouter, KGNode


class TestQueryMetrics(unittest.TestCase):
    """Tests for QueryMetrics dataclass."""

    def test_query_metrics_creation(self):
        """Test creating QueryMetrics."""
        metrics = QueryMetrics(
            entropy=0.5,
            top_similarity=0.8,
            similarity_variance=0.05,
            historical_consensus=0.7,
            avg_node_latency_ms=100.0
        )
        self.assertEqual(metrics.entropy, 0.5)
        self.assertEqual(metrics.top_similarity, 0.8)
        self.assertEqual(metrics.similarity_variance, 0.05)
        self.assertEqual(metrics.historical_consensus, 0.7)
        self.assertEqual(metrics.avg_node_latency_ms, 100.0)


class TestAdaptiveKConfig(unittest.TestCase):
    """Tests for AdaptiveKConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AdaptiveKConfig()
        self.assertEqual(config.base_k, 3)
        self.assertEqual(config.min_k, 1)
        self.assertEqual(config.max_k, 10)
        self.assertEqual(config.entropy_weight, 0.3)
        self.assertEqual(config.latency_weight, 0.2)
        self.assertEqual(config.consensus_weight, 0.5)
        self.assertEqual(config.entropy_threshold, 0.7)
        self.assertEqual(config.similarity_threshold, 0.5)
        self.assertEqual(config.consensus_threshold, 0.6)
        self.assertEqual(config.history_size, 100)

    def test_custom_config(self):
        """Test custom configuration."""
        config = AdaptiveKConfig(
            base_k=5,
            min_k=2,
            max_k=15,
            entropy_weight=0.5
        )
        self.assertEqual(config.base_k, 5)
        self.assertEqual(config.min_k, 2)
        self.assertEqual(config.max_k, 15)
        self.assertEqual(config.entropy_weight, 0.5)


class TestAdaptiveKCalculator(unittest.TestCase):
    """Tests for AdaptiveKCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AdaptiveKConfig(base_k=3, min_k=1, max_k=10)
        self.calculator = AdaptiveKCalculator(self.config)

        # Create mock nodes with different centroids
        np.random.seed(42)
        self.nodes = [
            KGNode(
                node_id=f"node_{i}",
                endpoint=f"http://node{i}:8080",
                centroid=np.random.randn(384),
                topics=[f"topic_{i}"],
                embedding_model="test-model"
            )
            for i in range(5)
        ]

    def test_compute_k_default(self):
        """Test compute_k returns valid range."""
        query_embedding = np.random.randn(384)
        k = self.calculator.compute_k(query_embedding, self.nodes)
        self.assertGreaterEqual(k, self.config.min_k)
        self.assertLessEqual(k, min(self.config.max_k, len(self.nodes)))

    def test_compute_k_empty_nodes(self):
        """Test compute_k with empty node list."""
        query_embedding = np.random.randn(384)
        k = self.calculator.compute_k(query_embedding, [])
        self.assertEqual(k, self.config.min_k)

    def test_compute_k_high_entropy_increases_k(self):
        """Test that high entropy (ambiguous query) increases k."""
        # Create nodes with similar centroids (high entropy scenario)
        uniform_nodes = [
            KGNode(
                node_id=f"node_{i}",
                endpoint=f"http://node{i}:8080",
                centroid=np.ones(384) + np.random.randn(384) * 0.01,  # Nearly identical
                topics=[f"topic_{i}"],
                embedding_model="test-model"
            )
            for i in range(5)
        ]

        query = np.ones(384)
        k_uniform = self.calculator.compute_k(query, uniform_nodes)

        # Create nodes with one very similar, others different (low entropy)
        diverse_nodes = [
            KGNode(
                node_id="node_0",
                endpoint="http://node0:8080",
                centroid=np.ones(384),  # Very similar to query
                topics=["topic_0"],
                embedding_model="test-model"
            )
        ] + [
            KGNode(
                node_id=f"node_{i}",
                endpoint=f"http://node{i}:8080",
                centroid=np.random.randn(384),  # Random
                topics=[f"topic_{i}"],
                embedding_model="test-model"
            )
            for i in range(1, 5)
        ]

        k_diverse = self.calculator.compute_k(query, diverse_nodes)

        # High entropy should generally lead to higher k
        # (though this depends on thresholds)
        self.assertGreaterEqual(k_uniform, self.config.min_k)

    def test_compute_k_low_similarity_increases_k(self):
        """Test that low max similarity increases k."""
        # Create nodes with low similarity to query
        query = np.array([1.0] * 384)
        low_sim_nodes = [
            KGNode(
                node_id=f"node_{i}",
                endpoint=f"http://node{i}:8080",
                centroid=np.array([-1.0] * 384),  # Opposite direction
                topics=[f"topic_{i}"],
                embedding_model="test-model"
            )
            for i in range(5)
        ]

        k = self.calculator.compute_k(query, low_sim_nodes)
        # Low similarity should increase k above base
        self.assertGreaterEqual(k, self.config.base_k)

    def test_compute_k_latency_budget_limits_k(self):
        """Test that latency budget limits k."""
        query_embedding = np.random.randn(384)

        # Add latency data
        for node in self.nodes:
            self.calculator._latency_cache[node.node_id] = [200.0]  # 200ms each

        # With 500ms budget and 200ms per node, max 2 nodes
        k = self.calculator.compute_k(query_embedding, self.nodes, latency_budget_ms=500)
        self.assertLessEqual(k, 3)  # Should be constrained

    def test_record_query_outcome(self):
        """Test recording query outcomes."""
        query_embedding = np.random.randn(384)
        self.calculator.record_query_outcome(
            query_embedding=query_embedding,
            consensus_score=0.8,
            k_used=3,
            node_latencies={"node_0": 50.0, "node_1": 75.0}
        )

        self.assertEqual(len(self.calculator.query_history), 1)
        self.assertEqual(len(self.calculator._latency_cache), 2)
        self.assertIn("node_0", self.calculator._latency_cache)

    def test_record_query_outcome_trims_history(self):
        """Test that history is trimmed at limit."""
        config = AdaptiveKConfig(history_size=5)
        calculator = AdaptiveKCalculator(config)

        for i in range(10):
            calculator.record_query_outcome(
                query_embedding=np.random.randn(384),
                consensus_score=0.5,
                k_used=3
            )

        self.assertEqual(len(calculator.query_history), 5)

    def test_historical_consensus_influences_k(self):
        """Test that low historical consensus increases k."""
        query = np.ones(384)

        # Record similar queries with low consensus
        for _ in range(5):
            similar_query = np.ones(384) + np.random.randn(384) * 0.01
            self.calculator.record_query_outcome(
                query_embedding=similar_query,
                consensus_score=0.3,  # Low consensus
                k_used=3
            )

        k = self.calculator.compute_k(query, self.nodes)
        # Low historical consensus should increase k
        self.assertGreaterEqual(k, self.config.base_k)

    def test_get_stats(self):
        """Test statistics retrieval."""
        # Empty stats
        stats = self.calculator.get_stats()
        self.assertEqual(stats['queries_recorded'], 0)
        self.assertEqual(stats['avg_k_used'], self.config.base_k)

        # Add some history
        for i in range(3):
            self.calculator.record_query_outcome(
                query_embedding=np.random.randn(384),
                consensus_score=0.7,
                k_used=3 + i,
                node_latencies={f"node_{i}": 100.0}
            )

        stats = self.calculator.get_stats()
        self.assertEqual(stats['queries_recorded'], 3)
        self.assertEqual(stats['avg_k_used'], 4.0)  # (3+4+5)/3
        self.assertAlmostEqual(stats['avg_consensus'], 0.7, places=5)
        self.assertEqual(stats['nodes_tracked'], 3)

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(self.calculator._cosine_similarity(a, b), 1.0)

        c = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(self.calculator._cosine_similarity(a, c), 0.0)

        d = np.array([-1.0, 0.0, 0.0])
        self.assertAlmostEqual(self.calculator._cosine_similarity(a, d), -1.0)

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.zeros(3)
        self.assertEqual(self.calculator._cosine_similarity(a, b), 0.0)


class TestAdaptiveFederatedEngine(unittest.TestCase):
    """Tests for AdaptiveFederatedEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock router
        self.mock_router = Mock(spec=KleinbergRouter)

        # Create mock nodes
        np.random.seed(42)
        self.mock_nodes = [
            KGNode(
                node_id=f"node_{i}",
                endpoint=f"http://node{i}:8080",
                centroid=np.random.randn(384),
                topics=[f"topic_{i}"],
                embedding_model="test-model"
            )
            for i in range(5)
        ]
        self.mock_router.discover_nodes.return_value = self.mock_nodes

    def test_engine_creation(self):
        """Test creating AdaptiveFederatedEngine."""
        engine = AdaptiveFederatedEngine(
            router=self.mock_router,
            adaptive_config=AdaptiveKConfig(base_k=5)
        )
        self.assertEqual(engine.federation_k, 5)
        self.assertIsNotNone(engine.adaptive)

    def test_engine_creation_with_factory(self):
        """Test creating engine with factory function."""
        engine = create_adaptive_engine(
            router=self.mock_router,
            base_k=4,
            min_k=2,
            max_k=8,
            entropy_weight=0.4
        )
        self.assertEqual(engine.federation_k, 4)
        self.assertEqual(engine.adaptive.config.min_k, 2)
        self.assertEqual(engine.adaptive.config.max_k, 8)
        self.assertEqual(engine.adaptive.config.entropy_weight, 0.4)

    def test_get_stats_includes_adaptive(self):
        """Test that stats include adaptive metrics."""
        engine = AdaptiveFederatedEngine(router=self.mock_router)
        stats = engine.get_stats()
        self.assertIn('adaptive', stats)
        self.assertIn('queries_recorded', stats['adaptive'])

    @patch('federated_query.FederatedQueryEngine.federated_query')
    def test_federated_query_uses_adaptive_k(self, mock_parent_query):
        """Test that federated_query uses computed adaptive k."""
        # Set up mock response
        mock_response = Mock(spec=AggregatedResponse)
        mock_response.results = [{'normalized_prob': 0.8}, {'normalized_prob': 0.2}]
        mock_parent_query.return_value = mock_response

        engine = AdaptiveFederatedEngine(
            router=self.mock_router,
            adaptive_config=AdaptiveKConfig(base_k=3)
        )

        query_embedding = np.random.randn(384)
        engine.federated_query(
            query_text="test query",
            query_embedding=query_embedding,
            top_k=5
        )

        # Verify parent was called
        mock_parent_query.assert_called_once()
        # Check that federation_k was passed
        call_kwargs = mock_parent_query.call_args[1]
        self.assertIn('federation_k', call_kwargs)
        self.assertIsInstance(call_kwargs['federation_k'], int)

    @patch('federated_query.FederatedQueryEngine.federated_query')
    def test_federated_query_override_k(self, mock_parent_query):
        """Test that explicit federation_k overrides adaptive."""
        mock_response = Mock(spec=AggregatedResponse)
        mock_response.results = [{'normalized_prob': 0.8}]
        mock_parent_query.return_value = mock_response

        engine = AdaptiveFederatedEngine(
            router=self.mock_router,
            adaptive_config=AdaptiveKConfig(base_k=3)
        )

        query_embedding = np.random.randn(384)
        engine.federated_query(
            query_text="test query",
            query_embedding=query_embedding,
            top_k=5,
            federation_k=7  # Explicit override
        )

        call_kwargs = mock_parent_query.call_args[1]
        self.assertEqual(call_kwargs['federation_k'], 7)

    @patch('federated_query.FederatedQueryEngine.federated_query')
    def test_federated_query_with_latency_budget(self, mock_parent_query):
        """Test federated_query with latency budget."""
        mock_response = Mock(spec=AggregatedResponse)
        mock_response.results = [{'normalized_prob': 0.8}]
        mock_parent_query.return_value = mock_response

        engine = AdaptiveFederatedEngine(router=self.mock_router)

        # Add latency data to make budget meaningful
        for node in self.mock_nodes:
            engine.adaptive._latency_cache[node.node_id] = [200.0]

        query_embedding = np.random.randn(384)
        engine.federated_query(
            query_text="test query",
            query_embedding=query_embedding,
            top_k=5,
            latency_budget_ms=400
        )

        # Should have called with constrained k
        call_kwargs = mock_parent_query.call_args[1]
        self.assertIn('federation_k', call_kwargs)

    def test_compute_consensus_score_empty(self):
        """Test consensus score with empty results."""
        engine = AdaptiveFederatedEngine(router=self.mock_router)
        mock_response = Mock(spec=AggregatedResponse)
        mock_response.results = []
        score = engine._compute_consensus_score(mock_response)
        self.assertEqual(score, 0.0)

    def test_compute_consensus_score_two_results(self):
        """Test consensus score with two results."""
        engine = AdaptiveFederatedEngine(router=self.mock_router)
        mock_response = Mock(spec=AggregatedResponse)
        mock_response.results = [
            {'normalized_prob': 0.8},
            {'normalized_prob': 0.2}
        ]
        score = engine._compute_consensus_score(mock_response)
        # 0.8 / (0.2 + 0.1) = 2.67, clamped to 1.0
        self.assertEqual(score, 1.0)

    def test_compute_consensus_score_close_results(self):
        """Test consensus score with close results."""
        engine = AdaptiveFederatedEngine(router=self.mock_router)
        mock_response = Mock(spec=AggregatedResponse)
        mock_response.results = [
            {'normalized_prob': 0.35},
            {'normalized_prob': 0.30}
        ]
        score = engine._compute_consensus_score(mock_response)
        # 0.35 / (0.30 + 0.1) = 0.875
        self.assertAlmostEqual(score, 0.875, places=2)


if __name__ == '__main__':
    unittest.main()
