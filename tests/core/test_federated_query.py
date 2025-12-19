# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Unit tests for KG Topology Phase 4: Federated Query Algebra

"""
Unit tests for federated query components.

Tests:
- Aggregation functions (SUM, MAX, AVG, etc.)
- Aggregator monoid properties (associativity, commutativity)
- NodeResult and NodeResponse serialization
- AggregatedResult merging
- FederatedQueryEngine aggregation logic
"""

import unittest
import tempfile
import os
import sys
import time
import math

import numpy as np

# Add source directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from unifyweaver.targets.python_runtime.federated_query import (
    AggregationStrategy,
    AggregationConfig,
    Aggregator,
    SumAggregator,
    MaxAggregator,
    MinAggregator,
    AvgAggregator,
    CountAggregator,
    FirstAggregator,
    get_aggregator,
    NodeResult,
    NodeResponse,
    ResultProvenance,
    AggregatedResult,
    AggregatedResponse,
    compute_answer_hash,
    compute_exp_scores,
    create_federated_engine,
)
from unifyweaver.targets.python_runtime.discovery_clients import LocalDiscoveryClient
from unifyweaver.targets.python_runtime.kleinberg_router import KleinbergRouter


class TestAggregationStrategy(unittest.TestCase):
    """Tests for AggregationStrategy enum."""

    def test_all_strategies_exist(self):
        """Test all expected strategies are defined."""
        strategies = [s.value for s in AggregationStrategy]
        self.assertIn('sum', strategies)
        self.assertIn('max', strategies)
        self.assertIn('min', strategies)
        self.assertIn('avg', strategies)
        self.assertIn('count', strategies)
        self.assertIn('first', strategies)
        self.assertIn('collect', strategies)
        self.assertIn('diversity', strategies)


class TestSumAggregator(unittest.TestCase):
    """Tests for SumAggregator."""

    def setUp(self):
        self.agg = SumAggregator()

    def test_identity(self):
        """Test identity element."""
        self.assertEqual(self.agg.identity(), 0.0)

    def test_merge(self):
        """Test merge operation."""
        self.assertEqual(self.agg.merge(2.0, 3.0), 5.0)

    def test_finalize(self):
        """Test finalize."""
        self.assertEqual(self.agg.finalize(5.0), 5.0)

    def test_associativity(self):
        """Test associative property: (a + b) + c = a + (b + c)"""
        a, b, c = 1.0, 2.0, 3.0
        left = self.agg.merge(self.agg.merge(a, b), c)
        right = self.agg.merge(a, self.agg.merge(b, c))
        self.assertEqual(left, right)

    def test_commutativity(self):
        """Test commutative property: a + b = b + a"""
        a, b = 1.5, 2.5
        self.assertEqual(self.agg.merge(a, b), self.agg.merge(b, a))

    def test_identity_element(self):
        """Test identity: a + 0 = a"""
        a = 5.0
        self.assertEqual(self.agg.merge(a, self.agg.identity()), a)


class TestMaxAggregator(unittest.TestCase):
    """Tests for MaxAggregator."""

    def setUp(self):
        self.agg = MaxAggregator()

    def test_merge(self):
        """Test merge takes maximum."""
        self.assertEqual(self.agg.merge(2.0, 3.0), 3.0)
        self.assertEqual(self.agg.merge(5.0, 1.0), 5.0)

    def test_finalize_handles_identity(self):
        """Test finalize handles -inf identity."""
        self.assertEqual(self.agg.finalize(float('-inf')), 0.0)
        self.assertEqual(self.agg.finalize(5.0), 5.0)

    def test_associativity(self):
        """Test associative property."""
        a, b, c = 1.0, 5.0, 3.0
        left = self.agg.merge(self.agg.merge(a, b), c)
        right = self.agg.merge(a, self.agg.merge(b, c))
        self.assertEqual(left, right)


class TestMinAggregator(unittest.TestCase):
    """Tests for MinAggregator."""

    def setUp(self):
        self.agg = MinAggregator()

    def test_merge(self):
        """Test merge takes minimum."""
        self.assertEqual(self.agg.merge(2.0, 3.0), 2.0)
        self.assertEqual(self.agg.merge(5.0, 1.0), 1.0)

    def test_finalize_handles_identity(self):
        """Test finalize handles +inf identity."""
        self.assertEqual(self.agg.finalize(float('inf')), 0.0)
        self.assertEqual(self.agg.finalize(5.0), 5.0)


class TestAvgAggregator(unittest.TestCase):
    """Tests for AvgAggregator."""

    def setUp(self):
        self.agg = AvgAggregator()

    def test_identity(self):
        """Test identity is (0, 0)."""
        self.assertEqual(self.agg.identity(), (0.0, 0))

    def test_merge(self):
        """Test merge combines (sum, count) tuples."""
        a = (10.0, 2)
        b = (15.0, 3)
        result = self.agg.merge(a, b)
        self.assertEqual(result, (25.0, 5))

    def test_finalize(self):
        """Test finalize computes average."""
        self.assertEqual(self.agg.finalize((10.0, 4)), 2.5)
        self.assertEqual(self.agg.finalize((0.0, 0)), 0.0)  # Avoid division by zero


class TestCountAggregator(unittest.TestCase):
    """Tests for CountAggregator."""

    def setUp(self):
        self.agg = CountAggregator()

    def test_merge(self):
        """Test merge adds counts."""
        self.assertEqual(self.agg.merge(2, 3), 5)

    def test_finalize(self):
        """Test finalize returns float."""
        self.assertIsInstance(self.agg.finalize(5), float)


class TestGetAggregator(unittest.TestCase):
    """Tests for get_aggregator factory."""

    def test_returns_correct_types(self):
        """Test factory returns correct aggregator types."""
        self.assertIsInstance(get_aggregator(AggregationStrategy.SUM), SumAggregator)
        self.assertIsInstance(get_aggregator(AggregationStrategy.MAX), MaxAggregator)
        self.assertIsInstance(get_aggregator(AggregationStrategy.MIN), MinAggregator)
        self.assertIsInstance(get_aggregator(AggregationStrategy.AVG), AvgAggregator)
        self.assertIsInstance(get_aggregator(AggregationStrategy.COUNT), CountAggregator)
        self.assertIsInstance(get_aggregator(AggregationStrategy.FIRST), FirstAggregator)


class TestNodeResult(unittest.TestCase):
    """Tests for NodeResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        result = NodeResult(
            answer_id=1,
            answer_text="Test answer",
            answer_hash="abc123",
            raw_score=0.85,
            exp_score=2.34,
            metadata={'source': 'test'}
        )
        d = result.to_dict()

        self.assertEqual(d['answer_id'], 1)
        self.assertEqual(d['answer_text'], "Test answer")
        self.assertEqual(d['answer_hash'], "abc123")
        self.assertEqual(d['raw_score'], 0.85)
        self.assertEqual(d['exp_score'], 2.34)
        self.assertEqual(d['metadata']['source'], 'test')

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            'answer_id': 2,
            'answer_text': "Another answer",
            'answer_hash': "def456",
            'raw_score': 0.75,
            'exp_score': 2.12
        }
        result = NodeResult.from_dict(d)

        self.assertEqual(result.answer_id, 2)
        self.assertEqual(result.answer_text, "Another answer")
        self.assertEqual(result.exp_score, 2.12)


class TestNodeResponse(unittest.TestCase):
    """Tests for NodeResponse dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        response = NodeResponse(
            source_node="node-a",
            results=[],
            partition_sum=5.0,
            node_metadata={'corpus_id': 'test'}
        )
        d = response.to_dict()

        self.assertEqual(d['__type'], 'kg_federated_response')
        self.assertEqual(d['source_node'], 'node-a')
        self.assertEqual(d['partition_sum'], 5.0)

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            'source_node': 'node-b',
            'results': [],
            'partition_sum': 3.0,
            'node_metadata': {}
        }
        response = NodeResponse.from_dict(d)

        self.assertEqual(response.source_node, 'node-b')
        self.assertEqual(response.partition_sum, 3.0)


class TestAggregatedResult(unittest.TestCase):
    """Tests for AggregatedResult dataclass."""

    def test_from_single(self):
        """Test creation from single result."""
        result = NodeResult(
            answer_id=1,
            answer_text="Answer",
            answer_hash="hash1",
            raw_score=0.8,
            exp_score=2.2
        )
        aggregated = AggregatedResult.from_single(
            result, 'node-a', {'corpus_id': 'test_corpus'}
        )

        self.assertEqual(aggregated.answer_text, "Answer")
        self.assertEqual(aggregated.combined_score, 2.2)
        self.assertEqual(aggregated.source_nodes, ['node-a'])
        self.assertEqual(len(aggregated.provenance), 1)
        self.assertEqual(aggregated.provenance[0].corpus_id, 'test_corpus')

    def test_to_dict(self):
        """Test serialization with normalized probability."""
        aggregated = AggregatedResult(
            answer_text="Test",
            answer_hash="abc",
            combined_score=5.0,
            source_nodes=['a', 'b'],
            provenance=[]
        )
        d = aggregated.to_dict(normalized_prob=0.5)

        self.assertEqual(d['combined_score'], 5.0)
        self.assertEqual(d['normalized_prob'], 0.5)
        self.assertEqual(d['node_count'], 2)


class TestComputeExpScores(unittest.TestCase):
    """Tests for compute_exp_scores helper."""

    def test_empty_list(self):
        """Test with empty input."""
        scores, partition = compute_exp_scores([])
        self.assertEqual(scores, [])
        self.assertEqual(partition, 0.0)

    def test_single_score(self):
        """Test with single score."""
        scores, partition = compute_exp_scores([1.0])
        self.assertAlmostEqual(scores[0], math.exp(1.0), places=5)
        self.assertAlmostEqual(partition, math.exp(1.0), places=5)

    def test_multiple_scores(self):
        """Test with multiple scores."""
        raw = [1.0, 2.0, 3.0]
        scores, partition = compute_exp_scores(raw)

        # Verify partition is sum of exp scores
        self.assertAlmostEqual(partition, sum(scores), places=5)

        # Verify normalization: sum of (exp/partition) should be 1
        probs = [s / partition for s in scores]
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_numerical_stability(self):
        """Test with large scores (log-sum-exp trick)."""
        raw = [100.0, 101.0, 102.0]  # Large values
        scores, partition = compute_exp_scores(raw)

        # Should not overflow
        self.assertFalse(math.isinf(partition))
        for s in scores:
            self.assertFalse(math.isinf(s))


class TestComputeAnswerHash(unittest.TestCase):
    """Tests for compute_answer_hash helper."""

    def test_deterministic(self):
        """Test same input gives same hash."""
        h1 = compute_answer_hash("Test answer")
        h2 = compute_answer_hash("Test answer")
        self.assertEqual(h1, h2)

    def test_different_inputs(self):
        """Test different inputs give different hashes."""
        h1 = compute_answer_hash("Answer A")
        h2 = compute_answer_hash("Answer B")
        self.assertNotEqual(h1, h2)

    def test_hash_length(self):
        """Test hash is 16 characters."""
        h = compute_answer_hash("Test")
        self.assertEqual(len(h), 16)


class TestAggregationConfig(unittest.TestCase):
    """Tests for AggregationConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = AggregationConfig()
        self.assertEqual(config.strategy, AggregationStrategy.SUM)
        self.assertEqual(config.dedup_key, "answer_hash")
        self.assertIsNone(config.consensus_threshold)
        self.assertEqual(config.diversity_field, "corpus_id")


class TestCreateFederatedEngine(unittest.TestCase):
    """Tests for create_federated_engine factory."""

    def test_creates_engine(self):
        """Test factory creates engine with correct config."""
        from unifyweaver.targets.python_runtime.federated_query import FederatedQueryEngine

        discovery = LocalDiscoveryClient()
        router = KleinbergRouter('test-node', discovery)

        engine = create_federated_engine(
            router,
            strategy='max',
            federation_k=5,
            timeout_ms=3000,
            consensus_threshold=2
        )

        self.assertIsInstance(engine, FederatedQueryEngine)
        self.assertEqual(engine.federation_k, 5)
        self.assertEqual(engine.timeout_ms, 3000)
        self.assertEqual(engine.config.strategy, AggregationStrategy.MAX)
        self.assertEqual(engine.config.consensus_threshold, 2)


class TestFederatedQueryEngineAggregation(unittest.TestCase):
    """Tests for FederatedQueryEngine aggregation logic."""

    def setUp(self):
        """Set up test engine."""
        from unifyweaver.targets.python_runtime.federated_query import FederatedQueryEngine

        discovery = LocalDiscoveryClient()
        self.router = KleinbergRouter('test-node', discovery)
        self.engine = FederatedQueryEngine(
            router=self.router,
            federation_k=3
        )

    def test_aggregate_sum(self):
        """Test SUM aggregation boosts duplicates."""
        responses = [
            NodeResponse(
                source_node='node-a',
                results=[NodeResult(1, "Answer", "hash1", 0.8, 2.0)],
                partition_sum=2.0,
                node_metadata={'corpus_id': 'corpus_a'}
            ),
            NodeResponse(
                source_node='node-b',
                results=[NodeResult(2, "Answer", "hash1", 0.7, 1.5)],  # Same hash
                partition_sum=1.5,
                node_metadata={'corpus_id': 'corpus_b'}
            )
        ]

        aggregated, total_partition = self.engine._aggregate(
            responses, AggregationStrategy.SUM
        )

        # Should have one result (deduped)
        self.assertEqual(len(aggregated), 1)

        # Combined score should be sum: 2.0 + 1.5 = 3.5
        result = aggregated['hash1']
        self.assertEqual(result.combined_score, 3.5)

        # Should have both nodes as sources
        self.assertEqual(len(result.source_nodes), 2)

        # Total partition should be sum
        self.assertEqual(total_partition, 3.5)

    def test_aggregate_max(self):
        """Test MAX aggregation takes best, no boost."""
        responses = [
            NodeResponse(
                source_node='node-a',
                results=[NodeResult(1, "Answer", "hash1", 0.8, 2.0)],
                partition_sum=2.0,
                node_metadata={}
            ),
            NodeResponse(
                source_node='node-b',
                results=[NodeResult(2, "Answer", "hash1", 0.7, 1.5)],
                partition_sum=1.5,
                node_metadata={}
            )
        ]

        aggregated, _ = self.engine._aggregate(
            responses, AggregationStrategy.MAX
        )

        # Combined score should be max: max(2.0, 1.5) = 2.0
        result = aggregated['hash1']
        self.assertEqual(result.combined_score, 2.0)

    def test_aggregate_skips_errors(self):
        """Test aggregation skips error responses."""
        responses = [
            NodeResponse(
                source_node='node-a',
                results=[NodeResult(1, "Answer", "hash1", 0.8, 2.0)],
                partition_sum=2.0,
                node_metadata={},
                error=None
            ),
            NodeResponse(
                source_node='node-b',
                results=[],
                partition_sum=0.0,
                node_metadata={},
                error="Connection refused"
            )
        ]

        aggregated, total_partition = self.engine._aggregate(
            responses, AggregationStrategy.SUM
        )

        # Only node-a's result
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(total_partition, 2.0)

    def test_empty_response(self):
        """Test empty response generation."""
        response = self.engine._empty_response('test-id', AggregationStrategy.SUM)

        self.assertEqual(response.query_id, 'test-id')
        self.assertEqual(response.results, [])
        self.assertEqual(response.nodes_queried, 0)

    def test_get_stats(self):
        """Test statistics retrieval."""
        stats = self.engine.get_stats()

        self.assertEqual(stats['query_count'], 0)
        self.assertEqual(stats['federation_k'], 3)
        self.assertEqual(stats['aggregation_strategy'], 'sum')


if __name__ == '__main__':
    unittest.main()
