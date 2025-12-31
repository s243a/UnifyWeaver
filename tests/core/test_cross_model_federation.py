# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Unit tests for Cross-Model Federation (Phase 6e).
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/unifyweaver/targets/python_runtime'))

from cross_model_federation import (
    # Enums
    FusionMethod,
    # Config
    ModelPoolConfig,
    CrossModelConfig,
    # Data structures
    PoolResult,
    PoolResponse,
    PoolContribution,
    FusedResult,
    CrossModelResponse,
    # Fusion functions
    weighted_sum_fusion,
    rrf_fusion,
    consensus_fusion,
    geometric_mean_fusion,
    max_fusion,
    geometric_mean,
    # Engine
    PoolRouter,
    CrossModelFederatedEngine,
    # Weight learning
    AdaptiveModelWeights
)
from federated_query import AggregationStrategy


class TestFusionMethod(unittest.TestCase):
    """Test FusionMethod enum."""

    def test_enum_values(self):
        """All fusion methods have correct string values."""
        self.assertEqual(FusionMethod.WEIGHTED_SUM.value, "weighted_sum")
        self.assertEqual(FusionMethod.RECIPROCAL_RANK.value, "rrf")
        self.assertEqual(FusionMethod.CONSENSUS.value, "consensus")
        self.assertEqual(FusionMethod.GEOMETRIC_MEAN.value, "geometric_mean")
        self.assertEqual(FusionMethod.MAX.value, "max")


class TestModelPoolConfig(unittest.TestCase):
    """Test ModelPoolConfig data class."""

    def test_default_values(self):
        """Default values are correct."""
        config = ModelPoolConfig(model_name="test-model")
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.weight, 1.0)
        self.assertEqual(config.federation_k, 5)
        self.assertEqual(config.aggregation_strategy, AggregationStrategy.DENSITY_FLUX)
        self.assertFalse(config.use_adaptive_k)
        self.assertFalse(config.use_hierarchical)
        self.assertIsNone(config.node_filter)

    def test_custom_values(self):
        """Custom values are stored correctly."""
        config = ModelPoolConfig(
            model_name="custom-model",
            weight=0.5,
            federation_k=10,
            aggregation_strategy=AggregationStrategy.SUM
        )
        self.assertEqual(config.weight, 0.5)
        self.assertEqual(config.federation_k, 10)
        self.assertEqual(config.aggregation_strategy, AggregationStrategy.SUM)


class TestCrossModelConfig(unittest.TestCase):
    """Test CrossModelConfig data class."""

    def test_default_values(self):
        """Default values are correct."""
        config = CrossModelConfig()
        self.assertEqual(config.pools, [])
        self.assertEqual(config.fusion_method, FusionMethod.WEIGHTED_SUM)
        self.assertEqual(config.rrf_k, 60)
        self.assertEqual(config.consensus_threshold, 0.1)
        self.assertEqual(config.min_pools_for_consensus, 2)
        self.assertEqual(config.consensus_boost_factor, 1.5)
        self.assertEqual(config.pool_timeout_ms, 30000)

    def test_with_pools(self):
        """Configuration with pools."""
        config = CrossModelConfig(
            pools=[
                ModelPoolConfig(model_name="model-a", weight=0.6),
                ModelPoolConfig(model_name="model-b", weight=0.4)
            ],
            fusion_method=FusionMethod.CONSENSUS
        )
        self.assertEqual(len(config.pools), 2)
        self.assertEqual(config.fusion_method, FusionMethod.CONSENSUS)


class TestPoolResult(unittest.TestCase):
    """Test PoolResult data class."""

    def test_creation(self):
        """PoolResult can be created with required fields."""
        result = PoolResult(
            answer_id="ans_1",
            answer_text="Test answer",
            answer_hash="hash123",
            raw_score=0.8,
            exp_score=2.23,
            density_adjusted_score=1.89
        )
        self.assertEqual(result.answer_id, "ans_1")
        self.assertEqual(result.density_score, 0.0)  # default
        self.assertEqual(result.cluster_id, -1)  # default

    def test_to_dict(self):
        """to_dict serializes correctly."""
        result = PoolResult(
            answer_id="ans_1",
            answer_text="Test answer",
            answer_hash="hash123",
            raw_score=0.8,
            exp_score=2.23,
            density_adjusted_score=1.89,
            density_score=0.7,
            cluster_id=2,
            source_nodes=["node_a", "node_b"],
            model_name="test-model"
        )
        d = result.to_dict()
        self.assertEqual(d['answer_id'], "ans_1")
        self.assertEqual(d['density_score'], 0.7)
        self.assertEqual(d['cluster_id'], 2)
        self.assertEqual(d['model_name'], "test-model")


class TestPoolResponse(unittest.TestCase):
    """Test PoolResponse data class."""

    def test_creation(self):
        """PoolResponse can be created."""
        response = PoolResponse(
            model_name="test-model",
            results=[],
            total_results=0
        )
        self.assertEqual(response.model_name, "test-model")
        self.assertIsNone(response.error)

    def test_with_error(self):
        """PoolResponse can capture errors."""
        response = PoolResponse(
            model_name="test-model",
            results=[],
            error="Connection timeout"
        )
        self.assertEqual(response.error, "Connection timeout")


class TestFusedResult(unittest.TestCase):
    """Test FusedResult data class."""

    def test_creation(self):
        """FusedResult can be created."""
        result = FusedResult(
            answer_id="ans_1",
            answer_text="Test answer",
            answer_hash="hash123",
            fused_score=0.85,
            num_pools=3,
            consensus_strength=1.0
        )
        self.assertEqual(result.fused_score, 0.85)
        self.assertEqual(result.num_pools, 3)
        self.assertEqual(result.consensus_strength, 1.0)

    def test_to_dict(self):
        """to_dict serializes correctly."""
        result = FusedResult(
            answer_id="ans_1",
            answer_text="Test answer",
            answer_hash="hash123",
            fused_score=0.85,
            num_pools=2,
            consensus_strength=0.67,
            pool_contributions={
                "model-a": PoolContribution(
                    model_name="model-a",
                    probability=0.8,
                    density=0.7,
                    cluster_size=5,
                    rank=1
                )
            }
        )
        d = result.to_dict()
        self.assertEqual(d['fused_score'], 0.85)
        self.assertIn('model-a', d['pool_contributions'])


class TestGeometricMean(unittest.TestCase):
    """Test geometric_mean helper function."""

    def test_empty_list(self):
        """Empty list returns 0."""
        self.assertEqual(geometric_mean([]), 0.0)

    def test_single_value(self):
        """Single value returns itself."""
        self.assertAlmostEqual(geometric_mean([0.5]), 0.5)

    def test_multiple_values(self):
        """Multiple values compute correctly."""
        # geometric_mean([4, 9]) = sqrt(36) = 6
        # But we're dealing with probabilities, so:
        # geometric_mean([0.5, 0.5]) = 0.5
        self.assertAlmostEqual(geometric_mean([0.5, 0.5]), 0.5)
        # geometric_mean([0.25, 1.0]) = sqrt(0.25) = 0.5
        self.assertAlmostEqual(geometric_mean([0.25, 1.0]), 0.5)

    def test_handles_zero(self):
        """Zero values are handled (clamped to small epsilon)."""
        result = geometric_mean([0.0, 1.0])
        self.assertGreater(result, 0)
        self.assertLess(result, 0.1)  # Should be very small


class TestWeightedSumFusion(unittest.TestCase):
    """Test weighted_sum_fusion function."""

    def _make_pool_responses(self) -> Dict[str, PoolResponse]:
        """Create test pool responses."""
        return {
            "model-a": PoolResponse(
                model_name="model-a",
                results=[
                    PoolResult(
                        answer_id="1", answer_text="Answer 1", answer_hash="hash1",
                        raw_score=0.8, exp_score=2.23, density_adjusted_score=0.8,
                        density_score=0.7, cluster_size=5
                    ),
                    PoolResult(
                        answer_id="2", answer_text="Answer 2", answer_hash="hash2",
                        raw_score=0.6, exp_score=1.82, density_adjusted_score=0.6,
                        density_score=0.5, cluster_size=3
                    )
                ]
            ),
            "model-b": PoolResponse(
                model_name="model-b",
                results=[
                    PoolResult(
                        answer_id="1", answer_text="Answer 1", answer_hash="hash1",
                        raw_score=0.7, exp_score=2.01, density_adjusted_score=0.7,
                        density_score=0.6, cluster_size=4
                    ),
                    PoolResult(
                        answer_id="3", answer_text="Answer 3", answer_hash="hash3",
                        raw_score=0.5, exp_score=1.65, density_adjusted_score=0.5,
                        density_score=0.4, cluster_size=2
                    )
                ]
            )
        }

    def test_basic_fusion(self):
        """Basic weighted sum fusion works."""
        config = CrossModelConfig(
            pools=[
                ModelPoolConfig(model_name="model-a", weight=0.5),
                ModelPoolConfig(model_name="model-b", weight=0.5)
            ]
        )
        pool_responses = self._make_pool_responses()

        results = weighted_sum_fusion(pool_responses, config, top_k=5)

        self.assertGreater(len(results), 0)
        # hash1 appears in both pools, should have highest score
        self.assertEqual(results[0].answer_hash, "hash1")
        self.assertEqual(results[0].num_pools, 2)

    def test_respects_weights(self):
        """Fusion respects model weights."""
        config = CrossModelConfig(
            pools=[
                ModelPoolConfig(model_name="model-a", weight=0.9),
                ModelPoolConfig(model_name="model-b", weight=0.1)
            ]
        )
        pool_responses = self._make_pool_responses()

        results = weighted_sum_fusion(pool_responses, config, top_k=5)

        # hash1 is in both, but model-a weighted higher
        # Check that consensus_strength is correct
        self.assertEqual(results[0].num_pools, 2)

    def test_handles_empty_pool(self):
        """Handles pools with no results."""
        config = CrossModelConfig(
            pools=[
                ModelPoolConfig(model_name="model-a", weight=0.5),
                ModelPoolConfig(model_name="model-b", weight=0.5)
            ]
        )
        pool_responses = {
            "model-a": PoolResponse(model_name="model-a", results=[]),
            "model-b": self._make_pool_responses()["model-b"]
        }

        results = weighted_sum_fusion(pool_responses, config, top_k=5)
        self.assertGreater(len(results), 0)


class TestRRFFusion(unittest.TestCase):
    """Test rrf_fusion function."""

    def test_basic_rrf(self):
        """Basic RRF fusion works."""
        config = CrossModelConfig(rrf_k=60)
        pool_responses = {
            "model-a": PoolResponse(
                model_name="model-a",
                results=[
                    PoolResult(
                        answer_id="1", answer_text="A1", answer_hash="h1",
                        raw_score=0.8, exp_score=2.2, density_adjusted_score=0.8
                    ),
                    PoolResult(
                        answer_id="2", answer_text="A2", answer_hash="h2",
                        raw_score=0.6, exp_score=1.8, density_adjusted_score=0.6
                    )
                ]
            ),
            "model-b": PoolResponse(
                model_name="model-b",
                results=[
                    PoolResult(
                        answer_id="2", answer_text="A2", answer_hash="h2",
                        raw_score=0.9, exp_score=2.5, density_adjusted_score=0.9
                    ),
                    PoolResult(
                        answer_id="1", answer_text="A1", answer_hash="h1",
                        raw_score=0.7, exp_score=2.0, density_adjusted_score=0.7
                    )
                ]
            )
        }

        results = rrf_fusion(pool_responses, config, top_k=5)

        # Both answers appear in both pools
        self.assertEqual(len(results), 2)
        # h2 is rank 1 in model-b, h1 is rank 1 in model-a
        # RRF should give them similar scores since they trade top spots
        for r in results:
            self.assertEqual(r.num_pools, 2)

    def test_rrf_rank_tracking(self):
        """RRF tracks ranks by pool."""
        config = CrossModelConfig(rrf_k=60)
        pool_responses = {
            "model-a": PoolResponse(
                model_name="model-a",
                results=[
                    PoolResult(
                        answer_id="1", answer_text="A1", answer_hash="h1",
                        raw_score=0.8, exp_score=2.2, density_adjusted_score=0.8
                    )
                ]
            )
        }

        results = rrf_fusion(pool_responses, config, top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].ranks_by_pool["model-a"], 1)


class TestConsensusFusion(unittest.TestCase):
    """Test consensus_fusion function."""

    def test_consensus_boost(self):
        """Answers in multiple pools get boosted."""
        config = CrossModelConfig(
            pools=[
                ModelPoolConfig(model_name="model-a", weight=0.5),
                ModelPoolConfig(model_name="model-b", weight=0.5)
            ],
            consensus_threshold=0.1,
            min_pools_for_consensus=2,
            consensus_boost_factor=1.5
        )
        pool_responses = {
            "model-a": PoolResponse(
                model_name="model-a",
                results=[
                    PoolResult(
                        answer_id="1", answer_text="A1", answer_hash="h1",
                        raw_score=0.8, exp_score=2.2, density_adjusted_score=0.8,
                        density_score=0.7, cluster_size=5
                    )
                ]
            ),
            "model-b": PoolResponse(
                model_name="model-b",
                results=[
                    PoolResult(
                        answer_id="1", answer_text="A1", answer_hash="h1",
                        raw_score=0.7, exp_score=2.0, density_adjusted_score=0.7,
                        density_score=0.6, cluster_size=4
                    )
                ]
            )
        }

        results = consensus_fusion(pool_responses, config, top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].num_pools, 2)
        # Score should be boosted (base * boost)
        # Base = 0.5*1.0 + 0.5*1.0 = 1.0 (normalized within each pool)
        # Boost = 1 + 0.5 * geom_mean([0.7, 0.6])
        self.assertGreater(results[0].fused_score, 0)


class TestMaxFusion(unittest.TestCase):
    """Test max_fusion function."""

    def test_takes_maximum(self):
        """Max fusion takes highest probability."""
        config = CrossModelConfig()
        pool_responses = {
            "model-a": PoolResponse(
                model_name="model-a",
                results=[
                    PoolResult(
                        answer_id="1", answer_text="A1", answer_hash="h1",
                        raw_score=0.3, exp_score=1.35, density_adjusted_score=0.3
                    )
                ]
            ),
            "model-b": PoolResponse(
                model_name="model-b",
                results=[
                    PoolResult(
                        answer_id="1", answer_text="A1", answer_hash="h1",
                        raw_score=0.9, exp_score=2.46, density_adjusted_score=0.9
                    )
                ]
            )
        }

        results = max_fusion(pool_responses, config, top_k=5)

        self.assertEqual(len(results), 1)
        # Should take the higher normalized probability (from model-b)
        self.assertGreater(results[0].fused_score, 0.5)


class TestAdaptiveModelWeights(unittest.TestCase):
    """Test AdaptiveModelWeights class."""

    def test_initial_uniform_weights(self):
        """Weights start uniform."""
        weights = AdaptiveModelWeights(["a", "b", "c"])
        w = weights.get_weights()
        self.assertAlmostEqual(w["a"], 1/3, places=5)
        self.assertAlmostEqual(w["b"], 1/3, places=5)
        self.assertAlmostEqual(w["c"], 1/3, places=5)

    def test_custom_initial_weights(self):
        """Custom initial weights are respected."""
        weights = AdaptiveModelWeights(
            ["a", "b"],
            initial_weights={"a": 0.8, "b": 0.2}
        )
        w = weights.get_weights()
        self.assertAlmostEqual(w["a"], 0.8)
        self.assertAlmostEqual(w["b"], 0.2)

    def test_update_increases_winner_weight(self):
        """Update increases weight of model that ranked answer highest."""
        weights = AdaptiveModelWeights(["a", "b"], learning_rate=0.1)

        # Model 'a' ranked answer at position 1, model 'b' at position 5
        weights.update(
            chosen_answer="ans1",
            pool_rankings={
                "a": ["ans1", "ans2", "ans3"],
                "b": ["ans2", "ans3", "ans4", "ans5", "ans1"]
            }
        )

        w = weights.get_weights()
        # Model 'a' should have higher weight now
        self.assertGreater(w["a"], w["b"])

    def test_update_increments_count(self):
        """Feedback count is incremented."""
        weights = AdaptiveModelWeights(["a", "b"])
        self.assertEqual(weights.feedback_count, 0)

        weights.update("ans1", {"a": ["ans1"], "b": ["ans2", "ans1"]})

        self.assertEqual(weights.feedback_count, 1)

    def test_serialization(self):
        """Weights can be serialized and deserialized."""
        weights = AdaptiveModelWeights(["a", "b"])
        weights.update("ans1", {"a": ["ans1"], "b": ["ans2"]})

        d = weights.to_dict()
        restored = AdaptiveModelWeights.from_dict(d)

        self.assertEqual(restored.get_weights(), weights.get_weights())
        self.assertEqual(restored.feedback_count, weights.feedback_count)


class TestCrossModelFederatedEngine(unittest.TestCase):
    """Test CrossModelFederatedEngine class."""

    def _create_mock_router(self, nodes: List[Dict]) -> Mock:
        """Create a mock router with specified nodes."""
        mock_router = Mock()

        # Create mock KGNode objects
        mock_nodes = []
        for n in nodes:
            node = Mock()
            node.node_id = n['node_id']
            node.metadata = n.get('metadata', {})
            mock_nodes.append(node)

        mock_router.discover_nodes.return_value = mock_nodes
        mock_router.get_stats.return_value = {}
        return mock_router

    def test_discover_pools(self):
        """discover_pools groups nodes by embedding_model."""
        router = self._create_mock_router([
            {'node_id': 'n1', 'metadata': {'embedding_model': 'model-a'}},
            {'node_id': 'n2', 'metadata': {'embedding_model': 'model-a'}},
            {'node_id': 'n3', 'metadata': {'embedding_model': 'model-b'}},
            {'node_id': 'n4', 'metadata': {}},  # unknown model
        ])

        config = CrossModelConfig()
        engine = CrossModelFederatedEngine(router, config)

        pools = engine.discover_pools()

        self.assertEqual(len(pools['model-a']), 2)
        self.assertEqual(len(pools['model-b']), 1)
        self.assertEqual(len(pools['unknown']), 1)

    def test_get_stats(self):
        """get_stats returns engine statistics."""
        router = self._create_mock_router([])
        config = CrossModelConfig()
        engine = CrossModelFederatedEngine(router, config)

        stats = engine.get_stats()

        self.assertIn('queries_executed', stats)
        self.assertIn('avg_latency_ms', stats)
        self.assertIn('pools_configured', stats)

    def test_empty_pools_returns_empty_response(self):
        """Query with no pools returns empty response."""
        router = self._create_mock_router([])
        config = CrossModelConfig()
        engine = CrossModelFederatedEngine(router, config)

        response = engine.federated_query("test query")

        self.assertEqual(len(response.results), 0)
        self.assertEqual(response.pools_responded, 0)


class TestPoolRouter(unittest.TestCase):
    """Test PoolRouter class."""

    def test_filters_by_model(self):
        """PoolRouter only returns nodes with matching model."""
        base_router = Mock()

        node_a = Mock()
        node_a.node_id = 'n1'
        node_a.metadata = {'embedding_model': 'model-a'}

        node_b = Mock()
        node_b.node_id = 'n2'
        node_b.metadata = {'embedding_model': 'model-b'}

        base_router.discover_nodes.return_value = [node_a, node_b]

        pool_router = PoolRouter(base_router, 'model-a')
        nodes = pool_router.discover_nodes()

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].node_id, 'n1')

    def test_caches_filtered_nodes(self):
        """Filtered nodes are cached."""
        base_router = Mock()
        base_router.discover_nodes.return_value = []

        pool_router = PoolRouter(base_router, 'model-a')

        # Call twice
        pool_router.discover_nodes()
        pool_router.discover_nodes()

        # Base router should only be called once
        base_router.discover_nodes.assert_called_once()

    def test_custom_node_filter(self):
        """Custom node filter is applied."""
        base_router = Mock()

        node_1 = Mock()
        node_1.node_id = 'n1'
        node_1.metadata = {'embedding_model': 'model-a', 'region': 'us'}

        node_2 = Mock()
        node_2.node_id = 'n2'
        node_2.metadata = {'embedding_model': 'model-a', 'region': 'eu'}

        base_router.discover_nodes.return_value = [node_1, node_2]

        # Filter to only 'us' region
        pool_router = PoolRouter(
            base_router, 'model-a',
            node_filter=lambda n: n.metadata.get('region') == 'us'
        )
        nodes = pool_router.discover_nodes()

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].node_id, 'n1')


class TestAdaptiveModelWeightsPersistence(unittest.TestCase):
    """Test AdaptiveModelWeights persistence features."""

    def setUp(self):
        """Set up temp file for tests."""
        import tempfile
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        self.temp_path = self.temp_file.name

    def tearDown(self):
        """Clean up temp file."""
        import os
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_save_and_load(self):
        """Weights can be saved and loaded from file."""
        weights = AdaptiveModelWeights(["model-a", "model-b", "model-c"])
        weights.update("ans1", {"model-a": ["ans1"], "model-b": ["ans2", "ans1"]})

        # Save to file
        weights.save_to_file(self.temp_path)

        # Load from file
        loaded = AdaptiveModelWeights.load_from_file(self.temp_path)

        self.assertEqual(loaded.get_weights(), weights.get_weights())
        self.assertEqual(loaded.feedback_count, weights.feedback_count)
        self.assertEqual(loaded.learning_rate, weights.learning_rate)

    def test_reset_weights(self):
        """reset_weights restores uniform distribution."""
        weights = AdaptiveModelWeights(["a", "b"], initial_weights={"a": 0.8, "b": 0.2})
        weights.feedback_count = 10

        weights.reset_weights()

        self.assertAlmostEqual(weights.weights["a"], 0.5)
        self.assertAlmostEqual(weights.weights["b"], 0.5)
        self.assertEqual(weights.feedback_count, 0)

    def test_set_weights(self):
        """set_weights updates and normalizes."""
        weights = AdaptiveModelWeights(["a", "b", "c"])

        weights.set_weights({"a": 3.0, "b": 1.0})

        # Should be normalized: a=0.6, b=0.2 (c unchanged at ~0.333, then renormalized)
        total = sum(weights.weights.values())
        self.assertAlmostEqual(total, 1.0)

    def test_set_weights_ignores_unknown(self):
        """set_weights ignores unknown model names."""
        weights = AdaptiveModelWeights(["a", "b"])

        weights.set_weights({"a": 0.9, "unknown": 0.5})

        # unknown should be ignored
        self.assertNotIn("unknown", weights.weights)


class TestPrologValidation(unittest.TestCase):
    """Test Prolog validation predicates load correctly."""

    def test_prolog_module_loads(self):
        """Service validation module loads without errors."""
        import subprocess
        result = subprocess.run(
            ['swipl', '-g',
             "use_module('src/unifyweaver/core/service_validation'), "
             "is_valid_cross_model_option(fusion_method(consensus)), "
             "is_valid_pool_config(pool('test-model', [weight(0.5)])), "
             "halt."],
            capture_output=True,
            text=True,
            cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver'
        )
        # Should exit successfully (no ERROR in output)
        self.assertNotIn('ERROR:', result.stderr)


class TestGeometricMeanFusion(unittest.TestCase):
    """Test geometric_mean_fusion function."""

    def test_rewards_consistency(self):
        """Geometric mean rewards consistent probabilities across pools."""
        config = CrossModelConfig()
        # Answer with consistent high scores in both pools
        pool_responses = {
            "model-a": PoolResponse(
                model_name="model-a",
                results=[
                    PoolResult(
                        answer_id="1", answer_text="A1", answer_hash="h1",
                        raw_score=0.9, exp_score=2.5, density_adjusted_score=0.9
                    )
                ]
            ),
            "model-b": PoolResponse(
                model_name="model-b",
                results=[
                    PoolResult(
                        answer_id="1", answer_text="A1", answer_hash="h1",
                        raw_score=0.9, exp_score=2.5, density_adjusted_score=0.9
                    )
                ]
            )
        }

        results = geometric_mean_fusion(pool_responses, config, top_k=5)

        self.assertEqual(len(results), 1)
        # Both pools have same answer at 100% probability
        self.assertAlmostEqual(results[0].fused_score, 1.0)


if __name__ == '__main__':
    unittest.main()
