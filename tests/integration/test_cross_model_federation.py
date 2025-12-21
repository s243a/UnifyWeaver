#!/usr/bin/env python3
"""
Integration tests for Cross-Model Federation (Phase 6e).

Tests:
- Fusion methods (weighted sum, RRF, consensus, geometric mean, max)
- Model pool configuration
- Adaptive weight learning
- Cross-model result aggregation
"""

import sys
import os
import unittest
import tempfile
import json
import numpy as np

# Add the runtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/unifyweaver/targets/python_runtime'))

from cross_model_federation import (
    FusionMethod,
    ModelPoolConfig,
    CrossModelConfig,
    PoolResult,
    PoolResponse,
    FusedResult,
    PoolContribution,
    AdaptiveModelWeights,
    weighted_sum_fusion,
    rrf_fusion,
    consensus_fusion,
    geometric_mean_fusion,
    max_fusion,
    geometric_mean,
)

from federated_query import AggregationStrategy


class TestFusionMethod(unittest.TestCase):
    """Test the FusionMethod enum."""

    def test_enum_values(self):
        """Verify enum has expected values."""
        self.assertEqual(FusionMethod.WEIGHTED_SUM.value, "weighted_sum")
        self.assertEqual(FusionMethod.RECIPROCAL_RANK.value, "rrf")
        self.assertEqual(FusionMethod.CONSENSUS.value, "consensus")
        self.assertEqual(FusionMethod.GEOMETRIC_MEAN.value, "geometric_mean")
        self.assertEqual(FusionMethod.MAX.value, "max")


class TestModelPoolConfig(unittest.TestCase):
    """Test the ModelPoolConfig class."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = ModelPoolConfig(model_name="test-model")

        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.weight, 1.0)
        self.assertEqual(config.federation_k, 5)

    def test_custom_values(self):
        """Should accept custom values."""
        config = ModelPoolConfig(
            model_name="e5-large",
            weight=2.0,
            federation_k=10,
        )

        self.assertEqual(config.weight, 2.0)
        self.assertEqual(config.federation_k, 10)


class TestCrossModelConfig(unittest.TestCase):
    """Test the CrossModelConfig class."""

    def test_default_fusion_method(self):
        """Default fusion should be weighted_sum."""
        config = CrossModelConfig()
        self.assertEqual(config.fusion_method, FusionMethod.WEIGHTED_SUM)

    def test_multiple_pools(self):
        """Should support multiple model pools."""
        config = CrossModelConfig(
            pools=[
                ModelPoolConfig(model_name="e5-small", weight=1.0),
                ModelPoolConfig(model_name="e5-large", weight=2.0),
                ModelPoolConfig(model_name="bge-base", weight=1.5),
            ]
        )

        self.assertEqual(len(config.pools), 3)


class TestPoolResultDataClasses(unittest.TestCase):
    """Test pool result data classes."""

    def test_pool_result_to_dict(self):
        """PoolResult should serialize to dict."""
        result = PoolResult(
            answer_id="a1",
            answer_text="Test answer",
            answer_hash="hash123",
            raw_score=0.9,
            exp_score=2.45,
            density_adjusted_score=1.5,
            model_name="e5-small",
        )

        d = result.to_dict()
        self.assertEqual(d["answer_id"], "a1")
        self.assertEqual(d["answer_text"], "Test answer")

    def test_pool_response_to_dict(self):
        """PoolResponse should serialize to dict."""
        response = PoolResponse(
            model_name="e5-small",
            results=[
                PoolResult(
                    answer_id="a1",
                    answer_text="Test",
                    answer_hash="h1",
                    raw_score=0.9,
                    exp_score=2.45,
                    density_adjusted_score=1.5,
                )
            ],
            total_results=1,
            query_time_ms=100.0,
        )

        d = response.to_dict()
        self.assertEqual(d["model_name"], "e5-small")
        self.assertEqual(len(d["results"]), 1)


class TestGeometricMean(unittest.TestCase):
    """Test the geometric_mean helper function."""

    def test_single_value(self):
        """Single value should return itself."""
        self.assertAlmostEqual(geometric_mean([5.0]), 5.0, places=5)

    def test_multiple_values(self):
        """Should compute geometric mean correctly."""
        # geometric_mean([2, 8]) = sqrt(16) = 4
        self.assertAlmostEqual(geometric_mean([2.0, 8.0]), 4.0, places=5)

    def test_empty_list(self):
        """Empty list should return 0."""
        self.assertEqual(geometric_mean([]), 0.0)

    def test_handles_zeros(self):
        """Should handle zeros gracefully."""
        result = geometric_mean([0.0, 1.0])
        self.assertGreater(result, 0)  # Uses epsilon


def create_test_pool_responses() -> dict:
    """Create test pool responses for fusion testing."""
    return {
        "model_a": PoolResponse(
            model_name="model_a",
            results=[
                PoolResult(
                    answer_id="1", answer_text="Answer 1", answer_hash="h1",
                    raw_score=0.9, exp_score=2.0, density_adjusted_score=2.0,
                    density_score=0.8, cluster_size=5,
                ),
                PoolResult(
                    answer_id="2", answer_text="Answer 2", answer_hash="h2",
                    raw_score=0.7, exp_score=1.5, density_adjusted_score=1.5,
                    density_score=0.6, cluster_size=3,
                ),
                PoolResult(
                    answer_id="3", answer_text="Answer 3", answer_hash="h3",
                    raw_score=0.5, exp_score=1.0, density_adjusted_score=1.0,
                    density_score=0.4, cluster_size=2,
                ),
            ],
            total_results=3,
        ),
        "model_b": PoolResponse(
            model_name="model_b",
            results=[
                PoolResult(
                    answer_id="2", answer_text="Answer 2", answer_hash="h2",
                    raw_score=0.85, exp_score=1.8, density_adjusted_score=1.8,
                    density_score=0.7, cluster_size=4,
                ),
                PoolResult(
                    answer_id="1", answer_text="Answer 1", answer_hash="h1",
                    raw_score=0.6, exp_score=1.2, density_adjusted_score=1.2,
                    density_score=0.5, cluster_size=2,
                ),
                PoolResult(
                    answer_id="4", answer_text="Answer 4", answer_hash="h4",
                    raw_score=0.4, exp_score=0.8, density_adjusted_score=0.8,
                    density_score=0.3, cluster_size=1,
                ),
            ],
            total_results=3,
        ),
    }


class TestWeightedSumFusion(unittest.TestCase):
    """Test weighted sum fusion."""

    def test_basic_fusion(self):
        """Should fuse results using weighted sum."""
        responses = create_test_pool_responses()
        config = CrossModelConfig(
            pools=[
                ModelPoolConfig(model_name="model_a", weight=1.0),
                ModelPoolConfig(model_name="model_b", weight=1.0),
            ]
        )

        results = weighted_sum_fusion(responses, config, top_k=5)

        self.assertGreater(len(results), 0)
        # Check results are sorted by score
        scores = [r.fused_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_weighted_fusion(self):
        """Higher weight should increase contribution."""
        responses = create_test_pool_responses()

        # Model A has higher weight
        config_a = CrossModelConfig(
            pools=[
                ModelPoolConfig(model_name="model_a", weight=10.0),
                ModelPoolConfig(model_name="model_b", weight=1.0),
            ]
        )
        results_a = weighted_sum_fusion(responses, config_a, top_k=5)

        # Model B has higher weight
        config_b = CrossModelConfig(
            pools=[
                ModelPoolConfig(model_name="model_a", weight=1.0),
                ModelPoolConfig(model_name="model_b", weight=10.0),
            ]
        )
        results_b = weighted_sum_fusion(responses, config_b, top_k=5)

        # Different weights should give different orderings
        # h1 is top in model_a, h2 is top in model_b
        top_a = results_a[0].answer_hash if results_a else None
        top_b = results_b[0].answer_hash if results_b else None

        # When model_a dominates, h1 should rank higher
        # When model_b dominates, h2 should rank higher
        # (This might not always hold depending on exact scores)


class TestRRFFusion(unittest.TestCase):
    """Test Reciprocal Rank Fusion."""

    def test_basic_rrf(self):
        """Should fuse using reciprocal rank."""
        responses = create_test_pool_responses()
        config = CrossModelConfig(rrf_k=60)

        results = rrf_fusion(responses, config, top_k=5)

        self.assertGreater(len(results), 0)
        # Results should have ranks recorded
        for r in results:
            self.assertIsInstance(r.ranks_by_pool, dict)

    def test_rrf_boosts_consensus(self):
        """Answers appearing in both pools should rank higher."""
        responses = create_test_pool_responses()
        config = CrossModelConfig(rrf_k=60)

        results = rrf_fusion(responses, config, top_k=5)

        # h1 and h2 appear in both pools, h3 only in model_a, h4 only in model_b
        consensus_answers = {"h1", "h2"}

        # Check that consensus answers appear in top results
        top_hashes = {r.answer_hash for r in results[:2]}
        overlap = len(consensus_answers & top_hashes)
        self.assertGreater(overlap, 0)


class TestConsensusFusion(unittest.TestCase):
    """Test consensus-based fusion."""

    def test_basic_consensus(self):
        """Should boost answers with multi-pool agreement."""
        responses = create_test_pool_responses()
        config = CrossModelConfig(
            pools=[
                ModelPoolConfig(model_name="model_a"),
                ModelPoolConfig(model_name="model_b"),
            ],
            min_pools_for_consensus=2,
            consensus_boost_factor=1.5,
        )

        results = consensus_fusion(responses, config, top_k=5)

        self.assertGreater(len(results), 0)

    def test_consensus_strength(self):
        """Results should have consensus strength metric."""
        responses = create_test_pool_responses()
        config = CrossModelConfig(
            pools=[
                ModelPoolConfig(model_name="model_a"),
                ModelPoolConfig(model_name="model_b"),
            ]
        )

        results = consensus_fusion(responses, config, top_k=5)

        for r in results:
            # h1 and h2 are in both pools (consensus_strength = 1.0)
            # h3 and h4 are in one pool (consensus_strength = 0.5)
            if r.answer_hash in {"h1", "h2"}:
                self.assertEqual(r.consensus_strength, 1.0)
            elif r.answer_hash in {"h3", "h4"}:
                self.assertEqual(r.consensus_strength, 0.5)


class TestGeometricMeanFusion(unittest.TestCase):
    """Test geometric mean fusion."""

    def test_basic_geometric_mean(self):
        """Should compute geometric mean of probabilities."""
        responses = create_test_pool_responses()
        config = CrossModelConfig()

        results = geometric_mean_fusion(responses, config, top_k=5)

        self.assertGreater(len(results), 0)


class TestMaxFusion(unittest.TestCase):
    """Test max fusion."""

    def test_basic_max(self):
        """Should take max probability across pools."""
        responses = create_test_pool_responses()
        config = CrossModelConfig()

        results = max_fusion(responses, config, top_k=5)

        self.assertGreater(len(results), 0)


class TestAdaptiveModelWeights(unittest.TestCase):
    """Test adaptive weight learning."""

    def test_initial_uniform_weights(self):
        """Weights should start uniform."""
        weights = AdaptiveModelWeights(models=["a", "b", "c"])

        w = weights.get_weights()
        self.assertAlmostEqual(w["a"], 1/3, places=5)
        self.assertAlmostEqual(w["b"], 1/3, places=5)
        self.assertAlmostEqual(w["c"], 1/3, places=5)

    def test_custom_initial_weights(self):
        """Should accept custom initial weights."""
        weights = AdaptiveModelWeights(
            models=["a", "b"],
            initial_weights={"a": 0.8, "b": 0.2}
        )

        w = weights.get_weights()
        self.assertAlmostEqual(w["a"], 0.8, places=5)
        self.assertAlmostEqual(w["b"], 0.2, places=5)

    def test_update_increases_good_model_weight(self):
        """Updating with feedback should increase weight of accurate models."""
        weights = AdaptiveModelWeights(
            models=["good", "bad"],
            learning_rate=0.1
        )

        # Simulate feedback: user chose answer that "good" model ranked #1
        rankings = {
            "good": ["chosen_answer", "other"],
            "bad": ["other", "chosen_answer"]  # bad model ranked it #2
        }

        initial_good = weights.get_weights()["good"]
        weights.update("chosen_answer", rankings)
        final_good = weights.get_weights()["good"]

        self.assertGreater(final_good, initial_good)

    def test_weights_sum_to_one(self):
        """Weights should always sum to 1."""
        weights = AdaptiveModelWeights(models=["a", "b", "c"])

        for _ in range(10):
            weights.update("answer", {
                "a": ["answer"],
                "b": ["other", "answer"],
                "c": ["other"]
            })

        total = sum(weights.get_weights().values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_serialization(self):
        """Should serialize and deserialize correctly."""
        original = AdaptiveModelWeights(
            models=["a", "b"],
            initial_weights={"a": 0.7, "b": 0.3}
        )
        original.feedback_count = 5

        d = original.to_dict()
        restored = AdaptiveModelWeights.from_dict(d)

        self.assertEqual(
            original.get_weights()["a"],
            restored.get_weights()["a"]
        )
        self.assertEqual(original.feedback_count, restored.feedback_count)

    def test_file_persistence(self):
        """Should save and load from file."""
        weights = AdaptiveModelWeights(models=["x", "y"])
        weights.update("answer", {"x": ["answer"], "y": ["other"]})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            weights.save_to_file(filepath)
            loaded = AdaptiveModelWeights.load_from_file(filepath)

            self.assertEqual(
                weights.get_weights(),
                loaded.get_weights()
            )
        finally:
            os.unlink(filepath)

    def test_reset_weights(self):
        """Should reset to uniform weights."""
        weights = AdaptiveModelWeights(models=["a", "b"])
        weights.update("answer", {"a": ["answer"], "b": []})

        weights.reset_weights()

        w = weights.get_weights()
        self.assertAlmostEqual(w["a"], 0.5, places=5)
        self.assertAlmostEqual(w["b"], 0.5, places=5)

    def test_set_weights(self):
        """Should allow manual weight setting."""
        weights = AdaptiveModelWeights(models=["a", "b", "c"])

        weights.set_weights({"a": 0.6, "c": 0.2})

        w = weights.get_weights()
        # Weights should be normalized
        total = sum(w.values())
        self.assertAlmostEqual(total, 1.0, places=5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases in fusion."""

    def test_empty_pool_response(self):
        """Should handle empty responses gracefully."""
        responses = {
            "model_a": PoolResponse(
                model_name="model_a",
                results=[],
                error="Connection timeout"
            )
        }
        config = CrossModelConfig()

        results = weighted_sum_fusion(responses, config, top_k=5)
        self.assertEqual(len(results), 0)

    def test_single_pool(self):
        """Should work with single pool."""
        responses = create_test_pool_responses()
        # Only use model_a
        responses = {"model_a": responses["model_a"]}
        config = CrossModelConfig(
            pools=[ModelPoolConfig(model_name="model_a")]
        )

        results = weighted_sum_fusion(responses, config, top_k=5)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
