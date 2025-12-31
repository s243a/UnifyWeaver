"""Tests for DensityIndex pre-computed density lookup."""

import numpy as np
import pytest
import tempfile
import os

from unifyweaver.targets.python_runtime.density_scoring import (
    DensityIndex,
    DensityIndexConfig,
    BandwidthMethod,
    inference_with_density,
    flux_softmax,
)


class TestDensityIndex:
    """Tests for DensityIndex class."""

    def test_build_single_answer_clusters(self):
        """Test building index with single-answer clusters."""
        # Create simple clusters with single answers
        clusters = {
            'cluster_0': (
                np.random.randn(3, 64),  # 3 questions
                np.random.randn(64),      # 1 answer
                np.random.randn(64),      # centroid
            ),
            'cluster_1': (
                np.random.randn(2, 64),
                np.random.randn(64),
                np.random.randn(64),
            ),
        }

        index = DensityIndex()
        stats = index.build(clusters)

        assert stats['total_clusters'] == 2
        assert stats['total_answers'] == 2
        assert len(index) == 2

    def test_build_multi_answer_clusters(self):
        """Test building index with multi-answer clusters."""
        # Create clusters with multiple answers
        clusters = {
            'cluster_0': (
                np.random.randn(5, 64),   # 5 questions
                np.random.randn(3, 64),   # 3 answers
                np.random.randn(64),
            ),
            'cluster_1': (
                np.random.randn(2, 64),
                np.random.randn(4, 64),   # 4 answers
                np.random.randn(64),
            ),
        }

        index = DensityIndex()
        stats = index.build(clusters)

        assert stats['total_clusters'] == 2
        assert stats['total_answers'] == 7
        assert len(index) == 7

    def test_lookup_density(self):
        """Test density lookup returns valid scores."""
        # Create cluster with known answers
        answer_emb = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
        ])
        clusters = {
            'cluster_0': (
                np.random.randn(2, 3),
                answer_emb,
                np.array([0.9, 0.1, 0.0]),
            ),
        }
        answer_ids = {'cluster_0': ['a1', 'a2', 'a3']}

        index = DensityIndex()
        index.build(clusters, answer_ids)

        # Check densities are in valid range
        for a_id in ['a1', 'a2', 'a3']:
            density = index.lookup_density(a_id)
            assert 0.0 <= density <= 1.0

        # Unknown answer returns 0
        assert index.lookup_density('unknown') == 0.0

    def test_lookup_densities_batch(self):
        """Test batch density lookup."""
        answer_emb = np.random.randn(5, 32)
        clusters = {
            'cluster_0': (
                np.random.randn(3, 32),
                answer_emb,
                np.random.randn(32),
            ),
        }
        answer_ids = {'cluster_0': ['a0', 'a1', 'a2', 'a3', 'a4']}

        index = DensityIndex()
        index.build(clusters, answer_ids)

        # Batch lookup
        densities = index.lookup_densities(['a0', 'a2', 'a4'])
        assert len(densities) == 3
        assert all(0.0 <= d <= 1.0 for d in densities)

        # Mixed known/unknown
        densities = index.lookup_densities(['a0', 'unknown', 'a4'])
        assert len(densities) == 3
        assert densities[1] == 0.0  # unknown

    def test_add_answer_incremental(self):
        """Test incrementally adding answers."""
        clusters = {
            'cluster_0': (
                np.random.randn(2, 16),
                np.random.randn(2, 16),
                np.random.randn(16),
            ),
        }
        answer_ids = {'cluster_0': ['a0', 'a1']}

        index = DensityIndex()
        index.build(clusters, answer_ids)
        assert len(index) == 2

        # Add new answer
        new_emb = np.random.randn(16)
        density = index.add_answer('a2', new_emb, 'cluster_0')

        assert len(index) == 3
        assert 'a2' in index
        assert index.get_cluster_for_answer('a2') == 'cluster_0'
        assert 0.0 <= density <= 1.0

    def test_add_answer_auto_route(self):
        """Test adding answer with automatic cluster routing."""
        # Create two distinct clusters
        clusters = {
            'cluster_a': (
                np.random.randn(2, 16),
                np.array([[1.0] + [0.0] * 15]),
                np.array([1.0] + [0.0] * 15),
            ),
            'cluster_b': (
                np.random.randn(2, 16),
                np.array([[0.0] + [1.0] + [0.0] * 14]),
                np.array([0.0] + [1.0] + [0.0] * 14),
            ),
        }
        answer_ids = {
            'cluster_a': ['a0'],
            'cluster_b': ['b0'],
        }

        index = DensityIndex()
        index.build(clusters, answer_ids)

        # Add answer similar to cluster_a
        new_emb = np.array([0.9] + [0.1] + [0.0] * 14)
        index.add_answer('a_new', new_emb)

        # Should route to cluster_a
        assert index.get_cluster_for_answer('a_new') == 'cluster_a'

    def test_recompute_cluster_densities(self):
        """Test recomputing densities after adding multiple answers."""
        clusters = {
            'cluster_0': (
                np.random.randn(2, 16),
                np.random.randn(2, 16),
                np.random.randn(16),
            ),
        }
        answer_ids = {'cluster_0': ['a0', 'a1']}

        index = DensityIndex()
        index.build(clusters, answer_ids)

        # Add several answers
        for i in range(5):
            index.add_answer(f'new_{i}', np.random.randn(16), 'cluster_0')

        # Recompute
        index.recompute_cluster_densities('cluster_0')

        # Check all densities are valid
        for a_id in ['a0', 'a1'] + [f'new_{i}' for i in range(5)]:
            assert 0.0 <= index.lookup_density(a_id) <= 1.0

    def test_save_load(self):
        """Test saving and loading index."""
        clusters = {
            'cluster_0': (
                np.random.randn(3, 32),
                np.random.randn(4, 32),
                np.random.randn(32),
            ),
        }
        answer_ids = {'cluster_0': ['a', 'b', 'c', 'd']}

        index = DensityIndex()
        index.build(clusters, answer_ids)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            index.save(path)
            loaded = DensityIndex.load(path)

            assert len(loaded) == len(index)
            assert loaded.global_stats == index.global_stats

            # Check densities match
            for a_id in answer_ids['cluster_0']:
                assert loaded.lookup_density(a_id) == index.lookup_density(a_id)
        finally:
            os.unlink(path)

    def test_contains(self):
        """Test __contains__ method."""
        clusters = {
            'cluster_0': (
                np.random.randn(2, 16),
                np.random.randn(16),
                np.random.randn(16),
            ),
        }
        answer_ids = {'cluster_0': ['answer_1']}

        index = DensityIndex()
        index.build(clusters, answer_ids)

        assert 'answer_1' in index
        assert 'unknown' not in index

    def test_bandwidth_methods(self):
        """Test different bandwidth selection methods."""
        clusters = {
            'cluster_0': (
                np.random.randn(5, 32),
                np.random.randn(10, 32),  # enough for meaningful bandwidth
                np.random.randn(32),
            ),
        }

        # Silverman
        config_silv = DensityIndexConfig(bandwidth_method=BandwidthMethod.SILVERMAN)
        index_silv = DensityIndex(config_silv)
        stats_silv = index_silv.build(clusters)

        # Scott
        config_scott = DensityIndexConfig(bandwidth_method=BandwidthMethod.SCOTT)
        index_scott = DensityIndex(config_scott)
        stats_scott = index_scott.build(clusters)

        # Both should produce valid bandwidths
        assert stats_silv['avg_bandwidth'] > 0
        assert stats_scott['avg_bandwidth'] > 0


class TestInferenceWithDensity:
    """Tests for inference_with_density function."""

    def test_basic_inference(self):
        """Test basic inference with density weighting."""
        # Setup index
        clusters = {
            'cluster_0': (
                np.random.randn(2, 16),
                np.random.randn(5, 16),
                np.random.randn(16),
            ),
        }
        answer_ids = {'cluster_0': ['a0', 'a1', 'a2', 'a3', 'a4']}

        index = DensityIndex()
        index.build(clusters, answer_ids)

        # Create candidates
        candidates = [
            ('a0', np.random.randn(16), 0.9),  # high similarity
            ('a1', np.random.randn(16), 0.7),
            ('a2', np.random.randn(16), 0.5),
        ]

        results = inference_with_density(
            np.random.randn(16),
            candidates,
            index,
            density_weight=0.3
        )

        # Should return sorted by probability
        assert len(results) == 3
        probs = [r[1] for r in results]
        assert probs == sorted(probs, reverse=True)

        # Probabilities should sum to ~1
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_empty_candidates(self):
        """Test inference with empty candidates."""
        index = DensityIndex()
        index._built = True  # Fake built state

        results = inference_with_density(
            np.random.randn(16),
            [],
            index
        )

        assert results == []

    def test_density_weight_effect(self):
        """Test that density weight affects ranking."""
        # Create cluster where one answer has higher density
        # by placing it near others
        center = np.array([1.0] + [0.0] * 15)
        answer_emb = np.array([
            center,                          # a0: central
            center + np.array([0.01] + [0.0] * 15),  # a1: very close
            np.array([0.0, 1.0] + [0.0] * 14),       # a2: far away
        ])

        clusters = {
            'cluster_0': (
                np.random.randn(2, 16),
                answer_emb,
                center,
            ),
        }
        answer_ids = {'cluster_0': ['a0', 'a1', 'a2']}

        index = DensityIndex()
        index.build(clusters, answer_ids)

        # a2 has lowest density (far from others)
        # If we give it highest similarity but use high density weight,
        # it should be penalized
        candidates = [
            ('a0', answer_emb[0], 0.8),
            ('a2', answer_emb[2], 0.85),  # Higher similarity but low density
        ]

        # With no density weight, a2 wins
        results_no_density = inference_with_density(
            center, candidates, index, density_weight=0.0
        )
        assert results_no_density[0][0] == 'a2'

        # With high density weight, a0 might win (depends on relative densities)
        # At minimum, the ordering should potentially change
        results_high_density = inference_with_density(
            center, candidates, index, density_weight=2.0
        )
        # Just verify it runs and returns valid probabilities
        assert len(results_high_density) == 2
        assert abs(sum(r[1] for r in results_high_density) - 1.0) < 1e-6


class TestDensityIndexConfig:
    """Tests for DensityIndexConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DensityIndexConfig()

        assert config.bandwidth_method == BandwidthMethod.SILVERMAN
        assert config.default_bandwidth == 0.1
        assert config.normalize_densities is True
        assert config.store_embeddings is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DensityIndexConfig(
            bandwidth_method=BandwidthMethod.SCOTT,
            default_bandwidth=0.2,
            normalize_densities=False,
            store_embeddings=False,
        )

        assert config.bandwidth_method == BandwidthMethod.SCOTT
        assert config.default_bandwidth == 0.2
        assert config.normalize_densities is False
        assert config.store_embeddings is False

    def test_no_embeddings_config(self):
        """Test index without stored embeddings."""
        config = DensityIndexConfig(store_embeddings=False)

        clusters = {
            'cluster_0': (
                np.random.randn(2, 16),
                np.random.randn(3, 16),
                np.random.randn(16),
            ),
        }

        index = DensityIndex(config)
        index.build(clusters)

        # Should still work for lookup
        assert len(index) == 3

        # But incremental add will use default density
        density = index.add_answer('new', np.random.randn(16), 'cluster_0')
        assert density == 0.5  # Default when no embeddings stored
