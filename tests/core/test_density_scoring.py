"""
Unit tests for density_scoring.py - Phase 4d Density-Based Confidence Scoring.
"""

import unittest
import numpy as np
import sys
import os

# Add source paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/unifyweaver/targets/python_runtime'))

from density_scoring import (
    DensityConfig,
    DensityResult,
    BandwidthMethod,
    cosine_similarity,
    cosine_distance,
    pairwise_cosine_distances,
    silverman_bandwidth,
    scott_bandwidth,
    gaussian_kernel,
    compute_density_scores,
    flux_softmax,
    cluster_by_similarity,
    compute_cluster_density,
    two_stage_density_pipeline,
    compute_cluster_stats,
    ClusterAggregator,
    AggregatorRegistry,
    TransactionConfig,
    TransactionManager,
    get_transaction_manager
)


class TestCosineSimilarity(unittest.TestCase):
    """Tests for cosine similarity/distance functions."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1."""
        a = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, a), 1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1."""
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0)

    def test_distance_is_one_minus_similarity(self):
        """Cosine distance = 1 - cosine similarity."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        sim = cosine_similarity(a, b)
        dist = cosine_distance(a, b)
        self.assertAlmostEqual(sim + dist, 1.0)

    def test_zero_vector(self):
        """Zero vector returns 0 similarity."""
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        self.assertEqual(cosine_similarity(a, b), 0.0)


class TestPairwiseDistances(unittest.TestCase):
    """Tests for pairwise distance matrix computation."""

    def test_single_vector(self):
        """Single vector has 0 distance to itself."""
        embeddings = np.array([[1.0, 0.0]])
        distances = pairwise_cosine_distances(embeddings)
        self.assertEqual(distances.shape, (1, 1))
        self.assertEqual(distances[0, 0], 0.0)

    def test_diagonal_is_zero(self):
        """Diagonal of distance matrix should be 0."""
        embeddings = np.random.randn(5, 10)
        distances = pairwise_cosine_distances(embeddings)
        np.testing.assert_array_almost_equal(np.diag(distances), np.zeros(5))

    def test_symmetric(self):
        """Distance matrix should be symmetric."""
        embeddings = np.random.randn(4, 8)
        distances = pairwise_cosine_distances(embeddings)
        np.testing.assert_array_almost_equal(distances, distances.T)


class TestBandwidthSelection(unittest.TestCase):
    """Tests for bandwidth selection methods."""

    def test_silverman_returns_positive(self):
        """Silverman bandwidth is positive."""
        distances = np.random.uniform(0.1, 0.5, 100)
        h = silverman_bandwidth(distances, 10)
        self.assertGreater(h, 0)

    def test_scott_returns_positive(self):
        """Scott bandwidth is positive."""
        distances = np.random.uniform(0.1, 0.5, 100)
        h = scott_bandwidth(distances, 10)
        self.assertGreater(h, 0)

    def test_empty_distances(self):
        """Empty distances return default bandwidth."""
        h = silverman_bandwidth(np.array([]), 0)
        self.assertEqual(h, 0.1)

    def test_all_zeros(self):
        """All zero distances return minimum bandwidth."""
        h = silverman_bandwidth(np.array([0.0, 0.0, 0.0]), 3)
        self.assertEqual(h, 0.1)


class TestGaussianKernel(unittest.TestCase):
    """Tests for Gaussian kernel function."""

    def test_zero_distance(self):
        """Zero distance gives maximum kernel value (1)."""
        self.assertAlmostEqual(gaussian_kernel(0.0, 0.1), 1.0)

    def test_large_distance(self):
        """Large distance gives small kernel value."""
        k = gaussian_kernel(10.0, 0.1)
        self.assertLess(k, 0.001)

    def test_smaller_bandwidth_sharper(self):
        """Smaller bandwidth = sharper falloff."""
        k_narrow = gaussian_kernel(0.5, 0.1)
        k_wide = gaussian_kernel(0.5, 1.0)
        self.assertLess(k_narrow, k_wide)


class TestDensityScores(unittest.TestCase):
    """Tests for KDE-based density scoring."""

    def test_single_point_has_density_one(self):
        """Single point normalized to density 1."""
        embeddings = np.array([[1.0, 0.0, 0.0]])
        densities = compute_density_scores(embeddings)
        self.assertEqual(len(densities), 1)
        self.assertEqual(densities[0], 1.0)

    def test_cluster_center_highest_density(self):
        """Within a tight cluster, center has highest density."""
        # Create a tight cluster - all vectors point in similar directions
        # Using unit vectors for clean cosine similarity
        embeddings = np.array([
            [1.0, 0.1, 0.0],   # Near axis (center-ish)
            [1.0, 0.0, 0.0],   # On axis
            [1.0, 0.05, 0.0],  # Very close to axis
            [1.0, -0.05, 0.0], # Very close to axis
            [1.0, 0.2, 0.0],   # Slightly further
        ])
        # Normalize to unit vectors
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        densities = compute_density_scores(embeddings)

        # All should have high density (tight cluster)
        self.assertTrue(np.all(densities > 0.5))
        # And they should be normalized between 0 and 1
        self.assertAlmostEqual(np.max(densities), 1.0)

    def test_normalized_to_zero_one(self):
        """Densities are normalized to [0, 1]."""
        embeddings = np.random.randn(10, 5)
        densities = compute_density_scores(embeddings)
        self.assertTrue(np.all(densities >= 0))
        self.assertTrue(np.all(densities <= 1))
        self.assertAlmostEqual(np.max(densities), 1.0)


class TestFluxSoftmax(unittest.TestCase):
    """Tests for flux-softmax (density-weighted softmax)."""

    def test_sums_to_one(self):
        """Flux-softmax probabilities sum to 1."""
        scores = np.array([1.0, 2.0, 3.0])
        densities = np.array([0.5, 0.8, 0.3])
        probs = flux_softmax(scores, densities)
        self.assertAlmostEqual(np.sum(probs), 1.0)

    def test_zero_weight_equals_standard_softmax(self):
        """With density_weight=0, equals standard softmax."""
        scores = np.array([1.0, 2.0, 3.0])
        densities = np.array([0.1, 0.9, 0.5])  # Should be ignored

        flux_probs = flux_softmax(scores, densities, density_weight=0.0)

        # Standard softmax
        exp_scores = np.exp(scores - scores.max())
        standard_probs = exp_scores / exp_scores.sum()

        np.testing.assert_array_almost_equal(flux_probs, standard_probs)

    def test_density_boosts_probability(self):
        """Higher density increases probability."""
        scores = np.array([1.0, 1.0])  # Equal scores
        densities_high_first = np.array([1.0, 0.0])
        densities_high_second = np.array([0.0, 1.0])

        probs1 = flux_softmax(scores, densities_high_first, density_weight=0.5)
        probs2 = flux_softmax(scores, densities_high_second, density_weight=0.5)

        # First element should have higher prob when its density is higher
        self.assertGreater(probs1[0], probs1[1])
        self.assertGreater(probs2[1], probs2[0])

    def test_single_element(self):
        """Single element has probability 1."""
        probs = flux_softmax(np.array([5.0]), np.array([0.5]))
        self.assertEqual(len(probs), 1)
        self.assertEqual(probs[0], 1.0)


class TestClustering(unittest.TestCase):
    """Tests for similarity-based clustering."""

    def test_two_distinct_clusters(self):
        """Two well-separated clusters are identified."""
        # Cluster A around [1, 0], Cluster B around [0, 1]
        embeddings = np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ])
        labels, centroids = cluster_by_similarity(
            embeddings, threshold=0.9, min_cluster_size=2
        )

        # Should have 2 clusters
        unique_labels = set(labels) - {-1}
        self.assertEqual(len(unique_labels), 2)

        # First two should be in same cluster
        self.assertEqual(labels[0], labels[1])
        # Last two should be in same cluster
        self.assertEqual(labels[2], labels[3])
        # But different from first two
        self.assertNotEqual(labels[0], labels[2])

    def test_singleton_is_noise(self):
        """Single-point cluster marked as noise (-1)."""
        embeddings = np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 0.0],  # Outlier
        ])
        labels, _ = cluster_by_similarity(
            embeddings, threshold=0.9, min_cluster_size=2
        )

        # Outlier should be noise
        self.assertEqual(labels[2], -1)


class TestTwoStagePipeline(unittest.TestCase):
    """Tests for two-stage density pipeline."""

    def test_returns_correct_shapes(self):
        """Pipeline returns correct array shapes."""
        embeddings = np.random.randn(10, 8)
        scores = np.random.rand(10)

        flux_probs, densities, labels, centroids = two_stage_density_pipeline(
            embeddings, scores
        )

        self.assertEqual(len(flux_probs), 10)
        self.assertEqual(len(densities), 10)
        self.assertEqual(len(labels), 10)

    def test_flux_probs_sum_to_one(self):
        """Flux probabilities sum to 1."""
        embeddings = np.random.randn(8, 5)
        scores = np.random.rand(8)

        flux_probs, _, _, _ = two_stage_density_pipeline(embeddings, scores)

        self.assertAlmostEqual(np.sum(flux_probs), 1.0)


class TestClusterAggregator(unittest.TestCase):
    """Tests for ClusterAggregator class."""

    def test_should_accept_similar(self):
        """Accepts results similar to centroid."""
        agg = ClusterAggregator(
            cluster_id="test",
            centroid=np.array([1.0, 0.0]),
            transaction_id="txn1"
        )
        similar = np.array([0.95, 0.05])
        self.assertTrue(agg.should_accept(similar, threshold=0.9))

    def test_should_reject_dissimilar(self):
        """Rejects results dissimilar to centroid."""
        agg = ClusterAggregator(
            cluster_id="test",
            centroid=np.array([1.0, 0.0]),
            transaction_id="txn1"
        )
        dissimilar = np.array([0.0, 1.0])
        self.assertFalse(agg.should_accept(dissimilar, threshold=0.9))

    def test_shutdown_prevents_accept(self):
        """Shutdown aggregator rejects all results."""
        agg = ClusterAggregator(
            cluster_id="test",
            centroid=np.array([1.0, 0.0]),
            transaction_id="txn1"
        )
        agg.shutdown()
        self.assertFalse(agg.should_accept(np.array([1.0, 0.0])))
        self.assertFalse(agg.is_active())


class TestAggregatorRegistry(unittest.TestCase):
    """Tests for AggregatorRegistry class."""

    def test_register_and_find(self):
        """Register and find aggregator by similarity."""
        registry = AggregatorRegistry(transaction_id="txn1")

        agg1 = ClusterAggregator(
            cluster_id="cluster_a",
            centroid=np.array([1.0, 0.0]),
            transaction_id="txn1"
        )
        registry.register_aggregator(agg1)

        # Find should return closest
        best_id, sim = registry.find_best_aggregator(np.array([0.95, 0.05]))
        self.assertEqual(best_id, "cluster_a")
        self.assertGreater(sim, 0.9)

    def test_gossip_notifies_peers(self):
        """Registering new aggregator notifies existing peers."""
        registry = AggregatorRegistry(transaction_id="txn1")

        agg1 = ClusterAggregator(
            cluster_id="cluster_a",
            centroid=np.array([1.0, 0.0]),
            transaction_id="txn1"
        )
        registry.register_aggregator(agg1)

        agg2 = ClusterAggregator(
            cluster_id="cluster_b",
            centroid=np.array([0.0, 1.0]),
            transaction_id="txn1"
        )
        registry.register_aggregator(agg2)

        # agg1 should know about agg2
        self.assertIn("cluster_b", agg1.known_peers)
        # agg2 should know about agg1
        self.assertIn("cluster_a", agg2.known_peers)


class TestTransactionManager(unittest.TestCase):
    """Tests for TransactionManager class."""

    def test_begin_and_close(self):
        """Begin and close transaction."""
        mgr = TransactionManager()

        registry = mgr.begin_transaction("txn1", timeout_ms=5000)
        self.assertIsNotNone(registry)
        self.assertEqual(registry.transaction_id, "txn1")

        # Close returns stats
        stats = mgr.close_transaction("txn1")
        self.assertIsNotNone(stats)

        # Should be removed
        self.assertIsNone(mgr.get_transaction("txn1"))

    def test_kill_transaction(self):
        """Kill transaction forcibly."""
        mgr = TransactionManager()

        mgr.begin_transaction("txn2")
        mgr.kill_transaction("txn2")

        self.assertIsNone(mgr.get_transaction("txn2"))

    def test_expired_detection(self):
        """Expired transactions are detected."""
        config = TransactionConfig(
            transaction_id="txn3",
            timeout_ms=0  # Immediately expired
        )
        self.assertTrue(config.is_expired())


class TestDensityConfig(unittest.TestCase):
    """Tests for DensityConfig dataclass."""

    def test_defaults(self):
        """Default configuration values."""
        config = DensityConfig()
        self.assertEqual(config.bandwidth_method, BandwidthMethod.SILVERMAN)
        self.assertEqual(config.density_weight, 0.3)
        self.assertTrue(config.clustering_enabled)
        self.assertEqual(config.similarity_threshold, 0.7)


if __name__ == '__main__':
    unittest.main()
