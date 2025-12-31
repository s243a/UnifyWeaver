"""
Tests for adversarial robustness module.

Tests output smoothing (soft collisions), semantic collision detection (hard collisions),
consensus voting, and trust-weighted consensus.
"""

import unittest
import numpy as np
from dataclasses import dataclass
from typing import Optional

from adversarial_robustness import (
    # Data structures
    EstablishedAnswer,
    VersionedAnswer,
    TwoDimensionalTrust,
    NodeTrust,
    # Output smoothing
    smooth_outliers,
    robust_aggregate,
    winsorize_scores,
    OutlierSmoother,
    # Collision detection
    SemanticCollisionDetector,
    ConsensusCollisionDetector,
    # Trust management
    DirectTrustManager,
    TrustWeightedConsensusDetector,
    # Pipeline
    AdversarialPipeline,
    create_adversarial_pipeline,
)


# =============================================================================
# TEST DATA STRUCTURES
# =============================================================================

class TestEstablishedAnswer(unittest.TestCase):
    """Tests for EstablishedAnswer dataclass."""

    def test_should_lock_meets_criteria(self):
        """Answer should lock when confidence and node count meet thresholds."""
        answer = EstablishedAnswer(
            answer_hash="abc123",
            answer_text="Test answer",
            semantic_region="region1",
            confidence=0.9,
            first_seen=1000.0,
            node_count=5,
            locked=False
        )
        self.assertTrue(answer.should_lock(0.8, 3))

    def test_should_lock_low_confidence(self):
        """Answer should not lock with low confidence."""
        answer = EstablishedAnswer(
            answer_hash="abc123",
            answer_text="Test answer",
            semantic_region="region1",
            confidence=0.5,
            first_seen=1000.0,
            node_count=5,
            locked=False
        )
        self.assertFalse(answer.should_lock(0.8, 3))

    def test_should_lock_low_node_count(self):
        """Answer should not lock with insufficient nodes."""
        answer = EstablishedAnswer(
            answer_hash="abc123",
            answer_text="Test answer",
            semantic_region="region1",
            confidence=0.9,
            first_seen=1000.0,
            node_count=2,
            locked=False
        )
        self.assertFalse(answer.should_lock(0.8, 3))


class TestTwoDimensionalTrust(unittest.TestCase):
    """Tests for FMS-style two-dimensional trust."""

    def test_effective_trust(self):
        """Effective trust is average of both dimensions."""
        trust = TwoDimensionalTrust(message_trust=80, trust_list_trust=40)
        self.assertEqual(trust.effective_trust(), 60)

    def test_normalized_trust(self):
        """Normalized trust maps -100..+100 to 0..1."""
        trust = TwoDimensionalTrust(message_trust=0, trust_list_trust=100)
        self.assertAlmostEqual(trust.normalized_message_trust(), 0.5)
        self.assertAlmostEqual(trust.normalized_trust_list_trust(), 1.0)

    def test_negative_trust(self):
        """Negative trust values work correctly."""
        trust = TwoDimensionalTrust(message_trust=-100, trust_list_trust=-50)
        self.assertEqual(trust.effective_trust(), -75)
        self.assertAlmostEqual(trust.normalized_message_trust(), 0.0)


# =============================================================================
# TEST OUTPUT SMOOTHING (SOFT COLLISIONS)
# =============================================================================

class TestOutputSmoothing(unittest.TestCase):
    """Tests for outlier rejection (soft collision detection)."""

    def test_zscore_outlier_rejection(self):
        """Z-score method rejects statistical outliers."""
        # Normal distribution with one extreme outlier
        scores = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 10.0])  # 10.0 is outlier
        mask = smooth_outliers(scores, method="zscore", threshold=2.0)

        self.assertEqual(mask.sum(), 6)  # 6 kept
        self.assertFalse(mask[-1])  # Last value (10.0) rejected

    def test_mad_outlier_rejection(self):
        """MAD method is robust to multiple outliers."""
        # Two outliers
        scores = np.array([1.0, 1.1, 0.9, 1.0, 10.0, 20.0])
        mask = smooth_outliers(scores, method="mad", threshold=2.5)

        self.assertFalse(mask[-1])  # 20.0 rejected
        self.assertFalse(mask[-2])  # 10.0 rejected

    def test_iqr_outlier_rejection(self):
        """IQR method handles skewed distributions."""
        scores = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 5.0])
        mask = smooth_outliers(scores, method="iqr", threshold=1.5)

        self.assertFalse(mask[-1])  # 5.0 rejected

    def test_no_outliers(self):
        """All values kept when no outliers present."""
        scores = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        mask = smooth_outliers(scores, method="mad", threshold=2.5)

        self.assertTrue(mask.all())

    def test_small_sample(self):
        """Small samples are not filtered."""
        scores = np.array([1.0, 10.0])  # Too small for reliable detection
        mask = smooth_outliers(scores, method="zscore", threshold=2.0)

        self.assertTrue(mask.all())  # All kept


class TestRobustAggregate(unittest.TestCase):
    """Tests for trimmed mean aggregation."""

    def test_trimmed_mean(self):
        """Trimmed mean removes extreme values."""
        scores = np.array([1.0, 1.1, 1.2, 1.0, 100.0])  # 100 is extreme
        result = robust_aggregate(scores, trim_fraction=0.2)

        # 20% trim removes the extreme
        self.assertLess(result, 2.0)

    def test_empty_array(self):
        """Empty array returns 0."""
        result = robust_aggregate(np.array([]))
        self.assertEqual(result, 0.0)


class TestOutlierSmoother(unittest.TestCase):
    """Tests for OutlierSmoother class."""

    def test_filter_results(self):
        """Filter results removes outliers."""
        @dataclass
        class MockResult:
            density_adjusted_score: float

        results = [
            MockResult(1.0),
            MockResult(1.1),
            MockResult(0.9),
            MockResult(10.0),  # Outlier
        ]

        smoother = OutlierSmoother(method="mad", threshold=2.5)
        filtered = smoother.filter_results(results)

        self.assertEqual(len(filtered), 3)

    def test_min_samples_threshold(self):
        """Results below min_samples are not filtered."""
        @dataclass
        class MockResult:
            density_adjusted_score: float

        results = [MockResult(1.0), MockResult(100.0)]

        smoother = OutlierSmoother(min_samples=3)
        filtered = smoother.filter_results(results)

        self.assertEqual(len(filtered), 2)  # All kept


# =============================================================================
# TEST SEMANTIC COLLISION DETECTOR (HARD COLLISIONS)
# =============================================================================

class TestSemanticCollisionDetector(unittest.TestCase):
    """Tests for KSK-style collision detection."""

    def setUp(self):
        self.detector = SemanticCollisionDetector(
            lock_threshold=0.8,
            min_nodes_to_lock=3,
            region_granularity=100
        )

    def test_region_computation_deterministic(self):
        """Same embedding produces same region hash."""
        embedding = np.array([0.1, 0.2, 0.3])

        region1 = self.detector._compute_region(embedding)
        region2 = self.detector._compute_region(embedding)

        self.assertEqual(region1, region2)

    def test_region_computation_different_embeddings(self):
        """Different embeddings produce different regions (usually)."""
        emb1 = np.array([0.1, 0.2, 0.3])
        emb2 = np.array([0.9, 0.8, 0.7])

        region1 = self.detector._compute_region(emb1)
        region2 = self.detector._compute_region(emb2)

        self.assertNotEqual(region1, region2)

    def test_first_answer_establishes(self):
        """First answer in a region becomes established."""
        embedding = np.array([0.1, 0.2, 0.3])

        self.detector.register_result(
            answer_hash="hash1",
            answer_text="Answer 1",
            embedding=embedding,
            confidence=0.9,
            node_count=5
        )

        established = self.detector.get_established(embedding)
        self.assertIsNotNone(established)
        self.assertEqual(established.answer_hash, "hash1")
        self.assertTrue(established.locked)  # Meets threshold

    def test_same_answer_reinforces(self):
        """Same answer reinforces existing establishment."""
        embedding = np.array([0.1, 0.2, 0.3])

        self.detector.register_result(
            answer_hash="hash1",
            answer_text="Answer 1",
            embedding=embedding,
            confidence=0.7,
            node_count=2
        )

        self.detector.register_result(
            answer_hash="hash1",
            answer_text="Answer 1",
            embedding=embedding,
            confidence=0.9,
            node_count=5
        )

        established = self.detector.get_established(embedding)
        self.assertEqual(established.confidence, 0.9)
        self.assertEqual(established.node_count, 5)

    def test_different_answer_rejected_in_locked_region(self):
        """Different answer is rejected in locked region."""
        embedding = np.array([0.1, 0.2, 0.3])

        # Establish and lock first answer
        self.detector.register_result(
            answer_hash="hash1",
            answer_text="Answer 1",
            embedding=embedding,
            confidence=0.9,
            node_count=5
        )

        # Try to check a different answer
        allowed, reason = self.detector.check_collision("hash2", embedding)

        self.assertFalse(allowed)
        self.assertIn("Collision", reason)

    def test_unlocked_region_accepts_different(self):
        """Unlocked region accepts different answers."""
        embedding = np.array([0.1, 0.2, 0.3])

        # Establish but don't lock (low confidence)
        self.detector.register_result(
            answer_hash="hash1",
            answer_text="Answer 1",
            embedding=embedding,
            confidence=0.5,
            node_count=1
        )

        # Different answer should be allowed
        allowed, reason = self.detector.check_collision("hash2", embedding)

        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_same_answer_allowed_in_locked_region(self):
        """Same answer is allowed in locked region (reinforces)."""
        embedding = np.array([0.1, 0.2, 0.3])

        self.detector.register_result(
            answer_hash="hash1",
            answer_text="Answer 1",
            embedding=embedding,
            confidence=0.9,
            node_count=5
        )

        allowed, reason = self.detector.check_collision("hash1", embedding)

        self.assertTrue(allowed)


# =============================================================================
# TEST CONSENSUS COLLISION DETECTOR
# =============================================================================

class TestConsensusCollisionDetector(unittest.TestCase):
    """Tests for consensus-based collision detection."""

    def setUp(self):
        self.detector = ConsensusCollisionDetector(
            quorum=3,
            supersede_margin=2,
            lock_threshold=0.8,
            min_nodes_to_lock=1
        )

    def test_quorum_required_to_lock(self):
        """Region not locked until quorum is reached."""
        embedding = np.array([0.1, 0.2, 0.3])

        # Two votes - below quorum
        self.detector.register_vote("hash1", "Answer 1", embedding, "node1")
        self.detector.register_vote("hash1", "Answer 1", embedding, "node2")

        self.assertFalse(self.detector.is_locked(embedding))

        # Third vote - reaches quorum
        self.detector.register_vote("hash1", "Answer 1", embedding, "node3")

        self.assertTrue(self.detector.is_locked(embedding))

    def test_supersede_margin_required(self):
        """New answer needs supersede margin to override."""
        embedding = np.array([0.1, 0.2, 0.3])

        # Establish first answer with 3 votes
        for i in range(3):
            self.detector.register_vote("hash1", "Answer 1", embedding, f"node{i}")

        # Try to supersede with 4 votes (only +1 margin)
        for i in range(4):
            self.detector.register_vote("hash2", "Answer 2", embedding, f"other{i}")

        # Should still be hash1 (need +2 margin)
        established = self.detector.get_established(embedding)
        self.assertEqual(established.answer_hash, "hash1")

        # Add more votes to reach +2 margin (total 5 votes)
        self.detector.register_vote("hash2", "Answer 2", embedding, "other4")

        # Now hash2 should win
        established = self.detector.get_established(embedding)
        self.assertEqual(established.answer_hash, "hash2")

    def test_multiple_versions_tracked(self):
        """All versions are tracked with vote counts."""
        embedding = np.array([0.1, 0.2, 0.3])

        self.detector.register_vote("hash1", "Answer 1", embedding, "node1")
        self.detector.register_vote("hash1", "Answer 1", embedding, "node2")
        self.detector.register_vote("hash2", "Answer 2", embedding, "node3")

        stats = self.detector.get_version_stats(embedding)

        self.assertEqual(stats["hash1"], 2)
        self.assertEqual(stats["hash2"], 1)


# =============================================================================
# TEST DIRECT TRUST MANAGER
# =============================================================================

class TestDirectTrustManager(unittest.TestCase):
    """Tests for direct trust management."""

    def setUp(self):
        self.trust = DirectTrustManager(default_trust=0.5, ema_weight=0.2)

    def test_default_trust(self):
        """Unknown nodes get default trust."""
        trust = self.trust.get_trust("unknown_node")
        self.assertEqual(trust, 0.5)

    def test_trust_update_success(self):
        """Successful verification increases trust."""
        self.trust.update_trust("node1", success=True)
        trust = self.trust.get_trust("node1")

        self.assertGreater(trust, 0.5)

    def test_trust_update_failure(self):
        """Failed verification decreases trust."""
        self.trust.update_trust("node1", success=False)
        trust = self.trust.get_trust("node1")

        self.assertLess(trust, 0.5)

    def test_trust_ema_convergence(self):
        """Trust converges toward outcome over time."""
        # Many successes
        for _ in range(20):
            self.trust.update_trust("good_node", success=True)

        trust = self.trust.get_trust("good_node")
        self.assertGreater(trust, 0.9)

    def test_weight_responses(self):
        """Responses are weighted by trust."""
        self.trust.update_trust("trusted", success=True)
        self.trust.update_trust("trusted", success=True)
        self.trust.update_trust("untrusted", success=False)

        responses = [
            ("trusted", "response1"),
            ("untrusted", "response2"),
        ]

        weighted = self.trust.weight_responses(responses)

        # Trusted should have higher weight
        self.assertGreater(weighted[0][1], weighted[1][1])


# =============================================================================
# TEST TRUST-WEIGHTED CONSENSUS
# =============================================================================

class TestTrustWeightedConsensus(unittest.TestCase):
    """Tests for trust-weighted consensus detection."""

    def setUp(self):
        self.trust_manager = DirectTrustManager(default_trust=0.5)
        # Build up trust for some nodes
        for _ in range(10):
            self.trust_manager.update_trust("trusted1", success=True)
            self.trust_manager.update_trust("trusted2", success=True)

        self.detector = TrustWeightedConsensusDetector(
            trust_manager=self.trust_manager,
            quorum=3,
            trust_quorum_factor=0.5  # Trust-based quorum is 1.5
        )

    def test_high_trust_locks_faster(self):
        """Highly trusted nodes can lock with fewer votes."""
        embedding = np.array([0.1, 0.2, 0.3])

        # Two trusted votes should be enough (trust ~0.9 each = ~1.8 total)
        self.detector.register_vote("hash1", "Answer 1", embedding, "trusted1")
        self.detector.register_vote("hash1", "Answer 1", embedding, "trusted2")

        # Should be locked (trust_sum > trust_quorum = 1.5)
        self.assertTrue(self.detector.is_locked(embedding))

    def test_low_trust_needs_more_votes(self):
        """Low-trust nodes need more votes."""
        embedding = np.array([0.5, 0.6, 0.7])

        # Three untrusted votes (0.5 each = 1.5 total)
        self.detector.register_vote("hash1", "Answer 1", embedding, "untrusted1")
        self.detector.register_vote("hash1", "Answer 1", embedding, "untrusted2")
        self.detector.register_vote("hash1", "Answer 1", embedding, "untrusted3")

        # Just at threshold
        self.assertTrue(self.detector.is_locked(embedding))

    def test_untrusted_cant_outvote_trusted(self):
        """Many untrusted nodes can't easily override trusted consensus."""
        embedding = np.array([0.1, 0.2, 0.3])

        # Trusted nodes establish first answer
        self.detector.register_vote("hash1", "Answer 1", embedding, "trusted1")
        self.detector.register_vote("hash1", "Answer 1", embedding, "trusted2")

        # Many untrusted try to override
        for i in range(5):
            self.detector.register_vote("hash2", "Answer 2", embedding, f"untrusted{i}")

        # hash1 should still win (trusted has higher trust_sum)
        established = self.detector.get_established(embedding)
        self.assertEqual(established.answer_hash, "hash1")


# =============================================================================
# TEST ADVERSARIAL PIPELINE
# =============================================================================

class TestAdversarialPipeline(unittest.TestCase):
    """Tests for combined adversarial pipeline."""

    def test_create_pipeline(self):
        """Pipeline can be created with factory function."""
        pipeline = create_adversarial_pipeline(
            outlier_method="mad",
            outlier_threshold=2.5
        )

        self.assertIsNotNone(pipeline.smoother)
        self.assertEqual(pipeline.smoother.method, "mad")

    def test_pipeline_stats(self):
        """Pipeline reports stats from all components."""
        detector = SemanticCollisionDetector()
        trust = DirectTrustManager()

        pipeline = AdversarialPipeline(
            collision_detector=detector,
            trust_manager=trust
        )

        stats = pipeline.get_stats()

        self.assertIn("smoother", stats)
        self.assertIn("collision_detector", stats)
        self.assertIn("trust_manager", stats)


if __name__ == "__main__":
    unittest.main()
