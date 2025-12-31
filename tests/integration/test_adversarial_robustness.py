#!/usr/bin/env python3
"""
Integration tests for Adversarial Robustness (Phase 6f).

Tests:
- Output smoothing (soft collision detection)
- Semantic collision detection (hard collisions)
- Consensus-based collision detection with quorum voting
- Trust management (direct and FMS-style)
"""

import sys
import os
import unittest
import numpy as np

# Add the runtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/unifyweaver/targets/python_runtime'))

from adversarial_robustness import (
    # Data structures
    EstablishedAnswer,
    VersionedAnswer,
    TwoDimensionalTrust,
    NodeTrust,
    # Outlier smoothing
    smooth_outliers,
    robust_aggregate,
    winsorize_scores,
    OutlierSmoother,
    # Collision detection
    SemanticCollisionDetector,
    ConsensusCollisionDetector,
    # Trust management
    DirectTrustManager,
)


class TestEstablishedAnswer(unittest.TestCase):
    """Test the EstablishedAnswer data class."""

    def test_should_lock_criteria(self):
        """Locking should require both confidence and node count."""
        answer = EstablishedAnswer(
            answer_hash="h1",
            answer_text="Test",
            semantic_region="r1",
            confidence=0.9,
            first_seen=0.0,
            node_count=5,
        )

        # Meets both criteria
        self.assertTrue(answer.should_lock(min_confidence=0.8, min_nodes=3))

        # Fails confidence
        self.assertFalse(answer.should_lock(min_confidence=0.95, min_nodes=3))

        # Fails node count
        self.assertFalse(answer.should_lock(min_confidence=0.8, min_nodes=10))


class TestVersionedAnswer(unittest.TestCase):
    """Test the VersionedAnswer data class."""

    def test_vote_count(self):
        """Should count votes correctly."""
        answer = VersionedAnswer(
            answer_hash="h1",
            answer_text="Test",
            node_votes={"n1", "n2", "n3"}
        )

        self.assertEqual(answer.vote_count, 3)

    def test_empty_votes(self):
        """Empty votes should give count 0."""
        answer = VersionedAnswer(answer_hash="h1", answer_text="Test")
        self.assertEqual(answer.vote_count, 0)


class TestTwoDimensionalTrust(unittest.TestCase):
    """Test the TwoDimensionalTrust class."""

    def test_effective_trust(self):
        """Should compute average of both dimensions."""
        trust = TwoDimensionalTrust(message_trust=50, trust_list_trust=30)
        self.assertEqual(trust.effective_trust(), 40)

    def test_normalized_trust(self):
        """Should normalize to 0-1 range."""
        trust = TwoDimensionalTrust(message_trust=0, trust_list_trust=100)

        self.assertAlmostEqual(trust.normalized_message_trust(), 0.5, places=5)
        self.assertAlmostEqual(trust.normalized_trust_list_trust(), 1.0, places=5)

    def test_extreme_values(self):
        """Should handle extreme trust values."""
        trust = TwoDimensionalTrust(message_trust=-100, trust_list_trust=100)

        self.assertAlmostEqual(trust.normalized_message_trust(), 0.0, places=5)
        self.assertAlmostEqual(trust.normalized_trust_list_trust(), 1.0, places=5)


class TestNodeTrust(unittest.TestCase):
    """Test the NodeTrust class."""

    def test_update_success(self):
        """Successful queries should increase trust."""
        node = NodeTrust(node_id="n1", trust_score=0.5)

        initial = node.trust_score
        node.update(success=True, weight=0.2)

        self.assertGreater(node.trust_score, initial)
        self.assertEqual(node.successful_queries, 1)

    def test_update_failure(self):
        """Failed queries should decrease trust."""
        node = NodeTrust(node_id="n1", trust_score=0.5)

        initial = node.trust_score
        node.update(success=False, weight=0.2)

        self.assertLess(node.trust_score, initial)
        self.assertEqual(node.failed_verifications, 1)


class TestSmoothOutliers(unittest.TestCase):
    """Test outlier smoothing functions."""

    def test_zscore_method(self):
        """Z-score should detect statistical outliers."""
        # Normal distribution with one outlier
        scores = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 10.0])  # 10.0 is outlier

        mask = smooth_outliers(scores, method="zscore", threshold=2.0)

        # Outlier should be rejected
        self.assertFalse(mask[-1])
        # Normal values should be kept
        self.assertTrue(all(mask[:-1]))

    def test_iqr_method(self):
        """IQR should detect outliers robustly."""
        scores = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 10.0])

        mask = smooth_outliers(scores, method="iqr", threshold=1.5)

        self.assertFalse(mask[-1])

    def test_mad_method(self):
        """MAD should be most robust to outliers."""
        # More data points with clear outliers - MAD needs non-zero deviation
        scores = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 50.0])

        mask = smooth_outliers(scores, method="mad", threshold=3.0)

        # Should identify 50.0 as outlier
        self.assertFalse(mask[-1])  # 50.0 is rejected
        self.assertTrue(all(mask[:-1]))  # Others kept

    def test_small_sample(self):
        """Small samples should return all True."""
        scores = np.array([1.0, 2.0])
        mask = smooth_outliers(scores, method="zscore")

        self.assertTrue(all(mask))


class TestRobustAggregate(unittest.TestCase):
    """Test robust aggregation."""

    def test_trimmed_mean(self):
        """Should trim extremes before averaging."""
        # Without trimming, mean = 100/10 = 10
        # With 10% trim, should be closer to 1
        scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 91.0, 1.0])

        result = robust_aggregate(scores, trim_fraction=0.1)

        # Should be closer to 1 than to 10
        self.assertLess(result, 5.0)

    def test_empty_array(self):
        """Empty array should return 0."""
        self.assertEqual(robust_aggregate(np.array([])), 0.0)


class TestOutlierSmoother(unittest.TestCase):
    """Test the OutlierSmoother class."""

    def test_filter_results(self):
        """Should filter outlier results."""
        from dataclasses import dataclass

        @dataclass
        class MockResult:
            density_adjusted_score: float

        # Need enough samples with extreme outlier for z-score to detect
        results = [
            MockResult(1.0),
            MockResult(1.1),
            MockResult(0.9),
            MockResult(1.05),
            MockResult(0.95),
            MockResult(100.0),  # Extreme outlier
        ]

        smoother = OutlierSmoother(method="zscore", threshold=2.0)
        filtered = smoother.filter_results(results)

        # 100.0 should be filtered out
        self.assertEqual(len(filtered), 5)

    def test_smooth_and_aggregate(self):
        """Should smooth outliers and aggregate."""
        scores = np.array([1.0, 1.1, 0.9, 1.0, 10.0])

        smoother = OutlierSmoother(method="mad", threshold=2.5)
        agg, mask = smoother.smooth_and_aggregate(scores)

        # Aggregated value should be close to 1
        self.assertLess(agg, 2.0)


class TestSemanticCollisionDetector(unittest.TestCase):
    """Test semantic collision detection."""

    def test_no_collision_new_region(self):
        """New regions should not have collisions."""
        detector = SemanticCollisionDetector()

        embedding = np.array([1.0, 0.0, 0.0])
        allowed, reason = detector.check_collision("answer1", embedding)

        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_collision_locked_region(self):
        """Locked regions should reject different answers."""
        detector = SemanticCollisionDetector(
            lock_threshold=0.5,
            min_nodes_to_lock=2
        )

        embedding = np.array([1.0, 0.0, 0.0])

        # Register and lock an answer
        detector.register_result(
            answer_hash="correct",
            answer_text="Correct answer",
            embedding=embedding,
            confidence=0.9,
            node_count=5
        )

        # Try to insert different answer
        allowed, reason = detector.check_collision("wrong", embedding)

        self.assertFalse(allowed)
        self.assertIn("Collision", reason)

    def test_same_answer_allowed(self):
        """Same answer should be allowed in locked region."""
        detector = SemanticCollisionDetector(
            lock_threshold=0.5,
            min_nodes_to_lock=2
        )

        embedding = np.array([1.0, 0.0, 0.0])

        detector.register_result(
            answer_hash="correct",
            answer_text="Correct",
            embedding=embedding,
            confidence=0.9,
            node_count=5
        )

        allowed, _ = detector.check_collision("correct", embedding)
        self.assertTrue(allowed)

    def test_region_isolation(self):
        """Different regions should be independent."""
        detector = SemanticCollisionDetector()

        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])

        # Lock region 1
        detector.register_result("a1", "Answer 1", emb1, 0.9, 5)

        # Region 2 should be independent
        allowed, _ = detector.check_collision("a2", emb2)
        self.assertTrue(allowed)

    def test_is_locked(self):
        """Should correctly report locked status."""
        detector = SemanticCollisionDetector(
            lock_threshold=0.5,
            min_nodes_to_lock=2
        )

        embedding = np.array([1.0, 0.0, 0.0])

        self.assertFalse(detector.is_locked(embedding))

        detector.register_result("a1", "Answer", embedding, 0.9, 5)

        self.assertTrue(detector.is_locked(embedding))


class TestConsensusCollisionDetector(unittest.TestCase):
    """Test consensus-based collision detection."""

    def test_quorum_required(self):
        """Should require quorum votes to lock."""
        detector = ConsensusCollisionDetector(quorum=3)

        embedding = np.array([1.0, 0.0, 0.0])

        # 2 votes - not enough
        detector.register_vote("a1", "Answer", embedding, "node1")
        detector.register_vote("a1", "Answer", embedding, "node2")

        self.assertFalse(detector.is_locked(embedding))

        # 3rd vote reaches quorum
        detector.register_vote("a1", "Answer", embedding, "node3")

        self.assertTrue(detector.is_locked(embedding))

    def test_competing_answers(self):
        """Should track multiple competing answers."""
        detector = ConsensusCollisionDetector(quorum=3)

        embedding = np.array([1.0, 0.0, 0.0])

        # Votes for answer A
        detector.register_vote("a", "A", embedding, "n1")
        detector.register_vote("a", "A", embedding, "n2")

        # Votes for answer B
        detector.register_vote("b", "B", embedding, "n3")
        detector.register_vote("b", "B", embedding, "n4")

        stats = detector.get_version_stats(embedding)

        self.assertEqual(stats["a"], 2)
        self.assertEqual(stats["b"], 2)

    def test_supersede_margin(self):
        """New consensus should need margin to override."""
        detector = ConsensusCollisionDetector(
            quorum=3,
            supersede_margin=2
        )

        embedding = np.array([1.0, 0.0, 0.0])

        # Establish answer A with 3 votes
        for i in range(3):
            detector.register_vote("a", "A", embedding, f"n{i}")

        established = detector.get_established(embedding)
        self.assertEqual(established.answer_hash, "a")

        # B gets 4 votes (not enough margin)
        for i in range(4):
            detector.register_vote("b", "B", embedding, f"m{i}")

        # A should still be established (4 < 3 + 2)
        established = detector.get_established(embedding)
        self.assertEqual(established.answer_hash, "a")

        # B gets 5th vote (now has margin)
        detector.register_vote("b", "B", embedding, "m4")

        established = detector.get_established(embedding)
        self.assertEqual(established.answer_hash, "b")


class TestDirectTrustManager(unittest.TestCase):
    """Test direct trust management."""

    def test_default_trust(self):
        """Unknown nodes should get default trust."""
        manager = DirectTrustManager(default_trust=0.5)

        trust = manager.get_trust("unknown_node")
        self.assertEqual(trust, 0.5)

    def test_trust_updates(self):
        """Trust should update based on outcomes."""
        manager = DirectTrustManager(default_trust=0.5)

        # Get initial trust
        manager.get_trust("node1")

        # Successful query increases trust
        manager.update_trust("node1", success=True)
        trust_after_success = manager.get_trust("node1")
        self.assertGreater(trust_after_success, 0.5)

        # Failed query decreases trust
        manager.update_trust("node1", success=False)
        manager.update_trust("node1", success=False)
        trust_after_failures = manager.get_trust("node1")
        self.assertLess(trust_after_failures, trust_after_success)

    def test_trust_bounds(self):
        """Trust should stay in valid range."""
        manager = DirectTrustManager(default_trust=0.5)

        # Many successes
        for _ in range(100):
            manager.update_trust("good_node", success=True)

        trust = manager.get_trust("good_node")
        self.assertLessEqual(trust, 1.0)

        # Many failures
        for _ in range(100):
            manager.update_trust("bad_node", success=False)

        trust = manager.get_trust("bad_node")
        self.assertGreaterEqual(trust, 0.0)


if __name__ == "__main__":
    unittest.main()
