"""
Test suite for fuzzy logic runtime.

Demonstrates integration with bookmark filing use case.
"""

import unittest
import numpy as np
from fuzzy_logic import (
    f_and, f_or, f_dist_or, f_union, f_not,
    f_and_batch, f_or_batch, f_dist_or_batch, f_union_batch,
    multiply_scores, blend_scores, top_k, apply_filter, apply_boost
)


class TestFuzzyLogicCore(unittest.TestCase):
    """Test core fuzzy logic operations."""

    def setUp(self):
        self.term_scores = {'bash': 0.8, 'shell': 0.6, 'linux': 0.7}
        self.terms = [('bash', 0.9), ('shell', 0.5)]

    def test_f_and(self):
        """Test fuzzy AND (product t-norm)."""
        # 0.9*0.8 * 0.5*0.6 = 0.216
        result = f_and(self.terms, self.term_scores)
        self.assertAlmostEqual(result, 0.216, places=6)

    def test_f_or(self):
        """Test fuzzy OR (probabilistic sum)."""
        # 1 - (1-0.72)(1-0.3) = 0.804
        result = f_or(self.terms, self.term_scores)
        self.assertAlmostEqual(result, 0.804, places=6)

    def test_f_dist_or(self):
        """Test distributed OR."""
        # 1 - (1-0.7*0.72)(1-0.7*0.3) = 0.60816
        result = f_dist_or(0.7, self.terms, self.term_scores)
        self.assertAlmostEqual(result, 0.60816, places=5)

    def test_f_union(self):
        """Test non-distributed OR (union)."""
        # 0.7 * 0.804 = 0.5628
        result = f_union(0.7, self.terms, self.term_scores)
        self.assertAlmostEqual(result, 0.5628, places=4)

    def test_f_not(self):
        """Test fuzzy NOT."""
        self.assertAlmostEqual(f_not(0.3), 0.7)
        self.assertAlmostEqual(f_not(0.0), 1.0)
        self.assertAlmostEqual(f_not(1.0), 0.0)

    def test_fallback_score(self):
        """Test default score for unknown terms."""
        result = f_and([('unknown', 1.0)], self.term_scores)
        self.assertAlmostEqual(result, 0.5)  # Default fallback

    def test_custom_fallback(self):
        """Test custom fallback score."""
        result = f_and([('unknown', 1.0)], self.term_scores, default_score=0.3)
        self.assertAlmostEqual(result, 0.3)


class TestBatchOperations(unittest.TestCase):
    """Test vectorized batch operations."""

    def setUp(self):
        # Batch of 3 items
        self.term_scores_batch = {
            'bash': np.array([0.8, 0.5, 0.9]),
            'shell': np.array([0.6, 0.7, 0.4])
        }
        self.terms = [('bash', 0.9), ('shell', 0.5)]

    def test_f_and_batch(self):
        """Test batch fuzzy AND."""
        result = f_and_batch(self.terms, self.term_scores_batch)
        expected = np.array([0.9*0.8*0.5*0.6, 0.9*0.5*0.5*0.7, 0.9*0.9*0.5*0.4])
        np.testing.assert_array_almost_equal(result, expected)

    def test_f_or_batch(self):
        """Test batch fuzzy OR."""
        result = f_or_batch(self.terms, self.term_scores_batch)
        # For first item: 1 - (1-0.72)(1-0.3) = 0.804
        self.assertAlmostEqual(result[0], 0.804, places=4)

    def test_f_dist_or_batch(self):
        """Test batch distributed OR."""
        base_scores = np.array([0.7, 0.8, 0.6])
        result = f_dist_or_batch(base_scores, self.terms, self.term_scores_batch)
        self.assertEqual(len(result), 3)


class TestScoreCombination(unittest.TestCase):
    """Test score combination utilities."""

    def test_multiply_scores(self):
        """Test element-wise multiplication."""
        s1 = np.array([0.8, 0.5, 0.9])
        s2 = np.array([0.5, 0.6, 0.7])
        result = multiply_scores(s1, s2)
        np.testing.assert_array_almost_equal(result, [0.4, 0.3, 0.63])

    def test_blend_scores(self):
        """Test score blending."""
        s1 = np.array([0.8, 0.5])
        s2 = np.array([0.4, 0.9])
        result = blend_scores(0.6, s1, s2)
        # 0.6*0.8 + 0.4*0.4 = 0.64, 0.6*0.5 + 0.4*0.9 = 0.66
        np.testing.assert_array_almost_equal(result, [0.64, 0.66])

    def test_top_k(self):
        """Test top-k selection."""
        items = ['a', 'b', 'c', 'd']
        scores = np.array([0.2, 0.8, 0.5, 0.9])
        top = top_k(items, scores, 2)
        self.assertEqual(top[0][0], 'd')
        self.assertEqual(top[1][0], 'b')


class TestFiltersAndBoosts(unittest.TestCase):
    """Test filter and boost operations."""

    def test_apply_filter(self):
        """Test boolean filtering."""
        items = [
            {'name': 'a', 'type': 'tree'},
            {'name': 'b', 'type': 'pearl'},
            {'name': 'c', 'type': 'tree'}
        ]
        scores = np.array([0.8, 0.9, 0.7])

        filtered_items, filtered_scores = apply_filter(
            items, scores,
            lambda item: item['type'] == 'tree'
        )

        self.assertEqual(len(filtered_items), 2)
        self.assertEqual(filtered_items[0]['name'], 'a')
        self.assertEqual(filtered_items[1]['name'], 'c')

    def test_apply_boost(self):
        """Test fuzzy boosting."""
        items = [
            {'name': 'a', 'depth': 1},
            {'name': 'b', 'depth': 3},
            {'name': 'c', 'depth': 2}
        ]
        scores = np.array([0.8, 0.6, 0.7])

        # Boost by inverse depth
        boosted = apply_boost(
            items, scores,
            lambda item: 1.0 / item['depth']
        )

        self.assertAlmostEqual(boosted[0], 0.8)      # 0.8 * 1/1
        self.assertAlmostEqual(boosted[1], 0.2)      # 0.6 * 1/3
        self.assertAlmostEqual(boosted[2], 0.35)     # 0.7 * 1/2


class TestBookmarkFilingIntegration(unittest.TestCase):
    """Test bookmark filing use case with fuzzy logic."""

    def test_semantic_search_with_boost(self):
        """Simulate semantic search with fuzzy boosting."""
        # Simulated semantic search results
        items = [
            {'id': 1, 'path': '/Unix/BASH/Scripts', 'type': 'tree', 'depth': 3},
            {'id': 2, 'path': '/Unix/Shell/Tutorials', 'type': 'pearl', 'depth': 3},
            {'id': 3, 'path': '/Programming/Python', 'type': 'tree', 'depth': 2},
            {'id': 4, 'path': '/Unix/Linux/Kernel', 'type': 'tree', 'depth': 3},
        ]
        base_scores = np.array([0.7, 0.5, 0.3, 0.6])

        # Term scores from query "bash scripting"
        term_scores = {'bash': 0.9, 'scripting': 0.7, 'shell': 0.5}
        terms = [('bash', 0.9), ('shell', 0.5)]

        # Apply distributed OR boost
        boosted = f_dist_or_batch(
            base_scores,
            terms,
            {
                'bash': np.array([0.9, 0.3, 0.1, 0.4]),  # Per-item bash relevance
                'shell': np.array([0.5, 0.8, 0.1, 0.3])  # Per-item shell relevance
            }
        )

        # Filter to trees only
        tree_items, tree_scores = apply_filter(
            items, boosted,
            lambda item: item['type'] == 'tree'
        )

        # Get top 2
        top = top_k(tree_items, tree_scores, 2)

        self.assertEqual(len(top), 2)
        # First result should be bash-related
        self.assertIn('BASH', top[0][0]['path'])

    def test_hierarchical_filtering(self):
        """Test filtering by path hierarchy."""
        items = [
            {'id': 1, 'path': '/Unix/BASH/Scripts'},
            {'id': 2, 'path': '/Unix/Shell/Tutorials'},
            {'id': 3, 'path': '/Programming/Python'},
        ]
        scores = np.array([0.8, 0.7, 0.9])

        # Filter to Unix subtree
        filtered, _ = apply_filter(
            items, scores,
            lambda item: 'Unix' in item['path']
        )

        self.assertEqual(len(filtered), 2)
        for item in filtered:
            self.assertIn('Unix', item['path'])


if __name__ == '__main__':
    unittest.main()
