#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Unit tests for KG Topology API

"""
Tests for kg_topology_api.py

Tests:
- KG schema creation (seed levels, anchor linking)
- 11 relation types (learning flow, scope, abstraction)
- Graph traversal API
- Anchor linking (content hash)
- Seed level tracking
- Direct search (baseline)
"""

import sys
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from kg_topology_api import (
    KGTopologyAPI,
    ALL_RELATION_TYPES,
    LEARNING_FLOW_RELATIONS,
    SCOPE_RELATIONS,
    ABSTRACTION_RELATIONS,
    INCOMING_RELATIONS,
    OUTGOING_RELATIONS,
)


class TestKGTopologySchema(unittest.TestCase):
    """Test KG topology schema creation."""

    def test_creates_kg_tables(self):
        """KG tables are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = KGTopologyAPI(db_path)

            # Check that KG tables exist
            cursor = db.conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('question_seed_levels', 'answer_anchors')
            """)
            tables = [row[0] for row in cursor.fetchall()]

            self.assertIn('question_seed_levels', tables)
            self.assertIn('answer_anchors', tables)
            db.close()

    def test_inherits_from_lda_database(self):
        """KGTopologyAPI inherits LDAProjectionDB methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = KGTopologyAPI(db_path)

            # Should have parent methods
            self.assertTrue(hasattr(db, 'add_answer'))
            self.assertTrue(hasattr(db, 'add_question'))
            self.assertTrue(hasattr(db, 'create_cluster'))
            self.assertTrue(hasattr(db, 'multi_head_search'))
            db.close()


class TestRelationTypeDefinitions(unittest.TestCase):
    """Test relation type definitions."""

    def test_all_relation_types_count(self):
        """All 11 relation types are defined."""
        self.assertEqual(len(ALL_RELATION_TYPES), 11)

    def test_learning_flow_relations(self):
        """Learning flow has 4 relations."""
        expected = {'foundational', 'preliminary', 'compositional', 'transitional'}
        self.assertEqual(LEARNING_FLOW_RELATIONS, expected)

    def test_scope_relations(self):
        """Scope has 2 relations."""
        expected = {'refined', 'general'}
        self.assertEqual(SCOPE_RELATIONS, expected)

    def test_abstraction_relations(self):
        """Abstraction has 5 relations."""
        expected = {'generalization', 'implementation', 'axiomatization', 'instance', 'example'}
        self.assertEqual(ABSTRACTION_RELATIONS, expected)

    def test_direction_coverage(self):
        """All relation types have a direction."""
        all_directional = INCOMING_RELATIONS | OUTGOING_RELATIONS
        self.assertEqual(all_directional, ALL_RELATION_TYPES)


class TestKGRelations(unittest.TestCase):
    """Test KG relation operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        # Create test answers
        self.a1 = self.db.add_answer("s.md", "Basic concept")
        self.a2 = self.db.add_answer("s.md", "Advanced concept")
        self.a3 = self.db.add_answer("s.md", "Prerequisite step")

    def tearDown(self):
        self.db.close()

    def test_add_kg_relation_foundational(self):
        """Can add foundational relation."""
        rid = self.db.add_kg_relation(self.a1, self.a2, 'foundational')
        self.assertIsInstance(rid, int)

    def test_add_kg_relation_all_types(self):
        """Can add all 11 relation types."""
        for rel_type in ALL_RELATION_TYPES:
            rid = self.db.add_kg_relation(self.a1, self.a2, rel_type)
            self.assertIsInstance(rid, int)

    def test_add_kg_relation_invalid_type(self):
        """Rejects invalid relation type."""
        with self.assertRaises(ValueError):
            self.db.add_kg_relation(self.a1, self.a2, 'invalid_type')

    def test_get_relations_outgoing(self):
        """Can get outgoing relations."""
        self.db.add_kg_relation(self.a1, self.a2, 'compositional')

        results = self.db.get_relations(self.a1, 'compositional', 'outgoing')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['answer_id'], self.a2)

    def test_get_relations_incoming(self):
        """Can get incoming relations."""
        self.db.add_kg_relation(self.a1, self.a2, 'foundational')

        # a1 is foundational TO a2, so query a2 for incoming
        results = self.db.get_relations(self.a2, 'foundational', 'incoming')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['answer_id'], self.a1)


class TestGraphTraversalAPI(unittest.TestCase):
    """Test graph traversal API methods."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        # Create a small knowledge graph
        # Basic -> Advanced (compositional)
        # Prereq -> Advanced (preliminary)
        # Basic is foundational to Advanced
        self.basic = self.db.add_answer("s.md", "Basic concept")
        self.advanced = self.db.add_answer("s.md", "Advanced concept")
        self.prereq = self.db.add_answer("s.md", "Prerequisite step")
        self.pattern = self.db.add_answer("s.md", "Abstract pattern")

        # Add relations
        self.db.add_kg_relation(self.basic, self.advanced, 'foundational')
        self.db.add_kg_relation(self.prereq, self.advanced, 'preliminary')
        self.db.add_kg_relation(self.basic, self.advanced, 'compositional')
        self.db.add_kg_relation(self.basic, self.pattern, 'generalization')

    def tearDown(self):
        self.db.close()

    def test_get_foundational(self):
        """get_foundational returns foundational concepts."""
        results = self.db.get_foundational(self.advanced)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['answer_id'], self.basic)

    def test_get_prerequisites(self):
        """get_prerequisites returns preliminary steps."""
        results = self.db.get_prerequisites(self.advanced)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['answer_id'], self.prereq)

    def test_get_extensions(self):
        """get_extensions returns compositional extensions."""
        results = self.db.get_extensions(self.basic)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['answer_id'], self.advanced)

    def test_get_generalizations(self):
        """get_generalizations returns abstract patterns."""
        results = self.db.get_generalizations(self.basic)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['answer_id'], self.pattern)

    def test_empty_relations(self):
        """Returns empty list when no relations exist."""
        results = self.db.get_next_steps(self.advanced)
        self.assertEqual(results, [])


class TestAnchorLinking(unittest.TestCase):
    """Test hash-based anchor linking."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)
        self.answer_id = self.db.add_answer("s.md", "Test answer")

    def tearDown(self):
        self.db.close()

    def test_compute_content_hash(self):
        """Content hash is computed correctly."""
        hash1 = KGTopologyAPI.compute_content_hash("Hello world")
        hash2 = KGTopologyAPI.compute_content_hash("Hello world")
        hash3 = KGTopologyAPI.compute_content_hash("Different text")

        # Same content = same hash
        self.assertEqual(hash1, hash2)
        # Different content = different hash
        self.assertNotEqual(hash1, hash3)
        # SHA-256 hex length
        self.assertEqual(len(hash1), 64)

    def test_set_anchor_question(self):
        """Can set anchor question for an answer."""
        question_text = "How do I do X?"
        hash_result = self.db.set_anchor_question(self.answer_id, question_text, seed_level=0)

        self.assertEqual(len(hash_result), 64)

    def test_get_anchor_question(self):
        """Can retrieve anchor question hash."""
        question_text = "How do I do X?"
        expected_hash = self.db.set_anchor_question(self.answer_id, question_text)

        retrieved_hash = self.db.get_anchor_question(self.answer_id)
        self.assertEqual(retrieved_hash, expected_hash)

    def test_get_anchor_question_none(self):
        """Returns None for answer without anchor."""
        result = self.db.get_anchor_question(9999)
        self.assertIsNone(result)


class TestSeedLevelTracking(unittest.TestCase):
    """Test seed level provenance tracking."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        # Create questions
        self.q1 = self.db.add_question("Original question", "medium")
        self.q2 = self.db.add_question("Discovered question", "medium")
        self.q3 = self.db.add_question("Second expansion", "medium")

    def tearDown(self):
        self.db.close()

    def test_set_seed_level(self):
        """Can set seed level for a question."""
        self.db.set_seed_level(self.q1, seed_level=0)
        level = self.db.get_seed_level(self.q1)
        self.assertEqual(level, 0)

    def test_seed_level_with_discovery(self):
        """Can track discovery provenance."""
        self.db.set_seed_level(self.q1, 0)
        self.db.set_seed_level(self.q2, 1, discovered_from=self.q1, discovery_relation='refined')

        level = self.db.get_seed_level(self.q2)
        self.assertEqual(level, 1)

    def test_get_seed_level_none(self):
        """Returns None for question without seed level."""
        result = self.db.get_seed_level(9999)
        self.assertIsNone(result)

    def test_get_questions_at_seed_level(self):
        """Can query questions by seed level."""
        self.db.set_seed_level(self.q1, 0)
        self.db.set_seed_level(self.q2, 1)
        self.db.set_seed_level(self.q3, 1)

        level_0 = self.db.get_questions_at_seed_level(0)
        level_1 = self.db.get_questions_at_seed_level(1)

        self.assertEqual(len(level_0), 1)
        self.assertEqual(len(level_1), 2)

    def test_get_questions_at_seed_level_with_cluster(self):
        """Can filter by cluster."""
        a1 = self.db.add_answer("s.md", "Answer")
        c1 = self.db.create_cluster("cluster1", [a1], [self.q1, self.q2])

        self.db.set_seed_level(self.q1, 0)
        self.db.set_seed_level(self.q2, 0)
        self.db.set_seed_level(self.q3, 0)  # Not in cluster

        in_cluster = self.db.get_questions_at_seed_level(0, cluster_id=c1)
        self.assertEqual(len(in_cluster), 2)


class TestDirectSearch(unittest.TestCase):
    """Test direct search (baseline)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)
        self.model_id = self.db.add_model("test-model", 4)

        # Create test answers with embeddings
        self.a1 = self.db.add_answer("s.md", "Answer about CSV", record_id="csv")
        self.a2 = self.db.add_answer("s.md", "Answer about JSON", record_id="json")

        # Orthogonal embeddings for clear testing
        self.db.store_embedding(self.model_id, "answer", self.a1,
                                np.array([1, 0, 0, 0], dtype=np.float32))
        self.db.store_embedding(self.model_id, "answer", self.a2,
                                np.array([0, 1, 0, 0], dtype=np.float32))

    def tearDown(self):
        self.db.close()

    def test_direct_search_with_mock_embedding(self):
        """Direct search works with pre-computed embeddings."""
        # Test the underlying search logic by directly testing get_all_answer_embeddings
        # and the scoring mechanism (without requiring a real embedding model)
        ids, matrix = self.db.get_all_answer_embeddings(self.model_id)

        self.assertEqual(len(ids), 2)
        self.assertEqual(matrix.shape, (2, 4))

        # Simulate a query closer to CSV [1,0,0,0]
        query = np.array([0.9, 0.1, 0, 0], dtype=np.float32)
        query = query / np.linalg.norm(query)

        # Normalize answer matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_normed = matrix / norms

        # Compute scores
        scores = matrix_normed @ query
        top_idx = np.argmax(scores)

        # Should match CSV (index 0)
        self.assertEqual(ids[top_idx], self.a1)

    @unittest.skip("Requires real embedding model - run manually with: python -m unittest tests.core.test_kg_topology.TestDirectSearch.test_direct_search_integration")
    def test_direct_search_integration(self):
        """Integration test: Direct search with real embedding model.

        This test requires sentence-transformers and network access.
        Run manually when testing full integration.
        """
        try:
            results = self.db._direct_search("csv query", "all-MiniLM-L6-v2", top_k=2)
            self.assertIsInstance(results, list)
        except Exception as e:
            self.skipTest(f"Embedding model not available: {e}")


class TestLearningPath(unittest.TestCase):
    """Test learning path generation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        # Create a dependency chain:
        # basics -> intermediate -> advanced
        self.basics = self.db.add_answer("s.md", "Basics")
        self.intermediate = self.db.add_answer("s.md", "Intermediate")
        self.advanced = self.db.add_answer("s.md", "Advanced")
        self.setup = self.db.add_answer("s.md", "Setup step")

        # Add relations
        self.db.add_kg_relation(self.basics, self.intermediate, 'foundational')
        self.db.add_kg_relation(self.intermediate, self.advanced, 'foundational')
        self.db.add_kg_relation(self.setup, self.advanced, 'preliminary')

    def tearDown(self):
        self.db.close()

    def test_get_learning_path(self):
        """Learning path includes dependencies in order."""
        path = self.db.get_learning_path(self.advanced, max_depth=3)

        # Should include basics, intermediate, and setup
        answer_ids = [p['answer_id'] for p in path]
        self.assertIn(self.basics, answer_ids)
        self.assertIn(self.intermediate, answer_ids)
        self.assertIn(self.setup, answer_ids)

    def test_learning_path_respects_depth(self):
        """Learning path respects max_depth."""
        path_shallow = self.db.get_learning_path(self.advanced, max_depth=1)
        path_deep = self.db.get_learning_path(self.advanced, max_depth=3)

        # Deeper path should have more or equal entries
        self.assertLessEqual(len(path_shallow), len(path_deep))


if __name__ == "__main__":
    unittest.main(verbosity=2)
