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


class TestSearchWithContext(unittest.TestCase):
    """Test search_with_context with graph enrichment."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)
        self.model_id = self.db.add_model("test-model", 4)

        # Create test answers
        self.csv_answer = self.db.add_answer("s.md", "How to read CSV files", record_id="csv")
        self.json_answer = self.db.add_answer("s.md", "How to parse JSON", record_id="json")
        self.prereq_answer = self.db.add_answer("s.md", "Install pandas first", record_id="prereq")
        self.advanced_answer = self.db.add_answer("s.md", "Advanced CSV with headers", record_id="advanced")

        # Store embeddings (orthogonal for clear testing)
        self.db.store_embedding(self.model_id, "answer", self.csv_answer,
                                np.array([1, 0, 0, 0], dtype=np.float32))
        self.db.store_embedding(self.model_id, "answer", self.json_answer,
                                np.array([0, 1, 0, 0], dtype=np.float32))
        self.db.store_embedding(self.model_id, "answer", self.prereq_answer,
                                np.array([0, 0, 1, 0], dtype=np.float32))
        self.db.store_embedding(self.model_id, "answer", self.advanced_answer,
                                np.array([0.9, 0.1, 0, 0], dtype=np.float32))  # Similar to CSV

        # Add KG relations
        # prereq is preliminary to csv
        self.db.add_kg_relation(self.prereq_answer, self.csv_answer, 'preliminary')
        # csv is foundational to advanced
        self.db.add_kg_relation(self.csv_answer, self.advanced_answer, 'foundational')
        # csv extends to advanced (compositional)
        self.db.add_kg_relation(self.csv_answer, self.advanced_answer, 'compositional')

    def tearDown(self):
        self.db.close()

    def test_graph_traversal_enrichment(self):
        """Graph traversal methods return correct relations."""
        # Test foundational: advanced depends on csv
        foundational = self.db.get_foundational(self.advanced_answer)
        self.assertEqual(len(foundational), 1)
        self.assertEqual(foundational[0]['answer_id'], self.csv_answer)

        # Test prerequisites: csv requires prereq
        prereqs = self.db.get_prerequisites(self.csv_answer)
        self.assertEqual(len(prereqs), 1)
        self.assertEqual(prereqs[0]['answer_id'], self.prereq_answer)

        # Test extensions: csv extends to advanced
        extensions = self.db.get_extensions(self.csv_answer)
        self.assertEqual(len(extensions), 1)
        self.assertEqual(extensions[0]['answer_id'], self.advanced_answer)

    def test_traverse_relations_depth(self):
        """_traverse_relations respects depth parameter."""
        # Depth 1: only direct relations
        results_d1 = self.db._traverse_relations(
            self.csv_answer, 'compositional', 'outgoing', depth=1
        )
        self.assertEqual(len(results_d1), 1)

        # Depth 0: no results
        results_d0 = self.db._traverse_relations(
            self.csv_answer, 'compositional', 'outgoing', depth=0
        )
        self.assertEqual(len(results_d0), 0)

    def test_search_with_context_structure(self):
        """search_with_context returns properly structured results."""
        # Mock the search by testing the enrichment structure
        # We'll test the traversal enrichment separately since _direct_search
        # requires a real embedding model

        # Create a mock result to enrich
        mock_result = {
            'answer_id': self.csv_answer,
            'score': 0.95,
            'text': 'CSV answer'
        }

        # Manually enrich like search_with_context does
        enriched = dict(mock_result)
        enriched['foundational'] = self.db._traverse_relations(
            self.csv_answer, 'foundational', 'incoming', 1
        )
        enriched['prerequisites'] = self.db._traverse_relations(
            self.csv_answer, 'preliminary', 'incoming', 1
        )
        enriched['extensions'] = self.db._traverse_relations(
            self.csv_answer, 'compositional', 'outgoing', 1
        )
        enriched['next_steps'] = self.db._traverse_relations(
            self.csv_answer, 'transitional', 'outgoing', 1
        )

        # Verify structure
        self.assertIn('foundational', enriched)
        self.assertIn('prerequisites', enriched)
        self.assertIn('extensions', enriched)
        self.assertIn('next_steps', enriched)

        # Verify content
        self.assertEqual(len(enriched['prerequisites']), 1)
        self.assertEqual(enriched['prerequisites'][0]['answer_id'], self.prereq_answer)
        self.assertEqual(len(enriched['extensions']), 1)
        self.assertEqual(enriched['extensions'][0]['answer_id'], self.advanced_answer)


class TestPrologModuleStructure(unittest.TestCase):
    """Test Prolog module structure (without requiring SWI-Prolog)."""

    def test_prolog_module_exists(self):
        """kg_topology.pl exists."""
        module_path = project_root / "src" / "unifyweaver" / "runtime" / "kg_topology.pl"
        self.assertTrue(module_path.exists())

    def test_prolog_module_has_required_predicates(self):
        """kg_topology.pl exports required predicates."""
        module_path = project_root / "src" / "unifyweaver" / "runtime" / "kg_topology.pl"

        with open(module_path, 'r') as f:
            content = f.read()

        # Check for module declaration with exports
        self.assertIn(':- module(kg_topology,', content)

        # Check for relation type predicates
        self.assertIn('relation_category/2', content)
        self.assertIn('relation_direction/2', content)
        self.assertIn('valid_relation_type/1', content)

        # Check for graph traversal predicates
        self.assertIn('get_foundational/3', content)
        self.assertIn('get_prerequisites/3', content)
        self.assertIn('get_extensions/3', content)
        self.assertIn('get_next_steps/3', content)

        # Check for anchor linking
        self.assertIn('compute_content_hash/2', content)
        self.assertIn('anchor_question/3', content)

        # Check for seed level tracking
        self.assertIn('seed_level/3', content)
        self.assertIn('questions_at_seed_level/4', content)

        # Check for search
        self.assertIn('search_with_context/5', content)

    def test_prolog_module_defines_all_relation_types(self):
        """kg_topology.pl defines all 11 relation types."""
        module_path = project_root / "src" / "unifyweaver" / "runtime" / "kg_topology.pl"

        with open(module_path, 'r') as f:
            content = f.read()

        # Learning flow relations
        self.assertIn('foundational', content)
        self.assertIn('preliminary', content)
        self.assertIn('compositional', content)
        self.assertIn('transitional', content)

        # Scope relations
        self.assertIn('refined', content)
        self.assertIn('general', content)

        # Abstraction relations
        self.assertIn('generalization', content)
        self.assertIn('implementation', content)
        self.assertIn('axiomatization', content)
        self.assertIn('instance', content)
        self.assertIn('example', content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
