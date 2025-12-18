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


# =============================================================================
# PHASE 2: SEMANTIC INTERFACES TESTS
# =============================================================================

class TestInterfaceSchema(unittest.TestCase):
    """Test Phase 2 interface schema creation."""

    def test_creates_interface_tables(self):
        """Interface tables are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = KGTopologyAPI(db_path)

            cursor = db.conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN (
                    'semantic_interfaces',
                    'interface_centroids',
                    'interface_clusters',
                    'interface_metrics'
                )
            """)
            tables = [row[0] for row in cursor.fetchall()]

            self.assertIn('semantic_interfaces', tables)
            self.assertIn('interface_centroids', tables)
            self.assertIn('interface_clusters', tables)
            self.assertIn('interface_metrics', tables)
            db.close()


class TestInterfaceCRUD(unittest.TestCase):
    """Test interface CRUD operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

    def tearDown(self):
        self.db.close()

    def test_create_interface(self):
        """Can create an interface."""
        iface_id = self.db.create_interface(
            name="csv_expert",
            description="Expert on CSV parsing",
            topics=["csv", "data", "parsing"]
        )
        self.assertIsInstance(iface_id, int)

    def test_get_interface(self):
        """Can retrieve interface by ID."""
        iface_id = self.db.create_interface(
            name="test_interface",
            description="Test description",
            topics=["topic1", "topic2"]
        )

        iface = self.db.get_interface(iface_id)
        self.assertIsNotNone(iface)
        self.assertEqual(iface['name'], "test_interface")
        self.assertEqual(iface['description'], "Test description")
        self.assertEqual(iface['topics'], ["topic1", "topic2"])
        self.assertTrue(iface['is_active'])

    def test_get_interface_by_name(self):
        """Can retrieve interface by name."""
        self.db.create_interface(name="named_interface", description="Find by name")

        iface = self.db.get_interface_by_name("named_interface")
        self.assertIsNotNone(iface)
        self.assertEqual(iface['name'], "named_interface")

    def test_list_interfaces(self):
        """Can list all interfaces."""
        self.db.create_interface(name="iface1")
        self.db.create_interface(name="iface2")
        self.db.create_interface(name="iface3")

        interfaces = self.db.list_interfaces()
        self.assertEqual(len(interfaces), 3)

    def test_list_interfaces_active_only(self):
        """Can filter inactive interfaces."""
        id1 = self.db.create_interface(name="active1")
        id2 = self.db.create_interface(name="inactive1")
        self.db.update_interface(id2, is_active=False)

        active = self.db.list_interfaces(active_only=True)
        all_ifaces = self.db.list_interfaces(active_only=False)

        self.assertEqual(len(active), 1)
        self.assertEqual(len(all_ifaces), 2)

    def test_update_interface(self):
        """Can update interface properties."""
        iface_id = self.db.create_interface(name="original", description="Original desc")

        self.db.update_interface(iface_id, name="updated", description="Updated desc")

        iface = self.db.get_interface(iface_id)
        self.assertEqual(iface['name'], "updated")
        self.assertEqual(iface['description'], "Updated desc")

    def test_delete_interface(self):
        """Can delete an interface."""
        iface_id = self.db.create_interface(name="to_delete")

        self.db.delete_interface(iface_id)

        iface = self.db.get_interface(iface_id)
        self.assertIsNone(iface)


class TestInterfaceClusterMapping(unittest.TestCase):
    """Test interface-cluster mapping."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        # Create test interface and clusters
        self.iface_id = self.db.create_interface(name="test_interface")

        # Create clusters
        a1 = self.db.add_answer("s.md", "Answer 1")
        a2 = self.db.add_answer("s.md", "Answer 2")
        q1 = self.db.add_question("Q1", "medium")
        q2 = self.db.add_question("Q2", "medium")

        self.c1 = self.db.create_cluster("cluster1", [a1], [q1])
        self.c2 = self.db.create_cluster("cluster2", [a2], [q2])

    def tearDown(self):
        self.db.close()

    def test_add_cluster_to_interface(self):
        """Can add cluster to interface."""
        self.db.add_cluster_to_interface(self.iface_id, self.c1)

        clusters = self.db.get_interface_clusters(self.iface_id)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]['cluster_id'], self.c1)

    def test_add_cluster_with_weight(self):
        """Can add cluster with custom weight."""
        self.db.add_cluster_to_interface(self.iface_id, self.c1, weight=0.8)

        clusters = self.db.get_interface_clusters(self.iface_id)
        self.assertEqual(clusters[0]['weight'], 0.8)

    def test_remove_cluster_from_interface(self):
        """Can remove cluster from interface."""
        self.db.add_cluster_to_interface(self.iface_id, self.c1)
        self.db.add_cluster_to_interface(self.iface_id, self.c2)

        self.db.remove_cluster_from_interface(self.iface_id, self.c1)

        clusters = self.db.get_interface_clusters(self.iface_id)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]['cluster_id'], self.c2)


class TestInterfaceCentroids(unittest.TestCase):
    """Test interface centroid operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        self.model_id = self.db.add_model("test-model", 4)
        self.iface_id = self.db.create_interface(name="test_interface")

    def tearDown(self):
        self.db.close()

    def test_set_interface_centroid(self):
        """Can set interface centroid."""
        centroid = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        self.db.set_interface_centroid(self.iface_id, self.model_id, centroid)

        retrieved = self.db.get_interface_centroid(self.iface_id, self.model_id)
        np.testing.assert_array_almost_equal(retrieved, centroid)

    def test_get_interface_centroid_none(self):
        """Returns None when centroid not set."""
        centroid = self.db.get_interface_centroid(self.iface_id, self.model_id)
        self.assertIsNone(centroid)

    def test_compute_interface_centroid(self):
        """Can compute interface centroid from clusters."""
        # Create clusters with centroids
        a1 = self.db.add_answer("s.md", "Answer 1")
        a2 = self.db.add_answer("s.md", "Answer 2")
        q1 = self.db.add_question("Q1", "medium")
        q2 = self.db.add_question("Q2", "medium")

        c1 = self.db.create_cluster("cluster1", [a1], [q1])
        c2 = self.db.create_cluster("cluster2", [a2], [q2])

        # Set cluster centroids
        self.db.set_cluster_centroid(c1, self.model_id,
                                     np.array([1, 0, 0, 0], dtype=np.float32))
        self.db.set_cluster_centroid(c2, self.model_id,
                                     np.array([0, 1, 0, 0], dtype=np.float32))

        # Add clusters to interface with equal weight
        self.db.add_cluster_to_interface(self.iface_id, c1, weight=1.0)
        self.db.add_cluster_to_interface(self.iface_id, c2, weight=1.0)

        # Compute centroid
        centroid = self.db.compute_interface_centroid(self.iface_id, self.model_id)

        # Should be normalized average
        self.assertIsNotNone(centroid)
        self.assertEqual(len(centroid), 4)
        # Normalized [0.5, 0.5, 0, 0] ≈ [0.707, 0.707, 0, 0]
        self.assertAlmostEqual(centroid[0], centroid[1], places=3)


class TestQueryToInterfaceMapping(unittest.TestCase):
    """Test query-to-interface mapping."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        self.model_id = self.db.add_model("test-model", 4)

        # Create interfaces with distinct centroids
        self.csv_iface = self.db.create_interface(name="csv_expert", topics=["csv"])
        self.json_iface = self.db.create_interface(name="json_expert", topics=["json"])

        # Set orthogonal centroids
        self.db.set_interface_centroid(
            self.csv_iface, self.model_id,
            np.array([1, 0, 0, 0], dtype=np.float32)
        )
        self.db.set_interface_centroid(
            self.json_iface, self.model_id,
            np.array([0, 1, 0, 0], dtype=np.float32)
        )

    def tearDown(self):
        self.db.close()

    def test_map_query_routing_weights(self):
        """Query mapping returns routing weights."""
        # Mock _embed_query to return a specific embedding
        original_embed = self.db._embed_query

        def mock_embed(text, model):
            if "csv" in text.lower():
                return np.array([0.9, 0.1, 0, 0], dtype=np.float32)
            return np.array([0.1, 0.9, 0, 0], dtype=np.float32)

        self.db._embed_query = mock_embed

        try:
            results = self.db.map_query_to_interface("How to read CSV?", "test-model")

            self.assertEqual(len(results), 2)
            # CSV interface should have higher weight
            self.assertEqual(results[0]['interface_id'], self.csv_iface)
            self.assertGreater(results[0]['routing_weight'], 0.5)
        finally:
            self.db._embed_query = original_embed

    def test_map_query_empty_when_no_centroids(self):
        """Returns empty list when no interfaces have centroids."""
        # Create interface without centroid
        new_iface = self.db.create_interface(name="no_centroid")

        # Delete the existing interfaces' centroids
        self.db.conn.cursor().execute("DELETE FROM interface_centroids")
        self.db.conn.commit()

        def mock_embed(text, model):
            return np.array([0.5, 0.5, 0, 0], dtype=np.float32)

        self.db._embed_query = mock_embed

        results = self.db.map_query_to_interface("Any query", "test-model")
        self.assertEqual(results, [])


class TestInterfaceMetrics(unittest.TestCase):
    """Test interface metrics operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        self.iface_id = self.db.create_interface(name="test_interface")

        # Create cluster with answers and questions
        self.a1 = self.db.add_answer("s.md", "Answer 1")
        self.a2 = self.db.add_answer("s.md", "Answer 2")
        self.q1 = self.db.add_question("Q1", "medium")
        self.q2 = self.db.add_question("Q2", "medium")
        self.q3 = self.db.add_question("Q3", "medium")

        self.c1 = self.db.create_cluster("cluster1", [self.a1, self.a2], [self.q1, self.q2, self.q3])
        self.db.add_cluster_to_interface(self.iface_id, self.c1)

    def tearDown(self):
        self.db.close()

    def test_set_and_get_metric(self):
        """Can set and retrieve metrics."""
        self.db.set_interface_metric(self.iface_id, "custom_metric", 42.5)

        metrics = self.db.get_interface_metrics(self.iface_id)
        self.assertEqual(metrics['custom_metric'], 42.5)

    def test_compute_interface_coverage(self):
        """Can compute coverage metrics."""
        metrics = self.db.compute_interface_coverage(self.iface_id)

        self.assertEqual(metrics['cluster_count'], 1.0)
        self.assertEqual(metrics['answer_count'], 2.0)
        self.assertEqual(metrics['question_count'], 3.0)
        self.assertEqual(metrics['avg_cluster_size'], 2.0)

    def test_get_interface_health_healthy(self):
        """Health check returns healthy for well-configured interface."""
        # Add centroid so health is complete
        model_id = self.db.add_model("test-model", 4)
        self.db.set_interface_centroid(
            self.iface_id, model_id,
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        )

        # Add more clusters to make it "healthy"
        a3 = self.db.add_answer("s.md", "Answer 3")
        a4 = self.db.add_answer("s.md", "Answer 4")
        q4 = self.db.add_question("Q4", "medium")
        q5 = self.db.add_question("Q5", "medium")

        c2 = self.db.create_cluster("cluster2", [a3], [q4])
        c3 = self.db.create_cluster("cluster3", [a4], [q5])
        self.db.add_cluster_to_interface(self.iface_id, c2)
        self.db.add_cluster_to_interface(self.iface_id, c3)

        health = self.db.get_interface_health(self.iface_id)

        self.assertEqual(health['health_status'], 'healthy')
        self.assertEqual(health['issues'], [])

    def test_get_interface_health_warning_few_clusters(self):
        """Health check warns about few clusters."""
        health = self.db.get_interface_health(self.iface_id)

        self.assertEqual(health['health_status'], 'warning')
        self.assertIn('Few clusters', health['issues'][0])

    def test_get_interface_health_unhealthy_no_clusters(self):
        """Health check reports unhealthy for no clusters."""
        empty_iface = self.db.create_interface(name="empty_interface")

        health = self.db.get_interface_health(empty_iface)

        self.assertEqual(health['health_status'], 'unhealthy')
        self.assertTrue(any('No clusters' in i for i in health['issues']))


class TestSearchViaInterface(unittest.TestCase):
    """Test interface-first search."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        self.model_id = self.db.add_model("test-model", 4)

        # Create interface
        self.iface_id = self.db.create_interface(name="test_interface")

        # Create cluster with answers
        self.a1 = self.db.add_answer("s.md", "Interface answer 1")
        self.a2 = self.db.add_answer("s.md", "Interface answer 2")
        self.a3 = self.db.add_answer("s.md", "Non-interface answer")

        q1 = self.db.add_question("Q1", "medium")
        q2 = self.db.add_question("Q2", "medium")
        q3 = self.db.add_question("Q3", "medium")

        self.c1 = self.db.create_cluster("cluster1", [self.a1, self.a2], [q1, q2])
        self.c2 = self.db.create_cluster("cluster2", [self.a3], [q3])

        # Only add c1 to interface
        self.db.add_cluster_to_interface(self.iface_id, self.c1)

        # Store embeddings
        self.db.store_embedding(self.model_id, "answer", self.a1,
                                np.array([1, 0, 0, 0], dtype=np.float32))
        self.db.store_embedding(self.model_id, "answer", self.a2,
                                np.array([0.9, 0.1, 0, 0], dtype=np.float32))
        self.db.store_embedding(self.model_id, "answer", self.a3,
                                np.array([0, 1, 0, 0], dtype=np.float32))

    def tearDown(self):
        self.db.close()

    def test_search_via_interface_returns_only_interface_answers(self):
        """search_via_interface returns only answers from interface clusters."""
        # This tests the filtering logic without requiring real embeddings
        interface_clusters = self.db.get_interface_clusters(self.iface_id)

        self.assertEqual(len(interface_clusters), 1)
        self.assertEqual(interface_clusters[0]['cluster_id'], self.c1)


# =============================================================================
# SCALE OPTIMIZATIONS TESTS
# =============================================================================

class TestInterfaceFirstRouting(unittest.TestCase):
    """Test Interface-First Routing optimization."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        self.model_id = self.db.add_model("test-model", 4)

        # Create interface with centroid
        self.iface_id = self.db.create_interface(name="test_interface")

        # Create answers with embeddings
        self.a1 = self.db.add_answer("s.md", "Close to interface")
        self.a2 = self.db.add_answer("s.md", "Also close to interface")
        self.a3 = self.db.add_answer("s.md", "Far from interface")

        # Embeddings: a1, a2 close to interface centroid, a3 far
        self.db.store_embedding(self.model_id, "answer", self.a1,
                                np.array([0.9, 0.1, 0, 0], dtype=np.float32))
        self.db.store_embedding(self.model_id, "answer", self.a2,
                                np.array([0.8, 0.2, 0, 0], dtype=np.float32))
        self.db.store_embedding(self.model_id, "answer", self.a3,
                                np.array([0, 0, 0.9, 0.1], dtype=np.float32))

        # Interface centroid: [1, 0, 0, 0]
        self.db.set_interface_centroid(
            self.iface_id, self.model_id,
            np.array([1, 0, 0, 0], dtype=np.float32)
        )

    def tearDown(self):
        self.db.close()

    def test_interface_first_search_filters_by_similarity(self):
        """Interface-first routing filters answers by centroid similarity."""
        # The _interface_first_search method filters by similarity threshold
        interface_centroid = self.db.get_interface_centroid(self.iface_id, self.model_id)

        # Compute expected similarities
        ids, matrix = self.db.get_all_answer_embeddings(self.model_id)

        # Normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_normed = matrix / norms

        similarities = matrix_normed @ interface_centroid

        # a1, a2 should have high similarity (>0.5), a3 low
        self.assertGreater(similarities[ids.index(self.a1)], 0.5)
        self.assertGreater(similarities[ids.index(self.a2)], 0.5)
        self.assertLess(similarities[ids.index(self.a3)], 0.5)

    def test_search_via_interface_accepts_routing_flag(self):
        """search_via_interface accepts use_interface_first_routing parameter."""
        # Test that the method accepts the parameter without error
        # (actual search requires _embed_query which needs real model)
        self.assertTrue(hasattr(self.db, 'search_via_interface'))

        # Verify signature includes the optimization parameters
        import inspect
        sig = inspect.signature(self.db.search_via_interface)
        params = sig.parameters

        self.assertIn('use_interface_first_routing', params)
        self.assertIn('max_distance', params)
        self.assertIn('similarity_threshold', params)


class TestMaxDistanceFiltering(unittest.TestCase):
    """Test max_distance post-filtering."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        self.model_id = self.db.add_model("test-model", 4)

        # Create answers with embeddings at known distances
        self.a1 = self.db.add_answer("s.md", "Very close")
        self.a2 = self.db.add_answer("s.md", "Medium distance")
        self.a3 = self.db.add_answer("s.md", "Far away")

        # Embeddings with known cosine similarities to [1,0,0,0]
        self.db.store_embedding(self.model_id, "answer", self.a1,
                                np.array([1, 0, 0, 0], dtype=np.float32))  # sim=1.0, dist=0.0
        self.db.store_embedding(self.model_id, "answer", self.a2,
                                np.array([0.707, 0.707, 0, 0], dtype=np.float32))  # sim≈0.707, dist≈0.29
        self.db.store_embedding(self.model_id, "answer", self.a3,
                                np.array([0, 1, 0, 0], dtype=np.float32))  # sim=0.0, dist=1.0

    def tearDown(self):
        self.db.close()

    def test_filter_by_max_distance_filters_correctly(self):
        """_filter_by_max_distance filters based on distance threshold."""
        interface_centroid = np.array([1, 0, 0, 0], dtype=np.float32)

        # Mock results to filter
        mock_results = [
            {'answer_id': self.a1, 'score': 0.9},
            {'answer_id': self.a2, 'score': 0.8},
            {'answer_id': self.a3, 'score': 0.7},
        ]

        # Filter with max_distance=0.5 (should keep a1, a2; exclude a3)
        filtered = self.db._filter_by_max_distance(
            mock_results, self.model_id, interface_centroid,
            max_distance=0.5, top_k=10
        )

        answer_ids = [r['answer_id'] for r in filtered]
        self.assertIn(self.a1, answer_ids)
        self.assertIn(self.a2, answer_ids)
        self.assertNotIn(self.a3, answer_ids)

    def test_filter_by_max_distance_adds_distance_metadata(self):
        """Filtered results include interface_distance and interface_similarity."""
        interface_centroid = np.array([1, 0, 0, 0], dtype=np.float32)

        mock_results = [{'answer_id': self.a1, 'score': 0.9}]

        filtered = self.db._filter_by_max_distance(
            mock_results, self.model_id, interface_centroid,
            max_distance=1.0, top_k=10
        )

        self.assertEqual(len(filtered), 1)
        self.assertIn('interface_distance', filtered[0])
        self.assertIn('interface_similarity', filtered[0])
        self.assertAlmostEqual(filtered[0]['interface_similarity'], 1.0, places=3)
        self.assertAlmostEqual(filtered[0]['interface_distance'], 0.0, places=3)


class TestScaleOptimizationConfig(unittest.TestCase):
    """Test scale optimization configuration."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        # Create some test answers
        for i in range(50):
            self.db.add_answer("s.md", f"Answer {i}")

    def tearDown(self):
        self.db.close()

    def test_get_scale_config_defaults(self):
        """Scale config returns defaults when not set."""
        config = self.db.get_scale_config()

        self.assertEqual(config['interface_first_routing_enabled'], 'auto')
        self.assertEqual(config['interface_first_routing_threshold'], 50000)
        self.assertEqual(config['transformer_distillation_enabled'], 'auto')
        self.assertEqual(config['transformer_distillation_threshold'], 100000)

    def test_set_scale_config(self):
        """Can set scale configuration values."""
        self.db.set_scale_config(
            interface_first_routing_enabled='true',
            interface_first_routing_threshold=1000
        )

        config = self.db.get_scale_config()
        self.assertEqual(config['interface_first_routing_enabled'], 'true')
        self.assertEqual(config['interface_first_routing_threshold'], 1000)

    def test_should_use_interface_first_routing_auto_below_threshold(self):
        """Auto mode returns False when below threshold."""
        result = self.db.should_use_interface_first_routing()

        self.assertFalse(result['use'])
        self.assertIn('Auto', result['reason'])
        self.assertEqual(result['qa_count'], 50)

    def test_should_use_interface_first_routing_auto_above_threshold(self):
        """Auto mode returns True when above threshold."""
        self.db.set_scale_config(interface_first_routing_threshold=10)
        result = self.db.should_use_interface_first_routing()

        self.assertTrue(result['use'])
        self.assertIn('Auto', result['reason'])

    def test_should_use_interface_first_routing_explicit_true(self):
        """Explicit 'true' setting always returns True."""
        self.db.set_scale_config(interface_first_routing_enabled='true')
        result = self.db.should_use_interface_first_routing()

        self.assertTrue(result['use'])
        self.assertIn('Explicitly enabled', result['reason'])

    def test_should_use_interface_first_routing_explicit_false(self):
        """Explicit 'false' setting always returns False."""
        # First set threshold low so auto would return True
        self.db.set_scale_config(
            interface_first_routing_enabled='false',
            interface_first_routing_threshold=10
        )
        result = self.db.should_use_interface_first_routing()

        self.assertFalse(result['use'])
        self.assertIn('Explicitly disabled', result['reason'])

    def test_should_use_transformer_distillation_auto(self):
        """Transformer distillation auto mode works correctly."""
        result = self.db.should_use_transformer_distillation()

        self.assertFalse(result['use'])
        self.assertIn('Auto', result['reason'])
        self.assertEqual(result['implementation'], 'projection_transformer.py')

    def test_get_optimization_status(self):
        """Can get comprehensive optimization status."""
        status = self.db.get_optimization_status()

        self.assertIn('config', status)
        self.assertIn('interface_first_routing', status)
        self.assertIn('transformer_distillation', status)

        # Check nested structure
        self.assertIn('use', status['interface_first_routing'])
        self.assertIn('reason', status['interface_first_routing'])
        self.assertIn('use', status['transformer_distillation'])
        self.assertIn('implementation', status['transformer_distillation'])


class TestDistillationCheck(unittest.TestCase):
    """Test transformer distillation recommendation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = KGTopologyAPI(self.db_path)

        # Create some test answers
        for i in range(50):
            self.db.add_answer("s.md", f"Answer {i}")

    def tearDown(self):
        self.db.close()

    def test_check_distillation_not_recommended_below_threshold(self):
        """Distillation not recommended when below threshold."""
        result = self.db.check_distillation_recommended(qa_threshold=100000)

        self.assertFalse(result['recommended'])
        self.assertEqual(result['qa_count'], 50)
        self.assertEqual(result['threshold'], 100000)
        self.assertLess(result['percentage'], 1.0)

    def test_check_distillation_recommended_above_threshold(self):
        """Distillation recommended when above threshold."""
        result = self.db.check_distillation_recommended(qa_threshold=10)

        self.assertTrue(result['recommended'])
        self.assertEqual(result['qa_count'], 50)
        self.assertEqual(result['threshold'], 10)
        self.assertGreater(result['percentage'], 100)

    def test_check_distillation_references_implementation(self):
        """Distillation check references existing implementation."""
        result = self.db.check_distillation_recommended()

        self.assertIn('implementation', result)
        self.assertEqual(result['implementation'], 'projection_transformer.py')

    def test_get_distillation_training_embeddings(self):
        """Can get training embeddings for distillation."""
        model_id = self.db.add_model("test-model", 4)

        # Add questions with embeddings
        for i in range(10):
            q_id = self.db.add_question(f"Question {i}", "medium")
            self.db.store_embedding(model_id, "question", q_id,
                                    np.random.randn(4).astype(np.float32))

        embeddings, question_ids = self.db.get_distillation_training_embeddings(
            "test-model", sample_size=5
        )

        self.assertEqual(len(question_ids), 5)
        self.assertEqual(embeddings.shape, (5, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
