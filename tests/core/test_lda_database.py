#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Unit tests for LDA projection database layer

"""
Tests for lda_database.py

Tests:
- Schema creation
- CRUD operations for answers, questions, clusters
- Embedding storage and retrieval
- Projection storage and retrieval
- Search API
- Query logging
- Graph traversal
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

from lda_database import LDAProjectionDB


class TestLDADatabaseSchema(unittest.TestCase):
    """Test database schema creation."""

    def test_creates_database_file(self):
        """Database file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = LDAProjectionDB(db_path)
            self.assertTrue(os.path.exists(db_path))
            db.close()

    def test_creates_embeddings_dir(self):
        """Embeddings directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = LDAProjectionDB(db_path)
            self.assertTrue(os.path.isdir(db.embeddings_dir))
            db.close()

    def test_context_manager(self):
        """Database can be used as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with LDAProjectionDB(db_path) as db:
                self.assertIsNotNone(db.conn)


class TestAnswers(unittest.TestCase):
    """Test answer CRUD operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = LDAProjectionDB(self.db_path)

    def tearDown(self):
        self.db.close()

    def test_add_answer(self):
        """Can add an answer."""
        aid = self.db.add_answer(
            source_file="test.md",
            text="Test answer",
            record_id="test_001"
        )
        self.assertIsInstance(aid, int)
        self.assertGreater(aid, 0)

    def test_get_answer(self):
        """Can retrieve an answer."""
        aid = self.db.add_answer(
            source_file="test.md",
            text="Test answer text",
            record_id="test_001"
        )
        answer = self.db.get_answer(aid)
        self.assertEqual(answer['text'], "Test answer text")
        self.assertEqual(answer['source_file'], "test.md")
        self.assertEqual(answer['record_id'], "test_001")

    def test_get_nonexistent_answer(self):
        """Returns None for nonexistent answer."""
        answer = self.db.get_answer(9999)
        self.assertIsNone(answer)

    def test_add_answer_with_parent(self):
        """Can add hierarchical answers."""
        parent_id = self.db.add_answer("doc.md", "Full document")
        chunk_id = self.db.add_answer(
            source_file="doc.md",
            text="Chunk 1",
            parent_id=parent_id,
            chunk_index=0
        )
        chunk = self.db.get_answer(chunk_id)
        self.assertEqual(chunk['parent_id'], parent_id)
        self.assertEqual(chunk['chunk_index'], 0)

    def test_find_answers_by_source(self):
        """Can find answers by source file."""
        self.db.add_answer("source1.md", "Answer 1")
        self.db.add_answer("source1.md", "Answer 2")
        self.db.add_answer("source2.md", "Answer 3")

        results = self.db.find_answers(source_file="source1.md")
        self.assertEqual(len(results), 2)

    def test_find_answers_by_record_id(self):
        """Can find answers by record ID."""
        self.db.add_answer("s.md", "A1", record_id="rec001")
        self.db.add_answer("s.md", "A2", record_id="rec002")

        results = self.db.find_answers(record_id="rec001")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text'], "A1")


class TestRelations(unittest.TestCase):
    """Test answer relations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = LDAProjectionDB(self.db_path)

    def tearDown(self):
        self.db.close()

    def test_add_relation(self):
        """Can create relations between answers."""
        a1 = self.db.add_answer("s.md", "Full doc")
        a2 = self.db.add_answer("s.md", "Summary")

        rid = self.db.add_relation(a2, a1, "summarizes")
        self.assertIsInstance(rid, int)

    def test_get_related_outgoing(self):
        """Can get outgoing relations."""
        a1 = self.db.add_answer("s.md", "Full doc")
        a2 = self.db.add_answer("s.md", "Chunk")
        self.db.add_relation(a2, a1, "chunk_of")

        related = self.db.get_related(a2, direction="from")
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0]['relation_type'], "chunk_of")

    def test_get_related_incoming(self):
        """Can get incoming relations."""
        a1 = self.db.add_answer("s.md", "Full doc")
        a2 = self.db.add_answer("s.md", "Chunk")
        self.db.add_relation(a2, a1, "chunk_of")

        related = self.db.get_related(a1, direction="to")
        self.assertEqual(len(related), 1)

    def test_get_full_version(self):
        """Can traverse hierarchy to find full document."""
        full = self.db.add_answer("s.md", "Full document")
        summary = self.db.add_answer("s.md", "Summary")
        self.db.add_relation(summary, full, "summarizes")

        result = self.db.get_full_version(summary)
        self.assertEqual(result['answer_id'], full)

    def test_relation_with_metadata(self):
        """Can store relation metadata."""
        a1 = self.db.add_answer("s.md", "English")
        a2 = self.db.add_answer("s.md", "Spanish")
        self.db.add_relation(a2, a1, "translates", {"from_lang": "en", "to_lang": "es"})

        related = self.db.get_related(a2, "translates", direction="from")
        self.assertEqual(len(related), 1)


class TestQuestions(unittest.TestCase):
    """Test question operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = LDAProjectionDB(self.db_path)

    def tearDown(self):
        self.db.close()

    def test_add_question(self):
        """Can add a question."""
        qid = self.db.add_question("How do I test?", "medium")
        self.assertIsInstance(qid, int)

    def test_get_question(self):
        """Can retrieve a question."""
        qid = self.db.add_question("Test question", "short")
        question = self.db.get_question(qid)
        self.assertEqual(question['text'], "Test question")
        self.assertEqual(question['length_type'], "short")


class TestClusters(unittest.TestCase):
    """Test cluster operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = LDAProjectionDB(self.db_path)

    def tearDown(self):
        self.db.close()

    def test_create_cluster(self):
        """Can create a cluster."""
        a1 = self.db.add_answer("s.md", "Answer")
        q1 = self.db.add_question("Question 1", "short")
        q2 = self.db.add_question("Question 2", "medium")

        cid = self.db.create_cluster(
            name="test_cluster",
            answer_ids=[a1],
            question_ids=[q1, q2],
            description="Test cluster"
        )
        self.assertIsInstance(cid, int)

    def test_get_cluster(self):
        """Can retrieve cluster with answers and questions."""
        a1 = self.db.add_answer("s.md", "Answer")
        q1 = self.db.add_question("Question", "short")
        cid = self.db.create_cluster("test", [a1], [q1])

        cluster = self.db.get_cluster(cid)
        self.assertEqual(cluster['name'], "test")
        self.assertEqual(len(cluster['answers']), 1)
        self.assertEqual(len(cluster['questions']), 1)

    def test_list_clusters(self):
        """Can list all clusters."""
        a1 = self.db.add_answer("s.md", "A")
        q1 = self.db.add_question("Q", "short")
        self.db.create_cluster("c1", [a1], [q1])
        self.db.create_cluster("c2", [a1], [q1])

        clusters = self.db.list_clusters()
        self.assertEqual(len(clusters), 2)


class TestEmbeddings(unittest.TestCase):
    """Test embedding storage and retrieval."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = LDAProjectionDB(self.db_path)
        self.model_id = self.db.add_model("test-model", 384)

    def tearDown(self):
        self.db.close()

    def test_add_model(self):
        """Can register an embedding model."""
        mid = self.db.add_model("another-model", 768, backend="python")
        self.assertIsInstance(mid, int)

    def test_get_model(self):
        """Can retrieve model by name."""
        model = self.db.get_model("test-model")
        self.assertEqual(model['dimension'], 384)

    def test_store_embedding(self):
        """Can store an embedding vector."""
        aid = self.db.add_answer("s.md", "Test")
        vec = np.random.randn(384).astype(np.float32)

        eid = self.db.store_embedding(self.model_id, "answer", aid, vec)
        self.assertIsInstance(eid, int)

    def test_get_embedding(self):
        """Can retrieve an embedding vector."""
        aid = self.db.add_answer("s.md", "Test")
        vec = np.random.randn(384).astype(np.float32)
        self.db.store_embedding(self.model_id, "answer", aid, vec)

        retrieved = self.db.get_embedding(self.model_id, "answer", aid)
        np.testing.assert_array_almost_equal(vec, retrieved, decimal=5)

    def test_get_all_answer_embeddings(self):
        """Can retrieve all answer embeddings at once."""
        a1 = self.db.add_answer("s.md", "A1")
        a2 = self.db.add_answer("s.md", "A2")
        v1 = np.random.randn(384).astype(np.float32)
        v2 = np.random.randn(384).astype(np.float32)
        self.db.store_embedding(self.model_id, "answer", a1, v1)
        self.db.store_embedding(self.model_id, "answer", a2, v2)

        ids, matrix = self.db.get_all_answer_embeddings(self.model_id)
        self.assertEqual(len(ids), 2)
        self.assertEqual(matrix.shape, (2, 384))


class TestProjections(unittest.TestCase):
    """Test projection storage and retrieval."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = LDAProjectionDB(self.db_path)
        self.input_model_id = self.db.add_model("input-model", 384)
        self.output_model_id = self.db.add_model("output-model", 1024)

    def tearDown(self):
        self.db.close()

    def test_store_projection_symmetric(self):
        """Can store symmetric projection (same input/output model)."""
        W = np.random.randn(384, 384).astype(np.float32)
        pid = self.db.store_projection(
            input_model_id=self.input_model_id,
            output_model_id=self.input_model_id,
            W=W,
            name="test_proj"
        )
        self.assertIsInstance(pid, int)

    def test_store_projection_asymmetric(self):
        """Can store asymmetric projection (different input/output models)."""
        W = np.random.randn(1024, 384).astype(np.float32)
        pid = self.db.store_projection(
            input_model_id=self.input_model_id,
            output_model_id=self.output_model_id,
            W=W,
            name="asymmetric_proj"
        )
        self.assertIsInstance(pid, int)

    def test_get_projection(self):
        """Can retrieve projection matrix and metadata."""
        W = np.random.randn(384, 384).astype(np.float32)
        pid = self.db.store_projection(
            input_model_id=self.input_model_id,
            output_model_id=self.input_model_id,
            W=W,
            name="test",
            lambda_reg=0.5,
            ridge=1e-6
        )

        W_retrieved, metadata = self.db.get_projection(pid)
        np.testing.assert_array_almost_equal(W, W_retrieved, decimal=5)
        self.assertEqual(metadata['name'], "test")
        self.assertAlmostEqual(metadata['lambda_reg'], 0.5)

    def test_list_projections(self):
        """Can list projections with filters."""
        W1 = np.random.randn(384, 384).astype(np.float32)
        W2 = np.random.randn(1024, 384).astype(np.float32)
        self.db.store_projection(self.input_model_id, self.input_model_id, W1, "sym")
        self.db.store_projection(self.input_model_id, self.output_model_id, W2, "asym")

        all_proj = self.db.list_projections()
        self.assertEqual(len(all_proj), 2)

        filtered = self.db.list_projections(output_model="output-model")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['name'], "asym")


class TestSearch(unittest.TestCase):
    """Test search API."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = LDAProjectionDB(self.db_path)
        self.model_id = self.db.add_model("test-model", 4)

        # Create simple test data
        # Answer 1: [1, 0, 0, 0]
        # Answer 2: [0, 1, 0, 0]
        self.a1 = self.db.add_answer("s.md", "Answer 1", record_id="a1")
        self.a2 = self.db.add_answer("s.md", "Answer 2", record_id="a2")
        self.db.store_embedding(self.model_id, "answer", self.a1, np.array([1, 0, 0, 0], dtype=np.float32))
        self.db.store_embedding(self.model_id, "answer", self.a2, np.array([0, 1, 0, 0], dtype=np.float32))

        # Identity projection
        W = np.eye(4, dtype=np.float32)
        self.proj_id = self.db.store_projection(self.model_id, self.model_id, W, "identity")

    def tearDown(self):
        self.db.close()

    def test_search_finds_closest(self):
        """Search returns closest answer."""
        query = np.array([0.9, 0.1, 0, 0], dtype=np.float32)
        results = self.db.search(query, self.proj_id, top_k=2, log=False)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['record_id'], "a1")  # Closer to [1,0,0,0]

    def test_search_logs_query(self):
        """Search logs queries when enabled."""
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        self.db.search(query, self.proj_id, log=True, query_text="test query")

        logs = self.db.get_query_log()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]['query_text'], "test query")


class TestQueryLog(unittest.TestCase):
    """Test query logging functionality."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = LDAProjectionDB(self.db_path)
        self.model_id = self.db.add_model("test", 4)
        W = np.eye(4, dtype=np.float32)
        self.proj_id = self.db.store_projection(self.model_id, self.model_id, W)

    def tearDown(self):
        self.db.close()

    def test_log_query(self):
        """Can log a query."""
        self.db.log_query(
            query_text="test query",
            projection_id=self.proj_id,
            results=[1, 2, 3]
        )
        logs = self.db.get_query_log()
        self.assertEqual(len(logs), 1)

    def test_update_feedback(self):
        """Can update query feedback."""
        self.db.log_query("test", self.proj_id, [1])
        logs = self.db.get_query_log()
        log_id = logs[0]['log_id']

        self.db.update_query_feedback(log_id, selected_id=1, was_helpful=True)
        updated = self.db.get_query_log()[0]
        self.assertEqual(updated['selected_answer_id'], 1)
        self.assertTrue(updated['was_helpful'])


class TestStats(unittest.TestCase):
    """Test statistics retrieval."""

    def test_get_stats(self):
        """Can get database statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with LDAProjectionDB(db_path) as db:
                db.add_answer("s.md", "Answer")
                db.add_question("Question", "short")

                stats = db.get_stats()
                self.assertEqual(stats['num_answers'], 1)
                self.assertEqual(stats['num_questions'], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
