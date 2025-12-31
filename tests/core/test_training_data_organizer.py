#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Unit tests for Training Data Organizer

"""
Tests for training_data_organizer.py

Tests:
- Folder structure creation
- Export by seed level
- Pruning
- Storage statistics
- Selective loading
"""

import sys
import os
import json
import tempfile
import unittest
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from kg_topology_api import KGTopologyAPI
from training_data_organizer import TrainingDataOrganizer, create_training_structure


class TestCreateTrainingStructure(unittest.TestCase):
    """Test folder structure creation."""

    def test_creates_seed_directories(self):
        """Creates seed level directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_training_structure(tmpdir, seed_levels=3)

            for i in range(3):
                seed_dir = Path(tmpdir) / f"seed_{i}"
                self.assertTrue(seed_dir.exists())
                self.assertTrue(seed_dir.is_dir())

    def test_creates_metadata_file(self):
        """Creates metadata.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_training_structure(tmpdir, seed_levels=2)

            metadata_file = Path(tmpdir) / "metadata.json"
            self.assertTrue(metadata_file.exists())

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.assertEqual(metadata['seed_levels'], 2)
            self.assertIn('created_at', metadata)

    def test_creates_gitkeep_files(self):
        """Creates .gitkeep files for empty directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_training_structure(tmpdir, seed_levels=2)

            for i in range(2):
                gitkeep = Path(tmpdir) / f"seed_{i}" / ".gitkeep"
                self.assertTrue(gitkeep.exists())


class TestTrainingDataOrganizer(unittest.TestCase):
    """Test TrainingDataOrganizer class."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.data_dir = os.path.join(self.tmpdir, "training_data")

        self.db = KGTopologyAPI(self.db_path)
        self.organizer = TrainingDataOrganizer(self.db, self.data_dir)

        # Create test data
        self.a1 = self.db.add_answer("s.md", "Answer 1")
        self.q1 = self.db.add_question("Question 1", "medium")
        self.q2 = self.db.add_question("Question 2", "medium")
        self.q3 = self.db.add_question("Question 3", "medium")

        self.c1 = self.db.create_cluster("cluster1", [self.a1], [self.q1, self.q2])

        # Set seed levels
        self.db.set_seed_level(self.q1, 0)
        self.db.set_seed_level(self.q2, 1)
        self.db.set_seed_level(self.q3, 1)

    def tearDown(self):
        self.db.close()

    def test_export_creates_directories(self):
        """Export creates seed level directories."""
        self.organizer.export_by_seed_level()

        seed_0 = Path(self.data_dir) / "seed_0"
        seed_1 = Path(self.data_dir) / "seed_1"

        self.assertTrue(seed_0.exists())
        self.assertTrue(seed_1.exists())

    def test_export_creates_cluster_directories(self):
        """Export creates cluster directories within seed levels."""
        self.organizer.export_by_seed_level()

        # q1 is seed_0 in cluster1
        cluster_dir = Path(self.data_dir) / "seed_0" / f"cluster_{self.c1:04d}"
        self.assertTrue(cluster_dir.exists())

    def test_export_creates_questions_file(self):
        """Export creates questions.jsonl files."""
        self.organizer.export_by_seed_level()

        questions_file = Path(self.data_dir) / "seed_0" / f"cluster_{self.c1:04d}" / "questions.jsonl"
        self.assertTrue(questions_file.exists())

        # Read and verify content
        with open(questions_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)  # Only q1 is seed_0 in cluster1

    def test_export_returns_stats(self):
        """Export returns statistics."""
        stats = self.organizer.export_by_seed_level()

        self.assertIn('seed_0', stats)
        self.assertIn('seed_1', stats)

    def test_export_respects_max_seed_level(self):
        """Export respects max_seed_level parameter."""
        stats = self.organizer.export_by_seed_level(max_seed_level=0)

        self.assertIn('seed_0', stats)
        self.assertNotIn('seed_1', stats)

    def test_export_creates_metadata(self):
        """Export creates metadata.json."""
        self.organizer.export_by_seed_level()

        metadata_file = Path(self.data_dir) / "metadata.json"
        self.assertTrue(metadata_file.exists())


class TestPruning(unittest.TestCase):
    """Test pruning functionality."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.data_dir = os.path.join(self.tmpdir, "training_data")

        self.db = KGTopologyAPI(self.db_path)
        self.organizer = TrainingDataOrganizer(self.db, self.data_dir)

        # Create and export test data
        a1 = self.db.add_answer("s.md", "Answer")
        q1 = self.db.add_question("Q1", "medium")
        q2 = self.db.add_question("Q2", "medium")
        self.db.create_cluster("c1", [a1], [q1, q2])

        self.db.set_seed_level(q1, 0)
        self.db.set_seed_level(q2, 1)

        self.organizer.export_by_seed_level()

    def tearDown(self):
        self.db.close()

    def test_prune_dry_run(self):
        """Dry run reports but doesn't delete."""
        stats = self.organizer.prune_seed_level(1, dry_run=True)

        self.assertTrue(stats['dry_run'])
        self.assertNotIn('deleted', stats)

        # Directory should still exist
        seed_1 = Path(self.data_dir) / "seed_1"
        self.assertTrue(seed_1.exists())

    def test_prune_actual(self):
        """Actual prune deletes directory."""
        stats = self.organizer.prune_seed_level(1, dry_run=False)

        self.assertFalse(stats['dry_run'])
        self.assertTrue(stats['deleted'])

        # Directory should be gone
        seed_1 = Path(self.data_dir) / "seed_1"
        self.assertFalse(seed_1.exists())

    def test_prune_nonexistent(self):
        """Pruning nonexistent level returns error."""
        stats = self.organizer.prune_seed_level(99, dry_run=True)
        self.assertIn('error', stats)


class TestStorageStats(unittest.TestCase):
    """Test storage statistics."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.data_dir = os.path.join(self.tmpdir, "training_data")

        self.db = KGTopologyAPI(self.db_path)
        self.organizer = TrainingDataOrganizer(self.db, self.data_dir)

        # Create and export test data
        a1 = self.db.add_answer("s.md", "Answer")
        q1 = self.db.add_question("Q1", "medium")
        self.db.create_cluster("c1", [a1], [q1])
        self.db.set_seed_level(q1, 0)

        self.organizer.export_by_seed_level()

    def tearDown(self):
        self.db.close()

    def test_get_storage_stats(self):
        """Can get storage statistics."""
        stats = self.organizer.get_storage_stats()

        self.assertIn('seed_0', stats)
        self.assertIn('size_bytes', stats['seed_0'])
        self.assertIn('size_mb', stats['seed_0'])
        self.assertIn('clusters', stats['seed_0'])
        self.assertIn('questions', stats['seed_0'])


class TestSelectiveLoading(unittest.TestCase):
    """Test selective loading by seed level."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.data_dir = os.path.join(self.tmpdir, "training_data")

        self.db = KGTopologyAPI(self.db_path)
        self.organizer = TrainingDataOrganizer(self.db, self.data_dir)

        # Create and export test data
        a1 = self.db.add_answer("s.md", "Answer")
        self.q1 = self.db.add_question("Seed 0 question", "medium")
        self.q2 = self.db.add_question("Seed 1 question", "medium")
        self.c1 = self.db.create_cluster("c1", [a1], [self.q1, self.q2])

        self.db.set_seed_level(self.q1, 0)
        self.db.set_seed_level(self.q2, 1)

        self.organizer.export_by_seed_level()

    def tearDown(self):
        self.db.close()

    def test_load_single_seed_level(self):
        """Can load single seed level."""
        data = self.organizer.load_seed_levels([0])

        # Should have cluster with only seed_0 questions
        self.assertIn(self.c1, data)
        questions = data[self.c1]
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]['_seed_level'], 0)

    def test_load_multiple_seed_levels(self):
        """Can load multiple seed levels."""
        data = self.organizer.load_seed_levels([0, 1])

        self.assertIn(self.c1, data)
        questions = data[self.c1]
        self.assertEqual(len(questions), 2)

    def test_load_with_cluster_filter(self):
        """Can filter by cluster."""
        # Create another cluster
        a2 = self.db.add_answer("s2.md", "Answer 2")
        q3 = self.db.add_question("Other question", "medium")
        c2 = self.db.create_cluster("c2", [a2], [q3])
        self.db.set_seed_level(q3, 0)

        self.organizer.export_by_seed_level()

        # Load only c1
        data = self.organizer.load_seed_levels([0], clusters=[self.c1])

        self.assertIn(self.c1, data)
        self.assertNotIn(c2, data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
