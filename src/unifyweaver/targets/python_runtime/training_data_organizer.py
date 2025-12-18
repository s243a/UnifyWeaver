# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Training Data Organizer by Seed Level

"""
Organize training data into folders by seed level.

This supports the pruning strategy from SEED_QUESTION_TOPOLOGY.md:
- seed_0/: Original curated dataset (highest value, preserve)
- seed_1/: First expansion
- seed_n/: Nth expansion (most distant, lowest priority, delete first)

Directory structure:
    training_data/
    ├── seed_0/
    │   ├── cluster_001/
    │   │   ├── questions.jsonl
    │   │   └── answers.jsonl
    │   ├── cluster_002/
    │   └── ...
    ├── seed_1/
    │   ├── cluster_001/
    │   └── ...
    └── seed_n/
        └── ...
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from kg_topology_api import KGTopologyAPI


class TrainingDataOrganizer:
    """
    Organize training data into folders by seed level.

    Supports:
    - Exporting from database to seed-level folders
    - Importing from seed-level folders to database
    - Pruning higher seed levels
    - Selective loading by seed level
    """

    def __init__(self, db: KGTopologyAPI, base_dir: str):
        """
        Initialize organizer.

        Args:
            db: KG Topology database instance
            base_dir: Base directory for training data
        """
        self.db = db
        self.base_dir = Path(base_dir)

    def export_by_seed_level(
        self,
        max_seed_level: int = None,
        clusters: List[int] = None
    ) -> Dict[str, int]:
        """
        Export training data organized by seed level.

        Args:
            max_seed_level: Maximum seed level to export (None = all)
            clusters: Specific clusters to export (None = all)

        Returns:
            Stats dict with counts per seed level
        """
        stats = {}

        # Get all clusters if not specified
        if clusters is None:
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT cluster_id FROM qa_clusters")
            clusters = [row['cluster_id'] for row in cursor.fetchall()]

        # Determine seed levels to export
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT DISTINCT seed_level FROM question_seed_levels ORDER BY seed_level")
        seed_levels = [row['seed_level'] for row in cursor.fetchall()]

        if max_seed_level is not None:
            seed_levels = [l for l in seed_levels if l <= max_seed_level]

        # Export each seed level
        for seed_level in seed_levels:
            seed_dir = self.base_dir / f"seed_{seed_level}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            level_count = 0

            for cluster_id in clusters:
                questions = self.db.get_questions_at_seed_level(seed_level, cluster_id)

                if not questions:
                    continue

                cluster_dir = seed_dir / f"cluster_{cluster_id:04d}"
                cluster_dir.mkdir(exist_ok=True)

                # Export questions
                questions_file = cluster_dir / "questions.jsonl"
                with open(questions_file, 'w', encoding='utf-8') as f:
                    for q in questions:
                        f.write(json.dumps(q, ensure_ascii=False) + '\n')
                        level_count += 1

                # Export associated answers
                cursor.execute("""
                    SELECT DISTINCT a.*
                    FROM answers a
                    JOIN cluster_answers ca ON a.answer_id = ca.answer_id
                    WHERE ca.cluster_id = ?
                """, (cluster_id,))

                answers = [dict(row) for row in cursor.fetchall()]

                if answers:
                    answers_file = cluster_dir / "answers.jsonl"
                    with open(answers_file, 'w', encoding='utf-8') as f:
                        for a in answers:
                            f.write(json.dumps(a, ensure_ascii=False) + '\n')

            stats[f"seed_{seed_level}"] = level_count

        # Write metadata
        metadata = {
            'exported_at': datetime.now().isoformat(),
            'seed_levels': seed_levels,
            'cluster_count': len(clusters),
            'stats': stats
        }

        with open(self.base_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return stats

    def prune_seed_level(self, seed_level: int, dry_run: bool = True) -> Dict[str, Any]:
        """
        Prune (delete) a specific seed level.

        Higher seed levels should be pruned first (most distant from original).

        Args:
            seed_level: Seed level to prune
            dry_run: If True, only report what would be deleted

        Returns:
            Stats about what was/would be deleted
        """
        seed_dir = self.base_dir / f"seed_{seed_level}"

        if not seed_dir.exists():
            return {'error': f'Seed level {seed_level} directory not found'}

        # Count what would be deleted
        question_count = 0
        answer_count = 0
        cluster_count = 0

        for cluster_dir in seed_dir.iterdir():
            if cluster_dir.is_dir():
                cluster_count += 1

                questions_file = cluster_dir / "questions.jsonl"
                if questions_file.exists():
                    with open(questions_file, 'r') as f:
                        question_count += sum(1 for _ in f)

                answers_file = cluster_dir / "answers.jsonl"
                if answers_file.exists():
                    with open(answers_file, 'r') as f:
                        answer_count += sum(1 for _ in f)

        stats = {
            'seed_level': seed_level,
            'clusters': cluster_count,
            'questions': question_count,
            'answers': answer_count,
            'dry_run': dry_run
        }

        if not dry_run:
            shutil.rmtree(seed_dir)
            stats['deleted'] = True

        return stats

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics by seed level.

        Returns:
            Dict with size and count info per seed level
        """
        stats = {}

        for seed_dir in sorted(self.base_dir.iterdir()):
            if seed_dir.is_dir() and seed_dir.name.startswith('seed_'):
                seed_level = seed_dir.name

                # Calculate size
                total_size = sum(
                    f.stat().st_size
                    for f in seed_dir.rglob('*')
                    if f.is_file()
                )

                # Count clusters
                cluster_count = sum(
                    1 for d in seed_dir.iterdir()
                    if d.is_dir() and d.name.startswith('cluster_')
                )

                # Count questions
                question_count = 0
                for cluster_dir in seed_dir.iterdir():
                    if cluster_dir.is_dir():
                        qf = cluster_dir / "questions.jsonl"
                        if qf.exists():
                            with open(qf, 'r') as f:
                                question_count += sum(1 for _ in f)

                stats[seed_level] = {
                    'size_bytes': total_size,
                    'size_mb': round(total_size / (1024 * 1024), 2),
                    'clusters': cluster_count,
                    'questions': question_count
                }

        return stats

    def load_seed_levels(
        self,
        seed_levels: List[int],
        clusters: List[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load training data from specific seed levels.

        Args:
            seed_levels: List of seed levels to load
            clusters: Specific clusters to load (None = all)

        Returns:
            Dict mapping cluster_id to list of questions
        """
        data = {}

        for seed_level in seed_levels:
            seed_dir = self.base_dir / f"seed_{seed_level}"

            if not seed_dir.exists():
                continue

            for cluster_dir in seed_dir.iterdir():
                if not cluster_dir.is_dir():
                    continue

                # Extract cluster ID from directory name
                try:
                    cluster_id = int(cluster_dir.name.split('_')[1])
                except (IndexError, ValueError):
                    continue

                if clusters is not None and cluster_id not in clusters:
                    continue

                questions_file = cluster_dir / "questions.jsonl"
                if questions_file.exists():
                    if cluster_id not in data:
                        data[cluster_id] = []

                    with open(questions_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            q = json.loads(line.strip())
                            q['_seed_level'] = seed_level
                            data[cluster_id].append(q)

        return data


def create_training_structure(base_dir: str, seed_levels: int = 3) -> None:
    """
    Create empty training data folder structure.

    Args:
        base_dir: Base directory for training data
        seed_levels: Number of seed levels to create (default: 3)
    """
    base_path = Path(base_dir)

    for level in range(seed_levels):
        seed_dir = base_path / f"seed_{level}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Create a .gitkeep to track empty directories
        (seed_dir / ".gitkeep").touch()

    # Create metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'seed_levels': seed_levels,
        'description': 'Training data organized by seed level'
    }

    with open(base_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
