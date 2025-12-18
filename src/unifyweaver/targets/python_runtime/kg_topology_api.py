# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Knowledge Graph Topology API for Q-A Systems

"""
Knowledge Graph Topology API extending LDA Projection Database.

Implements Phase 1 of the KG Topology Roadmap:
- 11 relation types across 3 categories
- Hash-based anchor linking
- Seed level provenance tracking
- Graph traversal API
- Search with knowledge graph context

See: docs/proposals/ROADMAP_KG_TOPOLOGY.md
     docs/proposals/QA_KNOWLEDGE_GRAPH.md
     docs/proposals/SEED_QUESTION_TOPOLOGY.md
"""

import sqlite3
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np

from lda_database import LDAProjectionDB


# =============================================================================
# RELATION TYPE DEFINITIONS
# =============================================================================

# Learning Flow relations (4)
LEARNING_FLOW_RELATIONS = {
    'foundational',   # A is foundational concept for B
    'preliminary',    # A is prerequisite step for B
    'compositional',  # B extends/builds upon A
    'transitional',   # B is natural next step after A
}

# Scope relations (2)
SCOPE_RELATIONS = {
    'refined',  # B is more specific variant of A
    'general',  # A is broader in scope than B
}

# Abstraction relations (5)
ABSTRACTION_RELATIONS = {
    'generalization',   # B is abstract pattern of A
    'implementation',   # A is code realizing pattern B
    'axiomatization',   # B is abstract theory of A
    'instance',         # A is domain satisfying theory B
    'example',          # A illustrates/demonstrates B
}

ALL_RELATION_TYPES = LEARNING_FLOW_RELATIONS | SCOPE_RELATIONS | ABSTRACTION_RELATIONS

# Direction mappings: incoming = target depends on source
INCOMING_RELATIONS = {
    'foundational', 'preliminary', 'general',
    'implementation', 'instance', 'example'
}
OUTGOING_RELATIONS = {
    'compositional', 'transitional', 'refined',
    'generalization', 'axiomatization'
}


class KGTopologyAPI(LDAProjectionDB):
    """
    Knowledge Graph Topology API extending LDA Projection Database.

    Adds:
    - 11 relation types for Q-A knowledge graphs
    - Hash-based anchor linking
    - Seed level provenance tracking
    - Graph traversal API
    - Search with knowledge graph context
    """

    SCHEMA_VERSION = 2  # Extends v1 with KG topology

    def __init__(self, db_path: str, embeddings_dir: str = None):
        """Initialize with KG topology extensions."""
        super().__init__(db_path, embeddings_dir)
        self._init_kg_schema()

    def _init_kg_schema(self):
        """Create KG topology tables if they don't exist."""
        cursor = self.conn.cursor()

        # Seed level tracking for questions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS question_seed_levels (
                question_id INTEGER PRIMARY KEY REFERENCES questions(question_id),
                seed_level INTEGER NOT NULL DEFAULT 0,
                discovered_from_question_id INTEGER REFERENCES questions(question_id),
                discovery_relation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Anchor linking: which question generated which answer
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answer_anchors (
                answer_id INTEGER PRIMARY KEY REFERENCES answers(answer_id),
                anchor_question_hash TEXT NOT NULL,
                seed_level INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for seed level queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_seed_level
            ON question_seed_levels(seed_level)
        """)

        # Index for cluster + seed level queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_question_seed_cluster
            ON question_seed_levels(seed_level)
        """)

        self.conn.commit()

    # =========================================================================
    # RELATION MANAGEMENT
    # =========================================================================

    def add_kg_relation(
        self,
        from_answer_id: int,
        to_answer_id: int,
        relation_type: str,
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Add a knowledge graph relation between answers.

        Args:
            from_answer_id: Source answer ID
            to_answer_id: Target answer ID
            relation_type: One of the 11 KG relation types
            metadata: Optional JSON metadata

        Returns:
            relation_id of the created relation
        """
        if relation_type not in ALL_RELATION_TYPES:
            raise ValueError(f"Invalid relation type: {relation_type}. "
                           f"Must be one of: {ALL_RELATION_TYPES}")

        return self.add_relation(from_answer_id, to_answer_id, relation_type, metadata)

    def get_relations(
        self,
        answer_id: int,
        relation_type: str,
        direction: str = 'outgoing'
    ) -> List[Dict[str, Any]]:
        """
        Get relations for an answer.

        Args:
            answer_id: The answer to get relations for
            relation_type: The type of relation
            direction: 'outgoing' (from this answer) or 'incoming' (to this answer)

        Returns:
            List of related answers with metadata
        """
        cursor = self.conn.cursor()

        if direction == 'outgoing':
            cursor.execute("""
                SELECT ar.relation_id, ar.to_answer_id as related_id,
                       ar.relation_type, ar.metadata,
                       a.text, a.source_file, a.record_id
                FROM answer_relations ar
                JOIN answers a ON ar.to_answer_id = a.answer_id
                WHERE ar.from_answer_id = ? AND ar.relation_type = ?
            """, (answer_id, relation_type))
        else:  # incoming
            cursor.execute("""
                SELECT ar.relation_id, ar.from_answer_id as related_id,
                       ar.relation_type, ar.metadata,
                       a.text, a.source_file, a.record_id
                FROM answer_relations ar
                JOIN answers a ON ar.from_answer_id = a.answer_id
                WHERE ar.to_answer_id = ? AND ar.relation_type = ?
            """, (answer_id, relation_type))

        results = []
        for row in cursor.fetchall():
            result = {
                'relation_id': row['relation_id'],
                'answer_id': row['related_id'],
                'relation_type': row['relation_type'],
                'text': row['text'][:500] if row['text'] else '',
                'source_file': row['source_file'],
                'record_id': row['record_id']
            }
            if row['metadata']:
                result['metadata'] = json.loads(row['metadata'])
            results.append(result)

        return results

    # =========================================================================
    # GRAPH TRAVERSAL API
    # =========================================================================

    def get_foundational(self, answer_id: int) -> List[Dict[str, Any]]:
        """Get foundational concepts this answer depends on."""
        return self.get_relations(answer_id, 'foundational', 'incoming')

    def get_prerequisites(self, answer_id: int) -> List[Dict[str, Any]]:
        """Get practical prerequisites required before this answer."""
        return self.get_relations(answer_id, 'preliminary', 'incoming')

    def get_extensions(self, answer_id: int) -> List[Dict[str, Any]]:
        """Get answers that extend or build upon this one."""
        return self.get_relations(answer_id, 'compositional', 'outgoing')

    def get_next_steps(self, answer_id: int) -> List[Dict[str, Any]]:
        """Get natural next steps after this answer."""
        return self.get_relations(answer_id, 'transitional', 'outgoing')

    def get_refined(self, answer_id: int) -> List[Dict[str, Any]]:
        """Get more specific variants of this answer."""
        return self.get_relations(answer_id, 'refined', 'outgoing')

    def get_general(self, answer_id: int) -> List[Dict[str, Any]]:
        """Get broader/more general versions of this answer."""
        return self.get_relations(answer_id, 'general', 'incoming')

    def get_generalizations(self, answer_id: int) -> List[Dict[str, Any]]:
        """Get abstract patterns derived from this answer."""
        return self.get_relations(answer_id, 'generalization', 'outgoing')

    def get_implementations(self, answer_id: int) -> List[Dict[str, Any]]:
        """Get code implementations of this pattern."""
        return self.get_relations(answer_id, 'implementation', 'incoming')

    def get_instances(self, answer_id: int) -> List[Dict[str, Any]]:
        """Get domain instances that satisfy this theory."""
        return self.get_relations(answer_id, 'instance', 'incoming')

    def get_examples(self, answer_id: int) -> List[Dict[str, Any]]:
        """Get pedagogical examples demonstrating this concept."""
        return self.get_relations(answer_id, 'example', 'incoming')

    # =========================================================================
    # ANCHOR LINKING
    # =========================================================================

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Compute SHA-256 hash of content for anchor linking."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def set_anchor_question(
        self,
        answer_id: int,
        question_text: str,
        seed_level: int = 0
    ) -> str:
        """
        Set the anchor question for an answer.

        Args:
            answer_id: The answer to set anchor for
            question_text: The original question text
            seed_level: The seed level of the question

        Returns:
            The computed question hash
        """
        question_hash = self.compute_content_hash(question_text)

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO answer_anchors
            (answer_id, anchor_question_hash, seed_level)
            VALUES (?, ?, ?)
        """, (answer_id, question_hash, seed_level))
        self.conn.commit()

        return question_hash

    def get_anchor_question(self, answer_id: int) -> Optional[str]:
        """Get the anchor question hash for an answer."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT anchor_question_hash
            FROM answer_anchors
            WHERE answer_id = ?
        """, (answer_id,))

        row = cursor.fetchone()
        return row['anchor_question_hash'] if row else None

    # =========================================================================
    # SEED LEVEL TRACKING
    # =========================================================================

    def set_seed_level(
        self,
        question_id: int,
        seed_level: int,
        discovered_from: int = None,
        discovery_relation: str = None
    ):
        """
        Set the seed level for a question.

        Args:
            question_id: The question ID
            seed_level: The seed level (0 = original, n = expansion depth)
            discovered_from: ID of question this was discovered from (optional)
            discovery_relation: How it was discovered (optional)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO question_seed_levels
            (question_id, seed_level, discovered_from_question_id, discovery_relation)
            VALUES (?, ?, ?, ?)
        """, (question_id, seed_level, discovered_from, discovery_relation))
        self.conn.commit()

    def get_seed_level(self, question_id: int) -> Optional[int]:
        """Get the seed level of a question."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT seed_level
            FROM question_seed_levels
            WHERE question_id = ?
        """, (question_id,))

        row = cursor.fetchone()
        return row['seed_level'] if row else None

    def get_questions_at_seed_level(
        self,
        seed_level: int,
        cluster_id: int = None
    ) -> List[Dict[str, Any]]:
        """
        Get all questions at a specific seed level.

        Args:
            seed_level: The seed level to query
            cluster_id: Optional cluster filter

        Returns:
            List of questions with their metadata
        """
        cursor = self.conn.cursor()

        if cluster_id is not None:
            cursor.execute("""
                SELECT q.question_id, q.text, q.length_type,
                       sl.seed_level, sl.discovered_from_question_id
                FROM questions q
                JOIN question_seed_levels sl ON q.question_id = sl.question_id
                JOIN cluster_questions cq ON q.question_id = cq.question_id
                WHERE sl.seed_level = ? AND cq.cluster_id = ?
            """, (seed_level, cluster_id))
        else:
            cursor.execute("""
                SELECT q.question_id, q.text, q.length_type,
                       sl.seed_level, sl.discovered_from_question_id
                FROM questions q
                JOIN question_seed_levels sl ON q.question_id = sl.question_id
                WHERE sl.seed_level = ?
            """, (seed_level,))

        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # ENHANCED SEARCH
    # =========================================================================

    def search_with_context(
        self,
        query_text: str,
        model_name: str,
        mh_projection_id: int = None,
        top_k: int = 5,
        include_foundational: bool = True,
        include_prerequisites: bool = True,
        include_extensions: bool = True,
        include_next_steps: bool = True,
        context_depth: int = 1,
        use_direct_search: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with knowledge graph context.

        Performs semantic search, then enriches results with related
        answers from the knowledge graph.

        Search Methods:
        - **multi_head_search** (default): Uses softmax routing over cluster
          centroids to project the query, then searches. This is the LDA
          projection approach from MULTI_HEAD_PROJECTION_THEORY.md.
        - **direct_search**: Raw cosine similarity without projection.
          Useful as baseline or when no projection is defined.

        Args:
            query_text: The search query
            model_name: Embedding model name
            mh_projection_id: Multi-head projection ID (uses multi_head_search)
            top_k: Number of results
            include_foundational: Include foundational concepts
            include_prerequisites: Include prerequisites
            include_extensions: Include extensions
            include_next_steps: Include next steps
            context_depth: How many hops to traverse (default: 1)
            use_direct_search: Force direct search without projection (baseline)

        Returns:
            List of results with knowledge graph context
        """
        query_embedding = self._embed_query(query_text, model_name)

        # Choose search method
        if use_direct_search or mh_projection_id is None:
            # Direct search: raw cosine similarity (baseline, no projection)
            base_results = self._direct_search(query_text, model_name, top_k)
        else:
            # Multi-head search: softmax routing + projection (recommended)
            # This uses the existing implementation from lda_database.py
            base_results = self.multi_head_search(
                query_embedding=query_embedding,
                mh_projection_id=mh_projection_id,
                top_k=top_k,
                log=False,
                query_text=query_text
            )

        # Enrich with graph context
        enriched_results = []
        for result in base_results:
            answer_id = result['answer_id']
            enriched = dict(result)

            if include_foundational:
                enriched['foundational'] = self._traverse_relations(
                    answer_id, 'foundational', 'incoming', context_depth
                )

            if include_prerequisites:
                enriched['prerequisites'] = self._traverse_relations(
                    answer_id, 'preliminary', 'incoming', context_depth
                )

            if include_extensions:
                enriched['extensions'] = self._traverse_relations(
                    answer_id, 'compositional', 'outgoing', context_depth
                )

            if include_next_steps:
                enriched['next_steps'] = self._traverse_relations(
                    answer_id, 'transitional', 'outgoing', context_depth
                )

            enriched_results.append(enriched)

        return enriched_results

    def _traverse_relations(
        self,
        answer_id: int,
        relation_type: str,
        direction: str,
        depth: int
    ) -> List[Dict[str, Any]]:
        """Traverse relations up to specified depth."""
        if depth <= 0:
            return []

        direct_relations = self.get_relations(answer_id, relation_type, direction)

        if depth == 1:
            return direct_relations

        # For depth > 1, recursively traverse
        all_relations = list(direct_relations)
        seen_ids = {answer_id}

        for rel in direct_relations:
            related_id = rel['answer_id']
            if related_id not in seen_ids:
                seen_ids.add(related_id)
                deeper = self._traverse_relations(
                    related_id, relation_type, direction, depth - 1
                )
                all_relations.extend(deeper)

        return all_relations

    def _embed_query(self, query_text: str, model_name: str) -> np.ndarray:
        """Embed a query using the specified model."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            embedding = model.encode(query_text, convert_to_numpy=True)
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except ImportError:
            raise RuntimeError("sentence-transformers not installed")

    def _direct_search(
        self,
        query_text: str,
        model_name: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Direct semantic search via matrix multiplication (BASELINE).

        This computes raw cosine similarity: query × all_answers.
        No learned projection is applied.

        Use cases:
        - Baseline comparison against multi_head_search
        - When no multi-head projection is defined
        - Future: 1:1 Q-A mappings after answer smoothing

        For production use, prefer multi_head_search() which applies
        learned LDA projection for better retrieval accuracy.
        See: MULTI_HEAD_PROJECTION_THEORY.md
        """
        query_emb = self._embed_query(query_text, model_name)

        # Get model info
        model_info = self.get_model(model_name)
        if not model_info:
            return []

        model_id = model_info['model_id']

        # Get all answer embeddings
        answer_ids, answer_matrix = self.get_all_answer_embeddings(model_id)

        if len(answer_ids) == 0:
            return []

        # Normalize answer embeddings
        answer_norms = np.linalg.norm(answer_matrix, axis=1, keepdims=True)
        answer_norms = np.where(answer_norms > 0, answer_norms, 1)
        answer_matrix_normed = answer_matrix / answer_norms

        # Direct matrix multiplication: query × all_answers
        scores = answer_matrix_normed @ query_emb

        # Get top-k
        top_indices = np.argsort(-scores)[:top_k]

        # Build results
        results = []
        for idx in top_indices:
            answer_id = answer_ids[idx]
            answer = self.get_answer(answer_id)
            results.append({
                'answer_id': answer_id,
                'score': float(scores[idx]),
                'text': answer['text'][:500] if answer else '',
                'record_id': answer.get('record_id', '') if answer else '',
                'source_file': answer.get('source_file', '') if answer else ''
            })

        return results

    # =========================================================================
    # LEARNING PATH GENERATION
    # =========================================================================

    def get_learning_path(
        self,
        answer_id: int,
        include_foundational: bool = True,
        include_prerequisites: bool = True,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate an ordered learning path leading to this answer.

        Traverses foundational and preliminary relations to build
        a suggested learning order.

        Args:
            answer_id: Target answer
            include_foundational: Include foundational concepts
            include_prerequisites: Include prerequisites
            max_depth: Maximum traversal depth

        Returns:
            Ordered list of answers to learn first
        """
        path = []
        visited = {answer_id}

        def collect_dependencies(aid: int, depth: int):
            if depth <= 0:
                return

            deps = []
            if include_foundational:
                deps.extend(self.get_foundational(aid))
            if include_prerequisites:
                deps.extend(self.get_prerequisites(aid))

            for dep in deps:
                dep_id = dep['answer_id']
                if dep_id not in visited:
                    visited.add(dep_id)
                    # Recursively collect deeper dependencies first
                    collect_dependencies(dep_id, depth - 1)
                    path.append(dep)

        collect_dependencies(answer_id, max_depth)

        # Reverse to get learning order (deepest dependencies first)
        return list(reversed(path))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_kg_database(db_path: str, embeddings_dir: str = None) -> KGTopologyAPI:
    """Create a new KG topology database."""
    return KGTopologyAPI(db_path, embeddings_dir)
