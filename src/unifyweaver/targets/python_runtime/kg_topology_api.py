# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Knowledge Graph Topology API for Q-A Systems

"""
Knowledge Graph Topology API extending LDA Projection Database.

Implements Phases 1, 2 & 3 of the KG Topology Roadmap:

Phase 1:
- 11 relation types across 3 categories
- Hash-based anchor linking
- Seed level provenance tracking
- Graph traversal API
- Search with knowledge graph context

Phase 2:
- Semantic interfaces (presentation layer for focused identities)
- Interface schema (centroid, topics, exposed clusters)
- Query-to-interface mapping
- Interface management (auto-generate, curation, metrics)

Phase 3:
- Distributed network with Kleinberg routing
- Node discovery and registration
- Inter-node query routing with HTL limits
- Path folding for shortcut creation

See: docs/proposals/ROADMAP_KG_TOPOLOGY.md
     docs/proposals/QA_KNOWLEDGE_GRAPH.md
     docs/proposals/SEED_QUESTION_TOPOLOGY.md
"""

import os
import sqlite3
import json
import hashlib
import base64
from typing import List, Dict, Any, Optional, Tuple, Set
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

    Phase 1 Features:
    - 11 relation types for Q-A knowledge graphs
    - Hash-based anchor linking
    - Seed level provenance tracking
    - Graph traversal API
    - Search with knowledge graph context

    Phase 2 Features:
    - Semantic interfaces (focused presentation layer)
    - Interface schema with centroids and topics
    - Query-to-interface mapping
    - Interface management (auto-generate, curation, metrics)
    """

    SCHEMA_VERSION = 3  # v2 = Phase 1, v3 = Phase 2 interfaces

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

        # =====================================================================
        # PHASE 2: SEMANTIC INTERFACES
        # =====================================================================

        # Semantic interfaces: presentation layer for focused identities
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_interfaces (
                interface_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                topics TEXT,  -- JSON array of topic keywords
                centroid_model_id INTEGER REFERENCES embedding_models(model_id),
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Interface centroids: embedding vectors for each interface
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interface_centroids (
                interface_id INTEGER REFERENCES semantic_interfaces(interface_id),
                model_id INTEGER REFERENCES embedding_models(model_id),
                centroid BLOB NOT NULL,  -- numpy array serialized
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (interface_id, model_id)
            )
        """)

        # Interface-cluster mapping: which clusters belong to which interface
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interface_clusters (
                interface_id INTEGER REFERENCES semantic_interfaces(interface_id),
                cluster_id INTEGER REFERENCES clusters(cluster_id),
                weight REAL DEFAULT 1.0,  -- how strongly this cluster belongs
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (interface_id, cluster_id)
            )
        """)

        # Interface metrics: health and coverage tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interface_metrics (
                interface_id INTEGER REFERENCES semantic_interfaces(interface_id),
                metric_name TEXT NOT NULL,
                metric_value REAL,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (interface_id, metric_name)
            )
        """)

        # Index for active interfaces
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_active_interfaces
            ON semantic_interfaces(is_active)
        """)

        # =====================================================================
        # ANSWER PREREQUISITES CENTROIDS
        # =====================================================================

        # Per-answer prerequisites centroids: computed from prerequisite relations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answer_prerequisites_centroids (
                answer_id INTEGER REFERENCES answers(answer_id),
                model_id INTEGER REFERENCES embedding_models(model_id),
                centroid BLOB NOT NULL,
                source_type TEXT DEFAULT 'metadata',  -- 'metadata', 'semantic', 'hybrid'
                source_answers TEXT,  -- JSON array of answer_ids used to compute
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (answer_id, model_id)
            )
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

    # =========================================================================
    # PHASE 2: SEMANTIC INTERFACES
    # =========================================================================

    def create_interface(
        self,
        name: str,
        description: str = None,
        topics: List[str] = None
    ) -> int:
        """
        Create a new semantic interface.

        An interface is a presentation layer that exposes a focused subset
        of the knowledge base. It has its own centroid and can route queries
        to relevant clusters.

        Args:
            name: Unique interface name
            description: Human-readable description
            topics: List of topic keywords

        Returns:
            interface_id of the created interface
        """
        cursor = self.conn.cursor()
        topics_json = json.dumps(topics) if topics else None

        cursor.execute("""
            INSERT INTO semantic_interfaces (name, description, topics)
            VALUES (?, ?, ?)
        """, (name, description, topics_json))

        self.conn.commit()
        return cursor.lastrowid

    def get_interface(self, interface_id: int) -> Optional[Dict[str, Any]]:
        """Get interface by ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT interface_id, name, description, topics,
                   centroid_model_id, is_active, created_at, updated_at
            FROM semantic_interfaces
            WHERE interface_id = ?
        """, (interface_id,))

        row = cursor.fetchone()
        if row:
            result = dict(row)
            if result['topics']:
                result['topics'] = json.loads(result['topics'])
            return result
        return None

    def get_interface_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get interface by name."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT interface_id, name, description, topics,
                   centroid_model_id, is_active, created_at, updated_at
            FROM semantic_interfaces
            WHERE name = ?
        """, (name,))

        row = cursor.fetchone()
        if row:
            result = dict(row)
            if result['topics']:
                result['topics'] = json.loads(result['topics'])
            return result
        return None

    def list_interfaces(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all interfaces."""
        cursor = self.conn.cursor()

        if active_only:
            cursor.execute("""
                SELECT interface_id, name, description, topics,
                       centroid_model_id, is_active, created_at
                FROM semantic_interfaces
                WHERE is_active = 1
                ORDER BY name
            """)
        else:
            cursor.execute("""
                SELECT interface_id, name, description, topics,
                       centroid_model_id, is_active, created_at
                FROM semantic_interfaces
                ORDER BY name
            """)

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result['topics']:
                result['topics'] = json.loads(result['topics'])
            results.append(result)

        return results

    def update_interface(
        self,
        interface_id: int,
        name: str = None,
        description: str = None,
        topics: List[str] = None,
        is_active: bool = None
    ):
        """Update interface properties."""
        cursor = self.conn.cursor()

        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if topics is not None:
            updates.append("topics = ?")
            params.append(json.dumps(topics))
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(is_active)

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(interface_id)

            cursor.execute(f"""
                UPDATE semantic_interfaces
                SET {', '.join(updates)}
                WHERE interface_id = ?
            """, params)
            self.conn.commit()

    def delete_interface(self, interface_id: int):
        """Delete an interface and its associated data."""
        cursor = self.conn.cursor()

        # Delete in order: metrics, clusters mapping, centroids, interface
        cursor.execute("DELETE FROM interface_metrics WHERE interface_id = ?",
                      (interface_id,))
        cursor.execute("DELETE FROM interface_clusters WHERE interface_id = ?",
                      (interface_id,))
        cursor.execute("DELETE FROM interface_centroids WHERE interface_id = ?",
                      (interface_id,))
        cursor.execute("DELETE FROM semantic_interfaces WHERE interface_id = ?",
                      (interface_id,))

        self.conn.commit()

    # =========================================================================
    # INTERFACE-CLUSTER MAPPING
    # =========================================================================

    def add_cluster_to_interface(
        self,
        interface_id: int,
        cluster_id: int,
        weight: float = 1.0
    ):
        """
        Add a cluster to an interface.

        Args:
            interface_id: Target interface
            cluster_id: Cluster to add
            weight: How strongly this cluster belongs (0-1)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO interface_clusters
            (interface_id, cluster_id, weight)
            VALUES (?, ?, ?)
        """, (interface_id, cluster_id, weight))
        self.conn.commit()

    def remove_cluster_from_interface(self, interface_id: int, cluster_id: int):
        """Remove a cluster from an interface."""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM interface_clusters
            WHERE interface_id = ? AND cluster_id = ?
        """, (interface_id, cluster_id))
        self.conn.commit()

    def get_interface_clusters(self, interface_id: int) -> List[Dict[str, Any]]:
        """Get all clusters belonging to an interface."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ic.cluster_id, ic.weight, c.name as cluster_name
            FROM interface_clusters ic
            JOIN qa_clusters c ON ic.cluster_id = c.cluster_id
            WHERE ic.interface_id = ?
            ORDER BY ic.weight DESC
        """, (interface_id,))

        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # CLUSTER CENTROIDS (Phase 2 extension)
    # =========================================================================

    def set_cluster_centroid(
        self,
        cluster_id: int,
        model_id: int,
        centroid: np.ndarray
    ):
        """
        Set the centroid embedding for a cluster.

        This is a simple centroid storage separate from multi-head projections.
        Used for Phase 2 interface centroid computation.

        Args:
            cluster_id: Cluster to set centroid for
            model_id: Embedding model ID
            centroid: Centroid embedding vector
        """
        cursor = self.conn.cursor()

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_centroids (
                cluster_id INTEGER REFERENCES qa_clusters(cluster_id),
                model_id INTEGER REFERENCES embedding_models(model_id),
                centroid BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (cluster_id, model_id)
            )
        """)

        centroid_blob = centroid.tobytes()

        cursor.execute("""
            INSERT OR REPLACE INTO cluster_centroids
            (cluster_id, model_id, centroid)
            VALUES (?, ?, ?)
        """, (cluster_id, model_id, centroid_blob))

        self.conn.commit()

    def get_cluster_centroid(
        self,
        cluster_id: int,
        model_id: int
    ) -> Optional[np.ndarray]:
        """Get the centroid embedding for a cluster."""
        cursor = self.conn.cursor()

        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='cluster_centroids'
        """)
        if not cursor.fetchone():
            return None

        cursor.execute("""
            SELECT centroid
            FROM cluster_centroids
            WHERE cluster_id = ? AND model_id = ?
        """, (cluster_id, model_id))

        row = cursor.fetchone()
        if row and row['centroid']:
            return np.frombuffer(row['centroid'], dtype=np.float32)
        return None

    # =========================================================================
    # INTERFACE CENTROIDS
    # =========================================================================

    def set_interface_centroid(
        self,
        interface_id: int,
        model_id: int,
        centroid: np.ndarray
    ):
        """
        Set the centroid embedding for an interface.

        Args:
            interface_id: Interface to update
            model_id: Embedding model ID
            centroid: Centroid embedding vector
        """
        cursor = self.conn.cursor()

        # Serialize numpy array
        centroid_blob = centroid.tobytes()

        cursor.execute("""
            INSERT OR REPLACE INTO interface_centroids
            (interface_id, model_id, centroid)
            VALUES (?, ?, ?)
        """, (interface_id, model_id, centroid_blob))

        # Update the interface's primary centroid model
        cursor.execute("""
            UPDATE semantic_interfaces
            SET centroid_model_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE interface_id = ?
        """, (model_id, interface_id))

        self.conn.commit()

    def get_interface_centroid(
        self,
        interface_id: int,
        model_id: int
    ) -> Optional[np.ndarray]:
        """Get the centroid embedding for an interface."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT centroid
            FROM interface_centroids
            WHERE interface_id = ? AND model_id = ?
        """, (interface_id, model_id))

        row = cursor.fetchone()
        if row and row['centroid']:
            return np.frombuffer(row['centroid'], dtype=np.float32)
        return None

    def compute_interface_centroid(
        self,
        interface_id: int,
        model_id: int
    ) -> Optional[np.ndarray]:
        """
        Compute interface centroid from its clusters' centroids.

        Weighted average of cluster centroids.
        """
        clusters = self.get_interface_clusters(interface_id)
        if not clusters:
            return None

        centroids = []
        weights = []

        for cluster_info in clusters:
            cluster_id = cluster_info['cluster_id']
            weight = cluster_info['weight']

            # Get cluster centroid
            cluster_centroid = self.get_cluster_centroid(cluster_id, model_id)
            if cluster_centroid is not None:
                centroids.append(cluster_centroid)
                weights.append(weight)

        if not centroids:
            return None

        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        centroid = np.zeros_like(centroids[0])
        for c, w in zip(centroids, weights):
            centroid += w * c

        # Normalize the resulting centroid
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Store and return
        self.set_interface_centroid(interface_id, model_id, centroid)
        return centroid

    # =========================================================================
    # ANSWER PREREQUISITES CENTROIDS
    # =========================================================================

    def set_prerequisites_centroid(
        self,
        answer_id: int,
        model_id: int,
        centroid: np.ndarray,
        source_type: str = 'metadata',
        source_answers: List[int] = None
    ) -> None:
        """
        Store a prerequisites centroid for an answer (chapter).

        Args:
            answer_id: The answer (chapter) to store centroid for
            model_id: Embedding model ID
            centroid: The centroid embedding vector
            source_type: How centroid was computed ('metadata', 'semantic', 'hybrid')
            source_answers: List of answer IDs used to compute this centroid
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO answer_prerequisites_centroids
            (answer_id, model_id, centroid, source_type, source_answers, computed_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            answer_id,
            model_id,
            centroid.astype(np.float32).tobytes(),
            source_type,
            json.dumps(source_answers) if source_answers else None
        ))
        self.conn.commit()

    def get_prerequisites_centroid(
        self,
        answer_id: int,
        model_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get the prerequisites centroid for an answer (chapter).

        Returns:
            Dict with 'centroid', 'source_type', 'source_answers', 'computed_at'
            or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT centroid, source_type, source_answers, computed_at
            FROM answer_prerequisites_centroids
            WHERE answer_id = ? AND model_id = ?
        """, (answer_id, model_id))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            'centroid': np.frombuffer(row['centroid'], dtype=np.float32),
            'source_type': row['source_type'],
            'source_answers': json.loads(row['source_answers']) if row['source_answers'] else [],
            'computed_at': row['computed_at']
        }

    def compute_prerequisites_centroid_from_metadata(
        self,
        answer_id: int,
        model_id: int
    ) -> Optional[np.ndarray]:
        """
        Compute prerequisites centroid from metadata relations.

        Uses 'preliminary' and 'foundational' relations to find prerequisites,
        then averages their embeddings.

        Args:
            answer_id: The answer (chapter) to compute prerequisites for
            model_id: Embedding model ID

        Returns:
            Computed centroid or None if no prerequisites found
        """
        # Get prerequisites via relations
        preliminary = self.get_prerequisites(answer_id)  # preliminary relations
        foundational = self.get_foundational(answer_id)  # foundational relations

        prereq_answer_ids = set()
        for rel in preliminary + foundational:
            prereq_answer_ids.add(rel['answer_id'])

        if not prereq_answer_ids:
            return None

        # Collect embeddings for prerequisites
        embeddings = []
        source_answers = []

        for prereq_id in prereq_answer_ids:
            emb = self._get_answer_embedding(prereq_id, model_id)
            if emb is not None:
                embeddings.append(emb)
                source_answers.append(prereq_id)

        if not embeddings:
            return None

        # Average embeddings
        centroid = np.mean(embeddings, axis=0)

        # Normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Store
        self.set_prerequisites_centroid(
            answer_id, model_id, centroid,
            source_type='metadata',
            source_answers=source_answers
        )

        return centroid

    def compute_prerequisites_centroid_from_interface(
        self,
        answer_id: int,
        model_id: int,
        prerequisites_interface_id: int
    ) -> Optional[np.ndarray]:
        """
        Compute prerequisites centroid by searching the prerequisites interface.

        Uses the answer's content to search the prerequisites interface,
        then averages top matches.

        Args:
            answer_id: The answer (chapter) to compute prerequisites for
            model_id: Embedding model ID
            prerequisites_interface_id: ID of the prerequisites interface

        Returns:
            Computed centroid or None
        """
        # Get answer text
        cursor = self.conn.cursor()
        cursor.execute("SELECT text FROM answers WHERE answer_id = ?", (answer_id,))
        row = cursor.fetchone()
        if not row:
            return None

        answer_text = row['text']

        # Get prerequisites interface centroid
        interface_centroid = self.get_interface_centroid(
            prerequisites_interface_id, model_id
        )
        if interface_centroid is None:
            interface_centroid = self.compute_interface_centroid(
                prerequisites_interface_id, model_id
            )

        if interface_centroid is None:
            return None

        # Search for semantically similar prerequisites
        # Get answer embedding
        answer_emb = self._get_answer_embedding(answer_id, model_id)
        if answer_emb is None:
            return None

        # Find prerequisites closest to both the answer and the interface
        # This is a simplified approach - search interface clusters
        interface_clusters = self.get_interface_clusters(prerequisites_interface_id)

        prereq_embeddings = []
        source_answers = []

        for cluster_info in interface_clusters:
            cluster_id = cluster_info['cluster_id']
            cluster_centroid = self.get_cluster_centroid(cluster_id, model_id)
            if cluster_centroid is not None:
                # Check similarity to answer
                similarity = np.dot(answer_emb, cluster_centroid)
                if similarity > 0.3:  # Threshold for relevance
                    prereq_embeddings.append(cluster_centroid)
                    # Get answer IDs from cluster
                    cursor.execute("""
                        SELECT answer_id FROM cluster_answers WHERE cluster_id = ?
                    """, (cluster_id,))
                    for r in cursor.fetchall():
                        source_answers.append(r['answer_id'])

        if not prereq_embeddings:
            return None

        # Average embeddings
        centroid = np.mean(prereq_embeddings, axis=0)

        # Normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Store
        self.set_prerequisites_centroid(
            answer_id, model_id, centroid,
            source_type='semantic',
            source_answers=list(set(source_answers))
        )

        return centroid

    def update_prerequisites_centroid(
        self,
        answer_id: int,
        model_id: int,
        method: str = 'metadata',
        prerequisites_interface_id: int = None
    ) -> Optional[np.ndarray]:
        """
        Update (recompute) the prerequisites centroid for an answer.

        Args:
            answer_id: The answer (chapter) to update
            model_id: Embedding model ID
            method: 'metadata' (from relations), 'semantic' (from interface), 'hybrid'
            prerequisites_interface_id: Required if method is 'semantic' or 'hybrid'

        Returns:
            Updated centroid or None
        """
        if method == 'metadata':
            return self.compute_prerequisites_centroid_from_metadata(answer_id, model_id)
        elif method == 'semantic':
            if prerequisites_interface_id is None:
                raise ValueError("prerequisites_interface_id required for semantic method")
            return self.compute_prerequisites_centroid_from_interface(
                answer_id, model_id, prerequisites_interface_id
            )
        elif method == 'hybrid':
            if prerequisites_interface_id is None:
                raise ValueError("prerequisites_interface_id required for hybrid method")
            # Compute both
            meta_centroid = self.compute_prerequisites_centroid_from_metadata(answer_id, model_id)
            sem_centroid = self.compute_prerequisites_centroid_from_interface(
                answer_id, model_id, prerequisites_interface_id
            )
            if meta_centroid is None and sem_centroid is None:
                return None
            elif meta_centroid is None:
                return sem_centroid
            elif sem_centroid is None:
                return meta_centroid
            else:
                # Average both
                centroid = (meta_centroid + sem_centroid) / 2
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                # Store hybrid result
                self.set_prerequisites_centroid(
                    answer_id, model_id, centroid,
                    source_type='hybrid',
                    source_answers=None
                )
                return centroid
        else:
            raise ValueError(f"Unknown method: {method}")

    def search_by_prerequisites_centroid(
        self,
        answer_id: int,
        model_id: int,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for content similar to an answer's prerequisites centroid.

        This finds content that is "prerequisite-like" for the given answer.

        Args:
            answer_id: The answer (chapter) whose prerequisites to search by
            model_id: Embedding model ID
            top_k: Number of results

        Returns:
            List of similar answers
        """
        prereq_info = self.get_prerequisites_centroid(answer_id, model_id)
        if prereq_info is None:
            # Try computing
            centroid = self.compute_prerequisites_centroid_from_metadata(answer_id, model_id)
            if centroid is None:
                return []
        else:
            centroid = prereq_info['centroid']

        # Search using centroid
        return self._search_by_embedding(centroid, model_id, top_k)

    def _get_answer_embedding(self, answer_id: int, model_id: int) -> Optional[np.ndarray]:
        """Get embedding for an answer."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT vector_path FROM embeddings
            WHERE entity_type = 'answer' AND entity_id = ? AND model_id = ?
        """, (answer_id, model_id))

        row = cursor.fetchone()
        if not row:
            return None

        vector_path = row['vector_path']
        if self.embeddings_dir:
            vector_path = os.path.join(self.embeddings_dir, vector_path)

        try:
            return np.load(vector_path)
        except Exception:
            return None

    def _search_by_embedding(
        self,
        query_embedding: np.ndarray,
        model_id: int,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search for answers similar to a query embedding."""
        cursor = self.conn.cursor()

        # Get all answer embeddings
        cursor.execute("""
            SELECT e.entity_id as answer_id, e.vector_path, a.text, a.source_file, a.record_id
            FROM embeddings e
            JOIN answers a ON e.entity_id = a.answer_id
            WHERE e.entity_type = 'answer' AND e.model_id = ?
        """, (model_id,))

        results = []
        for row in cursor.fetchall():
            vector_path = row['vector_path']
            if self.embeddings_dir:
                vector_path = os.path.join(self.embeddings_dir, vector_path)

            try:
                emb = np.load(vector_path)
                similarity = float(np.dot(query_embedding, emb))
                results.append({
                    'answer_id': row['answer_id'],
                    'text': row['text'][:500] if row['text'] else '',
                    'source_file': row['source_file'],
                    'record_id': row['record_id'],
                    'similarity': similarity
                })
            except Exception:
                continue

        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    # =========================================================================
    # INTERFACE UPDATE METHODS
    # =========================================================================

    def update_interface(
        self,
        interface_id: int,
        name: str = None,
        description: str = None,
        topics: List[str] = None,
        is_active: bool = None
    ) -> bool:
        """
        Update an existing interface's properties.

        Args:
            interface_id: Interface to update
            name: New name (optional)
            description: New description (optional)
            topics: New topics list (optional)
            is_active: New active status (optional)

        Returns:
            True if updated, False if interface not found
        """
        cursor = self.conn.cursor()

        # Check interface exists
        cursor.execute("SELECT interface_id FROM semantic_interfaces WHERE interface_id = ?",
                      (interface_id,))
        if not cursor.fetchone():
            return False

        # Build update query
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if topics is not None:
            updates.append("topics = ?")
            params.append(json.dumps(topics))
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(is_active)

        if not updates:
            return True  # Nothing to update

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(interface_id)

        cursor.execute(f"""
            UPDATE semantic_interfaces
            SET {', '.join(updates)}
            WHERE interface_id = ?
        """, params)

        self.conn.commit()
        return True

    def refresh_interface_centroid(
        self,
        interface_id: int,
        model_id: int
    ) -> Optional[np.ndarray]:
        """
        Refresh (recompute) an interface's centroid from its clusters.

        Args:
            interface_id: Interface to refresh
            model_id: Embedding model ID

        Returns:
            New centroid or None
        """
        return self.compute_interface_centroid(interface_id, model_id)

    def update_all_prerequisites_centroids(
        self,
        model_id: int,
        method: str = 'metadata',
        prerequisites_interface_id: int = None
    ) -> Dict[str, int]:
        """
        Update prerequisites centroids for all answers that have prerequisites.

        Args:
            model_id: Embedding model ID
            method: 'metadata', 'semantic', or 'hybrid'
            prerequisites_interface_id: Required for semantic/hybrid

        Returns:
            Stats dict with 'updated', 'skipped', 'errors'
        """
        stats = {'updated': 0, 'skipped': 0, 'errors': 0}

        cursor = self.conn.cursor()

        # Find all answers that have prerequisites relations
        cursor.execute("""
            SELECT DISTINCT to_answer_id as answer_id
            FROM answer_relations
            WHERE relation_type IN ('preliminary', 'foundational')
        """)

        for row in cursor.fetchall():
            try:
                result = self.update_prerequisites_centroid(
                    row['answer_id'], model_id, method, prerequisites_interface_id
                )
                if result is not None:
                    stats['updated'] += 1
                else:
                    stats['skipped'] += 1
            except Exception as e:
                stats['errors'] += 1

        return stats

    # =========================================================================
    # QUERY-TO-INTERFACE MAPPING
    # =========================================================================

    def map_query_to_interface(
        self,
        query_text: str,
        model_name: str,
        temperature: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Map a query to the most relevant interface(s).

        Uses softmax over interface centroids to determine routing.

        Args:
            query_text: The query to map
            model_name: Embedding model name
            temperature: Softmax temperature (lower = sharper routing)

        Returns:
            List of interfaces with routing weights, sorted by weight
        """
        query_emb = self._embed_query(query_text, model_name)

        model_info = self.get_model(model_name)
        if not model_info:
            return []
        model_id = model_info['model_id']

        # Get all active interfaces with centroids
        interfaces = self.list_interfaces(active_only=True)

        interface_similarities = []
        for iface in interfaces:
            centroid = self.get_interface_centroid(iface['interface_id'], model_id)
            if centroid is not None:
                # Cosine similarity
                similarity = float(np.dot(query_emb, centroid))
                interface_similarities.append({
                    'interface_id': iface['interface_id'],
                    'name': iface['name'],
                    'description': iface['description'],
                    'similarity': similarity
                })

        if not interface_similarities:
            return []

        # Softmax routing
        similarities = np.array([i['similarity'] for i in interface_similarities])
        exp_sims = np.exp(similarities / temperature)
        weights = exp_sims / exp_sims.sum()

        # Add routing weights
        for i, w in enumerate(weights):
            interface_similarities[i]['routing_weight'] = float(w)

        # Sort by routing weight
        interface_similarities.sort(key=lambda x: x['routing_weight'], reverse=True)

        return interface_similarities

    def search_via_interface(
        self,
        query_text: str,
        interface_id: int,
        model_name: str,
        mh_projection_id: int = None,
        top_k: int = 5,
        use_interface_first_routing: bool = False,
        max_distance: float = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific interface's context.

        Supports multiple modes:

        1. **Default mode**: Regular search, results naturally match interface
           topics due to semantic similarity.

        2. **Interface-First Routing** (optimization): Pre-filter answers by
           similarity to interface centroid, then search only that subset.
           Enable with `use_interface_first_routing=True`.

        3. **Max distance filtering** (convenience): Post-filter results to
           only include answers within `max_distance` of interface centroid.

        Args:
            query_text: The search query
            interface_id: Interface to search within
            model_name: Embedding model name
            mh_projection_id: Multi-head projection ID
            top_k: Number of results
            use_interface_first_routing: If True, pre-filter answers by
                similarity to interface centroid before searching (optimization)
            max_distance: If set, post-filter results to only include answers
                within this cosine distance (1 - similarity) from interface centroid
            similarity_threshold: For interface-first routing, minimum similarity
                to interface centroid to include an answer (default: 0.5)

        Returns:
            Search results, optionally filtered by interface proximity
        """
        model_info = self.get_model(model_name)
        if not model_info:
            return []
        model_id = model_info['model_id']

        # Get interface centroid for filtering modes
        interface_centroid = None
        if use_interface_first_routing or max_distance is not None:
            interface_centroid = self.get_interface_centroid(interface_id, model_id)
            if interface_centroid is None:
                # Fall back to computing it
                interface_centroid = self.compute_interface_centroid(interface_id, model_id)

        # Interface-First Routing: pre-filter answers by similarity to interface centroid
        if use_interface_first_routing and interface_centroid is not None:
            results = self._interface_first_search(
                query_text=query_text,
                model_name=model_name,
                model_id=model_id,
                interface_centroid=interface_centroid,
                mh_projection_id=mh_projection_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold or 0.5
            )
        else:
            # Default: regular search
            results = self.search_with_context(
                query_text=query_text,
                model_name=model_name,
                mh_projection_id=mh_projection_id,
                top_k=top_k if max_distance is None else top_k * 3,
                include_foundational=False,
                include_prerequisites=False,
                include_extensions=False,
                include_next_steps=False
            )

        # Max distance post-filtering (convenience option)
        if max_distance is not None and interface_centroid is not None:
            results = self._filter_by_max_distance(
                results, model_id, interface_centroid, max_distance, top_k
            )

        return results

    def _interface_first_search(
        self,
        query_text: str,
        model_name: str,
        model_id: int,
        interface_centroid: np.ndarray,
        mh_projection_id: int,
        top_k: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Interface-First Routing optimization.

        1. Filter answers by similarity to interface centroid
        2. Run multi_head_search on filtered subset

        This reduces computation for large Q-A databases by narrowing
        the search space before applying the more expensive multi-head routing.
        """
        # Get all answer embeddings
        answer_ids, answer_matrix = self.get_all_answer_embeddings(model_id)

        if len(answer_ids) == 0:
            return []

        # Normalize answer embeddings
        answer_norms = np.linalg.norm(answer_matrix, axis=1, keepdims=True)
        answer_norms = np.where(answer_norms > 0, answer_norms, 1)
        answer_matrix_normed = answer_matrix / answer_norms

        # Compute similarity to interface centroid
        interface_similarities = answer_matrix_normed @ interface_centroid

        # Filter to answers above threshold
        mask = interface_similarities >= similarity_threshold
        filtered_indices = np.where(mask)[0]

        if len(filtered_indices) == 0:
            # No answers meet threshold, fall back to top answers by interface similarity
            top_by_interface = np.argsort(-interface_similarities)[:top_k * 2]
            filtered_indices = top_by_interface

        filtered_answer_ids = [answer_ids[i] for i in filtered_indices]
        filtered_embeddings = answer_matrix[filtered_indices]

        # Now search within filtered subset
        query_emb = self._embed_query(query_text, model_name)

        # Normalize
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 0:
            query_emb = query_emb / query_norm

        filtered_norms = np.linalg.norm(filtered_embeddings, axis=1, keepdims=True)
        filtered_norms = np.where(filtered_norms > 0, filtered_norms, 1)
        filtered_normed = filtered_embeddings / filtered_norms

        # Direct search on filtered subset
        scores = filtered_normed @ query_emb

        # Get top-k from filtered set
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            answer_id = filtered_answer_ids[idx]
            answer = self.get_answer(answer_id)
            results.append({
                'answer_id': answer_id,
                'score': float(scores[idx]),
                'interface_similarity': float(interface_similarities[filtered_indices[idx]]),
                'text': answer['text'][:500] if answer else '',
                'record_id': answer.get('record_id', '') if answer else '',
                'source_file': answer.get('source_file', '') if answer else ''
            })

        return results

    def _filter_by_max_distance(
        self,
        results: List[Dict[str, Any]],
        model_id: int,
        interface_centroid: np.ndarray,
        max_distance: float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Post-filter results by maximum distance from interface centroid.

        Distance is computed as (1 - cosine_similarity).
        """
        filtered = []

        for result in results:
            answer_id = result['answer_id']

            # Get answer embedding using parent class method
            answer_emb = self.get_embedding(model_id, 'answer', answer_id)

            if answer_emb is not None:
                # Normalize
                norm = np.linalg.norm(answer_emb)
                if norm > 0:
                    answer_emb = answer_emb / norm

                # Compute distance (1 - similarity)
                similarity = float(np.dot(answer_emb, interface_centroid))
                distance = 1.0 - similarity

                if distance <= max_distance:
                    result['interface_distance'] = distance
                    result['interface_similarity'] = similarity
                    filtered.append(result)

                    if len(filtered) >= top_k:
                        break

        return filtered

    def _legacy_search_via_interface(
        self,
        query_text: str,
        interface_id: int,
        model_name: str,
        mh_projection_id: int = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Legacy: Search filtered to interface's explicit cluster membership."""
        # Get interface clusters
        interface_clusters = self.get_interface_clusters(interface_id)
        if not interface_clusters:
            return []

        cluster_ids = [c['cluster_id'] for c in interface_clusters]

        # Perform regular search
        all_results = self.search_with_context(
            query_text=query_text,
            model_name=model_name,
            mh_projection_id=mh_projection_id,
            top_k=top_k * 3,  # Get more, then filter
            include_foundational=False,
            include_prerequisites=False,
            include_extensions=False,
            include_next_steps=False
        )

        # Filter to interface clusters
        filtered_results = []
        for result in all_results:
            answer_id = result['answer_id']

            # Check if answer belongs to any interface cluster
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT DISTINCT ca.cluster_id
                FROM cluster_answers ca
                WHERE ca.answer_id = ? AND ca.cluster_id IN ({})
            """.format(','.join('?' * len(cluster_ids))),
            [answer_id] + cluster_ids)

            if cursor.fetchone():
                filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break

        return filtered_results

    # =========================================================================
    # INTERFACE MANAGEMENT - AUTO-GENERATION
    # =========================================================================

    def auto_generate_interfaces(
        self,
        model_name: str,
        min_clusters_per_interface: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[int]:
        """
        Auto-generate interfaces from cluster analysis.

        Groups semantically similar clusters into interfaces.

        Args:
            model_name: Embedding model for computing similarities
            min_clusters_per_interface: Minimum clusters per interface
            similarity_threshold: Minimum similarity to group clusters

        Returns:
            List of created interface IDs
        """
        model_info = self.get_model(model_name)
        if not model_info:
            return []
        model_id = model_info['model_id']

        # Get all clusters with centroids
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.cluster_id, c.name
            FROM qa_clusters c
        """)

        clusters = []
        for row in cursor.fetchall():
            centroid = self.get_cluster_centroid(row['cluster_id'], model_id)
            if centroid is not None:
                clusters.append({
                    'cluster_id': row['cluster_id'],
                    'name': row['name'],
                    'centroid': centroid
                })

        if len(clusters) < min_clusters_per_interface:
            return []

        # Simple greedy clustering: assign each cluster to most similar interface
        # or create a new interface if no similar one exists
        interfaces_created = []
        interface_clusters_map = {}  # interface_id -> list of cluster_ids

        for cluster in clusters:
            best_interface = None
            best_similarity = similarity_threshold

            # Check existing interfaces
            for iface_id, assigned_clusters in interface_clusters_map.items():
                iface_centroid = self.get_interface_centroid(iface_id, model_id)
                if iface_centroid is not None:
                    sim = float(np.dot(cluster['centroid'], iface_centroid))
                    if sim > best_similarity:
                        best_similarity = sim
                        best_interface = iface_id

            if best_interface is not None:
                # Add to existing interface
                self.add_cluster_to_interface(best_interface, cluster['cluster_id'])
                interface_clusters_map[best_interface].append(cluster['cluster_id'])
                # Recompute centroid
                self.compute_interface_centroid(best_interface, model_id)
            else:
                # Create new interface
                iface_name = f"auto_interface_{len(interfaces_created) + 1}"
                iface_id = self.create_interface(
                    name=iface_name,
                    description=f"Auto-generated from cluster {cluster['name']}"
                )
                self.add_cluster_to_interface(iface_id, cluster['cluster_id'])
                self.set_interface_centroid(iface_id, model_id, cluster['centroid'])
                interface_clusters_map[iface_id] = [cluster['cluster_id']]
                interfaces_created.append(iface_id)

        # Remove interfaces with too few clusters
        final_interfaces = []
        for iface_id, assigned in interface_clusters_map.items():
            if len(assigned) >= min_clusters_per_interface:
                final_interfaces.append(iface_id)
            else:
                self.delete_interface(iface_id)

        return final_interfaces

    # =========================================================================
    # SCALE OPTIMIZATIONS CONFIGURATION
    # =========================================================================
    #
    # Configuration for automatic optimization selection based on data scale.
    #
    # Default thresholds:
    #   - interface_first_routing: 50,000 Q-A pairs
    #   - transformer_distillation: 100,000 Q-A pairs
    #
    # See: docs/proposals/TRANSFORMER_DISTILLATION.md
    #      projection_transformer.py

    def get_scale_config(self) -> Dict[str, Any]:
        """
        Get current scale optimization configuration.

        Configuration is stored in a settings table, with defaults if not set.
        """
        cursor = self.conn.cursor()

        # Create settings table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

        # Default configuration
        defaults = {
            'interface_first_routing_enabled': 'auto',
            'interface_first_routing_threshold': '50000',
            'transformer_distillation_enabled': 'auto',
            'transformer_distillation_threshold': '100000',
        }

        config = {}
        for key, default in defaults.items():
            cursor.execute("SELECT value FROM kg_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            value = row['value'] if row else default

            # Convert types
            if key.endswith('_threshold'):
                config[key] = int(value)
            elif key.endswith('_enabled'):
                config[key] = value  # 'auto', 'true', 'false'
            else:
                config[key] = value

        return config

    def set_scale_config(self, **kwargs):
        """
        Set scale optimization configuration.

        Args:
            interface_first_routing_enabled: 'auto', 'true', or 'false'
            interface_first_routing_threshold: Q-A count threshold
            transformer_distillation_enabled: 'auto', 'true', or 'false'
            transformer_distillation_threshold: Q-A count threshold
        """
        cursor = self.conn.cursor()

        # Create settings table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        valid_keys = {
            'interface_first_routing_enabled',
            'interface_first_routing_threshold',
            'transformer_distillation_enabled',
            'transformer_distillation_threshold',
        }

        for key, value in kwargs.items():
            if key in valid_keys:
                cursor.execute("""
                    INSERT OR REPLACE INTO kg_settings (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (key, str(value)))

        self.conn.commit()

    def should_use_interface_first_routing(self) -> Dict[str, Any]:
        """
        Check if interface-first routing should be used based on configuration.

        Returns decision and reasoning.
        """
        config = self.get_scale_config()
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM answers")
        qa_count = cursor.fetchone()['count']

        enabled_setting = config['interface_first_routing_enabled']
        threshold = config['interface_first_routing_threshold']

        if enabled_setting == 'true':
            return {
                'use': True,
                'reason': 'Explicitly enabled in configuration',
                'qa_count': qa_count,
                'threshold': threshold
            }
        elif enabled_setting == 'false':
            return {
                'use': False,
                'reason': 'Explicitly disabled in configuration',
                'qa_count': qa_count,
                'threshold': threshold
            }
        else:  # auto
            use = qa_count >= threshold
            return {
                'use': use,
                'reason': f"Auto: {qa_count:,} Q-A pairs {'≥' if use else '<'} {threshold:,} threshold",
                'qa_count': qa_count,
                'threshold': threshold
            }

    def should_use_transformer_distillation(self) -> Dict[str, Any]:
        """
        Check if transformer distillation should be used based on configuration.

        Returns decision and reasoning.
        """
        config = self.get_scale_config()
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM answers")
        qa_count = cursor.fetchone()['count']

        enabled_setting = config['transformer_distillation_enabled']
        threshold = config['transformer_distillation_threshold']

        if enabled_setting == 'true':
            return {
                'use': True,
                'reason': 'Explicitly enabled in configuration',
                'qa_count': qa_count,
                'threshold': threshold,
                'implementation': 'projection_transformer.py'
            }
        elif enabled_setting == 'false':
            return {
                'use': False,
                'reason': 'Explicitly disabled in configuration',
                'qa_count': qa_count,
                'threshold': threshold,
                'implementation': 'projection_transformer.py'
            }
        else:  # auto
            use = qa_count >= threshold
            return {
                'use': use,
                'reason': f"Auto: {qa_count:,} Q-A pairs {'≥' if use else '<'} {threshold:,} threshold",
                'qa_count': qa_count,
                'threshold': threshold,
                'implementation': 'projection_transformer.py'
            }

    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of all scale optimizations.

        Use this during training/deployment to determine which optimizations
        to enable based on current data scale and configuration.
        """
        return {
            'config': self.get_scale_config(),
            'interface_first_routing': self.should_use_interface_first_routing(),
            'transformer_distillation': self.should_use_transformer_distillation(),
        }

    # =========================================================================
    # TRANSFORMER DISTILLATION (Scale Optimization)
    # =========================================================================
    #
    # Full implementation in: projection_transformer.py
    #
    # Usage:
    #     from projection_transformer import (
    #         ProjectionTransformer,
    #         train_distillation,
    #         evaluate_equivalence,
    #         optimal_architecture
    #     )
    #
    # This section provides database integration for tracking distillation status.

    def check_distillation_recommended(self, qa_threshold: int = 100000) -> Dict[str, Any]:
        """
        Check if transformer distillation is recommended based on Q-A count.

        Transformer distillation compresses the embedding + softmax routing
        into a smaller, faster model. See projection_transformer.py for the
        actual implementation.

        Args:
            qa_threshold: Q-A count threshold before distillation is recommended

        Returns:
            Dict with recommendation status and metrics
        """
        cursor = self.conn.cursor()

        # Get Q-A count
        cursor.execute("SELECT COUNT(*) as count FROM answers")
        qa_count = cursor.fetchone()['count']

        recommended = qa_count >= qa_threshold
        percentage = (qa_count / qa_threshold * 100) if qa_threshold > 0 else 0

        return {
            'recommended': recommended,
            'qa_count': qa_count,
            'threshold': qa_threshold,
            'percentage': round(percentage, 1),
            'message': f"Distillation {'recommended' if recommended else 'not needed'}: "
                      f"{qa_count:,} Q-A pairs ({percentage:.1f}% of {qa_threshold:,} threshold)",
            'implementation': 'projection_transformer.py'
        }

    def get_distillation_training_embeddings(
        self,
        model_name: str,
        sample_size: int = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Get query embeddings for transformer distillation training.

        Use with projection_transformer.train_distillation():

            embeddings, question_ids = db.get_distillation_training_embeddings(model_name)
            train_distillation(transformer, lda_projection, embeddings)

        Args:
            model_name: Embedding model name
            sample_size: Optional limit on training examples

        Returns:
            Tuple of (embeddings array, question_ids list)
        """
        model_info = self.get_model(model_name)
        if not model_info:
            return np.array([]), []

        model_id = model_info['model_id']
        cursor = self.conn.cursor()

        # Get questions that have embeddings stored
        query = """
            SELECT e.entity_id as question_id
            FROM embeddings e
            JOIN questions q ON e.entity_id = q.question_id
            WHERE e.model_id = ? AND e.entity_type = 'question'
        """

        if sample_size:
            query += f" LIMIT {sample_size}"

        cursor.execute(query, (model_id,))

        embeddings = []
        question_ids = []
        for row in cursor.fetchall():
            q_id = row['question_id']
            emb = self.get_embedding(model_id, 'question', q_id)
            if emb is not None:
                embeddings.append(emb)
                question_ids.append(q_id)

        if embeddings:
            return np.stack(embeddings), question_ids
        return np.array([]), []

    # =========================================================================
    # INTERFACE METRICS
    # =========================================================================

    def set_interface_metric(
        self,
        interface_id: int,
        metric_name: str,
        metric_value: float
    ):
        """Set a metric value for an interface."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO interface_metrics
            (interface_id, metric_name, metric_value)
            VALUES (?, ?, ?)
        """, (interface_id, metric_name, metric_value))
        self.conn.commit()

    def get_interface_metrics(self, interface_id: int) -> Dict[str, float]:
        """Get all metrics for an interface."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT metric_name, metric_value
            FROM interface_metrics
            WHERE interface_id = ?
        """, (interface_id,))

        return {row['metric_name']: row['metric_value']
                for row in cursor.fetchall()}

    def compute_interface_coverage(self, interface_id: int) -> Dict[str, float]:
        """
        Compute coverage metrics for an interface.

        Metrics:
        - cluster_count: Number of clusters
        - answer_count: Total answers accessible
        - question_count: Total questions accessible
        - avg_cluster_size: Average answers per cluster
        """
        cursor = self.conn.cursor()

        # Cluster count
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM interface_clusters
            WHERE interface_id = ?
        """, (interface_id,))
        cluster_count = cursor.fetchone()['count']

        # Answer and question counts
        cursor.execute("""
            SELECT
                COUNT(DISTINCT ca.answer_id) as answer_count,
                COUNT(DISTINCT cq.question_id) as question_count
            FROM interface_clusters ic
            LEFT JOIN cluster_answers ca ON ic.cluster_id = ca.cluster_id
            LEFT JOIN cluster_questions cq ON ic.cluster_id = cq.cluster_id
            WHERE ic.interface_id = ?
        """, (interface_id,))

        row = cursor.fetchone()
        answer_count = row['answer_count'] or 0
        question_count = row['question_count'] or 0

        # Compute metrics
        avg_cluster_size = answer_count / cluster_count if cluster_count > 0 else 0

        metrics = {
            'cluster_count': float(cluster_count),
            'answer_count': float(answer_count),
            'question_count': float(question_count),
            'avg_cluster_size': avg_cluster_size
        }

        # Store metrics
        for name, value in metrics.items():
            self.set_interface_metric(interface_id, name, value)

        return metrics

    def get_interface_health(self, interface_id: int) -> Dict[str, Any]:
        """
        Get comprehensive health status of an interface.

        Returns coverage metrics plus health indicators.
        """
        interface = self.get_interface(interface_id)
        if not interface:
            return {'error': 'Interface not found'}

        metrics = self.compute_interface_coverage(interface_id)

        # Health indicators
        health = {
            'interface_id': interface_id,
            'name': interface['name'],
            'is_active': interface['is_active'],
            'metrics': metrics,
            'health_status': 'healthy'
        }

        # Check for issues (unhealthy > warning > healthy)
        issues = []
        status = 'healthy'

        if metrics['cluster_count'] == 0:
            issues.append('No clusters assigned')
            status = 'unhealthy'
        elif metrics['cluster_count'] < 3:
            issues.append('Few clusters (consider adding more)')
            if status != 'unhealthy':
                status = 'warning'

        if metrics['answer_count'] == 0:
            issues.append('No answers accessible')
            status = 'unhealthy'

        if not interface.get('centroid_model_id'):
            issues.append('No centroid computed')
            if status != 'unhealthy':
                status = 'warning'

        health['health_status'] = status
        health['issues'] = issues

        return health


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_kg_database(db_path: str, embeddings_dir: str = None) -> KGTopologyAPI:
    """Create a new KG topology database."""
    return KGTopologyAPI(db_path, embeddings_dir)


# =============================================================================
# PHASE 3: DISTRIBUTED KG TOPOLOGY API
# =============================================================================

class DistributedKGTopologyAPI(KGTopologyAPI):
    """
    Distributed KG Topology API with Kleinberg routing support.

    Extends KGTopologyAPI with:
    - Node discovery and registration
    - Inter-node query routing via Kleinberg small-world routing
    - Path folding for shortcut creation
    - HTL-limited query propagation

    See: docs/proposals/ROADMAP_KG_TOPOLOGY.md (Phase 3)
    """

    SCHEMA_VERSION = 4  # v3 = Phase 2, v4 = Phase 3 distributed

    def __init__(
        self,
        db_path: str,
        embeddings_dir: str = None,
        node_id: str = None,
        discovery_backend: str = 'local',
        discovery_config: Dict[str, Any] = None
    ):
        """
        Initialize distributed KG topology API.

        Args:
            db_path: Path to SQLite database
            embeddings_dir: Directory for embedding files
            node_id: Unique identifier for this node (auto-generated if None)
            discovery_backend: Discovery backend ('local', 'consul', 'etcd')
            discovery_config: Backend-specific configuration
        """
        super().__init__(db_path, embeddings_dir)

        # Generate node ID from db_path if not provided
        self.node_id = node_id or f"node_{hashlib.sha256(db_path.encode()).hexdigest()[:8]}"
        self.discovery_backend = discovery_backend
        self.discovery_config = discovery_config or {}

        self._router = None
        self._discovery_client = None
        self._init_distributed_schema()

    def _init_distributed_schema(self):
        """Create distributed-specific tables."""
        cursor = self.conn.cursor()

        # Query shortcuts from path folding
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_shortcuts (
                query_hash TEXT PRIMARY KEY,
                target_node_id TEXT NOT NULL,
                target_interface_id INTEGER,
                hit_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Remote node cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS remote_nodes (
                node_id TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                centroid BLOB,
                topics TEXT,
                embedding_model TEXT,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                health_status TEXT DEFAULT 'unknown'
            )
        """)

        # Distributed query log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS distributed_query_log (
                query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT,
                query_hash TEXT,
                origin_node TEXT,
                hops INTEGER,
                result_count INTEGER,
                response_time_ms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for shortcut lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_shortcut_hash
            ON query_shortcuts(query_hash)
        """)

        self.conn.commit()

    def _get_discovery_client(self):
        """Get or create discovery client."""
        if self._discovery_client is None:
            try:
                from .discovery_clients import create_discovery_client
            except ImportError:
                from discovery_clients import create_discovery_client
            self._discovery_client = create_discovery_client(
                self.discovery_backend,
                **self.discovery_config
            )
        return self._discovery_client

    def get_router(self):
        """
        Get or create Kleinberg router instance.

        Returns:
            KleinbergRouter instance configured for this node
        """
        if self._router is None:
            try:
                from .kleinberg_router import KleinbergRouter
            except ImportError:
                from kleinberg_router import KleinbergRouter
            self._router = KleinbergRouter(
                local_node_id=self.node_id,
                discovery_client=self._get_discovery_client()
            )
        return self._router

    def register_node(
        self,
        interface_id: int = None,
        host: str = 'localhost',
        port: int = 8080,
        tags: List[str] = None,
        corpus_id: str = None,
        data_sources: List[str] = None
    ) -> bool:
        """
        Register this node with service discovery.

        Advertises interface centroid for routing and corpus info for
        diversity-weighted aggregation.

        Args:
            interface_id: Interface to advertise (uses first if None)
            host: Host address to advertise
            port: Port to advertise
            tags: Additional tags
            corpus_id: Unique identifier for this node's data corpus
            data_sources: List of upstream data sources

        Returns:
            True if registration succeeded
        """
        # Get interface(s) to advertise
        if interface_id:
            interfaces = [self.get_interface(interface_id)]
        else:
            interfaces = self.list_interfaces(active_only=True)

        if not interfaces or not interfaces[0]:
            return False

        interface = interfaces[0]

        # Get centroid for the interface
        centroid = None
        centroid_model_id = interface.get('centroid_model_id')
        if centroid_model_id:
            centroid = self.get_interface_centroid(
                interface['interface_id'],
                centroid_model_id
            )

        # Auto-generate corpus_id if not provided
        if corpus_id is None:
            corpus_id = self._generate_corpus_id()

        # Prepare metadata with Phase 4 corpus tracking
        metadata = {
            'interface_id': interface['interface_id'],
            'interface_name': interface['name'],
            'interface_topics': json.dumps(interface.get('topics', [])),
            'embedding_model': 'all-MiniLM-L6-v2',  # TODO: get from config
            # Phase 4: Corpus tracking for diversity-weighted aggregation
            'corpus_id': corpus_id,
            'data_sources': json.dumps(data_sources or []),
            'last_updated': datetime.now().isoformat()
        }

        if centroid is not None:
            metadata['semantic_centroid'] = base64.b64encode(
                centroid.astype(np.float32).tobytes()
            ).decode()

        # Store corpus info locally for provenance tracking
        self._corpus_id = corpus_id
        self._data_sources = data_sources or []

        # Register with discovery
        discovery = self._get_discovery_client()
        return discovery.register(
            service_name='kg_topology',
            service_id=self.node_id,
            host=host,
            port=port,
            tags=tags or ['kg_node'],
            metadata=metadata
        )

    def _generate_corpus_id(self) -> str:
        """Generate a corpus ID based on database content."""
        cursor = self.conn.cursor()

        # Hash based on source files and answer count
        cursor.execute("SELECT COUNT(*) FROM answers")
        answer_count = cursor.fetchone()[0]

        cursor.execute("SELECT DISTINCT source_file FROM answers LIMIT 10")
        sources = [row[0] for row in cursor.fetchall() if row[0]]

        content = f"{self.node_id}:{answer_count}:{','.join(sorted(sources))}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_corpus_info(self) -> Dict[str, Any]:
        """Get corpus information for this node."""
        return {
            'corpus_id': getattr(self, '_corpus_id', None),
            'data_sources': getattr(self, '_data_sources', []),
            'node_id': self.node_id
        }

    def set_corpus_info(
        self,
        corpus_id: str,
        data_sources: List[str] = None
    ):
        """Set corpus information for diversity tracking."""
        self._corpus_id = corpus_id
        self._data_sources = data_sources or []

    def deregister_node(self) -> bool:
        """Deregister this node from service discovery."""
        discovery = self._get_discovery_client()
        return discovery.deregister(self.node_id)

    def distributed_search(
        self,
        query_text: str,
        model_name: str = 'all-MiniLM-L6-v2',
        top_k: int = 5,
        max_hops: int = 10,
        local_first: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search across distributed KG network.

        Args:
            query_text: The search query
            model_name: Embedding model name
            top_k: Number of results
            max_hops: Maximum routing hops (HTL)
            local_first: Try local search before routing

        Returns:
            Combined results from local and remote nodes
        """
        import time
        start_time = time.time()

        results = []

        # Embed query
        query_emb = self._embed_query(query_text, model_name)
        query_hash = hashlib.sha256(query_text.encode()).hexdigest()[:16]

        # Check for shortcut
        shortcut = self._get_shortcut(query_hash)
        if shortcut:
            remote_results = self._query_via_shortcut(
                shortcut,
                query_emb,
                query_text,
                top_k
            )
            if remote_results:
                self._update_shortcut_usage(query_hash)
                self._log_query(query_text, query_hash, 'local', 1,
                               len(remote_results), time.time() - start_time)
                return remote_results

        # Local search first (if enabled)
        if local_first:
            local_results = self.search_with_context(
                query_text=query_text,
                model_name=model_name,
                top_k=top_k
            )

            # If local results are good, return them
            if local_results and local_results[0].get('score', 0) > 0.7:
                for r in local_results:
                    r['source_node'] = self.node_id
                    r['hops'] = 0
                self._log_query(query_text, query_hash, self.node_id, 0,
                               len(local_results), time.time() - start_time)
                return local_results

            results.extend(local_results)

        # Route to network
        router = self.get_router()
        from .kleinberg_router import RoutingEnvelope

        envelope = RoutingEnvelope(
            origin_node=self.node_id,
            htl=max_hops,
            path_folding_enabled=True
        )

        remote_results = router.route_query(
            query_embedding=query_emb,
            query_text=query_text,
            envelope=envelope,
            top_k=top_k
        )
        results.extend(remote_results)

        # Deduplicate and sort
        seen: Set[Tuple[int, str]] = set()
        unique_results = []
        for r in sorted(results, key=lambda x: x.get('score', 0), reverse=True):
            key = (r.get('answer_id'), r.get('source_node', 'local'))
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        final_results = unique_results[:top_k]

        # Log query
        max_hop = max((r.get('hops', 0) for r in final_results), default=0)
        self._log_query(query_text, query_hash, self.node_id, max_hop,
                       len(final_results), time.time() - start_time)

        return final_results

    def handle_remote_query(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle incoming query from another node.

        This is the HTTP endpoint handler for /kg/query.

        Args:
            request: Incoming request with routing envelope

        Returns:
            Response with results
        """
        routing = request.get('__routing', {})
        embedding_info = request.get('__embedding', {})
        payload = request.get('payload', {})

        htl = routing.get('htl', 0)
        visited = set(routing.get('visited', []))
        query_text = payload.get('query_text', '')
        top_k = payload.get('top_k', 5)

        # Extract query embedding
        vector = embedding_info.get('vector', [])
        if vector:
            query_emb = np.array(vector, dtype=np.float32)
        else:
            # Need to re-embed
            model_name = embedding_info.get('model', 'all-MiniLM-L6-v2')
            query_emb = self._embed_query(query_text, model_name)

        # Local search
        results = self.search_with_context(
            query_text=query_text,
            model_name=embedding_info.get('model', 'all-MiniLM-L6-v2'),
            top_k=top_k
        )

        # Add source node info
        for r in results:
            r['source_node'] = self.node_id
            r['hops'] = 0

        # If results are weak and HTL allows, forward to network
        if htl > 0 and (not results or results[0].get('score', 0) < 0.5):
            router = self.get_router()
            from .kleinberg_router import RoutingEnvelope

            envelope = RoutingEnvelope(
                origin_node=routing.get('origin_node', 'unknown'),
                htl=htl,
                visited=visited,
                path_folding_enabled=routing.get('path_folding_enabled', True)
            )
            envelope.visited.add(self.node_id)

            remote_results = router.route_query(
                query_embedding=query_emb,
                query_text=query_text,
                envelope=envelope,
                top_k=top_k
            )
            results.extend(remote_results)

        return {
            '__type': 'kg_response',
            '__id': request.get('__id'),
            'results': results[:top_k],
            'source_node': self.node_id
        }

    def _get_shortcut(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get shortcut for query hash."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT target_node_id, target_interface_id
            FROM query_shortcuts
            WHERE query_hash = ?
        """, (query_hash,))
        row = cursor.fetchone()
        if row:
            return {
                'target_node_id': row[0],
                'target_interface_id': row[1]
            }
        return None

    def _update_shortcut_usage(self, query_hash: str):
        """Update shortcut usage statistics."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE query_shortcuts
            SET hit_count = hit_count + 1,
                last_used_at = CURRENT_TIMESTAMP
            WHERE query_hash = ?
        """, (query_hash,))
        self.conn.commit()

    def _query_via_shortcut(
        self,
        shortcut: Dict[str, Any],
        query_emb: np.ndarray,
        query_text: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Query a node directly via shortcut."""
        target_node = shortcut['target_node_id']

        # If it's us, do local search
        if target_node == self.node_id:
            results = self.search_with_context(
                query_text=query_text,
                model_name='all-MiniLM-L6-v2',
                top_k=top_k
            )
            for r in results:
                r['source_node'] = self.node_id
                r['hops'] = 0
            return results

        # Otherwise, look up node and forward
        router = self.get_router()
        nodes = router.discover_nodes()

        for node in nodes:
            if node.node_id == target_node:
                from .kleinberg_router import RoutingEnvelope
                envelope = RoutingEnvelope(
                    origin_node=self.node_id,
                    htl=1,  # Direct query, no further routing
                    path_folding_enabled=False
                )
                return router._forward_to_node(
                    node, query_emb, query_text, envelope, top_k
                )

        return []

    def create_shortcut(
        self,
        query_text: str,
        target_node_id: str,
        target_interface_id: int = None
    ):
        """Create a path-folded shortcut."""
        query_hash = hashlib.sha256(query_text.encode()).hexdigest()[:16]
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO query_shortcuts
            (query_hash, target_node_id, target_interface_id)
            VALUES (?, ?, ?)
        """, (query_hash, target_node_id, target_interface_id))
        self.conn.commit()

    def _log_query(
        self,
        query_text: str,
        query_hash: str,
        origin_node: str,
        hops: int,
        result_count: int,
        response_time: float
    ):
        """Log a distributed query for analytics."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO distributed_query_log
            (query_text, query_hash, origin_node, hops, result_count, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (query_text, query_hash, origin_node, hops, result_count,
              response_time * 1000))
        self.conn.commit()

    def get_query_stats(self) -> Dict[str, Any]:
        """Get distributed query statistics."""
        cursor = self.conn.cursor()

        # Total queries
        cursor.execute("SELECT COUNT(*) FROM distributed_query_log")
        total_queries = cursor.fetchone()[0]

        # Average hops
        cursor.execute("SELECT AVG(hops) FROM distributed_query_log")
        avg_hops = cursor.fetchone()[0] or 0

        # Average response time
        cursor.execute("SELECT AVG(response_time_ms) FROM distributed_query_log")
        avg_response_ms = cursor.fetchone()[0] or 0

        # Shortcut stats
        cursor.execute("SELECT COUNT(*), SUM(hit_count) FROM query_shortcuts")
        row = cursor.fetchone()
        shortcut_count = row[0] or 0
        shortcut_hits = row[1] or 0

        return {
            'total_queries': total_queries,
            'avg_hops': round(avg_hops, 2),
            'avg_response_ms': round(avg_response_ms, 2),
            'shortcuts': {
                'count': shortcut_count,
                'total_hits': shortcut_hits
            },
            'node_id': self.node_id
        }

    def handle_federated_query(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle incoming federated query from another node.

        This is the HTTP endpoint handler for /kg/federated.
        Returns results with exp_scores and partition_sum for distributed
        softmax aggregation.

        Args:
            request: Incoming request with routing and aggregation config

        Returns:
            Response with results, exp_scores, and partition_sum
        """
        import math

        routing = request.get('__routing', {})
        embedding_info = request.get('__embedding', {})
        payload = request.get('payload', {})
        aggregation = routing.get('aggregation', {})

        query_text = payload.get('query_text', '')
        top_k = payload.get('top_k', 10)

        # Extract query embedding
        vector = embedding_info.get('vector', [])
        if vector:
            query_emb = np.array(vector, dtype=np.float32)
        else:
            model_name = embedding_info.get('model', 'all-MiniLM-L6-v2')
            query_emb = self._embed_query(query_text, model_name)

        # Local search
        results = self.search_with_context(
            query_text=query_text,
            model_name=embedding_info.get('model', 'all-MiniLM-L6-v2'),
            top_k=top_k
        )

        # Compute exp scores and partition sum for distributed softmax
        raw_scores = [r.get('score', 0.0) for r in results]

        # Log-sum-exp trick for numerical stability
        if raw_scores:
            max_score = max(raw_scores)
            exp_scores = [math.exp(s - max_score) for s in raw_scores]
            partition_sum = sum(exp_scores)
            # Scale back
            scale = math.exp(max_score)
            exp_scores = [e * scale for e in exp_scores]
            partition_sum *= scale
        else:
            exp_scores = []
            partition_sum = 0.0

        # Format results with exp_scores
        formatted_results = []
        for i, r in enumerate(results):
            answer_text = r.get('answer_text', r.get('text', ''))
            answer_hash = hashlib.sha256(answer_text.encode()).hexdigest()[:16]

            formatted_results.append({
                'answer_id': r.get('answer_id', r.get('id', i)),
                'answer_text': answer_text,
                'answer_hash': answer_hash,
                'raw_score': r.get('score', 0.0),
                'exp_score': exp_scores[i] if i < len(exp_scores) else 0.0,
                'metadata': {
                    'source_file': r.get('source_file'),
                    'record_id': r.get('record_id'),
                    'question_text': r.get('question_text'),
                    'interface_id': r.get('interface_id')
                }
            })

        # Get corpus info - prefer explicit corpus_id, fall back to generated
        corpus_id = getattr(self, '_corpus_id', None)
        data_sources = getattr(self, '_data_sources', [])

        if corpus_id is None:
            # Fall back to topic-based corpus ID
            interfaces = self.list_interfaces(active_only=True)
            if interfaces:
                corpus_id = '_'.join(interfaces[0].get('topics', [])[:3])
        else:
            interfaces = self.list_interfaces(active_only=True)

        return {
            '__type': 'kg_federated_response',
            '__id': request.get('__id'),
            'source_node': self.node_id,
            'results': formatted_results,
            'partition_sum': partition_sum,
            'node_metadata': {
                'corpus_id': corpus_id,
                'data_sources': data_sources,
                'embedding_model': embedding_info.get('model', 'all-MiniLM-L6-v2'),
                'interface_count': len(interfaces) if interfaces else 0
            }
        }

    def prune_shortcuts(self, max_age_days: int = 30, min_hits: int = 1):
        """
        Prune old or unused shortcuts.

        Args:
            max_age_days: Remove shortcuts older than this
            min_hits: Remove shortcuts with fewer hits
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM query_shortcuts
            WHERE last_used_at < datetime('now', '-' || ? || ' days')
               OR hit_count < ?
        """, (max_age_days, min_hits))
        self.conn.commit()
        return cursor.rowcount


def create_distributed_kg_database(
    db_path: str,
    embeddings_dir: str = None,
    node_id: str = None,
    discovery_backend: str = 'local',
    discovery_config: Dict[str, Any] = None
) -> DistributedKGTopologyAPI:
    """Create a new distributed KG topology database."""
    return DistributedKGTopologyAPI(
        db_path,
        embeddings_dir,
        node_id,
        discovery_backend,
        discovery_config
    )
