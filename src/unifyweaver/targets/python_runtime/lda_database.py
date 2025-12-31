# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# LDA Projection Database Layer

"""
SQLite database for storing embeddings, projections, and Q-A mappings.

Supports:
- Asymmetric embeddings (different models for input/output)
- Hierarchical answers with graph relations
- Multiple projection matrices per model pair
- Query logging for continuous learning
"""

import sqlite3
import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np


class LDAProjectionDB:
    """Database layer for LDA projection system."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str, embeddings_dir: str = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            embeddings_dir: Directory for storing large numpy arrays
                           (defaults to same directory as db)
        """
        self.db_path = db_path
        self.embeddings_dir = embeddings_dir or str(Path(db_path).parent / "embeddings")

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        cursor = self.conn.cursor()

        # Answers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                answer_id INTEGER PRIMARY KEY,
                parent_id INTEGER REFERENCES answers(answer_id),
                source_file TEXT NOT NULL,
                record_id TEXT,
                text TEXT NOT NULL,
                text_variant TEXT DEFAULT 'default',
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Answer relations (graph)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answer_relations (
                relation_id INTEGER PRIMARY KEY,
                from_answer_id INTEGER REFERENCES answers(answer_id),
                to_answer_id INTEGER REFERENCES answers(answer_id),
                relation_type TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(from_answer_id, to_answer_id, relation_type)
            )
        """)

        # Questions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                question_id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                length_type TEXT CHECK(length_type IN ('short', 'medium', 'long')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Clusters
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_clusters (
                cluster_id INTEGER PRIMARY KEY,
                name TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_answers (
                cluster_id INTEGER REFERENCES qa_clusters(cluster_id),
                answer_id INTEGER REFERENCES answers(answer_id),
                PRIMARY KEY (cluster_id, answer_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_questions (
                cluster_id INTEGER REFERENCES qa_clusters(cluster_id),
                question_id INTEGER REFERENCES questions(question_id),
                PRIMARY KEY (cluster_id, question_id)
            )
        """)

        # Embedding models
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_models (
                model_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                dimension INTEGER NOT NULL,
                backend TEXT,
                max_tokens INTEGER,
                notes TEXT
            )
        """)

        # Embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY,
                model_id INTEGER REFERENCES embedding_models(model_id),
                entity_type TEXT CHECK(entity_type IN ('answer', 'question')),
                entity_id INTEGER,
                vector_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_id, entity_type, entity_id)
            )
        """)

        # Projections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projections (
                projection_id INTEGER PRIMARY KEY,
                input_model_id INTEGER REFERENCES embedding_models(model_id),
                output_model_id INTEGER REFERENCES embedding_models(model_id),
                name TEXT,
                W_path TEXT NOT NULL,
                lambda_reg REAL,
                ridge REAL,
                num_clusters INTEGER,
                num_queries INTEGER,
                recall_at_1 REAL,
                mrr REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projection_clusters (
                projection_id INTEGER REFERENCES projections(projection_id),
                cluster_id INTEGER REFERENCES qa_clusters(cluster_id),
                PRIMARY KEY (projection_id, cluster_id)
            )
        """)

        # Query log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                log_id INTEGER PRIMARY KEY,
                query_text TEXT NOT NULL,
                model_id INTEGER REFERENCES embedding_models(model_id),
                projection_id INTEGER REFERENCES projections(projection_id),
                results TEXT,
                selected_answer_id INTEGER,
                was_helpful BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Training batches
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_batches (
                batch_id INTEGER PRIMARY KEY,
                source_file TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                status TEXT CHECK(status IN ('pending', 'importing', 'embedding', 'training', 'completed', 'failed')),
                model_name TEXT,
                num_clusters INTEGER,
                num_questions INTEGER,
                projection_id INTEGER REFERENCES projections(projection_id),
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_file, file_hash)
            )
        """)

        # Batch status history (tracks all status transitions with timestamps)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_status_history (
                history_id INTEGER PRIMARY KEY,
                batch_id INTEGER REFERENCES training_batches(batch_id),
                status TEXT NOT NULL,
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Multi-head projections (groups of per-cluster heads)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS multi_head_projections (
                mh_projection_id INTEGER PRIMARY KEY,
                model_id INTEGER REFERENCES embedding_models(model_id),
                name TEXT,
                temperature REAL DEFAULT 1.0,
                num_heads INTEGER,
                recall_at_1 REAL,
                mrr REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Per-cluster heads (attention heads)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_heads (
                head_id INTEGER PRIMARY KEY,
                mh_projection_id INTEGER REFERENCES multi_head_projections(mh_projection_id),
                cluster_id INTEGER REFERENCES qa_clusters(cluster_id),
                centroid_path TEXT NOT NULL,
                answer_emb_path TEXT NOT NULL,
                W_path TEXT,
                num_questions INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(mh_projection_id, cluster_id)
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(model_id, entity_type, entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_from ON answer_relations(from_answer_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_to ON answer_relations(to_answer_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_type ON answer_relations(relation_type)")

        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # Answers
    # =========================================================================

    def add_answer(
        self,
        source_file: str,
        text: str,
        record_id: str = None,
        text_variant: str = "default",
        parent_id: int = None,
        chunk_index: int = None
    ) -> int:
        """Add an answer document."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO answers (source_file, text, record_id, text_variant, parent_id, chunk_index)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (source_file, text, record_id, text_variant, parent_id, chunk_index))
        self.conn.commit()
        return cursor.lastrowid

    def get_answer(self, answer_id: int) -> Optional[Dict]:
        """Get answer by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM answers WHERE answer_id = ?", (answer_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def find_answers(self, source_file: str = None, record_id: str = None) -> List[Dict]:
        """Find answers matching criteria."""
        cursor = self.conn.cursor()
        conditions = []
        params = []

        if source_file:
            conditions.append("source_file = ?")
            params.append(source_file)
        if record_id:
            conditions.append("record_id = ?")
            params.append(record_id)

        where = " AND ".join(conditions) if conditions else "1=1"
        cursor.execute(f"SELECT * FROM answers WHERE {where}", params)
        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Relations
    # =========================================================================

    def add_relation(
        self,
        from_id: int,
        to_id: int,
        relation_type: str,
        metadata: Dict = None
    ) -> int:
        """Create a relation between answers."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO answer_relations (from_answer_id, to_answer_id, relation_type, metadata)
            VALUES (?, ?, ?, ?)
        """, (from_id, to_id, relation_type, json.dumps(metadata) if metadata else None))
        self.conn.commit()
        return cursor.lastrowid

    def get_related(
        self,
        answer_id: int,
        relation_type: str = None,
        direction: str = "from"
    ) -> List[Dict]:
        """Get related answers.

        Args:
            answer_id: The answer to find relations for
            relation_type: Filter by type (optional)
            direction: 'from' (outgoing) or 'to' (incoming)
        """
        cursor = self.conn.cursor()

        if direction == "from":
            col = "from_answer_id"
            other_col = "to_answer_id"
        else:
            col = "to_answer_id"
            other_col = "from_answer_id"

        if relation_type:
            cursor.execute(f"""
                SELECT ar.*, a.* FROM answer_relations ar
                JOIN answers a ON ar.{other_col} = a.answer_id
                WHERE ar.{col} = ? AND ar.relation_type = ?
            """, (answer_id, relation_type))
        else:
            cursor.execute(f"""
                SELECT ar.*, a.* FROM answer_relations ar
                JOIN answers a ON ar.{other_col} = a.answer_id
                WHERE ar.{col} = ?
            """, (answer_id,))

        return [dict(row) for row in cursor.fetchall()]

    def get_full_version(self, answer_id: int) -> Optional[Dict]:
        """Follow hierarchy relations to find the full document."""
        current = self.get_answer(answer_id)
        if not current:
            return None

        # Follow chunk_of, summarizes, abbreviates relations
        hierarchy_types = ('chunk_of', 'summarizes', 'abbreviates')

        visited = {answer_id}
        while True:
            related = self.get_related(answer_id, direction="from")
            parent = None
            for r in related:
                if r['relation_type'] in hierarchy_types and r['answer_id'] not in visited:
                    parent = r
                    break

            if not parent:
                return current

            answer_id = parent['answer_id']
            visited.add(answer_id)
            current = self.get_answer(answer_id)

    # =========================================================================
    # Questions
    # =========================================================================

    def add_question(self, text: str, length_type: str = "medium") -> int:
        """Add a question."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO questions (text, length_type)
            VALUES (?, ?)
        """, (text, length_type))
        self.conn.commit()
        return cursor.lastrowid

    def get_question(self, question_id: int) -> Optional[Dict]:
        """Get question by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM questions WHERE question_id = ?", (question_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    # =========================================================================
    # Clusters
    # =========================================================================

    def create_cluster(
        self,
        name: str,
        answer_ids: List[int],
        question_ids: List[int],
        description: str = None
    ) -> int:
        """Create a Q-A cluster."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO qa_clusters (name, description)
            VALUES (?, ?)
        """, (name, description))
        cluster_id = cursor.lastrowid

        for aid in answer_ids:
            cursor.execute("""
                INSERT INTO cluster_answers (cluster_id, answer_id)
                VALUES (?, ?)
            """, (cluster_id, aid))

        for qid in question_ids:
            cursor.execute("""
                INSERT INTO cluster_questions (cluster_id, question_id)
                VALUES (?, ?)
            """, (cluster_id, qid))

        self.conn.commit()
        return cluster_id

    def get_cluster(self, cluster_id: int) -> Optional[Dict]:
        """Get cluster with its answers and questions."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM qa_clusters WHERE cluster_id = ?", (cluster_id,))
        cluster = cursor.fetchone()
        if not cluster:
            return None

        result = dict(cluster)

        cursor.execute("""
            SELECT a.* FROM answers a
            JOIN cluster_answers ca ON a.answer_id = ca.answer_id
            WHERE ca.cluster_id = ?
        """, (cluster_id,))
        result['answers'] = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT q.* FROM questions q
            JOIN cluster_questions cq ON q.question_id = cq.question_id
            WHERE cq.cluster_id = ?
        """, (cluster_id,))
        result['questions'] = [dict(row) for row in cursor.fetchall()]

        return result

    def get_questions_for_answer(self, answer_id: int) -> List[Dict]:
        """Reverse lookup: Get all training questions that map to an answer.

        Args:
            answer_id: The answer ID to look up

        Returns:
            List of question dicts with cluster info
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT q.*, qc.cluster_id, qc.name as cluster_name
            FROM questions q
            JOIN cluster_questions cq ON q.question_id = cq.question_id
            JOIN cluster_answers ca ON cq.cluster_id = ca.cluster_id
            JOIN qa_clusters qc ON cq.cluster_id = qc.cluster_id
            WHERE ca.answer_id = ?
            ORDER BY q.length_type, q.question_id
        """, (answer_id,))
        return [dict(row) for row in cursor.fetchall()]

    def search_answers(self, text_pattern: str) -> List[Dict]:
        """Search answers by text pattern.

        Args:
            text_pattern: SQL LIKE pattern (use % for wildcards)

        Returns:
            List of matching answer dicts
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT a.*, qc.name as cluster_name, qc.cluster_id
            FROM answers a
            LEFT JOIN cluster_answers ca ON a.answer_id = ca.answer_id
            LEFT JOIN qa_clusters qc ON ca.cluster_id = qc.cluster_id
            WHERE a.text LIKE ? OR a.record_id LIKE ?
            ORDER BY a.answer_id
        """, (text_pattern, text_pattern))
        return [dict(row) for row in cursor.fetchall()]

    def list_clusters(self) -> List[Dict]:
        """List all clusters."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM qa_clusters")
        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Embedding Models
    # =========================================================================

    def add_model(
        self,
        name: str,
        dimension: int,
        backend: str = None,
        max_tokens: int = None,
        notes: str = None
    ) -> int:
        """Register an embedding model."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO embedding_models (name, dimension, backend, max_tokens, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (name, dimension, backend, max_tokens, notes))
        self.conn.commit()

        cursor.execute("SELECT model_id FROM embedding_models WHERE name = ?", (name,))
        return cursor.fetchone()[0]

    def get_model(self, name: str) -> Optional[Dict]:
        """Get model by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM embedding_models WHERE name = ?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_model_by_id(self, model_id: int) -> Optional[Dict]:
        """Get model by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM embedding_models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    # =========================================================================
    # Embeddings
    # =========================================================================

    def _embedding_path(self, model_id: int, entity_type: str, entity_id: int) -> str:
        """Generate path for embedding numpy file."""
        return os.path.join(
            self.embeddings_dir,
            f"{entity_type}_{entity_id}_model_{model_id}.npy"
        )

    def store_embedding(
        self,
        model_id: int,
        entity_type: str,
        entity_id: int,
        vector: np.ndarray
    ) -> int:
        """Store an embedding vector."""
        path = self._embedding_path(model_id, entity_type, entity_id)
        np.save(path, vector)

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings (model_id, entity_type, entity_id, vector_path)
            VALUES (?, ?, ?, ?)
        """, (model_id, entity_type, entity_id, path))
        self.conn.commit()
        return cursor.lastrowid

    def get_embedding(
        self,
        model_id: int,
        entity_type: str,
        entity_id: int
    ) -> Optional[np.ndarray]:
        """Load an embedding vector."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT vector_path FROM embeddings
            WHERE model_id = ? AND entity_type = ? AND entity_id = ?
        """, (model_id, entity_type, entity_id))
        row = cursor.fetchone()

        if not row:
            return None

        return np.load(row['vector_path'])

    def get_all_answer_embeddings(self, model_id: int) -> Tuple[List[int], np.ndarray]:
        """Get all answer embeddings for a model.

        Returns:
            Tuple of (answer_ids, embedding_matrix)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT entity_id, vector_path FROM embeddings
            WHERE model_id = ? AND entity_type = 'answer'
            ORDER BY entity_id
        """, (model_id,))

        ids = []
        vectors = []
        for row in cursor.fetchall():
            ids.append(row['entity_id'])
            vectors.append(np.load(row['vector_path']))

        if not vectors:
            return [], np.array([])

        return ids, np.stack(vectors)

    # =========================================================================
    # Projections
    # =========================================================================

    def _projection_path(self, projection_id: int) -> str:
        """Generate path for projection W matrix."""
        return os.path.join(self.embeddings_dir, f"W_projection_{projection_id}.npy")

    def store_projection(
        self,
        input_model_id: int,
        output_model_id: int,
        W: np.ndarray,
        name: str = None,
        cluster_ids: List[int] = None,
        lambda_reg: float = None,
        ridge: float = None,
        metrics: Dict = None
    ) -> int:
        """Store a trained projection matrix."""
        cursor = self.conn.cursor()

        # Insert projection record (without path first to get ID)
        cursor.execute("""
            INSERT INTO projections (
                input_model_id, output_model_id, name, W_path,
                lambda_reg, ridge, num_clusters, num_queries,
                recall_at_1, mrr
            ) VALUES (?, ?, ?, '', ?, ?, ?, ?, ?, ?)
        """, (
            input_model_id, output_model_id, name,
            lambda_reg, ridge,
            metrics.get('num_clusters') if metrics else None,
            metrics.get('total_queries') if metrics else None,
            metrics.get('recall_at_1') if metrics else None,
            metrics.get('mrr') if metrics else None
        ))
        projection_id = cursor.lastrowid

        # Save W matrix
        path = self._projection_path(projection_id)
        np.save(path, W)

        # Update with path
        cursor.execute("""
            UPDATE projections SET W_path = ? WHERE projection_id = ?
        """, (path, projection_id))

        # Link clusters
        if cluster_ids:
            for cid in cluster_ids:
                cursor.execute("""
                    INSERT INTO projection_clusters (projection_id, cluster_id)
                    VALUES (?, ?)
                """, (projection_id, cid))

        self.conn.commit()
        return projection_id

    def get_projection(self, projection_id: int) -> Optional[Tuple[np.ndarray, Dict]]:
        """Load a projection matrix and its metadata."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM projections WHERE projection_id = ?", (projection_id,))
        row = cursor.fetchone()

        if not row:
            return None

        metadata = dict(row)
        W = np.load(row['W_path'])

        return W, metadata

    def list_projections(self, input_model: str = None, output_model: str = None) -> List[Dict]:
        """List projections, optionally filtered by model names."""
        cursor = self.conn.cursor()

        query = """
            SELECT p.*, im.name as input_model_name, om.name as output_model_name
            FROM projections p
            JOIN embedding_models im ON p.input_model_id = im.model_id
            JOIN embedding_models om ON p.output_model_id = om.model_id
        """
        conditions = []
        params = []

        if input_model:
            conditions.append("im.name = ?")
            params.append(input_model)
        if output_model:
            conditions.append("om.name = ?")
            params.append(output_model)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Query Logging
    # =========================================================================

    def log_query(
        self,
        query_text: str,
        projection_id: int,
        results: List[int],
        selected_id: int = None,
        was_helpful: bool = None
    ):
        """Log a query for continuous learning."""
        cursor = self.conn.cursor()

        # Get model_id from projection
        cursor.execute("SELECT input_model_id FROM projections WHERE projection_id = ?", (projection_id,))
        row = cursor.fetchone()
        model_id = row['input_model_id'] if row else None

        cursor.execute("""
            INSERT INTO query_log (query_text, model_id, projection_id, results, selected_answer_id, was_helpful)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (query_text, model_id, projection_id, json.dumps(results), selected_id, was_helpful))
        self.conn.commit()

    def get_query_log(self, limit: int = 100) -> List[Dict]:
        """Get recent query logs."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM query_log ORDER BY created_at DESC LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def update_query_feedback(self, log_id: int, selected_id: int = None, was_helpful: bool = None):
        """Update feedback on a logged query."""
        cursor = self.conn.cursor()
        updates = []
        params = []

        if selected_id is not None:
            updates.append("selected_answer_id = ?")
            params.append(selected_id)
        if was_helpful is not None:
            updates.append("was_helpful = ?")
            params.append(was_helpful)

        if updates:
            params.append(log_id)
            cursor.execute(f"""
                UPDATE query_log SET {', '.join(updates)} WHERE log_id = ?
            """, params)
            self.conn.commit()

    # =========================================================================
    # Search API
    # =========================================================================

    def search(
        self,
        query_embedding: np.ndarray,
        projection_id: int,
        top_k: int = 5,
        log: bool = True,
        query_text: str = None
    ) -> List[Dict]:
        """Search for similar answers using LDA projection.

        Args:
            query_embedding: The embedded query vector (input model space)
            projection_id: Which projection to use
            top_k: Number of results to return
            log: Whether to log this query
            query_text: Original query text (for logging)

        Returns:
            List of dicts with answer_id, score, and answer metadata
        """
        # Load projection
        result = self.get_projection(projection_id)
        if not result:
            raise ValueError(f"Projection {projection_id} not found")

        W, metadata = result
        output_model_id = metadata['output_model_id']

        # Project query: projected = W @ query
        projected = W @ query_embedding

        # Normalize projected query
        proj_norm = np.linalg.norm(projected)
        if proj_norm > 0:
            projected = projected / proj_norm

        # Get all answer embeddings
        answer_ids, answer_matrix = self.get_all_answer_embeddings(output_model_id)

        if len(answer_ids) == 0:
            return []

        # Normalize answer embeddings
        answer_norms = np.linalg.norm(answer_matrix, axis=1, keepdims=True)
        answer_norms = np.where(answer_norms > 0, answer_norms, 1)  # Avoid division by zero
        answer_matrix_normed = answer_matrix / answer_norms

        # Compute similarities
        similarities = answer_matrix_normed @ projected

        # Get top-k indices
        top_indices = np.argsort(-similarities)[:top_k]

        # Build results
        results = []
        for idx in top_indices:
            answer_id = answer_ids[idx]
            answer = self.get_answer(answer_id)
            results.append({
                'answer_id': answer_id,
                'score': float(similarities[idx]),
                **answer
            })

        # Log query
        if log and query_text:
            self.log_query(
                query_text=query_text,
                projection_id=projection_id,
                results=[r['answer_id'] for r in results]
            )

        return results

    def search_with_embedder(
        self,
        query_text: str,
        projection_id: int,
        embedder,
        top_k: int = 5,
        log: bool = True
    ) -> List[Dict]:
        """Search using a provided embedder function/object.

        Args:
            query_text: The query string
            projection_id: Which projection to use
            embedder: Object with .encode() method or callable
            top_k: Number of results to return
            log: Whether to log this query

        Returns:
            List of dicts with answer_id, score, and answer metadata
        """
        # Embed query
        if hasattr(embedder, 'encode'):
            query_emb = embedder.encode(query_text, convert_to_numpy=True)
        else:
            query_emb = embedder(query_text)

        return self.search(
            query_embedding=query_emb,
            projection_id=projection_id,
            top_k=top_k,
            log=log,
            query_text=query_text
        )

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    def get_variants(self, answer_id: int) -> List[Dict]:
        """Get all variants of an answer (different text representations)."""
        variants = []

        # Get variant_of relations (outgoing)
        variants.extend(self.get_related(answer_id, "variant_of", direction="from"))

        # Get variant_of relations (incoming - this is a variant of others)
        variants.extend(self.get_related(answer_id, "variant_of", direction="to"))

        return variants

    def get_chunks(self, answer_id: int) -> List[Dict]:
        """Get all chunks of a document."""
        return self.get_related(answer_id, "chunk_of", direction="to")

    def get_next_chunk(self, answer_id: int) -> Optional[Dict]:
        """Get the next chunk in sequence."""
        related = self.get_related(answer_id, "next_chunk", direction="from")
        return related[0] if related else None

    def get_prev_chunk(self, answer_id: int) -> Optional[Dict]:
        """Get the previous chunk in sequence."""
        related = self.get_related(answer_id, "prev_chunk", direction="from")
        return related[0] if related else None

    def get_translations(self, answer_id: int) -> List[Dict]:
        """Get translations of an answer."""
        return self.get_related(answer_id, "translates", direction="to")

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def embed_and_store_batch(
        self,
        model_id: int,
        entity_type: str,
        items: List[Tuple[int, np.ndarray]]
    ) -> int:
        """Store multiple embeddings efficiently.

        Args:
            model_id: The embedding model ID
            entity_type: 'answer' or 'question'
            items: List of (entity_id, vector) tuples

        Returns:
            Number of embeddings stored
        """
        for entity_id, vector in items:
            self.store_embedding(model_id, entity_type, entity_id, vector)
        return len(items)

    def get_cluster_embeddings(
        self,
        cluster_id: int,
        model_id: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get all embeddings for a cluster.

        Returns:
            Tuple of (answer_embeddings, question_embeddings)
        """
        cluster = self.get_cluster(cluster_id)
        if not cluster:
            return [], []

        answer_embs = []
        for a in cluster['answers']:
            emb = self.get_embedding(model_id, 'answer', a['answer_id'])
            if emb is not None:
                answer_embs.append(emb)

        question_embs = []
        for q in cluster['questions']:
            emb = self.get_embedding(model_id, 'question', q['question_id'])
            if emb is not None:
                question_embs.append(emb)

        return answer_embs, question_embs

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM answers")
        stats['num_answers'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM questions")
        stats['num_questions'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM qa_clusters")
        stats['num_clusters'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM embeddings")
        stats['num_embeddings'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM projections")
        stats['num_projections'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM query_log")
        stats['num_queries_logged'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM answer_relations")
        stats['num_relations'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM embedding_models")
        stats['num_models'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM training_batches")
        stats['num_batches'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM training_batches WHERE status = 'completed'")
        stats['num_batches_completed'] = cursor.fetchone()[0]

        return stats

    # =========================================================================
    # Training Batches
    # =========================================================================

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def register_batch(
        self,
        source_file: str,
        model_name: str = None
    ) -> Tuple[int, str]:
        """Register a training batch file.

        Args:
            source_file: Path to the Q-A pairs JSON file
            model_name: Target embedding model name

        Returns:
            Tuple of (batch_id, status) where status is:
            - 'new': New batch registered
            - 'unchanged': File already processed with same hash
            - 'modified': File changed, new batch registered
        """
        file_hash = self.compute_file_hash(source_file)
        cursor = self.conn.cursor()

        # Check if we have this exact file+hash
        cursor.execute("""
            SELECT batch_id, status FROM training_batches
            WHERE source_file = ? AND file_hash = ?
        """, (source_file, file_hash))
        existing = cursor.fetchone()

        if existing:
            return existing['batch_id'], 'unchanged'

        # Check if file exists with different hash
        cursor.execute("""
            SELECT batch_id FROM training_batches
            WHERE source_file = ? AND file_hash != ?
        """, (source_file, file_hash))
        modified = cursor.fetchone()

        # Insert new batch
        cursor.execute("""
            INSERT INTO training_batches (source_file, file_hash, status, model_name)
            VALUES (?, ?, 'pending', ?)
        """, (source_file, file_hash, model_name))
        batch_id = cursor.lastrowid

        # Log initial status
        self._log_batch_status(batch_id, 'pending', 'Batch registered')

        self.conn.commit()

        status = 'modified' if modified else 'new'
        return batch_id, status

    def _log_batch_status(self, batch_id: int, status: str, message: str = None):
        """Log a status change to batch history."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO batch_status_history (batch_id, status, message)
            VALUES (?, ?, ?)
        """, (batch_id, status, message))

    def update_batch_status(
        self,
        batch_id: int,
        status: str,
        message: str = None,
        num_clusters: int = None,
        num_questions: int = None,
        projection_id: int = None,
        error_message: str = None
    ):
        """Update batch status and optionally other fields."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        updates = ["status = ?"]
        params = [status]

        if status == 'importing' or status == 'embedding' or status == 'training':
            updates.append("started_at = COALESCE(started_at, ?)")
            params.append(now)

        if status == 'completed':
            updates.append("completed_at = ?")
            params.append(now)

        if num_clusters is not None:
            updates.append("num_clusters = ?")
            params.append(num_clusters)

        if num_questions is not None:
            updates.append("num_questions = ?")
            params.append(num_questions)

        if projection_id is not None:
            updates.append("projection_id = ?")
            params.append(projection_id)

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(batch_id)
        cursor.execute(f"""
            UPDATE training_batches SET {', '.join(updates)}
            WHERE batch_id = ?
        """, params)

        # Log to history
        self._log_batch_status(batch_id, status, message or error_message)

        self.conn.commit()

    def get_batch(self, batch_id: int) -> Optional[Dict]:
        """Get batch by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM training_batches WHERE batch_id = ?", (batch_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_batch_by_file(self, source_file: str) -> List[Dict]:
        """Get all batches for a source file (may have multiple hashes)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM training_batches
            WHERE source_file = ?
            ORDER BY created_at DESC
        """, (source_file,))
        return [dict(row) for row in cursor.fetchall()]

    def get_pending_batches(self) -> List[Dict]:
        """Get all pending batches ready for processing."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM training_batches
            WHERE status = 'pending'
            ORDER BY created_at ASC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_failed_batches(self) -> List[Dict]:
        """Get all failed batches that could be retried."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM training_batches
            WHERE status = 'failed'
            ORDER BY created_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_batch_history(self, batch_id: int) -> List[Dict]:
        """Get status history for a batch."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM batch_status_history
            WHERE batch_id = ?
            ORDER BY created_at ASC
        """, (batch_id,))
        return [dict(row) for row in cursor.fetchall()]

    def scan_for_new_batches(self, directory: str, model_name: str = None) -> List[Tuple[int, str]]:
        """Scan a directory for new or modified Q-A JSON files.

        Args:
            directory: Path to scan for .json files
            model_name: Target embedding model

        Returns:
            List of (batch_id, status) tuples for registered batches
        """
        results = []
        dir_path = Path(directory)

        for json_file in dir_path.glob("*.json"):
            batch_id, status = self.register_batch(str(json_file), model_name)
            if status != 'unchanged':
                results.append((batch_id, status))

        return results

    def list_batches(self, status: str = None) -> List[Dict]:
        """List all batches, optionally filtered by status."""
        cursor = self.conn.cursor()

        if status:
            cursor.execute("""
                SELECT * FROM training_batches
                WHERE status = ?
                ORDER BY created_at DESC
            """, (status,))
        else:
            cursor.execute("""
                SELECT * FROM training_batches
                ORDER BY created_at DESC
            """)

        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Multi-Head Projections
    # =========================================================================

    def _head_path(self, mh_id: int, cluster_id: int, kind: str) -> str:
        """Generate path for cluster head vectors."""
        return os.path.join(
            self.embeddings_dir,
            f"mh_{mh_id}_cluster_{cluster_id}_{kind}.npy"
        )

    def create_multi_head_projection(
        self,
        model_id: int,
        name: str = None,
        temperature: float = 1.0
    ) -> int:
        """Create a new multi-head projection.

        Args:
            model_id: The embedding model ID
            name: Optional name for this projection
            temperature: Softmax temperature for routing (default 1.0)

        Returns:
            The mh_projection_id
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO multi_head_projections (model_id, name, temperature, num_heads)
            VALUES (?, ?, ?, 0)
        """, (model_id, name, temperature))
        self.conn.commit()
        return cursor.lastrowid

    def add_cluster_head(
        self,
        mh_projection_id: int,
        cluster_id: int,
        centroid: np.ndarray,
        answer_emb: np.ndarray,
        W: np.ndarray = None,
        num_questions: int = None
    ) -> int:
        """Add a cluster head to a multi-head projection.

        Args:
            mh_projection_id: The multi-head projection ID
            cluster_id: The cluster this head represents
            centroid: Question centroid vector (d,)
            answer_emb: Answer embedding vector (d,)
            W: Optional per-cluster projection matrix (d, d)
            num_questions: Number of questions in cluster

        Returns:
            The head_id
        """
        cursor = self.conn.cursor()

        # Save centroid
        centroid_path = self._head_path(mh_projection_id, cluster_id, 'centroid')
        np.save(centroid_path, centroid)

        # Save answer embedding
        answer_path = self._head_path(mh_projection_id, cluster_id, 'answer')
        np.save(answer_path, answer_emb)

        # Optionally save per-cluster W
        W_path = None
        if W is not None:
            W_path = self._head_path(mh_projection_id, cluster_id, 'W')
            np.save(W_path, W)

        cursor.execute("""
            INSERT INTO cluster_heads
            (mh_projection_id, cluster_id, centroid_path, answer_emb_path, W_path, num_questions)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (mh_projection_id, cluster_id, centroid_path, answer_path, W_path, num_questions))
        head_id = cursor.lastrowid

        # Update head count
        cursor.execute("""
            UPDATE multi_head_projections
            SET num_heads = num_heads + 1
            WHERE mh_projection_id = ?
        """, (mh_projection_id,))

        self.conn.commit()
        return head_id

    def get_multi_head_projection(self, mh_projection_id: int) -> Optional[Dict]:
        """Get multi-head projection metadata."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM multi_head_projections WHERE mh_projection_id = ?
        """, (mh_projection_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_cluster_heads(self, mh_projection_id: int) -> List[Dict]:
        """Get all cluster heads for a multi-head projection."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ch.*, qc.name as cluster_name
            FROM cluster_heads ch
            LEFT JOIN qa_clusters qc ON ch.cluster_id = qc.cluster_id
            WHERE ch.mh_projection_id = ?
            ORDER BY ch.cluster_id
        """, (mh_projection_id,))
        return [dict(row) for row in cursor.fetchall()]

    def load_multi_head_data(
        self,
        mh_projection_id: int
    ) -> Tuple[np.ndarray, np.ndarray, List[int], Optional[List[np.ndarray]]]:
        """Load all data needed for multi-head inference.

        Returns:
            Tuple of:
            - centroids: (num_heads, d) matrix of cluster centroids
            - answer_embs: (num_heads, d) matrix of answer embeddings
            - cluster_ids: List of cluster IDs in same order
            - W_matrices: Optional list of per-cluster W matrices
        """
        heads = self.get_cluster_heads(mh_projection_id)

        if not heads:
            return np.array([]), np.array([]), [], None

        centroids = []
        answer_embs = []
        cluster_ids = []
        W_matrices = []
        has_W = False

        for head in heads:
            centroids.append(np.load(head['centroid_path']))
            answer_embs.append(np.load(head['answer_emb_path']))
            cluster_ids.append(head['cluster_id'])

            if head['W_path']:
                W_matrices.append(np.load(head['W_path']))
                has_W = True
            else:
                W_matrices.append(None)

        return (
            np.stack(centroids),
            np.stack(answer_embs),
            cluster_ids,
            W_matrices if has_W else None
        )

    def multi_head_search(
        self,
        query_embedding: np.ndarray,
        mh_projection_id: int,
        top_k: int = 5,
        log: bool = True,
        query_text: str = None
    ) -> List[Dict]:
        """Search using multi-head projection with soft routing.

        The query is routed to cluster heads based on similarity to centroids,
        then projected using weighted combination of per-cluster projections.

        Args:
            query_embedding: The query vector (d,)
            mh_projection_id: Which multi-head projection to use
            top_k: Number of results to return
            log: Whether to log this query
            query_text: Original query text (for logging)

        Returns:
            List of dicts with answer_id, score, routing_weights, and metadata
        """
        mh_proj = self.get_multi_head_projection(mh_projection_id)
        if not mh_proj:
            raise ValueError(f"Multi-head projection {mh_projection_id} not found")

        centroids, answer_embs, cluster_ids, W_matrices = self.load_multi_head_data(mh_projection_id)

        if len(cluster_ids) == 0:
            return []

        temperature = mh_proj.get('temperature', 1.0)
        model_id = mh_proj['model_id']

        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_normed = query_embedding / query_norm
        else:
            query_normed = query_embedding

        # Normalize centroids
        centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroid_norms = np.where(centroid_norms > 0, centroid_norms, 1)
        centroids_normed = centroids / centroid_norms

        # Compute routing weights via softmax over centroid similarities
        similarities = centroids_normed @ query_normed
        scaled = similarities / temperature
        exp_scaled = np.exp(scaled - np.max(scaled))  # Numerical stability
        routing_weights = exp_scaled / np.sum(exp_scaled)

        # Project query using weighted combination
        if W_matrices:
            # Use per-cluster W matrices
            projected = np.zeros_like(query_embedding)
            for i, (w, W) in enumerate(zip(routing_weights, W_matrices)):
                if W is not None:
                    projected += w * (W @ query_embedding)
                else:
                    # Fallback: project toward answer embedding
                    projected += w * answer_embs[i]
        else:
            # No W matrices: project toward weighted combination of answer embeddings
            projected = routing_weights @ answer_embs

        # Normalize projected query
        proj_norm = np.linalg.norm(projected)
        if proj_norm > 0:
            projected = projected / proj_norm

        # Get all answer embeddings for comparison
        answer_ids, answer_matrix = self.get_all_answer_embeddings(model_id)

        if len(answer_ids) == 0:
            return []

        # Normalize answer matrix
        answer_norms = np.linalg.norm(answer_matrix, axis=1, keepdims=True)
        answer_norms = np.where(answer_norms > 0, answer_norms, 1)
        answer_matrix_normed = answer_matrix / answer_norms

        # Compute similarities
        scores = answer_matrix_normed @ projected

        # Get top-k
        top_indices = np.argsort(-scores)[:top_k]

        # Build results with routing info
        results = []
        routing_info = {
            cluster_ids[i]: float(routing_weights[i])
            for i in range(len(cluster_ids))
        }

        for idx in top_indices:
            answer_id = answer_ids[idx]
            answer = self.get_answer(answer_id)
            results.append({
                'answer_id': answer_id,
                'score': float(scores[idx]),
                'routing_weights': routing_info,
                **answer
            })

        return results

    def update_multi_head_metrics(
        self,
        mh_projection_id: int,
        recall_at_1: float = None,
        mrr: float = None
    ):
        """Update validation metrics for a multi-head projection."""
        cursor = self.conn.cursor()
        updates = []
        params = []

        if recall_at_1 is not None:
            updates.append("recall_at_1 = ?")
            params.append(recall_at_1)
        if mrr is not None:
            updates.append("mrr = ?")
            params.append(mrr)

        if updates:
            params.append(mh_projection_id)
            cursor.execute(f"""
                UPDATE multi_head_projections SET {', '.join(updates)}
                WHERE mh_projection_id = ?
            """, params)
            self.conn.commit()
