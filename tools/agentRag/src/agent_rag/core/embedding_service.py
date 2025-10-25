"""
Embedding Service for Agent-Based RAG
Optional service for semantic search enhancement
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

import json
import sqlite3
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import faiss
import pickle
import logging

# Get database path from config
try:
    from agent_rag.config import get_config
    config = get_config()
    DEFAULT_DB_PATH = config["db_path"]
except ImportError:
    DEFAULT_DB_PATH = os.getenv("DB_PATH", "rag_index.db")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class EmbeddingService:
    """Manages embeddings and semantic search"""
    
    def __init__(self, model_name: str = "nomic-ai/modernbert-base", 
                 db_path: str = "rag_index.db"):
        self.model_name = model_name
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunk_map = {}  # Maps index position to chunk_id
        self.init_db()
        self.load_or_create_index()
    
    def init_db(self):
        """Initialize database tables for embeddings"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_name TEXT,
                created_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        index_path = Path("faiss_index.bin")
        map_path = Path("chunk_map.pkl")
        
        if index_path.exists() and map_path.exists():
            logger.info("Loading existing FAISS index")
            self.index = faiss.read_index(str(index_path))
            with open(map_path, 'rb') as f:
                self.chunk_map = pickle.load(f)
        else:
            logger.info("Creating new FAISS index")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.rebuild_index()
    
    def rebuild_index(self):
        """Rebuild FAISS index from database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT chunk_id, embedding FROM embeddings")
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            logger.info("No embeddings in database")
            return
        
        # Reset index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunk_map = {}
        
        # Add embeddings to index
        for i, (chunk_id, embedding_blob) in enumerate(rows):
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            self.index.add(embedding.reshape(1, -1))
            self.chunk_map[i] = chunk_id
        
        logger.info(f"Rebuilt index with {len(rows)} embeddings")
        self.save_index()
    
    def save_index(self):
        """Save FAISS index and chunk map to disk"""
        faiss.write_index(self.index, "faiss_index.bin")
        with open("chunk_map.pkl", 'wb') as f:
            pickle.dump(self.chunk_map, f)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding"""
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts to embeddings"""
        return self.model.encode(texts, convert_to_numpy=True, 
                                normalize_embeddings=True, batch_size=32)
    
    def add_embeddings(self, chunks: List[Dict[str, Any]]):
        """Add chunk embeddings to database and index"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.encode_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            # Add to database
            c.execute("""
                INSERT OR REPLACE INTO embeddings 
                (chunk_id, embedding, model_name, created_at)
                VALUES (?, ?, ?, datetime('now'))
            """, (
                chunk['chunk_id'],
                embedding.tobytes(),
                self.model_name
            ))
            
            # Add to FAISS index
            index_pos = len(self.chunk_map)
            self.index.add(embedding.reshape(1, -1))
            self.chunk_map[index_pos] = chunk['chunk_id']
        
        conn.commit()
        conn.close()
        
        self.save_index()
        logger.info(f"Added {len(chunks)} embeddings")
    
    def search(self, query: str, top_k: int = 5, 
              chunk_type: Optional[str] = None) -> List[Dict]:
        """
        Search for similar chunks using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of results to return
            chunk_type: Optional filter for 'macro' or 'micro' chunks
        
        Returns:
            List of similar chunks with scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self.encode_text(query).reshape(1, -1)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))
        
        # Get chunk details from database
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for unfilled slots
                continue
                
            chunk_id = self.chunk_map.get(idx)
            if not chunk_id:
                continue
            
            # Get chunk details
            query_sql = """
                SELECT c.chunk_id, c.text, c.chunk_type, c.source_file, c.parent_id
                FROM chunks c
                WHERE c.chunk_id = ?
            """
            
            if chunk_type:
                query_sql += f" AND c.chunk_type = '{chunk_type}'"
            
            c.execute(query_sql, (chunk_id,))
            row = c.fetchone()
            
            if row:
                results.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "chunk_type": row[2],
                    "source_file": row[3],
                    "parent_id": row[4],
                    "similarity_score": float(1 / (1 + dist)),  # Convert distance to similarity
                    "distance": float(dist)
                })
        
        conn.close()
        
        # Sort by similarity and limit to top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """
        Two-stage hybrid search: macro then micro
        
        Args:
            query: Search query
            top_k: Number of results per stage
        
        Returns:
            Dict with 'macro' and 'micro' results
        """
        # Stage 1: Find relevant macro chunks
        macro_results = self.search(query, top_k=top_k, chunk_type='macro')
        
        # Stage 2: Find micro chunks within those macro chunks
        micro_results = []
        if macro_results:
            parent_ids = [r['chunk_id'] for r in macro_results]
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            placeholders = ','.join('?' * len(parent_ids))
            c.execute(f"""
                SELECT chunk_id, text, parent_id 
                FROM chunks 
                WHERE parent_id IN ({placeholders}) 
                AND chunk_type = 'micro'
            """, parent_ids)
            
            micro_chunks = []
            for row in c.fetchall():
                micro_chunks.append({
                    'chunk_id': row[0],
                    'text': row[1],
                    'parent_id': row[2]
                })
            
            conn.close()
            
            if micro_chunks:
                # Encode and score micro chunks
                micro_texts = [c['text'] for c in micro_chunks]
                query_emb = self.encode_text(query)
                micro_embs = self.encode_batch(micro_texts)
                
                # Calculate similarities
                similarities = np.dot(micro_embs, query_emb)
                
                for chunk, sim in zip(micro_chunks, similarities):
                    chunk['similarity_score'] = float(sim)
                
                # Sort and limit
                micro_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
                micro_results = micro_chunks[:top_k]
        
        return {
            "macro": macro_results,
            "micro": micro_results
        }

# Initialize service
embedding_service = EmbeddingService()

# Flask endpoints
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": embedding_service.model_name,
        "index_size": embedding_service.index.ntotal if embedding_service.index else 0
    })

@app.route("/embed", methods=["POST"])
def embed_text():
    """Embed a single text"""
    data = request.get_json()
    text = data.get("text")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    embedding = embedding_service.encode_text(text)
    
    return jsonify({
        "text": text,
        "embedding": embedding.tolist(),
        "dimension": len(embedding)
    })

@app.route("/embed_batch", methods=["POST"])
def embed_batch():
    """Embed multiple texts"""
    data = request.get_json()
    texts = data.get("texts", [])
    
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    
    embeddings = embedding_service.encode_batch(texts)
    
    return jsonify({
        "texts": texts,
        "embeddings": embeddings.tolist(),
        "count": len(texts)
    })

@app.route("/add_chunks", methods=["POST"])
def add_chunks():
    """Add chunks with embeddings to the index"""
    data = request.get_json()
    chunks = data.get("chunks", [])
    
    if not chunks:
        return jsonify({"error": "No chunks provided"}), 400
    
    embedding_service.add_embeddings(chunks)
    
    return jsonify({
        "status": "success",
        "chunks_added": len(chunks),
        "total_indexed": embedding_service.index.ntotal
    })

@app.route("/search", methods=["POST"])
def search():
    """Semantic search for similar chunks"""
    data = request.get_json()
    query = data.get("query")
    top_k = data.get("top_k", 5)
    chunk_type = data.get("chunk_type")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    results = embedding_service.search(query, top_k, chunk_type)
    
    return jsonify({
        "query": query,
        "results": results,
        "count": len(results)
    })

@app.route("/hybrid_search", methods=["POST"])
def hybrid_search():
    """Two-stage hybrid search"""
    data = request.get_json()
    query = data.get("query")
    top_k = data.get("top_k", 5)
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    results = embedding_service.hybrid_search(query, top_k)
    
    return jsonify({
        "query": query,
        "macro_results": results["macro"],
        "micro_results": results["micro"]
    })

@app.route("/rebuild_index", methods=["POST"])
def rebuild_index():
    """Rebuild the FAISS index from database"""
    embedding_service.rebuild_index()
    
    return jsonify({
        "status": "success",
        "index_size": embedding_service.index.ntotal
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get index statistics"""
    conn = sqlite3.connect(embedding_service.db_path)
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM embeddings")
    total_embeddings = c.fetchone()[0]
    
    c.execute("SELECT COUNT(DISTINCT chunk_id) FROM chunks WHERE chunk_type = 'macro'")
    macro_chunks = c.fetchone()[0]
    
    c.execute("SELECT COUNT(DISTINCT chunk_id) FROM chunks WHERE chunk_type = 'micro'")
    micro_chunks = c.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        "total_embeddings": total_embeddings,
        "macro_chunks": macro_chunks,
        "micro_chunks": micro_chunks,
        "index_size": embedding_service.index.ntotal if embedding_service.index else 0,
        "model": embedding_service.model_name,
        "dimension": embedding_service.dimension
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)