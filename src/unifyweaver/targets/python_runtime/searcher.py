import sqlite3
import struct
import numpy as np

class PtSearcher:
    def __init__(self, db_path, embedder):
        self.conn = sqlite3.connect(db_path)
        self.embedder = embedder

    def search(self, query, top_k=10):
        q_vec = self.embedder.get_embedding(query)
        
        cursor = self.conn.execute("SELECT id, vector FROM embeddings")
        results = []
        for obj_id, blob in cursor:
            # Assuming float32
            count = len(blob) // 4
            vec = np.array(struct.unpack(f'<{count}f', blob))
            
            score = self._cosine_similarity(q_vec, vec)
            results.append((score, obj_id))
        
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Enrich with content
        enriched = []
        for score, obj_id in results[:top_k]:
            row = self.conn.execute("SELECT data FROM objects WHERE id = ?", (obj_id,)).fetchone()
            data = row[0] if row else "{}"
            enriched.append({'id': obj_id, 'score': float(score), 'data': data})
            
        return enriched

    def _cosine_similarity(self, a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def close(self):
        self.conn.close()
