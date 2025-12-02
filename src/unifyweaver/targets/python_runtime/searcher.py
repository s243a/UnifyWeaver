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

    def graph_search(self, query, top_k=5, hops=1):
        # 1. Initial Vector Search
        seeds = self.search(query, top_k=top_k)
        
        # 2. Graph Traversal (1 hop for now)
        ids_to_fetch = set(s['id'] for s in seeds)
        
        # Map seed_id -> {parents: [], children: []}
        graph_context = {s['id']: {'item': s, 'parents': [], 'children': []} for s in seeds}
        
        for seed in seeds:
            sid = seed['id']
            
            # Find children (incoming links: who points to me?)
            # Crawler stores (Child, Parent). So children are source_id where target_id=sid.
            cursor = self.conn.execute("SELECT source_id FROM links WHERE target_id = ?", (sid,))
            for (kid_id,) in cursor:
                graph_context[sid]['children'].append(kid_id)
                ids_to_fetch.add(kid_id)
                
            # Find parents (outgoing links: who do I point to?)
            # Crawler stores (Child, Parent). So parents are target_id where source_id=sid.
            cursor = self.conn.execute("SELECT target_id FROM links WHERE source_id = ?", (sid,))
            for (pid_id,) in cursor:
                graph_context[sid]['parents'].append(pid_id)
                ids_to_fetch.add(pid_id)

        # 3. Fetch Content
        missing_ids = ids_to_fetch - set(s['id'] for s in seeds)
        neighbor_data = {}
        
        valid_ids = [mid for mid in missing_ids if mid]
        if valid_ids:
            placeholders = ','.join('?' for _ in valid_ids)
            cursor = self.conn.execute(f"SELECT id, data FROM objects WHERE id IN ({placeholders})", valid_ids)
            for oid, data in cursor:
                neighbor_data[oid] = data
                
        # 4. Assemble Result
        final_results = []
        for seed in seeds:
            sid = seed['id']
            entry = graph_context[sid]
            
            # Hydrate neighbors
            children_objs = [neighbor_data.get(kid_id) or next((s['data'] for s in seeds if s['id'] == kid_id), "{}") for kid_id in entry['children']]
            parent_objs = [neighbor_data.get(pid) or next((s['data'] for s in seeds if s['id'] == pid), "{}") for pid in entry['parents']]
            
            final_results.append({
                'focus': seed,
                'context': {
                    'children': children_objs,
                    'parents': parent_objs
                }
            })
            
        return final_results

    def _cosine_similarity(self, a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def close(self):
        self.conn.close()
