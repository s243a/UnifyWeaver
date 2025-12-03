import sqlite3
import struct
import json
import numpy as np
import sys

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

    def text_search(self, query, top_k=10):
        # Simple keyword search using LIKE
        # Searches in the raw JSON 'data' column
        search_term = f"%{query}%"
        cursor = self.conn.execute("SELECT id, data FROM objects WHERE data LIKE ? LIMIT ?", (search_term, top_k))
        
        results = []
        for obj_id, data_str in cursor:
            # Score is dummy 1.0 for text match
            results.append({'id': obj_id, 'score': 1.0, 'data': data_str})
            
            # Lazy Embedding: If we found it by text, ensure it has an embedding for next time
            if self.embedder:
                # Check if embedding exists
                has_emb = self.conn.execute("SELECT 1 FROM embeddings WHERE id = ?", (obj_id,)).fetchone()
                if not has_emb:
                    try:
                        data = json.loads(data_str)
                        text = data.get('title') or data.get('text') or ""
                        if text:
                            vec = self.embedder.get_embedding(text)
                            # Upsert embedding (Pack floats into bytes)
                            blob = struct.pack(f'<{len(vec)}f', *vec)
                            self.conn.execute("INSERT OR REPLACE INTO embeddings (id, vector) VALUES (?, ?)", (obj_id, blob))
                            self.conn.commit()
                    except Exception:
                        # Silent fail for lazy embedding to not break search
                        pass
            
        return results

    def graph_search(self, query, top_k=5, hops=1, mode='vector'):
        # 1. Initial Search (Anchor)
        if mode == 'text':
            seeds = self.text_search(query, top_k=top_k)
        else:
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
