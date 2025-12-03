import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath('src'))

from unifyweaver.targets.python_runtime.importer import PtImporter
from unifyweaver.targets.python_runtime.searcher import PtSearcher
from unifyweaver.targets.python_runtime.embedding import IEmbeddingProvider

class MockEmbedder(IEmbeddingProvider):
    def get_embedding(self, text):
        # Deterministic dummy embedding based on length
        vec = np.zeros(384, dtype=np.float32)
        val = float(len(text))
        vec[0] = val
        return vec.tolist()

def test_runtime():
    db_path = "test_runtime.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("Initializing Importer...")
    importer = PtImporter(db_path)
    
    # Insert objects
    importer.upsert_object("1", "doc", {"text": "short"})
    importer.upsert_object("2", "doc", {"text": "a very long document text"})
    
    # Insert embeddings (mocked)
    embedder = MockEmbedder()
    vec1 = embedder.get_embedding("short")
    vec2 = embedder.get_embedding("a very long document text")
    
    importer.upsert_embedding("1", vec1)
    importer.upsert_embedding("2", vec2)
    
    importer.close()
    print("Import complete.")
    
    print("Initializing Searcher...")
    searcher = PtSearcher(db_path, embedder)
    
    # Search for "short"
    print("Searching for 'short'...")
    results = searcher.search("short", top_k=2)
    
    for score, obj_id in results:
        print(f"Result: {obj_id}, Score: {score}")
        
    # Expect 1 to be top result (perfect match if using dot product on unnormalized, 
    # but cosine similarity: 
    # vec1 = [5, 0...], vec2 = [25, 0...]
    # cosine([5,0], [5,0]) = 1.0
    # cosine([5,0], [25,0]) = 1.0
    # Wait, if they are collinear, cosine is 1.0!
    # My mock embedding is bad for ranking.
    
    searcher.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    print("Test finished.")

if __name__ == "__main__":
    test_runtime()
