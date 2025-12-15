#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path

# Add paths for UnifyWeaver modules
script_dir = Path(__file__).resolve().parent
# scripts/skills/lookup_example.py -> scripts/skills -> scripts -> root
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from lda_database import LDAProjectionDB
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed.")
    sys.exit(1)

def lookup_example(query, top_k=3, mh_projection_id=1):
    """
    Look up examples using LDA multi-head semantic search.
    Uses softmax routing over cluster centroids.
    """
    db_path = str(project_root / "playbooks" / "lda-training-data" / "lda.db")
    model_name = "all-MiniLM-L6-v2"

    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    try:
        with LDAProjectionDB(db_path) as db:
            embedder = SentenceTransformer(model_name)
            
            # Embed query first as multi_head_search takes embedding
            query_emb = embedder.encode(query, convert_to_numpy=True)
            
            results = db.multi_head_search(
                query_embedding=query_emb,
                mh_projection_id=mh_projection_id,
                top_k=top_k,
                log=True,
                query_text=query
            )
            
            if results:
                print(f"Found {len(results)} relevant examples for: '{query}'\n")
                for i, r in enumerate(results):
                    score = r['score']
                    record_id = r['record_id']
                    # Clean up text for display
                    text = r['text'].strip().replace('\n', ' ')
                    if len(text) > 150:
                        text = text[:147] + "..."
                    
                    print(f"{i+1}. [{score:.3f}] {record_id}")
                    print(f"   {text}")
                    if r.get('source_file'):
                        print(f"   Source: {r['source_file']}")
                    print("")
            else:
                print(f"No examples found for: '{query}'")

    except Exception as e:
        print(f"An error occurred during search: {e}")

def main():
    parser = argparse.ArgumentParser(description="Look up UnifyWeaver examples using multi-head semantic search.")
    parser.add_argument("query", help="The search query (e.g., 'how to use csv source')")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--mh-proj", type=int, default=2, help="Multi-Head Projection ID to use (default: 2)")
    
    args = parser.parse_args()
    
    lookup_example(args.query, args.top_k, args.mh_proj)

if __name__ == "__main__":
    main()
