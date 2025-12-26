#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Embed all answers and questions in the database that are missing embeddings

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from lda_database import LDAProjectionDB

def embed_missing(db_path, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    
    print(f"Opening database: {db_path}")
    db = LDAProjectionDB(db_path)
    
    print(f"Loading model: {model_name}")
    embedder = SentenceTransformer(model_name)
    dimension = embedder.get_sentence_embedding_dimension()
    
    # Register model
    model_id = db.add_model(
        name=model_name,
        dimension=dimension,
        backend='python',
        max_tokens=256,
        notes='sentence-transformers mass-embedder'
    )
    
    cursor = db.conn.cursor()
    
    # 1. Find answers missing embeddings
    print("Checking for answers missing embeddings...")
    cursor.execute("""
        SELECT answer_id, text FROM answers 
        WHERE answer_id NOT IN (
            SELECT entity_id FROM embeddings 
            WHERE entity_type = 'answer' AND model_id = ?
        )
    """, (model_id,))
    missing_answers = cursor.fetchall()
    
    if missing_answers:
        print(f"Embedding {len(missing_answers)} answers...")
        for i, (aid, text) in enumerate(missing_answers):
            if i % 10 == 0:
                print(f"  [{i}/{len(missing_answers)}]...", end='\r')
            emb = embedder.encode(text, convert_to_numpy=True)
            db.store_embedding(model_id, 'answer', aid, emb)
        print(f"\nSuccessfully embedded {len(missing_answers)} answers.")
    else:
        print("No missing answer embeddings.")
        
    # 2. Find questions missing embeddings
    print("Checking for questions missing embeddings...")
    cursor.execute("""
        SELECT question_id, text FROM questions 
        WHERE question_id NOT IN (
            SELECT entity_id FROM embeddings 
            WHERE entity_type = 'question' AND model_id = ?
        )
    """, (model_id,))
    missing_questions = cursor.fetchall()
    
    if missing_questions:
        print(f"Embedding {len(missing_questions)} questions...")
        for i, (qid, text) in enumerate(missing_questions):
            if i % 50 == 0:
                print(f"  [{i}/{len(missing_questions)}]...", end='\r')
            emb = embedder.encode(text, convert_to_numpy=True)
            db.store_embedding(model_id, 'question', qid, emb)
        print(f"\nSuccessfully embedded {len(missing_questions)} questions.")
    else:
        print("No missing question embeddings.")
        
    db.close()

def main():
    parser = argparse.ArgumentParser(description="Embed missing entries in LDA database")
    parser.add_argument("--db", default="playbooks/lda-training-data/lda.db", help="Path to database")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    
    args = parser.parse_args()
    embed_missing(args.db, args.model)

if __name__ == "__main__":
    main()
