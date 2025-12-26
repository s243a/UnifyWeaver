#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Import all training data from training-data/ subfolder into lda.db

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from lda_database import LDAProjectionDB

def iter_jsonl_files(base_dir: Path):
    """Iterate over all JSONL files in the training data directory."""
    for subdir_path in base_dir.iterdir():
        if subdir_path.is_dir() and subdir_path.name.startswith('book-'):
            for jsonl_file in subdir_path.rglob('*.jsonl'):
                yield jsonl_file

def import_jsonl_to_db(db: LDAProjectionDB, jsonl_path: Path):
    """Import a single JSONL file into the database."""
    print(f"Processing: {jsonl_path.relative_to(project_root)}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            cluster_id_name = record.get('cluster_id')
            if not cluster_id_name:
                continue
            
            # answer_source
            source_files = record.get('source_files', [])
            source_file = source_files[0] if source_files else str(jsonl_path.relative_to(project_root))
            
            # Answer
            answer_data = record.get('answer', {})
            if isinstance(answer_data, str):
                answer_text = answer_data
            else:
                answer_text = answer_data.get('text', '')
            
            if not answer_text:
                continue
                
            # Questions
            questions = record.get('questions', [])
            if not questions and 'question' in record:
                questions = [record['question']]
            
            # 1. Add Answer
            aid = db.add_answer(
                source_file=source_file,
                text=answer_text,
                record_id=cluster_id_name,
                text_variant='default'
            )
            
            # 2. Add Questions
            question_ids = []
            for q_text in questions:
                qid = db.add_question(
                    text=q_text,
                    length_type='medium' # Default
                )
                question_ids.append(qid)
            
            # 3. Create Cluster
            db.create_cluster(
                name=cluster_id_name,
                answer_ids=[aid],
                question_ids=question_ids,
                description=f"Imported from {jsonl_path.name}"
            )

def main():
    parser = argparse.ArgumentParser(description="Mass import training data from training-data/ into lda.db")
    parser.add_argument("--db", default="playbooks/lda-training-data/lda.db", help="Path to database")
    parser.add_argument("--input", default="training-data", help="Input training data root")
    
    args = parser.parse_args()
    
    input_root = project_root / args.input
    if not input_root.exists():
        print(f"Error: Input directory not found: {input_root}")
        return
        
    db = LDAProjectionDB(args.db)
    
    try:
        for jsonl_file in iter_jsonl_files(input_root):
            import_jsonl_to_db(db, jsonl_file)
        
        print("\nImport complete.")
        print("Note: You must now run embedding and training to update the search model.")
        print(f"Example: python3 scripts/migrate_to_lda_db.py --db {args.db} --process-pending")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
