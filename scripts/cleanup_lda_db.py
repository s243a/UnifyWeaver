#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src" / "unifyweaver" / "targets" / "python_runtime"))

from lda_database import LDAProjectionDB

def cleanup_db(db_path, dry_run=True):
    print(f"Opening database: {db_path}")
    if not Path(db_path).exists():
        print("Error: Database file not found.")
        return

    db = LDAProjectionDB(db_path)
    cursor = db.conn.cursor()

    # 1. Identify clusters without embeddings
    # A valid cluster should have answers with embeddings in the 'embeddings' table.
    # We'll check for clusters where none of their answers have embeddings.
    
    print("Scanning for broken clusters (no embeddings)...")
    
    # Get all clusters
    cursor.execute("SELECT cluster_id, name FROM qa_clusters")
    all_clusters = cursor.fetchall()
    
    broken_clusters = []
    
    for cluster in all_clusters:
        cid = cluster['cluster_id']
        cname = cluster['name']
        
        # Check if any answer in this cluster has an embedding
        # We join cluster_answers -> embeddings (via answer_id = entity_id and entity_type='answer')
        cursor.execute("""
            SELECT COUNT(*) 
            FROM cluster_answers ca
            JOIN embeddings e ON ca.answer_id = e.entity_id
            WHERE ca.cluster_id = ? AND e.entity_type = 'answer'
        """, (cid,))
        
        count = cursor.fetchone()[0]
        
        if count == 0:
            broken_clusters.append((cid, cname))

    print(f"Found {len(broken_clusters)} broken clusters out of {len(all_clusters)} total.")

    if not broken_clusters:
        print("Database is clean.")
        return

    print("\nBroken Clusters:")
    for cid, cname in broken_clusters:
        print(f"  [ID: {cid}] {cname}")

    if dry_run:
        print("\n[Dry Run] No changes made. Use --force to delete.")
        return

    print("\nDeleting broken clusters...")
    
    deleted_count = 0
    for cid, cname in broken_clusters:
        try:
            # Delete from junction tables first
            cursor.execute("DELETE FROM cluster_answers WHERE cluster_id = ?", (cid,))
            cursor.execute("DELETE FROM cluster_questions WHERE cluster_id = ?", (cid,))
            cursor.execute("DELETE FROM projection_clusters WHERE cluster_id = ?", (cid,))
            cursor.execute("DELETE FROM cluster_heads WHERE cluster_id = ?", (cid,))
            
            # Delete the cluster itself
            cursor.execute("DELETE FROM qa_clusters WHERE cluster_id = ?", (cid,))
            
            deleted_count += 1
            print(f"  Deleted cluster {cid} ({cname})")
        except Exception as e:
            print(f"  Error deleting cluster {cid}: {e}")

    db.conn.commit()
    print(f"\nSuccessfully deleted {deleted_count} clusters.")
    
    # Optional: Vacuum to reclaim space
    print("Vacuuming database...")
    cursor.execute("VACUUM")
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Cleanup LDA database by removing broken clusters.")
    parser.add_argument("--db", default="playbooks/lda-training-data/lda.db", help="Path to database")
    parser.add_argument("--force", action="store_true", help="Execute deletion (default is dry-run)")
    
    args = parser.parse_args()
    
    # Resolve db path relative to project root if not absolute
    if not Path(args.db).is_absolute():
        db_path = str(project_root / args.db)
    else:
        db_path = args.db

    cleanup_db(db_path, dry_run=not args.force)

if __name__ == "__main__":
    main()
