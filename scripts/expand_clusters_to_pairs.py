#!/usr/bin/env python3
"""
Expand cluster-based Q-A data to 1-to-1 pairs for smoothing techniques.

Supports two input formats:

1. Legacy JSON (playbooks/lda-training-data/raw/):
   qa_pairs_v1.json -> expanded/qa_pairs_v1_expanded.json

2. JSONL (training-data/):
   book-01-foundations/*.jsonl -> expanded/book-01-foundations/*.jsonl

The expanded format preserves cluster_id for grouping/analysis while
enabling per-pair smoothing operations.
"""

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Iterator


def expand_jsonl_cluster(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand a JSONL cluster record to individual Q-A pairs.

    Supports two JSONL formats:

    Format 1 (cluster with multiple questions):
    {
        "cluster_id": "prolog-facts-rules",
        "source_files": ["..."],
        "questions": ["Q1", "Q2", ...],
        "answer": {"text": "...", ...}
    }

    Format 2 (single Q-A pair):
    {
        "cluster_id": "ps-cmdlet-001",
        "question": "How do I...",
        "answer": "Use ...",
        "source_file": "..."
    }
    """
    pairs = []
    cluster_id = record.get("cluster_id", "unknown")

    # Handle answer (can be string or dict with "text" key)
    answer_obj = record.get("answer", "")
    if isinstance(answer_obj, dict):
        answer_text = answer_obj.get("text", "")
    else:
        answer_text = str(answer_obj)

    # Handle source files (can be list or single string)
    source_files = record.get("source_files", [])
    source_file = record.get("source_file", "")
    if source_file and not source_files:
        source_files = [source_file]

    topics = record.get("topics", [])

    # Handle questions (can be array "questions" or single "question")
    questions = record.get("questions", [])
    single_question = record.get("question", "")

    if single_question and not questions:
        # Format 2: single Q-A pair (already 1-to-1)
        pairs.append({
            "pair_id": f"{cluster_id}_p0",
            "cluster_id": cluster_id,
            "question": single_question,
            "question_type": "medium",
            "answer": answer_text,
            "answer_variant": "default",
            "answer_source": source_files[0] if source_files else "",
            "topics": topics
        })
    else:
        # Format 1: cluster with multiple questions
        for idx, question in enumerate(questions):
            pair_id = f"{cluster_id}_p{idx}"
            pairs.append({
                "pair_id": pair_id,
                "cluster_id": cluster_id,
                "question": question,
                "question_type": "medium",
                "answer": answer_text,
                "answer_variant": "default",
                "answer_source": source_files[0] if source_files else "",
                "topics": topics
            })

    return pairs


def expand_jsonl_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """Expand a JSONL file to 1-to-1 pairs (still JSONL format)."""
    all_pairs = []
    cluster_count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pairs = expand_jsonl_cluster(record)
                all_pairs.extend(pairs)
                cluster_count += 1
            except json.JSONDecodeError:
                continue

    # Write as JSONL (one pair per line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + '\n')

    return {"num_clusters": cluster_count, "num_pairs": len(all_pairs)}


def expand_training_data_dir(input_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """Expand all JSONL files in training-data/ structure."""
    results = []

    # Find all JSONL files
    for jsonl_file in input_dir.rglob("*.jsonl"):
        # Skip if already in expanded/
        if "expanded" in str(jsonl_file):
            continue

        # Compute relative path and output location
        rel_path = jsonl_file.relative_to(input_dir)
        output_path = output_dir / rel_path

        print(f"Expanding {rel_path}")
        stats = expand_jsonl_file(jsonl_file, output_path)
        results.append({
            "source": str(rel_path),
            "output": str(output_path.relative_to(output_dir)),
            **stats
        })
        print(f"  {stats['num_clusters']} clusters -> {stats['num_pairs']} pairs")

    return results


def expand_cluster_to_pairs(cluster: Dict[str, Any], model_variant: str = "default") -> List[Dict[str, Any]]:
    """
    Expand a single cluster to individual Q-A pairs.

    Args:
        cluster: Cluster dict with id, answers, queries
        model_variant: Which answer variant to use (default, all-MiniLM-L6-v2, etc.)

    Returns:
        List of individual Q-A pair dicts
    """
    pairs = []
    cluster_id = cluster["id"]
    answer_source = cluster.get("answer_source", "")

    # Get answer text for this variant
    answers = cluster.get("answers", {})
    if isinstance(answers, str):
        # Simple string answer
        answer_text = answers
    else:
        # Dict with variants
        answer_text = answers.get(model_variant, answers.get("default", ""))

    # Expand each query
    queries = cluster.get("queries", {})
    pair_index = 0

    for length_type, query_list in queries.items():
        if not isinstance(query_list, list):
            continue

        for query in query_list:
            pair_id = f"{cluster_id}_p{pair_index}"
            pairs.append({
                "pair_id": pair_id,
                "cluster_id": cluster_id,
                "question": query,
                "question_type": length_type,
                "answer": answer_text,
                "answer_variant": model_variant,
                "answer_source": answer_source
            })
            pair_index += 1

    return pairs


def expand_file(input_path: Path, output_path: Path, model_variant: str = "default") -> dict:
    """
    Expand a cluster-based JSON file to 1-to-1 pairs.

    Args:
        input_path: Path to cluster-based JSON
        output_path: Path for expanded JSON output
        model_variant: Answer variant to use

    Returns:
        Stats dict with counts
    """
    with open(input_path) as f:
        data = json.load(f)

    clusters = data.get("clusters", [])
    all_pairs = []

    for cluster in clusters:
        pairs = expand_cluster_to_pairs(cluster, model_variant)
        all_pairs.extend(pairs)

    # Build expanded output
    expanded = {
        "version": data.get("version", "1.0") + "-expanded",
        "description": f"Expanded from {input_path.name} - 1-to-1 Q-A pairs",
        "source_file": str(input_path.name),
        "model_variant": model_variant,
        "expanded_at": datetime.now().isoformat(),
        "stats": {
            "num_clusters": len(clusters),
            "num_pairs": len(all_pairs)
        },
        "pairs": all_pairs
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(expanded, f, indent=2)

    return expanded["stats"]


def expand_all_files(raw_dir: Path, expanded_dir: Path, model_variant: str = "default"):
    """Expand all qa_pairs_*.json files in raw_dir."""

    results = []
    for input_path in sorted(raw_dir.glob("qa_pairs_*.json")):
        # Generate output filename
        stem = input_path.stem
        output_name = f"{stem}_expanded.json"
        output_path = expanded_dir / output_name

        print(f"Expanding {input_path.name} -> {output_name}")
        stats = expand_file(input_path, output_path, model_variant)
        results.append({
            "source": input_path.name,
            "output": output_name,
            **stats
        })
        print(f"  {stats['num_clusters']} clusters -> {stats['num_pairs']} pairs")

    return results


# --- Database Integration ---

def ensure_qa_pairs_table(db_path: Path):
    """Create qa_pairs table if it doesn't exist."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_pairs (
            pair_id INTEGER PRIMARY KEY,
            cluster_id INTEGER REFERENCES qa_clusters(cluster_id),
            question_id INTEGER REFERENCES questions(question_id),
            answer_id INTEGER REFERENCES answers(answer_id),
            pair_source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(question_id, answer_id)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_qa_pairs_cluster ON qa_pairs(cluster_id)
    """)

    conn.commit()
    conn.close()
    print(f"Ensured qa_pairs table exists in {db_path}")


def import_expanded_to_db(
    expanded_path: Path,
    db_path: Path,
    replace_cluster: bool = True
) -> dict:
    """
    Import expanded pairs to database.

    Args:
        expanded_path: Path to expanded JSON file
        db_path: Path to SQLite database
        replace_cluster: If True, delete existing data for clusters being imported

    Returns:
        Stats dict
    """
    ensure_qa_pairs_table(db_path)

    with open(expanded_path) as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    if not pairs:
        return {"imported": 0, "skipped": 0}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Track cluster IDs we're importing
    cluster_ids_in_file = set(p["cluster_id"] for p in pairs)

    # Map cluster names to DB IDs (create if needed)
    cluster_id_map = {}
    for cluster_name in cluster_ids_in_file:
        cursor.execute("SELECT cluster_id FROM qa_clusters WHERE name = ?", (cluster_name,))
        row = cursor.fetchone()
        if row:
            cluster_id_map[cluster_name] = row[0]
        else:
            cursor.execute(
                "INSERT INTO qa_clusters (name, description) VALUES (?, ?)",
                (cluster_name, f"Imported from {expanded_path.name}")
            )
            cluster_id_map[cluster_name] = cursor.lastrowid

    if replace_cluster:
        # Delete existing pairs for these clusters
        for db_cluster_id in cluster_id_map.values():
            cursor.execute("DELETE FROM qa_pairs WHERE cluster_id = ?", (db_cluster_id,))
            # Also delete orphaned questions/answers from junction tables
            cursor.execute("DELETE FROM cluster_questions WHERE cluster_id = ?", (db_cluster_id,))
            cursor.execute("DELETE FROM cluster_answers WHERE cluster_id = ?", (db_cluster_id,))

    imported = 0
    skipped = 0
    source_file = data.get("source_file", expanded_path.name)

    for pair in pairs:
        cluster_name = pair["cluster_id"]
        db_cluster_id = cluster_id_map[cluster_name]

        # Insert or get question
        cursor.execute(
            "SELECT question_id FROM questions WHERE text = ?",
            (pair["question"],)
        )
        row = cursor.fetchone()
        if row:
            question_id = row[0]
        else:
            cursor.execute(
                "INSERT INTO questions (text, length_type) VALUES (?, ?)",
                (pair["question"], pair.get("question_type", "medium"))
            )
            question_id = cursor.lastrowid

        # Insert or get answer
        cursor.execute(
            "SELECT answer_id FROM answers WHERE text = ? AND text_variant = ?",
            (pair["answer"], pair.get("answer_variant", "default"))
        )
        row = cursor.fetchone()
        if row:
            answer_id = row[0]
        else:
            cursor.execute(
                "INSERT INTO answers (source_file, text, text_variant) VALUES (?, ?, ?)",
                (pair.get("answer_source", ""), pair["answer"], pair.get("answer_variant", "default"))
            )
            answer_id = cursor.lastrowid

        # Insert pair (with conflict handling)
        try:
            cursor.execute(
                """INSERT INTO qa_pairs (cluster_id, question_id, answer_id, pair_source)
                   VALUES (?, ?, ?, ?)""",
                (db_cluster_id, question_id, answer_id, source_file)
            )
            imported += 1
        except sqlite3.IntegrityError:
            skipped += 1  # Duplicate pair

    conn.commit()
    conn.close()

    return {"imported": imported, "skipped": skipped, "clusters": len(cluster_id_map)}


def main():
    parser = argparse.ArgumentParser(
        description="Expand cluster-based Q-A data to 1-to-1 pairs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory (training-data/ for JSONL, playbooks/lda-training-data/raw/ for legacy JSON)"
    )
    parser.add_argument(
        "--expanded-dir",
        type=Path,
        default=None,
        help="Output directory for expanded files"
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "legacy", "auto"],
        default="auto",
        help="Input format: jsonl (training-data), legacy (playbooks), or auto-detect"
    )
    parser.add_argument(
        "--model-variant",
        default="default",
        help="Answer variant to use for legacy format"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite database path for import (optional)"
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Don't replace existing cluster data in DB"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Process single file instead of all"
    )
    # Legacy compatibility
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="[Legacy] Alias for --input-dir with legacy format"
    )

    args = parser.parse_args()

    # Determine input format and directories
    if args.raw_dir:
        # Legacy mode
        input_dir = args.raw_dir
        expanded_dir = args.expanded_dir or Path("playbooks/lda-training-data/expanded")
        use_jsonl = False
    elif args.input_dir:
        input_dir = args.input_dir
        expanded_dir = args.expanded_dir or (input_dir / "expanded")
        # Auto-detect format
        if args.format == "auto":
            # Check if directory has JSONL files
            use_jsonl = any(input_dir.rglob("*.jsonl"))
        else:
            use_jsonl = args.format == "jsonl"
    else:
        # Default: try training-data/ first, fall back to legacy
        if Path("training-data").exists() and any(Path("training-data").rglob("*.jsonl")):
            input_dir = Path("training-data")
            expanded_dir = args.expanded_dir or Path("training-data/expanded")
            use_jsonl = True
        else:
            input_dir = Path("playbooks/lda-training-data/raw")
            expanded_dir = args.expanded_dir or Path("playbooks/lda-training-data/expanded")
            use_jsonl = False

    if args.file:
        # Process single file
        if args.file.suffix == ".jsonl":
            output_path = expanded_dir / args.file.name
            stats = expand_jsonl_file(args.file, output_path)
        else:
            output_name = f"{args.file.stem}_expanded.json"
            output_path = expanded_dir / output_name
            stats = expand_file(args.file, output_path, args.model_variant)
        print(f"Expanded: {stats['num_clusters']} clusters -> {stats['num_pairs']} pairs")

        if args.db_path:
            db_stats = import_expanded_to_db(
                output_path,
                args.db_path,
                replace_cluster=not args.no_replace
            )
            print(f"DB import: {db_stats['imported']} imported, {db_stats['skipped']} skipped")
    else:
        # Process all files
        if use_jsonl:
            print(f"Processing JSONL files from {input_dir}")
            results = expand_training_data_dir(input_dir, expanded_dir)
        else:
            print(f"Processing legacy JSON files from {input_dir}")
            results = expand_all_files(input_dir, expanded_dir, args.model_variant)

        total_clusters = sum(r["num_clusters"] for r in results)
        total_pairs = sum(r["num_pairs"] for r in results)
        print(f"\nTotal: {total_clusters} clusters -> {total_pairs} pairs across {len(results)} files")

        if args.db_path:
            print(f"\nImporting to {args.db_path}...")
            # Find all expanded files
            pattern = "*.jsonl" if use_jsonl else "*_expanded.json"
            for expanded_file in sorted(expanded_dir.rglob(pattern)):
                db_stats = import_expanded_to_db(
                    expanded_file,
                    args.db_path,
                    replace_cluster=not args.no_replace
                )
                print(f"  {expanded_file.name}: {db_stats['imported']} imported")


if __name__ == "__main__":
    main()
