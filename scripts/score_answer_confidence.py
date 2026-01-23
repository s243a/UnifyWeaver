#!/usr/bin/env python3
"""
Score Answer Confidence for Q/A Pairs

Reviews tailored answers against original skill documents and assigns
confidence scores. Low-confidence answers are queued for deeper review.

Uses SQLite with B-tree index for efficient priority queue operations.

Usage:
    # Score all tailored answers
    python3 scripts/score_answer_confidence.py --input datasets/skills_qa/tailored/skills_qa.jsonl

    # Get lowest confidence items for review
    python3 scripts/score_answer_confidence.py --get-queue --limit 20

    # Export low-confidence items for deep review
    python3 scripts/score_answer_confidence.py --export-for-review --threshold 0.7
"""

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

project_root = Path(__file__).parent.parent

# Default paths
DEFAULT_DB = project_root / "datasets" / "skills_qa" / "confidence_queue.db"
SKILLS_DIR = project_root / "skills"


SCORING_PROMPT = '''You are reviewing a Q/A pair for accuracy and completeness.

ORIGINAL SKILL DOCUMENT:
{skill_content}

QUESTION: {question}

TAILORED ANSWER:
{answer}

ORIGINAL BASE ANSWER:
{original_answer}

Review the tailored answer against the skill document. Score confidence from 0.0 to 1.0:
- 1.0: Answer is accurate, complete, and well-supported by the skill document
- 0.8-0.9: Minor rephrasing issues but factually correct
- 0.6-0.7: Missing some context or slight inaccuracies
- 0.4-0.5: Significant gaps or potential hallucinations
- 0.0-0.3: Major errors or unsupported claims

Output ONLY a JSON object (no markdown, no explanation):
{{"confidence": 0.85, "issues": "brief note if any issues, or null"}}'''


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with review queue table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS review_queue (
            pair_id TEXT PRIMARY KEY,
            cluster_id TEXT,
            question TEXT,
            answer TEXT,
            original_answer TEXT,
            confidence REAL,
            issues TEXT,
            reviewed BOOLEAN DEFAULT FALSE,
            deep_reviewed BOOLEAN DEFAULT FALSE,
            skill_file TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # B-tree index on confidence for efficient priority queue
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_confidence
        ON review_queue(confidence) WHERE NOT reviewed
    ''')
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_deep_review
        ON review_queue(confidence) WHERE NOT deep_reviewed
    ''')
    conn.commit()
    return conn


def load_skill_content(skill_file: str) -> Optional[str]:
    """Load skill document content."""
    skill_path = SKILLS_DIR / skill_file
    if skill_path.exists():
        return skill_path.read_text(encoding='utf-8')

    # Try without .md extension
    if not skill_file.endswith('.md'):
        skill_path = SKILLS_DIR / f"{skill_file}.md"
        if skill_path.exists():
            return skill_path.read_text(encoding='utf-8')

    return None


def call_llm_for_score(
    question: str,
    answer: str,
    original_answer: str,
    skill_content: str,
    provider: str = "claude",
    model: str = "haiku"
) -> Dict[str, Any]:
    """Call LLM to score answer confidence."""
    prompt = SCORING_PROMPT.format(
        skill_content=skill_content[:4000],  # Truncate long docs
        question=question,
        answer=answer[:2000],
        original_answer=original_answer[:2000]
    )

    try:
        if provider == "claude":
            result = subprocess.run(
                ["claude", "-p", "--model", model, prompt],
                capture_output=True, text=True, timeout=60
            )
        else:  # gemini
            result = subprocess.run(
                ["gemini", "-p", prompt, "-m", model, "--output-format", "text"],
                capture_output=True, text=True, timeout=120
            )

        if result.returncode == 0:
            output = result.stdout.strip()
            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```" in output:
                output = output.split("```")[1]
                if output.startswith("json"):
                    output = output[4:]
            return json.loads(output)
        else:
            return {"confidence": 0.5, "issues": f"LLM error: {result.stderr[:100]}"}

    except json.JSONDecodeError as e:
        return {"confidence": 0.5, "issues": f"JSON parse error: {str(e)[:50]}"}
    except subprocess.TimeoutExpired:
        return {"confidence": 0.5, "issues": "Timeout"}
    except Exception as e:
        return {"confidence": 0.5, "issues": str(e)[:100]}


def score_pair(
    pair: Dict[str, Any],
    conn: sqlite3.Connection,
    provider: str = "claude",
    model: str = "haiku"
) -> Dict[str, Any]:
    """Score a single Q/A pair and insert into queue."""
    pair_id = pair.get("pair_id", "")

    # Check if already scored
    existing = conn.execute(
        "SELECT confidence FROM review_queue WHERE pair_id = ?",
        (pair_id,)
    ).fetchone()

    if existing:
        return {"pair_id": pair_id, "confidence": existing[0], "skipped": True}

    # Get skill content
    skill_files = pair.get("related_skills", [])
    skill_content = ""
    skill_file = ""
    for sf in skill_files:
        content = load_skill_content(sf)
        if content:
            skill_content = content
            skill_file = sf
            break

    if not skill_content:
        # No skill doc - assign medium confidence
        score_result = {"confidence": 0.6, "issues": "No skill document found for verification"}
    else:
        # Call LLM for scoring
        score_result = call_llm_for_score(
            question=pair.get("question", ""),
            answer=pair.get("answer", ""),
            original_answer=pair.get("original_answer", pair.get("answer", "")),
            skill_content=skill_content,
            provider=provider,
            model=model
        )

    # Insert into queue
    conn.execute('''
        INSERT OR REPLACE INTO review_queue
        (pair_id, cluster_id, question, answer, original_answer, confidence, issues, skill_file)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        pair_id,
        pair.get("cluster_id", ""),
        pair.get("question", ""),
        pair.get("answer", ""),
        pair.get("original_answer", ""),
        score_result.get("confidence", 0.5),
        score_result.get("issues"),
        skill_file
    ))
    conn.commit()

    return {
        "pair_id": pair_id,
        "confidence": score_result.get("confidence", 0.5),
        "issues": score_result.get("issues"),
        "skipped": False
    }


def process_file(
    input_path: Path,
    conn: sqlite3.Connection,
    provider: str = "claude",
    model: str = "haiku",
    batch_size: int = 10,
    delay: float = 0.3,
    limit: Optional[int] = None
) -> Dict[str, int]:
    """Process a JSONL file and score all pairs."""
    pairs = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    if limit:
        pairs = pairs[:limit]

    stats = {"scored": 0, "skipped": 0, "total": len(pairs)}

    for i, pair in enumerate(pairs):
        result = score_pair(pair, conn, provider, model)

        if result.get("skipped"):
            stats["skipped"] += 1
        else:
            stats["scored"] += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}] Confidence: {result['confidence']:.2f} - {pair.get('question', '')[:40]}...")

        if delay > 0 and not result.get("skipped"):
            time.sleep(delay)

    return stats


def get_queue(conn: sqlite3.Connection, limit: int = 20, threshold: float = 1.0) -> List[Dict]:
    """Get lowest confidence items from queue."""
    cursor = conn.execute('''
        SELECT pair_id, cluster_id, question, answer, confidence, issues, skill_file
        FROM review_queue
        WHERE NOT deep_reviewed AND confidence <= ?
        ORDER BY confidence ASC
        LIMIT ?
    ''', (threshold, limit))

    results = []
    for row in cursor:
        results.append({
            "pair_id": row[0],
            "cluster_id": row[1],
            "question": row[2],
            "answer": row[3],
            "confidence": row[4],
            "issues": row[5],
            "skill_file": row[6]
        })
    return results


def export_for_review(conn: sqlite3.Connection, threshold: float, output_path: Path):
    """Export low-confidence items for deep review."""
    items = get_queue(conn, limit=10000, threshold=threshold)

    with open(output_path, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')

    print(f"Exported {len(items)} items with confidence <= {threshold} to {output_path}")


def print_stats(conn: sqlite3.Connection):
    """Print queue statistics."""
    stats = conn.execute('''
        SELECT
            COUNT(*) as total,
            AVG(confidence) as avg_conf,
            MIN(confidence) as min_conf,
            MAX(confidence) as max_conf,
            SUM(CASE WHEN confidence < 0.5 THEN 1 ELSE 0 END) as low_conf,
            SUM(CASE WHEN confidence >= 0.8 THEN 1 ELSE 0 END) as high_conf,
            SUM(CASE WHEN deep_reviewed THEN 1 ELSE 0 END) as reviewed
        FROM review_queue
    ''').fetchone()

    print("\n=== Confidence Queue Statistics ===")
    print(f"Total scored: {stats[0]}")
    print(f"Average confidence: {stats[1]:.3f}" if stats[1] else "N/A")
    print(f"Range: {stats[2]:.2f} - {stats[3]:.2f}" if stats[2] else "N/A")
    print(f"Low confidence (<0.5): {stats[4]}")
    print(f"High confidence (>=0.8): {stats[5]}")
    print(f"Deep reviewed: {stats[6]}")


def main():
    parser = argparse.ArgumentParser(description="Score answer confidence for Q/A pairs")
    parser.add_argument("--input", type=Path, help="Input JSONL file with tailored answers")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--provider", default="claude", choices=["claude", "gemini"])
    parser.add_argument("--model", default="haiku", help="Model for scoring")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--limit", type=int, help="Limit number of pairs to process")

    # Queue operations
    parser.add_argument("--get-queue", action="store_true", help="Get lowest confidence items")
    parser.add_argument("--queue-limit", type=int, default=20, help="Number of items to retrieve")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--export-for-review", action="store_true", help="Export low-confidence items")
    parser.add_argument("--output", type=Path, help="Output path for export")
    parser.add_argument("--stats", action="store_true", help="Print queue statistics")

    args = parser.parse_args()

    conn = init_db(args.db)

    if args.stats:
        print_stats(conn)
    elif args.get_queue:
        items = get_queue(conn, args.queue_limit, args.threshold)
        for item in items:
            print(f"\n[{item['confidence']:.2f}] {item['pair_id']}")
            print(f"  Q: {item['question'][:60]}...")
            if item['issues']:
                print(f"  Issues: {item['issues']}")
    elif args.export_for_review:
        output = args.output or (project_root / "datasets" / "skills_qa" / "needs_review.jsonl")
        export_for_review(conn, args.threshold, output)
    elif args.input:
        print(f"Scoring {args.input} with {args.provider}/{args.model}")
        stats = process_file(
            args.input, conn,
            provider=args.provider,
            model=args.model,
            batch_size=args.batch_size,
            delay=args.delay,
            limit=args.limit
        )
        print(f"\nDone: {stats['scored']} scored, {stats['skipped']} skipped")
        print_stats(conn)
    else:
        print_stats(conn)

    conn.close()


if __name__ == "__main__":
    main()
