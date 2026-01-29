#!/usr/bin/env python3
"""
Convert Pearltrees API database to tree data JSONL format.

Reads api_responses.db and builds path hierarchies by traversing
parent-child relationships via contentTree references.

Usage:
    python3 scripts/convert_api_to_tree_data.py \
        --db .local/data/pearltrees_api/api_responses.db \
        --output .local/data/tree_paths.jsonl
"""

import argparse
import json
import sqlite3
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_api_responses(db_path: Path) -> Dict[str, dict]:
    """Load all API responses from database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT tree_id, title, response_json FROM api_responses")

    trees = {}
    for tree_id, title, response_json in cur.fetchall():
        try:
            resp = json.loads(response_json)
            trees[str(tree_id)] = {
                'tree_id': str(tree_id),
                'title': title,
                'response': resp
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for tree {tree_id}: {e}")

    conn.close()
    logger.info(f"Loaded {len(trees)} trees from {db_path}")
    return trees


def build_parent_map(trees: Dict[str, dict]) -> Dict[str, str]:
    """Build child_tree_id -> parent_tree_id mapping from contentTree references."""
    parent_map = {}

    for tree_id, tree_data in trees.items():
        resp = tree_data.get('response', {})
        tree_info = resp.get('tree', {})
        pearls = tree_info.get('pearls', [])

        for pearl in pearls:
            content_tree = pearl.get('contentTree', {})
            if content_tree and 'id' in content_tree:
                child_id = str(content_tree['id'])
                # This tree contains a reference to child_id
                parent_map[child_id] = tree_id

    logger.info(f"Built parent map with {len(parent_map)} child->parent relationships")
    return parent_map


def get_path_to_root(tree_id: str, trees: Dict[str, dict], parent_map: Dict[str, str],
                     max_depth: int = 20) -> List[str]:
    """Get path from tree to root as list of titles."""
    path = []
    current_id = tree_id
    visited = set()

    while current_id and len(path) < max_depth:
        if current_id in visited:
            logger.warning(f"Cycle detected at {current_id}")
            break
        visited.add(current_id)

        if current_id in trees:
            title = trees[current_id]['title']
            path.append(title)

        # Move to parent
        current_id = parent_map.get(current_id)

    # Reverse to get root -> leaf order
    path.reverse()
    return path


def build_target_text(path: List[str], tree_id: str) -> str:
    """Build target_text in the expected format."""
    lines = [f"/{tree_id}"]  # ID line
    indent = ""
    for title in path:
        lines.append(f"{indent}- {title}")
        indent += "  "
    return "\n".join(lines)


def convert_to_jsonl(trees: Dict[str, dict], parent_map: Dict[str, str],
                     output_path: Path, account: str = "s243a"):
    """Convert trees to JSONL format with target_text paths."""
    count = 0

    with open(output_path, 'w') as f:
        for tree_id, tree_data in trees.items():
            path = get_path_to_root(tree_id, trees, parent_map)

            if not path:
                # No path found, use just the title
                path = [tree_data['title']]

            target_text = build_target_text(path, tree_id)

            entry = {
                'tree_id': tree_id,
                'title': tree_data['title'],
                'account': account,
                'target_text': target_text,
                'path_depth': len(path)
            }

            f.write(json.dumps(entry) + '\n')
            count += 1

    logger.info(f"Wrote {count} entries to {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert Pearltrees API DB to tree data JSONL")
    parser.add_argument("--db", type=Path, required=True,
                       help="Path to api_responses.db")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--account", type=str, default="s243a",
                       help="Account name to use (default: s243a)")

    args = parser.parse_args()

    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        return 1

    # Load data
    trees = load_api_responses(args.db)

    # Build parent relationships
    parent_map = build_parent_map(trees)

    # Show some stats
    depths = defaultdict(int)
    for tree_id in trees:
        path = get_path_to_root(tree_id, trees, parent_map)
        depths[len(path)] += 1

    logger.info("Path depth distribution:")
    for depth in sorted(depths.keys()):
        logger.info(f"  depth {depth}: {depths[depth]} trees")

    # Convert and write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    convert_to_jsonl(trees, parent_map, args.output, args.account)

    return 0


if __name__ == "__main__":
    exit(main())
