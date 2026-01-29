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


def load_tree_files(trees_dir: Path) -> Dict[str, dict]:
    """Load tree data from individual JSON files.

    Expects files with 'api_response' key (Pearltrees API format).
    """
    trees = {}
    skipped = 0

    if not trees_dir.exists():
        logger.warning(f"Trees directory not found: {trees_dir}")
        return trees

    for f in trees_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)

            # Validate structure - must be a dict with api_response
            if not isinstance(data, dict):
                skipped += 1
                continue
            if 'api_response' not in data:
                skipped += 1
                continue

            tree_id = str(data.get('tree_id', f.stem))
            resp = data.get('api_response', {})
            title = resp.get('tree', {}).get('title', '')
            trees[tree_id] = {
                'tree_id': tree_id,
                'title': title,
                'response': resp
            }
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load {f}: {e}")

    if skipped:
        logger.info(f"Skipped {skipped} files without api_response structure")
    logger.info(f"Loaded {len(trees)} trees from {trees_dir}")
    return trees


def build_parent_map(trees: Dict[str, dict]) -> Dict[str, tuple]:
    """Build child_tree_id -> (parent_tree_id, parent_title) mapping.

    Uses info.parentTree for direct parent reference (preferred),
    falls back to contentTree references in pearls.
    """
    parent_map = {}

    for tree_id, tree_data in trees.items():
        resp = tree_data.get('response', {})

        # Primary: use info.parentTree (direct parent reference)
        info = resp.get('info', {})
        parent_tree = info.get('parentTree', {})
        if parent_tree and 'id' in parent_tree:
            parent_id = str(parent_tree['id'])
            parent_title = parent_tree.get('title', '')
            parent_map[tree_id] = (parent_id, parent_title)
            continue

        # Fallback: scan for contentTree references (less reliable)
        tree_info = resp.get('tree', {})
        pearls = tree_info.get('pearls', [])
        for pearl in pearls:
            content_tree = pearl.get('contentTree', {})
            if content_tree and 'id' in content_tree:
                child_id = str(content_tree['id'])
                if child_id not in parent_map:
                    parent_map[child_id] = (tree_id, tree_data['title'])

    logger.info(f"Built parent map with {len(parent_map)} child->parent relationships")
    return parent_map


def get_path_to_root(tree_id: str, trees: Dict[str, dict], parent_map: Dict[str, tuple],
                     max_depth: int = 20) -> List[str]:
    """Get path from tree to root as list of titles.

    Uses parent_map which maps tree_id -> (parent_id, parent_title).
    This allows building paths even when parent trees weren't fetched,
    because we store parent titles in the parent_map.
    """
    path = []
    current_id = tree_id
    visited = set()

    while current_id and len(path) < max_depth:
        if current_id in visited:
            logger.warning(f"Cycle detected at {current_id}")
            break
        visited.add(current_id)

        # Get title and parent info for current node
        if current_id in trees:
            title = trees[current_id]['title']
            # Get parent from the tree's own info.parentTree
            resp = trees[current_id].get('response', {})
            info = resp.get('info', {})
            parent_tree = info.get('parentTree', {})
            if parent_tree and 'id' in parent_tree:
                next_id = str(parent_tree['id'])
                next_title = parent_tree.get('title', '')
                # Store in parent_map for future lookups
                if next_id not in parent_map:
                    parent_map[next_id] = (None, next_title)  # No parent of parent known
            else:
                next_id = None
        else:
            # Node not in trees - check if we have its title from parent_map
            title = None
            next_id = None
            # Look for this ID as a parent in parent_map entries
            for child_id, (parent_id, parent_title) in parent_map.items():
                if parent_id == current_id and parent_title:
                    title = parent_title
                    break
            if not title:
                # Check if current_id is stored as a key with title info
                if current_id in parent_map:
                    _, stored_title = parent_map[current_id]
                    if stored_title:
                        title = stored_title
            if not title:
                title = f"Unknown({current_id})"

        path.append(title)

        # Move to parent - first check parent_map, then use next_id from tree
        parent_info = parent_map.get(current_id)
        if parent_info and parent_info[0]:
            current_id = parent_info[0]
        elif next_id:
            current_id = next_id
        else:
            current_id = None

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
    parser = argparse.ArgumentParser(
        description="Convert Pearltrees API data to tree data JSONL",
        epilog="""
Examples:
  # From database:
  python3 scripts/convert_api_to_tree_data.py \\
    --db .local/data/pearltrees_api/api_responses.db \\
    --output .local/data/api_tree_paths.jsonl

  # From tree files directory:
  python3 scripts/convert_api_to_tree_data.py \\
    --trees-dir .local/data/pearltrees_api/trees \\
    --output .local/data/api_tree_paths.jsonl

  # Combine both sources:
  python3 scripts/convert_api_to_tree_data.py \\
    --db .local/data/pearltrees_api/api_responses.db \\
    --trees-dir .local/data/pearltrees_api/trees \\
    --output .local/data/api_tree_paths.jsonl
        """
    )
    parser.add_argument("--db", type=Path, default=None,
                       help="Path to api_responses.db")
    parser.add_argument("--trees-dir", type=Path, default=None, dest="trees_dir",
                       help="Path to directory with tree JSON files")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--account", type=str, default="s243a",
                       help="Account name to use (default: s243a)")

    args = parser.parse_args()

    if not args.db and not args.trees_dir:
        parser.error("At least one of --db or --trees-dir is required")

    # Load data from all sources
    trees = {}

    if args.db and args.db.exists():
        db_trees = load_api_responses(args.db)
        trees.update(db_trees)
    elif args.db:
        logger.warning(f"Database not found: {args.db}")

    if args.trees_dir:
        file_trees = load_tree_files(args.trees_dir)
        # Merge, preferring file data (more likely to have info.parentTree)
        for tree_id, data in file_trees.items():
            if tree_id not in trees:
                trees[tree_id] = data
            else:
                # If file has info.parentTree and db doesn't, use file
                file_resp = data.get('response', {})
                db_resp = trees[tree_id].get('response', {})
                if file_resp.get('info', {}).get('parentTree') and not db_resp.get('info', {}).get('parentTree'):
                    trees[tree_id] = data

    if not trees:
        logger.error("No trees loaded from any source")
        return 1

    logger.info(f"Total trees loaded: {len(trees)}")

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
