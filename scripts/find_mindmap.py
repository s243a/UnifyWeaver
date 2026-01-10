#!/usr/bin/env python3
"""
Find mindmap files by tree ID or pattern.

Supports multiple filename formats:
  - id{number}.smmx
  - Title_id{number}.smmx
  - Title_{number}.smmx

Usage:
    # Find by tree ID
    python3 scripts/find_mindmap.py 75009241
    python3 scripts/find_mindmap.py 75009241 14165883 10088091

    # Find by title pattern (case-insensitive)
    python3 scripts/find_mindmap.py --title "psychology"
    python3 scripts/find_mindmap.py --title "science" --title "math"

    # Find by glob pattern
    python3 scripts/find_mindmap.py --glob "*Psychology*.smmx"

    # Use index for fast lookup
    python3 scripts/find_mindmap.py --index .local/data/scans/mindmap_index.json 75009241

    # List all IDs in directory
    python3 scripts/find_mindmap.py --list-ids

    # Search in specific directory
    python3 scripts/find_mindmap.py -d output/mindmaps_curated 75009241
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def extract_tree_id(filename: str) -> Optional[str]:
    """Extract tree ID from filename.

    Supports:
      - id{number}.smmx
      - Title_id{number}.smmx
      - Title_{number}.smmx (6+ digits)
    """
    stem = Path(filename).stem

    # id{number} prefix
    if stem.startswith('id') and stem[2:].isdigit():
        return stem[2:]

    # Title_id{number} suffix
    match = re.search(r'_id(\d+)$', stem)
    if match:
        return match.group(1)

    # Title_{number} suffix (6+ digits)
    match = re.search(r'_(\d{6,})$', stem)
    if match:
        return match.group(1)

    return None


def find_by_id(tree_id: str, search_dir: Path) -> List[Path]:
    """Find mindmap files matching a tree ID."""
    results = []

    # Try exact patterns first (faster)
    patterns = [
        f"id{tree_id}.smmx",
        f"*_id{tree_id}.smmx",
        f"*_{tree_id}.smmx",
    ]

    for pattern in patterns:
        results.extend(search_dir.rglob(pattern))

    return list(set(results))  # Dedupe


def find_by_title(title_pattern: str, search_dir: Path) -> List[Path]:
    """Find mindmap files with title matching pattern (case-insensitive)."""
    results = []
    pattern_lower = title_pattern.lower()

    for smmx in search_dir.rglob("*.smmx"):
        if pattern_lower in smmx.stem.lower():
            results.append(smmx)

    return results


def find_by_glob(glob_pattern: str, search_dir: Path) -> List[Path]:
    """Find mindmap files matching glob pattern."""
    return list(search_dir.rglob(glob_pattern))


def find_in_index(tree_id: str, index: Dict[str, str]) -> Optional[str]:
    """Look up tree ID in index."""
    return index.get(tree_id)


def list_all_ids(search_dir: Path) -> List[Tuple[str, Path]]:
    """List all tree IDs found in directory."""
    results = []

    for smmx in search_dir.rglob("*.smmx"):
        tree_id = extract_tree_id(smmx.name)
        if tree_id:
            results.append((tree_id, smmx))

    return sorted(results, key=lambda x: x[0])


def load_index(index_path: Path) -> Dict[str, str]:
    """Load mindmap index file."""
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description='Find mindmap files by ID or pattern',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('tree_ids', nargs='*', help='Tree IDs to find')
    parser.add_argument('-d', '--directory', type=Path,
                        default=Path('output/mindmaps_curated'),
                        help='Directory to search (default: output/mindmaps_curated)')
    parser.add_argument('--title', action='append', dest='titles',
                        help='Find by title pattern (case-insensitive, can repeat)')
    parser.add_argument('--glob', action='append', dest='globs',
                        help='Find by glob pattern (can repeat)')
    parser.add_argument('--index', type=Path, default=None,
                        help='Use index file for fast lookup')
    parser.add_argument('--list-ids', action='store_true',
                        help='List all tree IDs in directory')
    parser.add_argument('--format', choices=['path', 'name', 'id', 'full'],
                        default='path', help='Output format')
    parser.add_argument('--exists', action='store_true',
                        help='Only show files that exist (for index lookups)')

    args = parser.parse_args()

    # Load index if specified
    index = {}
    if args.index:
        index = load_index(args.index)
        if index:
            print(f"Loaded index: {len(index)} entries", file=__import__('sys').stderr)

    results = []

    # List all IDs mode
    if args.list_ids:
        id_list = list_all_ids(args.directory)
        print(f"Found {len(id_list)} mindmaps with IDs:\n")
        for tree_id, path in id_list:
            if args.format == 'id':
                print(tree_id)
            elif args.format == 'name':
                print(f"{tree_id}\t{path.name}")
            elif args.format == 'full':
                print(f"{tree_id}\t{path.name}\t{path}")
            else:
                print(f"{tree_id}\t{path}")
        return

    # Find by tree IDs
    for tree_id in args.tree_ids:
        # Try index first
        if index and tree_id in index:
            path_str = index[tree_id]
            path = Path(path_str)
            if not args.exists or path.exists():
                results.append(('index', tree_id, path))
            continue

        # Fall back to filesystem search
        found = find_by_id(tree_id, args.directory)
        for path in found:
            results.append(('fs', tree_id, path))

        if not found:
            results.append(('notfound', tree_id, None))

    # Find by title patterns
    if args.titles:
        for title in args.titles:
            found = find_by_title(title, args.directory)
            for path in found:
                tree_id = extract_tree_id(path.name) or '?'
                results.append(('title', tree_id, path))

    # Find by glob patterns
    if args.globs:
        for glob in args.globs:
            found = find_by_glob(glob, args.directory)
            for path in found:
                tree_id = extract_tree_id(path.name) or '?'
                results.append(('glob', tree_id, path))

    # Output results
    if not results:
        if not args.tree_ids and not args.titles and not args.globs:
            parser.print_help()
        return

    for source, tree_id, path in results:
        if path is None:
            print(f"NOT FOUND: {tree_id}")
        elif args.format == 'path':
            print(path)
        elif args.format == 'name':
            print(path.name)
        elif args.format == 'id':
            print(tree_id)
        elif args.format == 'full':
            print(f"{tree_id}\t{path.name}\t{path}\t[{source}]")


if __name__ == '__main__':
    main()
