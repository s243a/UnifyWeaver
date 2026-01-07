#!/usr/bin/env python3
"""
Build an index of mindmap files mapping tree IDs to file paths.

This index enables:
- Relative linking between mindmaps without full recursive regeneration
- Avoiding AI calls for human-readable names on reruns
- Fast lookup of existing mindmaps by tree ID

Usage:
    python3 build_index.py <mindmap_dir> [output_file]
    python3 build_index.py output/mindmaps_curated output/mindmaps_curated/index.json
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Dict


def extract_tree_id(filename: str) -> Optional[str]:
    """Extract tree ID from filename.

    Supported patterns:
    - id{number}.smmx -> returns the number
    - sha_{hash}.smmx -> returns sha_{hash}
    - {alphanumeric}.smmx -> returns the stem if long enough
    """
    stem = Path(filename).stem

    # Numeric ID: id{number}
    match = re.match(r'^id(\d+)$', stem)
    if match:
        return match.group(1)

    # Synthetic ID: sha_{hash}
    match = re.match(r'^(sha_[a-f0-9]+)$', stem)
    if match:
        return match.group(1)

    # Other alphanumeric patterns (synthetic IDs from old format)
    if re.match(r'^[a-z0-9]+$', stem) and len(stem) > 5:
        return stem

    return None


def build_index(search_dir: str) -> Dict[str, str]:
    """Build index mapping tree_id -> relative path.

    Args:
        search_dir: Directory to search for .smmx files

    Returns:
        Dict mapping tree IDs to relative file paths
    """
    index = {}
    base = Path(search_dir)

    for smmx_file in base.rglob("*.smmx"):
        tree_id = extract_tree_id(smmx_file.name)
        if tree_id:
            # Store relative path from base
            rel_path = str(smmx_file.relative_to(base))
            index[tree_id] = rel_path

    return index


def main():
    if len(sys.argv) < 2:
        print("Usage: build_index.py <mindmap_dir> [output_file]", file=sys.stderr)
        print("       build_index.py output/mindmaps_curated index.json", file=sys.stderr)
        sys.exit(1)

    search_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.isdir(search_dir):
        print(f"Error: {search_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    index = build_index(search_dir)

    result = {
        "base_dir": os.path.abspath(search_dir),
        "count": len(index),
        "index": index
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Written {len(index)} entries to {output_file}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
