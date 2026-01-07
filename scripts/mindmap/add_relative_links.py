#!/usr/bin/env python3
"""
Add relative links (cloudmapref) to mindmap files using the index.

For each Pearltrees URL in a mindmap, looks up the corresponding local
mindmap file and adds a cloudmapref attribute for local navigation.

Usage:
    python3 add_relative_links.py mindmap.smmx --index index.json
    python3 add_relative_links.py mindmap.smmx --index index.json --dry-run
    python3 add_relative_links.py output/mindmaps_curated/*.smmx --index index.json
"""

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from index_store import create_index_store, IndexStore


def extract_tree_id_from_url(url: str) -> Optional[str]:
    """Extract tree ID from Pearltrees URL.

    Patterns:
    - https://www.pearltrees.com/user/slug/id{tree_id}
    - https://www.pearltrees.com/user/slug/id{tree_id}/item{pearl_id}

    Returns:
        Tree ID string or None if not a tree URL
    """
    if not url or 'pearltrees.com' not in url:
        return None

    # Match /id{digits} pattern (tree ID)
    match = re.search(r'/id(\d+)(?:/|$)', url)
    if match:
        return match.group(1)

    return None


def compute_relative_path(source_path: str, target_path: str, base_dir: str) -> str:
    """Compute relative path from source mindmap to target mindmap.

    Args:
        source_path: Path to source .smmx file (absolute or relative to base_dir)
        target_path: Path to target .smmx file (relative to base_dir from index)
        base_dir: Base directory for the index

    Returns:
        Relative path from source to target (e.g., '../sibling/target.smmx')
    """
    # Make both paths absolute
    if not os.path.isabs(source_path):
        source_abs = os.path.join(base_dir, source_path)
    else:
        source_abs = source_path

    target_abs = os.path.join(base_dir, target_path)

    # Get directories
    source_dir = os.path.dirname(source_abs)
    target_dir = os.path.dirname(target_abs)
    target_filename = os.path.basename(target_abs)

    # Compute relative path from source dir to target dir
    rel_dir = os.path.relpath(target_dir, source_dir)

    # Combine with filename
    if rel_dir == '.':
        return target_filename
    else:
        return os.path.join(rel_dir, target_filename)


def extract_tree_id_from_filename(filename: str) -> Optional[str]:
    """Extract tree ID from mindmap filename like 'id75009241.smmx' or 'id75009241_repaired.smmx'."""
    match = re.search(r'id(\d+)', os.path.basename(filename))
    if match:
        return match.group(1)
    return None


def process_mindmap(
    smmx_path: str,
    store: IndexStore,
    dry_run: bool = False,
    verbose: bool = False,
    root_dir: str = None
) -> Tuple[int, int]:
    """Process a mindmap file and add cloudmapref links.

    Args:
        smmx_path: Path to .smmx file
        store: Index store for looking up mindmap paths
        dry_run: If True, don't save changes
        verbose: If True, print each link added
        root_dir: Override base directory for relative path computation
                  (use when source file is outside the index base_dir)

    Returns:
        Tuple of (links_found, links_added)
    """
    smmx_path = os.path.abspath(smmx_path)
    base_dir = root_dir or store.base_dir

    # Get this mindmap's tree ID to avoid self-links
    self_tree_id = extract_tree_id_from_filename(smmx_path)

    if not base_dir:
        print(f"Warning: No base_dir in index, using mindmap directory", file=sys.stderr)
        base_dir = os.path.dirname(smmx_path)

    # Read the smmx file (it's a ZIP)
    try:
        with zipfile.ZipFile(smmx_path, 'r') as zf:
            xml_content = zf.read('document/mindmap.xml').decode('utf-8')
    except (zipfile.BadZipFile, KeyError) as e:
        print(f"Error reading {smmx_path}: {e}", file=sys.stderr)
        return (0, 0)

    # Parse XML
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Error parsing XML in {smmx_path}: {e}", file=sys.stderr)
        return (0, 0)

    # Find the source path relative to base_dir
    try:
        source_rel = os.path.relpath(smmx_path, base_dir)
    except ValueError:
        # Different drives on Windows
        source_rel = smmx_path

    links_found = 0
    links_added = 0

    # Process all link elements
    for link in root.iter('link'):
        urllink = link.get('urllink')
        if not urllink:
            continue

        tree_id = extract_tree_id_from_url(urllink)
        if not tree_id:
            continue

        links_found += 1

        # Skip self-links
        if tree_id == self_tree_id:
            continue

        # Skip if already has cloudmapref
        if link.get('cloudmapref'):
            continue

        # Look up in index
        target_path = store.get(tree_id)
        if not target_path:
            if verbose:
                print(f"  Tree {tree_id} not in index", file=sys.stderr)
            continue

        # Compute relative path
        rel_path = compute_relative_path(source_rel, target_path, base_dir)

        # Add cloudmapref
        link.set('cloudmapref', rel_path)
        links_added += 1

        if verbose:
            print(f"  Added: {tree_id} -> {rel_path}")

    if links_added > 0 and not dry_run:
        # Write back to smmx
        # Preserve XML declaration
        xml_output = ET.tostring(root, encoding='unicode')
        xml_output = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE simplemind-mindmaps>\n' + xml_output

        with zipfile.ZipFile(smmx_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('document/mindmap.xml', xml_output.encode('utf-8'))

    return (links_found, links_added)


def main():
    parser = argparse.ArgumentParser(
        description='Add relative links (cloudmapref) to mindmap files'
    )
    parser.add_argument('files', nargs='+', help='Mindmap files to process')
    parser.add_argument('--index', '-i', required=True,
                       help='Path to index file (JSON, TSV, or SQLite)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print each link added')
    parser.add_argument('--cache', action='store_true',
                       help='Cache index in memory (faster for many files)')
    parser.add_argument('--root', '-r', default=None,
                       help='Root directory for relative path computation '
                            '(overrides index base_dir)')

    args = parser.parse_args()

    # Load index
    store = create_index_store(args.index, cache=args.cache)
    print(f"Loaded index with {store.count()} entries", file=sys.stderr)

    if not store.base_dir:
        print("Warning: Index has no base_dir set", file=sys.stderr)

    total_found = 0
    total_added = 0
    files_modified = 0

    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}", file=sys.stderr)
            continue

        if args.verbose:
            print(f"Processing: {filepath}")

        found, added = process_mindmap(
            filepath, store,
            dry_run=args.dry_run,
            verbose=args.verbose,
            root_dir=args.root
        )

        total_found += found
        total_added += added
        if added > 0:
            files_modified += 1

    action = "Would add" if args.dry_run else "Added"
    print(f"\n{action} {total_added} cloudmapref links ({total_found} Pearltrees links found)")
    print(f"Files {'that would be ' if args.dry_run else ''}modified: {files_modified}")


if __name__ == '__main__':
    main()
