#!/usr/bin/env python3
"""
Build a reverse index of mindmap links (backlinks).

Scans all mindmaps and records which mindmaps link to each other.
Useful for updating links when a mindmap is renamed or moved.

Usage:
    python3 build_reverse_index.py <mindmap_dir> [output_file]
    python3 build_reverse_index.py output/mindmaps_curated backlinks.json
    python3 build_reverse_index.py output/mindmaps_curated backlinks.tsv
"""

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from index_store import ReverseIndex


def extract_tree_id_from_url(url: str) -> Optional[str]:
    """Extract tree ID from Pearltrees URL."""
    if not url or 'pearltrees.com' not in url:
        return None
    match = re.search(r'/id(\d+)(?:/|$)', url)
    if match:
        return match.group(1)
    return None


def extract_tree_id_from_cloudmapref(ref: str) -> Optional[str]:
    """Extract tree ID from cloudmapref path like '../folder/id12345.smmx'."""
    match = re.search(r'id(\d+)\.smmx$', ref)
    if match:
        return match.group(1)
    return None


def extract_tree_id_from_filename(filename: str) -> Optional[str]:
    """Extract tree ID from mindmap filename."""
    match = re.match(r'^id(\d+)\.smmx$', os.path.basename(filename))
    if match:
        return match.group(1)
    return None


def scan_mindmap_links(smmx_path: str) -> set:
    """Scan a mindmap and return all tree IDs it links to.

    Checks both urllink (Pearltrees URLs) and cloudmapref (local links).
    """
    linked_ids = set()

    try:
        with zipfile.ZipFile(smmx_path, 'r') as zf:
            xml_content = zf.read('document/mindmap.xml').decode('utf-8')
    except (zipfile.BadZipFile, KeyError):
        return linked_ids

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return linked_ids

    for link in root.iter('link'):
        # Check urllink
        urllink = link.get('urllink')
        if urllink:
            tree_id = extract_tree_id_from_url(urllink)
            if tree_id:
                linked_ids.add(tree_id)

        # Check cloudmapref
        cloudmapref = link.get('cloudmapref')
        if cloudmapref:
            tree_id = extract_tree_id_from_cloudmapref(cloudmapref)
            if tree_id:
                linked_ids.add(tree_id)

    return linked_ids


def build_reverse_index(mindmap_dir: str, verbose: bool = False) -> ReverseIndex:
    """Build reverse index from all mindmaps in directory.

    Args:
        mindmap_dir: Directory containing .smmx files
        verbose: Print progress

    Returns:
        ReverseIndex with all backlinks
    """
    reverse = ReverseIndex()
    base = Path(mindmap_dir)

    smmx_files = list(base.rglob("*.smmx"))
    total = len(smmx_files)

    for i, smmx_file in enumerate(smmx_files):
        source_id = extract_tree_id_from_filename(str(smmx_file))
        if not source_id:
            continue

        if verbose and (i % 500 == 0 or i == total - 1):
            print(f"Scanning {i+1}/{total}...", file=sys.stderr)

        linked_ids = scan_mindmap_links(str(smmx_file))

        # Remove self-links
        linked_ids.discard(source_id)

        for target_id in linked_ids:
            reverse.add_link(source_id, target_id)

    return reverse


def main():
    parser = argparse.ArgumentParser(
        description='Build reverse index of mindmap links (backlinks)'
    )
    parser.add_argument('mindmap_dir', help='Directory containing .smmx files')
    parser.add_argument('output', nargs='?', help='Output file (JSON or TSV)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print progress')

    args = parser.parse_args()

    if not os.path.isdir(args.mindmap_dir):
        print(f"Error: {args.mindmap_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    reverse = build_reverse_index(args.mindmap_dir, verbose=args.verbose)

    # Count stats
    total_links = sum(len(sources) for _, sources in reverse.items())
    targets_with_backlinks = sum(1 for _ in reverse.items())

    print(f"Found {total_links} links to {targets_with_backlinks} targets", file=sys.stderr)

    if args.output:
        ext = Path(args.output).suffix.lower()
        if ext == '.tsv':
            reverse.save_tsv(args.output)
        else:
            reverse.save_json(args.output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        # Print summary
        print("\nTop 10 most linked-to mindmaps:")
        items = [(target, sources) for target, sources in reverse.items()]
        items.sort(key=lambda x: len(x[1]), reverse=True)
        for target, sources in items[:10]:
            print(f"  {target}: {len(sources)} backlinks")


if __name__ == '__main__':
    main()
