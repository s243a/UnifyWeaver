#!/usr/bin/env python3
"""
Repair broken cloudmapref links in mindmap files.

Scans mindmaps for cloudmapref attributes pointing to non-existent files,
then attempts to find the correct target by matching:
1. Tree ID from the filename
2. Title if no ID available

Usage:
    # Check for broken links (dry run)
    python3 repair_broken_links.py output/mindmaps_curated/ --dry-run

    # Repair all broken links
    python3 repair_broken_links.py output/mindmaps_curated/

    # Repair a single mindmap
    python3 repair_broken_links.py --mindmap output/mindmaps_curated/Science_and_technology_id75009241.smmx
"""

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET


def extract_tree_id(filename: str) -> Optional[str]:
    """Extract tree ID from filename like 'id75009241.smmx' or 'Title_id75009241.smmx'."""
    # Match id followed by digits, or _id followed by digits before .smmx
    match = re.search(r'(?:_id|^id)(\d+)\.smmx$', filename)
    if match:
        return match.group(1)
    # Also try underscore pattern without 'id' prefix
    match = re.search(r'_(\d{7,})\.smmx$', filename)
    if match:
        return match.group(1)
    return None


def extract_title(filename: str) -> Optional[str]:
    """Extract title from filename like 'Science_id10388356.smmx'."""
    basename = os.path.basename(filename)
    # Remove .smmx extension
    name = basename.replace('.smmx', '')
    # Remove ID suffix
    name = re.sub(r'_id\d+$', '', name)
    name = re.sub(r'_\d{7,}$', '', name)
    return name if name else None


def build_id_index(directory: Path) -> Dict[str, Path]:
    """Build index of tree_id -> mindmap path."""
    index = {}
    for smmx in directory.rglob('*.smmx'):
        tree_id = extract_tree_id(smmx.name)
        if tree_id:
            index[tree_id] = smmx
    return index


def build_title_index(directory: Path) -> Dict[str, Path]:
    """Build index of lowercase title -> mindmap path."""
    index = {}
    for smmx in directory.rglob('*.smmx'):
        title = extract_title(smmx.name)
        if title:
            # Normalize: lowercase, replace underscores with spaces
            normalized = title.lower().replace('_', ' ')
            index[normalized] = smmx
    return index


def find_broken_links(smmx_path: Path, search_dir: Path) -> List[Tuple[str, str]]:
    """Find all broken cloudmapref links in a mindmap.

    Returns list of (xpath_description, broken_ref) tuples.
    """
    try:
        with zipfile.ZipFile(smmx_path, 'r') as zf:
            xml_content = zf.read('document/mindmap.xml').decode('utf-8')
    except (zipfile.BadZipFile, KeyError) as e:
        print(f"Error reading {smmx_path}: {e}", file=sys.stderr)
        return []

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Error parsing XML in {smmx_path}: {e}", file=sys.stderr)
        return []

    broken = []
    for link in root.iter('link'):
        cloudmapref = link.get('cloudmapref', '')
        if not cloudmapref:
            continue

        # Resolve the reference relative to the mindmap's directory
        ref_path = (smmx_path.parent / cloudmapref).resolve()

        # Check if target exists
        if not ref_path.exists():
            # Get topic text for context
            parent = link
            topic_text = "unknown"
            for elem in root.iter():
                if link in list(elem):
                    topic_text = elem.get('text', 'unknown')[:50]
                    break
            broken.append((f"topic '{topic_text}'", cloudmapref))

    return broken


def resolve_broken_ref(
    broken_ref: str,
    source_path: Path,
    id_index: Dict[str, Path],
    title_index: Dict[str, Path]
) -> Optional[Path]:
    """Try to resolve a broken reference to an existing mindmap.

    Args:
        broken_ref: The broken cloudmapref value
        source_path: Path to the mindmap containing the broken link
        id_index: tree_id -> Path mapping
        title_index: normalized_title -> Path mapping

    Returns:
        Path to the resolved mindmap, or None if not found
    """
    basename = os.path.basename(broken_ref)

    # Try by tree ID first
    tree_id = extract_tree_id(basename)
    if tree_id and tree_id in id_index:
        return id_index[tree_id]

    # Try by title
    title = extract_title(basename)
    if title:
        normalized = title.lower().replace('_', ' ')
        if normalized in title_index:
            return title_index[normalized]

    return None


def compute_relative_path(from_path: Path, to_path: Path) -> str:
    """Compute relative path from one mindmap to another.

    SimpleMind expects same-directory references to start with './'
    """
    # Both paths should be resolved
    from_dir = from_path.parent.resolve()
    to_path = to_path.resolve()

    try:
        rel = os.path.relpath(to_path, from_dir)
        # SimpleMind expects './' prefix for same-directory files
        if not rel.startswith('.') and not rel.startswith('/'):
            rel = './' + rel
        return rel
    except ValueError:
        # On Windows, can't compute relative path across drives
        return str(to_path)


def repair_mindmap(
    smmx_path: Path,
    id_index: Dict[str, Path],
    title_index: Dict[str, Path],
    dry_run: bool = False,
    verbose: bool = False
) -> Tuple[int, int]:
    """Repair broken links in a single mindmap.

    Returns:
        Tuple of (links_found, links_repaired)
    """
    try:
        with zipfile.ZipFile(smmx_path, 'r') as zf:
            xml_content = zf.read('document/mindmap.xml').decode('utf-8')
    except (zipfile.BadZipFile, KeyError) as e:
        print(f"Error reading {smmx_path}: {e}", file=sys.stderr)
        return (0, 0)

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Error parsing XML in {smmx_path}: {e}", file=sys.stderr)
        return (0, 0)

    broken_found = 0
    repaired = 0
    modified = False

    for link in root.iter('link'):
        cloudmapref = link.get('cloudmapref', '')
        if not cloudmapref:
            continue

        # Check if target exists
        ref_path = (smmx_path.parent / cloudmapref).resolve()
        if ref_path.exists():
            continue

        broken_found += 1

        # Try to resolve
        resolved = resolve_broken_ref(cloudmapref, smmx_path, id_index, title_index)
        if resolved:
            new_ref = compute_relative_path(smmx_path, resolved)
            if verbose:
                print(f"  {cloudmapref}")
                print(f"    -> {new_ref}")
            link.set('cloudmapref', new_ref)
            repaired += 1
            modified = True
        else:
            if verbose:
                print(f"  {cloudmapref}")
                print(f"    -> [NOT FOUND]")

    if modified and not dry_run:
        # Write back to smmx
        xml_output = ET.tostring(root, encoding='unicode')
        xml_output = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE simplemind-mindmaps>\n' + xml_output

        with zipfile.ZipFile(smmx_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('document/mindmap.xml', xml_output.encode('utf-8'))

    return (broken_found, repaired)


def main():
    parser = argparse.ArgumentParser(
        description='Repair broken cloudmapref links in mindmaps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('directory', type=Path, nargs='?',
                        help='Directory to scan for mindmaps')
    parser.add_argument('--mindmap', '-m', type=Path,
                        help='Single mindmap to repair')
    parser.add_argument('--dry-run', '-d', action='store_true',
                        help='Show what would be fixed without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show each repair')

    args = parser.parse_args()

    if not args.directory and not args.mindmap:
        parser.error("Must specify directory or --mindmap")

    # Determine search directory
    if args.mindmap:
        search_dir = args.mindmap.parent
        mindmaps = [args.mindmap]
    else:
        search_dir = args.directory
        mindmaps = list(search_dir.rglob('*.smmx'))

    print(f"Building index from {search_dir}...")
    id_index = build_id_index(search_dir)
    title_index = build_title_index(search_dir)
    print(f"  {len(id_index)} mindmaps indexed by ID")
    print(f"  {len(title_index)} mindmaps indexed by title")

    total_broken = 0
    total_repaired = 0
    files_with_broken = 0

    print(f"\nScanning {len(mindmaps)} mindmaps...")
    for smmx_path in mindmaps:
        broken, repaired = repair_mindmap(
            smmx_path, id_index, title_index,
            dry_run=args.dry_run, verbose=args.verbose
        )

        if broken > 0:
            files_with_broken += 1
            total_broken += broken
            total_repaired += repaired

            status = f"{repaired}/{broken} repaired" if not args.dry_run else f"{repaired}/{broken} would repair"
            print(f"{smmx_path.name}: {status}")

    print(f"\nSummary:")
    action = "Would repair" if args.dry_run else "Repaired"
    print(f"  Files with broken links: {files_with_broken}")
    print(f"  Total broken links: {total_broken}")
    print(f"  {action}: {total_repaired}")

    if total_broken > total_repaired:
        print(f"  Could not resolve: {total_broken - total_repaired}")


if __name__ == '__main__':
    main()
