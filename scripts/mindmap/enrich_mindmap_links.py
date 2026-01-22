#!/usr/bin/env python3
"""
Enrich mindmap with Pearltrees links and RefPearl cross-references.

This script adds:
1. Pearltrees URL to the root node (if missing)
2. cloudmapref links for RefPearl/AliasPearl children pointing to other mindmaps

RefPearl/AliasPearl Link Resolution:
- Uses the mindmap index (index.json) to get expected filenames
- Links are relative cloudmapref paths (e.g., ./Title_id12345.smmx)
- Links work even if target mindmap doesn't exist yet (progress tracking)

Usage:
    # Enrich using index for path resolution
    python3 scripts/mindmap/enrich_mindmap_links.py \
        --mindmap output/mindmaps_curated/Science_id10388356.smmx \
        --children-db output/children_index.db \
        --index output/mindmaps_curated/index.json

    # Dry run
    python3 scripts/mindmap/enrich_mindmap_links.py \
        --mindmap output/mindmaps_curated/Science_id10388356.smmx \
        --children-db output/children_index.db \
        --index output/mindmaps_curated/index.json \
        --dry-run
"""

import argparse
import json
import os
import re
import sqlite3
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET


def extract_tree_id(filename: str) -> Optional[str]:
    """Extract tree ID from filename like 'id75009241.smmx' or 'Title_id75009241.smmx'."""
    match = re.search(r'(?:_id|^id)(\d+)\.smmx$', filename)
    if match:
        return match.group(1)
    match = re.search(r'_(\d{7,})\.smmx$', filename)
    if match:
        return match.group(1)
    return None


def load_mindmap_index(index_path: Path) -> Tuple[Dict[str, str], Optional[str]]:
    """Load mindmap index from JSON file.

    The index maps tree_id -> filename (e.g., "12345" -> "Title_id12345.smmx").
    This allows linking to mindmaps that don't exist yet.

    Returns:
        Tuple of (index dict, base_dir or None)
    """
    if not index_path.exists():
        return {}, None

    with open(index_path, 'r') as f:
        data = json.load(f)

    base_dir = data.get('base_dir')

    # Handle both formats: direct dict or {"index": {...}}
    if 'index' in data:
        return data['index'], base_dir
    return data, base_dir


def save_mindmap_index(index_path: Path, index: Dict[str, str], base_dir: Optional[str] = None):
    """Save mindmap index to JSON file."""
    data = {
        'base_dir': base_dir or str(index_path.parent),
        'count': len(index),
        'index': index
    }
    with open(index_path, 'w') as f:
        json.dump(data, f, indent=2)


def sanitize_filename(title: str, max_length: int = 50) -> str:
    """Sanitize a title for use as a filename."""
    if not title:
        return ""
    # Replace problematic characters with underscores
    sanitized = re.sub(r'[/\\:*?"<>|\s]+', '_', title)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')
    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Truncate
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')
    return sanitized


def generate_filename(title: str, tree_id: str) -> str:
    """Generate mindmap filename from title and tree ID.

    Returns filename like 'Title_id12345.smmx' or 'id12345.smmx' if no title.
    """
    sanitized = sanitize_filename(title)
    if sanitized:
        return f"{sanitized}_id{tree_id}.smmx"
    return f"id{tree_id}.smmx"


def build_mindmap_index(mindmap_dir: Path) -> Dict[str, Path]:
    """Build index of tree_id -> mindmap path by scanning directory."""
    index = {}
    for smmx in mindmap_dir.rglob('*.smmx'):
        tree_id = extract_tree_id(smmx.name)
        if tree_id:
            index[tree_id] = smmx
    return index


def get_tree_info(db_path: Path, tree_id: str) -> Optional[Dict]:
    """Get tree info including Pearltrees URL."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Look for RootPearl which has the tree's own info
        cursor.execute('''
            SELECT uri, title, parent_tree_uri
            FROM children
            WHERE parent_tree_id = ? AND pearl_type = 'RootPearl'
            LIMIT 1
        ''', (tree_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            uri, title, parent_tree_uri = row
            # Extract account from parent_tree_uri
            match = re.search(r'pearltrees\.com/([^/]+)/', parent_tree_uri or '')
            account = match.group(1) if match else 's243a'

            # Build Pearltrees URL
            # Format: https://www.pearltrees.com/{account}/{slug}/id{tree_id}
            slug = re.sub(r'[^\w]+', '-', title.lower()).strip('-') if title else 'tree'
            pearltrees_url = f"https://www.pearltrees.com/{account}/{slug}/id{tree_id}"

            return {
                'tree_id': tree_id,
                'title': title,
                'pearltrees_url': pearltrees_url,
                'account': account
            }
    except Exception as e:
        print(f"Warning: Error getting tree info: {e}")

    return None


def get_refpearl_children(db_path: Path, tree_id: str) -> List[Dict]:
    """Get RefPearl and AliasPearl children that point to other trees."""
    children = []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT pearl_type, title, see_also_uri, pos_order
            FROM children
            WHERE parent_tree_id = ?
              AND pearl_type IN ('RefPearl', 'AliasPearl')
              AND see_also_uri IS NOT NULL
              AND see_also_uri != ''
            ORDER BY pos_order
        ''', (tree_id,))

        for row in cursor.fetchall():
            pearl_type, title, see_also_uri, pos_order = row
            # Extract target tree ID from see_also_uri
            target_id = extract_tree_id_from_uri(see_also_uri)
            if target_id:
                children.append({
                    'type': pearl_type,
                    'title': title,
                    'target_uri': see_also_uri,
                    'target_tree_id': target_id,
                    'pos_order': pos_order
                })

        conn.close()
    except Exception as e:
        print(f"Warning: Error getting RefPearl children: {e}")

    return children


def extract_tree_id_from_uri(uri: str) -> Optional[str]:
    """Extract tree ID from Pearltrees URI."""
    match = re.search(r'id(\d+)', uri)
    return match.group(1) if match else None


def compute_relative_path(from_path: Path, to_path: Path) -> str:
    """Compute relative path with ./ prefix for same-directory."""
    from_dir = from_path.parent.resolve()
    to_path = to_path.resolve()

    try:
        rel = os.path.relpath(to_path, from_dir)
        if not rel.startswith('.') and not rel.startswith('/'):
            rel = './' + rel
        return rel
    except ValueError:
        return str(to_path)


def normalize_title(text: str) -> str:
    """Normalize title for matching.

    Handles:
    - \\N sequences (SimpleMind newlines)
    - &amp; entities
    - Multiple spaces
    - Case insensitivity
    """
    if not text:
        return ""
    # Replace \N (literal backslash-N) with space
    text = text.replace('\\N', ' ').replace('\n', ' ')
    # Replace XML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Collapse multiple spaces and strip
    text = ' '.join(text.split())
    return text.lower()


def find_topic_by_title(root: ET.Element, title: str) -> Optional[ET.Element]:
    """Find a topic element by its text attribute."""
    title_normalized = normalize_title(title)

    for topic in root.iter('topic'):
        topic_text = normalize_title(topic.get('text') or '')
        if topic_text == title_normalized:
            return topic

    return None


def enrich_mindmap(
    smmx_path: Path,
    db_path: Path,
    filename_index: Dict[str, str],
    dry_run: bool = False,
    verbose: bool = False
) -> Tuple[int, int, int]:
    """
    Enrich a mindmap with Pearltrees links.

    Also adds missing RefPearl targets to the filename_index (in-place).

    Returns:
        Tuple of (root_link_added, cloudmapref_links_added, index_entries_added)
    """
    tree_id = extract_tree_id(smmx_path.name)
    if not tree_id:
        print(f"Could not extract tree ID from {smmx_path.name}")
        return (0, 0, 0)

    # Load mindmap XML
    try:
        with zipfile.ZipFile(smmx_path, 'r') as zf:
            xml_content = zf.read('document/mindmap.xml').decode('utf-8')
    except (zipfile.BadZipFile, KeyError) as e:
        print(f"Error reading {smmx_path}: {e}")
        return (0, 0, 0)

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Error parsing XML in {smmx_path}: {e}")
        return (0, 0, 0)

    modified = False
    root_link_added = 0
    cloudmapref_added = 0

    # 1. Add Pearltrees URL to root node if missing
    tree_info = get_tree_info(db_path, tree_id)
    if tree_info:
        root_topic = None
        for topic in root.iter('topic'):
            if topic.get('id') == '0':
                root_topic = topic
                break

        if root_topic is not None:
            # Check if root has a link with urllink
            link = root_topic.find('link')
            if link is not None:
                current_url = link.get('urllink', '')
                if not current_url:
                    link.set('urllink', tree_info['pearltrees_url'])
                    root_link_added = 1
                    modified = True
                    if verbose:
                        print(f"  Added root URL: {tree_info['pearltrees_url']}")
            else:
                # Create link element
                link = ET.SubElement(root_topic, 'link')
                link.set('urllink', tree_info['pearltrees_url'])
                root_link_added = 1
                modified = True
                if verbose:
                    print(f"  Created root link: {tree_info['pearltrees_url']}")

    # 2. Add cloudmapref links for RefPearl/AliasPearl children
    refpearls = get_refpearl_children(db_path, tree_id)
    index_entries_added = 0

    for ref in refpearls:
        target_id = ref['target_tree_id']
        title = ref['title']

        # Find topic matching this RefPearl title
        topic = find_topic_by_title(root, title)
        if topic is None:
            if verbose:
                print(f"  No topic found for '{title}'")
            continue

        # Get or create link element
        link = topic.find('link')
        if link is None:
            link = ET.SubElement(topic, 'link')

        # Look up or generate filename, add to index if missing
        if target_id in filename_index:
            filename = filename_index[target_id]
        else:
            # Generate filename and add to index
            filename = generate_filename(title, target_id)
            filename_index[target_id] = filename
            index_entries_added += 1
            if verbose:
                print(f"  Added to index: {target_id} -> {filename}")

        # Add cloudmapref if not already set
        current_cloudmapref = link.get('cloudmapref', '')
        if not current_cloudmapref:
            rel_path = f"./{filename}"
            link.set('cloudmapref', rel_path)
            cloudmapref_added += 1
            modified = True
            if verbose:
                print(f"  Added cloudmapref for '{title}': {rel_path}")

    # Save if modified
    if modified and not dry_run:
        xml_output = ET.tostring(root, encoding='unicode')
        xml_output = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE simplemind-mindmaps>\n' + xml_output

        with zipfile.ZipFile(smmx_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('document/mindmap.xml', xml_output.encode('utf-8'))

    return (root_link_added, cloudmapref_added, index_entries_added)


def main():
    parser = argparse.ArgumentParser(
        description='Enrich mindmap with Pearltrees links and RefPearl cross-references',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--mindmap', '-m', type=Path, required=True,
                        help='Mindmap file to enrich')
    parser.add_argument('--children-db', '-c', type=Path, required=True,
                        help='Path to children_index.db')
    parser.add_argument('--index', '-i', type=Path,
                        help='Mindmap index file (JSON) for path resolution')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress')

    args = parser.parse_args()

    if not args.mindmap.exists():
        print(f"Error: Mindmap not found: {args.mindmap}")
        return 1

    if not args.children_db.exists():
        print(f"Error: Children database not found: {args.children_db}")
        return 1

    # Load mindmap filename index
    index_path = args.index
    base_dir = None
    if index_path:
        print(f"Loading index from {index_path}...")
        filename_index, base_dir = load_mindmap_index(index_path)
        print(f"  {len(filename_index)} entries")
    else:
        # Fallback: scan directory and extract filenames
        mindmap_dir = args.mindmap.parent
        print(f"No index provided, scanning {mindmap_dir}...")
        path_index = build_mindmap_index(mindmap_dir)
        filename_index = {k: v.name for k, v in path_index.items()}
        print(f"  {len(filename_index)} mindmaps found")

    # Enrich mindmap
    print(f"\nEnriching {args.mindmap.name}...")
    root_added, cloudmaprefs_added, index_added = enrich_mindmap(
        args.mindmap,
        args.children_db,
        filename_index,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    # Save updated index if entries were added
    if index_added > 0 and index_path and not args.dry_run:
        print(f"\nSaving updated index ({index_added} new entries)...")
        save_mindmap_index(index_path, filename_index, base_dir)

    action = "Would add" if args.dry_run else "Added"
    print(f"\n{action}:")
    print(f"  Root Pearltrees link: {root_added}")
    print(f"  RefPearl cloudmapref links: {cloudmaprefs_added}")
    print(f"  Index entries: {index_added}")

    return 0


if __name__ == '__main__':
    exit(main())
