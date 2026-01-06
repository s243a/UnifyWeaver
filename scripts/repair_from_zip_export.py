#!/usr/bin/env python3
"""
Repair JSONL data from Pearltrees zip exports.

Pearltrees zip exports contain:
- HTML files for each PagePearl (with PT link and source URL)
- Subdirectories for child Trees

This script parses the export folder structure to generate JSONL entries
for trees and pearls that may be missing from the RDF export.

Usage:
    # Extract repairs to separate file
    python3 scripts/repair_from_zip_export.py \
        --export-dir "context/PT/Zip_Exports/Academic disciplines" \
        --root-uri "https://www.pearltrees.com/s243a/academic-disciplines/id53344165" \
        --account s243a \
        --output data/repair_academic_disciplines.jsonl

    # Compare with existing JSONL (report what's missing)
    python3 scripts/repair_from_zip_export.py \
        --export-dir "context/PT/Zip_Exports/Academic disciplines" \
        --root-uri "https://www.pearltrees.com/s243a/academic-disciplines/id53344165" \
        --account s243a \
        --compare data/pearltrees_all.jsonl \
        --output /tmp/missing_items.jsonl

    # Auto-merge repairs into existing JSONL
    python3 scripts/repair_from_zip_export.py \
        --export-dir "context/PT/Zip_Exports/Academic disciplines" \
        --root-uri "https://www.pearltrees.com/s243a/academic-disciplines/id53344165" \
        --account s243a \
        --merge-into data/pearltrees_all.jsonl
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import date

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_pearl_data_from_html(html_file: Path) -> Optional[Dict]:
    """Extract pearl data from an HTML export file.

    Returns dict with:
        - title: Pearl title
        - pearl_uri: Full Pearltrees URI (with item ID)
        - parent_tree_uri: Parent tree URI (extracted from pearl URI)
        - source_url: The URL the pearl points to
        - item_id: The pearl's item ID
    """
    try:
        content = html_file.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        logger.warning(f"Could not read {html_file}: {e}")
        return None

    # Extract title from <title> tag
    title_match = re.search(r'<title>([^<]+)</title>', content)
    title = title_match.group(1).strip() if title_match else html_file.stem

    # Extract Pearltrees item link from see-medal div
    # Format: data-pt-skip="true"href="https://www.pearltrees.com/account/tree-name/id123/item456"
    pt_link_match = re.search(
        r'data-pt-skip="true"href="(https://www\.pearltrees\.com/[^"]+/item\d+)"', content)

    if not pt_link_match:
        # Try standard href format
        pt_link_match = re.search(
            r'href="(https://www\.pearltrees\.com/[^"]+/item\d+)"', content)

    if not pt_link_match:
        logger.debug(f"No PT link found in {html_file.name}")
        return None

    pearl_uri = pt_link_match.group(1)

    # Extract item ID and parent tree URI from pearl URI
    # e.g., .../scientific-disciplines/id72963112/item556601870
    item_match = re.search(r'(/item\d+)$', pearl_uri)
    item_id = item_match.group(1).replace('/item', '') if item_match else None

    # Parent tree URI is everything before /item...
    parent_tree_uri = pearl_uri.rsplit('/item', 1)[0] if '/item' in pearl_uri else None

    # Extract source URL (the actual webpage) from author-medal div
    # Format: <div class="author-medal"><a href="URL"
    source_match = re.search(r'<div class="author-medal"><a href="([^"]+)"', content)
    if not source_match:
        # Try: <a class="relative-url" href="URL"
        source_match = re.search(r'class="relative-url" href="([^"]+)"', content)

    source_url = source_match.group(1) if source_match else None

    return {
        'title': title,
        'pearl_uri': pearl_uri,
        'parent_tree_uri': parent_tree_uri,
        'source_url': source_url,
        'item_id': item_id
    }


def parse_export_folder(
    export_dir: Path,
    parent_tree_uri: str,
    account: str,
    depth: int = 0
) -> Tuple[List[Dict], List[Dict]]:
    """Recursively parse export folder to extract trees and pearls.

    Args:
        export_dir: Path to export folder
        parent_tree_uri: URI of the parent tree (used as cluster_id for this folder's items)
        account: Account name
        depth: Current recursion depth (for logging)

    Returns:
        (trees, pearls): Lists of tree and pearl dicts
    """
    trees = []
    pearls = []

    indent = "  " * depth
    logger.info(f"{indent}Parsing: {export_dir.name}")

    # Process HTML files (PagePearls)
    html_files = [f for f in export_dir.iterdir()
                  if f.suffix == '.html' and 'Zone.Identifier' not in f.name]

    # Track the actual tree URI discovered from pearl URIs
    # (all pearls in a folder should have the same parent tree URI)
    discovered_tree_uri = None

    for html_file in html_files:
        pearl_data = extract_pearl_data_from_html(html_file)
        if pearl_data:
            # Use the parent_tree_uri from the pearl itself (contains actual tree ID)
            actual_parent = pearl_data.get('parent_tree_uri') or parent_tree_uri

            # Track discovered URI for this folder
            if pearl_data.get('parent_tree_uri') and not discovered_tree_uri:
                discovered_tree_uri = pearl_data['parent_tree_uri']

            # Build PagePearl entry
            pearl_entry = {
                'type': 'PagePearl',
                'raw_title': pearl_data['title'],
                'query': pearl_data['title'],
                'cluster_id': actual_parent,
                'pearl_id': f"item{pearl_data['item_id']}" if pearl_data['item_id'] else '',
                'pearl_uri': pearl_data['pearl_uri'],
                'parent_tree_uri': actual_parent,
                'account': account,
                'url': pearl_data['source_url'] or '',
                '_source': 'zip_export',
                '_source_file': str(html_file.relative_to(export_dir.parent.parent))
            }
            pearls.append(pearl_entry)

    logger.info(f"{indent}  Found {len(html_files)} HTML files -> {len(pearls)} pearls")
    if discovered_tree_uri:
        logger.info(f"{indent}  Discovered tree URI: {discovered_tree_uri}")

    # Process subdirectories (child Trees)
    subdirs = [d for d in export_dir.iterdir() if d.is_dir()]

    for subdir in subdirs:
        # First, peek into the subdir to find the tree URI from its pearl files
        subdir_tree_uri = None
        for html_file in subdir.glob("*.html"):
            if 'Zone.Identifier' in str(html_file):
                continue
            pearl_data = extract_pearl_data_from_html(html_file)
            if pearl_data and pearl_data.get('parent_tree_uri'):
                subdir_tree_uri = pearl_data['parent_tree_uri']
                break

        # Extract tree_id from discovered URI
        tree_id = None
        if subdir_tree_uri:
            # URI format: .../tree-name/id12345
            id_match = re.search(r'/id(\d+)$', subdir_tree_uri)
            tree_id = id_match.group(1) if id_match else None

        # Create Tree entry with discovered URI
        tree_entry = {
            'type': 'Tree',
            'raw_title': subdir.name,
            'cluster_id': discovered_tree_uri or parent_tree_uri,
            'account': account,
            '_source': 'zip_export',
            '_source_folder': str(subdir.relative_to(export_dir.parent.parent)),
        }

        if subdir_tree_uri:
            tree_entry['uri'] = subdir_tree_uri
            tree_entry['tree_id'] = tree_id
        else:
            tree_entry['_needs_uri'] = True

        trees.append(tree_entry)

        # Recursively process child folder with discovered URI
        child_trees, child_pearls = parse_export_folder(
            subdir,
            subdir_tree_uri or f"{parent_tree_uri}/{subdir.name.lower().replace(' ', '-')}",
            account,
            depth + 1
        )

        trees.extend(child_trees)
        pearls.extend(child_pearls)

    return trees, pearls


def generate_target_text(pearl: Dict, parent_title: str = "") -> str:
    """Generate target_text for embedding (simplified version)."""
    title = pearl.get('raw_title', '')
    if parent_title:
        return f"- {parent_title}\n  - {title}"
    return f"- {title}"


def load_existing_data(jsonl_path: Path) -> Tuple[Dict[str, Dict], Dict[str, Dict], set]:
    """Load existing JSONL and build indices.

    Returns:
        (pearl_by_uri, tree_by_uri, all_uris): Indices for quick lookup
    """
    pearl_by_uri = {}
    tree_by_uri = {}
    all_uris = set()

    if not jsonl_path.exists():
        logger.warning(f"JSONL file not found: {jsonl_path}")
        return pearl_by_uri, tree_by_uri, all_uris

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            entry_type = entry.get('type', '')

            if entry_type == 'Tree':
                uri = entry.get('uri', '')
                if uri:
                    tree_by_uri[uri] = entry
                    all_uris.add(uri)
            elif entry_type in ('PagePearl', 'RefPearl', 'AliasPearl'):
                uri = entry.get('pearl_uri', '')
                if uri:
                    pearl_by_uri[uri] = entry
                    all_uris.add(uri)

    logger.info(f"Loaded {len(tree_by_uri)} trees, {len(pearl_by_uri)} pearls from {jsonl_path}")
    return pearl_by_uri, tree_by_uri, all_uris


def compare_with_existing(
    extracted_trees: List[Dict],
    extracted_pearls: List[Dict],
    pearl_by_uri: Dict[str, Dict],
    tree_by_uri: Dict[str, Dict]
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Compare extracted data with existing JSONL.

    Returns:
        (missing_trees, missing_pearls, existing_trees, existing_pearls)
    """
    missing_trees = []
    missing_pearls = []
    existing_trees = []
    existing_pearls = []

    for tree in extracted_trees:
        uri = tree.get('uri', '')
        if uri and uri in tree_by_uri:
            existing_trees.append(tree)
        else:
            missing_trees.append(tree)

    for pearl in extracted_pearls:
        uri = pearl.get('pearl_uri', '')
        if uri and uri in pearl_by_uri:
            existing_pearls.append(pearl)
        else:
            missing_pearls.append(pearl)

    return missing_trees, missing_pearls, existing_trees, existing_pearls


def merge_into_jsonl(
    jsonl_path: Path,
    new_entries: List[Dict],
    backup: bool = True
) -> int:
    """Merge new entries into existing JSONL file.

    Args:
        jsonl_path: Path to existing JSONL
        new_entries: New entries to add
        backup: Whether to create .bak backup

    Returns:
        Number of entries added
    """
    if not new_entries:
        logger.info("No new entries to merge")
        return 0

    # Create backup
    if backup and jsonl_path.exists():
        backup_path = jsonl_path.with_suffix('.jsonl.bak')
        import shutil
        shutil.copy(jsonl_path, backup_path)
        logger.info(f"Created backup: {backup_path}")

    # Append new entries
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"Merged {len(new_entries)} entries into {jsonl_path}")
    return len(new_entries)


def main():
    parser = argparse.ArgumentParser(
        description="Repair JSONL from Pearltrees zip exports")
    parser.add_argument("--export-dir", type=Path, required=True,
                       help="Path to zip export folder")
    parser.add_argument("--root-uri", type=str, required=True,
                       help="URI of the root tree (e.g., https://www.pearltrees.com/s243a/tree/id123)")
    parser.add_argument("--account", type=str, required=True,
                       help="Account name")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output JSONL file (required unless --merge-into)")
    parser.add_argument("--compare", type=Path, default=None,
                       help="Compare with existing JSONL and report what's missing")
    parser.add_argument("--merge-into", type=Path, default=None,
                       help="Auto-merge repairs directly into this JSONL file")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip backup when using --merge-into")
    parser.add_argument("--parent-title", type=str, default="",
                       help="Title of parent tree (for target_text generation)")

    args = parser.parse_args()

    # Validate arguments
    if not args.output and not args.merge_into:
        parser.error("Either --output or --merge-into is required")

    if not args.export_dir.exists():
        logger.error(f"Export directory not found: {args.export_dir}")
        return 1

    logger.info(f"Parsing export: {args.export_dir}")
    logger.info(f"Root URI: {args.root_uri}")

    # Parse the export folder
    trees, pearls = parse_export_folder(
        args.export_dir, args.root_uri, args.account)

    logger.info(f"\nExtracted from zip export:")
    logger.info(f"  Trees: {len(trees)}")
    logger.info(f"  Pearls: {len(pearls)}")

    # Add target_text to pearls
    for pearl in pearls:
        pearl['target_text'] = generate_target_text(pearl, args.parent_title)

    # Compare mode: check what's missing from existing JSONL
    compare_jsonl = args.compare or args.merge_into
    if compare_jsonl:
        pearl_by_uri, tree_by_uri, all_uris = load_existing_data(compare_jsonl)

        missing_trees, missing_pearls, existing_trees, existing_pearls = \
            compare_with_existing(trees, pearls, pearl_by_uri, tree_by_uri)

        print(f"\n=== Comparison with {compare_jsonl.name} ===")
        print(f"Trees in zip export:  {len(trees)}")
        print(f"  - Already in JSONL: {len(existing_trees)}")
        print(f"  - Missing (NEW):    {len(missing_trees)}")
        print(f"Pearls in zip export: {len(pearls)}")
        print(f"  - Already in JSONL: {len(existing_pearls)}")
        print(f"  - Missing (NEW):    {len(missing_pearls)}")

        # Show sample of missing items
        if missing_trees:
            print(f"\nMissing trees:")
            for tree in missing_trees[:10]:
                uri = tree.get('uri', '(no URI)')
                print(f"  - {tree['raw_title']}: {uri}")
            if len(missing_trees) > 10:
                print(f"  ... and {len(missing_trees) - 10} more")

        if missing_pearls:
            print(f"\nMissing pearls (sample):")
            for pearl in missing_pearls[:10]:
                print(f"  - {pearl['raw_title']}")
            if len(missing_pearls) > 10:
                print(f"  ... and {len(missing_pearls) - 10} more")

        # Only output missing entries
        all_entries = missing_trees + missing_pearls
    else:
        all_entries = trees + pearls

    # Handle merge mode
    if args.merge_into:
        if not all_entries:
            print("\nNo new entries to merge - JSONL is already complete!")
            return 0

        merged_count = merge_into_jsonl(
            args.merge_into, all_entries, backup=not args.no_backup)
        print(f"\n=== Merge Complete ===")
        print(f"Added {merged_count} entries to {args.merge_into}")
        return 0

    # Write output file
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            for entry in all_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"Wrote {len(all_entries)} entries to {args.output}")

    # Summary
    print(f"\n=== Repair Summary ===")
    print(f"Export folder: {args.export_dir}")
    print(f"Root tree: {args.root_uri}")
    print(f"Trees found: {len(trees)}")
    print(f"Pearls found: {len(pearls)}")
    if args.output:
        print(f"Output: {args.output}")

    if trees:
        print(f"\nChild trees discovered:")
        for tree in trees[:5]:
            uri = tree.get('uri', '(needs URI)')
            print(f"  - {tree['raw_title']}: {uri}")
        if len(trees) > 5:
            print(f"  ... and {len(trees) - 5} more")


if __name__ == "__main__":
    main()
