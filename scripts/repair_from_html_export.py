#!/usr/bin/env python3
"""
Repair/augment JSONL data from Pearltrees HTML (Netscape bookmark) exports.

HTML exports contain:
- Folder hierarchy (<H3> tags for trees)
- PagePearls (<A TARGET="_BLANK"> with external URLs but NO pearl_uri)
- AliasPearls (<A HREF="pearltrees.com/..."> with full Pearltrees URI)

This script can:
1. Extract AliasPearls with their full URIs
2. Build folder hierarchy (child folder names under each parent)
3. Match PagePearls by URL to find their pearl_uri from existing JSONL/RDF

Usage:
    # Extract what's available and compare with existing JSONL
    python3 scripts/repair_from_html_export.py \
        --html-export "context/PT/pearltrees_export_s243a_grous_2026-01-05.html" \
        --compare reports/pearltrees_targets_repaired.jsonl \
        --output /tmp/html_export_analysis.jsonl

    # Just analyze the HTML export structure
    python3 scripts/repair_from_html_export.py \
        --html-export "context/PT/pearltrees_export_s243a_grous_2026-01-05.html" \
        --analyze
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from html.parser import HTMLParser
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BookmarkItem:
    """Represents an item from the HTML export."""
    title: str
    url: str
    add_date: str = ""
    is_folder: bool = False
    is_pearltrees_link: bool = False
    parent_path: List[str] = field(default_factory=list)

    @property
    def pearltrees_uri(self) -> Optional[str]:
        """Extract Pearltrees URI if this is a Pearltrees link."""
        if self.is_pearltrees_link and 'pearltrees.com' in self.url:
            return self.url
        return None

    @property
    def tree_id(self) -> Optional[str]:
        """Extract tree ID from Pearltrees URI."""
        if not self.pearltrees_uri:
            return None
        match = re.search(r'/id(\d+)(?:/|$)', self.pearltrees_uri)
        return match.group(1) if match else None


class HierarchyBuilder:
    """Builds hybrid hierarchy with title paths and partial ID paths."""

    def __init__(self):
        # Map folder title to known tree ID (from AliasPearls)
        self.title_to_id: Dict[str, str] = {}
        # Map full title path tuple to known tree ID
        self.path_to_id: Dict[Tuple[str, ...], str] = {}

    def register_known_id(self, title: str, tree_id: str, parent_path: List[str] = None):
        """Register a known title -> tree_id mapping."""
        self.title_to_id[title] = tree_id
        if parent_path is not None:
            full_path = tuple(parent_path + [title])
            self.path_to_id[full_path] = tree_id

    def build_id_path_pattern(self, title_path: List[str]) -> str:
        """Build materialized ID path with regex for unknown parts.

        Args:
            title_path: List of folder titles from root to current

        Returns:
            Pattern like '/10311468/[^/]+/[^/]+/2492215'
        """
        parts = []
        for i, title in enumerate(title_path):
            # Try exact path match first
            path_tuple = tuple(title_path[:i+1])
            if path_tuple in self.path_to_id:
                parts.append(self.path_to_id[path_tuple])
            # Fall back to title-only match
            elif title in self.title_to_id:
                parts.append(self.title_to_id[title])
            else:
                # Unknown - use regex pattern
                parts.append('[^/]+')

        return '/' + '/'.join(parts) if parts else '/'

    def get_known_id(self, title: str, parent_path: List[str] = None) -> Optional[str]:
        """Get known ID for a title, preferring path-specific match."""
        if parent_path is not None:
            full_path = tuple(parent_path + [title])
            if full_path in self.path_to_id:
                return self.path_to_id[full_path]
        return self.title_to_id.get(title)


class NetscapeBookmarkParser(HTMLParser):
    """Parser for Netscape bookmark HTML format."""

    def __init__(self):
        super().__init__()
        self.items: List[BookmarkItem] = []
        self.folders: List[BookmarkItem] = []
        self.folder_stack: List[str] = []  # Current path of folder names
        self.current_attrs: Dict = {}
        self.in_h3 = False
        self.in_a = False
        self.current_text = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag == 'h3':
            self.in_h3 = True
            self.current_attrs = attrs_dict
            self.current_text = ""
        elif tag == 'a':
            self.in_a = True
            self.current_attrs = attrs_dict
            self.current_text = ""
        elif tag == 'dl':
            # Entering a new folder's content
            pass

    def handle_endtag(self, tag):
        if tag == 'h3' and self.in_h3:
            # Folder/Tree
            folder = BookmarkItem(
                title=self.current_text.strip(),
                url="",
                add_date=self.current_attrs.get('add_date', ''),
                is_folder=True,
                parent_path=self.folder_stack.copy()
            )
            self.folders.append(folder)
            self.folder_stack.append(folder.title)
            self.in_h3 = False

        elif tag == 'a' and self.in_a:
            # Bookmark link
            url = self.current_attrs.get('href', '')
            is_pt = 'pearltrees.com' in url and 'TARGET' not in str(self.current_attrs).upper()

            # Check if it's a Pearltrees internal link (no TARGET="_BLANK")
            has_target_blank = self.current_attrs.get('target', '').upper() == '_BLANK'
            is_pt = 'pearltrees.com' in url and not has_target_blank

            item = BookmarkItem(
                title=self.current_text.strip(),
                url=url,
                add_date=self.current_attrs.get('add_date', ''),
                is_folder=False,
                is_pearltrees_link=is_pt,
                parent_path=self.folder_stack.copy()
            )
            self.items.append(item)
            self.in_a = False

        elif tag == 'dl':
            # Exiting a folder
            if self.folder_stack:
                self.folder_stack.pop()

    def handle_data(self, data):
        if self.in_h3 or self.in_a:
            self.current_text += data


def parse_html_export(html_path: Path) -> Tuple[List[BookmarkItem], List[BookmarkItem]]:
    """Parse HTML export file.

    Returns:
        (items, folders): Lists of bookmark items and folders
    """
    content = html_path.read_text(encoding='utf-8', errors='ignore')

    parser = NetscapeBookmarkParser()
    parser.feed(content)

    return parser.items, parser.folders


def load_existing_jsonl(jsonl_path: Path) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Load existing JSONL and build indices.

    Returns:
        (by_url, by_uri): Indices for lookup by URL and URI
    """
    by_url = {}  # source URL -> entry
    by_uri = {}  # pearltrees URI -> entry

    if not jsonl_path.exists():
        return by_url, by_uri

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Index by URL (for PagePearls)
            url = entry.get('url', '')
            if url and url != 'null':
                by_url[url] = entry

            # Index by URI (for trees and pearls)
            uri = entry.get('uri') or entry.get('pearl_uri', '')
            if uri:
                by_uri[uri] = entry

    return by_url, by_uri


def analyze_export(items: List[BookmarkItem], folders: List[BookmarkItem]):
    """Print analysis of the HTML export."""

    pt_links = [i for i in items if i.is_pearltrees_link]
    external_links = [i for i in items if not i.is_pearltrees_link]

    print(f"\n=== HTML Export Analysis ===")
    print(f"Total folders (Trees):     {len(folders)}")
    print(f"Total bookmarks:           {len(items)}")
    print(f"  - Pearltrees links:      {len(pt_links)} (have URI)")
    print(f"  - External links:        {len(external_links)} (no pearl_uri)")

    # Show folder depth distribution
    max_depth = max((len(f.parent_path) for f in folders), default=0)
    print(f"\nMax folder depth: {max_depth}")

    # Show sample Pearltrees links
    if pt_links:
        print(f"\nSample Pearltrees links (AliasPearls):")
        for item in pt_links[:5]:
            path = " > ".join(item.parent_path[-3:]) if item.parent_path else "(root)"
            print(f"  - {item.title}")
            print(f"    URI: {item.url}")
            print(f"    Path: {path}")

    # Show top-level folders
    top_folders = [f for f in folders if len(f.parent_path) == 0]
    if top_folders:
        print(f"\nTop-level folders ({len(top_folders)}):")
        for f in top_folders[:10]:
            print(f"  - {f.title}")
        if len(top_folders) > 10:
            print(f"  ... and {len(top_folders) - 10} more")


def match_with_existing(
    items: List[BookmarkItem],
    folders: List[BookmarkItem],
    by_url: Dict[str, Dict],
    by_uri: Dict[str, Dict]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Match HTML export items with existing JSONL data.

    Returns:
        (matched_items, unmatched_items, alias_pearls_with_uri)
    """
    matched = []
    unmatched = []
    alias_pearls = []

    for item in items:
        if item.is_pearltrees_link:
            # This is an AliasPearl with full URI
            alias_entry = {
                'type': 'AliasPearl',
                'raw_title': item.title,
                'alias_target_uri': item.url,
                'parent_path': item.parent_path,
                '_source': 'html_export'
            }

            # Check if target exists in JSONL
            if item.url in by_uri:
                alias_entry['_target_exists'] = True
                alias_entry['_target_type'] = by_uri[item.url].get('type')
            else:
                alias_entry['_target_exists'] = False

            alias_pearls.append(alias_entry)
        else:
            # External link - try to match by URL
            if item.url in by_url:
                existing = by_url[item.url]
                matched.append({
                    'html_title': item.title,
                    'html_parent_path': item.parent_path,
                    'existing_entry': existing
                })
            else:
                unmatched.append({
                    'title': item.title,
                    'url': item.url,
                    'parent_path': item.parent_path,
                    '_source': 'html_export'
                })

    return matched, unmatched, alias_pearls


def build_hierarchy_with_patterns(
    items: List[BookmarkItem],
    folders: List[BookmarkItem],
    existing_by_uri: Dict[str, Dict] = None
) -> Tuple[HierarchyBuilder, List[Dict]]:
    """Build hierarchy with title paths and ID path patterns.

    Uses AliasPearls to learn title->ID mappings, then generates
    entries with hybrid paths for all folders.

    Returns:
        (hierarchy_builder, folder_entries)
    """
    hierarchy = HierarchyBuilder()

    # First pass: learn ID mappings from AliasPearls (items pointing to PT)
    for item in items:
        if item.is_pearltrees_link and item.tree_id:
            # This AliasPearl tells us the ID of the tree it points to
            hierarchy.register_known_id(
                title=item.title,
                tree_id=item.tree_id,
                parent_path=item.parent_path
            )

    # Also learn from existing JSONL if provided
    if existing_by_uri:
        for uri, entry in existing_by_uri.items():
            title = entry.get('raw_title', '')
            tree_id = entry.get('tree_id', '')
            if title and tree_id:
                hierarchy.title_to_id[title] = tree_id

    # Second pass: generate folder entries with hybrid paths
    folder_entries = []
    for folder in folders:
        full_path = folder.parent_path + [folder.title]
        id_pattern = hierarchy.build_id_path_pattern(full_path)
        known_id = hierarchy.get_known_id(folder.title, folder.parent_path)

        entry = {
            'type': 'Tree',
            'raw_title': folder.title,
            'title_path': full_path,
            'id_path_pattern': id_pattern,
            '_source': 'html_export',
        }

        if known_id:
            entry['tree_id'] = known_id
            entry['_id_source'] = 'alias_pearl'
        else:
            entry['_needs_id'] = True

        # Count known vs unknown parts
        parts = id_pattern.split('/')[1:]  # Skip empty first part
        known_count = sum(1 for p in parts if p != '[^/]+')
        entry['_known_ids'] = known_count
        entry['_unknown_ids'] = len(parts) - known_count

        folder_entries.append(entry)

    return hierarchy, folder_entries


def generate_target_text_with_pattern(entry: Dict) -> str:
    """Generate target_text using title path and ID pattern.

    Format:
        /id_pattern
        - title1
          - title2
            - title3
    """
    title_path = entry.get('title_path', [])
    id_pattern = entry.get('id_path_pattern', '')

    lines = [id_pattern]
    for i, title in enumerate(title_path):
        indent = '  ' * i
        lines.append(f"{indent}- {title}")

    return '\n'.join(lines)


def extract_missing_trees(alias_pearls: List[Dict]) -> List[Dict]:
    """Extract unique missing trees from AliasPearls that target non-existent entries.

    Returns list of Tree entries that can be added to JSONL.
    """
    seen_uris = set()
    missing_trees = []

    for alias in alias_pearls:
        if alias.get('_target_exists'):
            continue

        uri = alias.get('alias_target_uri', '')
        if not uri or uri in seen_uris:
            continue

        # Skip placeholder/broken links
        if '/id' not in uri or uri == 'https://www.pearltrees.com/ooops':
            continue

        seen_uris.add(uri)

        # Parse URI to extract tree info
        # Format: https://www.pearltrees.com/[t/]account/tree-name/id12345
        match = re.match(
            r'https://www\.pearltrees\.com/(?:t/)?([^/]+)/(?:([^/]+)/)?id(\d+)',
            uri
        )

        if not match:
            continue

        account = match.group(1)
        tree_name = match.group(2) or ''
        tree_id = match.group(3)

        tree_entry = {
            'type': 'Tree',
            'raw_title': alias.get('raw_title', tree_name),
            'tree_id': tree_id,
            'uri': uri,
            'account': account,
            '_source': 'html_export_alias_target',
            '_discovered_from_path': alias.get('parent_path', [])
        }

        missing_trees.append(tree_entry)

    return missing_trees


def main():
    parser = argparse.ArgumentParser(
        description="Parse Pearltrees HTML exports")
    parser.add_argument("--html-export", type=Path, required=True,
                       help="Path to HTML export file")
    parser.add_argument("--compare", type=Path, default=None,
                       help="Compare with existing JSONL")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output JSONL file for extracted/matched data")
    parser.add_argument("--extract-missing-trees", type=Path, default=None,
                       help="Extract missing trees from AliasPearls to JSONL")
    parser.add_argument("--build-hierarchy", type=Path, default=None,
                       help="Build hybrid hierarchy with title paths and ID patterns")
    parser.add_argument("--analyze", action="store_true",
                       help="Just analyze the export structure")

    args = parser.parse_args()

    if not args.html_export.exists():
        logger.error(f"HTML file not found: {args.html_export}")
        return 1

    logger.info(f"Parsing: {args.html_export}")
    items, folders = parse_html_export(args.html_export)

    logger.info(f"Found {len(items)} items, {len(folders)} folders")

    if args.analyze:
        analyze_export(items, folders)
        return 0

    # Compare with existing JSONL
    if args.compare:
        by_url, by_uri = load_existing_jsonl(args.compare)
        logger.info(f"Loaded {len(by_url)} URLs, {len(by_uri)} URIs from {args.compare}")

        matched, unmatched, alias_pearls = match_with_existing(
            items, folders, by_url, by_uri)

        print(f"\n=== Comparison Results ===")
        print(f"PagePearls matched by URL:    {len(matched)}")
        print(f"PagePearls unmatched:         {len(unmatched)}")
        print(f"AliasPearls with URI:         {len(alias_pearls)}")

        # Check alias targets
        targets_exist = sum(1 for a in alias_pearls if a.get('_target_exists'))
        print(f"  - Target exists in JSONL:   {targets_exist}")
        print(f"  - Target missing:           {len(alias_pearls) - targets_exist}")

        # Write output if requested
        if args.output:
            all_entries = alias_pearls + unmatched
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                for entry in all_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"Wrote {len(all_entries)} entries to {args.output}")

        # Show sample unmatched
        if unmatched:
            print(f"\nSample unmatched PagePearls:")
            for item in unmatched[:5]:
                path = " > ".join(item['parent_path'][-3:]) if item['parent_path'] else "(root)"
                print(f"  - {item['title']}")
                print(f"    URL: {item['url'][:80]}...")
                print(f"    Path: {path}")

        # Extract missing trees if requested
        if args.extract_missing_trees:
            missing_trees = extract_missing_trees(alias_pearls)
            args.extract_missing_trees.parent.mkdir(parents=True, exist_ok=True)
            with open(args.extract_missing_trees, 'w', encoding='utf-8') as f:
                for tree in missing_trees:
                    f.write(json.dumps(tree, ensure_ascii=False) + '\n')
            print(f"\n=== Missing Trees Extracted ===")
            print(f"Wrote {len(missing_trees)} missing trees to {args.extract_missing_trees}")

            # Show sample
            if missing_trees:
                print(f"\nSample missing trees:")
                for tree in missing_trees[:10]:
                    print(f"  - {tree['raw_title']} ({tree['account']})")
                    print(f"    URI: {tree['uri']}")

        # Build hybrid hierarchy if requested
        if args.build_hierarchy:
            hierarchy, folder_entries = build_hierarchy_with_patterns(
                items, folders, by_uri)

            # Add target_text to each entry
            for entry in folder_entries:
                entry['target_text'] = generate_target_text_with_pattern(entry)

            # Write output
            args.build_hierarchy.parent.mkdir(parents=True, exist_ok=True)
            with open(args.build_hierarchy, 'w', encoding='utf-8') as f:
                for entry in folder_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            # Stats
            known_count = sum(1 for e in folder_entries if not e.get('_needs_id'))
            unknown_count = sum(1 for e in folder_entries if e.get('_needs_id'))
            fully_resolved = sum(1 for e in folder_entries if e.get('_unknown_ids', 0) == 0)

            print(f"\n=== Hybrid Hierarchy Built ===")
            print(f"Total folders:           {len(folder_entries)}")
            print(f"With known tree_id:      {known_count}")
            print(f"Without tree_id:         {unknown_count}")
            print(f"Fully resolved ID paths: {fully_resolved}")
            print(f"Output: {args.build_hierarchy}")

            # Show samples at different resolution levels
            print(f"\nSample entries:")
            # Fully resolved
            resolved = [e for e in folder_entries if e.get('_unknown_ids', 0) == 0][:2]
            if resolved:
                print(f"\n  Fully resolved ({len([e for e in folder_entries if e.get('_unknown_ids', 0) == 0])}):")
                for e in resolved:
                    print(f"    {e['raw_title']}")
                    print(f"      ID pattern: {e['id_path_pattern']}")

            # Partially resolved
            partial = [e for e in folder_entries
                      if 0 < e.get('_unknown_ids', 0) < len(e.get('title_path', []))][:2]
            if partial:
                print(f"\n  Partially resolved ({len([e for e in folder_entries if 0 < e.get('_unknown_ids', 0) < len(e.get('title_path', []))])}):")
                for e in partial:
                    print(f"    {e['raw_title']}")
                    print(f"      ID pattern: {e['id_path_pattern']}")
                    print(f"      Title path: {' > '.join(e['title_path'][-4:])}")

            # Unresolved
            unresolved = [e for e in folder_entries
                         if e.get('_unknown_ids', 0) == len(e.get('title_path', []))][:2]
            if unresolved:
                print(f"\n  Unresolved ({len([e for e in folder_entries if e.get('_unknown_ids', 0) == len(e.get('title_path', []))])}):")
                for e in unresolved:
                    print(f"    {e['raw_title']}")
                    print(f"      Title path: {' > '.join(e['title_path'][-4:])}")

    else:
        analyze_export(items, folders)


if __name__ == "__main__":
    main()
