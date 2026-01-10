#!/usr/bin/env python3
"""
Repair incomplete mindmaps by loading children from the SQLite index.

Reads the scan results and regenerates mindmaps that have missing children.

Usage:
    python3 scripts/repair_incomplete_mindmaps.py \
        --scan .local/data/scans/incomplete_mindmaps.json \
        --children-index .local/data/children_index.db \
        --output-dir output/mindmaps_curated \
        --data reports/pearltrees_targets_combined_2026-01-02_trees.jsonl

    # Dry run to see what would be repaired:
    python3 scripts/repair_incomplete_mindmaps.py --scan ... --dry-run
"""

import argparse
import json
import sqlite3
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import uuid
import hashlib


@dataclass
class ChildPearl:
    """A pearl child of a tree."""
    uri: str
    pearl_type: str
    title: str
    pos_order: int
    external_url: str = ""
    see_also_uri: str = ""


def load_children_from_index(db_path: Path, tree_id: str) -> List[ChildPearl]:
    """Load children for a tree from the SQLite index."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT pearl_type, title, pos_order, external_url, see_also_uri, uri
        FROM children
        WHERE parent_tree_id = ?
        ORDER BY pos_order
    ''', (str(tree_id),))

    children = []
    for row in cursor.fetchall():
        pearl_type, title, pos_order, external_url, see_also_uri, uri = row
        # Skip RootPearl
        if pearl_type == 'RootPearl':
            continue
        children.append(ChildPearl(
            uri=uri or '',
            pearl_type=pearl_type,
            title=title or '',
            pos_order=pos_order or 0,
            external_url=external_url or '',
            see_also_uri=see_also_uri or ''
        ))

    conn.close()
    return children


def generate_guid() -> str:
    """Generate a unique GUID for a topic."""
    return hashlib.md5(uuid.uuid4().bytes).hexdigest()[:22]


def extract_domain(url: str) -> str:
    """Extract domain from URL for display."""
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        # Strip common prefixes
        for prefix in ('www.', 'en.', 'docs.', 'm.'):
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
                break
        return domain
    except Exception:
        return ""


def generate_mindmap_xml(tree_id: str, title: str, uri: str,
                         children: List[ChildPearl], expanded_mode: bool = True) -> str:
    """Generate SimpleMind XML for a tree with children."""

    # Root element
    root = Element('simplemind-mindmaps')
    root.set('generator', 'UnifyWeaver')
    root.set('gen-version', '1.0.0')
    root.set('doc-version', '3')

    mindmap = SubElement(root, 'mindmap')

    # Meta
    meta = SubElement(mindmap, 'meta')
    guid_elem = SubElement(meta, 'guid')
    guid_elem.set('guid', hashlib.md5(tree_id.encode()).hexdigest().upper()[:32])
    title_elem = SubElement(meta, 'title')
    title_elem.set('text', title)
    style = SubElement(meta, 'style')
    style.set('key', 'system.soft-palette')
    numbering = SubElement(meta, 'auto-numbering')
    numbering.set('style', 'disabled')
    scroll = SubElement(meta, 'scrollstate')
    scroll.set('zoom', '40')
    scroll.set('x', '0')
    scroll.set('y', '0')
    central = SubElement(meta, 'main-centraltheme')
    central.set('id', '0')

    # Topics
    topics = SubElement(mindmap, 'topics')

    # Root topic
    root_topic = SubElement(topics, 'topic')
    root_topic.set('id', '0')
    root_topic.set('parent', '-1')
    root_topic.set('guid', generate_guid())
    root_topic.set('x', '0.00')
    root_topic.set('y', '0.00')
    root_topic.set('palette', '1')
    root_topic.set('colorinfo', '1')
    # Wrap title for display
    display_title = title.replace(' ', '\\N') if len(title) > 15 else title
    root_topic.set('text', display_title)

    layout = SubElement(root_topic, 'layout')
    layout.set('mode', 'radial')
    layout.set('direction', 'auto')
    layout.set('flow', 'default')

    style = SubElement(root_topic, 'style')
    font = SubElement(style, 'font')
    font.set('scale', '1.50')

    link = SubElement(root_topic, 'link')
    link.set('urllink', uri)

    # Child topics
    next_id = 1
    child_node_id = 1000
    palette_idx = 2

    for child in children:
        topic = SubElement(topics, 'topic')
        topic.set('id', str(next_id))
        topic.set('parent', '0')
        topic.set('guid', generate_guid())
        topic.set('x', '0.00')
        topic.set('y', '0.00')
        topic.set('palette', str((palette_idx % 8) + 1))
        topic.set('colorinfo', str((palette_idx % 8) + 1))

        # Wrap title
        child_title = child.title.replace(' ', '\\N') if len(child.title) > 12 else child.title
        topic.set('text', child_title)

        style = SubElement(topic, 'style')
        # PagePearl and Section get no border
        if child.pearl_type in ('PagePearl', 'SectionPearl'):
            style.set('borderstyle', 'sbsNone')

        # Link to Pearltrees URI
        link = SubElement(topic, 'link')
        link.set('urllink', child.uri)

        # For PagePearl in expanded mode, add domain child node
        if expanded_mode and child.pearl_type == 'PagePearl' and child.external_url:
            domain = extract_domain(child.external_url)
            if domain:
                child_topic = SubElement(topics, 'topic')
                child_topic.set('id', str(child_node_id))
                child_topic.set('parent', str(next_id))
                child_topic.set('guid', generate_guid())
                child_topic.set('x', '30.00')
                child_topic.set('y', '20.00')
                child_topic.set('text', domain)

                child_style = SubElement(child_topic, 'style')
                child_style.set('borderstyle', 'sbsNone')

                child_link = SubElement(child_topic, 'link')
                child_link.set('urllink', child.external_url)

                child_node_id += 1

        next_id += 1
        palette_idx += 1

    # Relations (empty)
    SubElement(mindmap, 'relations')

    # Convert to string with pretty printing
    rough_string = tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    pretty = reparsed.toprettyxml(indent="  ")

    # Clean up extra blank lines and fix declaration
    lines = [l for l in pretty.split('\n') if l.strip()]
    lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
    lines.insert(1, '<!DOCTYPE simplemind-mindmaps>')

    return '\n'.join(lines)


def write_smmx(xml_content: str, output_path: Path):
    """Write XML to .smmx file (zip format)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('document/mindmap.xml', xml_content)


def main():
    parser = argparse.ArgumentParser(description='Repair incomplete mindmaps')
    parser.add_argument('--scan', type=Path, required=True,
                        help='Path to scan results JSON')
    parser.add_argument('--children-index', type=Path, required=True,
                        help='Path to children index SQLite database')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Base output directory for mindmaps')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be repaired without writing files')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of repairs (for testing)')
    parser.add_argument('--expanded', action='store_true', default=True,
                        help='Use expanded mode for PagePearls (default: True)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Load scan results
    with open(args.scan) as f:
        scan_data = json.load(f)

    maps = scan_data.get('maps', [])
    print(f"Loaded {len(maps)} incomplete mindmaps from scan")

    if args.limit:
        maps = maps[:args.limit]
        print(f"Limited to {args.limit} maps for testing")

    # Stats
    repaired = 0
    skipped_no_children = 0
    skipped_private = 0
    errors = 0

    for i, m in enumerate(maps):
        tree_id = m.get('tree_id', '')
        title = m.get('title', 'Unknown')
        uri = m.get('uri', '')
        rel_path = m.get('path', '')

        # Skip private trees (optional)
        if title == '*private*':
            skipped_private += 1
            continue

        # Load children from index
        children = load_children_from_index(args.children_index, tree_id)

        if not children:
            skipped_no_children += 1
            if args.verbose:
                print(f"  [{i+1}/{len(maps)}] {tree_id}: No children in index, skipping")
            continue

        output_path = args.output_dir / rel_path

        if args.dry_run:
            print(f"  [{i+1}/{len(maps)}] Would repair: {tree_id} ({title[:30]}) - {len(children)} children -> {rel_path}")
            repaired += 1
            continue

        try:
            # Generate and write mindmap
            xml = generate_mindmap_xml(tree_id, title, uri, children,
                                       expanded_mode=args.expanded)
            write_smmx(xml, output_path)
            repaired += 1

            if args.verbose or (repaired % 100 == 0):
                print(f"  [{i+1}/{len(maps)}] Repaired: {tree_id} ({title[:30]}) - {len(children)} children")
        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(maps)}] Error repairing {tree_id}: {e}")

    # Summary
    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Repair Summary:")
    print(f"  Repaired: {repaired}")
    print(f"  Skipped (no children in index): {skipped_no_children}")
    print(f"  Skipped (private): {skipped_private}")
    print(f"  Errors: {errors}")
    print(f"  Total processed: {repaired + skipped_no_children + skipped_private + errors}")


if __name__ == '__main__':
    main()
