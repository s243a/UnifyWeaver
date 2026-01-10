#!/usr/bin/env python3
"""
Build a parent→children index from Pearltrees RDF.

Groups all pearls (PagePearl, RefPearl, AliasPearl, SectionPearl, RootPearl)
by their parentTree and stores in SQLite or JSON for fast lookup.

Usage:
    python3 scripts/build_children_index.py context/PT/pearltrees_export_s243a_2026-01-02.rdf
    python3 scripts/build_children_index.py context/PT/*.rdf --output .local/data/children_index.db
    python3 scripts/build_children_index.py context/PT/*.rdf --format json --output .local/data/children_index.json
"""

import argparse
import json
import sqlite3
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

NAMESPACES = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'pt': 'http://www.pearltrees.com/rdf/0.1/#',
    'dcterms': 'http://purl.org/dc/elements/1.1/',
}


@dataclass
class ChildPearl:
    """Represents a pearl that is a child of a tree."""
    uri: str
    pearl_type: str  # PagePearl, RefPearl, AliasPearl, SectionPearl, RootPearl
    title: str
    pos_order: int
    external_url: Optional[str] = None  # For PagePearl
    see_also_uri: Optional[str] = None  # For RefPearl/AliasPearl - target tree


def extract_tree_id(uri: str) -> Optional[str]:
    """Extract tree ID from URI like https://www.pearltrees.com/s243a/foo/id12345"""
    match = re.search(r'id(\d+)', uri)
    return match.group(1) if match else None


def parse_rdf_children(rdf_file: Path) -> Dict[str, List[ChildPearl]]:
    """
    Parse RDF file and group children by parent tree.

    Returns:
        Dict mapping parent_tree_uri -> list of ChildPearl
    """
    children_by_parent: Dict[str, List[ChildPearl]] = defaultdict(list)

    try:
        tree = ET.parse(rdf_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {rdf_file}: {e}")
        return children_by_parent

    pearl_types = ['PagePearl', 'RefPearl', 'AliasPearl', 'SectionPearl', 'RootPearl']

    for pearl_type in pearl_types:
        for elem in root.findall(f".//{{{NAMESPACES['pt']}}}{pearl_type}"):
            uri = elem.get(f"{{{NAMESPACES['rdf']}}}about")
            if not uri:
                continue

            # Get parent tree
            parent_elem = elem.find(f"{{{NAMESPACES['pt']}}}parentTree")
            if parent_elem is None:
                continue
            parent_uri = parent_elem.get(f"{{{NAMESPACES['rdf']}}}resource")
            if not parent_uri:
                continue

            # Get title
            title_elem = elem.find(f"{{{NAMESPACES['dcterms']}}}title")
            title = title_elem.text if title_elem is not None and title_elem.text else ""

            # Get position
            pos_elem = elem.find(f"{{{NAMESPACES['pt']}}}posOrder")
            pos_order = int(pos_elem.text) if pos_elem is not None and pos_elem.text else 0

            # Get external URL for PagePearl
            external_url = None
            if pearl_type == 'PagePearl':
                id_elem = elem.find(f"{{{NAMESPACES['dcterms']}}}identifier")
                external_url = id_elem.text if id_elem is not None and id_elem.text else None

            # Get seeAlso for RefPearl/AliasPearl
            see_also_uri = None
            if pearl_type in ('RefPearl', 'AliasPearl'):
                see_also = elem.find(f"{{{NAMESPACES['rdfs']}}}seeAlso")
                if see_also is not None:
                    see_also_uri = see_also.get(f"{{{NAMESPACES['rdf']}}}resource")

            child = ChildPearl(
                uri=uri,
                pearl_type=pearl_type,
                title=title,
                pos_order=pos_order,
                external_url=external_url,
                see_also_uri=see_also_uri
            )

            children_by_parent[parent_uri].append(child)

    # Sort children by pos_order
    for parent_uri in children_by_parent:
        children_by_parent[parent_uri].sort(key=lambda c: c.pos_order)

    return children_by_parent


def save_to_sqlite(children_by_parent: Dict[str, List[ChildPearl]], db_path: Path):
    """Save children index to SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS children (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_tree_uri TEXT NOT NULL,
            parent_tree_id TEXT,
            uri TEXT NOT NULL,
            pearl_type TEXT NOT NULL,
            title TEXT,
            pos_order INTEGER,
            external_url TEXT,
            see_also_uri TEXT,
            UNIQUE(parent_tree_uri, uri)
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_parent_tree_uri ON children(parent_tree_uri)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_parent_tree_id ON children(parent_tree_id)')

    # Insert data
    for parent_uri, children in children_by_parent.items():
        parent_id = extract_tree_id(parent_uri)
        for child in children:
            cursor.execute('''
                INSERT OR REPLACE INTO children
                (parent_tree_uri, parent_tree_id, uri, pearl_type, title, pos_order, external_url, see_also_uri)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                parent_uri,
                parent_id,
                child.uri,
                child.pearl_type,
                child.title,
                child.pos_order,
                child.external_url,
                child.see_also_uri
            ))

    conn.commit()

    # Stats
    cursor.execute('SELECT COUNT(DISTINCT parent_tree_uri) FROM children')
    parent_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM children')
    child_count = cursor.fetchone()[0]

    conn.close()

    print(f"Saved {child_count} children across {parent_count} parent trees to {db_path}")


def save_to_json(children_by_parent: Dict[str, List[ChildPearl]], json_path: Path):
    """Save children index to JSON file."""
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = {}
    for parent_uri, children in children_by_parent.items():
        parent_id = extract_tree_id(parent_uri)
        data[parent_uri] = {
            'parent_id': parent_id,
            'children': [asdict(c) for c in children]
        }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} parent trees to {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Build parent→children index from RDF')
    parser.add_argument('rdf_files', type=Path, nargs='+', help='RDF files to parse')
    parser.add_argument('--output', '-o', type=Path,
                        default=Path('.local/data/children_index.db'),
                        help='Output file path')
    parser.add_argument('--format', choices=['sqlite', 'json'], default='sqlite',
                        help='Output format (default: sqlite)')

    args = parser.parse_args()

    # Collect children from all RDF files
    all_children: Dict[str, List[ChildPearl]] = defaultdict(list)

    for rdf_file in args.rdf_files:
        if not rdf_file.exists():
            print(f"Warning: {rdf_file} not found, skipping")
            continue

        print(f"Parsing {rdf_file}...")
        children = parse_rdf_children(rdf_file)

        for parent_uri, child_list in children.items():
            all_children[parent_uri].extend(child_list)

    # Remove duplicates and re-sort
    for parent_uri in all_children:
        seen = set()
        unique = []
        for child in all_children[parent_uri]:
            if child.uri not in seen:
                seen.add(child.uri)
                unique.append(child)
        unique.sort(key=lambda c: c.pos_order)
        all_children[parent_uri] = unique

    # Save
    if args.format == 'sqlite':
        save_to_sqlite(all_children, args.output)
    else:
        save_to_json(all_children, args.output)


if __name__ == '__main__':
    main()
