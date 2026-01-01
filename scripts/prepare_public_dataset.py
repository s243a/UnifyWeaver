#!/usr/bin/env python3
"""
Prepare Public Dataset with Privacy Propagation

Uses existing PearltreesParser for hierarchy, adds privacy propagation.

Usage:
    python scripts/prepare_public_dataset.py \
        --rdf context/PT/pearltrees_export_s243a_2025-12-27.rdf \
        --jsonl reports/pearltrees_targets_full_pearls.jsonl \
        --output-dir datasets/pearltrees_public
"""

import argparse
import json
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Set

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from pearltrees_target_generator import PearltreesParser, NAMESPACES


def extract_id_from_uri(uri: str) -> str:
    """Extract tree ID from Pearltrees URI."""
    if not uri:
        return ""
    if "/id" in uri:
        return uri.split("/id")[-1].split("#")[0].split("?")[0]
    return ""


def parse_rdf_with_privacy(rdf_path: Path) -> tuple:
    """Parse RDF using existing parser, add privacy extraction."""
    print(f"Parsing RDF: {rdf_path}")
    
    # Use existing parser for structure
    parser = PearltreesParser(rdf_path)
    parser.parse()
    
    # Extract privacy flags directly from XML
    tree = ET.parse(rdf_path)
    root = tree.getroot()
    
    privacy_map = {}  # uri -> privacy (0 or 1)
    
    for elem in root.findall(f".//{{{NAMESPACES['pt']}}}Tree"):
        uri = elem.get(f"{{{NAMESPACES['rdf']}}}about", "")
        privacy_elem = elem.find(f"{{{NAMESPACES['pt']}}}privacy")
        privacy = int(privacy_elem.text) if privacy_elem is not None else 0
        privacy_map[uri] = privacy
    
    # Build parent map from parser's trees
    parent_map = {}  # child_uri -> parent_uri
    for uri, tree_data in parser.trees.items():
        parent_uri = tree_data.get("parent_uri")
        if parent_uri:
            parent_map[uri] = parent_uri
    
    print(f"  Trees: {len(parser.trees)}")
    print(f"  Parent relationships: {len(parent_map)}")
    print(f"  Direct private: {sum(1 for p in privacy_map.values() if p == 1)}")
    
    return privacy_map, parent_map


def propagate_privacy(privacy_map: Dict[str, int], parent_map: Dict[str, str]) -> Set[str]:
    """Propagate privacy through tree hierarchy. Returns set of all private URIs."""
    private_uris = set()
    
    # Start with directly private trees
    for uri, privacy in privacy_map.items():
        if privacy == 1:
            private_uris.add(uri)
    
    # Build children map
    children_map = {}
    for child_uri, parent_uri in parent_map.items():
        if parent_uri not in children_map:
            children_map[parent_uri] = []
        children_map[parent_uri].append(child_uri)
    
    # BFS propagation
    queue = list(private_uris)
    while queue:
        current = queue.pop(0)
        for child in children_map.get(current, []):
            if child not in private_uris:
                private_uris.add(child)
                queue.append(child)
    
    # Convert URIs to IDs
    private_ids = set()
    for uri in private_uris:
        tree_id = extract_id_from_uri(uri)
        if tree_id:
            private_ids.add(tree_id)
    
    print(f"  Total private after propagation: {len(private_ids)}")
    return private_ids



def filter_jsonl(jsonl_path: Path, private_ids: Set[str], output_path: Path):
    """Filter JSONL to remove private items."""
    print(f"Filtering JSONL: {jsonl_path}")
    
    total = 0
    filtered = 0
    public_items = []
    
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            total += 1
            item = json.loads(line)
            tree_id = str(item.get("tree_id", ""))
            
            # Also check cluster_id for items without tree_id
            cluster_id = extract_id_from_uri(item.get("cluster_id", ""))
            
            if tree_id in private_ids or cluster_id in private_ids:
                filtered += 1
                continue
            
            # Check if title indicates private
            title = item.get("raw_title", "")
            if title == "*private*":
                filtered += 1
                continue
            
            # Check if URI contains /private/
            uri = item.get("uri", "")
            if "/private/" in uri:
                filtered += 1
                continue
            
            public_items.append(item)
    
    print(f"  Total: {total}, Filtered: {filtered}, Public: {len(public_items)}")
    
    # Save filtered JSONL
    with open(output_path, 'w') as f:
        for item in public_items:
            f.write(json.dumps(item) + '\n')
    
    return public_items


def save_sqlite(items, output_path: Path):
    """Save items to SQLite."""
    conn = sqlite3.connect(str(output_path))
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY,
        tree_id TEXT UNIQUE,
        raw_title TEXT,
        type TEXT,
        account TEXT,
        uri TEXT,
        cluster_id TEXT,
        target_text TEXT,
        query TEXT
    )''')
    
    c.execute('CREATE INDEX IF NOT EXISTS idx_tree_id ON items(tree_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_title ON items(raw_title)')
    
    for item in items:
        c.execute('''INSERT OR REPLACE INTO items 
            (tree_id, raw_title, type, account, uri, cluster_id, target_text, query)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (item.get('tree_id'), item.get('raw_title'), item.get('type'),
             item.get('account'), item.get('uri'), item.get('cluster_id'),
             item.get('target_text'), item.get('query')))
    
    conn.commit()
    conn.close()
    print(f"  SQLite: {output_path}")


def save_huggingface(items, output_path: Path):
    """Save as HuggingFace Dataset."""
    try:
        from datasets import Dataset
        dataset = Dataset.from_list(items)
        dataset.save_to_disk(str(output_path))
        parquet_path = output_path.parent / f"{output_path.name}.parquet"
        dataset.to_parquet(str(parquet_path))
        print(f"  HuggingFace: {output_path}")
    except ImportError:
        print("  Warning: 'datasets' not installed, skipping HuggingFace format")


def create_readme(output_dir: Path, stats: Dict):
    """Create README."""
    readme = f"""# Pearltrees Public Dataset

## Stats
- Total items: {stats['total']}
- Private filtered: {stats['filtered']}
- Public items: {stats['public']}

## Files
- `pearltrees_public.jsonl` (recommended)
- `pearltrees_public.db` (SQLite)
- `hf_dataset/` or `.parquet` (HuggingFace)

## Privacy Filtering
Items filtered based on:
1. RDF `<pt:privacy>1</pt:privacy>` flag
2. Children of private trees (cascaded)
3. Title = "*private*"
4. URI contains "/private/"
"""
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdf", type=Path, nargs="+",
                       default=[Path("context/PT/pearltrees_export_s243a_2025-12-27.rdf")])
    parser.add_argument("--jsonl", type=Path, 
                       default=Path("reports/pearltrees_targets_full_pearls.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                       default=Path("datasets/pearltrees_public"))
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse all RDF files and merge
    all_privacy = {}
    all_parents = {}
    
    for rdf_path in args.rdf:
        privacy, parents = parse_rdf_with_privacy(rdf_path)
        all_privacy.update(privacy)
        all_parents.update(parents)
    
    # Propagate privacy
    print("\nPropagating privacy...")
    private_ids = propagate_privacy(all_privacy, all_parents)
    
    # Filter JSONL
    print("\nFiltering...")
    jsonl_out = args.output_dir / "pearltrees_public.jsonl"
    public_items = filter_jsonl(args.jsonl, private_ids, jsonl_out)
    
    # Export formats
    print("\nExporting formats...")
    save_sqlite(public_items, args.output_dir / "pearltrees_public.db")
    save_huggingface(public_items, args.output_dir / "hf_dataset")
    
    # README
    stats = {'total': len(public_items) + len(private_ids), 
             'filtered': len(private_ids), 'public': len(public_items)}
    create_readme(args.output_dir, stats)
    
    print(f"\nDone! Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
