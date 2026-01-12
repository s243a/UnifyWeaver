#!/usr/bin/env python3
"""
Generate mindmaps organized by MST folder structure.

Each tree becomes a mindmap with its pearls as children, organized
into folders based on MST semantic clustering. Includes cloudmapref
links between related mindmaps in the same folder.

Usage:
    python3 scripts/mindmap/generate_mst_mindmaps.py \
        --mst-structure output/mst_folder_structure_trees.json \
        --embeddings datasets/pearltrees_combined_2026-01-02_all_fixed_embeddings.npz \
        --output output/mst_mindmaps/

    # Limit to first N trees for testing
    python3 scripts/mindmap/generate_mst_mindmaps.py \
        --mst-structure output/mst_folder_structure_trees.json \
        --limit 100 \
        --output output/mst_mindmaps/

    # With custom root folder name
    python3 scripts/mindmap/generate_mst_mindmaps.py \
        --mst-structure output/mst_folder_structure_trees.json \
        --root-name "Pearltrees_Collection" \
        --output output/mst_mindmaps/
"""

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

import numpy as np


def extract_parent_tree_id(uri: str) -> Optional[str]:
    """Extract parent tree ID from a pearl URI.

    Pearl URIs contain the parent tree ID before # or ?:
    - PagePearl: .../id2492374#item18082864
    - AliasPearl: .../id2492215#item18110176&show=item,18110176
    - RefPearl: .../id2492215?show=item,18049594

    Returns the numeric tree ID (without 'id' prefix) or None.
    """
    if not uri:
        return None

    # Remove fragment and query parts
    base_uri = uri.split('#')[0].split('?')[0]

    # Find id followed by digits
    match = re.search(r'/id(\d+)/?$', base_uri)
    if match:
        return match.group(1)

    return None


def extract_pearl_id(uri: str) -> Optional[str]:
    """Extract pearl ID from a pearl URI."""
    if not uri:
        return None

    # Look for #item or item, pattern
    match = re.search(r'#item(\d+)|item[,=](\d+)', uri)
    if match:
        return match.group(1) or match.group(2)

    return None


def load_tree_pearl_relationships(embeddings_path: Path) -> Tuple[Dict, Dict]:
    """Load embeddings and build tree->pearls mapping.

    Returns:
        tree_info: Dict[tree_id] -> {title, uri, type}
        tree_pearls: Dict[tree_id] -> List[{title, uri, type, external_url}]
    """
    print(f"Loading embeddings from {embeddings_path}...")
    data = np.load(embeddings_path, allow_pickle=True)

    titles = data['titles']
    uris = data['uris']
    item_types = data['item_types']
    tree_ids = data['tree_ids']

    # Build tree info
    tree_info = {}
    tree_mask = item_types == 'Tree'
    for i, is_tree in enumerate(tree_mask):
        if is_tree:
            tid = str(tree_ids[i])
            tree_info[tid] = {
                'title': str(titles[i]),
                'uri': str(uris[i]),
                'type': 'Tree'
            }

    # Build tree->pearls mapping
    tree_pearls = defaultdict(list)
    pearl_types = {'PagePearl', 'AliasPearl', 'RefPearl'}

    for i, item_type in enumerate(item_types):
        if str(item_type) in pearl_types:
            uri = str(uris[i])
            parent_tree_id = extract_parent_tree_id(uri)

            if parent_tree_id:
                pearl = {
                    'title': str(titles[i]),
                    'uri': uri,
                    'type': str(item_type),
                    'pearl_id': extract_pearl_id(uri),
                }
                tree_pearls[parent_tree_id].append(pearl)

    print(f"Found {len(tree_info)} trees")
    print(f"Found {sum(len(p) for p in tree_pearls.values())} pearls")

    return tree_info, dict(tree_pearls)


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """Sanitize a string for use as filename."""
    # Remove invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Replace whitespace with underscore
    name = re.sub(r'\s+', '_', name)
    # Truncate
    if len(name) > max_length:
        name = name[:max_length]
    return name or 'unnamed'


def create_mindmap_xml(tree_id: str, tree_info: Dict, pearls: List[Dict],
                       sibling_maps: List[Dict] = None) -> str:
    """Create SimpleMind XML for a tree with its pearls.

    Args:
        tree_id: The tree ID
        tree_info: Dict with title, uri, type for the tree
        pearls: List of pearl dicts with title, uri, type
        sibling_maps: List of sibling mindmaps for cloudmapref links
                      Each dict has: tree_id, title, relative_path

    Returns:
        XML string for the mindmap
    """
    sibling_maps = sibling_maps or []

    # Root element
    root = Element('simplemind-mindmaps')
    root.set('version', '3')
    root.set('creator', 'MST Folder Grouping')

    # Mindmap element
    mindmap = SubElement(root, 'mindmap')
    mindmap.set('guid', f"mst_{tree_id}")
    mindmap.set('version', '3')

    # Center topic (the tree)
    topics = SubElement(mindmap, 'topics')
    center = SubElement(topics, 'topic')
    center.set('id', '0')
    center.set('guid', f"tree_{tree_id}")

    # Center topic text
    text_elem = SubElement(center, 'text')
    text_elem.text = tree_info.get('title', f'Tree {tree_id}')

    # Note with metadata
    note = SubElement(center, 'note')
    note_lines = [
        f"Type: Tree",
        f"Tree ID: {tree_id}",
        f"URI: {tree_info.get('uri', '')}",
        f"Children: {len(pearls)}",
        f"Related: {len(sibling_maps)}"
    ]
    note.text = '\n'.join(note_lines)

    # Add pearls as child topics
    relations = SubElement(mindmap, 'relations')

    for i, pearl in enumerate(pearls):
        topic = SubElement(topics, 'topic')
        topic.set('id', str(i + 1))
        topic.set('guid', f"pearl_{pearl.get('pearl_id', i)}")

        # Topic text
        text_elem = SubElement(topic, 'text')
        text_elem.text = pearl.get('title', 'Untitled')

        # Note with metadata
        note = SubElement(topic, 'note')
        note_lines = [
            f"Type: {pearl.get('type', 'Unknown')}",
            f"URI: {pearl.get('uri', '')}",
        ]
        note.text = '\n'.join(note_lines)

        # Relation from center to pearl
        relation = SubElement(relations, 'relation')
        relation.set('topic1-id', '0')
        relation.set('topic2-id', str(i + 1))
        relation.set('direction', 'to')

    # Add sibling/related maps as linked topics with cloudmapref
    next_id = len(pearls) + 1
    for sibling in sibling_maps:
        if sibling.get('tree_id') == tree_id:
            continue  # Skip self

        topic = SubElement(topics, 'topic')
        topic.set('id', str(next_id))
        topic.set('guid', f"link_{sibling.get('tree_id', next_id)}")

        # Topic text (linked map title)
        text_elem = SubElement(topic, 'text')
        text_elem.text = f"→ {sibling.get('title', 'Related')}"

        # Link element with cloudmapref
        link = SubElement(topic, 'link')
        link.set('cloudmapref', sibling.get('relative_path', ''))

        # Note
        note = SubElement(topic, 'note')
        note.text = f"Related mindmap: {sibling.get('title', '')}\nTree ID: {sibling.get('tree_id', '')}"

        # Relation from center (dashed line for links)
        relation = SubElement(relations, 'relation')
        relation.set('topic1-id', '0')
        relation.set('topic2-id', str(next_id))
        relation.set('direction', 'to')
        relation.set('linestyle', 'rlsDashed')

        next_id += 1

    # Pretty print
    rough_string = tostring(root, encoding='unicode')
    parsed = minidom.parseString(rough_string)
    return parsed.toprettyxml(indent='  ', encoding=None)


def write_smmx(xml_content: str, output_path: Path):
    """Write XML to .smmx file (zip format)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('document/mindmap.xml', xml_content)


def process_circle(circle: Dict, tree_info: Dict, tree_pearls: Dict,
                   output_base: Path, path_parts: List[str],
                   stats: Dict, limit: Optional[int] = None,
                   root_name: str = None):
    """Recursively process a circle (folder) and generate mindmaps.

    Args:
        circle: Circle dict with name, items, children
        tree_info: Tree metadata dict
        tree_pearls: Tree->pearls mapping
        output_base: Base output directory
        path_parts: Current path components for folder hierarchy
        stats: Stats dict to update
        limit: Optional limit on number of mindmaps
        root_name: Override name for root folder
    """
    if limit and stats['generated'] >= limit:
        return

    # Create folder path from circle hierarchy
    circle_name = circle.get('name', 'unnamed')
    # Use root_name override for the first level
    if not path_parts and root_name:
        folder_name = sanitize_filename(root_name)
    else:
        folder_name = sanitize_filename(circle_name)

    current_path = path_parts + [folder_name]
    folder_path = output_base / '/'.join(current_path)

    # First pass: collect all items and compute filenames for cross-linking
    items = circle.get('items', [])
    sibling_maps = []

    for item in items:
        tree_id = str(item.get('tree_id', ''))
        if not tree_id:
            continue

        info = tree_info.get(tree_id, {'title': item.get('title', 'Unknown')})
        title = sanitize_filename(info.get('title', 'unnamed'))
        filename = f"{title}_id{tree_id}.smmx"

        sibling_maps.append({
            'tree_id': tree_id,
            'title': info.get('title', 'Unknown'),
            'relative_path': filename,  # Same folder, just filename
            'info': info,
        })

    # Second pass: generate mindmaps with cross-links
    for sibling in sibling_maps:
        if limit and stats['generated'] >= limit:
            return

        tree_id = sibling['tree_id']
        info = sibling['info']
        pearls = tree_pearls.get(tree_id, [])

        # Generate mindmap with sibling links
        xml = create_mindmap_xml(tree_id, info, pearls, sibling_maps)

        # Create output path
        output_path = folder_path / sibling['relative_path']

        write_smmx(xml, output_path)
        stats['generated'] += 1
        stats['total_pearls'] += len(pearls)
        stats['total_links'] += len(sibling_maps) - 1  # Exclude self

        if stats['generated'] % 500 == 0:
            print(f"Generated {stats['generated']} mindmaps...")

    # Process child circles
    for child in circle.get('children', []):
        process_circle(child, tree_info, tree_pearls, output_base,
                       current_path, stats, limit)


def main():
    parser = argparse.ArgumentParser(
        description='Generate mindmaps organized by MST folder structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--mst-structure', type=Path, required=True,
                        help='Path to MST folder structure JSON')
    parser.add_argument('--embeddings', type=Path,
                        default=Path('datasets/pearltrees_combined_2026-01-02_all_fixed_embeddings.npz'),
                        help='Path to embeddings file')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output directory for mindmaps')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of mindmaps to generate (for testing)')
    parser.add_argument('--root-name', type=str, default='Pearltrees_Collection',
                        help='Name for root folder (default: Pearltrees_Collection)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show what would be generated without writing files')

    args = parser.parse_args()

    # Load MST structure
    print(f"Loading MST structure from {args.mst_structure}...")
    with open(args.mst_structure) as f:
        mst_structure = json.load(f)

    # Load tree-pearl relationships
    tree_info, tree_pearls = load_tree_pearl_relationships(args.embeddings)

    # Generate mindmaps
    stats = {'generated': 0, 'total_pearls': 0, 'total_links': 0}

    if args.dry_run:
        print("\n[DRY RUN] Would generate mindmaps to:", args.output)
        # Just count
        def count_items(circle):
            count = len(circle.get('items', []))
            for child in circle.get('children', []):
                count += count_items(child)
            return count
        total = count_items(mst_structure)
        print(f"Would generate {total} mindmaps")
        print(f"Root folder: {args.root_name}")
    else:
        print(f"\nGenerating mindmaps to {args.output}...")
        print(f"Root folder: {args.root_name}")
        process_circle(mst_structure, tree_info, tree_pearls,
                       args.output, [], stats, args.limit, args.root_name)

        print(f"\nDone!")
        print(f"Generated {stats['generated']} mindmaps")
        print(f"Total pearls across all mindmaps: {stats['total_pearls']}")
        print(f"Total cloudmapref links: {stats['total_links']}")


if __name__ == '__main__':
    main()
