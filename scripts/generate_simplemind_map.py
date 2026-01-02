#!/usr/bin/env python3
"""
Generate SimpleMind mind maps from Pearltrees clusters.

Uses radial layout with hierarchical structure based on semantic similarity.
Output is a .smmx file (zip containing mindmap.xml).

Usage:
    python3 scripts/generate_simplemind_map.py \
        --cluster "Poles and Zeros" \
        --data reports/pearltrees_targets_full_multi_account.jsonl \
        --output output/poles_zeros.smmx

    # Or by cluster URL:
    python3 scripts/generate_simplemind_map.py \
        --cluster-url "https://www.pearltrees.com/s243a/poles-zeros-complex-numbers/id11563630" \
        --output output/poles_zeros.smmx
"""

import argparse
import json
import math
import uuid
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import numpy as np

@dataclass
class MindMapNode:
    """A node in the mind map."""
    id: int
    title: str
    tree_id: str = ""
    url: str = ""
    parent_id: int = -1
    x: float = 0.0
    y: float = 0.0
    palette: int = 1
    children: List['MindMapNode'] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

def generate_guid() -> str:
    """Generate a SimpleMind-style GUID."""
    return uuid.uuid4().hex.upper()[:32]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def build_hierarchy(nodes: List[MindMapNode], root_idx: int = 0) -> MindMapNode:
    """
    Build hierarchical tree from flat list using similarity.

    Algorithm:
    1. Root is the first node (usually the folder itself)
    2. Find N closest nodes to root -> direct children
    3. For remaining nodes, assign to most similar child
    4. Recurse for each child
    """
    if len(nodes) <= 1:
        return nodes[0] if nodes else None

    root = nodes[root_idx]
    others = [n for i, n in enumerate(nodes) if i != root_idx]

    if not others:
        return root

    # For small clusters, all nodes are direct children of root
    if len(others) <= 6:
        root.children = others
        for child in others:
            child.parent_id = root.id
        return root

    # For larger clusters, use similarity to build hierarchy
    if root.embedding is not None:
        # Sort by similarity to root
        similarities = [(n, cosine_similarity(root.embedding, n.embedding)) for n in others]
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Top N become direct children
        n_direct = min(5, len(others))
        direct_children = [s[0] for s in similarities[:n_direct]]
        remaining = [s[0] for s in similarities[n_direct:]]

        # Assign remaining to most similar direct child
        for node in remaining:
            best_child = max(direct_children,
                           key=lambda c: cosine_similarity(c.embedding, node.embedding))
            best_child.children.append(node)
            node.parent_id = best_child.id

        root.children = direct_children
        for child in direct_children:
            child.parent_id = root.id
    else:
        # No embeddings - just make all direct children
        root.children = others
        for child in others:
            child.parent_id = root.id

    return root

def wrap_title(title: str, target_ratio: float = 2.0) -> str:
    """
    Wrap title with \\N line breaks for rounder nodes.

    Targets a width:height ratio (default 2:1).
    For ratio r with n chars: optimal lines k = sqrt(n/r)
    """
    words = title.split()
    if len(words) <= 1:
        return title

    n = len(title)

    # Calculate optimal number of lines for target ratio
    # width/height = ratio, width ≈ n/k, height = k
    # (n/k)/k = ratio → k = sqrt(n/ratio)
    optimal_lines = max(1, int(round(math.sqrt(n / target_ratio))))

    if optimal_lines <= 1:
        return title

    # Target chars per line
    target_per_line = n / optimal_lines

    # Greedily build lines trying to stay close to target width
    lines = []
    current_line = []
    current_len = 0

    for word in words:
        word_len = len(word)
        # Would adding this word exceed target? Start new line if we have content
        if current_line and (current_len + 1 + word_len) > target_per_line * 1.3:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_len = word_len
        else:
            if current_line:
                current_len += 1  # space
            current_line.append(word)
            current_len += word_len

    # Don't forget last line
    if current_line:
        lines.append(' '.join(current_line))

    return '\\N'.join(lines)

def apply_radial_layout(node: MindMapNode, center_x: float = 500, center_y: float = 500,
                        radius: float = 300, start_angle: float = 0, depth: int = 0):
    """
    Apply radial layout to node and its children.

    Each level of children is arranged in a circle around the parent.
    Distance from parent is constant for all children at same level.
    """
    node.x = center_x
    node.y = center_y

    if not node.children:
        return

    n_children = len(node.children)
    angle_step = 2 * math.pi / n_children

    # Reduce radius for deeper levels
    child_radius = radius * (0.7 ** depth)

    for i, child in enumerate(node.children):
        angle = start_angle + i * angle_step
        child_x = center_x + child_radius * math.cos(angle)
        child_y = center_y + child_radius * math.sin(angle)

        # Recurse with offset angle to avoid overlap
        apply_radial_layout(child, child_x, child_y,
                           radius=child_radius * 0.8,
                           start_angle=angle - math.pi/4,
                           depth=depth + 1)

def node_to_xml(node: MindMapNode, parent_element: Element):
    """Convert a MindMapNode to SimpleMind XML topic element."""
    topic = SubElement(parent_element, 'topic')
    topic.set('id', str(node.id))
    topic.set('parent', str(node.parent_id))
    topic.set('guid', generate_guid())
    topic.set('x', f'{node.x:.2f}')
    topic.set('y', f'{node.y:.2f}')
    topic.set('palette', str(node.palette))
    topic.set('colorinfo', str(node.palette))

    # Wrap title for rounder nodes
    # Don't manually escape & - ElementTree handles XML escaping
    wrapped = wrap_title(node.title)
    text = wrapped.replace('\n', '\\N')
    topic.set('text', text)

    # Add link if URL present
    if node.url:
        link = SubElement(topic, 'link')
        link.set('urllink', node.url)

    # Recurse for children
    for child in node.children:
        node_to_xml(child, parent_element)

def generate_mindmap_xml(root: MindMapNode, title: str) -> str:
    """Generate complete SimpleMind XML document."""
    # Root element
    root_elem = Element('simplemind-mindmaps')
    root_elem.set('generator', 'UnifyWeaver')
    root_elem.set('gen-version', '1.0.0')
    root_elem.set('doc-version', '3')

    mindmap = SubElement(root_elem, 'mindmap')

    # Meta section
    meta = SubElement(mindmap, 'meta')
    guid = SubElement(meta, 'guid')
    guid.set('guid', generate_guid())
    title_elem = SubElement(meta, 'title')
    title_elem.set('text', title.replace('&', '&amp;'))
    style = SubElement(meta, 'style')
    style.set('key', 'system.soft-palette')
    auto_num = SubElement(meta, 'auto-numbering')
    auto_num.set('style', 'disabled')
    scroll = SubElement(meta, 'scrollstate')
    scroll.set('zoom', '70')
    scroll.set('x', str(int(-root.x)))
    scroll.set('y', str(int(-root.y)))
    main_theme = SubElement(meta, 'main-centraltheme')
    main_theme.set('id', '0')

    # Topics section
    topics = SubElement(mindmap, 'topics')
    node_to_xml(root, topics)

    # Relations section (empty for now)
    SubElement(mindmap, 'relations')

    # Pretty print
    xml_str = tostring(root_elem, encoding='unicode')
    parsed = minidom.parseString(xml_str)
    pretty = parsed.toprettyxml(indent='  ')

    # Remove extra blank lines and fix XML declaration
    lines = [l for l in pretty.split('\n') if l.strip()]
    lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
    lines.insert(1, '<!DOCTYPE simplemind-mindmaps>')

    return '\n'.join(lines)

def load_cluster_items(data_path: Path, cluster_query: str = None,
                       cluster_url: str = None) -> List[Dict]:
    """Load items from a cluster."""
    items = []
    folder_item = None

    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)

            # Match by cluster URL - find children
            if cluster_url and rec.get('cluster_id') == cluster_url:
                items.append(rec)
            # Also find the folder itself (where URI = cluster_url)
            if cluster_url and rec.get('uri') == cluster_url:
                folder_item = rec
            # Match by title search
            if cluster_query and cluster_query.lower() in rec.get('raw_title', '').lower():
                items.append(rec)

    # Add folder item if found and not already in items
    if folder_item and folder_item not in items:
        items.insert(0, folder_item)

    # If searching by title, get the cluster_id of the first match and find all items
    if cluster_query and items:
        # Find the folder itself
        folder = next((i for i in items if cluster_query.lower() in i.get('raw_title', '').lower()), None)
        if folder:
            folder_url = folder.get('uri', '')
            # Re-scan for all items in this cluster
            items = [folder]
            with open(data_path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get('cluster_id') == folder_url:
                        items.append(rec)

    return items

def create_nodes_from_items(items: List[Dict]) -> List[MindMapNode]:
    """Convert cluster items to MindMapNodes."""
    nodes = []

    # Find the root (the folder itself - usually has cluster_id == uri)
    root_item = None
    for item in items:
        if item.get('uri') == item.get('cluster_id') or item.get('type') == 'Tree':
            root_item = item
            break

    # If no clear root, use first item
    if not root_item and items:
        root_item = items[0]

    # Create root node
    if root_item:
        root_node = MindMapNode(
            id=0,
            title=root_item.get('raw_title', 'Root'),
            tree_id=root_item.get('tree_id', ''),
            url=root_item.get('uri', ''),
            parent_id=-1,
            palette=1
        )
        nodes.append(root_node)

    # Create child nodes
    palette_idx = 2
    for i, item in enumerate(items):
        if item == root_item:
            continue

        node = MindMapNode(
            id=len(nodes),
            title=item.get('raw_title', f'Item {i}'),
            tree_id=item.get('tree_id', ''),
            url=item.get('uri', ''),
            parent_id=0,  # Will be updated by build_hierarchy
            palette=(palette_idx % 8) + 1
        )
        nodes.append(node)
        palette_idx += 1

    return nodes

def write_smmx(xml_content: str, output_path: Path):
    """Write XML to .smmx file (zip format)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # SimpleMind expects document/mindmap.xml
        zf.writestr('document/mindmap.xml', xml_content)

    print(f"Written: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate SimpleMind mind maps from Pearltrees clusters"
    )
    parser.add_argument('--cluster', type=str, help='Cluster name/title to search for')
    parser.add_argument('--cluster-url', type=str, help='Cluster URL (Pearltrees URI)')
    parser.add_argument('--data', type=Path,
                        default=Path('reports/pearltrees_targets_full_multi_account.jsonl'),
                        help='Path to training data JSONL')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output .smmx file path')
    parser.add_argument('--xml-only', action='store_true',
                        help='Output raw XML instead of .smmx')

    args = parser.parse_args()

    if not args.cluster and not args.cluster_url:
        parser.error("Either --cluster or --cluster-url required")

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        return 1

    # Load cluster items
    items = load_cluster_items(args.data, args.cluster, args.cluster_url)

    if not items:
        print(f"No items found for cluster")
        return 1

    print(f"Found {len(items)} items in cluster")

    # Create nodes
    nodes = create_nodes_from_items(items)

    # Build hierarchy
    root = build_hierarchy(nodes)

    # Apply radial layout
    apply_radial_layout(root, center_x=500, center_y=500, radius=300)

    # Generate XML
    title = root.title if root else "Mind Map"
    xml_content = generate_mindmap_xml(root, title)

    # Write output
    if args.xml_only:
        output_path = args.output.with_suffix('.xml')
        output_path.write_text(xml_content)
        print(f"Written: {output_path}")
    else:
        write_smmx(xml_content, args.output)

    return 0

if __name__ == '__main__':
    exit(main())
