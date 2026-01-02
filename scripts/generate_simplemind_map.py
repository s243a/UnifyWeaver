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
from typing import List, Dict, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import numpy as np
from sklearn.cluster import KMeans

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

def load_embeddings(embeddings_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load embeddings indexed by tree_id, title, and uri.

    Returns:
        (tree_id_to_emb, title_to_emb, uri_to_emb) - three dicts for lookups
    """
    if not embeddings_path.exists():
        return {}, {}, {}

    data = np.load(embeddings_path, allow_pickle=True)
    tree_ids = data['tree_ids']
    titles = data['titles']
    embeddings = data['input_nomic']  # 768-dim Nomic embeddings

    # Index by tree_id (for Trees)
    tree_id_to_emb = {}
    for tid, emb in zip(tree_ids, embeddings):
        if tid and str(tid).strip():
            tree_id_to_emb[str(tid)] = emb

    # Index by title (fallback)
    title_to_emb = {str(t): emb for t, emb in zip(titles, embeddings)}

    # Index by uri if available (most precise for PagePearls)
    uri_to_emb = {}
    if 'uris' in data.files:
        uris = data['uris']
        for uri, emb in zip(uris, embeddings):
            if uri and str(uri).strip():
                uri_to_emb[str(uri)] = emb

    return tree_id_to_emb, title_to_emb, uri_to_emb

def build_hierarchy(nodes: List[MindMapNode], root_idx: int = 0,
                    min_children: int = 4, max_children: int = 8,
                    next_id: List[int] = None) -> MindMapNode:
    """
    Build hierarchical tree using recursive micro-clustering.

    Algorithm:
    1. Root is the first node (usually the folder itself)
    2. If <= max_children items, make all direct children
    3. Otherwise, use K-means to cluster into groups
    4. For each cluster, pick most central item as representative
    5. Recursively build hierarchy for each cluster
    """
    if next_id is None:
        next_id = [max(n.id for n in nodes) + 1]  # Mutable counter for new IDs

    if len(nodes) <= 1:
        return nodes[0] if nodes else None

    root = nodes[root_idx]
    others = [n for i, n in enumerate(nodes) if i != root_idx]

    if not others:
        return root

    # For small clusters, all nodes are direct children of root
    if len(others) <= max_children:
        root.children = others
        for child in others:
            child.parent_id = root.id
        return root

    # Need micro-clustering - check embedding coverage
    nodes_with_emb = [n for n in others if n.embedding is not None]
    embedding_ratio = len(nodes_with_emb) / len(others) if others else 0

    # Use K-means if we have enough embeddings (>50% coverage)
    if embedding_ratio > 0.5 and len(nodes_with_emb) >= min_children:
        # Use K-means clustering on nodes with embeddings
        embeddings = np.array([n.embedding for n in nodes_with_emb])

        # Target number of clusters: aim for min_children to max_children
        n_clusters = max(min_children, min(max_children, len(nodes_with_emb) // max_children + 1))
        n_clusters = min(n_clusters, len(nodes_with_emb))

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

        # Group nodes by cluster
        clusters: Dict[int, List[MindMapNode]] = {}
        for node, label in zip(others, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)

        # For each cluster, find most central node (closest to centroid)
        for label, cluster_nodes in clusters.items():
            if len(cluster_nodes) == 1:
                # Single node - add directly as child
                node = cluster_nodes[0]
                node.parent_id = root.id
                root.children.append(node)
            else:
                # Find node closest to centroid
                centroid = kmeans.cluster_centers_[label]
                cluster_embeddings = np.array([n.embedding for n in cluster_nodes])
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                central_idx = np.argmin(distances)
                central_node = cluster_nodes[central_idx]

                # Central node becomes child of root
                central_node.parent_id = root.id
                root.children.append(central_node)

                # Recursively build hierarchy for remaining nodes in cluster
                remaining = [n for i, n in enumerate(cluster_nodes) if i != central_idx]
                if remaining:
                    # Recursively cluster if still too many
                    sub_nodes = [central_node] + remaining
                    build_hierarchy(sub_nodes, root_idx=0,
                                   min_children=min_children, max_children=max_children,
                                   next_id=next_id)
    else:
        # No embeddings - create arbitrary groups
        n_groups = max(min_children, len(others) // max_children + 1)
        group_size = len(others) // n_groups

        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else len(others)
            group = others[start:end]

            if len(group) == 1:
                group[0].parent_id = root.id
                root.children.append(group[0])
            else:
                # First node in group becomes representative
                rep = group[0]
                rep.parent_id = root.id
                root.children.append(rep)
                for node in group[1:]:
                    node.parent_id = rep.id
                    rep.children.append(node)

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

def count_nodes_per_level(node: MindMapNode, level: int = 0,
                          counts: Dict[int, int] = None) -> Dict[int, int]:
    """Count total nodes at each depth level."""
    if counts is None:
        counts = {}

    counts[level] = counts.get(level, 0) + 1

    for child in node.children:
        count_nodes_per_level(child, level + 1, counts)

    return counts

def apply_radial_layout(node: MindMapNode, center_x: float = 500, center_y: float = 500,
                        min_spacing: float = 80, base_radius: float = 150):
    """
    Apply radial layout with consistent circumferential spacing.

    Radius at each level grows to maintain min_spacing between nodes.
    Formula: radius = max(base_radius, n_nodes * min_spacing / (2 * pi))
    """
    # First pass: count nodes at each level
    level_counts = count_nodes_per_level(node)

    # Calculate radius for each level to maintain spacing
    level_radii = {}
    cumulative_radius = 0
    for level in sorted(level_counts.keys()):
        if level == 0:
            level_radii[0] = 0  # Root is at center
        else:
            n_nodes = level_counts[level]
            # Radius needed for this level's circumference
            # circumference = 2 * pi * r, spacing = circumference / n_nodes
            # So r = n_nodes * spacing / (2 * pi)
            needed_radius = max(base_radius, n_nodes * min_spacing / (2 * math.pi))
            cumulative_radius += needed_radius
            level_radii[level] = cumulative_radius

    # Second pass: position nodes
    _position_nodes_by_level(node, center_x, center_y, level_radii)

def _position_nodes_by_level(node: MindMapNode, center_x: float, center_y: float,
                              level_radii: Dict[int, float], level: int = 0,
                              angle_start: float = 0, angle_span: float = 2 * math.pi):
    """Position nodes at their level's radius within their angular sector."""
    if level == 0:
        node.x = center_x
        node.y = center_y

    if not node.children:
        return

    n_children = len(node.children)
    child_level = level + 1
    child_radius = level_radii.get(child_level, 200)

    # Divide the angular span among children
    angle_per_child = angle_span / n_children

    for i, child in enumerate(node.children):
        # Child's angle is within parent's sector
        child_angle = angle_start + (i + 0.5) * angle_per_child

        child.x = center_x + child_radius * math.cos(child_angle)
        child.y = center_y + child_radius * math.sin(child_angle)

        # Recurse: each child gets a proportional angular sector
        _position_nodes_by_level(child, center_x, center_y, level_radii,
                                  level=child_level,
                                  angle_start=angle_start + i * angle_per_child,
                                  angle_span=angle_per_child)

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

def create_nodes_from_items(items: List[Dict],
                            tree_id_emb: Dict[str, np.ndarray] = None,
                            title_emb: Dict[str, np.ndarray] = None,
                            uri_emb: Dict[str, np.ndarray] = None) -> List[MindMapNode]:
    """Convert cluster items to MindMapNodes with optional embeddings.

    Looks up embeddings by: uri (most precise) > tree_id > title (fallback).
    """
    nodes = []
    tree_id_emb = tree_id_emb or {}
    title_emb = title_emb or {}
    uri_emb = uri_emb or {}

    def get_embedding(item):
        """Get embedding for an item, trying uri, then tree_id, then title."""
        # Try pearl_uri (for PagePearls) - most precise
        pearl_uri = item.get('pearl_uri', '')
        if pearl_uri and pearl_uri in uri_emb:
            return uri_emb[pearl_uri]
        # Try uri (for Trees)
        uri = item.get('uri', '')
        if uri and uri in uri_emb:
            return uri_emb[uri]
        # Try tree_id
        tree_id = item.get('tree_id', '')
        if tree_id and str(tree_id) in tree_id_emb:
            return tree_id_emb[str(tree_id)]
        # Fallback to title
        title = item.get('raw_title', '')
        if title in title_emb:
            return title_emb[title]
        return None

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
        tree_id = root_item.get('tree_id', '')
        root_node = MindMapNode(
            id=0,
            title=root_item.get('raw_title', 'Root'),
            tree_id=tree_id,
            url=root_item.get('uri', ''),
            parent_id=-1,
            palette=1,
            embedding=get_embedding(root_item)
        )
        nodes.append(root_node)

    # Create child nodes
    palette_idx = 2
    for i, item in enumerate(items):
        if item == root_item:
            continue

        tree_id = item.get('tree_id', '')
        node = MindMapNode(
            id=len(nodes),
            title=item.get('raw_title', f'Item {i}'),
            tree_id=tree_id,
            url=item.get('uri', ''),
            parent_id=0,  # Will be updated by build_hierarchy
            palette=(palette_idx % 8) + 1,
            embedding=get_embedding(item)
        )
        nodes.append(node)
        palette_idx += 1

    # Report embedding coverage
    with_emb = sum(1 for n in nodes if n.embedding is not None)
    print(f"Embeddings loaded: {with_emb}/{len(nodes)} nodes")

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
    parser.add_argument('--embeddings', type=Path,
                        default=Path('models/dual_embeddings_full.npz'),
                        help='Path to embeddings file (.npz)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output .smmx file path')
    parser.add_argument('--xml-only', action='store_true',
                        help='Output raw XML instead of .smmx')
    parser.add_argument('--max-children', type=int, default=8,
                        help='Max children per node before micro-clustering')
    parser.add_argument('--min-children', type=int, default=4,
                        help='Min clusters when micro-clustering')

    args = parser.parse_args()

    if not args.cluster and not args.cluster_url:
        parser.error("Either --cluster or --cluster-url required")

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        return 1

    # Load embeddings
    tree_id_emb, title_emb, uri_emb = {}, {}, {}
    if args.embeddings.exists():
        print(f"Loading embeddings from {args.embeddings}...")
        tree_id_emb, title_emb, uri_emb = load_embeddings(args.embeddings)
        print(f"Loaded {len(tree_id_emb)} tree_id, {len(title_emb)} title, {len(uri_emb)} uri embeddings")
    else:
        print(f"Warning: Embeddings file not found: {args.embeddings}")

    # Load cluster items
    items = load_cluster_items(args.data, args.cluster, args.cluster_url)

    if not items:
        print(f"No items found for cluster")
        return 1

    print(f"Found {len(items)} items in cluster")

    # Create nodes with embeddings
    nodes = create_nodes_from_items(items, tree_id_emb, title_emb, uri_emb)

    # Build hierarchy with micro-clustering
    root = build_hierarchy(nodes, min_children=args.min_children,
                          max_children=args.max_children)

    # Apply radial layout with consistent circumferential spacing
    apply_radial_layout(root, center_x=500, center_y=500, min_spacing=80, base_radius=150)

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
