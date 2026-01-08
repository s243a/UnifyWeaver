#!/usr/bin/env python3
"""
Generate mind maps from Pearltrees clusters.

Uses radial layout with hierarchical structure based on semantic similarity.
Output is a .smmx file (zip containing mindmap.xml).

Usage:
    python3 scripts/generate_mindmap.py \
        --cluster "Poles and Zeros" \
        --data reports/pearltrees_targets_full_multi_account.jsonl \
        --output output/poles_zeros.smmx

    # Or by cluster URL:
    python3 scripts/generate_mindmap.py \
        --cluster-url "https://www.pearltrees.com/s243a/poles-zeros-complex-numbers/id11563630" \
        --output output/poles_zeros.smmx
"""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

# Optional: import index store for relative links
try:
    from mindmap.index_store import create_index_store, IndexStore
    HAS_INDEX_STORE = True
except ImportError:
    HAS_INDEX_STORE = False
    IndexStore = None

# Optional: sqlite3 for children index
import sqlite3


def load_children_from_index(db_path: Path, tree_id: str) -> List[Dict]:
    """Load children for a tree from the SQLite children index.

    Args:
        db_path: Path to children_index.db
        tree_id: Tree ID to look up

    Returns:
        List of child dicts with type, title, external_url, see_also_uri, pos_order
    """
    if not db_path or not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT pearl_type, title, pos_order, external_url, see_also_uri, uri, parent_tree_uri
            FROM children
            WHERE parent_tree_id = ?
            ORDER BY pos_order
        ''', (str(tree_id),))

        children = []
        for row in cursor.fetchall():
            pearl_type, title, pos_order, external_url, see_also_uri, uri, parent_tree_uri = row
            # Convert to item dict format matching JSONL structure
            item = {
                'type': pearl_type,
                'raw_title': title or '',
                'pos_order': pos_order or 0,
                'uri': uri or '',
                'pearl_uri': uri or '',
                'cluster_id': parent_tree_uri or '',
                'url': external_url or '',  # For PagePearl
                'alias_target_uri': see_also_uri or '',  # For RefPearl/AliasPearl
            }
            children.append(item)

        conn.close()
        return children
    except Exception as e:
        print(f"Warning: Error loading children from index: {e}")
        return []


def extract_account_from_uri(uri: str) -> Optional[str]:
    """Extract account name from Pearltrees URI.

    Handles both regular /account/ and team /t/account/ formats.
    """
    if not uri:
        return None
    match = re.search(r'pearltrees\.com/(?:t/)?([^/]+)/', uri)
    return match.group(1) if match else None


def get_disambiguated_title(item: Dict) -> str:
    """Get display title with cross-account disambiguation for AliasPearl/RefPearl.

    When an AliasPearl points to a tree from a different account, prefix with
    the target account to help identify external references:
      "science" -> "vijayau:science"
    """
    title = item.get('raw_title', 'Unknown')

    # Only disambiguate AliasPearl and RefPearl types
    item_type = item.get('type', '')
    if item_type not in ('AliasPearl', 'RefPearl'):
        return title

    # Get the pearl's owning account and the target URI
    pearl_account = item.get('account', '')
    target_uri = item.get('alias_target_uri', '')

    if not target_uri:
        return title

    # Extract target account from URI
    target_account = extract_account_from_uri(target_uri)

    # If cross-account, prefix with target account
    if target_account and target_account != pearl_account:
        return f"{target_account}:{title}"

    return title


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
    item_type: str = ""  # PagePearl, Tree, etc.
    guid: str = ""  # Deterministic GUID for cross-map linking
    cluster_id: str = ""  # The cluster this node belongs to (for GUID generation)
    alias_target_uri: str = ""  # For AliasPearl/RefPearl - the target tree URI
    external_url: str = ""  # For PagePearl - the actual external URL (e.g., wikipedia.org)


# SimpleMind borderstyle values
BORDERSTYLES = {
    'half-round': 'sbsHalfRound',
    'ellipse': 'sbsEllipse',
    'rectangle': 'sbsRectangle',
    'diamond': 'sbsDiamond',
}

def generate_guid() -> str:
    """Generate a random SimpleMind-style GUID."""
    return uuid.uuid4().hex.upper()[:32]


def generate_deterministic_guid(cluster_id: str, node_id: int) -> str:
    """Generate a deterministic GUID from cluster_id and node_id.

    This allows linking to specific nodes before their map is generated,
    since the GUID is predictable from the cluster and node identifiers.

    Args:
        cluster_id: The cluster URL or tree_id (e.g., "https://...id10818216")
        node_id: The node's ID within the cluster (0 for root, 1+ for children)

    Returns:
        URL-safe base64 encoded hash, SimpleMind-compatible GUID format
    """
    key = f"{cluster_id}_{node_id}"
    hash_bytes = hashlib.sha256(key.encode()).digest()[:18]
    # URL-safe base64, no padding, ~24 chars like SimpleMind GUIDs
    guid = base64.urlsafe_b64encode(hash_bytes).decode().rstrip('=')
    return guid


def extract_tree_id(url: str) -> str:
    """Extract tree ID from Pearltrees URL.

    Examples:
        "https://www.pearltrees.com/s243a/hactivism/id10818216" -> "id10818216"
        "https://www.pearltrees.com/t/hacktivism/id2492215" -> "id2492215"
    """
    if not url:
        return ""
    # Extract the last path segment that starts with "id"
    parts = url.rstrip('/').split('/')
    for part in reversed(parts):
        if part.startswith('id') and part[2:].isdigit():
            return part
    # Fallback: use last segment
    return parts[-1] if parts else ""


def extract_domain(url: str) -> str:
    """Extract domain name from URL for display.

    Examples:
        "https://en.wikipedia.org/wiki/History" -> "wikipedia.org"
        "https://www.example.com/page" -> "example.com"
        "https://docs.python.org/3/" -> "python.org"
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove www. prefix and common subdomains
        if domain.startswith('www.'):
            domain = domain[4:]
        elif domain.startswith('en.'):
            domain = domain[3:]
        elif domain.startswith('docs.'):
            domain = domain[5:]
        return domain
    except Exception:
        return ""


def sanitize_filename(title: str, max_length: int = 50) -> str:
    """Sanitize a title for use as a filename.

    Args:
        title: The raw title to sanitize
        max_length: Maximum length for the filename (default: 50)

    Returns:
        A filesystem-safe filename string
    """
    import re
    if not title:
        return ""

    # Replace problematic characters with underscores
    # Forbidden in Windows: / \ : * ? " < > |
    # Also replace spaces and other problematic chars
    sanitized = re.sub(r'[/\\:*?"<>|\s]+', '_', title)

    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')

    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Truncate to max length, but try to break at underscore
    if len(sanitized) > max_length:
        # Try to find a good break point
        truncated = sanitized[:max_length]
        last_underscore = truncated.rfind('_')
        if last_underscore > max_length // 2:
            sanitized = truncated[:last_underscore]
        else:
            sanitized = truncated.rstrip('_')

    return sanitized


def generate_filename_from_title(title: str, tree_id: str, include_id: bool = True) -> str:
    """Generate a filename from tree title and ID.

    Args:
        title: The tree's raw title
        tree_id: The tree's ID (e.g., "id2492215")
        include_id: Whether to append ID for uniqueness (default: True)

    Returns:
        Filename without extension (e.g., "Hacktivism_id2492215")
    """
    sanitized = sanitize_filename(title)

    if not sanitized:
        return tree_id

    if include_id and tree_id:
        return f"{sanitized}_{tree_id}"
    else:
        return sanitized


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

    Uses output_nomic embeddings which encode the full hierarchical path context:
    - Materialized path with IDs (e.g., /10311468/11820376/...)
    - Structured list with full titles (indented hierarchy)

    This provides better semantic clustering than raw title embeddings (input_nomic).

    Returns:
        (tree_id_to_emb, title_to_emb, uri_to_emb) - three dicts for lookups
    """
    if not embeddings_path.exists():
        return {}, {}, {}

    data = np.load(embeddings_path, allow_pickle=True)
    tree_ids = data['tree_ids']
    titles = data['titles']
    # Use output_nomic: embeds materialized path + structured title list
    # This captures hierarchical context for better folder clustering
    embeddings = data['output_nomic']  # 768-dim Nomic embeddings of hierarchical context

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


def build_mst_tree_hierarchy(nodes: List[MindMapNode], root_idx: int = 0,
                              max_depth: int = 6, max_children: int = 12) -> MindMapNode:
    """
    Build a hierarchical structure using hierarchical clustering with depth limit.

    Uses Ward's method to create balanced clusters, then builds a tree with
    cluster representatives. This creates semantic grouping while keeping
    the tree balanced.

    This preserves the true Pearltrees hierarchy (only direct children) while
    organizing them visually based on semantic similarity.

    Args:
        nodes: List of MindMapNode items (root + direct children)
        root_idx: Index of the root node
        max_depth: Maximum tree depth (default 6)
        max_children: Maximum children per node before clustering (default 12)

    Returns:
        Root node with items organized hierarchically
    """
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, fcluster

    if len(nodes) <= 1:
        return nodes[0] if nodes else None

    root = nodes[root_idx]
    others = [n for i, n in enumerate(nodes) if i != root_idx]

    if not others:
        return root

    # Separate nodes with and without embeddings
    nodes_with_emb = [n for n in others if n.embedding is not None]
    nodes_without_emb = [n for n in others if n.embedding is None]

    if len(nodes_with_emb) < 2:
        # Not enough embeddings - all nodes as direct children
        root.children = others
        for child in others:
            child.parent_id = root.id
        return root

    # Build embeddings array
    embeddings = np.array([n.embedding for n in nodes_with_emb])

    def build_recursive(node_indices: List[int], parent_node: MindMapNode,
                        current_depth: int):
        """Recursively cluster and build hierarchy."""
        if len(node_indices) == 0:
            return

        if len(node_indices) == 1:
            # Single node - add directly
            node = nodes_with_emb[node_indices[0]]
            node.parent_id = parent_node.id
            parent_node.children.append(node)
            return

        # If few enough nodes or at max depth, add all as direct children
        if len(node_indices) <= max_children or current_depth >= max_depth:
            for idx in node_indices:
                node = nodes_with_emb[idx]
                node.parent_id = parent_node.id
                parent_node.children.append(node)
            return

        # Need to cluster - compute hierarchical clustering on subset
        subset_embeddings = embeddings[node_indices]
        subset_distances = pdist(subset_embeddings, metric='cosine')

        if len(subset_distances) == 0:
            # All same embedding - add directly
            for idx in node_indices:
                node = nodes_with_emb[idx]
                node.parent_id = parent_node.id
                parent_node.children.append(node)
            return

        subset_Z = linkage(subset_distances, method='ward')

        # Determine number of clusters (aim for balanced tree)
        n_clusters = min(max_children, max(2, len(node_indices) // max_children + 1))
        labels = fcluster(subset_Z, n_clusters, criterion='maxclust')

        # Group by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node_indices[i])

        # For each cluster, pick representative and recurse
        for label, cluster_indices in sorted(clusters.items()):
            if len(cluster_indices) == 1:
                # Single node cluster - add directly
                node = nodes_with_emb[cluster_indices[0]]
                node.parent_id = parent_node.id
                parent_node.children.append(node)
            else:
                # Find representative (closest to cluster centroid)
                cluster_embs = embeddings[cluster_indices]
                centroid = np.mean(cluster_embs, axis=0)
                distances_to_centroid = np.linalg.norm(cluster_embs - centroid, axis=1)
                rep_local_idx = int(np.argmin(distances_to_centroid))
                rep_global_idx = cluster_indices[rep_local_idx]

                # Add representative as child
                rep_node = nodes_with_emb[rep_global_idx]
                rep_node.parent_id = parent_node.id
                parent_node.children.append(rep_node)

                # Recurse with remaining nodes under representative
                remaining = [idx for idx in cluster_indices if idx != rep_global_idx]
                if remaining:
                    build_recursive(remaining, rep_node, current_depth + 1)

    # Build hierarchy starting from root
    all_indices = list(range(len(nodes_with_emb)))
    build_recursive(all_indices, root, 0)

    # Add nodes without embeddings as direct children of root
    for node in nodes_without_emb:
        node.parent_id = root.id
        root.children.append(node)

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

def apply_radial_tree_layout(node: MindMapNode, center_x: float = 500, center_y: float = 500,
                              min_node_spacing: float = 60, base_radius: float = 120):
    """
    Apply radial tree layout with subtree-weighted angular allocation.

    Each subtree gets angular space proportional to its size, ensuring no edge crossings.
    Radius grows per level to fit all children with minimum spacing.

    Args:
        node: Root node of the tree
        center_x, center_y: Center coordinates
        min_node_spacing: Minimum spacing between adjacent nodes
        base_radius: Base radius increment per level
    """
    # Step 1: Compute subtree sizes (leaf count for weighting)
    def compute_subtree_size(n: MindMapNode) -> int:
        """Return number of leaves (or 1 if leaf)."""
        if not n.children:
            return 1
        return sum(compute_subtree_size(c) for c in n.children)

    # Cache subtree sizes
    subtree_sizes = {}
    def cache_sizes(n: MindMapNode):
        subtree_sizes[id(n)] = compute_subtree_size(n)
        for c in n.children:
            cache_sizes(c)
    cache_sizes(node)

    # Step 2: Position nodes recursively
    def position_subtree(n: MindMapNode, angle_start: float, angle_span: float,
                         depth: int, parent_radius: float):
        """Position a node and its subtree within the given angular wedge."""
        # Calculate radius for this depth
        radius = parent_radius + base_radius if depth > 0 else 0

        # Position this node at center of its wedge
        angle_center = angle_start + angle_span / 2
        n.x = center_x + radius * math.cos(angle_center)
        n.y = center_y + radius * math.sin(angle_center)

        if not n.children:
            return

        # Calculate radius needed so children have min_node_spacing
        n_children = len(n.children)
        child_radius = radius + base_radius

        # Ensure children don't overlap: arc_length >= n_children * spacing
        # arc_length = angle_span * child_radius
        min_child_radius = (n_children * min_node_spacing) / angle_span if angle_span > 0 else base_radius
        child_radius = max(child_radius, min_child_radius)

        # Equal angles per parent - each child gets equal share of parent's wedge
        child_span = angle_span / n_children

        current_angle = angle_start
        for child in n.children:
            position_subtree(child, current_angle, child_span, depth + 1, child_radius - base_radius)
            current_angle += child_span

    # Start positioning from root
    position_subtree(node, 0, 2 * math.pi, 0, 0)


def collect_all_nodes(node: MindMapNode, nodes: List[MindMapNode] = None) -> List[MindMapNode]:
    """Flatten tree into list of all nodes."""
    if nodes is None:
        nodes = []
    nodes.append(node)
    for child in node.children:
        collect_all_nodes(child, nodes)
    return nodes

def count_descendants(node: MindMapNode) -> int:
    """Count total descendants of a node."""
    count = len(node.children)
    for child in node.children:
        count += count_descendants(child)
    return count

def force_directed_optimize(root: MindMapNode, center_x: float = 500, center_y: float = 500,
                            iterations: int = 300, repulsion: float = 100000,
                            attraction: float = 0.001, radial_weight: float = 0.0,
                            min_distance: float = 120, damping: float = 0.8):
    """
    Apply force-directed optimization to reduce overlaps while preserving structure.

    Forces:
    - Repulsion: nodes push apart (inverse square law, stronger when overlapping)
    - Attraction: weak pull to parent only when very far away (>500px)

    The radial constraint is disabled by default since the initial radial layout
    provides good starting positions. Pure force-directed then spreads nodes to
    eliminate overlaps.

    Args:
        root: Root node of the tree
        iterations: Number of simulation steps (default 300)
        repulsion: Repulsion force strength (default 100000)
        attraction: Attraction force strength when far from parent (default 0.001)
        radial_weight: Radial constraint weight (default 0.0 = disabled)
        min_distance: Target minimum distance between nodes (default 120px)
        damping: Velocity damping factor (default 0.8)
    """
    all_nodes = collect_all_nodes(root)
    n = len(all_nodes)

    if n <= 1:
        return

    # Calculate "mass" for each node based on descendants
    # Nodes with more descendants push harder
    node_mass = {}
    for node in all_nodes:
        descendants = count_descendants(node)
        # Mass scales with sqrt of descendants (1 + sqrt(d))
        # This gives hubs more push without being overwhelming
        node_mass[node.id] = 1 + math.sqrt(descendants)

    # Store original positions and radii for radial constraint
    original_positions = {node.id: (node.x, node.y) for node in all_nodes}
    original_radii = {}
    for node in all_nodes:
        dx = node.x - center_x
        dy = node.y - center_y
        original_radii[node.id] = math.sqrt(dx*dx + dy*dy)

    # Build parent lookup and connected set
    parent_map = {}
    connected_pairs = set()
    for node in all_nodes:
        for child in node.children:
            parent_map[child.id] = node
            # Store both directions for easy lookup
            connected_pairs.add((node.id, child.id))
            connected_pairs.add((child.id, node.id))

    # Initialize velocities
    velocities = {node.id: [0.0, 0.0] for node in all_nodes}

    for iteration in range(iterations):
        forces = {node.id: [0.0, 0.0] for node in all_nodes}

        # Repulsion between all pairs
        for i, node_a in enumerate(all_nodes):
            for node_b in all_nodes[i+1:]:
                dx = node_b.x - node_a.x
                dy = node_b.y - node_a.y
                dist_sq = dx*dx + dy*dy
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.1

                # Check if nodes are connected (parent-child)
                is_connected = (node_a.id, node_b.id) in connected_pairs

                if dist < min_distance * 5:  # Apply over larger range
                    # Repulsion scaled by product of masses
                    # Hub nodes push each other apart more strongly
                    mass_factor = node_mass[node_a.id] * node_mass[node_b.id]

                    # Non-connected nodes repel more strongly
                    connection_factor = 0.3 if is_connected else 1.5

                    # Inverse square repulsion, stronger when very close
                    force_mag = (repulsion * mass_factor * connection_factor) / (dist_sq + 100)
                    # Extra boost when overlapping
                    if dist < min_distance:
                        force_mag *= 3
                    if dist > 0:
                        fx = (dx / dist) * force_mag
                        fy = (dy / dist) * force_mag
                        forces[node_a.id][0] -= fx
                        forces[node_a.id][1] -= fy
                        forces[node_b.id][0] += fx
                        forces[node_b.id][1] += fy

        # Attraction to parent - stronger for leaf nodes (low mass)
        for node in all_nodes:
            if node.id in parent_map:
                parent = parent_map[node.id]
                dx = parent.x - node.x
                dy = parent.y - node.y
                dist = math.sqrt(dx*dx + dy*dy)

                # Ideal distance scales with node's mass
                # Leaf nodes (mass ~1) should stay close, hubs can be further
                mass = node_mass[node.id]
                ideal_dist = 100 * mass  # Leaves: ~100px, large hubs: ~1000px

                if dist > ideal_dist:
                    # Attraction strength inversely proportional to mass
                    # Leaves get pulled back hard, hubs resist
                    attract_strength = attraction * (5.0 / mass)
                    # Quadratic pull when very far (>2x ideal)
                    if dist > ideal_dist * 2:
                        force_mag = (dist - ideal_dist) * attract_strength * 2
                    else:
                        force_mag = (dist - ideal_dist) * attract_strength
                    if dist > 0:
                        forces[node.id][0] += (dx / dist) * force_mag
                        forces[node.id][1] += (dy / dist) * force_mag

        # Radial constraint - very weak pull toward original radius
        # (mostly disabled to allow free movement)
        if radial_weight > 0.01:
            for node in all_nodes:
                if node.id == root.id:
                    continue  # Don't move root

                dx = node.x - center_x
                dy = node.y - center_y
                current_radius = math.sqrt(dx*dx + dy*dy)
                original_radius = original_radii[node.id]

                if current_radius > 0:
                    # Pull toward original radius
                    radius_diff = original_radius - current_radius
                    force_mag = radius_diff * radial_weight
                    forces[node.id][0] += (dx / current_radius) * force_mag
                    forces[node.id][1] += (dy / current_radius) * force_mag

        # Update velocities and positions
        max_movement = 0
        for node in all_nodes:
            if node.id == root.id:
                continue  # Keep root fixed

            # Update velocity with damping
            velocities[node.id][0] = (velocities[node.id][0] + forces[node.id][0]) * damping
            velocities[node.id][1] = (velocities[node.id][1] + forces[node.id][1]) * damping

            # Limit velocity
            vel_mag = math.sqrt(velocities[node.id][0]**2 + velocities[node.id][1]**2)
            max_vel = 150  # Allow larger movements
            if vel_mag > max_vel:
                velocities[node.id][0] *= max_vel / vel_mag
                velocities[node.id][1] *= max_vel / vel_mag

            # Update position
            node.x += velocities[node.id][0]
            node.y += velocities[node.id][1]

            max_movement = max(max_movement, abs(velocities[node.id][0]), abs(velocities[node.id][1]))

        # Early termination if converged
        if max_movement < 0.5:
            print(f"Force-directed optimization converged at iteration {iteration}")
            break

    print(f"Force-directed optimization completed ({iterations} iterations)")

def segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float],
                       p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """
    Check if line segment p1-p2 intersects with segment p3-p4.
    Uses counter-clockwise orientation test.
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Check if segments share an endpoint (not a real crossing for straight lines)
    endpoints = {p1, p2}
    if p3 in endpoints or p4 in endpoints:
        return False

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def sibling_edges_cross(parent: MindMapNode, child1: MindMapNode, child2: MindMapNode) -> bool:
    """
    Check if two sibling edges (from same parent) would cross with curved lines.

    SimpleMind curves: radial from node center, tangent to quadrant axes.
    Two siblings cross if their angular positions would cause curve overlap.

    Heuristic: if one child is "between" the parent and another child angularly,
    and they're at similar distances, the curves likely cross.
    """
    # Get angles from parent to each child
    dx1 = child1.x - parent.x
    dy1 = child1.y - parent.y
    dx2 = child2.x - parent.x
    dy2 = child2.y - parent.y

    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)

    dist1 = math.sqrt(dx1*dx1 + dy1*dy1)
    dist2 = math.sqrt(dx2*dx2 + dy2*dy2)

    # Normalize angle difference to [-pi, pi]
    angle_diff = abs(angle1 - angle2)
    if angle_diff > math.pi:
        angle_diff = 2 * math.pi - angle_diff

    # If angles are very close but distances differ significantly,
    # the curves might cross (one goes "over" the other)
    if angle_diff < 0.3 and abs(dist1 - dist2) > 100:  # ~17 degrees
        return True

    # If angles are moderately close and one child is much closer,
    # check if the further one's curve might cross
    if angle_diff < 0.6 and min(dist1, dist2) < max(dist1, dist2) * 0.5:
        return True

    return False


def collect_edges(node: MindMapNode, edges: List[Tuple[MindMapNode, MindMapNode]] = None) -> List[Tuple[MindMapNode, MindMapNode]]:
    """Collect all parent-child edges in the tree."""
    if edges is None:
        edges = []
    for child in node.children:
        edges.append((node, child))
        collect_edges(child, edges)
    return edges


def count_edge_crossings(root: MindMapNode, include_siblings: bool = True,
                         verbose: bool = False) -> int:
    """Count total number of edge crossings in the layout.

    Args:
        root: Root node of the tree
        include_siblings: Also check sibling edges for curve crossings
        verbose: Print details of each crossing found
    """
    edges = collect_edges(root)
    all_nodes = collect_all_nodes(root)
    crossings = 0

    # Check non-sibling edge crossings (straight line intersection)
    for i, (a1, a2) in enumerate(edges):
        p1 = (a1.x, a1.y)
        p2 = (a2.x, a2.y)
        for b1, b2 in edges[i+1:]:
            p3 = (b1.x, b1.y)
            p4 = (b2.x, b2.y)
            if segments_intersect(p1, p2, p3, p4):
                crossings += 1
                if verbose:
                    print(f"  Crossing: [{a1.title[:20]}→{a2.title[:20]}] X [{b1.title[:20]}→{b2.title[:20]}]")

    # Check sibling edge crossings (curved line heuristic)
    if include_siblings:
        for node in all_nodes:
            if len(node.children) >= 2:
                for i, child1 in enumerate(node.children):
                    for child2 in node.children[i+1:]:
                        if sibling_edges_cross(node, child1, child2):
                            crossings += 1
                            if verbose:
                                print(f"  Sibling crossing: {node.title[:20]} → [{child1.title[:20]}, {child2.title[:20]}]")

    return crossings


def get_node_depth(node: MindMapNode, root: MindMapNode, depth_cache: Dict[int, int] = None) -> int:
    """Get depth of a node from root (via parent chain)."""
    if depth_cache is None:
        depth_cache = {}

    if node.id in depth_cache:
        return depth_cache[node.id]

    # Build depth by traversing tree
    def compute_depths(n: MindMapNode, d: int):
        depth_cache[n.id] = d
        for child in n.children:
            compute_depths(child, d + 1)

    if not depth_cache:
        compute_depths(root, 0)

    return depth_cache.get(node.id, 0)


def minimize_crossings(root: MindMapNode, center_x: float = 500, center_y: float = 500,
                       force_iterations: int = 50, max_passes: int = 10,
                       verbose: bool = True):
    """
    Minimize edge crossings by adjusting node positions one at a time.

    Algorithm:
    1. For each node (sorted by depth, shallowest first):
       a. Try angular adjustments within parent's sector
       b. Count crossings for each candidate position
       c. Accept position that minimizes crossings
       d. Run brief force-directed pass to restore overlap-free state
    2. Repeat until no improvement
    """
    all_nodes = collect_all_nodes(root)

    # Build depth cache
    depth_cache = {}
    get_node_depth(root, root, depth_cache)

    # Build parent map
    parent_map = {}
    for node in all_nodes:
        for child in node.children:
            parent_map[child.id] = node

    # Sort nodes by depth (shallowest first, skip root)
    nodes_by_depth = sorted([n for n in all_nodes if n.id != root.id],
                            key=lambda n: depth_cache[n.id])

    initial_crossings = count_edge_crossings(root)
    if verbose:
        print(f"Initial edge crossings: {initial_crossings}")

    if initial_crossings == 0:
        print("No crossings to minimize")
        return

    for pass_num in range(max_passes):
        improved = False
        pass_start_crossings = count_edge_crossings(root)

        for node in nodes_by_depth:
            # Save original position
            orig_x, orig_y = node.x, node.y
            parent = parent_map.get(node.id)

            if not parent:
                continue

            # Calculate current angle from parent
            dx = node.x - parent.x
            dy = node.y - parent.y
            current_dist = math.sqrt(dx*dx + dy*dy)
            current_angle = math.atan2(dy, dx)

            if current_dist < 1:
                continue

            # Try angular adjustments (radians): ±6°, ±15°, ±30°, ±45°, ±60°
            best_crossings = count_edge_crossings(root)
            best_x, best_y = orig_x, orig_y

            for angle_offset in [-1.0, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0]:
                test_angle = current_angle + angle_offset
                node.x = parent.x + current_dist * math.cos(test_angle)
                node.y = parent.y + current_dist * math.sin(test_angle)

                crossings = count_edge_crossings(root)
                if crossings < best_crossings:
                    best_crossings = crossings
                    best_x, best_y = node.x, node.y
                    improved = True

            # Also try distance adjustments: 0.8x, 1.2x distance
            for dist_factor in [0.8, 1.2]:
                test_dist = current_dist * dist_factor
                node.x = parent.x + test_dist * math.cos(current_angle)
                node.y = parent.y + test_dist * math.sin(current_angle)

                crossings = count_edge_crossings(root)
                if crossings < best_crossings:
                    best_crossings = crossings
                    best_x, best_y = node.x, node.y
                    improved = True

            # Apply best position
            node.x, node.y = best_x, best_y

        # Only run force-directed if we have overlaps (check nearby pairs)
        all_nodes = collect_all_nodes(root)
        overlap_count = 0
        for i, node_a in enumerate(all_nodes):
            for node_b in all_nodes[i+1:]:
                dx = node_b.x - node_a.x
                dy = node_b.y - node_a.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 80:  # Overlap threshold
                    overlap_count += 1

        # Only run force-directed if overlaps exist
        if overlap_count > 0:
            # Lighter force-directed to fix overlaps without destroying crossing gains
            # Use fewer iterations for minor overlaps
            iters = force_iterations // 2 if overlap_count > 2 else force_iterations // 4
            force_directed_optimize(root, center_x, center_y,
                                   iterations=iters, repulsion=50000,
                                   attraction=0.0005, min_distance=100)

        current_crossings = count_edge_crossings(root)
        if verbose:
            print(f"Pass {pass_num + 1}: {current_crossings} crossings (was {pass_start_crossings})")

        if not improved or current_crossings >= pass_start_crossings:
            break

    final_crossings = count_edge_crossings(root)
    if verbose:
        print(f"Final edge crossings: {final_crossings} (reduced from {initial_crossings})")
        if final_crossings > 0:
            print("Remaining crossings:")
            count_edge_crossings(root, verbose=True)


def calculate_node_scales(root: MindMapNode, min_scale: float = 1.2) -> Dict[int, float]:
    """
    Calculate font scale for each node based on descendant count.

    Scale formula: min_scale + log2(1 + descendants) * 0.4
    - Leaf nodes: min_scale (default 1.2)
    - Node with 7 descendants: ~2.1
    - Node with 63 descendants: ~3.0
    """
    scales = {}

    def compute_scale(node: MindMapNode):
        descendants = count_descendants(node)
        # Scale from min_scale (leaf) to ~3.5 (large hub)
        scale = min_scale + math.log2(1 + descendants) * 0.4
        scales[node.id] = min(scale, 3.5)  # Cap at 3.5

        for child in node.children:
            compute_scale(child)

    compute_scale(root)
    return scales


def compute_relative_cloudmapref_path(
    source_folder: Path,
    target_filename: str,
    target_folder: Path
) -> str:
    """Compute relative path from source to target .smmx file.

    Args:
        source_folder: Folder containing source .smmx (relative to output root)
        target_filename: Filename of target (e.g., 'Title_id123456.smmx' or 'id123456.smmx')
        target_folder: Folder containing target .smmx (relative to output root)

    Returns:
        Relative path string for cloudmapref (e.g., '../sibling/Title_id789.smmx')
    """
    import os
    # Both paths are relative to output root
    # source_folder: where the current .smmx is
    # target_folder: where the target .smmx is
    # Ensure filename has .smmx extension
    if not target_filename.endswith('.smmx'):
        target_filename = f'{target_filename}.smmx'
    target_file = target_folder / target_filename

    # Compute relative path from source folder to target file
    rel_path = os.path.relpath(target_file, source_folder)
    # Normalize to forward slashes for SimpleMind compatibility
    rel_path = rel_path.replace('\\', '/')

    # Add ./ prefix for same-directory or subdirectory paths (SimpleMind convention)
    if not rel_path.startswith('../') and not rel_path.startswith('./'):
        rel_path = './' + rel_path

    return rel_path


def create_parent_link_node(
    parent_element: Element,
    root_node: 'MindMapNode',
    parent_tree_id: str,
    parent_cloudmapref: str,
    node_id: int,
    parent_title: str = None
) -> None:
    """Create a parent link node connected to the root.

    Adds a small square node labeled "↑ ParentName" that links back to the parent map.

    Args:
        parent_element: XML element to add the node to
        root_node: The root MindMapNode (for positioning)
        parent_tree_id: Tree ID of the parent
        parent_cloudmapref: Relative path to parent .smmx file
        node_id: Unique ID for this node
        parent_title: Title of the parent folder (optional)
    """
    topic = SubElement(parent_element, 'topic')
    topic.set('id', str(node_id))
    topic.set('parent', str(root_node.id))  # Connect to root node
    topic.set('guid', generate_guid())
    # Position to the left of root
    topic.set('x', str(int(root_node.x - 120)))
    topic.set('y', str(int(root_node.y)))
    topic.set('palette', '0')  # Use palette 0 (typically gray/neutral)
    topic.set('colorinfo', '0')
    # Up arrow with optional parent title
    if parent_title:
        topic.set('text', f'↑ {parent_title}')
    else:
        topic.set('text', '↑')

    # Square borderstyle for distinction
    style = SubElement(topic, 'style')
    style.set('borderstyle', 'sbsRectangle')

    # Link with cloudmapref to parent
    link = SubElement(topic, 'link')
    link.set('cloudmapref', parent_cloudmapref)


def node_to_xml(node: MindMapNode, parent_element: Element, scales: Dict[int, float] = None,
                tree_style: str = None, pearl_style: str = None,
                enable_cloudmapref: bool = False, cluster_id: str = None,
                url_nodes_mode: str = None, next_id: List[int] = None,
                child_node_text: str = '',
                source_folder: Path = None,
                cluster_to_folder: Dict[str, Path] = None,
                url_to_filename: Dict[str, str] = None,
                layout: str = None,
                index_store = None,
                output_path: Path = None):
    """Convert a MindMapNode to SimpleMind XML topic element.

    Args:
        node: The node to convert
        parent_element: Parent XML element to add topic to
        scales: Font scale mapping by node ID
        tree_style: Borderstyle for Tree items (folders)
        pearl_style: Borderstyle for Pearl items (pages/links)
        enable_cloudmapref: If True, add cloudmapref links for Tree nodes
        cluster_id: The cluster ID for deterministic GUID generation
        url_nodes_mode: 'url' = URL on main/cloudmapref on child, 'map' = cloudmapref on main/URL on child, None = no child nodes
        next_id: Mutable counter [next_available_id] for generating child node IDs
        child_node_text: Text label for child link nodes (default: empty)
        source_folder: Current .smmx folder (relative to output root) for path computation
        cluster_to_folder: Dict mapping cluster URLs to folder paths for MST-based layout
        url_to_filename: Dict mapping cluster URLs to their actual filenames (for titled files)
        layout: Layout mode - 'radial-auto' adds native radial layout element to root topic
    """
    topic = SubElement(parent_element, 'topic')
    topic.set('id', str(node.id))
    topic.set('parent', str(node.parent_id))

    # Use deterministic GUID if cluster_id provided, otherwise use node.guid or generate random
    if cluster_id:
        guid = generate_deterministic_guid(cluster_id, node.id)
    elif node.guid:
        guid = node.guid
    else:
        guid = generate_guid()
    topic.set('guid', guid)

    topic.set('x', f'{node.x:.2f}')
    topic.set('y', f'{node.y:.2f}')
    topic.set('palette', str(node.palette))
    topic.set('colorinfo', str(node.palette))

    # Wrap title for rounder nodes
    # Don't manually escape & - ElementTree handles XML escaping
    wrapped = wrap_title(node.title)
    text = wrapped.replace('\n', '\\N')
    topic.set('text', text)

    # Add layout element to root topic for native radial layout
    if node.id == 0 and layout == 'radial-auto':
        layout_elem = SubElement(topic, 'layout')
        layout_elem.set('mode', 'radial')
        layout_elem.set('direction', 'auto')
        layout_elem.set('flow', 'default')

    # Determine borderstyle based on item type
    # PagePearl/Section/Note use sbsNone (no border) to indicate no relative link
    # Tree/AliasPearl/RefPearl can have cloudmapref links, so use configured style
    borderstyle = None
    if node.item_type in ('PagePearl', 'Section', 'Note'):
        borderstyle = 'sbsNone'
    elif node.item_type == 'Tree' and tree_style:
        borderstyle = BORDERSTYLES.get(tree_style)
    elif node.item_type in ('AliasPearl', 'RefPearl') and pearl_style:
        borderstyle = BORDERSTYLES.get(pearl_style)

    # Add style element (for font scaling and/or borderstyle)
    style = None
    if scales and node.id in scales:
        scale = scales[node.id]
        if scale > 1.2:  # Only add styling for non-leaf nodes
            style = SubElement(topic, 'style')
            font = SubElement(style, 'font')
            if scale > 2.0:
                font.set('bold', 'True')
            font.set('scale', f'{scale:.2f}')

    # Add borderstyle to existing style element or create new one
    if borderstyle:
        if style is None:
            style = SubElement(topic, 'style')
        style.set('borderstyle', borderstyle)

    # Add link - handle Tree nodes with cloudmapref and optional child nodes
    if node.url:
        is_tree_with_children = enable_cloudmapref and node.item_type == 'Tree' and node.id != 0
        target_tree_id = extract_tree_id(node.url) if is_tree_with_children else None

        # Compute cloudmapref path (uses folder structure, filename mapping, or index)
        def get_cloudmapref_path():
            # Get the actual filename (titled or just ID)
            if url_to_filename and node.url in url_to_filename:
                target_filename = url_to_filename[node.url]
            else:
                target_filename = target_tree_id

            if cluster_to_folder and source_folder is not None:
                # Use target's folder from mapping, or root folder as fallback
                target_folder = cluster_to_folder.get(node.url, Path('.'))
                return compute_relative_cloudmapref_path(source_folder, target_filename, target_folder)
            elif index_store and target_tree_id and output_path:
                # Use index to look up target path
                # Index keys are just digits (e.g., "11460410"), not "id11460410"
                lookup_id = target_tree_id[2:] if target_tree_id.startswith('id') else target_tree_id
                target_path = index_store.get(lookup_id)
                if target_path:
                    # Compute relative path from output_path to target
                    base_dir = index_store.base_dir or str(output_path.parent)
                    output_dir = str(output_path.parent.resolve())
                    target_abs = Path(base_dir) / target_path
                    try:
                        rel_path = os.path.relpath(target_abs, output_dir)
                        return rel_path
                    except ValueError:
                        return target_path
                return None  # Target not in index
            else:
                return f'./{target_filename}.smmx'

        if is_tree_with_children and target_tree_id:
            # Determine link layout based on url_nodes_mode
            # 'url': URL on main, cloudmapref on child (user's preferred hand-built style)
            # 'map': cloudmapref on main, URL on child
            # None: cloudmapref on main, no child node

            if url_nodes_mode == 'url':
                # URL on main node
                link = SubElement(topic, 'link')
                link.set('urllink', node.url)

                # cloudmapref on child node (square, unlabeled) - only if target is in index
                cloudmap_path = get_cloudmapref_path()
                if cloudmap_path and next_id is not None:
                    child_node_id = next_id[0]
                    next_id[0] += 1
                    child_topic = SubElement(parent_element, 'topic')
                    child_topic.set('id', str(child_node_id))
                    child_topic.set('parent', str(node.id))
                    child_topic.set('guid', generate_deterministic_guid(cluster_id, child_node_id) if cluster_id else generate_guid())
                    child_topic.set('x', f'{node.x + 30:.2f}')
                    child_topic.set('y', f'{node.y + 20:.2f}')
                    child_topic.set('palette', str(node.palette))
                    child_topic.set('colorinfo', str(node.palette))
                    child_topic.set('text', child_node_text)
                    # Square style
                    child_style = SubElement(child_topic, 'style')
                    child_style.set('borderstyle', 'sbsRectangle')
                    # cloudmapref link
                    child_link = SubElement(child_topic, 'link')
                    child_link.set('cloudmapref', cloudmap_path)
                    target_guid = generate_deterministic_guid(node.url, 0)
                    child_link.set('element', target_guid)

            elif url_nodes_mode == 'map':
                cloudmap_path = get_cloudmapref_path()
                if cloudmap_path:
                    # cloudmapref on main node
                    link = SubElement(topic, 'link')
                    link.set('cloudmapref', cloudmap_path)
                    target_guid = generate_deterministic_guid(node.url, 0)
                    link.set('element', target_guid)

                    # URL on child node (square)
                    if next_id is not None:
                        child_node_id = next_id[0]
                        next_id[0] += 1
                        child_topic = SubElement(parent_element, 'topic')
                        child_topic.set('id', str(child_node_id))
                        child_topic.set('parent', str(node.id))
                        child_topic.set('guid', generate_deterministic_guid(cluster_id, child_node_id) if cluster_id else generate_guid())
                        child_topic.set('x', f'{node.x + 30:.2f}')
                        child_topic.set('y', f'{node.y + 20:.2f}')
                        child_topic.set('palette', str(node.palette))
                        child_topic.set('colorinfo', str(node.palette))
                        child_topic.set('text', child_node_text)
                        # Square style
                        child_style = SubElement(child_topic, 'style')
                        child_style.set('borderstyle', 'sbsRectangle')
                        # URL link
                        child_link = SubElement(child_topic, 'link')
                        child_link.set('urllink', node.url)
                else:
                    # Fallback: target not in index, just use URL on main node
                    link = SubElement(topic, 'link')
                    link.set('urllink', node.url)

            elif url_nodes_mode == 'url-label':
                cloudmap_path = get_cloudmapref_path()
                if cloudmap_path:
                    # cloudmapref on main node (for map navigation)
                    link = SubElement(topic, 'link')
                    link.set('cloudmapref', cloudmap_path)
                    target_guid = generate_deterministic_guid(node.url, 0)
                    link.set('element', target_guid)

                    # URL on child node with "url" label and no border
                    if next_id is not None:
                        child_node_id = next_id[0]
                        next_id[0] += 1
                        child_topic = SubElement(parent_element, 'topic')
                        child_topic.set('id', str(child_node_id))
                        child_topic.set('parent', str(node.id))
                        child_topic.set('guid', generate_deterministic_guid(cluster_id, child_node_id) if cluster_id else generate_guid())
                        # Position along radial direction from center (500, 500)
                        center_x, center_y = 500, 500
                        dx = node.x - center_x
                        dy = node.y - center_y
                        dist = (dx*dx + dy*dy) ** 0.5
                        if dist > 0:
                            # Offset 60 pixels further along radial direction
                            url_x = node.x + (dx / dist) * 60
                            url_y = node.y + (dy / dist) * 60
                        else:
                            url_x = node.x + 60
                            url_y = node.y
                        child_topic.set('x', f'{url_x:.2f}')
                        child_topic.set('y', f'{url_y:.2f}')
                        child_topic.set('palette', str(node.palette))
                        child_topic.set('colorinfo', str(node.palette))
                        child_topic.set('text', 'url')
                        # No border style
                        child_style = SubElement(child_topic, 'style')
                        child_style.set('borderstyle', 'sbsNone')
                        # URL link
                        child_link = SubElement(child_topic, 'link')
                        child_link.set('urllink', node.url)
                else:
                    # Fallback: target not in index, just use URL on main node
                    link = SubElement(topic, 'link')
                    link.set('urllink', node.url)

            else:
                # No child node mode - cloudmapref on main only if available, else URL
                cloudmap_path = get_cloudmapref_path()
                link = SubElement(topic, 'link')
                if cloudmap_path:
                    link.set('cloudmapref', cloudmap_path)
                    target_guid = generate_deterministic_guid(node.url, 0)
                    link.set('element', target_guid)
                else:
                    link.set('urllink', node.url)

        # Handle RefPearl/AliasPearl with alias_target_uri (links to other mindmaps)
        elif enable_cloudmapref and node.alias_target_uri and node.item_type in ('RefPearl', 'AliasPearl'):
            # These pearls link to other trees - need cloudmapref to target mindmap
            target_url = node.alias_target_uri
            target_tree_id = extract_tree_id(target_url)

            def get_alias_cloudmapref_path():
                # Get the actual filename (titled or just ID)
                if url_to_filename and target_url in url_to_filename:
                    target_filename = url_to_filename[target_url]
                else:
                    target_filename = target_tree_id

                if cluster_to_folder and source_folder is not None:
                    # Use target's folder from mapping, or root folder as fallback
                    target_folder = cluster_to_folder.get(target_url, Path('.'))
                    return compute_relative_cloudmapref_path(source_folder, target_filename, target_folder)
                elif index_store and target_tree_id and output_path:
                    # Use index to look up target path
                    # Index keys are just digits (e.g., "11460410"), not "id11460410"
                    lookup_id = target_tree_id[2:] if target_tree_id.startswith('id') else target_tree_id
                    target_path = index_store.get(lookup_id)
                    if target_path:
                        base_dir = index_store.base_dir or str(output_path.parent)
                        output_dir = str(output_path.parent.resolve())
                        target_abs = Path(base_dir) / target_path
                        try:
                            return os.path.relpath(target_abs, output_dir)
                        except ValueError:
                            return target_path
                    return None  # Target not in index
                else:
                    return f'./{target_filename}.smmx'

            if target_tree_id:
                alias_path = get_alias_cloudmapref_path()
                if alias_path:
                    # cloudmapref on main node (to navigate to target mindmap)
                    link = SubElement(topic, 'link')
                    link.set('cloudmapref', alias_path)
                    target_guid = generate_deterministic_guid(target_url, 0)
                    link.set('element', target_guid)

                    # URL on child node (to open in browser)
                    if next_id is not None and node.url:
                        child_node_id = next_id[0]
                        next_id[0] += 1
                        child_topic = SubElement(parent_element, 'topic')
                        child_topic.set('id', str(child_node_id))
                        child_topic.set('parent', str(node.id))
                        child_topic.set('guid', generate_deterministic_guid(cluster_id, child_node_id) if cluster_id else generate_guid())
                        child_topic.set('x', f'{node.x + 30:.2f}')
                        child_topic.set('y', f'{node.y + 20:.2f}')
                        child_topic.set('palette', str(node.palette))
                        child_topic.set('colorinfo', str(node.palette))
                        child_topic.set('text', 'url')  # Label for URL node
                        # Square style
                        child_style = SubElement(child_topic, 'style')
                        child_style.set('borderstyle', 'sbsRectangle')
                        # URL link - use alias_target_uri (target tree URL) not node.url (pearl item URL)
                        child_link = SubElement(child_topic, 'link')
                        child_link.set('urllink', target_url)
                else:
                    # Alias target not in index - just use URL
                    link = SubElement(topic, 'link')
                    link.set('urllink', node.url)
            else:
                # No target tree ID - just use URL
                link = SubElement(topic, 'link')
                link.set('urllink', node.url)

        else:
            # Non-Tree nodes without alias_target_uri, or root: use urllink
            # Check for 'expanded' mode with PagePearl external URLs
            if url_nodes_mode == 'expanded' and node.item_type == 'PagePearl' and node.external_url:
                # Main node links to Pearltrees
                link = SubElement(topic, 'link')
                link.set('urllink', node.url)

                # Create child node with domain name linking to external URL
                if next_id is not None:
                    child_node_id = next_id[0]
                    next_id[0] += 1
                    child_topic = SubElement(parent_element, 'topic')
                    child_topic.set('id', str(child_node_id))
                    child_topic.set('parent', str(node.id))
                    child_topic.set('guid', generate_deterministic_guid(cluster_id, child_node_id) if cluster_id else generate_guid())
                    child_topic.set('x', f'{node.x + 30:.2f}')
                    child_topic.set('y', f'{node.y + 20:.2f}')
                    # Use domain name as the label
                    domain = extract_domain(node.external_url)
                    child_topic.set('text', domain or 'link')
                    # No border style (sbsNone)
                    child_style = SubElement(child_topic, 'style')
                    child_style.set('borderstyle', 'sbsNone')
                    # URL link to external site
                    child_link = SubElement(child_topic, 'link')
                    child_link.set('urllink', node.external_url)
            elif node.item_type == 'PagePearl' and node.external_url:
                # Default behavior: PagePearl links directly to external URL
                link = SubElement(topic, 'link')
                link.set('urllink', node.external_url)
            elif node.url:
                link = SubElement(topic, 'link')
                link.set('urllink', node.url)

    # Recurse for children
    for child in node.children:
        node_to_xml(child, parent_element, scales, tree_style, pearl_style,
                    enable_cloudmapref, cluster_id, url_nodes_mode, next_id, child_node_text,
                    source_folder, cluster_to_folder, url_to_filename, layout,
                    index_store, output_path)

def generate_mindmap_xml(root: MindMapNode, title: str, scales: Dict[int, float] = None,
                         tree_style: str = None, pearl_style: str = None,
                         enable_cloudmapref: bool = False, cluster_id: str = None,
                         url_nodes_mode: str = None, child_node_text: str = '',
                         source_folder: Path = None,
                         cluster_to_folder: Dict[str, Path] = None,
                         parent_tree_id: str = None,
                         parent_cloudmapref: str = None,
                         url_to_filename: Dict[str, str] = None,
                         parent_title: str = None,
                         layout: str = None,
                         index_store = None,
                         output_path: Path = None) -> str:
    """Generate complete SimpleMind XML document.

    Args:
        root: Root node of the mind map
        title: Mind map title
        scales: Font scale mapping by node ID
        tree_style: Borderstyle for Tree items (folders)
        pearl_style: Borderstyle for Pearl items (pages/links)
        enable_cloudmapref: If True, add cloudmapref links for Tree nodes
        cluster_id: The cluster ID for deterministic GUID generation
        url_nodes_mode: 'url', 'map', or None for child node behavior
        child_node_text: Text label for child nodes (default: empty/unlabeled)
        source_folder: Current .smmx folder (relative to output root) for path computation
        cluster_to_folder: Dict mapping cluster URLs to folder paths for MST-based layout
        parent_tree_id: Tree ID of the parent cluster (for parent links)
        parent_cloudmapref: Relative path to parent .smmx file (for parent links)
        url_to_filename: Dict mapping cluster URLs to their actual filenames
        parent_title: Title of the parent folder (for parent link label)
        layout: Layout mode - 'radial-auto' adds native radial layout element to root topic
    """
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
    scroll.set('zoom', '40')  # Zoom out more for larger maps
    scroll.set('x', str(int(-root.x)))
    scroll.set('y', str(int(-root.y)))
    main_theme = SubElement(meta, 'main-centraltheme')
    main_theme.set('id', '0')

    # Topics section
    topics = SubElement(mindmap, 'topics')
    # Initialize next_id counter for child link nodes (start after all regular nodes)
    all_nodes = collect_all_nodes(root)
    next_id = [max(n.id for n in all_nodes) + 1000]  # Leave gap for safety
    node_to_xml(root, topics, scales, tree_style, pearl_style,
                enable_cloudmapref, cluster_id, url_nodes_mode, next_id, child_node_text,
                source_folder, cluster_to_folder, url_to_filename, layout,
                index_store, output_path)

    # Add parent link node if parent info provided
    if parent_tree_id and parent_cloudmapref:
        parent_node_id = next_id[0]
        next_id[0] += 1
        create_parent_link_node(topics, root, parent_tree_id, parent_cloudmapref, parent_node_id, parent_title)

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
                       cluster_url: str = None,
                       pearls_only: bool = False,
                       children_index: Path = None) -> List[Dict]:
    """Load items from a cluster.

    Args:
        data_path: Path to JSONL data file
        cluster_query: Search by title (optional)
        cluster_url: Load items with this cluster_id (optional)
        pearls_only: If True, only load Pearls (PagePearl, AliasPearl, RefPearl),
            not child Trees. Use for curated mode where each Tree gets its own
            mindmap and child Trees are linked via cloudmapref.
        children_index: Path to SQLite children index for loading missing children

    Returns:
        List of item dicts (folder + children)
    """
    items = []
    folder_item = None

    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)

            # Find the folder itself (where URI = cluster_url) - always include
            if cluster_url and rec.get('uri') == cluster_url:
                folder_item = rec

            # Match by cluster URL - find children
            if cluster_url and rec.get('cluster_id') == cluster_url:
                # Skip the folder itself (it's added separately)
                if rec.get('uri') == cluster_url:
                    continue
                # In pearls_only mode, skip child Trees (they have their own mindmaps)
                if pearls_only and rec.get('type') == 'Tree':
                    continue
                items.append(rec)

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
                        if pearls_only and rec.get('type') == 'Tree':
                            continue
                        items.append(rec)

    # If we have a children index and few/no children, try loading from index
    if children_index and folder_item:
        tree_id = folder_item.get('tree_id', '')
        # Count non-folder items (children)
        child_count = len([i for i in items if i != folder_item])

        # If only 0-1 children (root only), try loading from index
        if child_count <= 1 and tree_id:
            index_children = load_children_from_index(children_index, tree_id)
            if index_children:
                # Get existing URIs to avoid duplicates
                existing_uris = {i.get('uri') for i in items}
                # Add children from index that aren't already present
                added = 0
                for child in index_children:
                    # Skip RootPearl (it's implicit in the folder)
                    if child.get('type') == 'RootPearl':
                        continue
                    if child.get('uri') not in existing_uris:
                        # In pearls_only mode, skip Trees
                        if pearls_only and child.get('type') == 'Tree':
                            continue
                        items.append(child)
                        added += 1
                if added > 0:
                    print(f"  Loaded {added} children from index for tree {tree_id}")

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

    def get_url(item):
        """Get URL for an item (main link).

        Returns Pearltrees URI for all items. For PagePearls, external URL
        is stored separately in external_url field and used based on url_nodes_mode.
        - Section: no URL (just organizational headers)
        - All others: Pearltrees URI
        """
        item_type = item.get('type', '')
        if item_type == 'Section':
            # Sections are organizational headers - no link needed
            return ''
        else:
            # Use Pearltrees URI for all items
            return item.get('uri') or item.get('pearl_uri', '')

    def get_external_url(item):
        """Get external URL for PagePearls (for 'expanded' url_nodes_mode)."""
        if item.get('type') == 'PagePearl':
            return item.get('url', '')
        return ''

    def get_pearltrees_uri(item):
        """Get Pearltrees URI for an item."""
        return item.get('uri') or item.get('pearl_uri', '')

    # Create root node
    if root_item:
        tree_id = root_item.get('tree_id', '')
        root_node = MindMapNode(
            id=0,
            title=get_disambiguated_title(root_item) or 'Root',
            tree_id=tree_id,
            url=get_url(root_item),
            parent_id=-1,
            palette=1,
            embedding=get_embedding(root_item),
            item_type=root_item.get('type', ''),
            alias_target_uri=root_item.get('alias_target_uri', ''),
            external_url=get_external_url(root_item)
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
            title=get_disambiguated_title(item) or f'Item {i}',
            tree_id=tree_id,
            url=get_url(item),
            parent_id=0,  # Will be updated by build_hierarchy
            palette=(palette_idx % 8) + 1,
            embedding=get_embedding(item),
            item_type=item.get('type', ''),
            alias_target_uri=item.get('alias_target_uri', ''),
            external_url=get_external_url(item)
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


def generate_single_map(cluster_url: str, data_path: Path, output_path: Path,
                        tree_id_emb: Dict[str, np.ndarray],
                        title_emb: Dict[str, np.ndarray],
                        uri_emb: Dict[str, np.ndarray],
                        min_children: int = 4, max_children: int = 8,
                        optimize: bool = False, optimize_iterations: int = 100,
                        do_minimize_crossings: bool = False, crossing_passes: int = 10,
                        no_scaling: bool = False, tree_style: str = None,
                        pearl_style: str = None, enable_cloudmapref: bool = False,
                        url_nodes_mode: str = None, child_node_text: str = '',
                        xml_only: bool = False,
                        source_folder: Path = None,
                        cluster_to_folder: Dict[str, Path] = None,
                        parent_tree_id: str = None,
                        parent_cloudmapref: str = None,
                        url_to_filename: Dict[str, str] = None,
                        parent_title: str = None,
                        layout: str = 'radial',
                        flat_hierarchy: bool = False,
                        children_index: Path = None) -> List[str]:
    """Generate a single mind map and return URLs of child Trees for recursion.

    Args:
        cluster_url: URL of the cluster to generate
        data_path: Path to JSONL data file
        output_path: Output .smmx file path
        tree_id_emb, title_emb, uri_emb: Embedding dictionaries
        min_children, max_children: Hierarchy parameters
        optimize, optimize_iterations: Force-directed optimization
        do_minimize_crossings, crossing_passes: Edge crossing minimization
        no_scaling: Disable font scaling
        tree_style, pearl_style: Node styles
        enable_cloudmapref: Add cloudmapref links
        url_nodes_mode: 'url', 'map', or None for child node behavior
        child_node_text: Text label for child link nodes
        xml_only: Output raw XML instead of .smmx
        source_folder: Current .smmx folder (relative to output root) for path computation
        cluster_to_folder: Dict mapping cluster URLs to folder paths for MST-based layout
        parent_tree_id: Tree ID of the parent cluster (for parent links)
        parent_cloudmapref: Relative path to parent .smmx file (for parent links)
        url_to_filename: Dict mapping cluster URLs to their actual filenames
        parent_title: Title of the parent folder (for parent link label)
        flat_hierarchy: If True, use hierarchical clustering to organize items.
            Items are still only direct children of the tree (per RDF), but
            organized into a balanced hierarchy using Ward's method.
            Use for curated folders mode to preserve true Pearltrees hierarchy.

    Returns:
        List of child Tree URLs for recursive generation
    """
    # Load cluster items
    # In curated mode (flat_hierarchy), only load pearls - child Trees have their own mindmaps
    items = load_cluster_items(data_path, cluster_url=cluster_url,
                               pearls_only=flat_hierarchy,
                               children_index=children_index)

    if not items:
        print(f"No items found for cluster: {cluster_url}")
        return []

    print(f"Generating map for: {cluster_url} ({len(items)} items)")

    # Create nodes with embeddings
    nodes = create_nodes_from_items(items, tree_id_emb, title_emb, uri_emb)

    # Build hierarchy - balanced clustering or K-means micro-clustering
    if flat_hierarchy:
        # Hierarchical clustering: balanced tree using Ward's method
        # Items are only direct children per RDF, organized semantically
        root = build_mst_tree_hierarchy(nodes)
    else:
        # K-means micro-clustering: creates nested groups with cluster representatives
        root = build_hierarchy(nodes, min_children=min_children, max_children=max_children)

    # Apply layout (skip for 'radial-auto' - native software handles it)
    if layout == 'radial':
        apply_radial_tree_layout(root, center_x=500, center_y=500, min_node_spacing=60, base_radius=120)
    elif layout == 'radial-freeform':
        apply_radial_layout(root, center_x=500, center_y=500, min_spacing=100, base_radius=180)
    # else: 'radial-auto' - use default positions, native software handles layout

    # Optional: force-directed optimization (only for algorithmic layouts)
    if optimize and layout != 'radial-auto':
        force_directed_optimize(root, center_x=500, center_y=500,
                               iterations=optimize_iterations)

    # Optional: edge crossing minimization (only for algorithmic layouts)
    if do_minimize_crossings and layout != 'radial-auto':
        if not optimize:
            force_directed_optimize(root, center_x=500, center_y=500, iterations=300)
        minimize_crossings(root, center_x=500, center_y=500, max_passes=crossing_passes)

    # Calculate node scales
    scales = None
    if not no_scaling:
        scales = calculate_node_scales(root)

    # Generate XML with cloudmapref if enabled
    title = root.title if root else "Mind Map"
    xml_content = generate_mindmap_xml(root, title, scales,
                                       tree_style=tree_style, pearl_style=pearl_style,
                                       enable_cloudmapref=enable_cloudmapref,
                                       cluster_id=cluster_url,
                                       url_nodes_mode=url_nodes_mode,
                                       child_node_text=child_node_text,
                                       source_folder=source_folder,
                                       cluster_to_folder=cluster_to_folder,
                                       parent_tree_id=parent_tree_id,
                                       parent_cloudmapref=parent_cloudmapref,
                                       url_to_filename=url_to_filename,
                                       parent_title=parent_title,
                                       layout=layout)

    # Write output
    if xml_only:
        out = output_path.with_suffix('.xml')
        out.write_text(xml_content)
        print(f"Written: {out}")
    else:
        write_smmx(xml_content, output_path)

    # Collect child Tree URLs for recursive generation
    # Include both direct Tree children and AliasPearl/RefPearl targets
    child_tree_urls = []
    all_nodes = collect_all_nodes(root)
    for node in all_nodes:
        if node.id == 0:
            continue
        # Direct Tree children
        if node.item_type == 'Tree' and node.url:
            child_tree_urls.append(node.url)
        # AliasPearl/RefPearl links to other trees
        elif node.alias_target_uri:
            child_tree_urls.append(node.alias_target_uri)

    return child_tree_urls


def discover_all_clusters(
    root_url: str,
    data_path: Path,
    max_depth: int = None
) -> List[str]:
    """Pre-discover all cluster URLs in hierarchy for MST computation.

    Uses BFS to find all Tree nodes that will be generated.

    Args:
        root_url: Starting cluster URL
        data_path: Path to JSONL data file
        max_depth: Maximum depth (None = unlimited)

    Returns:
        List of all cluster URLs in the hierarchy
    """
    discovered = []
    queue = [(root_url, 0)]
    visited = set()

    while queue:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        if max_depth is not None and depth > max_depth:
            continue

        visited.add(url)
        discovered.append(url)

        # Find child Trees and AliasPearl/RefPearl targets
        items = load_cluster_items(data_path, cluster_url=url)
        for item in items:
            child_url = None
            if item.get('type') == 'Tree':
                child_url = item.get('uri', '')
            elif item.get('alias_target_uri'):
                # AliasPearl/RefPearl link to another tree
                child_url = item.get('alias_target_uri')

            if child_url and child_url != url and child_url not in visited:
                queue.append((child_url, depth + 1))

    return discovered


def compute_cluster_centroids(
    cluster_urls: List[str],
    data_path: Path,
    tree_id_emb: Dict[str, np.ndarray],
    title_emb: Dict[str, np.ndarray],
    uri_emb: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute centroid embedding for each cluster.

    Args:
        cluster_urls: List of cluster URLs to compute centroids for
        data_path: Path to JSONL data file
        tree_id_emb, title_emb, uri_emb: Embedding dictionaries

    Returns:
        Tuple of (cluster_centroids dict, global_centroid)
    """
    def get_embedding(item):
        """Get embedding for an item."""
        pearl_uri = item.get('pearl_uri', '')
        if pearl_uri and pearl_uri in uri_emb:
            return uri_emb[pearl_uri]
        uri = item.get('uri', '')
        if uri and uri in uri_emb:
            return uri_emb[uri]
        tree_id = item.get('tree_id', '')
        if tree_id and str(tree_id) in tree_id_emb:
            return tree_id_emb[str(tree_id)]
        title = item.get('raw_title', '')
        if title in title_emb:
            return title_emb[title]
        return None

    cluster_centroids = {}
    all_embeddings = []

    # Pre-load and index data file for efficiency
    print("    Loading data file into memory...")
    cluster_to_items = {}
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)
            cluster_id = rec.get('cluster_id', '')
            if cluster_id:
                if cluster_id not in cluster_to_items:
                    cluster_to_items[cluster_id] = []
                cluster_to_items[cluster_id].append(rec)
    print(f"    Indexed {len(cluster_to_items)} clusters")

    total = len(cluster_urls)
    for i, cluster_url in enumerate(cluster_urls):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"    Computing centroids: {i+1}/{total} ({100*(i+1)//total}%)", flush=True)

        items = cluster_to_items.get(cluster_url, [])
        embeddings = []
        for item in items:
            emb = get_embedding(item)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            cluster_centroids[cluster_url] = centroid
            all_embeddings.append(centroid)

    # Global centroid is mean of all cluster centroids
    if all_embeddings:
        global_centroid = np.mean(all_embeddings, axis=0)
    else:
        global_centroid = np.zeros(768)  # Default embedding dimension

    return cluster_centroids, global_centroid


def build_mst_hierarchy(
    cluster_urls: List[str],
    cluster_centroids: Dict[str, np.ndarray],
    global_centroid: np.ndarray
) -> Tuple[Dict[str, List[str]], str]:
    """Build MST from cluster centroids and return parent->children mapping.

    Args:
        cluster_urls: List of cluster URLs
        cluster_centroids: Dict mapping cluster_url -> centroid embedding
        global_centroid: Mean of all cluster centroids

    Returns:
        Tuple of (parent_to_children dict, root_url)
    """
    # Filter to clusters that have centroids
    valid_urls = [url for url in cluster_urls if url in cluster_centroids]

    if len(valid_urls) == 0:
        return {}, cluster_urls[0] if cluster_urls else ""

    if len(valid_urls) == 1:
        return {valid_urls[0]: []}, valid_urls[0]

    # Stack centroids into matrix
    centroids_matrix = np.stack([cluster_centroids[url] for url in valid_urls])

    # Compute pairwise cosine distances
    distances = squareform(pdist(centroids_matrix, metric='cosine'))

    # Build MST
    mst = minimum_spanning_tree(distances)
    mst_array = mst.toarray()

    # Find root: cluster closest to global centroid (using cosine distance)
    dists_to_global = []
    for url in valid_urls:
        centroid = cluster_centroids[url]
        # Cosine distance = 1 - cosine similarity
        cos_sim = np.dot(centroid, global_centroid) / (
            np.linalg.norm(centroid) * np.linalg.norm(global_centroid) + 1e-10)
        dists_to_global.append(1 - cos_sim)
    root_idx = int(np.argmin(dists_to_global))
    root_url = valid_urls[root_idx]

    # BFS from root to build parent->children hierarchy
    # MST edges are directed from lower to higher index in scipy
    # Make symmetric for BFS
    mst_symmetric = mst_array + mst_array.T

    parent_to_children = {url: [] for url in valid_urls}
    visited = set()
    queue = [root_idx]
    visited.add(root_idx)

    while queue:
        current_idx = queue.pop(0)
        current_url = valid_urls[current_idx]

        # Find neighbors in MST
        for neighbor_idx in range(len(valid_urls)):
            if mst_symmetric[current_idx, neighbor_idx] > 0 and neighbor_idx not in visited:
                visited.add(neighbor_idx)
                neighbor_url = valid_urls[neighbor_idx]
                parent_to_children[current_url].append(neighbor_url)
                queue.append(neighbor_idx)

    return parent_to_children, root_url


def mst_to_folder_structure(
    parent_to_children: Dict[str, List[str]],
    root_url: str,
    max_folder_depth: int = None,
    min_folder_children: int = None,
    max_folder_children: int = None
) -> Dict[str, Path]:
    """Convert MST hierarchy to folder paths.

    Args:
        parent_to_children: MST structure mapping parent -> children URLs
        root_url: Root cluster URL
        max_folder_depth: Maximum folder nesting depth (None = unlimited)
        min_folder_children: Minimum children to create subfolders (None = no min)
        max_folder_children: Maximum children to put in subfolders (None = unlimited)

    Returns:
        Dict mapping cluster_url -> relative folder Path
    """
    cluster_to_folder = {}

    def assign_folders(url: str, current_path: Path, depth: int):
        # Assign this cluster to current path
        cluster_to_folder[url] = current_path

        # Process children
        children = parent_to_children.get(url, [])
        num_children = len(children)

        # Determine if we should create subfolders for children
        create_subfolders = True
        if max_folder_depth is not None and depth >= max_folder_depth:
            create_subfolders = False
        if min_folder_children is not None and num_children < min_folder_children:
            create_subfolders = False

        for i, child_url in enumerate(children):
            child_tree_id = extract_tree_id(child_url)
            if not child_tree_id:
                child_tree_id = f"cluster_{len(cluster_to_folder)}"

            # Determine child path
            if not create_subfolders:
                # Flatten: keep in current folder
                child_path = current_path
            elif max_folder_children is not None and i >= max_folder_children:
                # Exceeded max children: flatten remaining to current folder
                child_path = current_path
            else:
                # Create subfolder
                child_path = current_path / child_tree_id

            assign_folders(child_url, child_path, depth + 1)

    # Start from root at output root (Path('.'))
    assign_folders(root_url, Path('.'), 0)

    return cluster_to_folder


# =============================================================================
# Curated Folders: Hierarchy-Preserving Folder Organization
# =============================================================================

def build_user_hierarchy(
    data_path: Path,
    primary_account: str = None
) -> Tuple[Dict[str, List[str]], Dict[str, str], str, List[str], Dict[str, str], Dict[str, str]]:
    """Build user's actual hierarchy from JSONL cluster_id relationships.

    Args:
        data_path: Path to JSONL data file
        primary_account: Primary account to use as root (None = auto-detect)

    Returns:
        Tuple of:
        - parent_to_children: Dict mapping parent URI to list of child URIs
        - child_to_parent: Dict mapping child URI to parent URI
        - root_url: The root cluster URL
        - orphans: List of cluster URLs not connected to main hierarchy
        - url_to_title: Dict mapping URI to tree title
        - url_to_account: Dict mapping URI to account name
    """
    # Load all Trees from JSONL
    trees = {}  # uri -> tree data
    parent_to_children: Dict[str, List[str]] = {}
    child_to_parent: Dict[str, str] = {}
    url_to_title: Dict[str, str] = {}
    url_to_account: Dict[str, str] = {}

    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item.get('type') == 'Tree':
                uri = item.get('uri', '')
                if uri:
                    trees[uri] = item
                    parent_to_children[uri] = []
                    url_to_title[uri] = item.get('raw_title', '')
                    url_to_account[uri] = item.get('account', 'unknown')

    # Build parent-child relationships (skip self-references)
    self_refs = set()  # Trees that reference themselves (subtree roots)
    for uri, item in trees.items():
        cluster_id = item.get('cluster_id', '')
        if cluster_id == uri:
            # Self-referencing tree - this is a subtree root
            self_refs.add(uri)
        elif cluster_id and cluster_id in trees:
            parent_to_children[cluster_id].append(uri)
            child_to_parent[uri] = cluster_id

    # Find roots: trees with no parent OR self-referencing trees
    roots = [uri for uri in trees if uri not in child_to_parent]
    # Note: self_refs are included since they won't be in child_to_parent

    # Filter roots by primary account if specified
    if primary_account:
        account_roots = [uri for uri in roots
                        if trees[uri].get('account') == primary_account]
        if account_roots:
            roots = account_roots

    # Use path hierarchy from target_text to find the true root
    # The root should be the tree that appears first in most paths
    from collections import Counter
    first_ancestor_counts = Counter()
    for uri, item in trees.items():
        target_text = item.get('target_text', '')
        if target_text.startswith('/'):
            parts = target_text.split('\n')[0].strip('/').split('/')
            if parts:
                first_ancestor_counts[parts[0]] += 1

    # Map tree_id to uri for root candidates
    tree_id_to_uri = {item.get('tree_id'): uri for uri, item in trees.items()}

    # Pick root - prefer the one that appears first in most paths
    def count_descendants(uri: str) -> int:
        count = len(parent_to_children.get(uri, []))
        for child in parent_to_children.get(uri, []):
            count += count_descendants(child)
        return count

    if first_ancestor_counts:
        # Find best root from path analysis
        best_root_id, best_count = first_ancestor_counts.most_common(1)[0]
        if best_root_id in tree_id_to_uri:
            root_url = tree_id_to_uri[best_root_id]
        elif roots:
            root_url = max(roots, key=count_descendants)
        else:
            root_url = next(iter(trees.keys())) if trees else ""
    elif roots:
        root_url = max(roots, key=count_descendants)
    else:
        # Fallback: first tree
        root_url = next(iter(trees.keys())) if trees else ""

    # Find orphans: trees not reachable from root
    reachable = set()
    queue = [root_url]
    while queue:
        current = queue.pop(0)
        if current in reachable:
            continue
        reachable.add(current)
        queue.extend(parent_to_children.get(current, []))

    orphans = [uri for uri in trees if uri not in reachable]

    return parent_to_children, child_to_parent, root_url, orphans, url_to_title, url_to_account


def cluster_trees_into_folder_groups(
    tree_urls: List[str],
    tree_centroids: Dict[str, np.ndarray],
    folder_count: int,
    method: str = 'kmeans'
) -> Dict[str, int]:
    """K-cluster trees into folder groups by centroid similarity.

    Args:
        tree_urls: List of tree URLs to cluster
        tree_centroids: Dict mapping tree URL to centroid embedding
        folder_count: Target number of folder groups (K)
        method: Clustering method ('kmeans' | 'mst-cut')

    Returns:
        Dict mapping tree_url -> folder_group_id (0 to K-1)
    """
    # Filter to trees with centroids
    valid_urls = [url for url in tree_urls if url in tree_centroids]

    if len(valid_urls) == 0:
        return {}

    if len(valid_urls) <= folder_count:
        # Each tree gets its own folder group
        return {url: i for i, url in enumerate(valid_urls)}

    centroids_matrix = np.stack([tree_centroids[url] for url in valid_urls])

    if method == 'kmeans':
        kmeans = KMeans(n_clusters=folder_count, random_state=42, n_init=10)
        labels = kmeans.fit_predict(centroids_matrix)
        return {url: int(label) for url, label in zip(valid_urls, labels)}

    elif method == 'mst-cut':
        # Build MST and cut K-1 longest edges
        distances = squareform(pdist(centroids_matrix, metric='cosine'))
        mst = minimum_spanning_tree(distances)
        mst_array = mst.toarray()

        # Find edge weights and sort
        edges = []
        for i in range(len(valid_urls)):
            for j in range(i + 1, len(valid_urls)):
                weight = mst_array[i, j] + mst_array[j, i]
                if weight > 0:
                    edges.append((weight, i, j))
        edges.sort(reverse=True)

        # Cut K-1 longest edges
        cut_edges = set()
        for weight, i, j in edges[:folder_count - 1]:
            cut_edges.add((min(i, j), max(i, j)))

        # Find connected components after cuts
        from collections import deque
        visited = set()
        components = []
        for start in range(len(valid_urls)):
            if start in visited:
                continue
            component = []
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in range(len(valid_urls)):
                    edge = (min(node, neighbor), max(node, neighbor))
                    if edge in cut_edges:
                        continue
                    weight = mst_array[node, neighbor] + mst_array[neighbor, node]
                    if weight > 0 and neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)

        # Assign folder group IDs
        tree_to_group = {}
        for group_id, component in enumerate(components):
            for idx in component:
                tree_to_group[valid_urls[idx]] = group_id
        return tree_to_group

    else:
        raise ValueError(f"Unknown clustering method: {method}")


def build_mst_with_fixed_root(
    folder_group_centroids: Dict[int, np.ndarray],
    root_folder_group: int
) -> Dict[int, List[int]]:
    """Build MST from folder group centroids with fixed root.

    Args:
        folder_group_centroids: Dict mapping folder_group_id -> centroid
        root_folder_group: The folder group ID that must be root

    Returns:
        Dict mapping parent_group_id -> list of child_group_ids
    """
    group_ids = list(folder_group_centroids.keys())

    if len(group_ids) <= 1:
        return {gid: [] for gid in group_ids}

    # Stack centroids
    centroids_matrix = np.stack([folder_group_centroids[gid] for gid in group_ids])

    # Compute pairwise distances
    distances = squareform(pdist(centroids_matrix, metric='cosine'))

    # Build MST
    mst = minimum_spanning_tree(distances)
    mst_array = mst.toarray()
    mst_symmetric = mst_array + mst_array.T

    # BFS from fixed root to build parent->children
    root_idx = group_ids.index(root_folder_group)
    parent_to_children = {gid: [] for gid in group_ids}
    visited = set()
    queue = [root_idx]
    visited.add(root_idx)

    while queue:
        current_idx = queue.pop(0)
        current_gid = group_ids[current_idx]

        for neighbor_idx in range(len(group_ids)):
            if mst_symmetric[current_idx, neighbor_idx] > 0 and neighbor_idx not in visited:
                visited.add(neighbor_idx)
                neighbor_gid = group_ids[neighbor_idx]
                parent_to_children[current_gid].append(neighbor_gid)
                queue.append(neighbor_idx)

    return parent_to_children


def curated_to_folder_structure(
    tree_to_folder_group: Dict[str, int],
    folder_group_hierarchy: Dict[int, List[int]],
    root_folder_group: int,
    max_folder_depth: int = None
) -> Dict[str, Path]:
    """Convert curated folder groups to folder paths.

    Args:
        tree_to_folder_group: Dict mapping tree_url -> folder_group_id
        folder_group_hierarchy: Dict mapping parent_group -> child_groups
        root_folder_group: Root folder group ID
        max_folder_depth: Maximum folder nesting (None = unlimited)

    Returns:
        Dict mapping tree_url -> relative folder Path
    """
    # First build group_id -> folder path
    group_to_path: Dict[int, Path] = {}

    def assign_group_paths(group_id: int, current_path: Path, depth: int):
        group_to_path[group_id] = current_path
        children = folder_group_hierarchy.get(group_id, [])

        for child_gid in children:
            if max_folder_depth is not None and depth >= max_folder_depth:
                # Flatten: keep in current folder
                child_path = current_path
            else:
                child_path = current_path / f"group_{child_gid}"
            assign_group_paths(child_gid, child_path, depth + 1)

    assign_group_paths(root_folder_group, Path('.'), 0)

    # Now map trees to paths
    tree_to_folder = {}
    for tree_url, group_id in tree_to_folder_group.items():
        tree_to_folder[tree_url] = group_to_path.get(group_id, Path('.'))

    return tree_to_folder


def curated_to_folder_structure_named(
    tree_to_folder_group: Dict[str, int],
    folder_group_hierarchy: Dict[int, List[int]],
    root_folder_group: int,
    group_names: Dict[int, str],
    max_folder_depth: int = None
) -> Dict[str, Path]:
    """Convert curated folder groups to folder paths with meaningful names.

    Args:
        tree_to_folder_group: Dict mapping tree_url -> folder_group_id
        folder_group_hierarchy: Dict mapping parent_group -> child_groups
        root_folder_group: Root folder group ID
        group_names: Dict mapping group_id -> folder name
        max_folder_depth: Maximum folder nesting (None = unlimited)

    Returns:
        Dict mapping tree_url -> relative folder Path
    """
    # First build group_id -> folder path
    group_to_path: Dict[int, Path] = {}

    def assign_group_paths(group_id: int, current_path: Path, depth: int):
        group_to_path[group_id] = current_path
        children = folder_group_hierarchy.get(group_id, [])

        for child_gid in children:
            if max_folder_depth is not None and depth >= max_folder_depth:
                # Flatten: keep in current folder
                child_path = current_path
            else:
                # Use the named folder
                folder_name = group_names.get(child_gid, f"group_{child_gid}")
                child_path = current_path / folder_name
            assign_group_paths(child_gid, child_path, depth + 1)

    assign_group_paths(root_folder_group, Path('.'), 0)

    # Now map trees to paths
    tree_to_folder = {}
    for tree_url, group_id in tree_to_folder_group.items():
        tree_to_folder[tree_url] = group_to_path.get(group_id, Path('.'))

    return tree_to_folder


def generate_folder_name_llm(
    tree_titles: List[str],
    model: str = "gemini-2.0-flash",
    timeout: int = 30
) -> Optional[str]:
    """Generate a folder name from tree titles using LLM.

    Args:
        tree_titles: List of tree titles in this folder group
        model: LLM model to use (default: gemini-2.0-flash)
        timeout: Timeout in seconds

    Returns:
        A short folder name, or None if LLM fails
    """
    import subprocess

    if not tree_titles:
        return None

    # Limit to first 10 titles to keep prompt reasonable
    sample_titles = tree_titles[:10]
    titles_text = "\n".join(f"- {t}" for t in sample_titles)
    if len(tree_titles) > 10:
        titles_text += f"\n- ... and {len(tree_titles) - 10} more"

    prompt = f"""Given these folder/category titles, generate a short (1-3 words) folder name that captures their common theme.
Output ONLY the folder name, nothing else. Use lowercase with underscores for spaces.

Titles:
{titles_text}

Folder name:"""

    try:
        result = subprocess.run(
            ["gemini", "-p", prompt, "-m", model, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            name = result.stdout.strip()
            # Sanitize the LLM output
            name = sanitize_filename(name, max_length=30)
            return name if name else None
        return None
    except Exception:
        return None


def generate_folder_names_batch(
    group_to_trees: Dict[int, List[str]],
    url_to_title: Dict[str, str],
    use_llm: bool = True,
    llm_model: str = "gemini-2.0-flash"
) -> Dict[int, str]:
    """Generate folder names for all groups.

    Args:
        group_to_trees: Dict mapping group_id -> list of tree URLs
        url_to_title: Dict mapping URL -> title
        use_llm: Whether to use LLM for naming
        llm_model: LLM model to use

    Returns:
        Dict mapping group_id -> folder name
    """
    group_names = {}

    for group_id, tree_urls in group_to_trees.items():
        titles = [url_to_title.get(url, '') for url in tree_urls if url_to_title.get(url)]

        folder_name = None
        if use_llm and titles:
            folder_name = generate_folder_name_llm(titles, model=llm_model)

        if not folder_name:
            # Fallback: use first title or group_N
            if titles:
                folder_name = sanitize_filename(titles[0], max_length=30)
            if not folder_name:
                folder_name = f"group_{group_id}"

        group_names[group_id] = folder_name

    return group_names


def build_curated_folder_structure(
    data_path: Path,
    tree_id_emb: Dict[str, np.ndarray],
    title_emb: Dict[str, np.ndarray],
    uri_emb: Dict[str, np.ndarray],
    folder_count: int = 100,
    folder_method: str = 'kmeans',
    max_folder_depth: int = None,
    primary_account: str = None,
    use_llm_naming: bool = False,
    llm_model: str = "gemini-2.0-flash"
) -> Tuple[Dict[str, Path], str, List[str], Dict[str, str]]:
    """Build complete curated folder structure.

    Args:
        data_path: Path to JSONL data file
        tree_id_emb, title_emb, uri_emb: Embedding dictionaries
        folder_count: Target number of folder groups
        folder_method: Clustering method ('kmeans' | 'mst-cut')
        max_folder_depth: Maximum folder nesting depth
        primary_account: Primary account to use as root
        use_llm_naming: Whether to use LLM for folder naming
        llm_model: LLM model to use for folder naming

    Returns:
        Tuple of:
        - cluster_to_folder: Dict mapping cluster_url -> folder Path
        - root_url: The root cluster URL
        - orphans: List of orphan cluster URLs
        - url_to_title: Dict mapping URL -> tree title
    """
    print("Building curated folder structure...")

    # Phase 1: Build user hierarchy
    parent_to_children, child_to_parent, root_url, orphans, url_to_title, url_to_account = build_user_hierarchy(
        data_path, primary_account)
    print(f"  User hierarchy: root={extract_tree_id(root_url)}, "
          f"{len(parent_to_children)} trees, {len(orphans)} orphans")

    # Get all tree URLs in main hierarchy
    all_trees = list(parent_to_children.keys())

    # Phase 2: Compute tree centroids
    tree_centroids, _ = compute_cluster_centroids(
        all_trees, data_path, tree_id_emb, title_emb, uri_emb)
    print(f"  Computed centroids for {len(tree_centroids)} trees")

    # Phase 3: K-cluster trees into folder groups
    tree_to_group = cluster_trees_into_folder_groups(
        all_trees, tree_centroids, folder_count, folder_method)
    print(f"  Clustered into {len(set(tree_to_group.values()))} folder groups")

    # Assign trees without centroids to their parent's folder group
    # (trees without children have no items hence no centroid)
    trees_without_group = [url for url in all_trees if url not in tree_to_group]
    if trees_without_group:
        print(f"  Assigning {len(trees_without_group)} trees without centroids to parent folders")
        for url in trees_without_group:
            # Find parent and use its group
            parent = child_to_parent.get(url)
            if parent and parent in tree_to_group:
                tree_to_group[url] = tree_to_group[parent]
            elif tree_to_group:
                # No parent or parent has no group - use root's group
                tree_to_group[url] = tree_to_group.get(root_url, 0)
            else:
                # Fallback to group 0
                tree_to_group[url] = 0

    # Find which folder group contains the root
    root_group = tree_to_group.get(root_url, 0)

    # Phase 4: Compute folder group centroids
    group_centroids: Dict[int, np.ndarray] = {}
    group_to_trees: Dict[int, List[str]] = {}
    for group_id in set(tree_to_group.values()):
        group_trees = [url for url, gid in tree_to_group.items() if gid == group_id]
        group_to_trees[group_id] = group_trees
        embeddings = [tree_centroids[url] for url in group_trees if url in tree_centroids]
        if embeddings:
            group_centroids[group_id] = np.mean(embeddings, axis=0)

    # Phase 5: Build MST with fixed root
    folder_group_hierarchy = build_mst_with_fixed_root(group_centroids, root_group)
    print(f"  MST built with root at group {root_group}")

    # Phase 6: Generate folder names
    if use_llm_naming:
        print(f"  Generating folder names with LLM ({llm_model})...")
    group_names = generate_folder_names_batch(
        group_to_trees, url_to_title, use_llm=use_llm_naming, llm_model=llm_model)

    # Phase 7: Convert to folder paths with names
    cluster_to_folder = curated_to_folder_structure_named(
        tree_to_group, folder_group_hierarchy, root_group, group_names, max_folder_depth)

    # Handle orphans - put in _orphans/{account}/ folder
    for orphan_url in orphans:
        account = url_to_account.get(orphan_url, 'unknown')
        orphan_folder = Path('_orphans') / account
        cluster_to_folder[orphan_url] = orphan_folder

    print(f"  Folder structure computed")

    return cluster_to_folder, root_url, orphans, url_to_title


def generate_recursive(cluster_url: str, data_path: Path, output_dir: Path,
                       tree_id_emb: Dict[str, np.ndarray],
                       title_emb: Dict[str, np.ndarray],
                       uri_emb: Dict[str, np.ndarray],
                       max_depth: int = None, current_depth: int = 0,
                       visited: set = None,
                       mst_folders: bool = False,
                       curated_folders: bool = False,
                       cluster_to_folder: Dict[str, Path] = None,
                       max_folder_depth: int = None,
                       min_folder_children: int = None,
                       max_folder_children: int = None,
                       folder_count: int = 100,
                       folder_method: str = 'kmeans',
                       primary_account: str = None,
                       use_llm_naming: bool = False,
                       llm_model: str = "gemini-2.0-flash",
                       use_titled_files: bool = False,
                       url_to_title: Dict[str, str] = None,
                       url_to_filename: Dict[str, str] = None,
                       parent_links: bool = False,
                       parent_url: str = None,
                       url_to_folder: Dict[str, Path] = None,
                       curated_root_url: str = None,
                       save_folder_map: Path = None,
                       load_folder_map: Path = None,
                       children_index: Path = None,
                       **kwargs) -> int:
    """Recursively generate mind maps for a cluster hierarchy.

    Args:
        cluster_url: Starting cluster URL
        data_path: Path to JSONL data file
        output_dir: Output directory for all .smmx files
        tree_id_emb, title_emb, uri_emb: Embedding dictionaries
        max_depth: Maximum recursion depth (None = unlimited)
        current_depth: Current depth in recursion
        visited: Set of already-visited cluster URLs (prevents infinite loops)
        mst_folders: If True, organize output into MST-based subfolder hierarchy
        curated_folders: If True, use curated folder structure (preserves user hierarchy)
        cluster_to_folder: Pre-computed mapping of cluster URLs to folder paths
        max_folder_depth: Maximum subfolder nesting depth
        min_folder_children: Minimum children to create subfolders
        max_folder_children: Maximum children to put in subfolders
        folder_count: Target number of folder groups for curated folders
        folder_method: Clustering method for curated folders ('kmeans' | 'mst-cut')
        primary_account: Primary account for curated folders root selection
        parent_links: If True, add "back to parent" nodes in child maps
        parent_url: URL of the parent cluster (for parent links)
        url_to_folder: Mapping of cluster URLs to folder paths (for parent links)
        curated_root_url: Root URL determined by curated folders (for starting generation)
        save_folder_map: If provided, save computed folder structure to this JSON file
        load_folder_map: If provided, load folder structure from this JSON file (skips computation)
        **kwargs: Additional arguments passed to generate_single_map

    Returns:
        Total number of maps generated
    """
    if visited is None:
        visited = set()

        # Initialize url_to_folder for parent_links mode (tracks where each map ends up)
        if parent_links and url_to_folder is None:
            url_to_folder = {}

        # Initialize url_to_filename mapping (tracks actual filenames for cloudmapref links)
        if url_to_filename is None:
            url_to_filename = {}

        # On first call with curated_folders, build or load curated structure
        if curated_folders and cluster_to_folder is None:
            # Try to load from file if specified
            if load_folder_map and load_folder_map.exists():
                print(f"Loading folder map from {load_folder_map}...")
                with open(load_folder_map) as f:
                    folder_data = json.load(f)
                cluster_to_folder = {url: Path(path) for url, path in folder_data['cluster_to_folder'].items()}
                curated_root_url = folder_data['root_url']
                orphans = folder_data.get('orphans', [])
                url_to_title = folder_data.get('url_to_title', {})
                print(f"  Loaded {len(cluster_to_folder)} folder mappings, root={extract_tree_id(curated_root_url)}")
            else:
                # Build from scratch
                cluster_to_folder, curated_root_url, orphans, url_to_title = build_curated_folder_structure(
                    data_path, tree_id_emb, title_emb, uri_emb,
                    folder_count=folder_count,
                    folder_method=folder_method,
                    max_folder_depth=max_folder_depth,
                    primary_account=primary_account,
                    use_llm_naming=use_llm_naming,
                    llm_model=llm_model
                )

                # Save folder map if requested
                if save_folder_map:
                    print(f"Saving folder map to {save_folder_map}...")
                    folder_data = {
                        'cluster_to_folder': {url: str(path) for url, path in cluster_to_folder.items()},
                        'root_url': curated_root_url,
                        'orphans': orphans,
                        'url_to_title': url_to_title
                    }
                    save_folder_map.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_folder_map, 'w') as f:
                        json.dump(folder_data, f, indent=2)
                    print(f"  Saved {len(cluster_to_folder)} folder mappings")

            # Start from curated root, not the passed cluster_url
            cluster_url = curated_root_url
            if orphans:
                print(f"  Note: {len(orphans)} orphan trees will be placed in _orphans/")

            # Build url_to_filename upfront if titled_files is enabled
            if use_titled_files and url_to_title:
                for url, title in url_to_title.items():
                    tree_id = extract_tree_id(url)
                    if tree_id:
                        url_to_filename[url] = generate_filename_from_title(title, tree_id, include_id=True)

        # On first call with mst_folders, build MST structure
        elif mst_folders and cluster_to_folder is None:
            print("Building MST folder structure...")
            # Discover all clusters
            all_clusters = discover_all_clusters(cluster_url, data_path, max_depth)
            print(f"  Found {len(all_clusters)} clusters")

            # Compute centroids
            cluster_centroids, global_centroid = compute_cluster_centroids(
                all_clusters, data_path, tree_id_emb, title_emb, uri_emb)
            print(f"  Computed centroids for {len(cluster_centroids)} clusters")

            # Build MST hierarchy
            parent_to_children, mst_root = build_mst_hierarchy(
                all_clusters, cluster_centroids, global_centroid)
            print(f"  MST root: {extract_tree_id(mst_root)}")

            # Convert to folder structure
            cluster_to_folder = mst_to_folder_structure(
                parent_to_children, mst_root, max_folder_depth,
                min_folder_children, max_folder_children)
            print(f"  Folder structure computed")

        # For non-curated/mst modes, build url_to_title from data file
        # This is needed for parent link titles and titled files
        if url_to_title is None and (parent_links or use_titled_files):
            url_to_title = {}
            with open(data_path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get('type') == 'Tree':
                        uri = rec.get('uri', '')
                        title = rec.get('raw_title', '')
                        if uri and title:
                            url_to_title[uri] = title

            # Build url_to_filename upfront if titled_files is enabled
            if use_titled_files:
                for url, title in url_to_title.items():
                    tree_id = extract_tree_id(url)
                    if tree_id:
                        url_to_filename[url] = generate_filename_from_title(title, tree_id, include_id=True)

    # Prevent infinite loops (e.g., circular references)
    if cluster_url in visited:
        return 0

    # Check depth limit
    if max_depth is not None and current_depth > max_depth:
        return 0

    visited.add(cluster_url)

    # Determine output path
    tree_id = extract_tree_id(cluster_url)
    if not tree_id:
        tree_id = f"cluster_{len(visited)}"

    # Generate filename (with title if enabled)
    if use_titled_files and url_to_title and cluster_url in url_to_title:
        title = url_to_title[cluster_url]
        filename = generate_filename_from_title(title, tree_id, include_id=True)
    else:
        filename = tree_id

    # Record this cluster's filename for cloudmapref links
    if url_to_filename is not None:
        url_to_filename[cluster_url] = filename

    if (mst_folders or curated_folders) and cluster_to_folder and cluster_url in cluster_to_folder:
        folder = cluster_to_folder[cluster_url]
        output_path = output_dir / folder / f"{filename}.smmx"
        source_folder = folder
    else:
        output_path = output_dir / f"{filename}.smmx"
        source_folder = Path('.')

    # Track this cluster's folder for parent_links
    if parent_links and url_to_folder is not None:
        url_to_folder[cluster_url] = source_folder

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute parent link info if parent_links enabled and we have a parent
    parent_tree_id = None
    parent_cloudmapref = None
    parent_title = None
    if parent_links and parent_url:
        parent_tree_id = extract_tree_id(parent_url)
        if parent_tree_id:
            # Get parent's folder and filename (should have been recorded)
            parent_folder = url_to_folder.get(parent_url, Path('.')) if url_to_folder else Path('.')
            parent_filename = url_to_filename.get(parent_url, parent_tree_id) if url_to_filename else parent_tree_id
            parent_cloudmapref = compute_relative_cloudmapref_path(
                source_folder, parent_filename, parent_folder
            )
            # Get parent title from url_to_title if available
            if url_to_title:
                parent_title = url_to_title.get(parent_url)

    # Generate this cluster's map
    child_urls = generate_single_map(
        cluster_url=cluster_url,
        data_path=data_path,
        output_path=output_path,
        tree_id_emb=tree_id_emb,
        title_emb=title_emb,
        uri_emb=uri_emb,
        enable_cloudmapref=True,  # Always enable for recursive mode
        source_folder=source_folder,
        cluster_to_folder=cluster_to_folder,
        parent_tree_id=parent_tree_id,
        parent_cloudmapref=parent_cloudmapref,
        url_to_filename=url_to_filename,
        parent_title=parent_title,
        flat_hierarchy=curated_folders,  # Preserve true hierarchy in curated mode
        children_index=children_index,
        **kwargs
    )

    total_generated = 1

    # Recursively generate child maps
    for child_url in child_urls:
        total_generated += generate_recursive(
            cluster_url=child_url,
            data_path=data_path,
            output_dir=output_dir,
            tree_id_emb=tree_id_emb,
            title_emb=title_emb,
            uri_emb=uri_emb,
            max_depth=max_depth,
            current_depth=current_depth + 1,
            visited=visited,
            mst_folders=mst_folders,
            curated_folders=curated_folders,
            cluster_to_folder=cluster_to_folder,
            max_folder_depth=max_folder_depth,
            min_folder_children=min_folder_children,
            max_folder_children=max_folder_children,
            folder_count=folder_count,
            folder_method=folder_method,
            primary_account=primary_account,
            use_llm_naming=use_llm_naming,
            llm_model=llm_model,
            use_titled_files=use_titled_files,
            url_to_title=url_to_title,
            url_to_filename=url_to_filename,
            parent_links=parent_links,
            parent_url=cluster_url,  # Current cluster becomes parent for children
            url_to_folder=url_to_folder,
            children_index=children_index,
            **kwargs
        )

    return total_generated

def main():
    parser = argparse.ArgumentParser(
        description="Generate mind maps from Pearltrees clusters"
    )
    parser.add_argument('--cluster', type=str, help='Cluster name/title to search for')
    parser.add_argument('--cluster-url', type=str, help='Cluster URL (Pearltrees URI)')
    parser.add_argument('--data', type=Path,
                        default=Path('reports/pearltrees_targets_full_multi_account.jsonl'),
                        help='Path to training data JSONL')
    parser.add_argument('--embeddings', type=Path,
                        default=Path('models/dual_embeddings_full.npz'),
                        help='Path to embeddings file (.npz)')
    parser.add_argument('--index', type=Path, default=None,
                        help='Path to mindmap index file (JSON/TSV/SQLite) for relative links')
    parser.add_argument('--children-index', type=Path, default=None,
                        help='Path to SQLite children index for loading missing children from RDF')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output file path (required unless --recursive)')
    parser.add_argument('--format', choices=['smmx', 'mm', 'opml', 'graphml', 'vue'], default='smmx',
                        help='Output format: smmx (default), mm (FreeMind), opml, graphml, vue')
    parser.add_argument('--xml-only', action='store_true',
                        help='Output raw XML instead of packaged format')
    parser.add_argument('--max-children', type=int, default=8,
                        help='Max children per node before micro-clustering')
    parser.add_argument('--min-children', type=int, default=4,
                        help='Min clusters when micro-clustering')
    parser.add_argument('--optimize', action='store_true',
                        help='Apply force-directed optimization to reduce overlaps')
    parser.add_argument('--optimize-iterations', type=int, default=100,
                        help='Number of optimization iterations')
    parser.add_argument('--no-scaling', action='store_true',
                        help='Disable node size scaling by descendant count')
    parser.add_argument('--minimize-crossings', action='store_true',
                        help='Apply edge crossing minimization after force-directed')
    parser.add_argument('--crossing-passes', type=int, default=10,
                        help='Max passes for crossing minimization')
    parser.add_argument('--layout', choices=['radial-auto', 'radial', 'radial-freeform'], default='radial-auto',
                        help='Layout: "radial-auto" (native, default), "radial" (equal angles per parent), "radial-freeform" (force-directed)')
    parser.add_argument('--tree-style',
                        choices=['half-round', 'ellipse', 'rectangle', 'diamond'],
                        default=None,
                        help='Node style for Tree items (folders)')
    parser.add_argument('--pearl-style',
                        choices=['half-round', 'ellipse', 'rectangle', 'diamond'],
                        default=None,
                        help='Node style for Pearl items (pages/links)')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively generate linked maps for child Trees')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for recursive generation (required with --recursive)')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Maximum depth for recursive generation (default: unlimited)')
    parser.add_argument('--parent-links', action='store_true',
                        help='Add "back to parent" nodes in child maps')
    parser.add_argument('--mst-folders', action='store_true',
                        help='Organize output into subfolders based on MST of cluster centroids')
    parser.add_argument('--curated-folders', action='store_true',
                        help='Organize into folders preserving user hierarchy with K-clustered groups')
    parser.add_argument('--folder-count', type=int, default=100,
                        help='Target number of folder groups for --curated-folders (default: 100)')
    parser.add_argument('--folder-method', choices=['kmeans', 'mst-cut'], default='kmeans',
                        help='Clustering method for --curated-folders (default: kmeans)')
    parser.add_argument('--primary-account', type=str, default=None,
                        help='Primary account for --curated-folders root selection')
    parser.add_argument('--llm-folder-names', action='store_true',
                        help='Use LLM to generate folder names (requires gemini CLI)')
    parser.add_argument('--llm-model', type=str, default='gemini-2.0-flash',
                        help='LLM model for folder naming (default: gemini-2.0-flash)')
    parser.add_argument('--titled-files', action='store_true',
                        help='Use tree titles in filenames (e.g., Hacktivism_id2492215.smmx)')
    parser.add_argument('--max-folder-depth', type=int, default=None,
                        help='Maximum subfolder nesting depth (default: unlimited)')
    parser.add_argument('--min-folder-children', type=int, default=None,
                        help='Minimum children to create subfolders (default: no minimum)')
    parser.add_argument('--max-folder-children', type=int, default=None,
                        help='Maximum children to put in subfolders (default: unlimited)')
    parser.add_argument('--url-nodes', choices=['auto', 'expanded', 'direct', 'url', 'map', 'url-label'], nargs='?', const='auto', default='auto',
                        help='Link mode for nodes. "auto" (default): use expanded if <threshold PagePearls, else direct. "expanded": PagePearls link to Pearltrees with child node showing domain linking to external URL. "direct": PagePearls link directly to external URL. "url": URL on main, cloudmapref on child. "map": cloudmapref on main, URL on child.')
    parser.add_argument('--expanded-threshold', type=int, default=50,
                        help='PagePearl count threshold for auto mode (default: 50). Below this, use expanded mode.')
    parser.add_argument('--child-text', type=str, default='',
                        help='Text label for child link nodes (default: empty/unlabeled)')
    parser.add_argument('--save-folder-map', type=Path, default=None,
                        help='Save computed folder structure to JSON file for reuse')
    parser.add_argument('--load-folder-map', type=Path, default=None,
                        help='Load pre-computed folder structure from JSON file (skips computation)')

    args = parser.parse_args()

    if not args.cluster and not args.cluster_url:
        parser.error("Either --cluster or --cluster-url required")

    if args.recursive and not args.output_dir:
        parser.error("--output-dir required when using --recursive")

    if not args.recursive and not args.output:
        parser.error("--output required (or use --recursive with --output-dir)")

    # Format conversion support
    # For non-smmx formats, we generate smmx first then convert using export_mindmap.py
    convert_format = None
    export_script = Path(__file__).parent / 'export_mindmap.py'
    if args.format != 'smmx':
        # Supported formats in export_mindmap.py: mm, opml, graphml, vue
        supported_formats = {'mm', 'opml', 'graphml', 'vue'}
        if args.format not in supported_formats:
            print(f"Error: Format '{args.format}' is not supported.")
            print(f"Supported formats: smmx (native), {', '.join(sorted(supported_formats))}")
            return 1
        convert_format = args.format
        if not export_script.exists():
            print(f"Error: Export script not found: {export_script}")
            return 1

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

    # Load mindmap index for relative links
    index_store = None
    if args.index:
        if not HAS_INDEX_STORE:
            print("Error: index_store module not available. Install from scripts/mindmap/")
            return 1
        if args.index.exists():
            index_store = create_index_store(str(args.index))
            print(f"Loaded index with {index_store.count()} entries")
        else:
            print(f"Warning: Index file not found: {args.index}")

    # Resolve cluster URL if searching by name
    cluster_url = args.cluster_url
    if args.cluster and not cluster_url:
        # Find cluster by name search
        items = load_cluster_items(args.data, args.cluster, None)
        if items:
            # Find the folder itself
            folder = next((i for i in items
                          if args.cluster.lower() in i.get('raw_title', '').lower()
                          and i.get('type') == 'Tree'), None)
            if folder:
                cluster_url = folder.get('uri', '')
                print(f"Resolved cluster name '{args.cluster}' to: {cluster_url}")
        if not cluster_url:
            print(f"No cluster found matching: {args.cluster}")
            return 1

    # Recursive mode: generate entire hierarchy
    if args.recursive:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Recursive generation starting from: {cluster_url}")
        print(f"Output directory: {args.output_dir}")
        if args.max_depth is not None:
            print(f"Max depth: {args.max_depth}")
        if args.mst_folders:
            print(f"MST folder organization: enabled")
            if args.max_folder_depth is not None:
                print(f"Max folder depth: {args.max_folder_depth}")
        if args.curated_folders:
            print(f"Curated folder organization: enabled")
            print(f"  Folder count: {args.folder_count}")
            print(f"  Folder method: {args.folder_method}")
            if args.primary_account:
                print(f"  Primary account: {args.primary_account}")
            if args.max_folder_depth is not None:
                print(f"  Max folder depth: {args.max_folder_depth}")
            if args.llm_folder_names:
                print(f"  LLM folder names: enabled ({args.llm_model})")
            if args.titled_files:
                print(f"  Titled files: enabled")
        if args.parent_links:
            print(f"Parent links: enabled")

        total = generate_recursive(
            cluster_url=cluster_url,
            data_path=args.data,
            output_dir=args.output_dir,
            tree_id_emb=tree_id_emb,
            title_emb=title_emb,
            uri_emb=uri_emb,
            max_depth=args.max_depth,
            mst_folders=args.mst_folders,
            curated_folders=args.curated_folders,
            max_folder_depth=args.max_folder_depth,
            min_folder_children=args.min_folder_children,
            max_folder_children=args.max_folder_children,
            folder_count=args.folder_count,
            folder_method=args.folder_method,
            primary_account=args.primary_account,
            use_llm_naming=args.llm_folder_names,
            llm_model=args.llm_model,
            use_titled_files=args.titled_files,
            parent_links=args.parent_links,
            save_folder_map=args.save_folder_map,
            load_folder_map=args.load_folder_map,
            children_index=args.children_index,
            min_children=args.min_children,
            max_children=args.max_children,
            optimize=args.optimize,
            optimize_iterations=args.optimize_iterations,
            do_minimize_crossings=args.minimize_crossings,
            crossing_passes=args.crossing_passes,
            no_scaling=args.no_scaling,
            tree_style=args.tree_style,
            pearl_style=args.pearl_style,
            url_nodes_mode=args.url_nodes,
            child_node_text=args.child_text,
            xml_only=args.xml_only,
            layout=args.layout
        )

        print(f"\nGenerated {total} linked mind maps")

        # Convert to target format if needed
        if convert_format:
            print(f"\nConverting to {convert_format} format...")
            import subprocess
            smmx_files = list(args.output_dir.rglob('*.smmx'))
            for smmx_file in smmx_files:
                target_file = smmx_file.with_suffix(f'.{convert_format}')
                subprocess.run(['python3', str(export_script), str(smmx_file), str(target_file)], check=True)
            print(f"Converted {len(smmx_files)} files to {convert_format}")

        return 0

    # Single map mode (original behavior)
    # Load cluster items
    items = load_cluster_items(args.data, args.cluster, cluster_url,
                               children_index=args.children_index)

    if not items:
        print(f"No items found for cluster")
        return 1

    print(f"Found {len(items)} items in cluster")

    # Create nodes with embeddings
    nodes = create_nodes_from_items(items, tree_id_emb, title_emb, uri_emb)

    # Build hierarchy with micro-clustering
    root = build_hierarchy(nodes, min_children=args.min_children,
                          max_children=args.max_children)

    # Apply layout (skip for 'radial-auto' - native software handles it)
    if args.layout == 'radial':
        apply_radial_tree_layout(root, center_x=500, center_y=500, min_node_spacing=60, base_radius=120)
    elif args.layout == 'radial-freeform':
        apply_radial_layout(root, center_x=500, center_y=500, min_spacing=100, base_radius=180)
    # else: 'radial-auto' - use default positions, native software handles layout

    # Optional: force-directed optimization
    if args.optimize:
        print("Applying force-directed optimization...")
        force_directed_optimize(root, center_x=500, center_y=500,
                               iterations=args.optimize_iterations)

    # Optional: edge crossing minimization (requires force-directed first)
    if args.minimize_crossings:
        if not args.optimize:
            print("Running force-directed optimization first (required for crossing minimization)...")
            force_directed_optimize(root, center_x=500, center_y=500, iterations=300)
        print("Minimizing edge crossings...")
        minimize_crossings(root, center_x=500, center_y=500,
                          max_passes=args.crossing_passes)

    # Calculate node scales based on descendant count
    scales = None
    if not args.no_scaling:
        scales = calculate_node_scales(root)

    # Generate XML
    title = root.title if root else "Mind Map"
    # Enable cloudmapref if index is provided
    use_cloudmapref = index_store is not None

    # Resolve url_nodes mode
    url_nodes = args.url_nodes

    # Auto mode: choose expanded or direct based on PagePearl count
    if url_nodes == 'auto':
        pagepearl_count = sum(1 for n in nodes if n.item_type == 'PagePearl')
        if pagepearl_count < args.expanded_threshold:
            url_nodes = 'expanded'
            print(f"Auto mode: using 'expanded' ({pagepearl_count} PagePearls < {args.expanded_threshold} threshold)")
        else:
            url_nodes = 'direct'
            print(f"Auto mode: using 'direct' ({pagepearl_count} PagePearls >= {args.expanded_threshold} threshold)")

    # When index is provided with url/map modes, enable cloudmapref child nodes
    if index_store is not None and url_nodes in (None, 'url', 'map'):
        if url_nodes is None:
            url_nodes = 'url'
    xml_content = generate_mindmap_xml(root, title, scales,
                                       tree_style=args.tree_style,
                                       pearl_style=args.pearl_style,
                                       enable_cloudmapref=use_cloudmapref,
                                       cluster_id=cluster_url,
                                       url_nodes_mode=url_nodes,
                                       layout=args.layout,
                                       index_store=index_store,
                                       output_path=args.output)

    # Write output
    if args.xml_only:
        output_path = args.output.with_suffix('.xml')
        output_path.write_text(xml_content)
        print(f"Written: {output_path}")
    else:
        smmx_path = args.output if args.output.suffix == '.smmx' else args.output.with_suffix('.smmx')
        write_smmx(xml_content, smmx_path)

        # Convert to target format if needed
        if convert_format:
            print(f"Converting to {convert_format} format...")
            import subprocess
            target_file = smmx_path.with_suffix(f'.{convert_format}')
            subprocess.run(['python3', str(export_script), str(smmx_path), str(target_file)], check=True)
            print(f"Written: {target_file}")

    return 0

if __name__ == '__main__':
    exit(main())
