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
import base64
import hashlib
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
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

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
    target_tree_id: str,
    target_folder: Path
) -> str:
    """Compute relative path from source to target .smmx file.

    Args:
        source_folder: Folder containing source .smmx (relative to output root)
        target_tree_id: Tree ID of target (e.g., 'id123456')
        target_folder: Folder containing target .smmx (relative to output root)

    Returns:
        Relative path string for cloudmapref (e.g., '../sibling/id789.smmx')
    """
    import os
    # Both paths are relative to output root
    # source_folder: where the current .smmx is
    # target_folder: where the target .smmx is
    target_file = target_folder / f'{target_tree_id}.smmx'

    # Compute relative path from source folder to target file
    rel_path = os.path.relpath(target_file, source_folder)
    # Normalize to forward slashes for SimpleMind compatibility
    rel_path = rel_path.replace('\\', '/')

    # Add ./ prefix for paths that don't start with ../ (SimpleMind convention)
    if not rel_path.startswith('../') and not rel_path.startswith('./'):
        rel_path = './' + rel_path

    return rel_path


def create_parent_link_node(
    parent_element: Element,
    root_node: 'MindMapNode',
    parent_tree_id: str,
    parent_cloudmapref: str,
    node_id: int
) -> None:
    """Create a parent link node connected to the root.

    Adds a small square node labeled "↑" that links back to the parent map.

    Args:
        parent_element: XML element to add the node to
        root_node: The root MindMapNode (for positioning)
        parent_tree_id: Tree ID of the parent
        parent_cloudmapref: Relative path to parent .smmx file
        node_id: Unique ID for this node
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
    topic.set('text', '↑')  # Up arrow to indicate parent

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
                cluster_to_folder: Dict[str, Path] = None):
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

    # Determine borderstyle based on item type
    borderstyle = None
    if node.item_type == 'Tree' and tree_style:
        borderstyle = BORDERSTYLES.get(tree_style)
    elif 'Pearl' in node.item_type and pearl_style:
        # Matches PagePearl, and any other Pearl types
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

        # Compute cloudmapref path (uses folder structure if available)
        def get_cloudmapref_path():
            if cluster_to_folder and source_folder is not None and node.url in cluster_to_folder:
                target_folder = cluster_to_folder[node.url]
                return compute_relative_cloudmapref_path(source_folder, target_tree_id, target_folder)
            else:
                return f'./{target_tree_id}.smmx'

        if is_tree_with_children and target_tree_id:
            # Determine link layout based on url_nodes_mode
            # 'url': URL on main, cloudmapref on child (user's preferred hand-built style)
            # 'map': cloudmapref on main, URL on child
            # None: cloudmapref on main, no child node

            if url_nodes_mode == 'url':
                # URL on main node
                link = SubElement(topic, 'link')
                link.set('urllink', node.url)

                # cloudmapref on child node (square, unlabeled)
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
                    # cloudmapref link
                    child_link = SubElement(child_topic, 'link')
                    child_link.set('cloudmapref', get_cloudmapref_path())
                    target_guid = generate_deterministic_guid(node.url, 0)
                    child_link.set('element', target_guid)

            elif url_nodes_mode == 'map':
                # cloudmapref on main node
                link = SubElement(topic, 'link')
                link.set('cloudmapref', get_cloudmapref_path())
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
                # No child node - cloudmapref on main only
                link = SubElement(topic, 'link')
                link.set('cloudmapref', get_cloudmapref_path())
                target_guid = generate_deterministic_guid(node.url, 0)
                link.set('element', target_guid)
        else:
            # Non-Tree nodes or root: use urllink
            link = SubElement(topic, 'link')
            link.set('urllink', node.url)

    # Recurse for children
    for child in node.children:
        node_to_xml(child, parent_element, scales, tree_style, pearl_style,
                    enable_cloudmapref, cluster_id, url_nodes_mode, next_id, child_node_text,
                    source_folder, cluster_to_folder)

def generate_mindmap_xml(root: MindMapNode, title: str, scales: Dict[int, float] = None,
                         tree_style: str = None, pearl_style: str = None,
                         enable_cloudmapref: bool = False, cluster_id: str = None,
                         url_nodes_mode: str = None, child_node_text: str = '',
                         source_folder: Path = None,
                         cluster_to_folder: Dict[str, Path] = None,
                         parent_tree_id: str = None,
                         parent_cloudmapref: str = None) -> str:
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
                source_folder, cluster_to_folder)

    # Add parent link node if parent info provided
    if parent_tree_id and parent_cloudmapref:
        parent_node_id = next_id[0]
        next_id[0] += 1
        create_parent_link_node(topics, root, parent_tree_id, parent_cloudmapref, parent_node_id)

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

    def get_url(item):
        """Get URL for an item - uri for Trees, pearl_uri for PagePearls."""
        return item.get('uri') or item.get('pearl_uri', '')

    # Create root node
    if root_item:
        tree_id = root_item.get('tree_id', '')
        root_node = MindMapNode(
            id=0,
            title=root_item.get('raw_title', 'Root'),
            tree_id=tree_id,
            url=get_url(root_item),
            parent_id=-1,
            palette=1,
            embedding=get_embedding(root_item),
            item_type=root_item.get('type', '')
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
            url=get_url(item),
            parent_id=0,  # Will be updated by build_hierarchy
            palette=(palette_idx % 8) + 1,
            embedding=get_embedding(item),
            item_type=item.get('type', '')
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
                        parent_cloudmapref: str = None) -> List[str]:
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

    Returns:
        List of child Tree URLs for recursive generation
    """
    # Load cluster items
    items = load_cluster_items(data_path, cluster_url=cluster_url)

    if not items:
        print(f"No items found for cluster: {cluster_url}")
        return []

    print(f"Generating map for: {cluster_url} ({len(items)} items)")

    # Create nodes with embeddings
    nodes = create_nodes_from_items(items, tree_id_emb, title_emb, uri_emb)

    # Build hierarchy with micro-clustering
    root = build_hierarchy(nodes, min_children=min_children, max_children=max_children)

    # Apply radial layout
    apply_radial_layout(root, center_x=500, center_y=500, min_spacing=80, base_radius=150)

    # Optional: force-directed optimization
    if optimize:
        force_directed_optimize(root, center_x=500, center_y=500,
                               iterations=optimize_iterations)

    # Optional: edge crossing minimization
    if do_minimize_crossings:
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
                                       parent_cloudmapref=parent_cloudmapref)

    # Write output
    if xml_only:
        out = output_path.with_suffix('.xml')
        out.write_text(xml_content)
        print(f"Written: {out}")
    else:
        write_smmx(xml_content, output_path)

    # Collect child Tree URLs for recursive generation
    child_tree_urls = []
    all_nodes = collect_all_nodes(root)
    for node in all_nodes:
        if node.item_type == 'Tree' and node.url and node.id != 0:
            child_tree_urls.append(node.url)

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

        # Find child Trees
        items = load_cluster_items(data_path, cluster_url=url)
        for item in items:
            if item.get('type') == 'Tree':
                child_url = item.get('uri', '')
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

    for cluster_url in cluster_urls:
        items = load_cluster_items(data_path, cluster_url=cluster_url)
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


def generate_recursive(cluster_url: str, data_path: Path, output_dir: Path,
                       tree_id_emb: Dict[str, np.ndarray],
                       title_emb: Dict[str, np.ndarray],
                       uri_emb: Dict[str, np.ndarray],
                       max_depth: int = None, current_depth: int = 0,
                       visited: set = None,
                       mst_folders: bool = False,
                       cluster_to_folder: Dict[str, Path] = None,
                       max_folder_depth: int = None,
                       min_folder_children: int = None,
                       max_folder_children: int = None,
                       parent_links: bool = False,
                       parent_url: str = None,
                       url_to_folder: Dict[str, Path] = None,
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
        cluster_to_folder: Pre-computed mapping of cluster URLs to folder paths
        max_folder_depth: Maximum subfolder nesting depth
        min_folder_children: Minimum children to create subfolders
        max_folder_children: Maximum children to put in subfolders
        parent_links: If True, add "back to parent" nodes in child maps
        parent_url: URL of the parent cluster (for parent links)
        url_to_folder: Mapping of cluster URLs to folder paths (for parent links)
        **kwargs: Additional arguments passed to generate_single_map

    Returns:
        Total number of maps generated
    """
    if visited is None:
        visited = set()

        # Initialize url_to_folder for parent_links mode (tracks where each map ends up)
        if parent_links and url_to_folder is None:
            url_to_folder = {}

        # On first call with mst_folders, build MST structure
        if mst_folders and cluster_to_folder is None:
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

    if mst_folders and cluster_to_folder and cluster_url in cluster_to_folder:
        folder = cluster_to_folder[cluster_url]
        output_path = output_dir / folder / f"{tree_id}.smmx"
        source_folder = folder
    else:
        output_path = output_dir / f"{tree_id}.smmx"
        source_folder = Path('.')

    # Track this cluster's folder for parent_links
    if parent_links and url_to_folder is not None:
        url_to_folder[cluster_url] = source_folder

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute parent link info if parent_links enabled and we have a parent
    parent_tree_id = None
    parent_cloudmapref = None
    if parent_links and parent_url:
        parent_tree_id = extract_tree_id(parent_url)
        if parent_tree_id:
            # Get parent's folder (should have been recorded)
            parent_folder = url_to_folder.get(parent_url, Path('.')) if url_to_folder else Path('.')
            parent_cloudmapref = compute_relative_cloudmapref_path(
                source_folder, parent_tree_id, parent_folder
            )

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
            cluster_to_folder=cluster_to_folder,
            max_folder_depth=max_folder_depth,
            min_folder_children=min_folder_children,
            max_folder_children=max_folder_children,
            parent_links=parent_links,
            parent_url=cluster_url,  # Current cluster becomes parent for children
            url_to_folder=url_to_folder,
            **kwargs
        )

    return total_generated

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
    parser.add_argument('--output', type=Path, default=None,
                        help='Output .smmx file path (required unless --recursive)')
    parser.add_argument('--xml-only', action='store_true',
                        help='Output raw XML instead of .smmx')
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
    parser.add_argument('--max-folder-depth', type=int, default=None,
                        help='Maximum subfolder nesting depth (default: unlimited)')
    parser.add_argument('--min-folder-children', type=int, default=None,
                        help='Minimum children to create subfolders (default: no minimum)')
    parser.add_argument('--max-folder-children', type=int, default=None,
                        help='Maximum children to put in subfolders (default: unlimited)')
    parser.add_argument('--url-nodes', choices=['url', 'map'], nargs='?', const='url', default=None,
                        help='Attach small child nodes to Tree nodes. "url" (default): URL on main, cloudmapref on child. "map": cloudmapref on main, URL on child.')
    parser.add_argument('--child-text', type=str, default='',
                        help='Text label for child link nodes (default: empty/unlabeled)')

    args = parser.parse_args()

    if not args.cluster and not args.cluster_url:
        parser.error("Either --cluster or --cluster-url required")

    if args.recursive and not args.output_dir:
        parser.error("--output-dir required when using --recursive")

    if not args.recursive and not args.output:
        parser.error("--output required (or use --recursive with --output-dir)")

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
            max_folder_depth=args.max_folder_depth,
            min_folder_children=args.min_folder_children,
            max_folder_children=args.max_folder_children,
            parent_links=args.parent_links,
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
            xml_only=args.xml_only
        )

        print(f"\nGenerated {total} linked mind maps")
        return 0

    # Single map mode (original behavior)
    # Load cluster items
    items = load_cluster_items(args.data, args.cluster, cluster_url)

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
    xml_content = generate_mindmap_xml(root, title, scales,
                                       tree_style=args.tree_style,
                                       pearl_style=args.pearl_style,
                                       enable_cloudmapref=False,
                                       cluster_id=cluster_url)

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
