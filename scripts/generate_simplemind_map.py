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

    # Check if segments share an endpoint (not a real crossing)
    endpoints = {p1, p2}
    if p3 in endpoints or p4 in endpoints:
        return False

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def collect_edges(node: MindMapNode, edges: List[Tuple[MindMapNode, MindMapNode]] = None) -> List[Tuple[MindMapNode, MindMapNode]]:
    """Collect all parent-child edges in the tree."""
    if edges is None:
        edges = []
    for child in node.children:
        edges.append((node, child))
        collect_edges(child, edges)
    return edges


def count_edge_crossings(root: MindMapNode) -> int:
    """Count total number of edge crossings in the layout."""
    edges = collect_edges(root)
    crossings = 0

    for i, (a1, a2) in enumerate(edges):
        p1 = (a1.x, a1.y)
        p2 = (a2.x, a2.y)
        for b1, b2 in edges[i+1:]:
            p3 = (b1.x, b1.y)
            p4 = (b2.x, b2.y)
            if segments_intersect(p1, p2, p3, p4):
                crossings += 1

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

            # Try angular adjustments: -30, -15, 0, +15, +30 degrees
            best_crossings = count_edge_crossings(root)
            best_x, best_y = orig_x, orig_y

            for angle_offset in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:  # radians
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

        # Only run force-directed if significant overlaps exist
        if overlap_count > 2:
            # Lighter force-directed to fix overlaps without destroying crossing gains
            force_directed_optimize(root, center_x, center_y,
                                   iterations=force_iterations // 2, repulsion=50000,
                                   attraction=0.0005, min_distance=100)
        elif overlap_count > 0 and verbose:
            print(f"  (skipping force-directed, only {overlap_count} minor overlaps)")

        current_crossings = count_edge_crossings(root)
        if verbose:
            print(f"Pass {pass_num + 1}: {current_crossings} crossings (was {pass_start_crossings})")

        if not improved or current_crossings >= pass_start_crossings:
            break

    final_crossings = count_edge_crossings(root)
    if verbose:
        print(f"Final edge crossings: {final_crossings} (reduced from {initial_crossings})")


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

def node_to_xml(node: MindMapNode, parent_element: Element, scales: Dict[int, float] = None):
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

    # Add font scaling based on descendant count
    if scales and node.id in scales:
        scale = scales[node.id]
        if scale > 1.2:  # Only add styling for non-leaf nodes
            style = SubElement(topic, 'style')
            font = SubElement(style, 'font')
            if scale > 2.0:
                font.set('bold', 'True')
            font.set('scale', f'{scale:.2f}')

    # Add link if URL present
    if node.url:
        link = SubElement(topic, 'link')
        link.set('urllink', node.url)

    # Recurse for children
    for child in node.children:
        node_to_xml(child, parent_element, scales)

def generate_mindmap_xml(root: MindMapNode, title: str, scales: Dict[int, float] = None) -> str:
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
    scroll.set('zoom', '40')  # Zoom out more for larger maps
    scroll.set('x', str(int(-root.x)))
    scroll.set('y', str(int(-root.y)))
    main_theme = SubElement(meta, 'main-centraltheme')
    main_theme.set('id', '0')

    # Topics section
    topics = SubElement(mindmap, 'topics')
    node_to_xml(root, topics, scales)

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
            url=get_url(item),
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
    xml_content = generate_mindmap_xml(root, title, scales)

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
