#!/usr/bin/env python3
"""
MST Circle-Based Folder Grouping

Implements the MST circle partitioning algorithm for organizing mindmaps
into semantically coherent folder hierarchies.

Usage:
    # Test on physics subset
    python3 mst_folder_grouping.py --subset physics --max-depth 3 --target-size 8

    # Run on all trees
    python3 mst_folder_grouping.py --trees-only --max-depth 4 --target-size 10

    # Dry run with visualization
    python3 mst_folder_grouping.py --subset physics --visualize --dry-run
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cosine


@dataclass
class Circle:
    """A circle (folder) in the MST partition."""
    nodes: Set[int] = field(default_factory=set)
    boundary_nodes: Set[int] = field(default_factory=set)
    children: List['Circle'] = field(default_factory=list)
    parent: Optional['Circle'] = None
    name: str = ""

    def add_node(self, node: int):
        self.nodes.add(node)

    def add_boundary(self, node: int):
        self.boundary_nodes.add(node)

    @property
    def size(self) -> int:
        return len(self.nodes)


class MSTFolderGrouper:
    """MST-based folder grouping algorithm."""

    def __init__(
        self,
        embeddings: np.ndarray,
        titles: List[str],
        tree_ids: List[str],
        target_size: int = 8,
        max_depth: int = 4,
        min_size: int = 2,
        verbose: bool = False,
        subdivision_method: str = 'multilevel',
        internal_cost_mode: str = 'none',
        size_cost_mode: str = 'gm_maximize',
        tree_source: str = 'mst',
        hierarchy_paths: Optional[Dict[str, str]] = None,
        output_embeddings: Optional[np.ndarray] = None,
        embed_blend: float = 0.3
    ):
        """
        Args:
            embeddings: (N, D) array of embedding vectors (input embeddings)
            titles: List of item titles
            tree_ids: List of tree IDs
            target_size: Target number of items per folder (used as initial estimate for gm_maximize)
            max_depth: Maximum folder hierarchy depth
            min_size: Minimum items per folder
            verbose: Print progress
            subdivision_method: 'multilevel' (default) or 'bisection'
            internal_cost_mode: 'none' (default), 'arithmetic', or 'geometric'
            size_cost_mode: 'gm_maximize' (default, scale-invariant) or 'quadratic'
            tree_source: 'mst', 'curated', or 'hybrid'
            hierarchy_paths: Dict[tree_id] -> path string (e.g., '/id1/id2/.../idn')
                            Required when tree_source='curated' or 'hybrid'
            output_embeddings: (N, D) array of output embeddings for blending (hybrid mode)
            embed_blend: Weight for input embeddings in blend (default 0.3 = 30% input, 70% output)
        """
        self.embeddings = embeddings
        self.titles = titles
        self.tree_ids = tree_ids
        self.n_items = len(embeddings)
        self.target_size = target_size
        self.max_depth = max_depth
        self.min_size = min_size
        self.verbose = verbose
        self.subdivision_method = subdivision_method
        self.internal_cost_mode = internal_cost_mode  # 'none', 'arithmetic', or 'geometric'
        self.size_cost_mode = size_cost_mode  # 'quadratic' or 'geometric'
        self.tree_source = tree_source  # 'mst', 'curated', or 'hybrid'
        self.hierarchy_paths = hierarchy_paths or {}
        self.output_embeddings = output_embeddings
        self.embed_blend = embed_blend  # Weight for input embeddings (1 - this = output weight)

        # Track circles where subdivision was attempted but failed
        self._subdivision_failed: Set[int] = set()

        # Depth penalty parameters (soft constraint)
        # penalty(d) = depth_penalty_base * (exp(depth_penalty_k * excess) - 1)
        # Calibrated so penalty(max_depth+1) ≈ total size imbalance cost
        self.depth_penalty_k = 2.0  # Exponential growth rate
        self.depth_penalty_base = None  # Calibrated after seeing data

        # Computed structures
        self.mst = None
        self.mst_adjacency = None
        self.circles: List[Circle] = []

    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute pairwise cosine distances."""
        if self.verbose:
            print(f"Computing {self.n_items}x{self.n_items} distance matrix...")

        # Normalize embeddings for cosine distance
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized = self.embeddings / (norms + 1e-8)

        # Cosine distance = 1 - cosine_similarity
        similarity = normalized @ normalized.T
        distances = 1 - similarity

        # Ensure non-negative (numerical precision)
        distances = np.maximum(distances, 0)
        np.fill_diagonal(distances, 0)

        return distances

    def _compute_sparse_knn_graph(self, k: int = 30) -> 'csr_matrix':
        """
        Compute sparse k-NN graph for memory-efficient MST.

        Instead of O(N^2) memory, uses O(N*k) memory.
        """
        from scipy.sparse import coo_matrix

        if self.verbose:
            print(f"Computing sparse {k}-NN graph for {self.n_items} items...")

        # Normalize embeddings (use float32 to save memory)
        embeddings_f32 = self.embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True)
        normalized = embeddings_f32 / (norms + 1e-8)
        del embeddings_f32  # Free memory

        # Collect edges as lists (more memory efficient than lil_matrix)
        rows = []
        cols = []
        data = []

        # Process in smaller batches to manage memory
        batch_size = min(500, self.n_items)
        for i in range(0, self.n_items, batch_size):
            end_i = min(i + batch_size, self.n_items)
            batch = normalized[i:end_i]

            # Compute similarities to all other nodes
            similarities = batch @ normalized.T  # (batch, N)
            distances = (1 - similarities).astype(np.float32)
            del similarities  # Free memory

            # For each node in batch, keep k nearest neighbors
            for j in range(distances.shape[0]):
                node_idx = i + j
                row = distances[j]
                row[node_idx] = float('inf')  # Exclude self
                knn_indices = np.argpartition(row, k)[:k]

                for neighbor in knn_indices:
                    dist = float(row[neighbor])
                    rows.append(node_idx)
                    cols.append(neighbor)
                    data.append(dist)

            del distances  # Free batch memory

            if self.verbose and i % 2000 == 0:
                print(f"  Processed {end_i}/{self.n_items} nodes...")

        # Make symmetric by adding reverse edges
        sym_rows = rows + cols
        sym_cols = cols + rows
        sym_data = data + data

        # Build sparse matrix from COO format
        sparse_dist = coo_matrix((sym_data, (sym_rows, sym_cols)), shape=(self.n_items, self.n_items))

        if self.verbose:
            nnz = sparse_dist.nnz
            print(f"  Sparse graph has {nnz} edges ({nnz/self.n_items:.1f} per node)")

        return sparse_dist.tocsr()

    def _build_mst(self, distances):
        """Build minimum spanning tree from distance matrix."""
        from scipy.sparse import issparse
        from scipy.sparse.csgraph import connected_components

        if self.verbose:
            print("Building MST...")

        # Check connectivity for sparse graphs
        if issparse(distances):
            n_components, labels = connected_components(distances, directed=False)
            if self.verbose:
                print(f"  Graph has {n_components} connected components")
            if n_components > 1:
                # Count items per component
                from collections import Counter
                comp_sizes = Counter(labels)
                largest_comp = max(comp_sizes.values())
                if self.verbose:
                    print(f"  Largest component: {largest_comp} items ({100*largest_comp/self.n_items:.1f}%)")

        # scipy expects dense matrix, returns sparse CSR
        self.mst = minimum_spanning_tree(distances)

        # Convert to adjacency list for traversal
        self.mst_adjacency = defaultdict(list)
        cx = self.mst.tocoo()
        for i, j, w in zip(cx.row, cx.col, cx.data):
            self.mst_adjacency[i].append((j, w))
            self.mst_adjacency[j].append((i, w))

        if self.verbose:
            total_weight = self.mst.sum()
            mst_edges = self.mst.nnz
            print(f"MST total weight: {total_weight:.4f}")
            print(f"MST has {mst_edges} edges (expected {self.n_items - 1} for connected graph)")
            if len(self.mst_adjacency) < self.n_items:
                print(f"  WARNING: Only {len(self.mst_adjacency)} nodes in MST adjacency")

    def _build_curated_tree(self):
        """Build tree adjacency from curated hierarchy paths.

        Uses self.hierarchy_paths (dict tree_id -> path string like '/id1/id2/.../idn')
        to construct parent-child relationships. Edge weights are computed from
        embedding distances for subdivision decisions.
        """
        if self.verbose:
            print("Building curated tree from hierarchy paths...")

        # Build tree_id -> index mapping
        tree_id_to_idx = {tid: i for i, tid in enumerate(self.tree_ids)}

        # Parse paths to extract parent-child relationships
        # Path format: /ancestor1/ancestor2/.../parent/self
        parent_child_pairs = []  # [(parent_idx, child_idx), ...]

        for tree_id, path in self.hierarchy_paths.items():
            if tree_id not in tree_id_to_idx:
                continue  # Skip trees not in our subset

            child_idx = tree_id_to_idx[tree_id]

            # Parse path to get parent
            parts = [p for p in path.strip('/').split('/') if p]
            if len(parts) >= 2:
                parent_id = parts[-2]  # Second to last is parent
                if parent_id in tree_id_to_idx:
                    parent_idx = tree_id_to_idx[parent_id]
                    parent_child_pairs.append((parent_idx, child_idx))

        if self.verbose:
            print(f"  Found {len(parent_child_pairs)} parent-child pairs")

        # Build adjacency list with embedding-based edge weights
        self.mst_adjacency = defaultdict(list)

        # Normalize embeddings for cosine distance
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized = self.embeddings / (norms + 1e-8)

        for parent_idx, child_idx in parent_child_pairs:
            # Compute edge weight from embedding cosine distance
            similarity = np.dot(normalized[parent_idx], normalized[child_idx])
            weight = max(0, 1 - similarity)  # Cosine distance

            # Add bidirectional edges
            self.mst_adjacency[parent_idx].append((child_idx, weight))
            self.mst_adjacency[child_idx].append((parent_idx, weight))

        # Find orphan nodes (no parent in our subset) and connect them
        connected_nodes = set()
        for node, neighbors in self.mst_adjacency.items():
            connected_nodes.add(node)
            for neighbor, _ in neighbors:
                connected_nodes.add(neighbor)

        orphans = set(range(self.n_items)) - connected_nodes

        if orphans and self.verbose:
            print(f"  Found {len(orphans)} orphan nodes (no parent in subset)")

        # For orphans, find their nearest connected neighbor by embedding distance
        if orphans and connected_nodes:
            connected_list = list(connected_nodes)
            connected_embeddings = normalized[connected_list]

            for orphan in orphans:
                # Find nearest connected node
                similarities = np.dot(connected_embeddings, normalized[orphan])
                best_idx = np.argmax(similarities)
                best_neighbor = connected_list[best_idx]
                weight = max(0, 1 - similarities[best_idx])

                self.mst_adjacency[orphan].append((best_neighbor, weight))
                self.mst_adjacency[best_neighbor].append((orphan, weight))

        if self.verbose:
            n_edges = sum(len(v) for v in self.mst_adjacency.values()) // 2
            print(f"  Curated tree has {n_edges} edges")
            print(f"  {len(self.mst_adjacency)} nodes in adjacency")

        # Set mst to None (not computed for curated tree)
        self.mst = None

    def _build_hybrid_tree(self):
        """Build hybrid tree: curated structure + greedy orphan attachment.

        Uses blended embeddings (embed_blend * input + (1-embed_blend) * output)
        for computing distances. Curated hierarchy is fixed; orphan nodes are
        attached greedily to minimize semantic distance, with permutation to
        optimize attachment order.
        """
        if self.verbose:
            print("Building hybrid tree...")
            print(f"  Embedding blend: {self.embed_blend:.0%} input, {1-self.embed_blend:.0%} output")

        # Compute blended embeddings
        if self.output_embeddings is None:
            raise ValueError("output_embeddings required for hybrid mode")

        blended = (self.embed_blend * self.embeddings +
                   (1 - self.embed_blend) * self.output_embeddings)

        # Normalize blended embeddings for cosine distance
        norms = np.linalg.norm(blended, axis=1, keepdims=True)
        normalized = blended / (norms + 1e-8)

        # Build tree_id -> index mapping
        tree_id_to_idx = {tid: i for i, tid in enumerate(self.tree_ids)}

        # Parse paths to extract parent-child relationships (curated structure)
        parent_child_pairs = []
        curated_nodes = set()

        for tree_id, path in self.hierarchy_paths.items():
            if tree_id not in tree_id_to_idx:
                continue

            child_idx = tree_id_to_idx[tree_id]
            curated_nodes.add(child_idx)

            parts = [p for p in path.strip('/').split('/') if p]
            if len(parts) >= 2:
                parent_id = parts[-2]
                if parent_id in tree_id_to_idx:
                    parent_idx = tree_id_to_idx[parent_id]
                    parent_child_pairs.append((parent_idx, child_idx))
                    curated_nodes.add(parent_idx)

        if self.verbose:
            print(f"  Curated structure: {len(parent_child_pairs)} edges, {len(curated_nodes)} nodes")

        # Build initial adjacency from curated structure
        self.mst_adjacency = defaultdict(list)

        for parent_idx, child_idx in parent_child_pairs:
            similarity = np.dot(normalized[parent_idx], normalized[child_idx])
            weight = max(0, 1 - similarity)
            self.mst_adjacency[parent_idx].append((child_idx, weight))
            self.mst_adjacency[child_idx].append((parent_idx, weight))

        # Find orphan nodes (not in curated structure)
        orphans = set(range(self.n_items)) - curated_nodes

        if not orphans:
            if self.verbose:
                print("  No orphan nodes to attach")
            self.mst = None
            return

        if self.verbose:
            print(f"  Orphan nodes to attach: {len(orphans)}")

        # Greedy orphan attachment with permutation optimization
        # Sort orphans by their minimum distance to curated nodes (attach closest first)
        connected_nodes = list(curated_nodes)
        connected_embeddings = normalized[connected_nodes]

        orphan_list = list(orphans)
        orphan_embeddings = normalized[orphan_list]

        # Compute distances from each orphan to each connected node
        # Shape: (n_orphans, n_connected)
        similarities = orphan_embeddings @ connected_embeddings.T
        min_distances = 1 - similarities.max(axis=1)  # Min distance to any connected node

        # Sort orphans by minimum distance (closest first for better attachment)
        sorted_indices = np.argsort(min_distances)
        orphan_order = [orphan_list[i] for i in sorted_indices]

        if self.verbose:
            print(f"  Attaching orphans in optimized order (closest first)...")

        # Greedily attach orphans
        attached_count = 0
        for orphan in orphan_order:
            # Find best attachment point among all connected nodes
            connected_list = list(connected_nodes)
            if not connected_list:
                # First orphan with no curated nodes - shouldn't happen but handle it
                connected_nodes.add(orphan)
                continue

            connected_emb = normalized[connected_list]
            orphan_emb = normalized[orphan]

            similarities = np.dot(connected_emb, orphan_emb)
            best_idx = np.argmax(similarities)
            best_neighbor = connected_list[best_idx]
            weight = max(0, 1 - similarities[best_idx])

            # Attach orphan (non-binary: can attach to any node)
            self.mst_adjacency[orphan].append((best_neighbor, weight))
            self.mst_adjacency[best_neighbor].append((orphan, weight))

            # Orphan is now connected and can receive future attachments
            connected_nodes.add(orphan)
            attached_count += 1

        if self.verbose:
            n_edges = sum(len(v) for v in self.mst_adjacency.values()) // 2
            print(f"  Attached {attached_count} orphans")
            print(f"  Hybrid tree has {n_edges} edges")
            print(f"  {len(self.mst_adjacency)} nodes in adjacency")

        self.mst = None

    def _find_mst_root(self) -> int:
        """Find a good root for MST traversal (node with highest degree or most central)."""
        # Use node with highest degree as root
        degrees = [(len(self.mst_adjacency[i]), i) for i in range(self.n_items)]
        max_degree, root = max(degrees)
        return root

    def _compute_path_distance(self, node1: int, node2: int) -> float:
        """Compute distance between two nodes along MST path."""
        if node1 == node2:
            return 0.0

        # BFS to find path
        visited = {node1}
        queue = [(node1, 0.0)]

        while queue:
            current, dist = queue.pop(0)

            for neighbor, weight in self.mst_adjacency[current]:
                if neighbor == node2:
                    return dist + weight
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + weight))

        return float('inf')  # Shouldn't happen in connected MST

    def _compute_circle_cost(self, circle: Circle) -> float:
        """Compute cost of a circle (boundary + internal edge costs)."""
        total_cost = 0.0

        # Boundary cost: pairwise distances between boundary nodes
        if len(circle.boundary_nodes) > 1:
            boundary_list = list(circle.boundary_nodes)
            for i, n1 in enumerate(boundary_list):
                for n2 in boundary_list[i+1:]:
                    total_cost += self._compute_path_distance(n1, n2)

        # Internal cost: sum of MST edge weights inside the circle
        # This penalizes large/spread-out circles
        internal_cost = self._compute_internal_cost(circle)
        total_cost += internal_cost

        return total_cost

    def _compute_internal_cost(self, circle: Circle, use_geometric: bool = True) -> float:
        """Compute internal cost from MST edge weights inside the circle.

        Args:
            circle: The circle to compute cost for
            use_geometric: If True, use geometric average (sum of logs).
                          If False, use arithmetic sum.

        Geometric average favors even distribution of edge weights,
        penalizing folders with a few very high-weight (distant) edges.
        """
        if not circle.nodes or not self.mst_adjacency:
            return 0.0

        circle_nodes = set(circle.nodes)
        seen_edges = set()

        if use_geometric:
            # Geometric: sum of log(weights)
            log_sum = 0.0
            edge_count = 0

            for node in circle_nodes:
                for neighbor, weight in self.mst_adjacency.get(node, []):
                    if neighbor in circle_nodes:
                        edge = tuple(sorted([node, neighbor]))
                        if edge not in seen_edges:
                            seen_edges.add(edge)
                            # Add small epsilon to avoid log(0)
                            log_sum += np.log(weight + 1e-8)
                            edge_count += 1

            if edge_count == 0:
                return 0.0

            # Geometric mean of edge weights (weights are distances 0-1)
            geometric_mean = np.exp(log_sum / edge_count)

            # Low GM = some edges very tight (low weight), others loose
            #        = uneven distribution = natural cut points exist
            # High GM = uniform edge weights = no obvious cut point
            #
            # For subdivision: uneven (low GM) should trigger subdivision
            # Cost for NOT subdividing = 1/GM (high when GM low)
            # Scale by edge_count for size proportionality
            return edge_count / (geometric_mean + 1e-8)
        else:
            # Arithmetic: sum of weights
            total = 0.0
            for node in circle_nodes:
                for neighbor, weight in self.mst_adjacency.get(node, []):
                    if neighbor in circle_nodes:
                        total += weight
            return total / 2.0  # Each edge counted twice

    def _compute_size_cost(self, size: int) -> float:
        """Compute cost for a folder of given size.

        Modes:
        - 'quadratic': (size - target)² - penalizes oversized only
        - 'geometric': DEPRECATED - use gm_maximize instead
        - 'gm_maximize': handled separately in _should_subdivide
        """
        if self.size_cost_mode == 'geometric':
            # DEPRECATED: This doesn't correctly maximize GM.
            # Use --size-cost gm_maximize instead.
            import warnings
            warnings.warn(
                "geometric size cost is deprecated and incorrectly implemented. "
                "Use --size-cost gm_maximize for scale-invariant GM maximization.",
                DeprecationWarning, stacklevel=2
            )
            if size <= 0:
                return float('inf')
            log_ratio = np.log(size / self.target_size)
            return 2 * self.target_size ** 2 * log_ratio ** 2
        else:  # quadratic (default) or gm_maximize (handled elsewhere)
            excess = max(0, size - self.target_size)
            return excess ** 2

    def _compute_depth_penalty(self, depth: int) -> float:
        """Compute penalty for exceeding max_depth.

        Returns 0 for depth <= max_depth.
        Returns exponentially increasing penalty for depth > max_depth.
        """
        if depth <= self.max_depth:
            return 0.0

        excess = depth - self.max_depth

        if self.depth_penalty_base is None:
            # Default base if not calibrated
            self.depth_penalty_base = 1000.0

        # penalty = base * (exp(k * excess) - 1)
        return self.depth_penalty_base * (np.exp(self.depth_penalty_k * excess) - 1)

    def _calibrate_depth_penalty(self, circles: List[Circle]):
        """Calibrate depth penalty so it balances with per-folder size cost.

        We want: penalty(max_depth+1) ≈ cost of one large folder
        This allows subdivision when the benefit (reducing large folder) exceeds the penalty.
        """
        # Use the cost of a "typical large folder" as reference
        # A folder at 2x target_size is our threshold for concern
        reference_size = self.target_size * 2  # 20 items for target=10
        reference_cost = self._compute_size_cost(reference_size)

        if reference_cost > 0:
            # penalty(1) = base * (exp(k) - 1) = reference_cost
            self.depth_penalty_base = reference_cost / (np.exp(self.depth_penalty_k) - 1)
        else:
            self.depth_penalty_base = 100.0  # Default

        if self.verbose:
            print(f"Calibrated depth penalty: base={self.depth_penalty_base:.2f}, k={self.depth_penalty_k}")
            print(f"  reference_cost({reference_size} items) = {reference_cost:.2f}")
            print(f"  penalty(depth={self.max_depth+1}) = {self._compute_depth_penalty(self.max_depth+1):.2f}")
            print(f"  penalty(depth={self.max_depth+2}) = {self._compute_depth_penalty(self.max_depth+2):.2f}")

    def _compute_current_gm(self, circles: List[Circle], exclude_circle: Circle = None) -> float:
        """Compute geometric mean of folder sizes across all circles.

        Args:
            circles: List of all current circles
            exclude_circle: Optional circle to exclude from calculation

        Returns:
            Geometric mean of sizes (or target_size if empty)
        """
        sizes = [c.size for c in circles if c.size > 0 and c is not exclude_circle]
        if not sizes:
            return float(self.target_size)

        log_sum = sum(np.log(s) for s in sizes)
        return np.exp(log_sum / len(sizes))

    def _should_subdivide(self, circle: Circle, current_depth: int,
                          all_circles: List[Circle] = None) -> bool:
        """Decide whether to subdivide based on cost/benefit analysis.

        For 'gm_maximize' mode: uses dynamic threshold based on current GM.
        Split if: size > 4 × current_GM (derived from maximizing log(GM))
        This is SCALE INVARIANT - no fixed target_size needed.

        For other modes: compares cost of subdividing vs not subdividing.

        Returns True if subdividing improves the objective.
        """
        # Skip if subdivision was already attempted and failed for this circle
        if id(circle) in self._subdivision_failed:
            return False

        # GM-maximize mode: SCALE INVARIANT - uses dynamic threshold only
        if self.size_cost_mode == 'gm_maximize':
            # Can't split if too small (uses min_size as fraction of expected)
            if circle.size <= self.min_size * 2:
                return False

            if all_circles:
                current_gm = self._compute_current_gm(all_circles, exclude_circle=circle)
            else:
                # Bootstrap: use total/expected_folders as initial estimate
                expected_folders = max(1, self.n_items // self.target_size)
                current_gm = self.n_items / expected_folders

            # From derivation: split if n > 4 * GM
            # The 4× factor comes from log(GM) maximization math
            threshold = 4.0 * current_gm

            # Depth penalty raises threshold (less eager to split deep)
            depth_penalty = self._compute_depth_penalty(current_depth + 1)
            if depth_penalty > 0:
                threshold *= (1 + depth_penalty / 100)

            return circle.size > threshold

        # Cost-based modes: use target_size as floor
        if circle.size <= self.target_size:
            return False

        # Cost-based modes (quadratic, geometric)
        size_cost = self._compute_size_cost(circle.size)
        no_subdivide_cost = size_cost

        # Cost of subdividing (assuming roughly equal split)
        new_depth = current_depth + 1
        depth_penalty = self._compute_depth_penalty(new_depth)

        # Estimate size cost after subdivision (assume 2-way split)
        half_size = circle.size // 2
        subdivide_size_cost = 2 * self._compute_size_cost(half_size)
        subdivide_total_cost = depth_penalty + subdivide_size_cost

        # Optionally include internal edge costs (more accurate but slower)
        if self.internal_cost_mode != 'none':
            use_geometric = (self.internal_cost_mode == 'geometric')
            internal_cost = self._compute_internal_cost(circle, use_geometric=use_geometric)
            no_subdivide_cost += internal_cost
            # Internal cost after split: reduced by ~10% (cut edges removed)
            subdivide_total_cost += internal_cost * 0.9

        # Subdivide if it reduces cost
        return subdivide_total_cost < no_subdivide_cost

    def _incremental_partition(self) -> List[Circle]:
        """
        Partition MST using incremental growth (Option C from proposal).

        Grows circles from root, deciding at each step whether to:
        - Add node to current circle
        - Start a new child circle (cut)
        """
        root = self._find_mst_root()

        if self.verbose:
            print(f"Starting incremental partition from root node {root} ({self.titles[root]})")

        # Track which nodes are assigned
        assigned = set()

        # Root circle
        root_circle = Circle(name=self.titles[root])
        root_circle.add_node(root)
        assigned.add(root)

        # BFS queue: (node, parent_node, current_circle, depth)
        queue = []
        for neighbor, weight in self.mst_adjacency[root]:
            queue.append((neighbor, root, root_circle, 1))

        all_circles = [root_circle]

        while queue:
            node, parent, current_circle, depth = queue.pop(0)

            if node in assigned:
                continue

            # Decision: add to current circle or start new one?
            should_cut = self._should_cut(
                node, parent, current_circle, depth
            )

            if should_cut and depth < self.max_depth:
                # Start new child circle
                new_circle = Circle(name=self.titles[node])
                new_circle.add_node(node)
                new_circle.parent = current_circle
                current_circle.children.append(new_circle)

                # Mark boundary nodes
                current_circle.add_boundary(parent)
                new_circle.add_boundary(node)

                all_circles.append(new_circle)
                assigned.add(node)

                # Only increment depth if parent circle is reasonably sized
                # This prevents tiny circles from consuming depth budget
                new_depth = depth + 1 if current_circle.size >= self.target_size // 2 else depth

                # Continue with new circle
                for neighbor, weight in self.mst_adjacency[node]:
                    if neighbor not in assigned:
                        queue.append((neighbor, node, new_circle, new_depth))
            else:
                # Add to current circle
                current_circle.add_node(node)
                assigned.add(node)

                # Continue with same circle
                for neighbor, weight in self.mst_adjacency[node]:
                    if neighbor not in assigned:
                        queue.append((neighbor, node, current_circle, depth))

        self.circles = all_circles
        return all_circles

    def _should_cut(
        self,
        node: int,
        parent: int,
        current_circle: Circle,
        depth: int
    ) -> bool:
        """
        Decide whether to cut (start new circle) or continue current circle.

        Heuristics:
        1. Cut if circle exceeds target size
        2. Cut if semantic distance is large
        3. Don't cut if it would exceed max depth
        4. Don't cut if resulting circle would be too small
        """
        # Never cut at max depth
        if depth >= self.max_depth:
            return False

        # Cut if circle is getting too big
        if current_circle.size >= self.target_size:
            return True

        # Check semantic distance from circle centroid
        if current_circle.size > 0:
            circle_embeddings = self.embeddings[list(current_circle.nodes)]
            centroid = circle_embeddings.mean(axis=0)
            node_embedding = self.embeddings[node]

            # Cosine distance from centroid
            dist = 1 - np.dot(centroid, node_embedding) / (
                np.linalg.norm(centroid) * np.linalg.norm(node_embedding) + 1e-8
            )

            # Cut if semantically distant (threshold scales with depth)
            threshold = 0.3 + 0.1 * depth  # Stricter at root, looser at leaves
            if dist > threshold:
                return True

        return False

    def _subdivide_circle(self, circle: Circle, current_depth: int, subdivision_level: int = 0) -> List[Circle]:
        """
        Recursively subdivide a circle that's too large.

        Dispatches to the appropriate method based on self.subdivision_method.
        """
        if self.subdivision_method == 'multilevel':
            return self._subdivide_circle_multilevel(circle, current_depth, subdivision_level)
        else:
            return self._subdivide_circle_bisection(circle, current_depth, subdivision_level)

    def _subdivide_circle_bisection(self, circle: Circle, current_depth: int, subdivision_level: int = 0) -> List[Circle]:
        """
        Recursively subdivide a circle using balanced bisection.

        Uses the same MST-based approach but restricted to nodes in this circle.
        Uses cost-based decision: subdivide if it reduces total cost (size + depth penalty).
        """
        total_depth = current_depth + subdivision_level

        # Use cost-based decision for subdivision
        if not self._should_subdivide(circle, total_depth):
            return [circle]

        if self.verbose:
            print(f"  Subdividing '{circle.name}' ({circle.size} items) at depth {total_depth}/{self.max_depth}")

        # Get nodes in this circle
        nodes = list(circle.nodes)
        n = len(nodes)

        if n < self.min_size * 2:
            self._subdivision_failed.add(id(circle))
            return [circle]

        # Build local MST for this circle's nodes
        local_embeddings = self.embeddings[nodes]
        norms = np.linalg.norm(local_embeddings, axis=1, keepdims=True)
        normalized = local_embeddings / (norms + 1e-8)
        similarity = normalized @ normalized.T
        distances = np.maximum(1 - similarity, 0)
        np.fill_diagonal(distances, 0)

        local_mst = minimum_spanning_tree(distances)

        # Build adjacency from MST
        cx = local_mst.tocoo()
        mst_adj = defaultdict(list)
        all_edges = []
        for r, c, w in zip(cx.row, cx.col, cx.data):
            mst_adj[r].append((c, w))
            mst_adj[c].append((r, w))
            all_edges.append((r, c, w))

        # Use balanced bisection: find edge whose removal creates most balanced split
        def find_component_size(start, adj, excluded_edge):
            """BFS to find component size when one edge is excluded."""
            visited = {start}
            queue = [start]
            while queue:
                node = queue.pop(0)
                for neighbor, _ in adj[node]:
                    if neighbor not in visited:
                        # Check if this is the excluded edge
                        edge = tuple(sorted([node, neighbor]))
                        if edge != excluded_edge:
                            visited.add(neighbor)
                            queue.append(neighbor)
            return len(visited)

        def bisect_once(adj, n_nodes, edges):
            """Find the best edge to cut for balanced bisection."""
            best_edge = None
            best_balance = 0  # Higher is better (max is 0.5 for perfect split)

            for r, c, w in edges:
                edge = tuple(sorted([r, c]))
                # Find size of component containing r (without this edge)
                size_r = find_component_size(r, adj, edge)
                size_c = n_nodes - size_r

                # Balance score: min(size_r, size_c) / n_nodes
                balance = min(size_r, size_c) / n_nodes

                if balance > best_balance:
                    best_balance = balance
                    best_edge = edge

            return best_edge, best_balance

        # Iteratively bisect until we have enough components or can't improve
        cut_edges = set()
        target_components = max(2, n // self.target_size)

        for _ in range(target_components - 1):
            # Find edges still available (not cut)
            available_edges = [(r, c, w) for r, c, w in all_edges
                              if tuple(sorted([r, c])) not in cut_edges]

            if not available_edges:
                break

            best_edge, balance = bisect_once(mst_adj, n, available_edges)

            if best_edge is None or balance < 0.1:  # Stop if can't find good cut
                break

            cut_edges.add(best_edge)
            cut_edges.add((best_edge[1], best_edge[0]))

        # Build adjacency without cut edges
        adj = defaultdict(list)
        for r, c, w in zip(cx.row, cx.col, cx.data):
            edge = tuple(sorted([r, c]))
            if edge not in cut_edges:
                adj[r].append(c)
                adj[c].append(r)

        # Find connected components (each becomes a child circle)
        visited = set()
        components = []

        for start in range(n):
            if start in visited:
                continue
            component = []
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(nodes[node])  # Map back to global index
                stack.extend(adj[node])
            if component:
                components.append(component)

        if len(components) <= 1:
            if self.verbose and circle.size > self.target_size * 2:
                print(f"    WARNING: Could not subdivide '{circle.name}' ({circle.size} items) - graph not separable")
            self._subdivision_failed.add(id(circle))
            return [circle]

        # Create child circles
        result_circles = []
        for comp in components:
            child = Circle(name=self.titles[comp[0]])
            for node in comp:
                child.add_node(node)
            child.parent = circle  # Parent is the circle being subdivided

            # Recursively subdivide if still too large
            # Only increment subdivision_level (current_depth is fixed for this subdivision chain)
            subdivided = self._subdivide_circle(child, current_depth, subdivision_level + 1)
            result_circles.extend(subdivided)

            # Add as children of original circle for hierarchy
            circle.children.extend(subdivided)

        # Clear original nodes (they're now in child circles)
        circle.nodes.clear()
        circle.name = f"{circle.name} (subdivided)"

        return result_circles

    def _subdivide_circle_multilevel(self, circle: Circle, current_depth: int, subdivision_level: int = 0) -> List[Circle]:
        """
        Subdivide using multi-level cuts (top cut / bottom cuts).

        A circle is defined by:
        - Top cut: edge connecting to parent
        - Bottom cuts: edges connecting to children

        This allows circles to span multiple tree levels and provides
        more flexibility than simple bisection.
        """
        total_depth = current_depth + subdivision_level

        # Use cost-based decision for subdivision
        if not self._should_subdivide(circle, total_depth):
            return [circle]

        if self.verbose:
            print(f"  Subdividing '{circle.name}' ({circle.size} items) at depth {total_depth}/{self.max_depth} [multilevel]")

        # Get nodes in this circle
        nodes = list(circle.nodes)
        n = len(nodes)
        node_to_local = {node: i for i, node in enumerate(nodes)}
        local_to_node = {i: node for i, node in enumerate(nodes)}

        if n < self.min_size * 2:
            self._subdivision_failed.add(id(circle))
            return [circle]

        # Build local MST for this circle's nodes
        local_embeddings = self.embeddings[nodes]
        norms = np.linalg.norm(local_embeddings, axis=1, keepdims=True)
        normalized = local_embeddings / (norms + 1e-8)
        similarity = normalized @ normalized.T
        distances = np.maximum(1 - similarity, 0)
        np.fill_diagonal(distances, 0)

        local_mst = minimum_spanning_tree(distances)

        # Build adjacency from MST
        cx = local_mst.tocoo()
        mst_adj = defaultdict(list)
        for r, c, w in zip(cx.row, cx.col, cx.data):
            mst_adj[r].append((c, w))
            mst_adj[c].append((r, w))

        # Find a root node (highest degree or centroid)
        degrees = [(len(mst_adj[i]), i) for i in range(n)]
        _, root = max(degrees)

        # Build rooted tree structure
        parent_map = {}  # node -> parent
        children_map = defaultdict(list)  # node -> [children]
        depth_map = {root: 0}  # node -> depth in tree

        queue = [root]
        visited = {root}
        while queue:
            node = queue.pop(0)
            for neighbor, weight in mst_adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = node
                    children_map[node].append(neighbor)
                    depth_map[neighbor] = depth_map[node] + 1
                    queue.append(neighbor)

        # Compute subtree sizes
        subtree_size = {}
        def compute_subtree_size(node):
            size = 1
            for child in children_map[node]:
                size += compute_subtree_size(child)
            subtree_size[node] = size
            return size
        compute_subtree_size(root)

        # Find multi-level cuts: enumerate possible "bands" in the tree
        # A band is defined by: nodes reachable from a start node, stopping at certain cut points

        def find_band(start_node: int, max_size: int) -> Tuple[Set[int], List[int]]:
            """
            Grow a band from start_node, stopping when size exceeds max_size.
            Returns (band_nodes, bottom_cut_nodes).

            Uses BFS to grow, preferring to include children with smaller subtrees first.
            """
            band = {start_node}
            bottom_cuts = []
            frontier = []  # (subtree_size, node) - smaller subtrees first

            # Initialize frontier with children
            for child in children_map[start_node]:
                frontier.append((subtree_size[child], child))
            frontier.sort()  # Smallest first

            while frontier and len(band) < max_size:
                _, node = frontier.pop(0)

                # Check if adding this subtree would exceed max_size
                if len(band) + subtree_size[node] <= max_size:
                    # Add entire subtree
                    def add_subtree(n):
                        band.add(n)
                        for c in children_map[n]:
                            add_subtree(c)
                    add_subtree(node)
                else:
                    # Add just this node, children become new frontier
                    band.add(node)
                    for child in children_map[node]:
                        frontier.append((subtree_size[child], child))
                    frontier.sort()

            # Nodes in band with children outside band are bottom cuts
            for node in band:
                for child in children_map[node]:
                    if child not in band:
                        bottom_cuts.append(child)

            return band, bottom_cuts

        # Strategy: Create multiple bands of target_size
        # Start from root, greedily partition into bands

        bands = []
        remaining = set(range(n))

        # Sort nodes by depth (process shallower nodes first)
        nodes_by_depth = sorted(range(n), key=lambda x: depth_map[x])

        while remaining:
            # Find a good starting node (prefer nodes whose parent is not in remaining)
            start = None
            for node in nodes_by_depth:
                if node in remaining:
                    parent = parent_map.get(node)
                    if parent is None or parent not in remaining:
                        start = node
                        break

            if start is None:
                start = next(iter(remaining))

            # Grow band from start
            band, _ = find_band(start, self.target_size * 2)
            band = band & remaining  # Only include remaining nodes

            if band:
                bands.append(band)
                remaining -= band

        # If we only got one band, try harder to split
        if len(bands) == 1 and n > self.target_size * 2:
            # Fall back to balanced bisection for this case
            bands = self._bisect_into_bands(mst_adj, n, self.target_size)

        if len(bands) <= 1:
            if self.verbose and circle.size > self.target_size * 2:
                print(f"    WARNING: Could not subdivide '{circle.name}' ({circle.size} items) - multilevel failed")
            self._subdivision_failed.add(id(circle))
            return [circle]

        # Create child circles from bands
        result_circles = []
        for band in bands:
            global_nodes = [local_to_node[i] for i in band]
            child = Circle(name=self.titles[global_nodes[0]])
            for node in global_nodes:
                child.add_node(node)
            child.parent = circle

            # Recursively subdivide if still too large
            subdivided = self._subdivide_circle_multilevel(child, current_depth, subdivision_level + 1)
            result_circles.extend(subdivided)
            circle.children.extend(subdivided)

        # Clear original nodes
        circle.nodes.clear()
        circle.name = f"{circle.name} (subdivided)"

        return result_circles

    def _bisect_into_bands(self, mst_adj: Dict, n: int, target_size: int) -> List[Set[int]]:
        """
        Fallback: bisect the MST into roughly equal bands.
        Used when multi-level approach doesn't find good cuts.
        """
        # Collect all edges
        all_edges = set()
        for node in mst_adj:
            for neighbor, weight in mst_adj[node]:
                edge = tuple(sorted([node, neighbor]))
                all_edges.add((edge[0], edge[1], weight))
        all_edges = list(all_edges)

        def find_component(start, adj, cut_edges):
            """Find connected component containing start, avoiding cut edges."""
            visited = {start}
            queue = [start]
            while queue:
                node = queue.pop(0)
                for neighbor, _ in adj[node]:
                    if neighbor not in visited:
                        edge = tuple(sorted([node, neighbor]))
                        if edge not in cut_edges:
                            visited.add(neighbor)
                            queue.append(neighbor)
            return visited

        # Iteratively find best cuts
        cut_edges = set()
        target_bands = max(2, n // target_size)

        for _ in range(target_bands - 1):
            best_edge = None
            best_balance = 0

            for r, c, w in all_edges:
                edge = tuple(sorted([r, c]))
                if edge in cut_edges:
                    continue

                # Temporarily add this cut
                test_cuts = cut_edges | {edge}

                # Find components
                remaining = set(range(n))
                components = []
                while remaining:
                    start = next(iter(remaining))
                    comp = find_component(start, mst_adj, test_cuts)
                    components.append(comp)
                    remaining -= comp

                # Score by balance (how close to equal sizes)
                sizes = [len(c) for c in components]
                if len(sizes) > 1:
                    balance = min(sizes) / max(sizes)
                    if balance > best_balance:
                        best_balance = balance
                        best_edge = edge

            if best_edge and best_balance > 0.1:
                cut_edges.add(best_edge)
            else:
                break

        # Find final components
        remaining = set(range(n))
        bands = []
        while remaining:
            start = next(iter(remaining))
            comp = find_component(start, mst_adj, cut_edges)
            bands.append(comp)
            remaining -= comp

        return bands

    def partition(self, sparse_threshold: int = 5000, knn_k: int = 50) -> List[Circle]:
        """Run the full partitioning algorithm."""
        # Step 1-2: Build tree structure (MST, curated, or hybrid)
        if self.tree_source == 'curated':
            # Use curated hierarchy paths directly (no MST computation)
            if not self.hierarchy_paths:
                raise ValueError("hierarchy_paths required when tree_source='curated'")
            self._build_curated_tree()
        elif self.tree_source == 'hybrid':
            # Curated structure + greedy orphan attachment with blended embeddings
            if not self.hierarchy_paths:
                raise ValueError("hierarchy_paths required when tree_source='hybrid'")
            if self.output_embeddings is None:
                raise ValueError("output_embeddings required when tree_source='hybrid'")
            self._build_hybrid_tree()
        else:
            # Compute MST from embeddings
            if self.n_items > sparse_threshold:
                if self.verbose:
                    print(f"Using sparse k-NN mode (n={self.n_items} > {sparse_threshold})")
                distances = self._compute_sparse_knn_graph(k=knn_k)
            else:
                distances = self._compute_distance_matrix()

            self._build_mst(distances)

        # Step 3: Partition into circles
        circles = self._incremental_partition()

        # Step 4: Calibrate depth penalty based on initial partition
        self._calibrate_depth_penalty(circles)

        # Step 5: Recursively subdivide oversized circles (using cost-based decision)
        if self.verbose:
            oversized = [c for c in circles if c.size > self.target_size]
            if oversized:
                print(f"\nSubdividing {len(oversized)} oversized circles...")

        final_circles = []
        for circle in circles:
            depth = 0
            parent = circle.parent
            while parent:
                depth += 1
                parent = parent.parent
            final_circles.extend(self._subdivide_circle(circle, depth))

        # Keep subdividing until cost-based decision says stop
        max_iterations = 20
        for iteration in range(max_iterations):
            # Find circles where subdivision would reduce cost
            to_subdivide = []
            for c in final_circles:
                # Compute depth of this circle
                depth = 0
                p = c.parent
                while p:
                    depth += 1
                    p = p.parent
                # Pass all_circles for gm_maximize mode
                if self._should_subdivide(c, depth, all_circles=final_circles):
                    to_subdivide.append((c, depth))

            if not to_subdivide:
                break

            if self.verbose:
                print(f"  Pass {iteration + 2}: {len(to_subdivide)} circles benefit from subdivision...")

            new_circles = []
            subdivide_ids = {id(c) for c, _ in to_subdivide}
            for c in final_circles:
                if id(c) in subdivide_ids:
                    # Find depth for this circle
                    depth = 0
                    p = c.parent
                    while p:
                        depth += 1
                        p = p.parent
                    subdivided = self._subdivide_circle(c, depth, 0)
                    new_circles.extend(subdivided)
                else:
                    new_circles.append(c)
            final_circles = new_circles

        self.circles = final_circles

        # Step 5: Merge small sibling circles
        if self.verbose:
            small_count = sum(1 for c in final_circles if c.size < self.min_size)
            if small_count > 0:
                print(f"\nMerging {small_count} small circles (< {self.min_size} items)...")

        # Find root circle (the one with no parent in our circles list)
        root_circle = circles[0] if circles else None
        self._merge_small_circles(root_circle)

        # Rebuild final_circles list after merging
        final_circles = []
        def collect_circles(circle):
            if circle.size > 0:  # Only include non-empty circles
                final_circles.append(circle)
            for child in circle.children:
                collect_circles(child)

        if self.circles:
            collect_circles(self.circles[0])

        self.circles = final_circles

        if self.verbose:
            self._print_summary()

        return final_circles

    def _merge_small_circles(self, circle: Circle):
        """Recursively merge small sibling circles into their parents."""
        if circle is None:
            return

        # First, recurse into children
        for child in circle.children[:]:  # Copy list since we'll modify
            self._merge_small_circles(child)

        # Step 1: Rebalance - small circles steal from large siblings
        self._rebalance_siblings(circle.children)

        # Step 2: Small circles steal from parent if parent is large
        self._steal_from_parent(circle)

        # Step 3: Merge very small siblings together
        self._merge_tiny_siblings(circle.children)

        # Step 4: Merge remaining small children back into parent
        max_parent_size = self.target_size * 2
        children_to_remove = []

        for child in circle.children:
            # Only merge if child is small AND parent won't become too large
            if child.size < self.min_size and child.size > 0:
                if circle.size + child.size <= max_parent_size:
                    # Merge child's items into parent
                    for node in child.nodes:
                        circle.add_node(node)
                    # Also merge grandchildren up
                    for grandchild in child.children:
                        grandchild.parent = circle
                        circle.children.append(grandchild)
                    children_to_remove.append(child)

        for child in children_to_remove:
            circle.children.remove(child)

    def _rebalance_siblings(self, siblings: List[Circle]):
        """Rebalance items between sibling circles - small ones steal from large ones."""
        if not siblings:
            return

        # Find small and large siblings
        small = [c for c in siblings if c.size < self.min_size and c.size > 0]
        large = [c for c in siblings if c.size > self.target_size]

        if not small or not large:
            return

        for small_circle in small:
            items_needed = self.min_size - small_circle.size

            for large_circle in large:
                if items_needed <= 0:
                    break
                if large_circle.size <= self.target_size:
                    continue  # Skip if no longer oversized

                # Find items in large_circle closest to small_circle's centroid
                small_nodes = list(small_circle.nodes)
                large_nodes = list(large_circle.nodes)

                if not small_nodes or not large_nodes:
                    continue

                # Compute small circle centroid
                small_embeddings = self.embeddings[small_nodes]
                centroid = small_embeddings.mean(axis=0)
                centroid_norm = np.linalg.norm(centroid)

                # Find closest nodes in large circle
                large_embeddings = self.embeddings[large_nodes]
                similarities = (large_embeddings @ centroid) / (
                    np.linalg.norm(large_embeddings, axis=1) * centroid_norm + 1e-8
                )

                # Transfer closest items (up to items_needed, but keep large_circle >= target_size)
                transfer_count = min(
                    items_needed,
                    large_circle.size - self.target_size
                )

                if transfer_count <= 0:
                    continue

                closest_indices = np.argsort(similarities)[-transfer_count:]

                for idx in closest_indices:
                    node = large_nodes[idx]
                    large_circle.nodes.remove(node)
                    small_circle.add_node(node)
                    items_needed -= 1

    def _steal_from_parent(self, parent: Circle):
        """Allow small children to steal MST subtrees from an oversized parent.

        Instead of stealing individual items, this moves MST cuts to transfer
        entire connected subtrees. This keeps semantically connected items together.
        """
        if parent.size <= self.target_size:
            return

        # Find small children that need items
        small_children = [c for c in parent.children if c.size < self.min_size and c.size > 0]
        if not small_children:
            return

        parent_node_set = set(parent.nodes)
        if not parent_node_set:
            return

        for child in small_children:
            while child.size < self.min_size and parent.size > self.target_size:
                # Find MST edges from child's nodes to parent's nodes
                child_nodes = set(child.nodes)
                candidate_edges = []

                for c_node in child_nodes:
                    for neighbor, weight in self.mst_adjacency.get(c_node, []):
                        if neighbor in parent_node_set:
                            candidate_edges.append((c_node, neighbor, weight))

                if not candidate_edges:
                    break

                # For each candidate edge, compute the subtree size in parent
                best_edge = None
                best_subtree = None
                best_size = 0

                for c_node, p_node, weight in candidate_edges:
                    # Find subtree in parent starting from p_node, excluding edge to c_node
                    subtree = self._find_subtree(p_node, parent_node_set, exclude_node=c_node)

                    # Prefer subtrees that bring child closer to min_size
                    # but don't make it larger than target_size
                    if len(subtree) > 0:
                        new_child_size = child.size + len(subtree)
                        new_parent_size = parent.size - len(subtree)

                        # Skip if would make child too large
                        if new_child_size > self.target_size * 1.5:
                            continue
                        # Skip if would make parent too small
                        if new_parent_size < self.min_size and new_parent_size > 0:
                            continue

                        # Prefer subtrees that bring child closer to target
                        if len(subtree) > best_size:
                            best_size = len(subtree)
                            best_edge = (c_node, p_node)
                            best_subtree = subtree

                if best_subtree is None:
                    break

                # Move subtree from parent to child
                for node in best_subtree:
                    parent.nodes.discard(node)
                    child.add_node(node)

                # Also move any parent's children connected only through transferred nodes
                transferred_set = best_subtree
                children_to_move = []
                for pc in parent.children:
                    if pc is child:
                        continue
                    # Check if this child is connected to parent only through transferred nodes
                    pc_boundary = self._get_boundary_to_circle(pc, parent)
                    if pc_boundary and all(n in transferred_set for n in pc_boundary):
                        children_to_move.append(pc)

                for pc in children_to_move:
                    parent.children.remove(pc)
                    pc.parent = child
                    child.children.append(pc)

                # Update parent_node_set
                parent_node_set = set(parent.nodes)

    def _find_subtree(self, start_node: int, within_nodes: Set[int],
                      exclude_node: int = None) -> Set[int]:
        """Find all nodes in the MST subtree rooted at start_node.

        Only includes nodes that are in within_nodes.
        Does not traverse through exclude_node.
        """
        subtree = set()
        queue = [start_node]
        visited = {exclude_node} if exclude_node is not None else set()

        while queue:
            node = queue.pop(0)
            if node in visited or node not in within_nodes:
                continue

            visited.add(node)
            subtree.add(node)

            for neighbor, weight in self.mst_adjacency.get(node, []):
                if neighbor not in visited and neighbor in within_nodes:
                    queue.append(neighbor)

        return subtree

    def _get_boundary_to_circle(self, child: Circle, parent: Circle) -> Set[int]:
        """Find parent nodes that connect to this child via MST edges."""
        boundary = set()
        parent_nodes = set(parent.nodes)
        child_nodes = set(child.nodes)

        # Also include all nested descendants of child
        def collect_all_nodes(c):
            nodes = set(c.nodes)
            for cc in c.children:
                nodes |= collect_all_nodes(cc)
            return nodes

        all_child_nodes = collect_all_nodes(child)

        for c_node in all_child_nodes:
            for neighbor, weight in self.mst_adjacency.get(c_node, []):
                if neighbor in parent_nodes:
                    boundary.add(neighbor)

        return boundary

    def _merge_tiny_siblings(self, siblings: List[Circle]):
        """Merge very small sibling circles together.

        Groups tiny siblings (size 1-2) based on embedding similarity and merges
        them into larger combined folders.
        """
        if not siblings:
            return

        # Find tiny siblings (1-2 items)
        tiny = [c for c in siblings if 0 < c.size < self.min_size]
        if len(tiny) < 2:
            return

        # Compute centroids for tiny siblings
        tiny_centroids = []
        for c in tiny:
            nodes = list(c.nodes)
            if nodes:
                centroid = self.embeddings[nodes].mean(axis=0)
                tiny_centroids.append(centroid)
            else:
                tiny_centroids.append(np.zeros(self.embeddings.shape[1]))

        tiny_centroids = np.array(tiny_centroids)

        # Normalize for cosine similarity
        norms = np.linalg.norm(tiny_centroids, axis=1, keepdims=True) + 1e-8
        normalized = tiny_centroids / norms

        # Greedy merge: pair most similar tiny siblings
        merged_indices = set()

        for i in range(len(tiny)):
            if i in merged_indices:
                continue

            target_circle = tiny[i]

            # Find most similar unmerged siblings to merge with
            while target_circle.size < self.min_size:
                best_j = None
                best_sim = -1

                for j in range(len(tiny)):
                    if j == i or j in merged_indices:
                        continue
                    if tiny[j].size == 0:
                        continue

                    # Check if merging would exceed target size
                    if target_circle.size + tiny[j].size > self.target_size:
                        continue

                    sim = np.dot(normalized[i], normalized[j])
                    if sim > best_sim:
                        best_sim = sim
                        best_j = j

                if best_j is None:
                    break

                # Merge tiny[best_j] into target_circle
                source_circle = tiny[best_j]
                for node in list(source_circle.nodes):
                    target_circle.add_node(node)
                    source_circle.nodes.remove(node)

                # Move source's children to target
                for child in source_circle.children:
                    child.parent = target_circle
                    target_circle.children.append(child)
                source_circle.children = []

                merged_indices.add(best_j)

        # Remove empty siblings
        for c in list(siblings):
            if c.size == 0 and not c.children:
                siblings.remove(c)

    def _print_summary(self):
        """Print partition summary."""
        print(f"\n=== Partition Summary ===")
        print(f"Total items: {self.n_items}")
        print(f"Total circles: {len(self.circles)}")

        sizes = [c.size for c in self.circles]
        print(f"Circle sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")

        # Compute total cost
        total_cost = sum(self._compute_circle_cost(c) for c in self.circles)
        print(f"Total boundary cost: {total_cost:.4f}")

        print(f"\nCircle hierarchy:")
        self._print_circle_tree(self.circles[0], indent=0)

    def print_statistics(self):
        """Print detailed statistics tables."""
        from collections import Counter

        # Compute depths
        def get_depth(circle, depth=0):
            depths = [(circle, depth)]
            for child in circle.children:
                depths.extend(get_depth(child, depth + 1))
            return depths

        circle_depths = get_depth(self.circles[0]) if self.circles else []
        depth_counts = Counter(d for _, d in circle_depths)
        sizes = [c.size for c, _ in circle_depths if c.size > 0]
        size_counts = Counter(sizes)

        print(f"\n{'='*50}")
        print(f"FOLDER STATISTICS")
        print(f"{'='*50}")

        # Summary table
        print(f"\n| Metric | Value |")
        print(f"|--------|-------|")
        print(f"| Total items | {self.n_items:,} |")
        print(f"| Total folders | {len([c for c, _ in circle_depths if c.size > 0]):,} |")
        print(f"| Max depth | {max(depth_counts.keys()) if depth_counts else 0} |")
        print(f"| Folder size range | {min(sizes) if sizes else 0}–{max(sizes) if sizes else 0} |")
        print(f"| Average folder size | {np.mean(sizes):.1f} |" if sizes else "| Average folder size | 0 |")

        # Depth distribution
        print(f"\n| Depth | Folders |")
        print(f"|-------|---------|")
        for depth in sorted(depth_counts.keys()):
            print(f"| {depth} | {depth_counts[depth]} |")

        # Size distribution
        print(f"\n| Size | Folders |")
        print(f"|------|---------|")
        for size in sorted(size_counts.keys()):
            print(f"| {size} | {size_counts[size]} |")

    def _print_circle_tree(self, circle: Circle, indent: int):
        """Print circle hierarchy as tree."""
        prefix = "  " * indent
        boundary_str = f" [boundary: {len(circle.boundary_nodes)}]" if circle.boundary_nodes else ""
        print(f"{prefix}- {circle.name} ({circle.size} items){boundary_str}")

        # Show some items
        if indent < 2:
            for node in list(circle.nodes)[:3]:
                print(f"{prefix}    * {self.titles[node]}")
            if circle.size > 3:
                print(f"{prefix}    ... and {circle.size - 3} more")

        for child in circle.children:
            self._print_circle_tree(child, indent + 1)

    def to_folder_structure(self) -> Dict:
        """Convert circles to folder structure dict."""
        def circle_to_dict(circle: Circle) -> Dict:
            return {
                "name": circle.name,
                "items": [
                    {"title": self.titles[n], "tree_id": self.tree_ids[n]}
                    for n in circle.nodes
                ],
                "children": [circle_to_dict(c) for c in circle.children]
            }

        if self.circles:
            return circle_to_dict(self.circles[0])
        return {}


def load_physics_subset(embeddings_path: Path, targets_path: Path,
                        include_output: bool = False) -> Tuple[np.ndarray, List[str], List[str], Optional[np.ndarray]]:
    """Load physics trees subset with embeddings.

    Returns:
        embeddings: Input embeddings (N, D)
        titles: List of titles
        tree_ids: List of tree IDs
        output_embeddings: Output embeddings (N, D) if include_output=True, else None
    """
    # Load targets to get tree IDs
    tree_ids = []
    titles = []
    with open(targets_path) as f:
        for line in f:
            data = json.loads(line)
            tree_ids.append(data['tree_id'])
            titles.append(data['raw_title'])

    # Load full embeddings and filter
    full_data = np.load(embeddings_path, allow_pickle=True)
    full_tree_ids = list(full_data['tree_ids'])
    full_embeddings = full_data['input_nomic']
    full_output_embeddings = full_data['output_nomic'] if include_output else None

    # Create lookup
    id_to_idx = {tid: i for i, tid in enumerate(full_tree_ids)}

    # Filter to physics subset
    indices = [id_to_idx[tid] for tid in tree_ids if tid in id_to_idx]
    filtered_tree_ids = [tid for tid in tree_ids if tid in id_to_idx]
    filtered_titles = [titles[i] for i, tid in enumerate(tree_ids) if tid in id_to_idx]

    embeddings = full_embeddings[indices]
    output_embeddings = full_output_embeddings[indices] if include_output else None

    return embeddings, filtered_titles, filtered_tree_ids, output_embeddings


def load_trees_only(embeddings_path: Path, limit: Optional[int] = None,
                    include_output: bool = False) -> Tuple[np.ndarray, List[str], List[str], Optional[np.ndarray]]:
    """Load only Tree items from embeddings.

    Returns:
        embeddings: Input embeddings (N, D)
        titles: List of titles
        tree_ids: List of tree IDs
        output_embeddings: Output embeddings (N, D) if include_output=True, else None
    """
    data = np.load(embeddings_path, allow_pickle=True)

    item_types = data['item_types']
    tree_mask = item_types == 'Tree'

    embeddings = data['input_nomic'][tree_mask]
    output_embeddings = data['output_nomic'][tree_mask] if include_output else None
    titles = list(data['titles'][tree_mask])
    tree_ids = list(data['tree_ids'][tree_mask])

    if limit and limit < len(embeddings):
        embeddings = embeddings[:limit]
        titles = titles[:limit]
        tree_ids = tree_ids[:limit]
        if include_output:
            output_embeddings = output_embeddings[:limit]

    return embeddings, titles, tree_ids, output_embeddings


def load_hierarchy_paths(targets_path: Path) -> Dict[str, str]:
    """Load hierarchy paths from JSONL file.

    Returns dict mapping tree_id -> path string (e.g., '/id1/id2/.../idn').
    The path is extracted from the first line of target_text.
    """
    hierarchy_paths = {}
    with open(targets_path) as f:
        for line in f:
            data = json.loads(line)
            tree_id = data.get('tree_id')
            target_text = data.get('target_text', '')

            # First line of target_text is the path
            path = target_text.split('\n')[0].strip()
            if tree_id and path.startswith('/'):
                hierarchy_paths[tree_id] = path

    return hierarchy_paths


def main():
    parser = argparse.ArgumentParser(
        description='MST Circle-Based Folder Grouping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--subset', choices=['physics'], default=None,
                        help='Use a named subset (physics)')
    parser.add_argument('--trees-only', action='store_true',
                        help='Use all Tree items (no pearls)')
    parser.add_argument('--embeddings', type=Path,
                        default=Path('datasets/pearltrees_combined_2026-01-02_all_fixed_embeddings.npz'),
                        help='Path to embeddings file')
    parser.add_argument('--targets', type=Path,
                        default=Path('reports/pearltrees_targets_physics_trees.jsonl'),
                        help='Path to targets file (for subset)')

    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of items to process (for testing)')
    parser.add_argument('--target-size', type=int, default=8,
                        help='Target items per folder (default: 8)')
    parser.add_argument('--max-depth', type=int, default=4,
                        help='Maximum folder depth (default: 4)')
    parser.add_argument('--min-size', type=int, default=2,
                        help='Minimum items per folder (default: 2)')
    parser.add_argument('--subdivision-method', type=str, default='multilevel',
                        choices=['multilevel', 'bisection'],
                        help='Subdivision method: multilevel (default) or bisection')
    parser.add_argument('--internal-cost', type=str, default='none',
                        choices=['none', 'arithmetic', 'geometric'],
                        help='Internal cost mode: none (default), arithmetic, or geometric')
    parser.add_argument('--size-cost', type=str, default='gm_maximize',
                        choices=['gm_maximize', 'quadratic', 'geometric'],
                        help='Size cost: gm_maximize (default, scale-invariant), quadratic, or geometric (deprecated)')
    parser.add_argument('--tree-source', type=str, default='mst',
                        choices=['mst', 'curated', 'hybrid'],
                        help='Tree source: mst (compute from embeddings), curated (use hierarchy paths), or hybrid (curated + greedy orphan attachment)')
    parser.add_argument('--embed-blend', type=float, default=0.3,
                        help='Embedding blend for hybrid mode: weight for input embeddings (default: 0.3 = 30%% input, 70%% output)')

    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output JSON file for folder structure')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress')
    parser.add_argument('--stats', '-s', action='store_true',
                        help='Print statistics tables (markdown format)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show results without saving')

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    embeddings_path = project_root / args.embeddings

    # Load data
    hierarchy_paths = {}
    output_embeddings = None
    need_output = args.tree_source == 'hybrid'
    need_hierarchy = args.tree_source in ('curated', 'hybrid')

    if args.subset == 'physics':
        targets_path = project_root / args.targets
        print(f"Loading physics subset from {targets_path}...")
        embeddings, titles, tree_ids, output_embeddings = load_physics_subset(
            embeddings_path, targets_path, include_output=need_output)
        if need_hierarchy:
            hierarchy_paths = load_hierarchy_paths(targets_path)
    elif args.trees_only:
        print(f"Loading trees-only from {embeddings_path}...")
        embeddings, titles, tree_ids, output_embeddings = load_trees_only(
            embeddings_path, limit=args.limit, include_output=need_output)
        if need_hierarchy:
            # For trees-only mode with curated/hybrid, need a targets file
            # Default to the combined trees file
            trees_targets = project_root / 'reports/pearltrees_targets_combined_2026-01-02_trees_fixed.jsonl'
            if trees_targets.exists():
                hierarchy_paths = load_hierarchy_paths(trees_targets)
                print(f"Loaded {len(hierarchy_paths)} hierarchy paths from {trees_targets}")
            else:
                parser.error(f"--tree-source {args.tree_source} requires a targets file with hierarchy paths")
    else:
        parser.error("Must specify --subset or --trees-only")

    print(f"Loaded {len(embeddings)} items")
    if args.tree_source == 'curated':
        print(f"Using curated hierarchy ({len(hierarchy_paths)} paths)")
    elif args.tree_source == 'hybrid':
        print(f"Using hybrid mode: {len(hierarchy_paths)} hierarchy paths, "
              f"{args.embed_blend:.0%} input / {1-args.embed_blend:.0%} output blend")

    # Run grouping
    grouper = MSTFolderGrouper(
        embeddings=embeddings,
        titles=titles,
        tree_ids=tree_ids,
        target_size=args.target_size,
        max_depth=args.max_depth,
        min_size=args.min_size,
        verbose=args.verbose or args.dry_run,
        subdivision_method=args.subdivision_method,
        internal_cost_mode=args.internal_cost,
        size_cost_mode=args.size_cost,
        tree_source=args.tree_source,
        hierarchy_paths=hierarchy_paths,
        output_embeddings=output_embeddings,
        embed_blend=args.embed_blend
    )

    circles = grouper.partition()

    # Print statistics if requested
    if args.stats:
        grouper.print_statistics()

    # Output
    if args.output and not args.dry_run:
        folder_structure = grouper.to_folder_structure()
        with open(args.output, 'w') as f:
            json.dump(folder_structure, f, indent=2)
        print(f"\nSaved folder structure to {args.output}")
    elif args.dry_run:
        print("\n[DRY RUN] Would save folder structure")

    print(f"\nDone. Created {len(circles)} folders.")


if __name__ == '__main__':
    main()
