# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Proper Small-World Network with High Connectivity.

Each node has 10-20 connections:
- Local connections: k-nearest neighbors (semantic similarity)
- Long-range connections: Random distant nodes (small-world property)

This follows the Watts-Strogatz model adapted for semantic embeddings:
- High clustering (similar nodes connected)
- Short path lengths (long-range shortcuts)

See: Watts & Strogatz (1998) "Collective dynamics of small-world networks"
"""

import math
import random
import bisect
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from enum import Enum
import heapq

import numpy as np


class AngleOrdering(Enum):
    """
    Strategy for computing angles to order neighbors for binary search.

    COSINE_BASED: Use arccos(cosine_similarity) - accurate for high-dimensional vectors.
                  This is the recommended default.
    PROJECTION_2D: Use atan2 on first 2 dimensions - only accurate for 2D data.
                   Deprecated: loses discrimination in high-dimensional space.
    """
    COSINE_BASED = "cosine_based"
    PROJECTION_2D = "projection_2d"


def compute_angle_2d(direction: np.ndarray) -> float:
    """
    Compute angle from first two components of a direction vector.

    DEPRECATED: Use compute_cosine_angle for high-dimensional vectors.
    This is only accurate for 2D data.

    Returns angle in radians [-pi, pi].
    """
    if len(direction) < 2:
        return 0.0
    return math.atan2(direction[1], direction[0])


def compute_cosine_angle(vec: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute angle between vector and reference using cosine similarity.

    Returns angle in radians [0, pi]. This gives meaningful ordering
    in high-dimensional space based on actual vector similarity.

    Args:
        vec: The vector to measure
        reference: Reference direction (e.g., centroid)

    Returns:
        Angle in radians [0, pi]
    """
    norm_vec = np.linalg.norm(vec)
    norm_ref = np.linalg.norm(reference)
    if norm_vec == 0 or norm_ref == 0:
        return math.pi / 2  # Perpendicular if zero vector

    similarity = np.dot(vec, reference) / (norm_vec * norm_ref)
    # Clamp to [-1, 1] to handle numerical errors
    similarity = max(-1.0, min(1.0, similarity))
    return math.acos(similarity)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity)."""
    return 1.0 - cosine_similarity(a, b)


@dataclass
class SWNode:
    """
    A node in the small-world network.

    Supports optimized neighbor lookup via angle-sorted neighbor list.
    Neighbors are sorted by their angle from the node's centroid,
    enabling O(log k) lookup using binary search.
    """

    node_id: str
    vector: np.ndarray

    # All connections (no hierarchy distinction)
    neighbors: Set[str] = field(default_factory=set)

    # Sorted neighbor list: [(angle, neighbor_id), ...] sorted by angle
    # This enables O(log k) lookup via binary search
    sorted_neighbors: List[Tuple[float, str]] = field(default_factory=list)

    # Centroid tracking for angle-based neighbor ordering
    # The centroid is the node's representative vector, determined by its data
    # (not by routing connections). For a leaf node, this is the centroid of
    # stored documents. For routing, this is the node's embedding.
    # Note: Centroid updates happen when node DATA changes, not when routing
    # connections change (neighbors are just pointers, not data).
    _centroid: Optional[np.ndarray] = field(default=None, repr=False)
    _centroid_count: int = field(default=0, repr=False)

    # Configuration
    max_neighbors: int = 20

    # Statistics
    queries_routed: int = 0

    def __post_init__(self):
        """Initialize centroid from vector."""
        if self._centroid is None:
            self._centroid = self.vector.copy()
            self._centroid_count = 1

    @property
    def centroid(self) -> np.ndarray:
        """Get the current centroid."""
        return self._centroid if self._centroid is not None else self.vector

    def update_centroid_add(self, new_vector: np.ndarray) -> None:
        """
        Incrementally update centroid when adding a vector.

        Uses running average: C_new = (C * n + v) / (n + 1)
        """
        n = self._centroid_count
        self._centroid = (self._centroid * n + new_vector) / (n + 1)
        self._centroid_count = n + 1

    def update_centroid_remove(self, old_vector: np.ndarray) -> None:
        """
        Incrementally update centroid when removing a vector.

        Uses: C_new = (C * n - v) / (n - 1)
        """
        n = self._centroid_count
        if n <= 1:
            # Reset to node's own vector
            self._centroid = self.vector.copy()
            self._centroid_count = 1
            return

        self._centroid = (self._centroid * n - old_vector) / (n - 1)
        self._centroid_count = n - 1

    def compute_neighbor_angle(self, neighbor_vector: np.ndarray) -> float:
        """Compute angle of neighbor relative to centroid using cosine."""
        return compute_cosine_angle(neighbor_vector, self.centroid)

    def add_neighbor(self, neighbor_id: str, neighbor_vector: np.ndarray = None) -> bool:
        """
        Add a neighbor connection with insertion sort for angle ordering.

        Args:
            neighbor_id: ID of neighbor to add
            neighbor_vector: Vector of neighbor (for angle computation)

        Returns True if added successfully.
        """
        if neighbor_id == self.node_id:
            return False
        if len(self.neighbors) >= self.max_neighbors:
            return False
        if neighbor_id in self.neighbors:
            return False

        self.neighbors.add(neighbor_id)

        # Insert into sorted list if vector provided
        if neighbor_vector is not None:
            angle = self.compute_neighbor_angle(neighbor_vector)
            # Use bisect for insertion sort - O(log k) search, O(k) insert
            bisect.insort(self.sorted_neighbors, (angle, neighbor_id))

        return True

    def remove_neighbor(self, neighbor_id: str) -> bool:
        """Remove a neighbor connection."""
        if neighbor_id in self.neighbors:
            self.neighbors.discard(neighbor_id)

            # Remove from sorted list - O(k) scan
            self.sorted_neighbors = [
                (angle, nid) for angle, nid in self.sorted_neighbors
                if nid != neighbor_id
            ]
            return True
        return False

    def lookup_neighbors_by_angle(
        self,
        query_vector: np.ndarray,
        window_size: int = 5,
    ) -> List[str]:
        """
        Find neighbors near the query's angle using binary search.

        Args:
            query_vector: Query to find neighbors for
            window_size: Number of neighbors to return on each side

        Returns:
            List of neighbor IDs near the query angle
        """
        if not self.sorted_neighbors:
            # Fall back to all neighbors
            return list(self.neighbors)

        # Compute query angle
        query_angle = self.compute_neighbor_angle(query_vector)

        # Binary search for position
        idx = bisect.bisect_left(self.sorted_neighbors, (query_angle,))

        # Get neighbors in window around this position
        n = len(self.sorted_neighbors)
        start = max(0, idx - window_size)
        end = min(n, idx + window_size + 1)

        # Handle wraparound (angles are circular: -pi to pi)
        candidates = []

        # Main window
        for i in range(start, end):
            candidates.append(self.sorted_neighbors[i][1])

        # Wraparound: if near -pi, also check near +pi
        if idx < window_size:
            for i in range(n - (window_size - idx), n):
                if self.sorted_neighbors[i][1] not in candidates:
                    candidates.append(self.sorted_neighbors[i][1])

        # Wraparound: if near +pi, also check near -pi
        if idx > n - window_size:
            for i in range(0, window_size - (n - idx)):
                if self.sorted_neighbors[i][1] not in candidates:
                    candidates.append(self.sorted_neighbors[i][1])

        return candidates

    def rebuild_sorted_neighbors(self, get_vector_fn) -> None:
        """
        Rebuild the sorted neighbor list from scratch.

        Call this periodically to correct numeric drift in centroid
        or after major topology changes.

        Args:
            get_vector_fn: Function that takes neighbor_id and returns vector
        """
        self.sorted_neighbors = []
        for neighbor_id in self.neighbors:
            try:
                neighbor_vector = get_vector_fn(neighbor_id)
                angle = self.compute_neighbor_angle(neighbor_vector)
                self.sorted_neighbors.append((angle, neighbor_id))
            except (KeyError, ValueError):
                continue

        self.sorted_neighbors.sort()


class SmallWorldProper:
    """
    Proper small-world network with high connectivity.

    Each node maintains ~k_local + k_long connections:
    - k_local: Nearest neighbors by semantic similarity
    - k_long: Random long-range connections

    Routing uses greedy forwarding with backtracking.
    """

    def __init__(
        self,
        k_local: int = 10,      # Local neighbors per node
        k_long: int = 5,        # Long-range connections per node
        alpha: float = 2.0,     # Kleinberg exponent for long-range link probability
        max_neighbors: int = 20,
        rewire_prob: float = 0.1,  # Probability of rewiring
        angle_ordering: AngleOrdering = AngleOrdering.COSINE_BASED,
        embedding_dim: int = 384,  # For compatibility with generated code
    ):
        """
        Initialize small-world network.

        Args:
            k_local: Number of local (nearest) neighbors
            k_long: Number of long-range (random) connections
            alpha: Kleinberg exponent - P(link) ~ 1/distance^alpha
            max_neighbors: Maximum total neighbors per node
            rewire_prob: Probability of rewiring during evolution
            angle_ordering: Strategy for computing angles (COSINE_BASED recommended)
            embedding_dim: Embedding dimension (for compatibility)
        """
        self.nodes: Dict[str, SWNode] = {}
        self.k_local = k_local
        self.k_long = k_long
        self.alpha = alpha
        self.max_neighbors = max_neighbors
        self.rewire_prob = rewire_prob
        self.angle_ordering = angle_ordering
        self.embedding_dim = embedding_dim

    def add_node(
        self,
        node_id: str,
        vector: np.ndarray,
        connect: bool = True,
    ) -> SWNode:
        """
        Add a node and optionally connect to existing nodes.

        Args:
            node_id: Unique identifier
            vector: Embedding vector
            connect: If True, connect to k_local nearest + k_long random
        """
        node = SWNode(
            node_id=node_id,
            vector=vector,
            max_neighbors=self.max_neighbors,
        )
        self.nodes[node_id] = node

        if connect and len(self.nodes) > 1:
            self._connect_node(node_id)

        return node

    def _connect_node(self, node_id: str) -> None:
        """Connect a node to local and long-range neighbors."""
        node = self.nodes[node_id]
        other_ids = [nid for nid in self.nodes.keys() if nid != node_id]

        if not other_ids:
            return

        # Compute similarities to all other nodes
        similarities = []
        for other_id in other_ids:
            other = self.nodes[other_id]
            sim = cosine_similarity(node.vector, other.vector)
            similarities.append((other_id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Add k_local nearest neighbors
        local_count = min(self.k_local, len(similarities))
        for i in range(local_count):
            neighbor_id = similarities[i][0]
            self._add_bidirectional_edge(node_id, neighbor_id)

        # Add k_long random long-range connections
        # Prefer distant nodes (Kleinberg-style)
        remaining = [s for s in similarities[local_count:]]
        if remaining:
            # Weight by distance (prefer more distant)
            weights = [(1.0 - sim) for _, sim in remaining]
            total_weight = sum(weights)
            if total_weight > 0:
                probs = [w / total_weight for w in weights]

                long_count = min(self.k_long, len(remaining))
                try:
                    chosen_indices = np.random.choice(
                        len(remaining),
                        size=long_count,
                        replace=False,
                        p=probs,
                    )
                    for idx in chosen_indices:
                        neighbor_id = remaining[idx][0]
                        self._add_bidirectional_edge(node_id, neighbor_id)
                except ValueError:
                    # Fallback: random selection
                    for neighbor_id, _ in random.sample(remaining, min(long_count, len(remaining))):
                        self._add_bidirectional_edge(node_id, neighbor_id)

    def _add_bidirectional_edge(self, node_id1: str, node_id2: str) -> bool:
        """Add bidirectional edge between two nodes with angle tracking."""
        if node_id1 not in self.nodes or node_id2 not in self.nodes:
            return False

        n1 = self.nodes[node_id1]
        n2 = self.nodes[node_id2]

        # Pass vectors for angle computation and sorted insertion
        added1 = n1.add_neighbor(node_id2, neighbor_vector=n2.vector)
        added2 = n2.add_neighbor(node_id1, neighbor_vector=n1.vector)

        return added1 or added2

    def _remove_bidirectional_edge(self, node_id1: str, node_id2: str) -> bool:
        """Remove bidirectional edge between two nodes."""
        if node_id1 not in self.nodes or node_id2 not in self.nodes:
            return False

        n1 = self.nodes[node_id1]
        n2 = self.nodes[node_id2]

        removed1 = n1.remove_neighbor(node_id2)
        removed2 = n2.remove_neighbor(node_id1)

        return removed1 or removed2

    # =========================================================================
    # REWIRING (Watts-Strogatz style evolution)
    # =========================================================================

    def rewire_random(self, rng: random.Random = None) -> int:
        """
        Randomly rewire edges to improve small-world properties.

        For each edge, with probability rewire_prob:
        - Remove the edge
        - Add a new random edge

        Returns number of edges rewired.
        """
        if rng is None:
            rng = random.Random()

        rewired = 0
        all_node_ids = list(self.nodes.keys())

        for node_id, node in list(self.nodes.items()):
            for neighbor_id in list(node.neighbors):
                if rng.random() < self.rewire_prob:
                    # Pick a new random neighbor
                    candidates = [
                        nid for nid in all_node_ids
                        if nid != node_id and nid not in node.neighbors
                    ]
                    if candidates:
                        new_neighbor = rng.choice(candidates)
                        self._remove_bidirectional_edge(node_id, neighbor_id)
                        self._add_bidirectional_edge(node_id, new_neighbor)
                        rewired += 1

        return rewired

    def optimize_connections(self, num_rounds: int = 10, seed: int = None) -> Dict:
        """
        Optimize connections for better routing.

        Combines:
        1. Random rewiring (Watts-Strogatz)
        2. Quality-based connection updates

        Returns statistics.
        """
        rng = random.Random(seed)

        initial_stats = self.get_statistics()

        total_rewired = 0
        for _ in range(num_rounds):
            total_rewired += self.rewire_random(rng)

        final_stats = self.get_statistics()

        return {
            "rounds": num_rounds,
            "total_rewired": total_rewired,
            "initial_avg_neighbors": initial_stats["avg_neighbors"],
            "final_avg_neighbors": final_stats["avg_neighbors"],
        }

    # =========================================================================
    # ROUTING
    # =========================================================================

    def route_greedy(
        self,
        query: np.ndarray,
        start_node_id: Optional[str] = None,
        max_hops: int = 50,
    ) -> Tuple[List[str], int]:
        """
        Route query using greedy forwarding.

        Returns (path, comparisons).
        """
        if not self.nodes:
            return [], 0

        # Start from random node if not specified
        if start_node_id is None:
            start_node_id = random.choice(list(self.nodes.keys()))

        if start_node_id not in self.nodes:
            return [], 0

        path = [start_node_id]
        current_id = start_node_id
        comparisons = 0
        visited = {start_node_id}

        current_dist = cosine_distance(query, self.nodes[current_id].vector)

        for _ in range(max_hops):
            node = self.nodes[current_id]

            # Find closest unvisited neighbor
            best_neighbor = None
            best_dist = current_dist

            for neighbor_id in node.neighbors:
                if neighbor_id in visited:
                    continue
                if neighbor_id not in self.nodes:
                    continue

                comparisons += 1
                neighbor = self.nodes[neighbor_id]
                dist = cosine_distance(query, neighbor.vector)

                if dist < best_dist:
                    best_dist = dist
                    best_neighbor = neighbor_id

            if best_neighbor is None:
                break  # No improvement possible

            current_id = best_neighbor
            current_dist = best_dist
            path.append(current_id)
            visited.add(current_id)
            node.queries_routed += 1

        return path, comparisons

    def route_with_backtrack(
        self,
        query: np.ndarray,
        start_node_id: Optional[str] = None,
        max_hops: int = 100,
    ) -> Tuple[List[str], int]:
        """
        Route with backtracking for better success rate.
        """
        if not self.nodes:
            return [], 0

        if start_node_id is None:
            start_node_id = random.choice(list(self.nodes.keys()))

        if start_node_id not in self.nodes:
            return [], 0

        comparisons = 0
        best_node_id = start_node_id
        best_dist = cosine_distance(query, self.nodes[start_node_id].vector)

        # Stack: (node_id, tried neighbors)
        stack: List[Tuple[str, Set[str]]] = [(start_node_id, set())]
        visited: Set[str] = {start_node_id}
        path: List[str] = [start_node_id]

        total_visits = 0

        while stack and total_visits < max_hops:
            total_visits += 1
            current_id, tried = stack[-1]
            current_node = self.nodes[current_id]
            current_dist = cosine_distance(query, current_node.vector)

            if current_dist < best_dist:
                best_dist = current_dist
                best_node_id = current_id

            # Get untried neighbors
            untried = [
                nid for nid in current_node.neighbors
                if nid not in tried and nid in self.nodes
            ]

            if not untried:
                stack.pop()
                if path:
                    path.pop()
                continue

            # Find best untried neighbor
            best_neighbor = None
            best_neighbor_dist = float('inf')

            for neighbor_id in untried:
                comparisons += 1
                neighbor = self.nodes[neighbor_id]
                dist = cosine_distance(query, neighbor.vector)

                if dist < best_neighbor_dist:
                    best_neighbor_dist = dist
                    best_neighbor = neighbor_id

            if best_neighbor is None:
                stack.pop()
                if path:
                    path.pop()
                continue

            tried.add(best_neighbor)

            if best_neighbor not in visited:
                visited.add(best_neighbor)
                stack.append((best_neighbor, set()))
                path.append(best_neighbor)

        # Return path to best node
        if best_node_id in path:
            final_path = path[:path.index(best_node_id) + 1]
        else:
            final_path = [best_node_id]

        return final_path, comparisons

    def search_knn(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = 50,
        start_node_id: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, float]], int]:
        """
        Search for k nearest neighbors using beam search.

        Args:
            query: Query vector
            k: Number of neighbors to return
            ef: Beam width (explore this many candidates)
            start_node_id: Starting node (random if None)

        Returns:
            List of (node_id, distance) tuples, total comparisons
        """
        if not self.nodes:
            return [], 0

        if start_node_id is None:
            start_node_id = random.choice(list(self.nodes.keys()))

        comparisons = 0
        visited = {start_node_id}

        # Priority queue: (distance, node_id)
        start_dist = cosine_distance(query, self.nodes[start_node_id].vector)
        candidates = [(start_dist, start_node_id)]
        results = [(start_dist, start_node_id)]

        while candidates:
            candidates.sort()
            current_dist, current_id = candidates.pop(0)

            # Check if we can improve
            results.sort()
            if len(results) >= ef and current_dist > results[-1][0]:
                break

            # Explore neighbors
            current_node = self.nodes[current_id]
            for neighbor_id in current_node.neighbors:
                if neighbor_id in visited:
                    continue
                if neighbor_id not in self.nodes:
                    continue

                visited.add(neighbor_id)
                comparisons += 1
                neighbor = self.nodes[neighbor_id]
                dist = cosine_distance(query, neighbor.vector)

                if len(results) < ef or dist < results[-1][0]:
                    results.append((dist, neighbor_id))
                    candidates.append((dist, neighbor_id))

                    if len(results) > ef:
                        results.sort()
                        results = results[:ef]

        results.sort()
        return [(nid, dist) for dist, nid in results[:k]], comparisons

    def search_knn_optimized(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = 50,
        start_node_id: Optional[str] = None,
        window_size: int = 5,
    ) -> Tuple[List[Tuple[str, float]], int]:
        """
        Search for k nearest neighbors using angle-optimized lookup.

        Instead of checking all neighbors at each hop, uses binary search
        on angle-sorted neighbors to find candidates in the query direction.

        Args:
            query: Query vector
            k: Number of neighbors to return
            ef: Beam width
            start_node_id: Starting node (random if None)
            window_size: Neighbors to check around query angle

        Returns:
            List of (node_id, distance) tuples, total comparisons
        """
        if not self.nodes:
            return [], 0

        if start_node_id is None:
            start_node_id = random.choice(list(self.nodes.keys()))

        comparisons = 0
        visited = {start_node_id}

        start_dist = cosine_distance(query, self.nodes[start_node_id].vector)
        candidates = [(start_dist, start_node_id)]
        results = [(start_dist, start_node_id)]

        while candidates:
            candidates.sort()
            current_dist, current_id = candidates.pop(0)

            results.sort()
            if len(results) >= ef and current_dist > results[-1][0]:
                break

            current_node = self.nodes[current_id]

            # Use angle-optimized lookup if sorted neighbors available
            if current_node.sorted_neighbors:
                neighbor_ids = current_node.lookup_neighbors_by_angle(
                    query, window_size=window_size
                )
            else:
                neighbor_ids = list(current_node.neighbors)

            for neighbor_id in neighbor_ids:
                if neighbor_id in visited:
                    continue
                if neighbor_id not in self.nodes:
                    continue

                visited.add(neighbor_id)
                comparisons += 1
                neighbor = self.nodes[neighbor_id]
                dist = cosine_distance(query, neighbor.vector)

                if len(results) < ef or dist < results[-1][0]:
                    results.append((dist, neighbor_id))
                    candidates.append((dist, neighbor_id))

                    if len(results) > ef:
                        results.sort()
                        results = results[:ef]

        results.sort()
        return [(nid, dist) for dist, nid in results[:k]], comparisons

    def rebuild_all_sorted_neighbors(self) -> int:
        """
        Rebuild sorted neighbor lists for all nodes.

        Call periodically to correct numeric drift or after bulk updates.

        Returns:
            Number of nodes rebuilt
        """
        def get_vector(node_id):
            return self.nodes[node_id].vector

        count = 0
        for node in self.nodes.values():
            node.rebuild_sorted_neighbors(get_vector)
            count += 1

        return count

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict:
        """Get network statistics."""
        if not self.nodes:
            return {"num_nodes": 0}

        neighbor_counts = [len(n.neighbors) for n in self.nodes.values()]

        # Compute clustering coefficient
        clustering = self._compute_clustering_coefficient()

        # Estimate average path length (sample)
        avg_path = self._estimate_avg_path_length()

        return {
            "num_nodes": len(self.nodes),
            "total_edges": sum(neighbor_counts) // 2,
            "min_neighbors": min(neighbor_counts),
            "max_neighbors": max(neighbor_counts),
            "avg_neighbors": sum(neighbor_counts) / len(neighbor_counts),
            "clustering_coefficient": clustering,
            "estimated_avg_path_length": avg_path,
        }

    def _compute_clustering_coefficient(self) -> float:
        """Compute average clustering coefficient."""
        if len(self.nodes) < 3:
            return 0.0

        coefficients = []
        for node_id, node in self.nodes.items():
            neighbors = list(node.neighbors)
            if len(neighbors) < 2:
                continue

            # Count edges between neighbors
            edges = 0
            possible = len(neighbors) * (len(neighbors) - 1) / 2

            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if n1 in self.nodes and n2 in self.nodes[n1].neighbors:
                        edges += 1

            if possible > 0:
                coefficients.append(edges / possible)

        return sum(coefficients) / len(coefficients) if coefficients else 0.0

    def _estimate_avg_path_length(self, samples: int = 50) -> float:
        """Estimate average shortest path length via sampling."""
        if len(self.nodes) < 2:
            return 0.0

        node_ids = list(self.nodes.keys())
        total_length = 0
        valid_pairs = 0

        for _ in range(samples):
            start = random.choice(node_ids)
            end = random.choice(node_ids)
            if start == end:
                continue

            # BFS for shortest path
            path_len = self._bfs_path_length(start, end)
            if path_len is not None:
                total_length += path_len
                valid_pairs += 1

        return total_length / valid_pairs if valid_pairs > 0 else float('inf')

    def _bfs_path_length(self, start: str, end: str) -> Optional[int]:
        """Find shortest path length via BFS."""
        if start == end:
            return 0

        visited = {start}
        queue = [(start, 0)]

        while queue:
            current, depth = queue.pop(0)

            for neighbor in self.nodes[current].neighbors:
                if neighbor == end:
                    return depth + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return None  # No path found


def build_small_world(
    vectors: List[Tuple[str, np.ndarray]],
    k_local: int = 10,
    k_long: int = 5,
    seed: int = 42,
) -> SmallWorldProper:
    """
    Build a small-world network from vectors.

    Args:
        vectors: List of (id, vector) tuples
        k_local: Local neighbors per node
        k_long: Long-range connections per node
        seed: Random seed

    Returns:
        Built small-world network
    """
    np.random.seed(seed)
    random.seed(seed)

    network = SmallWorldProper(k_local=k_local, k_long=k_long)

    for node_id, vector in vectors:
        network.add_node(node_id, vector, connect=True)

    return network
