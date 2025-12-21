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
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import heapq

import numpy as np


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
    """A node in the small-world network."""

    node_id: str
    vector: np.ndarray

    # All connections (no hierarchy distinction)
    neighbors: Set[str] = field(default_factory=set)

    # Configuration
    max_neighbors: int = 20

    # Statistics
    queries_routed: int = 0

    def add_neighbor(self, neighbor_id: str) -> bool:
        """Add a neighbor connection."""
        if neighbor_id == self.node_id:
            return False
        if len(self.neighbors) >= self.max_neighbors:
            return False
        self.neighbors.add(neighbor_id)
        return True

    def remove_neighbor(self, neighbor_id: str) -> bool:
        """Remove a neighbor connection."""
        if neighbor_id in self.neighbors:
            self.neighbors.discard(neighbor_id)
            return True
        return False


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
        max_neighbors: int = 20,
        rewire_prob: float = 0.1,  # Probability of rewiring
    ):
        """
        Initialize small-world network.

        Args:
            k_local: Number of local (nearest) neighbors
            k_long: Number of long-range (random) connections
            max_neighbors: Maximum total neighbors per node
            rewire_prob: Probability of rewiring during evolution
        """
        self.nodes: Dict[str, SWNode] = {}
        self.k_local = k_local
        self.k_long = k_long
        self.max_neighbors = max_neighbors
        self.rewire_prob = rewire_prob

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
        """Add bidirectional edge between two nodes."""
        if node_id1 not in self.nodes or node_id2 not in self.nodes:
            return False

        n1 = self.nodes[node_id1]
        n2 = self.nodes[node_id2]

        added1 = n1.add_neighbor(node_id2)
        added2 = n2.add_neighbor(node_id1)

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
