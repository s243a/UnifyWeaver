# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
HNSW-Style Layered Small-World Network.

Combines hierarchical layers with small-world connections for O(log n) routing.

Key concepts:
1. Nodes exist on multiple layers (layer L → also on layers 0..L-1)
2. Upper layers are sparse (long-range jumps)
3. Lower layers are dense (fine-grained search)
4. Routing: greedy search at each layer, descend to next layer

For P2P/distributed:
- Start at any layer (not just top)
- More entry points at lower layers
- Backtracking enables success from any start point

See: Malkov & Yashunin (2018) "Efficient and robust approximate nearest neighbor search"
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - np.dot(a, b) / (norm_a * norm_b)


@dataclass
class HNSWNode:
    """A node in the HNSW graph."""

    node_id: str
    vector: np.ndarray

    # Layer assignment (node exists on layers 0..max_layer)
    max_layer: int = 0

    # Neighbors per layer: layer -> set of neighbor IDs
    neighbors: Dict[int, Set[str]] = field(default_factory=lambda: defaultdict(set))

    # Max neighbors per layer
    max_neighbors: int = 16  # M parameter in HNSW
    max_neighbors_layer0: int = 32  # M0 parameter (layer 0 can have more)

    def get_neighbors_at_layer(self, layer: int) -> Set[str]:
        """Get neighbors at a specific layer."""
        return self.neighbors.get(layer, set())

    def add_neighbor(self, neighbor_id: str, layer: int) -> bool:
        """Add a neighbor at a specific layer."""
        if neighbor_id == self.node_id:
            return False

        max_n = self.max_neighbors_layer0 if layer == 0 else self.max_neighbors
        if len(self.neighbors[layer]) >= max_n:
            return False

        self.neighbors[layer].add(neighbor_id)
        return True

    def remove_neighbor(self, neighbor_id: str, layer: int) -> bool:
        """Remove a neighbor at a specific layer."""
        if neighbor_id in self.neighbors[layer]:
            self.neighbors[layer].discard(neighbor_id)
            return True
        return False


class HNSWGraph:
    """
    HNSW-style layered graph for approximate nearest neighbor search.

    Supports both centralized (start at top) and distributed (start anywhere)
    routing modes.
    """

    def __init__(
        self,
        max_neighbors: int = 16,        # M parameter
        max_neighbors_layer0: int = 32,  # M0 parameter
        level_multiplier: float = 1.0 / math.log(2),  # mL parameter
        ef_construction: int = 100,      # Search width during construction
    ):
        """
        Initialize HNSW graph.

        Args:
            max_neighbors: Max neighbors per node per layer (M)
            max_neighbors_layer0: Max neighbors at layer 0 (M0)
            level_multiplier: Controls layer distribution (mL)
            ef_construction: Beam width during insertion
        """
        self.nodes: Dict[str, HNSWNode] = {}
        self.max_neighbors = max_neighbors
        self.max_neighbors_layer0 = max_neighbors_layer0
        self.level_multiplier = level_multiplier
        self.ef_construction = ef_construction

        # Entry point (node at highest layer)
        self.entry_point_id: Optional[str] = None
        self.max_layer: int = 0

        # Layer index: layer -> set of node IDs at that layer
        self.layer_index: Dict[int, Set[str]] = defaultdict(set)

    def _random_layer(self, rng: random.Random = None) -> int:
        """
        Assign a random layer to a new node.

        Uses exponential distribution: P(layer=L) ∝ exp(-L/mL)
        This ensures upper layers are exponentially sparser.
        """
        if rng is None:
            rng = random.Random()

        # Sample from geometric distribution
        r = rng.random()
        layer = int(-math.log(r) * self.level_multiplier)
        return layer

    def add_node(
        self,
        node_id: str,
        vector: np.ndarray,
        rng: random.Random = None,
    ) -> HNSWNode:
        """
        Add a node to the graph with HNSW insertion algorithm.

        1. Assign random layer L
        2. Start from entry point at top layer
        3. Greedy search down to layer L+1
        4. At each layer L..0, find and connect to nearest neighbors
        """
        if rng is None:
            rng = random.Random()

        # Assign layer
        node_layer = self._random_layer(rng)

        node = HNSWNode(
            node_id=node_id,
            vector=vector,
            max_layer=node_layer,
            max_neighbors=self.max_neighbors,
            max_neighbors_layer0=self.max_neighbors_layer0,
        )

        self.nodes[node_id] = node

        # Update layer index
        for layer in range(node_layer + 1):
            self.layer_index[layer].add(node_id)

        # First node becomes entry point
        if self.entry_point_id is None:
            self.entry_point_id = node_id
            self.max_layer = node_layer
            return node

        # Find entry point for insertion
        current_id = self.entry_point_id

        # Greedy descent from top to node's layer + 1
        for layer in range(self.max_layer, node_layer, -1):
            current_id = self._greedy_search_layer(
                vector, current_id, layer, k=1
            )[0][0]

        # For each layer from node_layer down to 0, find neighbors
        for layer in range(min(node_layer, self.max_layer), -1, -1):
            # Find k nearest at this layer
            candidates = self._search_layer(
                vector, current_id, layer, ef=self.ef_construction
            )

            # Select neighbors (simple: take closest)
            neighbors = self._select_neighbors(
                vector, candidates,
                self.max_neighbors_layer0 if layer == 0 else self.max_neighbors
            )

            # Add bidirectional edges
            for neighbor_id, _ in neighbors:
                node.add_neighbor(neighbor_id, layer)
                neighbor = self.nodes[neighbor_id]
                neighbor.add_neighbor(node_id, layer)

                # Prune if neighbor has too many connections
                self._prune_connections(neighbor, layer)

            # Use closest as entry for next layer
            if candidates:
                current_id = candidates[0][0]

        # Update entry point if new node is at higher layer
        if node_layer > self.max_layer:
            self.entry_point_id = node_id
            self.max_layer = node_layer

        return node

    def _greedy_search_layer(
        self,
        query: np.ndarray,
        entry_id: str,
        layer: int,
        k: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Greedy search at a single layer.

        Returns k nearest nodes found.
        """
        visited = {entry_id}
        candidates = [(entry_id, cosine_distance(query, self.nodes[entry_id].vector))]

        while True:
            # Get current best
            candidates.sort(key=lambda x: x[1])
            current_id, current_dist = candidates[0]

            # Check neighbors
            improved = False
            current_node = self.nodes[current_id]

            for neighbor_id in current_node.get_neighbors_at_layer(layer):
                if neighbor_id in visited:
                    continue
                if neighbor_id not in self.nodes:
                    continue

                visited.add(neighbor_id)
                neighbor = self.nodes[neighbor_id]
                dist = cosine_distance(query, neighbor.vector)

                if dist < current_dist:
                    candidates.append((neighbor_id, dist))
                    improved = True

            if not improved:
                break

        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

    def _search_layer(
        self,
        query: np.ndarray,
        entry_id: str,
        layer: int,
        ef: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Beam search at a single layer.

        Maintains ef candidates (beam width) for better recall.
        """
        visited = {entry_id}
        candidates = [(cosine_distance(query, self.nodes[entry_id].vector), entry_id)]
        results = list(candidates)

        while candidates:
            # Pop closest candidate
            candidates.sort()
            current_dist, current_id = candidates.pop(0)

            # Get furthest result
            results.sort()
            if results and current_dist > results[-1][0]:
                break  # No improvement possible

            # Explore neighbors
            current_node = self.nodes[current_id]
            for neighbor_id in current_node.get_neighbors_at_layer(layer):
                if neighbor_id in visited:
                    continue
                if neighbor_id not in self.nodes:
                    continue

                visited.add(neighbor_id)
                neighbor = self.nodes[neighbor_id]
                dist = cosine_distance(query, neighbor.vector)

                # Add to results if better than worst
                results.sort()
                if len(results) < ef or dist < results[-1][0]:
                    results.append((dist, neighbor_id))
                    candidates.append((dist, neighbor_id))

                    # Keep only ef best
                    if len(results) > ef:
                        results.sort()
                        results = results[:ef]

        # Return as (id, distance) pairs
        results.sort()
        return [(nid, dist) for dist, nid in results]

    def _select_neighbors(
        self,
        query: np.ndarray,
        candidates: List[Tuple[str, float]],
        max_neighbors: int,
    ) -> List[Tuple[str, float]]:
        """Select best neighbors from candidates."""
        # Simple heuristic: take closest
        candidates.sort(key=lambda x: x[1])
        return candidates[:max_neighbors]

    def _prune_connections(self, node: HNSWNode, layer: int) -> None:
        """Prune connections if node has too many neighbors."""
        max_n = self.max_neighbors_layer0 if layer == 0 else self.max_neighbors

        if len(node.neighbors[layer]) <= max_n:
            return

        # Keep closest neighbors
        neighbor_dists = []
        for neighbor_id in node.neighbors[layer]:
            if neighbor_id in self.nodes:
                dist = cosine_distance(node.vector, self.nodes[neighbor_id].vector)
                neighbor_dists.append((neighbor_id, dist))

        neighbor_dists.sort(key=lambda x: x[1])

        # Keep only max_n closest
        keep = set(nid for nid, _ in neighbor_dists[:max_n])
        node.neighbors[layer] = keep

    # =========================================================================
    # SEARCH (with distributed/P2P support)
    # =========================================================================

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = 50,
        start_layer: Optional[int] = None,
        start_node_id: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, float]], int]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vector
            k: Number of neighbors to return
            ef: Search beam width
            start_layer: Layer to start search (None = top layer)
            start_node_id: Node to start from (None = entry point)

        Returns:
            List of (node_id, distance) tuples, total comparisons
        """
        if not self.nodes:
            return [], 0

        comparisons = 0

        # Determine starting point
        if start_layer is None:
            start_layer = self.max_layer

        if start_node_id is None:
            # Find a node at the start layer
            if start_layer >= self.max_layer:
                start_node_id = self.entry_point_id
            else:
                # Pick any node at this layer
                layer_nodes = list(self.layer_index.get(start_layer, []))
                if layer_nodes:
                    start_node_id = layer_nodes[0]
                else:
                    start_node_id = self.entry_point_id

        if start_node_id is None or start_node_id not in self.nodes:
            return [], 0

        current_id = start_node_id

        # Greedy descent from start_layer to layer 1
        for layer in range(start_layer, 0, -1):
            result = self._greedy_search_layer(query, current_id, layer, k=1)
            comparisons += len(self.nodes[current_id].get_neighbors_at_layer(layer))
            if result:
                current_id = result[0][0]

        # Beam search at layer 0
        candidates = self._search_layer(query, current_id, layer=0, ef=ef)
        comparisons += ef * 2  # Approximate

        return candidates[:k], comparisons

    def search_from_any_node(
        self,
        query: np.ndarray,
        start_node_id: str,
        k: int = 10,
        ef: int = 50,
        use_backtrack: bool = True,
    ) -> Tuple[List[Tuple[str, float]], int]:
        """
        Search starting from any node (P2P mode).

        Uses the node's max_layer as the starting layer.
        """
        if start_node_id not in self.nodes:
            return [], 0

        node = self.nodes[start_node_id]

        if use_backtrack:
            return self._search_with_backtrack(
                query, start_node_id, node.max_layer, k, ef
            )
        else:
            return self.search(
                query, k, ef,
                start_layer=node.max_layer,
                start_node_id=start_node_id,
            )

    def _search_with_backtrack(
        self,
        query: np.ndarray,
        start_node_id: str,
        start_layer: int,
        k: int = 10,
        ef: int = 50,
        max_backtracks: int = 10,
    ) -> Tuple[List[Tuple[str, float]], int]:
        """
        Search with backtracking for P2P scenarios.

        If greedy search gets stuck, backtrack and try alternative paths.
        """
        comparisons = 0
        best_candidates = []

        # Try from start node
        candidates, comps = self.search(
            query, k, ef,
            start_layer=start_layer,
            start_node_id=start_node_id,
        )
        comparisons += comps
        best_candidates.extend(candidates)

        # If we didn't find k results, try from other nodes at this layer
        if len(best_candidates) < k:
            layer_nodes = list(self.layer_index.get(start_layer, []))
            random.shuffle(layer_nodes)

            for alt_node_id in layer_nodes[:max_backtracks]:
                if alt_node_id == start_node_id:
                    continue

                alt_candidates, comps = self.search(
                    query, k, ef,
                    start_layer=start_layer,
                    start_node_id=alt_node_id,
                )
                comparisons += comps
                best_candidates.extend(alt_candidates)

                if len(set(c[0] for c in best_candidates)) >= k:
                    break

        # Deduplicate and sort
        seen = set()
        unique = []
        for nid, dist in best_candidates:
            if nid not in seen:
                seen.add(nid)
                unique.append((nid, dist))

        unique.sort(key=lambda x: x[1])
        return unique[:k], comparisons

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        if not self.nodes:
            return {"num_nodes": 0}

        layer_counts = {layer: len(nodes) for layer, nodes in self.layer_index.items()}

        total_edges = sum(
            sum(len(node.neighbors[layer]) for layer in range(node.max_layer + 1))
            for node in self.nodes.values()
        )

        return {
            "num_nodes": len(self.nodes),
            "max_layer": self.max_layer,
            "entry_point": self.entry_point_id,
            "layer_distribution": layer_counts,
            "total_edges": total_edges // 2,  # Bidirectional
            "avg_edges_per_node": total_edges / len(self.nodes) if self.nodes else 0,
        }

    def get_entry_points_at_layer(self, layer: int) -> List[str]:
        """Get all possible entry points at a given layer."""
        return list(self.layer_index.get(layer, []))


def build_hnsw_index(
    vectors: List[Tuple[str, np.ndarray]],
    max_neighbors: int = 16,
    ef_construction: int = 100,
    seed: int = 42,
) -> HNSWGraph:
    """
    Build an HNSW index from a list of vectors.

    Args:
        vectors: List of (id, vector) tuples
        max_neighbors: M parameter
        ef_construction: Construction beam width
        seed: Random seed for reproducibility

    Returns:
        Built HNSW graph
    """
    rng = random.Random(seed)

    graph = HNSWGraph(
        max_neighbors=max_neighbors,
        ef_construction=ef_construction,
    )

    for node_id, vector in vectors:
        graph.add_node(node_id, vector, rng=rng)

    return graph
