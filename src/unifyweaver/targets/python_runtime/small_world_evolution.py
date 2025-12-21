# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Small-World Network Evolution.

Starts with a hierarchical network (from subdivision) and evolves toward
small-world topology through random link exchanges.

Key concepts:
1. Initial hierarchy: Master node partitions queries
2. Link exchange: Nodes swap connections to improve distance distribution
3. Mature network: Queries can start from any node with O(log n) routing

The goal is Kleinberg-optimal distribution where:
- P(link to node at distance d) ∝ d^(-α)
- For 1D: α=1, For 2D: α=2, etc.

See: Kleinberg (2000) "The Small-World Phenomenon: An Algorithmic Perspective"
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
    similarity = np.dot(a, b) / (norm_a * norm_b)
    return 1.0 - similarity


@dataclass
class SmallWorldNode:
    """A node in the evolving small-world network."""

    node_id: str
    centroid: np.ndarray

    # Connections
    parent_id: Optional[str] = None          # From hierarchy
    children_ids: List[str] = field(default_factory=list)  # From hierarchy
    shortcut_ids: Set[str] = field(default_factory=set)    # From link exchange

    # Configuration
    max_shortcuts: int = 10  # Limit shortcut connections
    target_alpha: float = 2.0  # Kleinberg exponent for distance distribution

    # Statistics
    queries_routed: int = 0
    successful_routes: int = 0

    def get_all_neighbors(self) -> Set[str]:
        """Get all connected node IDs."""
        neighbors = set(self.shortcut_ids)
        if self.parent_id:
            neighbors.add(self.parent_id)
        neighbors.update(self.children_ids)
        return neighbors

    def add_shortcut(self, node_id: str) -> bool:
        """Add a shortcut connection."""
        if node_id == self.node_id:
            return False
        if len(self.shortcut_ids) >= self.max_shortcuts:
            return False
        self.shortcut_ids.add(node_id)
        return True

    def remove_shortcut(self, node_id: str) -> bool:
        """Remove a shortcut connection."""
        if node_id in self.shortcut_ids:
            self.shortcut_ids.discard(node_id)
            return True
        return False


class SmallWorldNetwork:
    """
    Evolving small-world network.

    Starts hierarchical, evolves through link exchange toward
    Kleinberg-optimal small-world topology.
    """

    def __init__(
        self,
        target_alpha: float = 2.0,
        max_shortcuts_per_node: int = 10,
        exchange_probability: float = 0.1,
    ):
        """
        Initialize network.

        Args:
            target_alpha: Kleinberg exponent (2.0 for 2D embeddings)
            max_shortcuts_per_node: Maximum shortcut connections per node
            exchange_probability: Probability of attempting exchange per round
        """
        self.nodes: Dict[str, SmallWorldNode] = {}
        self.root_id: Optional[str] = None  # Master node for hierarchical queries
        self.target_alpha = target_alpha
        self.max_shortcuts = max_shortcuts_per_node
        self.exchange_probability = exchange_probability

        # Evolution tracking
        self.exchange_rounds = 0
        self.successful_exchanges = 0
        self.maturity_score = 0.0  # 0 = hierarchical, 1 = fully small-world

    # =========================================================================
    # NETWORK CONSTRUCTION
    # =========================================================================

    def add_node(
        self,
        node_id: str,
        centroid: np.ndarray,
        parent_id: Optional[str] = None,
        children_ids: List[str] = None,
    ) -> SmallWorldNode:
        """Add a node to the network."""
        node = SmallWorldNode(
            node_id=node_id,
            centroid=centroid,
            parent_id=parent_id,
            children_ids=children_ids or [],
            max_shortcuts=self.max_shortcuts,
            target_alpha=self.target_alpha,
        )
        self.nodes[node_id] = node

        # Set root if this is the first node or has no parent
        if self.root_id is None or parent_id is None:
            if self.root_id is None:
                self.root_id = node_id

        return node

    def import_from_hierarchy(self, hierarchy_data: Dict) -> None:
        """
        Import nodes from subdivision hierarchy.

        Args:
            hierarchy_data: Output from SubdivisionRegistry.to_node_hierarchy_data()
        """
        # Import regions as internal nodes
        for region in hierarchy_data.get("regions", []):
            self.add_node(
                node_id=region["region_id"],
                centroid=np.array(region["centroid"]) if region.get("centroid") is not None else np.zeros(1),
                parent_id=region.get("parent_region"),
                children_ids=region.get("child_nodes", []),
            )
            if region.get("parent_region") is None:
                self.root_id = region["region_id"]

        # Import leaf nodes
        for leaf in hierarchy_data.get("leaf_nodes", []):
            parent = hierarchy_data.get("node_to_region", {}).get(leaf.node_id)
            self.add_node(
                node_id=leaf.node_id,
                centroid=leaf.centroid if leaf.centroid is not None else np.zeros(1),
                parent_id=parent,
            )

    # =========================================================================
    # LINK EXCHANGE (Evolution toward small-world)
    # =========================================================================

    def compute_distance_distribution(self, node: SmallWorldNode) -> Dict[str, float]:
        """
        Compute distance distribution for a node's connections.

        Returns mapping of neighbor_id -> distance.
        """
        distances = {}
        for neighbor_id in node.get_all_neighbors():
            if neighbor_id in self.nodes:
                neighbor = self.nodes[neighbor_id]
                dist = cosine_distance(node.centroid, neighbor.centroid)
                distances[neighbor_id] = dist
        return distances

    def evaluate_distribution_quality(self, distances: Dict[str, float]) -> float:
        """
        Evaluate how close the distance distribution is to Kleinberg-optimal.

        Kleinberg-optimal: P(d) ∝ d^(-α), which means log-uniform spacing.

        Returns: Quality score (higher = better, max 1.0)
        """
        if len(distances) < 2:
            return 0.0

        # Sort distances
        sorted_dists = sorted(distances.values())

        # For Kleinberg-optimal, log(distances) should be roughly uniform
        log_dists = [math.log(max(d, 1e-6)) for d in sorted_dists]

        # Compute variance of gaps between consecutive log distances
        gaps = [log_dists[i+1] - log_dists[i] for i in range(len(log_dists)-1)]
        if not gaps:
            return 0.0

        mean_gap = sum(gaps) / len(gaps)
        variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)

        # Lower variance = more uniform = better
        # Normalize: score = 1 / (1 + variance)
        return 1.0 / (1.0 + variance)

    def attempt_link_exchange(
        self,
        node_id: str,
        rng: random.Random = None
    ) -> bool:
        """
        Attempt to improve a node's distance distribution via link exchange.

        Algorithm:
        1. Pick a random neighbor
        2. Get one of neighbor's connections
        3. If connecting to that node improves our distribution, add shortcut
        4. Optionally drop a redundant connection

        Returns: True if exchange improved distribution
        """
        if rng is None:
            rng = random.Random()

        node = self.nodes.get(node_id)
        if not node:
            return False

        neighbors = list(node.get_all_neighbors())
        if not neighbors:
            return False

        # Pick random neighbor
        neighbor_id = rng.choice(neighbors)
        neighbor = self.nodes.get(neighbor_id)
        if not neighbor:
            return False

        # Get neighbor's connections (potential new links)
        neighbor_connections = list(neighbor.get_all_neighbors())
        candidate_ids = [
            cid for cid in neighbor_connections
            if cid != node_id and cid not in node.get_all_neighbors()
        ]

        if not candidate_ids:
            return False

        # Pick a candidate
        candidate_id = rng.choice(candidate_ids)

        # Evaluate current distribution
        current_dists = self.compute_distance_distribution(node)
        current_quality = self.evaluate_distribution_quality(current_dists)

        # Simulate adding the candidate
        candidate = self.nodes.get(candidate_id)
        if not candidate:
            return False

        new_dist = cosine_distance(node.centroid, candidate.centroid)
        test_dists = dict(current_dists)
        test_dists[candidate_id] = new_dist

        new_quality = self.evaluate_distribution_quality(test_dists)

        # Accept if quality improves
        if new_quality > current_quality:
            # Check if we need to drop a connection
            if len(node.shortcut_ids) >= node.max_shortcuts:
                # Find most redundant shortcut (closest distance to another)
                redundant = self._find_redundant_shortcut(node, candidate_id)
                if redundant:
                    node.remove_shortcut(redundant)

            if node.add_shortcut(candidate_id):
                self.successful_exchanges += 1
                return True

        return False

    def _find_redundant_shortcut(
        self,
        node: SmallWorldNode,
        new_candidate_id: str
    ) -> Optional[str]:
        """Find the most redundant shortcut to remove."""
        if not node.shortcut_ids:
            return None

        # Compute distances
        distances = {}
        for sid in node.shortcut_ids:
            if sid in self.nodes:
                distances[sid] = cosine_distance(
                    node.centroid, self.nodes[sid].centroid
                )

        if not distances:
            return None

        # Find shortcut with closest neighbor (most redundant)
        sorted_shortcuts = sorted(distances.items(), key=lambda x: x[1])

        # Return the one that's most "covered" by others
        for sid, dist in sorted_shortcuts:
            # Check if another shortcut is close
            for other_sid, other_dist in distances.items():
                if other_sid != sid:
                    if abs(dist - other_dist) < 0.1:  # Close in distance
                        return sid

        # Default: remove shortest distance shortcut
        return sorted_shortcuts[0][0] if sorted_shortcuts else None

    def evolution_round(self, rng: random.Random = None) -> int:
        """
        Run one round of evolution across all nodes.

        Returns: Number of successful exchanges
        """
        if rng is None:
            rng = random.Random()

        self.exchange_rounds += 1
        exchanges = 0

        # Shuffle nodes for random order
        node_ids = list(self.nodes.keys())
        rng.shuffle(node_ids)

        for node_id in node_ids:
            if rng.random() < self.exchange_probability:
                if self.attempt_link_exchange(node_id, rng):
                    exchanges += 1

        # Update maturity score
        self._update_maturity_score()

        return exchanges

    def evolve(
        self,
        num_rounds: int = 100,
        seed: int = None
    ) -> Dict[str, float]:
        """
        Run multiple evolution rounds.

        Returns: Statistics about the evolution
        """
        rng = random.Random(seed)

        initial_quality = self._compute_network_quality()

        for _ in range(num_rounds):
            self.evolution_round(rng)

        final_quality = self._compute_network_quality()

        return {
            "rounds": num_rounds,
            "successful_exchanges": self.successful_exchanges,
            "initial_quality": initial_quality,
            "final_quality": final_quality,
            "maturity_score": self.maturity_score,
        }

    def _compute_network_quality(self) -> float:
        """Compute average distribution quality across all nodes."""
        qualities = []
        for node in self.nodes.values():
            dists = self.compute_distance_distribution(node)
            if dists:
                qualities.append(self.evaluate_distribution_quality(dists))
        return sum(qualities) / len(qualities) if qualities else 0.0

    def _update_maturity_score(self) -> None:
        """
        Update maturity score based on shortcut density and quality.

        Maturity ranges from 0 (pure hierarchy) to 1 (fully small-world).
        """
        if not self.nodes:
            self.maturity_score = 0.0
            return

        # Factor 1: Shortcut density
        total_shortcuts = sum(len(n.shortcut_ids) for n in self.nodes.values())
        max_shortcuts = len(self.nodes) * self.max_shortcuts
        shortcut_density = total_shortcuts / max(1, max_shortcuts)

        # Factor 2: Distribution quality
        quality = self._compute_network_quality()

        # Combine factors
        self.maturity_score = 0.5 * shortcut_density + 0.5 * quality

    # =========================================================================
    # CROSS-BRANCH LINK DISCOVERY (Semantic similarity across subtrees)
    # =========================================================================

    def find_cross_branch_candidates(
        self,
        node_id: str,
        similarity_threshold: float = 0.7,
        max_candidates: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find semantically similar nodes in OTHER branches.

        This enables "super-category" style links that connect
        related content across different parts of the hierarchy.

        Args:
            node_id: Node to find cross-branch links for
            similarity_threshold: Minimum similarity to consider
            max_candidates: Maximum candidates to return

        Returns:
            List of (candidate_node_id, similarity) tuples
        """
        node = self.nodes.get(node_id)
        if not node:
            return []

        # Find node's top-level branch
        my_branch = self._get_branch_id(node_id)

        # Exclude all nodes in the same branch (siblings, ancestors, descendants)
        same_branch = set()
        for other_id in self.nodes.keys():
            other_branch = self._get_branch_id(other_id)
            if other_branch == my_branch or other_id == self.root_id:
                same_branch.add(other_id)

        # Find similar nodes in other branches
        candidates = []
        for other_id, other_node in self.nodes.items():
            if other_id in same_branch:
                continue
            if other_id in node.get_all_neighbors():
                continue  # Already connected

            # Compute similarity
            similarity = 1.0 - cosine_distance(node.centroid, other_node.centroid)
            if similarity >= similarity_threshold:
                candidates.append((other_id, similarity))

        # Sort by similarity, return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]

    def _get_ancestors(self, node_id: str) -> Set[str]:
        """Get all ancestor node IDs."""
        ancestors = set()
        current = self.nodes.get(node_id)
        while current and current.parent_id:
            ancestors.add(current.parent_id)
            current = self.nodes.get(current.parent_id)
        return ancestors

    def _get_branch_id(self, node_id: str) -> Optional[str]:
        """Get the top-level branch (child of root) this node belongs to."""
        current = self.nodes.get(node_id)
        prev_id = node_id

        while current and current.parent_id:
            if current.parent_id == self.root_id:
                return current.node_id
            prev_id = current.node_id
            current = self.nodes.get(current.parent_id)

        return prev_id if prev_id != self.root_id else None

    def discover_cross_branch_links(
        self,
        similarity_threshold: float = 0.6,
        max_links_per_node: int = 3,
    ) -> int:
        """
        Discover and create cross-branch shortcuts based on semantic similarity.

        This implements "super-category" style links - connecting related
        content that happens to be in different branches of the hierarchy.

        Returns:
            Number of new shortcuts created
        """
        new_links = 0

        for node_id in list(self.nodes.keys()):
            node = self.nodes[node_id]

            # Skip if already has many shortcuts
            if len(node.shortcut_ids) >= node.max_shortcuts:
                continue

            # Find cross-branch candidates
            candidates = self.find_cross_branch_candidates(
                node_id,
                similarity_threshold=similarity_threshold,
                max_candidates=max_links_per_node,
            )

            for candidate_id, similarity in candidates:
                if node.add_shortcut(candidate_id):
                    new_links += 1

                    # Create bidirectional link for better routing
                    candidate = self.nodes.get(candidate_id)
                    if candidate:
                        candidate.add_shortcut(node_id)

        return new_links

    # =========================================================================
    # PATH FOLDING (Freenet-style query-driven shortcuts)
    # =========================================================================

    def record_successful_path(
        self,
        path: List[str],
        query_embedding: np.ndarray,
    ) -> int:
        """
        Create shortcuts from successful query path (Freenet path folding).

        When query successfully routes: A → B → C → target
        Create shortcuts: A → target, B → target (skip intermediates)

        Args:
            path: List of node IDs from query start to result
            query_embedding: The query that was successful

        Returns:
            Number of shortcuts created
        """
        if len(path) < 3:
            return 0  # No shortcuts needed for short paths

        shortcuts_created = 0
        target_id = path[-1]
        target = self.nodes.get(target_id)

        if not target:
            return 0

        # Create shortcuts from earlier nodes to target
        for i, node_id in enumerate(path[:-2]):  # Exclude last two (already connected)
            node = self.nodes.get(node_id)
            if not node:
                continue

            # Only create shortcut if it would save multiple hops
            hops_saved = len(path) - i - 2
            if hops_saved >= 2:  # Worth creating shortcut
                if node.add_shortcut(target_id):
                    shortcuts_created += 1

        return shortcuts_created

    # =========================================================================
    # ROUTING (with backtracking)
    # =========================================================================

    def route_greedy(
        self,
        query_embedding: np.ndarray,
        start_node_id: Optional[str] = None,
        max_hops: int = 20,
    ) -> Tuple[List[str], int]:
        """
        Route query using greedy forwarding.

        Args:
            query_embedding: Query vector
            start_node_id: Starting node (uses root if None and immature)
            max_hops: Maximum routing hops

        Returns:
            (path of node IDs, number of comparisons made)
        """
        # Choose starting point based on maturity
        if start_node_id is None:
            if self.maturity_score < 0.5:
                # Immature: start from root (hierarchical)
                start_node_id = self.root_id
            else:
                # Mature: can start from random node
                start_node_id = random.choice(list(self.nodes.keys()))

        if start_node_id is None or start_node_id not in self.nodes:
            return [], 0

        path = [start_node_id]
        current_id = start_node_id
        comparisons = 0
        visited = {start_node_id}

        current_node = self.nodes[current_id]
        current_dist = cosine_distance(query_embedding, current_node.centroid)

        for _ in range(max_hops):
            node = self.nodes[current_id]
            neighbors = node.get_all_neighbors()

            # Find closest neighbor to query
            best_neighbor = None
            best_dist = current_dist

            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue
                if neighbor_id not in self.nodes:
                    continue

                neighbor = self.nodes[neighbor_id]
                dist = cosine_distance(query_embedding, neighbor.centroid)
                comparisons += 1

                if dist < best_dist:
                    best_dist = dist
                    best_neighbor = neighbor_id

            if best_neighbor is None:
                # No progress possible
                break

            # Move to best neighbor
            current_id = best_neighbor
            current_dist = best_dist
            path.append(current_id)
            visited.add(current_id)

            # Update statistics
            node.queries_routed += 1

        # Mark final node as successful
        if path:
            self.nodes[path[-1]].successful_routes += 1

        return path, comparisons

    def route_to_top_k(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        start_node_id: Optional[str] = None,
        num_paths: int = 1,
    ) -> Tuple[List[Tuple[str, float]], int]:
        """
        Route to top-k closest nodes.

        Uses multiple greedy paths for better coverage.

        Returns:
            List of (node_id, distance) tuples, total comparisons
        """
        all_candidates = {}
        total_comparisons = 0

        # Run multiple paths from different starts
        starts = [start_node_id] if start_node_id else []
        if len(starts) < num_paths and self.maturity_score >= 0.5:
            # Add random starts for mature network
            available = [nid for nid in self.nodes.keys() if nid not in starts]
            random.shuffle(available)
            starts.extend(available[:num_paths - len(starts)])

        if not starts:
            starts = [self.root_id]

        for start in starts:
            if start is None:
                continue
            path, comps = self.route_greedy(query_embedding, start)
            total_comparisons += comps

            # Score all nodes in path
            for node_id in path:
                if node_id in self.nodes:
                    dist = cosine_distance(query_embedding, self.nodes[node_id].centroid)
                    if node_id not in all_candidates or dist < all_candidates[node_id]:
                        all_candidates[node_id] = dist

        # Sort by distance and return top-k
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1])
        return sorted_candidates[:k], total_comparisons

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict:
        """Get network statistics."""
        if not self.nodes:
            return {"num_nodes": 0}

        shortcut_counts = [len(n.shortcut_ids) for n in self.nodes.values()]
        neighbor_counts = [len(n.get_all_neighbors()) for n in self.nodes.values()]

        return {
            "num_nodes": len(self.nodes),
            "root_id": self.root_id,
            "exchange_rounds": self.exchange_rounds,
            "successful_exchanges": self.successful_exchanges,
            "maturity_score": self.maturity_score,
            "avg_shortcuts": sum(shortcut_counts) / len(shortcut_counts),
            "avg_neighbors": sum(neighbor_counts) / len(neighbor_counts),
            "max_shortcuts": max(shortcut_counts),
            "network_quality": self._compute_network_quality(),
        }
