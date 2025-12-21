# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Multi-Interface Node with Scale-Free Interface Distribution.

A physical node can expose multiple logical interfaces, each representing
a different semantic region of its data. This provides apparent scale
without proportional physical nodes.

Key concepts:
1. Each interface has 10-20 external connections
2. All interfaces share one binary search structure internally
3. Number of interfaces follows scale-free (power law) distribution

Scale-free benefits:
- Most nodes have 1-3 interfaces (leaves)
- Some nodes have 10-20 interfaces (regional hubs)
- Few nodes have 50-100+ interfaces (major hubs)
- Short paths via hub nodes, robust to random failures
"""

import math
import random
import bisect
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, NamedTuple
from collections import defaultdict

import numpy as np


def compute_angle_2d(direction: np.ndarray) -> float:
    """Compute angle from first two components of direction vector."""
    if len(direction) < 2:
        return 0.0
    return math.atan2(direction[1], direction[0])


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


class ExternalConnection(NamedTuple):
    """A connection from one interface to another node's interface."""
    angle: float              # Angle from source interface centroid
    source_interface: str     # Interface ID on this node
    target_node: str          # Target node ID
    target_interface: str     # Target interface ID on target node


@dataclass
class Interface:
    """A logical interface representing a semantic region."""

    interface_id: str
    centroid: np.ndarray

    # Documents/data covered by this interface
    document_count: int = 0

    # Statistics
    queries_handled: int = 0


@dataclass
class MultiInterfaceNode:
    """
    A node with multiple logical interfaces.

    Each interface represents a semantic region of the node's data.
    Interfaces share a unified binary search structure for efficient
    routing both internally (which interface?) and externally (which neighbor?).
    """

    node_id: str

    # Interfaces sorted by angle from node centroid
    interfaces: List[Tuple[float, Interface]] = field(default_factory=list)

    # All external connections, sorted by angle
    # This is ONE big sorted list across all interfaces
    connections: List[ExternalConnection] = field(default_factory=list)

    # Node-level centroid (average of all interface centroids)
    _centroid: Optional[np.ndarray] = field(default=None, repr=False)

    # Configuration
    max_connections_per_interface: int = 20

    @property
    def centroid(self) -> np.ndarray:
        """Get node-level centroid."""
        if self._centroid is not None:
            return self._centroid
        if not self.interfaces:
            return np.zeros(2)
        # Average of interface centroids
        centroids = [iface.centroid for _, iface in self.interfaces]
        return np.mean(centroids, axis=0)

    @property
    def num_interfaces(self) -> int:
        """Number of interfaces on this node."""
        return len(self.interfaces)

    @property
    def total_connections(self) -> int:
        """Total external connections across all interfaces."""
        return len(self.connections)

    def add_interface(self, interface: Interface) -> bool:
        """
        Add an interface to this node.

        Maintains sorted order by angle from node centroid.
        """
        # Compute angle from current centroid
        if self._centroid is not None:
            direction = interface.centroid - self._centroid
        else:
            direction = interface.centroid

        angle = compute_angle_2d(direction)

        # Insert in sorted order
        bisect.insort(self.interfaces, (angle, interface))

        # Update node centroid
        self._update_centroid()

        return True

    def _update_centroid(self) -> None:
        """Recompute node centroid from interface centroids."""
        if not self.interfaces:
            self._centroid = None
            return

        centroids = [iface.centroid for _, iface in self.interfaces]
        self._centroid = np.mean(centroids, axis=0)

    def add_connection(
        self,
        source_interface_id: str,
        target_node_id: str,
        target_interface_id: str,
        target_centroid: np.ndarray,
    ) -> bool:
        """
        Add an external connection from one of our interfaces.

        The connection is inserted into the unified sorted list.
        """
        # Find source interface
        source_iface = None
        for _, iface in self.interfaces:
            if iface.interface_id == source_interface_id:
                source_iface = iface
                break

        if source_iface is None:
            return False

        # Count existing connections for this interface
        iface_connections = sum(
            1 for c in self.connections
            if c.source_interface == source_interface_id
        )

        if iface_connections >= self.max_connections_per_interface:
            return False

        # Compute angle from source interface centroid
        direction = target_centroid - source_iface.centroid
        angle = compute_angle_2d(direction)

        conn = ExternalConnection(
            angle=angle,
            source_interface=source_interface_id,
            target_node=target_node_id,
            target_interface=target_interface_id,
        )

        # Insert in sorted order (by angle)
        bisect.insort(self.connections, conn)

        return True

    def find_closest_interfaces(
        self,
        query: np.ndarray,
        k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Find the k closest interfaces to a query vector.

        Uses binary search on angle-sorted interfaces, then refines
        by actual distance.

        Returns list of (interface_id, distance) tuples.
        """
        if not self.interfaces:
            return []

        # Compute query angle from node centroid
        direction = query - self.centroid
        query_angle = compute_angle_2d(direction)

        # Binary search for starting position
        idx = bisect.bisect_left(self.interfaces, (query_angle,))

        # Check a window around this position
        n = len(self.interfaces)
        window = min(k * 2, n)  # Check 2k interfaces

        candidates = []
        for offset in range(-window // 2, window // 2 + 1):
            i = (idx + offset) % n  # Wrap around
            _, iface = self.interfaces[i]
            dist = cosine_distance(query, iface.centroid)
            candidates.append((iface.interface_id, dist))

        # Sort by distance and return top k
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

    def find_connections_for_query(
        self,
        query: np.ndarray,
        interface_id: Optional[str] = None,
        window_size: int = 5,
    ) -> List[ExternalConnection]:
        """
        Find relevant external connections for a query.

        If interface_id is provided, only returns connections from that interface.
        Uses binary search on angle-sorted connections.
        """
        if not self.connections:
            return []

        # If interface specified, filter and search
        if interface_id:
            iface_connections = [
                c for c in self.connections
                if c.source_interface == interface_id
            ]
            if not iface_connections:
                return []

            # Find interface centroid
            iface_centroid = None
            for _, iface in self.interfaces:
                if iface.interface_id == interface_id:
                    iface_centroid = iface.centroid
                    break

            if iface_centroid is None:
                return iface_connections[:window_size * 2]

            # Compute query angle from interface centroid
            direction = query - iface_centroid
            query_angle = compute_angle_2d(direction)

            # Binary search within interface connections
            angles = [c.angle for c in iface_connections]
            idx = bisect.bisect_left(angles, query_angle)

            # Return window around this position
            n = len(iface_connections)
            start = max(0, idx - window_size)
            end = min(n, idx + window_size + 1)

            return iface_connections[start:end]

        # No interface specified - search all connections
        # First find closest interface, then search its connections
        closest = self.find_closest_interfaces(query, k=1)
        if closest:
            return self.find_connections_for_query(
                query,
                interface_id=closest[0][0],
                window_size=window_size
            )

        return []

    def route_query(
        self,
        query: np.ndarray,
        window_size: int = 5,
    ) -> Tuple[List[str], List[ExternalConnection]]:
        """
        Route a query through this node.

        1. Find closest interface(s)
        2. Find relevant external connections
        3. Return (interface_ids, connections)

        This is the main entry point for query routing.
        """
        # Find closest interfaces
        closest_interfaces = self.find_closest_interfaces(query, k=2)

        if not closest_interfaces:
            return [], []

        # Get connections for primary interface
        primary_interface = closest_interfaces[0][0]
        connections = self.find_connections_for_query(
            query,
            interface_id=primary_interface,
            window_size=window_size,
        )

        # Update statistics
        for _, iface in self.interfaces:
            if iface.interface_id == primary_interface:
                iface.queries_handled += 1
                break

        interface_ids = [iface_id for iface_id, _ in closest_interfaces]
        return interface_ids, connections

    def get_statistics(self) -> Dict:
        """Get node statistics."""
        connections_per_interface = defaultdict(int)
        for conn in self.connections:
            connections_per_interface[conn.source_interface] += 1

        return {
            "node_id": self.node_id,
            "num_interfaces": self.num_interfaces,
            "total_connections": self.total_connections,
            "avg_connections_per_interface": (
                self.total_connections / self.num_interfaces
                if self.num_interfaces > 0 else 0
            ),
            "connections_per_interface": dict(connections_per_interface),
            "queries_handled": sum(
                iface.queries_handled for _, iface in self.interfaces
            ),
        }


def generate_scale_free_interface_count(
    gamma: float = 2.5,
    min_interfaces: int = 1,
    max_interfaces: int = 100,
    rng: random.Random = None,
) -> int:
    """
    Generate interface count following power law distribution.

    P(k) ~ k^(-gamma)

    Args:
        gamma: Power law exponent (typically 2-3)
        min_interfaces: Minimum interfaces per node
        max_interfaces: Maximum interfaces per node
        rng: Random number generator

    Returns:
        Number of interfaces for a node
    """
    if rng is None:
        rng = random.Random()

    # Inverse transform sampling for power law
    # P(X > x) = (x / x_min)^(1 - gamma)
    u = rng.random()

    # Inverse CDF: x = x_min * (1 - u)^(1 / (1 - gamma))
    x = min_interfaces * (1 - u) ** (1 / (1 - gamma))

    # Clamp to range
    return min(max(int(x), min_interfaces), max_interfaces)


def create_scale_free_node(
    node_id: str,
    base_centroid: np.ndarray,
    gamma: float = 2.5,
    min_interfaces: int = 1,
    max_interfaces: int = 100,
    rng: random.Random = None,
) -> MultiInterfaceNode:
    """
    Create a node with scale-free number of interfaces.

    Interfaces are distributed around the base centroid.
    """
    if rng is None:
        rng = random.Random()

    num_interfaces = generate_scale_free_interface_count(
        gamma=gamma,
        min_interfaces=min_interfaces,
        max_interfaces=max_interfaces,
        rng=rng,
    )

    node = MultiInterfaceNode(node_id=node_id)

    dim = len(base_centroid)

    for i in range(num_interfaces):
        # Create interface centroid near base, with some spread
        offset = np.random.randn(dim) * 0.2
        centroid = base_centroid + offset
        centroid = centroid / np.linalg.norm(centroid)  # Normalize

        interface = Interface(
            interface_id=f"{node_id}_iface_{i}",
            centroid=centroid,
        )

        node.add_interface(interface)

    return node


# =============================================================================
# NETWORK OF MULTI-INTERFACE NODES
# =============================================================================

class MultiInterfaceNetwork:
    """
    A network of multi-interface nodes with scale-free properties.

    Nodes have varying numbers of interfaces (power law distribution).
    Connections are made between interfaces on different nodes.
    """

    def __init__(
        self,
        connections_per_interface: int = 15,
        gamma: float = 2.5,
    ):
        """
        Initialize network.

        Args:
            connections_per_interface: Target connections per interface
            gamma: Power law exponent for interface distribution
        """
        self.nodes: Dict[str, MultiInterfaceNode] = {}
        self.connections_per_interface = connections_per_interface
        self.gamma = gamma

        # Index: interface_id -> (node_id, Interface)
        self.interface_index: Dict[str, Tuple[str, Interface]] = {}

    def add_node(
        self,
        node_id: str,
        base_centroid: np.ndarray,
        num_interfaces: Optional[int] = None,
        rng: random.Random = None,
    ) -> MultiInterfaceNode:
        """
        Add a node to the network.

        If num_interfaces not specified, uses scale-free distribution.
        """
        if rng is None:
            rng = random.Random()

        if num_interfaces is None:
            node = create_scale_free_node(
                node_id=node_id,
                base_centroid=base_centroid,
                gamma=self.gamma,
                rng=rng,
            )
        else:
            node = MultiInterfaceNode(node_id=node_id)
            dim = len(base_centroid)

            for i in range(num_interfaces):
                offset = np.random.randn(dim) * 0.2
                centroid = base_centroid + offset
                centroid = centroid / np.linalg.norm(centroid)

                interface = Interface(
                    interface_id=f"{node_id}_iface_{i}",
                    centroid=centroid,
                )
                node.add_interface(interface)

        self.nodes[node_id] = node

        # Index interfaces
        for _, iface in node.interfaces:
            self.interface_index[iface.interface_id] = (node_id, iface)

        # Connect to existing nodes
        if len(self.nodes) > 1:
            self._connect_node(node_id)

        return node

    def _connect_node(self, node_id: str) -> None:
        """Connect a new node's interfaces to existing interfaces."""
        node = self.nodes[node_id]

        for _, source_iface in node.interfaces:
            # Find closest interfaces on other nodes
            candidates = []

            for other_id, other_node in self.nodes.items():
                if other_id == node_id:
                    continue

                for _, other_iface in other_node.interfaces:
                    dist = cosine_distance(
                        source_iface.centroid,
                        other_iface.centroid
                    )
                    candidates.append((dist, other_id, other_iface))

            # Sort by distance
            candidates.sort(key=lambda x: x[0])

            # Connect to closest interfaces
            connected = 0
            for dist, target_node_id, target_iface in candidates:
                if connected >= self.connections_per_interface:
                    break

                # Add bidirectional connection
                success = node.add_connection(
                    source_interface_id=source_iface.interface_id,
                    target_node_id=target_node_id,
                    target_interface_id=target_iface.interface_id,
                    target_centroid=target_iface.centroid,
                )

                if success:
                    # Reverse connection
                    target_node = self.nodes[target_node_id]
                    target_node.add_connection(
                        source_interface_id=target_iface.interface_id,
                        target_node_id=node_id,
                        target_interface_id=source_iface.interface_id,
                        target_centroid=source_iface.centroid,
                    )
                    connected += 1

    def route_query(
        self,
        query: np.ndarray,
        start_node_id: Optional[str] = None,
        max_hops: int = 10,
    ) -> Tuple[List[Tuple[str, str]], int]:
        """
        Route a query through the network.

        Returns:
            (path as [(node_id, interface_id), ...], comparisons)
        """
        if not self.nodes:
            return [], 0

        if start_node_id is None:
            start_node_id = random.choice(list(self.nodes.keys()))

        path = []
        comparisons = 0
        visited_interfaces: Set[str] = set()

        current_node_id = start_node_id

        for _ in range(max_hops):
            node = self.nodes[current_node_id]

            # Route through this node
            interface_ids, connections = node.route_query(query)
            comparisons += len(connections) + len(interface_ids)

            if not interface_ids:
                break

            primary_interface = interface_ids[0]
            path.append((current_node_id, primary_interface))
            visited_interfaces.add(primary_interface)

            # Find best unvisited connection
            best_conn = None
            best_dist = float('inf')

            for conn in connections:
                if conn.target_interface in visited_interfaces:
                    continue

                # Get target centroid
                if conn.target_interface in self.interface_index:
                    _, target_iface = self.interface_index[conn.target_interface]
                    dist = cosine_distance(query, target_iface.centroid)

                    if dist < best_dist:
                        best_dist = dist
                        best_conn = conn

            if best_conn is None:
                break  # No unvisited connections

            # Move to next node
            current_node_id = best_conn.target_node
            visited_interfaces.add(best_conn.target_interface)

        return path, comparisons

    def get_statistics(self) -> Dict:
        """Get network statistics."""
        interface_counts = [n.num_interfaces for n in self.nodes.values()]

        return {
            "num_nodes": len(self.nodes),
            "total_interfaces": sum(interface_counts),
            "avg_interfaces_per_node": (
                sum(interface_counts) / len(interface_counts)
                if interface_counts else 0
            ),
            "min_interfaces": min(interface_counts) if interface_counts else 0,
            "max_interfaces": max(interface_counts) if interface_counts else 0,
            "interface_distribution": {
                k: interface_counts.count(k)
                for k in sorted(set(interface_counts))
            },
            "total_connections": sum(
                n.total_connections for n in self.nodes.values()
            ) // 2,  # Bidirectional
        }
