# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Adaptive Node Subdivision for distributed KG topology.

"""
Adaptive Node Subdivision for KG Topology.

Nodes split when they exceed capacity thresholds, maintaining optimal
cluster density for consistent precision across network sizes.

Key concepts:
- Split triggers: document count, variance, latency, memory
- K-means partitioning into child nodes
- Parent becomes region node (routing only)
- Integrates with HierarchicalFederatedEngine

See: docs/proposals/ADAPTIVE_NODE_SUBDIVISION.md
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import deque

import numpy as np

try:
    from .kleinberg_router import KGNode
except ImportError:
    from kleinberg_router import KGNode


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SplitConfig:
    """Configuration for node subdivision."""

    enabled: bool = True

    # Split thresholds (ANY triggers split)
    max_documents: int = 1000
    max_variance: float = 0.5
    max_latency_p99_ms: float = 500.0
    max_memory_percent: float = 80.0

    # Split constraints
    min_child_documents: int = 100  # Don't split if children too small
    split_method: str = "kmeans"    # kmeans, medoid, random

    # Merge thresholds (optional)
    merge_enabled: bool = False
    min_documents_for_merge: int = 200  # Merge if both siblings below this
    min_query_rate_for_merge: float = 5.0  # Queries per minute

    # Monitoring
    check_interval_seconds: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "max_documents": self.max_documents,
            "max_variance": self.max_variance,
            "max_latency_p99_ms": self.max_latency_p99_ms,
            "max_memory_percent": self.max_memory_percent,
            "min_child_documents": self.min_child_documents,
            "split_method": self.split_method,
            "merge_enabled": self.merge_enabled,
        }


# =============================================================================
# NODE METRICS
# =============================================================================

@dataclass
class NodeMetrics:
    """Runtime metrics for split decision making."""

    document_count: int = 0
    centroid_variance: float = 0.0

    # Latency tracking (rolling window)
    latency_window_size: int = 100
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=100))

    # Query rate tracking
    query_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))

    # Memory (simplified - would use psutil in production)
    memory_percent: float = 0.0

    # Last check time
    last_check_time: float = 0.0

    def record_query(self, latency_ms: float) -> None:
        """Record a query execution."""
        self.latencies_ms.append(latency_ms)
        self.query_timestamps.append(time.time())

    def get_latency_p99(self) -> float:
        """Get P99 latency from recent queries."""
        if len(self.latencies_ms) < 10:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        p99_idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]

    def get_query_rate(self) -> float:
        """Get queries per minute from recent history."""
        if len(self.query_timestamps) < 2:
            return 0.0
        now = time.time()
        recent = [t for t in self.query_timestamps if now - t < 60]
        return len(recent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_count": self.document_count,
            "centroid_variance": self.centroid_variance,
            "latency_p99_ms": self.get_latency_p99(),
            "query_rate_per_min": self.get_query_rate(),
            "memory_percent": self.memory_percent,
        }


# =============================================================================
# NODE TYPES
# =============================================================================

class NodeType(Enum):
    """Type of node in the hierarchy."""
    LEAF = "leaf"      # Stores documents, answers queries
    REGION = "region"  # Routes to children, no documents


@dataclass
class SubdividableNode:
    """
    A node that can subdivide when thresholds are exceeded.

    Extends KGNode concept with subdivision capabilities.
    """

    node_id: str
    node_type: NodeType = NodeType.LEAF

    # Embedding data (for LEAF nodes)
    centroid: Optional[np.ndarray] = None
    embeddings: List[np.ndarray] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)

    # Hierarchy (for REGION nodes)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    level: int = 0

    # Metadata
    topics: List[str] = field(default_factory=list)
    endpoint: str = ""
    embedding_model: str = ""

    # Runtime
    config: SplitConfig = field(default_factory=SplitConfig)
    metrics: NodeMetrics = field(default_factory=NodeMetrics)

    def __post_init__(self):
        """Initialize metrics from current state."""
        self.metrics.document_count = len(self.document_ids)
        if self.embeddings:
            self._update_centroid_and_variance()

    def _update_centroid_and_variance(self) -> None:
        """Recompute centroid and variance from embeddings."""
        if not self.embeddings:
            self.centroid = None
            self.metrics.centroid_variance = 0.0
            return

        embeddings_array = np.array(self.embeddings)
        self.centroid = np.mean(embeddings_array, axis=0)

        # Variance = average squared distance from centroid
        if len(self.embeddings) > 1:
            distances = np.linalg.norm(embeddings_array - self.centroid, axis=1)
            self.metrics.centroid_variance = float(np.mean(distances ** 2))
        else:
            self.metrics.centroid_variance = 0.0

    def add_document(self, doc_id: str, embedding: np.ndarray) -> None:
        """Add a document to this node."""
        if self.node_type != NodeType.LEAF:
            raise ValueError("Cannot add documents to REGION node")

        self.document_ids.append(doc_id)
        self.embeddings.append(embedding)
        self.metrics.document_count = len(self.document_ids)
        self._update_centroid_and_variance()

    def to_kg_node(self) -> KGNode:
        """Convert to standard KGNode for routing."""
        return KGNode(
            node_id=self.node_id,
            endpoint=self.endpoint,
            centroid=self.centroid,
            topics=self.topics,
            embedding_model=self.embedding_model,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "document_count": len(self.document_ids),
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "level": self.level,
            "topics": self.topics,
            "endpoint": self.endpoint,
            "metrics": self.metrics.to_dict(),
        }


# =============================================================================
# SPLIT DETECTION
# =============================================================================

def should_split(node: SubdividableNode) -> Tuple[bool, str]:
    """
    Check if a node should subdivide.

    Args:
        node: The node to check

    Returns:
        Tuple of (should_split, reason)
    """
    if not node.config.enabled:
        return False, "subdivision disabled"

    if node.node_type != NodeType.LEAF:
        return False, "only LEAF nodes can split"

    # Check minimum size for split
    if len(node.document_ids) < node.config.min_child_documents * 2:
        return False, f"too few documents ({len(node.document_ids)})"

    # Check triggers
    if node.metrics.document_count > node.config.max_documents:
        return True, f"document count {node.metrics.document_count} > {node.config.max_documents}"

    if node.metrics.centroid_variance > node.config.max_variance:
        return True, f"variance {node.metrics.centroid_variance:.3f} > {node.config.max_variance}"

    latency_p99 = node.metrics.get_latency_p99()
    if latency_p99 > node.config.max_latency_p99_ms:
        return True, f"latency P99 {latency_p99:.1f}ms > {node.config.max_latency_p99_ms}ms"

    if node.metrics.memory_percent > node.config.max_memory_percent:
        return True, f"memory {node.metrics.memory_percent:.1f}% > {node.config.max_memory_percent}%"

    return False, "no threshold exceeded"


# =============================================================================
# K-MEANS SPLITTING
# =============================================================================

def kmeans_split(
    embeddings: List[np.ndarray],
    k: int = 2,
    max_iterations: int = 100,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Simple k-means clustering for splitting.

    Args:
        embeddings: List of embedding vectors
        k: Number of clusters (default 2 for binary split)
        max_iterations: Maximum iterations
        seed: Random seed for reproducibility

    Returns:
        List of cluster labels (0 to k-1) for each embedding
    """
    if len(embeddings) < k:
        return list(range(len(embeddings)))

    rng = np.random.default_rng(seed)
    embeddings_array = np.array(embeddings)
    n_samples = len(embeddings)

    # Initialize centroids randomly
    indices = rng.choice(n_samples, size=k, replace=False)
    centroids = embeddings_array[indices].copy()

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iterations):
        # Assign to nearest centroid
        old_labels = labels.copy()
        for i, emb in enumerate(embeddings_array):
            distances = [np.linalg.norm(emb - c) for c in centroids]
            labels[i] = np.argmin(distances)

        # Check convergence
        if np.array_equal(labels, old_labels):
            break

        # Update centroids
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centroids[j] = embeddings_array[mask].mean(axis=0)

    return labels.tolist()


# =============================================================================
# SPLIT EXECUTION
# =============================================================================

def split_node(
    node: SubdividableNode,
    seed: Optional[int] = None,
) -> Tuple[SubdividableNode, SubdividableNode]:
    """
    Split a node into two children using k-means.

    Args:
        node: The node to split (must be LEAF type)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (child_a, child_b)

    Side effects:
        - Modifies node to become REGION type
        - Clears node's embeddings and document_ids
    """
    if node.node_type != NodeType.LEAF:
        raise ValueError("Can only split LEAF nodes")

    if len(node.embeddings) < 2:
        raise ValueError("Need at least 2 documents to split")

    # 1. Cluster documents into 2 groups
    if node.config.split_method == "kmeans":
        labels = kmeans_split(node.embeddings, k=2, seed=seed)
    elif node.config.split_method == "random":
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, size=len(node.embeddings)).tolist()
    else:
        # Default to kmeans
        labels = kmeans_split(node.embeddings, k=2, seed=seed)

    # 2. Partition documents
    docs_a = []
    embeddings_a = []
    docs_b = []
    embeddings_b = []

    for doc_id, embedding, label in zip(node.document_ids, node.embeddings, labels):
        if label == 0:
            docs_a.append(doc_id)
            embeddings_a.append(embedding)
        else:
            docs_b.append(doc_id)
            embeddings_b.append(embedding)

    # 3. Create child nodes
    child_a = SubdividableNode(
        node_id=f"{node.node_id}_a",
        node_type=NodeType.LEAF,
        embeddings=embeddings_a,
        document_ids=docs_a,
        parent_id=node.node_id,
        level=node.level + 1,
        topics=node.topics.copy(),
        endpoint=f"{node.endpoint}/a",  # Would be different in production
        embedding_model=node.embedding_model,
        config=node.config,
    )

    child_b = SubdividableNode(
        node_id=f"{node.node_id}_b",
        node_type=NodeType.LEAF,
        embeddings=embeddings_b,
        document_ids=docs_b,
        parent_id=node.node_id,
        level=node.level + 1,
        topics=node.topics.copy(),
        endpoint=f"{node.endpoint}/b",
        embedding_model=node.embedding_model,
        config=node.config,
    )

    # 4. Convert parent to region node
    node.node_type = NodeType.REGION
    node.children_ids = [child_a.node_id, child_b.node_id]
    node.embeddings = []
    node.document_ids = []

    # Update parent centroid to average of children
    if child_a.centroid is not None and child_b.centroid is not None:
        node.centroid = (child_a.centroid + child_b.centroid) / 2

    node.metrics = NodeMetrics()  # Reset metrics

    return child_a, child_b


# =============================================================================
# MERGE DETECTION (Optional)
# =============================================================================

def should_merge(
    node_a: SubdividableNode,
    node_b: SubdividableNode,
) -> Tuple[bool, str]:
    """
    Check if sibling nodes should merge back together.

    Args:
        node_a: First sibling node
        node_b: Second sibling node

    Returns:
        Tuple of (should_merge, reason)
    """
    # Must be siblings
    if node_a.parent_id != node_b.parent_id:
        return False, "not siblings"

    if node_a.parent_id is None:
        return False, "no parent"

    # Must both be LEAF nodes
    if node_a.node_type != NodeType.LEAF or node_b.node_type != NodeType.LEAF:
        return False, "both must be LEAF nodes"

    config = node_a.config
    if not config.merge_enabled:
        return False, "merge disabled"

    # Check if both are underutilized
    total_docs = len(node_a.document_ids) + len(node_b.document_ids)
    if total_docs >= config.min_documents_for_merge * 2:
        return False, f"combined docs {total_docs} >= threshold"

    # Check query rates
    rate_a = node_a.metrics.get_query_rate()
    rate_b = node_b.metrics.get_query_rate()
    if rate_a >= config.min_query_rate_for_merge or rate_b >= config.min_query_rate_for_merge:
        return False, f"query rates too high ({rate_a:.1f}, {rate_b:.1f})"

    return True, "both siblings underutilized"


def merge_nodes(
    node_a: SubdividableNode,
    node_b: SubdividableNode,
    parent: SubdividableNode,
) -> SubdividableNode:
    """
    Merge sibling nodes back into parent.

    Args:
        node_a: First sibling
        node_b: Second sibling
        parent: Parent region node (will become LEAF)

    Returns:
        The merged parent node (now LEAF)
    """
    if parent.node_type != NodeType.REGION:
        raise ValueError("Parent must be REGION node")

    # Combine documents
    parent.document_ids = node_a.document_ids + node_b.document_ids
    parent.embeddings = node_a.embeddings + node_b.embeddings

    # Convert back to LEAF
    parent.node_type = NodeType.LEAF
    parent.children_ids = []

    # Recompute centroid
    parent._update_centroid_and_variance()
    parent.metrics.document_count = len(parent.document_ids)

    return parent


# =============================================================================
# NODE REGISTRY
# =============================================================================

class SubdivisionRegistry:
    """
    Registry of subdivided nodes.

    Tracks the hierarchy of nodes and provides routing helpers.
    """

    def __init__(self):
        self.nodes: Dict[str, SubdividableNode] = {}

    def register(self, node: SubdividableNode) -> None:
        """Register a node."""
        self.nodes[node.node_id] = node

    def unregister(self, node_id: str) -> None:
        """Unregister a node."""
        self.nodes.pop(node_id, None)

    def get(self, node_id: str) -> Optional[SubdividableNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_leaf_nodes(self) -> List[SubdividableNode]:
        """Get all LEAF nodes."""
        return [n for n in self.nodes.values() if n.node_type == NodeType.LEAF]

    def get_region_nodes(self) -> List[SubdividableNode]:
        """Get all REGION nodes."""
        return [n for n in self.nodes.values() if n.node_type == NodeType.REGION]

    def get_children(self, node_id: str) -> List[SubdividableNode]:
        """Get children of a region node."""
        node = self.nodes.get(node_id)
        if node is None or node.node_type != NodeType.REGION:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_descendants(self, node_id: str) -> List[SubdividableNode]:
        """Get all descendant LEAF nodes of a region."""
        result = []
        to_visit = [node_id]

        while to_visit:
            current_id = to_visit.pop()
            node = self.nodes.get(current_id)
            if node is None:
                continue

            if node.node_type == NodeType.LEAF:
                result.append(node)
            else:
                to_visit.extend(node.children_ids)

        return result

    def route_to_leaf(
        self,
        query_embedding: np.ndarray,
        start_node_id: Optional[str] = None,
    ) -> Optional[SubdividableNode]:
        """
        Route a query to the best LEAF node.

        Traverses from region to children based on similarity.
        """
        if start_node_id is None:
            # Start from root (node with no parent)
            roots = [n for n in self.nodes.values() if n.parent_id is None]
            if not roots:
                return None
            # Pick most similar root
            best_root = max(
                roots,
                key=lambda n: self._similarity(query_embedding, n.centroid)
            )
            start_node_id = best_root.node_id

        current = self.nodes.get(start_node_id)
        if current is None:
            return None

        while current.node_type == NodeType.REGION:
            children = self.get_children(current.node_id)
            if not children:
                return None

            # Pick most similar child
            current = max(
                children,
                key=lambda n: self._similarity(query_embedding, n.centroid)
            )

        return current

    def _similarity(self, a: np.ndarray, b: Optional[np.ndarray]) -> float:
        """Compute cosine similarity."""
        if b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def check_and_split_all(self, seed: Optional[int] = None) -> List[str]:
        """
        Check all nodes and split those exceeding thresholds.

        Returns:
            List of node IDs that were split
        """
        split_nodes = []

        # Get snapshot of leaf nodes (avoid modifying during iteration)
        leaves = list(self.get_leaf_nodes())

        for node in leaves:
            should, reason = should_split(node)
            if should:
                child_a, child_b = split_node(node, seed=seed)
                self.register(child_a)
                self.register(child_b)
                split_nodes.append(node.node_id)

        return split_nodes

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary."""
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "leaf_count": len(self.get_leaf_nodes()),
            "region_count": len(self.get_region_nodes()),
        }
