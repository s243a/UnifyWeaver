# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Synthetic Network Generator for Federation Benchmarks.

Creates mock node networks with controllable characteristics:
- Size: number of nodes
- Topic distribution: clustered vs uniform embeddings
- Latency profiles: fast, normal, slow, mixed
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class SyntheticNode:
    """A synthetic node for benchmarking."""

    node_id: str
    centroid: np.ndarray
    topics: List[str] = field(default_factory=list)
    latency_ms: float = 50.0
    result_count: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "centroid": self.centroid.tolist(),
            "topics": self.topics,
            "latency_ms": self.latency_ms,
            "result_count": self.result_count,
            "metadata": self.metadata,
        }


# Topic clusters for generating themed nodes
TOPIC_CLUSTERS = {
    "data_processing": ["csv", "json", "xml", "parsing", "etl", "pandas"],
    "machine_learning": ["neural", "training", "inference", "model", "pytorch", "tensorflow"],
    "web_development": ["http", "rest", "api", "flask", "django", "frontend"],
    "databases": ["sql", "nosql", "query", "index", "postgres", "mongodb"],
    "devops": ["docker", "kubernetes", "ci", "deploy", "aws", "cloud"],
}


def _generate_cluster_centroid(
    cluster_center: np.ndarray,
    cluster_radius: float,
    embedding_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a centroid near a cluster center."""
    offset = rng.normal(0, cluster_radius, embedding_dim)
    centroid = cluster_center + offset
    # Normalize to unit sphere
    return centroid / np.linalg.norm(centroid)


def _generate_uniform_centroid(
    embedding_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a uniformly distributed centroid on unit sphere."""
    centroid = rng.normal(0, 1, embedding_dim)
    return centroid / np.linalg.norm(centroid)


def _get_latency(
    profile: str,
    rng: np.random.Generator,
) -> float:
    """Get latency based on profile."""
    if profile == "fast":
        return max(5.0, rng.normal(10.0, 3.0))
    elif profile == "normal":
        return max(10.0, rng.normal(50.0, 15.0))
    elif profile == "slow":
        return max(50.0, rng.normal(200.0, 50.0))
    elif profile == "mixed":
        # 60% normal, 30% fast, 10% slow
        roll = rng.random()
        if roll < 0.6:
            return max(10.0, rng.normal(50.0, 15.0))
        elif roll < 0.9:
            return max(5.0, rng.normal(10.0, 3.0))
        else:
            return max(50.0, rng.normal(200.0, 50.0))
    else:
        return 50.0


def create_synthetic_network(
    num_nodes: int,
    embedding_dim: int = 384,
    topic_distribution: str = "clustered",
    latency_profile: str = "normal",
    seed: Optional[int] = None,
) -> List[SyntheticNode]:
    """
    Create a synthetic node network for benchmarking.

    Args:
        num_nodes: Number of nodes to create
        embedding_dim: Dimension of embedding vectors
        topic_distribution: "clustered" or "uniform"
        latency_profile: "fast", "normal", "slow", or "mixed"
        seed: Random seed for reproducibility

    Returns:
        List of SyntheticNode objects
    """
    rng = np.random.default_rng(seed)
    nodes = []

    # Generate cluster centers for clustered distribution
    cluster_names = list(TOPIC_CLUSTERS.keys())
    num_clusters = len(cluster_names)
    cluster_centers = {}

    if topic_distribution == "clustered":
        for name in cluster_names:
            center = rng.normal(0, 1, embedding_dim)
            cluster_centers[name] = center / np.linalg.norm(center)

    for i in range(num_nodes):
        node_id = f"node_{i:03d}"

        if topic_distribution == "clustered":
            # Assign to a cluster
            cluster_idx = i % num_clusters
            cluster_name = cluster_names[cluster_idx]
            cluster_center = cluster_centers[cluster_name]

            # Generate centroid near cluster center
            centroid = _generate_cluster_centroid(
                cluster_center,
                cluster_radius=0.3,
                embedding_dim=embedding_dim,
                rng=rng,
            )

            # Assign topics from the cluster
            topics = list(rng.choice(
                TOPIC_CLUSTERS[cluster_name],
                size=min(3, len(TOPIC_CLUSTERS[cluster_name])),
                replace=False,
            ))
        else:
            # Uniform distribution
            centroid = _generate_uniform_centroid(embedding_dim, rng)
            # Random topics from all clusters
            all_topics = [t for topics in TOPIC_CLUSTERS.values() for t in topics]
            topics = list(rng.choice(all_topics, size=3, replace=False))

        latency = _get_latency(latency_profile, rng)
        result_count = int(max(1, rng.normal(10, 3)))

        node = SyntheticNode(
            node_id=node_id,
            centroid=centroid,
            topics=topics,
            latency_ms=latency,
            result_count=result_count,
            metadata={
                "corpus_id": f"corpus_{cluster_name if topic_distribution == 'clustered' else 'mixed'}",
                "embedding_model": "benchmark-model",
            },
        )
        nodes.append(node)

    return nodes


def create_network_suite(
    sizes: List[int] = [10, 25, 50],
    seed: int = 42,
) -> Dict[int, List[SyntheticNode]]:
    """
    Create networks of various sizes for scalability testing.

    Args:
        sizes: List of network sizes to create
        seed: Base random seed

    Returns:
        Dict mapping size to network
    """
    networks = {}
    for i, size in enumerate(sizes):
        networks[size] = create_synthetic_network(
            num_nodes=size,
            topic_distribution="clustered",
            latency_profile="mixed",
            seed=seed + i,
        )
    return networks
