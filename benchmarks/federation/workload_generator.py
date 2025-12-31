# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""
Workload Generator for Federation Benchmarks.

Creates representative query patterns:
- SPECIFIC: High similarity to one node (focused queries)
- EXPLORATORY: Low variance across nodes (broad queries)
- CONSENSUS: Medium similarity, seeking agreement
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import numpy as np

from .synthetic_network import SyntheticNode


class QueryType(Enum):
    """Classification of query characteristics."""

    SPECIFIC = "specific"
    EXPLORATORY = "exploratory"
    CONSENSUS = "consensus"


@dataclass
class BenchmarkQuery:
    """A benchmark query with ground truth."""

    query_id: str
    embedding: np.ndarray
    expected_type: QueryType
    ground_truth_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "embedding": self.embedding.tolist(),
            "expected_type": self.expected_type.value,
            "ground_truth_nodes": self.ground_truth_nodes,
            "metadata": self.metadata,
        }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _generate_specific_query(
    nodes: List[SyntheticNode],
    query_id: str,
    rng: np.random.Generator,
) -> BenchmarkQuery:
    """
    Generate a SPECIFIC query - high similarity to one node.

    The query embedding is very close to a single node's centroid.
    """
    target_node = rng.choice(nodes)
    # Small perturbation from target centroid
    noise = rng.normal(0, 0.05, len(target_node.centroid))
    embedding = target_node.centroid + noise
    embedding = embedding / np.linalg.norm(embedding)

    # Ground truth: the target node and maybe 1-2 nearby nodes
    similarities = [
        (node.node_id, _cosine_similarity(embedding, node.centroid))
        for node in nodes
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    ground_truth = [nid for nid, sim in similarities[:3] if sim > 0.7]

    return BenchmarkQuery(
        query_id=query_id,
        embedding=embedding,
        expected_type=QueryType.SPECIFIC,
        ground_truth_nodes=ground_truth,
        metadata={"target_node": target_node.node_id},
    )


def _generate_exploratory_query(
    nodes: List[SyntheticNode],
    query_id: str,
    rng: np.random.Generator,
) -> BenchmarkQuery:
    """
    Generate an EXPLORATORY query - low variance across nodes.

    The query embedding is equidistant from multiple clusters.
    """
    # Average of several random node centroids
    sample_size = min(5, len(nodes))
    sample_nodes = rng.choice(nodes, size=sample_size, replace=False)
    embedding = np.mean([n.centroid for n in sample_nodes], axis=0)
    # Add some noise
    noise = rng.normal(0, 0.1, len(embedding))
    embedding = embedding + noise
    embedding = embedding / np.linalg.norm(embedding)

    # Ground truth: nodes with similarity > 0.5 (lower threshold for broad queries)
    similarities = [
        (node.node_id, _cosine_similarity(embedding, node.centroid))
        for node in nodes
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    ground_truth = [nid for nid, sim in similarities if sim > 0.5][:10]

    return BenchmarkQuery(
        query_id=query_id,
        embedding=embedding,
        expected_type=QueryType.EXPLORATORY,
        ground_truth_nodes=ground_truth,
        metadata={"sample_nodes": [n.node_id for n in sample_nodes]},
    )


def _generate_consensus_query(
    nodes: List[SyntheticNode],
    query_id: str,
    rng: np.random.Generator,
) -> BenchmarkQuery:
    """
    Generate a CONSENSUS query - medium similarity, seeking agreement.

    The query is close to a cluster but not a single node.
    """
    # Pick nodes from the same cluster (similar topics)
    target_node = rng.choice(nodes)
    same_topic_nodes = [
        n for n in nodes
        if any(t in n.topics for t in target_node.topics)
    ]

    if len(same_topic_nodes) < 2:
        same_topic_nodes = nodes[:5]

    # Average of cluster members
    sample = rng.choice(
        same_topic_nodes,
        size=min(3, len(same_topic_nodes)),
        replace=False,
    )
    embedding = np.mean([n.centroid for n in sample], axis=0)
    noise = rng.normal(0, 0.08, len(embedding))
    embedding = embedding + noise
    embedding = embedding / np.linalg.norm(embedding)

    # Ground truth: nodes with similarity > 0.6
    similarities = [
        (node.node_id, _cosine_similarity(embedding, node.centroid))
        for node in nodes
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    ground_truth = [nid for nid, sim in similarities if sim > 0.6][:5]

    return BenchmarkQuery(
        query_id=query_id,
        embedding=embedding,
        expected_type=QueryType.CONSENSUS,
        ground_truth_nodes=ground_truth,
        metadata={"cluster_sample": [n.node_id for n in sample]},
    )


def generate_workload(
    network: List[SyntheticNode],
    num_queries: int = 50,
    query_mix: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> List[BenchmarkQuery]:
    """
    Generate a workload of benchmark queries.

    Args:
        network: List of synthetic nodes
        num_queries: Total number of queries to generate
        query_mix: Distribution of query types, e.g. {"specific": 0.4, "exploratory": 0.3, "consensus": 0.3}
        seed: Random seed for reproducibility

    Returns:
        List of BenchmarkQuery objects
    """
    if query_mix is None:
        query_mix = {"specific": 0.4, "exploratory": 0.3, "consensus": 0.3}

    rng = np.random.default_rng(seed)
    queries = []

    # Calculate counts for each type
    type_counts = {
        QueryType.SPECIFIC: int(num_queries * query_mix.get("specific", 0.4)),
        QueryType.EXPLORATORY: int(num_queries * query_mix.get("exploratory", 0.3)),
        QueryType.CONSENSUS: int(num_queries * query_mix.get("consensus", 0.3)),
    }
    # Handle rounding
    remaining = num_queries - sum(type_counts.values())
    type_counts[QueryType.SPECIFIC] += remaining

    generators = {
        QueryType.SPECIFIC: _generate_specific_query,
        QueryType.EXPLORATORY: _generate_exploratory_query,
        QueryType.CONSENSUS: _generate_consensus_query,
    }

    query_idx = 0
    for query_type, count in type_counts.items():
        generator = generators[query_type]
        for _ in range(count):
            query_id = f"q_{query_idx:04d}"
            query = generator(network, query_id, rng)
            queries.append(query)
            query_idx += 1

    # Shuffle to mix query types
    rng.shuffle(queries)

    return queries
