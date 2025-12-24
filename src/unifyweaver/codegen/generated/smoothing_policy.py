"""
LDA Smoothing Policy - Python Implementation

Generated from lda_smoothing_policy.pl by UnifyWeaver.
Do not edit manually - regenerate from source.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SmoothingTechnique(Enum):
    """Available smoothing techniques."""
    FFT = "fft"
    BASIS_K4 = "basis_k4"
    BASIS_K8 = "basis_k8"
    BASIS_K16 = "basis_k16"
    BASELINE = "baseline"


@dataclass
class NodeInfo:
    """Information about a node in the smoothing tree."""
    node_id: str
    cluster_count: int
    total_pairs: int
    depth: int
    avg_pairs: float
    similarity_score: float = 0.5  # Default: moderately confusable


@dataclass
class SmoothingAction:
    """An action in the smoothing plan."""
    technique: SmoothingTechnique
    node_id: str


# Policy constants
FFT_THRESHOLD = 30
BASIS_SWEET_SPOT = (10, 50)
DISTINGUISH_THRESHOLD = 0.3
MAX_RECURSION_DEPTH = 4


def clusters_distinguishable(node: NodeInfo) -> bool:
    """Check if clusters within this node are well-separated after projection.

    Low similarity score means clusters are distinct and don't need refinement.
    """
    return node.similarity_score < DISTINGUISH_THRESHOLD


def refinement_needed(node: NodeInfo) -> bool:
    """Check if this node would benefit from further refinement.

    Refinement is needed when:
    - Enough clusters to potentially confuse (>10)
    - Not too deep in the tree (<4)
    - Clusters still too similar (>0.7 similarity)
    """
    return (
        node.cluster_count > 10 and
        node.depth < MAX_RECURSION_DEPTH and
        node.similarity_score > 0.7
    )


def sufficient_data(node: NodeInfo, technique: SmoothingTechnique) -> bool:
    """Check if node has enough data for the technique to be meaningful."""
    min_clusters, max_clusters = {
        SmoothingTechnique.FFT: (10, 100000),
        SmoothingTechnique.BASIS_K4: (5, 500),
        SmoothingTechnique.BASIS_K8: (10, 200),
        SmoothingTechnique.BASIS_K16: (20, 100),
        SmoothingTechnique.BASELINE: (1, 100000),
    }.get(technique, (1, 100000))

    return (
        min_clusters <= node.cluster_count <= max_clusters and
        node.avg_pairs >= 1.0
    )


def recommended_technique(node: NodeInfo) -> SmoothingTechnique:
    """Recommend a smoothing technique based on node properties.

    Rules (in priority order):
    1. Large clusters (>=30) at shallow depths (<3) -> FFT
    2. Medium clusters (10-50) at depth >=1 with good data -> basis_k8
    3. Smaller clusters (5-20) at depth >=2 -> basis_k4
    4. Very small clusters (<5) -> baseline
    5. Large clusters at deep levels (>=50, depth >=3) -> FFT
    6. Fallback -> basis_k4
    """
    c = node.cluster_count
    d = node.depth
    avg = node.avg_pairs

    # Rule 1: Large clusters at shallow depths -> FFT
    if c >= FFT_THRESHOLD and d < 3:
        return SmoothingTechnique.FFT

    # Rule 2: Medium clusters -> basis_k8
    if BASIS_SWEET_SPOT[0] <= c <= BASIS_SWEET_SPOT[1] and d >= 1 and avg >= 2:
        return SmoothingTechnique.BASIS_K8

    # Rule 3: Smaller clusters at deeper levels -> basis_k4
    if 5 <= c < 20 and d >= 2 and avg >= 2:
        return SmoothingTechnique.BASIS_K4

    # Rule 4: Very small clusters -> baseline
    if c < 5:
        return SmoothingTechnique.BASELINE

    # Rule 5: Large clusters at deep levels -> FFT
    if c >= 50 and d >= 3:
        return SmoothingTechnique.FFT

    # Rule 6: Fallback
    if c >= 5:
        return SmoothingTechnique.BASIS_K4

    return SmoothingTechnique.BASELINE


def generate_smoothing_plan(
    root: NodeInfo,
    children: Dict[str, List[NodeInfo]]
) -> List[SmoothingAction]:
    """Generate a complete smoothing plan for the tree.

    Args:
        root: Root node info
        children: Mapping from parent node_id to list of child NodeInfo

    Returns:
        List of SmoothingAction in execution order
    """
    plan = []
    _plan_recursive(root, children, plan)
    return plan


def _plan_recursive(
    node: NodeInfo,
    children: Dict[str, List[NodeInfo]],
    plan: List[SmoothingAction]
) -> None:
    """Recursively build the plan."""
    technique = recommended_technique(node)
    plan.append(SmoothingAction(technique=technique, node_id=node.node_id))

    if refinement_needed(node) and node.node_id in children:
        for child in children[node.node_id]:
            if not clusters_distinguishable(child):
                _plan_recursive(child, children, plan)


def estimate_cost_ms(plan: List[SmoothingAction], nodes: Dict[str, NodeInfo]) -> float:
    """Estimate total training cost in milliseconds.

    Cost estimates per cluster:
    - FFT: ~0.4ms
    - basis_k4: ~10ms
    - basis_k8: ~15ms
    - baseline: ~0.02ms
    """
    cost_per_cluster = {
        SmoothingTechnique.FFT: 0.4,
        SmoothingTechnique.BASIS_K4: 10.0,
        SmoothingTechnique.BASIS_K8: 15.0,
        SmoothingTechnique.BASIS_K16: 25.0,
        SmoothingTechnique.BASELINE: 0.02,
    }

    total = 0.0
    for action in plan:
        if action.node_id in nodes:
            c = nodes[action.node_id].cluster_count
            total += c * cost_per_cluster.get(action.technique, 1.0)
    return total


def plan_summary(plan: List[SmoothingAction]) -> Dict:
    """Get a summary of the plan."""
    from collections import Counter

    tech_counts = Counter(a.technique.value for a in plan)

    return {
        'num_actions': len(plan),
        'technique_counts': dict(tech_counts),
    }
