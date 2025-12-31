# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Hierarchical Smoothing using Federation-style Cluster Aggregation.

Uses the federation pattern to build larger clusters from smaller ones,
then applies smoothing at each hierarchy level. This provides:

1. Multi-resolution smoothing - coarse patterns at high levels, fine at low
2. Reuse of existing infrastructure (HNSW, smoothing_basis)
3. Natural handling of varying cluster densities

Hierarchy levels:
- Level 0: Raw clusters (e.g., 277 from training data)
- Level 1: Merged super-clusters (e.g., 50 groups)
- Level 2: Mega-clusters (e.g., 10 groups)
- ...

At each level, clusters with similar centroids are merged and smoothing
is applied to the combined Q-A pairs.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import logging

# Import existing smoothing
try:
    from .smoothing_basis import SmoothingBasisProjection
except ImportError:
    from smoothing_basis import SmoothingBasisProjection

logger = logging.getLogger(__name__)


@dataclass
class ClusterNode:
    """A node in the cluster hierarchy."""
    level: int
    cluster_id: str
    questions: np.ndarray  # Shape (n, d)
    answer: np.ndarray     # Shape (d,) or (1, d)
    centroid: np.ndarray   # Shape (d,)
    children: List['ClusterNode'] = field(default_factory=list)
    projection: Optional[np.ndarray] = None  # Trained W matrix


def compute_centroid(questions: np.ndarray) -> np.ndarray:
    """Compute centroid of question embeddings."""
    return np.mean(questions, axis=0)


def merge_clusters(clusters: List[ClusterNode], new_level: int,
                   group_id: str) -> ClusterNode:
    """
    Merge multiple clusters into a single super-cluster.

    The merged cluster contains:
    - All questions from children
    - Answer = weighted average of child answers (by question count)
    - Centroid = mean of all questions
    """
    all_questions = np.vstack([c.questions for c in clusters])

    # Weighted average of answers
    total_weight = sum(len(c.questions) for c in clusters)
    answer = np.zeros_like(clusters[0].answer)
    for c in clusters:
        weight = len(c.questions) / total_weight
        answer += weight * c.answer

    centroid = compute_centroid(all_questions)

    return ClusterNode(
        level=new_level,
        cluster_id=f"L{new_level}_{group_id}",
        questions=all_questions,
        answer=answer,
        centroid=centroid,
        children=clusters
    )


def build_hierarchy(base_clusters: List[ClusterNode],
                    merge_thresholds: List[float] = None,
                    num_levels: int = 3) -> List[List[ClusterNode]]:
    """
    Build cluster hierarchy using agglomerative clustering.

    Args:
        base_clusters: Level 0 clusters
        merge_thresholds: Distance thresholds for each level
                         (higher = more merging)
        num_levels: Number of hierarchy levels to create

    Returns:
        List of cluster lists, one per level
        [level_0_clusters, level_1_clusters, ...]
    """
    if merge_thresholds is None:
        # Default: progressively more aggressive merging
        merge_thresholds = [0.3, 0.5, 0.7][:num_levels-1]

    hierarchy = [base_clusters]
    current_clusters = base_clusters

    for level, threshold in enumerate(merge_thresholds, start=1):
        if len(current_clusters) <= 1:
            break

        # Compute centroids
        centroids = np.array([c.centroid for c in current_clusters])

        # Hierarchical clustering on centroids
        if len(centroids) > 1:
            linkage_matrix = linkage(centroids, method='average', metric='cosine')
            labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
        else:
            labels = [1]

        # Group clusters by label
        groups: Dict[int, List[ClusterNode]] = {}
        for idx, label in enumerate(labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(current_clusters[idx])

        # Merge groups
        merged_clusters = []
        for group_id, group_clusters in sorted(groups.items()):
            if len(group_clusters) == 1:
                # Single cluster - just promote to next level
                c = group_clusters[0]
                merged = ClusterNode(
                    level=level,
                    cluster_id=f"L{level}_{group_id}",
                    questions=c.questions,
                    answer=c.answer,
                    centroid=c.centroid,
                    children=[c]
                )
            else:
                merged = merge_clusters(group_clusters, level, str(group_id))
            merged_clusters.append(merged)

        logger.info(f"Level {level}: {len(current_clusters)} -> {len(merged_clusters)} clusters")
        hierarchy.append(merged_clusters)
        current_clusters = merged_clusters

    return hierarchy


class HierarchicalSmoothing:
    """
    Multi-level smoothing using cluster hierarchy.

    Trains smoothing projections at each hierarchy level.
    Inference combines projections from multiple levels.
    """

    def __init__(self, num_levels: int = 3,
                 merge_thresholds: List[float] = None,
                 level_weights: List[float] = None,
                 num_basis: int = 4):
        """
        Args:
            num_levels: Number of hierarchy levels
            merge_thresholds: Distance thresholds for merging at each level
            level_weights: Weights for combining projections from levels
                          (default: equal weights)
            num_basis: Number of basis matrices for smoothing
        """
        self.num_levels = num_levels
        self.merge_thresholds = merge_thresholds
        self.level_weights = level_weights
        self.num_basis = num_basis

        self.hierarchy: List[List[ClusterNode]] = []
        self.projections: List[SmoothingBasisProjection] = []

    def train(self, clusters: List[Tuple[np.ndarray, np.ndarray, str]],
              num_iterations: int = 50) -> Dict[str, Any]:
        """
        Train hierarchical smoothing.

        Args:
            clusters: List of (Q, A, cluster_id) tuples
            num_iterations: Iterations for smoothing basis training

        Returns:
            Training statistics
        """
        # Create base cluster nodes
        base_nodes = []
        for Q, A, cid in clusters:
            if Q.ndim == 1:
                Q = Q.reshape(1, -1)
            if A.ndim == 1:
                A = A.reshape(-1)

            node = ClusterNode(
                level=0,
                cluster_id=cid,
                questions=Q,
                answer=A,
                centroid=compute_centroid(Q)
            )
            base_nodes.append(node)

        logger.info(f"Training hierarchical smoothing: {len(base_nodes)} base clusters")

        # Build hierarchy
        self.hierarchy = build_hierarchy(
            base_nodes,
            merge_thresholds=self.merge_thresholds,
            num_levels=self.num_levels
        )

        # Train smoothing at each level
        self.projections = []
        stats = {'levels': []}

        for level_idx, level_clusters in enumerate(self.hierarchy):
            logger.info(f"Training level {level_idx} with {len(level_clusters)} clusters")

            # Prepare data for smoothing basis
            cluster_data = [
                (c.questions, c.answer.reshape(1, -1) if c.answer.ndim == 1 else c.answer)
                for c in level_clusters
            ]

            # Train smoothing
            smoother = SmoothingBasisProjection(
                num_basis=min(self.num_basis, len(level_clusters)),
                cosine_weight=0.5
            )

            if len(cluster_data) > 1:
                losses = smoother.train(
                    cluster_data,
                    num_iterations=num_iterations,
                    log_interval=num_iterations  # Only log final
                )
                final_loss = losses[-1] if losses else 0.0
            else:
                final_loss = 0.0

            self.projections.append(smoother)

            stats['levels'].append({
                'level': level_idx,
                'num_clusters': len(level_clusters),
                'final_loss': final_loss
            })

        # Default level weights if not specified
        if self.level_weights is None:
            # More weight on lower (finer) levels
            num_levels = len(self.hierarchy)
            self.level_weights = [1.0 / (i + 1) for i in range(num_levels)]
            total = sum(self.level_weights)
            self.level_weights = [w / total for w in self.level_weights]

        logger.info(f"Training complete. Level weights: {self.level_weights}")

        return stats

    def project(self, query_emb: np.ndarray,
                temperature: float = 0.1,
                use_levels: Optional[List[int]] = None) -> np.ndarray:
        """
        Project query using multi-level smoothing.

        Args:
            query_emb: Query embedding (d,)
            temperature: Softmax temperature for routing
            use_levels: Specific levels to use (None = all)

        Returns:
            Projected embedding (d,)
        """
        if len(self.projections) == 0:
            return query_emb

        if use_levels is None:
            use_levels = list(range(len(self.projections)))

        # Get projection from each level
        projections = []
        weights = []

        for level_idx in use_levels:
            if level_idx < len(self.projections):
                proj = self.projections[level_idx].project(query_emb, temperature)
                projections.append(proj)
                weights.append(self.level_weights[level_idx])

        if not projections:
            return query_emb

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Combine projections
        result = np.zeros_like(query_emb)
        for proj, w in zip(projections, weights):
            result += w * proj

        return result

    def project_at_level(self, query_emb: np.ndarray, level: int,
                         temperature: float = 0.1) -> np.ndarray:
        """Project using only a specific hierarchy level."""
        if level >= len(self.projections):
            return query_emb
        return self.projections[level].project(query_emb, temperature)


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=== Hierarchical Smoothing Test ===\n")

    np.random.seed(42)
    d = 64
    N = 20  # Base clusters

    # Create clusters with hierarchical structure
    clusters = []
    for i in range(N):
        # 4 groups of 5 clusters each
        group = i // 5
        n_q = np.random.randint(1, 4)

        base = np.random.randn(d) * 0.2 + group * 0.8
        Q = base + np.random.randn(n_q, d) * 0.1
        A = base + np.random.randn(d) * 0.05

        clusters.append((Q, A, f"cluster_{i}"))

    print(f"Created {N} base clusters in 4 groups")

    # Train hierarchical smoothing
    hs = HierarchicalSmoothing(
        num_levels=3,
        merge_thresholds=[0.4, 0.7],
        num_basis=3
    )

    stats = hs.train(clusters, num_iterations=30)

    print(f"\nHierarchy stats:")
    for level_stats in stats['levels']:
        print(f"  Level {level_stats['level']}: {level_stats['num_clusters']} clusters, "
              f"loss={level_stats['final_loss']:.4f}")

    # Test projection
    print("\nProjection test:")
    test_query = np.random.randn(d)

    proj_all = hs.project(test_query)
    proj_l0 = hs.project_at_level(test_query, 0)
    proj_l1 = hs.project_at_level(test_query, 1)

    print(f"  Query norm: {np.linalg.norm(test_query):.3f}")
    print(f"  Multi-level projection norm: {np.linalg.norm(proj_all):.3f}")
    print(f"  Level 0 (fine) norm: {np.linalg.norm(proj_l0):.3f}")
    print(f"  Level 1 (coarse) norm: {np.linalg.norm(proj_l1):.3f}")

    # Similarity between levels
    cos_01 = np.dot(proj_l0, proj_l1) / (np.linalg.norm(proj_l0) * np.linalg.norm(proj_l1) + 1e-8)
    print(f"  Cosine sim (L0 vs L1): {cos_01:.3f}")

    print("\n=== Test Complete ===")
