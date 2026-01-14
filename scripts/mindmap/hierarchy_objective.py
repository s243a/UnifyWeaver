#!/usr/bin/env python3
"""
Objective function for evaluating and constructing hierarchies.

Combines semantic distance D(T) and entropy gain H(T) to score hierarchy quality.
Supports multiple smoothing methods for both probability weighting and entropy estimation.

Usage:
    # Evaluate existing hierarchy
    python3 hierarchy_objective.py --tree hierarchy.json --embeddings embeds.npy

    # As library
    from hierarchy_objective import HierarchyObjective
    obj = HierarchyObjective(smoothing='dirichlet', alpha=1.0)
    score = obj.compute(tree, embeddings)

References:
    - Information-theoretic clustering (Scholarpedia)
    - Structural entropy for hierarchical clustering (Pan et al. 2025)
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np

# Smoothing methods
SmoothingMethod = Literal['none', 'add_one', 'dirichlet', 'jeffreys']
CombineMethod = Literal['product', 'sum', 'log_product']
ProbabilitySource = Literal['subtree_size', 'density_knn', 'density_kernel', 'logits']


def estimate_density_knn(
    embeddings: np.ndarray,
    k: int = 10,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Estimate probability/density via k-NN distances.

    For embedding-only models (Nomic, MiniLM) where we don't have logits.

    Args:
        embeddings: Node embeddings
        k: Number of neighbors
        metric: Distance metric

    Returns:
        Density estimates (higher = denser region = more probable)
    """
    n = len(embeddings)
    densities = np.zeros(n)

    # Normalize embeddings for cosine
    if metric == 'cosine':
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)

    for i in range(n):
        # Compute distances to all other points
        if metric == 'cosine':
            # Cosine distance = 1 - dot product (for normalized vectors)
            dists = 1 - embeddings @ embeddings[i]
        else:
            dists = np.linalg.norm(embeddings - embeddings[i], axis=1)

        # k-th nearest neighbor distance (excluding self)
        dists[i] = np.inf
        kth_dist = np.partition(dists, k)[k]

        # Density inversely proportional to k-NN distance
        # Add epsilon to avoid division by zero
        densities[i] = 1.0 / (kth_dist + 1e-10)

    # Normalize to probabilities
    return densities / densities.sum()


def estimate_density_kernel(
    embeddings: np.ndarray,
    bandwidth: Optional[float] = None,
    kernel: str = 'gaussian'
) -> np.ndarray:
    """
    Estimate probability/density via kernel density estimation.

    Args:
        embeddings: Node embeddings
        bandwidth: Kernel bandwidth (auto if None)
        kernel: Kernel type ('gaussian' or 'epanechnikov')

    Returns:
        Density estimates
    """
    n, d = embeddings.shape

    # Auto bandwidth: Silverman's rule
    if bandwidth is None:
        std = np.std(embeddings, axis=0).mean()
        bandwidth = std * (4 / (d + 2) / n) ** (1 / (d + 4))

    densities = np.zeros(n)

    for i in range(n):
        dists = np.linalg.norm(embeddings - embeddings[i], axis=1)

        if kernel == 'gaussian':
            weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
        elif kernel == 'epanechnikov':
            u = dists / bandwidth
            weights = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        densities[i] = weights.sum() / n

    # Normalize to probabilities
    return densities / densities.sum()


@dataclass
class HierarchyStats:
    """Statistics computed for a hierarchy."""
    semantic_distance: float      # D(T) - avg distance to parent
    semantic_distance_raw: float  # Before normalization
    entropy_gain: float           # H(T) - information gain between levels
    entropy_gain_raw: float       # Before normalization
    objective: float              # Combined objective J(T)
    n_nodes: int
    n_levels: int
    level_stats: Dict[int, dict]  # Per-level statistics


class HierarchyObjective:
    """
    Compute hierarchy quality objective combining semantic distance and entropy.

    The objective J(T) balances:
    - D(T): Semantic distance (want small - tight clusters)
    - H(T): Entropy gain between levels (want large - informative splits)

    Supports smoothing for robust estimation from finite samples.

    Probability Source Options:
        Different embedding models provide different ways to estimate probability:

        - subtree_size: Use branch size as probability proxy (model-agnostic)
        - density_knn: Estimate from k-NN distances (for Nomic, MiniLM)
        - density_kernel: KDE-based estimation (for any embedding model)
        - logits: Use model output logits (for ModernBERT with classification head)

        Note: Nomic and MiniLM are pure embedding models - they don't output
        logits/probabilities directly. ModernBERT with MLM/classification head
        CAN provide logit-based probabilities via softmax(logits).

        For embedding-only models, density estimation is the only option for
        "model-aware" probability. Otherwise, use subtree_size which is
        purely structural.
    """

    def __init__(
        self,
        smoothing: SmoothingMethod = 'dirichlet',
        alpha: float = 1.0,
        combine: CombineMethod = 'product',
        use_probability_weight: bool = True,
        probability_source: ProbabilitySource = 'subtree_size',
        knn_k: int = 10,
        kernel_bandwidth: Optional[float] = None,
        entropy_smoothing: SmoothingMethod = 'dirichlet',
        entropy_alpha: float = 1.0,
    ):
        """
        Initialize the objective function.

        Args:
            smoothing: Smoothing method for probability weights
                - 'none': No smoothing (raw counts/probabilities)
                - 'add_one': Laplace smoothing (+1 to all counts)
                - 'dirichlet': Dirichlet prior with concentration alpha
                - 'jeffreys': Jeffreys prior (alpha=0.5)
            alpha: Concentration parameter for Dirichlet smoothing
            combine: How to combine D and H
                - 'product': J = D̂(1 - Ĥ) - both must be good
                - 'sum': J = D̂ - λĤ - allows trade-offs
                - 'log_product': log(J) = log(D̂) + log(1-Ĥ)
            use_probability_weight: Weight nodes by subtree probability
            probability_source: Where to get probability estimates
                - 'subtree_size': Branch size (structural, model-agnostic)
                - 'density_knn': k-NN density (for Nomic, MiniLM)
                - 'density_kernel': KDE (any embedding model)
                - 'logits': Model output logits (ModernBERT only)
            knn_k: k for k-NN density estimation
            kernel_bandwidth: Bandwidth for KDE (auto if None)
            entropy_smoothing: Smoothing method for entropy estimation
            entropy_alpha: Alpha for entropy smoothing
        """
        self.smoothing = smoothing
        self.alpha = alpha
        self.combine = combine
        self.use_probability_weight = use_probability_weight
        self.probability_source = probability_source
        self.knn_k = knn_k
        self.kernel_bandwidth = kernel_bandwidth
        self.entropy_smoothing = entropy_smoothing
        self.entropy_alpha = entropy_alpha
        self._density_cache = None  # Cache for density estimates

    def get_node_probabilities(
        self,
        embeddings: np.ndarray,
        node_to_idx: Dict[str, int],
        subtree_sizes: Dict[str, int],
        logits: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Get probability estimates for each node.

        Args:
            embeddings: Node embeddings
            node_to_idx: Map node ID to embedding index
            subtree_sizes: Subtree size for each node
            logits: Optional logit outputs (for ModernBERT)

        Returns:
            Dict mapping node_id to probability estimate
        """
        if self.probability_source == 'subtree_size':
            # Use subtree size as probability proxy
            total = sum(subtree_sizes.values())
            return {
                node_id: size / total
                for node_id, size in subtree_sizes.items()
            }

        elif self.probability_source == 'density_knn':
            # Estimate from k-NN distances
            if self._density_cache is None:
                self._density_cache = estimate_density_knn(
                    embeddings, k=self.knn_k, metric='cosine'
                )
            densities = self._density_cache
            return {
                node_id: densities[idx]
                for node_id, idx in node_to_idx.items()
            }

        elif self.probability_source == 'density_kernel':
            # KDE-based estimation
            if self._density_cache is None:
                self._density_cache = estimate_density_kernel(
                    embeddings, bandwidth=self.kernel_bandwidth
                )
            densities = self._density_cache
            return {
                node_id: densities[idx]
                for node_id, idx in node_to_idx.items()
            }

        elif self.probability_source == 'logits':
            # Use model logits (must be provided)
            if logits is None:
                raise ValueError(
                    "probability_source='logits' requires logits array. "
                    "This is only available for models with classification heads "
                    "(e.g., ModernBERT). For Nomic/MiniLM, use 'density_knn' or "
                    "'density_kernel' instead."
                )
            # Softmax to convert logits to probabilities
            exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            # Use max probability per token as node probability
            node_probs = probs.max(axis=-1)
            node_probs = node_probs / node_probs.sum()
            return {
                node_id: node_probs[idx]
                for node_id, idx in node_to_idx.items()
            }

        else:
            raise ValueError(f"Unknown probability_source: {self.probability_source}")

    def smooth_counts(
        self,
        counts: np.ndarray,
        method: SmoothingMethod,
        alpha: float
    ) -> np.ndarray:
        """
        Apply smoothing to counts/probabilities.

        Args:
            counts: Raw counts or probabilities
            method: Smoothing method
            alpha: Concentration parameter

        Returns:
            Smoothed probabilities (normalized to sum to 1)
        """
        counts = np.asarray(counts, dtype=float)
        k = len(counts)

        if method == 'none':
            # No smoothing - just normalize
            total = counts.sum()
            if total == 0:
                return np.ones(k) / k
            return counts / total

        elif method == 'add_one':
            # Laplace smoothing: (c_i + 1) / (N + k)
            smoothed = counts + 1
            return smoothed / smoothed.sum()

        elif method == 'dirichlet':
            # Dirichlet prior: (c_i + α) / (N + kα)
            smoothed = counts + alpha
            return smoothed / smoothed.sum()

        elif method == 'jeffreys':
            # Jeffreys prior: (c_i + 0.5) / (N + k/2)
            smoothed = counts + 0.5
            return smoothed / smoothed.sum()

        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def compute_entropy(
        self,
        counts: np.ndarray,
        smoothing: Optional[SmoothingMethod] = None,
        alpha: Optional[float] = None
    ) -> float:
        """
        Compute Shannon entropy with optional smoothing.

        Args:
            counts: Counts or probabilities for each category
            smoothing: Override default smoothing method
            alpha: Override default alpha

        Returns:
            Entropy in nats (natural log)
        """
        method = smoothing if smoothing is not None else self.entropy_smoothing
        a = alpha if alpha is not None else self.entropy_alpha

        probs = self.smooth_counts(counts, method, a)

        # H = -sum(p * log(p)), handle p=0
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    def compute_mutual_information(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray
    ) -> float:
        """
        Compute mutual information between parent and child level labels.

        I(X;Y) = H(X) + H(Y) - H(X,Y)

        Measures how much knowing the child cluster tells you about the parent.
        High MI = informative refinement.
        """
        # Count joint occurrences
        n = len(parent_labels)
        parent_unique = np.unique(parent_labels)
        child_unique = np.unique(child_labels)

        # Marginal counts
        parent_counts = np.array([np.sum(parent_labels == p) for p in parent_unique])
        child_counts = np.array([np.sum(child_labels == c) for c in child_unique])

        # Joint counts
        joint_counts = []
        for p in parent_unique:
            for c in child_unique:
                joint_counts.append(np.sum((parent_labels == p) & (child_labels == c)))
        joint_counts = np.array(joint_counts)

        # Compute entropies with smoothing
        H_parent = self.compute_entropy(parent_counts)
        H_child = self.compute_entropy(child_counts)
        H_joint = self.compute_entropy(joint_counts)

        # MI = H(X) + H(Y) - H(X,Y)
        return H_parent + H_child - H_joint

    def build_tree_structure(
        self,
        tree: Dict,
        embeddings: np.ndarray
    ) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, List[str]]]:
        """
        Extract tree structure from hierarchy dict.

        Returns:
            node_to_idx: Map node ID to embedding index
            parent_of: Map node ID to parent ID
            children_of: Map node ID to list of child IDs
        """
        node_to_idx = {}
        parent_of = {}
        children_of = defaultdict(list)

        def traverse(node, parent_id=None):
            node_id = node.get('id') or node.get('tree_id') or str(id(node))

            # Get embedding index if available
            if 'embedding_idx' in node:
                node_to_idx[node_id] = node['embedding_idx']
            elif 'idx' in node:
                node_to_idx[node_id] = node['idx']

            if parent_id is not None:
                parent_of[node_id] = parent_id
                children_of[parent_id].append(node_id)

            for child in node.get('children', []):
                traverse(child, node_id)

        if isinstance(tree, dict):
            if 'root' in tree:
                traverse(tree['root'])
            else:
                traverse(tree)
        elif isinstance(tree, list):
            # Forest - multiple roots
            for root in tree:
                traverse(root)

        return node_to_idx, parent_of, dict(children_of)

    def compute_level_assignments(
        self,
        parent_of: Dict[str, str],
        children_of: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """
        Assign each node to a level (depth from root).
        """
        levels = {}

        # Find roots (nodes with no parent)
        all_nodes = set(parent_of.keys()) | set(children_of.keys())
        roots = all_nodes - set(parent_of.keys())

        def assign_level(node_id, level):
            levels[node_id] = level
            for child_id in children_of.get(node_id, []):
                assign_level(child_id, level + 1)

        for root in roots:
            assign_level(root, 0)

        return levels

    def compute_subtree_sizes(
        self,
        children_of: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """
        Compute size of subtree rooted at each node.
        """
        sizes = {}

        def compute_size(node_id):
            children = children_of.get(node_id, [])
            if not children:
                sizes[node_id] = 1
            else:
                sizes[node_id] = 1 + sum(compute_size(c) for c in children)
            return sizes[node_id]

        # Find roots
        all_nodes = set(children_of.keys())
        for children in children_of.values():
            all_nodes.update(children)
        roots = all_nodes - set(c for children in children_of.values() for c in children)

        for root in roots:
            compute_size(root)

        return sizes

    def compute_semantic_distance(
        self,
        tree: Dict,
        embeddings: np.ndarray,
        node_to_idx: Dict[str, int],
        parent_of: Dict[str, str],
        subtree_sizes: Dict[str, int]
    ) -> Tuple[float, Dict[int, float]]:
        """
        Compute average semantic distance D(T).

        D = weighted average of cosine distance from each node to its parent.

        Returns:
            (overall_D, per_level_D)
        """
        distances = []
        weights = []
        level_distances = defaultdict(list)
        level_weights = defaultdict(list)

        levels = self.compute_level_assignments(parent_of, {})

        for node_id, parent_id in parent_of.items():
            if node_id not in node_to_idx or parent_id not in node_to_idx:
                continue

            node_emb = embeddings[node_to_idx[node_id]]
            parent_emb = embeddings[node_to_idx[parent_id]]

            # Cosine distance
            cos_sim = np.dot(node_emb, parent_emb) / (
                np.linalg.norm(node_emb) * np.linalg.norm(parent_emb) + 1e-10
            )
            dist = 1 - cos_sim

            # Weight by subtree size (probability proxy)
            if self.use_probability_weight:
                weight = subtree_sizes.get(node_id, 1)
            else:
                weight = 1

            distances.append(dist)
            weights.append(weight)

            level = levels.get(node_id, 0)
            level_distances[level].append(dist)
            level_weights[level].append(weight)

        if not distances:
            return 0.0, {}

        # Apply smoothing to weights
        weights = self.smooth_counts(np.array(weights), self.smoothing, self.alpha)

        overall_D = np.average(distances, weights=weights)

        per_level_D = {}
        for level in level_distances:
            if level_distances[level]:
                w = self.smooth_counts(
                    np.array(level_weights[level]),
                    self.smoothing,
                    self.alpha
                )
                per_level_D[level] = np.average(level_distances[level], weights=w)

        return overall_D, per_level_D

    def compute_entropy_gain(
        self,
        parent_of: Dict[str, str],
        children_of: Dict[str, List[str]],
        subtree_sizes: Dict[str, int]
    ) -> Tuple[float, Dict[int, float]]:
        """
        Compute entropy gain H(T) between levels.

        Measures how informative the refinement is at each level.
        Uses mutual information between parent and child cluster assignments.

        Returns:
            (overall_H, per_level_H)
        """
        levels = self.compute_level_assignments(parent_of, children_of)

        if not levels:
            return 0.0, {}

        max_level = max(levels.values())

        per_level_H = {}
        total_H = 0.0
        n_transitions = 0

        for level in range(max_level):
            # Get nodes at this level and next level
            nodes_this = [n for n, l in levels.items() if l == level]
            nodes_next = [n for n, l in levels.items() if l == level + 1]

            if not nodes_this or not nodes_next:
                continue

            # Create label arrays
            # Parent label = which node at level k
            # Child label = which node at level k+1
            parent_labels = []
            child_labels = []

            for child_id in nodes_next:
                if child_id in parent_of:
                    parent_id = parent_of[child_id]
                    if parent_id in nodes_this:
                        parent_labels.append(nodes_this.index(parent_id))
                        child_labels.append(nodes_next.index(child_id))

            if len(parent_labels) < 2:
                continue

            # Compute mutual information
            mi = self.compute_mutual_information(
                np.array(parent_labels),
                np.array(child_labels)
            )

            per_level_H[level] = mi
            total_H += mi
            n_transitions += 1

        overall_H = total_H / n_transitions if n_transitions > 0 else 0.0

        return overall_H, per_level_H

    def normalize_to_01(
        self,
        values: List[float],
        value: float
    ) -> float:
        """Normalize a value to [0,1] based on observed range."""
        if not values:
            return 0.5

        v_min = min(values)
        v_max = max(values)

        if v_max - v_min < 1e-10:
            return 0.5

        return (value - v_min) / (v_max - v_min)

    def compute(
        self,
        tree: Dict,
        embeddings: np.ndarray,
        reference_stats: Optional[Dict] = None
    ) -> HierarchyStats:
        """
        Compute the full hierarchy objective.

        Args:
            tree: Hierarchy structure dict
            embeddings: Node embeddings array
            reference_stats: Optional stats from baseline for normalization

        Returns:
            HierarchyStats with all computed metrics
        """
        # Extract structure
        node_to_idx, parent_of, children_of = self.build_tree_structure(tree, embeddings)
        subtree_sizes = self.compute_subtree_sizes(children_of)
        levels = self.compute_level_assignments(parent_of, children_of)

        # Compute D and H
        D_raw, D_per_level = self.compute_semantic_distance(
            tree, embeddings, node_to_idx, parent_of, subtree_sizes
        )
        H_raw, H_per_level = self.compute_entropy_gain(
            parent_of, children_of, subtree_sizes
        )

        # Normalize
        if reference_stats:
            D_hat = self.normalize_to_01(
                [reference_stats.get('D_min', 0), reference_stats.get('D_max', 1)],
                D_raw
            )
            H_hat = self.normalize_to_01(
                [reference_stats.get('H_min', 0), reference_stats.get('H_max', 1)],
                H_raw
            )
        else:
            # Use raw values scaled by typical ranges
            D_hat = min(1.0, D_raw / 0.5)  # Assume max distance ~0.5
            H_hat = min(1.0, H_raw / 2.0)  # Assume max entropy ~2.0 nats

        # Combine
        eps = 1e-10
        if self.combine == 'product':
            # J = D̂(1 - Ĥ) - minimize
            # Small when D small AND H large
            objective = D_hat * (1 - H_hat)
        elif self.combine == 'sum':
            # J = D̂ - Ĥ - minimize
            objective = D_hat - H_hat
        elif self.combine == 'log_product':
            # log(J) = log(D̂) + log(1 - Ĥ)
            objective = np.log(D_hat + eps) + np.log(1 - H_hat + eps)
        else:
            raise ValueError(f"Unknown combine method: {self.combine}")

        # Compile level stats
        level_stats = {}
        for level in set(D_per_level.keys()) | set(H_per_level.keys()):
            level_stats[level] = {
                'semantic_distance': D_per_level.get(level, 0.0),
                'entropy_gain': H_per_level.get(level, 0.0),
                'n_nodes': sum(1 for n, l in levels.items() if l == level)
            }

        return HierarchyStats(
            semantic_distance=D_hat,
            semantic_distance_raw=D_raw,
            entropy_gain=H_hat,
            entropy_gain_raw=H_raw,
            objective=objective,
            n_nodes=len(node_to_idx),
            n_levels=max(levels.values()) + 1 if levels else 0,
            level_stats=level_stats
        )


def evaluate_hierarchy(
    tree_path: Path,
    embeddings_path: Path,
    smoothing: SmoothingMethod = 'dirichlet',
    alpha: float = 1.0,
    combine: CombineMethod = 'product',
    verbose: bool = True
) -> HierarchyStats:
    """
    Evaluate a hierarchy from files.

    Args:
        tree_path: Path to hierarchy JSON
        embeddings_path: Path to embeddings .npy file
        smoothing: Smoothing method
        alpha: Smoothing parameter
        combine: How to combine D and H
        verbose: Print detailed output

    Returns:
        HierarchyStats
    """
    # Load data
    with open(tree_path) as f:
        tree = json.load(f)

    embeddings = np.load(embeddings_path)

    # Compute
    obj = HierarchyObjective(
        smoothing=smoothing,
        alpha=alpha,
        combine=combine
    )

    stats = obj.compute(tree, embeddings)

    if verbose:
        print(f"Hierarchy Statistics:")
        print(f"  Nodes: {stats.n_nodes}")
        print(f"  Levels: {stats.n_levels}")
        print(f"  Semantic Distance D: {stats.semantic_distance:.4f} (raw: {stats.semantic_distance_raw:.4f})")
        print(f"  Entropy Gain H: {stats.entropy_gain:.4f} (raw: {stats.entropy_gain_raw:.4f})")
        print(f"  Objective J: {stats.objective:.4f}")
        print(f"\nPer-level stats:")
        for level, lstats in sorted(stats.level_stats.items()):
            print(f"  Level {level}: D={lstats['semantic_distance']:.4f}, "
                  f"H={lstats['entropy_gain']:.4f}, n={lstats['n_nodes']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate hierarchy quality objective',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--tree', '-t', type=Path, required=True,
                        help='Path to hierarchy JSON file')
    parser.add_argument('--embeddings', '-e', type=Path, required=True,
                        help='Path to embeddings .npy file')
    parser.add_argument('--smoothing', '-s',
                        choices=['none', 'add_one', 'dirichlet', 'jeffreys'],
                        default='dirichlet',
                        help='Smoothing method (default: dirichlet)')
    parser.add_argument('--alpha', '-a', type=float, default=1.0,
                        help='Smoothing alpha parameter (default: 1.0)')
    parser.add_argument('--combine', '-c',
                        choices=['product', 'sum', 'log_product'],
                        default='product',
                        help='How to combine D and H (default: product)')
    parser.add_argument('--no-probability-weight', action='store_true',
                        help='Disable probability weighting')
    parser.add_argument('--probability-source', '-p',
                        choices=['subtree_size', 'density_knn', 'density_kernel', 'logits'],
                        default='subtree_size',
                        help='Source for probability estimates (default: subtree_size). '
                             'Use density_knn or density_kernel for Nomic/MiniLM. '
                             'logits only works with ModernBERT.')
    parser.add_argument('--knn-k', type=int, default=10,
                        help='k for k-NN density estimation (default: 10)')
    parser.add_argument('--kernel-bandwidth', type=float, default=None,
                        help='Bandwidth for KDE (auto if not specified)')
    parser.add_argument('--entropy-smoothing',
                        choices=['none', 'add_one', 'dirichlet', 'jeffreys'],
                        default='dirichlet',
                        help='Smoothing for entropy estimation (default: dirichlet)')
    parser.add_argument('--entropy-alpha', type=float, default=1.0,
                        help='Alpha for entropy smoothing (default: 1.0)')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output JSON file for stats')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Build objective
    obj = HierarchyObjective(
        smoothing=args.smoothing,
        alpha=args.alpha,
        combine=args.combine,
        use_probability_weight=not args.no_probability_weight,
        probability_source=args.probability_source,
        knn_k=args.knn_k,
        kernel_bandwidth=args.kernel_bandwidth,
        entropy_smoothing=args.entropy_smoothing,
        entropy_alpha=args.entropy_alpha
    )

    # Load data
    with open(args.tree) as f:
        tree = json.load(f)
    embeddings = np.load(args.embeddings)

    # Compute stats
    stats = obj.compute(tree, embeddings)

    if not args.quiet:
        print(f"Hierarchy Statistics:")
        print(f"  Nodes: {stats.n_nodes}")
        print(f"  Levels: {stats.n_levels}")
        print(f"  Semantic Distance D: {stats.semantic_distance:.4f} (raw: {stats.semantic_distance_raw:.4f})")
        print(f"  Entropy Gain H: {stats.entropy_gain:.4f} (raw: {stats.entropy_gain_raw:.4f})")
        print(f"  Objective J: {stats.objective:.4f}")
        print(f"\nPer-level stats:")
        for level, lstats in sorted(stats.level_stats.items()):
            print(f"  Level {level}: D={lstats['semantic_distance']:.4f}, "
                  f"H={lstats['entropy_gain']:.4f}, n={lstats['n_nodes']}")

    if args.output:
        output = {
            'semantic_distance': stats.semantic_distance,
            'semantic_distance_raw': stats.semantic_distance_raw,
            'entropy_gain': stats.entropy_gain,
            'entropy_gain_raw': stats.entropy_gain_raw,
            'objective': stats.objective,
            'n_nodes': stats.n_nodes,
            'n_levels': stats.n_levels,
            'level_stats': {str(k): v for k, v in stats.level_stats.items()},
            'config': {
                'smoothing': args.smoothing,
                'alpha': args.alpha,
                'combine': args.combine,
                'probability_source': args.probability_source,
                'use_probability_weight': not args.no_probability_weight,
                'knn_k': args.knn_k,
                'kernel_bandwidth': args.kernel_bandwidth,
                'entropy_smoothing': args.entropy_smoothing,
                'entropy_alpha': args.entropy_alpha
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nStats written to: {args.output}")


if __name__ == '__main__':
    main()
