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
EntropySource = Literal['fisher', 'logits']
EntropyTextSource = Literal['raw_phrase', 'embedding_text', 'mixed']
DiagnosticSource = Literal['none', 'logits', 'density']

# Optional transformer support for computing logits from text
_transformer_model = None
_transformer_tokenizer = None


def load_transformer_model(model_name: str = "answerdotai/ModernBERT-base"):
    """
    Load a transformer model for computing logits from text.

    Args:
        model_name: HuggingFace model name (default: ModernBERT-base)

    Returns:
        (model, tokenizer) tuple
    """
    global _transformer_model, _transformer_tokenizer

    if _transformer_model is not None:
        return _transformer_model, _transformer_tokenizer

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError(
            "transformers and torch required for logits computation. "
            "Install with: pip install transformers torch"
        )

    _transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _transformer_model = AutoModelForMaskedLM.from_pretrained(model_name)
    _transformer_model.eval()

    return _transformer_model, _transformer_tokenizer


def compute_logits_from_text(
    texts: List[str],
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 32
) -> np.ndarray:
    """
    Compute logits from text using a transformer model.

    Args:
        texts: List of text strings (one per node)
        model_name: HuggingFace model name
        batch_size: Batch size for inference

    Returns:
        Logits array of shape [n_texts, seq_len, vocab_size]
        or [n_texts, vocab_size] if using CLS token only
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch required. Install with: pip install torch")

    model, tokenizer = load_transformer_model(model_name)

    all_logits = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Forward pass
            outputs = model(**inputs)

            # Get logits - shape [batch, seq_len, vocab_size]
            batch_logits = outputs.logits.numpy()

            # Average over sequence length to get per-node logits
            # Or use CLS token position (index 0)
            # Using mean for now - could be configurable
            mean_logits = batch_logits.mean(axis=1)  # [batch, vocab_size]
            all_logits.append(mean_logits)

    return np.vstack(all_logits)


def compute_entropy_from_text(
    texts: List[str],
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 32
) -> np.ndarray:
    """
    Compute Shannon entropy for each text using a transformer model.

    Lower entropy = more predictable/common concept = should be higher in hierarchy
    Higher entropy = less predictable/specific concept = should be deeper

    Args:
        texts: List of text strings (one per node)
        model_name: HuggingFace model name
        batch_size: Batch size for inference

    Returns:
        Entropy array of shape [n_texts]
    """
    logits = compute_logits_from_text(texts, model_name, batch_size)

    # Softmax to get probabilities
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)

    return entropy


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
    # Depth-probability alignment (if logits available)
    depth_surprisal_correlation: Optional[float] = None  # corr(depth, -log(p))
    depth_surprisal_slope: Optional[float] = None        # Does surprisal increase with depth?


class HierarchyObjective:
    """
    Compute hierarchy quality objective combining semantic distance and entropy.

    The objective J(T) balances:
    - D(T): Semantic distance (want small - tight clusters)
    - H(T): Entropy gain between levels (want large - informative splits)

    Supports smoothing for robust estimation from finite samples.

    Decoupled Model Architecture:
        The embedding model (for distance D) and entropy model (for H) are
        INDEPENDENT. This allows using:

        - Nomic/MiniLM embeddings for semantic distance computation
        - ModernBERT for entropy/probability estimation from text

        Example:
            obj = HierarchyObjective(
                entropy_source='logits',        # Use actual entropy, not proxy
                entropy_model='answerdotai/ModernBERT-base'
            )
            stats = obj.compute(
                tree,
                nomic_embeddings,               # Distances from Nomic
                texts={'node1': 'machine learning', ...}  # Entropy from text
            )

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

    Entropy Text Source Options:
        When using logits-based entropy, choose what text to feed to the model:

        - raw_phrase: Just the topic name (e.g., "machine learning")
            Captures inherent concept specificity
        - embedding_text: Full text used for embedding (may include context)
            Captures specificity in context
        - mixed: Average of both
            Balances inherent specificity with contextual specificity
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
        depth_normalize: bool = True,
        depth_decay: float = 0.5,
        depth_scale_mode: str = 'exponential',
        entropy_source: EntropySource = 'fisher',
        entropy_text_source: EntropyTextSource = 'raw_phrase',
        entropy_model: str = "answerdotai/ModernBERT-base",
        entropy_diagnostic_source: DiagnosticSource = 'none',
        min_probability_threshold: float = 0.0,
        min_samples_for_std: int = 30,
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
            depth_normalize: Normalize distances by depth-expected values
                At higher depths (more specific), expect tighter clustering
            depth_decay: λ controlling how fast expected distance shrinks with depth
                D_expected(d) = D_base * exp(-λd)
                Higher λ = stricter requirements at depth (default 0.5)
            depth_scale_mode: How to compute depth penalty for distances
                - 'exponential': penalty = exp(λd) - same distance penalized more at depth
                NOTE: 'level_std' and 'subtree_std' are for kernel bandwidth selection
                in density manifold computation, not for distance normalization here.
                See estimate_density_kernel() and kernel_methods_flux.md.
            entropy_source: How to compute H (entropy) for the objective function J = D/(1+H)
                - 'fisher': geometric proxy using between/within cluster variance
                - 'logits': actual entropy from transformer model output logits
                  Computes per-node entropy, aggregates by level, uses entropy
                  gain (slope of entropy vs depth) as H in the objective.
            entropy_text_source: What text to use for logits-based entropy computation
                - 'raw_phrase': Just the topic/node name (e.g., "machine learning")
                - 'embedding_text': The full text used for embedding (may include context)
                - 'mixed': Average of both (balance inherent specificity with context)
                NOTE: Entropy model is independent of embedding model. You can use
                Nomic for distances (D) while using ModernBERT for entropy (H).
            entropy_model: HuggingFace model name for logits-based entropy
                Default: "answerdotai/ModernBERT-base"
                Any transformer with a masked LM head can work (BERT, RoBERTa, etc.)
            entropy_diagnostic_source: Independent sanity check for depth-surprisal correlation
                - 'none': No diagnostic (default)
                - 'logits': Compute depth-surprisal correlation from transformer logits
                - 'density': Compute from embedding density estimates
                This is separate from entropy_source - allows checking that depth
                tracks probability regardless of which H is used in the objective.
            min_probability_threshold: Prune subtrees with cumulative probability below this
                When probability-weighted, small branches contribute negligibly.
                Default 0.0 (no pruning). Typical value: 0.01 (prune <1% branches).
            min_samples_for_std: Minimum samples needed for stable std estimate (default 30)
                Once enough samples collected at a level, can stop traversing negligible branches.
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
        self.depth_normalize = depth_normalize
        self.depth_decay = depth_decay
        self.depth_scale_mode = depth_scale_mode
        self.entropy_source = entropy_source
        self.entropy_text_source = entropy_text_source
        self.entropy_model = entropy_model
        self.entropy_diagnostic_source = entropy_diagnostic_source
        self.min_probability_threshold = min_probability_threshold
        self.min_samples_for_std = min_samples_for_std
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

    def compute_entropy_from_logits(
        self,
        logits: np.ndarray
    ) -> np.ndarray:
        """
        Compute Shannon entropy from model output logits.

        NOTE: The text used to generate logits affects entropy estimates:
        - Raw phrase (e.g., "machine learning"): inherent concept specificity
        - Embedding text (with context): specificity in context
        - Mix: balance both signals

        Longer/contextualized text may have higher entropy due to more tokens.
        Raw phrases may be too short for reliable estimates.
        The choice should match how embeddings were computed.

        Args:
            logits: Shape [n_nodes, seq_len, vocab_size] or [n_nodes, vocab_size]

        Returns:
            Per-node entropy values
        """
        # Softmax to get probabilities
        if logits.ndim == 3:
            # [n_nodes, seq_len, vocab_size] - average over sequence
            exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            # Per-token entropy, then average over sequence
            token_entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
            return token_entropy.mean(axis=-1)
        else:
            # [n_nodes, vocab_size]
            exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            return -np.sum(probs * np.log(probs + 1e-10), axis=-1)

    def compute_depth_probability_correlation(
        self,
        levels: Dict[str, int],
        node_probs: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Compute correlation between depth and surprisal (-log p).

        A good hierarchy should have: depth(node) ∝ -log(p(node))
        - High probability (common) concepts near root
        - Low probability (specific) concepts at leaves

        Returns:
            (correlation, mean_surprisal_per_depth_ratio)
            correlation: Pearson correlation between depth and surprisal
            ratio: How well depth tracks surprisal (1.0 = perfect)
        """
        depths = []
        surprisals = []

        for node_id, level in levels.items():
            if node_id in node_probs and node_probs[node_id] > 0:
                depths.append(level)
                surprisals.append(-np.log(node_probs[node_id]))

        if len(depths) < 2:
            return 0.0, 0.0

        depths = np.array(depths)
        surprisals = np.array(surprisals)

        # Pearson correlation
        if np.std(depths) > 0 and np.std(surprisals) > 0:
            correlation = np.corrcoef(depths, surprisals)[0, 1]
        else:
            correlation = 0.0

        # Mean surprisal per depth level
        unique_depths = np.unique(depths)
        if len(unique_depths) > 1:
            mean_surprisal_by_depth = []
            for d in sorted(unique_depths):
                mask = depths == d
                mean_surprisal_by_depth.append(surprisals[mask].mean())
            # Check if surprisal increases with depth (should be positive slope)
            slope = np.polyfit(range(len(mean_surprisal_by_depth)), mean_surprisal_by_depth, 1)[0]
            ratio = slope / (np.mean(surprisals) + 1e-10)  # Normalized slope
        else:
            ratio = 0.0

        return correlation, ratio

    def compute_entropy_gain_from_logits(
        self,
        levels: Dict[str, int],
        node_entropies: Dict[str, float]
    ) -> Tuple[float, Dict[int, float]]:
        """
        Compute entropy gain H for the objective using per-node entropies from logits.

        H represents how much entropy (specificity) increases with depth.
        A good hierarchy has increasing entropy at deeper levels (more specific concepts).

        The entropy gain is the slope of mean entropy vs depth, normalized to be
        comparable with Fisher entropy scale.

        Args:
            levels: Dict mapping node_id to depth level
            node_entropies: Dict mapping node_id to Shannon entropy from logits

        Returns:
            (overall_H, per_level_H) where H is entropy gain (higher = better hierarchy)
        """
        if not levels or not node_entropies:
            return 0.0, {}

        # Group entropies by level
        entropy_by_level = defaultdict(list)
        for node_id, level in levels.items():
            if node_id in node_entropies:
                entropy_by_level[level].append(node_entropies[node_id])

        if len(entropy_by_level) < 2:
            return 0.0, {}

        # Compute mean entropy per level
        per_level_H = {}
        level_means = []
        level_indices = []

        for level in sorted(entropy_by_level.keys()):
            entropies = entropy_by_level[level]
            mean_entropy = np.mean(entropies)
            per_level_H[level] = mean_entropy
            level_means.append(mean_entropy)
            level_indices.append(level)

        # Entropy gain = slope of entropy vs depth
        # Positive slope means deeper nodes have higher entropy (more specific)
        if len(level_means) >= 2:
            slope, _ = np.polyfit(level_indices, level_means, 1)
            # Normalize: scale so it's comparable to Fisher entropy (~0-2 range)
            # Use log1p to handle negative slopes gracefully
            overall_H = np.log1p(max(0, slope))
        else:
            overall_H = 0.0

        return overall_H, per_level_H

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
        subtree_sizes: Dict[str, int],
        depth_normalize: bool = True,
        depth_decay: float = 0.5,
        depth_scale_mode: str = 'exponential'
    ) -> Tuple[float, Dict[int, float]]:
        """
        Compute average semantic distance D(T).

        D = weighted average of cosine distance from each node to its parent,
        normalized by expected distance at that depth.

        At higher depths (more specific nodes), we expect tighter clustering,
        so the same raw distance is penalized more heavily.

        Args:
            depth_normalize: If True, normalize distances by depth-expected values
            depth_decay: λ parameter for exponential mode (higher = stricter at depth)
            depth_scale_mode: How to compute expected scale at each depth
                - 'exponential': σ(d) = exp(-λd) - simple exponential decay
                - 'subtree_std': σ(d) = std of distances within subtrees at depth d
                - 'level_std': σ(d) = std of all distances at level d

        Returns:
            (overall_D, per_level_D)
        """
        distances = []
        weights = []
        level_distances = defaultdict(list)
        level_weights = defaultdict(list)

        # Build children_of from parent_of for level assignment
        children_of_local = defaultdict(list)
        for child, parent in parent_of.items():
            children_of_local[parent].append(child)
        levels = self.compute_level_assignments(parent_of, dict(children_of_local))

        # Collect raw distances by level and by parent (for subtree_std mode)
        raw_distances_by_level = defaultdict(list)
        raw_distances_by_parent = defaultdict(list)

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
            level = levels.get(node_id, 0)
            raw_distances_by_level[level].append((node_id, parent_id, dist))
            raw_distances_by_parent[parent_id].append(dist)

        if not raw_distances_by_level:
            return 0.0, {}

        # Compute depth penalty
        # For exponential: scale(d) = exp(-λd), so D_normalized = D_raw / scale = D_raw * exp(λd)
        # Penalty grows with depth: same distance is penalized more at deeper levels
        level_scales = {}
        if depth_normalize:
            if depth_scale_mode == 'exponential':
                for level in raw_distances_by_level:
                    level_scales[level] = np.exp(-depth_decay * level)
            else:
                raise ValueError(
                    f"Unknown depth_scale_mode: {depth_scale_mode}. "
                    f"Only 'exponential' is supported for distance normalization. "
                    f"'level_std' and 'subtree_std' are for kernel bandwidth selection."
                )
        else:
            for level in raw_distances_by_level:
                level_scales[level] = 1.0

        # Compute depth-weighted distances
        for level, node_dists in raw_distances_by_level.items():
            scale = level_scales.get(level, 1.0)

            for node_id, parent_id, raw_dist in node_dists:
                if depth_normalize:
                    # D / exp(-λd) = D * exp(λd): penalty grows with depth
                    dist = raw_dist / scale
                else:
                    dist = raw_dist

                # Weight by subtree size (probability proxy)
                if self.use_probability_weight:
                    weight = subtree_sizes.get(node_id, 1)
                else:
                    weight = 1

                distances.append(dist)
                weights.append(weight)
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
        subtree_sizes: Dict[str, int],
        embeddings: Optional[np.ndarray] = None,
        node_to_idx: Optional[Dict[str, int]] = None
    ) -> Tuple[float, Dict[int, float]]:
        """
        Compute entropy gain H(T) between levels.

        Measures how informative the refinement is at each level.
        If embeddings provided, uses semantic cluster quality (within/between variance ratio).
        Otherwise falls back to mutual information between structural labels.

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

        # Use semantic entropy if embeddings available
        use_semantic = embeddings is not None and node_to_idx is not None

        for level in range(max_level):
            # Get nodes at this level and next level
            nodes_this = [n for n, l in levels.items() if l == level]
            nodes_next = [n for n, l in levels.items() if l == level + 1]

            if not nodes_this or not nodes_next:
                continue

            if use_semantic:
                # Semantic entropy: measure how well children cluster under parents
                # Higher = children are well-separated between parent groups
                h = self._compute_semantic_entropy_gain(
                    nodes_this, nodes_next, parent_of, embeddings, node_to_idx
                )
            else:
                # Structural mutual information
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

                h = self.compute_mutual_information(
                    np.array(parent_labels),
                    np.array(child_labels)
                )

            per_level_H[level] = h
            total_H += h
            n_transitions += 1

        overall_H = total_H / n_transitions if n_transitions > 0 else 0.0

        return overall_H, per_level_H

    def _compute_semantic_entropy_gain(
        self,
        parent_nodes: List[str],
        child_nodes: List[str],
        parent_of: Dict[str, str],
        embeddings: np.ndarray,
        node_to_idx: Dict[str, int]
    ) -> float:
        """
        Compute semantic entropy gain using embedding-based cluster quality.

        Uses the ratio of between-cluster variance to within-cluster variance.
        Higher ratio = better separation = more informative split.

        This is related to Fisher's criterion / Linear Discriminant Analysis.
        """
        # Group children by parent
        children_by_parent = defaultdict(list)
        for child_id in child_nodes:
            if child_id in parent_of and child_id in node_to_idx:
                parent_id = parent_of[child_id]
                if parent_id in parent_nodes:
                    children_by_parent[parent_id].append(child_id)

        if len(children_by_parent) < 2:
            return 0.0

        # Compute global centroid
        all_child_embeds = []
        for children in children_by_parent.values():
            for c in children:
                if c in node_to_idx:
                    all_child_embeds.append(embeddings[node_to_idx[c]])

        if len(all_child_embeds) < 2:
            return 0.0

        all_child_embeds = np.array(all_child_embeds)
        global_centroid = all_child_embeds.mean(axis=0)

        # Compute within-cluster variance (average distance to cluster centroid)
        within_var = 0.0
        n_within = 0

        # Compute between-cluster variance (cluster centroids to global centroid)
        cluster_centroids = []

        for parent_id, children in children_by_parent.items():
            child_embeds = []
            for c in children:
                if c in node_to_idx:
                    child_embeds.append(embeddings[node_to_idx[c]])

            if not child_embeds:
                continue

            child_embeds = np.array(child_embeds)
            cluster_centroid = child_embeds.mean(axis=0)
            cluster_centroids.append((cluster_centroid, len(child_embeds)))

            # Within-cluster: sum of squared distances to cluster centroid
            for emb in child_embeds:
                within_var += np.sum((emb - cluster_centroid) ** 2)
                n_within += 1

        if n_within == 0 or len(cluster_centroids) < 2:
            return 0.0

        within_var /= n_within

        # Between-cluster: weighted sum of squared distances from cluster centroids to global
        between_var = 0.0
        total_weight = 0
        for centroid, weight in cluster_centroids:
            between_var += weight * np.sum((centroid - global_centroid) ** 2)
            total_weight += weight

        if total_weight == 0:
            return 0.0

        between_var /= total_weight

        # Return ratio (with epsilon for numerical stability)
        # Log transform to get entropy-like scaling
        eps = 1e-10
        ratio = between_var / (within_var + eps)

        # Map to [0, ~2] range like entropy (log1p to handle ratio < 1)
        return np.log1p(ratio)

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
        reference_stats: Optional[Dict] = None,
        logits: Optional[np.ndarray] = None,
        texts: Optional[Dict[str, str]] = None,
        embedding_texts: Optional[Dict[str, str]] = None
    ) -> HierarchyStats:
        """
        Compute the full hierarchy objective.

        Args:
            tree: Hierarchy structure dict
            embeddings: Node embeddings array (can be from any model: Nomic, MiniLM, etc.)
            reference_stats: Optional stats from baseline for normalization
            logits: Optional pre-computed model output logits for entropy-based metrics
                    If provided, used directly for depth-surprisal correlation
            texts: Optional dict mapping node_id to raw phrase/name
                   Used for on-the-fly entropy computation via entropy_model
                   Independent of embedding model - can use ModernBERT for entropy
                   even when using Nomic embeddings for distances
            embedding_texts: Optional dict mapping node_id to full embedding text
                   (with any context used during embedding). If entropy_text_source
                   is 'embedding_text' or 'mixed', this is used.

        Returns:
            HierarchyStats with all computed metrics

        Note:
            The embedding model (used for D/distances) and entropy model are
            independent. You can compute distances with Nomic embeddings while
            using ModernBERT to compute entropy from the original text.
        """
        # Extract structure
        node_to_idx, parent_of, children_of = self.build_tree_structure(tree, embeddings)
        subtree_sizes = self.compute_subtree_sizes(children_of)
        levels = self.compute_level_assignments(parent_of, children_of)

        # Compute D
        D_raw, D_per_level = self.compute_semantic_distance(
            tree, embeddings, node_to_idx, parent_of, subtree_sizes,
            depth_normalize=self.depth_normalize, depth_decay=self.depth_decay,
            depth_scale_mode=self.depth_scale_mode
        )

        # Compute H based on entropy_source
        if self.entropy_source == 'logits':
            # Use transformer logits for H
            if texts is None:
                raise ValueError(
                    "entropy_source='logits' requires texts dict. "
                    "Provide texts={'node_id': 'node text', ...}"
                )
            # Get text list in node order
            node_ids = list(node_to_idx.keys())
            if self.entropy_text_source == 'raw_phrase':
                text_list = [texts.get(nid, "") for nid in node_ids]
            elif self.entropy_text_source == 'embedding_text':
                if embedding_texts is None:
                    raise ValueError(
                        "entropy_text_source='embedding_text' requires embedding_texts dict"
                    )
                text_list = [embedding_texts.get(nid, "") for nid in node_ids]
            elif self.entropy_text_source == 'mixed':
                # For mixed, we average the entropies from both sources later
                if embedding_texts is None:
                    raise ValueError(
                        "entropy_text_source='mixed' requires embedding_texts dict"
                    )
                text_list = [texts.get(nid, "") for nid in node_ids]
            else:
                raise ValueError(f"Unknown entropy_text_source: {self.entropy_text_source}")

            # Compute per-node entropy from text
            if text_list and any(t for t in text_list):
                node_entropy_array = compute_entropy_from_text(
                    text_list, model_name=self.entropy_model
                )
                node_entropies = {
                    nid: node_entropy_array[i]
                    for i, nid in enumerate(node_ids)
                }
                H_raw, H_per_level = self.compute_entropy_gain_from_logits(
                    levels, node_entropies
                )
            else:
                H_raw, H_per_level = 0.0, {}
        else:
            # Use Fisher criterion (geometric proxy)
            H_raw, H_per_level = self.compute_entropy_gain(
                parent_of, children_of, subtree_sizes, embeddings, node_to_idx
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
            # D is average of cosine distances, bounded [0, 2]
            # No transformation needed - just use raw values
            D_hat = D_raw
            H_hat = H_raw

        # Combine
        eps = 1e-10
        n_levels = max(levels.values()) + 1 if levels else 0

        # Handle degenerate cases: if no measurable structure (H=0, D=0, or no real hierarchy)
        # A flat tree with no measured edges should not be considered "best"
        is_degenerate = (
            (H_hat < eps and D_raw < eps) or  # No measured structure at all
            (H_hat < eps and n_levels <= 2)    # Flat tree (root + leaves only)
        )
        if is_degenerate:
            # Degenerate: no hierarchical structure worth measuring
            # Return maximum penalty
            objective = 1.0
        elif self.combine == 'product':
            # J = D̂ / (1 + Ĥ) - minimize
            # Small when D small AND H large
            # Unlike D(1-H), this doesn't break when H >= 1
            objective = D_hat / (1 + H_hat)
        elif self.combine == 'sum':
            # J = D̂ - Ĥ - minimize
            objective = D_hat - H_hat
        elif self.combine == 'log_product':
            # J = D̂ * exp(-Ĥ) - minimize
            # Exponential reward for high entropy
            objective = D_hat * np.exp(-H_hat)
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

        # Compute depth-surprisal diagnostic (independent sanity check)
        depth_corr = None
        depth_slope = None

        if self.entropy_diagnostic_source == 'logits':
            # Use transformer logits for diagnostic
            if texts is not None:
                node_ids = list(node_to_idx.keys())
                text_list = [texts.get(nid, "") for nid in node_ids]
                if text_list and any(t for t in text_list):
                    diag_logits = compute_logits_from_text(
                        text_list, model_name=self.entropy_model
                    )
                    node_probs = self.get_node_probabilities(
                        embeddings, node_to_idx, subtree_sizes, logits=diag_logits
                    )
                    depth_corr, depth_slope = self.compute_depth_probability_correlation(
                        levels, node_probs
                    )
            elif logits is not None:
                # Use pre-computed logits if available
                node_probs = self.get_node_probabilities(
                    embeddings, node_to_idx, subtree_sizes, logits=logits
                )
                depth_corr, depth_slope = self.compute_depth_probability_correlation(
                    levels, node_probs
                )

        elif self.entropy_diagnostic_source == 'density':
            # Use embedding density for diagnostic
            old_prob_source = self.probability_source
            self.probability_source = 'density_knn'
            node_probs = self.get_node_probabilities(
                embeddings, node_to_idx, subtree_sizes
            )
            self.probability_source = old_prob_source
            depth_corr, depth_slope = self.compute_depth_probability_correlation(
                levels, node_probs
            )
        # else: entropy_diagnostic_source == 'none', skip diagnostic

        return HierarchyStats(
            semantic_distance=D_hat,
            semantic_distance_raw=D_raw,
            entropy_gain=H_hat,
            entropy_gain_raw=H_raw,
            objective=objective,
            n_nodes=len(node_to_idx),
            n_levels=max(levels.values()) + 1 if levels else 0,
            level_stats=level_stats,
            depth_surprisal_correlation=depth_corr,
            depth_surprisal_slope=depth_slope
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
    parser.add_argument('--no-depth-normalize', action='store_true',
                        help='Disable depth-dependent distance normalization')
    parser.add_argument('--depth-decay', type=float, default=0.5,
                        help='Decay rate for expected distance with depth (default: 0.5). '
                             'Higher = stricter requirements at deeper levels.')
    parser.add_argument('--depth-scale-mode',
                        choices=['exponential', 'subtree_std', 'level_std'],
                        default='exponential',
                        help='How to compute expected scale at each depth: '
                             'exponential (exp(-λd)), subtree_std (from within-subtree variance), '
                             'level_std (from all distances at level). Default: exponential.')
    parser.add_argument('--entropy-source',
                        choices=['fisher', 'logits'],
                        default='fisher',
                        help='How to compute entropy for hierarchy quality: '
                             'fisher (geometric proxy using cluster variance) or '
                             'logits (actual entropy from transformer model). Default: fisher.')
    parser.add_argument('--entropy-text-source',
                        choices=['raw_phrase', 'embedding_text', 'mixed'],
                        default='raw_phrase',
                        help='What text to use for logits-based entropy: '
                             'raw_phrase (just topic name), embedding_text (full context), '
                             'or mixed (average of both). Default: raw_phrase.')
    parser.add_argument('--entropy-model', type=str,
                        default='answerdotai/ModernBERT-base',
                        help='HuggingFace model for logits-based entropy computation. '
                             'Independent of embedding model - can use ModernBERT for '
                             'entropy even with Nomic embeddings for distances.')
    parser.add_argument('--entropy-diagnostic-source',
                        choices=['none', 'logits', 'density'],
                        default='none',
                        help='Independent sanity check for depth-surprisal correlation: '
                             'none (skip), logits (transformer), density (embedding k-NN). '
                             'Separate from entropy_source used in objective.')
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
        entropy_alpha=args.entropy_alpha,
        depth_normalize=not args.no_depth_normalize,
        depth_decay=args.depth_decay,
        depth_scale_mode=args.depth_scale_mode,
        entropy_source=args.entropy_source,
        entropy_text_source=args.entropy_text_source,
        entropy_model=args.entropy_model,
        entropy_diagnostic_source=args.entropy_diagnostic_source
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
