"""
Per-Pair Routing: k-NN style routing over individual Q/A pair transforms.

Instead of routing to clusters, we:
1. Compute minimal transform W_j for each training pair (q_j, a_j)
2. At inference, route to similar training queries via softmax
3. Blend the corresponding W_j transforms

Key insight: This is k-NN in transformation space, where each training
example contributes its own Procrustes transform.

Temperature and other routing parameters can be tuned on held-out data.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

# Import from minimal_transform (assumes it's in the same directory)
from minimal_transform import compute_minimal_transform

logger = logging.getLogger(__name__)


@dataclass
class RoutingConfig:
    """Configuration for routing behavior."""
    temperature: float = 0.1          # Softmax temperature
    top_k: Optional[int] = None       # If set, only use top-k neighbors
    min_similarity: float = -1.0      # Minimum similarity threshold
    use_hard_routing: bool = False    # If True, use argmax instead of softmax


@dataclass
class TrainedPair:
    """A trained Q/A pair with its minimal transform."""
    query: np.ndarray           # Original query embedding
    answer: np.ndarray          # Target answer embedding
    W: np.ndarray              # Minimal transform (Procrustes)
    scale: float               # Scale factor from Procrustes
    cluster_id: Optional[str] = None  # Optional cluster label


class PerPairRouting:
    """
    Per-pair routing with learned temperature.

    Each training (q, a) pair gets its own W via Procrustes.
    At inference, blend W's weighted by query similarity.
    """

    def __init__(
        self,
        config: Optional[RoutingConfig] = None,
        allow_scaling: bool = True,
        normalize_queries: bool = True,
    ):
        """
        Initialize per-pair routing.

        Args:
            config: Routing configuration
            allow_scaling: Whether Procrustes includes scaling
            normalize_queries: Whether to L2-normalize queries for similarity
        """
        self.config = config or RoutingConfig()
        self.allow_scaling = allow_scaling
        self.normalize_queries = normalize_queries

        # Trained state
        self.pairs: List[TrainedPair] = []
        self.query_matrix: Optional[np.ndarray] = None  # (N, d) for fast similarity
        self.d: int = 0  # Embedding dimension

    def train(
        self,
        qa_pairs: List[Tuple[np.ndarray, np.ndarray]],
        cluster_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Train per-pair transforms.

        Args:
            qa_pairs: List of (query, answer) embedding pairs
            cluster_ids: Optional cluster labels for each pair

        Returns:
            Training statistics
        """
        N = len(qa_pairs)
        if N == 0:
            raise ValueError("No Q/A pairs provided")

        self.d = len(qa_pairs[0][0].flatten())
        logger.info(f"Training PerPairRouting: N={N}, d={self.d}")

        self.pairs = []
        queries = []
        scales = []

        for i, (q, a) in enumerate(qa_pairs):
            q = q.flatten()
            a = a.flatten()

            # Compute minimal transform
            W, scale, _ = compute_minimal_transform(
                q.reshape(1, -1),
                a.reshape(1, -1),
                allow_scaling=self.allow_scaling,
            )

            cluster_id = cluster_ids[i] if cluster_ids else None

            self.pairs.append(TrainedPair(
                query=q,
                answer=a,
                W=W,
                scale=scale,
                cluster_id=cluster_id,
            ))

            queries.append(q)
            scales.append(scale)

        # Build query matrix for fast similarity computation
        self.query_matrix = np.stack(queries)
        if self.normalize_queries:
            norms = np.linalg.norm(self.query_matrix, axis=1, keepdims=True)
            self.query_matrix = self.query_matrix / (norms + 1e-8)

        return {
            "num_pairs": N,
            "embedding_dim": self.d,
            "mean_scale": float(np.mean(scales)),
            "std_scale": float(np.std(scales)),
        }

    def compute_routing_weights(
        self,
        query: np.ndarray,
        config: Optional[RoutingConfig] = None,
    ) -> np.ndarray:
        """
        Compute routing weights for a query.

        Args:
            query: Query embedding (d,)
            config: Override routing config

        Returns:
            Routing weights (N,) summing to 1
        """
        if self.query_matrix is None:
            raise ValueError("Model not trained")

        config = config or self.config
        query = query.flatten()

        # Normalize query if needed
        if self.normalize_queries:
            query = query / (np.linalg.norm(query) + 1e-8)

        # Compute similarities (matrix multiply)
        similarities = self.query_matrix @ query  # (N,)

        # Apply minimum similarity threshold
        if config.min_similarity > -1.0:
            mask = similarities >= config.min_similarity
            if not mask.any():
                # Fall back to top-1 if nothing passes threshold
                mask[np.argmax(similarities)] = True
            similarities = np.where(mask, similarities, -np.inf)

        # Apply top-k if specified
        if config.top_k is not None and config.top_k < len(similarities):
            top_k_idx = np.argpartition(similarities, -config.top_k)[-config.top_k:]
            mask = np.zeros(len(similarities), dtype=bool)
            mask[top_k_idx] = True
            similarities = np.where(mask, similarities, -np.inf)

        # Hard or soft routing
        if config.use_hard_routing:
            weights = np.zeros(len(similarities))
            weights[np.argmax(similarities)] = 1.0
        else:
            # Softmax with temperature
            exp_sim = np.exp((similarities - np.max(similarities)) / config.temperature)
            weights = exp_sim / (np.sum(exp_sim) + 1e-8)

        return weights

    def project(
        self,
        query: np.ndarray,
        config: Optional[RoutingConfig] = None,
    ) -> np.ndarray:
        """
        Project query using weighted blend of per-pair transforms.

        Args:
            query: Query embedding (d,)
            config: Override routing config

        Returns:
            Projected embedding (d,)
        """
        query = query.flatten()
        weights = self.compute_routing_weights(query, config)

        # Blend projections
        projected = np.zeros(self.d)
        for i, (pair, weight) in enumerate(zip(self.pairs, weights)):
            if weight > 1e-10:  # Skip negligible weights
                projected += weight * (query @ pair.W)

        return projected

    def evaluate(
        self,
        val_pairs: List[Tuple[np.ndarray, np.ndarray]],
        config: Optional[RoutingConfig] = None,
        answer_pool: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            val_pairs: List of (query, answer) pairs
            config: Routing config to evaluate
            answer_pool: Optional answer pool for ranking. If None, uses
                        validation answers as the pool.

        Returns:
            Evaluation metrics
        """
        config = config or self.config

        cosines = []
        mses = []
        ranks = []

        # Build answer pool for ranking
        if answer_pool is None:
            # Use validation answers as the pool
            all_answers = np.stack([a.flatten() for _, a in val_pairs])
        else:
            all_answers = answer_pool

        answer_norms = np.linalg.norm(all_answers, axis=1, keepdims=True)
        all_answers_normalized = all_answers / (answer_norms + 1e-8)
        n_answers = len(all_answers)

        for idx, (q, a) in enumerate(val_pairs):
            a = a.flatten()
            projected = self.project(q, config)

            # Cosine similarity to target
            proj_norm = np.linalg.norm(projected)
            a_norm = np.linalg.norm(a)
            cosine = np.dot(projected, a) / (proj_norm * a_norm + 1e-8)
            cosines.append(cosine)

            # MSE to target
            mse = np.mean((projected - a) ** 2)
            mses.append(mse)

            # Rank: how does projected compare to all answers?
            proj_normalized = projected / (proj_norm + 1e-8)
            sims_to_all = all_answers_normalized @ proj_normalized

            # The target answer is at index 'idx' in the pool (if using val answers)
            if answer_pool is None:
                target_idx = idx
            else:
                # Find closest match in pool
                target_normalized = a / (a_norm + 1e-8)
                target_sims = all_answers_normalized @ target_normalized
                target_idx = np.argmax(target_sims)

            # Rank = how many answers have higher similarity than target
            target_sim = sims_to_all[target_idx]
            rank = np.sum(sims_to_all >= target_sim)
            ranks.append(rank)

        ranks = np.array(ranks)

        return {
            "mean_cosine": float(np.mean(cosines)),
            "std_cosine": float(np.std(cosines)),
            "mean_mse": float(np.mean(mses)),
            "mean_rank": float(np.mean(ranks)),
            "mrr": float(np.mean(1.0 / ranks)),
            "recall_at_1": float(np.mean(ranks == 1)),
            "recall_at_5": float(np.mean(ranks <= 5)),
            "recall_at_10": float(np.mean(ranks <= 10)),
            "n_answers_in_pool": n_answers,
        }

    def tune_temperature(
        self,
        val_pairs: List[Tuple[np.ndarray, np.ndarray]],
        temperatures: Optional[List[float]] = None,
        metric: str = "mrr",
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Tune temperature on validation set.

        Args:
            val_pairs: Validation (query, answer) pairs
            temperatures: Temperatures to try
            metric: Metric to optimize ("mrr", "mean_cosine", "recall_at_1")

        Returns:
            (best_temperature, results_dict)
        """
        if temperatures is None:
            temperatures = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

        results = {}
        best_temp = self.config.temperature
        best_score = -np.inf

        for temp in temperatures:
            config = RoutingConfig(
                temperature=temp,
                top_k=self.config.top_k,
                min_similarity=self.config.min_similarity,
                use_hard_routing=self.config.use_hard_routing,
            )

            metrics = self.evaluate(val_pairs, config)
            score = metrics[metric]

            results[temp] = metrics
            logger.info(f"  temp={temp:.3f}: {metric}={score:.4f}")

            if score > best_score:
                best_score = score
                best_temp = temp

        # Update config with best temperature
        self.config.temperature = best_temp

        return best_temp, results

    def tune_top_k(
        self,
        val_pairs: List[Tuple[np.ndarray, np.ndarray]],
        k_values: Optional[List[int]] = None,
        metric: str = "mrr",
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Tune top-k on validation set.

        Args:
            val_pairs: Validation pairs
            k_values: Values of k to try
            metric: Metric to optimize

        Returns:
            (best_k, results_dict)
        """
        N = len(self.pairs)
        if k_values is None:
            k_values = [1, 3, 5, 10, 20, 50, min(100, N), N]
            k_values = sorted(set(k for k in k_values if k <= N))

        results = {}
        best_k = N  # Default to all
        best_score = -np.inf

        for k in k_values:
            config = RoutingConfig(
                temperature=self.config.temperature,
                top_k=k,
                min_similarity=self.config.min_similarity,
                use_hard_routing=self.config.use_hard_routing,
            )

            metrics = self.evaluate(val_pairs, config)
            score = metrics[metric]

            results[k] = metrics
            logger.info(f"  k={k}: {metric}={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

        self.config.top_k = best_k if best_k < N else None

        return best_k, results


def train_val_split(
    qa_pairs: List[Tuple[np.ndarray, np.ndarray]],
    val_ratio: float = 0.2,
    seed: int = 42,
    cluster_ids: Optional[List[str]] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]],
           List[Tuple[np.ndarray, np.ndarray]],
           Optional[List[str]],
           Optional[List[str]]]:
    """
    Split Q/A pairs into train and validation sets.

    Args:
        qa_pairs: All (query, answer) pairs
        val_ratio: Fraction for validation
        seed: Random seed
        cluster_ids: Optional cluster labels

    Returns:
        (train_pairs, val_pairs, train_cluster_ids, val_cluster_ids)
    """
    np.random.seed(seed)
    N = len(qa_pairs)
    indices = np.random.permutation(N)

    val_size = int(N * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_pairs = [qa_pairs[i] for i in train_idx]
    val_pairs = [qa_pairs[i] for i in val_idx]

    train_clusters = None
    val_clusters = None
    if cluster_ids:
        train_clusters = [cluster_ids[i] for i in train_idx]
        val_clusters = [cluster_ids[i] for i in val_idx]

    return train_pairs, val_pairs, train_clusters, val_clusters
