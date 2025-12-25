"""
Per-Pair Routing: k-NN style routing over individual Q/A pair transforms.

Instead of routing to clusters, we:
1. Compute minimal transform W_j for each training pair (q_j, a_j)
2. At inference, route to similar training queries via softmax
3. Blend the corresponding W_j transforms

Key insight: This is k-NN in transformation space, where each training
example contributes its own Procrustes transform.

Temperature and other routing parameters can be tuned on held-out data.

## Held-Out Evaluation Methodology

We use held-out data to tune routing parameters (temperature, top-k):

1. **Split**: Randomly partition Q/A pairs into train (80%) and validation (20%)

2. **Training**: Compute Procrustes transforms only for training pairs.
   Each training (q_j, a_j) gets its own W_j. Note: transforms are computed
   independently per pair - there's no dependence on adjacent pairs, so we
   only need to hold out data from routing, not from transform computation.

3. **Routing**: Validation queries are routed using only training queries.
   The validation queries are NOT in the routing pool.

4. **Evaluation**: For each validation query:
   - Project using weighted blend of training W's
   - Rank projected query against validation answers
   - Check if EXACT target answer ranks #1 (R@1), top-5 (R@5), etc.

5. **Tuning**: Try different temperatures, select best by MRR on validation set.

## Evaluation Modes

The answer pool for ranking can be configured:

- **Validation answers only** (default): Rank against held-out answers.
  This measures retrieval among unseen answers.

- **All answers**: Rank against train + validation answers.
  This is a harder test - must find the exact answer among all candidates.

- **Unique cluster answers**: Rank against one answer per cluster.
  This measures cluster-level accuracy when answers are shared within clusters.

## Results (tailored data, 500 pairs)

### Validation Answers Only (100 candidates)

| Model | MRR | R@1 | R@5 |
|-------|-----|-----|-----|
| all-MiniLM (384d) | 0.901 | 83% | 98% |
| ModernBERT (768d) | 0.956 | 92% | 100% |

### Full Answer Pool (500 candidates - harder test)

| Model | MRR | R@1 | R@5 |
|-------|-----|-----|-----|
| all-MiniLM (384d) | 0.758 | 64% | 90% |
| ModernBERT (768d) | 0.852 | 74% | 97% |

The full pool test ranks against ALL answers (train + val), not just held-out.
This is harder because similar training answers compete with the target.

## Practical Use Cases

### R@1 vs R@5 Tradeoffs

Which metric matters depends on the application:

- **Single-turn QA**: R@5 is often sufficient. Users expect to scan a few results
  (like Google's top 10). MiniLM at 90% R@5 is practical and fast.

- **RAG with multiple queries**: R@1 matters more. Each retrieval adds to context:
  - 5 sub-questions × 5 answers = 25 chunks = 5-10K tokens
  - Cleaner retrieval = less noise for the LLM
  - ModernBERT's 74% R@1 reduces context bloat

- **Q/A pair storage**: When both questions and answers are stored, the LLM
  can compare retrieved questions to its query and do its own soft reranking.
  This makes R@5 more useful - the LLM sees the mapping context.

### Model Selection

| Use Case | Recommended | R@5 (full pool) |
|----------|-------------|-----------------|
| Fast prototyping | MiniLM | 90% |
| Production RAG | ModernBERT | 97% |
| Latency-critical | MiniLM | 90% |
| Accuracy-critical | ModernBERT | 97% |

ModernBERT is ~20x slower to embed but embeddings are cached after training.
Inference cost (single query) is negligible for both.

These results use simple softmax routing with temperature tuning.

## Logit-Flux Routing (Experimental)

Logit-flux routing combines similarity and density in log-odds space:

    P(i) ∝ odds(s_i)^(1/τ) × odds(c_i)^(w/τ)

Where:
- odds(x) = x / (1-x)
- c_i is the density-derived confidence for training query i
- w controls density influence (0 = ignore, 1 = equal weight)

**On this dataset, logit-flux does NOT improve results** because:
- Training queries have uniform density (std=0.02)
- 96% of queries have confidence < 0.5 (penalty region)
- No clear dense/sparse regions to exploit

Logit-flux would be more useful for:
- Noisy datasets with outliers to penalize
- Heterogeneous topic coverage (some topics over-represented)
- Real-world data with uneven sampling
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
    # Logit-flux routing parameters
    use_logit_flux: bool = False      # If True, use logit-flux routing
    density_weight: float = 1.0       # Weight for density term (w in the formula)
    density_bandwidth: Optional[float] = None  # KDE bandwidth (None = Silverman's rule)


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

        # Precompute training query densities for logit-flux routing
        self._precompute_densities()

        return {
            "num_pairs": N,
            "embedding_dim": self.d,
            "mean_scale": float(np.mean(scales)),
            "std_scale": float(np.std(scales)),
        }

    def _precompute_densities(self, bandwidth: Optional[float] = None):
        """
        Precompute KDE densities for training queries.

        Uses Gaussian kernel with Silverman's rule for bandwidth selection.
        Densities are converted to confidence scores in (0, 1).
        """
        if self.query_matrix is None:
            return

        N = len(self.query_matrix)

        # Compute pairwise similarities (already normalized)
        sim_matrix = self.query_matrix @ self.query_matrix.T  # (N, N)

        # Convert to distances (cosine distance = 1 - similarity)
        dist_matrix = 1 - sim_matrix

        # Silverman's rule for bandwidth
        if bandwidth is None:
            # Use upper triangle distances
            upper_dists = dist_matrix[np.triu_indices(N, k=1)]
            sigma = np.std(upper_dists)
            iqr = np.percentile(upper_dists, 75) - np.percentile(upper_dists, 25)
            scale = min(sigma, iqr / 1.34) if iqr > 0 else sigma
            bandwidth = 0.9 * scale * (N ** -0.2)
            bandwidth = max(bandwidth, 0.01)  # Minimum bandwidth

        self._kde_bandwidth = bandwidth

        # Gaussian kernel KDE
        kernel_values = np.exp(-(dist_matrix ** 2) / (2 * bandwidth ** 2))
        self._train_densities = kernel_values.mean(axis=1)  # (N,)

        # Convert to confidence in (0, 1) using sigmoid-like normalization
        baseline = self._train_densities.mean()
        self._train_confidences = self._train_densities / (self._train_densities + baseline)

        logger.info(f"KDE bandwidth: {bandwidth:.4f}, mean confidence: {self._train_confidences.mean():.4f}")

    def compute_query_density(self, query: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute density of query relative to training queries.

        Args:
            query: Query embedding (d,), should be normalized

        Returns:
            (per_train_densities, query_confidence)
            - per_train_densities: How dense each training query's neighborhood is
            - query_confidence: How confident we are about this query location
        """
        query = query.flatten()
        if self.normalize_queries:
            query = query / (np.linalg.norm(query) + 1e-8)

        # Similarity to all training queries
        sims = self.query_matrix @ query  # (N,)
        dists = 1 - sims

        # Query's density (how close is it to training queries)
        kernel_values = np.exp(-(dists ** 2) / (2 * self._kde_bandwidth ** 2))
        query_density = kernel_values.mean()

        # Convert to confidence
        baseline = self._train_densities.mean()
        query_confidence = query_density / (query_density + baseline)

        return self._train_confidences, query_confidence

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
        elif config.use_logit_flux:
            # Logit-flux routing: combine similarity and density in log-odds space
            weights = self._logit_flux_weights(similarities, config)
        else:
            # Standard softmax with temperature
            exp_sim = np.exp((similarities - np.max(similarities)) / config.temperature)
            weights = exp_sim / (np.sum(exp_sim) + 1e-8)

        return weights

    def _logit_flux_weights(
        self,
        similarities: np.ndarray,
        config: RoutingConfig,
        eps: float = 1e-7,
    ) -> np.ndarray:
        """
        Compute routing weights using logit-flux formulation.

        Combines similarity and density in log-odds space:
            P(i) ∝ odds(s_i)^(1/τ) × odds(c_i)^(w/τ)

        where odds(x) = x / (1-x) and c_i is the confidence (density-derived).

        Args:
            similarities: Raw cosine similarities to training queries
            config: Routing configuration
            eps: Small constant to avoid log(0)

        Returns:
            Routing weights (N,) summing to 1
        """
        # Clamp similarities to (0, 1) for logit transform
        # Cosine similarities are in [-1, 1], shift to (0, 1)
        s = (similarities + 1) / 2  # Now in [0, 1]
        s = np.clip(s, eps, 1 - eps)

        # Get precomputed confidence scores for training queries
        c = np.clip(self._train_confidences, eps, 1 - eps)

        # Compute logits
        s_logit = np.log(s / (1 - s))
        c_logit = np.log(c / (1 - c))

        # Combined logit with temperature
        # P(i) ∝ exp((logit(s) + w * logit(c)) / τ)
        combined = (s_logit + config.density_weight * c_logit) / config.temperature

        # Handle -inf from top-k/threshold masking
        combined = np.where(np.isinf(similarities), -np.inf, combined)

        # Softmax (with numerical stability)
        combined = combined - np.max(combined[~np.isinf(combined)])
        exp_combined = np.exp(combined)
        weights = exp_combined / (np.sum(exp_combined) + eps)

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


def build_full_answer_pool(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    val_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Build answer pool from all pairs (train + validation).

    This creates a harder ranking task where the model must find
    the exact target answer among ALL possible answers.

    Args:
        train_pairs: Training (query, answer) pairs
        val_pairs: Validation (query, answer) pairs

    Returns:
        Answer matrix (N_total × d)
    """
    all_answers = []
    for _, a in train_pairs:
        all_answers.append(a.flatten())
    for _, a in val_pairs:
        all_answers.append(a.flatten())
    return np.stack(all_answers)


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
