#!/usr/bin/env python3
"""
Answer-Level Benchmark for LDA Smoothing.

Unlike cluster-level benchmarks (P@1 = correct cluster), this measures
distance to the exact target answer:

1. Cosine Similarity: cos(projected_query, target_answer)
2. MSE: ||projected_query - target_answer||²
3. Rank of Target: Position of correct answer in similarity-ranked results
4. Recall@k: Is target answer in top-k results?
5. Per-cluster breakdown: Which clusters are hardest?

Usage:
    python scripts/benchmark_answer_level.py
    python scripts/benchmark_answer_level.py --max-clusters 50
    python scripts/benchmark_answer_level.py --output reports/answer_level.json

    # Real embeddings (requires sentence-transformers):
    python scripts/benchmark_answer_level.py --embedder all-minilm
    python scripts/benchmark_answer_level.py --embedder e5-small
    python scripts/benchmark_answer_level.py --embedder modernbert

Embedder Options:
    hash         - Hash-based synthetic embeddings (fast, no semantic meaning)
    all-minilm   - all-MiniLM-L6-v2 (384 dim, 512 context)
    e5-small     - intfloat/e5-small-v2 (384 dim, 512 context)
    modernbert   - nomic-ai/nomic-embed-text-v1.5 (768 dim, 8192 context)
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))

# Embedding model configurations
EMBEDDER_CONFIGS = {
    "hash": {
        "name": "Hash-based (synthetic)",
        "dim": 64,
        "model_id": None,
        "description": "Deterministic pseudo-random vectors (no semantic meaning)"
    },
    "all-minilm": {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        "model_id": "all-MiniLM-L6-v2",
        "description": "Small, fast BERT model (384 dim, 512 context)"
    },
    "e5-small": {
        "name": "intfloat/e5-small-v2",
        "dim": 384,
        "model_id": "intfloat/e5-small-v2",
        "description": "E5 small model with query/passage prefixes (384 dim)"
    },
    "modernbert": {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "description": "ModernBERT with 8192 token context (768 dim)"
    }
}

from smoothing_basis import (
    SmoothingBasisProjection,
    MultiHeadLDABaseline,
    ResidualBasisProjection,
    KernelSmoothingProjection,
    SmoothingKernel,
    UnifiedKernelBasisProjection,
    KernelSmoothedBasisProjection,
    AdaptiveConditioningProjection,
)
from fft_smoothing import FFTSmoothingProjection, AdaptiveFFTSmoothing
from hierarchical_smoothing import HierarchicalSmoothing
from minimal_transform import MinimalTransformProjection


@dataclass
class AnswerLevelMetrics:
    """Per-query answer-level metrics."""
    query_id: str
    cluster_id: str
    cosine_to_target: float
    mse_to_target: float
    target_rank: int  # 1-indexed rank of correct answer
    recall_at_1: bool
    recall_at_5: bool
    recall_at_10: bool


@dataclass
class MethodResult:
    """Aggregated results for a smoothing method."""
    method: str
    params: Dict[str, Any]

    # Answer-level metrics (mean across all queries)
    mean_cosine_to_target: float
    std_cosine_to_target: float
    mean_mse_to_target: float
    std_mse_to_target: float
    mean_target_rank: float
    median_target_rank: float

    # Recall metrics
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank

    # Timing
    train_time_ms: float
    inference_time_us: float

    # Per-cluster breakdown
    cluster_metrics: Dict[str, Dict[str, float]]

    # Data info
    num_queries: int
    num_answers: int
    num_clusters: int


def load_training_data(data_dir: Path, max_clusters: Optional[int] = None) -> Dict[str, List[Dict]]:
    """Load training data grouped by cluster_id."""
    clusters = defaultdict(list)

    for jsonl_file in sorted(data_dir.rglob("*.jsonl")):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        pair = json.loads(line)
                        cluster_id = pair.get("cluster_id", "unknown")
                        clusters[cluster_id].append(pair)
                    except json.JSONDecodeError:
                        pass

    if max_clusters and len(clusters) > max_clusters:
        cluster_ids = list(clusters.keys())[:max_clusters]
        clusters = {k: clusters[k] for k in cluster_ids}

    return dict(clusters)


def get_text(field: Any) -> str:
    """Extract text from field."""
    if isinstance(field, str):
        return field
    elif isinstance(field, dict):
        return field.get("text", str(field))
    return str(field) if field else ""


def simple_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic hash-based embedding."""
    np.random.seed(hash(text) % (2**32))
    emb = np.random.randn(dim)
    return emb / (np.linalg.norm(emb) + 1e-8)


class EmbeddingProvider:
    """Wrapper for different embedding providers."""

    def __init__(self, embedder_type: str = "hash", device: str = "auto"):
        self.embedder_type = embedder_type
        self.config = EMBEDDER_CONFIGS[embedder_type]
        self.dim = self.config["dim"]
        self._provider = None
        self._cache = {}  # Cache embeddings to avoid recomputing

        if embedder_type != "hash":
            self._init_model_provider(device)

    def _init_model_provider(self, device: str):
        """Initialize the actual model provider."""
        try:
            from modernbert_embedding import SentenceTransformerProvider
            model_id = self.config["model_id"]
            print(f"Loading embedding model: {model_id}...")
            self._provider = SentenceTransformerProvider(
                model_name=model_id,
                device=device
            )
            # Update dim from actual model
            self.dim = self._provider.dimension
            print(f"  Loaded: {model_id}, dim={self.dim}, device={self._provider.device}")
        except ImportError as e:
            raise ImportError(
                f"Failed to load embedding model. Install sentence-transformers:\n"
                f"  pip install sentence-transformers\n"
                f"Error: {e}"
            )

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, with caching."""
        if text in self._cache:
            return self._cache[text]

        if self.embedder_type == "hash":
            emb = simple_embedding(text, self.dim)
        else:
            # Add prefix for e5 models
            if "e5" in self.embedder_type:
                # e5 uses "query: " and "passage: " prefixes
                # For Q-A pairs, both can use query prefix for comparison
                text = f"query: {text}"
            emb = np.array(self._provider.get_embedding(text))

        self._cache[text] = emb
        return emb

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        if self.embedder_type == "hash":
            return [self.get_embedding(t) for t in texts]

        # Check cache first
        uncached_texts = [t for t in texts if t not in self._cache]
        if uncached_texts:
            # Add prefixes for e5
            if "e5" in self.embedder_type:
                prefixed = [f"query: {t}" for t in uncached_texts]
            else:
                prefixed = uncached_texts

            embeddings = self._provider.get_embeddings(prefixed)
            for t, emb in zip(uncached_texts, embeddings):
                self._cache[t] = np.array(emb)

        return [self._cache[t] for t in texts]

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def prepare_data(
    clusters: Dict[str, List[Dict]],
    embedding_provider: EmbeddingProvider
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],  # (Q, A) for smoothing
    List[Dict],  # Individual Q-A pairs with embeddings
    Dict[str, np.ndarray]  # answer_id -> embedding
]:
    """
    Prepare data for answer-level benchmarking.

    Args:
        clusters: Dict mapping cluster_id to list of Q-A pair dicts
        embedding_provider: Provider for generating embeddings

    Returns:
        clusters_for_smoothing: List of (Q, A) tuples per cluster
        qa_pairs: List of dicts with query/answer embeddings and IDs
        answer_index: Dict mapping answer_id to embedding
    """
    clusters_for_smoothing = []
    qa_pairs = []
    answer_index = {}

    # Collect all texts for batch embedding
    all_q_texts = []
    all_a_texts = []
    text_to_cluster = []  # Track which cluster each text belongs to

    for cluster_id, pairs in clusters.items():
        for i, pair in enumerate(pairs):
            q_text = get_text(pair.get("question", ""))
            a_text = get_text(pair.get("answer", ""))
            if q_text and a_text:
                all_q_texts.append(q_text)
                all_a_texts.append(a_text)
                text_to_cluster.append((cluster_id, i, q_text, a_text))

    # Generate embeddings (batch for efficiency with real models)
    print(f"  Generating embeddings for {len(all_q_texts)} Q-A pairs...")
    start_time = time.perf_counter()

    q_embeddings = embedding_provider.get_embeddings_batch(all_q_texts)
    a_embeddings = embedding_provider.get_embeddings_batch(all_a_texts)

    embed_time = time.perf_counter() - start_time
    print(f"  Embedding time: {embed_time:.2f}s ({len(all_q_texts)/embed_time:.1f} pairs/sec)")

    # Organize by cluster
    cluster_data = defaultdict(lambda: {'questions': [], 'answers': [], 'pairs': []})

    for idx, (cluster_id, i, q_text, a_text) in enumerate(text_to_cluster):
        q_emb = q_embeddings[idx]
        a_emb = a_embeddings[idx]

        answer_id = f"{cluster_id}_{i}"
        query_id = f"q_{cluster_id}_{i}"

        cluster_data[cluster_id]['questions'].append(q_emb)
        cluster_data[cluster_id]['answers'].append(a_emb)
        cluster_data[cluster_id]['pairs'].append({
            'query_id': query_id,
            'answer_id': answer_id,
            'cluster_id': cluster_id,
            'query_emb': q_emb,
            'answer_emb': a_emb,
            'query_text': q_text,
            'answer_text': a_text,
        })

        answer_index[answer_id] = a_emb

    # Build clusters_for_smoothing and qa_pairs
    for cluster_id, data in cluster_data.items():
        qa_pairs.extend(data['pairs'])

        if data['questions']:
            Q = np.array(data['questions'])
            # Use mean answer as cluster representation for smoothing
            A = np.mean(data['answers'], axis=0).reshape(1, -1)
            clusters_for_smoothing.append((Q, A))

    return clusters_for_smoothing, qa_pairs, answer_index


def compute_answer_level_metrics(
    projector,
    qa_pairs: List[Dict],
    answer_index: Dict[str, np.ndarray]
) -> List[AnswerLevelMetrics]:
    """Compute answer-level metrics for each query."""
    metrics = []

    # Build answer list for ranking
    answer_ids = list(answer_index.keys())
    answer_embs = np.array([answer_index[aid] for aid in answer_ids])

    for pair in qa_pairs:
        query_emb = pair['query_emb']
        target_answer_id = pair['answer_id']
        target_answer_emb = pair['answer_emb']

        # Project query
        projected = projector.project(query_emb)

        # Cosine similarity to target
        cos_to_target = cosine_similarity(projected, target_answer_emb)

        # MSE to target
        mse_to_target = float(np.mean((projected - target_answer_emb) ** 2))

        # Compute similarities to all answers for ranking
        sims = []
        for aid, a_emb in zip(answer_ids, answer_embs):
            sim = cosine_similarity(projected, a_emb)
            sims.append((aid, sim))

        # Sort by similarity descending
        sims.sort(key=lambda x: x[1], reverse=True)
        ranked_ids = [aid for aid, _ in sims]

        # Find rank of target answer (1-indexed)
        try:
            target_rank = ranked_ids.index(target_answer_id) + 1
        except ValueError:
            target_rank = len(ranked_ids) + 1

        metrics.append(AnswerLevelMetrics(
            query_id=pair['query_id'],
            cluster_id=pair['cluster_id'],
            cosine_to_target=cos_to_target,
            mse_to_target=mse_to_target,
            target_rank=target_rank,
            recall_at_1=target_rank <= 1,
            recall_at_5=target_rank <= 5,
            recall_at_10=target_rank <= 10,
        ))

    return metrics


def aggregate_metrics(
    method_name: str,
    params: Dict[str, Any],
    metrics: List[AnswerLevelMetrics],
    train_time_ms: float,
    inference_time_us: float,
    num_answers: int,
    num_clusters: int
) -> MethodResult:
    """Aggregate per-query metrics into method-level results."""

    cosines = [m.cosine_to_target for m in metrics]
    mses = [m.mse_to_target for m in metrics]
    ranks = [m.target_rank for m in metrics]

    # Per-cluster breakdown
    cluster_metrics = defaultdict(lambda: {'cosines': [], 'mses': [], 'ranks': []})
    for m in metrics:
        cluster_metrics[m.cluster_id]['cosines'].append(m.cosine_to_target)
        cluster_metrics[m.cluster_id]['mses'].append(m.mse_to_target)
        cluster_metrics[m.cluster_id]['ranks'].append(m.target_rank)

    # Aggregate per cluster
    cluster_summary = {}
    for cid, data in cluster_metrics.items():
        cluster_summary[cid] = {
            'mean_cosine': float(np.mean(data['cosines'])),
            'mean_mse': float(np.mean(data['mses'])),
            'mean_rank': float(np.mean(data['ranks'])),
            'num_queries': len(data['cosines']),
        }

    return MethodResult(
        method=method_name,
        params=params,
        mean_cosine_to_target=float(np.mean(cosines)),
        std_cosine_to_target=float(np.std(cosines)),
        mean_mse_to_target=float(np.mean(mses)),
        std_mse_to_target=float(np.std(mses)),
        mean_target_rank=float(np.mean(ranks)),
        median_target_rank=float(np.median(ranks)),
        recall_at_1=float(np.mean([m.recall_at_1 for m in metrics])),
        recall_at_5=float(np.mean([m.recall_at_5 for m in metrics])),
        recall_at_10=float(np.mean([m.recall_at_10 for m in metrics])),
        mrr=float(np.mean([1.0 / m.target_rank for m in metrics])),
        train_time_ms=train_time_ms,
        inference_time_us=inference_time_us,
        cluster_metrics=cluster_summary,
        num_queries=len(metrics),
        num_answers=num_answers,
        num_clusters=num_clusters,
    )


def benchmark_method(
    name: str,
    projector,
    clusters_for_smoothing: List[Tuple],
    qa_pairs: List[Dict],
    answer_index: Dict[str, np.ndarray],
    train_fn,
    params: Dict[str, Any]
) -> MethodResult:
    """Benchmark a single smoothing method."""

    # Training
    start = time.perf_counter()
    train_fn()
    train_time_ms = (time.perf_counter() - start) * 1000

    # Inference timing
    start = time.perf_counter()
    for pair in qa_pairs:
        _ = projector.project(pair['query_emb'])
    inference_time_us = (time.perf_counter() - start) * 1e6 / len(qa_pairs)

    # Compute answer-level metrics
    metrics = compute_answer_level_metrics(projector, qa_pairs, answer_index)

    return aggregate_metrics(
        name, params, metrics,
        train_time_ms, inference_time_us,
        len(answer_index), len(clusters_for_smoothing)
    )


def run_benchmarks(
    clusters_for_smoothing: List[Tuple],
    qa_pairs: List[Dict],
    answer_index: Dict[str, np.ndarray],
    k_values: List[int] = [4, 8, 16]
) -> List[MethodResult]:
    """Run all benchmarks."""
    results = []

    print(f"\nAnswer-Level Benchmark")
    print(f"Clusters: {len(clusters_for_smoothing)}, Queries: {len(qa_pairs)}, Answers: {len(answer_index)}")
    print("=" * 90)

    # 1. Baseline: Multi-head LDA
    print("\n1. MultiHeadLDA Baseline...")
    baseline = MultiHeadLDABaseline()
    result = benchmark_method(
        "MultiHeadLDA", baseline, clusters_for_smoothing, qa_pairs, answer_index,
        lambda: baseline.train(clusters_for_smoothing),
        {"type": "baseline"}
    )
    results.append(result)
    print(f"   Cosine: {result.mean_cosine_to_target:.4f} ± {result.std_cosine_to_target:.4f}")
    print(f"   MSE: {result.mean_mse_to_target:.4f}, Rank: {result.mean_target_rank:.1f}, MRR: {result.mrr:.4f}")

    # 2. FFT Smoothing
    for cutoff in [0.3, 0.5, 0.7]:
        print(f"\n2. FFT (cutoff={cutoff})...")
        fft = FFTSmoothingProjection(cutoff=cutoff, blend_factor=0.6)
        result = benchmark_method(
            f"FFT_c{cutoff}", fft, clusters_for_smoothing, qa_pairs, answer_index,
            lambda fft=fft: fft.train(clusters_for_smoothing),
            {"cutoff": cutoff, "blend": 0.6}
        )
        results.append(result)
        print(f"   Cosine: {result.mean_cosine_to_target:.4f} ± {result.std_cosine_to_target:.4f}")
        print(f"   MSE: {result.mean_mse_to_target:.4f}, Rank: {result.mean_target_rank:.1f}, MRR: {result.mrr:.4f}")

    # 3. Adaptive FFT
    print("\n3. AdaptiveFFT...")
    adaptive = AdaptiveFFTSmoothing(min_cutoff=0.2, max_cutoff=0.7)
    result = benchmark_method(
        "AdaptiveFFT", adaptive, clusters_for_smoothing, qa_pairs, answer_index,
        lambda: adaptive.train(clusters_for_smoothing),
        {"min_cutoff": 0.2, "max_cutoff": 0.7}
    )
    results.append(result)
    print(f"   Cosine: {result.mean_cosine_to_target:.4f} ± {result.std_cosine_to_target:.4f}")
    print(f"   MSE: {result.mean_mse_to_target:.4f}, Rank: {result.mean_target_rank:.1f}, MRR: {result.mrr:.4f}")

    # 4. Smoothing Basis
    for K in k_values:
        if K > len(clusters_for_smoothing):
            continue
        print(f"\n4. SmoothingBasis (K={K})...")
        sb = SmoothingBasisProjection(num_basis=K, cosine_weight=0.5)
        result = benchmark_method(
            f"Basis_K{K}", sb, clusters_for_smoothing, qa_pairs, answer_index,
            lambda sb=sb: sb.train(clusters_for_smoothing, num_iterations=50, log_interval=100),
            {"K": K, "iterations": 50}
        )
        results.append(result)
        print(f"   Cosine: {result.mean_cosine_to_target:.4f} ± {result.std_cosine_to_target:.4f}")
        print(f"   MSE: {result.mean_mse_to_target:.4f}, Rank: {result.mean_target_rank:.1f}, MRR: {result.mrr:.4f}")

    # 5. Residual Basis (FFT + learned residual)
    for K in [4, 8]:
        for alpha_reg in [0.01, 0.1]:
            print(f"\n5. ResidualBasis (K={K}, reg={alpha_reg})...")
            rb = ResidualBasisProjection(
                num_basis=K,
                fft_cutoff=0.5,
                fft_blend=0.7,
                alpha_reg=alpha_reg,
                cosine_weight=0.5
            )
            result = benchmark_method(
                f"Residual_K{K}_r{alpha_reg}", rb, clusters_for_smoothing, qa_pairs, answer_index,
                lambda rb=rb: rb.train(clusters_for_smoothing, num_iterations=30, log_interval=100),
                {"K": K, "alpha_reg": alpha_reg, "fft_cutoff": 0.5}
            )
            results.append(result)
            print(f"   Cosine: {result.mean_cosine_to_target:.4f} ± {result.std_cosine_to_target:.4f}")
            print(f"   MSE: {result.mean_mse_to_target:.4f}, Rank: {result.mean_target_rank:.1f}, MRR: {result.mrr:.4f}")

            # Print residual stats
            stats = rb.get_residual_stats()
            if stats.get('has_residuals'):
                print(f"   Residual: ΔW/FFT ratio={stats['delta_to_fft_ratio']:.4f}")

    # 6. Kernel Smoothing Methods
    kernel_configs = [
        (SmoothingKernel.GAUSSIAN_RBF, "Gaussian", [0.5, 1.0, 2.0]),
        (SmoothingKernel.MATERN_52, "Matern52", [0.5, 1.0, 2.0]),
        (SmoothingKernel.MATERN_32, "Matern32", [1.0]),
    ]

    for kernel, kernel_name, length_scales in kernel_configs:
        for ls in length_scales:
            print(f"\n6. KernelSmoothing ({kernel_name}, ls={ls})...")
            ks = KernelSmoothingProjection(
                kernel=kernel,
                length_scale=ls,
                regularization=1e-6,
            )
            result = benchmark_method(
                f"Kernel_{kernel_name}_ls{ls}", ks, clusters_for_smoothing, qa_pairs, answer_index,
                lambda ks=ks: ks.train(clusters_for_smoothing),
                {"kernel": kernel_name, "length_scale": ls}
            )
            results.append(result)
            print(f"   Cosine: {result.mean_cosine_to_target:.4f} ± {result.std_cosine_to_target:.4f}")
            print(f"   MSE: {result.mean_mse_to_target:.4f}, Rank: {result.mean_target_rank:.1f}, MRR: {result.mrr:.4f}")

            # Print kernel stats
            kernel_stats = ks.get_kernel_stats()
            print(f"   Kernel: cond={kernel_stats['condition_number']:.2f}, eff_rank={kernel_stats['effective_rank']:.0f}")

    # 7. Unified Kernel Basis (Matérn-5/2 + K=4 basis + CG)
    unified_configs = [
        # (K, length_scale, smoothing_strength, k_neighbors)
        (4, 0.5, 0.1, None),       # Dense kernel
        (4, 0.5, 0.1, 5),          # Sparse (k=5 neighbors)
        (4, 0.5, 0.01, None),      # Weaker smoothing
        (4, 1.0, 0.1, None),       # Larger length scale
        (8, 0.5, 0.1, None),       # More basis vectors
    ]

    for K, ls, lam, k_nn in unified_configs:
        sparse_str = f"_k{k_nn}" if k_nn else ""
        name = f"Unified_K{K}_ls{ls}_λ{lam}{sparse_str}"
        print(f"\n7. {name}...")

        unified = UnifiedKernelBasisProjection(
            num_basis=K,
            length_scale=ls,
            smoothing_strength=lam,
            k_neighbors=k_nn,
            cg_max_iter=50,
            cg_tol=1e-5,
            use_jacobi_precond=True,
            diagonal_init_scale=1.0,
            noise_scale=0.1,
            cosine_weight=0.5,
        )
        result = benchmark_method(
            name, unified, clusters_for_smoothing, qa_pairs, answer_index,
            lambda u=unified: u.train(clusters_for_smoothing, num_iterations=30, log_interval=100),
            {"K": K, "length_scale": ls, "smoothing_strength": lam, "k_neighbors": k_nn}
        )
        results.append(result)
        print(f"   Cosine: {result.mean_cosine_to_target:.4f} ± {result.std_cosine_to_target:.4f}")
        print(f"   MSE: {result.mean_mse_to_target:.4f}, Rank: {result.mean_target_rank:.1f}, MRR: {result.mrr:.4f}")

        # Print unified stats
        stats = unified.get_stats()
        if stats.get('trained'):
            print(f"   CG: total_iters={stats['total_cg_iterations']}, "
                  f"eff_neighbors={unified.K_matrix.sum(axis=1).mean() - 1:.1f}")

    # 8. KernelSmoothedBasis (simpler hybrid: basis first, then kernel smooth)
    hybrid_configs = [
        # (K, length_scale, kernel_blend)
        (4, 0.5, 0.3),   # Light smoothing
        (4, 0.5, 0.5),   # Medium smoothing
        (4, 0.5, 0.7),   # Strong smoothing
        (8, 0.5, 0.5),   # More basis, medium smoothing
        (4, 1.0, 0.5),   # Larger length scale
    ]

    for K, ls, blend in hybrid_configs:
        name = f"Hybrid_K{K}_ls{ls}_b{blend}"
        print(f"\n8. {name}...")

        hybrid = KernelSmoothedBasisProjection(
            num_basis=K,
            length_scale=ls,
            kernel_blend=blend,
            cosine_weight=0.5,
        )
        result = benchmark_method(
            name, hybrid, clusters_for_smoothing, qa_pairs, answer_index,
            lambda h=hybrid: h.train(clusters_for_smoothing, num_iterations=50, log_interval=100),
            {"K": K, "length_scale": ls, "kernel_blend": blend}
        )
        results.append(result)
        print(f"   Cosine: {result.mean_cosine_to_target:.4f} ± {result.std_cosine_to_target:.4f}")
        print(f"   MSE: {result.mean_mse_to_target:.4f}, Rank: {result.mean_target_rank:.1f}, MRR: {result.mrr:.4f}")

    # 9. Adaptive Conditioning Projection (bisection-based λ)
    adaptive_cond_configs = [
        # (K, target_cond)
        (4, 10.0),    # Strict conditioning
        (4, 50.0),    # Moderate conditioning (default)
        (4, 100.0),   # Relaxed conditioning
        (8, 50.0),    # More basis vectors
    ]

    for K, target_cond in adaptive_cond_configs:
        name = f"AdaptiveCond_K{K}_c{int(target_cond)}"
        print(f"\n9. {name}...")

        adaptive_proj = AdaptiveConditioningProjection(
            num_basis=K,
            target_cond=target_cond,
            bisection_tol=1e-3,
            bisection_max_iter=20,
            cosine_weight=0.5,
        )
        result = benchmark_method(
            name, adaptive_proj, clusters_for_smoothing, qa_pairs, answer_index,
            lambda a=adaptive_proj: a.train(clusters_for_smoothing, num_iterations=50, log_interval=100),
            {"K": K, "target_cond": target_cond}
        )
        results.append(result)
        print(f"   Cosine: {result.mean_cosine_to_target:.4f} ± {result.std_cosine_to_target:.4f}")
        print(f"   MSE: {result.mean_mse_to_target:.4f}, Rank: {result.mean_target_rank:.1f}, MRR: {result.mrr:.4f}")

        # Print adaptive stats
        stats = adaptive_proj.get_stats()
        if stats.get('trained'):
            print(f"   Regularized: {stats['clusters_needing_regularization']}/{len(clusters_for_smoothing)} clusters")
            print(f"   Mean λ: {stats['mean_lambda']:.4f}")

    # =========================================================================
    # 10. Minimal Transformation Projection (Procrustes-based)
    # =========================================================================
    print("\n" + "=" * 60)
    print("10. MINIMAL TRANSFORMATION PROJECTION (Procrustes)")
    print("=" * 60)

    # Test configurations: (smooth_method, fidelity_weight, per_pair)
    minimal_configs = [
        # Centroid-based (original)
        ("none", 1.0, False),      # Pure minimal transform, no smoothing
        ("fft", 0.0, False),       # Full FFT smoothing
        ("fft", 0.3, False),       # Mostly smoothed, some minimal
        ("fft", 0.5, False),       # Balanced
        ("fft", 0.7, False),       # Mostly minimal, some smoothing
        ("kernel", 0.5, False),    # Kernel smoothing, balanced
        # Per-pair: "transform then average"
        ("none", 1.0, True),       # Per-pair, then average (within-cluster smoothing)
        ("fft", 0.5, True),        # Per-pair + cross-cluster FFT smoothing
    ]

    for smooth_method, fidelity, per_pair in minimal_configs:
        pp_suffix = "_pp" if per_pair else ""
        name = f"MinTrans_{smooth_method}_f{int(fidelity*10)}{pp_suffix}"
        print(f"\n10. {name}...")

        min_proj = MinimalTransformProjection(
            smooth_method=smooth_method,
            fft_cutoff_ratio=0.3,
            fidelity_weight=fidelity,
            allow_scaling=True,
            per_pair=per_pair,
        )
        result = benchmark_method(
            name, min_proj, clusters_for_smoothing, qa_pairs, answer_index,
            lambda m=min_proj: m.train(clusters_for_smoothing),
            {"smooth_method": smooth_method, "fidelity_weight": fidelity, "per_pair": per_pair}
        )
        results.append(result)
        print(f"   Cosine: {result.mean_cosine_to_target:.4f} ± {result.std_cosine_to_target:.4f}")
        print(f"   MSE: {result.mean_mse_to_target:.4f}, Rank: {result.mean_target_rank:.1f}, MRR: {result.mrr:.4f}")

        # Print diagnostics
        diag = min_proj.get_diagnostics()
        if diag.get('trained'):
            print(f"   Rotation quality: {diag['mean_rotation_quality']:.4f}")
            train_stats = min_proj._compute_stats()
            print(f"   Dev from minimal: {train_stats.get('mean_deviation_from_minimal', 0):.4f}")

    return results


def print_metric_legend():
    """Print explanation of metrics."""
    print("\n" + "=" * 110)
    print("METRIC LEGEND")
    print("=" * 110)
    print("""
| Metric   | Meaning                                           | Good Value |
|----------|---------------------------------------------------|------------|
| Cosine   | Cosine similarity between projected query and     | Higher (1.0 = perfect alignment)
|          | target answer. Measures direction alignment.      |            |
| MSE      | Mean Squared Error between projected query and    | Lower (0.0 = identical vectors)
|          | target answer. Measures magnitude difference.     |            |
| Rank     | Position of correct answer when all answers are   | Lower (1 = correct answer is top result)
|          | sorted by similarity to projected query.          |            |
| MRR      | Mean Reciprocal Rank = average of 1/rank.         | Higher (1.0 = always rank 1)
|          | Rewards finding correct answer early.             |            |
| R@k      | Recall at k = % of queries where correct answer   | Higher (100% = always in top k)
|          | appears in top k results.                         |            |
""")


def print_summary(results: List[MethodResult]):
    """Print summary table."""
    print_metric_legend()

    print("\n" + "=" * 110)
    print("ANSWER-LEVEL SUMMARY")
    print("=" * 110)

    # Header
    print(f"{'Method':<20} {'Cosine':>10} {'MSE':>10} {'Rank':>8} {'MRR':>8} "
          f"{'R@1':>8} {'R@5':>8} {'R@10':>8} {'Train(ms)':>10}")
    print("-" * 110)

    # Sort by MRR descending
    for r in sorted(results, key=lambda x: x.mrr, reverse=True):
        print(f"{r.method:<20} {r.mean_cosine_to_target:<12.4f} {r.mean_mse_to_target:<10.4f} "
              f"{r.mean_target_rank:<8.1f} {r.mrr:<8.4f} {r.recall_at_1:<8.2%} "
              f"{r.recall_at_5:<8.2%} {r.recall_at_10:<8.2%} {r.train_time_ms:<10.1f}")

    print("-" * 110)

    # Best by metric
    print("\nBest by metric:")
    print(f"  Highest Cosine:  {max(results, key=lambda x: x.mean_cosine_to_target).method} "
          f"({max(results, key=lambda x: x.mean_cosine_to_target).mean_cosine_to_target:.4f})")
    print(f"  Lowest MSE:      {min(results, key=lambda x: x.mean_mse_to_target).method} "
          f"({min(results, key=lambda x: x.mean_mse_to_target).mean_mse_to_target:.4f})")
    print(f"  Lowest Rank:     {min(results, key=lambda x: x.mean_target_rank).method} "
          f"({min(results, key=lambda x: x.mean_target_rank).mean_target_rank:.1f})")
    print(f"  Highest MRR:     {max(results, key=lambda x: x.mrr).method} "
          f"({max(results, key=lambda x: x.mrr).mrr:.4f})")


def print_cluster_analysis(results: List[MethodResult], top_n: int = 5):
    """Print per-cluster analysis for best method."""
    best = max(results, key=lambda x: x.mrr)

    print(f"\n{'=' * 80}")
    print(f"PER-CLUSTER ANALYSIS (Method: {best.method})")
    print("=" * 80)

    # Sort clusters by difficulty (mean rank)
    sorted_clusters = sorted(
        best.cluster_metrics.items(),
        key=lambda x: x[1]['mean_rank']
    )

    # Easiest clusters
    print(f"\nEasiest Clusters (lowest mean rank):")
    print(f"{'Cluster':<30} {'Cosine':<10} {'MSE':<10} {'Rank':<8} {'N':<5}")
    print("-" * 65)
    for cid, m in sorted_clusters[:top_n]:
        print(f"{cid[:30]:<30} {m['mean_cosine']:<10.4f} {m['mean_mse']:<10.4f} "
              f"{m['mean_rank']:<8.1f} {m['num_queries']:<5}")

    # Hardest clusters
    print(f"\nHardest Clusters (highest mean rank):")
    print(f"{'Cluster':<30} {'Cosine':<10} {'MSE':<10} {'Rank':<8} {'N':<5}")
    print("-" * 65)
    for cid, m in sorted_clusters[-top_n:]:
        print(f"{cid[:30]:<30} {m['mean_cosine']:<10.4f} {m['mean_mse']:<10.4f} "
              f"{m['mean_rank']:<8.1f} {m['num_queries']:<5}")

    # Variance across clusters
    all_cosines = [m['mean_cosine'] for m in best.cluster_metrics.values()]
    all_ranks = [m['mean_rank'] for m in best.cluster_metrics.values()]

    print(f"\nCluster Variance:")
    print(f"  Cosine: min={min(all_cosines):.4f}, max={max(all_cosines):.4f}, "
          f"std={np.std(all_cosines):.4f}")
    print(f"  Rank:   min={min(all_ranks):.1f}, max={max(all_ranks):.1f}, "
          f"std={np.std(all_ranks):.1f}")


def save_results(results: List[MethodResult], output_path: Path):
    """Save results to JSON with metric explanations."""
    save_results_with_embedder(results, output_path, "hash", 64)


def save_results_with_embedder(
    results: List[MethodResult],
    output_path: Path,
    embedder: str,
    dim: int
):
    """Save results to JSON with metric explanations and embedder info."""

    # Metric legend for documentation
    metric_legend = {
        "cosine_to_target": {
            "description": "Cosine similarity between projected query and target answer",
            "good_value": "Higher is better (1.0 = perfect alignment)",
            "range": "[-1.0, 1.0]"
        },
        "mse_to_target": {
            "description": "Mean Squared Error between projected query and target answer",
            "good_value": "Lower is better (0.0 = identical vectors)",
            "range": "[0.0, inf)"
        },
        "target_rank": {
            "description": "Position of correct answer when sorted by similarity (1-indexed)",
            "good_value": "Lower is better (1 = correct answer is top result)",
            "range": "[1, num_answers]"
        },
        "mrr": {
            "description": "Mean Reciprocal Rank = average of 1/rank across all queries",
            "good_value": "Higher is better (1.0 = always rank 1)",
            "range": "(0.0, 1.0]"
        },
        "recall_at_k": {
            "description": "Fraction of queries where correct answer is in top k results",
            "good_value": "Higher is better (1.0 = always in top k)",
            "range": "[0.0, 1.0]"
        }
    }

    # Embedder info
    embedder_info = {
        "embedder": embedder,
        "model_name": EMBEDDER_CONFIGS[embedder]["name"],
        "model_id": EMBEDDER_CONFIGS[embedder]["model_id"],
        "dimension": dim,
        "description": EMBEDDER_CONFIGS[embedder]["description"],
    }

    # Build results list
    method_results = []
    for r in results:
        method_results.append({
            "method": r.method,
            "params": r.params,
            "mean_cosine_to_target": r.mean_cosine_to_target,
            "std_cosine_to_target": r.std_cosine_to_target,
            "mean_mse_to_target": r.mean_mse_to_target,
            "std_mse_to_target": r.std_mse_to_target,
            "mean_target_rank": r.mean_target_rank,
            "median_target_rank": r.median_target_rank,
            "recall_at_1": r.recall_at_1,
            "recall_at_5": r.recall_at_5,
            "recall_at_10": r.recall_at_10,
            "mrr": r.mrr,
            "train_time_ms": r.train_time_ms,
            "inference_time_us": r.inference_time_us,
            "num_queries": r.num_queries,
            "num_answers": r.num_answers,
            "num_clusters": r.num_clusters,
            "cluster_metrics": r.cluster_metrics,
        })

    # Find winner for each metric
    winners = {
        "highest_cosine": max(results, key=lambda x: x.mean_cosine_to_target).method,
        "lowest_mse": min(results, key=lambda x: x.mean_mse_to_target).method,
        "lowest_rank": min(results, key=lambda x: x.mean_target_rank).method,
        "highest_mrr": max(results, key=lambda x: x.mrr).method,
        "highest_recall_at_5": max(results, key=lambda x: x.recall_at_5).method,
    }

    output = {
        "embedder": embedder_info,
        "metric_legend": metric_legend,
        "winners": winners,
        "results": method_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Answer-level benchmark for LDA smoothing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Embedder Options:
  hash         Hash-based synthetic embeddings (fast, no semantic meaning)
  all-minilm   all-MiniLM-L6-v2 (384 dim, 512 context)
  e5-small     intfloat/e5-small-v2 (384 dim, 512 context)
  modernbert   nomic-ai/nomic-embed-text-v1.5 (768 dim, 8192 context)

Examples:
  python scripts/benchmark_answer_level.py --embedder hash
  python scripts/benchmark_answer_level.py --embedder all-minilm
  python scripts/benchmark_answer_level.py --embedder e5-small --max-clusters 50
  python scripts/benchmark_answer_level.py --embedder modernbert --device cuda
        """
    )
    parser.add_argument("--max-clusters", type=int, default=None,
                        help="Limit number of clusters")
    parser.add_argument("--embedder", type=str, default="hash",
                        choices=list(EMBEDDER_CONFIGS.keys()),
                        help="Embedding model to use (default: hash)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device for embedding model (default: auto)")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/answer_level_benchmark.json"),
                        help="Output file")
    parser.add_argument("--k-values", type=str, default="4,8,16",
                        help="K values for smoothing basis")
    parser.add_argument("--cluster-analysis", action="store_true",
                        help="Print per-cluster breakdown")

    args = parser.parse_args()

    # Initialize embedding provider
    print(f"\n{'=' * 80}")
    print(f"EMBEDDING MODEL: {EMBEDDER_CONFIGS[args.embedder]['name']}")
    print(f"Description: {EMBEDDER_CONFIGS[args.embedder]['description']}")
    print(f"{'=' * 80}")

    embedding_provider = EmbeddingProvider(args.embedder, args.device)
    dim = embedding_provider.dim
    print(f"Embedding dimension: {dim}")

    # Find training data
    project_root = Path(__file__).parent.parent
    data_dirs = [
        project_root / "training-data" / "tailored",
        project_root / "training-data" / "tailored-gemini",
        project_root / "training-data" / "expanded",
    ]

    data_dir = None
    for d in data_dirs:
        if d.exists():
            data_dir = d
            break

    if not data_dir:
        print("No training data found. Creating synthetic data...")
        clusters = {}
        for i in range(30):
            clusters[f"cluster_{i}"] = [
                {"question": f"Question {j} about topic {i}",
                 "answer": f"Answer {j} about topic {i}"}
                for j in range(np.random.randint(2, 6))
            ]
    else:
        print(f"Loading data from {data_dir}...")
        clusters = load_training_data(data_dir, args.max_clusters)

    print(f"Loaded {len(clusters)} clusters")

    # Prepare data with embeddings
    clusters_for_smoothing, qa_pairs, answer_index = prepare_data(clusters, embedding_provider)
    print(f"Prepared {len(qa_pairs)} Q-A pairs, {len(answer_index)} unique answers")

    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # Run benchmarks
    results = run_benchmarks(clusters_for_smoothing, qa_pairs, answer_index, k_values)

    # Print summary
    print_summary(results)

    # Per-cluster analysis
    if args.cluster_analysis:
        print_cluster_analysis(results)

    # Update output filename to include embedder
    output_path = args.output
    if args.embedder != "hash":
        output_path = output_path.parent / f"answer_level_{args.embedder}.json"

    # Save results with embedder info
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_with_embedder(results, output_path, args.embedder, dim)


if __name__ == "__main__":
    main()
