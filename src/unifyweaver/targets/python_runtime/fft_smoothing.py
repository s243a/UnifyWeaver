# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
FFT-based Smoothing for Cross-Cluster LDA Projection.

Extends per-cluster smoothing by treating clusters as a 1D signal in similarity
space and applying frequency-domain filtering to smooth projection matrices.

Key insight: Similar clusters should have similar projections. FFT enables:
1. Ordering clusters by similarity (creating 1D signal)
2. Filtering in frequency domain (smooth across cluster boundaries)
3. Reconstructing smoothed projections

This complements per-cluster approaches by adding inter-cluster regularization.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from scipy.fft import fft, ifft, fft2, ifft2
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import logging

logger = logging.getLogger(__name__)


def order_clusters_by_similarity(centroids: np.ndarray) -> np.ndarray:
    """
    Order clusters to form a 1D path through similarity space.

    Uses MST + DFS to find a path that keeps similar clusters adjacent.
    This creates a meaningful 1D ordering for FFT.

    Args:
        centroids: Cluster centroids, shape (N, d)

    Returns:
        Indices giving the ordering, shape (N,)
    """
    N = len(centroids)
    if N <= 2:
        return np.arange(N)

    # Compute pairwise distances (lower = more similar)
    distances = squareform(pdist(centroids, metric='cosine'))

    # Build MST for efficient path through clusters
    mst = minimum_spanning_tree(distances).toarray()
    mst = mst + mst.T  # Make symmetric

    # DFS from node with highest degree (most connected)
    degrees = (mst > 0).sum(axis=1)
    start = np.argmax(degrees)

    visited = set()
    order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        order.append(node)
        neighbors = np.where(mst[node] > 0)[0]
        # Visit neighbors sorted by distance (closest first)
        for neighbor in sorted(neighbors, key=lambda n: mst[node, n]):
            dfs(neighbor)

    dfs(start)

    return np.array(order)


def fft_lowpass_filter(signal: np.ndarray, cutoff: float = 0.5) -> np.ndarray:
    """
    Apply low-pass filter in frequency domain.

    Args:
        signal: Input signal, shape (N, ...) where N is the sequence length
        cutoff: Fraction of frequencies to keep (0.5 = keep lowest 50%)

    Returns:
        Filtered signal, same shape as input
    """
    N = len(signal)
    original_shape = signal.shape

    # Flatten to (N, -1) for processing
    signal_flat = signal.reshape(N, -1)

    # FFT along cluster dimension
    freq = fft(signal_flat, axis=0)

    # Create low-pass mask
    mask = np.zeros(N)
    cutoff_idx = max(1, int(N * cutoff))
    mask[:cutoff_idx] = 1.0
    mask[-cutoff_idx+1:] = 1.0  # Symmetric for real signal

    # Optional: smooth transition (Hann window on edges)
    if cutoff_idx > 1:
        transition = min(3, cutoff_idx // 2)
        for i in range(transition):
            fade = 0.5 * (1 + np.cos(np.pi * i / transition))
            mask[cutoff_idx - 1 - i] *= fade
            if N - cutoff_idx + i < N:
                mask[N - cutoff_idx + i] *= fade

    # Apply filter
    freq_filtered = freq * mask.reshape(-1, 1)

    # Inverse FFT
    filtered = np.real(ifft(freq_filtered, axis=0))

    return filtered.reshape(original_shape)


def fft_bandpass_filter(signal: np.ndarray, low: float = 0.0,
                        high: float = 0.5) -> np.ndarray:
    """
    Apply band-pass filter in frequency domain.

    Args:
        signal: Input signal, shape (N, ...)
        low: Lower cutoff (fraction of frequencies)
        high: Upper cutoff (fraction of frequencies)

    Returns:
        Filtered signal
    """
    N = len(signal)
    signal_flat = signal.reshape(N, -1)

    freq = fft(signal_flat, axis=0)

    # Create band-pass mask
    mask = np.zeros(N)
    low_idx = max(0, int(N * low))
    high_idx = max(1, int(N * high))
    mask[low_idx:high_idx] = 1.0
    mask[N-high_idx+1:N-low_idx+1 if low_idx > 0 else N] = 1.0

    freq_filtered = freq * mask.reshape(-1, 1)
    filtered = np.real(ifft(freq_filtered, axis=0))

    return filtered.reshape(signal.shape)


class FFTSmoothingProjection:
    """
    Cross-cluster smoothing using FFT in similarity-ordered space.

    Workflow:
    1. Train per-cluster projections (W matrices)
    2. Order clusters by similarity
    3. Apply FFT smoothing across ordered sequence
    4. Use smoothed projections for inference

    This provides inter-cluster regularization that complements
    per-cluster optimization.
    """

    def __init__(self, cutoff: float = 0.5,
                 blend_factor: float = 0.5,
                 use_2d_fft: bool = False):
        """
        Args:
            cutoff: Low-pass filter cutoff (0.5 = keep lowest 50% frequencies)
            blend_factor: Blend between original (0) and smoothed (1) projections
            use_2d_fft: Apply 2D FFT on flattened W matrices (experimental)
        """
        self.cutoff = cutoff
        self.blend_factor = blend_factor
        self.use_2d_fft = use_2d_fft

        self.W_original: List[np.ndarray] = []
        self.W_smoothed: List[np.ndarray] = []
        self.centroids: np.ndarray = None
        self.cluster_order: np.ndarray = None
        self.inverse_order: np.ndarray = None

    def train(self, clusters: List[Tuple[np.ndarray, np.ndarray]],
              per_cluster_fn: Optional[Callable] = None) -> None:
        """
        Train FFT-smoothed projection.

        Args:
            clusters: List of (Q_i, A_i) - question embeddings and answer embeddings
            per_cluster_fn: Optional function to compute W from (Q, A)
                           Default: least squares
        """
        N = len(clusters)
        logger.info(f"Training FFT smoothing for {N} clusters")

        # Normalize inputs
        processed = []
        for Q, A in clusters:
            if Q.ndim == 1:
                Q = Q.reshape(1, -1)
            if A.ndim == 1:
                A = A.reshape(1, -1)
            processed.append((Q, A))

        # Compute centroids
        self.centroids = np.array([np.mean(Q, axis=0) for Q, A in processed])

        # Compute per-cluster projections (regularized for sparse clusters)
        if per_cluster_fn is None:
            def per_cluster_fn(Q, A):
                # Q: (n, d), A: (1, d) or (n, d)
                # If single answer, broadcast to all questions
                if A.shape[0] == 1 and Q.shape[0] > 1:
                    A = np.tile(A, (Q.shape[0], 1))
                d = Q.shape[1]
                reg = 0.01 * np.eye(d)
                return np.linalg.solve(Q.T @ Q + reg, Q.T @ A)

        self.W_original = []
        for Q, A in processed:
            W = per_cluster_fn(Q, A)
            self.W_original.append(W)

        # Order clusters by similarity
        self.cluster_order = order_clusters_by_similarity(self.centroids)
        self.inverse_order = np.argsort(self.cluster_order)

        # Stack W matrices in similarity order
        W_stacked = np.stack([self.W_original[i] for i in self.cluster_order])

        # Apply FFT smoothing
        logger.info(f"Applying FFT smoothing with cutoff={self.cutoff}")
        W_filtered = fft_lowpass_filter(W_stacked, self.cutoff)

        # Reorder back to original cluster indices
        W_filtered_reordered = W_filtered[self.inverse_order]

        # Blend original and smoothed
        self.W_smoothed = []
        for i in range(N):
            W_blend = (1 - self.blend_factor) * self.W_original[i] + \
                      self.blend_factor * W_filtered_reordered[i]
            self.W_smoothed.append(W_blend)

        logger.info(f"FFT smoothing complete, blend={self.blend_factor}")

    def project(self, query_emb: np.ndarray,
                temperature: float = 0.1,
                use_smoothed: bool = True) -> np.ndarray:
        """
        Project query using (optionally smoothed) projections with soft routing.

        Args:
            query_emb: Query embedding (d,)
            temperature: Softmax temperature for routing
            use_smoothed: Use smoothed W matrices (True) or original (False)

        Returns:
            Projected embedding (d,)
        """
        W_matrices = self.W_smoothed if use_smoothed else self.W_original

        if len(W_matrices) == 0:
            return query_emb

        # Compute routing weights
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        similarities = []
        for centroid in self.centroids:
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            similarities.append(np.dot(query_norm, centroid_norm))

        similarities = np.array(similarities)

        # Softmax routing
        scaled = similarities / temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        weights = exp_scaled / np.sum(exp_scaled)

        # Weighted combination
        projected = np.zeros_like(query_emb)
        for i, w in enumerate(weights):
            projected += w * (query_emb @ W_matrices[i])

        return projected

    def get_frequency_analysis(self) -> dict:
        """
        Analyze frequency content of projection matrices.

        Returns:
            Dict with frequency analysis metrics
        """
        if len(self.W_original) == 0:
            return {}

        W_stacked = np.stack([self.W_original[i] for i in self.cluster_order])
        W_flat = W_stacked.reshape(len(W_stacked), -1)

        # FFT
        freq = fft(W_flat, axis=0)
        power = np.abs(freq) ** 2

        # Power distribution
        total_power = power.sum(axis=1)
        cumulative = np.cumsum(total_power) / total_power.sum()

        # Find frequency containing 90% of power
        freq_90 = np.searchsorted(cumulative, 0.9) / len(W_stacked)

        return {
            'total_power': float(total_power.sum()),
            'power_per_freq': total_power.tolist(),
            'freq_90_percent': float(freq_90),
            'recommended_cutoff': min(0.7, freq_90 + 0.1)
        }


class AdaptiveFFTSmoothing(FFTSmoothingProjection):
    """
    FFT smoothing with adaptive cutoff based on cluster density.

    Dense regions get more smoothing (lower cutoff).
    Sparse regions preserve original signal (higher cutoff).
    """

    def __init__(self, min_cutoff: float = 0.2, max_cutoff: float = 0.8,
                 blend_factor: float = 0.5):
        super().__init__(cutoff=0.5, blend_factor=blend_factor)
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff

    def train(self, clusters: List[Tuple[np.ndarray, np.ndarray]],
              per_cluster_fn: Optional[Callable] = None) -> None:
        """Train with adaptive cutoff based on local density."""
        N = len(clusters)

        # First pass: compute per-cluster W and centroids
        processed = []
        for Q, A in clusters:
            if Q.ndim == 1:
                Q = Q.reshape(1, -1)
            if A.ndim == 1:
                A = A.reshape(1, -1)
            processed.append((Q, A))

        self.centroids = np.array([np.mean(Q, axis=0) for Q, A in processed])

        if per_cluster_fn is None:
            def per_cluster_fn(Q, A):
                if A.shape[0] == 1 and Q.shape[0] > 1:
                    A = np.tile(A, (Q.shape[0], 1))
                d = Q.shape[1]
                reg = 0.01 * np.eye(d)
                return np.linalg.solve(Q.T @ Q + reg, Q.T @ A)

        self.W_original = [per_cluster_fn(Q, A) for Q, A in processed]

        # Order clusters
        self.cluster_order = order_clusters_by_similarity(self.centroids)
        self.inverse_order = np.argsort(self.cluster_order)

        # Compute local density (average distance to k nearest neighbors)
        k = min(5, N - 1)
        distances = squareform(pdist(self.centroids, metric='cosine'))

        densities = []
        for i in range(N):
            sorted_dists = np.sort(distances[i])
            avg_dist = np.mean(sorted_dists[1:k+1])  # Exclude self
            densities.append(avg_dist)

        densities = np.array(densities)

        # Map density to cutoff (high density = low cutoff = more smoothing)
        density_norm = (densities - densities.min()) / (densities.max() - densities.min() + 1e-8)
        cutoffs = self.min_cutoff + density_norm * (self.max_cutoff - self.min_cutoff)

        # Stack W in order
        W_stacked = np.stack([self.W_original[i] for i in self.cluster_order])

        # Apply per-position adaptive filtering
        # (Simplified: use weighted average of different cutoff results)
        W_filtered = np.zeros_like(W_stacked)

        for cutoff_val in np.linspace(self.min_cutoff, self.max_cutoff, 5):
            W_at_cutoff = fft_lowpass_filter(W_stacked, cutoff_val)

            for i, ordered_idx in enumerate(self.cluster_order):
                weight = np.exp(-((cutoffs[ordered_idx] - cutoff_val) ** 2) / 0.1)
                W_filtered[i] += weight * W_at_cutoff[i]

        # Normalize weights
        weight_sums = np.zeros(N)
        for cutoff_val in np.linspace(self.min_cutoff, self.max_cutoff, 5):
            for i, ordered_idx in enumerate(self.cluster_order):
                weight_sums[i] += np.exp(-((cutoffs[ordered_idx] - cutoff_val) ** 2) / 0.1)

        for i in range(N):
            W_filtered[i] /= weight_sums[i]

        # Reorder and blend
        W_filtered_reordered = W_filtered[self.inverse_order]

        self.W_smoothed = []
        for i in range(N):
            W_blend = (1 - self.blend_factor) * self.W_original[i] + \
                      self.blend_factor * W_filtered_reordered[i]
            self.W_smoothed.append(W_blend)

        logger.info(f"Adaptive FFT smoothing complete, cutoffs range: "
                   f"{cutoffs.min():.2f}-{cutoffs.max():.2f}")


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=== FFT Smoothing Test ===\n")

    np.random.seed(42)
    d = 64
    N = 10

    # Create clusters with local structure (sparse: 1-3 questions per cluster)
    clusters = []
    for i in range(N):
        n_q = np.random.randint(1, 4)  # Sparse clusters
        base = np.random.randn(d) * 0.3 + (i // 3) * 0.5  # Groups of 3
        Q = base + np.random.randn(n_q, d) * 0.1
        A = base.reshape(1, -1) + np.random.randn(1, d) * 0.05  # Answer near base
        clusters.append((Q, A))

    print(f"Created {N} clusters")

    # Train FFT smoothing
    fft_proj = FFTSmoothingProjection(cutoff=0.4, blend_factor=0.7)
    fft_proj.train(clusters)

    # Frequency analysis
    freq_info = fft_proj.get_frequency_analysis()
    print(f"\nFrequency analysis:")
    print(f"  90% power at: {freq_info['freq_90_percent']:.2%} of spectrum")
    print(f"  Recommended cutoff: {freq_info['recommended_cutoff']:.2f}")

    # Test projection
    test_query = np.random.randn(d)
    proj_smooth = fft_proj.project(test_query, use_smoothed=True)
    proj_orig = fft_proj.project(test_query, use_smoothed=False)

    print(f"\nProjection test:")
    print(f"  Original norm: {np.linalg.norm(proj_orig):.3f}")
    print(f"  Smoothed norm: {np.linalg.norm(proj_smooth):.3f}")
    print(f"  Cosine sim: {np.dot(proj_orig, proj_smooth) / (np.linalg.norm(proj_orig) * np.linalg.norm(proj_smooth)):.3f}")

    # Test adaptive
    print("\nTesting Adaptive FFT Smoothing...")
    adaptive = AdaptiveFFTSmoothing(min_cutoff=0.2, max_cutoff=0.7)
    adaptive.train(clusters)

    proj_adaptive = adaptive.project(test_query)
    print(f"  Adaptive norm: {np.linalg.norm(proj_adaptive):.3f}")

    print("\n=== Test Complete ===")
