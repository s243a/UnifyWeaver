"""
Minimal Transformation Projection using Procrustes analysis.

This module implements projection learning via minimal transformations
(rotations + scaling) rather than regularized least squares.

The key insight: instead of solving min ||QW - A||² + λ||W||², we find
the minimal transformation (fewest degrees of freedom) that maps Q to A.

Approach:
1. For each cluster, compute per-sample minimal transforms via Procrustes
2. Smooth these transforms across clusters (FFT or kernel smoothing)
3. Balance smoothness with fidelity to the minimal transform

This gives automatic null-space structure without hyperparameter tuning.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.linalg import svd, orthogonal_procrustes
from scipy.fft import fft, ifft
import logging

logger = logging.getLogger(__name__)


def compute_minimal_transform(
    source: np.ndarray,
    target: np.ndarray,
    allow_scaling: bool = True,
    allow_reflection: bool = False,
) -> Tuple[np.ndarray, float, dict]:
    """
    Compute minimal transformation from source to target via Procrustes.

    Args:
        source: Source points (n × d) or single point (d,)
        target: Target points (n × d) or single point (d,)
        allow_scaling: If True, include uniform scaling
        allow_reflection: If True, allow improper rotations (det = -1)

    Returns:
        W: Transformation matrix (d × d)
        scale: Scaling factor (1.0 if allow_scaling=False)
        info: Dict with diagnostics
    """
    source = np.atleast_2d(source)
    target = np.atleast_2d(target)

    n, d = source.shape

    if n == 1:
        # Single point: align directions + scale
        s_norm = np.linalg.norm(source)
        t_norm = np.linalg.norm(target)

        if s_norm < 1e-10 or t_norm < 1e-10:
            return np.eye(d), 1.0, {"degenerate": True}

        s_unit = source.flatten() / s_norm
        t_unit = target.flatten() / t_norm

        # Rotation to align s_unit with t_unit
        # _rotation_between_vectors gives R such that R @ s_unit = t_unit
        # For row vector multiplication (source @ W), we need W = R.T
        R = _rotation_between_vectors(s_unit, t_unit)

        scale = t_norm / s_norm if allow_scaling else 1.0
        W = scale * R.T  # Transpose for row vector multiplication

        return W, scale, {
            "degenerate": False,
            "source_norm": s_norm,
            "target_norm": t_norm,
        }

    # Multiple points: use scipy's orthogonal_procrustes
    # Center the data
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    # Compute optimal rotation
    R, _ = orthogonal_procrustes(source_centered, target_centered)

    if not allow_reflection:
        # Ensure proper rotation (det = +1)
        U, S, Vt = svd(R)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

    # Compute optimal scale
    if allow_scaling:
        source_rot = source_centered @ R
        scale = np.trace(target_centered.T @ source_rot) / np.trace(source_centered.T @ source_centered)
        scale = max(0.01, scale)  # Prevent negative/zero scaling
    else:
        scale = 1.0

    W = scale * R

    # Compute residual
    residual = np.linalg.norm(source @ W - target + (target.mean(axis=0) - scale * source.mean(axis=0) @ R))

    return W, scale, {
        "residual": residual,
        "scale": scale,
        "rotation_det": np.linalg.det(R),
    }


def _rotation_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that rotates v1 to v2.

    Uses the formula: R = I + [v]_× + [v]_×² * (1-c)/s²
    where v = v1 × v2, c = v1 · v2, s = ||v||

    For high dimensions, we use the Householder-based approach.
    """
    d = len(v1)
    c = np.dot(v1, v2)

    if c > 0.9999:
        # Nearly aligned, return identity
        return np.eye(d)

    if c < -0.9999:
        # Nearly opposite, return reflection through orthogonal vector
        # Find a vector orthogonal to v1
        ortho = np.zeros(d)
        min_idx = np.argmin(np.abs(v1))
        ortho[min_idx] = 1.0
        ortho = ortho - np.dot(ortho, v1) * v1
        ortho = ortho / np.linalg.norm(ortho)
        # Rotation by π in the plane spanned by v1 and ortho
        return 2 * np.outer(ortho, ortho) - np.eye(d)

    # General case: rotation in the plane spanned by v1 and v2
    # Construct orthonormal basis for this plane
    u1 = v1
    u2 = v2 - c * v1
    u2 = u2 / np.linalg.norm(u2)

    # Rotation in this plane by angle theta where cos(theta) = c
    s = np.sqrt(1 - c * c)  # sin(theta)

    # R = I + (cos-1)(u1⊗u1 + u2⊗u2) + sin(u2⊗u1 - u1⊗u2)
    R = np.eye(d)
    R += (c - 1) * (np.outer(u1, u1) + np.outer(u2, u2))
    R += s * (np.outer(u2, u1) - np.outer(u1, u2))

    return R


class MinimalTransformProjection:
    """
    Projection using minimal transformations with optional smoothing.

    For each cluster, computes the minimal (Procrustes) transformation,
    then optionally smooths across clusters.
    """

    def __init__(
        self,
        smooth_method: str = "fft",  # "fft", "kernel", or "none"
        fft_cutoff_ratio: float = 0.3,
        fidelity_weight: float = 0.5,  # Balance between smooth and minimal
        allow_scaling: bool = True,
        temperature: float = 0.1,
        per_pair: bool = False,  # If True, compute W per Q/A pair then smooth within cluster
    ):
        """
        Initialize minimal transform projection.

        Args:
            smooth_method: Smoothing method ("fft", "kernel", "none")
            fft_cutoff_ratio: For FFT, keep this fraction of frequencies
            fidelity_weight: Weight for staying close to minimal transform (0-1)
                            0 = fully smoothed, 1 = pure minimal transforms
            allow_scaling: Whether to allow scaling in Procrustes
            temperature: Softmax temperature for routing
            per_pair: If True, compute W per Q/A pair then smooth within cluster.
                     This is "transform then average" vs "average then transform".
        """
        self.smooth_method = smooth_method
        self.fft_cutoff_ratio = fft_cutoff_ratio
        self.fidelity_weight = fidelity_weight
        self.allow_scaling = allow_scaling
        self.temperature = temperature
        self.per_pair = per_pair

        # Trained state
        self.W_minimal: Dict[int, np.ndarray] = {}
        self.W_smoothed: Dict[int, np.ndarray] = {}
        self.W_final: Dict[int, np.ndarray] = {}
        self.centroids: List[np.ndarray] = []
        self.cluster_order: List[int] = []
        self.scales: List[float] = []
        self.W_per_pair: Dict[int, List[np.ndarray]] = {}  # Per-pair transforms before averaging

    def train(self, clusters: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
        """
        Train minimal transform projection.

        Args:
            clusters: List of (Q_i, A_i) tuples where:
                     Q_i = query embeddings for cluster i (n_i × d)
                     A_i = answer embedding for cluster i (d,) or (1 × d)

        Returns:
            Training statistics
        """
        N = len(clusters)
        if N == 0:
            raise ValueError("No clusters provided")

        d = clusters[0][0].shape[1] if clusters[0][0].ndim > 1 else len(clusters[0][0])

        logger.info(f"Training MinimalTransformProjection: N={N}, d={d}")

        # Step 1: Compute minimal transform for each cluster
        for i, (Q, A) in enumerate(clusters):
            if Q.ndim == 1:
                Q = Q.reshape(1, -1)
            if A.ndim == 1:
                A = A.reshape(1, -1)

            # Compute centroid (used for routing regardless of per_pair)
            centroid = Q.mean(axis=0)
            self.centroids.append(centroid)

            if self.per_pair:
                # Per-pair approach: compute W for each (q, a) pair, then average
                n_pairs = min(Q.shape[0], A.shape[0])
                if A.shape[0] == 1:
                    # Single answer for all queries - use same answer for each query
                    A_expanded = np.tile(A, (Q.shape[0], 1))
                else:
                    A_expanded = A

                pair_transforms = []
                pair_scales = []

                for j in range(n_pairs):
                    q_j = Q[j]
                    a_j = A_expanded[j] if j < A_expanded.shape[0] else A_expanded[0]

                    W_j, scale_j, _ = compute_minimal_transform(
                        q_j.reshape(1, -1),
                        a_j.reshape(1, -1),
                        allow_scaling=self.allow_scaling,
                    )
                    pair_transforms.append(W_j)
                    pair_scales.append(scale_j)

                self.W_per_pair[i] = pair_transforms

                # Average the per-pair transforms (this is the within-cluster smoothing)
                W_avg = np.mean(pair_transforms, axis=0)
                scale_avg = np.mean(pair_scales)

                self.W_minimal[i] = W_avg
                self.scales.append(scale_avg)

            else:
                # Original approach: centroid to answer
                answer = A[0] if A.shape[0] >= 1 else A.flatten()

                W, scale, info = compute_minimal_transform(
                    centroid.reshape(1, -1),
                    answer.reshape(1, -1),
                    allow_scaling=self.allow_scaling,
                )

                self.W_minimal[i] = W
                self.scales.append(scale)

        # Step 2: Order clusters by centroid similarity (for FFT)
        self._compute_cluster_order()

        # Step 3: Apply smoothing
        if self.smooth_method == "fft":
            self._apply_fft_smoothing()
        elif self.smooth_method == "kernel":
            self._apply_kernel_smoothing()
        else:
            # No smoothing
            self.W_smoothed = self.W_minimal.copy()

        # Step 4: Blend minimal and smoothed
        for i in range(N):
            self.W_final[i] = (
                self.fidelity_weight * self.W_minimal[i] +
                (1 - self.fidelity_weight) * self.W_smoothed[i]
            )

        return self._compute_stats()

    def _compute_cluster_order(self):
        """Order clusters by centroid similarity (greedy nearest neighbor)."""
        N = len(self.centroids)
        if N == 0:
            return

        # Normalize centroids
        norms = [np.linalg.norm(c) + 1e-8 for c in self.centroids]
        normalized = [c / n for c, n in zip(self.centroids, norms)]

        # Greedy ordering
        remaining = set(range(N))
        order = [0]
        remaining.remove(0)

        while remaining:
            last = order[-1]
            best_sim = -2
            best_next = None

            for candidate in remaining:
                sim = np.dot(normalized[last], normalized[candidate])
                if sim > best_sim:
                    best_sim = sim
                    best_next = candidate

            order.append(best_next)
            remaining.remove(best_next)

        self.cluster_order = order

    def _apply_fft_smoothing(self):
        """Apply FFT-based smoothing to W matrices."""
        N = len(self.W_minimal)
        if N == 0:
            return

        # Stack W matrices in similarity order
        W_stacked = np.stack([self.W_minimal[i] for i in self.cluster_order])
        original_shape = W_stacked.shape

        # Flatten for FFT
        W_flat = W_stacked.reshape(N, -1)

        # FFT along cluster axis
        W_freq = fft(W_flat, axis=0)

        # Low-pass filter
        cutoff = max(1, int(N * self.fft_cutoff_ratio))
        mask = np.zeros(N)
        mask[:cutoff] = 1
        if N > 1:
            mask[-cutoff + 1:] = 1  # Symmetric for real signal

        W_freq_filtered = W_freq * mask[:, np.newaxis]

        # Inverse FFT
        W_smoothed_flat = np.real(ifft(W_freq_filtered, axis=0))
        W_smoothed_stacked = W_smoothed_flat.reshape(original_shape)

        # Map back to original indices
        for i, orig_idx in enumerate(self.cluster_order):
            self.W_smoothed[orig_idx] = W_smoothed_stacked[i]

    def _apply_kernel_smoothing(self):
        """Apply kernel-based smoothing to W matrices."""
        N = len(self.W_minimal)
        if N == 0:
            return

        # Compute kernel matrix (cosine similarity)
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                c_i = self.centroids[i]
                c_j = self.centroids[j]
                K[i, j] = np.dot(c_i, c_j) / (np.linalg.norm(c_i) * np.linalg.norm(c_j) + 1e-8)

        # Normalize rows
        K_norm = K / (K.sum(axis=1, keepdims=True) + 1e-8)

        # Smooth each W
        for i in range(N):
            W_smooth = np.zeros_like(self.W_minimal[0])
            for j in range(N):
                W_smooth += K_norm[i, j] * self.W_minimal[j]
            self.W_smoothed[i] = W_smooth

    def _compute_stats(self) -> dict:
        """Compute training statistics."""
        N = len(self.W_minimal)

        # Measure deviation from minimal
        deviations = []
        for i in range(N):
            diff = np.linalg.norm(self.W_final[i] - self.W_minimal[i])
            norm = np.linalg.norm(self.W_minimal[i])
            deviations.append(diff / (norm + 1e-8))

        stats = {
            "num_clusters": N,
            "smooth_method": self.smooth_method,
            "fidelity_weight": self.fidelity_weight,
            "per_pair": self.per_pair,
            "mean_scale": float(np.mean(self.scales)),
            "std_scale": float(np.std(self.scales)),
            "mean_deviation_from_minimal": float(np.mean(deviations)),
            "max_deviation_from_minimal": float(np.max(deviations)),
        }

        # Add within-cluster variance for per_pair mode
        if self.per_pair and self.W_per_pair:
            within_cluster_vars = []
            for i, transforms in self.W_per_pair.items():
                if len(transforms) > 1:
                    W_stack = np.stack(transforms)
                    var = np.mean(np.var(W_stack, axis=0))
                    within_cluster_vars.append(var)
            if within_cluster_vars:
                stats["mean_within_cluster_variance"] = float(np.mean(within_cluster_vars))

        return stats

    def project(self, query_emb: np.ndarray) -> np.ndarray:
        """
        Project query embedding using soft routing.

        Args:
            query_emb: Query embedding (d,)

        Returns:
            Projected embedding (d,)
        """
        if not self.centroids or not self.W_final:
            raise ValueError("Model not trained. Call train() first.")

        query_emb = query_emb.flatten()

        # Compute routing weights
        similarities = []
        for centroid in self.centroids:
            c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            similarities.append(np.dot(q_norm, c_norm))

        similarities = np.array(similarities)
        exp_sim = np.exp((similarities - np.max(similarities)) / self.temperature)
        weights = exp_sim / (np.sum(exp_sim) + 1e-8)

        # Weighted projection
        projected = np.zeros_like(query_emb)
        for i, weight in enumerate(weights):
            projected += weight * (query_emb @ self.W_final[i])

        return projected

    def get_diagnostics(self) -> dict:
        """Get diagnostics about the learned transformations."""
        if not self.W_minimal:
            return {"trained": False}

        N = len(self.W_minimal)

        # Analyze transformation structure
        rotation_quality = []  # How close to orthogonal?
        effective_ranks = []

        for i in range(N):
            W = self.W_final[i]
            U, S, Vt = svd(W)

            # Orthogonality: ||W^T W - I|| (for scaled rotation, compare to s²I)
            WtW = W.T @ W
            s_sq = np.mean(np.diag(WtW))
            ortho_err = np.linalg.norm(WtW - s_sq * np.eye(W.shape[0])) / np.linalg.norm(WtW)
            rotation_quality.append(1 - ortho_err)

            # Effective rank
            S_norm = S / (S[0] + 1e-8)
            effective_ranks.append(np.sum(S_norm > 0.01))

        return {
            "trained": True,
            "num_clusters": N,
            "mean_rotation_quality": float(np.mean(rotation_quality)),
            "mean_effective_rank": float(np.mean(effective_ranks)),
            "scales": self.scales,
        }
